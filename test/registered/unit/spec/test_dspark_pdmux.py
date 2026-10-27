import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

WORKER_MODULE = "sglang.srt.speculative.dspark_components.dspark_worker_v2"


def make_prefill_worker(device, tp_sync):
    """Small prefill fixture; model execution is supplied by separate tests."""
    worker = object.__new__(DSparkWorkerV2)
    worker._target_hidden_projection_enabled = False
    worker._tp_sync = tp_sync
    worker.model_runner = SimpleNamespace(
        prefill_attention_backend_str=(
            "torch_native" if torch.device(device).type == "cpu" else "triton"
        )
    )
    worker._kv_injector = SimpleNamespace(inject_target_hidden=Mock())
    batch = SimpleNamespace(
        seq_lens=torch.tensor([5, 9], dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor([5, 9], dtype=torch.int32),
        extend_lens=[3, 2],
        prefix_lens=[2, 7],
        out_cache_loc=torch.arange(5, device=device),
        req_pool_indices=torch.tensor([1, 3], device=device),
    )
    output = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=torch.ones(5, 8, device=device)),
        next_token_ids=torch.tensor([11, 13], device=device),
    )
    return worker, batch, output


class TestDSparkPDMux(unittest.TestCase):
    def test_deepseek_v4_hicache_packs_all_draft_stage_pools(self):
        from sglang.srt.speculative import base_spec_worker as spec
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        pools = (object(), object())
        runners = tuple(
            SimpleNamespace(
                token_to_kv_pool=pool,
                model_config=SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["DeepseekV4ForCausalLMDSpark"]
                    )
                ),
            )
            for pool in pools
        )
        target = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(model_runner=target),
            _draft_model_runners=lambda: runners,
        )
        with patch.object(
            spec,
            "get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=True),
        ):
            plan = spec.BaseSpecWorker._build_hicache_draft_plan(worker)
        self.assertEqual(plan.mode, spec.HiCacheDraftMode.PACKED)
        self.assertEqual(plan.device_pools, pools)
        self.assertEqual(target.mtp_draft_device_pools, pools)

    def test_prefill_sync_selects_current_lane_and_preserves_cpu_lengths(self):
        for dp_attention in (False, True):
            for unified in (False, True):
                with self.subTest(dp_attention=dp_attention, unified=unified):
                    prefill_group, attn_group = object(), object()
                    sync, publish = Mock(), Mock()
                    worker, batch, output = make_prefill_worker("cpu", sync)
                    seq_lens, seq_lens_cpu = batch.seq_lens, batch.seq_lens_cpu
                    with (
                        patch(
                            f"{WORKER_MODULE}.get_tp_group", return_value=prefill_group
                        ),
                        patch(
                            f"{WORKER_MODULE}.get_parallel",
                            return_value=SimpleNamespace(
                                enable_dp_attention=dp_attention,
                                attn_cp_size=1,
                                attn_tp_group=attn_group,
                            ),
                        ),
                        patch(
                            f"{WORKER_MODULE}.is_unified_kv_triton",
                            return_value=unified,
                        ),
                    ):
                        result = worker._finalize_prefill(batch, output, publish)

                    sync.sync.assert_called_once_with(
                        SpecTpSyncSite.DSPARK_TARGET,
                        output.next_token_ids,
                        group=attn_group if dp_attention else prefill_group,
                    )
                    self.assertIs(result.new_seq_lens, seq_lens)
                    self.assertIs(batch.seq_lens_cpu, seq_lens_cpu)
                    torch.testing.assert_close(
                        seq_lens_cpu, torch.tensor([5, 9], dtype=torch.int32)
                    )
                    publish.assert_called_once_with(seq_lens)
                    kwargs = worker._kv_injector.inject_target_hidden.call_args.kwargs
                    torch.testing.assert_close(
                        kwargs["positions"], torch.tensor([2, 3, 4, 7, 8])
                    )
                    if unified:
                        torch.testing.assert_close(
                            kwargs["state_slot"], torch.tensor([1, 1, 1, 3, 3])
                        )
                        torch.testing.assert_close(
                            kwargs["final_pos"], torch.tensor([4, 4, 4, 8, 8])
                        )
                    else:
                        self.assertIsNone(kwargs["state_slot"])
                        self.assertIsNone(kwargs["final_pos"])
                    self.assertIsNone(output.logits_output.hidden_states)
                    self.assertIs(
                        result.next_draft_input.bonus_tokens, output.next_token_ids
                    )

    def test_verify_metadata_uses_decode_backend_during_split_prefill(self):
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            TargetVerifyExecutor,
        )

        executor = object.__new__(TargetVerifyExecutor)
        executor._verify_backend_self_adds_seq_lens_cache = None
        executor.target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=object(),
                decode_attn_backend=SimpleNamespace(
                    make_forward_metadata_from_raw_verify=Mock()
                ),
            )
        )
        with patch(
            "sglang.srt.runtime_context.get_disagg",
            return_value=SimpleNamespace(enable_pdmux=True),
        ):
            self.assertTrue(executor._verify_backend_self_adds_seq_lens())

    def test_scheduler_pp_proxy_argument_is_accepted_for_decode(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._forward_decode = Mock(return_value="decoded")
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
        )
        proxy = object()

        result = worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)

        self.assertEqual(result, "decoded")
        worker._forward_decode.assert_called_once_with(batch, None, None)

    def test_scheduler_pp_proxy_argument_reaches_target_prefill(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._verify_planner = SimpleNamespace(note_non_decode_step=Mock())
        worker._observers = SimpleNamespace(note_prefill_step=Mock())
        worker._forward_prefill = Mock(return_value="prefilled")
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=True,
        )
        proxy = object()

        result = worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)

        self.assertEqual(result, "prefilled")
        worker._forward_prefill.assert_called_once_with(
            batch, None, pp_proxy_tensors=proxy
        )

    def test_layer_split_rejects_target_without_aux_capture_support(self):
        target_worker = SimpleNamespace(
            device="cuda",
            model_runner=SimpleNamespace(model=object()),
        )

        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_disagg",
                return_value=SimpleNamespace(
                    enable_pdmux=True, pdmux_prefill_mode="layer_split"
                ),
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_schedule",
                return_value=SimpleNamespace(page_size=256),
            ),
            self.assertRaisesRegex(NotImplementedError, "auxiliary hidden states"),
        ):
            DSparkWorkerV2(
                server_args=SimpleNamespace(page_size=256),
                gpu_id=0,
                ps=SimpleNamespace(),
                nccl_port=0,
                target_worker=target_worker,
            )

    def test_split_prefill_finalizes_only_after_last_segment(self):
        intermediate = SimpleNamespace(logits_output=None)
        final = SimpleNamespace(logits_output=object())
        target_worker = SimpleNamespace(
            forward_batch_split_prefill=Mock(side_effect=[intermediate, final])
        )
        worker = object.__new__(DSparkWorkerV2)
        worker._target_worker = target_worker
        worker._verify_planner = SimpleNamespace(note_non_decode_step=Mock())
        worker._observers = SimpleNamespace(note_prefill_step=Mock())
        worker._finalize_prefill = Mock(return_value="finalized")
        batch = SimpleNamespace(
            split_index=0,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )

        first = worker.forward_batch_split_prefill(batch)
        batch.split_index = 1
        second = worker.forward_batch_split_prefill(batch)

        self.assertIs(first, intermediate)
        self.assertEqual(second, "finalized")
        worker._verify_planner.note_non_decode_step.assert_called_once_with()
        worker._observers.note_prefill_step.assert_called_once_with()
        self.assertEqual(
            target_worker.forward_batch_split_prefill.call_args_list[0].kwargs,
            {"capture_hidden_mode": CaptureHiddenMode.FULL},
        )
        worker._finalize_prefill.assert_called_once_with(batch, final, on_publish=None)

    def test_split_prefill_idle_rank_skips_finalization(self):
        intermediate = SimpleNamespace(logits_output=None)
        final = SimpleNamespace(logits_output=object())
        target_worker = SimpleNamespace(
            forward_batch_split_prefill=Mock(side_effect=[intermediate, final])
        )
        worker = object.__new__(DSparkWorkerV2)
        worker._target_worker = target_worker
        worker._verify_planner = SimpleNamespace(note_non_decode_step=Mock())
        worker._observers = SimpleNamespace(note_prefill_step=Mock())
        worker._decode_idle_result = Mock(return_value="idle")
        worker._finalize_prefill = Mock()
        batch = SimpleNamespace(
            split_index=0,
            forward_mode=SimpleNamespace(is_idle=lambda: True),
        )

        first = worker.forward_batch_split_prefill(batch)
        batch.split_index = 1
        second = worker.forward_batch_split_prefill(batch)

        self.assertIs(first, intermediate)
        self.assertEqual(second, "idle")
        worker._decode_idle_result.assert_called_once_with(on_publish=None)
        worker._finalize_prefill.assert_not_called()

    def test_stream_switch_updates_target_and_draft_runners(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.model_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        worker.draft_model_runner = SimpleNamespace(update_decode_attn_backend=Mock())

        worker.update_pdmux_decode_attn_backend(2)

        worker.model_runner.update_decode_attn_backend.assert_called_once_with(2)
        worker.draft_model_runner.update_decode_attn_backend.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
