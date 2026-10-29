"""MTP draft state must be seeded after PDMux's last split segment."""

import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPDMuxMTPVerifyCommit(CustomTestCase):
    def test_verify_commits_decode_slots_while_prefill_metadata_is_live(self):
        """Equal P/D batch sizes must not silently commit into prefill slots.

        Unequal sizes reproduce the reported scatter length mismatch. Exercise
        both PDMux modes and the ordinary scheduler with the same acceptance.
        """
        for mode in ("layer_split", "standard"):
            for pdmux in (False, True):
                for prefill_slots in ([0], [0, 1]):
                    with self.subTest(mode=mode, pdmux=pdmux, slots=prefill_slots):
                        states = torch.full((10,), -1)
                        intermediate = torch.tensor([[10, 11, 12], [20, 21, 22]])

                        def backend(slots):
                            def commit(*, last_correct_step_indices, **kwargs):
                                self.assertEqual(
                                    len(slots), len(last_correct_step_indices)
                                )
                                states[slots] = intermediate[
                                    torch.arange(len(slots)), last_correct_step_indices
                                ]

                            return SimpleNamespace(
                                update_mamba_state_after_mtp_verify=commit
                            )

                        decode = backend([3, 7])
                        runner = SimpleNamespace(
                            attn_backend=backend(prefill_slots) if pdmux else decode,
                            decode_attn_backend=decode,
                            model=object(),
                            model_config=object(),
                            req_to_token_pool=SimpleNamespace(mamba_pool=None),
                        )
                        batch = SimpleNamespace(
                            forward_mode=ForwardMode.DECODE,
                            mamba_track_indices=None,
                            req_pool_indices=torch.tensor([5, 8]),
                        )
                        with (
                            get_context().override_server_args(
                                enable_pdmux=pdmux, pdmux_prefill_mode=mode
                            ),
                            patch(
                                "sglang.srt.speculative.spec_utils.mambaish_config",
                                return_value={},
                            ),
                        ):
                            commit_mamba_states_after_verify(
                                SimpleNamespace(model_runner=runner),
                                batch,
                                accept_lens=torch.tensor([2, 3], dtype=torch.int32),
                                accept_index=torch.tensor([[0, 1, -1], [3, 4, 5]]),
                                draft_token_num=3,
                            )
                        expected = torch.full_like(states, -1)
                        expected[3], expected[7] = 11, 22
                        torch.testing.assert_close(states, expected)


class TestPDMuxMTPSplitPrefill(CustomTestCase):
    def test_overlap_merge_seeds_real_relay_before_finished_rows_are_filtered(self):
        """Prefill and decode publish on different lanes into one relay.

        The prefill rows must become visible only at merge, before filtering
        mutates their pool indices. Existing decode rows must survive the seed.
        """
        pool = SimpleNamespace(req_to_token=torch.zeros((5, 64), dtype=torch.int32))
        with patch("sglang.srt.managers.overlap_utils._is_cuda", False):
            relay = FutureMap("cpu", SpeculativeAlgorithm.DFLASH, pool)
        relay.publish_ready = Mock()
        relay.new_seq_lens_buf.fill_(-1)
        relay.output_tokens_buf.fill_(-1)
        relay.new_seq_lens_buf[2] = 41
        relay.output_tokens_buf[2] = 99
        indices = torch.tensor([1, 3])
        lengths = torch.tensor([13, 29])
        draft = DFlashWorkerV2._make_next_draft_input_prefill(
            None, bonus_tokens=torch.tensor([21, 34]), seq_lens=lengths
        )
        result = SimpleNamespace(next_draft_input=draft)
        batch = SimpleNamespace(
            req_pool_indices=indices,
            seq_lens=lengths,
            spec_info=draft,
            split_prefill_finished=True,
            beam_tail=None,
            chunked_req=None,
            batch_size=lambda: 2,
            is_empty=lambda: True,
        )
        observed = []

        def process(_batch, _result):
            observed.append(relay.new_seq_lens_buf.clone())
            torch.testing.assert_close(draft.future_indices, indices)

        def filter_batch(**_kwargs):
            batch.req_pool_indices = torch.empty(0, dtype=torch.int64)

        batch.filter_batch = filter_batch
        scheduler = SimpleNamespace(
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            future_map=relay,
            beam_coordinator=SimpleNamespace(
                maybe_select_and_relay=lambda *_a, **_k: None
            ),
            chunked_req=None,
            process_batch_result=process,
        )
        scheduler._relay_forward_payload = lambda *args: (
            Scheduler._relay_forward_payload(scheduler, *args)
        )
        running = SimpleNamespace(is_empty=lambda: True, batch_is_full=False)
        with get_context().override_server_args(speculative_algorithm="DFLASH"):
            returned = SchedulerMultiplexMixin._merge_completed_prefill_batch(
                scheduler,
                batch=batch,
                prefill_result=result,
                running_batch=running,
                all_segments_run=True,
            )
        self.assertIs(returned, running)
        torch.testing.assert_close(observed[0], torch.tensor([-1, 13, 41, 29, -1]))
        torch.testing.assert_close(
            relay.output_tokens_buf, torch.tensor([-1, 21, 99, 34, -1])
        )

    def _run_final_split(
        self, *, rebind_lengths=False, cpu_mirror=True, idle=False, overlap=False
    ):
        def forbid_readback(*_args, **_kwargs):
            raise AssertionError("prefill lengths must not be read back to the host")

        lengths = SimpleNamespace(to=forbid_readback)
        mirror = torch.tensor([13, 29]) if cpu_mirror else None
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE if idle else ForwardMode.SPLIT_PREFILL,
            split_index=39,
            split_forward_count=1,
            extend_num_tokens=16,
            seq_lens=lengths,
            seq_lens_cpu=mirror,
            seq_lens_sum=42 if cpu_mirror else None,
            return_logprob=False,
            return_hidden_states=False,
            input_ids=object(),
        )
        result = SimpleNamespace(
            next_draft_input=object(),
            new_seq_lens=object() if rebind_lengths else lengths,
        )
        scheduler = SimpleNamespace(
            metrics_reporter=Mock(),
            forward_ct=0,
            _sched_idled=False,
            processed_tokens_counter=0,
            scripted_scheduler_hook=None,
            profiler_manager=Mock(),
            forward_sleep_time=None,
            disaggregation_mode=None,
            is_generation=True,
            enable_overlap=overlap,
            pdmux_standard=False,
            enable_pdmux=True,
            split_prefill_batch=batch,
            model_worker=SimpleNamespace(
                forward_batch_split_prefill=Mock(return_value=result)
            ),
            model_config=SimpleNamespace(num_hidden_layers=40),
            _copy_auxiliary_output_to_cpu=Mock(),
            _launch_result_copy=Mock(),
            device_module=SimpleNamespace(Event=object),
            pdmux_prefill_stream=object(),
            pdmux_prefill_copy_stream=object(),
            pdmux_prefill_copy_stream_ctx=contextlib.nullcontext(),
            _maybe_report_active_ranks=Mock(),
        )
        returned = inspect.unwrap(Scheduler.run_batch)(scheduler, batch)
        return batch, returned, result, lengths, mirror

    def test_overlap_keeps_split_worker_and_prompt_lengths(self):
        """Global overlap must not route a split through the ordinary forward.

        Its final token is copied asynchronously; prompt lengths stay intact
        until merge seeds FutureMap for the first decode.
        """
        batch, returned, result, lengths, mirror = self._run_final_split(overlap=True)
        self.assertIs(returned, result)
        self.assertIs(batch.seq_lens, lengths)
        self.assertIs(batch.seq_lens_cpu, mirror)
        self.assertIsNotNone(result.copy_done)
        self.assertIs(batch.spec_info, result.next_draft_input)

    def test_final_split_preserves_lengths_without_host_readback(self):
        """Seeding draft state must not stall decode submission on prompt lengths.

        The worker preserves prompt lengths; their CPU mirror and cached sum
        must survive until merge, including the case without a CPU mirror.
        """
        for cpu_mirror in (True, False):
            with self.subTest(cpu_mirror=cpu_mirror):
                batch, returned, result, lengths, mirror = self._run_final_split(
                    cpu_mirror=cpu_mirror
                )
                self.assertIs(returned, result)
                self.assertIs(batch.seq_lens, lengths)
                self.assertIs(batch.seq_lens_cpu, mirror)
                self.assertEqual(batch.seq_lens_sum, 42 if cpu_mirror else None)
                self.assertIs(batch.spec_info, result.next_draft_input)
                self.assertIsNone(batch.input_ids)

    def test_final_split_rejects_rebound_lengths(self):
        """A worker changing the length tensor cannot leave a stale CPU mirror."""
        with self.assertRaisesRegex(AssertionError, "unchanged prompt lengths"):
            self._run_final_split(rebind_lengths=True)

    def test_idle_final_split_keeps_its_empty_length_mirror(self):
        """A DP-idle worker may return a fresh empty tensor without a readback."""
        batch, _, result, lengths, mirror = self._run_final_split(
            rebind_lengths=True, idle=True, cpu_mirror=False
        )
        self.assertIs(batch.seq_lens, lengths)
        self.assertIs(batch.seq_lens_cpu, mirror)
        self.assertIs(batch.spec_info, result.next_draft_input)

    def test_draft_extend_runs_only_after_final_target_segment(self):
        worker = object.__new__(EAGLEWorkerV2)
        partial = SimpleNamespace(logits_output=None)
        final = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states="target_hidden", mm_input_embeds=None
            ),
            next_token_ids="sampled_token",
            next_draft_input=None,
            new_seq_lens=None,
        )
        target = SimpleNamespace(
            forward_batch_split_prefill=Mock(side_effect=[partial, final])
        )
        draft_runner = SimpleNamespace(tp_group=object())
        draft_modes = []

        def draft_extend(batch, *_args):
            draft_modes.append(batch.forward_mode)
            return "next_draft"

        draft = SimpleNamespace(
            draft_runner=draft_runner,
            draft_tp_context=Mock(return_value=contextlib.nullcontext()),
            prefill_lane_draft_extend_backend=Mock(
                return_value=contextlib.nullcontext()
            ),
            _draft_extend_for_prefill=Mock(side_effect=draft_extend),
        )
        worker._target_worker = target
        worker._draft_worker = draft
        batch = SimpleNamespace(
            forward_mode=ForwardMode.SPLIT_PREFILL, seq_lens="prompt_lengths"
        )

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
                return_value=contextlib.nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
                return_value=contextlib.nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
                return_value=contextlib.nullcontext(),
            ),
        ):
            self.assertIs(worker.forward_batch_split_prefill(batch), partial)
            draft._draft_extend_for_prefill.assert_not_called()
            self.assertIs(worker.forward_batch_split_prefill(batch), final)

        draft._draft_extend_for_prefill.assert_called_once_with(
            batch, "target_hidden", "sampled_token", None
        )
        self.assertEqual(final.next_draft_input, "next_draft")
        self.assertEqual(final.new_seq_lens, "prompt_lengths")
        self.assertEqual(draft_modes, [ForwardMode.EXTEND])
        self.assertEqual(batch.forward_mode, ForwardMode.SPLIT_PREFILL)

    def test_layer_split_switches_draft_decode_backend_group(self):
        target_runner = SimpleNamespace(selected_group=0)
        target_runner.update_decode_attn_backend = lambda idx: setattr(
            target_runner, "selected_group", idx
        )
        draft_runner = SimpleNamespace(
            selected_group=0, decode_attn_backend_group=[object(), object()]
        )
        draft_runner.update_decode_attn_backend = lambda idx: setattr(
            draft_runner, "selected_group", idx
        )
        draft_worker = SimpleNamespace(_draft_model_runners=lambda: (draft_runner,))
        scheduler = SimpleNamespace(
            model_worker=None,
            tp_worker=SimpleNamespace(model_runner=target_runner),
            draft_worker=draft_worker,
            pdmux_standard=False,
        )

        SchedulerMultiplexMixin._update_decode_attn_backends(scheduler, 1)
        self.assertEqual(target_runner.selected_group, 1)
        self.assertEqual(draft_runner.selected_group, 1)

    def test_idle_rank_joins_draft_only_on_final_segment(self):
        worker = object.__new__(EAGLEWorkerV2)
        idle_result = SimpleNamespace(logits_output=None, next_draft_input=None)
        worker._target_worker = SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=43),
            forward_batch_split_prefill=Mock(return_value=idle_result),
        )
        draft = SimpleNamespace(
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_tp_context=Mock(return_value=contextlib.nullcontext()),
            prefill_lane_draft_extend_backend=Mock(
                return_value=contextlib.nullcontext()
            ),
            _draft_extend_for_prefill=Mock(return_value=None),
        )
        worker._draft_worker = draft
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE, split_index=0, split_forward_count=21
        )
        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
                return_value=contextlib.nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
                return_value=contextlib.nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
                return_value=contextlib.nullcontext(),
            ),
        ):
            worker.forward_batch_split_prefill(batch)
            draft._draft_extend_for_prefill.assert_not_called()
            batch.split_index = 21
            batch.split_forward_count = 22
            worker.forward_batch_split_prefill(batch)

        draft._draft_extend_for_prefill.assert_called_once_with(batch, None, None, None)
        self.assertIsNone(idle_result.next_draft_input)
        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)


class TestPDMuxDFlashSplitPrefill(CustomTestCase):
    def test_rejects_target_without_split_aux_capture_before_loading_draft(self):
        target = SimpleNamespace(model_runner=SimpleNamespace(model=object()))
        with (
            get_context().override_server_args(
                enable_pdmux=True, pdmux_prefill_mode="layer_split"
            ),
            self.assertRaisesRegex(NotImplementedError, "auxiliary hidden states"),
        ):
            DFlashWorkerV2(
                server_args=object(),
                gpu_id=0,
                ps=object(),
                nccl_port=0,
                target_worker=target,
            )

    def test_verify_commits_states_from_decode_metadata(self):
        """The prefill batch can have one slot while verify has two requests."""
        for pdmux in (False, True):
            with self.subTest(pdmux=pdmux):
                temporal = torch.full((10,), -1)
                intermediate = torch.tensor([[10, 11, 12], [20, 21, 22]])

                def backend(slots):
                    def commit(*, last_correct_step_indices, **kwargs):
                        if len(slots) != len(last_correct_step_indices):
                            raise ValueError(
                                "state indices and verify steps have different lengths"
                            )
                        temporal[slots] = intermediate[
                            torch.arange(len(slots)), last_correct_step_indices
                        ]

                    return SimpleNamespace(update_mamba_state_after_mtp_verify=commit)

                decode = backend([3, 7])
                runner = SimpleNamespace(
                    attn_backend=backend([0]) if pdmux else decode,
                    decode_attn_backend=decode,
                    model=object(),
                )
                worker = object.__new__(DFlashWorkerV2)
                worker._need_mamba_verify_commit = True
                worker._target_worker = SimpleNamespace(model_runner=runner)
                batch = SimpleNamespace(
                    mamba_track_indices=None, req_pool_indices=torch.tensor([5, 8])
                )
                with get_context().override_server_args(enable_pdmux=pdmux):
                    worker._update_target_mamba_state_after_verify(
                        batch=batch,
                        seq_lens_pre_verify=torch.tensor([9, 13]),
                        seq_lens_post_verify=torch.tensor([11, 16]),
                        commit_lens=torch.tensor([2, 3]),
                    )
                expected = torch.full_like(temporal, -1)
                expected[3], expected[7] = 11, 22
                torch.testing.assert_close(temporal, expected)

    def test_chunk_with_prefix_seeds_draft_pages_only_after_last_split(self):
        """A split result must seed precisely the uncached prompt suffix.

        Missing or premature injection leaves stale draft pages cacheable by
        HiCache; using decode's communicator can reorder P/D collectives.
        """
        lengths = torch.tensor([7, 10])
        features = torch.arange(10, dtype=torch.float32).view(5, 2)
        locations = torch.tensor([2, 5, 7, 11, 14])
        pages = torch.full((16, 2), -1.0)
        writes, broadcasts = [], []

        def inject(*, target_hidden, cache_loc, positions):
            pages[cache_loc] = target_hidden
            writes.append(positions.clone())

        decode_group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            broadcast=lambda *a, **k: self.fail("prefill used decode communicator"),
        )
        prefill_group = SimpleNamespace(
            broadcast=lambda values, src: broadcasts.append(values.clone())
        )
        partial = SimpleNamespace(logits_output=None, next_draft_input=None)
        final = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=features),
            next_token_ids=torch.tensor([21, 34]),
        )
        worker = object.__new__(DFlashWorkerV2)
        worker._target_worker = SimpleNamespace(
            forward_batch_split_prefill=Mock(side_effect=[partial, final])
        )
        worker.model_runner = SimpleNamespace(
            prefill_attention_backend_str="torch_native"
        )
        worker._tp_sync = SpecTpSync(decode_group)
        worker._tp_sync._sites = frozenset({SpecTpSyncSite.DFLASH_TARGET})
        worker._append_target_hidden_to_draft_kv_by_loc = inject
        batch = SimpleNamespace(
            forward_mode=ForwardMode.SPLIT_PREFILL,
            sampling_info=SimpleNamespace(is_all_greedy=True),
            seq_lens=lengths,
            seq_lens_cpu=lengths.clone(),
            seq_lens_sum=17,
            prefix_lens=[5, 7],
            extend_lens=[2, 3],
            out_cache_loc=locations,
        )
        with (
            get_context().override_server_args(enable_dp_attention=False),
            patch(
                "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
                return_value=prefill_group,
            ),
        ):
            self.assertIs(worker.forward_batch_split_prefill(batch), partial)
            self.assertEqual(writes, [])
            self.assertEqual(broadcasts, [])
            result = worker.forward_batch_split_prefill(batch)
        expected = torch.full_like(pages, -1)
        expected[locations] = features
        torch.testing.assert_close(pages, expected)
        torch.testing.assert_close(writes[0], torch.tensor([5, 6, 7, 8, 9]))
        torch.testing.assert_close(broadcasts[0], final.next_token_ids)
        torch.testing.assert_close(
            result.next_draft_input.bonus_tokens, final.next_token_ids
        )
        torch.testing.assert_close(result.next_draft_input.new_seq_lens, lengths)
        self.assertIs(result.new_seq_lens, lengths)
        self.assertIsNone(result.logits_output.hidden_states)
        self.assertEqual(batch.seq_lens_sum, 17)
        torch.testing.assert_close(batch.seq_lens_cpu, lengths)


if __name__ == "__main__":
    unittest.main()
