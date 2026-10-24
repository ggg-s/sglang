"""MTP draft state must be seeded after PDMux's last split segment."""

import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPDMuxMTPSplitPrefill(CustomTestCase):
    def _run_final_split(self, *, rebind_lengths=False, cpu_mirror=True, idle=False):
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
            enable_overlap=False,
            pdmux_standard=False,
            enable_pdmux=True,
            split_prefill_batch=batch,
            model_worker=SimpleNamespace(
                forward_batch_split_prefill=Mock(return_value=result)
            ),
            model_config=SimpleNamespace(num_hidden_layers=40),
            _copy_auxiliary_output_to_cpu=Mock(),
            _maybe_report_active_ranks=Mock(),
        )
        returned = inspect.unwrap(Scheduler.run_batch)(scheduler, batch)
        return batch, returned, result, lengths, mirror

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


if __name__ == "__main__":
    unittest.main()
