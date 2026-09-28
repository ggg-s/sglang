"""MTP draft state must be seeded after PDMux's last split segment."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPDMuxMTPSplitPrefill(unittest.TestCase):
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
        target_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        draft_runner = SimpleNamespace(
            decode_attn_backend_group=[object(), object()],
            update_decode_attn_backend=Mock(),
        )
        draft_worker = SimpleNamespace(_draft_model_runners=lambda: (draft_runner,))
        scheduler = SimpleNamespace(
            model_worker=None,
            tp_worker=SimpleNamespace(model_runner=target_runner),
            draft_worker=draft_worker,
            pdmux_standard=False,
        )

        SchedulerMultiplexMixin._update_decode_attn_backends(scheduler, 1)
        target_runner.update_decode_attn_backend.assert_called_once_with(1)
        draft_runner.update_decode_attn_backend.assert_called_once_with(1)

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
