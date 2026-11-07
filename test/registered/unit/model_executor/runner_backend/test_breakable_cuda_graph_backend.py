"""BCG row buffers must preserve the dense decode output/sampler contract."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBreakableOutput(CustomTestCase):
    def setUp(self):
        self.backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        self.backend._enable_hicache_event_capture = False
        self.backend._cuda_graph_runner = SimpleNamespace(
            cuda_graph_output_rows=lambda output: None
        )

    def output(self, size):
        return LogitsProcessorOutput(
            next_token_logits=torch.arange(size * 7).reshape(size, 7).float(),
            hidden_states=torch.arange(size * 3).reshape(size, 3).float(),
            hidden_states_token_indices=torch.arange(size),
        )

    def test_logits_hidden_states_and_indices_share_row_buffers(self):
        output = self.output(2)
        buffer = self.backend._alloc_full_buffer(output, 4)
        self.backend._copy_output_to_buffer(output, buffer, 2)
        result = self.backend._slice_output(buffer, 2)
        self.assertIsInstance(result, LogitsProcessorOutput)
        for name in (
            "next_token_logits",
            "hidden_states",
            "hidden_states_token_indices",
        ):
            actual, expected = getattr(result, name), getattr(output, name)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(getattr(buffer, name).shape[0], 4)
            self.assertEqual(actual.data_ptr(), getattr(buffer, name).data_ptr())
        self.assertIsNone(result.next_token_logprobs)
        self.assertIsNone(result.sampling_mask_output)

    def test_replay_does_not_reuse_sampler_mutations_across_batches(self):
        buffer = self.backend._alloc_full_buffer(self.output(16), 16)
        self.backend._outputs = {
            size: self.backend._slice_output(buffer, size) for size in (1, 2, 4, 8, 16)
        }
        self.backend._graphs = {
            size: SimpleNamespace(
                replay=lambda size=size: self.backend._copy_output_to_buffer(
                    self.output(size), buffer, size
                )
            )
            for size in (1, 2, 4, 8, 16)
        }
        previous = []
        with mock.patch(
            "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend.graph_pool_replay_scope",
            contextlib.nullcontext,
        ):
            for size in (16, 8, 4, 1, 2, 16, 8, 1):
                output = self.backend.replay(size, None)
                self.assertTrue(all(output is not item for item in previous))
                torch.testing.assert_close(
                    output.next_token_logits, self.output(size).next_token_logits
                )
                self.assertIsNone(output.next_token_logprobs)
                self.assertIsNone(output.next_token_top_logprobs_val)
                self.assertIsNone(output.sampling_mask_output)
                output.next_token_logprobs = torch.ones(size)
                output.next_token_top_logprobs_val = [[0.5]] * size
                output.sampling_mask_output = object()
                previous.append(output)
        self.assertTrue(all(item.sampling_mask_output is not None for item in previous))

    def test_output_rows_clamp_logits_and_honor_runner_override(self):
        self.assertEqual(self.backend._output_rows(self.output(2), 4), 2)
        self.assertEqual(self.backend._output_rows(self.output(4), 2), 2)
        self.backend._cuda_graph_runner.cuda_graph_output_rows = lambda output: 3
        self.assertEqual(self.backend._output_rows(self.output(4), 2), 3)

    def test_static_non_tensor_fields_pass_through(self):
        output = self.output(2)
        output.customized_info = {"static": ["decode"]}
        buffer = self.backend._alloc_full_buffer(output, 4)
        self.backend._copy_output_to_buffer(output, buffer, 2)
        result = self.backend._slice_output(buffer, 2)
        self.assertIs(result.customized_info, output.customized_info)

    def test_optional_tensor_structure_change_is_rejected(self):
        buffer = self.backend._alloc_full_buffer(self.output(4), 4)
        output = self.output(2)
        output.hidden_states = None
        with self.assertRaisesRegex(ValueError, "structure changed"):
            self.backend._copy_output_to_buffer(output, buffer, 2)

    def test_unsupported_tensor_layout_cannot_silently_truncate_logits(self):
        output = self.output(4)
        output.hidden_states = output.hidden_states[:1]
        with self.assertRaisesRegex(ValueError, "matching row counts"):
            self.backend._output_rows(output, 4)
        output.hidden_states = torch.tensor(1.0)
        with self.assertRaisesRegex(ValueError, "needs a row dimension"):
            self.backend._alloc_full_buffer(output, 4)

    def test_tensor_and_sequence_outputs_still_work(self):
        output = (torch.arange(6).reshape(2, 3), [None, torch.ones(2, 1)])
        buffer = self.backend._alloc_full_buffer(output, 4)
        self.backend._copy_output_to_buffer(output, buffer, 2)
        result = self.backend._slice_output(buffer, 2)
        torch.testing.assert_close(result[0], output[0])
        torch.testing.assert_close(result[1][1], output[1][1])
        self.assertIsNone(result[1][0])


if __name__ == "__main__":
    unittest.main()
