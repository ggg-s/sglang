"""CPU checks for yielding and resuming a captured PDMux prefill body."""

from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as runner_module
from sglang.srt.model_executor.model_runner import (
    _can_replay_split_prefill_segment,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg_module,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    is_split_layer_capture_enabled,
    mark_split_layer_boundary,
    split_layer_capture_context,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPdmuxSplitBCG(CustomTestCase):
    def test_split_capture_flag_is_scoped(self):
        self.assertFalse(is_split_layer_capture_enabled())
        with split_layer_capture_context(True):
            self.assertTrue(is_split_layer_capture_enabled())
        self.assertFalse(is_split_layer_capture_enabled())

    def test_graph_yields_at_layer_boundary_and_resumes_with_tail(self):
        calls = []
        graph = BreakableCUDAGraph()
        graph._segments = [
            SimpleNamespace(replay=lambda i=i: calls.append(f"segment-{i}"))
            for i in range(6)
        ]
        graph._break_fns = [lambda i=i: calls.append(f"break-{i}") for i in range(5)]
        graph.split_layer_segments = {0: 1, 1: 3, 2: 5}
        key = ShapeKey(size=8, stream_idx=1)
        backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
        backend._graphs = {key: graph}
        backend._outputs = {key: "body-output"}

        with patch.object(
            bcg_module,
            "get_device_module",
            return_value=SimpleNamespace(current_stream=lambda: object()),
        ):
            self.assertIsNone(backend.replay_split_layers(key, 0, 1, 2))
            calls.append("decode")
            self.assertEqual(backend.replay_split_layers(key, 1, 2, 2), "body-output")

        self.assertEqual(
            calls,
            [
                "segment-0",
                "break-0",
                "segment-1",
                "break-1",
                "segment-2",
                "break-2",
                "decode",
                "segment-3",
                "break-3",
                "segment-4",
                "break-4",
                "segment-5",
            ],
        )

    def test_capture_marker_records_next_segment(self):
        graph = BreakableCUDAGraph()
        graph.capture_split_layers = True
        barriers = []

        def end_segment():
            graph._segments.append(object())

        capture = SimpleNamespace(
            cuda_graph=graph,
            _end_current_segment=end_segment,
            _barrier_fn=lambda: barriers.append("barrier"),
            _begin_new_segment=lambda: None,
        )
        token = bcg_module._current_capture_var.set(capture)
        try:
            mark_split_layer_boundary(0)
            mark_split_layer_boundary(1)
        finally:
            bcg_module._current_capture_var.reset(token)

        self.assertEqual(graph.split_layer_segments, {0: 1, 1: 2})
        self.assertEqual(len(graph._break_fns), 2)
        self.assertEqual(barriers, ["barrier", "barrier"])

    def test_hicache_load_back_keeps_first_segment_eager(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_split_prefill=lambda: True),
            split_index=0,
            pdmux_hicache_consumer_index=3,
        )
        graph = SimpleNamespace(
            pdmux_segmented=True,
            can_run_split_segment=lambda *_: self.fail("graph must be gated"),
        )
        self.assertFalse(_can_replay_split_prefill_segment(batch, graph, 2))

    def test_graph_decision_persists_across_split_segments(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_split_prefill=lambda: True),
            split_index=0,
            pdmux_hicache_consumer_index=-1,
        )
        graph = SimpleNamespace(
            pdmux_segmented=True,
            can_run_split_segment=lambda _batch, layers: layers == 2,
        )
        with patch(
            "sglang.srt.model_executor.model_runner."
            "_prefill_cuda_graph_allows_context_parallel",
            return_value=True,
        ):
            self.assertTrue(_can_replay_split_prefill_segment(batch, graph, 2))
        batch.split_index = 1
        self.assertFalse(_can_replay_split_prefill_segment(batch, graph, 2))
        batch._pdmux_split_graph_state = object()
        self.assertTrue(_can_replay_split_prefill_segment(batch, graph, 2))

    def test_captured_attention_metadata_is_keyed_by_lane(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.pdmux_segmented = True
        with patch.object(runner_module, "get_current_stream_idx", return_value=1):
            self.assertEqual(runner._attn_metadata_key(512), (1, 512))
        with patch.object(runner_module, "get_current_stream_idx", return_value=2):
            self.assertEqual(runner._attn_metadata_key(512), (2, 512))
