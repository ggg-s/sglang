"""Breakable primitives — segmented CUDA graph capture with eager break points.

Public API (also reachable via the deeper module paths):
  - BreakableCUDAGraph, BreakableCUDAGraphCapture — capture/replay
  - eager_on_graph — decorator that marks a callable as a graph break
  - break_graph — helper that inserts a bare graph break
  - mark_split_layer_boundary — boundary for PDMux split-prefill replay
  - split_layer_capture_context — select per-layer DSV4 capture and warmup
  - enable_breakable_cuda_graph — context that flips the Breakable runtime flag
  - is_in_breakable_cuda_graph — runtime flag getter

"""

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (  # noqa: F401
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_graph,
    eager_on_graph,
    is_split_layer_capture_enabled,
    mark_split_layer_boundary,
    split_layer_capture_context,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (  # noqa: F401
    enable_breakable_cuda_graph,
    is_in_breakable_cuda_graph,
)
