"""Experimental device-side dispatch of fixed-shape compact draft tails.

IF uses PyTorch's conditional bindings. SWITCH uses a capture-only CUDA adapter
and PyTorch-owned fixed-shape child graphs. Model metadata mutates Python
objects, so neither path requires functional torch.cond tracing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


def require_conditional_graph(dispatch="if"):
    if dispatch not in ("if", "switch"):
        raise ValueError("conditional dispatch must be if or switch")
    version = torch.version.cuda
    minimum = (12, 8) if dispatch == "switch" else (12, 4)
    methods = (
        ("raw_cuda_graph", "instantiate")
        if dispatch == "switch"
        else ("begin_capture_to_if_node", "end_capture_to_conditional_node")
    )
    if (
        torch.version.hip is not None
        or version is None
        or tuple(int(part) for part in version.split(".")[:2]) < minimum
        or not all(hasattr(torch.cuda.CUDAGraph, name) for name in methods)
    ):
        raise ValueError(
            f"conditional {dispatch} requires NVIDIA CUDA >={minimum[0]}.{minimum[1]} "
            "and compatible PyTorch CUDAGraph conditional-node bindings/raw graph APIs; "
            "set conditional_draft_graph=false for ordinary split dispatch"
        )


@lru_cache(maxsize=1)
def _switch_adapter():
    from torch.utils.cpp_extension import load

    return load(
        name="hybrid_spec_switch_graph_v2",
        sources=[str(Path(__file__).with_name("switch_graph.cu"))],
        extra_cuda_cflags=["-O3"],
    )


class ConditionalTailGraph:
    def __init__(self, record, stream, *, dispatch="if"):
        require_conditional_graph(dispatch)
        # Compile before capture, on every TP rank. No rank-local fallback.
        adapter = _switch_adapter() if dispatch == "switch" else None
        self.dispatch = dispatch
        self.leaf_graphs = []
        self.batch_size = record.inputs.batch_size
        count = record.num_neural
        self.live_bs = torch.full_like(count, self.batch_size)
        # B+1 histogram bins and a mixed counter. SWITCH updates these in its
        # setter kernel; IF uses captured tensor ops. Only diagnostics read back.
        self.stats = torch.zeros(
            self.batch_size + 2, dtype=torch.int64, device=count.device
        )
        one = torch.ones(1, dtype=torch.int64, device=count.device)
        self.graph = torch.cuda.CUDAGraph()

        def branch(predicate, body):
            self.graph.begin_capture_to_if_node(predicate)
            try:
                body()
            finally:
                self.graph.end_capture_to_conditional_node()

        def capture_range(first, last):
            if first == last:
                record.tails[first].forward()
                return
            middle = (first + last) // 2
            # Capture leaves in ascending order on every TP rank. At runtime
            # the consensus mask plus stable row compaction selects the same
            # single leaf/collective sequence on every rank. Nesting keeps the
            # executed predicate work logarithmic in captured batch size.
            branch(count <= middle, lambda: capture_range(first, middle))
            branch(count > middle, lambda: capture_range(middle + 1, last))

        # Compile/warm up predicate and histogram ops before stream capture.
        def statistics():
            self.stats[:-1].scatter_add_(0, count.reshape(1).long(), one)
            self.stats[-1:].add_(((count > 0) & (count < self.live_bs)).reshape(1))

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            count.zero_()
            if adapter is None:
                statistics()
            self.stats.zero_()
        torch.cuda.synchronize()
        try:
            if adapter is not None:
                # Capture with PyTorch so allocator bookkeeping stays owned
                # by its graph objects. Keep every leaf's pool alive until reset.
                # These deterministic inference graphs are child graph nodes,
                # not nested calls to CUDAGraph.replay().
                for size in range(1, self.batch_size + 1):
                    leaf = torch.cuda.CUDAGraph(keep_graph=True)
                    self.leaf_graphs.append(leaf)
                    with torch.cuda.graph(
                        leaf, pool=torch.cuda.graph_pool_handle(), stream=stream
                    ):
                        record.tails[size].forward()
            with torch.cuda.graph(
                self.graph, pool=torch.cuda.graph_pool_handle(), stream=stream
            ):
                if adapter is None:
                    branch(count > 0, lambda: capture_range(1, self.batch_size))
                    statistics()
                else:
                    adapter.capture_switch(
                        count.data_ptr(),
                        [leaf.raw_cuda_graph() for leaf in self.leaf_graphs],
                        stream.cuda_stream,
                        self.stats.data_ptr(),
                        self.live_bs.data_ptr(),
                    )
        except Exception as exc:
            # Do not retry/fall back independently on a TP rank after partial
            # collective capture. Restart all ranks with the option disabled.
            raise RuntimeError(
                "conditional draft graph capture failed; restart with "
                "conditional_draft_graph=false. Check CUDA/driver support, "
                "communication backend and allocator settings."
            ) from exc
        self.staging = (one,)

    def replay(self, raw_bs, *, batch_prepared=False):
        # Production SWITCH compaction already wrote this on the same stream.
        # Direct callers retain the explicit fill to avoid a stale raw batch.
        if not batch_prepared:
            self.live_bs.fill_(raw_bs)
        self.graph.replay()

    def snapshot(self):
        """Explicit, blocking diagnostic readback; never called from decode."""
        values = self.stats.cpu().tolist()
        return values[:-1], values[-1]

    def reset(self):
        self.graph.reset()
        for leaf in self.leaf_graphs:
            leaf.reset()
        self.leaf_graphs.clear()
