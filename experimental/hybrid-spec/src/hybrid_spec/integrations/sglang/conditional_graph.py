"""Experimental device-side dispatch of fixed-shape compact draft tails.

Use the same CUDAGraph IF bindings as PyTorch's torch.cond capture adapter.
Capture model bodies directly: model metadata mutates Python objects and is
not a functional torch.cond branch. No nested graph replay or new CUDA ABI.
"""

from __future__ import annotations

import torch


def require_conditional_graph():
    version = torch.version.cuda
    if (
        torch.version.hip is not None
        or version is None
        or tuple(int(part) for part in version.split(".")[:2]) < (12, 4)
        or not all(
            hasattr(torch.cuda.CUDAGraph, name)
            for name in (
                "begin_capture_to_if_node",
                "end_capture_to_conditional_node",
            )
        )
    ):
        raise ValueError(
            "conditional_draft_graph requires NVIDIA CUDA >=12.4 and PyTorch "
            "CUDAGraph conditional-node bindings (PyTorch 2.14 API); "
            "set conditional_draft_graph=false for ordinary split dispatch"
        )


class ConditionalTailGraph:
    def __init__(self, record, stream):
        require_conditional_graph()
        self.batch_size = record.inputs.batch_size
        count = record.num_neural
        self.live_bs = torch.full_like(count, self.batch_size)
        # B+1 tail histogram bins followed by a mixed-batch counter. All
        # updates are captured; only an explicit diagnostic call reads them.
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
            statistics()
            self.stats.zero_()
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(
                self.graph, pool=torch.cuda.graph_pool_handle(), stream=stream
            ):
                branch(count > 0, lambda: capture_range(1, self.batch_size))
                statistics()
        except Exception as exc:
            # Do not retry/fall back independently on a TP rank after partial
            # collective capture. Restart all ranks with the option disabled.
            raise RuntimeError(
                "conditional draft graph capture failed; restart with "
                "conditional_draft_graph=false. Check CUDA/driver support, "
                "communication backend and allocator settings."
            ) from exc
        self.staging = (one,)

    def replay(self, raw_bs):
        self.live_bs.fill_(raw_bs)
        self.graph.replay()

    def snapshot(self):
        """Explicit, blocking diagnostic readback; never called from decode."""
        values = self.stats.cpu().tolist()
        return values[:-1], values[-1]

    def reset(self):
        self.graph.reset()
