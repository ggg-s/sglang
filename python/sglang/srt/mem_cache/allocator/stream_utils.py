"""Lifetime of allocator indices shared by CUDA scheduling lanes."""

import torch


def record_index_stream(*indices: torch.Tensor) -> None:
    """Keep index storage alive until reads on the current stream complete.

    PDMux alternates allocator calls between prefill and decode. A free list
    can be allocated on one stream, read on the other, then replaced by cat or
    sort on its original stream. Holding the allocator does not keep the old
    tensor alive. Register the actual inputs before enqueueing their reads.
    The scheduler still owns data-ready dependencies between these streams.
    """
    for tensor in indices:
        if tensor is not None and tensor.is_cuda:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
