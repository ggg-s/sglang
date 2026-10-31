"""Bind DSV4.1 native global byte entries to the native P pinned stager."""

import functools


def bind(pool, manager):
    # MLA skips the MHA tensor-binding path in native prefill initialization.
    # Preserve native registration order and *native* strides: FP4 index scales
    # and payload can span several physical rows for one logical global page.
    tensors = {}
    for ratio, buffers in pool.kv_pools.items():
        if buffers is None:
            continue
        entries = list(buffers.kv_buffer)
        index = pool.index_pools.get(ratio)
        if index is not None:
            entries += list(index.contiguous_page_row_buffers())
        for tensor in entries:
            pointer = tensor.data_ptr()
            if pointer in tensors:
                raise ValueError("aliased DSV4.1 host staging entries")
            tensors[pointer] = tensor
    args = manager.kv_args
    pointers = list(args.kv_data_ptrs)
    if (
        set(pointers) != set(tensors)
        or len(set(pointers)) != len(pointers)
        or len(pointers) != len(args.kv_data_lens)
        or len(pointers) != len(args.kv_item_lens)
        or not pointers
    ):
        raise ValueError("DSV4.1 host staging registration mismatch")
    ordered = [tensors[p] for p in pointers]
    device = ordered[0].device
    for tensor, size, stride in zip(ordered, args.kv_data_lens, args.kv_item_lens):
        if (
            tensor.device.type != "cuda"
            or tensor.device != device
            or not tensor.is_contiguous()
            or tensor.numel() * tensor.element_size() != size
            or stride <= 0
            or size <= 0
            or size % stride
        ):
            raise ValueError("DSV4.1 host staging tensor/stride mismatch")
    ranges = sorted(zip(pointers, args.kv_data_lens))
    if any(a + n > b for (a, n), (b, _) in zip(ranges, ranges[1:])):
        raise ValueError("overlapping DSV4.1 host staging entries")
    stager = manager.ensure_host_stager()
    if stager.poisoned:
        raise RuntimeError("DSV4.1 host staging poisoned; retaining KV")
    # This bag is send-only: SWA/carry/aux retain their native final-state path.
    manager.set_kv_buffer_tensors(ordered, [], pool.page_size)
    expected = sorted(
        (p, n, s) for p, n, s in zip(pointers, args.kv_data_lens, args.kv_item_lens)
    )
    if [r[:3] for r in stager.regions] != expected:
        raise ValueError("DSV4.1 host staging binding failed")
    stager.reserve(device)  # Once at startup, shared by native workers/chunks.
    manager._dsv41_host_stager = stager
    return stager


def transfer(manager):
    return functools.partial(manager._transfer_data, stage=True)


def is_transfer(manager, callback):
    """Only the native staged send adapter may override the byte sender."""
    return (
        isinstance(callback, functools.partial)
        and callback.func == manager._transfer_data
        and not callback.args
        and callback.keywords == {"stage": True}
        and getattr(manager, "_dsv41_host_stager", None) is not None
        and manager._dsv41_host_stager is manager._core_stager
    )
