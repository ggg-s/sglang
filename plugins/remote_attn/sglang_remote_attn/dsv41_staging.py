"""Bind DSV4.1 native global byte entries to the native P pinned stager."""

import functools


def buffers(pool, manager):
    if hasattr(pool, "full_attention_layer_id_mapping"):
        from .qwen_hybrid import kv_buffers

        ordered = kv_buffers(pool)
        if [t.data_ptr() for t in ordered] != list(manager.kv_args.kv_data_ptrs):
            raise ValueError("hybrid D2D manager registration mismatch")
        return ordered
    # MLA skips the MHA tensor-binding path in native prefill initialization.
    # Preserve native registration order and *native* strides: FP4 index scales
    # and payload can span several physical rows for one logical global page.
    tensors = {}
    owners = {}
    for ratio, buffers in pool.kv_pools.items():
        if buffers is None:
            continue
        groups = [("kv", buffers.kv_buffer)]
        index = pool.index_pools.get(ratio)
        if index is not None:
            groups.append(("index", index.contiguous_page_row_buffers()))
        for kind, entries in groups:
            for buffer_index, tensor in enumerate(entries):
                pointer = tensor.data_ptr()
                if pointer in tensors:
                    raise ValueError("aliased DSV4.1 host staging entries")
                tensors[pointer] = tensor
                owners[pointer] = (kind, ratio, buffer_index)
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
    for entry, (tensor, size, stride) in enumerate(
        zip(ordered, args.kv_data_lens, args.kv_item_lens)
    ):
        nbytes = tensor.numel() * tensor.element_size()
        contiguous = tensor.is_contiguous()
        remainder = size % stride if stride > 0 else None
        failed = [
            name
            for name, invalid in (
                ("not_cuda", tensor.device.type != "cuda"),
                ("device_mismatch", tensor.device != device),
                ("not_contiguous", not contiguous),
                ("registered_size_mismatch", nbytes != size),
                ("nonpositive_item_stride", stride <= 0),
                ("nonpositive_registered_size", size <= 0),
                ("size_not_divisible_by_item_stride", stride > 0 and remainder != 0),
            )
            if invalid
        ]
        if failed:
            pointer = tensor.data_ptr()
            kind, ratio, buffer_index = owners[pointer]
            layout = getattr(pool, "kv_layout", None)
            raise ValueError(
                "DSV4.1 host staging tensor/stride mismatch: "
                f"entry={entry} kind={kind} ratio={ratio} buffer_index={buffer_index} "
                f"pointer={pointer:#x} kv_layout={getattr(layout, 'value', layout)} "
                f"page_size={pool.page_size} failed_checks={','.join(failed)}; "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} expected_device={device} "
                f"tensor_stride={tuple(tensor.stride())} contiguous={contiguous} "
                f"tensor_nbytes={nbytes} registered_size={size} "
                f"item_stride={stride} size_mod_stride={remainder}"
            )
    ranges = sorted(zip(pointers, args.kv_data_lens))
    if any(a + n > b for (a, n), (b, _) in zip(ranges, ranges[1:])):
        raise ValueError("overlapping DSV4.1 host staging entries")
    return ordered


def bind(pool, manager):
    ordered = buffers(pool, manager)
    args = manager.kv_args
    pointers = list(args.kv_data_ptrs)
    device = ordered[0].device
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
