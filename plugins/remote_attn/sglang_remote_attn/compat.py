"""The small boundary between the experimental runtime and the target SGLang."""


def snapshot_layout(kv_len, *, fast, layer_num=0, key=None, kv_cpu=None):
    """Describe the wire snapshot without reading a nonexistent CPU fast copy."""
    if fast:
        if key is None or layer_num <= 0:
            raise ValueError("fast freeze requires a nonempty GPU KV layout")
        return (
            [[kv_len] for _ in range(layer_num)],
            key.shape[1],
            key.shape[2],
            key.dtype,
        )
    if not kv_cpu or not kv_cpu[0]:
        raise ValueError("copy freeze requires a nonempty CPU KV snapshot")
    key = kv_cpu[0][0][0]
    return (
        [[len(k) for k, _ in layer] for layer in kv_cpu],
        key.shape[1],
        key.shape[2],
        key.dtype,
    )


def set_extend_range(req, prefix_len, extend_end):
    req.prefix_len = prefix_len
    req.extend_end = extend_end


def bind_decode_prefix(req, pool, mapping, kv_len):
    """Allocate a row without replacing Req.kv and losing the allocated row id."""
    if kv_len < 0 or len(mapping) < kv_len:
        raise ValueError("imported mapping is shorter than committed KV")
    old_row = req.kv.req_pool_idx
    ids = pool.alloc([req])
    if ids is None:
        raise RuntimeError("decode admission: req_to_token pool is full")
    row = pool.req_to_token
    try:
        row[int(req.kv.req_pool_idx), :kv_len] = mapping[:kv_len].to(
            dtype=row.dtype, device=row.device
        )
    except Exception:
        if old_row is None:
            pool.free(req)
        raise
    req.kv.kv_committed_len = kv_len
    req.kv.kv_allocated_len = kv_len
    req._handoff_prefix_indices = mapping
    req.handoff_prefix_indices = None
    set_extend_range(req, kv_len, kv_len)
