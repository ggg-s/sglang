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


def decode_queue_items(queue):
    """Read native DecodeReq wrappers; ReqList iteration yields bare requests."""
    pending = queue.queue
    return tuple(pending.items()) if hasattr(pending, "items") else tuple(pending)


def remove_decode_items(queue, ids):
    """Keep the native request-list owner while removing admitted/quarantined rows."""
    pending = queue.queue
    items = decode_queue_items(queue)
    if hasattr(pending, "leave"):
        pending.leave([item.req for item in items if id(item) in ids])
    else:
        # Lightweight fixtures and callers predating the ReqList migration.
        queue.queue = [item for item in items if id(item) not in ids]


def restore_detokenize_prefix(req, prefix_len):
    """Build the destination's first native header around already-owned output."""
    if isinstance(prefix_len, bool) or not isinstance(prefix_len, int):
        raise ValueError("D2D detokenization prefix must be an integer")
    if not 0 <= prefix_len <= len(req.output_ids):
        raise ValueError("D2D detokenization prefix exceeds frozen output")
    req.send_token_offset = prefix_len
    # Detokenizer state belongs to the destination process. Its first header
    # carries the output prefix as context, rather than emitting it again.
    req.detokenize_header_sent = False
