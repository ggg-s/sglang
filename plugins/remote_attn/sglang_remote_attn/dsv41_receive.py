"""DSV4.1 flat-entry adapter for native PD ring/watermark/scatter lifetimes.

The native MHA staging flag remains off. Enable a pinned ring on D after native
MLA initialization; ordinary same-TP chunks and all state/aux retain native PD.
"""

import logging
import struct

from . import dsv41_staging

logger = logging.getLogger(__name__)
WIRE = "dsv41-flat-v2"


def registration_message(original, manager):
    message = original(manager)
    if getattr(manager, "_dsv41_layerwise_schema", None) is not None:
        # Native code omits strides without customized-info. The progressive
        # state/aux barrier needs D's actual row strides on the FIRST request.
        # The controller installs the schema before any request is accepted.
        lens = manager.kv_args.aux_item_lens
        if len(lens) != len(manager.kv_args.aux_data_ptrs) or any(n <= 0 for n in lens):
            raise ValueError("invalid DSV auxiliary registration strides")
        message[6] = struct.pack(f"{len(lens)}Q", *lens)
    return message


def install(controller, manager, role):
    if role == "prefill" and getattr(manager, "_dsv41_receive_sender_ready", False):
        return

    manager._dsv41_layerwise_schema = controller._schema(manager)
    # Needed even without P D2H staging: a D host ring may be selected later.
    tensors = dsv41_staging.buffers(controller.pool, manager)
    bag = getattr(manager, "kv_buffer_tensors", None)
    if bag is None:
        manager.set_kv_buffer_tensors(tensors, [], controller.pool.page_size)
    elif [t.data_ptr() for t in [*bag["k_buffers"], *bag["v_buffers"]]] != list(
        manager.kv_args.kv_data_ptrs
    ) or bag["page_size"] != controller.pool.page_size:
        raise ValueError("DSV4.1 staging tensor binding changed")
    if role == "prefill":
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingContext,
        )

        if manager._staging_ctx is None:
            manager._staging_ctx = PrefillStagingContext()
        manager._dsv41_receive_sender_ready = True
        return
    progressive = getattr(controller, "dsv41_remote", False)
    if not controller.config.recv_host_staging and not progressive:
        return
    manager._dsv41_progressive_pool = controller.pool if progressive else None
    from sglang.srt.disaggregation.common.staging_handler import (
        DecodeStagingContext,
        DecodeStagingHandler,
        init_staging_allocator,
    )

    ctx = DecodeStagingContext()
    ctx.allocator = init_staging_allocator(
        manager._register_staging_memory,
        manager.kv_args,
        device_type="cuda",
        pin_memory=controller.config.recv_host_staging,
    )
    manager._staging_ctx = ctx
    manager.enable_staging = True
    handler = DecodeStagingHandler(
        manager,
        ctx.allocator,
        manager.kv_buffer_tensors,
        controller.tp_size,
        1,
        controller.rank,
        controller.scheduler,
    )
    manager._staging_handler = handler
    prealloc = controller.scheduler.disagg_decode_prealloc_queue
    transfer = prealloc.transfer_queue
    prealloc.enable_staging = transfer.enable_staging = True
    transfer.staging_handler = handler
    # TE registration alone does not own torch storage. Retain the ring/KV
    # until native worker shutdown and explicit deregistration are both proven.
    owners = getattr(manager.engine, "_dsv41_receive_owners", None)
    if owners is None:
        owners = manager.engine._dsv41_receive_owners = []
    owners.append((manager, ctx.allocator, tensors))
    logger.info(
        "[dsv41-receive] native ring enabled pinned=%s bytes=%s entries=%s",
        controller.config.recv_host_staging,
        ctx.allocator.total_size,
        len(tensors),
    )


def strategy(original, manager, staging_buffer):
    if getattr(manager, "_dsv41_layerwise_schema", None) is None:
        return original(manager, staging_buffer)
    from sglang.srt.disaggregation.common.staging_handler import PrefillStagingStrategy

    return PrefillStagingStrategy(manager, staging_buffer)


def selected(manager, indices):
    indices = tuple(indices)
    count = len(manager.kv_args.kv_item_lens)
    if (
        not indices
        or tuple(sorted(set(indices))) != indices
        or any(i < 0 or i >= count for i in indices)
    ):
        raise ValueError("invalid DSV4.1 staging entries")
    return indices


def prefix_pages(dr, page):
    from .destination_cache import lease_for

    lease = lease_for(dr.req)
    if lease is not None:
        lease.check()
    prefix = int(getattr(dr.req.kv, "cache_protected_len", 0))
    if prefix < 0 or prefix > len(dr.req.origin_input_ids) or prefix % page:
        raise ValueError("DSV4.1 cached prefix is not native page aligned")
    stamp = getattr(dr.req, "_remote_attn_decode_match", None)
    if prefix and (stamp is None or stamp[:2] != (dr.req.cache_request_handle, prefix)):
        raise RuntimeError("DSV4.1 cached prefix changed after native metadata")
    return prefix // page


def cache_metadata(queue, dr, state_indices):
    """Request-window SWA uses request rows, never the FULL page mapping."""
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if not getattr(controller, "dsv41_layerwise", False):
        return state_indices
    window = controller.pool.request_window
    if window is None:
        return state_indices
    from sglang.srt.disaggregation.base.conn import StateType

    types = queue.kv_manager.kv_args.state_types
    if (
        types.count(StateType.SWA) != 1
        or state_indices is None
        or len(types) != len(state_indices)
    ):
        raise ValueError("DSV4.1 request window requires one native SWA payload")
    seq_len = len(dr.req.origin_input_ids)
    start = max(dr.req.kv.cache_protected_len, seq_len - controller.pool.sliding_window)
    page = controller.pool.page_size
    start = start // page * page
    updated = list(state_indices)
    updated[types.index(StateType.SWA)] = window.page_indices(
        int(dr.req.kv.req_pool_idx), start, seq_len
    )
    return updated


def send_group(original, manager, req, chunk, target_info, offset, executor):
    if getattr(manager, "_dsv41_layerwise_schema", None) is None:
        return original(manager, req, chunk, target_info, offset, executor)
    import numpy as np

    entries = selected(manager, chunk.staging_layer_ids)
    if (
        entries != tuple(chunk.kv_buffer_indices)
        or manager.pp_size != 1
        or manager.attn_tp_size != target_info.dst_attn_tp_size
        or target_info.requires_dcp_relayout
        or manager.decode_kv_args_table.get(req.mooncake_session_id) is not target_info
        or not target_info.staging_base_ptr
        or offset < 0
        or (
            chunk.transfer_fn is not None
            and not dsv41_staging.is_transfer(manager, chunk.transfer_fn)
        )
    ):
        raise ValueError("DSV4.1 staging peer/layout changed")
    manager._validate_envelope_kv_layout(
        target_info.dst_kv_ptrs,
        target_info.dst_kv_item_len,
        target_info.dst_attn_tp_size,
    )
    cursor = target_info.staging_base_ptr + offset
    destinations = []
    for i in entries:
        destinations.append(cursor)
        cursor += len(chunk.prefill_kv_indices) * manager.kv_args.kv_item_lens[i]
    if cursor > target_info.staging_base_ptr + target_info.staging_total_size:
        return -1
    return manager._send_kvcache_generic(
        req.mooncake_session_id,
        [manager.kv_args.kv_data_ptrs[i] for i in entries],
        destinations,
        [manager.kv_args.kv_item_lens[i] for i in entries],
        chunk.prefill_kv_indices,
        np.arange(len(chunk.prefill_kv_indices), dtype=np.int32),
        executor,
        force_flat=True,
        transfer_fn=chunk.transfer_fn,
    )


def allocate(original, manager, msg):
    from .prefix_stream import register_range

    streamed = register_range(manager, msg)
    if getattr(manager, "_dsv41_layerwise_schema", None) is None:
        if streamed:
            from .prefix_stream import allocate_mha_range

            return allocate_mha_range(manager, msg)
        if (
            not streamed
            and len(msg) == 8
            and getattr(manager, "_staging_handler", None) is not None
        ):
            dr = manager._staging_handler._room_to_decode_req.get(int(msg[1]))
            offer = getattr(dr, "_pre_layerwise_offer", None)
            if offer is not None:
                from .prefix_stream import descriptor

                entries = struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])
                descriptor(offer, int(msg[2]), entries, int(msg[7]), int(msg[3]))
        return original(manager, msg)
    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

    room, index, pages = (int(msg[i].decode("ascii")) for i in (1, 2, 3))
    session = msg[4].decode("ascii")
    handler = manager._staging_handler
    with handler._scatter_lock:
        dr = handler._room_to_decode_req.get(room)
        if dr is None:
            return  # Late message after native room cleanup.
        receiver = dr.kv_receiver
        if len(msg) not in (8, 9) or not msg[6] or len(msg[6]) % 8:
            raise ValueError("DSV4.1 staging requires explicit physical entries")
        entries = selected(manager, struct.unpack(f"{len(msg[6]) // 8}Q", msg[6]))
        start = int(msg[7].decode("ascii"))
        page = manager.kv_args.page_size
        cached = prefix_pages(dr, page)
        from .prefix_stream import descriptor

        offered = getattr(dr, "_pre_layerwise_offer", None)
        streamed = streamed or (
            offered is not None and descriptor(offered, index, entries, start, pages)
        )
        if (
            index < 0
            or (not streamed and index >= len(manager.kv_args.kv_item_lens))
            or manager.pp_size != 1
            or int(msg[5]) != 0
            or receiver.prefill_info.attn_tp_size != manager.attn_tp_size
            or receiver.prefill_info.pp_size != 1
            or start < 0
            or pages <= 0
            or (
                cached + start + pages
                > (len(dr.req.origin_input_ids) + page - 1) // page
            )
            or (
                not streamed
                and cached + start + pages
                != (len(dr.req.origin_input_ids) + page - 1) // page
            )
        ):
            raise ValueError("invalid DSV4.1 final-chunk staging envelope")
        descriptor = (entries, start, pages)
        previous = receiver.chunk_staging_layers.get(index)
        if previous is not None and previous != descriptor:
            raise ValueError("DSV4.1 staging identity changed")
        # Reject overlapping entry ownership even when chunk ids differ.
        for other, (used, lo, count) in receiver.chunk_staging_layers.items():
            if (
                other != index
                and set(used).intersection(entries)
                and (not streamed or max(lo, start) < min(lo + count, start + pages))
            ):
                raise ValueError("duplicate DSV4.1 staging entries")
        receiver.chunk_staging_layers[index] = descriptor
        infos = receiver.chunk_staging_infos
        if previous is not None:
            # A retransmitted allocation must never reassign an already freed
            # slot. No response is needed after CHUNK_READY consumed this id.
            if (
                index >= len(infos)
                or infos[index][0] < 0
                and infos[index][1] != StagingAllocator.ALLOC_OVERSIZED
            ):
                return
            _, offset, rnd, end, _ = infos[index]
        else:
            required = pages * sum(manager.kv_args.kv_item_lens[i] for i in entries)
            if (
                getattr(manager, "_dsv41_progressive_pool", None) is not None
                and required > manager._staging_ctx.allocator.total_size
            ):
                raise ValueError(
                    "progressive scatter cannot fall back to direct page writes"
                )
            result = manager._staging_ctx.allocator.assign(required)
            while len(infos) <= index:
                infos.append((-1, -1, 0, -1, 0))
            if result is None:
                offset, rnd, end = StagingAllocator.ALLOC_OVERSIZED, 0, -1
                infos[index] = (-1, offset, rnd, end, pages)
            else:
                alloc_id, offset, rnd = result
                end = offset + required
                infos[index] = (alloc_id, offset, rnd, end, pages)
        for bi in manager._staging_ctx.room_bootstrap.get(room, ()):
            if bi["pp_rank"] != 0:
                continue
            sock, lock = receiver._connect_to_bootstrap_server(bi)
            with lock:
                sock.send_multipart(
                    [
                        b"STAGING_RSP",
                        str(room).encode(),
                        str(index).encode(),
                        str(offset).encode(),
                        str(rnd).encode(),
                        str(end).encode(),
                        session.encode(),
                    ]
                )
        handler.register_wm_subscriber(receiver, session)


def scatter(
    original,
    handler,
    staging_offset,
    page_start,
    num_pages,
    decode_req,
    receiver,
    stream=None,
    *,
    layer_ids=None,
    staging_bytes=None,
):
    manager = handler.kv_manager
    if getattr(manager, "_dsv41_layerwise_schema", None) is None:
        return original(
            handler,
            staging_offset,
            page_start,
            num_pages,
            decode_req,
            receiver,
            stream,
            layer_ids=layer_ids,
            staging_bytes=staging_bytes,
        )
    import torch
    from sglang.srt.utils import create_device_stream, device_stream_context

    entries = selected(manager, layer_ids or ())
    required = num_pages * sum(manager.kv_args.kv_item_lens[i] for i in entries)
    allocator = handler.staging_allocator
    page = manager.kv_args.page_size
    cached = prefix_pages(decode_req, page)
    from .prefix_stream import header

    offered = getattr(decode_req, "_pre_layerwise_offer", None)
    streamed = offered is not None and header(offered) is not None
    if (
        staging_bytes != required
        or required <= 0
        or staging_offset < 0
        or staging_offset + required > allocator.total_size
        or receiver.prefill_info.attn_tp_size != handler.decode_tp
        or receiver.prefill_info.pp_size != 1
        or page_start < 0
        or num_pages <= 0
        or cached + page_start + num_pages
        > (len(decode_req.req.origin_input_ids) + page - 1) // page
        or (
            not streamed
            and cached + page_start + num_pages
            != (len(decode_req.req.origin_input_ids) + page - 1) // page
        )
    ):
        raise ValueError("DSV4.1 scatter layout mismatch")
    handler.device_module.set_device(handler.gpu_id)
    if stream is None:
        if allocator._scatter_stream is None:
            allocator._scatter_stream = create_device_stream(handler.device)
        stream = allocator._scatter_stream
    # Native submit_chunk_scatter retains an explicit stream on the request
    # before entering here, and records the entry completion event on it.
    # release_room drains that retained stream even if event.record fails;
    # using another stream here would invalidate both completion and release.
    if decode_req._req_to_token_ready is not None:
        stream.wait_event(decode_req._req_to_token_ready)
    with device_stream_context(stream):
        end = staging_offset + required
        host = allocator.buffer.buffer[staging_offset:end]
        gpu = allocator.buffer.gpu_buffer[staging_offset:end]
        gpu.copy_(host, non_blocking=True)
        row = handler.scheduler.req_to_token_pool.req_to_token
        page_ids = (
            row[
                decode_req.req.kv.req_pool_idx,
                (cached + page_start) * page : (cached + page_start + num_pages)
                * page : page,
            ]
            // page
        ).to(torch.int64)
        if page_ids.numel() != num_pages:
            raise ValueError("DSV4.1 scatter destination row truncated")
        cursor = 0
        tensors = handler.kv_buffer_info["k_buffers"]
        for i in entries:
            stride = manager.kv_args.kv_item_lens[i]
            size = num_pages * stride
            dst = tensors[i].view(torch.uint8).reshape(-1, stride)
            src = gpu[cursor : cursor + size].reshape(num_pages, stride)
            pool = getattr(manager, "_dsv41_progressive_pool", None)
            entry = manager._dsv41_layerwise_schema["entries"][i]
            if pool is not None and entry[0] == "kv":
                layout = pool.get_extra_key_layout(entry[2])
                scatter_prefix(
                    dst,
                    src,
                    page_ids,
                    cached + page_start,
                    len(decode_req.req.origin_input_ids),
                    page,
                    entry[1],
                    layout,
                )
            else:
                dst.index_copy_(0, page_ids, src)
            cursor += size
    return True


def scatter_prefix(dst, src, page_ids, start, prefix, page, ratio, layout):
    """Write only immutable compressed rows; preserve D tail and native padding.

    Global layout may be interleaved or split data/scales. No whole-page writes
    are legal after D owns carry or newly compressed rows of the partial page.
    """
    import torch

    cpp = page // ratio
    full = min(src.shape[0], max(0, prefix // page - start))
    if full:
        dst.index_copy_(0, page_ids[:full], src[:full])
    rows = min(cpp, max(0, prefix // ratio - (start + full) * cpp))
    if full == src.shape[0] or not rows:
        return
    data_stride, scale_base, scale_stride = layout.row_addressing(cpp)
    for base, step, length in (
        (0, data_stride, layout.data_bytes),
        (scale_base, scale_stride, layout.scale_bytes),
    ):
        columns = (
            base
            + torch.arange(rows, device=src.device)[:, None] * step
            + torch.arange(length, device=src.device)
        ).reshape(-1)
        # Keep the page id as a vector: scalar CUDA tensor indexing may call
        # item() and synchronize the scatter stream on the scheduler thread.
        dst[page_ids[full : full + 1, None], columns[None, :]] = src[
            full : full + 1, columns
        ]


def teardown(original, manager):
    owners = getattr(manager.engine, "_dsv41_receive_owners", ())
    own = next((entry for entry in owners if entry[0] is manager), None)
    if own is None:
        return original(manager)
    handler = manager._staging_handler
    if handler._room_to_decode_req or own[1].allocations:
        raise RuntimeError(
            "DSV4.1 receive ring still active; retaining KV and registration"
        )
    threads = list(manager._worker_threads)
    result = original(manager)
    if any(t.is_alive() for t in threads):
        raise RuntimeError(
            "DSV4.1 receive workers remain active; retaining registration"
        )
    if own[1]._scatter_stream is not None:
        own[1]._scatter_stream.synchronize()
    if manager.engine.batch_deregister([own[1].get_base_ptr()]) != 0:
        raise RuntimeError(
            "DSV4.1 receive ring deregistration failed; retaining storage"
        )
    owners.remove(own)
    return result
