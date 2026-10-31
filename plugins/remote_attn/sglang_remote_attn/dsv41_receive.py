"""DSV4.1 flat-entry adapter for native PD ring/watermark/scatter lifetimes.

The native MHA staging flag remains off. Enable a pinned ring on D after native
MLA initialization; ordinary same-TP chunks and all state/aux retain native PD.
"""

import logging
import struct

from . import dsv41_staging

logger = logging.getLogger(__name__)
WIRE = "dsv41-flat-v1"


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
    if not controller.config.recv_host_staging:
        return
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
        pin_memory=True,
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
        "[dsv41-receive] pinned native ring enabled bytes=%s entries=%s",
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
    if getattr(manager, "_dsv41_layerwise_schema", None) is None:
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
        if len(msg) != 8 or not msg[6] or len(msg[6]) % 8:
            raise ValueError("DSV4.1 staging requires explicit physical entries")
        entries = selected(manager, struct.unpack(f"{len(msg[6]) // 8}Q", msg[6]))
        start = int(msg[7].decode("ascii"))
        page = manager.kv_args.page_size
        if (
            index < 0
            or index >= len(manager.kv_args.kv_item_lens)
            or manager.pp_size != 1
            or int(msg[5]) != 0
            or receiver.prefill_info.attn_tp_size != manager.attn_tp_size
            or receiver.prefill_info.pp_size != 1
            or start < 0
            or pages <= 0
            or start + pages != (len(dr.req.origin_input_ids) + page - 1) // page
            or dr.req.kv.cache_protected_len != 0
        ):
            raise ValueError("invalid DSV4.1 final-chunk staging envelope")
        descriptor = (entries, start, pages)
        previous = receiver.chunk_staging_layers.get(index)
        if previous is not None and previous != descriptor:
            raise ValueError("DSV4.1 staging identity changed")
        # Reject overlapping entry ownership even when chunk ids differ.
        for other, (used, _, _) in receiver.chunk_staging_layers.items():
            if other != index and set(used).intersection(entries):
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
            layer_ids=layer_ids,
            staging_bytes=staging_bytes,
        )
    import torch

    from sglang.srt.utils import create_device_stream, device_stream_context

    entries = selected(manager, layer_ids or ())
    required = num_pages * sum(manager.kv_args.kv_item_lens[i] for i in entries)
    allocator = handler.staging_allocator
    if (
        staging_bytes != required
        or required <= 0
        or staging_offset < 0
        or staging_offset + required > allocator.total_size
        or receiver.prefill_info.attn_tp_size != handler.decode_tp
        or receiver.prefill_info.pp_size != 1
        or decode_req.req.kv.cache_protected_len != 0
    ):
        raise ValueError("DSV4.1 scatter layout mismatch")
    handler.device_module.set_device(handler.gpu_id)
    if allocator._scatter_stream is None:
        allocator._scatter_stream = create_device_stream(handler.device)
    stream = allocator._scatter_stream
    if decode_req._req_to_token_ready is not None:
        stream.wait_event(decode_req._req_to_token_ready)
    with device_stream_context(stream):
        end = staging_offset + required
        host = allocator.buffer.buffer[staging_offset:end]
        gpu = allocator.buffer.gpu_buffer[staging_offset:end]
        gpu.copy_(host, non_blocking=True)
        page = manager.kv_args.page_size
        row = handler.scheduler.req_to_token_pool.req_to_token
        page_ids = (
            row[
                decode_req.req.kv.req_pool_idx,
                page_start * page : (page_start + num_pages) * page : page,
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
            tensors[i].view(torch.uint8).reshape(-1, stride).index_copy_(
                0, page_ids, gpu[cursor : cursor + size].reshape(num_pages, stride)
            )
            cursor += size
    return True


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
