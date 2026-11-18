"""Paged SWA and compressor-state handoff for DeepSeek-V4 split prefill."""

import hashlib
import json
import logging
import os
from dataclasses import replace

from . import qwen_split_prefill as split

WIRE = b"REMOTE_ATTN_DSV4_SPLIT_V1:"
logger = logging.getLogger(__name__)


def dpa_requested():
    value = os.environ.get("HANDOFF_SPLIT_DPA", "0")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_SPLIT_DPA must be 0 or 1")
    return value == "1"


def decode_pcp_requested():
    value = os.environ.get("HANDOFF_SPLIT_DECODE_PCP", "0")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_SPLIT_DECODE_PCP must be 0 or 1")
    return value == "1"


def init_decode_queue(original, queue, *args, **kwargs):
    if not decode_pcp_requested():
        return original(queue, *args, **kwargs)

    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    # Only these native PD constructors see a TP wire topology. They retain
    # the rank-local KV pools and use the four-rank CPU group for preallocation
    # and receive polling. The model and draft runners keep their CP topology.
    with parallel.override(
        attn_tp_size=parallel.tp_size,
        attn_tp_rank=parallel.tp_rank,
        attn_tp_group=parallel.tp_group,
        attn_cp_size=1,
        attn_cp_rank=0,
        attn_cp_group=parallel.attn_tp_group,
    ):
        return original(queue, *args, **kwargs)


def dspark_requested():
    from sglang.srt.runtime_context import get_disagg, get_spec

    disagg = get_disagg()
    return (
        get_spec().speculative_algorithm == "DSPARK"
        and os.environ.get("HANDOFF_HYBRID_SPLIT_PREFILL") == "1"
        and os.environ.get("HANDOFF_PRE_LAYERWISE") == "remote"
        and os.environ.get("HANDOFF_EXPERIMENTAL_PD_REMOTE", "0") == "0"
        and disagg.disaggregation_mode in ("prefill", "decode")
        and disagg.disaggregation_transfer_backend == "mooncake"
    )


def native_spec_layout(original, kv_args, *, include_flash=False):
    # CommonKVManager publishes this descriptor even for requests that never
    # split. A draft-less peer must fail bootstrap, including native PD fallback.
    return original(kv_args, include_flash=include_flash or dspark_requested())


def schema_id(controller, manager, runner):
    from sglang.srt.runtime_context import get_disagg

    if (
        os.environ.get("HANDOFF_HYBRID_SPLIT_PREFILL") != "1"
        or getattr(controller.hf_config, "model_type", None) != "deepseek_v4"
        or controller.config.mode != "remote"
        or controller.pd_remote
        or get_disagg().disaggregation_transfer_backend != "mooncake"
    ):
        return None

    from sglang.srt.disaggregation.base.conn import StateType

    pool = controller.pool
    backend = runner.attn_backend
    ratios = set(controller.schema_signature["ratios"])
    dspark = getattr(controller, "dsv4_split_dspark", False)
    state_types = manager.kv_args.state_types
    compatible_states = (
        bool(manager.dsv41_spec_layout)
        if dspark
        else len(state_types) == 2
        and set(state_types) == {StateType.SWA, StateType.DSV4_REQUEST_STATE}
    )
    if (
        not {4, 128} <= ratios <= {0, 4, 128}
        or pool.request_window is not None
        or pool.page_size % 128
        or pool.swa_page_size != pool.page_size
        or not controller.scheduler.sliding_window_size
        or getattr(controller.scheduler, "enable_hisparse", False)
        or manager.kv_shard_size != 1
        or runner.prefill_attention_backend_str != "dsv4"
        or runner.decode_attention_backend_str != "dsv4"
        or getattr(backend, "trtllm_attn", False)
        or getattr(backend, "mega_attn", False)
        or not compatible_states
    ):
        return None
    schema = (
        controller.schema_signature,
        pool.swa_page_size,
        controller.scheduler.sliding_window_size,
    )
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()


def window_start(boundary, window, page):
    return max(0, boundary - window) // page * page


def select_swa_prealloc_start(
    original, queue, decode_req, *, prefix_len, fill_len, swa_tail_len, extra_swa_budget
):
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if not getattr(controller, "split_prefill", False) or not getattr(
        controller, "dsv4_split_schema", None
    ):
        return original(
            queue,
            decode_req,
            prefix_len=prefix_len,
            fill_len=fill_len,
            swa_tail_len=swa_tail_len,
            extra_swa_budget=extra_swa_budget,
        )

    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver

    req, receiver = decode_req.req, decode_req.kv_receiver
    eligible = (
        prefix_len == 0
        and fill_len == len(req.origin_input_ids)
        and not decode_req.is_rebootstrap
        and type(receiver) is MooncakeKVReceiver
        and not receiver.require_staging
        and split.request_blocker(req) is None
        and controller.split_prefill_outstanding
        < controller.split_prefill_max_outstanding
        and queue.scheduler.running_batch.batch_size()
        < queue.scheduler.max_running_requests
    )
    page = controller.split_prefill_page_size
    window = queue.scheduler.sliding_window_size
    normal_start = fill_len - swa_tail_len
    allocator = queue.token_to_kv_pool_allocator
    # Native admission already reclaimed its ordinary tail. Spend only unused
    # admission budget and free pages on the additional part of the window.
    free_extra = (
        allocator.swa_available_size() - ((swa_tail_len + page - 1) // page) * page
    )
    allowed, extra = controller.agree(
        [int(eligible), max(0, min(extra_swa_budget, free_extra))]
    )
    if not allowed:
        return None
    earliest_start = max(0, normal_start - extra // page * page)
    first_cut = max((fill_len + 1) // 2, earliest_start + window)
    offer, reason = split.negotiate_decode_offer(
        controller, req, None, min_cut=first_cut
    )
    if offer is None:
        split.log_decision(controller, req, "decode_offer", reason)
        return None
    if offer.suffix_cap:
        first_cut = max(first_cut, fill_len - offer.suffix_cap)
    min_cut = (first_cut + page - 1) // page * page
    floor = (
        split.MIN_SUFFIX_TOKENS
        if controller.split_prefill_adaptive
        else split.min_suffix_tokens(fill_len)
    )
    if min_cut > fill_len - floor:
        split.log_decision(controller, req, "decode_offer", "dsv4_suffix_too_short")
        return None

    controller.split_prefill_outstanding += 1
    req._hybrid_split_reservation = controller
    receiver._dsv4_split_offer = min_cut, offer
    return min(normal_start, window_start(min_cut, window, page))


def advertise(queue, dr):
    receiver = dr.kv_receiver
    pending = getattr(receiver, "_dsv4_split_offer", None)
    if pending is None:
        return
    del receiver._dsv4_split_offer
    min_cut, offer = pending

    from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    req = dr.req
    controller = req._hybrid_split_reservation
    page = controller.split_prefill_page_size
    start = req.kv.get_evicted_seqlen(ComponentType.SWA)
    rows = queue.req_to_token_pool
    allocator = queue.token_to_kv_pool_allocator
    storage = kv_storage_of(rows, allocator)
    full = rows.req_to_token[
        req.kv.req_pool_idx, start : len(req.origin_input_ids) : page
    ]
    pages = (storage.window_ids("swa", full) // page).tolist()
    valid = bool(pages) and min(pages) > 0 and len(set(pages)) == len(pages)
    if not controller.agree([int(valid)])[0]:
        trim_swa(controller, req, 0)
        split.release_reservation(req)
        split.log_decision(controller, req, "decode_offer", "invalid_swa_mapping")
        return
    split.publish_decode_offer(
        controller, receiver, len(req.origin_input_ids), min_cut, offer
    )
    receiver._dsv4_split_window = (controller.dsv4_split_schema, start, pages)
    split.log_decision(
        controller,
        req,
        "decode_offer",
        "offered",
        min_cut=min_cut,
        swa_start=start,
        chunk_size=offer.chunk,
        suffix_cap=receiver._hybrid_split_suffix_cap,
    )


def encode_offer(receiver):
    schema, start, pages = receiver._dsv4_split_window
    min_cut, chunk_cap = receiver._hybrid_split_limits
    return (
        WIRE
        + json.dumps(
            [schema, min_cut, chunk_cap, start, pages], separators=(",", ":")
        ).encode()
    )


def parse_offer(peer, frame):
    try:
        offer = json.loads(frame[len(WIRE) :])
        if not isinstance(offer, list) or len(offer) != 5:
            return
        schema, min_cut, chunk_cap, start, pages = offer
        if (
            type(schema) is str
            and len(schema) == 64
            and all(
                type(n) is int and 0 <= n <= 2**31 - 1
                for n in (min_cut, chunk_cap, start)
            )
            and isinstance(pages, list)
            and pages
            and all(type(p) is int and 0 < p <= 2**31 - 1 for p in pages)
            and len(set(pages)) == len(pages)
        ):
            peer._hybrid_split_limits = min_cut, chunk_cap
            peer._dsv4_split_window = schema, start, pages
    except (ValueError, RecursionError):
        pass  # An unrecognized offer leaves native PD available.


def compatible_offer(controller, peer, length, min_cut):
    schema, start, pages = getattr(peer, "_dsv4_split_window", (None, 0, ()))
    if schema is None:
        return not getattr(controller, "dsv4_split_schema", None)
    page = controller.split_prefill_page_size
    return (
        schema == getattr(controller, "dsv4_split_schema", None)
        and min_cut % page == start % page == 0
        and start
        <= window_start(min_cut, controller.scheduler.sliding_window_size, page)
        and len(pages) == (length - start + page - 1) // page
    )


def seal(controller, req, cut):
    sender = req.disagg_kv_sender
    peers = sender.kv_mgr.transfer_infos.get(req.bootstrap_room)
    if not peers:
        return  # Native cancellation may already have removed the room.
    page = controller.split_prefill_page_size
    first = window_start(cut, controller.scheduler.sliding_window_size, page)
    destinations = []
    for peer in peers.values():
        _, start, pages = peer._dsv4_split_window
        if first < start or cut % page or cut > start + len(pages) * page:
            raise RuntimeError("DSV4 split prefill cut exceeds its reserved SWA window")
        destinations.append(
            (peer, pages[(first - start) // page : (cut - start) // page])
        )
    for peer, pages in destinations:
        peer._dsv4_split_swa_destination = pages


def send_states(
    original,
    manager,
    req,
    prefill_state_indices,
    executor,
    target_rank_registration_info=None,
):
    pages = getattr(req, "_dsv4_split_swa_destination", None)
    if pages is None:
        return original(
            manager, req, prefill_state_indices, executor, target_rank_registration_info
        )

    from sglang.srt.disaggregation.base.conn import StateType

    types = manager.kv_args.state_types
    src, dst = list(prefill_state_indices), list(req.dst_state_indices)
    if len(src) != len(types) or len(dst) != len(types):
        logger.error("DSV4 split prefill transfer has incompatible state indices")
        return -1
    for i, kind in enumerate(types):
        if kind == StateType.SWA:
            # Native paged DSpark uses the target's index mapping for both
            # SWA components. Both destinations must move to the cut window.
            if (
                src[i] is None
                or len(src[i]) != len(pages)
                or any(p <= 0 for p in src[i])
            ):
                logger.error(
                    "DSV4 split prefill SWA component %s does not cover the cut", i
                )
                return -1
            dst[i] = pages
    # The cut ends on a full C128 block. Native PD's C128 indices describe
    # the full prompt's pending block, which the suffix forward will produce.
    src[types.index(StateType.DSV4_REQUEST_STATE)] = None
    return original(
        manager,
        replace(req, dst_state_indices=dst),
        src,
        executor,
        target_rank_registration_info,
    )


def trim_swa(controller, req, cut):
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    scheduler = controller.scheduler
    queue = scheduler.disagg_decode_prealloc_queue
    page = controller.split_prefill_page_size
    start = req.kv.get_evicted_seqlen(ComponentType.SWA)
    end = (
        window_start(cut, max(page, scheduler.sliding_window_size), page)
        if cut
        else req.kv.kv_allocated_len - queue._swa_tail_len(req.kv.kv_allocated_len)
    )
    if end > start:
        rows = queue.req_to_token_pool.req_to_token[req.kv.req_pool_idx, start:end]
        queue.token_to_kv_pool_allocator.free_swa_segment(rows, start_pos=start)
        req.kv.set_evicted_seqlen(ComponentType.SWA, end)
