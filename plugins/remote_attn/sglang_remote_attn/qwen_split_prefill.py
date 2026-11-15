"""Cache-aware split PD using native Mooncake completion and local KV."""

import logging
import math
import os
import time
from http import HTTPStatus
from typing import NamedTuple

from .split_decode_budget import (
    AUTO_SUFFIX_CAP_MAX,
    MIN_SUFFIX_TOKENS,
    RATE_LIMIT,
    DecodeBudget,
    ema,
    read_budget,
)

logger = logging.getLogger(__name__)
# V3 advertises the cut bound and D's chunk cap. V2 assumes a fixed suffix floor.
WIRE = b"REMOTE_ATTN_HYBRID_SPLIT_V3:"
DECODE_PRIORITY = b"REMOTE_ATTN_SPLIT_DECODE_PRIORITY"
SUFFIX_CAP_WIRE = b"REMOTE_ATTN_SPLIT_SUFFIX_CAP:"
SUFFIX_RATE_WIRE = b"REMOTE_ATTN_SPLIT_RATE:"
# output_ids rows are already transferred in full; native uses only slot 0.
CUT_SLOT = 1
MIN_PROMPT_TOKENS = 4096
BALANCED_CUT_RATIO = 0.75
PREFIX_GROWTH_TOKENS = 1024
# 2P1D SWE-bench calibration on RTX 5090 TP2 / Qwen3.8-27B-FP8.
# For a target suffix time fraction f, size the average suffix at most
# f * suffix_prefill_rate / admitted_split_rps. Recalibrate for other loads.
DEFAULT_SUFFIX_CAP_TOKENS = 4096
# Legacy balanced basis: 128 decode steps at 14 ms, 6800 prefill tok/s.
COLLISION_BUDGET_TOKENS = 128 * 14 * 6800 // 1000
HOT_DEFER_TICKS = 45
DEFER_PATIENCE_TICKS = 15
AUTO_SUFFIX_CAP_GRANULARITY = 512
AUTO_SUFFIX_RATE_EMA = 0.7
TP_MIN_IDENTITY = 2**31 - 1


class DecodeOffer(NamedTuple):
    suffix_cap: int
    chunk: int
    schedule: tuple[int, int, int, int, int] | None


def min_suffix_tokens(length):
    return min(MIN_SUFFIX_TOKENS, max(length // 16, 1))


def quantize_suffix_cap(tokens):
    tokens = int(min(tokens, AUTO_SUFFIX_CAP_MAX))
    return max(
        MIN_SUFFIX_TOKENS,
        tokens // AUTO_SUFFIX_CAP_GRANULARITY * AUTO_SUFFIX_CAP_GRANULARITY,
    )


def configure(controller):
    value = os.environ.get("HANDOFF_HYBRID_SPLIT_PREFILL", "1")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_HYBRID_SPLIT_PREFILL must be 0 or 1")
    dsv4 = getattr(controller, "dsv4_split_schema", None)
    # The paged DSV4 adapter is enabled explicitly while full-model parity
    # and performance are being validated.
    dsv4_enabled = dsv4 and os.environ.get("HANDOFF_HYBRID_SPLIT_PREFILL") == "1"
    controller.split_prefill = (
        value == "1"
        and controller.config.mode == "remote"
        and (
            getattr(controller, "hybrid_gdn", False)
            or not getattr(controller, "dsv41_layerwise", False)
            or bool(dsv4_enabled)
        )
    )
    controller.split_prefill_page_size = controller.pool.page_size if dsv4 else 1
    if controller.split_prefill:
        from sglang.srt.runtime_context import get_disagg

        disagg = get_disagg()
        if (
            disagg.disaggregation_decode_allocation_policy == "prefill_complete"
            or disagg.disaggregation_decode_host_receive_threshold != 1
            or disagg.disaggregation_decode_enable_offload_kvcache
        ):
            raise ValueError("split prefill requires early device KV allocation")
    scheduler = controller.scheduler
    role = scheduler.disaggregation_mode.value
    adaptive = os.environ.get("HANDOFF_SPLIT_ADAPTIVE", "0")
    if adaptive not in ("0", "1"):
        raise ValueError("HANDOFF_SPLIT_ADAPTIVE must be 0 or 1")
    controller.split_prefill_adaptive = (
        controller.split_prefill and role in ("prefill", "decode") and adaptive == "1"
    )
    controller.split_prefill_decode_budget = None
    controller.split_prefill_prefix_rate = None
    controller.split_prefill_budget = None
    controller.split_prefill_chunk_size = None
    controller.split_prefill_cut_ratio = None
    controller.split_prefill_cut_basis = None
    controller.split_prefill_suffix_tokens = None
    controller.split_prefill_suffix_cap = None
    controller.split_prefill_min_queue_tokens = 0
    controller.split_prefill_max_outstanding = None
    controller.split_prefill_defer_ticks = 0
    controller.split_prefill_patience_ticks = DEFER_PATIENCE_TICKS
    controller.split_prefill_suffix_budget_s = None
    controller.split_prefill_auto_cap = 0
    controller.split_prefill_suffix_rate = None
    controller.split_prefill_rate_samples = 0
    if controller.split_prefill and role == "prefill":
        cut_env = "HANDOFF_HYBRID_SPLIT_PREFILL_CUT_RATIO"
        try:
            cut_ratio = float(os.environ.get(cut_env, str(BALANCED_CUT_RATIO)))
        except ValueError as exc:
            raise ValueError(f"{cut_env} must be in [0.5, 1)") from exc
        if not 0.5 <= cut_ratio < 1:
            raise ValueError(f"{cut_env} must be in [0.5, 1)")
        basis = os.environ.get("HANDOFF_SPLIT_CUT_BASIS", "miss")
        if basis not in ("balanced", "miss", "whole"):
            raise ValueError("HANDOFF_SPLIT_CUT_BASIS must be balanced, miss or whole")
        suffix_env = "HANDOFF_SPLIT_SUFFIX_TOKENS"
        try:
            suffix_tokens = int(os.environ.get(suffix_env, "0"))
        except ValueError as exc:
            raise ValueError(f"{suffix_env} must be a nonnegative integer") from exc
        if suffix_tokens < 0:
            raise ValueError(f"{suffix_env} must be a nonnegative integer")
        cap_env = "HANDOFF_SPLIT_SUFFIX_CAP_TOKENS"
        default_cap = (
            "auto"
            if controller.split_prefill_adaptive
            else str(DEFAULT_SUFFIX_CAP_TOKENS)
        )
        raw_cap = os.environ.get(cap_env, default_cap)
        suffix_cap = None
        if raw_cap != "auto":
            try:
                suffix_cap = int(raw_cap)
            except ValueError as exc:
                raise ValueError(f"{cap_env} must be 'auto' or an integer") from exc
            if suffix_cap < MIN_SUFFIX_TOKENS:
                raise ValueError(f"{cap_env} must be >= {MIN_SUFFIX_TOKENS}")
        # Instantaneous P backlog missed most useful splits on 2P1D. Keep it
        # as an opt-in experiment.
        gate_env = "HANDOFF_SPLIT_MIN_P_QUEUE_TOKENS"
        try:
            min_queue_tokens = int(os.environ.get(gate_env, "0"))
        except ValueError as exc:
            raise ValueError(f"{gate_env} must be a nonnegative integer") from exc
        if min_queue_tokens < 0:
            raise ValueError(f"{gate_env} must be a nonnegative integer")
        controller.split_prefill_cut_ratio = cut_ratio
        controller.split_prefill_cut_basis = basis
        controller.split_prefill_suffix_tokens = suffix_tokens
        controller.split_prefill_suffix_cap = suffix_cap
        controller.split_prefill_min_queue_tokens = min_queue_tokens
    if controller.split_prefill and role == "decode":
        defer_env = "HANDOFF_SPLIT_DEFER_TICKS"
        try:
            defer_ticks = int(os.environ.get(defer_env, "150"))
        except ValueError as exc:
            raise ValueError(f"{defer_env} must be a nonnegative integer") from exc
        if defer_ticks < 0:
            raise ValueError(f"{defer_env} must be a nonnegative integer")
        controller.split_prefill_defer_ticks = defer_ticks
        patience_env = "HANDOFF_SPLIT_DEFER_PATIENCE_TICKS"
        try:
            patience = int(os.environ.get(patience_env, str(DEFER_PATIENCE_TICKS)))
        except ValueError as exc:
            raise ValueError(f"{patience_env} must be a nonnegative integer") from exc
        if patience < 0:
            raise ValueError(f"{patience_env} must be a nonnegative integer")
        controller.split_prefill_patience_ticks = patience
        chunk_env = "HANDOFF_HYBRID_SPLIT_PREFILL_CHUNK_SIZE"
        minimum_chunk = 0 if controller.split_prefill_adaptive else 1
        default_chunk = "0" if controller.split_prefill_adaptive else "4096"
        try:
            chunk_size = int(os.environ.get(chunk_env, default_chunk))
        except ValueError as exc:
            raise ValueError(f"{chunk_env} must be an integer") from exc
        if chunk_size < minimum_chunk:
            raise ValueError(f"{chunk_env} must be >= {minimum_chunk}")
        limit_env = "HANDOFF_HYBRID_SPLIT_PREFILL_MAX_OUTSTANDING"
        try:
            limit = int(os.environ.get(limit_env, str(scheduler.max_running_requests)))
        except ValueError as exc:
            raise ValueError(f"{limit_env} must be a positive integer") from exc
        if limit <= 0:
            raise ValueError(f"{limit_env} must be a positive integer")
        controller.split_prefill_max_outstanding = limit
        controller.split_prefill_outstanding = 0
        controller.split_prefill_budget = scheduler.max_prefill_tokens
        controller.split_prefill_chunk_size = min(
            chunk_size or 4096, controller.split_prefill_budget
        )
        page = controller.split_prefill_page_size
        controller.split_prefill_chunk_size -= (
            controller.split_prefill_chunk_size % page
        )
        if not controller.split_prefill_chunk_size:
            raise ValueError(
                "split prefill needs a decode chunk of at least one KV page"
            )
        auto_env = "HANDOFF_SPLIT_CAP_AUTO"
        auto = os.environ.get(auto_env, "0")
        if auto not in ("0", "1"):
            raise ValueError(f"{auto_env} must be 0 or 1")
        if controller.split_prefill_adaptive:
            g = read_budget("HANDOFF_SPLIT_DECODE_BUDGET_G", 1.0, allow_zero=True)
            # A paged cut can need up to one extra partial page of suffix.
            # Round the floor up to a cap quantum so unaligned prompts remain
            # eligible for coverage probes at the bottom of the feedback loop.
            min_cap = quantize_suffix_cap(
                MIN_SUFFIX_TOKENS + page - 1 + AUTO_SUFFIX_CAP_GRANULARITY - 1
            )
            controller.split_prefill_decode_budget = DecodeBudget(
                target=g / (1 + g),
                stall_ms=read_budget("HANDOFF_SPLIT_STALL_MS", 600),
                defer_ms=read_budget("HANDOFF_SPLIT_DEFER_MS", 2000, allow_zero=True),
                patience_ms=read_budget(
                    "HANDOFF_SPLIT_DEFER_PATIENCE_MS", 200, allow_zero=True
                ),
                pinned_chunk=bool(chunk_size),
                cap=max(
                    min_cap, quantize_suffix_cap(controller.split_prefill_chunk_size)
                ),
                min_cap=min_cap,
            )
            budget = controller.split_prefill_decode_budget
            if not budget.defer_ms:
                controller.split_prefill_defer_ticks = 0
            if not budget.patience_ms:
                controller.split_prefill_patience_ticks = 0
        elif auto == "1":
            budget_env = "HANDOFF_SPLIT_SUFFIX_BUDGET_S"
            try:
                budget = float(os.environ.get(budget_env, "2.0"))
            except ValueError as exc:
                raise ValueError(
                    f"{budget_env} must be a finite positive number"
                ) from exc
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError(f"{budget_env} must be a finite positive number")
            controller.split_prefill_suffix_budget_s = budget
            controller.split_prefill_auto_cap = quantize_suffix_cap(
                controller.split_prefill_chunk_size
            )
        if not getattr(controller, "hybrid_gdn", False):
            from .remote_data_plane import Visibility

            controller.pd_visibility = Visibility(
                scheduler.req_to_token_pool.req_to_token.device
            )
    if controller.rank == 0:
        logger.info(
            "[handoff-split-config] role=%s enabled=%s max_prefill_tokens=%s "
            "chunk_size=%s cut_ratio=%s cut_basis=%s suffix_tokens=%s suffix_cap=%s "
            "max_outstanding=%s defer_ticks=%s min_p_queue_tokens=%s "
            "auto_cap=%s suffix_budget_s=%s patience_ticks=%s adaptive=%s decode_budget=%s",
            role,
            controller.split_prefill,
            controller.split_prefill_budget,
            controller.split_prefill_chunk_size,
            controller.split_prefill_cut_ratio,
            controller.split_prefill_cut_basis,
            controller.split_prefill_suffix_tokens,
            controller.split_prefill_suffix_cap,
            controller.split_prefill_max_outstanding,
            controller.split_prefill_defer_ticks,
            controller.split_prefill_min_queue_tokens,
            controller.split_prefill_auto_cap,
            controller.split_prefill_suffix_budget_s,
            controller.split_prefill_patience_ticks,
            controller.split_prefill_adaptive,
            controller.split_prefill_decode_budget,
        )


def request_blocker(req):
    if len(req.origin_input_ids) < MIN_PROMPT_TOKENS:
        return "prompt_too_short"
    if req.output_ids or req.is_retracted or req.retracted_stain:
        return "request_already_started"
    if req.return_logprob:
        return "return_logprob"
    if (
        req.return_hidden_states_mode.need_capture()
        or req.return_sampling_mask
        or req.return_routed_experts
        or req.return_indexer_topk
    ):
        return "auxiliary_outputs"
    if (
        req.grammar is not None
        or req.beam_group is not None
        or req.custom_logit_processor
    ):
        return "structured_or_custom_sampling"
    if req.session is not None:
        return "session_request"
    if (
        req.multimodal_inputs is not None
        or req.input_embeds is not None
        or req.positional_embed_overrides is not None
    ):
        return "non_token_inputs"
    if req.sampling_params.max_new_tokens <= 0:
        return "no_generation"
    return None


def log_decision(controller, req, stage, reason, **details):
    if controller.rank == 0:
        logger.info(
            "[handoff-split-decision] stage=%s reason=%s rid=%s room=%s prompt=%s %s",
            stage,
            reason,
            req.rid,
            req.bootstrap_room,
            len(req.origin_input_ids),
            " ".join(f"{k}={v}" for k, v in details.items()),
        )


def create_sender(original, queue, req, num_kv_heads):
    created = original(queue, req, num_kv_heads)
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if created and (
        getattr(controller, "hybrid_gdn", False)
        or getattr(controller, "split_prefill", False)
    ):
        req._hybrid_prefill_controller = controller
    return created


def can_prefill_before_bootstrap(original, queue, req):
    controller = getattr(req, "_hybrid_prefill_controller", None)
    if getattr(controller, "split_prefill", False) and request_blocker(req) is None:
        # An optimistic forward can pass the cut before D publishes its offer.
        return False
    return original(queue, req)


def negotiate_decode_offer(controller, req, reason):
    length = len(req.origin_input_ids)
    if length < MIN_PROMPT_TOKENS:
        return None, reason or "decode_suffix_too_short"
    chunk = controller.split_prefill_chunk_size
    budget = controller.split_prefill_decode_budget
    if budget is None:
        suffix_cap = controller.split_prefill_auto_cap
        if suffix_cap and controller.rank != 0:
            suffix_cap = TP_MIN_IDENTITY
        allowed, suffix_cap = controller.agree([int(reason is None), suffix_cap])
        schedule = None
    else:
        if controller.rank == 0:
            parameters = budget.offer(chunk, controller.split_prefill_budget)
        else:
            # Publish rank 0's estimates through MIN while keeping this rank's
            # prefill limit and request veto in the same vote.
            parameters = (
                TP_MIN_IDENTITY,
                controller.split_prefill_budget,
                TP_MIN_IDENTITY,
                TP_MIN_IDENTITY,
                TP_MIN_IDENTITY,
            )
        allowed, suffix_cap, chunk, rate, neg_step_us, coverage = controller.agree(
            [int(reason is None), *parameters]
        )
        chunk -= chunk % controller.split_prefill_page_size
        if allowed and length < 2 * chunk:
            return None, "decode_split_short_for_chunk"
        if allowed and req.bootstrap_room % 1000 >= coverage:
            return None, "decode_split_budget"
        step_us = -neg_step_us
        defer, patience = budget.defer_steps(
            step_us,
            (
                controller.split_prefill_defer_ticks,
                controller.split_prefill_patience_ticks,
            ),
        )
        schedule = chunk, defer, patience, rate, step_us
    if not allowed:
        return None, reason or "decode_peer_rejected"
    return DecodeOffer(suffix_cap, chunk, schedule), None


def publish_decode_offer(controller, receiver, length, min_cut, offer):
    chunk = offer.chunk
    suffix_cap = length - min_cut if offer.suffix_cap else 0
    receiver._hybrid_split_limits = (
        min_cut,
        chunk if suffix_cap else max(chunk, MIN_SUFFIX_TOKENS),
    )
    receiver._hybrid_split_suffix_cap = suffix_cap
    defer = controller.split_prefill_defer_ticks
    if offer.schedule is not None:
        receiver._hybrid_split_schedule = offer.schedule
        _, defer, _, rate, _ = offer.schedule
        receiver._hybrid_split_suffix_rate = rate
    receiver._hybrid_split_decode_priority = defer > 0


def advertise(queue, dr, decode_prefix_len):
    scheduler = queue.scheduler
    controller = getattr(scheduler, "_pre_layerwise", None)
    if not getattr(controller, "split_prefill", False):
        return
    if getattr(controller, "dsv4_split_schema", None):
        from . import dsv4_split_prefill

        return dsv4_split_prefill.advertise(queue, dr)
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver

    receiver, req = dr.kv_receiver, dr.req
    if getattr(receiver, "_hybrid_split_offer_checked", False):
        return
    receiver._hybrid_split_offer_checked = True
    length = len(req.origin_input_ids)
    # D accepts up to half the prompt as suffix, processed in bounded chunks.
    # P chooses the actual cut after matching its own prefix cache.
    min_cut = (length + 1) // 2
    reason = request_blocker(req)
    if dr.is_rebootstrap or decode_prefix_len:
        reason = "decode_already_owns_prefix"
    elif type(receiver) is not MooncakeKVReceiver:
        reason = "non_mooncake_receiver"
    elif reason is None and length - min_cut < min_suffix_tokens(length):
        reason = "decode_suffix_too_short"
    if reason is None and (
        controller.split_prefill_outstanding >= controller.split_prefill_max_outstanding
        or scheduler.running_batch.batch_size() >= scheduler.max_running_requests
    ):
        reason = "decode_busy"
    # One vote publishes the offer, including a rank-consistent schedule for
    # the admitted request. Keep rank-local timing estimates out of next_batch.
    offer, reason = negotiate_decode_offer(controller, req, reason)
    if offer is not None:
        if offer.suffix_cap:
            min_cut = max(min_cut, length - offer.suffix_cap)
        controller.split_prefill_outstanding += 1
        req._hybrid_split_reservation = controller
        publish_decode_offer(controller, receiver, length, min_cut, offer)
    log_decision(
        controller,
        req,
        "decode_offer",
        reason or "offered",
        min_cut=min_cut if offer else 0,
        outstanding=controller.split_prefill_outstanding,
        chunk_size=offer.chunk if offer else controller.split_prefill_chunk_size,
        suffix_cap=getattr(receiver, "_hybrid_split_suffix_cap", 0),
    )


def release_reservation(req):
    controller = getattr(req, "_hybrid_split_reservation", None)
    if controller is not None:
        req._hybrid_split_reservation = None
        controller.split_prefill_outstanding -= 1


def offered_limits(req, controller):
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender

    reason = request_blocker(req)
    if reason is not None:
        return None, reason
    sender = req.disagg_kv_sender
    if type(sender) is not MooncakeKVSender:
        return None, "non_mooncake_sender"
    if req.start_send_idx or req.kv.holds_kv:
        return None, "prefill_already_started"
    manager = sender.kv_mgr
    peers = tuple(manager.transfer_infos.get(req.bootstrap_room, {}).values())
    if len(peers) != 1 or peers[0].is_dummy:
        return None, "unexpected_destination_peers"
    peer = peers[0]
    limits = getattr(peer, "_hybrid_split_limits", None)
    if limits is None:
        return None, "decode_did_not_offer"
    cut, chunk_cap = limits
    length = len(req.origin_input_ids)
    suffix_cap = getattr(peer, "_hybrid_split_suffix_cap", 0)
    if (
        not (length + 1) // 2 <= cut <= length - min_suffix_tokens(length)
        or chunk_cap < (1 if suffix_cap else MIN_SUFFIX_TOKENS)
        or (suffix_cap and suffix_cap != length - cut)
    ):
        return None, "invalid_decode_split_limits"
    page = controller.split_prefill_page_size
    if len(peer.dst_kv_indices) != (length + page - 1) // page:
        return None, "destination_kv_length_mismatch"
    if page > 1 or getattr(peer, "_dsv4_split_window", None) is not None:
        from . import dsv4_split_prefill

        if not dsv4_split_prefill.compatible_offer(controller, peer, length, cut):
            return None, "incompatible_destination_split_layout"
    registration = manager.decode_kv_args_table.get(peer.mooncake_session_id)
    if (
        registration is None
        or registration.requires_dcp_relayout
        or registration.dst_attn_tp_size != controller.tp_size
    ):
        return None, "incompatible_destination_registration"
    # The legacy decode-priority marker still describes D's configured defer
    # behavior to older peers. Cut selection on this P uses its own basis.
    # Caps above the prompt length are equivalent for the TP vote.
    rate = getattr(peer, "_hybrid_split_suffix_rate", 0)
    return (cut, min(chunk_cap, length), suffix_cap, rate), None


def finalize_bootstrap(original, queue, req):
    result = original(queue, req)
    controller = getattr(req, "_hybrid_prefill_controller", None)
    if not result or not getattr(controller, "split_prefill", False):
        return result
    limits, reason = offered_limits(req, controller)
    min_cut, chunk_cap, suffix_cap, remote_rate = limits or (0, 0, 0, 0)
    vote = [min_cut, -min_cut, chunk_cap, -chunk_cap, suffix_cap, -suffix_cap]
    if controller.split_prefill_adaptive:
        vote.extend([remote_rate, int(controller.split_prefill_prefix_rate or 0)])
    cut, negative_cut, chunk, negative_chunk, suffix, negative_suffix, *rates = (
        controller.agree(vote)
    )
    if (
        cut > 0
        and cut == -negative_cut
        and chunk > 0
        and chunk == -negative_chunk
        and suffix == -negative_suffix
    ):
        req._hybrid_split_limits = cut, chunk, suffix
        req._hybrid_split_rates = rates if rates and all(rates) else None
    else:
        log_decision(
            controller,
            req,
            "prefill_bootstrap",
            reason or "tp_offer_disagreement",
            min_cut=min_cut,
        )
    return result


def prefill_backlog_tokens(scheduler, exclude):
    """Estimate queued P work, including the current chunked request."""
    chunked = scheduler.chunked_req

    def remaining(req):
        if req is None or req is exclude:
            return 0
        end = req.extend_end
        if end is None:
            end = req.prefix_len
        length = getattr(req, "_hybrid_split_at", 0) or len(req.origin_input_ids)
        return max(length - end, 0)

    return remaining(chunked) + sum(
        remaining(req) for req in scheduler.waiting_queue if req is not chunked
    )


def select_cut(controller, length, prefix, limits, rates=None):
    min_cut, chunk_cap, offered_suffix_cap = limits
    floor = min_suffix_tokens(length)
    ratio = controller.split_prefill_cut_ratio
    pinned_suffix = controller.split_prefill_suffix_tokens
    suffix_cap = length
    if pinned_suffix:
        cut = length - pinned_suffix
    elif controller.split_prefill_cut_basis == "whole":
        cut = max(int(length * ratio), prefix + PREFIX_GROWTH_TOKENS)
    elif controller.split_prefill_cut_basis == "balanced":
        cut = min((length + prefix + COLLISION_BUDGET_TOKENS) // 2, length - floor)
    else:
        miss = max(length - prefix, 0)
        if rates is not None:
            remote_rate, local_rate = rates
            suffix = miss * remote_rate // (remote_rate + local_rate)
        else:
            suffix = int(miss * (1 - ratio))
        suffix_cap = controller.split_prefill_suffix_cap
        if suffix_cap is None:
            suffix_cap = offered_suffix_cap or DEFAULT_SUFFIX_CAP_TOKENS
        elif offered_suffix_cap:
            suffix_cap = min(suffix_cap, offered_suffix_cap)
        if miss <= chunk_cap:
            suffix_cap = min(suffix_cap, chunk_cap, length * 15 // 100)
        if controller.split_prefill_adaptive:
            # Leave at least one uncached token for P and honor D's offer.
            suffix_cap = min(suffix_cap, length - min_cut, miss - 1)
            if suffix_cap < MIN_SUFFIX_TOKENS:
                return 0, "split_too_small"
            floor = MIN_SUFFIX_TOKENS
        cut = length - min(max(suffix, floor), suffix_cap)
    earliest = max(prefix + 1, min_cut, length - suffix_cap)
    latest = length - floor
    page = controller.split_prefill_page_size
    if page > 1:
        earliest = (earliest + page - 1) // page * page
        latest = latest // page * page
        if (
            earliest > latest
            and controller.split_prefill_adaptive
            and not pinned_suffix
            and controller.split_prefill_cut_basis == "miss"
        ):
            return 0, "split_too_small"
        cut = min((cut + page - 1) // page * page, latest)
        if not pinned_suffix:
            cut = max(cut, earliest)
    if earliest <= cut <= latest:
        return cut, "selected"
    return 0, "no_legal_cut"


def init_input(original, req, tree_cache=None, cow_mamba=None):
    original(req, tree_cache, cow_mamba)
    controller = getattr(req, "_hybrid_prefill_controller", None)
    if controller is None:
        return
    cut = getattr(req, "_hybrid_split_at", 0)
    limits = getattr(req, "_hybrid_split_limits", None)
    if (
        limits is not None
        and tree_cache is not None
        and cow_mamba is not False
        and not req.kv.holds_kv
        and not req.retracted_stain
        and not hasattr(req, "_hybrid_prefill_cached_tokens")
    ):
        length = len(req.origin_input_ids)
        prefix = max(req.prefix_len, req.kv.cache_protected_len)
        gate = controller.split_prefill_min_queue_tokens
        if gate and prefill_backlog_tokens(controller.scheduler, req) < gate:
            selected, reason = 0, "p_backlog_below_threshold"
        else:
            rates = getattr(req, "_hybrid_split_rates", None)
            selected, reason = select_cut(controller, length, prefix, limits, rates)
        agreed, negative_max = controller.agree([selected, -selected])
        accepted = agreed > 0 and agreed == -negative_max
        cut = req._hybrid_split_at = agreed if accepted else 0
        if not accepted and selected:
            reason = "tp_cut_disagreement"
        req._hybrid_split_match_reason = reason
    if cut:
        req.full_untruncated_fill_ids = req.full_untruncated_fill_ids[:cut]


def seal(req, end_idx):
    cut = getattr(req, "_hybrid_split_at", 0)
    if cut:
        sender = req.disagg_kv_sender
        controller = req._hybrid_prefill_controller
        page = controller.split_prefill_page_size
        count = cut // page
        if (
            req.extend_end != cut
            or end_idx not in (None, cut)
            or cut % page
            or sender.curr_idx > count
        ):
            raise RuntimeError("hybrid split prefill frontier changed before send")
        # Native sender derives its final envelope from curr_idx == total.
        # D registered the entire row; native Mooncake writes the prefix slice.
        sender.num_kv_indices = count
        if page > 1:
            from . import dsv4_split_prefill

            dsv4_split_prefill.seal(controller, req, cut)


def set_metadata(original, buffers, req):
    cut = getattr(req, "_hybrid_split_at", 0)
    if cut:
        buffers._hybrid_split_used = True
    if getattr(buffers, "_hybrid_split_used", False):
        buffers.output_ids[req.metadata_buffer_index, CUT_SLOT] = cut
    return original(buffers, req)


def clear_metadata(original, buffers, idx):
    buffers.output_ids[idx, CUT_SLOT] = 0
    return original(buffers, idx)


def commit(queue, dr, *, keep_receiver):
    receiver = dr.kv_receiver
    limits = getattr(receiver, "_hybrid_split_limits", None)
    if limits is None:
        return False
    req = dr.req
    if keep_receiver:
        release_reservation(req)
        return False
    idx = dr.metadata_buffer_index
    cut = int(queue.metadata_buffers.output_ids[idx, CUT_SLOT].item())
    controller = queue.scheduler._pre_layerwise
    length = len(req.origin_input_ids)
    min_cut, _ = limits
    page = controller.split_prefill_page_size
    paged = getattr(receiver, "_dsv4_split_window", None) is not None
    valid_cut = min_cut <= cut <= length - min_suffix_tokens(length) and cut % page == 0
    split, ordinary, agreed, negative_max = controller.agree(
        [int(valid_cut), int(cut == 0), cut, -cut]
    )
    if ordinary:
        if paged:
            from .dsv4_split_prefill import trim_swa

            trim_swa(controller, req, 0)
        release_reservation(req)
        return False
    room = int(queue.metadata_buffers.bootstrap_room[idx, 0].item())
    valid = (
        split
        and agreed == -negative_max
        and room == req.bootstrap_room
        and req.kv.holds_kv
        and (
            not getattr(controller, "hybrid_gdn", False)
            or (req.kv.holds_mamba and req.kv.mamba_cow_src_index is None)
        )
        and req.kv.kv_allocated_len == length
        and not req.kv.cache_protected_len
    )
    receiver.clear()
    dr.kv_receiver = None
    if not controller.agree([int(valid)])[0]:
        from sglang.srt.disaggregation.utils import prepare_abort

        release_reservation(req)
        prepare_abort(
            req,
            "Invalid hybrid split-prefill transfer",
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        return True
    controller.pd_visibility.receive()
    if paged:
        from .dsv4_split_prefill import trim_swa

        trim_swa(controller, req, cut)
    req.kv.truncate_committed(cut)
    req.kv.reserved_prefill_suffix = True
    req.kv.mamba_needs_clear = False
    req.prefix_len = cut
    req.already_computed = cut
    req._cache_breakdown_computed = True
    req.cached_tokens = int(queue.metadata_buffers.cached_tokens[idx, 0].item())
    req.cached_tokens_device = int(queue.metadata_buffers.cached_tokens[idx, 1].item())
    req._hybrid_split_defer_ticks = 0
    req._hybrid_split_ready_at = None
    req._hybrid_split_schedule = getattr(receiver, "_hybrid_split_schedule", None)
    req.time_stats.set_wait_queue_entry_time()
    return True


def time_suffix_forward(original, worker, batch, *args, **kwargs):
    controller = getattr(worker.model_runner, "_pre_layerwise", None)
    adaptive = getattr(controller, "split_prefill_adaptive", False)
    if batch is None or not (
        adaptive or getattr(controller, "split_prefill_auto_cap", 0)
    ):
        return original(worker, batch, *args, **kwargs)
    phase = None
    if getattr(batch, "_hybrid_split_extend", False) and batch.forward_mode.is_extend():
        phase = "suffix"
    elif adaptive:
        budget = controller.split_prefill_decode_budget
        if budget is None and batch.forward_mode.is_extend():
            phase = "prefix"
        elif budget is not None and budget.target and batch.forward_mode.is_decode():
            phase = "decode"
    if phase is None or (phase != "prefix" and controller.rank != 0):
        return original(worker, batch, *args, **kwargs)

    import torch

    # The scheduler has already entered the forward stream in overlap mode.
    # Keep the events with this result, even when another forward launches
    # before the scheduler processes it.
    tokens = batch.batch_size() if phase == "decode" else batch.extend_num_tokens
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream()
    start.record(stream)
    result = original(worker, batch, *args, **kwargs)
    end.record(stream)
    result._split_suffix_timing = (tokens, start, end, phase)
    return result


def record_completed_suffix(controller, result):
    timing = getattr(result, "_split_suffix_timing", None)
    if timing is None:
        return
    del result._split_suffix_timing
    tokens, start, end, phase = timing
    # The native result path has consumed the forward and its output copy.
    # Skip any incomplete sample without adding a CUDA synchronization.
    if tokens <= 0 or not end.query():
        return
    elapsed_ms = start.elapsed_time(end)
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        return
    rate = tokens * 1000 / elapsed_ms
    if controller.split_prefill_adaptive:
        if phase == "prefix":
            controller.split_prefill_prefix_rate = ema(
                controller.split_prefill_prefix_rate, min(rate, RATE_LIMIT)
            )
        else:
            controller.split_prefill_decode_budget.observe(
                tokens, elapsed_ms, suffix=phase == "suffix", rank=controller.rank
            )
        return
    previous = controller.split_prefill_suffix_rate
    if previous is not None:
        rate = AUTO_SUFFIX_RATE_EMA * previous + (1 - AUTO_SUFFIX_RATE_EMA) * rate
    controller.split_prefill_suffix_rate = rate
    budget = controller.split_prefill_suffix_budget_s
    controller.split_prefill_auto_cap = quantize_suffix_cap(budget * rate)
    controller.split_prefill_rate_samples += 1
    if controller.rank == 0 and controller.split_prefill_rate_samples % 50 == 0:
        logger.info(
            "[handoff-split-auto] samples=%s rate=%.0f tok/s budget=%.2fs local_cap=%s",
            controller.split_prefill_rate_samples,
            rate,
            budget,
            controller.split_prefill_auto_cap,
        )


def next_batch(original, scheduler, running_batch):
    controller = getattr(scheduler, "_pre_layerwise", None)
    if not getattr(controller, "split_prefill", False):
        return original(scheduler, running_batch)

    last = scheduler.last_batch
    last_was_split = getattr(last, "_hybrid_split_extend", False)
    chunked = scheduler.chunked_req
    if chunked is not None:
        if chunked.inflight_steps:
            # The overlap loop consumes the suffix result after submitting
            # the independent decode batch on the native forward stream.
            return original(scheduler, running_batch)
        scheduler.process_pending_chunked_abort()
        scheduler.chunked_req = None
        if not chunked.finished():
            chunked.prefix_len = chunked.kv.kv_committed_len
            scheduler.waiting_queue.prepend([chunked])
        if last_was_split:
            last._hybrid_split_extend = False
            if not running_batch.is_empty():
                return original(scheduler, running_batch)
    elif last_was_split:
        # A promoted batch may become running_batch itself. Consume the
        # marker before native Decode reuses that object on subsequent ticks.
        last._hybrid_split_extend = False
        if last is not running_batch and last.forward_mode.is_extend():
            for req in last.reqs:
                release_reservation(req)
            last.filter_batch()
            if not last.is_empty():
                scheduler.prefill_launched_reqs.leave(last.reqs)
                scheduler.running_reqs.join(last.reqs)
                if running_batch.is_empty():
                    running_batch = last
                else:
                    running_batch.merge_batch(last)
            return original(scheduler, running_batch)

    req = next(
        (r for r in scheduler.waiting_queue if r.kv.reserved_prefill_suffix),
        None,
    )
    if req is None:
        return original(scheduler, running_batch)
    running_bs = running_batch.batch_size()
    if running_bs >= scheduler.max_running_requests:
        return original(scheduler, running_batch)

    prompt_len = len(req.origin_input_ids)
    remaining = prompt_len - req.prefix_len
    schedule = getattr(req, "_hybrid_split_schedule", None)
    if schedule is None:
        chunk_size = controller.split_prefill_chunk_size
        defer_budget = controller.split_prefill_defer_ticks
        patience = controller.split_prefill_patience_ticks
        if req.cached_tokens * 10 >= prompt_len * 7:
            defer_budget = min(defer_budget, HOT_DEFER_TICKS)
    else:
        chunk_size, defer_budget, patience, rate, step_us = schedule
        if rate and step_us:
            # Limit the wait to the estimated forward time of the next chunk.
            # Both inputs are frozen by the TP vote at admission.
            chunk_steps = min(chunk_size, remaining) * 1_000_000 // (rate * step_us)
            defer_budget = min(defer_budget, chunk_steps)
    ticks = req._hybrid_split_defer_ticks
    if ticks == 0:
        req._hybrid_split_defer_start_bs = running_bs
    # A stable or growing decode batch is unlikely to leave an empty slot
    # before the full budget expires. Use rank-consistent scheduler state.
    stalled = (
        patience > 0
        and ticks >= patience
        and running_bs >= req._hybrid_split_defer_start_bs
    )
    if running_bs and ticks < defer_budget and not stalled:
        # All TP ranks schedule the same requests. Count decisions and use
        # rank-local time only for logging.
        if controller.rank == 0 and ticks == 0:
            req._hybrid_split_ready_at = time.monotonic()
            logger.info(
                "[handoff-split-defer] rid=%s running_bs=%s budget=%s",
                req.rid,
                running_bs,
                defer_budget,
            )
        req._hybrid_split_defer_ticks = ticks + 1
        return original(scheduler, running_batch)
    req._hybrid_split_defer_ticks = 0
    ready_at = req._hybrid_split_ready_at
    req._hybrid_split_ready_at = None

    from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
    from sglang.srt.observability.req_time_stats import set_schedule_time_batch

    scheduler.waiting_queue.leave([req])
    req.refresh_fill_ids()
    if getattr(controller, "dsv4_split_schema", None):
        from .dsv4_split_prefill import trim_swa

        trim_swa(controller, req, req.prefix_len)
    req.extend_end = req.prefix_len + min(chunk_size, remaining)
    middle = req.extend_end < prompt_len
    scheduler.chunked_req = req if middle else None
    req.time_stats.set_forward_entry_time()
    batch = ScheduleBatch.init_new(
        [req],
        scheduler.req_to_token_pool,
        scheduler.token_to_kv_pool_allocator,
        scheduler.tree_cache,
        scheduler.model_config,
        scheduler.enable_overlap,
        scheduler.spec_algorithm,
        chunked_req=scheduler.chunked_req,
    )
    # The native middle-chunk result path skips token publication. Also avoid
    # sampling here so intermediate forwards do not advance the request RNG.
    batch.is_prefill_only = middle
    first_chunk = req.extend_batch_idx == 0
    batch.prepare_for_extend(reuse_preallocated_kv=True)
    batch._hybrid_split_extend = True
    set_schedule_time_batch(batch)
    if first_chunk:
        logger.info(
            "[handoff-split] hybrid PD rid=%s cut=%s suffix=%s",
            req.rid,
            req.prefix_len,
            prompt_len - req.prefix_len,
        )
    if controller.rank == 0:
        logger.info(
            "[handoff-split-chunk] hybrid PD rid=%s prefix=%s suffix_chunk=%s end=%s "
            "remaining=%s running_bs=%s defer_ticks=%s waited_ms=%.0f",
            req.rid,
            req.prefix_len,
            req.extend_len,
            req.extend_end,
            prompt_len - req.extend_end,
            running_bs,
            ticks,
            (time.monotonic() - ready_at) * 1000 if ready_at is not None else 0,
        )
    return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)


def prebuilt(original, scheduler, running_batch):
    if not getattr(getattr(scheduler, "_pre_layerwise", None), "split_prefill", False):
        return original(scheduler, running_batch)
    if scheduler.chunked_req is not None:
        # Native PREBUILT admission requires chunked_req to be clear. While
        # the previous chunk is in flight, only the running decode batch runs.
        return None
    # Native prebuilt admission assumes a sampled token. Keep split requests on
    # the real waiting list for abort/accounting, but exclude them from PREBUILT.
    pending = scheduler.waiting_queue.take(lambda req: req.kv.reserved_prefill_suffix)
    try:
        return original(scheduler, running_batch)
    finally:
        scheduler.waiting_queue.join(pending)
