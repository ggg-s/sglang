"""Cache-aware split PD using native Mooncake completion and local KV."""

import logging
import os
from http import HTTPStatus

logger = logging.getLogger(__name__)
# V3 advertises the cut bound and D's chunk cap. V2 assumes a fixed suffix floor.
WIRE = b"REMOTE_ATTN_HYBRID_SPLIT_V3:"
# output_ids rows are already transferred in full; native uses only slot 0.
CUT_SLOT = 1
MIN_PROMPT_TOKENS = 4096
MIN_SUFFIX_TOKENS = 2048
SHORT_PROMPT_TOKENS = 32768
BALANCED_CUT_RATIO = 0.75
PREFIX_GROWTH_TOKENS = 1024


def min_suffix_tokens(length):
    return min(MIN_SUFFIX_TOKENS, max(length // 16, 1))


def configure(controller):
    value = os.environ.get("HANDOFF_HYBRID_SPLIT_PREFILL", "1")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_HYBRID_SPLIT_PREFILL must be 0 or 1")
    # Dense MHA and native FULL+MAMBA have a continuation adapter. Compressed
    # DSV KV/state layouts retain native PD until their adapter supports cuts.
    controller.split_prefill = (
        value == "1"
        and controller.config.mode == "remote"
        and (
            getattr(controller, "hybrid_gdn", False)
            or not getattr(controller, "dsv41_layerwise", False)
        )
    )
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
    controller.split_prefill_budget = None
    controller.split_prefill_chunk_size = None
    controller.split_prefill_cut_ratio = None
    controller.split_prefill_cut_basis = None
    controller.split_prefill_suffix_tokens = None
    controller.split_prefill_max_outstanding = None
    if controller.split_prefill and role == "prefill":
        cut_env = "HANDOFF_HYBRID_SPLIT_PREFILL_CUT_RATIO"
        try:
            cut_ratio = float(os.environ.get(cut_env, str(BALANCED_CUT_RATIO)))
        except ValueError as exc:
            raise ValueError(f"{cut_env} must be in [0.5, 1)") from exc
        if not 0.5 <= cut_ratio < 1:
            raise ValueError(f"{cut_env} must be in [0.5, 1)")
        basis = os.environ.get("HANDOFF_SPLIT_CUT_BASIS", "miss")
        if basis not in ("miss", "whole"):
            raise ValueError("HANDOFF_SPLIT_CUT_BASIS must be miss or whole")
        suffix_env = "HANDOFF_SPLIT_SUFFIX_TOKENS"
        try:
            suffix_tokens = int(os.environ.get(suffix_env, "0"))
        except ValueError as exc:
            raise ValueError(f"{suffix_env} must be a nonnegative integer") from exc
        if suffix_tokens < 0:
            raise ValueError(f"{suffix_env} must be a nonnegative integer")
        controller.split_prefill_cut_ratio = cut_ratio
        controller.split_prefill_cut_basis = basis
        controller.split_prefill_suffix_tokens = suffix_tokens
    if controller.split_prefill and role == "decode":
        chunk_env = "HANDOFF_HYBRID_SPLIT_PREFILL_CHUNK_SIZE"
        try:
            chunk_size = int(os.environ.get(chunk_env, "4096"))
        except ValueError as exc:
            raise ValueError(f"{chunk_env} must be a positive integer") from exc
        if chunk_size <= 0:
            raise ValueError(f"{chunk_env} must be a positive integer")
        limit_env = "HANDOFF_HYBRID_SPLIT_PREFILL_MAX_OUTSTANDING"
        try:
            limit = int(os.environ.get(limit_env, "4"))
        except ValueError as exc:
            raise ValueError(f"{limit_env} must be a positive integer") from exc
        if limit <= 0:
            raise ValueError(f"{limit_env} must be a positive integer")
        controller.split_prefill_max_outstanding = limit
        controller.split_prefill_outstanding = 0
        controller.split_prefill_budget = scheduler.max_prefill_tokens
        controller.split_prefill_chunk_size = min(
            chunk_size, controller.split_prefill_budget
        )
        if not getattr(controller, "hybrid_gdn", False):
            from .remote_data_plane import Visibility

            controller.pd_visibility = Visibility(
                scheduler.req_to_token_pool.req_to_token.device
            )
    if controller.rank == 0:
        logger.info(
            "[handoff-split-config] role=%s enabled=%s max_prefill_tokens=%s "
            "chunk_size=%s cut_ratio=%s cut_basis=%s suffix_tokens=%s "
            "max_outstanding=%s",
            role,
            controller.split_prefill,
            controller.split_prefill_budget,
            controller.split_prefill_chunk_size,
            controller.split_prefill_cut_ratio,
            controller.split_prefill_cut_basis,
            controller.split_prefill_suffix_tokens,
            controller.split_prefill_max_outstanding,
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


def advertise(queue, dr, decode_prefix_len):
    scheduler = queue.scheduler
    controller = getattr(scheduler, "_pre_layerwise", None)
    if not getattr(controller, "split_prefill", False):
        return
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
    # D commits metadata collectively, so every TP rank must publish the same
    # admission decision even if its local capacity check differs.
    allowed = length >= MIN_PROMPT_TOKENS and controller.agree([int(reason is None)])[0]
    if allowed:
        controller.split_prefill_outstanding += 1
        req._hybrid_split_reservation = controller
        receiver._hybrid_split_limits = (
            min_cut,
            max(controller.split_prefill_chunk_size, MIN_SUFFIX_TOKENS),
        )
    log_decision(
        controller,
        req,
        "decode_offer",
        "offered" if allowed else reason or "decode_peer_rejected",
        min_cut=min_cut if allowed else 0,
        outstanding=controller.split_prefill_outstanding,
        chunk_size=controller.split_prefill_chunk_size,
    )


def release_reservation(req):
    controller = getattr(req, "_hybrid_split_reservation", None)
    if controller is not None:
        req._hybrid_split_reservation = None
        controller.split_prefill_outstanding -= 1


def offered_limits(req, tp_size):
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
    if (
        not (length + 1) // 2 <= cut <= length - min_suffix_tokens(length)
        or chunk_cap < MIN_SUFFIX_TOKENS
    ):
        return None, "invalid_decode_split_limits"
    if len(peer.dst_kv_indices) != length:
        return None, "destination_kv_length_mismatch"
    registration = manager.decode_kv_args_table.get(peer.mooncake_session_id)
    if (
        registration is None
        or registration.requires_dcp_relayout
        or registration.dst_attn_tp_size != tp_size
    ):
        return None, "incompatible_destination_registration"
    # Caps above the prompt length are equivalent and need not enter the TP vote.
    return (cut, min(chunk_cap, length)), None


def finalize_bootstrap(original, queue, req):
    result = original(queue, req)
    controller = getattr(req, "_hybrid_prefill_controller", None)
    if not result or not getattr(controller, "split_prefill", False):
        return result
    limits, reason = offered_limits(req, controller.tp_size)
    min_cut, chunk_cap = limits or (0, 0)
    cut, negative_cut, cap, negative_cap = controller.agree(
        [min_cut, -min_cut, chunk_cap, -chunk_cap]
    )
    if cut > 0 and cut == -negative_cut and cap > 0 and cap == -negative_cap:
        req._hybrid_split_limits = cut, cap
    else:
        log_decision(
            controller,
            req,
            "prefill_bootstrap",
            reason or "tp_offer_disagreement",
            min_cut=min_cut,
        )
    return result


def select_cut(controller, length, prefix, limits):
    min_cut, chunk_cap = limits
    floor = min_suffix_tokens(length)
    ratio = controller.split_prefill_cut_ratio
    pinned_suffix = controller.split_prefill_suffix_tokens
    if pinned_suffix:
        cut = length - pinned_suffix
    elif controller.split_prefill_cut_basis == "whole":
        cut = max(int(length * ratio), prefix + PREFIX_GROWTH_TOKENS)
    else:
        miss = max(length - prefix, 0)
        suffix = max(int(miss * (1 - ratio)), floor)
        if prefix or length < SHORT_PROMPT_TOKENS:
            suffix = min(suffix, chunk_cap, length * 15 // 100)
        cut = length - suffix
    if prefix < cut and min_cut <= cut <= length - floor:
        return cut
    return 0


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
        selected = select_cut(controller, length, prefix, limits)
        agreed, negative_max = controller.agree([selected, -selected])
        accepted = agreed > 0 and agreed == -negative_max
        cut = req._hybrid_split_at = agreed if accepted else 0
        if accepted:
            reason = "selected"
        elif selected:
            reason = "tp_cut_disagreement"
        else:
            reason = "no_legal_cut"
        req._hybrid_split_match_reason = reason
    if cut:
        req.full_untruncated_fill_ids = req.full_untruncated_fill_ids[:cut]


def seal(req, end_idx):
    cut = getattr(req, "_hybrid_split_at", 0)
    if cut:
        sender = req.disagg_kv_sender
        if req.extend_end != cut or end_idx not in (None, cut) or sender.curr_idx > cut:
            raise RuntimeError("hybrid split prefill frontier changed before send")
        # Native sender derives its final envelope from curr_idx == total.
        # D registered the entire row; native Mooncake writes the prefix slice.
        sender.num_kv_indices = cut


def set_metadata(original, buffers, req):
    cut = getattr(req, "_hybrid_split_at", 0)
    if cut:
        buffers._hybrid_split_used = True
    if getattr(buffers, "_hybrid_split_used", False):
        buffers.output_ids[req.metadata_buffer_index, CUT_SLOT] = cut
    return original(buffers, req)


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
    valid_cut = min_cut <= cut <= length - min_suffix_tokens(length)
    split, ordinary, agreed, negative_max = controller.agree(
        [int(valid_cut), int(cut == 0), cut, -cut]
    )
    if ordinary:
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
    req.kv.truncate_committed(cut)
    req.kv.reserved_prefill_suffix = True
    req.kv.mamba_needs_clear = False
    req.prefix_len = cut
    req.already_computed = cut
    req._cache_breakdown_computed = True
    req.cached_tokens = int(queue.metadata_buffers.cached_tokens[idx, 0].item())
    req.cached_tokens_device = int(queue.metadata_buffers.cached_tokens[idx, 1].item())
    req.time_stats.set_wait_queue_entry_time()
    return True


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
    if req is None or running_batch.batch_size() >= scheduler.max_running_requests:
        return original(scheduler, running_batch)

    from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
    from sglang.srt.observability.req_time_stats import set_schedule_time_batch

    scheduler.waiting_queue.leave([req])
    req.refresh_fill_ids()
    prompt_len = len(req.origin_input_ids)
    chunk_size = controller.split_prefill_chunk_size
    req.extend_end = min(req.prefix_len + chunk_size, prompt_len)
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
            "remaining=%s",
            req.rid,
            req.prefix_len,
            req.extend_len,
            req.extend_end,
            prompt_len - req.extend_end,
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
