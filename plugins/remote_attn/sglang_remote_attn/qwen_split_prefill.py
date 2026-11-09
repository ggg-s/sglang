"""Opt-in token-axis Hybrid PD using native Mooncake completion and local KV."""

import logging
import os
from http import HTTPStatus

logger = logging.getLogger(__name__)
WIRE = b"REMOTE_ATTN_HYBRID_SPLIT_V1:"
# output_ids rows are already transferred in full; native uses only slot 0.
CUT_SLOT = 1
MIN_PROMPT_TOKENS = 4096


def configure(controller):
    value = os.environ.get("HANDOFF_HYBRID_SPLIT_PREFILL", "0")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_HYBRID_SPLIT_PREFILL must be 0 or 1")
    if value == "1":
        from sglang.srt.runtime_context import get_disagg

        disagg = get_disagg()
        if (
            disagg.disaggregation_decode_allocation_policy == "prefill_complete"
            or disagg.disaggregation_decode_host_receive_threshold != 1
            or disagg.disaggregation_decode_enable_offload_kvcache
        ):
            raise ValueError("hybrid split prefill requires early device KV allocation")
    controller.split_prefill = value == "1"
    controller.split_prefill_budget = (
        controller.scheduler.max_prefill_tokens if controller.split_prefill else None
    )
    if controller.rank == 0:
        logger.info(
            "[handoff-split-config] role=%s enabled=%s max_prefill_tokens=%s",
            controller.scheduler.disaggregation_mode.value,
            controller.split_prefill,
            controller.split_prefill_budget,
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
    if created and getattr(controller, "hybrid_gdn", False):
        req._hybrid_prefill_controller = controller
    return created


def can_prefill_before_bootstrap(original, queue, req):
    controller = getattr(req, "_hybrid_prefill_controller", None)
    if getattr(controller, "split_prefill", False) and request_blocker(req) is None:
        # An optimistic forward can pass the cut before D publishes its offer.
        return False
    return original(queue, req)


def advertise(queue, dr, decode_prefix_len):
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if not getattr(controller, "split_prefill", False):
        return
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver

    receiver, req = dr.kv_receiver, dr.req
    if getattr(receiver, "_hybrid_split_offer_checked", False):
        return
    receiver._hybrid_split_offer_checked = True
    length = len(req.origin_input_ids)
    budget = controller.split_prefill_budget
    suffix = min(length - length // 2, budget)
    reason = request_blocker(req)
    if dr.is_rebootstrap or decode_prefix_len:
        reason = "decode_already_owns_prefix"
    elif type(receiver) is not MooncakeKVReceiver:
        reason = "non_mooncake_receiver"
    elif reason is None and suffix < MIN_PROMPT_TOKENS // 2:
        reason = "decode_prefill_budget_too_small"
    cut = 0
    if reason is None:
        cut = receiver._hybrid_split_offer = length - suffix
    log_decision(
        controller, req, "decode_offer", reason or "offered", cut=cut, budget=budget
    )


def offered_cut(req, tp_size):
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender

    reason = request_blocker(req)
    if reason is not None:
        return 0, reason
    sender = req.disagg_kv_sender
    if type(sender) is not MooncakeKVSender:
        return 0, "non_mooncake_sender"
    if req.start_send_idx or req.kv.holds_kv:
        return 0, "prefill_already_started"
    manager = sender.kv_mgr
    peers = tuple(manager.transfer_infos.get(req.bootstrap_room, {}).values())
    if len(peers) != 1 or peers[0].is_dummy:
        return 0, "unexpected_destination_peers"
    peer = peers[0]
    cut = getattr(peer, "_hybrid_split_offer", 0)
    if not cut:
        return 0, "decode_did_not_offer"
    length = len(req.origin_input_ids)
    if not length // 2 <= cut <= length - MIN_PROMPT_TOKENS // 2:
        return 0, "invalid_decode_cut"
    if len(peer.dst_kv_indices) != length:
        return 0, "destination_kv_length_mismatch"
    registration = manager.decode_kv_args_table.get(peer.mooncake_session_id)
    if (
        registration is None
        or registration.requires_dcp_relayout
        or registration.dst_attn_tp_size != tp_size
    ):
        return 0, "incompatible_destination_registration"
    return cut, None


def finalize_bootstrap(original, queue, req):
    result = original(queue, req)
    controller = getattr(req, "_hybrid_prefill_controller", None)
    if not result or not getattr(controller, "split_prefill", False):
        return result
    cut, reason = offered_cut(req, controller.tp_size)
    agreed, negative_max = controller.agree([cut, -cut])
    if agreed > 0 and agreed == -negative_max:
        req._hybrid_split_proposal = agreed
    else:
        log_decision(
            controller,
            req,
            "prefill_bootstrap",
            reason or "tp_offer_disagreement",
            cut=cut,
        )
    return result


def init_input(original, req, tree_cache=None, cow_mamba=None):
    original(req, tree_cache, cow_mamba)
    controller = getattr(req, "_hybrid_prefill_controller", None)
    if controller is None:
        return
    cut = getattr(req, "_hybrid_split_at", 0)
    proposal = getattr(req, "_hybrid_split_proposal", 0)
    if (
        tree_cache is not None
        and cow_mamba is not False
        and not req.kv.holds_kv
        and not req.retracted_stain
        and not hasattr(req, "_hybrid_prefill_cached_tokens")
    ):
        # Native lookup may run repeatedly while the request is queued. Only
        # the match used by the first actual forward becomes the snapshot.
        req._hybrid_prefill_cache_match = req.prefix_len
        if proposal:
            hot = req.prefix_len > (
                len(req.origin_input_ids) * controller.remote_min_cache_hit_ratio
            )
            split = not hot and req.prefix_len < proposal
            accepted = controller.agree([int(split)])[0]
            cut = req._hybrid_split_at = proposal if accepted else 0
            req._hybrid_split_match_reason = (
                "selected"
                if accepted
                else (
                    "warm_cache"
                    if hot
                    else (
                        "matched_prefix_reaches_cut"
                        if not split
                        else "tp_match_disagreement"
                    )
                )
            )
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
    offered = getattr(receiver, "_hybrid_split_offer", 0)
    if not offered or keep_receiver:
        return False
    req = dr.req
    idx = dr.metadata_buffer_index
    cut = int(queue.metadata_buffers.output_ids[idx, CUT_SLOT].item())
    controller = queue.scheduler._pre_layerwise
    split, ordinary = controller.agree([int(cut == offered), int(cut == 0)])
    if ordinary:
        return False
    room = int(queue.metadata_buffers.bootstrap_room[idx, 0].item())
    length = len(req.origin_input_ids)
    valid = (
        split
        and room == req.bootstrap_room
        and req.kv.holds_kv
        and req.kv.holds_mamba
        and req.kv.mamba_cow_src_index is None
        and req.kv.kv_allocated_len == length
        and not req.kv.cache_protected_len
    )
    receiver.clear()
    dr.kv_receiver = None
    if not controller.agree([int(valid)])[0]:
        from sglang.srt.disaggregation.utils import prepare_abort

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
    if last is not None and getattr(last, "_hybrid_split_extend", False):
        last.filter_batch()
        if not last.is_empty():
            scheduler.prefill_launched_reqs.leave(last.reqs)
            scheduler.running_reqs.join(last.reqs)
            if running_batch.is_empty():
                running_batch = last
            else:
                running_batch.merge_batch(last)
        # Give existing decodes a turn between suffix batches.
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
    req.extend_end = len(req.origin_input_ids)
    req.time_stats.set_forward_entry_time()
    batch = ScheduleBatch.init_new(
        [req],
        scheduler.req_to_token_pool,
        scheduler.token_to_kv_pool_allocator,
        scheduler.tree_cache,
        scheduler.model_config,
        scheduler.enable_overlap,
        scheduler.spec_algorithm,
    )
    batch.prepare_for_extend(reuse_preallocated_kv=True)
    batch._hybrid_split_extend = True
    set_schedule_time_batch(batch)
    logger.info(
        "[handoff-split] hybrid PD rid=%s cut=%s suffix=%s",
        req.rid,
        req.prefix_len,
        req.extend_len,
    )
    return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)


def prebuilt(original, scheduler, running_batch):
    if not getattr(getattr(scheduler, "_pre_layerwise", None), "split_prefill", False):
        return original(scheduler, running_batch)
    # Native prebuilt admission assumes a sampled token. Keep split requests on
    # the real waiting list for abort/accounting, but exclude them from PREBUILT.
    pending = scheduler.waiting_queue.take(lambda req: req.kv.reserved_prefill_suffix)
    try:
        return original(scheduler, running_batch)
    finally:
        scheduler.waiting_queue.join(pending)
