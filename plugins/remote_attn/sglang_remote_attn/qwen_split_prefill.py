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


def eligible(req):
    return (
        len(req.origin_input_ids) >= MIN_PROMPT_TOKENS
        and not req.output_ids
        and not req.is_retracted
        and not req.return_logprob
        and not req.return_hidden_states
        and not req.return_sampling_mask
        and not req.return_routed_experts
        and not req.return_indexer_topk
        and req.custom_logit_processor is None
        and req.sampling_params.max_new_tokens > 0
        and req.session is None
        and req.grammar is None
        and req.beam_group is None
        and req.multimodal_inputs is None
        and req.input_embeds is None
        and req.positional_embed_overrides is None
    )


def advertise(queue, dr, decode_prefix_len):
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if not getattr(controller, "split_prefill", False):
        return
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver

    receiver, req = dr.kv_receiver, dr.req
    if (
        type(receiver) is MooncakeKVReceiver
        and not dr.is_rebootstrap
        and not decode_prefix_len
        and eligible(req)
    ):
        length = len(req.origin_input_ids)
        cut = length // 2
        if length - cut <= queue.scheduler.max_prefill_tokens:
            receiver._hybrid_split_offer = cut


def finalize_bootstrap(original, queue, req):
    result = original(queue, req)
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if not result or not getattr(controller, "split_prefill", False):
        return result
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender

    sender = req.disagg_kv_sender
    cut = len(req.origin_input_ids) // 2
    offered = False
    if (
        type(sender) is MooncakeKVSender
        and not req.start_send_idx
        and not req.kv.holds_kv
        and eligible(req)
    ):
        manager = sender.kv_mgr
        peers = tuple(manager.transfer_infos.get(req.bootstrap_room, {}).values())
        if len(peers) == 1:
            peer = peers[0]
            registration = manager.decode_kv_args_table.get(peer.mooncake_session_id)
            offered = (
                getattr(peer, "_hybrid_split_offer", 0) == cut
                and not peer.is_dummy
                and len(peer.dst_kv_indices) == len(req.origin_input_ids)
                and registration is not None
                and not registration.requires_dcp_relayout
                and registration.dst_attn_tp_size == controller.tp_size
            )
    if controller.agree([int(offered)])[0]:
        req._hybrid_split_proposal = cut
        req._hybrid_split_hot_ratio = controller.remote_min_cache_hit_ratio
    return result


def init_input(original, req, tree_cache=None, cow_mamba=None):
    original(req, tree_cache, cow_mamba)
    cut = getattr(req, "_hybrid_split_at", 0)
    proposal = getattr(req, "_hybrid_split_proposal", 0)
    if (
        proposal
        and tree_cache is not None
        and cow_mamba is not False
        and not req.kv.holds_kv
    ):
        # Let native matching and CoW see the full prompt. Preserve the existing
        # warm-cache Remote path and always leave at least one token to run on P.
        if (
            req.prefix_len < proposal
            and req.prefix_len
            <= len(req.origin_input_ids) * req._hybrid_split_hot_ratio
        ):
            cut = req._hybrid_split_at = proposal
        else:
            cut = req._hybrid_split_at = 0
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
