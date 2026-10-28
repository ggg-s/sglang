"""Function-boundary hooks; no copies of SGLang scheduler loops."""

import inspect
import logging
import pkgutil
import threading
import time

logger = logging.getLogger(__name__)


def _runtime():
    from . import handoff_control

    return handoff_control


def _scheduler_init(original, scheduler, *args, **kwargs):
    from sglang.srt.plugins.hook_registry import HookRegistry

    missing = set(HOOKS) - HookRegistry._patched
    if missing:
        raise RuntimeError(f"remote_attn hooks were not applied: {sorted(missing)}")
    validate_config()
    original(scheduler, *args, **kwargs)
    # The transferred snapshots contain dense, per-layer K/V only. Reject
    # incompatible cache layouts before publishing ready.<rank>.json.
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    import torch

    if (
        type(pool) is not MHATokenToKVPool
        or pool.kv_cache_layout != "nhd"
        or pool.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise RuntimeError(
            "remote_attn currently requires dense NHD MHA KV in BF16/FP16"
        )
    _runtime().get_remote_hub(scheduler)


def validate_config():
    from mooncake.engine import TransferEngine

    from sglang.srt.runtime_context import (
        get_device,
        get_disagg,
        get_exec,
        get_memory,
        get_parallel,
        get_schedule,
        get_spec,
    )

    parallel = get_parallel()
    memory = get_memory()
    schedule = get_schedule()
    if get_device().device != "cuda":
        raise RuntimeError("remote_attn requires CUDA")
    if (
        parallel.pp_size != 1
        or parallel.dp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.dcp_enabled
        or parallel.attn_dp_enabled
    ):
        raise RuntimeError("remote_attn supports plain TP, without PP/DP/CP/DCP")
    if (
        not schedule.disable_overlap_schedule
        or get_exec().overlap.enable_two_batch_overlap
    ):
        raise RuntimeError("remote_attn freeze requires --disable-overlap-schedule")
    if not memory.disable_radix_cache:
        raise RuntimeError("remote_attn requires --disable-radix-cache")
    if memory.enable_unified_memory or memory.enable_hierarchical_cache:
        raise RuntimeError("remote_attn does not yet support unified KV or HiCache")
    if schedule.page_size != 1 or schedule.enable_mixed_chunk:
        raise RuntimeError("remote_attn requires --page-size 1 without mixed chunks")
    if get_spec().speculative_algorithm is not None:
        raise RuntimeError("remote_attn does not yet support speculative decoding")
    if get_disagg().enable_pd_role_switch:
        raise RuntimeError("remote_attn does not yet support live PD role switching")
    graph = get_exec().graph.cuda_graph_config
    if parallel.tp_size > 1 and (
        graph.decode.backend != "disabled" or graph.prefill.backend != "disabled"
    ):
        raise RuntimeError(
            "remote_attn TP requires --disable-cuda-graph until replay decisions are synchronized"
        )
    if not callable(getattr(TransferEngine, "send_probe", None)):
        raise RuntimeError("remote_attn requires Mooncake TransferEngine.send_probe")


def _event_loop(original, scheduler, *args, **kwargs):
    try:
        return original(scheduler, *args, **kwargs)
    finally:
        _runtime().shutdown_handoff(scheduler)


def _broadcast(original, receiver, recv_reqs, local_reqs=None):
    from sglang.srt.runtime_context import get_parallel

    from .tp import inject_control

    parallel = get_parallel()
    leader = (
        parallel.pp_rank == 0
        and parallel.attn_tp_rank == 0
        and parallel.attn_cp_rank == 0
    )
    recv_reqs = inject_control(
        recv_reqs, leader=leader, poll=_runtime().poll_broadcast_control
    )
    return original(receiver, recv_reqs, local_reqs)


def _process_requests(original, scheduler, recv_reqs):
    runtime = _runtime()
    requests, controls = runtime.split_handoff_controls(recv_reqs)
    result = original(scheduler, requests)
    for control in controls:
        runtime.execute_handoff_control(scheduler, control)
    runtime.maybe_handle_handoff_control(scheduler)
    return result


def _init_round(original, req, *args, **kwargs):
    prefix = getattr(req, "handoff_prefix_indices", None)
    if prefix is None:
        return original(req, *args, **kwargs)
    req.refresh_fill_ids()
    req.prefix_len = len(prefix)
    req.extend_end = len(req.full_untruncated_fill_ids)
    if req.prefix_len > req.extend_end:
        raise RuntimeError("imported KV prefix exceeds request length")


def _prefix_indices(original, batch, req):
    prefix = getattr(req, "handoff_prefix_indices", None)
    if prefix is None:
        return original(batch, req)
    # Admission may shorten extend_end for chunking. The imported prefix is
    # still the full prefix, independent of the chosen extend/chunk length.
    if len(prefix) != req.prefix_len:
        raise RuntimeError("imported KV prefix length changed before allocation")
    return prefix


def _alloc_extend(original, batch):
    result = original(batch)
    runtime = _runtime()
    # Upstream write_cache_indices has now bound the supplied prefix into the
    # admitted row. Clear only after success, so a re-init cannot reuse it.
    for req in batch.reqs:
        if getattr(req, "handoff_prefix_indices", None) is not None:
            req.handoff_prefix_indices = None
            runtime.HANDOFF_PREFIX_CLEARED += 1
    return result


def _reset_retracted(original, req):
    result = original(req)
    req.handoff_prefix_indices = None
    req._handoff_prefix_indices = None
    return result


def _release_kv(original, req, tree_cache, *, checkpoint):
    runtime = _runtime()
    runtime.stop_target_receivers(req)
    prefix = getattr(req, "handoff_prefix_indices", None)
    if req.kv.req_pool_idx is None and prefix is not None:
        # Decode's waiting-queue abort path assumes native PD requests have
        # already acquired a row. A handoff import deliberately has not.
        tree_cache.token_to_kv_pool_allocator.free(prefix)
        req.handoff_prefix_indices = None
        req._handoff_prefix_indices = None
        req._handoff_prefix_freed = True
        req.kv.mark_kv_released()
        req.kv.kv_committed_len = 0
        return
    return original(req, tree_cache, checkpoint=checkpoint)


def _stream_output(original, streamer, reqs, *args, **kwargs):
    # The controller reads these requests through status/status_many. Sending
    # them to the normal detokenizer would create a second output owner.
    reqs = [req for req in reqs if not getattr(req, "_handoff_req", False)]
    return original(streamer, reqs, *args, **kwargs)


def _decode_graph(original, runner, forward_batch):
    if _runtime().has_remote_work():
        return False
    return original(runner, forward_batch)


def _prefill_graph(original, runner, forward_batch):
    if _runtime().has_target_states():
        return False
    return original(runner, forward_batch)


def _graph_execute(original, runner, forward_batch, *args, **kwargs):
    _runtime().graph_step_begin(forward_batch)
    return original(runner, forward_batch, *args, **kwargs)


def _mooncake_init(original, manager, *args, **kwargs):
    # The constructor starts sender threads, so install the mutex first.
    manager._remote_attn_send_lock = threading.Lock()
    return original(manager, *args, **kwargs)


def _transfer(original, manager, mooncake_session_id, transfer_blocks):
    from sglang.srt.environ import envs

    if not transfer_blocks or not envs.SGLANG_MOONCAKE_SEND_LOCK.get():
        return original(manager, mooncake_session_id, transfer_blocks)
    with manager._remote_attn_send_lock:
        return original(manager, mooncake_session_id, transfer_blocks)


def _publish_peer(original, manager, mooncake_session_id, decode_kv_args):
    # Coordinate registration and sends on the same process-wide mutex.
    # Preserve upstream's validation, blacklist and publication ordering.
    with manager._remote_attn_send_lock:
        old = manager.decode_kv_args_table.get(mooncake_session_id)
        changed = (
            old is None
            or old.dst_kv_ptrs != decode_kv_args.dst_kv_ptrs
            or old.dst_kv_item_len != decode_kv_args.dst_kv_item_len
        )
        if changed:
            probe = manager.engine.send_probe
            rc = -1
            for attempt in range(3):
                try:
                    rc = probe(mooncake_session_id)
                except Exception:
                    logger.exception(
                        "Mooncake pre-send probe failed for %s", mooncake_session_id
                    )
                if rc == 0:
                    break
                if attempt < 2:
                    time.sleep(0.05)
        result = original(manager, mooncake_session_id, decode_kv_args)
        if changed and rc != 0:
            # The bootstrap callback runs on a long-lived receiver thread.
            # Keep that thread alive and let upstream failed-session recovery
            # reject sends until a later probe or registration repairs it.
            with manager.session_lock:
                manager.failed_sessions.add(mooncake_session_id)
            logger.warning(
                "Mooncake pre-send probe failed: %s, rc=%s", mooncake_session_id, rc
            )
        return result


HOOKS = {
    "sglang.srt.managers.scheduler.Scheduler.__init__": _scheduler_init,
    "sglang.srt.managers.scheduler.Scheduler.run_event_loop": _event_loop,
    "sglang.srt.managers.scheduler.Scheduler.process_input_requests": _process_requests,
    "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver._broadcast_reqs_across_ranks": _broadcast,
    "sglang.srt.managers.schedule_batch.Req.init_next_round_input": _init_round,
    "sglang.srt.managers.schedule_batch.Req.reset_for_retract": _reset_retracted,
    "sglang.srt.mem_cache.allocation._prefix_kv_indices": _prefix_indices,
    "sglang.srt.mem_cache.allocation.alloc_for_extend": _alloc_extend,
    "sglang.srt.mem_cache.common.release_kv_cache": _release_kv,
    "sglang.srt.managers.scheduler_components.output_streamer.SchedulerOutputStreamer.stream_output": _stream_output,
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.can_run_graph": _decode_graph,
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.execute": _graph_execute,
    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner.PrefillCudaGraphRunner.can_run_graph": _prefill_graph,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.__init__": _mooncake_init,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._transfer_data": _transfer,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._publish_peer_registration": _publish_peer,
}


def register_hooks():
    from sglang.srt.environ import envs
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # Resolve and signature-check the whole set before changing any target.
    # SGLang logs plugin failures, so the Scheduler init hook also verifies
    # that the loader actually applied every registration before serving.
    envs.SGLANG_MOONCAKE_SEND_LOCK.get()
    for target, hook in HOOKS.items():
        function = pkgutil.resolve_name(target)
        positional = []
        keywords = {}
        for param in inspect.signature(function).parameters.values():
            if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                positional.append(None)
            elif param.kind == param.KEYWORD_ONLY:
                keywords[param.name] = None
        inspect.signature(hook).bind(function, *positional, **keywords)
    for target, hook in HOOKS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
