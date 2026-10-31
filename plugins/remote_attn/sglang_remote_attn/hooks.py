"""Function-boundary hooks; no copies of SGLang scheduler loops."""

import inspect
import logging
import os
import pkgutil
import threading
import time

from . import (
    decode_overlap,
    dsv41_layerwise,
    dsv41_progressive,
    dsv41_receive,
    dsv41_remote,
    pre_layerwise,
    prefill_overlap,
    segmented_graph,
    source_cache,
)

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

    runner = scheduler.tp_worker.model_runner
    if dsv41_layerwise.is_dsv41(runner):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        if type(pool) is not DeepSeekV4TokenToKVPool:
            raise RuntimeError("DSV4.1 layerwise requires the native DSV4 KV pool")
        pre_layerwise.initialize(scheduler)
        return
    from sglang.srt.runtime_context import get_disagg, get_schedule

    if (
        get_disagg().disaggregation_mode == "prefill"
        and not get_schedule().disable_overlap_schedule
    ):
        raise RuntimeError("P overlap is supported only for the DSV4.1 PD adapter")
    if (
        get_schedule().page_size != 1
        or type(pool) is not MHATokenToKVPool
        or pool.kv_cache_layout != "nhd"
        or pool.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise RuntimeError(
            "remote_attn currently requires dense NHD MHA KV in BF16/FP16"
        )
    _runtime().get_remote_hub(scheduler)
    from .pre_layerwise import initialize

    initialize(scheduler)


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
    if get_exec().overlap.enable_two_batch_overlap:
        raise RuntimeError("remote_attn does not support two-batch overlap")
    if not schedule.disable_overlap_schedule and (
        not (
            get_disagg().disaggregation_mode == "prefill"
            and os.environ.get("HANDOFF_PRE_LAYERWISE", "off")
            in ("layerwise", "remote")
        )
        and (
            get_disagg().disaggregation_mode != "decode"
            or os.environ.get("HANDOFF_PRE_LAYERWISE", "off")
            not in ("remote", "layerwise")
        )
    ):
        raise RuntimeError(
            "remote_attn requires --disable-overlap-schedule except progressive D or DSV4.1 PD P"
        )
    if not memory.disable_radix_cache and (
        get_disagg().disaggregation_mode != "prefill"
        or os.environ.get("HANDOFF_PRE_LAYERWISE", "off") not in ("remote", "layerwise")
    ):
        raise RuntimeError(
            "remote_attn requires --disable-radix-cache except progressive P"
        )
    if memory.enable_unified_memory or memory.enable_hierarchical_cache:
        raise RuntimeError("remote_attn does not yet support unified KV or HiCache")
    # Paged DSV4.1 layerwise is validated against the concrete pool after init.
    # Dense MHA still checks page_size=1 in _scheduler_init.
    if schedule.enable_mixed_chunk or (
        schedule.page_size != 1
        and os.environ.get("HANDOFF_PRE_LAYERWISE", "off")
        not in ("layerwise", "remote")
    ):
        raise RuntimeError("remote_attn requires --page-size 1 without mixed chunks")
    if get_spec().speculative_algorithm is not None:
        raise RuntimeError("remote_attn does not yet support speculative decoding")
    if get_disagg().enable_pd_role_switch:
        raise RuntimeError("remote_attn does not yet support live PD role switching")
    if not callable(getattr(TransferEngine, "send_probe", None)):
        raise RuntimeError("remote_attn requires Mooncake TransferEngine.send_probe")


def _event_loop(original, scheduler, *args, **kwargs):
    try:
        return original(scheduler, *args, **kwargs)
    finally:
        from .pre_layerwise import shutdown

        try:
            shutdown(scheduler)
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
    from .retirement import split_controls

    requests, intents = split_controls(recv_reqs)
    from .prefill_admission import split_controls as split_admissions

    requests, admissions = split_admissions(requests)
    requests, controls = runtime.split_handoff_controls(requests)
    controller = getattr(scheduler, "_pre_layerwise", None)
    p_overlap = getattr(controller, "prefill_overlap", None)
    if p_overlap is not None:
        for envelope in admissions:
            p_overlap.admission.arm(envelope)
        for intent in intents:
            p_overlap.retirement.arm(intent)
    elif admissions:
        raise RuntimeError("P admission command requires DSV4.1 P overlap")
    result = original(scheduler, requests)
    if p_overlap is not None:
        p_overlap.finalize_control()
    for intent in intents:
        if p_overlap is not None:
            continue
        overlap = getattr(controller, "decode_overlap", None)
        if overlap is None:
            raise RuntimeError("ordered retirement requires progressive D overlap")
        overlap.retirement.arm(intent)
    for control in controls:
        controller = getattr(scheduler, "_pre_layerwise", None)
        if getattr(controller, "dsv41_layerwise", False) and control.get("cmd") not in {
            "status",
            "status_lite",
            "pd_queues",
            "debug_queue",
            "kv_ledger",
        }:
            raise RuntimeError(
                "DSV4.1 supports native PD only; no manual migration/dense-MHA Remote commands"
            )
        if getattr(controller, "decode_overlap", None) is not None and control.get(
            "cmd"
        ) not in {
            "status",
            "status_lite",
            "pd_queues",
            "debug_queue",
            "kv_ledger",
        }:
            # Legacy reset/release/import can bypass the native GPU/TP leases.
            raise RuntimeError(
                "D overlap accepts native progressive admissions only; manual state mutations are unsupported"
            )
        if controller is not None and control.get("cmd") in {
            "import",
            "import_many",
            "import_wave",
            "resume",
            "resume_many_decode",
            "admit_decode",
        }:
            # Controls were broadcast to the whole TP group. Revoke before
            # execution, even if a command fails before installing a shard.
            controller.graph_fast_path = False
        runtime.execute_handoff_control(scheduler, control)
    if not getattr(
        getattr(scheduler, "_pre_layerwise", None), "dsv41_layerwise", False
    ):
        runtime.maybe_handle_handoff_control(scheduler)
    from .pre_layerwise import poll

    poll(scheduler)
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
    target = getattr(req, "_pre_layerwise_target", None)
    if target is not None and getattr(target.owner, "dsv41_remote", False):
        # Direct native reset callers must respect the same TP-symmetric fence
        # as release_req; do not let a new attempt inherit pending PD writes.
        target.owner.drain_targets(required=target)
    dsv41_remote.release(req, "retraction")
    lease = getattr(req, "_dsv41_p_lease", None)
    if lease is not None:
        lease.overlap.drain(req)
    else:
        cert = getattr(req, "_dsv41_p_certificate", None)
        if cert is not None:
            cert.admission.overlap.drain(req)
    result = original(req)
    req.handoff_prefix_indices = None
    req._handoff_prefix_indices = None
    return result


def _release_kv(original, req, tree_cache, *, checkpoint):
    from .pre_layerwise import defer_release

    if defer_release(
        req,
        lambda: original(
            req, tree_cache, checkpoint=source_cache.checkpoint_allowed(req, checkpoint)
        ),
        checkpoint=checkpoint,
    ):
        return
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
    # Plugin cleanup can precede native decode's deferred release. The receiver
    # fence and pending-prefix cleanup above still apply without an admitted row.
    # Mamba state is independent of KV and must retain upstream's release path.
    if req.kv.is_kv_released and not req.kv.holds_kv and not req.kv.holds_mamba:
        return
    return original(req, tree_cache, checkpoint=checkpoint)


def _stream_output(original, streamer, reqs, *args, **kwargs):
    # The controller reads these requests through status/status_many. Sending
    # them to the normal detokenizer would create a second output owner.
    selected = []
    for req in reqs:
        if getattr(req, "_handoff_req", False):
            continue
        source = getattr(req, "_pre_layerwise_source", None)
        snapshot = getattr(req, "_pre_layerwise_response_snapshot", False)
        if source is not None and source.responded and not snapshot:
            continue
        selected.append(req)
    result = original(streamer, selected, *args, **kwargs)
    for req in selected:
        source = getattr(req, "_pre_layerwise_source", None)
        if source is not None:
            source.responded = True
    return result


def _decode_graph(original, runner, forward_batch):
    return _graph_decision(original, runner, forward_batch, "decode")


def _prefill_graph(original, runner, forward_batch):
    return _graph_decision(original, runner, forward_batch, "prefill")


def _graph_decision(original, runner, forward_batch, kind):
    from sglang.srt.runtime_context import get_parallel

    from .tp import agree_graph

    parallel = get_parallel()
    controller = getattr(getattr(runner, "model_runner", None), "_pre_layerwise", None)
    if (
        kind == "prefill"
        and getattr(controller, "dsv41_layerwise", False)
        and (
            controller.plan is not None
            or controller.executing is not None
            or bool(
                getattr(
                    getattr(
                        getattr(controller, "prefill_overlap", None), "current", None
                    ),
                    "jobs",
                    [],
                )
            )
        )
    ):
        # prepare agreed the batch across TP before any rank reaches this guard.
        return False
    if getattr(controller, "dsv41_remote", False) and controller.remote_targets:
        return False  # TP resource/lifetime leases still require eager decode.
    if controller is not None and controller.graph_fast_path:
        # early_targets admits/removes leases only after TP agreement. Keep
        # even a locally complete lease eager until its slowest peer retires.
        # The empty set is a shared Local certificate, not a rank-local guess.
        all_local = not controller.targets
        backend = getattr(runner, "backend", None)
        remote_graph = (
            not all_local
            and kind == "decode"
            and isinstance(backend, segmented_graph.RemoteGraphBackend)
            and backend.supports_remote(forward_batch)
        )
        can_replay = (all_local or remote_graph) and bool(
            original(runner, forward_batch)
        )
    else:
        # Legacy/manual imports have no progressive lease certificate. Every
        # rank must still join this conservative per-forward vote.
        eligible = bool(original(runner, forward_batch))
        can_replay, all_local = agree_graph(
            eligible,
            not _runtime().has_remote_work(),
            group=parallel.tp_group.cpu_group if parallel.tp_size > 1 else None,
            world_size=parallel.tp_size,
        )
    verdict = (can_replay, all_local)
    if getattr(runner, "_remote_attn_graph_verdict", None) != verdict:
        runner._remote_attn_graph_verdict = verdict
        logger.info(
            "[remote-attn-graph] kind=%s rank=%s tp=%s path=%s all_local=%s",
            kind,
            parallel.tp_rank,
            parallel.tp_size,
            ("graph-eligible" if all_local else "remote-segmented")
            if can_replay
            else "eager",
            all_local,
        )
    return can_replay


def _graph_execute(original, runner, forward_batch, *args, **kwargs):
    _runtime().graph_step_begin(forward_batch)
    backend = getattr(runner, "backend", None)
    if isinstance(backend, segmented_graph.RemoteGraphBackend):
        with backend.execution(forward_batch):
            return original(runner, forward_batch, *args, **kwargs)
    return original(runner, forward_batch, *args, **kwargs)


def _mooncake_init(original, manager, *args, **kwargs):
    # The constructor starts sender threads, so install the mutex first.
    manager._remote_attn_send_lock = threading.Lock()
    return original(manager, *args, **kwargs)


def _transfer(original, manager, mooncake_session_id, transfer_blocks, **kwargs):
    from sglang.srt.environ import envs

    if not transfer_blocks or not envs.SGLANG_MOONCAKE_SEND_LOCK.get():
        return original(manager, mooncake_session_id, transfer_blocks, **kwargs)
    with manager._remote_attn_send_lock:
        return original(manager, mooncake_session_id, transfer_blocks, **kwargs)


def _publish_peer(original, manager, mooncake_session_id, decode_kv_args):
    guard = getattr(manager, "_dsv41_admission_guard", None)
    if guard is not None:
        return guard.publish(
            mooncake_session_id,
            decode_kv_args,
            lambda: _publish_peer_now(
                original, manager, mooncake_session_id, decode_kv_args
            ),
        )
    return _publish_peer_now(original, manager, mooncake_session_id, decode_kv_args)


def _publish_peer_now(original, manager, mooncake_session_id, decode_kv_args):
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
    "sglang.srt.disaggregation.decode.DecodeTransferQueue._commit_transfer_to_req": dsv41_remote.commit,
    "sglang.srt.disaggregation.common.staging_handler.StagingManagerMixin._handle_staging_req": dsv41_receive.allocate,
    "sglang.srt.disaggregation.common.staging_handler.DecodeStagingHandler._scatter_region": dsv41_receive.scatter,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._try_create_staging_strategy": dsv41_receive.strategy,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.send_layer_group_to_staging": dsv41_receive.send_group,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.teardown": dsv41_receive.teardown,
    "sglang.srt.model_executor.runner_backend.utils.resolve_decode_backend": segmented_graph.resolve_backend,
    "sglang.srt.managers.scheduler.Scheduler.__init__": _scheduler_init,
    "sglang.srt.managers.scheduler.Scheduler.run_event_loop": _event_loop,
    "sglang.srt.managers.scheduler.Scheduler.process_input_requests": _process_requests,
    "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver._broadcast_reqs_across_ranks": _broadcast,
    "sglang.srt.managers.schedule_batch.Req.init_next_round_input": _init_round,
    "sglang.srt.managers.schedule_batch.Req.reset_for_retract": _reset_retracted,
    "sglang.srt.managers.schedule_batch.release_req": decode_overlap.before_retract,
    "sglang.srt.mem_cache.allocation._prefix_kv_indices": _prefix_indices,
    "sglang.srt.mem_cache.allocation.alloc_for_extend": _alloc_extend,
    "sglang.srt.mem_cache.common.release_kv_cache": _release_kv,
    "sglang.srt.managers.scheduler_components.output_streamer.SchedulerOutputStreamer.stream_output": _stream_output,
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.can_run_graph": _decode_graph,
    "sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.execute": _graph_execute,
    "sglang.srt.model_executor.runner.prefill_cuda_graph_runner.PrefillCudaGraphRunner.can_run_graph": _prefill_graph,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.__init__": _mooncake_init,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._transfer_data": _transfer,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.send_kvcache": dsv41_layerwise.send_selected,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.send_aux": dsv41_progressive.send_aux,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager.add_transfer_request": dsv41_progressive.canonical_track,
    "sglang.srt.models.deepseek_v4_replay_tail.DecoderReplayTail.plan": prefill_overlap.tail_plan,
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.clear_pending_chunk_send": prefill_overlap.before_mutation,
    "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue": prefill_overlap.before_mutation,
    "sglang.srt.disaggregation.mooncake.conn.MooncakeKVManager._publish_peer_registration": _publish_peer,
}

# Thin adapters; layer-pipeline state is owned by the scheduler instance.
HOOKS.update(
    {
        "sglang.srt.disaggregation.common.staging_handler.init_staging_allocator": pre_layerwise.init_staging_allocator,
        "sglang.srt.disaggregation.common.staging_handler.init_staging_buffers": pre_layerwise.init_staging_buffers,
        "sglang.srt.managers.scheduler.Scheduler.run_batch": pre_layerwise.run_batch,
        "sglang.srt.managers.scheduler.Scheduler.process_batch_result": decode_overlap.after_result,
        "sglang.srt.managers.scheduler.Scheduler.launch_batch_sample_if_needed": prefill_overlap.after_sample,
        "sglang.srt.managers.scheduler.Scheduler.is_disable_overlap_for_batch": decode_overlap.disable_overlap,
        "sglang.srt.managers.scheduler.Scheduler.is_fully_idle": pre_layerwise.is_fully_idle,
        "sglang.srt.model_executor.model_runner.ModelRunner.forward": pre_layerwise.forward,
        "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.send_kv_chunk": pre_layerwise.send_chunk,
        "sglang.srt.disaggregation.decode.DecodeTransferQueue.pop_transferred": pre_layerwise.pop_transferred,
        "sglang.srt.disaggregation.decode.DecodeTransferQueue.resolve_deferred_releases": decode_overlap.reap,
    }
)


def register_hooks():
    from sglang.srt.environ import envs
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from .interfaces import check_runtime_interfaces

    # Resolve and signature-check the whole set before changing any target.
    # SGLang logs plugin failures, so the Scheduler init hook also verifies
    # that the loader actually applied every registration before serving.
    check_runtime_interfaces()
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
