"""Subprocess fixture for the CPU-only TP control agreement regression test."""

import argparse
import json
import sys
import threading
import time
from dataclasses import make_dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
from sglang_remote_attn.tp import agree_control_result, inject_control


def progressive_graph_leases(args, context):
    """Real admission/retirement votes with transport and model boundaries on CPU."""
    import torch
    from sglang_remote_attn import handoff_control as runtime
    from sglang_remote_attn import hooks
    from sglang_remote_attn.pre_layerwise import (
        Config,
        Controller,
        LayerCoverage,
        Target,
        write_json,
    )

    root = Path(args.output).with_suffix(".leases")
    config = make_dataclass("GraphConfig", [("decode", str), ("prefill", str)])(
        "full", "breakable"
    )
    context.get_exec = lambda: SimpleNamespace(
        graph=SimpleNamespace(cuda_graph_config=config)
    )
    model_runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: []))
    decode = SimpleNamespace(model_runner=model_runner, capture_bs=[1, 2, 4], max_bs=4)
    prefill = SimpleNamespace(
        model_runner=model_runner, capture_num_tokens=[4, 8], max_bs=4
    )
    model_runner.decode_cuda_graph_runner, model_runner.prefill_cuda_graph_runner = (
        decode,
        prefill,
    )
    scheduler = SimpleNamespace(
        attn_tp_cpu_group=dist.group.WORLD,
        tp_worker=SimpleNamespace(model_runner=model_runner),
    )
    controller = Controller.__new__(Controller)
    controller.scheduler, controller.tp_size, controller.rank = scheduler, 2, args.rank
    controller.layers, controller.early_admissions, controller.targets = 1, 0, []
    controller.config = Config("remote", root)
    scheduler._pre_layerwise = model_runner._pre_layerwise = controller
    freed = []
    queue = SimpleNamespace(
        queue=[],
        metadata_buffers=SimpleNamespace(bootstrap_room=torch.full((3, 1), 7)),
        req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=freed.append),
        _commit_transfer_to_req=lambda *a, **kw: None,
    )

    def offer(index):
        req = SimpleNamespace(
            rid=str(index),
            bootstrap_room=7,
            output_ids=[],
            kv=SimpleNamespace(req_pool_idx=index),
            finished=lambda: False,
        )
        receiver = SimpleNamespace(status=2, clear=lambda: None)
        receiver.poll = lambda: receiver.status
        dr = SimpleNamespace(req=req, kv_receiver=receiver, metadata_buffer_index=index)
        target = Target(
            controller, queue, dr, root / str(index), str(index), LayerCoverage(2, 1)
        )
        dr._pre_layerwise_offer = target
        queue.queue.append(dr)
        return target

    def ready(target):
        write_json(
            target.path / "ready.json",
            {
                "generation": target.generation,
                "prefix_len": 2,
                "service": {"handoff_id": target.generation},
            },
        )

    def complete(target):
        target.decode_req.kv_receiver.status = 4
        write_json(
            target.path / "layer-0.json",
            {"generation": target.generation, "layer": 0, "start": 0, "end": 2},
        )

    reductions = 0
    actual_reduce = dist.all_reduce

    def counted_reduce(*a, **kw):
        nonlocal reductions
        reductions += 1
        return actual_reduce(*a, **kw)

    snapshots = []
    with (
        patch.dict(
            sys.modules,
            {
                "sglang.srt.runtime_context": context,
                "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                    KVPoll=SimpleNamespace(Success=4, Failed=0)
                ),
                "sglang.srt.layers.radix_attention": SimpleNamespace(
                    RadixAttention=type("RadixAttention", (), {})
                ),
            },
        ),
        patch.object(runtime, "Path", return_value=root),
        patch.object(runtime, "PROFILE", False),
        patch.object(runtime, "RPC_PROFILE", False),
        patch.object(runtime, "_TARGET_STATES", []),
        patch.object(dist, "all_reduce", counted_reduce),
    ):
        controller.validate_graphs()
        previous = controller.config
        controller.dsv41_layerwise = True
        controller.config = Config("remote" if args.rank == 0 else "layerwise", root)
        try:
            controller.validate_graphs()
            sparse_mode_mismatch_rejected = False
        except RuntimeError:
            sparse_mode_mismatch_rejected = True
        controller.config = previous
        del controller.dsv41_layerwise
        with patch.object(
            runtime, "_TARGET_STATES", [object()] if args.rank == 0 else []
        ):
            controller.validate_graphs()
            existing_state_fallback = not controller.graph_fast_path
        # The same capture mismatch must fail startup on *both* ranks.
        if args.rank == 1:
            decode.capture_bs.append(8)
        try:
            controller.validate_graphs()
            mismatch_rejected = False
        except RuntimeError:
            mismatch_rejected = True
        if args.rank == 1:
            decode.capture_bs.pop()
        scheduler.tree_cache = SimpleNamespace(disable=bool(args.rank))
        try:
            controller.validate_graphs()
            cache_mismatch_rejected = False
        except RuntimeError:
            cache_mismatch_rejected = True
        del scheduler.tree_cache
        scheduler.enable_overlap = bool(args.rank)
        try:
            controller.validate_graphs()
            overlap_mismatch_rejected = False
        except RuntimeError:
            overlap_mismatch_rejected = True
        scheduler.enable_overlap = True
        controller.validate_graphs()

        def sample():
            before = reductions
            verdicts = []
            # Include a Local steady-state burst, not just one lucky step.
            for _ in range(16):
                verdicts.append(
                    [
                        hooks._decode_graph(lambda *a: True, decode, None),
                        hooks._prefill_graph(lambda *a: True, prefill, None),
                    ]
                )
            snapshots.append(
                {
                    "leases": len(controller.targets),
                    "verdicts": verdicts,
                    "forward_reductions": reductions - before,
                }
            )

        a = offer(0)
        if args.rank == 0:
            ready(a)
        controller.early_targets(queue, None)  # One rank ready: neither admits.
        sample()
        ready(a)
        controller.early_targets(queue, None)
        sample()
        b = offer(1)
        ready(b)
        controller.early_targets(queue, None)
        sample()
        complete(a)
        controller.early_targets(queue, None)
        sample()  # A retired; B still prevents replay.
        if args.rank == 0:
            complete(b)
        controller.early_targets(queue, None)
        sample()  # The fast shard must keep B's lease.
        if args.rank == 1:
            complete(b)
        controller.early_targets(queue, None)
        sample()
        c = offer(2)
        ready(c)
        controller.early_targets(queue, None)
        sample()  # A new admission revokes Local before the next forward.
        complete(c)
        controller.early_targets(queue, None)
        sample()
    return {
        "snapshots": snapshots,
        "mismatch_rejected": mismatch_rejected,
        "sparse_mode_mismatch_rejected": sparse_mode_mismatch_rejected,
        "cache_mismatch_rejected": cache_mismatch_rejected,
        "overlap_mismatch_rejected": overlap_mismatch_rejected,
        "existing_state_fallback": existing_state_fallback,
        "boundary_reductions": reductions,
        "freed": freed,
    }


def decode_overlap_lifetimes(args):
    """Real native broadcast + local UDS notices, with no polling retirement vote."""
    import ast
    import dataclasses

    import torch
    from sglang_remote_attn import hooks
    from sglang_remote_attn.decode_overlap import (
        DecodeOverlap,
        before_retract,
        disable_overlap,
    )
    from sglang_remote_attn.pre_layerwise import (
        Config,
        Controller,
        LayerCoverage,
        Target,
        defer_release,
        run_batch,
    )
    from sglang_remote_attn.retirement import split_controls, wrap_receiver

    controller = Controller.__new__(Controller)
    controller.config = Config(
        "remote", Path(args.output).with_suffix(".ordered"), drain_timeout=5
    )
    controller.tp_size, controller.rank, controller.graph_fast_path = 2, args.rank, True
    controller.targets, controller.sources = [], []
    scheduler = SimpleNamespace(
        attn_tp_cpu_group=dist.group.WORLD,
        enable_overlap=True,
        _pre_layerwise=controller,
    )
    controller.scheduler = scheduler
    controller.decode_overlap = DecodeOverlap(controller)
    protocol = controller.decode_overlap.retirement
    root = Path(__file__).resolve().parents[3]
    path = root / "python/sglang/srt/managers/scheduler_components/request_receiver.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SchedulerRequestReceiver"
    )
    node.body = [
        n
        for n in node.body
        if isinstance(n, ast.AnnAssign)
        or getattr(n, "name", None) == "_broadcast_reqs_across_ranks"
    ]

    def broadcast(obj, rank, group, src=0):
        values = [obj]
        dist.broadcast_object_list(values, src=src, group=group)
        return values[0]

    parallel = SimpleNamespace(
        attn_dp_enabled=False,
        tp_size=2,
        tp_group=SimpleNamespace(
            rank=args.rank, cpu_group=dist.group.WORLD, ranks=[0, 1]
        ),
    )
    ns = dict(
        __name__=__name__,
        dataclass=dataclasses.dataclass,
        get_parallel=lambda: parallel,
        broadcast_pyobj=broadcast,
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        node,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        ns,
    )
    base = ns["SchedulerRequestReceiver"](
        **{f.name: None for f in dataclasses.fields(ns["SchedulerRequestReceiver"])}
    )
    receiver = wrap_receiver(base, protocol)
    freed, vote_order, commands = [], [], []
    notice_attempts = []
    native_send = protocol.mailbox.send

    def send_with_backpressure(notice):
        if args.rank == 1:
            notice_attempts.append(notice["version"])
            if len(notice_attempts) == 1:
                return False
        return native_send(notice)

    native_reduce = dist.all_reduce

    def counted_reduce(tensor, *a, **kw):
        vote_order.append([protocol.last_seq + 1, tensor.numel()])
        return native_reduce(tensor, *a, **kw)

    def target(index):
        req = SimpleNamespace(rid=f"req-{index}")
        io = SimpleNamespace(done=False, clear=lambda: None)
        io.poll = lambda: 4 if io.done else 2
        value = Target(
            controller,
            SimpleNamespace(
                metadata_buffers=SimpleNamespace(bootstrap_room=torch.ones((1, 1))),
                req_to_metadata_buffer_idx_allocator=SimpleNamespace(
                    free=lambda i: None
                ),
            ),
            SimpleNamespace(req=req, kv_receiver=io, metadata_buffer_index=0),
            controller.config.directory / str(index),
            f"g-{index}-rank{args.rank}",
            LayerCoverage(2, 1),
            admission_id=index,
        )
        req._pre_layerwise_target = value
        controller.targets.append(value)
        protocol.register(value)
        return value, req, io

    def cycle(*, finish=True):
        controller.retire_targets()  # Local checks/notifications only.
        messages = receiver._broadcast_reqs_across_ranks(
            ["native-request"] if args.rank == 0 else None
        )
        ordinary, intents = split_controls(messages)
        assert ordinary == ["native-request"]
        for command in intents:
            commands.append(command["seq"])
        # Exercise the real input hook: internal intents do not pass through
        # generic handoff controls (which have a separate all-gather contract).
        hooks._process_requests(lambda scheduler, requests: None, scheduler, messages)
        if finish:
            protocol.finalize()
        return bool(intents)

    imports = {
        "sglang.srt.disaggregation.base.conn": SimpleNamespace(
            KVPoll=SimpleNamespace(Success=4, Failed=0)
        ),
        "sglang.srt.managers.schedule_batch": SimpleNamespace(FINISH_LENGTH=None),
    }
    try:
        with (
            patch.dict(sys.modules, imports),
            patch.object(dist, "all_reduce", counted_reduce),
            patch.object(protocol.mailbox, "send", side_effect=send_with_backpressure),
        ):
            first, req, io = target(1)
            batch = SimpleNamespace(
                reqs=[req], forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
            )
            result = SimpleNamespace(
                copy_done=SimpleNamespace(query=lambda: True),
                delay_sample_func=object() if args.rank == 1 else None,
            )
            run_batch(lambda *a: result, scheduler, batch)
            for _ in range(24):
                cycle()
            assert not vote_order and not commands
            io.done = args.rank == 0
            for _ in range(8):
                cycle()
            assert not vote_order  # One IO-ready rank cannot schedule a vote.
            io.done = True
            if args.rank == 0:
                defer_release(req, lambda: freed.append("KV"))
            for _ in range(8):
                cycle()
            assert not vote_order  # Release-class mismatch also blocks proposals.
            if args.rank == 1:
                defer_release(req, lambda: freed.append("KV"))
            for _ in range(8):
                if cycle(finish=False):
                    break
            else:
                raise AssertionError("missing ordered intent")
            assert len(controller.targets) == 1 and not vote_order
            # Stale READY: rank 1 still has an unrecorded deferred-sampling event.
            protocol.finalize()
            assert protocol.retries == 1 and not freed
            for _ in range(16):
                cycle()
            assert (
                len(vote_order) == 1
            )  # No retry storm while GPU readiness is unchanged.
            result.delay_sample_func = None
            for _ in range(8):
                cycle()
                if not controller.targets:
                    break
            else:
                raise AssertionError("target did not retire")
            assert freed == ["KV"] and protocol.commits == 1
            with (
                patch.object(
                    dist,
                    "all_reduce",
                    side_effect=AssertionError("Local retirement collective"),
                ),
                patch.object(
                    protocol.mailbox,
                    "receive",
                    side_effect=AssertionError("Local mailbox polling"),
                ),
                patch.object(
                    protocol.mailbox,
                    "send",
                    side_effect=AssertionError("Local notification"),
                ),
            ):
                for _ in range(32):
                    cycle()
                    assert disable_overlap(
                        lambda *a: False, scheduler, batch, batch
                    ) in (True, False)
                    run_batch(lambda *a: result, scheduler, batch)
            second, req2, io2 = target(2)
            io2.done = True
            polls = 0

            def gpu_ready():
                nonlocal polls
                polls += 1
                return polls >= (3 if args.rank == 1 else 1)

            result2 = SimpleNamespace(
                copy_done=SimpleNamespace(query=gpu_ready), delay_sample_func=None
            )
            batch2 = SimpleNamespace(
                reqs=[req2], forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
            )
            run_batch(lambda *a: result2, scheduler, batch2)
            reentry = disable_overlap(lambda *a: False, scheduler, batch2, batch)

            def backup(*, req):
                assert req._pre_layerwise_target is None
                return True

            assert before_retract(backup, req=req2)
            # A normal intent already armed at intake must be consumed before
            # the nested retraction boundary can reserve another command seq.
            third, req3, io3 = target(3)
            io3.done = True
            for _ in range(8):
                if cycle(finish=False):
                    break
            else:
                raise AssertionError("missing pending intent for retraction")
            assert before_retract(backup, req=req3)
            fourth, _, io4 = target(4)
            fifth, _, io5 = target(5)
            io4.done = io5.done = True
            # Coalescing requires both ranks' READY frames to be available
            # before leader intake. Without this channel boundary the leader
            # can legitimately propose target 4 while rank 1 is still posting
            # target 5, giving two safe commands instead of the expected one.
            with patch.object(protocol.mailbox, "receive", return_value=[]):
                controller.retire_targets()
            dist.barrier()
            for _ in range(8):
                cycle()
                if not controller.targets:
                    break
            else:
                raise AssertionError("coalesced targets did not retire")
            if args.rank == 1:
                assert (
                    len(notice_attempts) >= 2
                    and notice_attempts[0] == notice_attempts[1]
                )
            return {
                "vote_order": vote_order,
                "commands": commands,
                "freed": freed,
                "retries": protocol.retries,
                "commits": protocol.commits,
                "last_seq": protocol.last_seq,
                "reentry": reentry,
                "gpu_proven": first.gpu_proven_forwards,
                "notice_retry_ok": True,
                "registered": len(protocol.registered),
            }
    finally:
        protocol.close()


def prefill_overlap_lifetimes(args):
    """P source adapter over real Gloo, asymmetric GPU/IO and rank generations."""
    from sglang_remote_attn.pre_layerwise import Config, Controller
    from sglang_remote_attn.prefill_overlap import (
        Chunk,
        LaunchPlan,
        PrefillOverlap,
        SourceLease,
    )
    from sglang_remote_attn.retirement import split_controls

    controller = Controller.__new__(Controller)
    controller.scheduler = SimpleNamespace(attn_tp_cpu_group=dist.group.WORLD)
    controller.rank, controller.tp_size, controller.sources = args.rank, 2, []
    controller.config = Config("layerwise", Path(args.output), drain_timeout=3)
    overlap = PrefillOverlap(controller)
    protocol = overlap.retirement
    votes, freed = [], []
    actual = dist.all_reduce

    def counted(tensor, *a, **kw):
        votes.append(tensor.numel())
        return actual(tensor, *a, **kw)

    def admit(index):
        req = SimpleNamespace(
            rid=str(index),
            bootstrap_room=index,
            cache_request_handle=(index, 0),
            kv=SimpleNamespace(req_pool_idx=index),
        )
        req.disagg_kv_sender = SimpleNamespace(poll=lambda: 4, kv_mgr=SimpleNamespace())
        chunk = Chunk(req, req.disagg_kv_sender, (index, 0), index, 0, 16, True)
        lease = SourceLease(overlap, chunk, index, f"{index}-rank-{args.rank}")
        event = SimpleNamespace(query=lambda: args.rank == 0)
        plan = LaunchPlan(index, None, (chunk,), [], [])
        plan.proof.result = SimpleNamespace(delay_sample_func=None, copy_done=event)
        plan.forward_done = event
        lease.plans.append(plan)
        lease.finalized = True
        lease.release = lambda: freed.append(index)
        req._dsv41_p_lease = lease
        overlap.leases.append(lease)
        overlap.by_room[index] = lease
        protocol.register(lease)
        return lease, event

    try:
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                        KVPoll=SimpleNamespace(Success=4, Failed=0)
                    )
                },
            ),
            patch.object(dist, "all_reduce", counted),
        ):
            lease, event = admit(1)
            for _ in range(16):
                overlap.poll()
            dist.barrier()
            values = [protocol.inject([]) if args.rank == 0 else None]
            dist.broadcast_object_list(values, src=0)
            _, commands = split_controls(values[0])
            assert not commands and not votes  # slow rank has no GPU proof
            event.query = lambda: True
            overlap.poll()
            dist.barrier()
            values = [protocol.inject([]) if args.rank == 0 else None]
            dist.broadcast_object_list(values, src=0)
            _, commands = split_controls(values[0])
            assert len(commands) == 1
            protocol.arm(commands[0])
            protocol.finalize()
            assert freed == [1]
            second, event = admit(2)
            second.plans[0].proof.result.delay_sample_func = lambda: None
            # Explicit cancellation/retry boundary uses forward_done, not an
            # unrecorded copy_done from delayed sampling. Wait for slow GPU.
            timer = threading.Timer(0.04, lambda: setattr(event, "query", lambda: True))
            timer.start()
            overlap.drain(second.chunk.req)
            timer.join()
            before = len(votes)
            for _ in range(32):
                overlap.poll()
                protocol.inject([])
            assert len(votes) == before
            return dict(
                votes=votes,
                freed=freed,
                commits=protocol.commits,
                registered=len(protocol.registered),
                local_votes=len(votes) - before,
            )
    finally:
        overlap.admission.close()
        protocol.close()


def prefill_certificate_lifetimes(args, *, prefix_cache=False):
    """Real TP2: cached chunks, asymmetric bootstrap, lost notice, revoke/retire."""
    import ast
    import copy
    from typing import NamedTuple

    import torch
    from sglang_remote_attn import hooks
    from sglang_remote_attn import prefill_admission as admission
    from sglang_remote_attn import prefill_overlap as po
    from sglang_remote_attn.pre_layerwise import Config, Controller, write_json
    from sglang_remote_attn.retirement import CONTROL as RETIRE

    controller = Controller.__new__(Controller)
    controller.rank, controller.tp_size = args.rank, 2
    controller.sources, controller.layers, controller.dsv41_layerwise = [], 2, True
    controller.config = Config(
        "layerwise", Path(args.output).with_suffix(".certificates"), 1, drain_timeout=3
    )
    scheduler = controller.scheduler = SimpleNamespace(
        attn_tp_cpu_group=dist.group.WORLD, _pre_layerwise=controller
    )
    cache = None
    if prefix_cache:
        from sglang_remote_attn.source_cache import CacheGuard

        refs = {}

        def lock(node):
            refs[node] = refs.get(node, 0) + 1
            return SimpleNamespace(node=node, closed=False)

        def unlock(receipt):
            assert not receipt.closed and refs[receipt.node] > 0
            refs[receipt.node] -= 1
            receipt.closed = True

        cache = SimpleNamespace(
            lock=lock,
            unlock=unlock,
            checkpoint=lambda *a, **kw: None,
            finish=lambda *a: None,
            reset=lambda: None,
            refs=refs,
        )
        controller.cache_guard = CacheGuard(cache)
    overlap = controller.prefill_overlap = po.PrefillOverlap(controller)
    manager = SimpleNamespace(
        transfer_infos={}, decode_kv_args_table={}, _dsv41_p_overlap=overlap
    )
    old_registration = SimpleNamespace(dst_kv_ptrs=[11, 21], dst_kv_item_len=8)
    manager.decode_kv_args_table["peer:1"] = old_registration
    peer = SimpleNamespace(mooncake_session_id="peer:1", is_dummy=False)
    tasks, frees, aborts, votes = [], [], [], []
    original_reduce = dist.all_reduce
    events = []

    class Event:
        def __init__(self):
            self.done = True
            events.append(self)

        def record(self, stream):
            pass

        def query(self):
            return self.done

    def counted(tensor, *a, **kw):
        votes.append(tensor.numel())
        return original_reduce(tensor, *a, **kw)

    class Handle(NamedTuple):
        rid: str
        attempt_id: int = 0

    def make_req(room, pending=False):
        req = SimpleNamespace(
            rid=str(room),
            bootstrap_room=room,
            cache_request_handle=Handle(str(room)),
            retraction_count=0,
            kv=SimpleNamespace(req_pool_idx=room),
            start_send_idx=0,
            extend_end=8,
            extend_len=8,
            pending_bootstrap=pending,
            origin_input_ids=list(range(1025)),
            return_logprob=False,
        )
        req.disagg_kv_sender = SimpleNamespace(
            kv_mgr=manager, poll=lambda: 4, curr_idx=0
        )

        def abort():
            assert manager.decode_kv_args_table["peer:1"] is old_registration
            aborts.append(room)

        req.disagg_kv_sender.abort = abort
        manager.transfer_infos[room] = {"peer:1": copy.copy(peer)}
        root = controller.config.directory / str(room) / str(args.rank)
        offer = dict(generation=f"{room}-rank-{args.rank}", prefix_len=1025)
        write_json(root / "offer.json", offer)
        req.offer = (root, offer)
        if cache is not None:
            req.kv.cache_protected_len = 8 + args.rank * 8
            req.last_node, req.lock = room, cache.lock(room)
        return req

    def candidates(batch, **kwargs):
        return [
            (req, req.disagg_kv_sender, *req.offer, ((peer, old_registration),), {})
            if not req.pending_bootstrap
            else None
            for req in batch.reqs
        ]

    controller.candidates = candidates
    controller.prepare = lambda *a, **kw: []
    batch = SimpleNamespace(
        reqs=[],
        forward_mode=SimpleNamespace(
            is_extend=lambda: True,
            is_prebuilt=lambda: False,
            is_split_prefill=lambda: False,
            is_mixed=lambda: False,
        ),
        is_prefill_only=False,
        return_hidden_states=False,
        input_embeds=None,
        replace_embeds=None,
    )
    batch.copy = lambda: copy.copy(batch)

    def add(manager, room, indices, span, last, **kwargs):
        tasks.append((room, kwargs["on_complete"]))

    def send(scheduler, req, *, last_chunk, end_idx=None):
        po.track_transfer(add, manager, req.bootstrap_room, [], slice(0, 0), last_chunk)
        req.start_send_idx = req.extend_end
        req.disagg_kv_sender.curr_idx = req.start_send_idx // 8

    scheduler.send_kv_chunk = lambda req, **kw: overlap.send_chunk(
        send, scheduler, req, end_idx=kw.pop("end_idx", None), **kw
    )

    def launch(scheduler, batch):
        overlap.forward(lambda *a: None, None, batch)
        return SimpleNamespace(delay_sample_func=None, copy_done=Event())

    def run(reqs):
        batch.reqs = reqs
        result = overlap.run_batch(launch, scheduler, batch)
        po.after_result(lambda *a: None, scheduler, batch, result)
        return result

    def receive(payload, original=lambda *a: None):
        hooks._process_requests(original, scheduler, payload)

    def pump():
        overlap.poll()
        dist.barrier()
        values = [overlap.inject([]) if args.rank == 0 else None]
        dist.broadcast_object_list(values, src=0)
        receive(values[0])
        return values[0]

    def complete(room=None):
        for task in list(tasks):
            if room is None or task[0] == room:
                task[1]()
                tasks.remove(task)

    def freed(req):
        lease = req._dsv41_p_lease

        def release():
            assert not lease.ledger.pending and lease.gpu_ready()
            if cache is not None:
                if not lease.cancelling:
                    cache.checkpoint(req, up_to=req.extend_end)
                cache.finish(
                    req.cache_request_handle,
                    SimpleNamespace(name="ABORT" if lease.cancelling else "SUCCESS"),
                )
                cache.unlock(req.lock)
                req.lock = None
            frees.append(req.bootstrap_room)

        overlap.defer_release(req, release)

    # Real native abort metadata function; only the finish-reason class is a boundary.
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / "python/sglang/srt/disaggregation/utils.py").read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == "prepare_abort")
    ns = {}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        node,
                    ],
                    type_ignores=[],
                )
            ),
            "<native-prepare-abort>",
            "exec",
        ),
        ns,
    )
    modules = {
        "sglang.srt.disaggregation.utils": SimpleNamespace(
            prepare_abort=ns["prepare_abort"]
        ),
        "sglang.srt.managers.schedule_batch": SimpleNamespace(
            FINISH_ABORT=lambda message, status: SimpleNamespace(message=message)
        ),
        "sglang.srt.disaggregation.base.conn": SimpleNamespace(
            KVPoll=SimpleNamespace(Success=4, Failed=0)
        ),
    }
    lost = [False]
    actual_send = overlap.admission.mailbox.send

    def lose_first(notice):
        if args.rank == 1 and not lost[0]:
            lost[0] = True
            return (
                True  # accepted by send boundary but deliberately lost before receive
            )
        return actual_send(notice)

    overlap.admission.mailbox.send = lose_first
    try:
        with (
            patch.dict(sys.modules, modules),
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "current_stream", lambda: None),
            patch.object(dist, "all_reduce", counted),
        ):
            a = make_req(101)
            run([a])
            freed(a)
            pump()
            pump()  # ACK loss must resend, without a revalidation collective
            assert a._dsv41_p_certificate.acked >= 1 or args.rank == 0
            before = len(votes)
            for end in range(16, 520, 8):
                a.extend_end = end
                run([a])
                pump()
            stable_votes = len(votes) - before
            assert stable_votes == 0
            b = make_req(102, pending=True)
            a.extend_end += 8
            run([a, b])
            before = len(votes)
            for _ in range(4):
                run([a, b])
                pump()
            assert len(votes) == before
            if args.rank == 0:
                b.pending_bootstrap = False
            pump()
            assert len(votes) == before  # one ready shard cannot activate
            b.pending_bootstrap = False
            pump()
            assert b._dsv41_p_certificate.active
            run([b])
            freed(b)
            c = make_req(103)
            c.extend_end = len(c.origin_input_ids)
            run([c])
            scheduler.send_kv_chunk(c, last_chunk=True)
            freed(c)
            complete(103)
            overlap.poll()
            dist.barrier()
            saved = [overlap.inject([]) if args.rank == 0 else None]
            dist.broadcast_object_list(saved, src=0)
            assert any(RETIRE in item for item in saved[0])
            # While retirement is in flight, a registration replacement may
            # notify, but cannot issue a second unordered collective command.
            replacement = SimpleNamespace(dst_kv_ptrs=[31, 41], dst_kv_item_len=8)
            if args.rank == 1:
                guard = overlap.admission.guard(manager)
                guard.publish(
                    "peer:1",
                    replacement,
                    lambda: manager.decode_kv_args_table.__setitem__(
                        "peer:1", replacement
                    ),
                )
            payload = pump()
            assert not any(
                admission.CONTROL in item and item[admission.CONTROL].get("command")
                for item in payload or []
            )
            # A native cancellation in the same intake consumes the older
            # retirement before its own drain, on both TP ranks.
            for req in (a, b):
                req._dsv41_p_lease.plans[-1].forward_done.done = False

            def finish():
                for event in events:
                    event.done = True
                complete()

            timer = threading.Timer(0.02 + args.rank * 0.03, finish)
            timer.start()
            try:
                receive(
                    saved[0],
                    lambda *unused: po.before_mutation(
                        lambda scheduler, req: req.disagg_kv_sender.abort(),
                        scheduler,
                        a,
                    ),
                )
            finally:
                timer.join()
            assert frees == [103, 101], (frees, votes)
            pump()
            assert aborts == [101, 102]
            assert frees == [103, 101, 102]
            assert b.finished_reason.message == "PD admission qualification changed"
            assert not overlap.leases and not overlap.admission.records
            assert not overlap.admission.guard(manager).pins
            assert manager.decode_kv_args_table["peer:1"] is (
                replacement if args.rank == 1 else old_registration
            )
            before = len(votes)
            for _ in range(16):
                pump()
            assert len(votes) == before
            empty_votes = len(votes) - before
            active_cache_closed = cache is None or (
                not controller.cache_guard.leases
                and all(cache.refs.get(room, 0) == 0 for room in (101, 102, 103))
            )
            assert active_cache_closed
            d = make_req(104, pending=True)
            run([d])
            d.pending_bootstrap = False
            cert = d._dsv41_p_certificate
            envelope = dict(
                epoch=overlap.admission.epoch,
                acks=[],
                command=dict(
                    epoch=overlap.admission.epoch,
                    seq=overlap.admission.last_seq + 1,
                    entries=[
                        dict(key=list(cert.key), version=cert.version, revoke=False)
                    ],
                ),
            )
            if args.rank == 1:
                cert.version += (
                    1  # corrupted/stale local state must vote, never return early
                )
            rejected = False
            try:
                receive([{admission.CONTROL: envelope}])
            except RuntimeError as exc:
                rejected = "validation failed" in str(exc)
            assert rejected and not cert.active
            return dict(
                votes=votes,
                prefix_cache=prefix_cache,
                active_cache_closed=active_cache_closed,
                stable_chunks=63,
                stable_votes=stable_votes,
                frees=frees,
                aborts=aborts,
                admission_checks=overlap.admission.checks,
                revalidations=overlap.admission.revalidations,
                revocations=overlap.admission.revocations,
                ack_loss_recovered=True,
                stale_version_rejected=True,
                empty_votes=empty_votes,
            )
    finally:
        overlap.admission.close()
        overlap.retirement.close()


def prefill_staging_failure(args):
    """An asymmetric poisoned slot must reach the same ordered abort vote."""
    import time

    from sglang_remote_attn.pre_layerwise import Config, Controller
    from sglang_remote_attn.prefill_overlap import SourceLease
    from sglang_remote_attn.retirement import Retirement

    owner = Controller.__new__(Controller)
    owner.rank, owner.tp_size = args.rank, 2
    owner.scheduler = SimpleNamespace(attn_tp_cpu_group=dist.group.WORLD)
    owner.config = Config("layerwise", Path(args.output).with_suffix(".staging"))
    slot = SimpleNamespace(poisoned=False)
    manager = SimpleNamespace(_dsv41_host_stager=slot)
    chunk = SimpleNamespace(
        req=SimpleNamespace(rid="staged", bootstrap_room=501),
        sender=SimpleNamespace(kv_mgr=manager),
        check=lambda: None,
    )
    lease = SourceLease(None, chunk, 1, f"stage-{args.rank}")
    lease.ledger.pending.add(1)  # Failed sends never run their callback.
    lease.finish = lambda **kw: (_ for _ in ()).throw(AssertionError("unsafe free"))
    owner.targets = [lease]
    protocol = Retirement(owner)
    protocol.register(lease)
    votes = []
    actual_reduce = dist.all_reduce

    def counted(tensor, *a, **kw):
        votes.append(tensor.numel())
        return actual_reduce(tensor, *a, **kw)

    try:
        with patch.object(dist, "all_reduce", counted):
            for _ in range(32):
                protocol.observe()
                assert protocol.poll_command() is None
            assert not votes  # Stable pending IO never votes per poll/chunk.
            dist.barrier()  # Complete both stable windows before injecting failure.
            slot.poisoned = args.rank == 1
            command = None
            for _ in range(100):
                protocol.observe()
                candidate = protocol.poll_command()
                message = [candidate]
                dist.broadcast_object_list(message, src=0)
                if message[0] is not None:
                    command = message[0]
                    break
                time.sleep(0.005)
            assert command is not None and command["kind"] == "abort"
            protocol.arm(command)
            rejected = False
            try:
                protocol.finalize()
            except RuntimeError as exc:
                rejected = "retaining KV" in str(exc)
            assert rejected and votes == [5]
            assert owner.targets == [lease] and not lease.closed
            assert lease.ledger.pending == {1}
            return dict(
                stable_votes=0,
                final_vote_shapes=votes,
                both_rejected=True,
                kv_retained=True,
            )
    finally:
        protocol.close()


def receive_staging_consensus(args):
    """Use native D poll reduction: H2D and metadata gate before shared Success."""
    import ast

    import torch

    path = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/disaggregation/utils.py"
    )
    names = {
        "poll_and_all_reduce_with_staging",
        "_poll_with_failure_injection",
        "_apply_metadata_gate",
        "_is_fake_transfer",
        "_all_reduce_polls",
        "all_reduce_flags",
    }
    nodes = [
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names
    ]
    namespace = dict(
        torch=torch,
        dist=dist,
        KVPoll=SimpleNamespace(Success=4, Transferring=3, Failed=0),
        FAKE_BOOTSTRAP_HOST="fake",
        envs=SimpleNamespace(
            SGLANG_TEST_DISAGG_FAILURE_PROB=SimpleNamespace(get=lambda: 0)
        ),
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    state = SimpleNamespace(done=args.rank == 0, failed=False)
    dr = SimpleNamespace(
        kv_receiver=SimpleNamespace(require_staging=True, poll=lambda: 4),
        req=SimpleNamespace(bootstrap_host="real-peer"),
        metadata_buffer_index=0,
    )
    handler = SimpleNamespace(
        is_done=lambda req: state.done,
        is_failed=lambda req: state.failed,
        advance_scatter=lambda req: None,
    )
    metadata = SimpleNamespace(bootstrap_room=torch.ones((1, 1), dtype=torch.int64))
    polls, shapes = [], []
    actual = dist.all_reduce

    def counted(tensor, *a, **kw):
        shapes.append(tensor.numel())
        return actual(tensor, *a, **kw)

    with patch.object(dist, "all_reduce", counted):
        poll = namespace["poll_and_all_reduce_with_staging"]
        polls.append(poll([dr], handler, dist.group.WORLD, metadata))
        state.done = True
        if args.rank == 1:
            metadata.bootstrap_room.zero_()
        polls.append(poll([dr], handler, dist.group.WORLD, metadata))
        metadata.bootstrap_room.fill_(501)
        polls.append(poll([dr], handler, dist.group.WORLD, metadata))
        state.failed = args.rank == 1
        polls.append(poll([dr], handler, dist.group.WORLD, metadata))
    assert polls == [[3], [3], [4], [0]] and shapes == [1, 1, 1, 1]
    return dict(polls=polls, native_collective_shapes=shapes)


def sparse_remote_lifetimes(args):
    """Shared post-PD eager window and one TP retirement after last P reader."""
    from sglang_remote_attn import handoff_control as rt
    from sglang_remote_attn import hooks
    from sglang_remote_attn.dsv41_layerwise import DSV41Source
    from sglang_remote_attn.pre_layerwise import Config, Controller, write_json
    from sglang_remote_attn.prefill_overlap import SourceLease
    from sglang_remote_attn.retirement import Retirement
    from sglang_remote_attn.source_cache import CacheGuard

    owner = Controller.__new__(Controller)
    owner.rank, owner.tp_size, owner.dsv41_remote = args.rank, 2, True
    owner.scheduler = SimpleNamespace(attn_tp_cpu_group=dist.group.WORLD)
    owner.config = Config("remote", Path(args.output).with_suffix(".sparse"))
    registry = rt.HandoffRegistry()
    owner.scheduler._handoff_hub = SimpleNamespace(registry=registry)
    req = SimpleNamespace(
        rid="sparse",
        bootstrap_room=705,
        cache_request_handle="attempt",
        lock=object(),
        last_node="cache",
        kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=8),
        disagg_kv_sender=SimpleNamespace(poll=lambda: 4, kv_mgr=SimpleNamespace()),
    )
    pins = set()
    cache = SimpleNamespace(
        lock=lambda node: pins.add(node) or node,
        unlock=pins.remove,
        checkpoint=lambda *a, **kw: None,
        finish=lambda *a: None,
        reset=lambda: None,
    )
    guard = CacheGuard(cache)
    source = DSV41Source(
        owner,
        req,
        owner.config.directory,
        f"g{args.rank}",
        None,
        None,
        (),
        33,
        finalized=True,
        cache_lease=guard.hold(req),
    )
    req._pre_layerwise_source = source
    ctx = registry.register(
        "sparse",
        req,
        0,
        33,
        33,
        "token",
        on_release=lambda ctx: source.remote_closed.set(),
    )
    source.service = dict(handoff_id="sparse", epoch=ctx.owner_epoch)
    if args.rank == 1:
        registry.acquire("sparse", ctx.owner_epoch)
    write_json(source.path / "remote-done.json", dict(generation=source.generation))
    freed = []
    chunk = SimpleNamespace(req=req, sender=req.disagg_kv_sender, check=lambda: None)
    overlap = SimpleNamespace(
        controller=SimpleNamespace(sources=[source]),
        by_room={705: None},
        ledgers={},
        admission=SimpleNamespace(retired=lambda req: None),
    )
    lease = SourceLease(
        overlap,
        chunk,
        1,
        source.generation,
        source=source,
        finalized=True,
        cache_lease=source.cache_lease,
        release=lambda: freed.append(705),
    )
    owner.targets = [lease]
    protocol = Retirement(owner)
    protocol.register(lease)
    d = SimpleNamespace(
        dsv41_remote=True,
        dsv41_layerwise=True,
        remote_targets={0: True},
        targets=[],
        plan=None,
        executing=None,
        graph_fast_path=True,
    )
    runner = SimpleNamespace(model_runner=SimpleNamespace(_pre_layerwise=d))
    votes = []
    actual = dist.all_reduce

    def counted(tensor, *a, **kw):
        votes.append(tensor.numel())
        return actual(tensor, *a, **kw)

    try:
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": SimpleNamespace(
                        get_parallel=lambda: SimpleNamespace(
                            tp_size=2,
                            tp_rank=args.rank,
                            tp_group=SimpleNamespace(cpu_group=dist.group.WORLD),
                        )
                    ),
                    "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                        KVPoll=SimpleNamespace(Success=4, Failed=0)
                    ),
                },
            ),
            patch.object(dist, "all_reduce", counted),
        ):
            for _ in range(32):
                assert not hooks._decode_graph(lambda *a: True, runner, None)
                protocol.observe()
                assert protocol.poll_command() is None
            assert not votes and pins and not freed
            dist.barrier()  # Test phase boundary: rank 1 is still a live reader.
            if args.rank == 1:
                registry.release_ref("sparse", ctx.owner_epoch)
            command = None
            for _ in range(64):
                protocol.observe()
                message = [protocol.poll_command()]
                dist.broadcast_object_list(message, src=0)
                if message[0] is not None:
                    command = message[0]
                    break
            assert command is not None
            protocol.arm(command)
            protocol.finalize()
            assert votes == [5] and freed == [705] and not pins
            d.remote_targets.clear()
            for _ in range(32):
                assert hooks._decode_graph(lambda *a: True, runner, None)
            assert votes == [5]
            return dict(
                stable_remote_votes=0,
                final_vote_shapes=votes,
                freed=freed,
                local_graph_votes=0,
                pins_closed=True,
            )
    finally:
        protocol.close()


def sparse_progressive_lifetimes(args):
    """Real TP2 commands, staggered scatter/GPU and cancellation release edges."""
    import torch
    from sglang_remote_attn import dsv41_progressive as pg
    from sglang_remote_attn.decode_overlap import DecodeOverlap
    from sglang_remote_attn.pre_layerwise import Config, Controller

    owner = Controller.__new__(Controller)
    owner.rank, owner.tp_size, owner.graph_fast_path = args.rank, 2, True
    owner.dsv41_remote = True
    owner.scheduler = SimpleNamespace(
        attn_tp_cpu_group=dist.group.WORLD, enable_overlap=True
    )
    owner.config = Config("remote", Path(args.output).with_suffix(".progressive"))
    owner.targets, owner.remote_targets = [], {0: True}
    status = [3]
    dr = SimpleNamespace(
        req=SimpleNamespace(
            rid="progressive", bootstrap_room=801, kv=SimpleNamespace(req_pool_idx=0)
        ),
        metadata_buffer_index=0,
        _staging_completed_layers=[],
        kv_receiver=SimpleNamespace(poll=lambda: status[0], clear=lambda: None),
    )
    queue = SimpleNamespace(
        staging_handler=SimpleNamespace(
            advance_scatter=lambda dr: None,
            is_failed=lambda dr: False,
            is_done=lambda dr: status[0] == 4,
            unregister_decode_req=lambda room: None,
        ),
        metadata_buffers=SimpleNamespace(bootstrap_room=torch.tensor([[801]])),
        req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=lambda idx: None),
    )
    closed = []
    rpc = SimpleNamespace(
        local_entries=set(),
        calls=0,
        check=lambda: None,
        close=lambda reason: (closed.append(reason), owner.remote_targets.clear()),
    )
    target = pg.ProgressiveTarget(
        owner,
        queue,
        dr,
        owner.config.directory,
        f"pg-{args.rank}",
        SimpleNamespace(prefix_len=33),
        native_staging=True,
        rpc=rpc,
    )
    resources = [pg.Resource(target, i, i + 1) for i in (0, 1)]
    target.resources = resources
    target.admission_id = 3
    gpu = [False]
    target.forward_proofs = [SimpleNamespace(ready=lambda: gpu[0])]
    owner.targets = [*resources, target]
    overlap = DecodeOverlap(owner)
    owner.decode_overlap = overlap
    assert overlap.serial()
    protocol = overlap.retirement
    for item in owner.targets:
        protocol.register(item)
    votes = []
    actual = dist.all_reduce

    def counted(tensor, *a, **kw):
        votes.append(tensor.numel())
        return actual(tensor, *a, **kw)

    def command():
        for _ in range(128):
            protocol.observe()
            message = [protocol.poll_command()]
            dist.broadcast_object_list(message, src=0)
            if message[0] is not None:
                protocol.arm(message[0])
                protocol.finalize()
                return
            time.sleep(0.001)
        raise AssertionError("ordered progressive command missing")

    def stable():
        for _ in range(32):
            protocol.observe()
            assert protocol.poll_command() is None

    freed = []
    try:
        with (
            patch.object(dist, "all_reduce", counted),
            patch(
                "sglang_remote_attn.pre_layerwise._runtime",
                return_value=SimpleNamespace(_drop_target_state_for=lambda req: None),
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                        KVPoll=SimpleNamespace(Success=4, Failed=0)
                    )
                },
            ),
        ):
            if args.rank == 0:
                dr._staging_completed_layers = [((0,), 0, 5)]
            stable()
            dist.barrier()
            assert not votes and not rpc.local_entries
            if args.rank == 1:
                dr._staging_completed_layers = [((0,), 0, 5)]
            command()
            assert rpc.local_entries == {0} and votes == [5]
            dr._staging_completed_layers.append(((1,), 0, 5))
            command()
            assert rpc.local_entries == {0, 1} and votes == [5, 5]
            status[0] = 4
            if args.rank == 0:
                target.release = lambda: freed.append("KV")
            stable()
            dist.barrier()  # Release mismatch never frees or votes per loop.
            if args.rank == 1:
                target.release = lambda: freed.append("KV")
            command()  # Both release, but one or both GPU proofs still pending.
            assert not freed and votes == [5, 5, 5]
            gpu[0] = True
            command()
            assert freed == ["KV"] and not owner.targets and not owner.remote_targets
            stable()
            assert overlap.serial()  # Final Remote result boundary.
            assert not overlap.serial()
            return dict(
                votes=votes,
                freed=freed,
                local_entries=sorted(rpc.local_entries),
                retries=protocol.retries,
                closed=len(closed),
                stable_votes=0,
            )
    finally:
        protocol.close()


def dsv41_graph_settings(args):
    """Exchange native ABI/capture choices before any segmented capture barrier."""
    import importlib.util

    from sglang_remote_attn.dsv41_graph import capture_signature
    from sglang_remote_attn.segmented_graph import agree_capture_settings

    spec = importlib.util.spec_from_file_location(
        "dsv41_graph_probe_support",
        Path(__file__).with_name("dsv41_graph_probe_support.py"),
    )
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    f = support.fixture()
    outcomes = []
    for scenario in ("same", "flag", "buckets", "abi", "variants"):
        f.runner.attention_graph_variants = (
            SimpleNamespace(capture_labels=("candidate_filtered",), graph_limits=())
            if scenario == "variants" and args.rank == 1
            else None
        )
        abi = capture_signature(f.mr, f.runner)
        if scenario == "abi" and args.rank == 1:
            abi = (False, abi[1])
        report = (
            False if scenario == "flag" and args.rank == 1 else True,
            (1, 2) if scenario == "buckets" and args.rank == 1 else (1, 2, 4),
            abi,
        )
        try:
            agree_capture_settings(report, group=dist.group.WORLD, world_size=2)
            outcomes.append(True)
        except RuntimeError:
            outcomes.append(False)
    assert outcomes == [True, False, False, False, False]
    from sglang_remote_attn import hooks
    from sglang_remote_attn.dsv41_graph import DSV41GraphBackend

    backend = DSV41GraphBackend.__new__(DSV41GraphBackend)
    backend.remote_keys = {n: SimpleNamespace(size=n) for n in (1, 2, 4)}
    backend.remote = SimpleNamespace(can_run=lambda *a: True)
    f.runner.backend = backend
    policies = []
    votes = []
    native_reduce = dist.all_reduce

    def counted(*a, **kw):
        votes.append(True)
        return native_reduce(*a, **kw)

    context = SimpleNamespace(
        get_parallel=lambda: SimpleNamespace(tp_size=2, tp_rank=args.rank)
    )
    with (
        patch.dict(sys.modules, {"sglang.srt.runtime_context": context}),
        patch.object(dist, "all_reduce", counted),
    ):
        for stage in ("remote", "partial-local", "all-copied", "retired"):
            f.controller.targets = [] if stage == "retired" else [object()]
            f.controller.remote_targets = {} if stage == "retired" else {0: True}
            policies.append(
                [
                    hooks._decode_graph(lambda *a: True, f.runner, f.batch(n))
                    for n in (1, 2, 3, 4, 8)
                ]
            )
        assert not votes
    assert policies == [[True, True, False, True, False]] * 3 + [[True] * 5]
    return dict(settings=outcomes, policies=policies, per_decode_votes=len(votes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("rendezvous")
    parser.add_argument("output")
    args = parser.parse_args()
    dist.init_process_group(
        "gloo",
        init_method=Path(args.rendezvous).as_uri(),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    polls = []

    def poll():
        polls.append(args.rank)
        return {
            "__handoff_ctl__": True,
            "data": {"cmd": "import", "nonce": "same-on-all-ranks"},
        }

    try:
        requests = inject_control(
            ["request"] if args.rank == 0 else None, leader=args.rank == 0, poll=poll
        )
        # These are the upstream work-channel timeout aborts, appended after
        # plugin injection and before the same CPU-group broadcast.
        if requests is not None:
            requests = [*requests, "timeout-abort"]
        message = [requests]
        dist.broadcast_object_list(message, src=0)
        outcomes = []
        for mode in ("success", "failed-shard", "changed-schedule", "both-reject"):
            result = {
                "ok": not (
                    mode == "both-reject" or mode == "failed-shard" and args.rank == 1
                )
            }
            if not result["ok"]:
                result["error"] = "injected import rejection"
            state = (
                ("request",)
                if not (mode == "changed-schedule" and args.rank == 1)
                else ()
            )
            agreed, fatal = agree_control_result(
                result, state, group=dist.group.WORLD, world_size=2
            )
            outcomes.append({"mode": mode, "ok": agreed["ok"], "fatal": fatal})
        # Real two-rank retirement: the faster rank must not return allocator
        # capacity before its peer has drained, nor shrink the next collective.
        from sglang_remote_attn.pre_layerwise import Config, Controller, Source

        controller = Controller.__new__(Controller)
        controller.config = Config("remote", Path(args.output).with_suffix(".journal"))
        controller.scheduler = SimpleNamespace(attn_tp_cpu_group=dist.group.WORLD)
        controller.tp_size = 2
        from sglang_remote_attn.source_cache import CacheGuard

        released = []
        pins = set()
        cache = SimpleNamespace(
            checkpoint=lambda *a, **kw: None,
            finish=lambda *a: None,
            reset=lambda: None,
            lock=lambda node: pins.add(node) or node,
            unlock=pins.remove,
        )
        guard = CacheGuard(cache)
        sources = []
        for name in ("a", "b"):
            req = SimpleNamespace(
                disagg_kv_sender=SimpleNamespace(poll=lambda: 4),
                cache_request_handle=name,
                lock=name,
                last_node=name,
                kv=SimpleNamespace(cache_protected_len=1),
            )
            source = Source(
                controller,
                req,
                controller.config.directory / name,
                name,
                [],
                slice(0, 1),
                (),
                1,
                responded=True,
                finalized=True,
                release=lambda name=name: released.append(name),
                cache_lease=guard.hold(req),
            )
            req._pre_layerwise_source = source
            sources.append(source)
        controller.sources = list(sources)
        retirements = []
        pin_counts = []
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.schedule_batch": SimpleNamespace(
                    FINISH_LENGTH=None
                ),
                "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                    KVPoll=SimpleNamespace(Success=4, Failed=0)
                ),
            },
        ):
            for step in range(3):
                if step >= args.rank:
                    sources[0].remote_closed.set()
                if step >= 2 - args.rank:
                    sources[1].remote_closed.set()
                controller.poll_sources()
                retirements.append(list(released))
                pin_counts.append(len(pins))
        # Exercise the actual plugin can_run_graph hooks over real Gloo. The
        # native predicate is rank-local too, so it must participate in the vote.
        from sglang_remote_attn import handoff_control as runtime
        from sglang_remote_attn import hooks

        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(
                tp_size=2,
                tp_rank=args.rank,
                tp_group=SimpleNamespace(cpu_group=dist.group.WORLD),
            )
        )
        event = threading.Event()
        state = SimpleNamespace(
            mode="remote", events=[event], error=None, aborted=False
        )
        decode, prefill = SimpleNamespace(), SimpleNamespace()
        graphs = []
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": context}):
            for step in range(9):
                # Start Local, then admit only rank 1's target, then give both
                # ranks a target with rank 0 already ready. Later only rank 0
                # sees a newly admitted incomplete target.
                states = [state] if step in (1, 2, 3, 4, 5, 6, 7) else []
                if (step == 1 and args.rank == 0) or (step == 6 and args.rank == 1):
                    states = []
                if (step >= 3 and step != 6) or (step == 2 and args.rank == 0):
                    event.set()
                else:
                    event.clear()
                state.error = "failed import" if step == 7 and args.rank == 1 else None
                eligible = not (step == 4 and args.rank == 1)
                with patch.object(runtime, "_TARGET_STATES", states):
                    graphs.append(
                        [
                            hooks._decode_graph(lambda *a: eligible, decode, None),
                            hooks._prefill_graph(lambda *a: eligible, prefill, None),
                        ]
                    )
        leases = progressive_graph_leases(args, context)
        from sglang_remote_attn.segmented_graph import (
            RemoteGraphBackend,
            agree_capture_settings,
        )

        startup = []
        for scenario in ("same", "flag-mismatch", "bucket-mismatch"):
            report = (
                not (scenario == "flag-mismatch" and args.rank == 1),
                (1, 2)
                if scenario == "bucket-mismatch" and args.rank == 1
                else (1, 2, 4),
            )
            try:
                agree_capture_settings(report, group=dist.group.WORLD, world_size=2)
                startup.append(True)
            except RuntimeError:
                startup.append(False)
        controller = SimpleNamespace(graph_fast_path=True, targets=[])
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(_pre_layerwise=controller)
        )
        backend = RemoteGraphBackend.__new__(RemoteGraphBackend)
        backend.runner, backend.remote_keys = (
            runner,
            {size: SimpleNamespace(size=size) for size in (1, 2, 4)},
        )
        backend.remote = SimpleNamespace(can_run=lambda *a: True)
        runner.backend = backend
        routing = []
        import torch

        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": context}),
            patch.object(
                dist, "all_reduce", side_effect=AssertionError("per-forward graph vote")
            ),
        ):
            for remote_active in (False, True, False, True):
                controller.targets = [object()] if remote_active else []
                row = []
                for size in (1, 2, 3, 4, 8):
                    batch = SimpleNamespace(
                        batch_size=size,
                        input_ids=torch.zeros(size, dtype=torch.int64),
                        input_embeds=None,
                        replace_embeds=None,
                        forward_mode=SimpleNamespace(is_decode=lambda: True),
                    )
                    row.append(hooks._decode_graph(lambda *a: True, runner, batch))
                routing.append(row)
        overlap_lifetimes = decode_overlap_lifetimes(args)
        p_overlap = prefill_overlap_lifetimes(args)
        p_certificates = prefill_certificate_lifetimes(args)
        p_cached_certificates = prefill_certificate_lifetimes(args, prefix_cache=True)
        p_staging_failure = prefill_staging_failure(args)
        d_receive_staging = receive_staging_consensus(args)
        sparse_remote = sparse_remote_lifetimes(args)
        sparse_progressive = sparse_progressive_lifetimes(args)
        dsv41_segments = dsv41_graph_settings(args)
        Path(args.output).write_text(
            json.dumps(
                {
                    "requests": message[0],
                    "polls": polls,
                    "outcomes": outcomes,
                    "retirements": retirements,
                    "p_radix_pins": pin_counts,
                    "graphs": graphs,
                    "leases": leases,
                    "d_overlap": overlap_lifetimes,
                    "p_overlap": p_overlap,
                    "p_certificates": p_certificates,
                    "p_cached_certificates": p_cached_certificates,
                    "p_staging_failure": p_staging_failure,
                    "d_receive_staging": d_receive_staging,
                    "sparse_remote": sparse_remote,
                    "sparse_progressive": sparse_progressive,
                    "dsv41_segments": dsv41_segments,
                    "segmented_startup": startup,
                    "segmented_routing": routing,
                }
            )
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
