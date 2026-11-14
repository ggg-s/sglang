"""Real two-rank Gloo votes around D2D frozen rows and asymmetric completion."""

import argparse
import json
import sys
import time
import types
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch.distributed as dist

# Load lightweight core transport contracts without importing SGLang's serving
# package in these isolated CPU/Gloo pool fixtures.
_package = types.ModuleType("sglang")
_package.__path__ = [str(Path(__file__).resolve().parents[3] / "python/sglang")]
sys.modules.setdefault("sglang", _package)
from dsv_d2d_probe_support import Environment
from sglang_remote_attn import dsv_d2d as d2d
from sglang_remote_attn.pre_layerwise import Controller
from sglang_remote_attn.retirement import Retirement


def partial_freeze(root, rank):
    from dsv_d2d_partial_probe_support import SharedEnvironment

    e = SharedEnvironment(root, 4)
    try:
        for c in e.nodes:
            c.decode_overlap.retirement.close()
            c.rank, c.tp_size = rank, 2
            c.scheduler.attn_tp_cpu_group = dist.group.WORLD
            c.agree = Controller.agree.__get__(c)
            protocol = c.decode_overlap.retirement = Retirement(c)
            for target in c.targets:
                protocol.register(target)
            e.stack.callback(protocol.close)
        commands = []
        for i, req in enumerate(e.requests):
            common = dict(
                migration_epoch=f"partial-{i}",
                destination="D1",
                target_rid=f"partial-target-{i}",
            )
            plan = d2d.handle(
                e.source, dict(cmd="d2d_plan", rid=req.rid, growth_tokens=0, **common)
            )["plan"]
            plans = [None, None]
            dist.all_gather_object(plans, plan)
            ticket = d2d.handle(
                e.target, dict(cmd="d2d_reserve", plans=plans, **common)
            )
            certificates = [None, None]
            dist.all_gather_object(certificates, ticket["reservation"])
            cmd = dict(
                cmd="d2d_freeze", rid=req.rid, reservations=certificates, **common
            )
            assert d2d.handle(e.source, cmd)["pending_retirement"]
            commands.append(cmd)
        proof = [rank == 0]
        e.parents[0].forward_proofs.append(NS(ready=lambda: proof[0]))
        for i in range(4):
            e.arrive(i)

        def pump():
            e.handler.flush_pending_scatters(e.stream)
            p = e.source.decode_overlap.retirement
            p.observe()
            command = [p.poll_command() if rank == 0 else None]
            dist.broadcast_object_list(command, src=0)
            if command[0]:
                p.arm(command[0])
                p.finalize()

        # Rank 0 has all bytes/proofs, but rank 1's last GPU consumer is live.
        pump()
        source = e.source.d2d_sources["partial-0"]
        assert d2d.handle(e.source, commands[0])["pending_retirement"]
        assert source.service is None and not e.parents[0].closed and not e.freed
        proof[0] = True
        deadline = time.monotonic() + 5
        while not e.source.agree([int(all(t.closed for t in e.parents))])[0]:
            assert time.monotonic() < deadline
            pump()
        for cmd in commands:
            reply = d2d.handle(e.source, cmd)
            assert reply["transfer"]["source_partial_remote"]
        assert not e.allocator.allocations and len(e.metadata_freed) == 4
        assert not e.freed
        for cmd in commands:
            assert d2d.handle(
                e.source, dict(cmd="d2d_resume", migration_epoch=cmd["migration_epoch"])
            )["resumed"]
        return True
    finally:
        e.close()


def pre_sample_reservation(root, rank):
    """Real TP2 votes keep an unsampled source runnable while D1 owns capacity."""
    e = Environment(root)
    try:
        for c in e.nodes:
            c.rank, c.tp_size = rank, 2
            c.scheduler.attn_tp_cpu_group = dist.group.WORLD
            c.agree = Controller.agree.__get__(c)
        e.req.output_ids.clear()
        e.req.kv.kv_committed_len = e.req.kv.kv_allocated_len = 0
        common = dict(
            migration_epoch="pre-sample",
            rid=e.req.rid,
            target_rid="target",
            destination="D1",
        )
        plan = d2d.handle(e.source, dict(cmd="d2d_plan", growth_tokens=8, **common))[
            "plan"
        ]
        plans = [None, None]
        dist.all_gather_object(plans, plan)
        reserved = d2d.handle(e.target, dict(cmd="d2d_reserve", plans=plans, **common))
        certificates = [None, None]
        dist.all_gather_object(certificates, reserved["reservation"])
        reply = d2d.handle(
            e.source, dict(cmd="d2d_freeze", reservations=certificates, **common)
        )
        assert reply["migration_deferred_reason"] == "sampled_output_pending"
        ticket = e.target.d2d_reservations[common["migration_epoch"]]
        assert ticket.req.kv.kv_committed_len == plan["upper"]
        assert not e.req.output_ids and not e.req._handoff_frozen and not e.freed
        assert not e.source.d2d_sources
        assert d2d.handle(e.target, dict(cmd="d2d_reserve_cancel", **common))[
            "cancelled"
        ]
        assert d2d.handle(e.source, dict(cmd="d2d_resume", **common))["resumed"]
        assert e.freed == ["target"] and not e.source.d2d_plans
        return True
    finally:
        e.close()


def precopy(root, rank):
    """A peer without a closed page prevents every shard's native read/write."""
    import torch
    from sglang_remote_attn import d2d_precopy as pc

    e = Environment(root)
    try:
        for c in e.nodes:
            c.decode_overlap.retirement.close()
            c.rank, c.tp_size = rank, 2
            c.scheduler.attn_tp_cpu_group = dist.group.WORLD
            c.agree = Controller.agree.__get__(c)
            c.decode_overlap.retirement = Retirement(c)
            e.stack.callback(c.decode_overlap.retirement.close)
        epoch = e.epoch = "precopy-tp"

        def cmd(c, name, **payload):
            return d2d.handle(
                c, dict(cmd="d2d_" + name, migration_epoch=epoch, **payload)
            )

        def gather(value):
            values = [None, None]
            dist.all_gather_object(values, value)
            return values

        plan = cmd(
            e.source,
            "plan",
            rid="source",
            target_rid="target",
            destination="D1",
            growth_tokens=16,
        )["plan"]
        plans = gather(plan)
        cmd(e.target, "reserve", plans=plans)
        frame = dict(entry=0, start=0, count=4, chunk="a" * 32)
        receive = cmd(e.target, "precopy_prepare", frame=frame)["receive"]
        receives = gather(receive)
        if rank == 1:
            e.req.kv.kv_committed_len = 31
        available = gather(cmd(e.source, "precopy_status")["available"])
        assert available[0][0] == 4 and available[1][0] == 3
        try:
            cmd(e.source, "precopy_copy", frame=frame, receives=receives)
            raise AssertionError("asymmetric stable page was read")
        except ValueError:
            pass
        assert pc.sources(e.source)[epoch].sender is None
        assert not e.req._handoff_frozen and not e.freed
        e.req.kv.kv_committed_len = 33
        receipts = []
        for i in range(len(plan["schema"]["entries"])):
            if i:
                frame = dict(entry=i, start=0, count=4, chunk=f"{i:032x}")
                receive = cmd(e.target, "precopy_prepare", frame=frame)["receive"]
                receives = gather(receive)
            proof = cmd(e.source, "precopy_copy", frame=frame, receives=receives)[
                "proof"
            ]
            assert (
                cmd(e.source, "precopy_copy", frame=frame, receives=receives)["proof"]
                == proof
            )
            proofs = gather(proof)
            receipt = cmd(e.target, "precopy_apply", frame=frame, proofs=proofs)[
                "receipt"
            ]
            assert (
                cmd(e.target, "precopy_apply", frame=frame, proofs=proofs)["receipt"]
                == receipt
            )
            receipts.append(receipt)
        meta = cmd(
            e.source,
            "freeze",
            rid="source",
            destination="D1",
            target_rid="target",
            reservations=plans,
            precopy_receipts=gather(receipts),
        )["transfer"]
        assert meta["precopy"]["prefix_pages"] == [4] * len(plan["schema"]["entries"])
        cmd(e.target, "import", destination="D1", transfers=gather(meta))
        target = e.target.d2d_targets[epoch]
        assert not target.admitted and not e.freed
        cmd(e.target, "commit")
        e.gate.set()
        e.wait_copy()
        for a, b in zip(pc.tables(e.source), pc.tables(e.target)):
            si = pc.page_ids(e.source, e.req, 0, 4)
            di = pc.page_ids(e.target, target.decode_req.req, 0, 4)
            assert torch.equal(a[si], b[di])
        e.target.drain_targets(required=target)
        source = e.source.d2d_sources[epoch]
        deadline = time.monotonic() + 5
        while not e.source.agree([int(source.poll())])[0]:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        source.retire()
        assert e.freed == ["source"] and not pc.bulk.owners(
            d2d.manager(e.source).engine
        )
        return True
    finally:
        e.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("rendezvous")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dist.init_process_group(
        "gloo",
        init_method="file://" + args.rendezvous,
        world_size=2,
        rank=args.rank,
        timeout=timedelta(seconds=20),
    )
    root = Path(__file__).resolve().parents[3]
    precopy_ok = precopy(root, args.rank)
    pre_sample_ok = pre_sample_reservation(root, args.rank)
    partial_ok = partial_freeze(root, args.rank)
    e = Environment(root)
    try:
        for c in e.nodes:
            c.decode_overlap.retirement.close()
            c.rank, c.tp_size = args.rank, 2
            c.scheduler.attn_tp_cpu_group = dist.group.WORLD
            c.agree = Controller.agree.__get__(c)
            c.decode_overlap.retirement = Retirement(c)
            e.stack.callback(c.decode_overlap.retirement.close)

        def prepare(epoch):
            plan = d2d.handle(
                e.source,
                dict(
                    cmd="d2d_plan",
                    migration_epoch=epoch,
                    rid="source",
                    destination="D1",
                    target_rid="target",
                    growth_tokens=0,
                ),
            )["plan"]
            plans = [None, None]
            dist.all_gather_object(plans, plan)
            return plans

        # A stale worker_status can leave just one rank with Remote entries.
        # All shards must defer before any plan or new owner is published.
        if args.rank == 1:
            e.req._dsv41_remote = NS(all_local=lambda: False)
        deferred = d2d.handle(
            e.source,
            dict(
                cmd="d2d_plan",
                migration_epoch="remote-pending",
                rid="source",
                destination="D1",
                target_rid="target",
                growth_tokens=0,
            ),
        )
        assert deferred["deferred"]
        assert deferred["migration_deferred_reason"] == (
            "remote_entries_pending" if args.rank == 1 else "peer_not_ready"
        )
        assert not e.source.d2d_plans and not e.req._handoff_frozen and not e.freed
        e.req._dsv41_remote = None

        # A request may finish on either shard after the status snapshot.
        # Both ranks must return the same window-loss result without publishing
        # an owner, changing batch membership, or releasing source KV.
        for finished_rank in (0, 1):
            batch = e.source.scheduler.running_batch
            before = list(batch.reqs)
            with patch.object(
                e.req, "finished", return_value=args.rank == finished_rank
            ):
                skipped = d2d.handle(
                    e.source,
                    dict(
                        cmd="d2d_freeze",
                        migration_epoch=f"finished-window-{finished_rank}",
                        rid="source",
                        destination="D1",
                        target_rid="target",
                    ),
                )
            assert skipped == dict(
                deferred=True,
                migration_deferred_reason="request_finished",
                retryable=False,
            )
            assert not e.source.d2d_sources and not e.req._handoff_frozen
            assert batch.reqs == before and e.req.kv.holds_kv and not e.freed

        # One rank lacks capacity: neither shard may freeze or allocate a row.
        plans = prepare("no-capacity")
        available = e.target.scheduler.req_to_token_pool.available_size
        if args.rank == 1:
            e.target.scheduler.req_to_token_pool.available_size = lambda: 0
        try:
            d2d.handle(
                e.target,
                dict(
                    cmd="d2d_reserve",
                    migration_epoch="no-capacity",
                    plans=plans,
                ),
            )
            raise AssertionError("asymmetric capacity accepted")
        except ValueError as exc:
            assert "reserve preflight" in str(exc)
        finally:
            e.target.scheduler.req_to_token_pool.available_size = available
        assert not e.req._handoff_frozen and not e.freed
        assert not e.target.d2d_reservations

        e.epoch = "migration"
        plans = prepare(e.epoch)
        reserved = d2d.handle(
            e.target,
            dict(
                cmd="d2d_reserve",
                migration_epoch=e.epoch,
                plans=plans,
            ),
        )
        assert reserved["reserved"] and not e.req._handoff_frozen
        certificates = [None, None]
        dist.all_gather_object(certificates, reserved["reservation"])
        meta = d2d.handle(
            e.source,
            dict(
                cmd="d2d_freeze",
                migration_epoch=e.epoch,
                rid="source",
                destination="D1",
                target_rid="target",
                reservations=certificates,
            ),
        )["transfer"]
        transfers = [None, None]
        dist.all_gather_object(transfers, meta)
        assert d2d.handle(
            e.target,
            dict(
                cmd="d2d_import",
                migration_epoch=e.epoch,
                destination="D1",
                transfers=transfers,
                async_handoff=True,
            ),
        )["import_pending"]
        target = e.target.d2d_targets[e.epoch]
        target.setup_worker.join(5)
        assert not target.setup_worker.is_alive()
        assert e.command(e.target, "d2d_import")["imported"]
        call = target.reader.call

        def partial_commit(action, **fields):
            if args.rank == 1 and action == "commit":
                raise ConnectionError("injected missing source confirmation")
            return call(action, **fields)

        def commit():
            return d2d.handle(
                e.target,
                dict(
                    cmd="d2d_control_wave",
                    migration_epoch="worker",
                    commands=[
                        dict(cmd="d2d_commit", migration_epoch=e.epoch),
                        dict(cmd="d2d_commit", migration_epoch="absent-epoch"),
                    ],
                ),
            )["results"]

        target.reader.call = partial_commit
        # Rank 1 has no background slot. Rank 0 must wait instead of issuing
        # its source confirmation on an epoch that its peer cannot start.
        if args.rank == 1:
            e.target.d2d_targets.update(
                {
                    f"busy-{i}": NS(
                        setup_worker=NS(is_alive=lambda: True), commit_worker=None
                    )
                    for i in range(d2d.ASYNC_HANDOFF_LIMIT)
                }
            )
        assert commit()[e.epoch]["commit_pending"]
        assert target.commit_worker is None and not target.commit_started
        if args.rank == 1:
            for i in range(d2d.ASYNC_HANDOFF_LIMIT):
                del e.target.d2d_targets[f"busy-{i}"]

        with patch.object(e.target, "agree", wraps=e.target.agree) as agree:
            reply = commit()
            assert agree.call_count == 1  # One prepare collective for the wave.
        assert reply[e.epoch]["commit_pending"]
        assert reply["absent-epoch"]["rejected"]
        target.commit_worker.join(5)
        assert not target.commit_worker.is_alive()
        assert "confirmation pending" in commit()[e.epoch]["error"]
        assert not target.admitted and not e.resumed
        target.reader.call = call
        assert commit()[e.epoch]["commit_pending"]
        if target.commit_worker is not None:
            target.commit_worker.join(5)
            assert not target.commit_worker.is_alive()
        assert commit()[e.epoch]["admitted"]
        assert len(e.resumed) == 1
        source = e.source.d2d_sources[e.epoch]
        assert not target.copied and not e.freed
        e.gate.set()
        e.wait_copy()
        proof = [args.rank == 0]
        target.forward_proofs.append(NS(ready=lambda: proof[0]))
        protocol = e.target.decode_overlap.retirement
        message = [
            protocol._command(
                [dict(**protocol.identity(target), generations=None, cache=None)],
                "drain",
            )
            if args.rank == 0
            else None
        ]
        dist.broadcast_object_list(message, src=0)
        protocol.arm(message[0])
        protocol.finalize()
        assert not target.closed and not source.poll() and not e.freed
        proof[0] = True
        e.target.drain_targets(required=target)
        deadline = time.monotonic() + 3
        while not source.poll():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        # Last reader exists only on one TP shard. A MIN vote cannot free either.
        ready = [args.rank == 0]
        assert not e.source.agree([int(ready[0])])[0]
        assert not e.freed
        assert e.source.agree([int(source.poll())])[0]
        source.retire()
        source.retire()
        assert e.freed == ["source"]
        # A real TP quarantine command must retain the target allocation on both
        # ranks even though only rank 1 lost a transport completion proof.
        # Use a fresh prepared target so no previous Local transition can hide it.
        e.source.scheduler.running_batch.reqs = [e.req]
        # The separate request fixture makes this a new generation/row owner.
        from dsv_d2d_probe_support import Sampling, request

        e.req = request("source2", list(range(e.prefix - 2)), [31, 32, 33], Sampling())
        e.req.kv = NS(
            req_pool_idx=0,
            kv_committed_len=e.prefix,
            kv_allocated_len=e.prefix,
            cache_protected_len=0,
            holds_kv=True,
            is_kv_released=False,
        )
        e.source.scheduler.running_batch.reqs = [e.req]
        e.source.scheduler.handoff_tracked_reqs["source2"] = e.req
        # Reuse the actual handle API with a distinct rid on both endpoints.
        meta = d2d.handle(
            e.source,
            dict(
                cmd="d2d_freeze",
                migration_epoch="failure",
                allow_unreserved=True,
                rid="source2",
                destination="D1",
                target_rid="target2",
            ),
        )["transfer"]
        metas = [None, None]
        dist.all_gather_object(metas, meta)
        d2d.handle(
            e.target,
            dict(
                cmd="d2d_import",
                migration_epoch="failure",
                destination="D1",
                transfers=metas,
            ),
        )
        bad = e.target.d2d_targets["failure"]
        bad.worker.join(5)
        assert not bad.worker.is_alive()
        if args.rank == 1:
            bad.error = "lost transfer proof"
        message = [
            protocol._command(
                [dict(**protocol.identity(bad), generations=None)], "quarantine"
            )
            if args.rank == 0
            else None
        ]
        dist.broadcast_object_list(message, src=0)

        # Native abort is the only mocked boundary; ordered quarantine and
        # defer_release holding are the real implementation.
        def abort(req, *a, **kw):
            req.finished = lambda: True

        with patch.dict(
            __import__("sys").modules,
            {"sglang.srt.disaggregation.utils": NS(prepare_abort=abort)},
        ):
            e.target.scheduler.output_streamer = NS(stream_output=lambda *a: None)
            protocol.arm(message[0])
            protocol.finalize()
        assert (
            getattr(bad.decode_req.req, "_remote_attn_transport_quarantine", None)
            is not None
        )
        assert e.freed == ["source"] and bad.decode_req.req.kv.holds_kv
        args.output.write_text(
            json.dumps(
                dict(
                    tp_partial_remote_waits_for_peer_gpu=partial_ok,
                    tp_partial_commit_never_admits=True,
                    reserved_before_freeze=True,
                    tp_finished_window_ranks=[0, 1],
                    tp_pre_sample_reservation=pre_sample_ok,
                    tp_precopy_asymmetric_pages_and_exact_bytes=precopy_ok,
                    tp_preflight_rejected_before_freeze=True,
                    asymmetric_gpu_blocks_both=True,
                    decode_before_copy=True,
                    source_free_count=1,
                    tp_quarantine_retains_both=True,
                )
            )
        )
    finally:
        e.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
