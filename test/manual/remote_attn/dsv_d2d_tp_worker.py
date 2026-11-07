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
        d2d.handle(
            e.target,
            dict(
                cmd="d2d_import",
                migration_epoch=e.epoch,
                destination="D1",
                transfers=transfers,
            ),
        )
        target = e.target.d2d_targets[e.epoch]
        call = target.reader.call

        def partial_commit(action, **fields):
            if args.rank == 1 and action == "commit":
                raise ConnectionError("injected missing source confirmation")
            return call(action, **fields)

        target.reader.call = partial_commit
        try:
            e.command(e.target, "d2d_commit")
            raise AssertionError("partial TP commit admitted")
        except d2d.TransportFailure:
            pass
        assert not target.admitted and not e.resumed
        target.reader.call = call
        e.command(e.target, "d2d_commit")
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
                    tp_partial_commit_never_admits=True,
                    reserved_before_freeze=True,
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
