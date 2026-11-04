"""Real two-rank Gloo votes around D2D frozen rows and asymmetric completion."""

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch.distributed as dist
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
        transfers = [None, None]
        dist.all_gather_object(transfers, e.freeze())
        d2d.handle(
            e.target,
            dict(
                cmd="d2d_import",
                migration_epoch=e.epoch,
                destination="D1",
                transfers=transfers,
            ),
        )
        e.command(e.target, "d2d_commit")
        target = e.target.d2d_targets[e.epoch]
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
        from dsv_d2d_probe_support import request, Sampling

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
