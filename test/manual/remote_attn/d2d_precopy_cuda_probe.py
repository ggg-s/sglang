"""c=4/8 live stable-page copies using real CUDA scatters and Mooncake TCP.

Native allocator/sampler ABI fixtures, not full model serving or H20/RDMA.
"""

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[3]
package = types.ModuleType("sglang")
package.__path__ = [str(ROOT / "python/sglang")]
sys.modules.setdefault("sglang", package)

from dsv_d2d_partial_probe_support import SharedEnvironment
from p4_dsv_probe_support import Driver, gateway
from sglang_remote_attn import d2d_precopy as pc
from sglang_remote_attn import dsv_d2d as d2d


def run(concurrency):
    e = SharedEnvironment(ROOT, concurrency, "cuda")
    gw = None
    try:
        e.source.scheduler.forward_stream = torch.cuda.current_stream()
        consumer_done = [False]
        for parent in e.parents:
            # Hold the original PD/RPC owner through pre-copy and freeze.
            parent.forward_proofs.append(NS(ready=lambda: consumer_done[0]))
        for i, req in enumerate(e.requests):
            req.output_ids.clear()
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 31
            req.sampling_params.max_new_tokens = 128
            e.arrive(i)
        driver = Driver.__new__(Driver)
        driver.env = e
        driver.calls, driver.before, driver.after = [], {}, {}
        driver.controllers = dict(src=e.source, dst=e.target)
        gw = gateway(driver, ROOT, Path(e.tmp) / "gateway", async_handoff=True)
        gw.tick()
        rids = []
        for i, req in enumerate(e.requests):
            rid = gw.accept(
                dict(
                    rid=f"precopy-{i}",
                    prompt_ids=req.origin_input_ids,
                    max_new_tokens=128,
                    cut=32,
                )
            )
            gw._update(rid, phase="SOURCE", source_rid=req.rid)
            rids.append(rid)
        # Drive the serving gateway's precopy scheduler with explicit polls.
        gw.background = True
        # Source can have no sampled token while closed PD pages are copied.
        deadline = time.monotonic() + 30
        while not all(
            gw.entry(rid).get("precopy_bytes", 0) > 0
            and gw.entry(rid).get("precopy_pending") is None
            for rid in rids
        ):
            assert time.monotonic() < deadline, [gw.entry(r) for r in rids]
            gw._refresh()
            gw.tick()
        assert not e.freed and not any(req._handoff_frozen for req in e.requests)
        assert all(not req.output_ids for req in e.requests)
        assert all(req.kv.holds_kv for req in e.requests)
        # Sampler ABI: source continues to cut after those same rows were copied.
        for req in e.requests:
            req.output_ids[:] = list(range(31, 63))
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 62
        while not all(gw.entry(r)["phase"] == "FREEZE" for r in rids):
            assert time.monotonic() < deadline, [gw.entry(r) for r in rids]
            gw._refresh()
            gw.tick()
        freeze_dispatched = time.monotonic()
        # Freeze every member for the partial-PD fixture before allowing the
        # prior receiver to retire, regardless of the serving tick budget.
        while not all(req._handoff_frozen for req in e.requests):
            assert time.monotonic() < deadline
            gw._refresh()
            gw.tick()
        epochs = [gw.entry(r)["epoch"] for r in rids]
        sources = [e.source.d2d_sources[x] for x in epochs]
        assert all(s.partial_remote and s.service is None for s in sources)
        assert all(req._handoff_frozen for req in e.requests) and not e.freed
        assert all(s.precopy_seal["receipts"] for s in sources)
        consumer_done[0] = True
        e.complete_pd()
        e.gate.set()
        while not all(gw.entry(r)["phase"] == "TARGET" for r in rids):
            assert time.monotonic() < deadline, [gw.entry(r) for r in rids]
            gw._refresh()
            gw.tick()
        targets = [e.target.d2d_targets[x] for x in epochs]
        assert all(t.async_handoff and t.source_confirmed for t in targets)
        for source, target in zip(sources, targets):
            target.worker.join(10)
            assert not target.error and not target.worker.is_alive(), target.error
            n = source.prefix_len // e.f.page
            si = pc.page_ids(e.source, source.req, 0, n)
            di = pc.page_ids(e.target, target.decode_req.req, 0, n)
            for a, b in zip(pc.tables(e.source), pc.tables(e.target)):
                assert torch.equal(a[si], b[di])
            target.decode_req.req.output_ids.append(63)  # continuation sampler ABI
        e.target.drain_targets()
        while not all(s.poll() for s in sources):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        for source in sources:
            source.retire()
            source.retire()
            e.source.sources.remove(source)
        released = time.monotonic()
        assert len(e.freed) == concurrency and len(set(e.freed)) == concurrency
        assert not any(req.kv.holds_kv for req in e.requests)
        before = sum(gw.entry(r)["precopy_bytes"] for r in rids)
        after = sum(s.postfreeze_bytes for s in sources)
        full_global = sum(
            (s.prefix_len + e.f.page - 1)
            // e.f.page
            * sum(x[-1] for x in s.schema["entries"])
            for s in sources
        )
        state_bytes = sum(
            sum(x["rows"] * x["stride"] for x in s.meta["states"]) for s in sources
        )
        assert after == full_global + state_bytes - before, (
            before,
            after,
            full_global,
            state_bytes,
        )
        assert not getattr(e.source, "d2d_precopy_unsafe", False)
        assert not getattr(e.target, "d2d_precopy_unsafe", False)
        assert not pc.bulk.owners(d2d.manager(e.source).engine)
        assert not pc.bulk.owners(d2d.manager(e.target).engine)
        return dict(
            concurrency=concurrency,
            migrated=concurrency,
            before_first_sample=True,
            source_continued_after_precopy=True,
            partial_remote_at_freeze=True,
            source_frees=len(e.freed),
            async_handoff=True,
            precopy_bytes=before,
            postfreeze_bytes=after,
            full_snapshot_bytes=full_global + state_bytes,
            freeze_dispatch_to_native_source_release_ms=(released - freeze_dispatched)
            * 1000,
            native_byte_equality=True,
            private_slots_released=True,
        )
    finally:
        if gw is not None:
            gw.close()
        e.close()


def cancelled_slot():
    """A real late Mooncake write cannot target a recycled native CUDA row."""
    from dsv_d2d_probe_support import Environment
    from d2d_precopy_probe_support import command, reserve
    import uuid

    e = Environment(ROOT, "cuda")
    try:
        reserve(e)
        frame = dict(entry=0, start=0, count=1, chunk=uuid.uuid4().hex)
        receive = command(e.target, "precopy", "precopy_prepare", frame=frame)[
            "receive"
        ]
        ticket = e.target.d2d_reservations["precopy"]
        private = ticket.precopy.receiver
        # No positive source-stop proof: retain the private registered backing,
        # while returning the native reservation after its own GPU fence.
        command(e.target, "precopy", "reserve_cancel")
        assert not ticket.req.kv.holds_kv and private.unsafe
        native = pc.tables(e.target)[0]
        native.fill_(199)
        torch.cuda.synchronize()
        sent = command(
            e.source, "precopy", "precopy_copy", frame=frame, receive=receive
        )["proof"]
        expected = pc.bulk.marker(private.nonce, private.sequence, sent["bytes"])
        deadline = time.monotonic() + 5
        while (
            bytes(
                private.tensor[sent["bytes"] : sent["bytes"] + pc.bulk.FOOTER].tolist()
            )
            != expected
        ):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert torch.all(native == 199).item()
        stopped = command(
            e.source, "precopy", "precopy_close", pending_chunk=frame["chunk"]
        )
        assert stopped["safe"] and e.req.kv.holds_kv and not e.req._handoff_frozen
        return dict(
            late_native_write_arrived=True,
            recycled_cuda_kv_unchanged=True,
            native_reservation_freed=True,
            private_slot_quarantined=True,
        )
    finally:
        e.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [run(c) for c in (4, 8)]
    cancellation = cancelled_slot()
    files = [
        "plugins/remote_attn/sglang_remote_attn/" + name + ".py"
        for name in ("d2d_precopy", "dsv_d2d", "dsv_d2d_reservation")
    ]
    args.output.write_text(
        json.dumps(
            dict(
                scope=__doc__,
                results=results,
                cancellation=cancellation,
                sha256={
                    f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                    for f in files
                },
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
