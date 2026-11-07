"""c=4/8 early preparation -> cut=32 -> partial PD retirement -> real D2D.

Small shared CUDA pools and native allocator/sampler ABI fixtures. Gateway
commands, receive ring/scatter, TP retirement and Mooncake TCP copies are real.
This is an ownership/control-order probe, not a model latency benchmark.
"""

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
package = types.ModuleType("sglang")
package.__path__ = [str(ROOT / "python/sglang")]
sys.modules.setdefault("sglang", package)

from dsv_d2d_partial_probe_support import SharedEnvironment
from p4_dsv_probe_support import Driver, gateway


def run(concurrency):
    e = SharedEnvironment(ROOT, concurrency, "cuda")
    gw = None
    try:
        for req in e.requests:
            req.output_ids[:] = []
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 0
            req.sampling_params.max_new_tokens = 128
        driver = Driver.__new__(Driver)
        driver.env = e
        driver.calls, driver.before, driver.after = [], {}, {}
        driver.controllers = dict(src=e.source, dst=e.target)
        gw = gateway(driver, ROOT, Path(e.tmp) / "gateway")
        gw.tick()
        rids = []
        for i, req in enumerate(e.requests):
            rid = gw.accept(
                dict(
                    rid=f"control-{i}",
                    prompt_ids=list(req.origin_input_ids),
                    max_new_tokens=128,
                    cut=32,
                    growth_tokens=0,
                )
            )
            gw._update(rid, phase="SOURCE", source_rid=req.rid)
            rids.append(rid)
        for _ in range(10):
            if all(gw.entry(rid)["phase"] == "WAIT_CUT" for rid in rids):
                break
            gw.tick()
        assert all(gw.entry(rid)["phase"] == "WAIT_CUT" for rid in rids)
        assert driver.count("d2d_plan") == driver.count("d2d_reserve") == concurrency
        assert driver.count("d2d_freeze") == 0
        assert all(not req._handoff_frozen for req in e.requests)
        epochs = [gw.entry(rid)["epoch"] for rid in rids]
        from sglang_remote_attn.source_resources import owned_requests

        tickets = [e.target.d2d_reservations[epoch] for epoch in epochs]
        assert {id(t.req.kv) for t in tickets} == {
            id(r.kv) for r in owned_requests(e.target)
        }
        assert all(gw.entry(rid)["plan_before_first_token"] for rid in rids)
        assert all(not req.output_ids for req in e.requests)
        # Model/sampling boundary: reach the cut on the same shared source batch.
        for req in e.requests:
            req.output_ids[:] = list(range(31, 63))
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 62
        gw.tick()
        assert all(gw.entry(rid)["phase"] == "FREEZE" for rid in rids)
        gw.tick()
        sources = [e.source.d2d_sources[epoch] for epoch in epochs]
        assert all(s.partial_remote and s.service is None for s in sources)
        assert all(req._handoff_frozen for req in e.requests)
        assert not e.source.scheduler.running_batch.reqs and not e.freed
        # The original P receiver must retire normally, even with no runnable D0.
        for i in range(concurrency):
            e.arrive(i)
        e.complete_pd()
        for _ in range(10):
            if all(gw.entry(rid)["phase"] == "TARGET" for rid in rids):
                break
            gw.tick()
        assert all(gw.entry(rid)["phase"] == "TARGET" for rid in rids), [
            gw.entry(r) for r in rids
        ]
        targets = [e.target.d2d_targets[epoch] for epoch in epochs]
        owners = owned_requests(e.target)
        assert {id(t.decode_req.req.kv) for t in targets} == {id(r.kv) for r in owners}
        assert not ({id(t.req) for t in tickets} & {id(r) for r in owners})
        assert len({t.decode_req.req.kv.req_pool_idx for t in targets}) == concurrency
        assert all(s.committed and not s.poll() for s in sources)
        assert not e.freed and len(e.resumed) == concurrency
        for t in targets:
            assert t.decode_req.req.output_ids == list(range(31, 63))
            t.decode_req.req.output_ids.append(63)
            t.decode_req.req.kv.kv_committed_len += 1
            t.decode_req.req.kv.kv_allocated_len += 1
        gw.tick()
        for rid in rids:
            events, _ = gw.events(rid, after=31)
            assert [(x["seq"], x["token_id"], x["owner"]) for x in events] == [
                (32, 63, "dst")
            ]
        e.gate.set()
        for t in targets:
            t.worker.join(15)
            assert not t.worker.is_alive() and not t.error, t.error
            e.target.drain_targets(required=t)
        deadline = time.monotonic() + 10
        while not all(s.poll() for s in sources):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert not e.freed
        for s in sources:
            s.retire()
            s.retire()
        gw.tick()
        assert all(gw.entry(rid)["phase"] == "LOCAL" for rid in rids)
        assert len(e.freed) == concurrency and len(set(e.freed)) == concurrency
        return dict(
            concurrency=concurrency,
            cut=32,
            pre_sample_preparations=concurrency,
            reservation_owners_accounted=concurrency,
            consumed_owners_transferred=True,
            migrations=concurrency,
            partial_remote_frozen=concurrency,
            destination_first_tokens=concurrency,
            source_free_count=len(e.freed),
            returned_reservation_tokens=len(e.returned_tokens[1]),
            handoff_observations=gw.latency_metrics()["handoff_latency_ms"]["count"],
            freeze_ack_observations=sum(
                gw.entry(rid)["freeze_ack_latency_ms"] is not None for rid in rids
            ),
            source_native_pd_retired_before_export=True,
            source_retained_until_reader_retirement=True,
            mooncake_transfers=len(e.transport.calls),
        )
    finally:
        if gw:
            gw.close()
        e.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = dict(
        results=[run(c) for c in (4, 8)],
        full_model=False,
        real_rdma=False,
        control_driver="in-process command dispatch, not file RPC latency",
        sampler="native ABI boundary fixture",
    )
    paths = [
        "plugins/remote_attn/control_plane/p4_dsv_gateway.py",
        "plugins/remote_attn/sglang_remote_attn/dsv_d2d.py",
        "plugins/remote_attn/sglang_remote_attn/dsv_d2d_reservation.py",
        "plugins/remote_attn/sglang_remote_attn/dsv_gateway_control.py",
        "plugins/remote_attn/sglang_remote_attn/source_resources.py",
        "test/manual/remote_attn/dsv_d2d_partial_probe_support.py",
        "test/manual/remote_attn/dsv_d2d_control_cuda_probe.py",
    ]
    result["sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
