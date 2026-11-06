"""Real native paged allocator kernels and PD preallocation on small Qwen pools."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from qwen_reservation_probe_support import Fixture, Sampling
from sglang_remote_attn import handoff_control as rt, qwen_reservation as reservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    results = []
    for page in (1, 4, 16):
        f = Fixture(args.source_root, "cuda", page=page, capacity=128)
        try:
            req = f.new(f.scheduler, "source", list(range(5)), [11, 12, 13], Sampling())
            req.kv.req_pool_idx = 0
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 7
            f.scheduler.handoff_tracked_reqs[req.rid] = req
            before = f.allocator.available_size()
            p = reservation.plan(
                f.scheduler,
                dict(
                    epoch="native-range",
                    items=[dict(rid="source", target_rid="target", growth_tokens=16)],
                ),
            )["plans"][0]
            with patch.object(
                f.queue, "_pre_alloc", wraps=f.queue._pre_alloc
            ) as allocate:
                result = reservation.reserve(
                    f.scheduler, dict(epoch=p["epoch"], plans=[[p]])
                )
                assert allocate.call_count == 1
                expected = (p["upper"] + page - 1) // page * page
                assert before - f.allocator.available_size() == expected
                reservation.reserve(f.scheduler, dict(epoch=p["epoch"], plans=[[p]]))
                assert allocate.call_count == 1
                reservation.confirm(
                    f.scheduler,
                    dict(epoch=p["epoch"], certificates=[result["reservations"]]),
                )
                req.output_ids.extend([14, 15])
                req.kv.kv_committed_len = req.kv.kv_allocated_len = 9
                reservation.check_freeze(f.scheduler, req, p["epoch"])
                frozen = dict(p, kv_committed_len=9, output_ids=list(req.output_ids))
                frozen.update(
                    d2d_range_protocol="layer-token-range-v1",
                    chunks=[[9]] * 3,
                    kv_heads=1,
                    head_dim=8,
                    dtype="float16",
                )
                target, mapping = reservation.consume(
                    f.scheduler, "target", frozen, p["epoch"]
                )
                held = (9 + page - 1) // page * page
                assert before - f.allocator.available_size() == held
                row = target.kv.req_pool_idx
                generation = int(f.pool.req_generation[row])
                rt.bind_decode_prefix(target, f.pool, mapping, 9)
                assert (
                    target.kv.req_pool_idx == row
                    and int(f.pool.req_generation[row]) == generation
                )
                assert torch.equal(f.pool.req_to_token[row, :9].long(), mapping.long())
                rt._free_source_request(f.scheduler, target)
                rt._free_source_request(f.scheduler, target)
                assert (
                    f.allocator.available_size() == before
                    and f.pool.available_size() == 3
                )
                assert allocate.call_count == 1
                results.append(
                    dict(
                        page_size=page,
                        upper=p["upper"],
                        frozen_kv=9,
                        allocated_tokens=expected,
                        retained_tokens=held,
                        native_allocations=1,
                        row_reused=True,
                        whole_tail_pages_trimmed=True,
                        source_boundary_checked=True,
                        free_once=True,
                    )
                )
        finally:
            f.close()
    report = dict(
        status="PASS",
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cases=results,
        scope=__doc__,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", cases=len(results))))


if __name__ == "__main__":
    main()
