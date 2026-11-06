"""D2D commands + native state bytes + actual GPU Remote during blocked KV copy.

Small pools and scheduler/allocator ABI fixtures; no model weights, SM100
FlashMLA, native HTTP stream rerouting, or H20 RDMA performance claim.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from dsv_d2d_probe_support import Environment
from sglang_remote_attn import dsv_d2d as d2d, dsv41_remote as remote
from sglang_remote_attn import dsv41_paged as paged, dsv41_sparse_math as math
from sglang_remote_attn.decode_overlap import BatchProof


def run(root, ratio, gpu, kv_layout="v4", p4=False):
    e = Environment(root, "cuda", ratio, gpu, kv_layout=kv_layout)
    gateway = None
    try:
        f = e.f
        pool = f.d_pool
        # Compile before the request's read lease exists; allocator metadata and
        # byte formats are source-loaded native definitions.
        for p, role in ((f.p_pool, "prefill"), (pool, "decode")):
            paged.warmup(p, (3,), {2}, f.query.dtype, role)
        if p4:
            from p4_dsv_probe_support import Driver, gateway as make_gateway, until

            driver = Driver(e)
            gateway = make_gateway(driver, root, Path(e.tmp) / "gateway")
            gateway.tick()
            rid = gateway.accept(
                dict(
                    rid="cuda-p4",
                    prompt_ids=list(e.req.origin_input_ids),
                    max_new_tokens=8,
                    cut=2,
                    growth_tokens=0,
                )
            )
            until(gateway, rid, "COMMIT")
            e.epoch = gateway.entry(rid)["epoch"]
        else:
            meta = e.reserve_and_freeze()
            e.import_(meta)
        source = e.source.d2d_sources[e.epoch]
        target = e.target.d2d_targets[e.epoch]
        assert e.blocked.wait(5) and not target.copied
        for (_, a, ai), (_, b, bi) in zip(
            d2d.state_rows(e.source, e.req, e.prefix),
            d2d.state_rows(e.target, target.decode_req.req, e.prefix),
        ):
            assert torch.equal(a[ai.long()], b[bi.long()])
        assert not e.freed and not source.poll()
        # Independent union reference, before removing D1's global prefix.
        ids = f.core.sparse_page_indices(ratio)[0]
        live = ids[ids >= 0]
        cpp = pool.page_size // ratio
        layout = pool.get_extra_key_layout(3)
        raw = pool.get_extra_key_buffer(3)
        keys = math.read_slots(raw, live, cpp, layout)
        swa = math.read_slots(
            pool.get_swa_key_buffer_radix(3),
            f.core.swa_page_indices[0, :2],
            pool.get_swa_key_page_size(),
            pool.kv_layout,
        )
        expected = math.partial(f.query[0], torch.cat((swa, keys)), 512**-0.5, f.sink)[
            0
        ].to(f.query.dtype)
        # Every completed prefix row is absent locally, including duplicate
        # top-k positions. The incomplete compression block stays D1-owned.
        positions = (
            list(range(e.prefix // ratio))
            if ratio == 128
            else ([0, e.prefix // ratio - 1] if ratio == 4 else [0, 15])
        )
        table = e.target.scheduler.req_to_token_pool.req_to_token[1]
        pos = torch.tensor(positions, device="cuda", dtype=torch.long)
        physical = table[pos // cpp * pool.page_size] // pool.page_size
        selected = physical * cpp + pos % cpp
        data, base, scale = layout.row_addressing(cpp)
        for off, stride, width in (
            (0, data, layout.data_bytes),
            (base, scale, layout.scale_bytes),
        ):
            cols = (
                off
                + (selected % cpp)[:, None] * stride
                + torch.arange(width, device="cuda")[None, :]
            )
            raw[(selected // cpp)[:, None], cols] = 0
        poisoned = raw.clone()
        if p4:
            driver.after["d2d_commit"] = [TimeoutError("injected lost commit reply")]
            gateway.tick()
            assert gateway.entry(rid)["phase"] == "COMMIT" and source.committed
            gateway.close()
            gateway = make_gateway(driver, root, Path(e.tmp) / "gateway")
            until(gateway, rid, "TARGET")
            assert gateway.entry(rid)["owner"] == "dst" and len(e.resumed) == 1
            assert not gateway.events(rid, after=2)[0]
            assert driver.count("d2d_resume") == 0
        else:
            e.command(e.target, "d2d_commit")
        assert source.committed and not target.copied
        native_calls = []

        def native(*args, **kwargs):
            native_calls.append(True)
            return expected[None]

        kv = torch.zeros((1, 512), device="cuda", dtype=torch.bfloat16)
        got = remote.attention(
            native,
            f.backend,
            f.query,
            kv,
            kv,
            NS(layer_id=3, tp_q_head_num=2),
            f.fb,
            ratio,
            attn_sink=f.sink,
        )
        torch.testing.assert_close(got[0], expected, atol=0.015625, rtol=0.01)
        torch.cuda.synchronize()
        assert not native_calls and target.rpc.calls == 1 and not e.freed
        error = float((got[0].float() - expected.float()).abs().max())
        # Real delayed compute completion: finished copying is insufficient.
        delayed = torch.cuda.Stream()
        with torch.cuda.stream(delayed):
            torch.cuda._sleep(500_000_000)
            fence = torch.cuda.Event()
            fence.record()
        target.forward_proofs.append(
            BatchProof(result=NS(delay_sample_func=None), serial_done=fence)
        )
        e.gate.set()
        e.wait_copy()
        assert not target.ready_to_finish() and not source.poll() and not e.freed
        # Copy covers only completed prefix slots, leaving D1's new tail and
        # padding bitwise unchanged. Compare bytes outside those slot columns.
        mask = torch.zeros_like(raw, dtype=torch.bool)
        count = e.prefix // ratio
        allpos = torch.arange(count, device="cuda")
        phys = table[allpos // cpp * pool.page_size] // pool.page_size
        allids = phys * cpp + allpos % cpp
        for off, stride, width in (
            (0, data, layout.data_bytes),
            (base, scale, layout.scale_bytes),
        ):
            cols = (
                off
                + (allids % cpp)[:, None] * stride
                + torch.arange(width, device="cuda")[None, :]
            )
            mask[(allids // cpp)[:, None], cols] = True
        assert torch.equal(raw[~mask], poisoned[~mask])
        fence.synchronize()
        e.target.drain_targets(required=target)
        deadline = time.monotonic() + 5
        while not source.poll():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert not e.freed
        source.retire()
        source.retire()
        assert e.freed == [source.req.rid]
        if p4:
            gateway.tick()
            assert gateway.entry(rid)["phase"] == "LOCAL"
        local = remote.attention(
            native,
            f.backend,
            f.query,
            kv,
            kv,
            NS(layer_id=3),
            f.fb,
            ratio,
            attn_sink=f.sink,
        )
        assert native_calls == [True] and torch.equal(local, expected[None])
        return dict(
            ratio=ratio,
            kv_layout=kv_layout,
            gpu_query_transport=gpu,
            bulk_transport=d2d.bulk.PROTOCOL,
            p4_gateway=p4,
            gateway_restart_after_lost_commit=p4,
            native_decode_admissions=len(e.resumed),
            reserved_before_freeze=True,
            reserved_row_consumed=e.target.d2d_reservations[e.epoch].consumed,
            state_bytes_exact=True,
            decode_before_bulk_copy=True,
            remote_calls=target.rpc.calls,
            max_error=error,
            tail_and_padding_exact=True,
            delayed_forward_retained_kv=True,
            source_free_count=1,
            local_restored=True,
            mooncake_transfers=0 if e.transport is None else len(e.transport.calls),
        )
    finally:
        if gateway is not None:
            gateway.close()
        e.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratio", type=int, choices=(2, 4, 128))
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--p4", action="store_true")
    parser.add_argument("--kv-layout", choices=("v4", "v41"), default="v4")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.testing.assert_close(torch.zeros(1), torch.zeros(1))
    root = args.source_root.resolve()
    if args.ratio is not None:
        results = [run(root, args.ratio, args.gpu, args.kv_layout, p4=args.p4)]
    else:
        # Isolate module-import patches and native Mooncake engine ownership.
        results = []
        for ratio, layout in ((2, "v4"), (2, "v41"), (4, "v4"), (128, "v4")):
            for gpu in (False, True):
                output = args.output.with_name(
                    f"d2d-case-{ratio}-{layout}-{int(gpu)}.json"
                )
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--source-root",
                    str(root),
                    "--output",
                    str(output),
                    "--ratio",
                    str(ratio),
                    "--kv-layout",
                    layout,
                ]
                if gpu:
                    cmd.append("--gpu")
                if args.p4:
                    cmd.append("--p4")
                subprocess.run(cmd, check=True, timeout=120)
                results.extend(json.loads(output.read_text())["cases"])

    files = (
        list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py"))
        + list(Path(__file__).parent.glob("*probe_support.py"))
        + [root / "plugins/remote_attn/control_plane/p4_dsv_gateway.py"]
        + [Path(__file__)]
    )
    report = dict(
        device=torch.cuda.get_device_name(),
        scope=__doc__,
        cases=results,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
