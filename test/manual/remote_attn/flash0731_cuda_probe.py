"""Flash C4/C128 native bytes, real RPC and immutable-tail scatter; no model weights."""

import argparse
import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from flash0731_probe_support import flash_fixture
from sglang_remote_attn import dsv41_fast, dsv41_remote as remote, dsv41_rpc
from sglang_remote_attn import dsv41_paged as paged, dsv41_sparse_math as math
from sglang_remote_attn import handoff_control as rt, remote_data_plane as dp
from sglang_remote_attn.dsv41_receive import scatter_prefix
from sglang_remote_attn.source_cache import CacheGuard


def mixed_maps(root):
    f = flash_fixture(root, "cuda", 128)
    table = f.d_row
    pages = table.shape[1] // 256
    target = NS(
        row=0,
        handle="mixed",
        prefix_len=f.prefix,
        decode_prefix_len=0,
        check=lambda: None,
    )
    by_ratio = {}
    for ratio in (4, 128, 4, 128):
        capacity = pages + ratio
        m = paged.page_map(target, table, 256, capacity, f.prefix + 128, ratio=ratio)
        if ratio in by_ratio:
            assert m is by_ratio[ratio]
        by_ratio[ratio] = m
        cpp = 256 // ratio
        positions = torch.tensor(
            [0, f.prefix // ratio - 1, f.prefix // ratio], device="cuda"
        )
        physical = table[0, positions // cpp * 256] // 256
        ids = (physical * cpp + positions % cpp).to(torch.int32)
        li, re = torch.empty_like(ids), torch.empty_like(ids)
        m.partition(ids, torch.tensor([3], device="cuda"), ratio, li, re)
        paged.check([m.error])
        assert re.tolist() == [0, f.prefix // ratio - 1, -1]
        assert li.tolist() == [-1, -1, int(ids[-1])]
    assert len(target._sparse_page_maps) == 2
    for ratio in (4, 128):
        try:
            paged.page_map(
                target, table, 256, pages + ratio + 1, f.prefix + 128, ratio=ratio
            )
            raise AssertionError("pool drift accepted")
        except RuntimeError:
            pass
    torch.cuda.synchronize()
    target._sparse_page_maps.clear()
    return dict(
        alternating_ratios=True,
        native_guard_capacities=[pages + 4, pages + 128],
        incomplete_blocks_stay_local=True,
        same_ratio_capacity_drift_rejected=True,
    )


def run(root, ratio, gpu):
    from remote_gpu_probe_support import Transport

    f = flash_fixture(root, "cuda", ratio)
    f.p_controller.dsv41_remote = True
    os.environ["HANDOFF_REMOTE_GPU_TRANSFER"] = "1" if gpu else "0"
    transport = Transport(root) if gpu else None
    if transport:
        transport.attach(f.p_controller, "prefill")
        engine = transport.attach(f.controller, "decode")
    else:
        f.controller.scheduler.disagg_decode_prealloc_queue = NS(kv_manager=None)
    schema = {"entries": [["kv", ratio, 3]]}
    f.controller._schema = lambda m: schema
    pool = f.d_pool
    ids = f.core.sparse_page_indices(ratio)[0]
    live = ids[ids >= 0]
    keys = math.read_slots(
        pool.get_extra_key_buffer(3), live, 256 // ratio, f.layout.V4
    )
    swa = math.read_slots(
        pool.get_swa_key_buffer_radix(3),
        f.core.swa_page_indices[0, :2],
        pool.get_swa_key_page_size(),
        f.layout.V4,
    )
    expected = math.partial(f.query[0], torch.cat((swa, keys)), 512**-0.5, f.sink)[
        0
    ].to(f.query.dtype)
    # D's immutable prefix is deliberately unavailable. Only P can supply it.
    raw = pool.get_extra_key_buffer(3)
    completed = f.prefix // ratio
    selected = live[:completed] if ratio == 128 else live[:3]
    cpp = 256 // ratio
    data, base, scale = f.layout.V4.row_addressing(cpp)
    for offset, stride, width in (
        (0, data, f.layout.V4.data_bytes),
        (base, scale, f.layout.V4.scale_bytes),
    ):
        cols = (
            offset
            + (selected % cpp)[:, None] * stride
            + torch.arange(width, device="cuda")[None, :]
        )
        raw[(selected // cpp)[:, None], cols] = 0
    paged.warmup(f.p_pool, (3,), {2, 3}, f.query.dtype, "prefill")
    paged.warmup(f.d_pool, (3,), {2, 3}, f.query.dtype, "decode")
    hub = rt.RemoteHub(
        f.p_controller.scheduler,
        stream_factory=lambda n: [(torch.cuda.Stream(), None, None) for _ in range(n)],
    )
    f.p_controller.scheduler._handoff_hub = hub
    pins = [0]

    def lock(node):
        pins[0] += 1
        return object()

    def unlock(pin):
        pins[0] -= 1

    guard = CacheGuard(
        NS(
            lock=lock,
            unlock=unlock,
            checkpoint=lambda *a, **kw: None,
            finish=lambda *a: None,
            reset=lambda: None,
        )
    )
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.dict(
            __import__("sys").modules,
            {"sglang.srt.disaggregation.base.conn": NS(KVPoll=NS(Success=4, Failed=0))},
        ),
    ):
        req = NS(
            kv=NS(req_pool_idx=0, cache_protected_len=0),
            cache_request_handle="flash",
            disagg_kv_sender=NS(poll=lambda: 3),
            last_node=1,
            lock=object(),
        )
        src = NS(
            owner=f.p_controller,
            req=req,
            prefix_len=f.prefix,
            decode_prefix_len=0,
            generation="g",
            path=Path(tmp),
            service=None,
            error=None,
            cache_lease=guard.hold(req),
            remote_closed=threading.Event(),
            schema=schema,
            barrier_ready=True,
            peers=((NS(mooncake_session_id=engine.get_session_id()), None),)
            if gpu
            else (),
        )
        target = None
        try:
            remote.register(src)
            f.req.cache_request_handle = "d"
            target = remote.Target(
                f.controller,
                f.req,
                src.path,
                "g",
                src.service,
                f.prefix,
                progressive=True,
            )
            f.controller.remote_targets[0] = target
            kv = torch.zeros((1, 512), device="cuda")
            native_calls = []

            def native(*args, **kw):
                native_calls.append(True)
                return expected[None]

            # Poison padded TP heads: the model discards them; Remote must not read NaNs.
            q = torch.cat(
                (
                    f.query,
                    torch.full(
                        (1, 1, 512), float("nan"), device="cuda", dtype=f.query.dtype
                    ),
                ),
                dim=1,
            )
            sink = torch.cat((f.sink, torch.zeros(1, device="cuda")))
            layer = NS(layer_id=3, tp_q_head_num=2)
            got = remote.attention(
                native, f.backend, q, kv, kv, layer, f.fb, ratio, attn_sink=sink
            )
            torch.testing.assert_close(got[0, :2], expected, atol=0.015625, rtol=0.01)
            torch.cuda.synchronize()
            assert not native_calls and target.calls == 1 and pins == [1]
            # Tombstone a real in-flight P GPU read. Neither early close nor
            # cancellation is a positive completion proof for source storage.
            ctx = hub.registry.get(src.service["handoff_id"], src.service["epoch"])
            delayed = torch.cuda.Stream()
            with torch.cuda.stream(delayed):
                torch.cuda._sleep(500_000_000)
                fence = torch.cuda.Event()
                fence.record()
            ctx.dsv41_ready = fence
            outputs, errors = [], []
            pending = dsv41_rpc.pool(f.controller).submit(
                [dsv41_rpc.Row(0, target, f.query[0], [0], 3, ratio, 512**-0.5)]
            )

            def call():
                try:
                    torch.cuda.set_device(0)
                    outputs.append(pending.result())
                except BaseException as exc:
                    errors.append(repr(exc))

            worker = threading.Thread(target=call)
            worker.start()
            deadline = time.monotonic() + 5
            while not ctx.inflight:
                assert time.monotonic() < deadline
                time.sleep(0.001)
            assert not dp.ready(f.req)
            try:
                target.close("cancel-before-proof")
                raise AssertionError("in-flight target retired")
            except RuntimeError:
                pass
            assert not hub.registry.release(ctx.handoff_id, expected_req=req)
            assert pins == [1] and not src.remote_closed.is_set()
            fence.synchronize()
            worker.join(5)
            assert not worker.is_alive() and not errors and len(outputs) == 1, errors
            torch.cuda.synchronize()
            assert dp.ready(f.req) and pins == [1]
            # Short/tail range scatter cannot touch D's new block or padding.
            for start_page in (0, 1):
                for prefix in (
                    ratio - 1,
                    ratio,
                    ratio + 1,
                    256 + ratio - 1,
                    256 + ratio,
                ):
                    dst = torch.full(
                        (4, f.layout.V4.page_bytes(cpp)),
                        253,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    before = dst.clone()
                    source = torch.zeros(
                        (2, dst.shape[1]), dtype=torch.uint8, device="cuda"
                    )
                    page_ids = torch.tensor([2, 1], device="cuda")
                    scatter_prefix(
                        dst,
                        source,
                        page_ids,
                        start_page,
                        prefix,
                        256,
                        ratio,
                        f.layout.V4,
                    )
                    immutable = prefix // ratio
                    # Exact byte oracle, including page-tail and scale padding.
                    oracle = before.clone()
                    for p in range(2):
                        if p + start_page < prefix // 256:
                            oracle[page_ids[p]] = source[p]
                        else:
                            rows = min(cpp, max(0, immutable - (p + start_page) * cpp))
                            for off, stride, width in (
                                (0, data, f.layout.V4.data_bytes),
                                (base, scale, f.layout.V4.scale_bytes),
                            ):
                                for row in range(rows):
                                    oracle[
                                        page_ids[p],
                                        off + row * stride : off + row * stride + width,
                                    ] = 0
                    assert torch.equal(dst, oracle)
            target.local_entries.add(0)
            remote.attention(
                native, f.backend, f.query, kv, kv, layer, f.fb, ratio, attn_sink=f.sink
            )
            assert native_calls == [True] and target.calls == 2
            target.close("gpu-proven")
            assert not target._sparse_page_maps
            assert remote.source_ready(src)
            src.cache_lease.retire(lambda: None)
            assert pins == [0]
            result = dict(
                ratio=ratio,
                gpu_transport=gpu,
                selected=len(f.positions),
                remote_rows=completed if ratio == 128 else 3,
                poisoned_destination_prefix=True,
                native_metadata=True,
                padded_heads_safe=True,
                tail_scatter_exact=True,
                nonzero_scatter_start=True,
                local_restored=True,
                pins_closed=True,
                early_close_rejected=True,
                tombstone_waited_for_gpu_reader=True,
            )
        finally:
            if hasattr(f.controller, "dsv41_rpc_batches"):
                f.controller.dsv41_rpc_batches.close()
            if transport:
                details = transport.close()
                if "result" in locals():
                    result["transport"] = details
            hub.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.testing.assert_close(torch.zeros(1), torch.zeros(1))
    root = args.source_root.resolve()
    result = dict(
        device=torch.cuda.get_device_name(),
        scope=__doc__,
        mixed_maps=mixed_maps(root),
        cases=[run(root, r, g) for r in (4, 128) for g in (False, True)],
    )
    files = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(__file__).with_name("flash0731_probe_support.py").resolve(),
    ]
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
