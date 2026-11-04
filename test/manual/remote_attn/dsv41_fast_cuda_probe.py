"""Fused sparse kernels and real batched RPC on CUDA; native byte-layout fixtures."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from dsv41_remote_probe_support import encoded, fixture
from sglang_remote_attn import dsv41_paged as paged
from sglang_remote_attn import dsv41_sparse_math as reference


def kernels(root):
    torch.manual_seed(29)
    f = fixture(root, "cuda", kv_layout="v4")
    cases = []
    for layout in (
        f.layout.V4,
        f.layout.V41,
        f.layout.V41_NEW,
        f.layout.V41_FP4,
        f.layout.V41_FP4_NEW,
    ):
        raw = encoded(layout, 7, 8, "cuda")
        for heads in (2, 16, 33, 64):
            for dtype in (torch.bfloat16, torch.float16):
                q = torch.randn(heads, 512, device="cuda", dtype=dtype) * 0.2
                sink = torch.randn(heads, device="cuda")
                ids = torch.tensor([0, 5, 53, -1, 5, 2, 15], device="cuda")
                extra = torch.tensor([17, -1, 18], device="cuda")
                keys = reference.read_slots(
                    raw, torch.cat((ids[ids >= 0], extra[extra >= 0])), 8, layout
                )
                expected = reference.partial(q, keys, 512**-0.5, sink)
                got, lse, error = paged.partial(
                    q, [(raw, ids, 8, layout), (raw, extra, 8, layout)], 512**-0.5, sink
                )
                paged.check([error])
                torch.testing.assert_close(got, expected[0], atol=3e-4, rtol=3e-4)
                torch.testing.assert_close(lse, expected[1], atol=3e-4, rtol=3e-4)
                cases.append(
                    dict(
                        layout=layout.value,
                        heads=heads,
                        dtype=str(dtype),
                        error=float((got - expected[0]).abs().max()),
                    )
                )
    empty = torch.full((8,), -1, device="cuda", dtype=torch.int64)
    q = f.query[0]
    raw = f.p_pool.get_extra_key_buffer(3)
    out, lse, error = paged.partial(q, [(raw, empty, 4, f.extra_layout)], 512**-0.5)
    paged.check([error])
    assert torch.equal(out, torch.zeros_like(out)) and torch.isneginf(lse).all()
    out, lse, error = paged.partial(
        q, [(raw, empty, 4, f.extra_layout)], 512**-0.5, f.sink
    )
    paged.check([error])
    torch.testing.assert_close(lse, f.sink)
    merged = paged.merge(
        (out, lse), (out, torch.full_like(lse, -float("inf"))), q.dtype
    )
    assert torch.equal(merged, torch.zeros_like(merged))
    bad = torch.tensor([raw.shape[0] * 4], device="cuda")
    _, _, error = paged.partial(q, [(raw, bad, 4, f.extra_layout)], 512**-0.5)
    try:
        paged.check([error])
        raise AssertionError("out-of-range slot accepted")
    except RuntimeError:
        pass
    for bad_q, bad_sink in (
        (q.clone().fill_(float("nan")), None),
        (q, f.sink.clone().fill_(float("nan"))),
    ):
        _, _, error = paged.partial(
            bad_q, [(raw, empty, 4, f.extra_layout)], 512**-0.5, bad_sink
        )
        try:
            paged.check([error])
            raise AssertionError("nonfinite query/sink accepted")
        except RuntimeError:
            pass
    target = NS(
        row=0, handle="a", prefix_len=33, decode_prefix_len=8, check=lambda: None
    )
    mapping = paged.page_map(target, f.d_row, 8, 8, 34)
    ids = f.core.c2_sparse_page_indices[0]
    local, remote = torch.empty_like(ids), torch.empty_like(ids)
    mapping.partition(ids, f.core.c2_sparse_topk_lengths, 2, local, remote)
    paged.check([mapping.error])
    assert remote.tolist() == [-1, 15, -1, 15, -1]
    assert local.tolist() == [28, -1, 24, -1, -1]
    assert paged.page_map(target, f.d_row, 8, 8, 35) is mapping
    # Microbenchmark only: selected rows, no model or networking.
    from triton.testing import do_bench

    layout = f.layout.V4
    raw = encoded(layout, 64, 8, "cuda")
    ids = torch.arange(512, device="cuda")
    q = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)

    def old():
        return reference.partial(
            q, reference.read_slots(raw, ids, 8, layout), 512**-0.5
        )

    def new():
        out, lse, error = paged.partial(q, [(raw, ids, 8, layout)], 512**-0.5)
        paged.check([error])
        return out, lse

    old()
    new()
    torch.cuda.synchronize()
    return dict(
        cases=cases,
        invalid_slot_rejected=True,
        nonfinite_payload_rejected=True,
        empty_and_sink=True,
        page_map_and_partition=True,
        micro_ms=dict(reference=do_bench(old), fused=do_bench(new)),
    )


def rpc_roundtrip(root, gpu_transport, output):
    import os
    import tempfile
    import threading
    from unittest.mock import patch

    from sglang_remote_attn import dsv41_fast, dsv41_rpc
    from sglang_remote_attn import dsv41_remote as remote

    assert dsv41_fast.rpc is dsv41_rpc
    from sglang_remote_attn import handoff_control as rt

    f = fixture(root, "cuda", kv_layout="v4")
    f.p_controller.dsv41_remote = True
    paged.warmup(f.p_pool, (1, 2), {2}, torch.bfloat16, "prefill")
    paged.warmup(f.d_pool, (1, 2), {2}, torch.bfloat16, "decode")
    count = 5
    f.controller.scheduler.req_to_token_pool.req_to_token = f.d_row.repeat(count, 1)
    f.p_controller.scheduler.req_to_token_pool.req_to_token = f.p_row.repeat(count, 1)
    transport = None
    old_mode = os.environ.get("HANDOFF_REMOTE_GPU_TRANSFER")
    os.environ["HANDOFF_REMOTE_GPU_TRANSFER"] = "1" if gpu_transport else "0"
    if gpu_transport:
        from remote_gpu_probe_support import Transport

        transport = Transport(root)
        transport.attach(f.p_controller, "prefill")
        d_engine = transport.attach(f.controller, "decode")
        from sglang_remote_attn.transfer_priority import TransferLock

        for owner in (f.p_controller, f.controller):
            owner.gpu_data_plane.manager._remote_attn_send_lock = TransferLock()
    else:
        f.controller.scheduler.disagg_decode_prealloc_queue = NS(kv_manager=None)
    schema = dict(entries=[["kv", 1, 1], ["kv", 2, 2]])
    f.controller._schema = lambda manager: schema
    f.fb.req_pool_indices = torch.arange(count, device="cuda")
    f.fb.req_pool_indices_cpu = list(range(count))
    f.fb.seq_lens = torch.full((count,), 34, device="cuda")
    f.fb.seq_lens_cpu = torch.full((count,), 34)
    for name in ("swa_page_indices", "c2_sparse_page_indices"):
        setattr(f.core, name, getattr(f.core, name).repeat(count, 1))
    for name in ("swa_topk_lengths", "c2_sparse_topk_lengths"):
        setattr(f.core, name, getattr(f.core, name).repeat(count))
    hub = rt.RemoteHub(
        f.p_controller.scheduler,
        stream_factory=lambda n: [(torch.cuda.Stream(), None, None) for _ in range(n)],
    )
    f.p_controller.scheduler._handoff_hub = hub
    query = f.query.repeat(count, 1, 1)
    query *= torch.arange(1, count + 1, device="cuda")[:, None, None] / count
    kv = torch.zeros(count, 512, device="cuda")
    local_keys = reference.read_slots(
        f.d_pool.get_swa_key_buffer_radix(3),
        torch.tensor([13, 14], device="cuda"),
        8,
        f.main_layout,
    )
    extra = reference.read_slots(
        f.d_pool.get_extra_key_buffer(3),
        f.core.c2_sparse_page_indices[0, :4],
        4,
        f.extra_layout,
    )
    expected = torch.stack(
        [
            reference.partial(q, torch.cat((local_keys, extra)), 512**-0.5, f.sink)[
                0
            ].to(query.dtype)
            for q in query
        ]
    )
    sources = []
    with (
        tempfile.TemporaryDirectory(dir=output.parent) as tmp,
        patch.dict(
            __import__("sys").modules,
            {"sglang.srt.disaggregation.base.conn": NS(KVPoll=NS(Success=4, Failed=0))},
        ),
    ):
        try:
            for i in range(count):
                name = str(i)
                req = NS(
                    kv=NS(req_pool_idx=i),
                    cache_request_handle=name,
                    disagg_kv_sender=NS(poll=lambda: 3),
                )
                src = NS(
                    owner=f.p_controller,
                    req=req,
                    prefix_len=33,
                    decode_prefix_len=0,
                    generation=name,
                    path=Path(tmp) / name,
                    service=None,
                    error=None,
                    cache_lease=None,
                    remote_closed=threading.Event(),
                    schema=schema,
                    barrier_ready=True,
                    peers=((NS(mooncake_session_id=d_engine.get_session_id()), None),)
                    if transport
                    else (),
                )
                remote.register(src)
                sources.append(src)
                target_req = NS(kv=NS(req_pool_idx=i), cache_request_handle=name)
                target = remote.Target(
                    f.controller,
                    target_req,
                    src.path,
                    name,
                    src.service,
                    33,
                    progressive=True,
                )
                f.controller.remote_targets[i] = target

            def native(*args, **kwargs):
                assert all(
                    1 in t.local_entries for t in f.controller.remote_targets.values()
                )
                return expected

            # Run the production dispatcher, including CPU/GPU lane selection.
            def step():
                return remote.attention(
                    native,
                    f.backend,
                    query,
                    kv,
                    kv,
                    NS(layer_id=3),
                    f.fb,
                    2,
                    attn_sink=f.sink,
                )

            warm = step()
            torch.testing.assert_close(warm, expected, atol=0.015625, rtol=0.01)
            # Prove real Q RPC can execute while D local GPU work is still live.
            delayed = [False]
            overlap, sending_overlap = [], []
            local_done = torch.cuda.Event()
            local_queued = threading.Event()
            producer_events = []
            from sglang_remote_attn import remote_data_plane as dp

            actual_event = dp.event

            def event(device):
                value = actual_event(device)
                if threading.get_ident() == main_thread:
                    producer_events.append(value)
                return value

            actual_partial, actual_execute = paged.partial, remote.execute
            main_thread = threading.get_ident()

            def partial(*args, **kwargs):
                if threading.get_ident() == main_thread and not delayed[0]:
                    delayed[0] = True
                    torch.cuda._sleep(200_000_000)
                    local_done.record()
                    local_queued.set()
                    assert producer_events
                    producer_events[-1].synchronize()
                    assert not local_done.query(), (
                        "Q producer event includes local work"
                    )
                return actual_partial(*args, **kwargs)

            def execute(*args, **kwargs):
                if delayed[0]:
                    overlap.append(not local_done.query())
                return actual_execute(*args, **kwargs)

            from contextlib import nullcontext

            actual_write = f.controller.gpu_data_plane.write if transport else None

            def write(*args, **kwargs):
                assert local_queued.wait(10), "D local work was never queued"
                sending_overlap.append(delayed[0] and not local_done.query())
                return actual_write(*args, **kwargs)

            with (
                patch.object(f.controller.gpu_data_plane, "write", write)
                if transport
                else nullcontext(),
                patch.object(dp, "event", event),
                patch.object(paged, "partial", partial),
                patch.object(remote, "execute", execute),
            ):
                got = step()
            torch.testing.assert_close(got, expected, atol=0.015625, rtol=0.01)
            if not transport:
                assert any(overlap), (
                    "Remote was serialized behind D local GPU completion"
                )
            batches = f.controller.dsv41_rpc_batches
            assert batches.stats["batches"] == 4 and batches.stats["rows"] == 10
            for t in f.controller.remote_targets.values():
                t.local_entries.add(1)
            local = step()
            torch.testing.assert_close(local, expected)
            assert batches.stats["batches"] == 4
            assert f.controller.sparse_stats["native_layers"] == 1
            torch.cuda.synchronize()
            for target in list(f.controller.remote_targets.values()):
                target.close("probe-local")
            deadline = time.monotonic() + 10
            while not all(remote.source_ready(src) for src in sources):
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert all(src.remote_closed.is_set() for src in sources)
            result = dict(
                gpu_transport=gpu_transport,
                five_rows_two_rpcs=True,
                q_producer_excludes_local=True,
                local_remote_gpu_overlap=any(overlap),
                q_send_overlaps_local=any(sending_overlap) if transport else True,
                native_layer_recovered=True,
                source_refs_drained=True,
                stats=dict(batches.stats),
            )
        finally:
            if hasattr(f.controller, "dsv41_rpc_batches"):
                f.controller.dsv41_rpc_batches.close()
            if transport:
                transport_result = transport.close()
                if "result" in locals():
                    result["transport"] = transport_result
            hub.close()
            if old_mode is None:
                os.environ.pop("HANDOFF_REMOTE_GPU_TRANSFER", None)
            else:
                os.environ["HANDOFF_REMOTE_GPU_TRANSFER"] = old_mode
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    result = kernels(args.source_root.resolve())
    result["rpc"] = [
        rpc_roundtrip(args.source_root.resolve(), mode, args.output)
        for mode in (False, True)
    ]
    result["device"] = torch.cuda.get_device_name()
    result["torch"] = torch.__version__
    result["scope"] = __doc__
    root = args.source_root.resolve()
    files = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(__file__).with_name("dsv41_remote_probe_support.py").resolve(),
        Path(__file__).with_name("remote_gpu_probe_support.py").resolve(),
    ]
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
