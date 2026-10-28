"""Single-GPU CUDA/Mooncake probe; no model or TP2 inference is implied.

PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/cuda_handoff_probe.py \
    --output /tmp/remote-attn-cuda

Each transfer uses separate source/target processes with the real plugin wire
protocol and Mooncake TCP engine. Small scheduler/pool fixtures supply only KV
storage; CUDA copies, streams, FlashInfer and transport are not mocked.
"""

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

LAYERS, TOKENS, HEADS, DIM = 6, 2048, 2, 64


def trace_overlap(path):
    events = json.loads(path.read_text())["traceEvents"]
    copies = [e for e in events if e.get("cat") == "gpu_memcpy" and "HtoD" in e["name"]]
    compute = [
        e for e in events if e.get("cat") == "kernel" and "gemm" in e["name"].lower()
    ]
    overlaps = [
        sum(
            max(
                0,
                min(copy["ts"] + copy["dur"], kernel["ts"] + kernel["dur"])
                - max(copy["ts"], kernel["ts"]),
            )
            for kernel in compute
        )
        for copy in copies
    ]
    return dict(
        h2d_copies=len(copies),
        gemm_kernels=len(compute),
        overlapping_copies=sum(value > 0 for value in overlaps),
        total_h2d_us=sum(copy["dur"] for copy in copies),
        h2d_gemm_overlap_us=sum(overlaps),
        copy_streams=sorted({e["args"]["stream"] for e in copies}),
        compute_streams=sorted({e["args"]["stream"] for e in compute}),
    )


def kv_pool(dtype):
    shape = (2 * TOKENS + 32, HEADS, DIM)
    base = torch.arange(shape[0] * HEADS * DIM, device="cuda").reshape(shape)
    keys = [((base + 7 * layer) % 97).to(dtype) / 97 for layer in range(LAYERS)]
    values = [((base + 11 * layer) % 89).to(dtype) / 89 for layer in range(LAYERS)]
    return SimpleNamespace(
        layer_num=LAYERS,
        get_key_buffer=keys.__getitem__,
        get_value_buffer=values.__getitem__,
        keys=keys,
        values=values,
    )


def source_worker(
    mode,
    dtype_name,
    sender_buffers,
    result_queue,
    cancel_request=None,
    source_release=None,
):
    try:
        os.environ.update(
            HANDOFF_FAST_BUFFERS=str(sender_buffers),
            HANDOFF_FAST_COPY_STREAMS="2",
            HANDOFF_FAST_ASYNC="1" if mode != "fast-sync" else "0",
            MC_FORCE_TCP="1",
        )
        from sglang_remote_attn import handoff_control as rt

        torch.cuda.set_device(0)
        pool = kv_pool(getattr(torch, dtype_name))
        indices = torch.arange(TOKENS, device="cuda") * 2 + 1
        req = SimpleNamespace(
            rid="cuda-probe",
            kv=SimpleNamespace(req_pool_idx=0, kv_committed_len=TOKENS),
        )
        scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=indices.unsqueeze(0)),
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
        )
        meta = dict(
            rid=req.rid,
            dtype=dtype_name,
            kv_committed_len=TOKENS,
            kv_heads=HEADS,
            head_dim=DIM,
            chunks=[[TOKENS] for _ in range(LAYERS)],
            fast_freeze=mode.startswith("fast"),
        )
        if mode.startswith("fast"):
            transfer = rt._start_layer_gpu_server(
                scheduler, req, meta, source_release is not None
            )
        else:
            snapshot = [
                [[pool.keys[layer][indices].cpu(), pool.values[layer][indices].cpu()]]
                for layer in range(LAYERS)
            ]
            transfer = rt._start_layer_mooncake_server(
                scheduler, req, meta, snapshot, False
            )
        result_queue.put({"transfer": transfer})
        if cancel_request is not None:
            assert cancel_request.wait(timeout=45), "no cancellation request"
            rt.stop_source_transfer(req)
        req._handoff_transfer_thread.join(timeout=45)
        if req._handoff_transfer_thread.is_alive():
            rt.stop_source_transfer(req)
            raise TimeoutError("source sender did not finish")
        if cancel_request is None:
            assert not rt._TRANSFER_ERRORS, rt._TRANSFER_ERRORS
        registrations = rt._get_registration_manager().stats()
        if cancel_request is None:
            assert registrations["live"] == 0, registrations
        hub_stats = None
        if source_release is not None:
            assert source_release.wait(timeout=60), "target did not finish remote reads"
            hub = rt.get_remote_hub(scheduler)
            deadline = time.monotonic() + 5
            while hub.stats()["in_flight"] and time.monotonic() < deadline:
                time.sleep(0.001)
            hub_stats = hub.stats()
            rt.close_handoff_hub(scheduler)
        result_queue.put(
            {
                "source": {
                    "registration": registrations,
                    "stats": rt._TRANSFER_STATS,
                    "hub": hub_stats,
                }
            }
        )
    except BaseException:
        result_queue.put({"error": traceback.format_exc()})
        raise


def target_state():
    return SimpleNamespace(
        events=[threading.Event() for _ in range(LAYERS)],
        error=None,
        req=None,
        aborted=False,
        abort_event=threading.Event(),
        import_done_at=None,
        layer_wait_ms=[None] * LAYERS,
        layer_copy_ms=[None] * LAYERS,
        layer_sync_ms=[None] * LAYERS,
        layer_bytes=[None] * LAYERS,
        layer_ready_at=[None] * LAYERS,
        layer_msg_at=[None] * LAYERS,
    )


def transfer_probe(
    mode,
    dtype_name,
    slots,
    buffers,
    output,
    profile=False,
    cancel=False,
    remote_drop=False,
):
    from sglang_remote_attn import handoff_control as rt

    rt.TARGET_STAGING_SLOTS = slots
    context = mp.get_context("spawn")
    messages = context.Queue()
    cancel_request = context.Event() if cancel else None
    source_release = context.Event() if remote_drop else None
    source = context.Process(
        target=source_worker,
        args=(mode, dtype_name, buffers, messages, cancel_request, source_release),
    )
    source.start()
    receiver = None
    state = target_state()
    conn = None
    profiler = None
    try:
        packet = messages.get(timeout=45)
        assert "error" not in packet, packet
        conn, meta, mirror, _, _, owner, staging, _, _ = rt._recv_transfer_layer_setup(
            packet["transfer"]
        )
        if remote_drop:
            state = rt._TargetHandoffState(
                0,
                TOKENS,
                [threading.Event() for _ in range(LAYERS)],
                "remote",
                remote_service=packet["transfer"]["remote_service"],
                handoff_id=meta["rid"],
            )
            os.environ["HANDOFF_LAYER_DROP"] = "0"
        dtype = getattr(torch, dtype_name)
        shape = (TOKENS + 32, HEADS, DIM)
        keys = [
            torch.full(shape, float("nan"), device="cuda", dtype=dtype)
            for _ in range(LAYERS)
        ]
        values = [torch.full_like(keys[0], float("nan")) for _ in range(LAYERS)]
        pool = SimpleNamespace(
            get_key_buffer=keys.__getitem__, get_value_buffer=values.__getitem__
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool)
        )
        mapping = torch.arange(TOKENS, device="cuda").flip(0) + 5
        state.indices_ready = torch.cuda.Event()
        state.indices_ready.record()
        # Warm compute before capture so initialization does not consume the
        # transfer window. A small GEMM leaves room for the copy engines.
        a = torch.randn(512, 512, device="cuda", dtype=dtype)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        torch.mm(a, b, out=c)
        torch.cuda.current_stream().synchronize()
        if profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            profiler.__enter__()
        target = rt._slot_layer_receiver if staging is not None else rt._layer_receiver
        receiver = threading.Thread(
            target=target,
            args=(
                scheduler,
                state,
                conn,
                meta,
                staging if staging is not None else mirror,
                mapping,
                owner,
                0,
            ),
            daemon=True,
        )
        state.receiver_thread = receiver
        state.receiver_conn = conn
        if cancel:
            rt._layer_delay_s = lambda: 0.025
        started = time.monotonic()
        receiver.start()
        compute_steps = 0
        if cancel:
            assert state.events[0].wait(timeout=15), "first layer never arrived"
            assert state.import_done_at is None, "cancel missed the import window"
            cancel_request.set()
            rt._TargetHandoffState.abort(state, "CUDA cancellation probe")
        deadline = started + 45
        while receiver.is_alive() and time.monotonic() < deadline:
            torch.mm(a, b, out=c)
            torch.cuda.current_stream().synchronize()
            compute_steps += 1
        receiver.join(timeout=1)
        assert not receiver.is_alive(), "target receiver stalled"
        if profiler is not None:
            profiler.__exit__(None, None, None)
            profiler.export_chrome_trace(str(output / "overlap-trace.json"))
            profiler = None
        if cancel:
            assert state.error is not None and state.import_done_at is None
            assert owner in rt._TRANSFER_QUARANTINE
            # Simulate allocator reuse only AFTER abort has fenced GPU writes.
            for tensor in keys + values:
                tensor.fill_(-5)
            torch.cuda.current_stream().synchronize()
            time.sleep(0.1)
            assert all(bool((tensor == -5).all()) for tensor in keys + values)
            packet = messages.get(timeout=15)
            assert "error" not in packet, packet
            source.join(timeout=10)
            assert source.exitcode == 0, source.exitcode
            return dict(
                mode=mode,
                cancelled=True,
                gpu_reuse_safe=True,
                target_quarantined=True,
                **packet,
            )
        if remote_drop:
            assert state.error is not None and state.import_done_at is None
            assert not state.events[0].is_set(), "dropped layer falsely became LOCAL"
            assert all(event.is_set() for event in state.events[1:])
            assert owner in rt._TRANSFER_QUARANTINE
            assert torch.isnan(keys[0]).all(), (
                "dropped layer should still have no local KV"
            )
            reference = kv_pool(dtype)
            source_indices = torch.arange(TOKENS, device="cuda") * 2 + 1
            for layer in range(1, LAYERS):
                torch.testing.assert_close(
                    keys[layer][mapping],
                    reference.keys[layer][source_indices],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    values[layer][mapping],
                    reference.values[layer][source_indices],
                    rtol=0,
                    atol=0,
                )
            errors = []
            service = packet["transfer"]["remote_service"]
            state.client.attach(state.handoff_id, service["token"], state.owner_epoch)
            state._pending_fallback = True
            for _ in range(3):
                state.begin_step(SimpleNamespace(seq_lens=torch.tensor([TOKENS + 17])))
                assert state.mode == "remote" and state.is_remote_layer(0)
                assert not any(
                    state.is_remote_layer(layer) for layer in range(1, LAYERS)
                )
                q = torch.randn(4, DIM, device="cuda", dtype=dtype)
                raw = q.cpu().view(torch.uint8).numpy().tobytes()
                header, remote_o, remote_lse = state.client.call(
                    dict(
                        handoff_id=state.handoff_id,
                        epoch=state.owner_epoch,
                        layer=0,
                        q_dtype=dtype_name,
                        q_shape=list(q.shape),
                        q_bytes=len(raw),
                        sm_scale=DIM**-0.5,
                    ),
                    raw,
                )
                rt._check_remote_header(header, state.handoff_id)
                tail_k = torch.randn(17, HEADS, DIM, device="cuda", dtype=dtype)
                tail_v = torch.randn_like(tail_k)
                local_o, local_lse = rt._single_decode(q, tail_k, tail_v, DIM**-0.5)
                remote_o, remote_lse = rt._h2d_pair(remote_o, remote_lse, q.device)
                merged, _ = rt._merge_states(local_o, local_lse, remote_o, remote_lse)
                k = torch.cat([reference.keys[0][source_indices], tail_k])
                v = torch.cat([reference.values[0][source_indices], tail_v])
                expected = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q[None, :, None, :].float(),
                        k.permute(1, 0, 2)[None].repeat_interleave(2, dim=1).float(),
                        v.permute(1, 0, 2)[None].repeat_interleave(2, dim=1).float(),
                    )
                    .squeeze(0)
                    .squeeze(1)
                )
                torch.testing.assert_close(
                    merged.squeeze(0).float(), expected, rtol=0.03, atol=0.004
                )
                errors.append(float((merged.squeeze(0).float() - expected).abs().max()))
            state.close()
            rt._close_target_clients()
            source_release.set()
            source_packet = messages.get(timeout=15)
            assert "error" not in source_packet, source_packet
            source.join(timeout=10)
            assert source.exitcode == 0, source.exitcode
            assert source_packet["source"]["hub"]["rpc_completed"] == 3
            assert source_packet["source"]["hub"]["in_flight"] == 0
            return dict(
                mode=mode,
                slots=slots,
                remote_drop=True,
                rpc_calls=3,
                local_layers=LAYERS - 1,
                max_abs_errors=errors,
                dropped_layer_still_remote=True,
                target_quarantined=True,
                **source_packet,
            )
        assert state.error is None, state.error
        assert state.import_done_at is not None
        assert all(event.is_set() for event in state.events)
        assert compute_steps > 0, "no compute was submitted during transfer"
        reference = kv_pool(dtype)
        source_indices = torch.arange(TOKENS, device="cuda") * 2 + 1
        for layer in range(LAYERS):
            torch.testing.assert_close(
                keys[layer][mapping],
                reference.keys[layer][source_indices],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                values[layer][mapping],
                reference.values[layer][source_indices],
                rtol=0,
                atol=0,
            )
            assert torch.isnan(keys[layer][:5]).all()
        source_packet = messages.get(timeout=15)
        assert "error" not in source_packet, source_packet
        source.join(timeout=10)
        assert source.exitcode == 0, source.exitcode
        registrations = rt._get_registration_manager().stats()
        assert registrations["live"] == 0, registrations
        assert not rt._TRANSFER_QUARANTINE, rt._TRANSFER_QUARANTINE.keys()
        return dict(
            mode=mode,
            dtype=dtype_name,
            slots=slots,
            source_buffers=buffers,
            compute_steps=compute_steps,
            overlap=trace_overlap(output / "overlap-trace.json") if profile else None,
            layer_ready_at=state.layer_ready_at,
            target_registration=registrations,
            **source_packet,
        )
    finally:
        if source_release is not None:
            source_release.set()
        if remote_drop:
            os.environ.pop("HANDOFF_LAYER_DROP", None)
            if hasattr(state, "close"):
                state.close()
            rt._close_target_clients()
        if profiler is not None:
            profiler.__exit__(None, None, None)
        if receiver is not None and receiver.is_alive():
            rt._stop_receiver_io(state)
        elif conn is not None:
            conn.close()
        if source.is_alive():
            source.terminate()
        source.join(timeout=10)
        messages.close()
        if cancel:
            rt._layer_delay_s = lambda: 0.0


def attention_probe(dtype_name):
    """Exercise real source attention and split-prefix/tail LSE merge."""
    from sglang_remote_attn import handoff_control as rt

    torch.manual_seed(42)
    dtype = getattr(torch, dtype_name)
    n, split, q_heads = 257, 193, 4
    q = torch.randn(q_heads, DIM, device="cuda", dtype=dtype)
    k = torch.randn(n, HEADS, DIM, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    # Noncontiguous source indices exercise the gather path.
    mapping = torch.arange(split, device="cuda").flip(0)
    stored_k, stored_v = k[:split].flip(0).contiguous(), v[:split].flip(0).contiguous()
    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=mapping.unsqueeze(0)),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(
                get_key_buffer=lambda layer: stored_k,
                get_value_buffer=lambda layer: stored_v,
            )
        ),
    )
    ctx = SimpleNamespace(
        req=SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0)),
        kv_len=split,
        contiguous=False,
    )
    job = SimpleNamespace(
        hub=SimpleNamespace(
            scheduler=scheduler, _stats_lock=threading.Lock(), _first_rpc_logged=True
        ),
        ctx=ctx,
        header={"layer": 0, "sm_scale": DIM**-0.5},
        q_cpu=q.cpu(),
        recv_delay_ms=0,
        recv_ms=0,
    )
    stream = rt._create_hub_streams(1)[0][0]
    stream.wait_stream(torch.cuda.current_stream())
    _, remote_o, remote_lse = rt.execute_remote_attention(job, stream, {})
    local_o, local_lse = rt._single_decode(q, k[split:], v[split:], DIM**-0.5)
    remote_o, remote_lse = rt._h2d_pair(remote_o, remote_lse, q.device)
    merged_o, _ = rt._merge_states(local_o, local_lse, remote_o, remote_lse)
    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q[None, :, None, :].float(),
            k.permute(1, 0, 2)[None].repeat_interleave(2, dim=1).float(),
            v.permute(1, 0, 2)[None].repeat_interleave(2, dim=1).float(),
        )
        .squeeze(0)
        .squeeze(1)
    )
    error = float((merged_o.squeeze(0).float() - expected).abs().max())
    torch.testing.assert_close(
        merged_o.squeeze(0).float(),
        expected,
        rtol=0.03,
        atol=0.004 if dtype == torch.float16 else 0.012,
    )
    return {"dtype": dtype_name, "max_abs_error": error, "tokens": n, "prefix": split}


def main():
    import flashinfer
    from sglang_remote_attn import handoff_control as rt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    report = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu_count=torch.cuda.device_count(),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        compute_capability=torch.cuda.get_device_capability(),
        flashinfer=flashinfer.__version__,
        mooncake=importlib.metadata.version("mooncake-transfer-engine-cuda13"),
        send_probe=callable(getattr(rt._get_te(), "send_probe", None)),
        source_sha256={
            "handoff_control.py": hashlib.sha256(
                Path(rt.__file__).read_bytes()
            ).hexdigest(),
            "cuda_handoff_probe.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        transfers=[],
        attention=[],
    )
    for dtype in ("float16", "bfloat16"):
        result = attention_probe(dtype)
        report["attention"].append(result)
        print("ATTENTION", json.dumps(result), flush=True)
    cases = [
        ("fast-async", "float16", 1, 3),
        ("fast-async", "bfloat16", 2, 2),
        ("fast-sync", "float16", 1, 1),
        ("fast-async", "float16", 0, 2),
        ("cpu-layer", "bfloat16", 0, 1),
    ]
    for index, (mode, dtype, slots, buffers) in enumerate(cases):
        result = transfer_probe(
            mode, dtype, slots, buffers, args.output, args.profile and index == 0
        )
        report["transfers"].append(result)
        print("TRANSFER", json.dumps(result), flush=True)
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
    for slots in (1, 0):
        result = transfer_probe(
            "fast-async", "float16", slots, 3, args.output, remote_drop=True
        )
        report["transfers"].append(result)
        (args.output / "results.json").write_text(json.dumps(report, indent=2))
        print("REMOTE DROP", json.dumps(result), flush=True)
    result = transfer_probe("fast-async", "float16", 1, 3, args.output, cancel=True)
    report["transfers"].append(result)
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    print("CANCELLATION", json.dumps(result), flush=True)
    print(
        "PASS: two attention comparisons, five transfers, two Remote Drop RPC probes and a cancellation fence",
        flush=True,
    )


if __name__ == "__main__":
    main()
