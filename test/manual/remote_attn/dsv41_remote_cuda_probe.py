"""Real CUDA + RemoteHub TCP eager sparse read closure; small native-byte pools.

No checkpoint/full SMG-PD server or SM100 FlashMLA/indexer kernel is launched.
Native PD-complete/top-k inputs are fixtures; RPC, registry, GPU reads and math
are real. D owns native-state continuation; no early admission is claimed.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from dsv41_remote_probe_support import encoded, fixture
from sglang_remote_attn import dsv41_remote as remote
from sglang_remote_attn import dsv41_sparse_math as math
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import hooks
from sglang_remote_attn.dsv41_layerwise import DSV41Source
from sglang_remote_attn.source_cache import CacheGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.sglang_root.resolve()
    torch.cuda.set_device(0)
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.disaggregation",
        "sglang.srt.disaggregation.base",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package
    sys.modules["sglang.srt.disaggregation.base.conn"] = SimpleNamespace(
        KVPoll=SimpleNamespace(Success=4, Failed=0)
    )
    sys.modules["sglang.srt.runtime_context"] = SimpleNamespace(
        get_parallel=lambda: SimpleNamespace(tp_size=1, tp_rank=0)
    )
    f = fixture(root, "cuda")
    f.p_controller.dsv41_remote = True
    f.controller.dsv41_layerwise = f.controller.graph_fast_path = True
    f.controller.plan = f.controller.executing = None
    layout_cases = []
    for layout in (
        f.layout.V41,
        f.layout.V41_NEW,
        f.layout.V41_FP4,
        f.layout.V41_FP4_NEW,
    ):
        split = encoded(layout.encoding, 5, 4, "cuda")
        reader = (
            f.oracle.dequantize_k_cache_v41_fp4
            if layout.is_fp4
            else f.oracle.dequantize_k_cache_v41
        )
        expected = reader(split.cpu(), 4).reshape(-1, 512)[
            torch.tensor([7, 1, 19, 0, 7])
        ]
        raw = (
            f.oracle.interleave_pages(split, 4, layout.data_bytes, layout.scale_bytes)
            if layout.is_interleaved
            else split
        )
        got = math.read_slots(
            raw, torch.tensor([7, 1, 19, 0, 7], device="cuda"), 4, layout
        )
        assert torch.equal(got.cpu().view(torch.int16), expected.view(torch.int16))
        layout_cases.append(layout.value)
    rt._CTL_DIR = ""
    hub = rt.RemoteHub(
        f.p_controller.scheduler,
        stream_factory=lambda n: [(torch.cuda.Stream(), None, None) for _ in range(n)],
    )
    f.p_controller.scheduler._handoff_hub = hub
    pins = {"FULL": 0, "SWA": 0}

    def lock(node):
        for key in pins:
            pins[key] += 1
        return object()

    def unlock(receipt):
        for key in pins:
            pins[key] -= 1

    cache = SimpleNamespace(
        lock=lock,
        unlock=unlock,
        checkpoint=lambda *a, **kw: None,
        finish=lambda *a: None,
        reset=lambda: None,
    )
    guard = CacheGuard(cache)
    sources = []
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        path = Path(directory)

        def source(generation):
            req = SimpleNamespace(
                kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=8),
                disagg_kv_sender=SimpleNamespace(poll=lambda: 4),
                cache_request_handle=generation,
                lock=object(),
                last_node="prefix",
            )
            src = DSV41Source(
                f.p_controller,
                req,
                path / generation,
                generation,
                None,
                None,
                (),
                33,
                cache_lease=guard.hold(req),
            )
            remote.register(src)
            sources.append(src)
            return src

        src = source("success")
        dr = SimpleNamespace(
            req=f.req,
            _pre_layerwise_offer=SimpleNamespace(
                path=src.path, generation=src.generation
            ),
        )
        queue = SimpleNamespace(scheduler=SimpleNamespace(_pre_layerwise=f.controller))
        remote.commit(lambda *a, **kw: None, queue, dr)
        target = f.req._dsv41_remote
        src.req.disagg_kv_sender.poll = lambda: 3
        rejected_before_pd = False
        try:
            target.call(f.query[0], [0], 3, 2, 512**-0.5)
        except RuntimeError:
            rejected_before_pd = True
        src.req.disagg_kv_sender.poll = lambda: 4
        rejected_half_pair = False
        try:
            target.call(f.query[0], [16], 3, 2, 512**-0.5)
        except RuntimeError:
            rejected_half_pair = True
        assert rejected_before_pd and rejected_half_pair
        assert not src.remote_closed.is_set() and pins == {"FULL": 1, "SWA": 1}
        graph_runner = SimpleNamespace(
            model_runner=SimpleNamespace(_pre_layerwise=f.controller)
        )
        assert not hooks._decode_graph(lambda *a: True, graph_runner, f.fb)
        global_before = [p.kv_buffer[0].clone() for p in f.p_pool.kv_pools.values()]
        swa_before = [x.clone() for x in f.p_swa]
        index_before = f.core.c2_sparse_page_indices.clone()
        max_error = 0.0
        kv = torch.zeros((1, 512), device="cuda", dtype=torch.bfloat16)

        def local_native_boundary(backend, q, k, v, layer, fb, ratio, **kwargs):
            # The 4060 cannot run the model's SM100 FlashMLA kernel. Keep the
            # SWA-only native-compute boundary explicit in this small probe.
            assert ratio == 0
            backend.store_cache(layer.layer_id, k, fb)
            keys = math.read_slots(
                f.d_pool.get_swa_key_buffer_radix(layer.layer_id),
                torch.tensor([13, 14], device="cuda"),
                8,
                f.layout.V41,
            )
            return (
                math.partial(q[0], keys, 512**-0.5, f.sink)[0].to(q.dtype).unsqueeze(0)
            )

        def decode(runner, fb):
            nonlocal max_error
            for layer_id, ratio in enumerate((0, 1, 2, 2)):
                got = remote.attention(
                    local_native_boundary,
                    f.backend,
                    f.query,
                    kv,
                    kv,
                    SimpleNamespace(layer_id=layer_id),
                    fb,
                    ratio,
                    attn_sink=f.sink,
                )
                swa = f.oracle.dequantize_k_cache_v41(
                    f.d_pool.get_swa_key_buffer_radix(layer_id).cpu(), 8
                ).reshape(-1, 512)[[13, 14]]
                rows = [swa]
                if ratio:
                    ids = (
                        f.core.sparse_page_indices(ratio)[
                            0, : int(f.core.sparse_topk_lengths(ratio)[0])
                        ]
                        .cpu()
                        .long()
                    )
                    extra = f.oracle.dequantize_k_cache_v41_fp4(
                        f.d_pool.get_extra_key_buffer(layer_id).cpu(), 8 // ratio
                    ).reshape(-1, 512)[ids]
                    rows.append(extra)
                keys = torch.cat(rows).float()
                logits = f.query[0].cpu().float() @ keys.T * 512**-0.5
                logits = torch.cat((logits, f.sink.cpu()[:, None]), dim=-1)
                values = torch.cat((keys, torch.zeros((1, 512))))
                expected = (torch.softmax(logits, dim=-1) @ values).to(torch.bfloat16)
                error = float((got[0].cpu().float() - expected.float()).abs().max())
                max_error = max(error, max_error)
                torch.testing.assert_close(
                    got[0].cpu(), expected, atol=0.015625, rtol=0.01
                )
            return "native-model-result"

        assert remote.forward(f.controller, decode, None, f.fb) == "native-model-result"
        assert target.calls == 3 and target.closed and not f.controller.remote_targets
        for _ in range(32):
            assert hooks._decode_graph(lambda *a: True, graph_runner, f.fb)
        assert torch.equal(index_before, f.core.c2_sparse_page_indices)
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                global_before, [p.kv_buffer[0] for p in f.p_pool.kv_pools.values()]
            )
        )
        assert all(torch.equal(a, b) for a, b in zip(swa_before, f.p_swa))
        deadline = time.monotonic() + 5
        while not remote.source_ready(src):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert pins == {"FULL": 1, "SWA": 1}
        src.cache_lease.retire(lambda: None)
        assert pins == {"FULL": 0, "SWA": 0}

        # A cancelled socket while a P GPU reader is live must tombstone the
        # registry immediately, retain the cache pins, then wait for GPU fence.
        cancel_src = source("cancel")
        route = cancel_src.service
        cancel_req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))
        cancel = remote.Target(
            f.controller, cancel_req, cancel_src.path, cancel_src.generation, route, 33
        )
        f.controller.remote_targets[0] = cancel
        warm = torch.cuda.Stream()
        with torch.cuda.stream(warm):
            torch.cuda._sleep(1_000_000_000)
            ready = torch.cuda.Event()
            ready.record(warm)
        ctx = hub.registry.get(route["handoff_id"], route["epoch"])
        ctx.dsv41_ready = ready
        errors = []

        def call():
            try:
                torch.cuda.set_device(0)
                cancel.call(f.query[0], [0, 15], 3, 2, 512**-0.5)
            except Exception as exc:
                errors.append(type(exc).__name__)

        worker = threading.Thread(target=call)
        worker.start()
        deadline = time.monotonic() + 5
        while not ctx.inflight:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        cancel.close("cancel-during-RPC")
        retained = not remote.source_ready(cancel_src)
        assert retained and pins == {"FULL": 1, "SWA": 1}
        assert hub.registry.acquire(ctx.handoff_id, ctx.owner_epoch) is None
        warm.synchronize()
        worker.join(5)
        assert not worker.is_alive()
        deadline = time.monotonic() + 5
        while not remote.source_ready(cancel_src):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        cancel_src.cache_lease.retire(lambda: None)
        assert pins == {"FULL": 0, "SWA": 0}
        assert not hub.registry.stats()["handoffs"]
        hub.close()
        files = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py"))
        files += [
            Path(__file__).resolve(),
            Path(__file__).with_name("dsv41_remote_probe_support.py").resolve(),
        ]
        files += [
            root / p
            for p in (
                "python/sglang/kernels/ops/attention/dsv4/kv_layout.py",
                "python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py",
                "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
                "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
            )
        ]
        result = dict(
            device=torch.cuda.get_device_name(),
            torch=torch.__version__,
            layouts_bit_exact=layout_cases,
            remote_calls=target.calls,
            rejected_before_pd=rejected_before_pd,
            rejected_half_pair=rejected_half_pair,
            max_abs_error=max_error,
            odd_half_pair_stays_local=True,
            native_topk_unchanged=True,
            shared_source_consumer=True,
            p_kv_swa_unchanged=True,
            local_graph_restored=True,
            cancel_inflight_gpu_retained=retained,
            cancel_rpc_errors=errors,
            source_pins_closed=True,
            scope=__doc__,
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
