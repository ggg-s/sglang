"""Real CUDA full/segmented banks plus DSV sparse RemoteHub RPC on RTX4060.

Native full/breakable backends and byte-layout sparse reader are source-loaded.
The 3-layer model, metadata/indexer/carry planning and pool manager boundaries
are small fixtures. No DSV checkpoint, SM100 kernel, SMG, NCCL or perf claim.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import types
import typing
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from dsv41_graph_probe_support import fixture
from dsv41_remote_probe_support import fixture as sparse_fixture
from segmented_graph_probe import definitions, load
from sglang_remote_attn import dsv41_graph as dg
from sglang_remote_attn import dsv41_remote as sparse
from sglang_remote_attn import dsv41_sparse_math as math
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import hooks
from sglang_remote_attn.dsv41_layerwise import DSV41Source


def main(root, output):
    torch.cuda.set_device(0)
    native = root / "python/sglang/srt/model_executor"
    prefix = "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
    primitives = native / "runner_backend_utils/breakable_cuda_graph"
    with (
        patch.dict(sys.modules, {}),
        tempfile.TemporaryDirectory(dir=output.parent) as temp,
    ):
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.model_executor",
            "sglang.srt.model_executor.runner_backend_utils",
            prefix,
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
        sys.modules["sglang.srt.environ"] = types.SimpleNamespace(
            envs=types.SimpleNamespace(
                SGLANG_LOG_DECODE_GRAPH_KEY=types.SimpleNamespace(get=lambda: False)
            )
        )
        sys.modules["sglang.srt.utils"] = types.SimpleNamespace(
            get_device_module=lambda: torch.cuda,
            is_hip=lambda: False,
            is_xpu=lambda: False,
        )
        # Real native weak tensor views; never silently substitute strong refs.
        from sgl_kernel import weak_ref_tensor

        sys.modules[
            "sglang.srt.model_executor.runner_backend_utils.weak_ref_tensor"
        ] = types.SimpleNamespace(weak_ref_tensors=weak_ref_tensor)
        load(prefix + ".cuda_utils", primitives / "cuda_utils.py")
        mode = types.ModuleType("sglang.srt.model_executor.runner_utils.capture_mode")
        mode.__dict__.update(
            contextmanager=contextmanager, _capture_attention_variant=None
        )
        definitions(native / "runner_utils/capture_mode.py", mode.__dict__)
        sys.modules[mode.__name__] = mode
        labels = (
            "candidate_all",
            "candidate_c2_all",
            "candidate_unfiltered",
            "candidate_filtered",
        )
        primitive = load(
            prefix + ".breakable_cuda_graph", primitives / "breakable_cuda_graph.py"
        )

        class Dedup:
            deduped_cuda_graph = None

            def begin_cuda_graph_capture(self):
                pass

            def end_cuda_graph_capture(self):
                pass

            def close(self):
                pass

        class Precarve:
            def measure(self):
                return nullcontext()

            def mint(self):
                pass

        pools = {}
        ns = {k: v for k, v in vars(typing).items() if not k.startswith("__")}
        ns.update(
            torch=torch,
            contextmanager=contextmanager,
            AbstractContextManager=typing.ContextManager,
            BaseCudaGraphBackend=object,
            DedupedCudaGraphMixin=Dedup,
            PPProxyTensors=type("PPProxyTensors", (), {}),
            GraphPoolPrecarve=Precarve,
            TorchMemorySaverAdapter=types.SimpleNamespace(create=lambda **kw: None),
            get_bool_env_var=lambda *a: False,
            set_graph_pool_id=lambda *a: None,
            get_parallel=lambda: types.SimpleNamespace(
                tp_group=types.SimpleNamespace(barrier=lambda: None)
            ),
            get_or_create_global_graph_memory_pool=lambda device: pools.setdefault(
                "full", device.graph_pool_handle()
            ),
            graph_pool_capture_scope=nullcontext,
            graph_pool_replay_scope=nullcontext,
            eager_on_graph=primitive.eager_on_graph,
            BreakableCUDAGraph=primitive.BreakableCUDAGraph,
            BreakableCUDAGraphCapture=primitive.BreakableCUDAGraphCapture,
            enable_breakable_cuda_graph=nullcontext,
        )
        definitions(native / "runner_backend/full_cuda_graph_backend.py", ns)
        definitions(native / "runner_backend/breakable_cuda_graph_backend.py", ns)
        ShapeKey = load("dsv_segment_shape", native / "runner/shape_key.py").ShapeKey
        sf = sparse_fixture(root, "cuda")
        sf.p_controller.dsv41_remote = True
        hub = rt.RemoteHub(
            sf.p_controller.scheduler,
            stream_factory=lambda n: [
                (torch.cuda.Stream(), None, None) for _ in range(n)
            ],
        )
        sf.p_controller.scheduler._handoff_hub = hub
        sys.modules["sglang.srt.disaggregation.base.conn"] = types.SimpleNamespace(
            KVPoll=types.SimpleNamespace(Success=4, Failed=0)
        )
        source = DSV41Source(
            sf.p_controller,
            types.SimpleNamespace(
                kv=types.SimpleNamespace(req_pool_idx=0),
                cache_request_handle="graph",
                disagg_kv_sender=types.SimpleNamespace(poll=lambda: 3),
            ),
            Path(temp),
            "graph",
            None,
            None,
            (),
            33,
            schema=dict(entries=[["kv", 1, 1], ["kv", 2, 2]]),
            barrier_ready=True,
        )
        sparse.register(source)
        rpc_calls = []
        seen_variants = set()
        live_targets = {}

        def selected(marker):
            return [0, 15] if marker % 2 else [15, 0, 15]

        def keys(layer, marker):
            ratio = layer
            cpp = 8 // ratio
            ids = torch.tensor(
                [sf.p_pages[p // cpp] * cpp + p % cpp for p in selected(marker)],
                device="cuda",
            )
            return math.read_slots(
                sf.p_pool.get_extra_key_buffer(layer), ids, cpp, sf.layout.V41_FP4
            )

        def attention(q, layer, fb):
            assert (
                not torch.cuda.is_current_stream_capturing()
                and not primitive.is_capturing()
            )
            assert mode.get_capture_attention_variant() == fb.variant
            seen_variants.add(fb.variant)
            if layer.layer_id == 0:
                return q
            outs = []
            for row in range(fb.batch_size):
                target = live_targets[row]
                query = q[row].repeat(128).reshape(1, 512).to(torch.bfloat16)
                if layer.layer_id in target.local_entries:
                    out, _ = math.partial(
                        query, keys(layer.layer_id, fb.marker), 512**-0.5
                    )
                else:
                    out, _ = target.call(
                        query,
                        selected(fb.marker),
                        layer.layer_id,
                        layer.layer_id,
                        512**-0.5,
                    )
                    rpc_calls.append((fb.marker, row, layer.layer_id))
                outs.append(out[0, :4].to(q.dtype) + q[row])
            return torch.stack(outs)

        f = fixture("cuda", torch.bfloat16, on_attention=attention)
        f.runner.attention_graph_variants = types.SimpleNamespace(
            capture_labels=labels, graph_limits=()
        )
        full = ns["FullCudaGraphBackend"](f.runner)
        remote = ns["BreakableCudaGraphBackend"](f.runner)
        remote._pool = torch.cuda.graph_pool_handle()
        backend = dg.DSV41GraphBackend(
            f.runner,
            full,
            remote,
            primitive.eager_on_graph,
            primitive.is_capturing,
            lambda: None,
        )
        f.attach(backend)
        inputs = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with backend.capture_session(stream):
            for size in (8, 4, 2, 1):
                x = torch.ones((size, 4), device="cuda", dtype=torch.bfloat16)
                positions = torch.zeros(size, device="cuda", dtype=torch.bfloat16)
                dummy = f.batch(size, -100)
                inputs[size] = (x, positions)
                for label in labels:
                    with dg.attention_variant(label):
                        backend.capture_one(
                            ShapeKey(size, attention_variant=label),
                            lambda x=x, p=positions, fb=dummy: f.model(x, p, fb),
                            capture_inputs=(x, positions),
                        )
        torch.cuda.synchronize()
        assert not rpc_calls and not f.calls and not f.prepared
        assert sorted(backend.remote_keys) == [1, 2, 4]
        segments = {
            str(key.size): len(value._segments) for key, value in remote._graphs.items()
        }
        assert segments == {"1": 8, "2": 8, "4": 8}, segments
        f.controller.targets = [object()]
        for row in range(4):
            req = types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=row))
            target = sparse.Target(
                f.controller,
                req,
                Path(temp),
                "graph",
                source.service,
                33,
                progressive=True,
            )
            live_targets[row] = target
        f.controller.remote_targets = live_targets
        sys.modules["sglang.srt.runtime_context"] = types.SimpleNamespace(
            get_parallel=lambda: types.SimpleNamespace(tp_size=1, tp_rank=0)
        )
        max_error = 0.0
        for phase, local in enumerate(((), (1,), (1, 2))):
            for target in live_targets.values():
                target.local_entries = set(local)
            for size in (1, 2, 4):
                for marker in (3, 8):
                    live = f.batch(size, marker)
                    live.variant = labels[(phase + marker) % 4]
                    x, p = inputs[size]
                    x.fill_(marker * 0.1)
                    p.fill_(marker)
                    # Independent eager union computation; no graph/RPC state.
                    expected = x.clone()
                    for layer in (0, 1, 2):
                        expected = expected * 2 + p[:, None]
                        if layer:
                            queries = (
                                expected.repeat(1, 128)
                                .reshape(size, 1, 512)
                                .to(torch.bfloat16)
                            )
                            k = keys(layer, marker)
                            expected = (
                                torch.stack(
                                    [
                                        math.partial(q, k, 512**-0.5)[0][0, :4]
                                        for q in queries
                                    ]
                                ).to(x.dtype)
                                + expected
                            )
                        expected = expected + 1
                    assert hooks._decode_graph(lambda *a: True, f.runner, live)
                    with backend.execution(live), backend.replay_session():
                        got = backend.replay(
                            ShapeKey(size, attention_variant=live.variant), live
                        )
                    err = float((got.float() - expected.float()).abs().max())
                    max_error = max(max_error, err)
                    torch.testing.assert_close(got, expected, atol=0.015625, rtol=0.01)
                    assert mode.get_capture_attention_variant() is None
                    assert f.metadata[-1] == marker
                    assert f.prepared[-3:] == [(marker, i) for i in range(3)]
        assert source.req.disagg_kv_sender.poll() == 3
        count = len(rpc_calls)
        for size in (3, 8):
            assert not hooks._decode_graph(lambda *a: True, f.runner, f.batch(size))
        f.controller.targets = []
        for target in list(live_targets.values()):
            target.close("tp-local-graph-probe")
        for step in range(8):
            live = f.batch(4, 42)
            x, p = inputs[4]
            x.fill_(0.25)
            p.fill_(2)
            expected = x.clone()
            for _ in range(3):
                expected = expected * 2 + p[:, None] + 1
            assert hooks._decode_graph(lambda *a: True, f.runner, live)
            with backend.execution(live), backend.replay_session():
                got = backend.replay(
                    ShapeKey(4, attention_variant=labels[step % 4]), live
                )
            torch.testing.assert_close(got, expected)
        assert len(rpc_calls) == count
        assert seen_variants == set(labels)
        assert len(full._graphs) == 16
        backend.cleanup()
        hub.close()
        paths = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py"))
        paths += [
            Path(__file__),
            Path(__file__).with_name("dsv41_graph_probe_support.py"),
            Path(__file__).with_name("dsv41_remote_probe_support.py"),
            Path(__file__).with_name("segmented_graph_probe.py"),
        ]
        paths += [
            root / p
            for p in (
                "python/sglang/kernels/ops/attention/dsv4/kv_layout.py",
                "python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py",
                "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
                "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
            )
        ]
        paths += [
            native / "runner_backend/full_cuda_graph_backend.py",
            native / "runner_backend/breakable_cuda_graph_backend.py",
            native / "runner/shape_key.py",
            native / "runner_utils/capture_mode.py",
            primitives / "breakable_cuda_graph.py",
            primitives / "cuda_utils.py",
        ]
        result = dict(
            device=torch.cuda.get_device_name(),
            torch=torch.__version__,
            remote_buckets=[1, 2, 4],
            native_full_shapes=16,
            native_candidate_variants=sorted(seen_variants),
            remote_segments=segments,
            remote_replays=backend.replays["remote"],
            full_replays=backend.replays["full"],
            rpc_calls=count,
            capture_rpc_calls=0,
            max_abs_error=max_error,
            live_metadata=True,
            live_sparse_topk=True,
            unsupported_remote_eager=[3, 8],
            local_full_restored=True,
            scope=__doc__,
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths
            },
        )
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root.resolve(), args.output.resolve())
