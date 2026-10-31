"""Real CUDA full/segmented banks plus DSV sparse RemoteHub RPC on RTX4060.

Native full/breakable backends, QKV/state methods and sparse reader are source-loaded.
Linear/norm/store kernels and the 3-layer model/metadata/indexer/carry planning
boundaries are small fixtures. No DSV checkpoint, SM100, SMG, NCCL or perf claim.
"""

import argparse
import hashlib
import json
import os
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


def main(root, output, *, fused=False, bf16_source=False, mesh=False):
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
                SGLANG_LOG_DECODE_GRAPH_KEY=types.SimpleNamespace(get=lambda: False),
                SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE=types.SimpleNamespace(
                    get=lambda: bf16_source
                ),
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

        def make_hub(fixture, index):
            scheduler = fixture.p_controller.scheduler
            if mesh:
                scheduler.disaggregation_mode = types.SimpleNamespace(value="prefill")
                scheduler.disagg_prefill_bootstrap_queue = types.SimpleNamespace(
                    kv_manager=types.SimpleNamespace(local_ip=f"127.0.0.{3 + index}")
                )
            with patch.dict(
                os.environ,
                {"HANDOFF_PD_NETWORK": "1", "HANDOFF_INSTANCE_ID": f"graph-p{index}"}
                if mesh
                else {},
            ):
                return rt.RemoteHub(
                    scheduler,
                    stream_factory=lambda n: [
                        (torch.cuda.Stream(), None, None) for _ in range(n)
                    ],
                )

        hub = make_hub(sf, 0)
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
        source_fixtures, sources, hubs = [sf], [source], [hub]
        cluster = None
        if mesh:
            from pd_mesh_probe_support import Cluster

            cluster = Cluster(root, Path(temp) / "mesh")
            second = sparse_fixture(root, "cuda")
            second.p_controller.dsv41_remote = True
            # Distinct native bytes ensure a mixed-source row cannot silently
            # read another P's cache and still pass the numerical comparison.
            for pool in second.p_pool.kv_pools.values():
                pool.kv_buffer[0].bitwise_xor_(1)
            second_hub = make_hub(second, 1)
            second.p_controller.scheduler._handoff_hub = second_hub
            second_source = DSV41Source(
                second.p_controller,
                types.SimpleNamespace(
                    kv=types.SimpleNamespace(req_pool_idx=0),
                    cache_request_handle="graph",
                    disagg_kv_sender=types.SimpleNamespace(poll=lambda: 3),
                ),
                Path(temp) / "p1",
                "graph-p1",
                None,
                None,
                (),
                33,
                schema=dict(entries=[["kv", 1, 1], ["kv", 2, 2]]),
                barrier_ready=True,
            )
            sparse.register(second_source)
            source_fixtures.append(second)
            sources.append(second_source)
            hubs.append(second_hub)
        rpc_calls = []
        seen_variants = set()
        live_targets = {}

        def selected(marker):
            return [0, 15] if marker % 2 else [15, 0, 15]

        def keys(layer, marker, row=0):
            sf = source_fixtures[row % len(source_fixtures)]
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
                        query, keys(layer.layer_id, fb.marker, row), 512**-0.5
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

        f = fixture(
            "cuda",
            torch.bfloat16,
            on_attention=attention,
            fused=fused,
            bf16_source=bf16_source,
        )
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
        # Native runner owns these metadata buffers for the lifetime of its full
        # graph bank. Keep the fixture's equivalents alive as well: CUDA graph
        # kernels hold raw addresses, not Python tensor ownership.
        capture_batches = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with backend.capture_session(stream):
            for size in (8, 4, 2, 1):
                x = torch.ones((size, 4), device="cuda", dtype=torch.bfloat16)
                positions = torch.zeros(size, device="cuda", dtype=torch.int64)
                dummy = f.batch(size, -100)
                capture_batches[size] = dummy
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
        projection_capture_calls = len(f.projections)
        for buffer in f.cache:
            buffer.fill_(-99)
        for buffer in f.carry:
            buffer.zero_()
        carry_expected = [buf.clone() for buf in f.carry]
        assert sorted(backend.remote_keys) == [1, 2, 4]
        segments = {
            str(key.size): len(value._segments) for key, value in remote._graphs.items()
        }
        assert segments == {"1": 8, "2": 8, "4": 8}, segments
        f.controller.targets = [object()]
        for row in range(4):
            req = types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=row))
            current_source = sources[row % len(sources)]
            path, generation, route = Path(temp), "graph", current_source.service
            if cluster:
                path, generation, route = cluster.route(row % 2, 0, route, 100 + row)
            target = sparse.Target(
                f.controller,
                req,
                path,
                generation,
                route,
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
                    kv_expected = []
                    for layer in (0, 1, 2):
                        if bf16_source:
                            kv = expected * 3 + 0.125
                            kv.mul_(0.5).add_(p[:, None] * 0.03125)
                        else:
                            kv = (
                                (expected * 3 + 0.125).float() * 0.5
                                + p.float()[:, None] * 0.03125
                            ).to(x.dtype)
                        kv_expected.append(kv)
                        if layer:
                            carry_expected[layer].index_add_(
                                0, live.out_cache_loc, expected.sum(-1)
                            )
                        expected = expected * 2 + p[:, None]
                        if layer:
                            queries = (
                                expected.repeat(1, 128)
                                .reshape(size, 1, 512)
                                .to(torch.bfloat16)
                            )
                            expected = (
                                torch.stack(
                                    [
                                        math.partial(
                                            q, keys(layer, marker, row), 512**-0.5
                                        )[0][0, :4]
                                        for row, q in enumerate(queries)
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
                    assert len(f.projections) == projection_capture_calls, (
                        "QKV executed in eager replay"
                    )
                    for layer in (0, 1, 2):
                        torch.testing.assert_close(
                            f.cache[layer][live.out_cache_loc],
                            kv_expected[layer],
                            atol=0,
                            rtol=0,
                        )
                        torch.testing.assert_close(
                            f.carry[layer], carry_expected[layer], atol=0, rtol=0
                        )
                    assert mode.get_capture_attention_variant() is None
                    assert f.metadata[-1] == marker
                    assert f.prepared[-3:] == [(marker, i) for i in range(3)]
        assert all(s.req.disagg_kv_sender.poll() == 3 for s in sources)
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
        if cluster:
            from sglang_remote_attn.pre_layerwise import read_json

            # A second D independently uses both P endpoints after D0 retires.
            # No D0 cancellation/close may detach D1's readers.
            for row in (0, 1):
                path, generation, route = cluster.route(
                    row, 1, sources[row].service, 200 + row
                )
                req = types.SimpleNamespace(kv=types.SimpleNamespace(req_pool_idx=row))
                target = sparse.Target(
                    f.controller, req, path, generation, route, 33, progressive=True
                )
                q = torch.ones((1, 512), device="cuda", dtype=torch.bfloat16)
                out, _ = target.call(q, selected(8), 1, 1, 512**-0.5)
                expected, _ = math.partial(q, keys(1, 8, row), 512**-0.5)
                torch.testing.assert_close(out, expected)
                target.close("d1-complete")
            cluster.wait(
                lambda: all(read_json(p / "remote-done.json") for p, _ in cluster.links)
            )
        backend.cleanup()
        for hub in hubs:
            hub.close()
        if cluster:
            cluster.close()
        paths = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py"))
        paths += [
            Path(__file__),
            Path(__file__).with_name("dsv41_graph_probe_support.py"),
            Path(__file__).with_name("dsv41_remote_probe_support.py"),
            Path(__file__).with_name("segmented_graph_probe.py"),
        ]
        if cluster:
            paths += [
                Path(__file__).with_name("pd_mesh_probe_support.py"),
                root / "python/sglang/srt/disaggregation/mooncake/conn.py",
                root / "python/sglang/srt/disaggregation/common/conn.py",
            ]
        paths += [
            root / p
            for p in (
                "python/sglang/kernels/ops/attention/dsv4/kv_layout.py",
                "python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py",
                "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
                "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
                "python/sglang/srt/models/deepseek_v4.py",
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
            qkv_projection_replay_host_calls=0,
            native_projection_state_source_loaded=True,
            fused_projection=fused,
            bf16_kv_source=bf16_source,
            live_swa_and_carry_verified=True,
            unsupported_remote_eager=[3, 8],
            local_full_restored=True,
            mesh=dict(
                prefills=2,
                decodes=2,
                mixed_source_batch=True,
                separate_journals=True,
                independent_d1_after_d0_retirement=True,
            )
            if cluster
            else None,
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
    parser.add_argument("--fused-projection", action="store_true")
    parser.add_argument("--bf16-kv-source", action="store_true")
    parser.add_argument("--mesh", action="store_true")
    args = parser.parse_args()
    main(
        args.source_root.resolve(),
        args.output.resolve(),
        fused=args.fused_projection,
        bf16_source=args.bf16_kv_source,
        mesh=args.mesh,
    )
