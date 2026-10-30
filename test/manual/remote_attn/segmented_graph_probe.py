"""Small real-CUDA probe for RemoteGraphBackend, without loading a model.

Source-loads the native full/breakable backends and breakable primitives.
TP, graph-pool borrowing/precarving, and attention/RPC are isolated boundaries;
all graph capture, replay, tensor copies, arithmetic and stream events are real.
This is not a PD inference/accuracy or NCCL test. Default weak tensor views use
sgl_kernel; --allow-strong-refs explicitly tests the reduced-dependency boundary.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
import types
import typing
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(path, namespace):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def run(root, allow_strong_refs, dtype):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prefix = "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
    native = root / "python/sglang/srt/model_executor"
    primitive_dir = native / "runner_backend_utils/breakable_cuda_graph"
    with patch.dict(sys.modules, {}):
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.model_executor",
            "sglang.srt.model_executor.runner_backend_utils",
            prefix,
        ):
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        sys.modules["sglang.srt.utils"] = types.SimpleNamespace(
            get_device_module=lambda: torch.cuda,
            is_hip=lambda: False,
            is_xpu=lambda: False,
        )
        strong = False
        try:
            from sgl_kernel import weak_ref_tensor
        except ImportError:
            if not allow_strong_refs:
                raise RuntimeError(
                    "sgl_kernel is missing; use --allow-strong-refs only for the reduced-dependency probe"
                )
            weak_ref_tensor = lambda tensor: tensor
            strong = True
        sys.modules[
            "sglang.srt.model_executor.runner_backend_utils.weak_ref_tensor"
        ] = types.SimpleNamespace(weak_ref_tensors=weak_ref_tensor)
        load(prefix + ".cuda_utils", primitive_dir / "cuda_utils.py")
        primitive = load(
            prefix + ".breakable_cuda_graph", primitive_dir / "breakable_cuda_graph.py"
        )

        class Precarve:
            def measure(self):
                return nullcontext()

            def mint(self):
                pass

        class Dedup:
            deduped_cuda_graph = None

            def begin_cuda_graph_capture(self):
                pass

            def end_cuda_graph_capture(self):
                pass

            def close(self):
                pass

        pools = {}
        namespace = {
            key: value
            for key, value in vars(typing).items()
            if not key.startswith("__")
        }
        namespace.update(
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
        definitions(native / "runner_backend/full_cuda_graph_backend.py", namespace)
        definitions(
            native / "runner_backend/breakable_cuda_graph_backend.py", namespace
        )
        shape = load("probe_shape_key", native / "runner/shape_key.py").ShapeKey
        sys.path.insert(0, str(root / "plugins/remote_attn"))
        from sglang_remote_attn import segmented_graph as sg

        torch.manual_seed(19)
        callbacks = []
        inputs = {}
        layers = 3
        width = 8

        class Attention(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                self.layer_id, self.tp_q_head_num, self.v_head_dim = index, 1, width

            def forward(self, q, k, v, batch, **kwargs):
                return q

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attentions = torch.nn.ModuleList(
                    Attention(i) for i in range(layers)
                )
                self.weights = [
                    torch.randn(width, width, device="cuda", dtype=dtype) * 0.1
                    for _ in range(layers)
                ]

            def forward(self, x, batch, reference=False):
                for module, weight in zip(self.attentions, self.weights):
                    q = torch.nn.functional.linear(x, weight) + 0.1
                    x = q if reference else module(q, q, q, batch)
                    x = torch.relu(x) + 0.2
                return x

        model = Model()
        controller = types.SimpleNamespace(graph_fast_path=True, targets=[])
        mr = types.SimpleNamespace(
            model=model,
            model_config=types.SimpleNamespace(num_hidden_layers=layers),
            _pre_layerwise=controller,
            device_timer=None,
            is_draft_worker=False,
        )
        runner = types.SimpleNamespace(
            model_runner=mr,
            device_module=torch.cuda,
            cuda_graph_output_capacity_rows=lambda output: output.shape[0],
            cuda_graph_output_rows=lambda output: output.shape[0],
        )
        full = namespace["FullCudaGraphBackend"](runner)
        remote = namespace["BreakableCudaGraphBackend"](runner)
        remote._pool = torch.cuda.graph_pool_handle()
        backend = sg.RemoteGraphBackend(
            runner,
            full,
            remote,
            primitive.eager_on_graph,
            primitive.is_capturing,
            lambda: None,
        )
        mr._remote_segmented_backend = backend
        runner.backend = backend

        def batch(size, marker=-1, local=(), fail=None):
            return types.SimpleNamespace(
                batch_size=size,
                input_ids=torch.zeros(size, dtype=torch.int64, device="cuda"),
                input_embeds=None,
                replace_embeds=None,
                forward_mode=types.SimpleNamespace(is_decode=lambda: True),
                marker=marker,
                local_layers=local,
                fail=fail,
            )

        def attention(module, q, k, v, live, **kwargs):
            assert (
                not primitive.is_capturing()
                and not torch.cuda.is_current_stream_capturing()
            )
            assert live.marker >= 0, "capture-time request escaped into replay"
            if module.layer_id in live.local_layers:
                return q
            callbacks.append((live.marker, module.layer_id))
            if module.layer_id == live.fail:
                raise RuntimeError("injected RPC failure")
            # Synchronous host round trip models an RPC boundary. D's local
            # partial and the returned partial are merged outside the graph.
            source = q.detach().cpu() * 0.5
            return q * 0.5 + source.to(q.device)

        runtime = types.SimpleNamespace(
            has_target_states=lambda: bool(controller.targets),
            _row_meta=lambda fb: {"states": []},
            _handoff_attention_forward=attention,
        )
        sys.modules["sglang.srt.layers.radix_attention"] = types.SimpleNamespace(
            RadixAttention=Attention
        )
        sys.modules["sglang.srt.environ"] = types.SimpleNamespace(
            envs=types.SimpleNamespace(
                SGLANG_LOG_DECODE_GRAPH_KEY=types.SimpleNamespace(get=lambda: False)
            )
        )
        with patch.object(sg, "_runtime", return_value=runtime), torch.inference_mode():
            sg.install_attention(mr)
            stream = torch.cuda.Stream()
            with backend.capture_session(stream):
                for size in (8, 4, 2, 1):
                    inputs[size] = torch.randn(size, width, device="cuda", dtype=dtype)
                    dummy = batch(size)
                    backend.capture_one(
                        shape(size),
                        lambda size=size, dummy=dummy: model(inputs[size], dummy),
                    )
            assert not callbacks, "capture performed an RPC"
            segment_counts = {
                str(key.size): len(graph._segments)
                for key, graph in remote._graphs.items()
            }
            assert segment_counts == {
                "1": layers + 1,
                "2": layers + 1,
                "4": layers + 1,
            }, segment_counts
            controller.targets.append(object())
            for size in (1, 2, 4, 1):
                for marker, local in ((11, ()), (12, (0,)), (13, (0, 1, 2))):
                    live = batch(size, marker, local)
                    inputs[size].normal_()
                    expected = model(inputs[size], live, reference=True)
                    with backend.execution(live), backend.replay_session():
                        output = backend.replay(shape(size), live)
                    torch.testing.assert_close(output, expected)
            assert not backend.supports_remote(
                batch(3)
            ) and not backend.supports_remote(batch(8))
            controller.targets.clear()
            before = len(callbacks)
            for size in (1, 2, 4, 8):
                live = batch(size, 21)
                inputs[size].normal_()
                expected = model(inputs[size], live, reference=True)
                with backend.execution(live), backend.replay_session():
                    output = backend.replay(shape(size), live)
                torch.testing.assert_close(output, expected)
            assert len(callbacks) == before, "Local full replay invoked RPC"
            controller.targets.append(object())
            live = batch(1, 99, fail=1)
            before = len(callbacks)
            try:
                with backend.execution(live), backend.replay_session():
                    backend.replay(shape(1), live)
                raise AssertionError("RPC failure was swallowed")
            except RuntimeError as error:
                assert str(error) == "injected RPC failure"
            assert callbacks[before:] == [(99, 0), (99, 1)]
            assert backend.active_batch is None and not backend.use_remote
            replay_counts = dict(backend.replays)
            backend.cleanup()
            assert backend.closed
        return {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": str(dtype),
            "layers": layers,
            "remote_segments": segment_counts,
            "replays": replay_counts,
            "rpc_calls": len(callbacks),
            "capture_rpc_calls": 0,
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "strong_ref_boundary": strong,
            "source_sha256": {
                str(path.relative_to(root)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in (
                    root / "plugins/remote_attn/sglang_remote_attn/segmented_graph.py",
                    Path(__file__),
                    native / "runner_backend/full_cuda_graph_backend.py",
                    native / "runner_backend/breakable_cuda_graph_backend.py",
                    primitive_dir / "breakable_cuda_graph.py",
                )
            },
            "scope": "real CUDA; source-loaded native backends; toy attention/RPC; TP1; no pool borrowing/precarving",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--allow-strong-refs", action="store_true")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.source_root.resolve(),
                args.allow_strong_refs,
                getattr(torch, args.dtype),
            ),
            indent=2,
        )
    )
