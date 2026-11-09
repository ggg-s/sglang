"""Small real-CUDA probe for RemoteGraphBackend, without loading a model.

Source-loads the native full/breakable backends and breakable primitives.
TP, graph-pool borrowing/precarving, and attention/RPC are isolated boundaries;
all graph capture, replay, tensor copies, arithmetic and stream events are real.
This is not a PD inference/accuracy or NCCL test. Default weak tensor views use
sgl_kernel; --allow-strong-refs explicitly tests the reduced-dependency boundary.
"""

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import sys
import threading
import types
import typing
from abc import ABC, abstractmethod
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


def definitions(path, namespace, names=None):
    nodes = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and (names is None or node.name in names)
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


def load_native_backend_base(root, namespace):
    namespace.update(ABC=ABC, abstractmethod=abstractmethod)
    definitions(
        root
        / "python/sglang/srt/model_executor/runner_backend/base_cuda_graph_backend.py",
        namespace,
        names={"BaseCudaGraphBackend"},
    )


def workspace_namespace(root):
    module = load(
        "sglang.srt.model_executor.runner_backend_utils.capture_workspace",
        root
        / "python/sglang/srt/model_executor/runner_backend_utils/capture_workspace.py",
    )
    return {
        "CaptureWorkspacePool": module.CaptureWorkspacePool,
        "get_capture_workspace": module.get_capture_workspace,
        "nullcontext": nullcontext,
    }


def background_allocation_probe(primitive):
    """Allocate unrelated device memory on another thread during capture."""
    ready, start, done, release = (threading.Event() for _ in range(4))
    errors = []
    device = torch.cuda.current_device()

    def worker():
        pointer = None
        try:
            primitive.checkCudaErrors(primitive.rt.cudaSetDevice(device))
            ready.set()
            if not start.wait(10):
                raise TimeoutError("background CUDA allocation did not start")
            pointer = primitive.checkCudaErrors(primitive.rt.cudaMalloc(4096))
        except BaseException as error:
            errors.append(repr(error))
        finally:
            ready.set()
            done.set()
            release.wait(10)
            if pointer is not None:
                try:
                    primitive.checkCudaErrors(primitive.rt.cudaFree(pointer))
                except BaseException as error:
                    errors.append(repr(error))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    graph = primitive.BreakableCUDAGraph()
    try:
        if not ready.wait(10) or errors:
            raise RuntimeError(f"background CUDA setup failed: {errors}")
        x = torch.ones(4, device="cuda")
        with primitive.BreakableCUDAGraphCapture(
            graph, stream=torch.cuda.Stream(), capture_error_mode="thread_local"
        ):
            output = x + 1
            start.set()
            if not done.wait(10) or errors:
                raise RuntimeError(f"background CUDA allocation failed: {errors}")
        x.fill_(4)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, torch.full_like(x, 5))
    finally:
        start.set()
        release.set()
        thread.join(timeout=10)
    if thread.is_alive() or errors:
        raise RuntimeError(f"background CUDA worker failed: {errors}")


def run(
    root, allow_strong_refs, dtype, logits_output=False, marlin_workspaces=False,
    hybrid=False,
):
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
            is_musa=lambda: False,
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
        if marlin_workspaces:
            background_allocation_probe(primitive)

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
            dataclasses=dataclasses,
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
        namespace.update(workspace_namespace(root))
        definitions(
            root / "python/sglang/srt/layers/logits_processor.py",
            namespace,
            names={"LogitsProcessorOutput"},
        )
        load_native_backend_base(root, namespace)
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
        layers = 4 if hybrid else 3
        attention_ids = [0, 2] if hybrid else list(range(layers))
        width = 8
        marlin_allocations = []

        def marlin_properties(device):
            if backend.capture_remote and primitive.is_capturing():
                raise AssertionError("Marlin device query during Remote capture")
            return torch.cuda.get_device_properties(device)

        def marlin_zeros(*args, **kwargs):
            if backend.capture_remote and primitive.is_capturing():
                raise AssertionError("Marlin allocation during Remote capture")
            tensor = torch.zeros(*args, **kwargs)
            if backend.capture_remote:
                # Keep only addresses: the backend, not this probe's log,
                # must retain the scratch tensors until all replay is done.
                marlin_allocations.append(tensor.data_ptr())
            return tensor

        marlin_namespace = {
            "torch": types.SimpleNamespace(
                device=torch.device,
                Tensor=torch.Tensor,
                int=torch.int,
                cuda=types.SimpleNamespace(get_device_properties=marlin_properties),
                zeros=marlin_zeros,
            ),
            "get_capture_workspace": namespace["get_capture_workspace"],
        }
        definitions(
            root / "python/sglang/srt/layers/quantization/marlin_utils.py",
            marlin_namespace,
            names={"marlin_make_workspace"},
        )

        class Attention(torch.nn.Module):
            def __init__(self, index):
                super().__init__()
                self.layer_id, self.tp_q_head_num, self.v_head_dim = index, 1, width

            def forward(self, q, k, v, batch, **kwargs):
                return q

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attentions = torch.nn.ModuleDict(
                    {str(i): Attention(i) for i in attention_ids}
                )
                self.weights = [
                    torch.randn(width, width, device="cuda", dtype=dtype) * 0.1
                    for _ in range(layers)
                ]

            def forward(self, x, batch, reference=False):
                for i, weight in enumerate(self.weights):
                    q = torch.nn.functional.linear(x, weight) + 0.1
                    if marlin_workspaces:
                        scratch = marlin_namespace["marlin_make_workspace"](x.device, 4)
                        q = q + scratch[0].to(q.dtype)
                        # Simulate dirty reduction locks. Repeated replay must
                        # execute the captured reset before reading this slot.
                        scratch.fill_(7)
                    if i not in attention_ids:
                        # Keep a distinct recurrent-like GPU operation in the
                        # graph between full-attention callbacks.
                        q = q + x * 0.125
                    x = (
                        q if reference or i not in attention_ids
                        else self.attentions[str(i)](q, q, q, batch)
                    )
                    x = torch.relu(x) + 0.2
                if logits_output:
                    return namespace["LogitsProcessorOutput"](
                        next_token_logits=x,
                        hidden_states=x + 0.3,
                        hidden_states_token_indices=torch.arange(
                            x.shape[0], device=x.device
                        ),
                    )
                return x

        def tensor_output(output):
            return output.next_token_logits if logits_output else output

        def check_output(output, expected):
            torch.testing.assert_close(tensor_output(output), tensor_output(expected))
            if logits_output:
                torch.testing.assert_close(output.hidden_states, expected.hidden_states)
                torch.testing.assert_close(
                    output.hidden_states_token_indices,
                    expected.hidden_states_token_indices,
                )
                assert output.next_token_logprobs is None
                assert output.next_token_top_logprobs_val is None
                assert output.sampling_mask_output is None
                # Exercise sampler mutations: another replay must return a clean
                # wrapper over the same graph-owned tensors.
                output.next_token_logprobs = torch.ones_like(output.next_token_logits)
                output.next_token_top_logprobs_val = [[0.5]]
                output.sampling_mask_output = object()

        model = Model()
        controller = types.SimpleNamespace(
            graph_fast_path=True, targets=[], remote_targets={},
        )
        mr = types.SimpleNamespace(
            model=model,
            model_config=types.SimpleNamespace(
                num_hidden_layers=layers,
                hf_config=types.SimpleNamespace(full_attention_layer_ids=attention_ids),
            ),
            _pre_layerwise=controller,
            device_timer=None,
            is_draft_worker=False,
        )
        runner = types.SimpleNamespace(
            model_runner=mr,
            device_module=torch.cuda,
            capture_bs=[1, 2, 3, 4, 6, 8, 14, 16, 32],
            attn_backend=types.SimpleNamespace(init_forward_metadata=lambda fb: None),
            cuda_graph_output_capacity_rows=lambda output: tensor_output(output).shape[
                0
            ],
            cuda_graph_output_rows=lambda output: tensor_output(output).shape[0],
        )
        full = namespace["FullCudaGraphBackend"](runner)
        remote = namespace["BreakableCudaGraphBackend"](
            runner, capture_error_mode="thread_local", preallocate_workspaces=True
        )
        remote._pool = torch.cuda.graph_pool_handle()
        backend_type = sg.HybridGraphBackend if hybrid else sg.RemoteGraphBackend
        backend = backend_type(
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
                req_pool_indices=torch.arange(size, dtype=torch.int32, device="cuda"),
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
            assert q.shape[0] == k.shape[0] == v.shape[0] == live.batch_size
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
        hybrid_context = nullcontext()
        if hybrid:
            from sglang_remote_attn import qwen_hybrid

            def hybrid_attention(owner, original, module, q, k, v, live,
                                 graph_rows=None, **kwargs):
                assert owner is controller
                if graph_rows is None:
                    return original(q, k, v, live, **kwargs)
                assert graph_rows == list(range(live.batch_size))
                return attention(module, q, k, v, live, **kwargs)

            hybrid_context = patch.object(qwen_hybrid, "attention", hybrid_attention)
        with (
            patch.object(sg, "_runtime", return_value=runtime),
            hybrid_context,
            torch.inference_mode(),
        ):
            sg.install_attention(mr, hybrid=hybrid)
            stream = torch.cuda.Stream()
            with backend.capture_session(stream):
                for size in reversed(runner.capture_bs):
                    inputs[size] = torch.randn(size, width, device="cuda", dtype=dtype)
                    dummy = batch(size)
                    backend.capture_one(
                        shape(size),
                        lambda size=size, dummy=dummy: model(inputs[size], dummy),
                    )
            assert not callbacks, "capture performed an RPC"
            if marlin_workspaces:
                assert len(marlin_allocations) == layers * 5
                assert len(set(marlin_allocations)) == layers * 5
            segment_counts = {
                str(key.size): len(graph._segments)
                for key, graph in remote._graphs.items()
            }
            assert segment_counts == {
                str(size): len(attention_ids) + 1 for size in (1, 2, 4, 8, 16)
            }, segment_counts
            assert all(ids == tuple(attention_ids) for ids in backend.boundaries.values())
            controller.targets.append(object())
            padded_replays = []
            for size in (16, 3, 5, 6, 7, 13, 14, 15, 8, 4, 1, 2, 16, 3):
                for marker, local in ((11, ()), (12, (0,)), (13, attention_ids)):
                    live = batch(size, marker, local)
                    key = backend.remote_key(live)
                    inputs[key.size].normal_()
                    expected = model(inputs[key.size][:size], live, reference=True)
                    with backend.execution(live), backend.replay_session():
                        output = backend.replay(key, live)
                    # Native decode output slicing is covered by the DSV probe;
                    # inspect this toy graph's raw bridge as well as live rows.
                    if key.size != size:
                        padded_replays.append([size, key.size])
                    if logits_output:
                        output = dataclasses.replace(
                            output,
                            next_token_logits=output.next_token_logits[:size],
                            hidden_states=output.hidden_states[:size],
                            hidden_states_token_indices=output.hidden_states_token_indices[
                                :size
                            ],
                        )
                    else:
                        output = output[:size]
                    check_output(output, expected)
            assert not backend.supports_remote(
                batch(17)
            ) and not backend.supports_remote(batch(32))
            controller.targets.clear()
            before = len(callbacks)
            for size in (16, 8, 4, 2, 1, 32):
                live = batch(size, 21)
                inputs[size].normal_()
                expected = model(inputs[size], live, reference=True)
                with backend.execution(live), backend.replay_session():
                    output = backend.replay(shape(size), live)
                check_output(output, expected)
            assert len(callbacks) == before, "Local full replay invoked RPC"
            controller.targets.append(object())
            live = batch(1, 99, fail=attention_ids[1])
            before = len(callbacks)
            try:
                with backend.execution(live), backend.replay_session():
                    backend.replay(shape(1), live)
                raise AssertionError("RPC failure was swallowed")
            except RuntimeError as error:
                assert str(error) == "injected RPC failure"
            assert callbacks[before:] == [(99, i) for i in attention_ids[:2]]
            assert backend.active_batch is None and not backend.use_remote
            replay_counts = dict(backend.replays)
            backend.cleanup()
            assert backend.closed
        return {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": str(dtype),
            "output_type": "LogitsProcessorOutput" if logits_output else "Tensor",
            "layers": layers,
            "hybrid": hybrid,
            "attention_layers": attention_ids,
            "remote_segments": segment_counts,
            "replays": replay_counts,
            "padded_replays": padded_replays,
            "marlin_workspace_allocations": len(marlin_allocations),
            "remote_capture_error_mode": remote._capture_error_mode,
            "background_allocation_during_capture": marlin_workspaces,
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
                    native / "runner_backend_utils/capture_workspace.py",
                    root / "python/sglang/srt/layers/quantization/marlin_utils.py",
                    root / "python/sglang/srt/layers/logits_processor.py",
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
    parser.add_argument("--logits-output", action="store_true")
    parser.add_argument("--marlin-workspaces", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
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
                args.logits_output,
                args.marlin_workspaces,
                args.hybrid,
            ),
            indent=2,
        )
    )
