"""Interleaved native full/Remote CUDA capture with an injected background API.

This diagnoses capture error scope, not the unknown triggering API in H20
serving. Model math, TP, dedup and allocator precarving are fixture boundaries.
"""

import argparse
import dataclasses
import hashlib
import json
import sys
import threading
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from segmented_graph_probe import definitions, load
from sglang_remote_attn import segmented_graph as sg


def run(root, buckets, full_mode):
    prefix = "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
    native = root / "python/sglang/srt/model_executor"
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
        get_device_module=lambda: torch.cuda, is_hip=lambda: False, is_xpu=lambda: False
    )
    sys.modules["sglang.srt.model_executor.runner_backend_utils.weak_ref_tensor"] = (
        types.SimpleNamespace(weak_ref_tensors=lambda x: x)
    )
    primitive_dir = native / "runner_backend_utils/breakable_cuda_graph"
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

    ns = dict(
        torch=torch,
        dataclasses=dataclasses,
        contextmanager=contextmanager,
        nullcontext=nullcontext,
        BaseCudaGraphBackend=object,
        DedupedCudaGraphMixin=Dedup,
        PPProxyTensors=type("PPProxyTensors", (), {}),
        GraphPoolPrecarve=Precarve,
        TorchMemorySaverAdapter=types.SimpleNamespace(create=lambda **kw: None),
        get_parallel=lambda: types.SimpleNamespace(
            tp_group=types.SimpleNamespace(barrier=lambda: None)
        ),
        get_bool_env_var=lambda *a: False,
        set_graph_pool_id=lambda *a: None,
        get_or_create_global_graph_memory_pool=lambda device: (
            device.graph_pool_handle()
        ),
        graph_pool_capture_scope=nullcontext,
        graph_pool_replay_scope=nullcontext,
        BreakableCUDAGraph=primitive.BreakableCUDAGraph,
        BreakableCUDAGraphCapture=primitive.BreakableCUDAGraphCapture,
        eager_on_graph=primitive.eager_on_graph,
        enable_breakable_cuda_graph=nullcontext,
    )
    definitions(native / "runner_backend/full_cuda_graph_backend.py", ns)
    definitions(native / "runner_backend/breakable_cuda_graph_backend.py", ns)
    runner = types.SimpleNamespace(
        device_module=torch.cuda,
        model_runner=types.SimpleNamespace(
            model_config=types.SimpleNamespace(num_hidden_layers=1)
        ),
        cuda_graph_output_capacity_rows=lambda out: out.shape[0],
        cuda_graph_output_rows=lambda out: out.shape[0],
    )
    full = ns["FullCudaGraphBackend"](runner)
    remote = ns["BreakableCudaGraphBackend"](runner, capture_error_mode="thread_local")
    remote._pool = torch.cuda.graph_pool_handle()
    backend = sg.RemoteGraphBackend(
        runner,
        full,
        remote,
        primitive.eager_on_graph,
        primitive.is_capturing,
        lambda: None,
    )
    if full_mode is not None:
        full._capture_error_mode = full_mode

    @dataclasses.dataclass(frozen=True)
    class Shape:
        size: int
        stream_idx: object = None
        variant_label: object = None
        attention_variant: object = None

    eager = primitive.eager_on_graph(lambda x: x + 1)
    start, initialized, done, release = (threading.Event() for _ in range(4))
    allocation, errors = {}, []

    def worker():
        primitive.rt.cudaSetDevice(0)
        initialized.set()
        start.wait()
        result = primitive.rt.cudaMalloc(4096)
        allocation["error"] = int(result[0])
        done.set()
        release.wait()
        if int(result[0]) == 0:
            primitive.rt.cudaFree(result[1])

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    assert initialized.wait(10)
    inputs, steps, injected = {}, [], False

    def forward(size):
        nonlocal injected
        if (
            size == 1
            and backend.remote_keys
            and not backend.capture_remote
            and torch.cuda.is_current_stream_capturing()
            and not injected
        ):
            injected = True
            start.set()
            assert done.wait(10)
            info = primitive.rt.cudaStreamGetCaptureInfo(
                torch.cuda.current_stream().cuda_stream
            )
            allocation["capture_status"] = int(info[1])
        x = inputs[size] * 2
        if backend.capture_remote:
            if primitive.is_capturing():
                backend.capture_layers.append(0)
            x = eager(x)
        else:
            x = x + 1
        return x * 3

    try:
        with patch.object(sg, "REMOTE_BATCHES", tuple(buckets)):
            with backend.capture_session(torch.cuda.Stream()):
                for size in sorted(set(buckets), reverse=True):
                    inputs[size] = torch.randn(size, 32, device="cuda")
                    backend.capture_one(Shape(size), lambda size=size: forward(size))
                    steps.append(size)
        for size in steps:
            expected = (inputs[size] * 2 + 1) * 3
            for bank in (full, remote):
                actual = bank.replay(Shape(size), None)
                torch.cuda.synchronize()
                torch.testing.assert_close(actual, expected)
    except BaseException as exc:
        errors.append(str(exc))
    finally:
        # Initialize the idle worker only after capture for the bs=1-only control.
        start.set()
        release.set()
        thread.join(10)
        backend.cleanup()
    files = [
        root / "plugins/remote_attn/sglang_remote_attn/segmented_graph.py",
        native / "runner_backend/full_cuda_graph_backend.py",
        native / "runner_backend/breakable_cuda_graph_backend.py",
        Path(__file__),
    ]
    return dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        buckets=buckets,
        captured=steps,
        injected_during_full_bs1=injected,
        allocation=allocation,
        full_mode=full._capture_error_mode,
        remote_mode=remote._capture_error_mode,
        errors=errors,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
        scope=__doc__,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--buckets", default="1,2,4,8,16")
    parser.add_argument("--full-mode", choices=("global", "thread_local"))
    args = parser.parse_args()
    result = run(
        args.source_root.resolve(),
        [int(x) for x in args.buckets.split(",")],
        args.full_mode,
    )
    print(json.dumps(result, indent=2))
    if result["errors"] and args.full_mode != "global":
        raise SystemExit(1)
