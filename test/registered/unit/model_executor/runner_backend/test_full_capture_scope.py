"""Full capture scope contracts without constructing a CUDA context."""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[5]


class TestFullCaptureScope(unittest.TestCase):
    def backend(self, mode="global", saver=False):
        path = (
            ROOT
            / "python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py"
        )
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "FullCudaGraphBackend"
        )
        cls.bases = []
        calls = []

        @contextmanager
        def graph(**kw):
            calls.append(kw)
            yield

        device = SimpleNamespace(graph=graph, synchronize=lambda: None)
        saver_adapter = (
            SimpleNamespace(enabled=True, cuda_graph=graph) if saver else None
        )
        ns = dict(
            contextmanager=contextmanager,
            partial=__import__("functools").partial,
            torch=SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=lambda: object())),
            get_parallel=lambda: SimpleNamespace(
                tp_group=SimpleNamespace(barrier=lambda: None)
            ),
            GraphPoolPrecarve=lambda: SimpleNamespace(
                measure=nullcontext, mint=lambda: None
            ),
            TorchMemorySaverAdapter=SimpleNamespace(create=lambda **kw: saver_adapter),
            get_bool_env_var=lambda *a: True,
            graph_pool_capture_scope=nullcontext,
            GPU_MEMORY_TYPE_CUDA_GRAPH="graph",
        )
        tree = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                cls,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)
        backend = ns["FullCudaGraphBackend"](
            SimpleNamespace(device_module=device),
            capture_error_mode=mode,
            enable_memory_saver=saver,
        )
        return backend, calls

    def test_native_default_keeps_existing_graph_factory_call(self):
        backend, calls = self.backend()
        backend.capture_one(1, lambda: "output")
        self.assertNotIn("capture_error_mode", calls[0])
        self.assertEqual(backend._outputs[1], "output")

    def test_thread_local_reaches_graph_factory_for_consecutive_buckets(self):
        backend, calls = self.backend("thread_local")
        for size in (16, 8, 4, 2, 1):
            backend.capture_one(size, lambda size=size: size)
        self.assertEqual([x["capture_error_mode"] for x in calls], ["thread_local"] * 5)
        self.assertEqual(backend._outputs[1], 1)

    def test_explicit_scope_also_reaches_memory_saver_factory(self):
        backend, calls = self.backend("thread_local", saver=True)
        backend.capture_one(1, lambda: None)
        self.assertEqual(calls[0]["capture_error_mode"], "thread_local")
        self.assertTrue(calls[0]["enable_cpu_backup"])


if __name__ == "__main__":
    unittest.main()
