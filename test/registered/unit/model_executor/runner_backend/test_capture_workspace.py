"""CPU contracts for graph-private Marlin scratch allocation and lock reset."""

import ast
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner_backend_utils.capture_workspace import (
    CaptureWorkspacePool,
    get_capture_workspace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCaptureWorkspace(unittest.TestCase):
    def marlin(self):
        # Run the real allocator body without importing optional CUDA kernels.
        path = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/srt/layers/quantization/marlin_utils.py"
        )
        tree = ast.parse(path.read_text())
        function = next(
            n for n in tree.body if getattr(n, "name", None) == "marlin_make_workspace"
        )
        namespace = {"torch": torch, "get_capture_workspace": get_capture_workspace}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        return namespace["marlin_make_workspace"]

    def test_marlin_warmup_reserves_all_calls_without_capture_allocation(self):
        make_workspace = self.marlin()
        capturing = False
        allocations = []
        zeros = torch.zeros

        def properties(device):
            self.assertFalse(capturing, "device query during graph capture")
            return SimpleNamespace(multi_processor_count=132)

        def allocate(*a, **kw):
            self.assertFalse(capturing, "allocation during graph capture")
            kw["device"] = "cpu"
            out = zeros(*a, **kw)
            allocations.append(out)
            return out

        pool = CaptureWorkspacePool()
        with (
            patch.object(torch.cuda, "get_device_properties", side_effect=properties),
            patch.object(torch, "zeros", side_effect=allocate),
        ):
            for _ in range(2):
                with pool.scope():
                    first = make_workspace(torch.device("cuda:0"), 4)
                    second = make_workspace(torch.device("cuda:0"), 4)
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertEqual(first.numel(), 132 * 4)
                    first.fill_(7)
                    second.fill_(9)
            capturing = True
            with pool.scope(capturing=True):
                actual = make_workspace(torch.device("cuda:0"), 4)
                self.assertEqual(actual.data_ptr(), first.data_ptr())
                self.assertEqual(torch.count_nonzero(actual).item(), 0)
                actual = make_workspace(torch.device("cuda:0"), 4)
                self.assertEqual(actual.data_ptr(), second.data_ptr())
                self.assertEqual(torch.count_nonzero(actual).item(), 0)
            self.assertEqual(len(allocations), 2)
            capturing = False
            # Another graph/bank must not reuse these reduction locks.
            other = CaptureWorkspacePool()
            with other.scope():
                independent = make_workspace(torch.device("cuda:0"), 4)
            self.assertNotIn(
                independent.data_ptr(), [t.data_ptr() for t in allocations[:2]]
            )
            # Non-capture callers retain the original per-call allocation.
            eager_a = make_workspace(torch.device("cuda:0"), 4)
            eager_b = make_workspace(torch.device("cuda:0"), 4)
            self.assertNotEqual(eager_a.data_ptr(), eager_b.data_ptr())

    def test_unwarmed_or_changed_trace_cannot_allocate_inside_capture(self):
        allocated = []

        def allocate():
            allocated.append(True)
            return torch.zeros(4)

        empty = CaptureWorkspacePool()
        with self.assertRaisesRegex(RuntimeError, "not allocated"):
            with empty.scope(capturing=True):
                get_capture_workspace("new", allocate)
        self.assertEqual(allocated, [])
        pool = CaptureWorkspacePool()
        with pool.scope():
            get_capture_workspace("a", allocate)
        for key, error in (("b", "request differs"), (None, "call count differs")):
            with self.assertRaisesRegex(RuntimeError, error):
                with pool.scope(capturing=True):
                    if key:
                        get_capture_workspace(key, allocate)
        self.assertEqual(allocated, [True])
        with self.assertRaisesRegex(RuntimeError, "not allocated"):
            with pool.scope(capturing=True):
                get_capture_workspace("a", allocate)
                get_capture_workspace("a", allocate)
        self.assertEqual(allocated, [True])
        # Exception cleanup restores the caller's ordinary allocation path.
        get_capture_workspace("outside", allocate)
        self.assertEqual(len(allocated), 2)

    def test_workspace_scope_is_thread_local(self):
        pool = CaptureWorkspacePool()
        values = []
        with pool.scope():
            current = get_capture_workspace("same", lambda: torch.zeros(4))
            thread = threading.Thread(
                target=lambda: values.append(
                    get_capture_workspace("same", lambda: torch.zeros(4))
                )
            )
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertNotEqual(current.data_ptr(), values[0].data_ptr())
        self.assertEqual(len(pool.entries), 1)


if __name__ == "__main__":
    unittest.main()
