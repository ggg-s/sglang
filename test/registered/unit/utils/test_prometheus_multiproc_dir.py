"""Verify metrics directory lifetime without importing the GPU runtime."""

import ast
import logging
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/utils/common.py"
)


def load_setter():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "set_prometheus_multiproc_dir"
    )
    namespace = {
        "os": os,
        "tempfile": tempfile,
        "logger": logging.getLogger(__name__),
        "_prometheus_multiproc_dirs": [],
        "_prometheus_multiproc_dir_lock": threading.Lock(),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["set_prometheus_multiproc_dir"], namespace


class TestPrometheusMultiprocDir(unittest.TestCase):
    def test_repeated_call_keeps_active_directory(self):
        set_dir, namespace = load_setter()
        with patch.dict(os.environ):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            try:
                set_dir()
                first = os.environ["PROMETHEUS_MULTIPROC_DIR"]
                set_dir()
                self.assertEqual(os.environ["PROMETHEUS_MULTIPROC_DIR"], first)
                self.assertTrue(Path(first).is_dir())
                self.assertEqual(len(namespace["_prometheus_multiproc_dirs"]), 1)

                # An explicit reset starts a fresh Engine without deleting a
                # directory that older workers may still be using.
                os.environ.pop("PROMETHEUS_MULTIPROC_DIR")
                set_dir()
                self.assertNotEqual(os.environ["PROMETHEUS_MULTIPROC_DIR"], first)
                self.assertTrue(Path(first).is_dir())
            finally:
                for directory in namespace["_prometheus_multiproc_dirs"]:
                    directory.cleanup()

    def test_caller_owned_directory_is_not_replaced(self):
        set_dir, namespace = load_setter()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": directory}):
                set_dir()
                set_dir()
                self.assertEqual(os.environ["PROMETHEUS_MULTIPROC_DIR"], directory)
                self.assertTrue(Path(directory).is_dir())
                self.assertEqual(namespace["_prometheus_multiproc_dirs"], [])

    def test_missing_caller_directory_fails_early(self):
        set_dir, _ = load_setter()
        with tempfile.TemporaryDirectory() as parent:
            missing = str(Path(parent) / "missing")
            with patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": missing}):
                with self.assertRaises(FileNotFoundError):
                    set_dir()


if __name__ == "__main__":
    unittest.main()
