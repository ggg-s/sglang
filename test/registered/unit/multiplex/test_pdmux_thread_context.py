"""Thread-local control state required by PDMux host-side submission."""

from __future__ import annotations

import ast
import importlib.util
import sys
import threading
import types
import unittest
from contextvars import ContextVar
from unittest.mock import patch

from test_pdmux_sxf_port import SRT, register_cpu_ci


def _control_module(path, names, **state):
    module = types.ModuleType("_pdmux_control_test")
    module.__dict__.update(state)
    tree = ast.parse((SRT / path).read_text())
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in definitions} == set(names)
    tree.body = definitions
    exec(compile(tree, str(SRT / path), "exec"), module.__dict__)
    return module


parallel_state = _control_module(
    "distributed/parallel_state.py",
    {"set_pdmux_status", "is_pdmux_prefill_enabled"},
    _ENABLE_PDMUX_P_TP=ContextVar("test_prefill_lane", default=False),
)
pdmux_context = _control_module(
    "multiplex/pdmux_context.py",
    {"set_current_stream_idx", "get_current_stream_idx"},
    STREAM_GROUPS=[],
    CURRENT_STREAM_IDX=ContextVar("test_stream_idx", default=0),
    CURRENT_STREAM_GROUP=None,
)
_spec = importlib.util.spec_from_file_location(
    "_pdmux_forward_context_test", SRT / "model_executor/forward_context.py"
)
_forward = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _forward
_spec.loader.exec_module(_forward)
ForwardContext = _forward.ForwardContext
get_forward_context = _forward.get_forward_context
has_forward_context = _forward.has_forward_context
forward_context = _forward.forward_context

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPDMuxThreadContexts(unittest.TestCase):
    def test_prefill_status_does_not_leak_to_decode_submitter(self):
        parallel_state.set_pdmux_status(False)
        child_value = []

        def prefill_submitter():
            parallel_state.set_pdmux_status(True)
            child_value.append(parallel_state.is_pdmux_prefill_enabled())

        thread = threading.Thread(target=prefill_submitter)
        thread.start()
        thread.join()

        self.assertEqual(child_value, [True])
        self.assertFalse(parallel_state.is_pdmux_prefill_enabled())

    def test_stream_index_does_not_leak_to_decode_submitter(self):
        child_value = []
        with patch.object(pdmux_context, "STREAM_GROUPS", [object(), object()]):
            pdmux_context.set_current_stream_idx(0)

            def prefill_submitter():
                pdmux_context.set_current_stream_idx(1)
                child_value.append(pdmux_context.get_current_stream_idx())

            thread = threading.Thread(target=prefill_submitter)
            thread.start()
            thread.join()

            self.assertEqual(child_value, [1])
            self.assertEqual(pdmux_context.get_current_stream_idx(), 0)

    def test_forward_context_does_not_leak_between_submitters(self):
        decode_backend = object()
        prefill_backend = object()
        child_backend = []

        with forward_context(ForwardContext(attn_backend=decode_backend)):

            def prefill_submitter():
                self.assertFalse(has_forward_context())
                with forward_context(ForwardContext(attn_backend=prefill_backend)):
                    child_backend.append(get_forward_context().attn_backend)

            thread = threading.Thread(target=prefill_submitter)
            thread.start()
            thread.join()

            self.assertEqual(get_forward_context().attn_backend, decode_backend)

        self.assertEqual(child_backend, [prefill_backend])


if __name__ == "__main__":
    unittest.main()
