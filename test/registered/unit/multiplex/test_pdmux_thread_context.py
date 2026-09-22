"""Thread-local control state required by PDMux host-side submission."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from sglang.srt.distributed import parallel_state
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    get_forward_context,
    has_forward_context,
    forward_context,
)
from sglang.srt.multiplex import pdmux_context
from sglang.test.ci.ci_register import register_cpu_ci

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
