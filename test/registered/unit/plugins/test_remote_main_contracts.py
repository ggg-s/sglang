"""Contracts at the native-main/plugin boundary after scheduler and graph refactors."""

import ast
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
with patch.object(sys, "path", [str(ROOT / "plugins/remote_attn"), *sys.path]):
    from sglang_remote_attn import destination_cache, hooks, prefix_stream
    from sglang_remote_attn.dsv41_graph import allocate_output_layout, output_layout
    from sglang_remote_attn.interfaces import SourceAPI, check_source_interfaces
    from sglang_remote_attn.segmented_graph import RemoteGraphBackend


class TestRemoteMainContracts(CustomTestCase):
    def test_every_native_decode_backend_access_is_implemented(self):
        source = (
            ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
        )
        tree = ast.parse(source.read_text())
        required = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "backend"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
        }
        self.assertTrue(required)
        self.assertEqual(required - set(dir(RemoteGraphBackend)), set())

    def test_capture_capabilities_and_selected_replay_are_forwarded(self):
        backend = RemoteGraphBackend.__new__(RemoteGraphBackend)
        full, remote = Mock(), Mock()
        backend.full, backend.remote = full, remote
        full.cache_loc_dtype = torch.int64
        self.assertEqual(backend.cache_loc_dtype, torch.int64)
        # Eager breaks still read live metadata after a native full-bank event.
        # A single full-bank event must never license early storage reuse.
        self.assertFalse(backend.supports_shared_read_events)
        model, options, context, batch = (object() for _ in range(4))
        self.assertIs(
            backend.prepare_capture_forward(model, False, num_tokens=2),
            full.prepare_capture_forward.return_value,
        )
        full.prepare_capture_forward.assert_called_once_with(model, False, num_tokens=2)
        self.assertIs(
            backend.capture_profile_context(options),
            full.capture_profile_context.return_value,
        )
        backend.finish_capture_profile(options, context)
        full.finish_capture_profile.assert_called_once_with(options, context)
        for selected, use_remote in ((full, False), (remote, True)):
            backend.use_remote = use_remote
            self.assertIs(
                backend.prepare_replay(batch), selected.prepare_replay.return_value
            )
            selected.prepare_replay.assert_called_once_with(batch)
        self.assertFalse(backend.supports_shared_read_events)

    def test_dsv_output_layout_owns_no_eager_tensor_storage(self):
        values = (torch.ones(2, 4), [None, torch.zeros(2, 1, dtype=torch.int64)])
        layout = output_layout(values)
        out = allocate_output_layout(layout)
        self.assertEqual(type(out), tuple)
        self.assertEqual(type(out[1]), list)
        self.assertIsNone(out[1][0])
        for original, copied in ((values[0], out[0]), (values[1][1], out[1][1])):
            self.assertEqual(original.shape, copied.shape)
            self.assertEqual(original.dtype, copied.dtype)
            self.assertNotEqual(original.data_ptr(), copied.data_ptr())

    def test_all_installed_hook_signatures_bind_native_explicit_parameters(self):
        api = SourceAPI(ROOT)
        for target, hook in hooks.HOOKS.items():
            with self.subTest(target=target):
                node = api.resolve(target)
                args = [None] * len(node.args.posonlyargs + node.args.args)
                kwargs = {arg.arg: None for arg in node.args.kwonlyargs}
                inspect.signature(hook).bind(None, *args, **kwargs)
        self.assertGreater(len(check_source_interfaces(ROOT)), 100)

    def test_cached_prefix_honors_native_loadback_send_guard(self):
        req = NS(pending_bootstrap=False)
        controller = NS(config=NS(mode="remote"), cache_guard=object())
        scheduler = NS(_pre_layerwise=controller)
        original = Mock(return_value="guarded")
        with patch.object(prefix_stream, "enabled", return_value=True):
            self.assertEqual(
                prefix_stream.cached_prefix(original, scheduler, req, False), "guarded"
            )
        original.assert_called_once_with(scheduler, req, allow_send=False)
        self.assertFalse(hasattr(req, "_remote_attn_prefix_stream_seen"))

    def test_destination_metadata_preserves_preassigned_and_async_flags(self):
        req = NS()
        queue, indices = object(), object()
        decode_req = NS(req=req)
        original = Mock(return_value="submitted")
        self.assertEqual(
            destination_cache.send_metadata(
                original,
                queue,
                decode_req,
                indices,
                16,
                None,
                True,
                True,
                decode_prefix_len=32,
            ),
            "submitted",
        )
        original.assert_called_once_with(
            queue,
            decode_req,
            indices,
            16,
            None,
            metadata_buffer_preassigned=True,
            non_blocking=True,
            decode_prefix_len=32,
        )


if __name__ == "__main__":
    unittest.main()
