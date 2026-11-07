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


class TestFlashSplitAndControl(CustomTestCase):
    def test_native_c4_c128_projection_state_order(self):
        # Load the production split bodies, with small real CPU tensors at the
        # CUDA cache/indexer boundary. C128 must never invent an indexer call.
        tree = ast.parse((ROOT / "python/sglang/srt/models/deepseek_v4.py").read_text())
        owner = next(n for n in tree.body if getattr(n, "name", "") == "MQALayer")
        nodes = [
            n
            for n in owner.body
            if getattr(n, "name", "") in {"project_decode_qkv", "update_decode_state"}
        ]
        namespace = {}
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), "<native-split>", "exec"),
            namespace,
        )
        for ratio in (0, 4, 128):
            for fused in (False, True):
                with self.subTest(ratio=ratio, fused=fused):
                    trace = []
                    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
                    owner = NS(
                        is_dsv41=False,
                        compress_ratio=ratio,
                        use_mega_attention=False,
                        use_fused_qk_norm_rope=False,
                        fuse_wqa_wkv=fused,
                        q_lora_rank=4,
                        layer_id=7,
                    )
                    owner.wqkv_a = lambda v: (torch.cat((v, v * 3), -1), None)
                    owner.wkv = lambda v: (v * 3, None)
                    owner._compute_q_a = lambda v, qkv_a=None: (v + 1, v + 1)
                    owner._compute_q_b = lambda v, pos, out: v * 2
                    q, lora, kv = namespace["project_decode_qkv"](owner, x, None)
                    torch.testing.assert_close(q, (x + 1) * 2)
                    torch.testing.assert_close(kv, x * 3)
                    owner._compute_kv_to_cache = lambda *a, projected_kv: trace.append(
                        ("swa", projected_kv.clone())
                    )
                    owner.indexer = (
                        (lambda **kw: trace.append(("index", kw["q_lora"])))
                        if ratio == 4
                        else None
                    )
                    owner.compressor = object() if ratio else None
                    backend = NS(
                        forward_core_compressor=lambda *a: trace.append(
                            ("compress", a[2])
                        )
                    )
                    fb = NS(forward_mode=NS(is_decode=lambda: True))
                    namespace["update_decode_state"](
                        owner, x, lora, kv, None, fb, backend
                    )
                    self.assertEqual(
                        [a[0] for a in trace],
                        ["swa"]
                        + (["index"] if ratio == 4 else [])
                        + (["compress"] if ratio else []),
                    )
                    torch.testing.assert_close(trace[0][1], x * 3)

    def test_d2d_without_ready_never_enters_nonce_free_legacy_control(self):
        import tempfile
        import time

        tree = ast.parse(
            (ROOT / "plugins/remote_attn/control_plane/p2d_driver.py").read_text()
        )
        node = next(n for n in tree.body if getattr(n, "name", "") == "_send_unlocked")
        with tempfile.TemporaryDirectory() as folder:
            namespace = dict(
                control_dir=lambda role: Path(folder),
                _ready=lambda role: [],
                _BARE_LEGACY_WAIT_S=0,
                time=time,
                _send_legacy=lambda *a: "unsafe-legacy",
            )
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), "<driver>", "exec"),
                namespace,
            )
            with self.assertRaisesRegex(RuntimeError, "refusing nonce-free"):
                namespace["_send_unlocked"]("src", {"cmd": "d2d_freeze"}, 1)

    def test_control_directory_overrides_preserve_role_isolation(self):
        tree = ast.parse(
            (ROOT / "plugins/remote_attn/control_plane/p2d_driver.py").read_text()
        )
        node = next(n for n in tree.body if getattr(n, "name", "") == "control_dir")
        import os

        namespace = dict(os=os, Path=Path, CTL=Path("/legacy"))
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "<control-dir>", "exec"),
            namespace,
        )
        with patch.dict(
            os.environ,
            {
                "HANDOFF_SRC_CTL_DIR": "/existing/D0",
                "HANDOFF_DST_CTL_DIR": "/existing/D1",
            },
        ):
            self.assertEqual(namespace["control_dir"]("src"), Path("/existing/D0"))
            self.assertEqual(namespace["control_dir"]("dst"), Path("/existing/D1"))
            with self.assertRaises(ValueError):
                namespace["control_dir"]("other")
        with patch.dict(
            os.environ, {"HANDOFF_SRC_CTL_DIR": "", "HANDOFF_DST_CTL_DIR": ""}
        ):
            self.assertEqual(namespace["control_dir"]("src"), Path("/legacy/src"))


class TestGatewayControlPaths(CustomTestCase):
    @staticmethod
    def functions(relative, names, namespace, owner=None):
        tree = ast.parse((ROOT / relative).read_text())
        body = (
            tree.body
            if owner is None
            else next(n.body for n in tree.body if getattr(n, "name", "") == owner)
        )
        nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise AssertionError("production function missing")
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), str(ROOT / relative), "exec"),
            namespace,
        )
        return NS(**{name: namespace[name] for name in names})

    def driver(self, root):
        import contextlib
        import fcntl
        import json
        import os
        import time
        import uuid

        names = {"control_dir", "_ready", "_role_lock", "_send_unlocked", "send"}
        ns = dict(
            CTL=root / "legacy",
            Path=Path,
            contextlib=contextlib,
            fcntl=fcntl,
            os=os,
            time=time,
            json=json,
            uuid=uuid,
            _BARE_LEGACY_WAIT_S=0,
            _READY_WAIT_S=0,
        )
        driver = self.functions(
            "plugins/remote_attn/control_plane/p2d_driver.py", names, ns
        )
        # Keep CTL on the namespace too: the buggy gateway directly accesses it.
        driver.CTL = ns["CTL"]
        for role in ("src", "dst"):
            directory = driver.control_dir(role)
            directory.mkdir(parents=True)
            (directory / "ready.0.json").write_text(
                json.dumps(dict(pid=os.getpid(), rank=0, tp_size=1))
            )
        return driver

    def test_heartbeat_and_qwen_dsv_commands_share_selected_directories(self):
        import json
        import os
        import tempfile
        import threading
        import time

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(
                os.environ,
                {
                    "HANDOFF_SRC_CTL_DIR": str(root / "existing-D0"),
                    "HANDOFF_DST_CTL_DIR": str(root / "existing-D1"),
                },
            ):
                driver = self.driver(root)
                gateway = NS(_hb_stop=False)
                clock = NS(
                    time=time.time, sleep=lambda _: setattr(gateway, "_hb_stop", True)
                )
                heartbeat = self.functions(
                    "plugins/remote_attn/control_plane/p4_gateway.py",
                    {"_heartbeat_loop"},
                    dict(d=driver, os=os, json=json, time=clock),
                    "Gateway",
                )
                heartbeat._heartbeat_loop(gateway)
                for role in ("src", "dst"):
                    directory = driver.control_dir(role)
                    hb = json.loads((directory / "heartbeat.json").read_text())
                    self.assertEqual(hb["pid"], os.getpid())
                    self.assertGreater(hb["ts"], 0)
                    self.assertFalse((directory / "heartbeat.tmp").exists())
                # Both backends use the real nonce/ready/flock command writer.
                # No worker responds, so a zero timeout deterministically leaves
                # the command on disk and proves the path without a fake ACK.
                for role in ("src", "dst"):
                    with self.assertRaises(TimeoutError):
                        driver.send(
                            role, {"cmd": "cancel_import", "items": []}, timeout=0
                        )
                    command = json.loads(
                        (driver.control_dir(role) / "cmd.0.json").read_text()
                    )
                    self.assertEqual(command["cmd"], "cancel_import")
                    self.assertEqual(len(command["nonce"]), 32)
                native_send = driver.send
                driver.send = lambda role, payload, timeout: native_send(
                    role, payload, timeout=0
                )
                dsv = self.functions(
                    "plugins/remote_attn/control_plane/p4_dsv_gateway.py",
                    {"_call", "rows"},
                    dict(),
                    "DsvGateway",
                )
                coordinator = NS(
                    driver=driver,
                    rows=dsv.rows,
                    role_locks={r: threading.Lock() for r in ("src", "dst")},
                    facts={
                        r: [dict(instance_id=r + "-instance")] for r in ("src", "dst")
                    },
                )
                with self.assertRaises(TimeoutError):
                    dsv._call(coordinator, "dst", "d2d_abort", "migration-1")
                command = json.loads(
                    (driver.control_dir("dst") / "cmd.0.json").read_text()
                )
                self.assertEqual(command["cmd"], "d2d_abort")
                self.assertEqual(command["migration_epoch"], "migration-1")
                self.assertEqual(command["expected_instances"], ["dst-instance"])
                self.assertTrue(coordinator.role_locks["dst"].acquire(blocking=False))
                coordinator.role_locks["dst"].release()
                self.assertFalse(
                    driver.CTL.exists(),
                    "legacy path must not receive heartbeat or commands",
                )

    def test_cancel_timeout_fences_already_waiting_native_import(self):
        import concurrent.futures
        import json
        import os
        import tempfile
        import threading
        import time

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(
                os.environ,
                {
                    "HANDOFF_SRC_CTL_DIR": str(root / "existing-D0"),
                    "HANDOFF_DST_CTL_DIR": str(root / "existing-D1"),
                    "HANDOFF_TEST_IMPORT_DELAY_MS": "1",
                },
            ):
                driver = self.driver(root)
                directory = driver.control_dir("dst")
                entered, proceed = threading.Event(), threading.Event()

                def delayed_import(_):
                    entered.set()
                    if not proceed.wait(5):
                        raise TimeoutError("test failed to release native import")

                clock = NS(
                    sleep=delayed_import,
                    monotonic=time.monotonic,
                    time=time.time,
                    perf_counter=time.perf_counter,
                )
                ns = dict(
                    os=os,
                    time=clock,
                    Path=Path,
                    _CTL_DIR=str(directory),
                    _CANCELLED_IMPORTS={},
                    _CANCEL_TTL_S=120,
                    _tracked=lambda scheduler: {},
                    _tp_rank=lambda scheduler: 0,
                    _tp_size=lambda scheduler: 1,
                )
                native = self.functions(
                    "plugins/remote_attn/sglang_remote_attn/handoff_control.py",
                    {"_handle", "_import_cancelled"},
                    ns,
                )
                attempted = []

                def transfer(*args):
                    attempted.append("transfer")
                    raise AssertionError("cancelled import reached allocation/transfer")

                ns["_select_transfer"] = transfer
                timeouts = []

                def cancel_send(payload, timeout):
                    # The marker must be visible BEFORE submitting the RPC.
                    timeouts.append(
                        (directory / "cancel.dst-request.epoch-1.json").exists()
                    )
                    return driver.send("dst", payload, timeout=0)

                cancel = self.functions(
                    "plugins/remote_attn/control_plane/p4_gateway.py",
                    {"_cancel_epochs"},
                    dict(d=driver, os=os, json=json, time=time),
                    "Gateway",
                )
                from types import ModuleType

                runtime = ModuleType("sglang.srt.runtime_context")
                runtime.get_schedule = Mock(
                    side_effect=AssertionError("cancelled import read scheduler config")
                )
                with (
                    patch.dict(sys.modules, {runtime.__name__: runtime}),
                    concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool,
                ):
                    pending = pool.submit(
                        native._handle,
                        NS(),
                        dict(cmd="import", rid="dst-request", epoch="epoch-1"),
                    )
                    try:
                        if not entered.wait(5):
                            pending.result(timeout=0)
                            self.fail("native import did not reach its delay")
                        cancel._cancel_epochs(
                            NS(dst=NS(send=cancel_send)),
                            [NS(drid="dst-request", freeze_epoch="epoch-1")],
                        )
                    finally:
                        proceed.set()
                    with self.assertRaisesRegex(RuntimeError, "import epoch cancelled"):
                        pending.result(timeout=5)
                self.assertEqual(timeouts, [True])
                command = json.loads((directory / "cmd.0.json").read_text())
                self.assertEqual(command["cmd"], "cancel_import")
                self.assertEqual(
                    command["items"], [dict(rid="dst-request", epoch="epoch-1")]
                )
                self.assertEqual(len(command["nonce"]), 32)
                self.assertEqual(attempted, [])
                self.assertTrue(native._import_cancelled("dst-request", "epoch-1"))
                self.assertFalse(native._import_cancelled("dst-request", "epoch-2"))
                self.assertFalse(driver.CTL.exists())


if __name__ == "__main__":
    unittest.main()
