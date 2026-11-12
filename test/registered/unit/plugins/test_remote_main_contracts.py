"""Contracts at the native-main/plugin boundary after scheduler and graph refactors."""

import ast
import inspect
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
with patch.object(sys, "path", [str(ROOT / "plugins/remote_attn"), *sys.path]):
    from sglang_remote_attn import destination_cache, hooks, prefix_stream
    from sglang_remote_attn.dsv41_graph import allocate_output_layout, output_layout
    from sglang_remote_attn.interfaces import SourceAPI, check_source_interfaces
    from sglang_remote_attn.segmented_graph import RemoteGraphBackend


class TestRemoteMainContracts(CustomTestCase):
    def test_split_suffix_uses_native_breakable_prefill_eligibility(self):
        from sglang.srt.model_executor import cuda_graph_config
        from sglang.srt.runtime_context import get_forward

        def launch(original, scheduler, batch):
            return original(scheduler, batch)

        controller = NS(decode_overlap=NS(run_batch=launch))
        scheduler = NS(_pre_layerwise=controller)
        batch = NS(_hybrid_split_extend=True, forward_mode=NS(is_extend=lambda: True))
        for backend, expected in (
            ("disabled", False),
            ("full", False),
            ("breakable", True),
        ):
            with self.subTest(backend=backend):
                config = NS(prefill=NS(backend=backend))
                with (
                    patch.object(
                        cuda_graph_config,
                        "get_exec",
                        return_value=NS(graph=NS(cuda_graph_config=config)),
                    ),
                    patch.object(hooks, "_graph_decision", return_value=True),
                ):
                    self.assertEqual(
                        hooks.pre_layerwise.run_batch(
                            lambda *_: hooks._prefill_graph(None, None, NS()),
                            scheduler,
                            batch,
                        ),
                        expected,
                    )
                self.assertFalse(get_forward().force_eager_prefill)

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
        queue, indices = NS(scheduler=NS()), object()
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

    def test_idle_file_controls_wake_native_poller_without_inference_traffic(self):
        """An idle worker must answer file controls without waiting for real requests."""
        import json
        import os
        import tempfile
        import threading
        import time
        from sglang.srt.environ import envs
        from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper
        from sglang.srt.runtime_context import get_parallel
        from sglang_remote_attn import control_wakeup, handoff_control as runtime

        entered, stopped = threading.Event(), threading.Event()
        failures, executed = [], []

        class TracedPoller:
            def __init__(self, raw):
                self.raw = raw

            def __getattr__(self, name):
                return getattr(self.raw, name)

            def poll(self, timeout):
                entered.set()
                return self.raw.poll(timeout)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(
                os.environ, {"HANDOFF_SRC_CTL_DIR": "", "HANDOFF_DST_CTL_DIR": ""}
            ),
            patch.object(runtime, "_CTL_DIR", str(Path(folder) / "legacy/src")),
            envs.SGLANG_EMPTY_CACHE_INTERVAL.override(0),
        ):
            driver = self.driver(Path(folder))
            scheduler = NS(
                idle_sleeper=IdleSleeper([]),
                running_batch=None,
                last_batch=None,
                waiting_queue=[],
            )
            scheduler.idle_sleeper.poller = TracedPoller(scheduler.idle_sleeper.poller)

            def handle(s, command):
                executed.append(threading.get_ident())
                return dict(protocol="dsv-d2d-v2", command=command["cmd"])

            def loop():
                try:
                    with get_parallel().override(tp_rank=0, tp_size=1):
                        while not stopped.is_set():
                            hooks.HOOKS[
                                "sglang.srt.managers.scheduler.Scheduler.maybe_sleep_on_idle"
                            ](lambda s: s.idle_sleeper.maybe_sleep(), scheduler)
                            if stopped.is_set():
                                break
                            item = runtime.poll_broadcast_control()
                            if item:
                                runtime.execute_handoff_control(scheduler, item["data"])
                except BaseException as exc:
                    failures.append(exc)

            with patch.object(runtime, "_handle", side_effect=handle):
                worker = threading.Thread(target=loop, name="test-native-scheduler")
                worker.start()
                latencies = []
                try:
                    for _ in range(8):
                        self.assertTrue(entered.wait(2))
                        entered.clear()
                        started = time.monotonic()
                        result = driver.send(
                            "src", dict(cmd="d2d_worker_status"), timeout=2
                        )
                        latencies.append((time.monotonic() - started) * 1000)
                        self.assertEqual(result["command"], "d2d_worker_status")
                    self.assertLess(max(latencies), 500, latencies)
                    self.assertEqual(set(executed), {worker.ident})
                    self.assertFalse(failures)
                    print("[control-wakeup-ms] " + json.dumps(latencies))
                finally:
                    stopped.set()
                    (Path(folder) / "legacy/src/cmd.0.json").write_text("{}")
                    worker.join(2)
                    control_wakeup.close(scheduler)
                self.assertFalse(worker.is_alive())
            # Rust's request ring has no zmq FD; retain it with a bounded wait.
            rust = NS(timeout_ms=1000)
            scheduler.idle_sleeper = rust
            calls = []
            control_wakeup.sleep(
                lambda s: calls.append(s.idle_sleeper.timeout_ms), scheduler
            )
            self.assertEqual(calls, [10])

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
                ns = dict(time=time)
                self.functions(
                    "plugins/remote_attn/control_plane/p4_dsv_gateway.py",
                    {"latency_histogram"},
                    ns,
                )
                dsv = self.functions(
                    "plugins/remote_attn/control_plane/p4_dsv_gateway.py",
                    {"_call", "rows"},
                    ns,
                    "DsvGateway",
                )
                coordinator = NS(
                    driver=driver,
                    rows=dsv.rows,
                    lock=threading.RLock(),
                    command_waiters=dict(src=0, dst=0),
                    control_latencies={},
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


class TestQwenHybridD2D(CustomTestCase):
    """GDN state and pending-token ownership must survive partial KV migration."""

    def environment(self):
        with patch.object(
            sys, "path", [str(ROOT / "test/manual/remote_attn"), *sys.path]
        ):
            from qwen_hybrid_probe_support import HybridEnvironment
        env = HybridEnvironment(ROOT, device=getattr(self, "device", "cpu"))
        self.addCleanup(env.close)
        return env

    def test_native_qwen_config_starts_prefill_and_decode_controllers(self):
        """Native config omits layer_types; startup must follow the model's map."""
        f = self.environment()
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                controller = f.startup(role)
                config = (
                    controller.scheduler.tp_worker.model_runner.model_config.hf_config
                )
                self.assertEqual(type(config).__name__, "Qwen3_5Config")
                self.assertEqual(type(config.text_config).__name__, "Qwen3_5TextConfig")
                self.assertFalse(hasattr(config.text_config, "layer_types"))
                expected = {
                    "full_attention_layer_ids": [3, 7],
                    "linear_layer_ids": [0, 1, 2, 4, 5, 6],
                }
                self.assertEqual(controller.schema_signature["layer_mapping"], expected)
                self.assertNotIn("layer_types", controller.schema_signature)
                self.assertEqual(controller.layer_entries, {3: (0, 2), 7: (1, 3)})
                self.assertEqual(
                    [m(None, None, None, NS()) for m in controller.attention_modules],
                    [3, 7],
                )
                peer = controller._schema(
                    controller.scheduler.disagg_prefill_bootstrap_queue.kv_manager
                )
                self.assertEqual(peer, controller.schema_signature)
                self.assertEqual(sorted({e[2] for e in peer["entries"]}), [3, 7])
                self.assertEqual(
                    {e[1] for e in peer["states"]}, set(expected["linear_layer_ids"])
                )

    def test_hybrid_pd_requires_gdn_barrier_and_retires_remote_before_free(self):
        """PD must use P attention while K/V is incomplete, with GDN already local."""
        from sglang_remote_attn import pre_layerwise
        from sglang_remote_attn import qwen_hybrid as hybrid
        from sglang_remote_attn import qwen_hybrid_pd as pd
        from sglang_remote_attn.dsv_d2d_partial import can_freeze

        with patch.object(
            sys, "path", [str(ROOT / "test/manual/remote_attn"), *sys.path]
        ):
            from qwen_hybrid_pd_probe_support import Environment
        with self.subTest(cache_hit="cold-or-threshold"):
            e = Environment(ROOT, device=getattr(self, "device", "cpu"))
            self.addCleanup(e.close)
            for cached in (0, e.prefix // 2):
                e.req.__dict__.pop("_hybrid_prefill_cached_tokens", None)
                scheduler = NS(
                    _pre_layerwise=e.source, disaggregation_mode=NS(value="prefill")
                )
                schedule = lambda *_: NS(batch_to_run=NS(reqs=[e.req]))
                e.req.kv.cache_protected_len = cached
                e.req.extend_batch_idx = 1
                e.req.retracted_stain = False
                pd.snapshot_prefill_admission(schedule, scheduler, None)
                # The request's own chunks can make the live counters look hot.
                e.req.cached_tokens_device = e.req.prefix_len = e.prefix - 1
                e.req.kv.cache_protected_len = e.prefix - 1
                e.req.extend_batch_idx = 15
                pd.snapshot_prefill_admission(schedule, scheduler, None)
                pd.prepare(e.source, [e.req])
                self.assertEqual(e.req._hybrid_prefill_cached_tokens, cached)
                self.assertIsNone(getattr(e.req, "_pre_layerwise_source", None))
                self.assertFalse(getattr(e.pm, "_prefix_streams", {}))
                self.assertFalse(e.source.sources)
                self.assertEqual(pd.early_targets(e.target, e.queue, None)[0], [])
            e.req.kv.cache_protected_len = 0

            def native_pop(queue, rids):
                self.assertEqual(queue.queue, [e.dr])
                if e.sender_status != e.poll.Success:
                    return []
                queue.queue.clear()
                return [e.dr_req]

            self.assertEqual(pre_layerwise.pop_transferred(native_pop, e.queue), [])
            e.sender_status = e.poll.Success
            self.assertEqual(
                pre_layerwise.pop_transferred(native_pop, e.queue), [e.dr_req]
            )
            self.assertFalse(e.target.remote_targets)

        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                e = Environment(ROOT, device=getattr(self, "device", "cpu"))
                self.addCleanup(e.close)
                e.req.start_send_idx = 1
                pd.prepare(e.source, [e.req])
                self.assertIsNone(getattr(e.req, "_pre_layerwise_source", None))
                e.req.start_send_idx = 0
                e.allow_state.clear()
                e.start()
                self.assertEqual(pd.early_targets(e.target, e.queue, None)[0], [])
                self.assertFalse(e.dr_req.output_ids)
                self.assertTrue(e.dr_req.kv.mamba_needs_clear)
                e.allow_state.set()
                e.wait_barrier()
                parent = e.admit()
                self.assertEqual(e.state_copies, 1)
                self.assertFalse(e.dr_req.kv.mamba_needs_clear)
                self.assertFalse(parent.rpc.all_local())
                self.assertTrue(can_freeze(e.dr_req))
                for (_, a, ai), (_, b, bi) in zip(
                    hybrid.state_rows(e.source, e.req, 8),
                    hybrid.state_rows(e.target, e.dr_req, 8),
                ):
                    self.assertTrue(torch.equal(a[ai.long()], b[bi.long()]))
                self.assertTrue(e.req.kv.holds_kv and e.req.kv.holds_mamba)
                if cancel:
                    from sglang.srt.mem_cache.common import release_kv_cache

                    e.dr_req.finished = lambda: True
                    release_kv_cache(
                        e.dr_req, e.target.scheduler.tree_cache, checkpoint=False
                    )
                    self.assertTrue(e.dr_req.kv.holds_kv and e.dr_req.kv.holds_mamba)
                    self.assertIsNotNone(parent.release)
                    self.assertFalse(parent.ready_to_finish())
                e.local(parent)
                if cancel:
                    self.assertFalse(e.dr_req.kv.holds_kv or e.dr_req.kv.holds_mamba)
                self.assertTrue(parent.closed)
                self.assertFalse(e.req.kv.holds_kv or e.req.kv.holds_mamba)

    def test_hybrid_prefill_prefix_cache_enabled_decode_stays_disabled(self):
        """A P cache hit must not be rejected by a D-only ownership restriction."""
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        # Startup validates cache type/policy; the native CUDA probe exercises
        # actual checkpoint, match, CoW and native PD send/retirement.
        cache = object.__new__(UnifiedRadixCache)
        cache.disable = False
        cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        cache.cache_controller = cache.linker = None
        cache.enable_session_radix_cache = False
        f = self.environment()
        controller = f.startup("prefill", prefix_cache=cache)
        self.assertIs(controller.scheduler.tree_cache, cache)
        self.assertIsNone(controller.cache_guard)
        self.assertEqual(controller.sources, [])
        with self.assertRaisesRegex(ValueError, "hybrid Decode Remote requires"):
            f.startup("decode", prefix_cache=cache)
        for field, value in (
            ("cache_controller", object()),
            ("linker", object()),
            ("enable_session_radix_cache", True),
            ("tree_components", (ComponentType.FULL,)),
        ):
            with self.subTest(field=field):
                before = getattr(cache, field)
                setattr(cache, field, value)
                with self.assertRaisesRegex(ValueError, r"device-only FULL\+MAMBA"):
                    f.startup("prefill", prefix_cache=cache)
                setattr(cache, field, before)

    def test_native_layer_mapping_mismatch_rejected_at_startup_and_manifest(self):
        from sglang_remote_attn import dsv_d2d as d2d

        f = self.environment()
        config = f.source.scheduler.tp_worker.model_runner.model_config.hf_config
        # Change the native model's attention interval without rebuilding pools.
        # Both validation and the wire schema must reject stale pool ownership.
        config.text_config.full_attention_interval = 2
        for role in ("prefill", "decode"):
            with (
                self.subTest(role=role),
                self.assertRaisesRegex(ValueError, "logical layer ownership"),
            ):
                f.startup(role)
        for controller in f.nodes:
            with self.assertRaisesRegex(ValueError, "logical layer ownership"):
                controller._schema(d2d.manager(controller))

    def test_native_qwen_language_only_pd_gate(self):
        tree = ast.parse((ROOT / "python/sglang/srt/server_args.py").read_text())
        owner = next(n for n in tree.body if getattr(n, "name", None) == "ServerArgs")
        allowed = next(
            n.value
            for n in owner.body
            if isinstance(n, ast.Assign)
            and any(
                getattr(t, "id", None) == "LANGUAGE_MODEL_ONLY_ARCHITECTURES"
                for t in n.targets
            )
        )
        args = NS(
            language_model_only=True,
            encoder_only=False,
            language_only=False,
            enable_prefix_mm_cache=False,
            enable_broadcast_mm_inputs_process=False,
            mm_enable_dp_encoder=False,
            disaggregation_mode="decode",
            LANGUAGE_MODEL_ONLY_ARCHITECTURES=ast.literal_eval(allowed),
        )
        hf = NS(model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"])
        native = TestGatewayControlPaths.functions(
            "python/sglang/srt/arg_groups/model_hook.py",
            {"handle_language_model_only"},
            dict(
                resolving_view=lambda value: value,
                model_config_of=lambda value: NS(hf_config=hf),
            ),
        )
        self.assertIsNone(native.handle_language_model_only(args))
        hf.model_type = "MuseGlimmer"
        with self.assertRaisesRegex(ValueError, "disaggregation-mode"):
            native.handle_language_model_only(args)

    def test_split_prefill_promotes_decode_rows_once(self):
        """Reusing the first Decode batch must not double its request rows."""
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang_remote_attn.qwen_split_prefill import next_batch

        class Batch:
            def __init__(self, reqs, mode):
                self.reqs = reqs
                self.forward_mode = mode
                self.chunked_req = None

            def filter_batch(self):
                pass

            def is_empty(self):
                return not self.reqs

            def merge_batch(self, other):
                self.reqs = self.reqs + other.reqs

        controller = NS(split_prefill=True, split_prefill_outstanding=1)
        req = NS(_hybrid_split_reservation=controller)
        suffix = Batch([req], ForwardMode.EXTEND)
        suffix._hybrid_split_extend = True
        promoted = []
        scheduler = NS(
            _pre_layerwise=controller,
            last_batch=suffix,
            chunked_req=None,
            waiting_queue=[],
            prefill_launched_reqs=NS(leave=lambda _: None),
            running_reqs=NS(join=promoted.extend),
        )

        def decode(_, running_batch):
            running_batch.forward_mode = ForwardMode.DECODE
            return running_batch

        running_batch = Batch([], ForwardMode.DECODE)
        for _ in range(6):
            running_batch = next_batch(decode, scheduler, running_batch)
            scheduler.last_batch = running_batch
        self.assertEqual(running_batch.reqs, [req])
        self.assertEqual(promoted, [req])
        self.assertEqual(controller.split_prefill_outstanding, 0)

    def test_native_text_batch_accepts_none_multimodal_placeholders(self):
        from sglang_remote_attn.qwen_hybrid import Controller

        controller = Controller.__new__(Controller)
        controller.scheduler = NS(disaggregation_mode=NS(value="decode"))
        sentinel = object()
        forward = lambda runner, batch: sentinel
        self.assertIs(
            controller.forward(forward, None, NS(mm_inputs=[None, None])), sentinel
        )
        with self.assertRaisesRegex(ValueError, "pure-text"):
            controller.forward(forward, None, NS(mm_inputs=[None, object()]))
        with self.assertRaisesRegex(ValueError, "pure-text"):
            controller.forward(
                forward, None, NS(mm_inputs=[None], input_embeds=torch.ones(1))
            )

    def test_pre_sample_reservation_keeps_native_gdn_capacity_until_cancel(self):
        from sglang_remote_attn import dsv_d2d as d2d

        f = self.environment()
        del f.req.output_ids[:]
        f.req.kv.kv_committed_len = f.req.kv.kv_allocated_len = 0
        native = f.native[1]
        before = (
            native.allocator.available_size(),
            native.allocator.state.slots.available_size(),
        )
        common = dict(
            migration_epoch="pre-gdn",
            destination="D1",
            target_rid="target",
            rid=f.req.rid,
        )
        plan = d2d.handle(f.source, dict(cmd="d2d_plan", growth_tokens=2, **common))[
            "plan"
        ]
        reserved = d2d.handle(f.target, dict(cmd="d2d_reserve", plan=plan, **common))
        ticket = f.target.d2d_reservations[common["migration_epoch"]]
        self.assertEqual(ticket.req.kv.kv_committed_len, plan["upper"])
        self.assertTrue(ticket.req.kv.holds_mamba)
        self.assertEqual(native.allocator.state.slots.available_size(), before[1] - 1)
        reply = d2d.handle(
            f.source,
            dict(cmd="d2d_freeze", reservation=reserved["reservation"], **common),
        )
        self.assertEqual(reply["migration_deferred_reason"], "sampled_output_pending")
        self.assertFalse(f.req.output_ids or f.req._handoff_frozen)
        d2d.handle(f.target, dict(cmd="d2d_reserve_cancel", **common))
        self.assertEqual(
            (
                native.allocator.available_size(),
                native.allocator.state.slots.available_size(),
            ),
            before,
        )
        self.assertFalse(ticket.req.kv.holds_kv or ticket.req.kv.holds_mamba)
        self.assertTrue(f.req.kv.holds_mamba)

    def test_reserve_accounts_for_native_working_and_tracking_slots(self):
        from sglang_remote_attn import dsv_d2d as d2d

        f = self.environment()
        native = f.native[1]
        # Three free state slots: main+two tracking slots fit exactly. A second
        # live reservation must not pass just because KV and request rows fit.
        native.pool.enable_mamba_extra_buffer = True
        native.pool.mamba_ping_pong_track_buffer_size = 2
        native.pool.enable_mamba_extra_buffer_lazy = False
        native.pool.get_mamba_ping_pong_other_idx = lambda index: 1 - index
        native.pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
            (4, 2), dtype=torch.int64, device=native.pool.device
        )
        common = dict(
            migration_epoch="slots", destination="D1", target_rid="target", rid="source"
        )
        plan = d2d.handle(f.source, dict(cmd="d2d_plan", growth_tokens=2, **common))[
            "plan"
        ]
        baseline = native.allocator.available_size()
        reserved = d2d.handle(
            f.target, dict(cmd="d2d_reserve", migration_epoch="slots", plan=plan)
        )
        self.assertTrue(reserved["reserved"])
        self.assertFalse(f.req._handoff_frozen)
        self.assertEqual(native.allocator.state.slots.available_size(), 0)
        req = native.make(
            native.scheduler,
            "second",
            list(range(8)),
            [31, 32, 33],
            f.req.sampling_params,
        )
        with self.assertRaisesRegex(ValueError, "state/request-row budget"):
            d2d.allocation_preflight(f.target, req)
        self.assertEqual(native.allocator.available_size(), baseline - plan["upper"])
        d2d.handle(f.target, dict(cmd="d2d_reserve_cancel", migration_epoch="slots"))
        self.assertEqual(native.allocator.available_size(), baseline)
        self.assertEqual(native.allocator.state.slots.available_size(), 3)
        d2d.handle(f.target, dict(cmd="d2d_reserve_cancel", migration_epoch="slots"))
        self.assertEqual(native.allocator.state.slots.available_size(), 3)

    def test_all_gdn_state_precedes_admission_and_kv_entries_switch_in_pairs(self):
        from sglang_remote_attn import qwen_hybrid as hybrid

        f = self.environment()
        tr = f.reserve_and_freeze(growth=3)
        ticket = f.target.d2d_reservations[f.epoch]
        row, slot = ticket.req.kv.req_pool_idx, ticket.req.kv.mamba_pool_idx.clone()
        self.assertNotEqual(slot.item(), f.req.kv.mamba_pool_idx.item())
        self.assertTrue(ticket.req.kv.mamba_needs_clear)
        f.import_(tr)
        target = f.target.d2d_targets[f.epoch]
        req = target.decode_req.req
        self.assertEqual(req.kv.req_pool_idx, row)
        self.assertEqual(req.kv.mamba_pool_idx.item(), slot.item())
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(list(req.output_ids), list(f.req.output_ids))
        self.assertEqual(
            req.kv.kv_committed_len, len(req.origin_input_ids) + len(req.output_ids) - 1
        )
        for (a, source, si), (b, dest, di) in zip(
            hybrid.state_rows(f.source, f.req, f.prefix),
            hybrid.state_rows(f.target, req, f.prefix),
        ):
            self.assertEqual(a, b)
            self.assertTrue(torch.equal(source[si], dest[di]))
        self.assertTrue(f.blocked.wait(5))
        self.assertFalse(target.copied)
        self.assertFalse(target.rpc.local_entries)
        self.assertTrue(f.command(f.target, "d2d_commit")["admitted"])
        self.assertTrue(f.req.kv.holds_mamba)
        self.assertTrue(f.req.kv.holds_kv)
        f.gate.set()
        f.wait_copy()
        k, v = f.target.layer_entries[3]
        next(r for r in target.resources if r.entry == k).finish()
        self.assertFalse(set((k, v)) <= target.rpc.local_entries)
        next(r for r in target.resources if r.entry == v).finish()
        self.assertTrue(set((k, v)) <= target.rpc.local_entries)
        for i, tensor in enumerate(hybrid.kv_buffers(f.target.pool)):
            source = hybrid.kv_buffers(f.source.pool)[i]
            src_ids = f.source.scheduler.req_to_token_pool.req_to_token[
                f.req.kv.req_pool_idx, : f.prefix
            ].long()
            dst_ids = f.target.scheduler.req_to_token_pool.req_to_token[
                row, : f.prefix
            ].long()
            self.assertTrue(torch.equal(tensor[dst_ids], source[src_ids]))
        self.assertTrue(
            f.req.kv.holds_mamba
        )  # copied alone is not a remote-close proof.
        target.finish()
        source = f.source.d2d_sources[f.epoch]
        self.assertTrue(source.poll())
        source.retire()
        self.assertFalse(f.req.kv.holds_mamba)
        self.assertFalse(f.req.kv.holds_kv)
        self.assertTrue(
            req.kv.holds_mamba
        )  # live destination owns its recurrent state.

    def test_cancel_waits_for_inflight_copy_and_tombstones_late_import(self):
        f = self.environment()
        tr = f.reserve_and_freeze()
        f.import_(tr)
        target = f.target.d2d_targets[f.epoch]
        req = target.decode_req.req
        self.assertTrue(f.blocked.wait(5))
        before = f.native[1].allocator.state.slots.available_size()
        result = f.command(f.target, "d2d_abort")
        self.assertTrue(result["pending_reader"])
        self.assertTrue(req.kv.holds_mamba)
        self.assertEqual(f.native[1].allocator.state.slots.available_size(), before)
        f.gate.set()
        target.worker.join(5)
        self.assertTrue(f.command(f.target, "d2d_abort")["aborted"])
        self.assertTrue(req.kv.holds_mamba)  # still waits for retirement certificate.
        target.finish()
        self.assertFalse(req.kv.holds_mamba)
        self.assertEqual(f.native[1].allocator.state.slots.available_size(), before + 1)
        with self.assertRaisesRegex(ValueError, "cancelled"):
            f.import_(tr)
        self.assertEqual(f.native[1].allocator.state.slots.available_size(), before + 1)
        self.assertTrue(f.command(f.source, "d2d_resume")["resumed"])
        self.assertTrue(f.req.kv.holds_mamba)

    def test_partial_state_exception_never_reuses_slot_before_abort(self):
        from sglang_remote_attn import dsv_d2d as d2d

        f = self.environment()
        tr = f.reserve_and_freeze()
        original = d2d.copy_rows

        def broken(*args):
            original(*args)
            raise RuntimeError("reply lost after first native state write")

        before = f.native[1].allocator.state.slots.available_size()
        with (
            patch.object(d2d, "copy_rows", broken),
            self.assertRaisesRegex(RuntimeError, "reply lost"),
        ):
            f.import_(tr)
        target = f.target.d2d_targets[f.epoch]
        req = target.decode_req.req
        self.assertTrue(req.kv.holds_mamba)
        self.assertTrue(req.kv.mamba_needs_clear)
        self.assertIsNone(target.rpc)
        self.assertFalse(target.admitted)
        with self.assertRaisesRegex(ValueError, "failed/aborted"):
            f.import_(tr)
        self.assertEqual(f.native[1].allocator.state.slots.available_size(), before)
        self.assertTrue(f.command(f.target, "d2d_abort")["aborted"])
        target.finish()
        self.assertFalse(req.kv.holds_mamba)
        self.assertEqual(f.native[1].allocator.state.slots.available_size(), before + 1)
        self.assertTrue(f.command(f.source, "d2d_resume")["resumed"])
        self.assertTrue(f.req.kv.holds_mamba)

    def test_late_native_state_write_keeps_kv_gdn_and_staging_owned(self):
        from sglang_remote_attn import dsv_d2d as d2d

        f = self.environment()
        tr = f.reserve_and_freeze()
        engine = d2d.manager(f.source).engine
        pending = []

        def submit(*args):
            pending.append(args)
            return 1

        # Inject at the native completion boundary: submit once, fail the CQ
        # query, and execute that same write only AFTER cancellation is tried.
        # CPU fixtures use registered host regions; the CUDA variant uses the
        # real Mooncake engine for the delayed write after a failed completion.
        native = getattr(engine, "engine", engine)
        with (
            patch.object(native, "batch_transfer_async_write", submit),
            patch.object(native, "get_batch_transfer_status", return_value=-1),
        ):
            with self.assertRaises(RuntimeError):
                f.import_(tr)
        source = f.source.d2d_sources[f.epoch]
        target = f.target.d2d_targets[f.epoch]
        req = target.decode_req.req
        self.assertEqual(len(pending), 1)
        sender, receiver = source.bulk_sender, target.reader.bulk
        self.assertTrue(sender.unsafe and receiver.unsafe)
        self.assertTrue(f.req.kv.holds_mamba and req.kv.holds_mamba)
        budgets = [
            (n.allocator.available_size(), n.allocator.state.slots.available_size())
            for n in f.native
        ]
        with self.assertRaisesRegex(RuntimeError, "completion unknown|quarantine"):
            f.command(f.target, "d2d_abort")
        with self.assertRaises(ValueError):
            f.command(f.source, "d2d_resume")
        if getattr(self, "device", "cpu") == "cpu":
            self.assertEqual(engine.batch_transfer_sync(*pending[0]), 0)
        else:
            # One actual single-submit late write, with the existing endpoint
            # buffers still registered. This is not a retry of the failed call.
            ticket = engine.engine.batch_transfer_async_write(*pending[0])
            self.assertGreaterEqual(ticket, 0)
            self.assertEqual(engine.engine.get_batch_transfer_status([ticket]), 0)
        self.assertFalse(sender.closed or receiver.closed)
        self.assertEqual(
            [
                (n.allocator.available_size(), n.allocator.state.slots.available_size())
                for n in f.native
            ],
            budgets,
        )
        self.assertTrue(f.req.kv.holds_mamba and req.kv.holds_mamba)
        # A late write/negative CQ is never retroactively a Local certificate.
        self.assertIsNone(target.rpc)
        self.assertFalse(target.admitted)

    def test_unreserved_freeze_and_incomplete_state_manifest_reject_without_allocation(
        self,
    ):
        import copy

        f = self.environment()
        before = f.native[1].allocator.available_size()
        with self.assertRaisesRegex(ValueError, "reservation"):
            f.freeze()
        self.assertFalse(f.req._handoff_frozen)
        tr = f.reserve_and_freeze()
        allocated = f.native[1].allocator.available_size()
        bad = copy.deepcopy(tr)
        bad["states"].pop()
        with self.assertRaisesRegex(ValueError, "GDN state manifest"):
            f.import_(bad)
        self.assertEqual(f.native[1].allocator.available_size(), allocated)
        self.assertFalse(f.target.d2d_reservations[f.epoch].consumed)
        self.assertNotIn(f.epoch, f.target.d2d_targets)
        self.assertLess(allocated, before)


class TestSplitPD(CustomTestCase):
    """The bootstrap bound and matched prefix must constrain one TP-agreed cut."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with patch.object(
            sys, "path", [str(ROOT / "test/manual/remote_attn"), *sys.path]
        ):
            from dsv41_remote_probe_support import definitions

        cls.Kv = definitions(
            ROOT, "python/sglang/srt/managers/schedule_batch.py", owner="ReqKvInfo"
        )

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        from sglang_remote_attn import pd_mesh, pre_layerwise, qwen_split_prefill

        self.split, self.mesh, self.pl = qwen_split_prefill, pd_mesh, pre_layerwise
        self.sender_type = type("MooncakeKVSender", (), {})
        self.receiver_type = type("MooncakeKVReceiver", (), {})
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.mooncake.conn": NS(
                        MooncakeKVSender=self.sender_type,
                        MooncakeKVReceiver=self.receiver_type,
                    ),
                    "sglang.srt.runtime_context": NS(
                        get_disagg=lambda: NS(
                            disaggregation_decode_allocation_policy="early",
                            disaggregation_decode_host_receive_threshold=1,
                            disaggregation_decode_enable_offload_kvcache=False,
                        )
                    ),
                },
            )
        )
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))

    def controller(self, role="prefill", *, hybrid=False, max_prefill_tokens=16384):
        scheduler = NS(
            disaggregation_mode=NS(value=role),
            max_prefill_tokens=max_prefill_tokens,
            max_running_requests=16,
            running_batch=NS(batch_size=lambda: 0),
            req_to_token_pool=NS(req_to_token=torch.zeros((1, 1))),
        )
        controller = NS(
            scheduler=scheduler,
            config=NS(mode="remote"),
            rank=0,
            tp_size=1,
            hybrid_gdn=hybrid,
            agree=lambda flags: flags,
            pd_visibility=NS(receive=lambda: None),
        )
        scheduler._pre_layerwise = controller
        self.split.configure(controller)
        return controller

    def request(self, *, prefix=0, length=32768):
        return NS(
            rid="split",
            bootstrap_room=7,
            origin_input_ids=list(range(length)),
            output_ids=[],
            kv=self.Kv(cache_protected_len=prefix),
            prefix_len=prefix,
            start_send_idx=0,
            is_retracted=False,
            retracted_stain=False,
            return_logprob=False,
            return_hidden_states_mode=NS(need_capture=lambda: False),
            return_sampling_mask=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            grammar=None,
            beam_group=None,
            custom_logit_processor=None,
            session=None,
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            sampling_params=NS(max_new_tokens=16),
        )

    def offer(self, d, *, length=32768):
        dr = NS(
            req=self.request(length=length),
            kv_receiver=self.receiver_type(),
            is_rebootstrap=False,
        )
        self.split.advertise(NS(scheduler=d.scheduler), dr, 0)
        return dr

    def bootstrap(self, *, prefix=0, length=32768, hybrid=False, decode_budget=16384):
        p = self.controller(hybrid=hybrid)
        d = self.controller("decode", hybrid=hybrid, max_prefill_tokens=decode_budget)
        req = self.request(prefix=prefix, length=length)
        dr = self.offer(d, length=length)
        receiver = dr.kv_receiver
        receiver.kv_mgr = NS()
        frames = self.mesh.bootstrap(
            lambda _, **kwargs: kwargs["message"],
            receiver,
            bootstrap_info={"is_dummy": False},
            message=[b""] * 10,
            operation="send_metadata",
        )
        peer = self.mesh.parse_transfer(lambda **_: NS(), msg=frames)
        peer.is_dummy = False
        peer.mooncake_session_id = "decode"
        peer.dst_kv_indices = list(range(length))
        sender = self.sender_type()
        sender.kv_mgr = NS(
            transfer_infos={7: {"decode": peer}},
            decode_kv_args_table={
                "decode": NS(requires_dcp_relayout=False, dst_attn_tp_size=1)
            },
        )
        sender.curr_idx = 0
        req.disagg_kv_sender = sender
        self.split.create_sender(lambda *_: True, NS(scheduler=p.scheduler), req, 8)
        self.split.finalize_bootstrap(lambda *_: True, None, req)
        return p, d, req, dr

    @staticmethod
    def match(req, *_):
        req.full_untruncated_fill_ids = list(req.origin_input_ids)

    def test_default_cut_uses_misses_and_caps_short_or_warm_suffixes(self):
        for hybrid in (False, True):
            for length, prefix, expected in (
                (16384, 0, 13927),
                (16384, 14745, 15360),
                (32768, 0, 22938),
                (32768, 13107, 28672),
                (32768, 26214, 30720),
                (65536, 26214, 61440),
            ):
                with self.subTest(hybrid=hybrid, length=length, prefix=prefix):
                    p, d, req, dr = self.bootstrap(
                        prefix=prefix, length=length, hybrid=hybrid
                    )
                    self.assertTrue(p.split_prefill)
                    self.assertEqual(d.split_prefill_chunk_size, 4096)
                    self.assertFalse(
                        self.split.can_prefill_before_bootstrap(
                            lambda *_: True, None, req
                        )
                    )
                    self.split.init_input(self.match, req, object())
                    self.assertEqual(req._hybrid_split_at, expected)
                    self.assertEqual(len(req.full_untruncated_fill_ids), expected)
                    self.assertGreater(expected, req.kv.cache_protected_len)
                    self.assertGreaterEqual(
                        expected, dr.kv_receiver._hybrid_split_limits[0]
                    )

    def test_prefill_cap_uses_the_decode_budget_from_bootstrap(self):
        _, d, req, _ = self.bootstrap(prefix=13107, decode_budget=1024)
        self.assertEqual(d.split_prefill_chunk_size, 1024)
        self.split.init_input(self.match, req, object())
        self.assertEqual(req._hybrid_split_at, 30720)

    def test_small_suffix_and_tp_cut_disagreement_use_full_native_input(self):
        for prefix, disagreement in ((30720, False), (32000, False), (0, True)):
            with self.subTest(prefix=prefix, disagreement=disagreement):
                p, _, req, _ = self.bootstrap(prefix=prefix)
                if disagreement:
                    p.agree = lambda flags: [flags[0], -flags[0] - 1]
                self.split.init_input(self.match, req, object())
                self.assertEqual(req._hybrid_split_at, 0)
                self.assertEqual(req.full_untruncated_fill_ids, req.origin_input_ids)

    def test_split_opt_out_and_prefill_cut_override(self):
        with patch.dict(os.environ, {"HANDOFF_HYBRID_SPLIT_PREFILL": "0"}):
            self.assertFalse(self.controller().split_prefill)
            self.assertFalse(self.controller("decode").split_prefill)
        with patch.dict(os.environ, {"HANDOFF_HYBRID_SPLIT_PREFILL_CUT_RATIO": "0.75"}):
            _, _, req, _ = self.bootstrap()
            self.split.init_input(self.match, req, object())
            self.assertEqual(req._hybrid_split_at, 24576)
        with patch.dict(os.environ, {"HANDOFF_SPLIT_CUT_BASIS": "whole"}):
            _, _, req, _ = self.bootstrap(prefix=26214)
            self.split.init_input(self.match, req, object())
            self.assertEqual(req._hybrid_split_at, 27238)
        for suffix, expected in ((8192, 8192), (1023, 0)):
            with (
                self.subTest(suffix=suffix),
                patch.dict(os.environ, {"HANDOFF_SPLIT_SUFFIX_TOKENS": str(suffix)}),
            ):
                _, _, req, _ = self.bootstrap(length=16384, prefix=6553)
                self.split.init_input(self.match, req, object())
                self.assertEqual(req._hybrid_split_at, expected)

    def test_probe_and_admission_retry_preserve_native_matching_order(self):
        _, _, req, _ = self.bootstrap()
        cache = object()
        self.split.init_input(self.match, req, cache, cow_mamba=False)
        self.assertEqual(len(req.full_untruncated_fill_ids), 32768)
        self.assertFalse(hasattr(req, "_hybrid_split_at"))
        self.split.init_input(self.match, req, cache)
        self.assertEqual(req._hybrid_split_at, 22938)
        req.prefix_len = req.kv.cache_protected_len = 29491
        self.split.init_input(self.match, req, cache)
        self.assertEqual(req._hybrid_split_at, 30720)
        req.kv.req_pool_idx = 0
        req.prefix_len = req.kv.cache_protected_len = 30000
        self.split.init_input(self.match, req, cache)
        self.assertEqual(req._hybrid_split_at, 30720)

    def test_decode_offers_reserve_capacity_before_transfer(self):
        d = self.controller("decode")
        active = [self.offer(d) for _ in range(4)]
        self.split.advertise(NS(scheduler=d.scheduler), active[0], 0)
        self.assertTrue(all(dr.kv_receiver._hybrid_split_limits for dr in active))
        self.assertEqual(d.split_prefill_outstanding, 4)
        self.assertFalse(hasattr(self.offer(d).kv_receiver, "_hybrid_split_limits"))
        req = active[0].req
        req.kv.req_pool_idx = 0
        req.kv.set_row_len(32768)
        with (
            patch.object(self.pl, "defer_release", return_value=False),
            patch.object(
                hooks, "_runtime", return_value=NS(stop_target_receivers=lambda _: None)
            ),
        ):
            hooks._release_kv(lambda *_, **__: None, req, None, checkpoint=False)
        self.split.release_reservation(req)
        self.assertEqual(d.split_prefill_outstanding, 3)
        self.assertTrue(self.offer(d).kv_receiver._hybrid_split_limits)

    def test_decode_capacity_and_tp_rejection_do_not_create_reservations(self):
        d = self.controller("decode")
        d.scheduler.running_batch.batch_size = lambda: d.scheduler.max_running_requests
        self.assertFalse(hasattr(self.offer(d).kv_receiver, "_hybrid_split_limits"))
        d.scheduler.running_batch.batch_size = lambda: 0
        d.agree = lambda _: [0]
        self.assertFalse(hasattr(self.offer(d).kv_receiver, "_hybrid_split_limits"))
        self.assertEqual(d.split_prefill_outstanding, 0)

    def test_inflight_suffix_chunk_allows_decode_before_result_processing(self):
        from sglang_remote_attn import decode_overlap

        class Waiting(list):
            def prepend(self, reqs):
                self[:0] = reqs

        req = self.request()
        req.inflight_steps = 1
        req.kv.kv_committed_len = 30000
        req.kv.reserved_prefill_suffix = True
        req.finished = lambda: False
        running = NS(is_empty=lambda: False, batch_size=lambda: 4)
        suffix = NS(
            _hybrid_split_extend=True,
            forward_mode=NS(is_extend=lambda: True),
            chunked_req=req,
        )
        abort_checks = []
        scheduler = NS(
            _pre_layerwise=NS(split_prefill=True),
            last_batch=suffix,
            chunked_req=req,
            running_batch=running,
            max_running_requests=4,
            waiting_queue=Waiting(),
            process_pending_chunked_abort=lambda: abort_checks.append(req),
        )
        self.assertFalse(
            decode_overlap.process_last_batch_early(lambda *_: False, scheduler, suffix)
        )
        self.assertTrue(
            decode_overlap.process_last_batch_early(lambda *_: True, scheduler, suffix)
        )
        self.assertIsNone(
            self.split.prebuilt(
                lambda *_: self.fail("prebuilt admission"), scheduler, running
            )
        )

        self.assertIs(
            self.split.next_batch(lambda _, batch: batch, scheduler, running), running
        )
        self.assertIs(scheduler.chunked_req, req)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(abort_checks, [])

        # The overlap loop consumed the suffix result after launching decode.
        scheduler.last_batch = running
        req.inflight_steps = 0
        self.split.next_batch(lambda _, batch: batch, scheduler, running)
        self.assertIsNone(scheduler.chunked_req)
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertEqual(req.prefix_len, 30000)
        self.assertEqual(abort_checks, [req])

        suffix.chunked_req = None
        self.assertTrue(
            decode_overlap.process_last_batch_early(lambda *_: False, scheduler, suffix)
        )

    def test_legacy_and_malformed_offers_leave_native_pd_available(self):
        for extension in (
            b"REMOTE_ATTN_HYBRID_SPLIT_V1:16384",
            b"REMOTE_ATTN_HYBRID_SPLIT_V2:16384",
            self.split.WIRE + b"-1:4096",
            self.split.WIRE + b"16384",
            self.split.WIRE + b"16384:99999999999",
        ):
            with self.subTest(extension=extension):
                peer = self.mesh.parse_transfer(
                    lambda **_: NS(), msg=[b""] * 10 + [extension]
                )
                self.assertFalse(hasattr(peer, "_hybrid_split_limits"))

    def commit_fixture(self, cut, *, hybrid=False, room=7, length=32768):
        _, d, _, dr = self.bootstrap(hybrid=hybrid, length=length)
        dr.metadata_buffer_index = 0
        dr.req.kv.req_pool_idx = 0
        dr.req.kv.set_row_len(length)
        if hybrid:
            dr.req.kv.mamba_pool_idx = torch.tensor([0])
        dr.req.time_stats = NS(set_wait_queue_entry_time=lambda: None)
        buffers = NS(
            output_ids=torch.tensor([[0, cut]], dtype=torch.int64),
            bootstrap_room=torch.tensor([[room]]),
            cached_tokens=torch.zeros((1, 2), dtype=torch.int64),
        )
        dr.kv_receiver.clear = lambda: None
        return d, NS(scheduler=d.scheduler, metadata_buffers=buffers), dr

    def test_dynamic_cut_commit_keeps_full_reservation_and_native_state(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                d, queue, dr = self.commit_fixture(15360, hybrid=hybrid, length=16384)
                self.assertTrue(self.split.commit(queue, dr, keep_receiver=False))
                self.assertEqual(dr.req.kv.kv_committed_len, 15360)
                self.assertEqual(dr.req.kv.kv_allocated_len, 16384)
                self.assertTrue(dr.req.kv.reserved_prefill_suffix)
                self.assertEqual(dr.req.prefix_len, 15360)
                self.assertIsNone(dr.kv_receiver)
                self.assertEqual(d.split_prefill_outstanding, 1)

    def test_invalid_transfer_or_tp_cut_mismatch_aborts_before_continuation(self):
        for length, cut, room, mismatch, missing_state in (
            (16384, 15361, 7, False, False),
            (32768, 16000, 7, False, False),
            (32768, 32000, 7, False, False),
            (32768, 27852, 8, False, False),
            (32768, 27852, 7, True, False),
            (32768, 27852, 7, False, True),
        ):
            with self.subTest(
                length=length,
                cut=cut,
                room=room,
                mismatch=mismatch,
                state=missing_state,
            ):
                d, queue, dr = self.commit_fixture(
                    cut, hybrid=True, room=room, length=length
                )
                if mismatch:
                    d.agree = lambda flags: (
                        [flags[0], flags[1], flags[2], -flags[2] - 1]
                        if len(flags) == 4
                        else flags
                    )
                if missing_state:
                    dr.req.kv.mamba_pool_idx = None
                aborted = []
                with patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.disaggregation.utils": NS(
                            prepare_abort=lambda *args, aborted=aborted, **kwargs: (
                                aborted.append(args[0])
                            )
                        )
                    },
                ):
                    self.assertTrue(self.split.commit(queue, dr, keep_receiver=False))
                self.assertEqual(aborted, [dr.req])
                self.assertFalse(dr.req.kv.reserved_prefill_suffix)
                self.assertEqual(dr.req.kv.kv_committed_len, length)
                self.assertEqual(d.split_prefill_outstanding, 0)

    def test_native_completion_clears_reused_cut_metadata(self):
        d, queue, dr = self.commit_fixture(0)
        receiver = dr.kv_receiver
        self.assertFalse(self.split.commit(queue, dr, keep_receiver=False))
        self.assertIs(dr.kv_receiver, receiver)
        self.assertEqual(d.split_prefill_outstanding, 0)
        req = self.request()
        req.metadata_buffer_index = 0
        buffers = NS(output_ids=torch.tensor([[0, 27852]]), _hybrid_split_used=True)
        self.split.set_metadata(lambda *_: None, buffers, req)
        self.assertEqual(buffers.output_ids[0, self.split.CUT_SLOT].item(), 0)

    def test_pd_remote_requires_opt_in_while_native_pd_remains_available(self):
        runtime = NS(_tp_rank=lambda _: 0, _tp_size=lambda _: 1)
        scheduler = NS(
            tp_worker=NS(model_runner=NS(model_config=NS(num_hidden_layers=2)))
        )
        with patch.object(self.pl, "_runtime", return_value=runtime):
            controller = self.pl.Controller(scheduler, NS(mode="remote"))
            self.assertFalse(controller.pd_remote)
            scheduler._pre_layerwise = controller
            native = object()
            self.assertIs(
                self.pl.pop_transferred(lambda *_: native, NS(scheduler=scheduler)),
                native,
            )
            self.assertIs(
                prefix_stream.cached_prefix(lambda *_, **__: native, scheduler, NS()),
                native,
            )
            controller.prepare = lambda _: self.fail("experimental PD prepare")
            controller.cache_guard = object()
            prefix_sends = []
            scheduler.maybe_send_cached_prefix_chunk = (
                lambda req, allow_send=True: prefix_sends.append((req, allow_send))
            )
            batch = NS(reqs=[NS()])

            def native_run(s, b):
                s.maybe_send_cached_prefix_chunk(b.reqs[0], allow_send=False)
                return native

            self.assertIs(self.pl.run_batch(native_run, scheduler, batch), native)
            self.assertEqual(prefix_sends, [(batch.reqs[0], False)])
            with patch.dict(os.environ, {"HANDOFF_EXPERIMENTAL_PD_REMOTE": "1"}):
                self.assertTrue(
                    self.pl.Controller(scheduler, NS(mode="remote")).pd_remote
                )
            self.assertTrue(
                self.pl.Controller(scheduler, NS(mode="layerwise")).pd_remote
            )


if __name__ == "__main__":
    unittest.main()
