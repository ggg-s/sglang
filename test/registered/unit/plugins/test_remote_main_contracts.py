"""Contracts at the native-main/plugin boundary after scheduler and graph refactors."""

import ast
import inspect
import os
import sys
import unittest
from contextlib import ExitStack
from dataclasses import dataclass
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
    def test_split_graph_startup_serializes_only_budget_configuration(self):
        import pickle
        from dataclasses import replace

        from sglang.srt import runtime_context
        from sglang_remote_attn import pd_mesh, pre_layerwise, source_cache
        from sglang_remote_attn.split_decode_budget import DecodeBudget

        @dataclass
        class GraphConfig:
            pass

        budget = DecodeBudget(
            target=0.75,
            stall_ms=400,
            patience_ms=200,
            pinned_chunk=False,
            max_cap=16384,
        )
        budget.ledger = torch.distributed.PrefixStore(
            "split", torch.distributed.HashStore()
        )
        control = NS(
            scheduler=NS(
                tp_worker=NS(
                    model_runner=NS(
                        decode_cuda_graph_runner=None, prefill_cuda_graph_runner=None
                    )
                )
            ),
            config=NS(),
            pd_remote=False,
            split_vote_size=2,
            split_vote_group=object(),
            split_prefill_decode_budget=budget,
        )
        execution = NS(graph=NS(cuda_graph_config=GraphConfig()))
        reports = []

        def gather(result, report, group):
            result[0] = pickle.loads(pickle.dumps(report))
            reports.append(result[0])
            result[1] = reports[0]

        with (
            patch.object(runtime_context, "get_exec", return_value=execution),
            patch.object(pd_mesh, "startup_signature", return_value=None),
            patch.object(source_cache, "cache_signature", return_value=None),
            patch.object(
                pre_layerwise,
                "_runtime",
                return_value=NS(has_target_states=lambda: False),
            ),
            patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        ):
            pre_layerwise.Controller.validate_graphs(control)
            self.assertTrue(control.graph_fast_path)
            budget.suffix_rate = 6000
            pre_layerwise.Controller.validate_graphs(control)
            self.assertEqual(reports[0], reports[1])
            control.split_prefill_decode_budget = replace(budget, stall_ms=500)
            with self.assertRaisesRegex(RuntimeError, "config or captures differ"):
                pre_layerwise.Controller.validate_graphs(control)

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

    def test_split_dpa_uses_native_graph_admission_without_global_tp_vote(self):
        """A draft graph probe on one DP group must not wait for idle groups."""
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.runtime_context import get_parallel
        from sglang_remote_attn import tp

        draft = NS(model_runner=NS())
        target = NS(
            model_runner=NS(_pre_layerwise=NS(graph_fast_path=False, split_dpa=True))
        )
        verify = NS(
            forward_mode=ForwardMode.TARGET_VERIFY, can_run_decode_cuda_graph=True
        )
        idle = NS(forward_mode=ForwardMode.IDLE, can_run_decode_cuda_graph=True)
        native = Mock(return_value=True)
        with (
            patch.dict(os.environ, {"HANDOFF_SPLIT_DPA": "1"}),
            get_parallel().override(tp_size=4, tp_rank=0, tp_group=NS(cpu_group=None)),
            patch.object(
                hooks, "_runtime", return_value=NS(has_remote_work=lambda: False)
            ),
            patch.object(
                tp, "agree_graph", side_effect=AssertionError("global TP vote")
            ),
        ):
            self.assertTrue(hooks._decode_graph(native, draft, verify))
            self.assertTrue(hooks._decode_graph(native, target, verify))
            self.assertTrue(hooks._decode_graph(native, target, idle))
            self.assertTrue(hooks._prefill_graph(native, target, verify))

            verify.can_run_decode_cuda_graph = False
            self.assertFalse(hooks._decode_graph(native, draft, verify))
            self.assertFalse(hooks._decode_graph(native, target, verify))
            self.assertEqual(native.call_count, 4)

            verify.can_run_decode_cuda_graph = True
            native.return_value = False
            self.assertFalse(hooks._decode_graph(native, target, verify))
            self.assertEqual(
                verify._remote_attn_graph_fallback, "native_graph_ineligible"
            )

    def test_flash_decode_pcp_uses_all_native_receive_ranks(self):
        """A width-one PD view collapses all D destinations into rank zero."""
        from sglang.srt.runtime_context import get_parallel

        with patch.object(
            sys, "path", [str(ROOT / "test/manual/remote_attn"), *sys.path]
        ):
            from dsv41_remote_probe_support import definitions

        manager = definitions(
            ROOT,
            "python/sglang/srt/disaggregation/common/conn.py",
            owner="CommonKVManager",
            members={"_resolve_rank_mapping"},
        )
        parallel = get_parallel()
        singleton = NS(world_size=1, cpu_group=object())
        prefix = "sglang.srt.disaggregation.decode."
        transfer_init = hooks.HOOKS[prefix + "DecodeTransferQueue.__init__"]
        prealloc_init = hooks.HOOKS[prefix + "DecodePreallocQueue.__init__"]

        def init_transfer(queue):
            queue.group = parallel.attn_tp_group.cpu_group

        def init_prealloc(queue):
            queue.group = parallel.attn_tp_group.cpu_group
            queue.args = NS(
                engine_rank=parallel.tp_rank % parallel.attn_tp_size,
            )
            queue.manager = NS(
                attn_cp_size=parallel.attn_cp_size,
                attn_tp_size=parallel.attn_tp_size,
                is_mla_backend=True,
                kv_args=queue.args,
                enable_all_cp_ranks_for_transfer=False,
                pp_size=1,
                pp_rank=0,
            )

        for width in (2, 4, 6, 8):
            with self.subTest(width=width):
                full_group = NS(world_size=width, cpu_group=object())
                registered = []
                with patch.dict(os.environ, {"HANDOFF_SPLIT_DECODE_PCP": "1"}):
                    for rank in range(width):
                        with parallel.override(
                            tp_size=width,
                            tp_rank=rank,
                            tp_group=full_group,
                            attn_tp_size=1,
                            attn_tp_rank=0,
                            attn_tp_group=singleton,
                            attn_cp_size=width,
                            attn_cp_rank=rank,
                            attn_cp_group=full_group,
                            attn_dp_size=1,
                            attn_dp_rank=0,
                        ):
                            transfer, prealloc = NS(), NS()
                            transfer_init(init_transfer, transfer)
                            prealloc_init(init_prealloc, prealloc)
                            self.assertIs(transfer.group, full_group.cpu_group)
                            self.assertIs(prealloc.group, full_group.cpu_group)
                            self.assertEqual(parallel.attn_cp_size, width)
                            self.assertIs(parallel.attn_tp_group, singleton)
                            prefill = NS(
                                attn_tp_size=1,
                                attn_cp_size=width,
                                enable_kv_cache_sharding=False,
                                enable_dsa_cache_layer_split=False,
                                pp_size=1,
                            )
                            manager._resolve_rank_mapping(prealloc.manager, prefill)
                            self.assertEqual(prefill.required_dst_info_num, width)
                            registered.append(prealloc.args.engine_rank)
                self.assertEqual(registered, list(range(width)))

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
        from collections import deque

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
                    {"_call", "_log_trace", "_observe", "rows"},
                    ns,
                    "DsvGateway",
                )
                coordinator = NS(
                    driver=driver,
                    rows=dsv.rows,
                    lock=threading.RLock(),
                    command_waiters=dict(src=0, dst=0),
                    dst_slot_holder=None,
                    dst_slot_history=deque(maxlen=256),
                    precopy_active={},
                    control_latencies={},
                    control_breakdown={},
                    _observe=dsv._observe,
                    out=None,
                    role_locks={r: threading.Lock() for r in ("src", "dst")},
                    facts={
                        r: [dict(instance_id=r + "-instance")] for r in ("src", "dst")
                    },
                )
                coordinator._log_trace = lambda trace: dsv._log_trace(
                    coordinator, trace
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
                            disaggregation_transfer_backend="mooncake",
                        )
                    ),
                },
            )
        )
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))

    def controller(
        self, role="prefill", *, hybrid=False, dsv4=False, max_prefill_tokens=16384
    ):
        scheduler = NS(
            disaggregation_mode=NS(value=role),
            max_req_input_len=131072,
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
            dsv41_layerwise=dsv4,
            dsv4_split_schema="a" * 64 if dsv4 else None,
            pd_remote=False,
            pool=NS(page_size=256),
            agree=lambda flags: flags,
            pd_visibility=NS(receive=lambda: None),
        )
        scheduler._pre_layerwise = controller
        self.split.configure(controller)
        return controller

    def test_plain_flash_split_keeps_native_receive_without_dspark(self):
        from tempfile import TemporaryDirectory

        from sglang_remote_attn import dsv41_layerwise

        for role in ("prefill", "decode"):
            for experimental_remote in (False, True):
                with self.subTest(role=role, remote=experimental_remote):
                    with TemporaryDirectory() as folder:
                        controller = self.controller(role, dsv4=True)
                        controller.split_dpa = controller.split_pcp = False
                        controller.pd_remote = experimental_remote
                        controller.config = NS(
                            mode="remote",
                            host_staging=False,
                            recv_host_staging=False,
                            directory=Path(folder),
                        )
                        controller._schema = lambda _: {"ratios": [0, 4, 128]}
                        controller.scheduler.sliding_window_size = 128
                        controller.scheduler.kv_checksum_computer = None
                        controller.pool.swa_page_size = 256
                        controller.pool.request_window = None
                        state_type = NS(SWA="swa", DSV4_REQUEST_STATE="state")
                        manager = NS(
                            enable_staging=False,
                            kv_shard_size=1,
                            kv_args=NS(state_types=["swa", "state"]),
                        )
                        controller.scheduler.disagg_decode_prealloc_queue = NS(
                            kv_manager=manager
                        )
                        controller.scheduler.disagg_prefill_bootstrap_queue = NS(
                            kv_manager=manager
                        )
                        controller.scheduler.token_to_kv_pool_allocator = NS(
                            get_kvcache=lambda: controller.pool
                        )
                        controller.scheduler.tp_worker = NS(
                            model_runner=NS(
                                lora_manager=None,
                                model_config=NS(hf_config=NS(model_type="deepseek_v4")),
                                model=NS(model=NS(layers=[Mock()])),
                                attn_backend=NS(),
                                prefill_attention_backend_str="dsv4",
                                decode_attention_backend_str="dsv4",
                            )
                        )
                        controller.layers = 1
                        with (
                            patch.dict(
                                os.environ,
                                {"HANDOFF_HYBRID_SPLIT_PREFILL": "1"},
                            ),
                            patch.dict(
                                sys.modules,
                                {
                                    "sglang.srt.runtime_context": NS(
                                        get_disagg=lambda: NS(
                                            language_model_only=True,
                                            disaggregation_decode_allocation_policy="early",
                                            disaggregation_transfer_backend="mooncake",
                                        ),
                                        get_memory=lambda: NS(
                                            disable_radix_cache=True,
                                            enable_unified_memory=False,
                                            enable_hierarchical_cache=False,
                                        ),
                                        get_spec=lambda: NS(speculative_algorithm=None),
                                        get_exec=lambda: NS(
                                            graph=NS(enable_torch_compile=False)
                                        ),
                                    ),
                                    "sglang.srt.disaggregation.base.conn": NS(
                                        StateType=state_type
                                    ),
                                },
                            ),
                            patch.object(
                                dsv41_layerwise.dsv41_remote, "validate"
                            ) as bind,
                            patch.object(
                                dsv41_layerwise.dsv41_receive, "install"
                            ) as receive,
                        ):
                            dsv41_layerwise.DSV41Controller.validate(controller)
                        self.assertEqual(controller.dsv41_remote, experimental_remote)
                        self.assertEqual(
                            bool(controller.dsv4_split_schema), not experimental_remote
                        )
                        self.assertEqual(bind.call_count, int(experimental_remote))
                        self.assertEqual(
                            receive.call_count,
                            int(experimental_remote and role == "decode"),
                        )
                        self.assertEqual(controller._layer_handles, [])

    def test_dsv4_split_negotiates_swa_capacity_and_commits_page_aligned_cut(self):
        self.assertFalse(self.controller(dsv4=True).split_prefill)
        with patch.dict(os.environ, {"HANDOFF_HYBRID_SPLIT_PREFILL": "1"}):
            self._check_dsv4_split(adaptive=False)

    def test_dsv4_split_freezes_page_aligned_adaptive_offer(self):
        with patch.dict(
            os.environ,
            {"HANDOFF_HYBRID_SPLIT_PREFILL": "1", "HANDOFF_SPLIT_ADAPTIVE": "1"},
        ):
            self._check_dsv4_split(adaptive=True)

    def test_dsv4_adaptive_keeps_offer_with_invalid_optional_hints(self):
        """Bad cap/rate hints previously erased a valid paged cut on P."""
        with patch.dict(
            os.environ,
            {"HANDOFF_HYBRID_SPLIT_PREFILL": "1", "HANDOFF_SPLIT_ADAPTIVE": "1"},
        ):
            self._check_dsv4_split(adaptive=True, invalid_hints=True)

    def test_dsv4_dspark_split_remaps_both_swa_components_and_keeps_native_fallback(
        self,
    ):
        with patch.dict(os.environ, {"HANDOFF_HYBRID_SPLIT_PREFILL": "1"}):
            for dpa in (False, True):
                with self.subTest(dpa=dpa):
                    self._check_dsv4_split(adaptive=False, dspark=True, dpa=dpa)

    def test_dsv4_pcp_followers_use_cp_zero_split_offer(self):
        """A dummy CP sender has no D metadata and must still stop at the cut."""
        with patch.dict(
            os.environ,
            {"HANDOFF_HYBRID_SPLIT_PREFILL": "1", "HANDOFF_SPLIT_ADAPTIVE": "1"},
        ):
            self._check_dsv4_split(adaptive=True, dspark=True, dpa=True, pcp=True)

    def test_dsv4_pcp_split_covers_all_decode_tp_destinations(self):
        """CP0 must accept all compatible offers and use each D rank's pages."""
        with patch.dict(
            os.environ,
            {"HANDOFF_HYBRID_SPLIT_PREFILL": "1", "HANDOFF_SPLIT_ADAPTIVE": "1"},
        ):
            for width in (4, 8):
                with self.subTest(width=width):
                    self._check_dsv4_split(
                        adaptive=True, dspark=True, pcp=True, fanout=True, pcp_width=width
                    )

    def _check_dsv4_split(
        self,
        *,
        adaptive,
        invalid_hints=False,
        dspark=False,
        dpa=False,
        pcp=False,
        fanout=False,
        pcp_width=4,
    ):
        from sglang_remote_attn import dsv4_split_prefill as flash

        with patch.object(
            sys, "path", [str(ROOT / "test/manual/remote_attn"), *sys.path]
        ):
            from dsv41_remote_probe_support import definitions

        state_type = definitions(
            ROOT, "python/sglang/srt/disaggregation/base/conn.py", owner="StateType"
        )
        if pcp:
            manager_type = definitions(
                ROOT,
                "python/sglang/srt/disaggregation/common/conn.py",
                owner="CommonKVManager",
                members={"_resolve_rank_mapping"},
            )
            decoder = NS(
                attn_tp_size=pcp_width if fanout else 1,
                attn_cp_size=1,
                is_mla_backend=True,
                kv_args=NS(engine_rank=3),
                enable_all_cp_ranks_for_transfer=False,
                pp_size=1,
                pp_rank=0,
            )
            prefill = NS(
                attn_tp_size=1,
                attn_cp_size=pcp_width,
                enable_kv_cache_sharding=False,
                enable_dsa_cache_layer_split=False,
                pp_size=1,
            )
            manager_type._resolve_rank_mapping(decoder, prefill)
            self.assertEqual(
                (prefill.target_cp_ranks, prefill.target_tp_ranks), ([0], [0])
            )
            self.assertEqual(prefill.required_prefill_response_num, 1)
            self.assertEqual(prefill.required_dst_info_num, pcp_width if fanout else 1)
        component_type = definitions(
            ROOT,
            "python/sglang/srt/mem_cache/unified_cache/component_type.py",
            owner="ComponentType",
        )
        storage_module = "sglang.srt.mem_cache.static_kv_storage"
        storage = NS()
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.base.conn": NS(StateType=state_type),
                    "sglang.srt.mem_cache.unified_cache.component_type": NS(
                        ComponentType=component_type
                    ),
                    storage_module: storage,
                },
            )
        )

        @dataclass
        class Peer:
            dst_state_indices: list

        length = 8192 + 19
        page, window = 256, 128
        p = self.controller(dsv4=True)
        d = self.controller("decode", dsv4=True, max_prefill_tokens=4200)
        chunk = 1280 if invalid_hints else 2560
        if adaptive:
            p.split_prefill_prefix_rate = 6000
            budget = d.split_prefill_decode_budget
            budget.share = 0.05
            for _ in range(8):
                budget.observe(2560, 200, suffix=True, rank=1)
            self.assertLess(budget.coverage, 1)
            d.agree = lambda vote: (
                [*vote[:2], chunk + 140, 4500, -20_000, vote[-1]]
                if len(vote) == 6
                else vote
            )
        state_types = [state_type.SWA, state_type.DSV4_REQUEST_STATE]
        dst_state_indices = [[48, 91], [7]]
        src_state_indices = [[25], [3]]
        if dspark:
            state_types += [state_type.SWA, state_type.REQUEST_ROW]
            dst_state_indices += [[48, 91], [9]]
            src_state_indices += [[25], [4]]
        manager = NS(
            kv_shard_size=1,
            kv_args=NS(state_types=state_types),
            dsv41_spec_layout={"state_types": state_types} if dspark else None,
        )
        for controller in (p, d):
            if pcp and controller is p:
                p.tp_size = pcp_width
                p.attn_tp_rank = 0
                p.attn_tp_size = 1
                p.split_pcp = True
                p.agree = Mock(side_effect=lambda vote: vote)
            elif dpa:
                controller.rank = 3
                controller.tp_size = 4
                controller.attn_tp_rank = 0
                controller.attn_tp_size = 1
                controller.split_dpa = True
            controller.scheduler.sliding_window_size = window
            controller.pool.swa_page_size = page
            controller.pool.request_window = None
            controller.hf_config = NS(model_type="deepseek_v4")
            controller.dsv4_split_dspark = dspark
            controller.schema_signature = {"ratios": [0, 4, 128]}
            controller.dsv4_split_schema = flash.schema_id(
                controller,
                manager,
                NS(
                    attn_backend=NS(),
                    prefill_attention_backend_str="dsv4",
                    decode_attention_backend_str="dsv4",
                ),
            )
            self.assertIsNotNone(controller.dsv4_split_schema)
        req = self.request(length=length)
        receiver = self.receiver_type()
        receiver.require_staging = False
        receiver.clear = lambda: None
        dr = NS(
            req=req, kv_receiver=receiver, is_rebootstrap=False, metadata_buffer_index=0
        )
        d.scheduler.disagg_decode_prealloc_queue = prealloc = NS(
            scheduler=d.scheduler,
            token_to_kv_pool_allocator=NS(
                swa_available_size=lambda: 4608 if adaptive else 2304
            ),
            _swa_tail_len=lambda _: length - 7936,
        )

        def reserve(extra):
            return flash.select_swa_prealloc_start(
                lambda *_a, **_kw: None,
                prealloc,
                dr,
                prefix_len=0,
                fill_len=length,
                swa_tail_len=length - 7936,
                extra_swa_budget=extra,
            )

        self.assertIsNone(reserve(0))
        self.assertEqual(d.split_prefill_outstanding, 0)
        start = reserve(3072 if adaptive else 2048)
        self.assertEqual(start, 5632 if adaptive else 6144)
        self.assertEqual(receiver._dsv4_split_offer[0], 5888 if adaptive else 6400)
        req.kv.req_pool_idx = 0
        req.kv.set_row_len(length)
        req.kv.set_evicted_seqlen(component_type.SWA, start)
        req.time_stats = NS(set_wait_queue_entry_time=lambda: None)
        rows = torch.arange(length).reshape(1, -1)
        prealloc.req_to_token_pool = NS(req_to_token=rows)
        mapped = (
            list(range(93, 82, -1))
            if adaptive
            else [81, 44, 38, 92, 71, 15, 30, 48, 91]
        )
        storage.kv_storage_of = lambda *_: NS(
            window_ids=lambda _, tokens: torch.tensor(mapped) * page
        )
        self.split.advertise(prealloc, dr, 0)
        self.assertEqual(d.split_prefill_chunk_size, 4096)
        if adaptive:
            self.assertEqual(
                receiver._hybrid_split_schedule, (chunk, 150, 10, 4500, 20_000)
            )
            self.assertEqual(receiver._hybrid_split_suffix_cap, 2323)

        receiver.kv_mgr = NS()
        frames = self.mesh.bootstrap(
            lambda _, **kw: kw["message"],
            receiver,
            bootstrap_info={"is_dummy": False},
            message=[b""] * 10,
            operation="send_metadata",
        )
        if invalid_hints:
            frames = [
                frame
                for frame in frames
                if not frame.startswith(self.split.SUFFIX_RATE_WIRE)
            ]
            frames.extend(
                (
                    self.split.SUFFIX_CAP_WIRE + b"2147483648",
                    self.split.SUFFIX_RATE_WIRE + b"invalid",
                )
            )
        peer = self.mesh.parse_transfer(lambda **_: Peer(dst_state_indices), msg=frames)
        peer.is_dummy = False
        peer.mooncake_session_id = "decode"
        peer.dst_kv_indices = list(range(33))
        malformed = self.mesh.parse_transfer(
            lambda **_: NS(),
            msg=[b""] * 10 + [flash.WIRE + b"[]"],
        )
        self.assertFalse(hasattr(malformed, "_hybrid_split_limits"))
        self.assertEqual(
            malformed._hybrid_split_offer_error, "invalid_decode_split_offer"
        )
        sender = self.sender_type()
        sender.kv_mgr = NS(
            transfer_infos={7: {"decode": peer}},
            decode_kv_args_table={
                "decode": NS(requires_dcp_relayout=False, dst_attn_tp_size=1)
            },
        )
        if fanout:
            peers = sender.kv_mgr.transfer_infos[7]
            registrations = sender.kv_mgr.decode_kv_args_table
            for rank in range(pcp_width):
                session = f"decode-{rank}"
                destination = self.mesh.parse_transfer(
                    lambda **_: Peer(
                        [
                            [index + 100 * rank for index in row]
                            for row in dst_state_indices
                        ]
                    ),
                    msg=frames,
                )
                destination.is_dummy = False
                destination.required_dst_info_num = pcp_width
                destination.mooncake_session_id = session
                destination.dst_kv_indices = list(range(33))
                schema, start_page, pages = destination._dsv4_split_window
                destination._dsv4_split_window = (
                    schema,
                    start_page,
                    [index + 100 * rank for index in pages],
                )
                if rank == pcp_width - 1:
                    destination._hybrid_split_limits = (6144, 2048)
                    destination._hybrid_split_suffix_cap = length - 6144
                    destination._hybrid_split_suffix_rate = 4200
                peers[session] = destination
                registrations[session] = NS(
                    requires_dcp_relayout=False, dst_attn_tp_size=pcp_width, dst_tp_rank=rank
                )
            del peers["decode"], registrations["decode"]
            peer = peers["decode-0"]
        sender.curr_idx = 11  # pages, not tokens
        source = self.request(length=length, prefix=4096 if adaptive else 6656)
        source.disagg_kv_sender = sender
        self.split.create_sender(lambda *_: True, NS(scheduler=p.scheduler), source, 1)
        if fanout:
            self.assertEqual(
                self.split.offered_limits(source, p),
                ((6144, 2048, length - 6144, 4200), None),
            )
            with patch.object(peers[f"decode-{pcp_width - 1}"], "_hybrid_split_limits", None):
                self.assertEqual(
                    self.split.offered_limits(source, p), (None, "decode_did_not_offer")
                )
            with patch.object(registrations[f"decode-{pcp_width - 1}"], "dst_tp_rank", 0):
                self.assertEqual(
                    self.split.offered_limits(source, p),
                    (None, "incompatible_destination_registration"),
                )
            with patch.object(peer, "_dsv4_split_window", ("b" * 64, start, mapped)):
                self.assertEqual(
                    self.split.offered_limits(source, p),
                    (None, "incompatible_destination_split_layout"),
                )
        self.split.finalize_bootstrap(lambda *_: True, None, source)
        if pcp:
            leader_offer_vote = p.agree.call_args.args[0]
        if adaptive:
            self.assertEqual(
                source._hybrid_split_rates,
                None if invalid_hints else [4200 if fanout else 4500, 6000],
            )
        self.split.init_input(self.match, source, object())
        cut = source._hybrid_split_at
        self.assertEqual(cut, 6144 if adaptive else 7680)
        self.assertEqual(len(source.full_untruncated_fill_ids), cut)
        source.extend_end = cut
        self.split.seal(source, cut)
        self.assertEqual(sender.num_kv_indices, 24 if adaptive else 30)
        if pcp:
            # Native transports bootstrap on CP0 only. Exercise the other
            # three ranks with no peer or decoded SWA mapping at all.
            for rank in range(1, 4):
                follower = NS(**vars(p))
                follower.rank = rank
                follower.agree = Mock(
                    side_effect=lambda vote: [
                        min(local, leader)
                        for local, leader in zip(vote, leader_offer_vote, strict=True)
                    ]
                )
                dummy = self.request(length=length, prefix=4096)
                dummy.disagg_kv_sender = dummy_sender = self.sender_type()
                dummy_sender.kv_mgr = NS(is_dummy_cp_rank=True, transfer_infos={})
                dummy_sender.curr_idx = 11
                self.split.create_sender(
                    lambda *_: True, NS(scheduler=NS(_pre_layerwise=follower)), dummy, 1
                )
                self.split.finalize_bootstrap(lambda *_: True, None, dummy)
                follower.agree.assert_called_once_with([self.split.TP_MIN_IDENTITY] * 8)
                self.assertEqual(
                    dummy._hybrid_split_limits, source._hybrid_split_limits
                )
                self.assertEqual(dummy._hybrid_split_rates, source._hybrid_split_rates)
                follower.agree.side_effect = lambda vote: [
                    min(vote[0], cut),
                    min(vote[1], -cut),
                ]
                self.split.init_input(self.match, dummy, object())
                self.assertEqual(dummy._hybrid_split_at, cut)
                self.assertEqual(len(dummy.full_untruncated_fill_ids), cut)
                dummy.extend_end = cut
                self.split.seal(dummy, cut)
                self.assertEqual(dummy_sender.num_kv_indices, sender.num_kv_indices)
        sent = flash.send_states(
            lambda _, target, indices, *_args: (target.dst_state_indices, indices),
            manager,
            peer,
            src_state_indices,
            None,
        )
        cut_pages = [92 if adaptive else 15]
        expected_dst, expected_src = [cut_pages, [7]], [[25], None]
        if dspark:
            expected_dst += [cut_pages, [9]]
            expected_src += [[25], [4]]
            self.assertEqual(
                flash.send_states(
                    lambda *_: self.fail("draft SWA must be present at the cut"),
                    manager,
                    peer,
                    [[25], [3], None, [4]],
                    None,
                ),
                -1,
            )
            self.assertEqual(
                flash.send_states(
                    lambda _, target, indices, *_args: (
                        target.dst_state_indices,
                        indices,
                    ),
                    manager,
                    Peer(dst_state_indices),
                    src_state_indices,
                    None,
                ),
                (dst_state_indices, src_state_indices),
            )
        self.assertEqual(sent, (expected_dst, expected_src))
        self.assertEqual(peer.dst_state_indices, dst_state_indices)
        if fanout:
            for rank, destination in enumerate(peers.values()):
                pages = [92 + 100 * rank]
                sent = flash.send_states(
                    lambda _, target, indices, *_args: (
                        target.dst_state_indices,
                        indices,
                    ),
                    manager,
                    destination,
                    src_state_indices,
                    None,
                )
                offset = 100 * rank
                self.assertEqual(
                    sent, ([pages, [7 + offset], pages, [9 + offset]], expected_src)
                )
                self.assertEqual(
                    destination.dst_state_indices,
                    [[index + offset for index in row] for row in dst_state_indices],
                )

        freed = []
        prealloc.token_to_kv_pool_allocator.free_swa_segment = (
            lambda tokens, *, start_pos: freed.append((start_pos, tokens.tolist()))
        )
        queue = NS(
            scheduler=d.scheduler,
            metadata_buffers=NS(
                output_ids=torch.tensor([[99, cut]]),
                bootstrap_room=torch.tensor([[7]]),
                cached_tokens=torch.zeros((1, 2), dtype=torch.int32),
            ),
        )
        self.assertTrue(self.split.commit(queue, dr, keep_receiver=False))
        self.assertEqual(freed, [(start, list(range(start, cut - page)))])
        self.assertEqual(req.kv.get_evicted_seqlen(component_type.SWA), cut - page)
        self.assertEqual(
            (req.kv.kv_committed_len, req.kv.kv_allocated_len), (cut, length)
        )
        self.assertTrue(req.kv.reserved_prefill_suffix)
        if adaptive:
            self.assertEqual(req._hybrid_split_schedule, (chunk, 150, 10, 4500, 20_000))
        self.assertEqual(req.output_ids, [])
        self.split.release_reservation(req)
        self.assertEqual(d.split_prefill_outstanding, 0)

        # The native PD completion keeps the full-prompt SWA tail.
        req.kv.set_row_len(length)
        req.kv.set_evicted_seqlen(component_type.SWA, start)
        req._hybrid_split_reservation = d
        d.split_prefill_outstanding = 1
        dr.kv_receiver = receiver
        self.split.clear_metadata(lambda *_: None, queue.metadata_buffers, 0)
        self.assertFalse(self.split.commit(queue, dr, keep_receiver=False))
        self.assertEqual(req.kv.get_evicted_seqlen(component_type.SWA), 7936)
        self.assertEqual(d.split_prefill_outstanding, 0)

    def test_flash_dspark_reuses_native_bootstrap_layout_check_for_fallback(self):
        import logging
        from enum import StrEnum

        from sglang_remote_attn import dsv4_split_prefill as flash

        with patch.object(
            sys, "path", [str(ROOT / "test/manual/remote_attn"), *sys.path]
        ):
            from dsv41_remote_probe_support import definitions

        state_type = definitions(
            ROOT, "python/sglang/srt/disaggregation/base/conn.py", owner="StateType"
        )
        state_type = StrEnum(
            "StateType",
            {key: value for key, value in vars(state_type).items() if key.isupper()},
        )
        spec = NS(speculative_algorithm="DSPARK", speculative_num_draft_tokens=8)
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.base.conn": NS(StateType=state_type),
                    "sglang.srt.runtime_context": NS(
                        get_spec=lambda: spec,
                        get_disagg=lambda: NS(
                            disaggregation_mode="decode",
                            disaggregation_transfer_backend="mooncake",
                        ),
                    ),
                },
            )
        )
        layout_for = definitions(
            ROOT,
            "python/sglang/srt/disaggregation/utils.py",
            owner="get_dsv41_spec_layout",
        )
        layout_for.__globals__["get_spec"] = lambda: spec
        kv_args = NS(
            page_size=256,
            mla_compression_ratios=[0, 4, 128],
            kv_layer_ids=[0, 1, 2],
            kv_item_lens=[128, 256, 512],
            state_types=[
                state_type.SWA,
                state_type.DSV4_REQUEST_STATE,
                state_type.SWA,
                state_type.REQUEST_ROW,
            ],
            state_item_lens=[[4096], [128], [4096], [8]],
        )
        self.assertIsNone(layout_for(kv_args))
        with patch.dict(
            os.environ,
            {"HANDOFF_HYBRID_SPLIT_PREFILL": "1", "HANDOFF_PRE_LAYERWISE": "remote"},
        ):
            layout = flash.native_spec_layout(layout_for, kv_args)
            self.assertEqual(
                layout["state_types"], [kind.value for kind in kv_args.state_types]
            )
            spec.speculative_algorithm = None
            self.assertIsNone(flash.native_spec_layout(layout_for, kv_args))
            spec.speculative_algorithm = "DSPARK"
            kv_args.state_types.pop()
            with self.assertRaisesRegex(RuntimeError, "Flash DSpark PD requires"):
                flash.native_spec_layout(layout_for, kv_args)
            kv_args.state_types.append(state_type.REQUEST_ROW)

        conn = "python/sglang/srt/disaggregation/common/conn.py"
        ensure = definitions(
            ROOT, conn, owner="CommonKVManager", members={"try_ensure_parallel_info"}
        ).try_ensure_parallel_info
        ensure.__globals__.update(
            PrefillServerInfo=NS,
            StateType=state_type,
            logger=logging.getLogger(__name__),
        )
        d = NS(
            prefill_info_table={},
            kv_args=kv_args,
            kv_cache_dtype_str="fp8",
            kv_layout="v4",
            dsv41_spec_layout=layout,
            attn_cp_size=1,
            attn_tp_size=2,
            is_mla_backend=True,
            is_hybrid_mla_backend=False,
            dcp_size=1,
            _resolve_rank_mapping=lambda _: None,
        )
        info = dict(
            page_size=256,
            kv_cache_dtype="fp8",
            kv_layout="v4",
            attn_tp_size=2,
            attn_cp_size=1,
        )
        for remote_layout in (None, dict(layout, num_draft_tokens=4), layout):
            info["dsv41_spec_layout"] = remote_layout
            ensure.__globals__["requests"] = NS(
                get=lambda *_args, **_kwargs: NS(status_code=200, json=lambda: info)
            )
            if remote_layout != layout:
                with self.assertRaisesRegex(RuntimeError, "DSpark PD layout mismatch"):
                    ensure(d, "P")
                self.assertEqual(d.prefill_info_table, {})
            else:
                self.assertTrue(ensure(d, "P"))
                self.assertEqual(d.prefill_info_table["P"].dsv41_spec_layout, layout)

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

    def bootstrap(
        self,
        *,
        prefix=0,
        length=32768,
        hybrid=False,
        decode_budget=16384,
        legacy=False,
        decode=None,
        prefill=None,
    ):
        p = prefill or self.controller(hybrid=hybrid)
        d = decode or self.controller(
            "decode", hybrid=hybrid, max_prefill_tokens=decode_budget
        )
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
        if legacy:
            frames.remove(self.split.DECODE_PRIORITY)
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

    def test_dpa_split_requires_opt_in_and_native_dspark_coordination(self):
        from sglang.srt.environ import envs
        from sglang_remote_attn import hooks

        runtime = NS(
            get_device=lambda: NS(device="cuda"),
            get_parallel=lambda: NS(
                pp_size=1,
                dp_size=1,
                attn_cp_size=1,
                dcp_enabled=False,
                attn_dp_enabled=True,
                attn_dp_size=2,
            ),
            get_disagg=lambda: NS(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="mooncake",
                enable_pd_role_switch=False,
            ),
            get_exec=lambda: NS(overlap=NS(enable_two_batch_overlap=False)),
            get_memory=lambda: NS(
                disable_radix_cache=True,
                enable_unified_memory=False,
                enable_hierarchical_cache=False,
            ),
            get_schedule=lambda: NS(
                disable_overlap_schedule=False, enable_mixed_chunk=False, page_size=256
            ),
            get_spec=lambda: NS(speculative_algorithm="DSPARK"),
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": runtime,
                    "mooncake.engine": NS(TransferEngine=NS(send_probe=lambda: None)),
                },
            ),
            patch.dict(
                os.environ,
                {
                    "HANDOFF_HYBRID_SPLIT_PREFILL": "1",
                    "HANDOFF_PRE_LAYERWISE": "remote",
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "HANDOFF_SPLIT_DPA=1"):
                hooks.validate_config()
            os.environ["HANDOFF_SPLIT_DPA"] = "1"
            with self.assertRaisesRegex(
                RuntimeError, "SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION=1"
            ):
                hooks.validate_config()
            with envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(True):
                hooks.validate_config()
                os.environ["HANDOFF_EXPERIMENTAL_PD_REMOTE"] = "1"
                with self.assertRaisesRegex(RuntimeError, "native Mooncake PD"):
                    hooks.validate_config()

    def test_split_pcp_requires_native_replicated_topology(self):
        from sglang_remote_attn import hooks

        parallel = NS(
            pp_size=1,
            tp_size=4,
            dp_size=1,
            dcp_enabled=False,
            attn_cp_size=4,
            attn_dp_size=1,
            attn_dp_enabled=False,
            enable_prefill_cp=True,
            enable_cp_decode_attn_tp=False,
            cp_strategy="interleave",
            cp_kv_layout=None,
            enable_kv_cache_sharding=False,
            enable_dsa_cache_layer_split=False,
            enable_cp_tp_group_sharing=False,
        )
        disagg = NS(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
            enable_pd_role_switch=False,
            disaggregation_decode_host_receive_threshold=1,
        )
        runtime = NS(
            get_device=lambda: NS(device="cuda"),
            get_parallel=lambda: parallel,
            get_disagg=lambda: disagg,
            get_exec=lambda: NS(overlap=NS(enable_two_batch_overlap=False)),
            get_memory=lambda: NS(
                disable_radix_cache=True,
                enable_unified_memory=False,
                enable_hierarchical_cache=False,
            ),
            get_schedule=lambda: NS(
                disable_overlap_schedule=False, enable_mixed_chunk=False, page_size=256
            ),
            get_spec=lambda: NS(speculative_algorithm="DSPARK"),
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": runtime,
                    "mooncake.engine": NS(TransferEngine=NS(send_probe=lambda: None)),
                },
            ),
            patch.dict(
                os.environ,
                {
                    "HANDOFF_HYBRID_SPLIT_PREFILL": "1",
                    "HANDOFF_PRE_LAYERWISE": "remote",
                },
            ),
        ):
            for width in (2, 4, 6, 8):
                with (
                    self.subTest(role=disagg.disaggregation_mode, width=width),
                    patch.multiple(parallel, tp_size=width, attn_cp_size=width),
                ):
                    hooks.validate_config()
            for config, name, value in (
                (disagg, "disaggregation_mode", "decode"),
                (parallel, "attn_cp_size", 2),
                (parallel, "enable_kv_cache_sharding", True),
            ):
                with (
                    self.subTest(name=name),
                    patch.object(config, name, value),
                    self.assertRaisesRegex(RuntimeError, "TP=CP,"),
                ):
                    hooks.validate_config()

            disagg.disaggregation_mode = "decode"
            os.environ["HANDOFF_SPLIT_DECODE_PCP"] = "1"
            with self.assertRaisesRegex(RuntimeError, "--enable-cp-decode-attn-tp"):
                hooks.validate_config()
            parallel.enable_cp_decode_attn_tp = True
            for width in (2, 4, 6, 8):
                with (
                    self.subTest(role=disagg.disaggregation_mode, width=width),
                    patch.multiple(parallel, tp_size=width, attn_cp_size=width),
                ):
                    hooks.validate_config()
            disagg.disaggregation_decode_host_receive_threshold = 0
            with self.assertRaisesRegex(RuntimeError, "requires GPU KV receive"):
                hooks.validate_config()

    def test_dpa_split_negotiates_inside_the_request_attention_group(self):
        with patch.dict(os.environ, {"HANDOFF_SPLIT_ADAPTIVE": "1"}):
            p, d = self.controller(), self.controller("decode")
        for controller in (p, d):
            controller.rank = 3
            controller.tp_size = 4
            controller.attn_tp_rank = 0
            controller.attn_tp_size = 1
            controller.split_dpa = True
        p.split_prefill_prefix_rate = 6000
        d.split_prefill_decode_budget.suffix_rate = 4500
        _, _, req, dr = self.bootstrap(prefill=p, decode=d)
        self.assertEqual(req._hybrid_split_rates, [4500, 6000])
        self.assertEqual(dr.kv_receiver._hybrid_split_schedule[3], 4500)
        self.split.init_input(self.match, req, object())
        self.assertEqual(req._hybrid_split_at, 24576)

    def test_dpa_prefill_samples_only_the_attention_group_leader(self):
        with patch.dict(os.environ, {"HANDOFF_SPLIT_ADAPTIVE": "1"}):
            p = self.controller()
        p.split_dpa = p.dsv4_split_dspark = True
        p.rank = 2
        worker = NS(target_worker=NS(model_runner=NS(_pre_layerwise=p)))
        batch = NS(
            forward_mode=NS(is_extend=lambda: True),
            extend_num_tokens=4096,
        )
        start, end = Mock(), Mock()
        end.query.return_value = True
        start.elapsed_time.return_value = 500
        with (
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.cuda, "Event", side_effect=[start, end]) as event,
        ):
            p.attn_tp_rank = 0
            result = self.split.time_dspark_forward(lambda *_: NS(), worker, batch)
            self.split.record_completed_suffix(p, result)
            self.assertEqual(p.split_prefill_prefix_rate, 8192)
            p.attn_tp_rank = 1
            result = self.split.time_dspark_forward(lambda *_: NS(), worker, batch)
            self.assertFalse(hasattr(result, "_split_suffix_timing"))
        self.assertEqual(event.call_count, 2)

    def test_dpa_suffix_and_native_steps_share_one_batch_sync(self):
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang_remote_attn import decode_overlap

        d = self.controller("decode")
        d.split_dpa = d.dsv4_split_dspark = True
        s = d.scheduler
        s.last_batch = s.chunked_req = None
        s.token_to_kv_pool_allocator = s.tree_cache = s.model_config = None
        s.enable_overlap = True
        s.spec_algorithm = "DSPARK"
        req = self.request(length=8192, prefix=6144)
        req.kv.reserved_prefill_suffix = True
        req.refresh_fill_ids = Mock()
        req.time_stats = Mock()
        req.extend_len = 2048
        req.extend_batch_idx = req.cached_tokens = req._hybrid_split_defer_ticks = 0
        req._hybrid_split_ready_at = None

        class WaitingQueue(list):
            def leave(self, reqs):
                for req in reqs:
                    self.remove(req)

        s.waiting_queue = WaitingQueue([req])
        suffix = NS(
            forward_mode=ForwardMode.EXTEND, prepare_for_extend=Mock(), extend_len=2048
        )
        s.dp_attn_adapter = NS()

        def sync_batch(batch):
            if batch is None:
                batch = NS(forward_mode=ForwardMode.IDLE)
            batch.is_extend_in_batch = True
            return batch

        sync = s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = Mock(
            side_effect=sync_batch
        )

        def native(scheduler, running):
            batch = None if running.is_empty() else running
            return NS(
                batch_to_run=scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
                    batch
                ),
                running_batch=running,
            )

        native = Mock(side_effect=native)
        idle = NS(is_empty=lambda: True, batch_size=lambda: 0)
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.schedule_batch": NS(
                    NextBatchPlan=NS,
                    ScheduleBatch=NS(init_new=lambda *args, **kwargs: suffix),
                ),
                "sglang.srt.observability.req_time_stats": NS(
                    set_schedule_time_batch=lambda _: None
                ),
            },
        ):
            result = self.split.next_batch(native, s, idle)
        self.assertIs(result.batch_to_run, suffix)
        self.assertTrue(suffix._hybrid_split_extend)
        sync.assert_called_once_with(suffix)
        native.assert_not_called()

        running = NS(forward_mode=ForwardMode.DECODE, is_empty=lambda: False)
        for active in (running, idle):
            result = self.split.next_batch(native, s, active)
            self.assertTrue(result.batch_to_run.is_extend_in_batch)
            self.assertTrue(
                decode_overlap.process_last_batch_early(
                    lambda *_: False, s, result.batch_to_run
                )
            )
        self.assertEqual(
            [call.args[0] for call in sync.call_args_list], [suffix, running, None]
        )
        self.assertEqual(native.call_count, 2)

    def test_default_cut_uses_prompt_bound_without_a_fixed_cap(self):
        for hybrid in (False, True):
            for length, prefix, expected in (
                (16384, 0, 12288),
                (16384, 14745, 15360),
                (32768, 0, 24576),
                (32768, 13107, 27853),
                (32768, 19660, 29491),
                (32768, 26214, 30720),
                (65408, 0, 49056),
                (65536, 39321, 58983),
                (131072, 0, 98304),
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

    def test_suffix_can_span_multiple_decode_chunks(self):
        with patch.dict(os.environ, {"HANDOFF_SPLIT_SUFFIX_CAP_TOKENS": "8192"}):
            _, d, req, _ = self.bootstrap(prefix=13107, decode_budget=1024)
            self.assertEqual(d.split_prefill_chunk_size, 1024)
            self.split.init_input(self.match, req, object())
            self.assertEqual(req._hybrid_split_at, 27853)

    def test_adaptive_offer_uses_tp_minimum_without_replacing_chunk_cap(self):
        with patch.dict(
            os.environ,
            {
                "HANDOFF_SPLIT_CAP_AUTO": "1",
                "HANDOFF_HYBRID_SPLIT_PREFILL_CHUNK_SIZE": "8192",
                "HANDOFF_SPLIT_SUFFIX_CAP_TOKENS": "auto",
            },
        ):
            for local_cap in (10240, 32768):
                with self.subTest(local_cap=local_cap):
                    d = self.controller("decode")
                    d.tp_size = 4
                    d.split_prefill_auto_cap = local_cap
                    d.agree = lambda vote: [vote[0], min(vote[1], 10240)]
                    _, _, req, dr = self.bootstrap(length=65536, prefix=16384, decode=d)
                    self.assertEqual(dr.kv_receiver._hybrid_split_limits, (55296, 8192))
                    self.assertEqual(req._hybrid_split_limits, (55296, 8192, 10240))
                    self.split.init_input(self.match, req, object())
                    self.assertEqual(req._hybrid_split_at, 55296)

            # An integer on P still bounds its cut under an adaptive offer.
            with patch.dict(os.environ, {"HANDOFF_SPLIT_SUFFIX_CAP_TOKENS": "4096"}):
                _, _, req, _ = self.bootstrap(length=65536, decode=d)
                self.split.init_input(self.match, req, object())
                self.assertEqual(req._hybrid_split_at, 61440)

    def test_dynamic_suffix_cap_is_clamped_per_prompt_and_round_trips(self):
        for mode in ("HANDOFF_SPLIT_ADAPTIVE", "HANDOFF_SPLIT_CAP_AUTO"):
            with (
                self.subTest(mode=mode),
                patch.dict(
                    os.environ,
                    {
                        mode: "1",
                        "HANDOFF_HYBRID_SPLIT_PREFILL_CHUNK_SIZE": "4096",
                        "HANDOFF_SPLIT_SUFFIX_CAP_TOKENS": "auto",
                    },
                ),
            ):
                d = self.controller("decode")
                budget = d.split_prefill_decode_budget
                if budget is None:
                    d.split_prefill_auto_cap = 60_000
                else:
                    self.assertEqual(budget.share, 0.25)
                    for length, cap in ((8193, 2048), (98305, 24576)):
                        receiver = self.offer(d, length=length).kv_receiver
                        self.assertEqual(receiver._hybrid_split_suffix_cap, cap)

                    for _ in range(7):
                        budget.observe(1024, 25, suffix=True, rank=0)
                    budget.observe(1, 1400, suffix=False, rank=0)
                    self.assertEqual(budget.share, 0.25)
                    budget.observe(1024, 25, suffix=True, rank=0)
                    self.assertAlmostEqual(budget.share, 0.275)
                    for _ in range(8):
                        budget.observe(1, 1400, suffix=False, rank=0)
                        for _ in range(8):
                            budget.observe(1024, 25, suffix=True, rank=0)
                    self.assertEqual(budget.share, 0.5)

                short = self.offer(d, length=8193).kv_receiver
                self.assertEqual(short._hybrid_split_suffix_cap, 4096)
                _, _, req, long = self.bootstrap(length=98305, decode=d)
                self.assertEqual(long.kv_receiver._hybrid_split_suffix_cap, 49152)
                self.assertEqual(req._hybrid_split_limits, (49153, 4096, 49152))
                if budget is None:
                    self.assertEqual(d.split_prefill_auto_cap, 60_000)
                else:
                    self.assertEqual(budget.share, 0.5)

                d.split_prefill_page_size = 256
                offer, reason = self.split.negotiate_decode_offer(
                    d, self.request(length=98305), None, min_cut=64020
                )
                self.assertIsNone(reason)
                self.assertEqual(offer.suffix_cap, 98305 - 64256)

    def test_auto_prefill_falls_back_to_the_negotiated_bound(self):
        with patch.dict(os.environ, {"HANDOFF_SPLIT_SUFFIX_CAP_TOKENS": "auto"}):
            _, d, req, dr = self.bootstrap()
        self.assertEqual(d.split_prefill_auto_cap, 0)
        self.assertEqual(dr.kv_receiver._hybrid_split_suffix_cap, 0)
        self.split.init_input(self.match, req, object())
        self.assertEqual(req._hybrid_split_at, 24576)

    def test_feedback_offer_agrees_admission_and_freezes_selected_peer_rates(self):
        with patch.dict(os.environ, {"HANDOFF_SPLIT_ADAPTIVE": "1"}):
            p = self.controller()
            d = self.controller("decode")
            other = self.controller("decode")
        # A busy D rank must veto the split even when the local rank has room.
        d.agree = lambda vote: [0, *vote[1:]]
        self.assertFalse(hasattr(self.offer(d).kv_receiver, "_hybrid_split_limits"))
        self.assertEqual(d.split_prefill_outstanding, 0)

        p.split_prefill_prefix_rate = 6000
        for worker, rate in ((d, 6000), (other, 3000)):
            worker.split_prefill_decode_budget.suffix_rate = rate
            worker.split_prefill_decode_budget.decode_ms = 10
        d.agree = lambda vote: [vote[0], 8192, 3072, 4500, -20_000, 1000]
        _, _, first, dr = self.bootstrap(prefill=p, decode=d, prefix=20480)
        _, _, second, _ = self.bootstrap(prefill=p, decode=other, prefix=20480)
        self.assertEqual(dr.kv_receiver._hybrid_split_limits, (24576, 3072))
        self.assertEqual(
            dr.kv_receiver._hybrid_split_schedule, (3072, 150, 10, 4500, 20_000)
        )
        self.split.init_input(self.match, first, object())
        self.split.init_input(self.match, second, object())
        self.assertEqual(first._hybrid_split_at, 27502)
        self.assertEqual(second._hybrid_split_at, 28672)

    def test_decode_timing_offer_publishes_rank_zero_estimates_and_preserves_veto(self):
        def reduce_min(vote, peer_vote):
            return [
                min(local, peer) for local, peer in zip(vote, peer_vote, strict=True)
            ]

        for adaptive in ("0", "1"):
            with (
                self.subTest(adaptive=adaptive),
                patch.dict(
                    os.environ,
                    {
                        "HANDOFF_SPLIT_CAP_AUTO": "1",
                        "HANDOFF_SPLIT_ADAPTIVE": adaptive,
                    },
                ),
            ):
                leader = self.controller("decode")
                follower = self.controller(
                    "decode", max_prefill_tokens=2048 if adaptive == "1" else 16384
                )
                leader.tp_size = follower.tp_size = 2
                follower.rank = 1
                if adaptive == "1":
                    budget = leader.split_prefill_decode_budget
                    budget.suffix_rate = 5000
                    budget.decode_ms = 20
                    expected_schedule = (2048, 150, 10, 5000, 20_000)
                else:
                    leader.split_prefill_auto_cap = 8192
                    expected_schedule = None

                leader.agree = Mock(side_effect=lambda vote: vote)
                self.split.negotiate_decode_offer(leader, self.request(), None)
                leader_vote = leader.agree.call_args.args[0]
                follower.agree = Mock(
                    side_effect=lambda vote: reduce_min(vote, leader_vote)
                )
                follower_receiver = self.offer(follower).kv_receiver

                follower_vote = follower.agree.call_args.args[0]
                leader.agree.side_effect = lambda vote: reduce_min(vote, follower_vote)
                leader_receiver = self.offer(leader).kv_receiver
                for receiver in (leader_receiver, follower_receiver):
                    self.assertEqual(receiver._hybrid_split_suffix_cap, 8192)
                    self.assertEqual(
                        getattr(receiver, "_hybrid_split_schedule", None),
                        expected_schedule,
                    )

                follower.scheduler.running_batch.batch_size = lambda: 16
                self.assertFalse(
                    hasattr(self.offer(follower).kv_receiver, "_hybrid_split_limits")
                )
                follower_vote = follower.agree.call_args.args[0]
                self.assertFalse(
                    hasattr(self.offer(leader).kv_receiver, "_hybrid_split_limits")
                )
                self.assertEqual(leader.split_prefill_outstanding, 1)
                self.assertEqual(follower.split_prefill_outstanding, 1)

    def test_feedback_suffix_floor_preserves_splits_after_large_cache_hits(self):
        with patch.dict(os.environ, {"HANDOFF_SPLIT_ADAPTIVE": "1"}):
            for length, prefix, rate, cut, reason in (
                (36000, 32400, 0, 33952, "selected"),
                (36000, 32400, 6000, 33952, "selected"),
                (36000, 33952, 6000, 0, "split_too_small"),
                (8192, 5200, 6000, 0, "split_too_small"),
            ):
                with self.subTest(length=length, prefix=prefix, rate=rate):
                    p = self.controller()
                    d = self.controller("decode")
                    p.split_prefill_prefix_rate = 6000
                    d.split_prefill_decode_budget.suffix_rate = rate
                    _, _, req, _ = self.bootstrap(
                        prefix=prefix, length=length, prefill=p, decode=d
                    )
                    self.split.init_input(self.match, req, object())
                    self.assertEqual(req._hybrid_split_at, cut)
                    self.assertEqual(req._hybrid_split_match_reason, reason)

    def test_feedback_chunk_floor_respects_scheduler_limit_and_explicit_pin(self):
        for max_tokens, pin, expected in (
            (16384, "0", 1024),
            (768, "0", 768),
            (16384, "512", 512),
        ):
            with (
                self.subTest(max_tokens=max_tokens, pin=pin),
                patch.dict(
                    os.environ,
                    {
                        "HANDOFF_SPLIT_ADAPTIVE": "1",
                        "HANDOFF_HYBRID_SPLIT_PREFILL_CHUNK_SIZE": pin,
                    },
                ),
            ):
                d = self.controller("decode", max_prefill_tokens=max_tokens)
                budget = d.split_prefill_decode_budget
                budget.suffix_rate = 1000
                receiver = self.offer(d).kv_receiver
                self.assertEqual(receiver._hybrid_split_limits[1], expected)

    def test_feedback_coverage_recovers_after_reaching_share_floor(self):
        with patch.dict(
            os.environ,
            {"HANDOFF_SPLIT_ADAPTIVE": "1", "HANDOFF_SPLIT_DECODE_BUDGET_G": "0.2"},
        ):
            d = self.controller("decode")
        budget = d.split_prefill_decode_budget
        for _ in range(8):
            budget.observe(2048, 200, suffix=True, rank=1)
        self.assertAlmostEqual(budget.share, 0.225)
        self.assertEqual(budget.coverage, 1)
        for _ in range(49 * 8):
            budget.observe(2048, 200, suffix=True, rank=1)
        self.assertEqual(budget.share, 0.05)
        self.assertAlmostEqual(budget.coverage, 0.05)
        for _ in range(35 * 8):
            budget.observe(1, 200, suffix=False, rank=1)
            budget.observe(2048, 10, suffix=True, rank=1)
        self.assertEqual(budget.coverage, 1)
        self.assertGreater(budget.share, 0.05)

    def test_feedback_observes_prefill_overlap_and_decode_only_when_enabled(self):
        from sglang_remote_attn import decode_overlap

        decode = NS(
            forward_mode=NS(is_decode=lambda: True, is_extend=lambda: False),
            batch_size=lambda: 4,
        )
        prefix = NS(forward_mode=NS(is_extend=lambda: True), extend_num_tokens=6000)
        suffix = NS(
            _hybrid_split_extend=True,
            forward_mode=NS(is_extend=lambda: True),
            extend_num_tokens=4096,
        )
        with patch.dict(os.environ, {"HANDOFF_SPLIT_ADAPTIVE": "1"}):
            p = self.controller()
            d = self.controller("decode")
            follower = self.controller("decode")
        with patch.dict(os.environ, {"HANDOFF_SPLIT_CAP_AUTO": "1"}):
            rate_only_follower = self.controller("decode")
        p.rank = follower.rank = rate_only_follower.rank = 1
        p.prefill_overlap = NS()
        static = self.controller("decode")
        for controller, batch, ms, phase in (
            (static, decode, 10, None),
            (p, prefix, 1000, "prefix"),
            (d, decode, 20, "decode"),
            (d, suffix, 500, "suffix"),
            (follower, decode, 20, None),
            (follower, suffix, 500, None),
            (rate_only_follower, suffix, 500, None),
        ):
            worker = NS(model_runner=NS(_pre_layerwise=controller))
            start = Mock()
            start.elapsed_time.return_value = ms
            end = Mock()
            end.query.return_value = True
            with (
                patch.object(torch.cuda, "current_stream", return_value=object()),
                patch.object(torch.cuda, "Event", side_effect=[start, end]) as event,
            ):
                result = self.split.time_suffix_forward(lambda *_: NS(), worker, batch)
                timing = getattr(result, "_split_suffix_timing", None)
                self.assertEqual(timing[-1] if timing else None, phase)
                self.assertEqual(event.call_count, 2 if phase else 0)
                decode_overlap.after_result(
                    lambda *_: None, controller.scheduler, batch, result
                )
        self.assertEqual(p.split_prefill_prefix_rate, 6000)
        self.assertEqual(d.split_prefill_decode_budget.decode_ms, 20)
        self.assertEqual(d.split_prefill_decode_budget.suffix_rate, 8192)
        self.assertIsNone(follower.split_prefill_decode_budget.decode_ms)
        self.assertIsNone(follower.split_prefill_decode_budget.suffix_rate)
        self.assertEqual(rate_only_follower.split_prefill_rate_samples, 0)

    def test_flash_dspark_times_the_outer_worker_once(self):
        from sglang_remote_attn import decode_overlap

        with patch.dict(os.environ, {"HANDOFF_SPLIT_ADAPTIVE": "1"}):
            d = self.controller("decode")
        d.dsv4_split_dspark = True
        target = NS(model_runner=NS(_pre_layerwise=d))
        worker = NS(target_worker=target)
        batch = NS(
            forward_mode=NS(is_extend=lambda: True),
            _hybrid_split_extend=True,
            extend_num_tokens=4096,
        )
        trace = []
        start, end = Mock(), Mock()
        start.record.side_effect = lambda _: trace.append("start")
        end.record.side_effect = lambda _: trace.append("end")
        end.query.return_value = True
        start.elapsed_time.return_value = 500

        def forward(_worker, batch):
            output = self.split.time_suffix_forward(lambda *_: NS(), target, batch)
            trace.append("draft KV written")
            return output

        with (
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.cuda, "Event", side_effect=[start, end]),
        ):
            result = self.split.time_dspark_forward(forward, worker, batch)
        self.assertEqual(trace, ["start", "draft KV written", "end"])
        decode_overlap.after_result(lambda *_: None, d.scheduler, batch, result)
        self.assertEqual(d.split_prefill_decode_budget.suffix_rate, 8192)

    def test_suffix_events_follow_native_results_not_scheduler_iterations(self):
        from sglang_remote_attn import decode_overlap

        with patch.dict(os.environ, {"HANDOFF_SPLIT_CAP_AUTO": "1"}):
            d = self.controller("decode")
        worker = NS(model_runner=NS(_pre_layerwise=d))
        stream = object()
        batch = NS(
            _hybrid_split_extend=True,
            forward_mode=NS(is_extend=lambda: True),
            extend_num_tokens=4096,
        )
        first_start, first_end, second_start, second_end = (
            Mock(),
            Mock(),
            Mock(),
            Mock(),
        )
        first_start.elapsed_time.return_value = 500
        second_start.elapsed_time.return_value = 500
        first_end.query.return_value = False
        second_end.query.return_value = False
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(
                torch.cuda,
                "Event",
                side_effect=[first_start, first_end, second_start, second_end],
            ),
        ):
            first = self.split.time_suffix_forward(lambda *_: NS(), worker, batch)
            batch.extend_num_tokens = 2048
            second = self.split.time_suffix_forward(lambda *_: NS(), worker, batch)
        self.assertEqual(d.split_prefill_auto_cap, 4096)

        def consume(_, _batch, result):
            if result is first:
                first_end.query.return_value = True
            else:
                second_end.query.return_value = True

        decode_overlap.after_result(consume, d.scheduler, batch, first)
        self.assertEqual(d.split_prefill_auto_cap, 16384)
        self.assertEqual(d.split_prefill_rate_samples, 1)
        decode_overlap.after_result(consume, d.scheduler, batch, first)
        self.assertEqual(d.split_prefill_rate_samples, 1)
        decode_overlap.after_result(consume, d.scheduler, batch, second)
        self.assertEqual(d.split_prefill_auto_cap, 13824)
        self.assertEqual(d.split_prefill_rate_samples, 2)

        # A result which is still incomplete must not affect a later sample.
        incomplete = NS(_split_suffix_timing=(8192, Mock(), Mock(), "suffix"))
        incomplete._split_suffix_timing[2].query.return_value = False
        decode_overlap.after_result(lambda *_: None, d.scheduler, batch, incomplete)
        self.assertFalse(hasattr(incomplete, "_split_suffix_timing"))
        self.assertEqual(d.split_prefill_auto_cap, 13824)

    def test_balanced_cut_does_not_depend_on_decode_deferral(self):
        for defer, legacy in (("150", False), ("0", False), ("150", True)):
            with (
                self.subTest(defer=defer, legacy=legacy),
                patch.dict(
                    os.environ,
                    {
                        "HANDOFF_SPLIT_CUT_BASIS": "balanced",
                        "HANDOFF_SPLIT_DEFER_TICKS": defer,
                    },
                ),
            ):
                _, _, req, _ = self.bootstrap(legacy=legacy)
                self.split.init_input(self.match, req, object())
                self.assertEqual(req._hybrid_split_at, 22476)

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
        with patch.dict(
            os.environ,
            {
                "HANDOFF_HYBRID_SPLIT_PREFILL_CUT_RATIO": "0.70",
                "HANDOFF_SPLIT_SUFFIX_CAP_TOKENS": "16384",
            },
        ):
            _, _, req, _ = self.bootstrap()
            self.split.init_input(self.match, req, object())
            self.assertEqual(req._hybrid_split_at, 22938)
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
        self.assertEqual(req._hybrid_split_at, 24576)
        req.prefix_len = req.kv.cache_protected_len = 29491
        self.split.init_input(self.match, req, cache)
        self.assertEqual(req._hybrid_split_at, 30720)
        req.kv.req_pool_idx = 0
        req.prefix_len = req.kv.cache_protected_len = 30000
        self.split.init_input(self.match, req, cache)
        self.assertEqual(req._hybrid_split_at, 30720)

    def test_opt_in_backlog_gate_counts_other_prefill_work(self):
        with patch.dict(os.environ, {"HANDOFF_SPLIT_MIN_P_QUEUE_TOKENS": "4096"}):
            p, _, req, _ = self.bootstrap()
        p.scheduler.waiting_queue = [req]
        p.scheduler.chunked_req = None
        self.split.init_input(self.match, req, object())
        self.assertEqual(req._hybrid_split_at, 0)
        self.assertEqual(req._hybrid_split_match_reason, "p_backlog_below_threshold")

        active = self.request(length=16384)
        active.extend_end = 8192
        p.scheduler.chunked_req = active
        self.split.init_input(self.match, req, object())
        self.assertEqual(req._hybrid_split_at, 24576)

    def test_decode_offers_reserve_capacity_before_transfer(self):
        d = self.controller("decode")
        limit = d.scheduler.max_running_requests
        active = [self.offer(d) for _ in range(limit)]
        self.split.advertise(NS(scheduler=d.scheduler), active[0], 0)
        self.assertTrue(all(dr.kv_receiver._hybrid_split_limits for dr in active))
        self.assertEqual(d.split_prefill_outstanding, limit)
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
        self.assertEqual(d.split_prefill_outstanding, limit - 1)
        self.assertTrue(self.offer(d).kv_receiver._hybrid_split_limits)

    def test_decode_capacity_and_tp_rejection_do_not_create_reservations(self):
        d = self.controller("decode")
        d.scheduler.running_batch.batch_size = lambda: d.scheduler.max_running_requests
        self.assertFalse(hasattr(self.offer(d).kv_receiver, "_hybrid_split_limits"))
        d.scheduler.running_batch.batch_size = lambda: 0
        d.agree = lambda _: [0, 0]
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
        peer = self.mesh.parse_transfer(
            lambda **_: NS(),
            msg=[b""] * 10
            + [
                self.split.WIRE + b"16384:4096",
                self.split.SUFFIX_CAP_WIRE + b"2147483648",
            ],
        )
        self.assertEqual(peer._hybrid_split_limits, (16384, 4096))
        self.assertFalse(hasattr(peer, "_hybrid_split_suffix_cap"))

    def commit_fixture(self, cut, *, hybrid=False, room=7, length=32768, decode=None):
        _, d, _, dr = self.bootstrap(hybrid=hybrid, length=length, decode=decode)
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

    def test_decode_priority_expires_per_chunk(self):
        for rate, hits, budget, end_bs, defer_ticks in (
            (None, 19660, 150, 1, 150),
            (None, 26214, 45, 1, 150),
            (None, 19660, 2, 0, 150),
            (None, 19660, 15, 2, 150),
            (2560, 19660, 80, 1, 150),
            (2560, 26214, 80, 1, 150),
            (1280, 19660, 160, 1, 150),
            (2560, 19660, 0, 1, 0),
            (0, 19660, 17, 1, 17),
        ):
            adaptive = rate is not None
            with self.subTest(rate=rate, hits=hits, budget=budget, end_bs=end_bs):
                decode = None
                cut = 24576
                if adaptive:
                    with patch.dict(
                        os.environ,
                        {
                            "HANDOFF_SPLIT_ADAPTIVE": "1",
                            "HANDOFF_SPLIT_DEFER_TICKS": str(defer_ticks),
                            "HANDOFF_SPLIT_DEFER_PATIENCE_MS": "0",
                        },
                    ):
                        decode = self.controller("decode")
                    # The TP-voted rate and step time steer both ranks even
                    # when the local estimates differ.
                    feedback = decode.split_prefill_decode_budget
                    feedback.suffix_rate = 20_000
                    feedback.decode_ms = 10
                    decode.agree = lambda vote: (
                        [vote[0], 8192, 4096, rate, -20_000, 1000]
                        if len(vote) == 6
                        else vote
                    )
                    cut = 26624
                d, queue, dr = self.commit_fixture(cut, decode=decode)
                queue.metadata_buffers.cached_tokens[0, 0] = hits
                self.assertTrue(self.split.commit(queue, dr, keep_receiver=False))
                req = dr.req
                req.extend_batch_idx = 0
                req.refresh_fill_ids = lambda: None
                req.time_stats.set_forward_entry_time = lambda: None
                req.finished = lambda: False
                req.inflight_steps = 0
                scheduler = d.scheduler
                scheduler.last_batch = scheduler.chunked_req = None
                scheduler.process_pending_chunked_abort = lambda: None
                scheduler.token_to_kv_pool_allocator = scheduler.tree_cache = None
                scheduler.model_config = scheduler.spec_algorithm = None
                scheduler.enable_overlap = True

                class WaitingQueue(list):
                    def leave(self, reqs):
                        self.remove(reqs[0])

                    def prepend(self, reqs):
                        self[:0] = reqs

                scheduler.waiting_queue = waiting = WaitingQueue([req])
                running = Mock()
                running.batch_size.return_value = 2

                def prepare_for_extend(**_):
                    req.extend_len = req.extend_end - req.prefix_len

                batch = NS(prepare_for_extend=prepare_for_extend)
                native_decode = Mock(return_value=running)
                modules = {
                    "sglang.srt.managers.schedule_batch": NS(
                        ScheduleBatch=NS(init_new=Mock(return_value=batch)),
                        NextBatchPlan=NS,
                    ),
                    "sglang.srt.observability.req_time_stats": NS(
                        set_schedule_time_batch=lambda _: None
                    ),
                }
                with patch.dict(sys.modules, modules):
                    for tick in range(budget):
                        if tick == 1 and end_bs:
                            running.batch_size.return_value = end_bs
                        self.assertIs(
                            self.split.next_batch(native_decode, scheduler, running),
                            running,
                        )
                    self.assertEqual(waiting, [req])
                    running.batch_size.return_value = end_bs
                    plan = self.split.next_batch(native_decode, scheduler, running)
                    self.assertIs(plan.batch_to_run, batch)
                    self.assertEqual(req.extend_end, cut + 4096)
                    self.assertEqual(req._hybrid_split_defer_ticks, 0)
                    self.assertEqual(waiting, [])
                    self.assertIs(scheduler.chunked_req, req)
                    req.kv.commit_prealloc(req.extend_end)
                    running.batch_size.return_value = 1
                    if adaptive:
                        final_wait = budget // 2 if rate else defer_ticks
                    else:
                        final_wait = 1
                    for _ in range(final_wait):
                        self.assertIs(
                            self.split.next_batch(native_decode, scheduler, running),
                            running,
                        )
                    self.assertEqual(req._hybrid_split_defer_ticks, final_wait)
                    if adaptive:
                        plan = self.split.next_batch(native_decode, scheduler, running)
                        self.assertIs(plan.batch_to_run, batch)
                        self.assertEqual(req.extend_end, 32768)
                        self.assertIsNone(scheduler.chunked_req)

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
            scheduler.maybe_send_cached_prefix_chunk = lambda req, allow_send=True: (
                prefix_sends.append((req, allow_send))
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
