"""CPU contracts for the optional plugin; GPU/model imports stay at the boundary.

Source-loaded pool methods are the real upstream implementations. This avoids
loading Triton/CUDA just to exercise row ownership and token-map lifetimes.
"""

import ast
import dataclasses
import importlib
import importlib.util
import json
import os
import pickle
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from array import array
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
PLUGIN = ROOT / "plugins" / "remote_attn"


def _source_class(relative, owner, members=None):
    path = ROOT / "python" / "sglang" / "srt" / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node.bases = []
    node.keywords = []
    if members is not None:
        node.body = [n for n in node.body if getattr(n, "name", None) in members]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "dataclasses": dataclasses,
        "dataclass": dataclasses.dataclass,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[owner]


class TestRemoteAttn(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.stack = ExitStack()
        cls.stack.enter_context(patch.dict(sys.modules, {}))
        spec = importlib.util.spec_from_file_location(
            "sglang_remote_attn",
            PLUGIN / "sglang_remote_attn" / "__init__.py",
            submodule_search_locations=[str(PLUGIN / "sglang_remote_attn")],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        cls.hooks = importlib.import_module("sglang_remote_attn.hooks")
        cls.runtime = importlib.import_module("sglang_remote_attn.handoff_control")
        cls.compat = importlib.import_module("sglang_remote_attn.compat")
        cls.check = importlib.import_module("sglang_remote_attn.check")
        cls.Kv = _source_class("managers/schedule_batch.py", "ReqKvInfo")
        cls.Pool = _source_class(
            "mem_cache/memory_pool.py",
            "ReqToTokenPool",
            {
                "alloc",
                "alloc_rows",
                "free",
                "free_rows",
                "available_size",
                "write",
            },
        )
        cls.Allocator = _source_class(
            "mem_cache/allocator/token.py",
            "TokenToKVPoolAllocator",
            {
                "clear",
                "alloc",
                "free",
                "available_size",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "stack"):
            cls.stack.close()
        super().tearDownClass()

    def _pool(self, capacity=2):
        pool = self.Pool()
        pool.free_slots = list(range(1, capacity + 1))
        pool.req_generation = torch.zeros(capacity + 1, dtype=torch.int64)
        pool.req_to_token = torch.full((capacity + 1, 32), -1, dtype=torch.int32)
        pool._on_alloc_rows = None
        pool._aux_cache = None
        return pool

    def _allocator(self):
        allocator = self.Allocator()
        allocator.size = 32
        allocator.device = "cpu"
        allocator.need_sort = False
        allocator.clear()
        return allocator

    def _req(self, mapping=None):
        req = SimpleNamespace(
            rid="imported",
            kv=self.Kv(),
            prefix_len=0,
            extend_end=None,
            origin_input_ids=array("q", [1, 2, 3]),
            output_ids=array("q", [4]),
            full_untruncated_fill_ids=array("q"),
            handoff_prefix_indices=mapping,
            _handoff_prefix_indices=mapping,
        )
        req.refresh_fill_ids = lambda: setattr(
            req, "full_untruncated_fill_ids", req.origin_input_ids + req.output_ids
        )
        return req

    def test_decode_admission_keeps_row_and_kv_ownership(self):
        """Replacing Req.kv after pool.alloc silently loses the newly bound row."""
        pool = self._pool()
        reqs = [self._req(torch.tensor([7, 8, 9])) for _ in range(2)]
        for req in reqs:
            kv = req.kv
            self.compat.bind_decode_prefix(req, pool, req.handoff_prefix_indices, 3)
            self.assertIs(req.kv, kv)
            self.assertEqual(req.kv.kv_committed_len, 3)
            self.assertEqual(req.kv.kv_allocated_len, 3)
            self.assertEqual(req.extend_end, 3)
            self.assertEqual(
                pool.req_to_token[req.kv.req_pool_idx, :3].tolist(), [7, 8, 9]
            )
        self.assertEqual(pool.available_size(), 0)
        ids = pool.alloc(reqs)
        self.assertEqual(len(set(ids)), 2)
        for req in reqs:
            pool.free(req)
        self.assertEqual(pool.available_size(), 2)

    def test_bad_import_cannot_consume_request_row(self):
        pool = self._pool()
        req = self._req(torch.tensor([7]))
        with self.assertRaises(ValueError):
            self.compat.bind_decode_prefix(req, pool, req.handoff_prefix_indices, 3)
        self.assertEqual(pool.available_size(), 2)
        self.assertIsNone(req.kv.req_pool_idx)

    def test_release_before_admission_frees_pending_kv_once(self):
        """KV is allocated at import while row admission is deliberately deferred."""
        allocator = self._allocator()
        pool = self._pool()
        req = self._req(allocator.alloc(3))
        req.kv.kv_committed_len = 3
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=pool,
            waiting_queue=[req],
            running_batch=None,
            last_batch=None,
        )
        self.runtime._free_source_request(scheduler, req)
        self.runtime._free_source_request(scheduler, req)
        self.assertEqual(allocator.available_size(), 32)
        self.assertEqual(torch.unique(allocator.free_pages).numel(), 32)
        self.assertEqual(pool.available_size(), 2)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertIsNone(req.handoff_prefix_indices)

    def test_upstream_release_waits_for_receiver_before_reusing_kv(self):
        allocator = self._allocator()
        req = self._req(allocator.alloc(3))
        req.kv.kv_committed_len = 3
        tree = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        live = [True]
        order = []

        def abort(reason):
            order.append("receiver_stopped")
            live[0] = False

        state = SimpleNamespace(
            req=req,
            handoff_id="imported",
            abort=abort,
            receiver_thread=SimpleNamespace(is_alive=lambda: live[0]),
            notify_source=lambda kind: order.append(kind),
            metrics=lambda: {},
            close=lambda: None,
        )
        with patch.object(self.runtime, "_TARGET_STATES", [state]):
            self.hooks._release_kv(
                lambda *a, **kw: self.fail("no row to release upstream"),
                req,
                tree,
                checkpoint=True,
            )
            self.assertEqual(self.runtime._TARGET_STATES, [])
        self.assertEqual(order, ["receiver_stopped", "done"])
        self.assertEqual(allocator.available_size(), 32)
        self.assertTrue(req.kv.is_kv_released)
        self.assertIsNone(req.handoff_prefix_indices)

    def test_running_receiver_blocks_kv_reuse(self):
        allocator = self._allocator()
        req = self._req(allocator.alloc(3))
        tree = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        state = SimpleNamespace(
            req=req,
            abort=lambda reason: None,
            receiver_thread=SimpleNamespace(is_alive=lambda: True),
        )
        with patch.object(self.runtime, "_TARGET_STATES", [state]):
            with self.assertRaisesRegex(RuntimeError, "receiver did not stop"):
                self.hooks._release_kv(
                    lambda *a, **kw: None, req, tree, checkpoint=True
                )
        self.assertEqual(allocator.available_size(), 29)
        self.assertIsNotNone(req.handoff_prefix_indices)

    def test_prefix_survives_until_successful_extend(self):
        req = self._req(torch.tensor([7, 8, 9]))
        self.hooks._init_round(lambda *a: self.fail("must bypass tree matching"), req)
        self.assertEqual((req.prefix_len, req.extend_end), (3, 4))
        req.extend_end = 3  # a scheduler-chosen zero-token boundary
        prefix = self.hooks._prefix_indices(
            lambda *a: self.fail("unbound prefix has no tree node"), None, req
        )
        self.assertEqual(prefix.tolist(), [7, 8, 9])
        batch = SimpleNamespace(reqs=[req])

        def failed_allocation(_):
            raise RuntimeError("injected allocation failure")

        with self.assertRaises(RuntimeError):
            self.hooks._alloc_extend(failed_allocation, batch)
        self.assertIs(req.handoff_prefix_indices, prefix)
        before = self.runtime.HANDOFF_PREFIX_CLEARED
        self.hooks._alloc_extend(lambda _: (None, None, None), batch)
        self.assertIsNone(req.handoff_prefix_indices)
        self.assertEqual(self.runtime.HANDOFF_PREFIX_CLEARED, before + 1)

    def test_retraction_invalidates_import_mapping(self):
        req = self._req(torch.tensor([7, 8, 9]))
        self.hooks._reset_retracted(lambda r: None, req)
        self.assertIsNone(req.handoff_prefix_indices)
        self.assertIsNone(req._handoff_prefix_indices)

    def test_claimed_command_preserves_the_next_command(self):
        """A newly published cmd.0.json must survive retirement of its predecessor."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(self.runtime, "_CTL_DIR", directory),
        ):
            path = Path(directory) / "cmd.0.json"
            path.write_text(
                json.dumps({"cmd": "freeze", "rid": "one", "nonce": "first"})
            )
            sentinel = self.runtime.poll_broadcast_control()
            path.write_text(json.dumps({"cmd": "status", "nonce": "second"}))
            ranks = [pickle.loads(pickle.dumps(sentinel)) for _ in range(2)]
            for copy in ranks:
                ordinary, controls = self.runtime.split_handoff_controls(
                    ["request", copy]
                )
                self.assertEqual(ordinary, ["request"])
                self.assertEqual(controls[0]["nonce"], "first")
            second = self.runtime.poll_broadcast_control()
            self.assertEqual(second["data"]["nonce"], "second")
            self.assertIsNone(self.runtime.poll_broadcast_control())

    def test_controls_execute_after_regular_input_and_before_housekeeping(self):
        order = []
        runtime = SimpleNamespace(
            split_handoff_controls=self.runtime.split_handoff_controls,
            execute_handoff_control=lambda s, c: order.append(c["cmd"]),
            maybe_handle_handoff_control=lambda s: order.append("housekeeping"),
        )
        sentinel = {self.runtime._HANDOFF_CTL_KEY: True, "data": {"cmd": "freeze"}}

        def process(scheduler, requests):
            order.extend(requests)

        with patch.object(self.hooks, "_runtime", return_value=runtime):
            self.hooks._process_requests(process, None, ["request", sentinel])
            self.hooks._process_requests(process, None, [])
        self.assertEqual(order, ["request", "freeze", "housekeeping", "housekeeping"])

    def test_output_has_one_owner(self):
        ordinary = SimpleNamespace(rid="ordinary")
        migrated = SimpleNamespace(rid="migrated", _handoff_req=True)
        emitted = self.hooks._stream_output(
            lambda s, reqs, *a: [r.rid for r in reqs], None, [ordinary, migrated], False
        )
        self.assertEqual(emitted, ["ordinary"])

    def test_graph_guards_reenable_only_after_remote_layers_finish(self):
        event = threading.Event()
        state = SimpleNamespace(
            mode="remote", events=[event], error=None, aborted=False
        )
        with patch.object(self.runtime, "_TARGET_STATES", [state]):
            self.assertFalse(self.hooks._decode_graph(lambda *a: True, None, None))
            self.assertFalse(self.hooks._prefill_graph(lambda *a: True, None, None))

            event.set()
            self.assertTrue(self.hooks._decode_graph(lambda *a: True, None, None))
            self.assertFalse(self.hooks._prefill_graph(lambda *a: True, None, None))

    def test_failed_import_cannot_become_local_or_replayable(self):
        event = threading.Event()
        event.set()  # the receiver wakes waiters when transport fails
        state = SimpleNamespace(
            mode="remote", events=[event], error="broken pipe", aborted=False
        )
        with (
            patch.object(self.runtime, "_TARGET_STATES", [state]),
            patch.object(self.runtime, "_row_meta", return_value={"states": [state]}),
        ):
            self.assertTrue(self.runtime.has_remote_work())
            with self.assertRaisesRegex(RuntimeError, "broken pipe"):
                self.runtime._handoff_attention_forward(None, None, None, None, None)

    def test_fast_and_cpu_snapshot_wire_layouts(self):
        key = torch.empty(3, 2, 4, dtype=torch.bfloat16)
        fast = self.compat.snapshot_layout(
            3, fast=True, layer_num=2, key=key, kv_cpu=None
        )
        full = [[(key[:1], key[:1]), (key[1:], key[1:])]] * 2
        copied = self.compat.snapshot_layout(3, fast=False, kv_cpu=full)
        self.assertEqual(fast, ([[3], [3]], 2, 4, torch.bfloat16))
        self.assertEqual(copied, ([[1, 2], [1, 2]], 2, 4, torch.bfloat16))

    def _remote_state(self):
        # The real state constructor writes diagnostic paths; isolate that I/O.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(self.runtime, "Path", return_value=Path(directory)),
            patch.object(self.runtime, "PROFILE", False),
            patch.object(self.runtime, "RPC_PROFILE", False),
        ):
            state = self.runtime._TargetHandoffState(
                0, 2, [threading.Event(), threading.Event()], "remote"
            )
        state.client = SimpleNamespace()
        self.addCleanup(state.close)
        return state

    def test_remote_import_failure_keeps_unready_layers_remote(self):
        """LOCAL Drop/EOF must preserve Remote routing instead of killing D1
        or turning error-wakeup events into locally readable KV.
        """
        runtime = self.runtime
        for slot_receiver in (False, True):
            for fault in ("drop", "eof"):
                with self.subTest(slot_receiver=slot_receiver, fault=fault):
                    state = self._remote_state()
                    keys = [torch.full((4, 1, 2), -99.0) for _ in range(2)]
                    values = [torch.full_like(keys[0], -99.0) for _ in range(2)]
                    scheduler = SimpleNamespace(
                        token_to_kv_pool_allocator=SimpleNamespace(
                            get_kvcache=lambda: SimpleNamespace(
                                get_key_buffer=keys.__getitem__,
                                get_value_buffer=values.__getitem__,
                            )
                        )
                    )
                    k = torch.arange(4.0).reshape(2, 1, 2)
                    v = k + 10
                    index = torch.tensor([1, 3])
                    # On EOF the first layer was copied; on Drop it is skipped.
                    messages = (
                        [{"layer": 0, "slot": 0}, EOFError("transfer EOF")]
                        if fault == "eof"
                        else [
                            {"layer": 0, "slot": 0},
                            {"layer": 1, "slot": 0},
                            {"done": True},
                        ]
                    )
                    acknowledgements = []
                    stream = SimpleNamespace(synchronize=lambda: None)
                    cuda = SimpleNamespace(
                        set_device=lambda device: None,
                        stream=lambda stream: nullcontext(),
                    )
                    conn = SimpleNamespace(close=lambda: None)
                    with (
                        patch.object(torch, "cuda", cuda),
                        patch.object(
                            runtime, "_fast_copy_streams", return_value=[stream]
                        ),
                        patch.object(runtime, "_layer_delay_s", return_value=0),
                        patch.object(
                            runtime,
                            "_layer_dropped",
                            side_effect=lambda scheduler, layer: (
                                fault == "drop" and layer == 0
                            ),
                        ),
                        patch.object(runtime, "_recv_json", side_effect=messages),
                        patch.object(
                            runtime,
                            "_send_json",
                            side_effect=lambda conn, message: acknowledgements.append(
                                message
                            ),
                        ),
                        patch.object(runtime, "_TRANSFER_QUARANTINE", {}),
                        patch.object(runtime, "_TRANSFER_ERRORS", {}),
                    ):
                        receiver = (
                            runtime._slot_layer_receiver
                            if slot_receiver
                            else runtime._layer_receiver
                        )
                        buffers = [[(k, v)]] if slot_receiver else [[(k, v)], [(k, v)]]
                        receiver(
                            scheduler,
                            state,
                            conn,
                            {"chunks": [[2], [2]]},
                            buffers,
                            index,
                            "fault",
                            0,
                        )
                        self.assertIn("fault", runtime._TRANSFER_QUARANTINE)
                    expected = [False, True] if fault == "drop" else [True, False]
                    self.assertEqual(
                        [event.is_set() for event in state.events], expected
                    )
                    ready_layer = 1 if fault == "drop" else 0
                    torch.testing.assert_close(keys[ready_layer][index], k)
                    self.assertTrue((keys[1 - ready_layer] == -99).all())
                    self.assertIsNone(state.import_done_at)
                    self.assertIsNotNone(state.error)
                    if slot_receiver:
                        self.assertEqual(
                            len(acknowledgements), 2 if fault == "drop" else 1
                        )
                    state._pending_fallback = True
                    fb = SimpleNamespace(
                        seq_lens=torch.tensor([3]),
                        forward_mode=SimpleNamespace(is_mixed=lambda: False),
                    )
                    state.begin_step(fb)
                    self.assertEqual(state.mode, "remote")
                    self.assertEqual(state._step_mask, expected)
                    self.assertTrue(state.is_remote_layer(1 - ready_layer))
                    self.assertFalse(state.is_remote_layer(ready_layer))
                    with (
                        patch.object(runtime, "_TARGET_STATES", [state]),
                        patch.object(
                            runtime,
                            "_row_meta",
                            return_value={"states": [state], "rp": [0]},
                        ),
                        patch.object(
                            runtime,
                            "_batch_remote_attention",
                            return_value="remote-result",
                        ),
                    ):
                        self.assertTrue(runtime.has_remote_work())
                        module = SimpleNamespace(layer_id=1 - ready_layer)
                        out = runtime._handoff_attention_forward(
                            module, torch.zeros(1, 2), None, None, fb
                        )
                        self.assertEqual(out, "remote-result")
                        module = SimpleNamespace(
                            layer_id=ready_layer,
                            _handoff_orig_forward=lambda *a, **kw: "local-result",
                        )
                        self.assertEqual(
                            runtime._handoff_attention_forward(
                                module, None, None, None, fb
                            ),
                            "local-result",
                        )

    def test_remote_downgrade_still_rejects_abort_missing_client_and_unsafe_copy(self):
        for condition in (
            "aborted",
            "abort-event",
            "no-client",
            "unsafe",
            "layer",
            "copy",
        ):
            with self.subTest(condition=condition):
                state = self._remote_state()
                state.error = "local transfer failed"
                if condition == "aborted":
                    state.aborted = True
                elif condition == "abort-event":
                    state.abort_event.set()
                elif condition == "no-client":
                    state.client = None
                elif condition == "unsafe":
                    state.receiver_unsafe = True
                else:
                    state.mode = condition
                with self.assertRaises(RuntimeError):
                    state.begin_step(SimpleNamespace())

    def test_launchers_default_to_one_native_transfer_batch_per_prefill_chunk(self):
        """N=8 used to split each native PD chunk into synchronous TE calls."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "capture.py"
            capture.write_text(
                "import json,os,sys\nwith open(os.environ['CAPTURE'], 'a') as f: f.write(json.dumps({'args':sys.argv[1:], 'batch':os.environ['SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES']})+'\\n')\n"
            )
            executable = root / "python"
            executable.write_text(
                f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} "$@"\n'
            )
            executable.chmod(0o755)
            for role in ("p", "d0", "d1"):
                for override in (None, "4096"):
                    with self.subTest(role=role, override=override):
                        log = root / "capture.jsonl"
                        log.write_text("")
                        env = dict(
                            os.environ,
                            HANDOFF_PYTHON=str(executable),
                            HANDOFF_CTL_ROOT=str(root / "ctl"),
                            CAPTURE=str(log),
                        )
                        for name in (
                            "SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES",
                            "HANDOFF_CTL_DIR",
                            "HANDOFF_SERVER_EXTRA",
                        ):
                            env.pop(name, None)
                        if override is not None:
                            env["SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES"] = override
                        subprocess.run(
                            [
                                "bash",
                                str(PLUGIN / "deploy" / f"start_p511_{role}_tp2.sh"),
                            ],
                            env=env,
                            check=True,
                        )
                        preflight, server = map(
                            json.loads, log.read_text().splitlines()
                        )
                        self.assertEqual(
                            preflight["args"], ["-m", "sglang_remote_attn.check"]
                        )
                        args = server["args"]
                        chunk = int(args[args.index("--chunked-prefill-size") + 1])
                        self.assertGreaterEqual(int(server["batch"]), chunk)
                        self.assertEqual(server["batch"], override or "2048")

    def test_early_layer_done_wakes_waiters_with_error(self):
        runtime = self.runtime
        state = SimpleNamespace(
            events=[threading.Event(), threading.Event()],
            error=None,
            import_done_at=None,
            abort_event=threading.Event(),
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None)
        )
        stream = SimpleNamespace(synchronize=lambda: None)
        cuda = SimpleNamespace(set_device=lambda device: None)
        conn = SimpleNamespace(close=lambda: None)
        with (
            patch.object(torch, "cuda", cuda),
            patch.object(runtime, "_fast_copy_streams", return_value=[stream]),
            patch.object(runtime, "_layer_delay_s", return_value=0),
            patch.object(runtime, "_recv_json", return_value={"done": True}),
            patch.object(runtime, "_TRANSFER_QUARANTINE", {}),
        ):
            runtime._layer_receiver(
                scheduler, state, conn, {}, [[], []], None, "early", 1
            )
            self.assertIn("early", runtime._TRANSFER_QUARANTINE)
        self.assertIn("incomplete", state.error)
        self.assertIsNone(state.import_done_at)
        self.assertTrue(all(event.is_set() for event in state.events))
        with self.assertRaises(RuntimeError):
            runtime._check_target_state(state)

    def test_full_copy_failure_after_gpu_load_rolls_back_allocation(self):
        allocator = self._allocator()
        allocator.load_cpu_copy = lambda *args: None  # GPU copy boundary
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator, tokenizer=None
        )
        meta = {
            "origin_input_ids": [1, 2, 3],
            "output_ids": [4],
            "kv_committed_len": 3,
            "sampling": {"max_new_tokens": 8},
        }
        with (
            patch.object(
                self.runtime, "_recv_transfer_mooncake", return_value=(meta, [])
            ),
            patch.object(
                self.runtime, "_sampling_from", side_effect=ValueError("bad sampling")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "bad sampling"):
                self.runtime._full_import_one(
                    scheduler, {}, {"mode": "mooncake"}, "imported"
                )
        self.assertEqual(allocator.available_size(), 32)
        self.assertEqual(torch.unique(allocator.free_pages).numel(), 32)

    def test_copy_target_keeps_registered_buffers_until_positive_completion(self):
        class Conn:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def settimeout(self, *args):
                pass

            def setsockopt(self, *args):
                pass

        meta = {
            "dtype": "bfloat16",
            "kv_heads": 1,
            "head_dim": 2,
            "chunks": [[3]],
            "kv_committed_len": 3,
        }
        for ack in ({"ok": True}, EOFError("connection dropped")):
            with self.subTest(ack=repr(ack)), ExitStack() as stack:
                unregistered = []
                manager = SimpleNamespace(
                    register=lambda *a, **kw: None, unregister=unregistered.append
                )
                for name, value in {
                    "_get_te": lambda: SimpleNamespace(get_rpc_port=lambda: 123),
                    "_get_registration_manager": lambda manager=manager: manager,
                    "_send_json": lambda *a: None,
                    "_TRANSFER_QUARANTINE": {},
                    "_TRANSFER_ERRORS": {},
                }.items():
                    stack.enter_context(patch.object(self.runtime, name, value))
                stack.enter_context(
                    patch.object(
                        self.runtime.socket, "create_connection", return_value=Conn()
                    )
                )
                stack.enter_context(
                    patch.object(self.runtime, "_recv_json", side_effect=[meta, ack])
                )
                transfer = {"host": "localhost", "port": 123, "token": "test"}
                if isinstance(ack, Exception):
                    with self.assertRaises(EOFError):
                        self.runtime._recv_transfer_mooncake(transfer)
                    self.assertEqual(unregistered, [])
                    buffers, reason = self.runtime._TRANSFER_QUARANTINE[
                        "target-mooncake:test"
                    ]
                    self.assertEqual(buffers[0][0][0].numel(), 6)
                    self.assertIn("completion", reason)
                else:
                    self.runtime._recv_transfer_mooncake(transfer)
                    self.assertEqual(unregistered, ["target-mooncake:test"])
                    self.assertEqual(self.runtime._TRANSFER_QUARANTINE, {})

    def test_clean_completion_is_not_reported_as_aborted(self):
        req = self._req()
        notifications = []
        state = SimpleNamespace(
            req=req,
            error=None,
            aborted=False,
            import_done_at=1.0,
            receiver_conn=None,
            receiver_thread=None,
            notify_source=notifications.append,
            handoff_id="clean",
            metrics=lambda: {},
            close=lambda: None,
            abort=lambda reason: self.fail(
                "clean completion must not be marked aborted"
            ),
        )
        with patch.object(self.runtime, "_TARGET_STATES", [state]):
            self.runtime.stop_target_receivers(req)
            self.assertEqual(self.runtime._TARGET_STATES, [])
        self.assertEqual(notifications, ["done"])
        self.assertIsNone(state.error)
        self.assertFalse(state.aborted)

    def test_live_source_reader_prevents_freeing_frozen_kv(self):
        allocator = self._allocator()
        req = self._req(allocator.alloc(3))
        req._handoff_transfer_thread = SimpleNamespace(
            join=lambda timeout: None, is_alive=lambda: True
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator, req_to_token_pool=self._pool()
        )
        with self.assertRaisesRegex(RuntimeError, "source transfer did not stop"):
            self.runtime._free_source_request(scheduler, req)
        self.assertEqual(allocator.available_size(), 29)
        self.assertFalse(getattr(req, "_handoff_freed", False))

    def test_two_transfer_lifetimes_do_not_clear_the_shared_active_flag(self):
        scheduler = SimpleNamespace()
        mark = self.runtime._mark_source_transfer
        mark(scheduler, "one", True)
        mark(scheduler, "two", True)
        mark(scheduler, "one", False)
        mark(scheduler, "one", False)  # idempotent exception cleanup
        self.assertTrue(scheduler._handoff_transfer_active)
        mark(scheduler, "two", False)
        self.assertFalse(scheduler._handoff_transfer_active)

    def test_tp2_broadcast_and_command_failure_agreement(self):
        worker = ROOT / "test/manual/remote_attn/tp_control_worker.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(
                    [str(PLUGIN), os.environ.get("PYTHONPATH", "")]
                ),
            )
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        str(worker),
                        str(rank),
                        str(root / "rendezvous"),
                        str(root / f"rank{rank}.json"),
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for rank in range(2)
            ]
            try:
                outputs = [process.communicate(timeout=25)[0] for process in processes]
                diagnostics = "\n".join(outputs)
                for process in processes:
                    self.assertEqual(process.returncode, 0, diagnostics)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        process.communicate(timeout=5)
                    elif process.stdout is not None and not process.stdout.closed:
                        process.stdout.close()
            rows = [
                json.loads((root / f"rank{rank}.json").read_text()) for rank in range(2)
            ]
        self.assertEqual(rows[0]["requests"], rows[1]["requests"])
        self.assertEqual(rows[0]["requests"][::2], ["request", "timeout-abort"])
        self.assertEqual(rows[0]["polls"], [0])
        self.assertEqual(rows[1]["polls"], [])
        self.assertEqual(rows[0]["outcomes"], rows[1]["outcomes"])
        self.assertEqual(
            [r["fatal"] for r in rows[0]["outcomes"]], [False, True, True, False]
        )

    def test_fast_sender_uses_rank_device_and_flushes_controls_before_slot_wait(self):
        """Three source buffers / one target slot used to form an ACK cycle."""
        self._fast_sender_contract()

    def test_fast_sender_retains_staging_if_submission_completion_is_unknown(self):
        for failure in ("sync", "async"):
            with self.subTest(failure=failure):
                self._fast_sender_contract(failure)

    def _fast_sender_contract(self, failure=None):
        device = threading.local()
        device.value = 1
        tensors = {}
        copied = []
        completed = []
        controls = []
        messages = queue.Queue()
        messages.put({"token": "regression"})
        messages.put({"session": "peer", "addrs": [100, 200], "slots": 1})

        class Stream:
            def wait_event(self, event):
                copied.append(("wait", device.value))

            def synchronize(self):
                copied.append(("sync", device.value))

        class Event:
            def record(self, stream=None):
                copied.append(("record", device.value))

            def synchronize(self):
                copied.append(("event_sync", device.value))

        class Conn:
            def settimeout(self, *args):
                pass

            def setsockopt(self, *args):
                pass

            def shutdown(self, *args):
                messages.put(None)

            def close(self):
                pass

        conn = Conn()
        listener = SimpleNamespace(accept=lambda: (conn, None), close=lambda: None)

        def receive(conn):
            message = messages.get(timeout=3)
            if message is None:
                raise EOFError()
            return message

        def send(conn, message):
            if "layer" in message:
                controls.append(message["layer"])
                messages.put({"free": message["slot"]})

        def empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            tensor = original_empty(*args, **kwargs)
            tensors[tensor.data_ptr()] = tensor
            return tensor

        class Engine:
            def __init__(self):
                self.next = 0
                self.pending = {}

            def batch_transfer_async_write(self, session, addresses, peer, sizes):
                if failure == "async":
                    raise RuntimeError("submission completion unknown")
                self.next += 1
                self.pending[self.next] = [
                    (tensors[a], tensors[a].clone()) for a in addresses
                ]
                return self.next

            def batch_transfer_sync_write(self, *args):
                return -1

            def get_batch_transfer_status(self, batches):
                for batch in batches:
                    for tensor, snapshot in self.pending.pop(batch):
                        if not torch.equal(tensor, snapshot):
                            raise AssertionError(
                                "source staging reused before TE completion"
                            )
                    completed.append(batch)
                return 0

        engine = Engine()
        fake_cuda = SimpleNamespace(
            current_device=lambda: getattr(device, "value", 0),
            set_device=lambda value: setattr(device, "value", value),
            Event=Event,
            current_stream=lambda *args: Stream(),
            stream=lambda stream: nullcontext(),
        )

        def streams(owner, count):
            self.assertEqual(owner, 1)
            self.assertEqual(device.value, 1)
            return [Stream() for _ in range(count)]

        pool = self._pool()
        req = self._req()
        pool.alloc([req])
        pool.req_to_token[req.kv.req_pool_idx, :3] = torch.tensor([1, 2, 3])
        keys = [
            torch.arange(64, dtype=torch.float32).reshape(32, 1, 2) + i
            for i in range(5)
        ]
        kv = SimpleNamespace(
            layer_num=5,
            get_key_buffer=lambda layer: keys[layer],
            get_value_buffer=lambda layer: keys[layer],
        )
        scheduler = SimpleNamespace(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: kv),
        )
        unregistered = []
        manager = SimpleNamespace(
            register=lambda *a, **kw: None, unregister=unregistered.append
        )
        original_empty = torch.empty
        with ExitStack() as stack:
            for name, value in {
                "FAST_BUFFERS": 3,
                "FAST_COPY_STREAMS": 2,
                "FAST_ASYNC": failure != "sync",
                "_get_te": lambda: engine,
                "_get_registration_manager": lambda: manager,
                "_open_transfer_listener": lambda: (listener, 123),
                "_recv_json": receive,
                "_send_json": send,
                "_fast_copy_streams": streams,
                "_TRANSFER_ERRORS": {},
                "_TRANSFER_QUARANTINE": {},
            }.items():
                stack.enter_context(patch.object(self.runtime, name, value))
            stack.enter_context(patch.object(torch, "cuda", fake_cuda))
            stack.enter_context(patch.object(torch, "empty", empty))
            stack.enter_context(
                patch.object(
                    self.runtime.secrets, "token_hex", return_value="regression"
                )
            )
            self.runtime._start_layer_gpu_server(
                scheduler, req, {"kv_committed_len": 3, "chunks": [[3]] * 5}, False
            )
            try:
                req._handoff_transfer_thread.join(timeout=4)
                self.assertFalse(
                    req._handoff_transfer_thread.is_alive(), "target-slot ACK cycle"
                )
                if failure is None:
                    self.assertEqual(self.runtime._TRANSFER_ERRORS, {})
                else:
                    self.assertIn("regression", self.runtime._TRANSFER_QUARANTINE)
            finally:
                self.runtime.stop_source_transfer(req)
        if failure is None:
            self.assertEqual(controls, list(range(5)))
            self.assertEqual(completed, list(range(1, 6)))
            self.assertEqual(unregistered, ["regression"])
        else:
            self.assertEqual(controls, [])
            self.assertEqual(unregistered, [])
        self.assertTrue(all(owner == 1 for _, owner in copied))

    def test_rpc_completion_fences_only_its_stream_and_retains_unsafe_kv(self):
        runtime = self.runtime
        for batch in (False, True):
            for failed in (False, True):
                with self.subTest(batch=batch, failed=failed):
                    req = self._req()
                    ctx = SimpleNamespace(req=req, handoff_id="rpc-source")
                    calls = []

                    def fence(calls=calls, failed=failed):
                        calls.append("stream-fence")
                        if failed:
                            raise RuntimeError("CUDA reader failed")

                    hub = SimpleNamespace(
                        registry=SimpleNamespace(
                            release_ref=lambda *args, calls=calls: calls.append(
                                "release"
                            )
                        ),
                        _job_finished=lambda *args, calls=calls: calls.append(
                            "finished"
                        ),
                    )
                    if batch:
                        item = SimpleNamespace(ctx=ctx, epoch=1)
                        job = runtime.RemoteBatchJob(hub, None, None, None, [item], 1)
                    else:
                        job = runtime.RemoteJob(
                            hub, None, None, None, ctx, 1, 1, {}, None
                        )
                    job.stream = SimpleNamespace(synchronize=fence)
                    with patch.object(
                        torch.cuda,
                        "synchronize",
                        side_effect=AssertionError("device fence"),
                    ):
                        job.finish()
                        job.finish()
                    self.assertEqual(
                        calls,
                        ["stream-fence"]
                        if failed
                        else ["stream-fence", "release", "finished"],
                    )
                    self.assertEqual(
                        bool(getattr(req, "_handoff_transfer_unsafe", False)), failed
                    )
        runtime._TRANSFER_ERRORS.pop("rpc-source", None)

    def test_normal_priority_hub_still_uses_dedicated_streams(self):
        runtime = self.runtime
        streams = [object(), object()]
        with (
            patch.dict(os.environ, {"HANDOFF_REMOTE_HIGH_PRIORITY": "0"}),
            patch.object(runtime, "_HUB_STREAM_CACHE", {}),
            patch.object(torch.cuda, "current_device", return_value=1),
            patch.object(torch.cuda, "Stream", side_effect=streams) as make_stream,
            patch.dict(sys.modules, {"cuda.bindings": None}),
        ):
            slots = runtime._create_hub_streams(2)
        self.assertEqual([slot[0] for slot in slots], streams)
        self.assertEqual(make_stream.call_count, 2)
        self.assertEqual(make_stream.call_args.kwargs, {"priority": 0})

    def test_scheduler_exit_fences_all_workers_even_after_cleanup_failure(self):
        runtime = self.runtime
        scheduler = SimpleNamespace(handoff_tracked_reqs={"r": self._req()})
        calls = []

        def close_hub(_scheduler):
            calls.append("hub")
            raise RuntimeError("live RPC reader")

        with (
            patch.object(runtime, "close_handoff_hub", side_effect=close_hub),
            patch.object(
                runtime,
                "_reset_target_states",
                side_effect=lambda: calls.append("target"),
            ),
            patch.object(
                runtime,
                "stop_source_transfer",
                side_effect=lambda req: calls.append("source"),
            ),
            patch.object(self.hooks, "_runtime", return_value=runtime),
        ):
            with self.assertRaisesRegex(RuntimeError, "live RPC reader"):
                self.hooks._event_loop(
                    lambda scheduler: calls.append("loop"), scheduler
                )
        self.assertEqual(calls, ["loop", "hub", "target", "source"])

    def test_peer_repair_is_probed_before_publication(self):
        old = SimpleNamespace(dst_kv_ptrs=[10, 20], dst_kv_item_len=8)
        new = SimpleNamespace(dst_kv_ptrs=[10, 21], dst_kv_item_len=8)
        events = []
        responses = iter([-1, 0])

        def probe(session):
            events.append("probe")
            return next(responses)

        manager = SimpleNamespace(
            _remote_attn_send_lock=threading.Lock(),
            session_lock=threading.Lock(),
            decode_kv_args_table={"peer": old},
            failed_sessions=set(),
            engine=SimpleNamespace(send_probe=probe),
        )

        def publish(manager, session, peer):
            events.append("publish")
            manager.decode_kv_args_table[session] = peer

        self.hooks._publish_peer(publish, manager, "peer", new)
        self.hooks._publish_peer(publish, manager, "peer", new)
        self.assertEqual(events, ["probe", "probe", "publish", "publish"])
        manager.engine.send_probe = lambda s: -1
        self.hooks._publish_peer(publish, manager, "other", old)
        self.assertIn("other", manager.failed_sessions)
        self.assertIs(manager.decode_kv_args_table["other"], old)

    def test_transfer_batches_share_one_process_lock(self):
        """Force two peers to overlap at admission; only one may reach TE."""
        from sglang.srt.environ import envs

        entered = threading.Event()
        second_attempt = threading.Event()
        release = threading.Event()
        calls = []
        errors = []

        class ObservedLock:
            def __init__(self):
                self.lock = threading.Lock()
                self.attempts = 0
                self.guard = threading.Lock()

            def __enter__(self):
                with self.guard:
                    self.attempts += 1
                    if self.attempts == 2:
                        second_attempt.set()
                self.lock.acquire()

            def __exit__(self, *args):
                self.lock.release()

        manager = SimpleNamespace(_remote_attn_send_lock=ObservedLock())

        def send(manager, peer, blocks):
            calls.append(peer)
            if peer == "first":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test failed to release first send")
            return 0

        def worker(peer):
            try:
                self.hooks._transfer(send, manager, peer, [(1, 2, 3)])
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(target=worker, args=("first",))
        second = threading.Thread(target=worker, args=("second",))
        with envs.SGLANG_MOONCAKE_SEND_LOCK.override(True):
            try:
                first.start()
                self.assertTrue(entered.wait(5))
                second.start()
                self.assertTrue(second_attempt.wait(5))
                self.assertEqual(calls, ["first"])
            finally:
                release.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(calls, ["first", "second"])

    def test_hook_targets_match_current_main(self):
        self.check.check_sources(ROOT)


if __name__ == "__main__":
    unittest.main()
