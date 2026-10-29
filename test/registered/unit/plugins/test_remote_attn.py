"""CPU contracts for the optional plugin; GPU/model imports stay at the boundary.

Source-loaded pool methods are the real upstream implementations. This avoids
loading Triton/CUDA just to exercise row ownership and token-map lifetimes.
"""

import ast
import copy
import ctypes
import dataclasses
import functools
import gc
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
import time
import unittest
import uuid
import weakref
from array import array
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
PLUGIN = ROOT / "plugins" / "remote_attn"


class _HostStagingSlot:
    """CUDA-copy boundary stand-in; copy real bytes for address-plan tests."""

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.storage = ctypes.create_string_buffer(capacity)
        self.ptr = ctypes.addressof(self.storage)
        self.closed = False

    def copy(self, blocks, ready_event):
        if not ready_event.wait(3):
            raise TimeoutError("KV producer did not finish")
        offset = 0
        for src, _, size in blocks:
            ctypes.memmove(self.ptr + offset, src, size)
            offset += size

    def copy_to_device(self, blocks):
        self.h2d_blocks = blocks
        offset = 0
        for dst, size in blocks:
            ctypes.memmove(dst, self.ptr + offset, size)
            offset += size

    def close(self):
        self.closed = True


def _host_staging_manager():
    registered = {}

    def register(ptr, size):
        registered[ptr] = size

    def deregister(ptr):
        del registered[ptr]

    return SimpleNamespace(
        engine=SimpleNamespace(register=register, deregister=deregister),
        _remote_attn_send_lock=threading.Lock(),
        registrations=registered,
    )


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


def _source_function(relative, name):
    path = ROOT / "python" / "sglang" / "srt" / relative
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


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

    def test_deferred_release_after_plugin_cleanup_is_idempotent(self):
        # Run the actual upstream entry point used by decode's deferred releases.
        release = _source_function("mem_cache/common.py", "release_kv_cache")
        for cleanup in ("source", "pending_import"):
            with self.subTest(cleanup=cleanup):
                allocator = self._allocator()
                pool = self._pool()
                req = self._req(allocator.alloc(3))
                req.kv.kv_committed_len = 3
                tree = SimpleNamespace(
                    token_to_kv_pool_allocator=allocator,
                    supports_mamba=lambda: False,
                )
                # A pending import has no row yet, but still owns prefix slots.
                self.assertTrue(req.kv.is_kv_released)
                if cleanup == "source":
                    scheduler = SimpleNamespace(
                        token_to_kv_pool_allocator=allocator,
                        req_to_token_pool=pool,
                        waiting_queue=[req],
                        running_batch=None,
                        last_batch=None,
                    )
                    self.runtime._free_source_request(scheduler, req)
                else:
                    self.hooks._release_kv(release, req, tree, checkpoint=True)
                self.assertEqual(allocator.available_size(), 32)
                self.assertIsNone(req.handoff_prefix_indices)
                # New requests can reuse those slots before deferred cleanup runs.
                in_use = allocator.alloc(5)
                for checkpoint in (False, True):
                    self.hooks._release_kv(release, req, tree, checkpoint=checkpoint)
                self.assertEqual(allocator.available_size(), 27)
                free = torch.cat((allocator.free_pages, allocator.release_pages))
                self.assertFalse(torch.isin(in_use, free).any())
                self.assertEqual(torch.unique(free).numel(), 27)
                self.assertEqual(pool.available_size(), 2)

    def test_release_without_kv_still_frees_independent_mamba_state(self):
        release = _source_function("mem_cache/common.py", "release_kv_cache")
        req = self._req()
        req.kv.mamba_pool_idx = torch.tensor([7])
        freed = []
        tree = SimpleNamespace(
            supports_mamba=lambda: True,
            req_to_token_pool=SimpleNamespace(
                mamba_allocator=SimpleNamespace(free=freed.append)
            ),
        )
        self.hooks._release_kv(release, req, tree, checkpoint=False)
        self.hooks._release_kv(release, req, tree, checkpoint=False)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(len(freed), 1)
        self.assertEqual(freed[0].tolist(), [[7]])

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
                    released = []
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
                        patch.object(
                            runtime,
                            "_get_registration_manager",
                            return_value=SimpleNamespace(
                                release_safe=lambda owner: (
                                    released.append(owner) or True
                                )
                            ),
                        ),
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
                        if fault == "drop":
                            self.assertEqual(released, ["fault"])
                            self.assertNotIn("fault", runtime._TRANSFER_QUARANTINE)
                        else:
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

    def test_c1_cutover_reuses_drain_reservation_epoch(self):
        """Pre-reserve and C1 cutover must reuse one allocation/epoch."""
        path = PLUGIN / "control_plane/p4_gateway.py"
        tree = ast.parse(path.read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            or isinstance(n, ast.ClassDef)
            and n.name in ("Gateway", "RequestState")
        ]
        namespace = dict(
            os=os,
            threading=threading,
            functools=functools,
            time=time,
            uuid=uuid,
            copy=copy,
            Path=Path,
            json=json,
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        gateway = namespace["Gateway"].__new__(namespace["Gateway"])
        gateway.migration_lock = threading.RLock()
        gateway.journal_lock = threading.Lock()
        gateway.admission_lock = threading.Lock()
        st = namespace["RequestState"]("c1", 2, 4, 1)
        gateway.journal = {st.rid: {"drid": st.drid, "prompt_len": 2, "max_new": 4}}
        allocator = self._allocator()
        allocator.size = 2048
        allocator.clear()
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=self._pool(),
            max_running_requests=2,
        )
        requests = []

        def send(payload, **kwargs):
            if payload["cmd"] == "status":
                return {"max_running": 2, "reqs": []}
            requests.append(payload)
            return self.runtime._handle(scheduler, payload)

        gateway.dst = SimpleNamespace(send=send)

        class FreezeReached(Exception):
            pass

        def timed(role, payload, *args):
            if payload["cmd"] == "freeze":
                self.assertEqual(payload["epoch"], "drain-epoch")
                raise FreezeReached()
            return role.send(payload)

        gateway._send_timed = timed
        gateway.src = SimpleNamespace()
        gateway._maybe_kill = lambda stage: None
        context = SimpleNamespace(
            get_schedule=lambda: SimpleNamespace(max_running_requests=2)
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"sglang.srt.runtime_context": context}),
        ):
            gateway.journal_path = Path(directory) / "journal.json"
            drain = {"epoch": "drain-epoch", "state": "REQUESTED", "members": [st.rid]}
            self.assertTrue(gateway._pre_reserve_drain(drain, [st.rid]))
            gateway.journal["__drain__"] = drain
            before = allocator.available_size()
            reservation = scheduler._handoff_reserved[st.drid]["indices"]
            with self.assertRaises(FreezeReached):
                gateway._cutover(None, st, "remote", "none")
            self.assertEqual(allocator.available_size(), before)
            self.assertIs(scheduler._handoff_reserved[st.drid]["indices"], reservation)
            self.assertEqual({r["epoch"] for r in requests}, {"drain-epoch"})
            # A genuinely different epoch still cannot steal the reservation.
            rejected = send(
                {
                    "cmd": "reserve",
                    "epoch": "stale",
                    "items": [{"rid": st.drid, "tokens": 10}],
                }
            )
            self.assertIn(st.drid, rejected["errors"])
            # Drain must wait for an already-started cutover before taking its
            # snapshot, while ordinary admission remains available.
            frozen = threading.Event()
            continue_freeze = threading.Event()
            drain_waiting = threading.Event()
            order = []
            errors = []

            class MigrationGate:
                def __init__(self):
                    self.lock = threading.RLock()

                def __enter__(self):
                    if threading.current_thread().name == "drain-race":
                        drain_waiting.set()
                    self.lock.acquire()

                def __exit__(self, *args):
                    self.lock.release()

            gateway.migration_lock = MigrationGate()

            def freeze_gate(role, payload, *args):
                if payload["cmd"] == "freeze":
                    frozen.set()
                    if not continue_freeze.wait(timeout=3):
                        raise TimeoutError("test gate")
                    order.append("freeze-exit")
                    raise FreezeReached()
                return role.send(payload)

            gateway._send_timed = freeze_gate
            gateway.drain_status = lambda: (
                order.append("drain-snapshot") or {"state": "REQUESTED"}
            )

            def cutover():
                try:
                    gateway._cutover(None, st, "remote", "none")
                except FreezeReached:
                    pass
                except Exception as exc:
                    errors.append(exc)

            def drain_call():
                try:
                    gateway.start_drain(["dst"])
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=cutover),
                threading.Thread(target=drain_call, name="drain-race"),
            ]
            try:
                threads[0].start()
                self.assertTrue(frozen.wait(timeout=3))
                threads[1].start()
                self.assertTrue(drain_waiting.wait(timeout=3))
                self.assertTrue(gateway.admission_lock.acquire(blocking=False))
                gateway.admission_lock.release()
                self.assertEqual(order, [])
            finally:
                continue_freeze.set()
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(timeout=3)
            self.assertEqual(errors, [])
            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(order, ["freeze-exit", "drain-snapshot"])

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

    def test_launchers_preserve_native_batching_and_retraction_configuration(self):
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
                            "HANDOFF_RETRACTION_BACKUP",
                        ):
                            env.pop(name, None)
                        if override is not None:
                            env["SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES"] = override
                            env["HANDOFF_RETRACTION_BACKUP"] = "cpu_tensor"
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
                        backup_flag = "--disaggregation-decode-retraction-backup"
                        if role == "p":
                            self.assertNotIn(backup_flag, args)
                        else:
                            self.assertEqual(args.count(backup_flag), 1)
                            self.assertEqual(
                                args[args.index(backup_flag) + 1],
                                "cpu_tensor" if override else "none",
                            )

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
            self.assertNotIn("early", runtime._TRANSFER_QUARANTINE)
        self.assertIn("incomplete", state.error)
        self.assertIsNone(state.import_done_at)
        self.assertTrue(all(event.is_set() for event in state.events))
        with self.assertRaises(RuntimeError):
            runtime._check_target_state(state)

    def test_quarantined_host_buffer_waits_for_matching_sender_and_gpu_fences(self):
        rt = self.runtime
        for gpu_safe in (False, True):
            with (
                self.subTest(gpu_safe=gpu_safe),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                for name in (
                    "_TRANSFER_TICKETS",
                    "_TRANSFER_RELEASES",
                    "_TRANSFER_QUARANTINE",
                    "_TRANSFER_ERRORS",
                ):
                    stack.enter_context(patch.object(rt, name, {}))
                stack.enter_context(
                    patch.object(rt, "_CTL_DIR", str(Path(directory) / "src"))
                )
                token = "a" * 32
                meta = rt._with_transfer_ticket(token, {})
                rt._CTL_DIR = str(Path(directory) / "dst")
                owner = "target:" + token
                rt._bind_transfer_ticket(owner, token, meta)
                released = []
                stack.enter_context(
                    patch.object(
                        rt,
                        "_get_registration_manager",
                        return_value=SimpleNamespace(
                            release_safe=lambda who: released.append(who) or True
                        ),
                    )
                )
                tensor = torch.zeros(4)
                rt._release_transfer_buffers(owner, tensor, "EOF", gpu_safe=gpu_safe)
                rt.reap_transfer_buffers()
                self.assertEqual(released, [])
                path = Path(meta["host_completion"]["path"])
                path.write_text("[]")
                rt.reap_transfer_buffers()
                self.assertEqual(released, [])
                path.write_text(json.dumps({"token": "stale", "complete": True}))
                rt.reap_transfer_buffers()
                self.assertEqual(released, [])
                rt._publish_transfer_complete(token, True)
                rt.reap_transfer_buffers()
                self.assertEqual(released, [owner] if gpu_safe else [])
                self.assertEqual(owner in rt._TRANSFER_QUARANTINE, not gpu_safe)
                if gpu_safe:
                    self.assertFalse(path.exists())
                    self.assertFalse(path.with_suffix(".ack").exists())

    def test_native_unregister_failure_keeps_storage_until_retry(self):
        rt = self.runtime
        returns = iter([-1, 0])
        engine = SimpleNamespace(
            batch_register_memory=lambda *args: 0,
            batch_unregister_memory=lambda *args: next(returns),
        )
        manager = rt.RegistrationManager(engine)
        tensor = torch.zeros(4)
        pointer = weakref.ref(tensor)
        manager.register(
            "failed", [tensor.data_ptr()], [tensor.numel() * tensor.element_size()]
        )
        with (
            patch.object(rt, "_TRANSFER_QUARANTINE", {}),
            patch.object(rt, "_TRANSFER_RELEASES", {}),
            patch.object(rt, "_TRANSFER_TICKETS", {}),
            patch.object(rt, "_TRANSFER_ERRORS", {}),
            patch.object(rt, "_get_registration_manager", return_value=manager),
        ):
            self.assertFalse(
                rt._release_transfer_buffers(
                    "failed", tensor, "retry", remote_complete=True
                )
            )
            del tensor
            gc.collect()
            self.assertIsNotNone(pointer())
            self.assertEqual(manager.stats()["stale"], 1)
            rt.reap_transfer_buffers()
            gc.collect()
            self.assertIsNone(pointer())
            self.assertEqual(manager.stats()["stale"], 0)
            self.assertEqual(manager.stats()["live"], 0)
            self.assertEqual(rt._TRANSFER_QUARANTINE, {})

    def test_completion_proof_cleans_up_in_both_ack_orders(self):
        rt = self.runtime
        for consumer_first in (False, True):
            with (
                self.subTest(consumer_first=consumer_first),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                for name in (
                    "_TRANSFER_TICKETS",
                    "_TRANSFER_RELEASES",
                    "_TRANSFER_QUARANTINE",
                    "_TRANSFER_ERRORS",
                ):
                    stack.enter_context(patch.object(rt, name, {}))
                stack.enter_context(
                    patch.object(rt, "_CTL_DIR", str(Path(directory) / "src"))
                )
                token = "b" * 32
                meta = rt._with_transfer_ticket(token, {})
                rt._CTL_DIR = str(Path(directory) / "dst")
                rt._bind_transfer_ticket("target", token, meta)
                stack.enter_context(
                    patch.object(
                        rt,
                        "_get_registration_manager",
                        return_value=SimpleNamespace(release_safe=lambda owner: True),
                    )
                )
                if not consumer_first:
                    rt._publish_transfer_complete(token, True)
                rt._release_transfer_buffers(
                    "target", torch.zeros(4), "done", remote_complete=True
                )
                if consumer_first:
                    rt._publish_transfer_complete(token, True)
                self.assertEqual(
                    list((Path(directory) / "src" / "transfer-completion").iterdir()),
                    [],
                )

    def test_undersized_reservation_returns_allocation_once(self):
        allocator = self._allocator()
        index = allocator.alloc(4)
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            _handoff_reserved={"small": {"indices": index}},
        )
        with self.assertRaises(RuntimeError):
            self.runtime._take_reserved(scheduler, "small", 8)
        self.assertEqual(allocator.available_size(), 32)
        self.assertEqual(torch.unique(allocator.free_pages).numel(), 32)
        self.assertEqual(scheduler._handoff_reserved, {})

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
                    register=lambda *a, **kw: None,
                    unregister=unregistered.append,
                    release_safe=lambda owner: unregistered.append(owner) or True,
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
        self.assertEqual(rows[0]["retirements"], [[], ["a"], ["a", "b"]])
        self.assertEqual(rows[0]["retirements"], rows[1]["retirements"])
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
            register=lambda *a, **kw: None,
            unregister=unregistered.append,
            release_safe=lambda owner: unregistered.append(owner) or True,
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

    def test_prelayer_coverage_rejects_holes_and_stale_generations(self):
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            state = SimpleNamespace(
                events=[threading.Event(), threading.Event()],
                layer_ready_at=[None, None],
                import_done_at=None,
            )
            target = pl.Target(
                None, None, None, path, "current", pl.LayerCoverage(9, 2), state
            )

            def proof(layer, start, end, generation="current"):
                pl.write_json(
                    path / f"layer-{layer}.json",
                    dict(generation=generation, layer=layer, start=start, end=end),
                )

            proof(0, 0, 9, "old")
            proof(1, 0, 4)
            target.refresh()
            self.assertFalse(any(e.is_set() for e in state.events))
            proof(0, 0, 9)
            proof(1, 5, 9)
            target.refresh()
            self.assertTrue(state.events[0].is_set())
            self.assertFalse(state.events[1].is_set())
            self.assertIsNone(state.import_done_at)
            proof(1, 4, 5)
            target.refresh()
            self.assertTrue(all(e.is_set() for e in state.events))
            self.assertIsNotNone(state.import_done_at)
            with self.assertRaises(ValueError):
                target.coverage.complete(2, 0, 9)
            with self.assertRaises(ValueError):
                target.coverage.complete(0, 0, 10)

    def test_prelayer_selects_matching_kv_buffers_without_mutating_pool(self):
        pl = self.hooks.pre_layerwise
        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py", "MooncakeKVManager", {"send_kvcache"}
        )
        manager = manager_cls()
        manager.is_mla_backend = manager.is_hybrid_mla_backend = False
        manager.pp_size = manager.attn_tp_size = 1
        manager.kv_args = SimpleNamespace(
            kv_layer_ids=[], kv_data_ptrs=[10, 20, 30, 40, 50, 60], kv_item_lens=[8] * 6
        )
        manager._validate_envelope_kv_layout = lambda *args: None
        manager._send_kvcache_generic = lambda **kw: kw
        dst = [110, 120, 130, 140, 150, 160]
        groups = pl.buffer_groups([0, 1, 2, 0, 1, 2], 3, 2)
        self.assertEqual(groups, [((0, 1), (0, 1, 3, 4)), ((2,), (2, 5))])
        result = manager.send_kvcache(
            "peer",
            [3, 8],
            dst,
            [1, 9],
            None,
            dst_attn_tp_size=1,
            kv_buffer_indices=groups[1][1],
        )
        self.assertEqual(result["src_data_ptrs"], [30, 60])
        self.assertEqual(result["dst_data_ptrs"], [130, 160])
        self.assertEqual(result["prefill_data_indices"], [3, 8])
        self.assertEqual(manager.kv_args.kv_data_ptrs, [10, 20, 30, 40, 50, 60])
        self.assertEqual(dst, [110, 120, 130, 140, 150, 160])
        for bad in [(), (0, 0), (6,), (-1,)]:
            with self.assertRaises(ValueError):
                manager.send_kvcache(
                    "peer",
                    [3],
                    dst,
                    [1],
                    None,
                    dst_attn_tp_size=1,
                    kv_buffer_indices=bad,
                )
        with self.assertRaises(ValueError):
            pl.buffer_groups([0, 1, 2], 3, 2)

    def test_prelayer_source_release_requires_native_and_rpc_drain(self):
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            owner = SimpleNamespace(
                config=pl.Config("remote", Path(directory)), scheduler=object()
            )
            status = SimpleNamespace(done=False)
            req = SimpleNamespace(disagg_kv_sender=status)
            source = pl.Source(
                owner, req, Path(directory), "g", [], slice(0, 3), (), 3, finalized=True
            )
            req._pre_layerwise_source = source
            registry = self.runtime.HandoffRegistry()
            ctx = registry.register(
                "h",
                req,
                1,
                3,
                3,
                "token",
                on_release=lambda _: source.remote_closed.set(),
            )
            source.service = dict(handoff_id="h", epoch=ctx.owner_epoch)
            registry.acquire("h", ctx.owner_epoch)
            released = []
            self.assertTrue(pl.defer_release(req, lambda: released.append("KV")))
            self.assertTrue(pl.defer_release(req, lambda: released.append("duplicate")))
            pl.write_json(source.path / "done.json", {"generation": "old"})
            with (
                patch.object(
                    pl, "_poll_success", side_effect=lambda sender: sender.done
                ),
                patch.object(
                    self.runtime,
                    "get_remote_hub",
                    return_value=SimpleNamespace(registry=registry),
                ),
            ):
                self.assertFalse(source.poll())
                self.assertIsNotNone(registry.get("h", ctx.owner_epoch))
                pl.write_json(source.path / "done.json", {"generation": "g"})
                self.assertFalse(source.poll())
                self.assertIsNone(registry.acquire("h", ctx.owner_epoch))
                registry.release_ref("h", ctx.owner_epoch)
                self.assertFalse(source.poll())  # RPC drained, native still writing.
                status.done = True
                self.assertTrue(source.poll())
                self.assertEqual(released, [])  # TP agreement must precede freeing.
                source.retire()
                source.retire()
            self.assertEqual(released, ["KV"])
            self.assertIsNone(req._pre_layerwise_source)

    def test_prelayer_target_keeps_metadata_until_native_completion(self):
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            freed, cleared, dropped = [], [], []
            receiver = SimpleNamespace(done=False, clear=lambda: cleared.append(1))
            req = SimpleNamespace()
            dr = SimpleNamespace(req=req, kv_receiver=receiver, metadata_buffer_index=0)
            queue_ = SimpleNamespace(
                metadata_buffers=SimpleNamespace(bootstrap_room=torch.tensor([[7]])),
                req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=freed.append),
            )
            state = SimpleNamespace(
                events=[threading.Event()],
                layer_ready_at=[None],
                import_done_at=None,
                remote_calls=2,
                admitted_at=1.0,
            )
            owner = SimpleNamespace(config=pl.Config("remote", path))
            target = pl.Target(
                owner, queue_, dr, path, "g", pl.LayerCoverage(3, 1), state
            )
            req._pre_layerwise_target = target
            pl.write_json(
                path / "layer-0.json", dict(generation="g", layer=0, start=0, end=3)
            )
            with (
                patch.object(pl, "_poll_success", side_effect=lambda r: r.done),
                patch.object(
                    self.runtime, "_drop_target_state_for", side_effect=dropped.append
                ),
            ):
                target.finish(wait=False)
                self.assertTrue(state.events[0].is_set())
                self.assertEqual(freed + cleared + dropped, [])
                self.assertEqual(queue_.metadata_buffers.bootstrap_room.item(), 7)
                receiver.done = True
                target.finish(wait=False)
                target.finish(wait=False)
            self.assertEqual(freed, [0])
            self.assertEqual(cleared, [1])
            self.assertEqual(dropped, [req])
            self.assertEqual(queue_.metadata_buffers.bootstrap_room.item(), 0)
            self.assertIsNone(dr.kv_receiver)
            self.assertEqual(pl.read_json(path / "done.json")["remote_calls"], 2)

    def test_prelayer_target_failure_never_frees_destination(self):
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            freed = []
            req = SimpleNamespace()
            receiver = SimpleNamespace()
            target = pl.Target(
                SimpleNamespace(config=pl.Config("remote", Path(directory))),
                SimpleNamespace(
                    req_to_metadata_buffer_idx_allocator=SimpleNamespace(
                        free=freed.append
                    )
                ),
                SimpleNamespace(req=req, kv_receiver=receiver),
                Path(directory),
                "g",
                pl.LayerCoverage(3, 1),
            )
            req._pre_layerwise_target = target
            with patch.object(
                pl, "_poll_success", side_effect=RuntimeError("unsettled transport")
            ):
                with self.assertRaisesRegex(RuntimeError, "unsettled"):
                    pl.defer_release(req, lambda: freed.append("KV"))
            self.assertEqual(freed, [])
            self.assertFalse(target.closed)
            self.assertIs(target.decode_req.kv_receiver, receiver)

    def test_prelayer_streams_prefill_response_exactly_once(self):
        for native_first in (False, True):
            with self.subTest(native_first=native_first):
                source = SimpleNamespace(responded=False)
                req = SimpleNamespace(_pre_layerwise_source=source)
                snapshot = copy.copy(req)
                snapshot._pre_layerwise_response_snapshot = True
                outputs = []
                emit = lambda streamer, reqs: outputs.extend(reqs)
                if native_first:
                    self.hooks._stream_output(emit, None, [req])
                    self.assertTrue(source.responded)  # Controller must skip snapshot.
                else:
                    self.hooks._stream_output(emit, None, [snapshot])
                    self.hooks._stream_output(emit, None, [req])
                self.assertEqual(len(outputs), 1)

    def test_prelayer_worker_proves_completion_only_after_all_writes(self):
        import logging
        from collections import defaultdict

        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py", "MooncakeKVManager", {"transfer_worker"}
        )
        polls = _source_class("disaggregation/base/conn.py", "KVPoll")
        manager_cls.transfer_worker.__globals__.update(
            time=time, logger=logging.getLogger(__name__), KVPoll=polls
        )
        for outcome in (
            "success",
            "staged_success",
            "failed_peer",
            "cancelled",
            "short_destination",
        ):
            with self.subTest(outcome=outcome):
                events = []
                manager = manager_cls()
                manager.enable_trace = manager.enable_staging = (
                    manager.enable_deferred_decode_kv_release
                ) = False
                manager.is_mla_backend = manager.is_hybrid_mla_backend = False
                manager.attn_tp_size = 1
                manager.bootstrap_port = 1
                manager.session_lock = threading.Lock()
                manager.state_layout_rejections = manager.failed_session_probes = {}
                manager.failed_sessions = {"b"} if outcome == "failed_peer" else set()
                manager._staging_outstanding = defaultdict(int)
                manager.kv_args = SimpleNamespace(kv_data_ptrs=[10, 20])
                manager.request_status = {7: polls.Transferring}
                manager.req_to_decode_prefix_len = {}
                manager.check_status = lambda room: manager.request_status[room]
                manager._get_dsa_cache_transfer_skip_flags = lambda peer: (False, False)
                manager.conclude_failure = lambda bootstrap_room, **kw: (
                    manager.request_status.update({bootstrap_room: polls.Failed})
                )
                manager.poison_deferred_ack_room = lambda room: None
                manager.decode_kv_args_table = {
                    session: SimpleNamespace(
                        requires_dcp_relayout=False,
                        dst_attn_tp_size=1,
                        dst_kv_ptrs=[30, 40],
                        dst_kv_layer_ids=[],
                        dst_kv_item_len=8,
                    )
                    for session in ("a", "b")
                }
                manager.transfer_infos = {
                    7: {
                        session: SimpleNamespace(
                            is_dummy=False,
                            mooncake_session_id=session,
                            dst_device_kv_indices=None,
                            dst_kv_indices=(
                                [5] if outcome == "short_destination" else [5, 6]
                            ),
                        )
                        for session in ("a", "b")
                    }
                }

                def send(session, *args, **kwargs):
                    self.assertIs(kwargs["transfer_fn"], transfer_fn)
                    events.append(session)
                    if outcome == "cancelled":
                        manager.request_status.pop(7, None)
                    return 0

                manager.send_kvcache = send
                transfer_fn = object() if outcome == "staged_success" else None
                chunk = SimpleNamespace(
                    room=7,
                    staging_counted=False,
                    wait_event=SimpleNamespace(
                        synchronize=lambda: events.append("gpu-ready")
                    ),
                    indices_event=None,
                    prefill_kv_indices=[1, 2],
                    index_slice=slice(0, 2),
                    is_last_chunk=False,
                    kv_buffer_indices=(0, 1),
                    transfer_fn=transfer_fn,
                    on_complete=lambda: events.append("proof"),
                )
                tasks = queue.Queue()
                tasks.put(chunk)
                tasks.put(None)
                if outcome == "short_destination":
                    with self.assertRaisesRegex(RuntimeError, "does not cover"):
                        manager.transfer_worker(tasks, None)
                else:
                    manager.transfer_worker(tasks, None)
                if outcome in ("success", "staged_success"):
                    self.assertEqual(events, ["gpu-ready", "a", "b", "proof"])
                else:
                    self.assertNotIn("proof", events)

    def test_prelayer_cuda_slot_orders_private_copy_before_native_send(self):
        """Exercise the real CUDA adapter with driver calls at the test boundary."""
        from sglang_remote_attn.pinned_staging import PinnedKVStager

        manager = _host_staging_manager()
        ready = object()
        pending, received, flags, destroyed, waited = [], [], [], [], []
        make_empty = torch.empty
        test = self

        class Stream:
            def wait_event(self, event):
                test.assertIs(event, ready)
                waited.append(event)

            def synchronize(self):
                test.assertEqual(pending, [])

        stream = Stream()

        class Event:
            recorded = False

            def record(self, actual):
                test.assertIs(actual, stream)
                self.recorded = True

            def synchronize(self):
                test.assertTrue(self.recorded)
                for dst, src, size in pending:
                    ctypes.memmove(dst, src, size)
                pending.clear()

        def memcpy(dst, src, size, kind, raw):
            self.assertEqual(raw, 17)
            self.assertIn(kind, (1, 2))
            if kind == 2:
                self.assertEqual(waited[-1:], [ready])
            pending.append((dst, src, size))
            return (0,)

        def empty(*args, **kwargs):
            self.assertTrue(kwargs.pop("pin_memory"))
            return make_empty(*args, **kwargs)

        cudart = SimpleNamespace(
            cudaStreamNonBlocking=1,
            cudaStreamCreateWithFlags=lambda f: (flags.append(f) or 0, 17),
            cudaStreamDestroy=lambda raw: (destroyed.append(raw) or 0,),
            cudaMemcpyKind=SimpleNamespace(
                cudaMemcpyDeviceToHost=2, cudaMemcpyHostToDevice=1
            ),
            cudaMemcpyAsync=memcpy,
        )
        with (
            patch.dict(sys.modules, {"cuda.bindings": SimpleNamespace(runtime=cudart)}),
            patch.object(torch, "empty", side_effect=empty),
            patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()),
            patch.object(torch.cuda, "ExternalStream", return_value=stream),
            patch.object(torch.cuda, "Event", side_effect=Event),
        ):
            stager = PinnedKVStager(manager, 1)
            stager.reserve(4)
            stager.reserve(8)
            self.assertEqual(len(manager.registrations), 1)

            def transfer(session, blocks):
                self.assertEqual(pending, [])
                received.extend(ctypes.string_at(ptr, size) for ptr, _, size in blocks)
                return 0

            manager._transfer_data = transfer
            data = ctypes.create_string_buffer(b"abcdefgh")
            blocks = [
                (ctypes.addressof(data) + 4, 100, 4),
                (ctypes.addressof(data), 200, 4),
            ]
            self.assertEqual(stager.send("peer", blocks, ready_event=ready), 0)
            self.assertEqual(received, [b"efgh", b"abcd"])
            destination = ctypes.create_string_buffer(8)
            ptr = ctypes.addressof(destination)
            stager.slot.copy_to_device([(ptr + 4, 4), (ptr, 4)])
            # Event completion, not async submission, makes these bytes visible.
            self.assertEqual(destination.raw, b"abcdefgh")
            self.assertEqual(pending, [])
            self.assertTrue(stager.close())
        self.assertEqual(flags, [1, 1])
        self.assertEqual(destroyed, [17, 17])
        self.assertEqual(manager.registrations, {})

    def test_prelayer_staging_preserves_native_mapping_and_batch_bound(self):
        """Staging changes only source pointers, including scattered TP rows."""
        import numpy as np
        from sglang_remote_attn.pinned_staging import PinnedKVStager

        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"send_kvcache", "_send_kvcache_generic"},
        )
        group = _source_function(
            "disaggregation/common/utils.py", "group_concurrent_contiguous"
        )
        group.__globals__["np"] = np
        flat = _source_function(
            "disaggregation/mooncake/conn.py", "_uses_flat_entry_layout"
        )
        flat.__globals__["StateType"] = SimpleNamespace(SWA="swa")
        manager_cls._send_kvcache_generic.__globals__.update(
            StateType=SimpleNamespace(SWA="swa", DSA="dsa"),
            group_concurrent_contiguous=group,
            _uses_flat_entry_layout=flat,
            is_elided_entry=_source_function(
                "disaggregation/utils.py", "is_elided_entry"
            ),
        )
        get_ptrs = _source_class(
            "disaggregation/common/conn.py",
            "CommonKVManager",
            {"get_mla_kv_ptrs_with_pp"},
        ).get_mla_kv_ptrs_with_pp
        manager_cls.get_mla_kv_ptrs_with_pp = get_ptrs
        src_indices = np.array(
            [7, 2, 3, 15, 0, 21, 9, 4, 28, 29, 11, 5, 6, 13, 14, 8, 10], dtype=np.int32
        )
        dst_indices = np.array(
            [1, 8, 9, 14, 3, 20, 4, 5, 26, 27, 12, 17, 18, 6, 7, 22, 23], dtype=np.int32
        )
        for rank in (0, 1):
            with self.subTest(rank=rank):
                manager = manager_cls()
                manager.__dict__.update(_host_staging_manager().__dict__)
                manager.is_mla_backend = manager.is_hybrid_mla_backend = False
                manager.enable_custom_mem_pool = False
                manager.max_transfer_batch_indices = 8
                manager.pp_size, manager.attn_tp_size = 1, 2
                manager._validate_envelope_kv_layout = lambda *args: None
                src = [
                    torch.arange(64, dtype=torch.uint8).reshape(32, 2)
                    + rank * 70
                    + i * 4
                    for i in range(4)
                ]
                dst = [torch.full((32, 2), 255, dtype=torch.uint8) for _ in range(4)]
                manager.kv_args = SimpleNamespace(
                    kv_layer_ids=[],
                    kv_data_ptrs=[t.data_ptr() for t in src],
                    kv_item_lens=[2] * 4,
                )
                stager = PinnedKVStager(manager, rank, slot_factory=_HostStagingSlot)
                stager.reserve(8 * 2 * 2)
                ready = threading.Event()
                ready.set()
                batches = []

                def transfer(session, blocks):
                    self.assertEqual(session, f"peer-{rank}")
                    with manager._remote_attn_send_lock:
                        for source, target, size in blocks:
                            self.assertTrue(
                                stager.slot.ptr
                                <= source
                                < source + size
                                <= stager.slot.ptr + stager.slot.capacity
                            )
                            ctypes.memmove(target, source, size)
                        batches.append(sum(n for _, _, n in blocks))
                    return 0

                manager._transfer_data = transfer
                rc = manager.send_kvcache(
                    f"peer-{rank}",
                    src_indices,
                    [t.data_ptr() for t in dst],
                    dst_indices,
                    None,
                    dst_attn_tp_size=2,
                    kv_buffer_indices=(1, 3),
                    transfer_fn=functools.partial(stager.send, ready_event=ready),
                )
                self.assertEqual(rc, 0)
                self.assertEqual(batches, [32, 32, 4])
                for i in range(4):
                    expected = torch.full_like(dst[i], 255)
                    if i in (1, 3):
                        expected[dst_indices] = src[i][src_indices]
                    self.assertTrue(torch.equal(dst[i], expected))
                with self.assertRaisesRegex(ValueError, "subset"):
                    manager.send_kvcache(
                        "peer",
                        src_indices,
                        [],
                        dst_indices,
                        None,
                        transfer_fn=stager.send,
                    )
                # Exercise the same native Batch=8 mapping through BOTH adapters.
                from sglang_remote_attn.receive_staging import (
                    HostDestination,
                    ReceiveStaging,
                )

                pl = self.hooks.pre_layerwise
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    rx_manager = _host_staging_manager()
                    rx_manager.kv_args = SimpleNamespace(
                        kv_data_ptrs=[t.data_ptr() for t in dst],
                        kv_item_lens=[2] * 4,
                        kv_layer_ids=[],
                    )
                    rx = ReceiveStaging(
                        rx_manager,
                        rank,
                        path,
                        "native",
                        dst_indices,
                        2,
                        3,
                        slot_factory=_HostStagingSlot,
                    )
                    source = pl.Source(
                        SimpleNamespace(
                            config=pl.Config("remote", path, drain_timeout=3)
                        ),
                        SimpleNamespace(
                            disagg_kv_sender=SimpleNamespace(kv_mgr=manager)
                        ),
                        path,
                        "native",
                        src_indices,
                        slice(0, len(src_indices)),
                        ((SimpleNamespace(mooncake_session_id=f"peer-{rank}"), None),),
                        len(src_indices),
                        stager=stager,
                    )
                    groups = pl.buffer_groups([0, 1, 0, 1], 2, 1)
                    source.destination = HostDestination(source, groups)
                    batches.clear()
                    for tensor in dst:
                        tensor.fill_(255)
                    for layers, buffers in groups:
                        rc = manager.send_kvcache(
                            f"peer-{rank}",
                            src_indices,
                            [t.data_ptr() for t in dst],
                            dst_indices,
                            None,
                            dst_attn_tp_size=2,
                            kv_buffer_indices=buffers,
                            transfer_fn=functools.partial(
                                source.destination.transfer, layers, ready
                            ),
                        )
                        self.assertEqual(rc, 0)
                        source.complete(layers)
                        # Sorted target runs merge across native batch boundaries.
                        self.assertEqual(len(rx.storage.slot.h2d_blocks), 16)
                    self.assertEqual(batches, [32, 32, 4] * 2)
                    for i in range(4):
                        expected = torch.full_like(dst[i], 255)
                        expected[dst_indices] = src[i][src_indices]
                        self.assertTrue(torch.equal(dst[i], expected))
                    rx.close()
                    self.assertFalse(rx_manager.registrations)
                self.assertTrue(stager.close())
                self.assertFalse(manager.registrations)

    def test_prelayer_staging_quarantines_unknown_completion_and_unregister(self):
        """A failed native call must never turn pinned storage into reusable memory."""
        from sglang_remote_attn.pinned_staging import PinnedKVStager

        for failure in ("register", "copy", "send", "send_exception", "unregister"):
            with self.subTest(failure=failure):
                manager = _host_staging_manager()
                stager = PinnedKVStager(manager, 0, slot_factory=_HostStagingSlot)
                ready = threading.Event()
                ready.set()

                def fail(*args):
                    raise RuntimeError("injected native failure")

                if failure == "register":
                    manager.engine.register = fail
                    with self.assertRaises(RuntimeError):
                        stager.reserve(16)
                else:
                    stager.reserve(16)
                    manager._transfer_data = (
                        fail if failure == "send_exception" else lambda *args: -1
                    )
                    if failure == "copy":
                        stager.slot.copy = fail
                    if failure == "unregister":
                        manager.engine.deregister = fail
                        self.assertFalse(stager.close())
                    else:
                        data = ctypes.create_string_buffer(b"abcd")
                        blocks = [(ctypes.addressof(data), 100, 4)]
                        if failure == "send":
                            self.assertEqual(
                                stager.send("peer", blocks, ready_event=ready), -1
                            )
                        else:
                            with self.assertRaises(RuntimeError):
                                stager.send("peer", blocks, ready_event=ready)
                slot = stager.slot
                self.assertFalse(stager.close())
                self.assertFalse(slot.closed)
                self.assertIn(stager, manager.engine._pre_layerwise_staging_owners)
                with self.assertRaisesRegex(RuntimeError, "quarantined"):
                    stager.reserve(16)
                pl = self.hooks.pre_layerwise
                req = SimpleNamespace(
                    _pre_layerwise_source=SimpleNamespace(
                        stager=stager, error=None, destination=None
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "retaining source KV"):
                    pl.defer_release(req, lambda: self.fail("unsafe KV release"))

    def test_prelayer_staging_serializes_slot_reuse_until_send_completes(self):
        from sglang_remote_attn.pinned_staging import PinnedKVStager

        manager = _host_staging_manager()
        stager = PinnedKVStager(manager, 0, slot_factory=_HostStagingSlot)
        stager.reserve(4)
        entered, release, second_attempt = (threading.Event() for _ in range(3))
        ready = threading.Event()
        ready.set()
        payloads, errors = [], []

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "second-stage":
                    second_attempt.set()
                stager_lock.acquire()

            def __exit__(self, *args):
                stager_lock.release()

        stager_lock = stager.lock
        stager.lock = ObservedLock()

        def transfer(session, blocks):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("test send did not drain")
            payloads.append(ctypes.string_at(blocks[0][0], 4))
            return 0

        manager._transfer_data = transfer
        buffers = [ctypes.create_string_buffer(s) for s in (b"aaaa", b"bbbb")]

        def send(index):
            try:
                stager.send(
                    "peer",
                    [(ctypes.addressof(buffers[index]), 100, 4)],
                    ready_event=ready,
                )
            except Exception as exc:
                errors.append(exc)

        first = threading.Thread(target=send, args=(0,))
        second = threading.Thread(target=send, args=(1,), name="second-stage")
        try:
            first.start()
            self.assertTrue(entered.wait(3))
            second.start()
            self.assertTrue(second_attempt.wait(3))
            self.assertEqual(ctypes.string_at(stager.slot.ptr, 4), b"aaaa")
        finally:
            release.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(payloads, [b"aaaa", b"bbbb"])
        self.assertTrue(stager.close())

    def test_prelayer_receive_staging_gates_local_and_reuses_after_h2d(self):
        """A completed network write cannot expose a layer or overwrite its slot."""
        from sglang_remote_attn.pinned_staging import PinnedKVStager
        from sglang_remote_attn.receive_staging import HostDestination, ReceiveStaging

        pl = self.hooks.pre_layerwise
        for rank, source_staging in ((0, False), (1, True)):
            with (
                self.subTest(rank=rank, source_staging=source_staging),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                started, allow_copy = threading.Event(), threading.Event()

                class Slot(_HostStagingSlot):
                    def copy_to_device(self, blocks):
                        started.set()
                        if not allow_copy.wait(3):
                            raise TimeoutError("test H2D fence")
                        offset = 0
                        for dst, n in blocks:
                            ctypes.memmove(dst, self.ptr + offset, n)
                            offset += n

                destination = [
                    ctypes.create_string_buffer(b"........") for _ in range(4)
                ]
                ptrs = [ctypes.addressof(b) for b in destination]
                manager = _host_staging_manager()
                manager.kv_args = SimpleNamespace(
                    kv_data_ptrs=ptrs, kv_item_lens=[1] * 4, kv_layer_ids=[]
                )
                receiver = ReceiveStaging(
                    manager,
                    rank,
                    path,
                    "current",
                    [6, 2, 4, 1],
                    2,
                    3,
                    slot_factory=Slot,
                )
                self.addCleanup(allow_copy.set)
                sender = _host_staging_manager()
                calls = []

                def transfer(session, blocks):
                    calls.append(blocks)
                    for src, dst, n in blocks:
                        ctypes.memmove(dst, src, n)
                    return 0

                sender._transfer_data = transfer
                stager = (
                    PinnedKVStager(sender, rank, slot_factory=_HostStagingSlot)
                    if source_staging
                    else None
                )
                if stager:
                    stager.reserve(4)
                source = pl.Source(
                    SimpleNamespace(config=pl.Config("remote", path, drain_timeout=3)),
                    SimpleNamespace(disagg_kv_sender=SimpleNamespace(kv_mgr=sender)),
                    path,
                    "current",
                    None,
                    slice(1, 4),
                    ((SimpleNamespace(mooncake_session_id="peer"), None),),
                    4,
                    stager=stager,
                )
                groups = pl.buffer_groups([0, 1, 0, 1], 2, 1)
                adapter = source.destination = HostDestination(source, groups)
                ready = threading.Event()
                ready.set()
                state = SimpleNamespace(
                    events=[threading.Event(), threading.Event()],
                    layer_ready_at=[None, None],
                    import_done_at=None,
                )
                target = pl.Target(
                    None,
                    None,
                    None,
                    path,
                    "current",
                    pl.LayerCoverage(4, 2),
                    state,
                    receive_staging=receiver,
                )
                payload = ctypes.create_string_buffer(b"abcXYZ")
                expected = [b"........"] * 4
                for group_index, (layers, buffers) in enumerate(groups):
                    blocks = [
                        (ctypes.addressof(payload) + j, ptrs[buffer] + token, 1)
                        for j, (buffer, token) in enumerate(
                            (b, t) for b in buffers for t in (2, 4, 1)
                        )
                    ]
                    # Distinct native sub-batches accumulate into ONE group slot.
                    for chunk in (blocks[:2], blocks[2:]):
                        self.assertEqual(
                            adapter.transfer(layers, ready, "peer", chunk), 0
                        )
                    self.assertTrue(
                        all(
                            adapter.ptr <= dst < adapter.ptr + adapter.capacity
                            for call in calls
                            for _, dst, _ in call
                        )
                    )
                    errors = []

                    def complete():
                        try:
                            source.complete(layers)
                        except Exception as exc:
                            errors.append(exc)

                    completion = threading.Thread(target=complete)
                    completion.start()
                    self.assertTrue(started.wait(3))
                    if group_index == 0:
                        target.refresh()
                        self.assertFalse(state.events[0].is_set())
                        self.assertFalse((path / "rx-ack-0.json").exists())
                        self.assertTrue(completion.is_alive())
                        self.assertEqual([b.raw[:8] for b in destination], expected)
                    allow_copy.set()
                    completion.join(3)
                    self.assertFalse(completion.is_alive())
                    self.assertEqual(errors, [])
                    target.refresh()
                    self.assertTrue(state.events[group_index].is_set())
                    # Prefix/middle-chunk token 6 and unrelated/decode rows stay intact.
                    expected[buffers[0]] = b".ca.b..."
                    expected[buffers[1]] = b".ZX.Y..."
                    self.assertEqual([b.raw[:8] for b in destination], expected)
                    self.assertEqual(adapter.offset, 0)
                receiver.close()
                receiver.close()
                self.assertEqual(manager.registrations, {})
                if stager:
                    self.assertTrue(stager.close())

    def test_prelayer_receive_staging_rejects_wrong_rows_and_retains_on_failure(self):
        """Wrong generation, missing bytes or H2D failure cannot publish GPU-ready."""
        from sglang_remote_attn.receive_staging import ReceiveStaging, _wait

        pl = self.hooks.pre_layerwise
        for failure in ("wrong-row", "missing-bytes", "copy", "stale-generation"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                dst = ctypes.create_string_buffer(b"........")
                ptr = ctypes.addressof(dst)
                manager = _host_staging_manager()
                manager.kv_args = SimpleNamespace(
                    kv_data_ptrs=[ptr, ptr + 4], kv_item_lens=[1, 1], kv_layer_ids=[]
                )

                class Slot(_HostStagingSlot):
                    def copy_to_device(self, blocks):
                        raise RuntimeError("injected H2D failure")

                rx = ReceiveStaging(
                    manager, 0, path, "current", [0, 1], 1, 0.1, slot_factory=Slot
                )
                pl.write_json(
                    path / "rx-plan.json",
                    dict(generation="current", start=1, end=2, groups=[[0]]),
                )
                _wait(path / "rx-slot.json", "current", 3)
                blocks = [[ptr + 1, 1], [ptr + 5, 1]]
                if failure == "wrong-row":
                    blocks[0][0] = ptr + 2
                if failure == "missing-bytes":
                    blocks.pop()
                pl.write_json(
                    path / "rx-group-0.json",
                    dict(
                        generation="old"
                        if failure == "stale-generation"
                        else "current",
                        layers=[0],
                        blocks=blocks,
                    ),
                )
                rx.thread.join(3)
                self.assertFalse(rx.thread.is_alive())
                self.assertIsNotNone(rx.error)
                self.assertFalse((path / "rx-ack-0.json").exists())
                self.assertEqual(dst.raw[:8], b"........")
                with self.assertRaisesRegex(RuntimeError, "retaining KV"):
                    pl.defer_release(
                        SimpleNamespace(_pre_layerwise_receive=rx), lambda: None
                    )
                self.assertTrue(manager.registrations)
                self.assertIn(rx.storage, manager.engine._pre_layerwise_staging_owners)

    def test_prelayer_receive_ack_failure_never_publishes_layer_proof(self):
        """Host-write success alone cannot authorize Local or source KV recycling."""
        from sglang_remote_attn.receive_staging import HostDestination

        pl = self.hooks.pre_layerwise
        for failure in ("stale-ack", "failed-send", "missing-bytes", "wrong-session"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory)
                pl.write_json(
                    path / "rx-slot.json",
                    dict(generation="current", ptr=100, capacity=2, groups=[[[10, 2]]]),
                )
                manager = SimpleNamespace(
                    _transfer_data=lambda session, blocks: (
                        -1 if failure == "failed-send" else 0
                    )
                )
                source = pl.Source(
                    SimpleNamespace(
                        config=pl.Config("layerwise", path, drain_timeout=0.02)
                    ),
                    SimpleNamespace(disagg_kv_sender=SimpleNamespace(kv_mgr=manager)),
                    path,
                    "current",
                    None,
                    slice(0, 1),
                    ((SimpleNamespace(mooncake_session_id="peer"), None),),
                    1,
                )
                adapter = source.destination = HostDestination(source, [((0,), (0, 1))])
                if failure == "wrong-session":
                    with self.assertRaisesRegex(RuntimeError, "session changed"):
                        adapter.transfer((0,), None, "other", [(1, 10, 2)])
                else:
                    adapter.transfer(
                        (0,),
                        None,
                        "peer",
                        [(1, 10, 1 if failure == "missing-bytes" else 2)],
                    )
                pl.write_json(
                    path / "rx-ack-0.json", dict(generation="old", layers=[0])
                )
                with self.assertRaises((RuntimeError, TimeoutError, ValueError)):
                    source.complete((0,))
                self.assertFalse((path / "layer-0.json").exists())
                self.assertTrue(adapter.poisoned)
                with self.assertRaisesRegex(RuntimeError, "retaining source KV"):
                    pl.defer_release(
                        SimpleNamespace(_pre_layerwise_source=source), lambda: None
                    )

    def test_prelayer_receive_unused_offer_has_no_pinned_allocation(self):
        """Ordinary PD / auto-group fallback must not leak a receive worker or slot."""
        from sglang_remote_attn.receive_staging import ReceiveStaging

        with tempfile.TemporaryDirectory() as directory:
            manager = _host_staging_manager()
            manager.kv_args = SimpleNamespace(
                kv_data_ptrs=[10, 20], kv_item_lens=[1, 1], kv_layer_ids=[]
            )
            rx = ReceiveStaging(
                manager,
                0,
                Path(directory),
                "current",
                [0],
                1,
                1,
                slot_factory=lambda *args: self.fail("unused offer allocated memory"),
            )
            rx.close()
            self.assertFalse(rx.thread.is_alive())
            self.assertEqual(manager.registrations, {})
            self.assertEqual(manager.engine._pre_layerwise_staging_owners, [])

    def test_prelayer_auto_grouping_uses_chunk_work_not_prompt_length(self):
        """Small tails keep Remote; long chunks use fewer, larger transfers."""
        pl = self.hooks.pre_layerwise
        for mode in ("layerwise", "remote"):
            with patch.dict(
                os.environ,
                {"HANDOFF_PRE_LAYERWISE": mode, "HANDOFF_PD_DIR": "/tmp/unused"},
                clear=True,
            ):
                config = pl.Config.from_env()
                self.assertFalse(config.host_staging)
                self.assertFalse(config.recv_host_staging)
                with patch.dict(
                    os.environ, {"HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING": "1"}
                ):
                    self.assertTrue(pl.Config.from_env().recv_host_staging)
                with patch.dict(
                    os.environ, {"HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING": "yes"}
                ):
                    with self.assertRaises(ValueError):
                        pl.Config.from_env()
                with patch.dict(os.environ, {"MC_FORCE_TCP": "1"}):
                    self.assertTrue(pl.Config.from_env().host_staging)
                    with patch.dict(
                        os.environ, {"HANDOFF_PRE_LAYERWISE_HOST_STAGING": "0"}
                    ):
                        self.assertFalse(pl.Config.from_env().host_staging)
            with self.subTest(mode=mode):
                self.assertEqual(
                    config.select_group_size(36, 2048, 2048),
                    36 if mode == "remote" else 0,
                )
                # A large middle chunk must not make a tiny eligible tail
                # cross the minimum pipelined-transfer threshold.
                self.assertEqual(
                    config.select_group_size(36, 2048, 32768),
                    36 if mode == "remote" else 0,
                )
                sizes = [
                    config.select_group_size(36, tokens, tokens)
                    for tokens in (3072, 4096, 8192, 16384, 32768)
                ]
                self.assertEqual(sizes, sorted(sizes))
                self.assertEqual(sizes[-1], 9)
                # Every layer's K and V is still covered exactly once.
                for size in sizes:
                    groups = pl.buffer_groups(list(range(36)) * 2, 36, size)
                    self.assertEqual(
                        sorted(i for _, buffers in groups for i in buffers),
                        list(range(72)),
                    )
                self.assertGreater(config.select_group_size(36, 4096, 2048), 0)
                fixed = pl.Config(mode, Path("unused"), 4)
                self.assertEqual(fixed.select_group_size(36, 2048, 2048), 4)

    def test_prelayer_chunk_tail_reserves_once_and_preserves_middle_chunks(self):
        pl = self.hooks.pre_layerwise
        sender_cls = _source_class(
            "disaggregation/common/conn.py", "CommonKVSender", {"_prepare_send_indices"}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            peer = SimpleNamespace(
                is_dummy=False, mooncake_session_id="peer", dst_device_kv_indices=None
            )
            registration = SimpleNamespace(
                dst_attn_tp_size=1,
                requires_dcp_relayout=False,
                dst_kv_layer_ids=[],
                dst_kv_ptrs=list(range(100, 172)),
            )
            sender = sender_cls()
            sender.kv_mgr = SimpleNamespace(
                transfer_infos={7: {0: peer}},
                decode_kv_args_table={"peer": registration},
                enable_staging=False,
                attn_tp_size=1,
                kv_shard_size=1,
                enable_all_cp_ranks_for_transfer=False,
                is_dummy_cp_rank=False,
                kv_args=SimpleNamespace(
                    kv_layer_ids=[], kv_data_ptrs=list(range(72)), state_data_ptrs=[]
                ),
            )
            sender.curr_idx, sender.num_kv_indices = 4, 9
            sender._record_transfer_indices = lambda *args: None
            sender._transfer_start_time = None
            req = SimpleNamespace(
                disagg_kv_sender=sender,
                pending_bootstrap=False,
                inflight_middle_chunks=1,
                extend_end=9,
                extend_len=5,
                origin_input_ids=list(range(9)),
                multimodal_inputs=None,
                input_embeds=None,
                bootstrap_room=7,
                start_send_idx=4,
                kv=SimpleNamespace(req_pool_idx=0),
            )
            batch = SimpleNamespace(
                reqs=[req],
                forward_mode=SimpleNamespace(
                    is_extend=lambda: True,
                    is_split_prefill=lambda: False,
                    is_mixed=lambda: False,
                ),
                is_prefill_only=False,
                return_hidden_states=False,
                input_embeds=None,
                replace_embeds=None,
            )
            controller = pl.Controller.__new__(pl.Controller)
            controller.scheduler = SimpleNamespace(
                disaggregation_mode="prefill",
                chunked_prefill_size=32768,
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.arange(100, 109, dtype=torch.int32).reshape(1, 9)
                ),
            )
            controller.config = pl.Config("remote", root, 1)
            controller.rank, controller.tp_size, controller.layers = 0, 1, 36
            controller.sources, controller.batches = [], 0
            pl.write_json(
                root / "7/0/offer.json",
                dict(
                    generation="g",
                    session="peer",
                    prefix_len=9,
                    tp_size=1,
                    layers=36,
                    mode="remote",
                ),
            )
            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.mooncake.conn": SimpleNamespace(
                        MooncakeKVSender=sender_cls
                    ),
                    "sglang.srt.disaggregation.utils": SimpleNamespace(
                        DisaggregationMode=SimpleNamespace(PREFILL="prefill")
                    ),
                },
            ):
                self.assertIsNone(controller.prepare(batch))
                self.assertEqual(sender.curr_idx, 4)
                req.inflight_middle_chunks = 0
                # Deployment allows 32K, but this actual tail has five tokens.
                # A skipped layerwise plan must leave native sending untouched.
                offer_path = root / "7/0/offer.json"
                offer = pl.read_json(offer_path)
                offer["mode"] = "layerwise"
                pl.write_json(offer_path, offer)
                controller.config = pl.Config("layerwise", root)
                batch.reqs.append(
                    SimpleNamespace(pending_bootstrap=True, extend_len=32768)
                )
                self.assertIsNone(controller.prepare(batch))
                self.assertEqual(sender.curr_idx, 4)
                self.assertEqual(req.start_send_idx, 4)
                self.assertEqual(controller.sources, [])
                batch.reqs.pop()
                offer["mode"] = "remote"
                pl.write_json(offer_path, offer)
                controller.config = pl.Config("remote", root)
                jobs = controller.prepare(batch)
                self.assertEqual(jobs[0][0].indices.tolist(), [104, 105, 106, 107, 108])
                self.assertEqual(jobs[0][0].index_slice, slice(4, 9))
                self.assertEqual(sender.curr_idx, 9)
                self.assertEqual(req.start_send_idx, 9)
                self.assertEqual(jobs[0][1], [(tuple(range(36)), tuple(range(72)))])
                self.assertEqual(jobs[0][0].prefill_chunk_tokens, 5)
                self.assertEqual(jobs[0][0].group_size, 36)
                self.assertEqual(controller.scheduler.chunked_prefill_size, 32768)
                self.assertIsNone(controller.prepare(batch))
                self.assertEqual(len(controller.sources), 1)
                self.assertEqual(sender.curr_idx, 9)

    def test_prelayer_leases_prevent_idle_cache_reset(self):
        pl = self.hooks.pre_layerwise
        controller = SimpleNamespace(sources=[], targets=[])
        scheduler = SimpleNamespace(_pre_layerwise=controller)
        idle = lambda scheduler, **kwargs: True
        self.assertTrue(pl.is_fully_idle(idle, scheduler))
        controller.sources.append(object())
        self.assertFalse(pl.is_fully_idle(idle, scheduler))
        self.assertTrue(pl.is_fully_idle(idle, scheduler, for_health_check=True))
        controller.sources.clear()
        controller.targets.append(object())
        self.assertFalse(pl.is_fully_idle(idle, scheduler))
        controller.targets.clear()
        self.assertTrue(pl.is_fully_idle(idle, scheduler))
        self.assertFalse(pl.is_fully_idle(lambda scheduler, **kwargs: False, scheduler))

    def test_hook_targets_match_current_main(self):
        self.check.check_sources(ROOT)


if __name__ == "__main__":
    unittest.main()
