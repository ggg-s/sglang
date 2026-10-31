"""CPU contracts for the optional plugin; GPU/model imports stay at the boundary.

Source-loaded pool methods are the real upstream implementations. This avoids
loading Triton/CUDA just to exercise row ownership and token-map lifetimes.
"""

import ast
import copy
import ctypes
import dataclasses
import enum
import functools
import gc
import importlib
import importlib.util
import io
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

register_cpu_ci(est_time=16, suite="base-a-test-cpu")

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
        if ready_event is not None and not ready_event.wait(3):
            raise TimeoutError("KV producer did not finish")
        offset = 0
        for src, _, size in blocks:
            ctypes.memmove(self.ptr + offset, src, size)
            offset += size

    def warmup(self):
        pass

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


def _source_class(relative, owner, members=None, *, strip_method_decorators=False):
    path = ROOT / "python" / "sglang" / "srt" / relative
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node.bases = []
    node.keywords = []
    if members is not None:
        node.body = [n for n in node.body if getattr(n, "name", None) in members]
    if strip_method_decorators:
        for method in node.body:
            if isinstance(method, ast.FunctionDef):
                method.decorator_list = []
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


def _reference_reaper_class():
    import logging
    from collections import deque

    cls = _source_class("disaggregation/mooncake/conn.py", "_TransferReferenceReaper")
    cls.__init__.__globals__.update(
        os=os,
        queue_module=queue,
        deque=deque,
        threading=threading,
        logger=logging.getLogger(__name__),
    )
    return cls


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
        cls.interfaces = importlib.import_module("sglang_remote_attn.interfaces")
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

    def _radix(self):
        """Execute native tree, locks, allocator and release; omit GPU imports/events."""
        import heapq
        import logging
        from abc import abstractmethod
        from collections import defaultdict
        from enum import Enum, auto
        from typing import NamedTuple

        ns = dict(
            __name__=__name__,
            torch=torch,
            dataclasses=dataclasses,
            array=array,
            Enum=Enum,
            auto=auto,
            NamedTuple=NamedTuple,
            abstractmethod=abstractmethod,
            heapq=heapq,
            sys=sys,
            time=time,
            defaultdict=defaultdict,
            logger=logging.getLogger(__name__),
        )

        def load(relative, names=None):
            path = ROOT / "python/sglang/srt" / relative
            nodes = [
                n
                for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and (names is None or n.name in names)
            ]
            for node in nodes:
                if node.name == "BasePrefixCache":
                    node.bases = []
            module = ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    *nodes,
                ],
                type_ignores=[],
            )
            exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)

        load("mem_cache/unified_cache/component_type.py")
        load(
            "mem_cache/base_prefix_cache.py",
            {
                "CacheRequestHandle",
                "CacheRequestOutcome",
                "MatchPrefixParams",
                "InsertParams",
                "InsertResult",
                "EvictParams",
                "EvictResult",
                "IncLockRefResult",
                "DecLockRefParams",
                "DecLockRefResult",
                "TreeLock",
                "MatchResult",
                "BasePrefixCache",
            },
        )
        ns["EvictionStrategy"] = object
        load("mem_cache/evict_policy.py", {"LRUStrategy"})
        load("mem_cache/utils.py", {"split_node_hash_value"})
        load("mem_cache/radix_cache.py")
        load(
            "mem_cache/common.py",
            {
                "release_kv_cache",
                "coalesce_ranges",
                "free_kv_row_segments",
                "_release_overallocated_kv_indices",
            },
        )
        ns.update(
            get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
            get_serving=lambda: SimpleNamespace(strip_thinking_cache=False),
        )
        allocator = self._allocator()
        allocator.page_size = 1
        base = _source_class(
            "mem_cache/allocator/base.py",
            "BaseTokenToKVPoolAllocator",
            {"free_segment", "free_segments", "_page_disjoint"},
        )
        for name in ("free_segment", "free_segments", "_page_disjoint"):
            setattr(type(allocator), name, getattr(base, name))
        cache = ns["RadixCache"].__new__(ns["RadixCache"])
        cache.disable, cache.page_size, cache.is_eagle = False, 1, False
        cache.device = "cpu"
        cache.req_to_token_pool = self._pool(8)
        cache.token_to_kv_pool_allocator = allocator
        cache.eviction_strategy = ns["LRUStrategy"]()
        cache.evictable_leaves = set()
        # Event publication is an external boundary, not tree behavior.
        cache.kv_events = SimpleNamespace(
            record_all_cleared=lambda: None,
            record_store=lambda *a: None,
            record_remove=lambda *a: None,
        )
        cache.reset()
        common = SimpleNamespace(
            coalesce_ranges=ns["coalesce_ranges"],
            free_kv_row_segments=ns["free_kv_row_segments"],
        )
        patcher = patch.dict(sys.modules, {"sglang.srt.mem_cache.common": common})
        patcher.start()
        self.addCleanup(patcher.stop)

        def request(tokens, rid):
            req = self._req()
            req.rid, req.cache_request_handle = rid, ns["CacheRequestHandle"](rid, 0)
            req.origin_input_ids, req.output_ids = array("q", tokens), array("q")
            req.extra_key, req.cache_salt, req.priority = None, None, 0
            req.skip_radix_cache_insert = False
            req.finished = lambda: True
            req.owned_kv_len = lambda: len(tokens)
            req.finished_len = None
            req.refresh_fill_ids()
            matched = cache.match_prefix(
                ns["MatchPrefixParams"](ns["RadixKey"](array("q", tokens)))
            )
            req.last_node = matched.last_device_node
            req.lock = cache.lock(req.last_node)
            req.kv.cache_protected_len = matched.device_prefix_len
            cache.req_to_token_pool.alloc([req])
            indices = torch.cat(
                (
                    cache.path_device_indices(req.last_node),
                    allocator.alloc(len(tokens) - matched.device_prefix_len),
                )
            )
            cache.req_to_token_pool.write(
                (req.kv.req_pool_idx, slice(0, len(tokens))), indices
            )
            req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
            return req, indices.clone()

        return SimpleNamespace(cache=cache, allocator=allocator, request=request, **ns)

    def test_p_radix_read_pin_survives_tree_split_and_eviction(self):
        from sglang_remote_attn.source_cache import CacheGuard

        f = self._radix()
        prefix, _ = f.request([1, 2, 3], "warm")
        f.release_kv_cache(prefix, f.cache, checkpoint=True)
        req, indices = f.request([1, 2, 3, 4, 5], "source")
        guard = CacheGuard(f.cache)
        lease = guard.hold(req)
        f.cache.unlock(req.lock)
        req.lock = None  # Independent read pin outlives native request pin.
        f.cache.match_prefix(f.MatchPrefixParams(f.RadixKey(array("q", [1, 2]))))
        f.cache.evict(f.EvictParams(num_tokens=32))
        self.assertEqual(f.allocator.available_size(), 27)
        self.assertEqual(
            f.cache.path_device_indices(req.last_node).tolist(), indices[:3].tolist()
        )
        lease.retire(lambda: f.release_kv_cache(req, f.cache, checkpoint=False))
        f.cache.evict(f.EvictParams(num_tokens=32))
        self.assertEqual(f.allocator.available_size(), 32)
        self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)
        self.assertTrue(lease.closed)
        self.assertEqual(guard.leases, {})

    def test_p_radix_checkpoint_waits_for_native_and_rpc_readers_then_dedups(self):
        from sglang_remote_attn.source_cache import CacheGuard, checkpoint_allowed

        f = self._radix()
        warm, _ = f.request([1, 2], "warm")
        f.release_kv_cache(warm, f.cache, checkpoint=True)
        req, original = f.request([1, 2, 3, 4], "source")
        lease = CacheGuard(f.cache).hold(req)
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            status = SimpleNamespace(done=False)
            req.disagg_kv_sender = status
            source = pl.Source(
                SimpleNamespace(
                    config=pl.Config("remote", Path(directory)), scheduler=object()
                ),
                req,
                Path(directory),
                "g",
                original.numpy(),
                slice(0, 4),
                (),
                4,
                finalized=True,
                cache_lease=lease,
            )
            req._pre_layerwise_source = source
            registry = self.runtime.HandoffRegistry()
            ctx = registry.register(
                "cache-reader",
                req,
                1,
                4,
                4,
                "token",
                on_release=lambda _: source.remote_closed.set(),
            )
            registry.acquire("cache-reader", ctx.owner_epoch)
            source.service = dict(handoff_id="cache-reader", epoch=ctx.owner_epoch)
            # Native final-prefill checkpoint must not publish/free the source suffix.
            f.cache.checkpoint(req, up_to=4)
            other, canonical = f.request([1, 2, 3, 4], "concurrent")
            self.assertNotEqual(canonical[2:].tolist(), original[2:].tolist())
            self.assertEqual(lease.checkpoints, 1)
            f.release_kv_cache(other, f.cache, checkpoint=True)
            release = lambda: f.release_kv_cache(
                req, f.cache, checkpoint=checkpoint_allowed(req, True)
            )
            self.assertTrue(pl.defer_release(req, release, checkpoint=True))
            f.cache.finish(req.cache_request_handle, f.CacheRequestOutcome.SUCCESS)
            pl.write_json(source.path / "done.json", {"generation": "g"})
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
                registry.release_ref("cache-reader", ctx.owner_epoch)
                self.assertFalse(source.poll())
                f.cache.evict(f.EvictParams(num_tokens=32))
                self.assertEqual(
                    f.cache.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, :4
                    ].tolist(),
                    original.tolist(),
                )
                self.assertTrue(
                    set(original.tolist()).isdisjoint(f.allocator.free_pages.tolist())
                )
                status.done = True
                self.assertTrue(source.poll())
                source.retire()
                source.retire()
            hit = f.cache.match_prefix(
                f.MatchPrefixParams(f.RadixKey(array("q", [1, 2, 3, 4])))
            )
            self.assertEqual(hit.device_prefix_len, 4)
            self.assertTrue(req.kv.is_kv_released)
            self.assertTrue(lease.closed)
            self.assertEqual(f.cache.protected_size_, 0)
            f.cache.evict(f.EvictParams(num_tokens=32))
            self.assertEqual(f.allocator.available_size(), 32)
            self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)

    def test_p_radix_abort_and_failed_release_keep_ownership_safe(self):
        from sglang_remote_attn.source_cache import CacheGuard, checkpoint_allowed

        for fail in (False, True):
            with self.subTest(fail=fail):
                f = self._radix()
                req, indices = f.request([9, 8, 7], "abort")
                guard = CacheGuard(f.cache)
                lease = guard.hold(req)
                req._pre_layerwise_source = SimpleNamespace(cache_lease=lease)
                f.cache.finish(req.cache_request_handle, f.CacheRequestOutcome.ABORT)
                f.cache.finish(req.cache_request_handle, f.CacheRequestOutcome.SUCCESS)
                self.assertFalse(checkpoint_allowed(req, True))
                with self.assertRaisesRegex(RuntimeError, "cannot reset"):
                    f.cache.reset()
                if fail:

                    def release():
                        raise RuntimeError("release failed")

                    with self.assertRaisesRegex(RuntimeError, "release failed"):
                        lease.retire(release)
                    with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                        lease.retire(release)
                    self.assertEqual(f.allocator.available_size(), 29)
                    self.assertEqual(len(guard.leases), 1)
                else:
                    lease.retire(
                        lambda: f.release_kv_cache(
                            req, f.cache, checkpoint=checkpoint_allowed(req, True)
                        )
                    )
                    self.assertEqual(f.cache.total_size(), 0)
                    self.assertEqual(f.allocator.available_size(), 32)
                    f.cache.reset()

    def test_p_radix_attempt_change_cannot_bypass_guard(self):
        from sglang_remote_attn.source_cache import CacheGuard

        f = self._radix()
        req, _ = f.request([1, 3], "retry")
        guard = CacheGuard(f.cache)
        guard.hold(req)
        req.cache_request_handle = f.CacheRequestHandle("retry", 1)
        with self.assertRaisesRegex(RuntimeError, "changed attempt"):
            f.cache.checkpoint(req, up_to=2)
        with self.assertRaisesRegex(RuntimeError, "already has"):
            guard.hold(req)
        self.assertEqual(f.allocator.available_size(), 30)

    def test_p_radix_layerwise_release_also_waits_for_transfer(self):
        from sglang_remote_attn.source_cache import CacheGuard

        f = self._radix()
        req, _ = f.request([2, 4], "layerwise")
        lease = CacheGuard(f.cache).hold(req)
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            req.disagg_kv_sender = SimpleNamespace(send=lambda *a: None)
            source = pl.Source(
                SimpleNamespace(
                    config=pl.Config("layerwise", Path(directory)),
                    scheduler=SimpleNamespace(
                        disagg_metadata_buffers=SimpleNamespace(set_buf=lambda r: None)
                    ),
                ),
                req,
                Path(directory) / "7/0/g",
                "g",
                [],
                slice(0, 2),
                (),
                2,
                cache_lease=lease,
            )
            req._pre_layerwise_source = source
            source.finalize()
            self.hooks._release_kv(f.release_kv_cache, req, f.cache, checkpoint=True)
            self.assertIsNotNone(source.release)
            self.assertEqual(f.allocator.available_size(), 30)
            # A late cancellation must override the already stored checkpoint=True.
            self.hooks._release_kv(f.release_kv_cache, req, f.cache, checkpoint=False)
            source.retire()
            self.assertEqual(f.cache.total_size(), 0)
            self.assertEqual(f.allocator.available_size(), 32)
            spec = importlib.util.spec_from_file_location(
                "pd_report", ROOT / "test/manual/remote_attn/pre_layerwise_report.py"
            )
            report = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(report)
            result = report.summarize(Path(directory))
            self.assertEqual(result["p_radix_leased_shards"], 1)
            self.assertEqual(result["p_radix_closed_shards"], 1)
            self.assertEqual(result["rows"][0]["cached_prefix_tokens"], 0)

    def test_p_radix_install_checks_backend_and_disabled_passthrough(self):
        from sglang_remote_attn.source_cache import install

        f = self._radix()
        unified = type("UnifiedRadixCache", (), {})
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.radix_cache": SimpleNamespace(
                    RadixCache=f.RadixCache
                ),
                "sglang.srt.mem_cache.unified_radix_cache": SimpleNamespace(
                    UnifiedRadixCache=unified
                ),
                "sglang.srt.mem_cache.unified_cache.component_type": SimpleNamespace(
                    ComponentType=f.ComponentType
                ),
            },
        ):
            self.assertIsNone(install(SimpleNamespace(disable=True)))
            self.assertIs(install(f.cache), install(f.cache))
            with self.assertRaisesRegex(ValueError, "built-in dense"):
                install(SimpleNamespace(disable=False))
            for field, value in (
                ("tree_components", (f.ComponentType.FULL, f.ComponentType.SWA)),
                ("cache_controller", object()),
                ("linker", object()),
                ("enable_session_radix_cache", True),
            ):
                cache = unified()
                cache.disable, cache.tree_components = False, (f.ComponentType.FULL,)
                cache.cache_controller = cache.linker = None
                cache.enable_session_radix_cache = False
                setattr(cache, field, value)
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, "device-only FULL"),
                ):
                    install(cache)

    def test_p_radix_native_prefix_send_precedes_final_reservation(self):
        pl = self.hooks.pre_layerwise
        native = _source_class(
            "disaggregation/prefill.py",
            "SchedulerDisaggregationPrefillMixin",
            {"maybe_send_cached_prefix_chunk", "_send_kv_chunk"},
        )
        kv_pages = _source_function("mem_cache/common.py", "kv_to_page_indices")
        native.maybe_send_cached_prefix_chunk.__globals__.update(
            envs=SimpleNamespace(
                SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=SimpleNamespace(
                    get=lambda: True
                )
            ),
            kv_to_page_indices=kv_pages,
        )
        scheduler = native()
        scheduler.enable_staging = scheduler.enable_overlap = False
        scheduler.disagg_prefill_pending_chunk_rids = set()
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1, translate_kv_indices_for_transfer=lambda t: t
        )
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[11, 12, 13, 14]])
        )
        scheduler.send_kv_chunk = scheduler._send_kv_chunk
        sent = []
        req = SimpleNamespace(
            rid="cached",
            pending_bootstrap=False,
            prefix_len=2,
            host_hit_length=0,
            start_send_idx=0,
            extend_end=4,
            origin_input_ids=[1, 2, 3, 4],
            kv=SimpleNamespace(req_pool_idx=0),
            disagg_kv_sender=SimpleNamespace(
                get_max_transfer_tokens=lambda: None,
                should_send_kv_chunk=lambda n, last: n > 0,
                send=lambda pages, state, **kw: sent.append(pages.tolist()),
            ),
        )
        reserved = []

        def prepare(batch):
            reserved.append((req.start_send_idx, sent[:]))
            req.start_send_idx = 4

        scheduler._pre_layerwise = SimpleNamespace(
            cache_guard=object(), prepare=prepare
        )

        def run(scheduler, batch):
            scheduler.maybe_send_cached_prefix_chunk(
                req
            )  # Native run_batch repeats it.

        pl.run_batch(run, scheduler, SimpleNamespace(reqs=[req]))
        self.assertEqual(reserved, [(2, [[11, 12]])])
        self.assertEqual(sent, [[11, 12]])

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
        """Replay must not skip layer waits, and completion must restore both runners."""
        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_size=1, tp_rank=0)
        )
        for mode in ("remote", "layer"):
            for hook in (self.hooks._decode_graph, self.hooks._prefill_graph):
                with self.subTest(mode=mode, hook=hook.__name__):
                    events = [threading.Event(), threading.Event()]
                    state = SimpleNamespace(
                        mode=mode, events=events, error=None, aborted=False
                    )
                    runner = SimpleNamespace()
                    with (
                        patch.object(self.runtime, "_TARGET_STATES", [state]),
                        patch.dict(
                            sys.modules, {"sglang.srt.runtime_context": context}
                        ),
                    ):
                        self.assertFalse(hook(lambda *a: True, runner, None))
                        events[0].set()
                        self.assertFalse(hook(lambda *a: True, runner, None))
                        events[1].set()
                        self.assertTrue(hook(lambda *a: True, runner, None))
                        # Completion does not override the native shape limits.
                        self.assertFalse(hook(lambda *a: False, runner, None))
                        state.error = "failed after waking waiters"
                        self.assertFalse(hook(lambda *a: True, runner, None))
                        state.error, state.aborted = None, True
                        self.assertFalse(hook(lambda *a: True, runner, None))
                    with (
                        patch.object(self.runtime, "_TARGET_STATES", []),
                        patch.dict(
                            sys.modules, {"sglang.srt.runtime_context": context}
                        ),
                    ):
                        self.assertTrue(hook(lambda *a: True, runner, None))

    def test_graph_dispatch_recovers_without_recapture_and_split_prefill_stays_eager(
        self,
    ):
        """Exercise the native dispatch: Remote->Local->new Remote, with one runner."""
        from enum import IntEnum, auto

        tree = ast.parse(
            (
                ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"
            ).read_text()
        )
        node = next(n for n in tree.body if getattr(n, "name", None) == "ForwardMode")
        namespace = {"IntEnum": IntEnum, "auto": auto}
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "<ForwardMode>", "exec"),
            namespace,
        )
        mode = namespace["ForwardMode"]
        cls = _source_class(
            "model_executor/model_runner.py", "ModelRunner", {"_forward_raw"}
        )
        cls._forward_raw.__globals__.update(
            has_forward_context=lambda: False,
            forward_context=lambda *a, **kw: nullcontext(),
            ForwardContext=SimpleNamespace,
            get_model=lambda: SimpleNamespace(override=lambda **kw: nullcontext()),
            get_global_dwdp_manager=lambda: None,
            EagerRunner=type("EagerRunner", (), {}),
            ModelRunnerOutput=SimpleNamespace,
            _prefill_cuda_graph_allows_context_parallel=lambda *a: True,
            device_timer_ctx=lambda *a: nullcontext(),
        )
        paths = []

        def execute(path):
            def run(*a, **kw):
                paths.append(path)
                return "output"

            return run

        decode, prefill = SimpleNamespace(), SimpleNamespace()
        decode.can_run_graph = lambda fb: self.hooks._decode_graph(
            lambda *a: True, decode, fb
        )
        prefill.can_run_graph = lambda fb: self.hooks._prefill_graph(
            lambda *a: True, prefill, fb
        )
        decode.execute, prefill.execute = (
            execute("decode-graph"),
            execute("prefill-graph"),
        )
        runner = cls()
        runner.model = SimpleNamespace(shared_experts_config=None)
        runner.attn_backend = runner.device_timer = runner.hisparse_coordinator = None
        runner.device = "cuda"
        runner.decode_cuda_graph_runner, runner.prefill_cuda_graph_runner = (
            decode,
            prefill,
        )
        runner.eager_runner = SimpleNamespace(execute=execute("eager"))
        runner._prepare_eager_forward_batch = lambda fb: None
        runner._maybe_execute_deferred_mamba_cow_and_clear = lambda fb: None
        runner._extend_forward_kwargs = lambda *a: {}
        runner.forward_split_prefill = execute("split-eager")
        fb = SimpleNamespace(global_num_tokens_cpu=None, token_indices_to_pool=None)
        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_size=1, tp_rank=0)
        )
        event = threading.Event()
        state = SimpleNamespace(
            mode="remote", events=[event], error=None, aborted=False
        )
        with (
            patch.object(self.runtime, "_TARGET_STATES", [state]),
            patch.dict(sys.modules, {"sglang.srt.runtime_context": context}),
        ):
            for forward_mode in (mode.DECODE, mode.EXTEND):
                fb.forward_mode = forward_mode
                for complete in (False, True, False):
                    if complete:
                        event.set()
                    else:
                        event.clear()
                    result = runner._forward_raw(fb, None)
                    self.assertEqual(result.can_run_graph, complete)
                    self.assertEqual(result.logits_output, "output")
            event.set()
            fb.forward_mode = mode.SPLIT_PREFILL
            result = runner._forward_raw(fb, None, split_forward_count=2)
            self.assertFalse(result.can_run_graph)
        self.assertEqual(
            paths,
            [
                "eager",
                "decode-graph",
                "eager",
                "eager",
                "prefill-graph",
                "eager",
                "split-eager",
            ],
        )

    def _overlap_target(self, directory, req=None):
        from sglang_remote_attn.decode_overlap import DecodeOverlap

        pl = self.hooks.pre_layerwise
        controller = pl.Controller.__new__(pl.Controller)
        controller.tp_size, controller.rank = 1, 0
        controller.graph_fast_path = True
        controller.config = pl.Config("remote", Path(directory))
        controller.scheduler = SimpleNamespace(enable_overlap=True)
        controller.decode_overlap = DecodeOverlap(controller)
        controller.scheduler._pre_layerwise = controller
        self.addCleanup(controller.decode_overlap.retirement.close)
        freed = []
        req = req or SimpleNamespace(rid="overlap")
        receiver = SimpleNamespace(done=False, clear=lambda: freed.append("receiver"))
        receiver.poll = lambda: 4 if receiver.done else 2
        target = pl.Target(
            controller,
            SimpleNamespace(
                metadata_buffers=SimpleNamespace(bootstrap_room=torch.ones((1, 1))),
                req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=freed.append),
            ),
            SimpleNamespace(req=req, kv_receiver=receiver, metadata_buffer_index=0),
            Path(directory) / "target",
            "g",
            pl.LayerCoverage(2, 1),
            admission_id=1,
        )
        controller.decode_overlap.retirement.register(target)
        req._pre_layerwise_target = target
        controller.targets = [target]
        patcher = patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                    KVPoll=SimpleNamespace(Success=4, Failed=0)
                )
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return controller, target, freed

    def _retire_step(self, controller, *, finalize=True):
        controller.retire_targets()
        protocol = controller.decode_overlap.retirement
        command = protocol.poll_command()
        if command is not None:
            protocol.arm(command)
        if finalize:
            protocol.finalize()
        return command

    def test_ordered_retirement_preserves_native_receiver_and_waits_for_command(self):
        from sglang_remote_attn.decode_overlap import after_result
        from sglang_remote_attn.retirement import split_controls, wrap_receiver

        native = _source_class(
            "managers/scheduler_components/request_receiver.py",
            "SchedulerRequestReceiver",
            strip_method_decorators=True,
        )
        native._broadcast_reqs_across_ranks.__globals__["get_parallel"] = lambda: (
            SimpleNamespace(attn_dp_enabled=False, tp_size=1)
        )
        values = {f.name: None for f in dataclasses.fields(native)}
        values["max_recv_per_poll"] = 17
        values["get_last_batch"] = lambda: "native batch"
        receiver = native(**values)
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory)
            protocol = controller.decode_overlap.retirement
            target.decode_req.kv_receiver.done = True
            for _ in range(32):
                controller.retire_targets()
            self.assertEqual(protocol.final_votes, 0)
            self.assertEqual(freed, [])
            wrapped = wrap_receiver(receiver, protocol)
            for field in dataclasses.fields(native):
                self.assertIs(
                    getattr(wrapped, field.name), getattr(receiver, field.name)
                )
            messages = wrapped._broadcast_reqs_across_ranks(
                ["request"], ["local-abort"]
            )
            ordinary, commands = split_controls(messages)
            self.assertEqual(ordinary, ["request", "local-abort"])
            self.assertEqual(len(commands), 1)
            with patch.object(
                self.runtime, "maybe_handle_handoff_control", return_value=None
            ):
                result = self.hooks._process_requests(
                    lambda scheduler, reqs: reqs, controller.scheduler, messages
                )
            self.assertEqual(result, ordinary)
            self.assertFalse(target.closed)  # Intake arms; it does not enter a vote.
            after_result(lambda *a: None, controller.scheduler, None, None)
            self.assertTrue(target.closed)
            self.assertEqual(freed, ["receiver", 0])
            self.assertEqual(protocol.final_votes, 1)
            self.assertEqual(protocol.last_seq, 1)

    def test_ordered_retirement_rechecks_gpu_and_avoids_retry_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory)
            protocol = controller.decode_overlap.retirement
            target.decode_req.kv_receiver.done = True
            controller.retire_targets()
            command = protocol.poll_command()
            # A later forward invalidates the earlier advisory GPU snapshot.
            gpu = SimpleNamespace(done=False)
            result = SimpleNamespace(
                copy_done=SimpleNamespace(query=lambda: gpu.done),
                delay_sample_func=None,
            )
            batch = SimpleNamespace(
                reqs=[target.decode_req.req],
                forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
            )
            self.hooks.pre_layerwise.run_batch(
                lambda *a: result, controller.scheduler, batch
            )
            protocol.arm(command)
            protocol.finalize()
            self.assertEqual(protocol.retries, 1)
            self.assertEqual(freed, [])
            for _ in range(32):
                self._retire_step(controller)
            self.assertEqual(protocol.final_votes, 1)
            gpu.done = True
            self._retire_step(controller)
            self.assertEqual(protocol.final_votes, 2)
            self.assertEqual(protocol.commits, 1)
            self.assertEqual(target.gpu_proven_forwards, 1)
            self.assertEqual(freed, ["receiver", 0])

    def test_ordered_retirement_cancel_during_pending_retraction_does_not_free_twice(
        self,
    ):
        from sglang_remote_attn.decode_overlap import before_retract

        f = self._radix()
        f.cache.disable = True
        req, _ = f.request([2, 3], "cancel-retract")
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory, req)
            protocol = controller.decode_overlap.retirement
            target.decode_req.kv_receiver.done = True
            controller.retire_targets()
            command = protocol.poll_command()  # Snapshot says no queued release.
            self.hooks._release_kv(f.release_kv_cache, req, f.cache, checkpoint=False)
            protocol.arm(command)
            self.assertFalse(
                before_retract(
                    lambda **kw: self.fail("backup after cancellation free"), req=req
                )
            )
            self.assertEqual(protocol.last_seq, 1)
            self.assertEqual(protocol.final_votes, 1)
            self.assertEqual(freed, ["receiver", 0])
            self.assertEqual(f.allocator.available_size(), 32)
            self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)

    def test_ordered_retirement_rejects_stale_identity_and_out_of_order_commands(self):
        for corruption in ("epoch", "seq", "generation", "admission", "duplicates"):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                controller, target, freed = self._overlap_target(directory)
                protocol = controller.decode_overlap.retirement
                target.decode_req.kv_receiver.done = True
                controller.retire_targets()
                command = protocol.poll_command()
                if corruption == "epoch":
                    command["epoch"] = "old-run"
                elif corruption == "seq":
                    command["seq"] += 1
                elif corruption == "generation":
                    command["targets"][0]["generations"][0] = "old-generation"
                elif corruption == "admission":
                    target.admission_id += 1
                else:
                    command["targets"].append(copy.deepcopy(command["targets"][0]))
                protocol.arm(command)
                with self.assertRaisesRegex(RuntimeError, "validation"):
                    protocol.finalize()
                self.assertFalse(target.closed)
                self.assertEqual(freed, [])
                self.assertEqual(protocol.final_votes, 1)
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory)
            protocol = controller.decode_overlap.retirement
            target.decode_req.kv_receiver.done = True
            command = self._retire_step(controller)
            protocol.arm(command)
            with self.assertRaisesRegex(RuntimeError, "validation"):
                protocol.finalize()
            self.assertEqual(freed, ["receiver", 0])
            self.assertEqual(protocol.last_seq, 1)

    def test_ordered_retirement_notifications_cannot_regress_or_retire_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory)
            protocol = controller.decode_overlap.retirement
            target.decode_req.kv_receiver.done = True
            controller.retire_targets()
            old = copy.deepcopy(protocol.reports[1][0])
            newer = {**old, "release": True, "version": old["version"] + 1}
            protocol._accept(newer)
            protocol._accept(old)
            protocol._accept({**newer, "epoch": "old-run", "version": 999})
            protocol._accept(
                {**newer, "identity": {**newer["identity"], "id": 99}, "version": 999}
            )
            self.assertEqual(protocol.reports[1][0], newer)
            self.assertEqual(protocol.final_votes, 0)
            self.assertEqual(freed, [])
            self.assertFalse(target.closed)

    def test_ordered_retirement_local_mailbox_lifecycle(self):
        from sglang_remote_attn.retirement import Mailbox

        epoch = uuid.uuid4().hex
        receiver, sender = Mailbox(epoch, 0), Mailbox(epoch, 1)
        self.addCleanup(receiver.close)
        self.addCleanup(sender.close)
        notice = {"epoch": epoch, "rank": 1, "version": 1}
        self.assertTrue(sender.send(notice))
        self.assertEqual(list(receiver.receive()), [notice])
        with self.assertRaisesRegex(RuntimeError, "mailbox limit"):
            sender.send({"data": "x" * 65536})
        receiver.close()
        self.assertFalse(receiver.path.exists())
        self.assertFalse(sender.send(notice))  # Startup/restart absence is retryable.

    def test_decode_overlap_native_loop_serializes_remote_and_restores_local(self):
        """Native PD loop must drain before Remote launches, then pipeline Local again."""
        from collections import deque

        from sglang_remote_attn.decode_overlap import disable_overlap

        native = _source_class(
            "disaggregation/decode.py",
            "SchedulerDisaggregationDecodeMixin",
            {"event_loop_overlap_disagg_decode"},
        )
        native.event_loop_overlap_disagg_decode.__wrapped__.__globals__["deque"] = deque
        predicate = _source_class(
            "managers/scheduler.py", "Scheduler", {"is_disable_overlap_for_batch"}
        ).is_disable_overlap_for_batch
        predicate.__globals__["envs"] = SimpleNamespace(
            SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=SimpleNamespace(
                get=lambda: False
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory)
            controller.targets = []
            scheduler = native()
            scheduler.enable_overlap = True
            controller.scheduler = scheduler
            scheduler._pre_layerwise = controller
            scheduler.gracefully_exit = scheduler._engine_paused = False
            scheduler.require_mlp_sync = False
            scheduler.running_batch = None
            scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
                prefetch_prefill_dp_rank_queries=lambda: None
            )
            scheduler.ngram_embedding_manager = SimpleNamespace(
                prepare_for_forward=lambda b, **kw: b
            )
            scheduler.chunked_req, scheduler.is_generation = None, True
            events, launches, processed = {}, [], []
            iteration = -1

            def batch(name):
                b = SimpleNamespace(
                    name=name,
                    reqs=[target.decode_req.req],
                    spec_algorithm=SimpleNamespace(is_none=lambda: True),
                    forward_mode=SimpleNamespace(
                        is_prebuilt=lambda: False, is_extend=lambda: False
                    ),
                )
                b.copy = lambda: b
                return b

            batches = [
                batch("local-0"),
                batch("remote-1"),
                batch("remote-2"),
                None,
                batch("local-3"),
                batch("local-4"),
                None,
            ]

            def ingest():
                nonlocal iteration
                iteration += 1
                if iteration == 1:
                    controller.targets.append(target)
                if iteration == 3:
                    target.decode_req.kv_receiver.done = True
                self._retire_step(controller, finalize=False)
                if iteration == len(batches) - 1:
                    scheduler.gracefully_exit = True

            def launch(scheduler, b):
                launches.append(
                    (b.name, [name for name, e in events.items() if not e.done])
                )
                event = SimpleNamespace(done=False)
                event.query = lambda: event.done
                event.synchronize = lambda: setattr(event, "done", True)
                events[b.name] = event
                return SimpleNamespace(copy_done=event, delay_sample_func=None)

            def process(b, result):
                result.copy_done.synchronize()  # Device boundary of native result processing.
                processed.append(b.name)

            scheduler.ingest_requests = ingest
            scheduler.process_decode_queue = controller.retire_targets
            scheduler.get_next_disagg_decode_batch_to_run = lambda **kw: (
                SimpleNamespace(running_batch=None, batch_to_run=batches[iteration])
            )
            scheduler.is_disable_overlap_for_batch = lambda b, last_batch: (
                disable_overlap(predicate, scheduler, b, last_batch)
            )
            scheduler.run_batch = lambda b: self.hooks.pre_layerwise.run_batch(
                launch, scheduler, b
            )
            from sglang_remote_attn.decode_overlap import after_result

            scheduler.process_batch_result = lambda *args: after_result(
                lambda scheduler, *a: process(*a), scheduler, *args
            )
            scheduler.launch_batch_sample_if_needed = lambda *args: after_result(
                lambda *a: None, scheduler, *args
            )
            scheduler._apply_war_barrier = lambda: None  # GPU stream boundary.
            scheduler.on_idle = lambda: None
            scheduler.event_loop_overlap_disagg_decode()
            self.assertEqual(
                launches,
                [
                    ("local-0", []),
                    ("remote-1", []),
                    ("remote-2", []),
                    ("local-3", []),
                    ("local-4", ["local-3"]),
                ],
            )
            self.assertEqual(processed, [b.name for b in batches if b])
            self.assertEqual(controller.targets, [])
            self.assertEqual(freed, ["receiver", 0])
            self.assertEqual(target.gpu_proven_forwards, 2)
            self.assertEqual(target.forward_count, 2)
            # Empty Local never reaches result-proof or collective work.
            with (
                patch(
                    "sglang_remote_attn.decode_overlap.BatchProof.ready",
                    side_effect=AssertionError("Local proof query"),
                ),
                patch.object(
                    torch.distributed,
                    "all_reduce",
                    side_effect=AssertionError("Local vote"),
                ),
            ):
                for _ in range(16):
                    controller.retire_targets()
                    self.assertFalse(
                        disable_overlap(predicate, scheduler, batches[4], batches[5])
                    )
                    scheduler.run_batch(batches[4])

    def test_decode_overlap_defers_native_kv_free_until_gpu_and_io_drained(self):
        f = self._radix()
        f.cache.disable = True  # D retains native disabled-radix release.
        req, _ = f.request([2, 3], "decode")
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory, req)
            gpu = SimpleNamespace(done=False)
            gpu.query = lambda: gpu.done
            result = SimpleNamespace(copy_done=gpu, delay_sample_func=None)
            batch = SimpleNamespace(
                reqs=[req], forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
            )
            self.hooks.pre_layerwise.run_batch(
                lambda *a: result, controller.scheduler, batch
            )
            self.hooks._release_kv(f.release_kv_cache, req, f.cache, checkpoint=False)
            self.hooks._release_kv(f.release_kv_cache, req, f.cache, checkpoint=False)
            self._retire_step(controller)
            self.assertEqual(f.allocator.available_size(), 30)
            target.decode_req.kv_receiver.done = True
            self._retire_step(controller)  # Native writes are complete; GPU is not.
            self.assertEqual(freed, [])
            self.assertEqual(f.allocator.available_size(), 30)
            gpu.done = True
            from sglang_remote_attn.decode_overlap import reap

            # The native queue can return early while retractions need memory.
            # Reap before that gate so a held KV free can make progress.
            process_queue = _source_class(
                "disaggregation/decode.py",
                "SchedulerDisaggregationDecodeMixin",
                {"process_decode_queue"},
                strip_method_decorators=True,
            ).process_decode_queue
            process_queue.__globals__["get_disagg"] = lambda: SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            )
            scheduler = controller.scheduler
            scheduler.enable_decode_hicache = False
            scheduler.waiting_queue = []
            scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
                resume_retracted_reqs=lambda: [], retracted_queue=[object()]
            )
            queue = SimpleNamespace(scheduler=scheduler)
            scheduler.disagg_decode_transfer_queue = SimpleNamespace(
                resolve_deferred_releases=lambda: reap(lambda q: None, queue)
            )
            process_queue(scheduler)
            self._retire_step(controller)
            self.assertEqual(freed, ["receiver", 0])
            self.assertEqual(f.allocator.available_size(), 32)
            self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)
            self.assertTrue(req.kv.is_kv_released)

    def test_decode_overlap_completion_rejects_empty_events_and_copy_errors(self):
        from sglang_remote_attn.decode_overlap import BatchProof

        # A new CUDA event can query True before delayed sampling records it.
        result = SimpleNamespace(
            delay_sample_func=object(), copy_done=SimpleNamespace(query=lambda: True)
        )
        proof = BatchProof(result)
        self.assertFalse(proof.ready())
        result.delay_sample_func = None
        result.copy_done = None
        with self.assertRaisesRegex(RuntimeError, "no GPU completion"):
            proof.ready()
        copy_done = _source_class("managers/async_d2h_copy_worker.py", "HostCopyDone")
        copy_done.__init__.__globals__["threading"] = threading
        result.copy_done = copy_done()
        self.assertFalse(proof.ready())
        result.copy_done.set_done(error=ValueError("copy failed"))
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            proof.ready()
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            BatchProof().ready()

    def test_decode_overlap_failed_forward_retains_target_and_pending_release(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, target, freed = self._overlap_target(directory)
            req = target.decode_req.req
            native_prebuilt = _source_class(
                "disaggregation/decode.py",
                "SchedulerDisaggregationDecodeMixin",
                {"_run_batch_prebuilt"},
            )._run_batch_prebuilt
            native_prebuilt.__globals__["GenerationBatchResult"] = _source_class(
                "managers/utils.py", "GenerationBatchResult"
            )
            prebuilt = SimpleNamespace(
                reqs=[req],
                inner_idle_batch=None,
                forward_mode=SimpleNamespace(is_prebuilt=lambda: True),
            )
            result = self.hooks.pre_layerwise.run_batch(
                native_prebuilt, controller.scheduler, prebuilt
            )
            self.assertIsNone(result.copy_done)
            self.assertEqual(target.forward_proofs, [])
            batch = SimpleNamespace(
                reqs=[req], forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
            )

            def fail(*args):
                raise RuntimeError("partial forward")

            with self.assertRaisesRegex(RuntimeError, "partial forward"):
                self.hooks.pre_layerwise.run_batch(fail, controller.scheduler, batch)
            self.assertTrue(
                self.hooks.pre_layerwise.defer_release(req, lambda: freed.append("KV"))
            )
            target.decode_req.kv_receiver.done = True
            with self.assertRaisesRegex(RuntimeError, "validation"):
                self._retire_step(controller)
            self.assertEqual(freed, [])
            self.assertEqual(controller.targets, [target])
            self.assertFalse(target.closed)

    def test_decode_overlap_failed_native_release_never_retries_allocator_mutation(
        self,
    ):
        f = self._radix()
        f.cache.disable = True
        req, _ = f.request([2, 3], "failed-release")
        with tempfile.TemporaryDirectory() as directory:
            controller, target, _ = self._overlap_target(directory, req)
            target.decode_req.kv_receiver.done = True

            def partial_release():
                f.release_kv_cache(req, f.cache, checkpoint=False)
                raise RuntimeError("native release completion unknown")

            self.assertTrue(
                self.hooks.pre_layerwise.defer_release(req, partial_release)
            )
            with self.assertRaisesRegex(RuntimeError, "completion unknown"):
                self._retire_step(controller)
            with self.assertRaisesRegex(RuntimeError, "validation"):
                self._retire_step(controller)
            self.assertEqual(f.allocator.available_size(), 32)
            self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)
            self.assertEqual(controller.targets, [target])
            self.assertTrue((target.path / "error.json").exists())

    def test_decode_overlap_retraction_drains_before_native_backup_and_reset(self):
        from sglang_remote_attn.decode_overlap import before_retract

        f = self._radix()
        f.cache.disable = True
        req, _ = f.request([2, 3], "retract")
        req.finished = lambda: False
        req.input_embeds, req.retraction_count = None, 0
        native_reset = _source_class(
            "managers/schedule_batch.py", "Req", {"reset_for_retract"}
        ).reset_for_retract
        req.reset_for_retract = lambda: native_reset(req)
        native_release = _source_function("managers/schedule_batch.py", "release_req")
        with tempfile.TemporaryDirectory() as directory:
            controller, target, _ = self._overlap_target(directory, req)
            ready = SimpleNamespace(done=False)
            result = SimpleNamespace(
                copy_done=SimpleNamespace(query=lambda: ready.done),
                delay_sample_func=None,
            )
            batch = SimpleNamespace(
                reqs=[req], forward_mode=SimpleNamespace(is_prebuilt=lambda: False)
            )
            self.hooks.pre_layerwise.run_batch(
                lambda *a: result, controller.scheduler, batch
            )
            # Timeout must leave the native backup/reset untouched and preserve KV.
            controller.config = dataclasses.replace(controller.config, drain_timeout=0)
            with self.assertRaisesRegex(RuntimeError, "validation"):
                before_retract(lambda **kw: self.fail("unsafe native backup"), req=req)
            self.assertEqual(req.retraction_count, 0)
            self.assertTrue(req.kv.holds_kv)
            controller.config = dataclasses.replace(controller.config, drain_timeout=1)
            receiver = target.decode_req.kv_receiver

            def device_progress(_):
                ready.done = receiver.done = True

            backed_up = []

            def backup(*args):
                self.assertTrue(target.closed)
                self.assertTrue(ready.done)
                backed_up.append(req.kv.req_pool_idx)
                return True

            native_release.__globals__.update(
                get_disagg=lambda: SimpleNamespace(
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup="host_pool",
                ),
                backup_kv_cache=backup,
                release_kv_cache=lambda req, cache, **kw: self.hooks._release_kv(
                    f.release_kv_cache, req, cache, **kw
                ),
                evict_from_tree_cache=lambda cache, n: cache.evict(
                    f.EvictParams(num_tokens=n)
                ),
                envs=SimpleNamespace(
                    SGLANG_RETRACT_DECODE_STEPS=SimpleNamespace(get=lambda: 1)
                ),
            )
            with patch.object(
                self.hooks.pre_layerwise.time, "sleep", side_effect=device_progress
            ):
                self.assertTrue(
                    before_retract(
                        native_release,
                        req=req,
                        remaing_req_count=1,
                        req_to_token_pool=f.cache.req_to_token_pool,
                        token_to_kv_pool_allocator=f.allocator,
                        tree_cache=f.cache,
                        hisparse_coordinator=None,
                    )
                )
            self.assertEqual(len(backed_up), 1)
            self.assertEqual(req.retraction_count, 1)
            self.assertTrue(req.is_retracted)
            self.assertFalse(req.kv.holds_kv)
            self.assertEqual(f.allocator.available_size(), 32)

    def test_decode_overlap_finished_overshoot_keeps_remote_state_until_retirement(
        self,
    ):
        """A prepared overshoot must not use incomplete local KV after Req finishes."""
        closed = []
        owner = SimpleNamespace(decode_overlap=object())
        target = SimpleNamespace(owner=owner, closed=False)
        req = SimpleNamespace(
            finished=lambda: True,
            kv=SimpleNamespace(req_pool_idx=0),
            _pre_layerwise_target=target,
        )
        state = SimpleNamespace(
            req=req,
            req_pool_idx=0,
            _admitted=True,
            close=lambda: closed.append("closed"),
        )
        with (
            patch.object(self.runtime, "_TARGET_STATES", [state]),
            patch.object(self.runtime, "_snapshot_state", return_value=None),
        ):
            self.assertEqual(self.runtime._bind_and_select_states([0]), [state])
            self.assertEqual(closed, [])
            req.kv.req_pool_idx = 1
            with self.assertRaisesRegex(RuntimeError, "row changed"):
                self.runtime._bind_and_select_states([1])
            req.kv.req_pool_idx = 0
            target.closed = True
            self.assertEqual(self.runtime._bind_and_select_states([0]), [])
            self.assertEqual(closed, ["closed"])

    def test_progressive_graph_lease_skips_per_forward_collectives(self):
        """TP-agreed leases, including locally complete ones, own the eager window."""
        controller = SimpleNamespace(graph_fast_path=True, targets=[])
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(_pre_layerwise=controller)
        )
        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(
                tp_size=2, tp_rank=0, tp_group=SimpleNamespace(cpu_group=None)
            )
        )
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": context}),
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=AssertionError("unexpected per-forward vote"),
            ),
            patch.object(
                self.runtime,
                "has_remote_work",
                side_effect=AssertionError(
                    "rank-local readiness is not a shared certificate"
                ),
            ),
        ):
            for hook in (self.hooks._decode_graph, self.hooks._prefill_graph):
                for _ in range(16):
                    self.assertTrue(hook(lambda *a: True, runner, None))
                controller.targets.append(SimpleNamespace(closed=True))
                self.assertFalse(hook(lambda *a: True, runner, None))
                # A local close cannot lift the lease before TP retirement.
                controller.targets.clear()
                self.assertTrue(hook(lambda *a: True, runner, None))
                self.assertFalse(hook(lambda *a: False, runner, None))

    def test_unmanaged_import_revokes_progressive_graph_fast_path(self):
        """Manual imports cannot silently bypass the progressive lease protocol."""
        for owned in (True, False):
            with self.subTest(owned=owned):
                controller = SimpleNamespace(graph_fast_path=True, targets=[])
                model = SimpleNamespace(modules=lambda: [])
                runner = SimpleNamespace(_pre_layerwise=controller, model=model)
                scheduler = SimpleNamespace(
                    _pre_layerwise=controller,
                    tp_worker=SimpleNamespace(model_runner=runner),
                )
                state = SimpleNamespace(req=SimpleNamespace())
                if owned:
                    state.req._pre_layerwise_target = SimpleNamespace(
                        owner=controller, state=state
                    )
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "sglang.srt.layers.radix_attention": SimpleNamespace(
                                RadixAttention=type("RadixAttention", (), {})
                            )
                        },
                    ),
                    patch.object(self.runtime, "_TARGET_STATES", []),
                    patch.object(self.runtime, "_FB_CACHE", {}),
                    patch.object(self.runtime, "PROFILE", False),
                ):
                    self.runtime._install_attention_wrappers(scheduler, state)
                    self.assertEqual(controller.graph_fast_path, owned)
                    self.assertEqual(self.runtime._TARGET_STATES, [state])
        # Broadcast control revokes on every rank before a shard might fail
        # early, rather than relying on that shard reaching wrapper install.
        controller.graph_fast_path = True
        controller.poll_sources = lambda: None
        runtime = SimpleNamespace(
            split_handoff_controls=self.runtime.split_handoff_controls,
            execute_handoff_control=lambda *a: self.assertFalse(
                controller.graph_fast_path
            ),
            maybe_handle_handoff_control=lambda *a: None,
        )
        sentinel = {self.runtime._HANDOFF_CTL_KEY: True, "data": {"cmd": "import"}}
        with patch.object(self.hooks, "_runtime", return_value=runtime):
            self.hooks._process_requests(lambda *a: None, scheduler, [sentinel])
            controller.decode_overlap = object()
            controller.graph_fast_path = True
            for command in ("import", "reset", "release", "abort_import"):
                with (
                    self.subTest(command=command),
                    self.assertRaisesRegex(
                        RuntimeError, "native progressive admissions only"
                    ),
                ):
                    message = {
                        self.runtime._HANDOFF_CTL_KEY: True,
                        "data": {"cmd": command},
                    }
                    self.hooks._process_requests(lambda *a: None, scheduler, [message])
            self.assertTrue(controller.graph_fast_path)

    def _segmented_graph_fixture(self):
        sg = self.hooks.segmented_graph
        trace = []
        controller = SimpleNamespace(graph_fast_path=True, targets=[])

        class Event:
            broken = False

            def record(self, stream):
                if self.broken:
                    raise RuntimeError("event record failed")
                trace.append("record")

            def synchronize(self):
                trace.append("sync")

        class Attention:
            def __init__(self, index):
                self.layer_id, self.tp_q_head_num, self.v_head_dim = index, 1, 4

            def forward(self, q, k, v, batch, **kw):
                return q * 2

        modules = [Attention(0), Attention(1)]
        mr = SimpleNamespace(
            _pre_layerwise=controller,
            model_config=SimpleNamespace(num_hidden_layers=2),
            model=SimpleNamespace(modules=lambda: modules),
        )
        runner = SimpleNamespace(
            model_runner=mr,
            device_module=SimpleNamespace(Event=Event, current_stream=lambda: "stream"),
        )

        class Bank:
            def __init__(self, name):
                self.name, self.keys, self.outputs, self.capturing = name, {}, {}, False

            def capture_session(self, stream):
                return nullcontext()

            def capture_one(self, key, fn, *a):
                self.capturing = True
                try:
                    self.outputs[key.size] = fn()
                    self.keys[key.size] = key
                finally:
                    self.capturing = False

            def can_run(self, batch, key):
                return self.keys.get(key.size) == key

            def replay_session(self):
                return nullcontext()

            def replay(self, key, batch):
                if self.name == "remote":
                    q = torch.ones(batch.batch_size, 4)
                    for module in modules:
                        q = backend._attention(module, q, q, q, save_kv_cache=True)
                    self.outputs[key.size].copy_(q)
                return self.outputs[key.size]

            def cleanup(self):
                trace.append("clean_" + self.name)

        full, remote = Bank("full"), Bank("remote")
        backend = sg.RemoteGraphBackend(
            runner,
            full,
            remote,
            lambda fn: fn,
            lambda: remote.capturing,
            lambda: trace.append("restore"),
        )
        mr._remote_segmented_backend = backend
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.layers.radix_attention": SimpleNamespace(
                    RadixAttention=Attention
                )
            },
        ):
            sg.install_attention(mr)
        runner.backend = backend

        def batch(size, marker=0):
            return SimpleNamespace(
                batch_size=size,
                input_ids=torch.zeros(size, dtype=torch.int64),
                input_embeds=None,
                replace_embeds=None,
                forward_mode=SimpleNamespace(is_decode=lambda: True),
                marker=marker,
            )

        def attention(module, q, k, v, live, **kwargs):
            trace.append(("rpc", live.marker, module.layer_id))
            return q + live.marker

        runtime = SimpleNamespace(
            has_target_states=lambda: bool(controller.targets),
            _row_meta=lambda fb: {"states": []},
            _handoff_attention_forward=attention,
        )
        with (
            patch.object(sg, "_runtime", return_value=runtime),
            backend.capture_session("stream"),
        ):
            for size in (8, 4, 2, 1):
                captured = batch(size, -100)

                def forward():
                    q = torch.ones(captured.batch_size, 4)
                    for module in modules:
                        q = module.forward(q, q, q, captured)
                    return q

                key = SimpleNamespace(
                    size=size,
                    stream_idx=None,
                    variant_label=None,
                    attention_variant=None,
                )
                backend.capture_one(key, forward)
        return SimpleNamespace(
            backend=backend,
            controller=controller,
            runner=runner,
            full=full,
            remote=remote,
            trace=trace,
            batch=batch,
            runtime=runtime,
        )

    def test_remote_segments_capture_only_three_buckets_and_use_live_requests(self):
        """Captured dummy requests must never leak into a later RPC callback."""
        f = self._segmented_graph_fixture()
        b = f.backend
        self.assertEqual(sorted(f.full.keys), [1, 2, 4, 8])
        self.assertEqual(sorted(f.remote.keys), [1, 2, 4])
        self.assertEqual(b.manifest(), ((1, (0, 1)), (2, (0, 1)), (4, (0, 1))))
        self.assertIsNone(b.capture_batch)
        self.assertFalse(any(isinstance(item, tuple) for item in f.trace))
        f.controller.targets.append(object())
        sg = self.hooks.segmented_graph
        with (
            patch.object(sg, "_runtime", return_value=f.runtime),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": SimpleNamespace(
                        get_parallel=lambda: SimpleNamespace(tp_size=1, tp_rank=0)
                    )
                },
            ),
        ):
            for size in (1, 2, 4):
                ptr = f.remote.outputs[size].data_ptr()
                for marker in (3, 7):
                    live = f.batch(size, marker)
                    self.assertTrue(
                        self.hooks._decode_graph(lambda *a: True, f.runner, live)
                    )
                    with b.execution(live), b.replay_session():
                        output = b.replay(f.remote.keys[size], live)
                    self.assertEqual(output.data_ptr(), ptr)
                    self.assertTrue(
                        torch.equal(output, torch.full((size, 4), 1 + marker * 2))
                    )
            for size in (3, 8):
                self.assertFalse(b.supports_remote(f.batch(size)))
                self.assertFalse(
                    self.hooks._decode_graph(lambda *a: True, f.runner, f.batch(size))
                )
                with self.assertRaisesRegex(RuntimeError, "fall back"):
                    with b.execution(f.batch(size)):
                        self.fail("unsupported batch entered a graph")
                self.assertIsNone(b.active_batch)
                self.assertFalse(b.use_remote)
            f.controller.targets.clear()
            before = len(f.trace)
            live = f.batch(4, 999)
            self.assertTrue(self.hooks._decode_graph(lambda *a: True, f.runner, live))
            with b.execution(live), b.replay_session():
                output = b.replay(f.full.keys[4], live)
            self.assertTrue(torch.equal(output, torch.full((4, 4), 4.0)))
            self.assertEqual(
                len(f.trace), before
            )  # No RPC or event on Local full replay.
        b.cleanup()
        self.assertEqual(f.trace[-3:], ["sync", "clean_remote", "clean_full"])
        self.assertEqual(b.replays, {"remote": 6, "full": 1})

    def test_remote_segments_fail_without_retry_and_retain_unfenced_banks(self):
        """An RPC/event failure cannot restart the token or free queued segment storage."""
        f = self._segmented_graph_fixture()
        f.controller.targets.append(object())
        calls = []

        def broken(module, q, *a, **kw):
            calls.append(module.layer_id)
            raise RuntimeError("RPC failed")

        f.runtime._handoff_attention_forward = broken
        with patch.object(
            self.hooks.segmented_graph, "_runtime", return_value=f.runtime
        ):
            with self.assertRaisesRegex(RuntimeError, "RPC failed"):
                live = f.batch(1)
                with f.backend.execution(live), f.backend.replay_session():
                    f.backend.replay(f.remote.keys[1], live)
            self.assertEqual(calls, [0])
            self.assertIsNone(f.backend.active_batch)
            self.assertFalse(f.backend.use_remote)
            self.assertEqual(f.backend.replays["remote"], 0)
            f.backend.done.broken = True
            with self.assertRaisesRegex(RuntimeError, "event record failed"):
                with f.backend.execution(f.batch(1)):
                    pass
        with self.assertRaisesRegex(RuntimeError, "retaining graph banks"):
            f.backend.cleanup()
        self.assertFalse(f.backend.closed)
        self.assertFalse(
            any(isinstance(item, str) and item.startswith("clean_") for item in f.trace)
        )

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

    def test_progressive_startup_accepts_graphs_without_relaxing_other_constraints(
        self,
    ):
        """Graph-enabled TP must pass both startup validators; overlap remains gated."""
        pl = self.hooks.pre_layerwise
        parallel = SimpleNamespace(
            tp_size=2,
            pp_size=1,
            dp_size=1,
            attn_cp_size=1,
            dcp_enabled=False,
            attn_dp_enabled=False,
        )
        schedule = SimpleNamespace(
            disable_overlap_schedule=True, page_size=1, enable_mixed_chunk=False
        )
        context = SimpleNamespace(
            get_parallel=lambda: parallel,
            get_device=lambda: SimpleNamespace(device="cuda"),
            get_schedule=lambda: schedule,
            get_memory=lambda: SimpleNamespace(
                disable_radix_cache=True,
                enable_unified_memory=False,
                enable_hierarchical_cache=False,
            ),
            get_exec=lambda: SimpleNamespace(
                overlap=SimpleNamespace(enable_two_batch_overlap=False),
                graph=SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend="full"),
                        prefill=SimpleNamespace(backend="breakable"),
                    )
                ),
            ),
            get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
            get_disagg=lambda: SimpleNamespace(
                disaggregation_mode="prefill",
                enable_pd_role_switch=False,
                disaggregation_decode_allocation_policy="early",
            ),
        )
        model = type(
            "Qwen3ForCausalLM", (), {"forward_split_prefill": lambda *a: None}
        )()
        controller = pl.Controller.__new__(pl.Controller)
        controller.scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(model=model, lora_manager=None)
            ),
            kv_checksum_computer=None,
            enable_pdmux=False,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": context,
                    "mooncake.engine": SimpleNamespace(
                        TransferEngine=SimpleNamespace(send_probe=lambda: None)
                    ),
                },
            ),
        ):
            controller.config = pl.Config("remote", Path(directory) / "journal")
            self.hooks.validate_config()
            controller.validate()
            self.assertTrue(controller.config.directory.is_dir())
            memory = context.get_memory()
            memory.disable_radix_cache = False
            context.get_memory = lambda: memory
            disagg = context.get_disagg()
            context.get_disagg = lambda: disagg
            for mode in ("remote", "layerwise", "off"):
                for role in ("prefill", "decode"):
                    with (
                        self.subTest(mode=mode, role=role),
                        patch.dict(os.environ, {"HANDOFF_PRE_LAYERWISE": mode}),
                    ):
                        disagg.disaggregation_mode = role
                        if role == "prefill" and mode != "off":
                            self.hooks.validate_config()
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "disable-radix-cache"
                            ):
                                self.hooks.validate_config()
            schedule.disable_overlap_schedule = False
            memory.disable_radix_cache = True
            for mode in ("remote", "layerwise", "off"):
                for role in ("prefill", "decode"):
                    with (
                        self.subTest(mode=mode, role=role),
                        patch.dict(os.environ, {"HANDOFF_PRE_LAYERWISE": mode}),
                    ):
                        disagg.disaggregation_mode = role
                        if (role == "decode" and mode != "off") or (
                            role == "prefill" and mode in ("layerwise", "remote")
                        ):
                            self.hooks.validate_config()
                        else:
                            with self.assertRaisesRegex(
                                RuntimeError, "disable-overlap-schedule"
                            ):
                                self.hooks.validate_config()
            execution = context.get_exec()
            execution.overlap.enable_two_batch_overlap = True
            context.get_exec = lambda: execution
            with self.assertRaisesRegex(RuntimeError, "two-batch overlap"):
                self.hooks.validate_config()

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

    @staticmethod
    def _gateway_namespace():
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
            ADMISSION_POLICY="decoupled",
            PD_PREFILL_URL="",
            WAVE_DEFAULT=True,
            KILL_AT="",
            DRAIN_KILL_AT="",
            DRAIN_NOOP=False,
            DRAIN_IMPORT_FAIL=False,
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
        return namespace

    def _gateway(self):
        namespace = self._gateway_namespace()
        gateway = namespace["Gateway"].__new__(namespace["Gateway"])
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        gateway.journal_path = Path(directory.name) / "journal.json"
        gateway.out = Path(directory.name) / "metrics.jsonl"
        gateway.proxy_dir = Path(directory.name) / "proxy"
        gateway.lock = threading.Lock()
        gateway.journal_lock = threading.Lock()
        gateway.admission_lock = threading.Lock()
        gateway.migration_lock = threading.RLock()
        gateway.group_lock = threading.Lock()
        gateway.groups = {}
        gateway.active = {}
        gateway.journal = {
            "long": {
                "rid": "long",
                "drid": "dst-long",
                "owner": "src",
                "phase": "accepted",
                "prompt_len": 32768,
                "max_new": 128,
                "finished": False,
                "epoch": "src",
            },
        }
        gateway.draining = False
        gateway._facts_at = 0.0
        gateway._src_facts = {"reqs": [{"rid": "long", "frozen": True}]}
        gateway._dst_facts = {"reqs": []}
        gateway._target_ok = True
        gateway._target_fail_since = None
        gateway._dst_fact_fails = 0
        gateway._target_dbg_t = time.monotonic()
        gateway._adopt_at = time.monotonic()
        gateway._admit_pending = 0
        gateway.drain_kv_baseline = 0
        calls = []

        def source_send(payload, **kwargs):
            calls.append(("src", copy.deepcopy(payload)))
            if payload["cmd"] == "status":
                return copy.deepcopy(gateway._src_facts)
            raise AssertionError(f"unexpected source RPC {payload}")

        def target_send(payload, **kwargs):
            calls.append(("dst", copy.deepcopy(payload)))
            if payload["cmd"] == "status":
                return {"reqs": [], "max_running": 8}
            if payload["cmd"] == "reserve":
                return {"errors": {}}
            if payload["cmd"] == "admit_ordinary":
                return {"admitted": True}
            if payload["cmd"] in ("submit", "release", "capacity_cancel"):
                return {}
            raise AssertionError(f"unexpected target RPC {payload}")

        gateway.src = SimpleNamespace(
            send=source_send,
            targets={},
            completed_targets={},
            transfer_stats={},
        )
        gateway.dst = SimpleNamespace(
            send=target_send,
            targets={},
            completed_targets={},
            wait_output=lambda *args: ([42], True, 0),
        )
        return gateway, namespace, calls

    def test_gateway_drain_fences_before_migration_lock_with_short_traffic(self):
        for arm in ("remote", "layer", "copy"):
            with self.subTest(arm=arm):
                gateway, namespace, calls = self._gateway()
                waiting = threading.Event()
                release = threading.Event()
                admitted = threading.Event()
                real_lock = threading.RLock()

                class MigrationGate:
                    def __enter__(self):
                        waiting.set()
                        if not release.wait(3):
                            raise TimeoutError("test migration gate")
                        real_lock.acquire()

                    def __exit__(self, *args):
                        real_lock.release()

                gateway.migration_lock = MigrationGate()
                errors = []
                handlers = [SimpleNamespace(wfile=io.BytesIO()) for _ in range(8)]
                original_send = gateway.dst.send
                capacity_lock = threading.Lock()
                tickets = set()
                reserved = set()

                def target(payload, **kwargs):
                    with capacity_lock:
                        cmd = payload["cmd"]
                        if cmd == "reserve":
                            self.assertEqual(tickets, set())
                            reserved.update(item["rid"] for item in payload["items"])
                        elif cmd == "admit_ordinary":
                            admitted.set()
                            self.assertIn("dst-long", reserved)
                            if len(tickets | reserved) >= 8:
                                return {"admitted": False, "reason": "slots"}
                            tickets.add(payload["rid"])
                        elif cmd == "release":
                            tickets.discard(payload["rid"])
                    return original_send(payload, **kwargs)

                gateway.dst.send = target

                def run(function, *args):
                    try:
                        function(*args)
                    except Exception as exc:
                        errors.append(exc)

                drain_thread = threading.Thread(
                    target=run, args=(gateway.start_drain, ["dst"], arm)
                )
                shorts = [
                    threading.Thread(
                        target=run,
                        args=(
                            gateway.stream,
                            handler,
                            {
                                "rid": f"short-{i}",
                                "prompt_ids": [1] * 256,
                                "max_new_tokens": 1,
                                "cut": 1000,
                            },
                        ),
                    )
                    for i, handler in enumerate(handlers)
                ]
                try:
                    drain_thread.start()
                    self.assertTrue(waiting.wait(3))
                    self.assertTrue(
                        gateway.journal_path.is_file(),
                        "drain fence not persisted before migration lock",
                    )
                    durable = json.loads(gateway.journal_path.read_text())
                    self.assertEqual(durable["__drain__"]["members"], ["long"])
                    self.assertEqual(gateway.src_worker_state(), "DRAINING")
                    duplicate = gateway.start_drain(["dst"], arm)
                    self.assertEqual(
                        duplicate["drain_id"], durable["__drain__"]["drain_id"]
                    )
                    self.assertTrue(gateway.admission_lock.acquire(blocking=False))
                    gateway.admission_lock.release()
                    for thread in shorts:
                        thread.start()
                    self.assertFalse(admitted.wait(0.05))
                finally:
                    release.set()
                    for thread in [drain_thread, *shorts]:
                        if thread.ident is not None:
                            thread.join(3)
                self.assertEqual(errors, [])
                self.assertTrue(all(not t.is_alive() for t in [drain_thread, *shorts]))
                self.assertEqual(gateway.journal["__drain__"]["members"], ["long"])
                self.assertEqual(
                    sum(role == "dst" and p["cmd"] == "submit" for role, p in calls), 8
                )
                self.assertFalse(any(role == "src" for role, p in calls))
                for handler in handlers:
                    self.assertIn(b'"token_id": 42', handler.wfile.getvalue())
                    events = [
                        json.loads(line[6:])
                        for line in handler.wfile.getvalue().decode().splitlines()
                        if line.startswith("data: {")
                    ]
                    self.assertTrue(all("error" not in event for event in events))
                self.assertEqual(tickets, set())
                # Continue the real gateway state machine through freeze,
                # import, local-ready proof and source release. Only worker
                # RPCs stand in for GPUs; no controller methods are replaced.
                epoch = gateway.journal["__drain__"]["epoch"]
                source_rows = [
                    {
                        "rid": "long",
                        "output_len": 32,
                        "kv_committed_len": 32799,
                        "frozen": False,
                    }
                ]
                target_states = []

                def source_rpc(payload, **kwargs):
                    cmd = payload["cmd"]
                    if cmd in ("status", "status_lite"):
                        return {"reqs": copy.deepcopy(source_rows)}
                    if cmd == "freeze_all":
                        self.assertEqual(payload["epoch"], epoch)
                        source_rows[0]["frozen"] = True
                        return {
                            "results": [
                                {
                                    "rid": "long",
                                    "output_len": 32,
                                    "migration_epoch": epoch,
                                    "transfer": {"token": "frozen-kv"},
                                }
                            ]
                        }
                    if cmd == "release":
                        self.assertTrue(target_states[0]["local_ready"])
                        source_rows.clear()
                        return {"released": True}
                    raise AssertionError(payload)

                def target_rpc(payload, **kwargs):
                    cmd = payload["cmd"]
                    if cmd == "status":
                        return {
                            "reqs": [],
                            "reserved": {"dst-long": {}},
                            "target_states": copy.deepcopy(target_states),
                        }
                    if cmd == "import_wave":
                        self.assertEqual(payload["items"][0]["epoch"], epoch)
                        target_states.append(
                            {
                                "handoff_id": "long",
                                "migration_epoch": epoch,
                                "new_tokens": 1,
                                "import_done_at": time.time(),
                                "local_ready": True,
                            }
                        )
                        return {}
                    raise AssertionError(payload)

                gateway.src.send = source_rpc
                gateway.dst.send = target_rpc
                for _ in range(3):
                    gateway._facts_at = 0.0
                    gateway._adopt_at = time.monotonic()
                    gateway._reconcile_once()
                final = gateway.drain_status()
                self.assertEqual(final["state"], "SAFE_TO_STOP")
                self.assertTrue(final["source_clean_all_ranks"])
                self.assertTrue(gateway.journal["long"]["source_released"])

    def test_gateway_admission_rechecks_fence_before_submit(self):
        gateway, namespace, calls = self._gateway()
        waiting = threading.Event()
        proceed = threading.Event()
        lock = threading.Lock()

        class AdmissionGate:
            def acquire(self, *args, **kwargs):
                return lock.acquire(*args, **kwargs)

            def release(self):
                lock.release()

            def __enter__(self):
                if threading.current_thread().name == "short-race":
                    waiting.set()
                    if not proceed.wait(3):
                        raise TimeoutError("test admission gate")
                lock.acquire()

            def __exit__(self, *args):
                lock.release()

        gateway.admission_lock = AdmissionGate()
        handler = SimpleNamespace(wfile=io.BytesIO())
        errors = []

        def short():
            try:
                gateway.stream(
                    handler,
                    {
                        "rid": "short",
                        "prompt_ids": [1],
                        "max_new_tokens": 1,
                        "cut": 1000,
                    },
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=short, name="short-race")
        try:
            thread.start()
            self.assertTrue(waiting.wait(3))
            self.assertEqual(calls, [])  # early admission chose source, no ticket
            gateway.start_drain(["dst"])
        finally:
            proceed.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(any(role == "src" for role, p in calls))
        target_commands = [p["cmd"] for role, p in calls]
        self.assertLess(
            target_commands.index("reserve"), target_commands.index("admit_ordinary")
        )
        submit = next(p for role, p in calls if p["cmd"] == "submit")
        self.assertEqual(submit["capacity_ticket"], "dst-short")
        self.assertIn(b'"token_id": 42', handler.wfile.getvalue())

    def test_gateway_preflight_restart_reserves_outside_admission_lock(self):
        gateway, namespace, calls = self._gateway()
        gateway.start_drain(["dst"])
        drain = gateway.journal["__drain__"]
        epoch = drain["epoch"]
        # Crash after reserve RPC, before preflight completion was persisted.
        drain.update(preflight_pending=True, state="RECONCILING")
        gateway._drain_persist(drain)
        reserved = threading.Event()
        finish = threading.Event()
        original = gateway.dst.send
        errors = []

        def target(payload, **kwargs):
            if payload["cmd"] == "reserve":
                reserved.set()
                if not finish.wait(3):
                    raise TimeoutError("test reserve gate")
            return original(payload, **kwargs)

        def reconcile():
            try:
                gateway._reconcile_once()
            except Exception as exc:
                errors.append(exc)

        gateway.dst.send = target
        thread = threading.Thread(target=reconcile)
        try:
            thread.start()
            self.assertTrue(reserved.wait(3))
            self.assertTrue(gateway.admission_lock.acquire(blocking=False))
            gateway.admission_lock.release()
            status = gateway.start_drain(["dst"])
            self.assertTrue(status["preflight_pending"])
            self.assertEqual(status["worker_state"], "DRAINING")
        finally:
            finish.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        reserves = [p for role, p in calls if p["cmd"] == "reserve"]
        self.assertEqual(len(reserves), 2)
        self.assertTrue(all(p["epoch"] == epoch for p in reserves))
        self.assertEqual(reserves[0]["items"], reserves[1]["items"])
        self.assertFalse(gateway.drain_status()["preflight_pending"])
        self.assertFalse(any(p["cmd"] == "freeze_many" for role, p in calls))

    def test_gateway_old_rollback_cannot_overwrite_new_drain_fence(self):
        gateway, namespace, calls = self._gateway()
        old = {"drain_id": "old", "state": "ROLLED_BACK", "members": ["long"]}
        gateway._drain_persist(old)
        snapshot, entries = gateway._drain_snapshot()
        new = gateway.start_drain(["dst"])
        self.assertNotEqual(new["drain_id"], "old")
        entries["long"].update(owner="src", drain_member=False)
        gateway._drain_persist(snapshot, entries)  # late old reconcile write
        status = gateway.drain_status()
        self.assertEqual(status["drain_id"], new["drain_id"])
        self.assertEqual(status["state"], "REQUESTED")
        self.assertEqual(status["worker_state"], "DRAINING")
        self.assertTrue(gateway.journal["long"]["drain_member"])
        durable = json.loads(gateway.journal_path.read_text())
        self.assertEqual(durable["__drain__"]["drain_id"], new["drain_id"])

    def test_gateway_active_stream_failure_keeps_migration_resources_fenced(self):
        gateway, namespace, calls = self._gateway()
        handler = SimpleNamespace(wfile=io.BytesIO())

        def submit(payload, **kwargs):
            calls.append(("src", copy.deepcopy(payload)))
            self.assertEqual(payload["cmd"], "submit")
            # A drain reaches an unsafe terminal outcome while this already
            # admitted source stream is live. The stream must exit with error,
            # without release/abort/resume or a successful finished marker.
            gateway.journal["__drain__"] = {
                "state": "FAILED_UNSAFE",
                "members": [payload["rid"]],
            }
            gateway._target_ok = False
            gateway._target_fail_since = time.monotonic() - 31
            return {}

        gateway.src.send = submit
        gateway.stream(
            handler, {"rid": "live-long", "prompt_ids": [1], "max_new_tokens": 128}
        )
        self.assertIn(b"target_unreachable_timeout", handler.wfile.getvalue())
        self.assertIn(b"[DONE]", handler.wfile.getvalue())
        self.assertFalse(gateway.journal["live-long"]["finished"])
        self.assertEqual([p["cmd"] for role, p in calls], ["submit"])
        self.assertNotIn("live-long", gateway.active)

    def test_gateway_capacity_wait_refreshes_finished_source_members(self):
        gateway, namespace, calls = self._gateway()
        template = copy.deepcopy(gateway.journal["long"])
        longs = ["long", "long-1", "long-2", "long-3"]
        shorts = [f"short-{i}" for i in range(5)]
        for rid in longs[1:] + shorts:
            gateway.journal[rid] = dict(template, rid=rid, drid=f"dst-{rid}")
        original = gateway.dst.send

        def target(payload, **kwargs):
            if payload["cmd"] == "status":
                return {"reqs": [{"rid": "background"}], "max_running": 8}
            return original(payload, **kwargs)

        def finish_shorts(_delay):
            self.assertEqual(gateway.src_worker_state(), "DRAINING")
            for rid in shorts:
                gateway.journal[rid]["finished"] = True

        gateway.dst.send = target
        with (
            patch.dict(os.environ, {"HANDOFF_DRAIN_SLOT_WAIT_S": "1"}),
            patch.object(namespace["time"], "sleep", side_effect=finish_shorts),
        ):
            status = gateway.start_drain(["dst"])
        self.assertEqual(status["state"], "REQUESTED")
        self.assertEqual(status["migration_members"], longs)
        self.assertEqual(status["slot_gate"]["members"], 4)
        self.assertEqual(status["members"], longs + shorts)
        reservations = [p for role, p in calls if p["cmd"] == "reserve"]
        self.assertEqual(len(reservations), 1)
        self.assertEqual(
            [x["rid"] for x in reservations[0]["items"]],
            [f"dst-{rid}" for rid in longs],
        )
        self.assertTrue(all(gateway.journal[r]["drain_member"] for r in shorts))
        # Run the real reconcile state machine to SAFE_TO_STOP. Completed
        # shorts still have source tracking/KV that must be released.
        epoch = gateway.journal["__drain__"]["epoch"]
        source_rows = {
            rid: {
                "rid": rid,
                "output_len": 128 if rid in shorts else 32,
                "kv_committed_len": 32799,
                "frozen": False,
                "finished": rid in shorts,
            }
            for rid in longs + shorts
        }
        target_states = {}
        frozen, imported, released = [], [], []

        def source_rpc(payload, **kwargs):
            cmd = payload["cmd"]
            if cmd in ("status", "status_lite"):
                return {"reqs": copy.deepcopy(list(source_rows.values()))}
            if cmd == "freeze_all":
                self.assertEqual(payload["epoch"], epoch)
                self.assertEqual(payload["rids"], longs)
                frozen.extend(payload["rids"])
                for rid in payload["rids"]:
                    source_rows[rid]["frozen"] = True
                return {
                    "results": [
                        {
                            "rid": rid,
                            "output_len": 32,
                            "migration_epoch": epoch,
                            "transfer": {"token": f"kv-{rid}"},
                        }
                        for rid in payload["rids"]
                    ]
                }
            if cmd == "release":
                rid = payload["rid"]
                if rid in longs:
                    self.assertTrue(target_states[rid]["local_ready"])
                released.append(rid)
                source_rows.pop(rid, None)
                return {"released": True}
            raise AssertionError(payload)

        def target_rpc(payload, **kwargs):
            cmd = payload["cmd"]
            if cmd == "status":
                return {
                    "reqs": [{"rid": "background"}],
                    "reserved": {f"dst-{rid}": {} for rid in longs},
                    "target_states": copy.deepcopy(list(target_states.values())),
                }
            if cmd == "import_wave":
                for item in payload["items"]:
                    self.assertEqual(item["epoch"], epoch)
                    rid = item["rid"].removeprefix("dst-")
                    self.assertIn(rid, longs)
                    imported.append(rid)
                    target_states[rid] = {
                        "handoff_id": rid,
                        "migration_epoch": epoch,
                        "new_tokens": 1,
                        "import_done_at": time.time(),
                        "local_ready": True,
                    }
                return {}
            raise AssertionError(payload)

        gateway.src.send = source_rpc
        gateway.dst.send = target_rpc
        for _ in range(4):
            gateway._facts_at = 0.0
            gateway._adopt_at = time.monotonic()
            gateway._reconcile_once()
        final = gateway.drain_status()
        self.assertEqual(final["state"], "SAFE_TO_STOP")
        self.assertTrue(final["source_clean_all_ranks"])
        self.assertCountEqual(frozen, longs)
        self.assertCountEqual(imported, longs)
        self.assertCountEqual(released, longs + shorts)
        self.assertEqual(source_rows, {})
        self.assertTrue(
            all(gateway.journal[r]["source_released"] for r in longs + shorts)
        )

    def test_gateway_completed_members_keep_release_bookkeeping(self):
        gateway, namespace, calls = self._gateway()
        gateway.journal["__drain__"] = {
            "drain_id": "pending",
            "wave_id": "wave",
            "members": ["long"],
            "preflight_pending": True,
            "epoch": "epoch",
            "wave": {},
            "state": "REQUESTED",
            "admission_fenced_at": time.time(),
        }
        gateway.journal["long"]["finished"] = True
        gateway._prepare_drain()
        status = gateway.drain_status()
        self.assertEqual(status["members"], ["long"])
        self.assertEqual(status["migration_members"], [])
        self.assertEqual(status["reserve_tokens"], 0)
        self.assertTrue(gateway.journal["long"]["drain_member"])
        self.assertFalse(any(p["cmd"] == "reserve" for role, p in calls))
        # The existing reconcile/release paths must still see this row.
        drain, entries = gateway._drain_snapshot()
        gateway._sync_member(
            entries["long"],
            {
                "long": {
                    "rid": "long",
                    "finished": True,
                    "output_len": 128,
                    "frozen": False,
                },
            },
            {},
            {},
            {},
        )
        self.assertEqual(entries["long"]["phase"], "done")

        def release(payload, **kwargs):
            self.assertEqual(payload["cmd"], "release")
            calls.append(("src", payload))
            return {"released": True}

        gateway.src.send = release
        gateway._drain_release(drain, entries)
        self.assertTrue(entries["long"]["source_released"])

    def test_gateway_preflight_tracks_existing_target_owned_members(self):
        gateway, namespace, calls = self._gateway()
        gateway.journal["long"]["owner"] = "dst"
        # Simulate the cutover completing while preflight waits on migration_lock.
        drain = {
            "drain_id": "pending",
            "wave_id": "wave",
            "members": ["long"],
            "preflight_pending": True,
            "epoch": "epoch",
            "wave": {},
            "state": "REQUESTED",
            "admission_fenced_at": time.time(),
        }
        gateway.journal["__drain__"] = drain
        gateway._prepare_drain()
        self.assertEqual(gateway.journal["__drain__"]["members"], ["long"])
        self.assertEqual(gateway.journal["__drain__"]["migration_members"], [])
        self.assertTrue(gateway.journal["long"]["drain_member"])
        self.assertFalse(any(p["cmd"] == "reserve" for role, p in calls))

    def test_gateway_capacity_failure_reopens_only_before_freeze(self):
        gateway, namespace, calls = self._gateway()
        original = gateway.dst.send

        def full(payload, **kwargs):
            if payload["cmd"] == "status":
                return {
                    "tp_rank_results": [
                        {"max_running": 8, "reqs": [None] * n} for n in (2, 8)
                    ]
                }
            return original(payload, **kwargs)

        gateway.dst.send = full
        with patch.dict(os.environ, {"HANDOFF_DRAIN_SLOT_WAIT_S": "0"}):
            status = gateway.start_drain(["dst"])
        self.assertEqual(status["state"], "ROLLED_BACK")
        self.assertEqual(gateway.src_worker_state(), "READY")
        self.assertFalse(gateway.draining)
        self.assertFalse(status["preflight_pending"])
        self.assertIn("load=8 + members=1", status["failed_reason"])
        self.assertFalse(any(p["cmd"] in ("freeze", "reserve") for role, p in calls))
        self.assertTrue(status["terminal"])

    def test_gateway_target_loss_after_import_has_bounded_fenced_failure(self):
        gateway, namespace, calls = self._gateway()
        gateway.start_drain(["dst"])
        drain = gateway.journal["__drain__"]
        drain["state"] = "BLOCKED"
        drain["wave"]["phase"] = "imported"
        gateway.journal["long"].update(phase="imported", epoch=drain["epoch"])
        gateway._target_ok = False
        gateway._target_fail_since = time.monotonic() - 31
        gateway._dst_fact_fails = 2
        gateway._facts_at = 0.0
        calls.clear()

        def offline(payload, **kwargs):
            calls.append(("dst", copy.deepcopy(payload)))
            raise ConnectionError("D1 disappeared after import")

        gateway.dst.send = offline
        gateway._reconcile_once()
        status = gateway.drain_status()
        self.assertEqual(status["state"], "FAILED_UNSAFE")
        self.assertTrue(status["terminal"])
        self.assertTrue(status["requires_intervention"])
        self.assertFalse(status["source_safe_to_stop"])
        self.assertEqual(status["failure_evidence"]["wave"]["phase"], "imported")
        self.assertEqual(gateway.src_worker_state(), "DRAINING")
        self.assertEqual(gateway.stop_drain()["error"], "not_safe_to_stop")
        self.assertTrue(all(p["cmd"] == "status" for role, p in calls))
        self.assertNotEqual(gateway.journal["long"].get("phase"), "done")
        self.assertFalse(gateway.journal["long"].get("source_released"))
        calls.clear()
        gateway._reconcile_once()  # terminal: no RPC storm or automatic rollback
        self.assertEqual(calls, [])
        handler = SimpleNamespace(
            wfile=io.BytesIO(),
            send_response=lambda *a: None,
            send_header=lambda *a: None,
            end_headers=lambda: None,
        )
        gateway.attach(handler, "long", -1)
        self.assertIn(b"target_unreachable_timeout", handler.wfile.getvalue())
        self.assertIn(b"[DONE]", handler.wfile.getvalue())
        self.assertEqual(calls, [])  # reconnect cannot resume, abort or release
        with self.assertRaisesRegex(RuntimeError, "drain_failed_unsafe"):
            gateway._cutover(
                None,
                namespace["RequestState"]("long", 32768, 128, 32),
                "remote",
                "none",
            )
        self.assertEqual(calls, [])
        self.assertFalse(gateway.journal["long"]["finished"])
        self.assertEqual(
            gateway._target_ticket_wait(64, "dst-short")["error"], "drain_failed_unsafe"
        )
        persisted = json.loads(gateway.journal_path.read_text())
        restarted, _, _ = self._gateway()
        restarted.journal = persisted
        self.assertEqual(restarted.src_worker_state(), "DRAINING")
        restarted._reconcile_once()
        self.assertEqual(restarted.drain_status()["state"], "FAILED_UNSAFE")

    def test_gateway_target_timeout_preserves_preimport_rollback_and_transient_recovery(
        self,
    ):
        gateway, namespace, calls = self._gateway()
        gateway.start_drain(["dst"])
        drain, entries = gateway._drain_snapshot()
        gateway._target_ok = False
        gateway._target_fail_since = time.monotonic() - 3
        drain["wave"]["phase"] = "imported"
        entries["long"]["phase"] = "imported"
        self.assertFalse(gateway._precommit_rollback_safe(drain, entries))
        self.assertFalse(gateway._fail_drain_target(drain, entries))
        gateway._refresh_facts(force=True)  # peer returns before timeout
        self.assertTrue(gateway._target_ok)
        self.assertIsNone(gateway._target_fail_since)
        self.assertFalse(gateway._fail_drain_target(drain, entries))
        gateway._target_ok = False
        gateway._target_fail_since = time.monotonic() - 31
        drain["wave"]["phase"] = "idle"
        entries["long"]["phase"] = "accepted"
        self.assertTrue(gateway._precommit_rollback_safe(drain, entries))
        self.assertFalse(gateway._fail_drain_target(drain, entries))
        drain["mode"] = "natural"
        entries["long"]["phase"] = "imported"
        self.assertFalse(gateway._fail_drain_target(drain, entries))

    def test_c1_cutover_reuses_drain_reservation_epoch(self):
        """Pre-reserve and C1 cutover must reuse one allocation/epoch."""
        namespace = self._gateway_namespace()
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
            # Drain fences admission before waiting for the already-started
            # cutover. Backend preparation stays in migration lock order.
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
                gateway.journal.pop("__drain__")
                gateway.active = {st.rid: st}
                gateway._facts_at = 0.0
                gateway._src_facts = {}
                gateway._maybe_drain_kill = lambda stage: None
                threads[1].start()
                self.assertTrue(drain_waiting.wait(timeout=3))
                self.assertEqual(gateway.src_worker_state(), "DRAINING")
                self.assertTrue(gateway.journal["__drain__"]["preflight_pending"])
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
                        self.assertNotIn("--chunked-prefill-size", args)
                        self.assertEqual(
                            args[args.index("--max-running-requests") + 1], "16"
                        )
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
        self.assertEqual(rows[0]["p_radix_pins"], [2, 1, 0])
        self.assertEqual(rows[0]["p_radix_pins"], rows[1]["p_radix_pins"])
        self.assertEqual(rows[0]["outcomes"], rows[1]["outcomes"])
        # A ready/no-state shard, a missing layer, a native shape rejection,
        # and failure wakeups must all lead to the same branch on both ranks.
        self.assertEqual(rows[0]["graphs"], rows[1]["graphs"])
        self.assertEqual(rows[0]["leases"], rows[1]["leases"])
        self.assertEqual(rows[0]["segmented_startup"], [True, False, False])
        self.assertEqual(rows[0]["segmented_startup"], rows[1]["segmented_startup"])
        self.assertEqual(rows[0]["segmented_routing"], rows[1]["segmented_routing"])
        self.assertEqual(
            rows[0]["segmented_routing"],
            [[True] * 5, [True, True, False, True, False]] * 2,
        )
        self.assertTrue(rows[0]["leases"]["mismatch_rejected"])
        self.assertTrue(rows[0]["leases"]["sparse_mode_mismatch_rejected"])
        self.assertEqual(rows[0]["leases"]["mesh_settings_rejected"], [True] * 4)
        self.assertTrue(rows[0]["leases"]["cache_mismatch_rejected"])
        self.assertTrue(rows[0]["leases"]["overlap_mismatch_rejected"])
        self.assertEqual(rows[0]["d_overlap"], rows[1]["d_overlap"])
        self.assertEqual(
            rows[0]["d_overlap"]["vote_order"], [[1, 5], [2, 5], [3, 5], [4, 5], [5, 8]]
        )
        self.assertEqual(rows[0]["d_overlap"]["commands"], [1, 2, 4, 5])
        self.assertEqual(rows[0]["d_overlap"]["freed"], ["KV"])
        self.assertEqual(rows[0]["d_overlap"]["retries"], 1)
        self.assertEqual(rows[0]["d_overlap"]["commits"], 4)
        self.assertEqual(rows[0]["d_overlap"]["last_seq"], 5)
        self.assertTrue(rows[0]["d_overlap"]["reentry"])
        self.assertTrue(rows[0]["d_overlap"]["notice_retry_ok"])
        self.assertEqual(rows[0]["d_overlap"]["registered"], 0)
        self.assertEqual(rows[0]["d_overlap"]["gpu_proven"], 1)
        self.assertEqual(rows[0]["p_overlap"], rows[1]["p_overlap"])
        self.assertEqual(
            rows[0]["p_overlap"],
            dict(votes=[5, 5], freed=[1, 2], commits=2, registered=0, local_votes=0),
        )
        self.assertEqual(
            rows[0]["dsv41_segments"],
            dict(
                settings=[True, False, False, False, False, False, False],
                policies=[[True, True, False, True, False]] * 3 + [[True] * 5],
                per_decode_votes=0,
            ),
        )
        self.assertEqual(rows[0]["dsv41_segments"], rows[1]["dsv41_segments"])
        self.assertEqual(rows[0]["sparse_progressive"], rows[1]["sparse_progressive"])
        self.assertEqual(
            rows[0]["sparse_progressive"],
            dict(
                votes=[5, 5, 5, 5],
                freed=["KV"],
                local_entries=[0, 1],
                retries=1,
                closed=1,
                stable_votes=0,
            ),
        )
        self.assertEqual(rows[0]["sparse_remote"], rows[1]["sparse_remote"])
        self.assertEqual(
            rows[0]["sparse_remote"],
            dict(
                stable_remote_votes=0,
                final_vote_shapes=[5],
                freed=[705],
                local_graph_votes=0,
                pins_closed=True,
            ),
        )
        self.assertEqual(rows[0]["d_receive_staging"], rows[1]["d_receive_staging"])
        self.assertEqual(
            rows[0]["d_receive_staging"],
            dict(polls=[[3], [3], [4], [0]], native_collective_shapes=[1, 1, 1, 1]),
        )
        self.assertEqual(rows[0]["p_staging_failure"], rows[1]["p_staging_failure"])
        self.assertEqual(
            rows[0]["p_staging_failure"],
            dict(
                stable_votes=0,
                final_vote_shapes=[5],
                both_rejected=True,
                kv_retained=True,
            ),
        )
        self.assertEqual(rows[0]["p_certificates"], rows[1]["p_certificates"])
        self.assertEqual(
            rows[0]["p_certificates"]["votes"], [2, 2, 2, 2, 5, 5, 2, 5, 2, 2]
        )
        self.assertEqual(rows[0]["p_certificates"]["stable_votes"], 0)
        self.assertEqual(
            rows[0]["p_cached_certificates"], rows[1]["p_cached_certificates"]
        )
        self.assertEqual(
            rows[0]["p_cached_certificates"]["votes"],
            rows[0]["p_certificates"]["votes"],
        )
        self.assertTrue(rows[0]["p_cached_certificates"]["active_cache_closed"])
        self.assertEqual(rows[0]["p_cached_certificates"]["stable_votes"], 0)
        self.assertEqual(rows[0]["p_certificates"]["empty_votes"], 0)
        self.assertEqual(rows[0]["p_certificates"]["frees"], [103, 101, 102])
        self.assertEqual(rows[0]["p_certificates"]["revocations"], 1)
        self.assertTrue(rows[0]["p_certificates"]["ack_loss_recovered"])
        self.assertTrue(rows[0]["leases"]["existing_state_fallback"])
        self.assertGreater(rows[0]["leases"]["boundary_reductions"], 0)
        self.assertEqual(rows[0]["leases"]["freed"], [0, 1, 2])
        for snapshot, count in zip(
            rows[0]["leases"]["snapshots"], (0, 1, 2, 1, 1, 0, 1, 0)
        ):
            self.assertEqual(snapshot["leases"], count)
            self.assertEqual(snapshot["verdicts"], [[count == 0, count == 0]] * 16)
            self.assertEqual(snapshot["forward_reductions"], 0)
        self.assertEqual(
            rows[0]["graphs"],
            [
                [value, value]
                for value in (True, False, False, True, False, True, False, False, True)
            ],
        )
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

    @patch.dict(os.environ, HANDOFF_PL_RETIRE="async")
    def test_prelayer_worker_proves_completion_only_after_all_writes(self):
        """Slow event destruction must not run on the worker after native sends."""
        import logging
        from collections import defaultdict

        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py", "MooncakeKVManager", {"transfer_worker"}
        )
        polls = _source_class("disaggregation/base/conn.py", "KVPoll")
        manager_cls.transfer_worker.__globals__.update(
            time=time,
            logger=logging.getLogger(__name__),
            KVPoll=polls,
            _TransferReferenceReaper=_reference_reaper_class(),
        )
        for outcome in (
            "success",
            "staged_success",
            "native_staged",
            "failed_peer",
            "cancelled",
            "short_destination",
        ):
            with self.subTest(outcome=outcome):
                events = []
                retirement_started = threading.Event()
                retirement_release = threading.Event()
                retirement_finished = threading.Event()
                retirement_threads = []

                class ReadyEvent:
                    def synchronize(self):
                        events.append("gpu-ready")

                    def __del__(self):
                        retirement_threads.append(threading.get_ident())
                        retirement_started.set()
                        retirement_release.wait(3)
                        retirement_finished.set()

                manager = manager_cls()
                manager._core_stager = object() if outcome == "native_staged" else None
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
                    if outcome == "staged_success":
                        self.assertIsInstance(kwargs["transfer_fn"], functools.partial)
                    else:
                        self.assertIsNone(kwargs["transfer_fn"])
                    events.append(session)
                    if outcome == "cancelled":
                        manager.request_status.pop(7, None)
                    return 0

                manager.send_kvcache = send
                chunk = SimpleNamespace(
                    room=7,
                    staging_counted=False,
                    wait_event=(
                        ReadyEvent()
                        if outcome in ("success", "staged_success", "native_staged")
                        else SimpleNamespace(
                            synchronize=lambda: events.append("gpu-ready")
                        )
                    ),
                    indices_event=None,
                    prefill_kv_indices=[1, 2],
                    index_slice=slice(0, 2),
                    is_last_chunk=False,
                    kv_buffer_indices=None if outcome == "native_staged" else (0, 1),
                    transfer_fn=None,
                    staging_chunk_idx=None,
                    staging_layer_ids=None,
                    on_complete=lambda: events.append("proof"),
                )
                if outcome == "staged_success":
                    chunk.transfer_fn = functools.partial(
                        lambda *, ready_event: None, ready_event=chunk.wait_event
                    )
                tasks = queue.Queue()
                tasks.put(chunk)
                del chunk
                tasks.put(None)
                try:
                    if outcome == "short_destination":
                        with self.assertRaisesRegex(RuntimeError, "does not cover"):
                            manager.transfer_worker(tasks, None)
                    else:
                        manager.transfer_worker(tasks, None)
                    if outcome in ("success", "staged_success", "native_staged"):
                        self.assertTrue(retirement_started.wait(3))
                        self.assertNotEqual(retirement_threads, [threading.get_ident()])
                        self.assertFalse(retirement_finished.is_set())
                finally:
                    retirement_release.set()
                    if outcome in ("success", "staged_success", "native_staged"):
                        self.assertTrue(retirement_finished.wait(3))
                if outcome in ("success", "staged_success", "native_staged"):
                    self.assertEqual(events, ["gpu-ready", "a", "b", "proof"])
                else:
                    self.assertNotIn("proof", events)

    def test_prelayer_reaper_bounds_pending_destructors(self):
        """A blocked destructor cannot create an unbounded overflow or inline free."""
        cls = _reference_reaper_class()
        started, release, saturated, submitted = (threading.Event() for _ in range(4))
        freed = []

        class SlowReference:
            def __del__(self):
                started.set()
                release.wait(3)

        class Reference:
            def __del__(self):
                freed.append(threading.get_ident())

        with patch.dict(os.environ, HANDOFF_PL_RETIRE="async"):
            reaper = cls()
        reaper.submit([SlowReference()])
        self.assertTrue(started.wait(3))
        for _ in range(64):
            reaper.submit([Reference()])
        original_put_nowait = reaper.queue.put_nowait

        def observe_full(envelope):
            try:
                original_put_nowait(envelope)
            except queue.Full:
                saturated.set()
                raise

        def producer():
            reaper.submit([Reference()])
            submitted.set()

        producer_thread = threading.Thread(target=producer)
        try:
            with patch.object(reaper.queue, "put_nowait", side_effect=observe_full):
                producer_thread.start()
                self.assertTrue(saturated.wait(3))
                self.assertFalse(submitted.is_set())
                self.assertEqual(reaper.queue.qsize(), 64)
                self.assertEqual(freed, [])
                release.set()
                producer_thread.join(3)
        finally:
            release.set()
            producer_thread.join(3)
            reaper.close()
            reaper.thread.join(3)
        self.assertTrue(submitted.is_set())
        self.assertFalse(reaper.thread.is_alive())
        self.assertEqual(freed, [reaper.thread.ident] * 65)
        for mode in ("off", "keep64"):
            with (
                self.subTest(mode=mode),
                patch.dict(os.environ, HANDOFF_PL_RETIRE=mode),
            ):
                freed.clear()
                control = cls()
                for _ in range(65):
                    control.submit([Reference()])
                self.assertIsNone(control.thread)
                self.assertEqual(len(freed), 65 if mode == "off" else 1)
                control.close()
                self.assertEqual(freed, [threading.get_ident()] * 65)

    def test_core_staging_preserves_bytes_bounds_and_host_batches(self):
        spec = importlib.util.spec_from_file_location(
            "core_staging_test",
            ROOT / "python/sglang/srt/disaggregation/mooncake/pinned_staging_core.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        engine = SimpleNamespace()
        stager = module.CorePinnedStager(engine, capacity=8)
        source = torch.arange(20, dtype=torch.uint8)

        class TensorBoundary:
            device = SimpleNamespace(type="cuda")

            def data_ptr(self):
                return source.data_ptr()

            def is_contiguous(self):
                return True

            def numel(self):
                return source.numel()

            def element_size(self):
                return 1

            def view(self, dtype):
                return source.view(dtype)

        tensor = TensorBoundary()
        stager.set_buffers([tensor], [source.data_ptr()], [4])
        self.assertIsNone(stager.resolve(source.data_ptr() + 1, 4))
        self.assertIsNone(stager.resolve(source.data_ptr() + 16, 8))
        self.assertIsNone(stager.resolve(1, 4))
        from sglang.srt.disaggregation.pinned_staging_buffer import PinnedKVStager

        proxy = _host_staging_manager()
        stager.buffer = PinnedKVStager(proxy, 0, slot_factory=_HostStagingSlot)
        stager.buffer.reserve(8)
        stager._reserve = lambda device: stager.buffer._check_live()
        destination = ctypes.create_string_buffer(24)
        dest = ctypes.addressof(destination)
        host = ctypes.create_string_buffer(b"host")
        batches = []

        def transfer(blocks):
            batches.append(list(blocks))
            for src, dst, n in blocks:
                ctypes.memmove(dst, src, n)
            return 0

        with (
            patch.object(torch.cuda, "device", side_effect=lambda _: nullcontext()),
            patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()),
        ):
            rc = stager.send(
                [
                    (source.data_ptr() + 4, dest, 16),
                    (ctypes.addressof(host), dest + 16, 4),
                ],
                transfer,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(destination.raw[:20], bytes(range(4, 20)) + b"host")
            self.assertEqual([sum(b[2] for b in batch) for batch in batches], [8, 8, 4])
            batches.clear()
            blocks = [
                (ctypes.addressof(host), dest, 2),
                (ctypes.addressof(host) + 2, dest + 2, 2),
            ]
            self.assertEqual(stager.send(blocks, transfer), 0)
            self.assertEqual(batches, [blocks])  # Already staged sources stay batched.
            manager_cls = _source_class(
                "disaggregation/mooncake/conn.py",
                "MooncakeKVManager",
                {"_transfer_data", "ensure_host_stager"},
            )
            manager = manager_cls()
            manager._core_stager = stager
            manager.engine = SimpleNamespace(
                batch_transfer_sync=lambda session, srcs, dsts, sizes: transfer(
                    list(zip(srcs, dsts, sizes))
                )
            )
            shared_slot = stager.buffer.slot
            self.assertIs(manager.ensure_host_stager(), stager)
            plan = [(source.data_ptr(), dest, 4)]
            for all_sends, layer_send in ((False, False), (False, True), (True, False)):
                manager._host_staging_all = all_sends
                batches.clear()
                self.assertEqual(
                    manager._transfer_data("peer", plan, stage=layer_send), 0
                )
                expected = (
                    shared_slot.ptr if all_sends or layer_send else source.data_ptr()
                )
                self.assertEqual(batches[0][0][0], expected)
                self.assertIs(stager.buffer.slot, shared_slot)
            self.assertEqual(
                stager.send([(source.data_ptr(), dest, 4)], lambda b: -1), -1
            )
            self.assertTrue(stager.poisoned)
        stager.set_buffers([tensor], [source.data_ptr() + 4], [4])
        self.assertIsNone(stager.resolve(source.data_ptr() + 4, 4))

    def test_core_staging_records_readiness_at_enqueue(self):
        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"add_transfer_request"},
        )
        method = manager_cls.add_transfer_request
        method.__globals__.update(
            DisaggregationMode=SimpleNamespace(PREFILL="p"),
            KVPoll=SimpleNamespace(Failed="failed"),
            TransferKVChunk=lambda **kw: SimpleNamespace(**kw),
            TraceNullContext=lambda: None,
        )
        manager = manager_cls()
        manager.disaggregation_mode = "p"
        manager._core_stager = object()
        manager.request_status = {1: "sending"}
        manager.check_status = lambda room: "sending"
        manager.transfer_infos = {1: {"host:10": object()}}
        manager.transfer_queues = [queue.Queue()]
        recorded = []
        event = SimpleNamespace(
            record=lambda stream: recorded.append((threading.get_ident(), stream))
        )
        with (
            patch.object(torch.cuda, "Event", return_value=event),
            patch.object(torch.cuda, "current_stream", return_value="producer"),
        ):
            manager.add_transfer_request(1, [0], slice(0, 1), False)
            self.assertEqual(recorded, [(threading.get_ident(), "producer")])
            self.assertIs(manager.transfer_queues[0].get().wait_event, event)
            manager.add_transfer_request(1, [0], slice(0, 1), False, wait_event=event)
            self.assertEqual(len(recorded), 1)
            manager._core_stager = None
            manager.add_transfer_request(1, [0], slice(0, 1), False)
            self.assertEqual(len(recorded), 1)

    def test_prelayer_cuda_slot_orders_private_copy_before_native_send(self):
        """Exercise the real CUDA adapter with driver calls at the test boundary."""
        from sglang.srt.disaggregation.pinned_staging_buffer import PinnedKVStager

        manager = _host_staging_manager()
        ready = object()
        pending, received, flags, destroyed, waited = [], [], [], [], []
        warmups = []
        make_empty = torch.empty
        test = self

        class Stream:
            def wait_event(self, event):
                test.assertTrue(event is ready or isinstance(event, Event))
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
            if waited[-1] is not ready:
                # Registration must precede the first real D2H to this slot.
                self.assertIn(dst, manager.registrations)
                warmups.append(dst)
            elif kind == 2:
                self.assertEqual(waited[-1:], [ready])
            pending.append((dst, src, size))
            return (0,)

        def empty(*args, **kwargs):
            if "device" in kwargs:
                self.assertEqual(kwargs.pop("device"), "cuda:1")
            else:
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
            patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "ExternalStream", return_value=stream),
            patch.object(torch.cuda, "Event", side_effect=Event),
        ):
            stager = PinnedKVStager(manager, 1)
            stager.reserve(4)
            stager.reserve(8)
            stager.reserve(8)
            self.assertEqual(len(warmups), 2)
            self.assertEqual(pending, [])
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
            self.assertTrue(stager.close())
        self.assertEqual(flags, [1, 1])
        self.assertEqual(destroyed, [17, 17])
        self.assertEqual(manager.registrations, {})

    def test_prelayer_staging_preserves_native_mapping_and_batch_bound(self):
        """Staging changes only source pointers, including scattered TP rows."""
        import numpy as np

        from sglang.srt.disaggregation.pinned_staging_buffer import PinnedKVStager

        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"send_kvcache", "_send_kvcache_generic", "send_layer_group_to_staging"},
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
            np=np,
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
                # Route the same native Batch=8 plan into the existing ring,
                # then execute the native scatter handler (both TP rank maps).
                f = self._native_ring_fixture(rank=rank, pinned=True, capacity=136)
                f.handler.kv_buffer_info.update(
                    k_buffers=[t.reshape(32, 1, 2) for t in dst[:2]],
                    v_buffers=[t.reshape(32, 1, 2) for t in dst[2:]],
                )
                f.handler.scheduler.req_to_token_pool.req_to_token = torch.from_numpy(
                    dst_indices
                ).reshape(1, -1)
                target = SimpleNamespace(
                    dst_kv_ptrs=[t.data_ptr() for t in dst],
                    dst_kv_layer_ids=[],
                    dst_kv_item_len=2,
                    dst_attn_tp_size=2,
                    staging_base_ptr=f.allocator.base_ptr,
                    staging_total_size=136,
                )
                batches.clear()
                for tensor in dst:
                    tensor.fill_(255)
                for index, selected in enumerate(((0, 2), (1, 3))):
                    info = f.allocate(index, (index,), start=0, pages=17)
                    chunk = SimpleNamespace(
                        staging_layer_ids=(index,),
                        kv_buffer_indices=selected,
                        prefill_kv_indices=src_indices,
                        transfer_fn=functools.partial(stager.send, ready_event=ready),
                    )
                    rc = manager.send_layer_group_to_staging(
                        SimpleNamespace(mooncake_session_id=f"peer-{rank}"),
                        chunk,
                        target,
                        info[1],
                        None,
                    )
                    self.assertEqual(rc, 0)
                    f.handler.handle_chunk_arrived(7, index, 0, 17, "peer")
                    f.events[-1].complete = True
                    f.handler.advance_scatter(f.dr)
                self.assertEqual(batches, [32, 32, 4] * 2)
                for i in range(4):
                    expected = torch.full_like(dst[i], 255)
                    expected[dst_indices] = src[i][src_indices]
                    self.assertTrue(torch.equal(dst[i], expected))
                self.assertFalse(f.allocator.allocations)
                self.assertTrue(stager.close())
                self.assertFalse(manager.registrations)

    def test_prelayer_staging_quarantines_unknown_completion_and_unregister(self):
        """A failed native call must never turn pinned storage into reusable memory."""
        from sglang.srt.disaggregation.pinned_staging_buffer import PinnedKVStager

        for failure in (
            "register",
            "warmup",
            "copy",
            "send",
            "send_exception",
            "unregister",
        ):
            with self.subTest(failure=failure):
                manager = _host_staging_manager()
                stager = PinnedKVStager(manager, 0, slot_factory=_HostStagingSlot)
                ready = threading.Event()
                ready.set()

                def fail(*args):
                    raise RuntimeError("injected native failure")

                if failure in ("register", "warmup"):
                    if failure == "register":
                        manager.engine.register = fail
                    with (
                        patch.object(
                            _HostStagingSlot,
                            "warmup",
                            side_effect=fail if failure == "warmup" else None,
                        ),
                        self.assertRaises(RuntimeError),
                    ):
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
                    _pre_layerwise_source=SimpleNamespace(stager=stager, error=None)
                )
                with self.assertRaisesRegex(RuntimeError, "retaining source KV"):
                    pl.defer_release(req, lambda: self.fail("unsafe KV release"))

    def test_prelayer_staging_serializes_slot_reuse_until_send_completes(self):
        from sglang.srt.disaggregation.pinned_staging_buffer import PinnedKVStager

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

    def _native_ring_fixture(self, *, rank=0, pinned=True, capacity=24):
        """Real allocator, wire allocation and scatter handler; CUDA is the boundary."""
        import logging
        import struct

        module_path = "disaggregation/common/staging_buffer.py"
        functions = {
            name: _source_function(module_path, name)
            for name in (
                "compute_staging_layout",
                "compute_head_slice_params",
                "scatter_kv_head_slices",
                "_scatter_staging_to_kv_torch",
                "resolve_total_kv_heads",
                "staging_grid_tokens",
            )
        }
        for fn in functions.values():
            fn.__globals__.update(functions, torch=torch)
        functions["scatter_staging_to_kv"] = functions["_scatter_staging_to_kv_torch"]
        stream_log, events, messages = [], [], []

        class Event:
            complete = False

            def record(self, stream):
                stream_log.append("event")

            def query(self):
                return self.complete

        class Stream:
            def wait_event(self, event):
                stream_log.append("wait-row")

            def synchronize(self):
                for event in events:
                    event.complete = True

        stream = Stream()

        class Buffer:
            def __init__(self, size, device, gpu_id, custom=None, *, pin_memory=False):
                self.device = torch.device("cpu")
                self.size_bytes, self.pin_memory = size, pin_memory
                self.buffer = torch.zeros(size, dtype=torch.uint8)
                self.gpu_buffer = (
                    torch.zeros_like(self.buffer) if pin_memory else self.buffer
                )
                self.data_ptr = self.buffer.data_ptr()

        allocator_cls = _source_class(module_path, "StagingAllocator")
        allocator_cls.__init__.__globals__.update(
            StagingBuffer=Buffer,
            threading=threading,
            logger=logging.getLogger(__name__),
            _device_ptr_type_str=lambda *args: "",
        )
        allocator = allocator_cls(capacity, "cuda", rank, pin_memory=pinned)
        functions["StagingAllocator"] = allocator_cls
        functions["scatter_staging_to_kv"] = functions["_scatter_staging_to_kv_torch"]
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.common.staging_buffer": SimpleNamespace(
                        **functions
                    ),
                },
            )
        )
        handler_cls = _source_class(
            "disaggregation/common/staging_handler.py", "DecodeStagingHandler"
        )

        def event(device):
            result = Event()
            events.append(result)
            return result

        handler_cls.__init__.__globals__.update(
            torch=torch,
            threading=threading,
            logger=logging.getLogger(__name__),
            time=time,
            create_device_event=event,
            create_device_stream=lambda device: stream,
            device_stream_context=lambda stream: nullcontext(),
        )
        handler = handler_cls.__new__(handler_cls)
        handler._scatter_lock = threading.RLock()
        handler._room_to_decode_req, handler._room_to_receiver = {}, {}
        handler._writer_counts, handler._wm_subscribers = {}, {}
        handler.staging_allocator = allocator
        handler.completion_timeout = 3
        handler.decode_tp, handler.total_kv_heads, handler.tp_rank = 2, 2, rank
        handler.device = torch.device("cuda", rank)
        handler.gpu_id = rank
        handler.device_module = SimpleNamespace(set_device=lambda device: None)
        k = [torch.full((8, 1, 2), 255, dtype=torch.uint8) for _ in range(2)]
        v = [torch.full_like(t, 255) for t in k]
        handler.kv_buffer_info = dict(k_buffers=k, v_buffers=v, page_size=1)
        kv_args = SimpleNamespace(
            kv_item_lens=[2] * 4, page_size=1, engine_rank=rank, total_kv_head_num=2
        )
        manager = SimpleNamespace(
            kv_args=kv_args,
            pp_size=1,
            attn_tp_size=2,
            _staging_ctx=SimpleNamespace(room_receivers={}, room_bootstrap={}),
        )
        handler.kv_manager = manager
        rows = torch.tensor([[6, 2, 4, 1]], dtype=torch.int64)
        handler.scheduler = SimpleNamespace(
            schedule_stream=stream, req_to_token_pool=SimpleNamespace(req_to_token=rows)
        )
        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            chunk_staging_layers={},
            prefill_info=SimpleNamespace(attn_tp_size=2, pp_size=1),
            bootstrap_infos=[{"pp_rank": 0}],
        )
        receiver._connect_to_bootstrap_server = lambda bi: (
            SimpleNamespace(send_multipart=lambda parts: messages.append(parts)),
            threading.Lock(),
        )
        dr = SimpleNamespace(
            req=SimpleNamespace(
                bootstrap_room=7,
                kv=SimpleNamespace(cache_protected_len=0, req_pool_idx=0),
            ),
            kv_receiver=receiver,
        )
        handler.register_decode_req(7, dr)
        manager._staging_ctx.room_receivers[7] = receiver
        manager._staging_ctx.room_bootstrap[7] = receiver.bootstrap_infos
        handler.register_wm_subscriber(receiver, "peer")
        request = _source_function(
            "disaggregation/common/staging_handler.py", "handle_staging_req"
        )
        request.__globals__.update(struct=struct, logger=logging.getLogger(__name__))

        def allocate(index, layers, start=1, pages=3):
            request(
                [
                    b"STAGING_REQ",
                    b"7",
                    str(index).encode(),
                    str(pages).encode(),
                    b"peer",
                    b"0",
                    struct.pack(f"{len(layers)}Q", *layers),
                    str(start).encode(),
                ],
                allocator,
                kv_args,
                2,
                2,
                handler.kv_buffer_info,
                manager._staging_ctx.room_receivers,
                manager._staging_ctx.room_bootstrap,
            )
            return receiver.chunk_staging_infos[index]

        return SimpleNamespace(
            allocator=allocator,
            handler=handler,
            manager=manager,
            receiver=receiver,
            dr=dr,
            allocate=allocate,
            events=events,
            messages=messages,
            k=k,
            v=v,
            rows=rows,
            stream_log=stream_log,
        )

    def test_native_ring_scatter_event_gates_local_and_watermark(self):
        """Network success and completed H2D cannot release a pending scatter slot."""
        pl = self.hooks.pre_layerwise
        for rank, pinned in ((0, False), (1, True)):
            with (
                self.subTest(rank=rank, pinned=pinned),
                tempfile.TemporaryDirectory() as directory,
            ):
                f = self._native_ring_fixture(rank=rank, pinned=pinned)
                first = f.allocate(0, (0,))
                second = f.allocate(1, (1,))
                self.assertEqual((first[1], second[1]), (0, 12))
                # Repeating native STAGING_REQ must not consume another slot.
                self.assertEqual(f.allocate(0, (0,)), first)
                self.assertEqual(len(f.allocator.allocations), 2)
                state = SimpleNamespace(
                    events=[threading.Event(), threading.Event()],
                    layer_ready_at=[None, None],
                    import_done_at=None,
                    mode="remote",
                    error=None,
                    aborted=False,
                )
                target = pl.Target(
                    None,
                    SimpleNamespace(staging_handler=f.handler),
                    f.dr,
                    Path(directory),
                    "current",
                    pl.LayerCoverage(4, 2),
                    state,
                    native_staging=True,
                )
                payload = torch.arange(12, dtype=torch.uint8)
                f.allocator.buffer.buffer[:12].copy_(payload)
                f.handler.handle_chunk_arrived(7, 0, 1, 3, "peer")
                f.handler.submit_last_scatter_async(7)
                target.refresh()
                self.assertFalse(state.events[0].is_set())
                with patch.object(self.runtime, "_TARGET_STATES", [state]):
                    self.assertTrue(self.runtime.has_remote_work())
                self.assertFalse(f.handler.is_done(f.dr))
                with patch.object(pl, "_poll_success", return_value=True):
                    self.assertFalse(target.ready_to_finish())
                self.assertEqual(f.allocator.get_watermark(), (0, 0))
                self.assertFalse((Path(directory) / "layer-0.json").exists())
                # Real native torch scatter writes exactly the selected K/V rows.
                for buf, values in ((f.k[0], payload[:6]), (f.v[0], payload[6:])):
                    expected = torch.full_like(buf, 255)
                    expected[f.rows[0, 1:]] = values.reshape(3, 1, 2)
                    self.assertTrue(torch.equal(buf, expected))
                self.assertTrue(torch.all(f.k[1] == 255).item())
                self.assertTrue(torch.all(f.v[1] == 255).item())
                f.events[-1].complete = True
                target.refresh()
                self.assertTrue(state.events[0].is_set())
                self.assertFalse(state.events[1].is_set())
                with patch.object(self.runtime, "_TARGET_STATES", [state]):
                    self.assertTrue(self.runtime.has_remote_work())
                self.assertEqual(f.allocator.get_watermark(), (0, 12))
                self.assertFalse(f.handler.is_done(f.dr))
                # The last group's arrival can follow native Success.
                f.allocator.buffer.buffer[12:24].copy_(payload + 20)
                f.handler.handle_chunk_arrived(7, 1, 1, 3, "peer")
                target.refresh()
                self.assertFalse(state.events[1].is_set())
                f.events[-1].complete = True
                target.refresh()
                self.assertTrue(state.events[1].is_set())
                with patch.object(self.runtime, "_TARGET_STATES", [state]):
                    self.assertFalse(self.runtime.has_remote_work())
                self.assertTrue(f.handler.is_done(f.dr))
                self.assertEqual(f.allocator.get_watermark(), (1, 0))
                self.assertEqual(f.messages[-1], [b"WATERMARK", b"1", b"0", b"peer"])
                self.assertIn("wait-row", f.stream_log)

    def test_native_ring_wrap_waits_for_scatter_not_host_copy(self):
        """Watermark reuse must preserve the old payload while scatter is pending."""
        f = self._native_ring_fixture(capacity=12)
        first, wrapped = f.allocate(0, (0,)), f.allocate(1, (1,))
        self.assertEqual((first[1], first[2], wrapped[1], wrapped[2]), (0, 0, 0, 1))
        ready = _source_function(
            "disaggregation/common/staging_handler.py", "is_watermark_ready"
        )
        context = SimpleNamespace(
            remote_watermarks={"peer": f.allocator.get_watermark()}
        )
        self.assertTrue(ready(context, "peer", first[2], first[3]))
        self.assertFalse(ready(context, "peer", wrapped[2], wrapped[3]))
        f.allocator.buffer.buffer.copy_(torch.arange(12, dtype=torch.uint8))
        f.handler.handle_chunk_arrived(7, 0, 1, 3, "peer")
        f.handler.advance_scatter(f.dr)
        context.remote_watermarks["peer"] = f.allocator.get_watermark()
        self.assertFalse(ready(context, "peer", wrapped[2], wrapped[3]))
        f.events[-1].complete = True
        f.handler.advance_scatter(f.dr)
        context.remote_watermarks["peer"] = f.allocator.get_watermark()
        self.assertTrue(ready(context, "peer", wrapped[2], wrapped[3]))
        # Mismatched ranges cannot launch another scatter or retire its slot.
        with self.assertRaisesRegex(ValueError, "does not match"):
            f.handler.submit_chunk_scatter(7, 1, 0, 3)
        self.assertIn(wrapped[0], f.allocator.allocations)

    def test_native_ring_registers_host_storage_without_changing_gpu_control(self):
        """Transport must register the host ring, while scatter keeps a GPU shadow."""
        import logging

        buffer_cls = _source_class(
            "disaggregation/common/staging_buffer.py", "StagingBuffer"
        )
        buffer_cls.__init__.__globals__.update(
            torch=torch,
            logger=logging.getLogger(__name__),
            _STAGING_SUPPORTED_DEVICE_TYPES=("cuda", "xpu"),
            _device_ptr_type_str=lambda *args: "",
        )
        empty = torch.empty
        for pinned in (False, True):
            allocations = []

            def allocate(*args, **kwargs):
                allocations.append(
                    (kwargs.pop("device", None), kwargs.pop("pin_memory", False))
                )
                return empty(*args, **kwargs)

            with (
                patch.object(
                    torch,
                    "get_device_module",
                    return_value=SimpleNamespace(set_device=lambda _: None),
                ),
                patch.object(torch, "empty", side_effect=allocate),
            ):
                buf = buffer_cls(64, "cuda", 1, pin_memory=pinned)
            self.assertEqual(buf.get_ptr(), buf.buffer.data_ptr())
            self.assertEqual(buf.get_ptr() != buf.gpu_buffer.data_ptr(), pinned)
            self.assertEqual(allocations[0], (torch.device("cuda", 1), False))
            self.assertEqual(len(allocations), 2 if pinned else 1)
            if pinned:
                self.assertEqual(allocations[1], (None, True))

    def test_native_ring_group_identity_and_oversize_fail_closed(self):
        f = self._native_ring_fixture(capacity=12)
        f.allocate(0, (0,))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            f.allocate(0, (1,))
        with self.assertRaisesRegex(ValueError, "invalid"):
            f.allocate(1, (2,))
        oversized = f.allocate(2, (0, 1))
        self.assertEqual(oversized[1], -2)
        self.assertFalse(f.handler.submit_chunk_scatter(7, 2, 1, 3))
        self.assertEqual(f.dr._staging_completed_layers, [])

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

    def _dsv41_fixture(self, length=33):
        """Real native page/state planners with CPU tensors at the GPU boundary."""
        import numpy as np

        ds = self.hooks.dsv41_layerwise
        native_types = _source_class("disaggregation/base/conn.py", "StateType")
        state_type = enum.Enum(
            "StateType",
            {k: v for k, v in vars(native_types).items() if not k.startswith("_")},
        )
        pool_cls = _source_class(
            "mem_cache/deepseek_v4_memory_pool.py",
            "DeepSeekV4TokenToKVPool",
            {
                "get_contiguous_buf_infos",
                "get_state_buf_infos",
                "get_request_state_buf_infos",
                "request_state_transfer_indices",
            },
        )
        index_cls = _source_class(
            "mem_cache/deepseek_v4_memory_pool.py",
            "DeepSeekV4IndexerPool",
            {"contiguous_page_row_buffers"},
        )
        carry_cls = _source_class(
            "mem_cache/deepseek_v4_compress_state.py",
            "CompressStatePool",
            {"transfer_indices"},
        )
        carry_cls.transfer_indices.__globals__["np"] = np
        pool = pool_cls()
        pool._unified_kv = pool._unified_kv_fp8 = pool.uniform_fp8 = False
        pool.page_size = pool.swa_page_size = 8
        pool._stage_start, pool._stage_end = 0, 40
        pool.compression_ratios = [0] * 2 + [2] * 18 + [1] * 20
        pool.sources_by_ratio = {2: [2, 8, 14], 1: [20]}
        pool.kv_pools, pool.index_pools = {}, {}
        for ratio, sources in pool.sources_by_ratio.items():
            pool.kv_pools[ratio] = SimpleNamespace(
                kv_buffer=[
                    torch.arange(128, dtype=torch.uint8).reshape(8, 16) + i
                    for i in sources
                ],
                kv_layout=SimpleNamespace(value="v41_fp4"),
            )
            index = index_cls()
            index.page_size, index.index_k_page8 = 2, False
            index.index_k_with_scale_buffer = None
            index_rows = 8 * pool.page_size // ratio // index.page_size
            index.index_k_payload_buffer = [
                torch.full((index_rows, 4), i + 31, dtype=torch.uint8) for i in sources
            ]
            index.index_k_scale_buffer = [
                torch.full((index_rows, 1), i + 71, dtype=torch.uint8) for i in sources
            ]
            pool.index_pools[ratio] = index
        pool.request_window = None
        pool.swa_kv_pool = SimpleNamespace(
            page_size=8, kv_buffer=[torch.full((8, 8), 23, dtype=torch.uint8)]
        )
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        for _ in range(3):
            carry = carry_cls()
            carry.ratio, carry.online, carry.ring_size, carry.request_scoped = (
                2,
                False,
                2,
                True,
            )
            carry.kv_score_buffer = SimpleNamespace(
                kv_score=torch.full((4, 4), 19, dtype=torch.uint8)
            )
            pool.compress_state_pools.append(carry)
        config = SimpleNamespace(
            model_type="deepseek_v41",
            num_hidden_layers=40,
            compress_ratios=pool.compression_ratios,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
            candidate_source_layer_id=20,
        )
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        swa = pool.get_state_buf_infos()
        carry = pool.get_request_state_buf_infos()
        kv_args = SimpleNamespace(
            kv_data_ptrs=ptrs,
            kv_data_lens=lens,
            kv_item_lens=item_lens,
            kv_layer_ids=[],
            num_draft_entries=0,
            state_types=[state_type.SWA, state_type.DSV4_REQUEST_STATE],
            state_data_ptrs=[swa[0], carry[0]],
            state_item_lens=[swa[2], carry[2]],
        )
        destination = [torch.zeros(n, dtype=torch.uint8) for n in lens]
        target_states = [
            [torch.zeros_like(t) for t in pool.swa_kv_pool.kv_buffer],
            [torch.zeros(16, dtype=torch.uint8) for _ in range(3)],
        ]
        registration = SimpleNamespace(
            dst_kv_ptrs=[t.data_ptr() for t in destination],
            dst_kv_layer_ids=[],
            dst_attn_tp_size=1,
            requires_dcp_relayout=False,
        )
        peer = SimpleNamespace(
            is_dummy=False, mooncake_session_id="peer:1", dst_device_kv_indices=None
        )
        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"_send_kvcache_generic", "send_kvcache"},
        )
        group = _source_function(
            "disaggregation/common/utils.py", "group_concurrent_contiguous"
        )
        group.__globals__["np"] = np
        flat = _source_function(
            "disaggregation/mooncake/conn.py", "_uses_flat_entry_layout"
        )
        flat.__globals__["StateType"] = state_type
        manager_cls._send_kvcache_generic.__globals__.update(
            np=np,
            StateType=state_type,
            group_concurrent_contiguous=group,
            _uses_flat_entry_layout=flat,
            get_memory=lambda: SimpleNamespace(enable_unified_memory=False),
            is_elided_entry=_source_function(
                "disaggregation/utils.py", "is_elided_entry"
            ),
        )
        manager_cls.get_mla_kv_ptrs_with_pp = _source_class(
            "disaggregation/common/conn.py",
            "CommonKVManager",
            {"get_mla_kv_ptrs_with_pp"},
        ).get_mla_kv_ptrs_with_pp
        manager = manager_cls()
        manager.kv_args, manager.attn_tp_size, manager.pp_size = kv_args, 1, 1
        manager.is_mla_backend = manager.is_hybrid_mla_backend = True
        manager.enable_custom_mem_pool = manager.enable_staging = False
        manager.kv_shard_size = 1
        manager.enable_all_cp_ranks_for_transfer = manager.is_dummy_cp_rank = False
        manager.max_transfer_batch_indices = 2
        manager.transfer_infos = {7: {"peer:1": peer}}
        manager.decode_kv_args_table = {"peer:1": registration}
        manager._validate_envelope_kv_layout = lambda *a: None
        manager.prepare_send_state_indices = lambda state: state
        transfers = []

        def copy_blocks(session, blocks):
            transfers.append(list(blocks))
            src_spans = list(zip(kv_args.kv_data_ptrs, kv_args.kv_data_lens))
            src_spans += list(zip(swa[0], swa[1])) + list(zip(carry[0], carry[1]))
            dst_spans = [(t.data_ptr(), t.nbytes) for t in destination]
            dst_spans += [
                (t.data_ptr(), t.nbytes)
                for component in target_states
                for t in component
            ]
            for src, dst, size in blocks:
                self.assertTrue(
                    any(p <= src and src + size <= p + n for p, n in src_spans)
                )
                self.assertTrue(
                    any(p <= dst and dst + size <= p + n for p, n in dst_spans)
                )
                ctypes.memmove(dst, src, size)
            return 0

        manager._transfer_data = copy_blocks
        tasks = queue.Queue()
        manager.add_transfer_request = lambda room, indices, index_slice, last, **kw: (
            tasks.put(
                SimpleNamespace(
                    indices=indices, index_slice=index_slice, last=last, **kw
                )
            )
        )
        sender_cls = _source_class(
            "disaggregation/common/conn.py",
            "CommonKVSender",
            {
                "_prepare_send_indices",
                "send",
                "_record_transfer_indices",
                "should_send_kv_chunk",
            },
        )
        polls = SimpleNamespace(
            Success="success", Failed="failed", Transferring="transferring"
        )
        sender_cls.send.__globals__.update(KVPoll=polls, time=time)
        sender = sender_cls()
        sender.kv_mgr, sender.bootstrap_room, sender.curr_idx = manager, 7, 0
        sender.num_kv_indices = (length + 7) // 8
        sender._transfer_num_kv_indices = sender._transfer_num_state_indices = 0
        sender._transfer_start_time = sender._early_send_wait_event = (
            sender.conclude_state
        ) = None
        sender.trace_ctx = SimpleNamespace(copy_for_thread=lambda: None)
        sender.aux_index = 0
        sender.get_max_transfer_tokens = lambda: None
        native_status = [polls.Transferring]
        sender.poll = lambda: native_status[0]
        req = SimpleNamespace(
            disagg_kv_sender=sender,
            pending_bootstrap=False,
            inflight_middle_chunks=0,
            extend_end=16,
            extend_len=16,
            origin_input_ids=list(range(length)),
            multimodal_inputs=None,
            input_embeds=None,
            bootstrap_room=7,
            start_send_idx=0,
            disagg_decode_prefix_len=0,
            kv=SimpleNamespace(req_pool_idx=0, is_kv_released=False),
        )
        scheduler_cls = _source_class(
            "disaggregation/prefill.py",
            "SchedulerDisaggregationPrefillMixin",
            {"send_kv_chunk", "_send_kv_chunk"},
        )
        kv_to_page = _source_function("mem_cache/common.py", "kv_to_page_indices")
        kv_to_page.__globals__.update(torch=torch, np=np)
        scheduler_cls._send_kv_chunk.__globals__.update(
            StateType=state_type,
            _is_npu=False,
            np=np,
            torch=torch,
            kv_to_page_indices=kv_to_page,
        )
        scheduler = scheduler_cls()
        scheduler.kv_checksum_computer = None
        scheduler.enable_staging = scheduler.enable_pdmux = False
        scheduler.sliding_window_size = 128
        scheduler.chunked_prefill_size = 16
        scheduler.disagg_prefill_pending_chunk_rids = set()
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(kv_manager=manager)
        scheduler.disaggregation_mode = "prefill"
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=(torch.arange(length, dtype=torch.int32) + 8).reshape(1, -1)
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=8,
            get_kvcache=lambda: pool,
            translate_kv_indices_for_transfer=lambda x: x,
            translate_swa_indices_for_transfer=lambda x: x,
        )
        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(kv_manager=manager)
        metadata = []
        scheduler.disagg_metadata_buffers = SimpleNamespace(
            set_buf=lambda r: metadata.append(r.rid)
        )
        req.rid = "dsv-long"
        layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(40)])
        model = SimpleNamespace(layers=layers, decoder_replay_tail=object())
        runner = SimpleNamespace(
            model=SimpleNamespace(model=model),
            model_config=SimpleNamespace(hf_config=config, num_hidden_layers=40),
            lora_manager=None,
        )
        scheduler.tp_worker = SimpleNamespace(model_runner=runner)
        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_rank=0, tp_size=1),
            get_disagg=lambda: SimpleNamespace(
                language_model_only=True,
                disaggregation_decode_allocation_policy="immediate",
            ),
            get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
            get_exec=lambda: SimpleNamespace(
                graph=SimpleNamespace(enable_torch_compile=False)
            ),
            get_memory=lambda: SimpleNamespace(
                disable_radix_cache=True,
                enable_unified_memory=False,
                enable_hierarchical_cache=False,
            ),
        )
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": context}):
            controller = ds.DSV41Controller(
                scheduler,
                self.hooks.pre_layerwise.Config("layerwise", Path(root.name), 1),
            )
            # Enum boundary: production uses DisaggregationMode.PREFILL.value.
            scheduler.disaggregation_mode = SimpleNamespace(value="prefill")
            controller.validate()
            scheduler.disaggregation_mode = "prefill"
        scheduler._pre_layerwise = runner._pre_layerwise = controller
        self.addCleanup(
            lambda: [handle.remove() for handle in controller._layer_handles]
        )
        schema = ds.manifest(pool, kv_args, config)
        self.hooks.pre_layerwise.write_json(
            Path(root.name) / "7/0/offer.json",
            {
                "generation": "attempt-1",
                "session": "peer:1",
                "dsv41": schema,
                "mode": "layerwise",
                "tp_size": 1,
                "prefix_len": length,
            },
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
        modules = {
            "sglang.srt.disaggregation.mooncake.conn": SimpleNamespace(
                MooncakeKVSender=sender_cls
            ),
            "sglang.srt.disaggregation.utils": SimpleNamespace(
                DisaggregationMode=SimpleNamespace(PREFILL="prefill")
            ),
            "sglang.srt.disaggregation.base.conn": SimpleNamespace(KVPoll=polls),
        }
        return SimpleNamespace(
            ds=ds,
            controller=controller,
            sender=sender,
            scheduler=scheduler,
            pool=pool,
            schema=schema,
            req=req,
            batch=batch,
            manager=manager,
            tasks=tasks,
            runner=runner,
            modules=modules,
            destination=destination,
            states=target_states,
            native_status=native_status,
            polls=polls,
            metadata=metadata,
            transfers=transfers,
            registration=registration,
            context=context,
            native_send=scheduler_cls.send_kv_chunk,
        )

    def _dsv41_overlap_fixture(self):
        from sglang_remote_attn import prefill_overlap as po

        f = self._dsv41_fixture()
        f.po = po
        f.controller.prefill_overlap = po.PrefillOverlap(f.controller)
        self.addCleanup(f.controller.prefill_overlap.retirement.close)
        f.overlap = f.controller.prefill_overlap
        f.manager._dsv41_p_overlap = f.overlap
        add = f.manager.add_transfer_request
        f.manager.add_transfer_request = lambda *a, **kw: po.track_transfer(
            lambda manager, *args, **kwargs: add(*args, **kwargs), f.manager, *a, **kw
        )
        f.scheduler.send_kv_chunk = lambda req, **kw: (
            self.hooks.pre_layerwise.send_chunk(f.native_send, f.scheduler, req, **kw)
        )
        f.batch.forward_mode.is_prebuilt = lambda: False
        f.batch.copy = lambda: copy.copy(f.batch)
        f.runner.model.model._dsv41_p_overlap = f.overlap
        return f

    def _dsv41_cache(self):
        """Native hybrid component locks/receipts, with storage/LRU at the boundary."""
        from sglang_remote_attn.source_cache import install

        r = self._radix()
        tree_cls = _source_class(
            "mem_cache/unified_cache/unified_tree_core.py",
            "UnifiedTreeCore",
            {
                "inc_lock_ref",
                "dec_lock_ref",
                "_assert_receipt_anchor",
                "_release_components",
            },
        )
        tree_cls.inc_lock_ref.__globals__.update(
            IncLockRefResult=r.IncLockRefResult,
            DecLockRefResult=r.DecLockRefResult,
            ComponentType=r.ComponentType,
        )
        cache_cls = _source_class(
            "mem_cache/unified_radix_cache.py",
            "UnifiedRadixCache",
            {"lock", "unlock", "inc_lock_ref", "dec_lock_ref"},
        )
        cache_cls.lock.__globals__.update(
            TreeLock=r.TreeLock, is_virtual_node=lambda node: False
        )
        base = _source_class(
            "mem_cache/unified_cache/components/base.py",
            "TreeComponent",
            {"value_len", "reclaimable_value_len"},
        )
        classes = [
            (
                _source_class(
                    "mem_cache/unified_cache/components/full.py",
                    "FullComponent",
                    {
                        "acquire_component_lock",
                        "release_component_lock",
                        "redistribute_on_node_split",
                    },
                ),
                r.ComponentType.FULL,
            ),
            (
                _source_class(
                    "mem_cache/unified_cache/components/swa.py",
                    "SWAComponent",
                    {
                        "acquire_component_lock",
                        "release_component_lock",
                        "redistribute_on_node_split",
                    },
                ),
                r.ComponentType.SWA,
            ),
        ]
        node_cls = type("Node", (), {})

        def node(index, size, parent=None):
            value = node_cls()
            value.id, value.parent, value.key = index, parent, list(range(size))
            value.component_data = {
                ct: SimpleNamespace(
                    value=torch.arange(size),
                    host_value=None,
                    lock_ref=0,
                    host_lock_ref=0,
                    session_ref=0,
                    session_ids=None,
                    metadata={},
                )
                for _, ct in classes
            }
            return value

        root, leaf = node(0, 0), node(1, 8)
        leaf.parent = root
        tree = tree_cls()
        tree.root_node = root
        tree.nodes = {0: root, 1: leaf}
        tree.node_by_id = tree.nodes.__getitem__
        tree._update_evictable_leaf_sets = lambda node: None
        tree.evictable_device_leaves = set()
        tree.component_evictable_size_ = {ct: 8 for _, ct in classes}
        tree.component_protected_size_ = {ct: 0 for _, ct in classes}
        tree.lru_lists = tree.host_lru_lists = {ct: None for _, ct in classes}
        components = []
        for cls, ct in classes:
            cls.value_len, cls.reclaimable_value_len = (
                base.value_len,
                base.reclaimable_value_len,
            )
            cls.acquire_component_lock.__globals__.update(
                next_component_uuid=lambda ct: 17
            )
            component = cls()
            component.component_type, component.tree_core = ct, tree
            component.sliding_window_size = 8
            components.append(component)
        tree.components = components
        tree.components_by_type = {c.component_type: c for c in components}
        cache = cache_cls()
        cache.disable, cache.tree_components, cache.tree_core = (
            False,
            (r.ComponentType.FULL, r.ComponentType.SWA),
            tree,
        )
        cache.cache_controller = cache.linker = None
        cache.enable_session_radix_cache = False
        operations = []
        cache.checkpoint = lambda req, **kw: operations.append("checkpoint")
        cache.finish = lambda *a: operations.append("finish")
        cache.reset = lambda: operations.append("reset")
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.radix_cache": SimpleNamespace(
                    RadixCache=r.RadixCache
                ),
                "sglang.srt.mem_cache.unified_radix_cache": SimpleNamespace(
                    UnifiedRadixCache=cache_cls
                ),
                "sglang.srt.mem_cache.unified_cache.component_type": SimpleNamespace(
                    ComponentType=r.ComponentType
                ),
            },
        ):
            guard = install(cache, allow_swa=True)
        return SimpleNamespace(
            cache=cache,
            guard=guard,
            tree=tree,
            leaf=leaf,
            node=node,
            components=components,
            operations=operations,
            native=r,
        )

    def test_dsv41_prefix_cache_pin_keeps_full_swa_receipt_valid_across_split(self):
        f = self._dsv41_cache()
        req = SimpleNamespace(
            cache_request_handle=f.native.CacheRequestHandle("prefix", 0),
            last_node=1,
            lock=f.cache.lock(1),
            kv=SimpleNamespace(cache_protected_len=8),
        )
        lease = f.guard.hold(req)
        parent = f.node(2, 4, f.tree.root_node)
        for component in f.components:
            component.redistribute_on_node_split(parent, f.leaf)
        f.leaf.key = [4, 5, 6, 7]
        f.leaf.parent = parent
        f.tree.nodes[2] = parent
        f.cache.unlock(req.lock)
        req.lock = None
        for ct in f.cache.tree_components:
            self.assertEqual(f.tree.component_protected_size_[ct], 8)
            self.assertEqual(f.leaf.component_data[ct].lock_ref, 1)
            self.assertEqual(parent.component_data[ct].lock_ref, 1)
        f.cache.checkpoint(req, up_to=8)
        self.assertEqual(f.operations, [])
        lease.retire(lambda: f.cache.checkpoint(req, up_to=8))
        self.assertEqual(f.operations, ["checkpoint"])
        for ct in f.cache.tree_components:
            self.assertEqual(f.tree.component_protected_size_[ct], 0)
            self.assertEqual(f.tree.component_evictable_size_[ct], 8)
        self.assertTrue(lease.closed)

    def test_dsv41_cached_request_waits_for_gpu_io_and_tp_before_checkpoint(self):
        f = self._dsv41_overlap_fixture()
        c = self._dsv41_cache()
        f.controller.cache_guard = c.guard
        f.req.cache_request_handle = c.native.CacheRequestHandle(f.req.rid, 0)
        f.req.kv.cache_protected_len = 8
        f.req.last_node, f.req.lock = 1, c.cache.lock(1)

        class Event:
            def record(self, stream):
                pass

            def query(self):
                return True

        def launch(scheduler, batch):
            f.controller.forward(lambda *a: None, f.runner, batch)
            return SimpleNamespace(delay_sample_func=None, copy_done=Event())

        with (
            patch.dict(sys.modules, f.modules),
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "current_stream", lambda: None),
        ):
            first = f.overlap.run_batch(launch, f.scheduler, f.batch)
            lease = f.req._dsv41_p_lease
            f.po.after_result(lambda *a: None, f.scheduler, f.batch, first)
            self.assertIsNotNone(lease.cache_lease)
            f.req.finished = lambda: False
            f.req.skip_radix_cache_insert = False
            _source_function("mem_cache/common.py", "checkpoint_kv_cache")(
                f.req, c.cache
            )
            self.assertEqual(
                f.req.prefix_len, 16
            )  # native chunk progress survives deferred publication
            self.assertEqual(c.operations, [])
            f.req.extend_end, f.req.extend_len = 33, 17
            with patch.object(
                f.po,
                "page_indices",
                lambda raw, page: (
                    (raw[::page] // page).numpy().copy(),
                    Event(),
                    (raw,),
                ),
            ):
                final = f.overlap.run_batch(launch, f.scheduler, f.batch)
            f.po.after_result(
                lambda scheduler, *a: scheduler.send_kv_chunk(f.req, last_chunk=True),
                f.scheduler,
                f.batch,
                final,
            )
            f.overlap.defer_release(
                f.req,
                lambda: (
                    c.cache.checkpoint(f.req, up_to=33),
                    c.cache.unlock(f.req.lock),
                ),
            )
            c.cache.finish(
                f.req.cache_request_handle, c.native.CacheRequestOutcome.SUCCESS
            )
            f.native_status[0] = f.polls.Success
            self.assertFalse(lease.ready_to_finish())
            self.assertEqual(c.operations, [])
            for _ in range(f.tasks.qsize()):
                f.tasks.get_nowait().on_complete()
            self.assertTrue(lease.ready_to_finish())
            f.overlap.poll()
            self.assertEqual(c.operations, [])
            f.overlap.retirement.arm(f.overlap.retirement.poll_command())
            f.overlap.retirement.finalize()
            self.assertEqual(c.operations, ["checkpoint", "finish"])
            self.assertTrue(lease.cache_lease.closed)
            self.assertFalse(c.guard.leases)
            self.assertFalse(f.overlap.leases)
            self.assertEqual(f.overlap.admission.checks, 1)
            for ct in c.cache.tree_components:
                self.assertEqual(c.tree.component_protected_size_[ct], 0)

    def test_dsv41_prefix_cache_serial_final_retains_native_state_completion(self):
        f, c = self._dsv41_fixture(), self._dsv41_cache()
        f.controller.cache_guard = c.guard
        f.req.cache_request_handle = c.native.CacheRequestHandle(f.req.rid, 0)
        f.req.kv.cache_protected_len = 8
        f.req.last_node, f.req.lock = 1, c.cache.lock(1)
        f.req.extend_end, f.req.extend_len = 33, 25
        pl = self.hooks.pre_layerwise
        with patch.dict(sys.modules, f.modules):
            source = f.controller.prepare(f.batch)[0]
            source.enqueue(f.controller.layers - 1, object(), flush=True)
            source.finalize_native(
                f.native_send, f.scheduler, f.req, last_chunk=True, end_idx=None
            )
            self.assertEqual(
                pl.read_json(source.path / "prefill.json")["cached_prefix_tokens"], 8
            )
            self.assertTrue(
                pl.read_json(source.path / "prefill.json")["p_radix_read_lease"]
            )
            pl.defer_release(
                f.req,
                lambda: (
                    c.cache.checkpoint(f.req, up_to=33),
                    c.cache.unlock(f.req.lock),
                ),
            )
            c.cache.finish(
                f.req.cache_request_handle, c.native.CacheRequestOutcome.SUCCESS
            )
            self.assertFalse(source.poll())
            for _ in range(f.tasks.qsize()):
                task = f.tasks.get_nowait()
                if getattr(task, "on_complete", None) is not None:
                    task.on_complete()
            f.native_status[0] = f.polls.Success
            self.assertTrue(source.poll())
            source.retire()
            self.assertEqual(c.operations, ["checkpoint", "finish"])
            self.assertTrue(
                pl.read_json(source.path / "released.json")["p_radix_lease_closed"]
            )
            self.assertFalse(c.guard.leases)

    def test_dsv41_prefix_cache_late_admission_waits_old_native_io_edge(self):
        f, c = self._dsv41_overlap_fixture(), self._dsv41_cache()
        f.controller.cache_guard = c.guard
        f.req.cache_request_handle = c.native.CacheRequestHandle(f.req.rid, 0)
        f.req.kv.cache_protected_len = 8
        f.req.last_node, f.req.lock = 1, c.cache.lock(1)
        f.manager.add_transfer_request(7, [], slice(0, 0), False)
        task = f.tasks.get_nowait()
        with patch.dict(sys.modules, f.modules):
            self.assertEqual(f.overlap.admission.select(f.batch)[1], [0])
            self.assertFalse(c.guard.leases)
            for _ in range(8):
                f.overlap.poll()
                self.hooks._process_requests(
                    lambda *a: None, f.scheduler, f.overlap.inject([])
                )
            self.assertEqual(f.overlap.admission.revalidations, 0)
            task.on_complete()
            f.overlap.poll()
            self.hooks._process_requests(
                lambda *a: None, f.scheduler, f.overlap.inject([])
            )
            self.assertEqual(f.overlap.admission.select(f.batch)[1], [1])
            self.assertEqual(f.overlap.admission.revalidations, 1)
            self.assertIsNotNone(f.req._dsv41_p_certificate.cache_lease)
            f.overlap.admission.retired(f.req)
            c.cache.unlock(f.req.lock)
            self.assertFalse(c.guard.leases)
            for ct in c.cache.tree_components:
                self.assertEqual(c.tree.component_protected_size_[ct], 0)

    def test_dsv41_prefix_cache_cancel_without_release_keeps_native_request_pin(self):
        f, c = self._dsv41_overlap_fixture(), self._dsv41_cache()
        f.controller.cache_guard = c.guard
        f.req.cache_request_handle = c.native.CacheRequestHandle(f.req.rid, 0)
        f.req.kv.cache_protected_len = 8
        f.req.last_node, f.req.lock = 1, c.cache.lock(1)
        with patch.dict(sys.modules, f.modules):
            f.overlap.admission.select(f.batch)
            cert = f.req._dsv41_p_certificate
            chunk = f.po.Chunk(
                f.req, f.sender, f.req.cache_request_handle, 0, 0, 16, False
            )
            lease = f.po.SourceLease(
                f.overlap, chunk, 1, "attempt-1", cache_lease=cert.cache_lease
            )
            event = SimpleNamespace(query=lambda: False)
            plan = f.po.LaunchPlan(1, f.batch, (chunk,), [], [])
            plan.forward_done = event
            plan.proof.result = SimpleNamespace(
                delay_sample_func=lambda: None, copy_done=None
            )
            lease.plans.append(plan)
            marker = object()
            lease.ledger.pending.add(marker)
            f.req._dsv41_p_lease = lease
            f.overlap.leases.append(lease)
            f.overlap.by_room[7] = lease
            f.overlap.retirement.register(lease)

            def complete():
                event.query = lambda: True
                with lease.ledger.lock:
                    lease.ledger.pending.remove(marker)

            timer = threading.Timer(0.04, complete)
            timer.start()
            trace = []

            def native_reset(scheduler, req):
                self.assertTrue(lease.closed)
                self.assertTrue(lease.cache_lease.closed)
                for ct in c.cache.tree_components:
                    self.assertEqual(c.tree.component_protected_size_[ct], 8)
                c.cache.unlock(req.lock)
                c.cache.finish(
                    req.cache_request_handle, c.native.CacheRequestOutcome.ABORT
                )
                trace.append("native-reset")

            try:
                f.po.before_mutation(native_reset, f.scheduler, f.req)
            finally:
                timer.join()
            self.assertEqual(trace, ["native-reset"])
            self.assertEqual(c.operations, ["finish"])
            self.assertFalse(c.guard.leases)
            self.assertEqual(f.overlap.retirement.final_votes, 1)
            for ct in c.cache.tree_components:
                self.assertEqual(c.tree.component_protected_size_[ct], 0)

    def test_dsv41_certificate_stable_chunks_and_final_reuse_one_tp_decision(self):
        f = self._dsv41_overlap_fixture()
        votes = []
        original = f.controller.agree
        f.controller.agree = lambda flags: votes.append(list(flags)) or original(flags)
        # SourceOwner was constructed before the counter; count admission only.
        with patch.dict(sys.modules, f.modules):
            for end in (8, 16, 24, 32, 33):
                f.req.extend_end = end
                candidates, accepted = f.overlap.admission.select(f.batch)
                self.assertEqual(accepted, [1])
                self.assertIs(candidates[0][0], f.req)
                if end != 33:
                    f.req.start_send_idx, f.sender.curr_idx = end, end // 8
                f.overlap.poll()
                # ACK-only traffic must never become a revalidation vote.
                for message in f.overlap.inject([]) or []:
                    from sglang_remote_attn.prefill_admission import CONTROL

                    if CONTROL in message:
                        f.overlap.admission.arm(message[CONTROL])
                f.overlap.finalize_control()
            self.assertEqual(votes, [[1, 1]])
            self.assertEqual(f.overlap.admission.checks, 1)
            self.assertEqual(f.overlap.admission.revalidations, 0)
            self.assertFalse(hasattr(f.req._dsv41_p_certificate.batch, "input_ids"))
            # TP-symmetric native retry/reset clears the old attempt's pins.
            self.hooks._reset_retracted(
                lambda req: setattr(req, "retraction_count", 1), f.req
            )
            f.req.start_send_idx, f.sender.curr_idx = 0, 0
            f.overlap.admission.select(f.batch)
            self.assertEqual(len(votes), 2)
            self.assertEqual(len(f.overlap.admission.records), 1)

    def test_dsv41_certificate_negative_cache_revalidates_only_after_bootstrap_edge(
        self,
    ):
        f = self._dsv41_overlap_fixture()
        f.req.pending_bootstrap = True
        votes = []
        original = f.controller.agree
        f.controller.agree = lambda flags: votes.append(list(flags)) or original(flags)
        with patch.dict(sys.modules, f.modules):
            for _ in range(8):
                self.assertEqual(f.overlap.admission.select(f.batch)[1], [0])
                f.overlap.poll()
                self.hooks._process_requests(
                    lambda *a: None, f.scheduler, f.overlap.inject([])
                )
            self.assertEqual(votes, [[1, 0]])
            f.req.pending_bootstrap = False
            f.overlap.poll()
            command = f.overlap.inject([])
            self.assertIsNotNone(command)
            self.hooks._process_requests(lambda *a: None, f.scheduler, command)
            self.assertEqual(f.overlap.admission.select(f.batch)[1], [1])
            for _ in range(16):
                f.overlap.poll()
                self.hooks._process_requests(
                    lambda *a: None, f.scheduler, f.overlap.inject([])
                )
            self.assertEqual(len(votes), 2)
            self.assertEqual(f.overlap.admission.revalidations, 1)

    def test_dsv41_certificate_old_notice_cannot_revoke_reused_rid_and_room(self):
        f = self._dsv41_overlap_fixture()
        with patch.dict(sys.modules, f.modules):
            f.overlap.admission.select(f.batch)
            f.overlap.admission.observe()
            old = f.req._dsv41_p_certificate
            delayed = dict(old.notice, revoke=True, revision=old.revision + 1)
            f.overlap.admission.retired(f.req)
            fresh = copy.copy(f.req)
            fresh._dsv41_p_certificate = None
            f.batch.reqs = [fresh]
            f.overlap.admission.select(f.batch)
            new = fresh._dsv41_p_certificate
            self.assertEqual(old.logical_key, new.logical_key)
            self.assertNotEqual(old.key, new.key)
            f.overlap.admission.accept(delayed)
            self.assertIsNone(f.overlap.admission.proposal(allow_command=True))
            self.assertTrue(new.active)

    def test_dsv41_negative_certificate_reset_does_not_drain_another_request(self):
        f = self._dsv41_overlap_fixture()
        with patch.dict(sys.modules, f.modules):
            f.overlap.admission.select(f.batch)
            chunk = f.po.Chunk(f.req, f.sender, None, 0, 0, 16, False)
            active = f.po.SourceLease(f.overlap, chunk, 1, "attempt-1")
            active.plans.append(f.po.LaunchPlan(1, f.batch, (chunk,), [], []))
            f.req._dsv41_p_lease = active
            f.overlap.leases.append(active)
            f.overlap.by_room[7] = active
            f.overlap.retirement.register(active)
            other = copy.copy(f.req)
            other._dsv41_p_lease = other._dsv41_p_certificate = None
            other.pending_bootstrap = True
            other.rid, other.bootstrap_room = "waiting", 8
            f.batch.reqs = [other]
            self.assertEqual(f.overlap.admission.select(f.batch)[1], [0])
            mutated = []
            f.po.before_mutation(lambda *a: mutated.append(True), f.scheduler, other)
            self.assertEqual(mutated, [True])
            self.assertFalse(active.closed)
            self.assertFalse(active.cancelling)
            self.assertEqual(f.overlap.retirement.final_votes, 0)
            self.assertTrue(other._dsv41_p_certificate.done)

    def test_dsv41_certificate_pins_native_registration_until_ordered_gpu_io_revoke(
        self,
    ):
        f = self._dsv41_overlap_fixture()
        f.req.return_logprob = False
        native_abort = _source_function("disaggregation/utils.py", "prepare_abort")
        abort_reason = _source_class("managers/schedule_batch.py", "FINISH_ABORT")
        f.modules["sglang.srt.disaggregation.utils"].prepare_abort = native_abort
        f.modules["sglang.srt.managers.schedule_batch"] = SimpleNamespace(
            FINISH_ABORT=abort_reason
        )
        f.manager._remote_attn_send_lock = threading.Lock()
        f.manager.engine = SimpleNamespace(send_probe=lambda *a: 0)
        native_publish = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"_publish_peer_registration"},
        )._publish_peer_registration
        f.manager.session_lock = threading.Lock()
        f.manager._validate_peer_state_layout = lambda *a: (True, None)
        f.manager._unblacklist_session = lambda *a: None
        f.manager.state_layout_rejections = {}
        f.manager.state_strides_validated = set()
        trace = []
        with patch.dict(sys.modules, f.modules):
            f.overlap.admission.select(f.batch)
            cert = f.req._dsv41_p_certificate
            chunk = f.po.Chunk(f.req, f.sender, None, 0, 0, 16, False)
            lease = f.po.SourceLease(f.overlap, chunk, 1, "attempt-1")
            event = SimpleNamespace(query=lambda: False)
            plan = f.po.LaunchPlan(1, f.batch, (chunk,), [], [])
            plan.forward_done = event
            plan.proof.result = SimpleNamespace(
                delay_sample_func=lambda: None, copy_done=None
            )
            lease.plans.append(plan)
            marker = object()
            lease.ledger.pending.add(marker)
            f.req._dsv41_p_lease = lease
            f.overlap.leases.append(lease)
            f.overlap.by_room[7] = lease
            f.overlap.retirement.register(lease)
            old = f.manager.decode_kv_args_table["peer:1"]
            new = copy.copy(old)
            new.dst_kv_ptrs = [p + 8 for p in old.dst_kv_ptrs]
            self.hooks._publish_peer(native_publish, f.manager, "peer:1", new)
            self.assertIs(f.manager.decode_kv_args_table["peer:1"], old)
            peer = f.manager.transfer_infos[7]["peer:1"]
            replacement = copy.copy(peer)
            replacement.is_dummy = True
            f.manager.transfer_infos[7]["peer:1"] = replacement
            self.assertIs(f.manager.transfer_infos[7]["peer:1"], peer)

            def abort():
                self.assertTrue(lease.closed)
                self.assertFalse(lease.ledger.pending)
                self.assertIs(f.manager.decode_kv_args_table["peer:1"], old)
                trace.append("abort-old-registration")

            f.sender.abort = abort
            f.scheduler.chunked_req = f.req

            def finish_io():
                event.query = lambda: True
                with lease.ledger.lock:
                    lease.ledger.pending.remove(marker)

            timer = threading.Timer(0.04, finish_io)
            timer.start()
            try:
                f.overlap.poll()
                self.hooks._process_requests(
                    lambda *a: None, f.scheduler, f.overlap.inject([])
                )
            finally:
                timer.join()
            self.assertEqual(trace, ["abort-old-registration"])
            self.assertIs(f.manager.decode_kv_args_table["peer:1"], new)
            self.assertIs(f.manager.transfer_infos[7]["peer:1"], replacement)
            self.assertEqual(f.req.finished_reason.to_json()["type"], "abort")
            self.assertIs(f.scheduler._pending_chunked_abort_req, f.req)
            self.assertTrue(cert.done)
            self.assertEqual(f.overlap.admission.revocations, 1)
            self.assertFalse(f.overlap.leases)
            self.assertFalse(f.overlap.admission.guards[id(f.manager)].pins)
            self.assertEqual(f.overlap.admission.select(f.batch)[1], [0])

    def test_dsv41_overlap_interleaves_middle_final_and_other_batch_without_cursor_replay(
        self,
    ):
        f = self._dsv41_overlap_fixture()
        events = []

        class GPUEvent:
            def __init__(self):
                self.done = False
                events.append(self)

            def record(self, stream):
                pass

            def query(self):
                return self.done

        def indices(raw, page):
            event = GPUEvent()
            return (raw[::page] // page).numpy().copy(), event, (raw, event)

        def forward(runner, batch):
            for layer in runner.model.model.layers:
                layer(torch.zeros(1))
            return None

        def launch(scheduler, batch):
            f.controller.forward(forward, f.runner, batch)
            return SimpleNamespace(delay_sample_func=lambda: None, copy_done=GPUEvent())

        with (
            patch.dict(sys.modules, f.modules),
            patch.object(torch.cuda, "Event", GPUEvent),
            patch.object(torch.cuda, "current_stream", lambda: None),
            patch.object(f.po, "page_indices", indices),
        ):
            # Native A-middle sends immediately; A's result remains unprocessed.
            a = f.overlap.run_batch(launch, f.scheduler, f.batch)
            lease = f.req._dsv41_p_lease
            self.assertEqual((f.req.start_send_idx, f.sender.curr_idx), (16, 2))
            self.assertEqual(f.tasks.qsize(), 1)
            # Launch another row before processing A, as the native overlap loop does.
            req_b = copy.copy(f.req)
            req_b._dsv41_p_certificate = None
            req_b.kv = SimpleNamespace(req_pool_idx=1)
            req_b._dsv41_p_lease = None
            req_b.disagg_kv_sender = copy.copy(f.sender)
            req_b.disagg_kv_sender.bootstrap_room = req_b.bootstrap_room = 8
            req_b.rid = "other-row"
            f.batch.reqs = [req_b]  # no matching offer: ordinary native launch
            b = f.overlap.run_batch(launch, f.scheduler, f.batch)
            self.assertFalse(hasattr(b, "_dsv41_p_plan"))
            self.assertIs(a._dsv41_p_plan.chunks[0].req, f.req)
            self.assertIs(a._dsv41_p_plan.batch.reqs[0], f.req)
            f.batch.reqs = [f.req]
            f.req.extend_end, f.req.extend_len = 33, 17
            final = f.overlap.run_batch(launch, f.scheduler, f.batch)
            self.assertIsNot(a._dsv41_p_plan, final._dsv41_p_plan)
            self.assertIs(f.req._dsv41_p_lease, lease)
            self.assertEqual(len(lease.plans), 2)
            self.assertEqual((f.req.start_send_idx, f.sender.curr_idx), (33, 5))
            before = f.tasks.qsize()
            f.po.after_result(
                lambda s, batch, result: s.send_kv_chunk(
                    f.req, last_chunk=False, end_idx=16
                ),
                f.scheduler,
                a._dsv41_p_plan.batch,
                a,
            )
            self.assertEqual(f.tasks.qsize(), before)  # no duplicate middle
            f.po.after_result(
                lambda s, batch, result: s.send_kv_chunk(f.req, last_chunk=True),
                f.scheduler,
                final._dsv41_p_plan.batch,
                final,
            )
            envelopes = [f.tasks.get_nowait() for _ in range(f.tasks.qsize())]
            self.assertEqual(len(envelopes), 6)  # middle + 4 producers + final states
            self.assertEqual(envelopes[0].indices.tolist(), [1, 2])
            self.assertTrue(
                all(t.indices.tolist() == [3, 4, 5] for t in envelopes[1:-1])
            )
            self.assertEqual(envelopes[-1].state_indices[1].tolist(), [0])
            self.assertEqual(envelopes[-1].indices.size, 0)
            self.assertEqual(f.sender._transfer_num_kv_indices, 5)
            freed = []
            self.assertTrue(f.overlap.defer_release(f.req, lambda: freed.append("A")))
            # A ready sender alone is insufficient while CUDA/IO proofs lag.
            f.native_status[0] = f.polls.Success
            self.assertFalse(lease.ready_to_finish())
            for event in events:
                event.done = True
            self.assertFalse(
                lease.ready_to_finish()
            )  # deferred sampling event unrecorded
            a.delay_sample_func = final.delay_sample_func = None
            self.assertFalse(lease.ready_to_finish())  # queued tasks, not just inflight
            for task in envelopes:
                task.on_complete()
            self.assertTrue(lease.ready_to_finish())
            for _ in range(20):
                f.overlap.poll()
            self.assertEqual(f.overlap.retirement.final_votes, 0)
            command = f.overlap.retirement.poll_command()
            f.overlap.retirement.arm(command)
            f.overlap.retirement.finalize()
            self.assertEqual(freed, ["A"])
            self.assertEqual(f.overlap.retirement.final_votes, 1)
            self.assertIsNone(f.req._pre_layerwise_source)
            self.assertIsNone(f.req._dsv41_p_lease)
            self.assertFalse(f.overlap.leases)
            f.po.after_result(lambda *a: None, f.scheduler, f.batch, b)
            b.delay_sample_func = None
            f.overlap.poll()
            self.assertFalse(f.overlap.plans)

    def test_dsv41_overlap_two_final_batches_keep_producers_and_result_owners_separate(
        self,
    ):
        f = self._dsv41_overlap_fixture()
        f.scheduler.req_to_token_pool.req_to_token = (
            f.scheduler.req_to_token_pool.req_to_token.repeat(2, 1)
        )
        f.req.extend_end, f.req.extend_len = 33, 33
        other = copy.copy(f.req)
        other.kv = SimpleNamespace(req_pool_idx=1, is_kv_released=False)
        other.rid, other.bootstrap_room = "other-final", 8
        other.disagg_kv_sender = copy.copy(f.sender)
        other.disagg_kv_sender.bootstrap_room = 8
        f.manager.transfer_infos[8] = f.manager.transfer_infos[7]
        pl = self.hooks.pre_layerwise
        offer = pl.read_json(f.controller.config.directory / "7/0/offer.json")
        pl.write_json(
            f.controller.config.directory / "8/0/offer.json",
            dict(offer, generation="attempt-B"),
        )
        model = f.runner.model.model
        saved_tail = model.decoder_replay_tail
        trace = []

        class Event:
            def record(self, stream):
                pass

            def query(self):
                return True

        def index(raw, page):
            return (raw[::page] // page).numpy().copy(), Event(), (raw,)

        def compute(runner, batch):
            self.assertIs(model.decoder_replay_tail, saved_tail)
            self.assertEqual(
                f.po.tail_plan(
                    lambda *a, **kw: (True, None), SimpleNamespace(_model=model)
                ),
                (False, None),
            )
            for layer in model.layers:
                layer(torch.zeros(1))
            trace.append([s.req.rid for s in f.overlap.current.jobs])

        def launch(scheduler, batch):
            f.controller.forward(compute, f.runner, batch)
            return SimpleNamespace(delay_sample_func=None, copy_done=Event())

        with (
            patch.dict(sys.modules, f.modules),
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "current_stream", lambda: None),
            patch.object(f.po, "page_indices", index),
        ):
            first = f.overlap.run_batch(launch, f.scheduler, f.batch)
            a = first._dsv41_p_plan
            f.batch.reqs = [other]
            second = f.overlap.run_batch(launch, f.scheduler, f.batch)
            b = second._dsv41_p_plan
            self.assertEqual(trace, [["dsv-long"], ["other-final"]])
            self.assertIs(a.jobs[0].req, f.req)
            self.assertIs(b.jobs[0].req, other)
            self.assertEqual(f.tasks.qsize(), 8)
            freed = []
            for req, result, plan in ((f.req, first, a), (other, second, b)):
                f.po.after_result(
                    lambda scheduler, batch, result: scheduler.send_kv_chunk(
                        req, last_chunk=True
                    ),
                    f.scheduler,
                    plan.batch,
                    result,
                )
                f.overlap.defer_release(req, lambda req=req: freed.append(req.rid))
            for _ in range(f.tasks.qsize()):
                f.tasks.get_nowait().on_complete()
            f.native_status[0] = f.polls.Success
            f.overlap.poll()
            f.overlap.retirement.arm(f.overlap.retirement.poll_command())
            f.overlap.retirement.finalize()
            self.assertEqual(freed, ["dsv-long", "other-final"])
            self.assertFalse(f.overlap.leases)
            self.assertEqual(
                f.po.tail_plan(
                    lambda *a, **kw: (True, None), SimpleNamespace(_model=model)
                ),
                (True, None),
            )
            f.overlap.poll()
            self.assertFalse(f.overlap.plans)

    def test_dsv41_overlap_retraction_and_reset_fence_before_native_mutation(self):
        from sglang_remote_attn.decode_overlap import before_retract

        f = self._dsv41_overlap_fixture()
        for hook in (
            lambda original, req: before_retract(original, req=req),
            lambda original, req: self.hooks._reset_retracted(original, req),
            lambda original, req: f.po.before_mutation(
                lambda scheduler, req: original(req), f.scheduler, req
            ),
        ):
            with self.subTest(hook=hook):
                chunk = f.po.Chunk(f.req, f.sender, None, 0, 0, 16, False)
                lease = f.po.SourceLease(
                    f.overlap, chunk, f.overlap.retirement.last_admission + 1, "attempt"
                )
                plan = f.po.LaunchPlan(1, f.batch, (chunk,), [], [])
                event = SimpleNamespace(query=lambda: True)
                plan.forward_done = event
                plan.proof.result = SimpleNamespace(
                    delay_sample_func=None, copy_done=event
                )
                lease.plans.append(plan)
                f.req._dsv41_p_lease = lease
                f.overlap.leases.append(lease)
                f.overlap.by_room[7] = lease
                f.overlap.retirement.register(lease)

                def mutate(*args, **kwargs):
                    self.assertTrue(lease.closed)
                    self.assertIsNone(f.req._dsv41_p_lease)
                    return "native"

                self.assertEqual(hook(mutate, f.req), "native")
                self.assertFalse(f.overlap.leases)
        # A bypassed reset/row reuse is detected rather than freeing the new owner.
        lease.chunk.req.kv.req_pool_idx = 2
        with self.assertRaisesRegex(RuntimeError, "attempt/row changed"):
            lease.check()

    def test_dsv41_overlap_cancel_inherits_queued_native_work_and_fences_before_reset(
        self,
    ):
        f = self._dsv41_overlap_fixture()
        done = SimpleNamespace(query=lambda: False)
        # A native prefix may be queued before plugin admission. Its ledger must survive.
        f.manager.add_transfer_request(7, [], slice(0, 0), False)
        task = f.tasks.get_nowait()
        ledger = f.overlap.ledgers[7]
        chunk = f.po.Chunk(f.req, f.sender, None, 0, 0, 16, False)
        lease = f.po.SourceLease(f.overlap, chunk, 1, "attempt-1", ledger=ledger)
        result = SimpleNamespace(delay_sample_func=lambda: None, copy_done=None)
        plan = f.po.LaunchPlan(1, f.batch, (chunk,), [], [])
        plan.proof.result, plan.forward_done = result, done
        lease.plans.append(plan)
        f.req._dsv41_p_lease = lease
        f.overlap.leases.append(lease)
        f.overlap.by_room[7] = lease
        f.overlap.retirement.register(lease)
        reset = []
        lease.cancelling = True
        self.assertFalse(lease.ready_to_finish())
        done.query = lambda: True
        self.assertFalse(lease.ready_to_finish())  # KVPoll.Failed is not an IO fence
        f.native_status[0] = f.polls.Failed
        task.on_complete()
        # Cancellation need not submit delayed sampling; it never reads source KV.
        f.po.before_mutation(lambda *a: reset.append(True), f.scheduler, f.req)
        self.assertEqual(reset, [True])
        self.assertTrue(lease.closed)
        self.assertFalse(f.overlap.leases)
        self.assertIsNotNone(result.delay_sample_func)

    def test_dsv41_overlap_failed_gpu_or_transfer_retains_lease(self):
        f = self._dsv41_overlap_fixture()
        chunk = f.po.Chunk(f.req, f.sender, None, 0, 0, 16, False)
        lease = f.po.SourceLease(f.overlap, chunk, 1, "attempt-1")
        plan = f.po.LaunchPlan(1, f.batch, (chunk,), [], [])
        plan.proof.result = SimpleNamespace(
            delay_sample_func=None,
            copy_done=SimpleNamespace(
                query=lambda: True, error=ValueError("copy failed")
            ),
        )
        plan.forward_done = SimpleNamespace(query=lambda: True)
        lease.plans.append(plan)
        f.overlap.leases.append(lease)
        f.overlap.by_room[7] = lease
        f.overlap.retirement.register(lease)
        f.req._dsv41_p_lease = lease
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            lease.gpu_ready()
        plan.error = "partial forward"
        with self.assertRaisesRegex(RuntimeError, "retaining KV"):
            lease.gpu_ready()
        self.assertFalse(lease.closed)
        plan.error = None

        def failed(*args, **kwargs):
            raise OSError("queue failed")

        with self.assertRaises(OSError):
            f.po.track_transfer(failed, f.manager, 7, [], slice(0, 0), False)
        with self.assertRaisesRegex(RuntimeError, "retaining KV"):
            f.overlap.drain(f.req)
        self.assertIs(f.req._dsv41_p_lease, lease)
        self.assertFalse(lease.closed)

    def test_dsv41_overlap_release_failure_cannot_retry_allocator_callback(self):
        f = self._dsv41_overlap_fixture()
        chunk = f.po.Chunk(f.req, f.sender, None, 0, 0, 16, False)
        lease = f.po.SourceLease(f.overlap, chunk, 1, "attempt")
        lease.cancelling = True
        calls = []

        def failed_release():
            calls.append(True)
            raise OSError("allocator partially changed")

        lease.release = failed_release
        with self.assertRaises(OSError):
            lease.finish()
        with self.assertRaisesRegex(RuntimeError, "retaining KV"):
            lease.finish()
        self.assertEqual(calls, [True])
        self.assertFalse(lease.closed)

    def test_dsv41_overlap_long_prompt_short_final_tail_uses_prompt_policy(self):
        f = self._dsv41_overlap_fixture()
        selected = []
        f.req.origin_input_ids = list(range(32768))
        f.req.extend_end, f.req.extend_len = 32768, 17
        chunk = f.po.Chunk(f.req, f.sender, None, 0, 32751, 32768, True)
        plan = f.po.LaunchPlan(1, f.batch, (chunk,), [None], [1])
        with patch.object(
            type(f.controller.config),
            "select_group_size",
            lambda self, layers, tokens, chunk: selected.append((tokens, chunk)) or 0,
        ):
            self.assertIsNone(f.controller.prepare(f.batch, plan=plan))
        self.assertEqual(selected, [(32768, 17)])
        auto = dataclasses.replace(f.controller.config, group_size=0)
        self.assertGreater(auto.select_group_size(40, selected[0][0], 17), 0)
        self.assertEqual(auto.select_group_size(40, 17, 17), 0)

    def test_dsv41_native_manifest_maps_shared_kv_and_fp4_scales_once(self):
        f = self._dsv41_fixture()
        groups = f.ds.resource_groups(f.schema, 1)
        self.assertEqual([p for p, _ in groups], [2, 8, 14, 20])
        self.assertEqual(
            sorted(i for _, entries in groups for i in entries), list(range(12))
        )
        self.assertEqual([e[2] for e in f.schema["entries"]].count(20), 3)
        self.assertEqual(f.schema["state_types"], ["SWA", "DSV4_REQUEST_STATE"])
        # Native index-page tiling yields 8/2/2 * payload/scale bytes at ratio 2.
        self.assertEqual([e[-1] for e in f.schema["entries"]][3:9], [8, 8, 8, 2, 2, 2])
        swa = f.pool.swa_kv_pool
        f.pool.request_window = SimpleNamespace(state=swa)
        f.pool.swa_kv_pool = None
        self.assertEqual(
            f.ds.manifest(f.pool, f.manager.kv_args, f.runner.model_config.hf_config),
            f.schema,
        )
        f.pool.request_window, f.pool.swa_kv_pool = None, swa
        f.manager.kv_args.kv_item_lens[0] += 1
        with self.assertRaisesRegex(ValueError, "native PD buffers"):
            f.ds.manifest(f.pool, f.manager.kv_args, f.runner.model_config.hf_config)

    def test_dsv41_layerwise_preserves_native_chunks_final_states_and_gpu_fences(self):
        """Odd final page needs ratio-2 carry; no D admission before native states/aux."""
        import numpy as np

        f = self._dsv41_fixture()
        pl = self.hooks.pre_layerwise
        with patch.dict(sys.modules, f.modules):
            # The original middle chunk still sends every native global entry.
            f.native_send(f.scheduler, f.req, last_chunk=False)
            middle = f.tasks.get_nowait()
            self.assertFalse(middle.last)
            self.assertEqual(middle.indices.tolist(), [1, 2])
            dst_pages = np.array([5, 6, 1, 2, 3], dtype=np.int32)
            f.ds.send_selected(
                type(f.manager).send_kvcache,
                f.manager,
                "peer:1",
                middle.indices,
                f.registration.dst_kv_ptrs,
                dst_pages[middle.index_slice],
                None,
                dst_attn_tp_size=1,
            )
            f.req.extend_end, f.req.extend_len = 33, 17
            jobs = f.controller.prepare(f.batch)
            self.assertEqual(f.sender.curr_idx, 5)
            self.assertEqual(f.req.start_send_idx, 33)
            self.assertEqual(jobs[0].indices.tolist(), [3, 4, 5])
            f.controller.plan = jobs
            events = []

            class GPUEvent:
                def __init__(self):
                    self.done = threading.Event()
                    events.append(self)

                def record(self, stream):
                    pass

                def synchronize(self):
                    if not self.done.wait(3):
                        raise TimeoutError("test GPU proof")

            model = f.runner.model.model
            saved_tail = model.decoder_replay_tail
            trace = []

            def compute(runner, batch):
                self.assertIsNone(model.decoder_replay_tail)
                for i, layer in enumerate(model.layers):
                    layer(torch.zeros(1))
                    trace.append((i, f.tasks.qsize()))
                return "native-output"

            with (
                patch.object(torch.cuda, "Event", GPUEvent),
                patch.object(torch.cuda, "current_stream", lambda: None),
            ):
                self.assertEqual(
                    f.controller.forward(compute, f.runner, f.batch), "native-output"
                )
            self.assertIs(model.decoder_replay_tail, saved_tail)
            self.assertGreater(trace[2][1], 0)
            pl.send_chunk(f.native_send, f.scheduler, f.req, last_chunk=True)
            source = jobs[0]
            self.assertFalse(source.poll())
            freed = []
            self.assertTrue(pl.defer_release(f.req, lambda: freed.append(True)))
            self.assertEqual(freed, [])
            envelopes = [f.tasks.get_nowait() for _ in range(f.tasks.qsize())]
            self.assertEqual(len(envelopes), 5)
            self.assertTrue(envelopes[-1].last)
            self.assertEqual(envelopes[-1].indices.size, 0)
            # Native last-chunk state planner carries an odd ratio-2 half-pair.
            self.assertEqual(envelopes[-1].state_indices[1].tolist(), [0])
            dst_pages = np.array([5, 6, 1, 2, 3], dtype=np.int32)
            completed = threading.Event()
            errors = []

            def transfer():
                try:
                    for task in envelopes:
                        if getattr(task, "wait_event", None) is not None:
                            task.wait_event.synchronize()
                        if task.indices.size:
                            f.ds.send_selected(
                                None,
                                f.manager,
                                "peer:1",
                                task.indices,
                                f.registration.dst_kv_ptrs,
                                dst_pages[task.index_slice],
                                None,
                                dst_attn_tp_size=1,
                                kv_buffer_indices=task.kv_buffer_indices,
                            )
                            task.on_complete()
                        if task.last:
                            for src_ptrs, dst, sizes, indices in zip(
                                f.manager.kv_args.state_data_ptrs,
                                f.states,
                                f.manager.kv_args.state_item_lens,
                                task.state_indices,
                            ):
                                f.manager._send_kvcache_generic(
                                    "peer:1",
                                    src_ptrs,
                                    [t.data_ptr() for t in dst],
                                    sizes,
                                    indices,
                                    indices,
                                    None,
                                    force_flat=True,
                                )
                            f.native_status[0] = f.polls.Success
                    completed.set()
                except BaseException as exc:
                    errors.append(exc)
                    completed.set()

            thread = threading.Thread(target=transfer)
            try:
                thread.start()
                self.assertFalse(completed.wait(0.05))
                self.assertTrue(
                    all(
                        not t[stride : 4 * stride].any()
                        for t, stride in zip(
                            f.destination, f.manager.kv_args.kv_item_lens
                        )
                    )
                )
                self.assertFalse(source.poll())
                # A request flag alone cannot bypass native state completion.
                f.req.kv.is_kv_released = True
                with patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.managers.schedule_batch": SimpleNamespace(
                            FINISH_LENGTH=object
                        )
                    },
                ):
                    f.controller.poll_sources()
                self.assertEqual(freed, [])
                self.assertEqual(f.controller.sources, [source])
            finally:
                for event in events:
                    event.done.set()
                thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(source.poll())
            self.assertEqual(len(source.completed), 12)
            for pointer, stride, dst in zip(
                f.manager.kv_args.kv_data_ptrs,
                f.manager.kv_args.kv_item_lens,
                f.destination,
            ):
                for src_page, dst_page in zip([1, 2, 3, 4, 5], [5, 6, 1, 2, 3]):
                    self.assertEqual(
                        bytes(dst[dst_page * stride : (dst_page + 1) * stride]),
                        ctypes.string_at(pointer + src_page * stride, stride),
                    )
            self.assertTrue(all(t.any() for t in f.states[1]))
            self.assertEqual(f.metadata, ["dsv-long"])
            self.assertEqual(f.scheduler.chunked_prefill_size, 16)
            source.retire()
            self.assertEqual(freed, [True])
            self.assertTrue(source.released)

    def test_dsv41_mismatched_peer_and_ineligible_rows_leave_native_cursors_untouched(
        self,
    ):
        f = self._dsv41_fixture()
        with patch.dict(sys.modules, f.modules):
            f.req.extend_end, f.req.extend_len = 33, 33
            path = f.controller.config.directory / "7/0/offer.json"
            offer = self.hooks.pre_layerwise.read_json(path)
            offer["dsv41"]["entries"][0][-1] += 1
            self.hooks.pre_layerwise.write_json(path, offer)
            self.assertIsNone(f.controller.prepare(f.batch))
            self.assertEqual((f.sender.curr_idx, f.req.start_send_idx), (0, 0))
            offer["dsv41"] = f.schema
            self.hooks.pre_layerwise.write_json(path, offer)
            f.req.multimodal_inputs = object()
            self.assertIsNone(f.controller.prepare(f.batch))
            self.assertEqual((f.sender.curr_idx, f.req.start_send_idx), (0, 0))
            f.req.multimodal_inputs = None
            f.manager.kv_shard_size = 2
            self.assertIsNone(f.controller.prepare(f.batch))
            self.assertEqual((f.sender.curr_idx, f.req.start_send_idx), (0, 0))
            f.manager.kv_shard_size = 1
            f.controller.agree = lambda flags: [
                0
            ]  # another TP rank cannot take this row
            self.assertIsNone(f.controller.prepare(f.batch))
            self.assertEqual(f.controller.sources, [])
            self.assertEqual((f.sender.curr_idx, f.req.start_send_idx), (0, 0))

    def test_dsv41_even_prefix_omits_carry_and_changed_registration_rejects_writes(
        self,
    ):
        import numpy as np

        f = self._dsv41_fixture(length=32)
        with patch.dict(sys.modules, f.modules):
            f.req.extend_end = 32
            f.native_send(f.scheduler, f.req, last_chunk=True)
            final = f.tasks.get_nowait()
            self.assertEqual(final.state_indices[1].size, 0)
        f.manager._dsv41_layerwise_schema = f.schema
        bad = list(f.registration.dst_kv_ptrs)
        bad[0] += 1
        with self.assertRaisesRegex(ValueError, "peer/layout changed"):
            f.ds.send_selected(
                None,
                f.manager,
                "peer:1",
                np.array([1]),
                bad,
                np.array([1]),
                None,
                dst_attn_tp_size=1,
                kv_buffer_indices=(0,),
            )
        self.assertEqual(f.transfers, [])
        with self.assertRaisesRegex(RuntimeError, "finalization"):
            source = f.ds.DSV41Source(
                f.controller,
                f.req,
                Path("unused"),
                "g",
                None,
                None,
                (),
                32,
                groups=((2, (0,)),),
            )
            source.finalize_native(
                f.native_send, f.scheduler, f.req, last_chunk=True, end_idx=None
            )

    def _dsv41_stage_fixture(self, f):
        """Real native byte stager with only CUDA allocation/copy mocked."""
        from sglang.srt.disaggregation.pinned_staging_buffer import PinnedKVStager

        spec = importlib.util.spec_from_file_location(
            "dsv41_core_staging_test",
            ROOT / "python/sglang/srt/disaggregation/mooncake/pinned_staging_core.py",
        )
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)

        class TensorBoundary:
            device = torch.device("cuda", 0)

            def __init__(self, tensor):
                self.tensor = tensor

            def __getattr__(self, key):
                return getattr(self.tensor, key)

            def __getitem__(self, key):
                return self.tensor[key]

            def view(self, *args):
                return TensorBoundary(self.tensor.view(*args))

            def flatten(self, *args):
                return TensorBoundary(self.tensor.flatten(*args))

        for ratio, buffers in f.pool.kv_pools.items():
            buffers.kv_buffer = [TensorBoundary(t) for t in buffers.kv_buffer]
            index = f.pool.index_pools[ratio]
            for name in ("index_k_payload_buffer", "index_k_scale_buffer"):
                setattr(index, name, [TensorBoundary(t) for t in getattr(index, name)])
        proxy = _host_staging_manager()
        stager = core.CorePinnedStager(proxy.engine, capacity=23)
        stager.buffer = PinnedKVStager(proxy, 0, slot_factory=_HostStagingSlot)
        stager.buffer.reserve(23)
        stager._reserve = lambda device: stager.buffer._check_live()
        native = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"ensure_host_stager", "set_kv_buffer_tensors", "_transfer_data"},
        )
        manager = f.manager
        manager._core_stager, manager._host_staging_all = stager, False
        for name in ("ensure_host_stager", "set_kv_buffer_tensors", "_transfer_data"):
            setattr(manager, name, getattr(native, name).__get__(manager))
        calls = []
        slot = stager.buffer.slot
        allowed = list(zip(manager.kv_args.kv_data_ptrs, manager.kv_args.kv_data_lens))
        allowed.append((slot.ptr, slot.capacity))
        allowed += [
            (p, n)
            for pointers, targets in zip(manager.kv_args.state_data_ptrs, f.states)
            for p, target in zip(pointers, targets)
            for n in (target.nbytes,)
        ]

        def te(session, sources, targets, sizes):
            for src, dst, size in zip(sources, targets, sizes):
                self.assertTrue(
                    any(p <= src and src + size <= p + n for p, n in allowed)
                )
                ctypes.memmove(dst, src, size)
            calls.append(list(zip(sources, targets, sizes)))
            return 0

        manager.engine = SimpleNamespace(batch_transfer_sync=te)
        f.controller.config = dataclasses.replace(
            f.controller.config, host_staging=True
        )
        f.scheduler.disaggregation_mode = SimpleNamespace(value="prefill")
        for handle in f.controller._layer_handles:
            handle.remove()
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}):
            f.controller.validate()
        f.scheduler.disaggregation_mode = "prefill"
        stack = ExitStack()
        stack.enter_context(patch.object(torch.cuda, "device", lambda _: nullcontext()))
        stack.enter_context(patch.object(torch.cuda, "stream", lambda _: nullcontext()))
        self.addCleanup(stack.close)
        return stager, calls

    def test_dsv41_host_staging_native_bytes_chunks_states_and_bounded_slot(self):
        import numpy as np

        f = self._dsv41_fixture()
        stager, calls = self._dsv41_stage_fixture(f)
        self.assertIs(f.controller.stager, stager)
        self.assertEqual(len(stager.regions), 12)
        self.assertEqual(f.manager.max_transfer_batch_indices, 2)
        dst_pages = np.array([5, 6, 1, 2, 3], dtype=np.int32)
        with patch.dict(sys.modules, f.modules):
            # Ordinary middle chunks keep native policy: direct unless native
            # host-staging-all is on. Both share the same bound rank slot.
            f.native_send(f.scheduler, f.req, last_chunk=False)
            middle = f.tasks.get_nowait()
            f.manager._host_staging_all = True
            f.ds.send_selected(
                type(f.manager).send_kvcache,
                f.manager,
                "peer:1",
                middle.indices,
                f.registration.dst_kv_ptrs,
                dst_pages[middle.index_slice],
                None,
                dst_attn_tp_size=1,
            )
            f.manager._host_staging_all = False
            f.req.extend_end, f.req.extend_len = 33, 17
            source = f.controller.prepare(f.batch)[0]
            self.assertIs(source.stager, stager)
            source.enqueue(39, None, flush=True)
            for _ in source.groups:
                task = f.tasks.get_nowait()
                self.assertTrue(
                    f.ds.dsv41_staging.is_transfer(f.manager, task.transfer_fn)
                )
                f.ds.send_selected(
                    None,
                    f.manager,
                    "peer:1",
                    task.indices,
                    f.registration.dst_kv_ptrs,
                    dst_pages[task.index_slice],
                    None,
                    dst_attn_tp_size=1,
                    kv_buffer_indices=task.kv_buffer_indices,
                    transfer_fn=task.transfer_fn,
                )
                task.on_complete()
            self.assertEqual(len(source.completed), 12)
            self.assertTrue(calls)
            slot = stager.buffer.slot
            self.assertTrue(
                all(
                    slot.ptr <= p and p + n <= slot.ptr + slot.capacity
                    for blocks in calls
                    for p, _, n in blocks
                )
            )
            self.assertTrue(all(sum(n for _, _, n in blocks) <= 23 for blocks in calls))
            for pointer, stride, dst in zip(
                f.manager.kv_args.kv_data_ptrs,
                f.manager.kv_args.kv_item_lens,
                f.destination,
            ):
                for src_page, dst_page in zip([1, 2, 3, 4, 5], dst_pages):
                    self.assertEqual(
                        bytes(dst[dst_page * stride : (dst_page + 1) * stride]),
                        ctypes.string_at(pointer + src_page * stride, stride),
                    )
            source.finalize_native(
                f.native_send, f.scheduler, f.req, last_chunk=True, end_idx=None
            )
            final = f.tasks.get_nowait()
            self.assertTrue(final.last)
            self.assertEqual(final.indices.size, 0)
            self.assertEqual(final.state_indices[1].tolist(), [0])
            before = len(calls)
            # SWA/carry retain native state planners and are not repacked/staged.
            f.manager._host_staging_all = True
            for srcs, dsts, strides, indices in zip(
                f.manager.kv_args.state_data_ptrs,
                f.states,
                f.manager.kv_args.state_item_lens,
                final.state_indices,
            ):
                f.manager._send_kvcache_generic(
                    "peer:1",
                    srcs,
                    [t.data_ptr() for t in dsts],
                    strides,
                    indices,
                    indices,
                    None,
                    force_flat=True,
                )
            self.assertTrue(
                all(
                    not (slot.ptr <= p < slot.ptr + slot.capacity)
                    for blocks in calls[before:]
                    for p, _, _ in blocks
                )
            )
            self.assertTrue(
                json.loads((source.path / "prefill.json").read_text())["host_staging"]
            )

    def _dsv41_remote_support(self):
        spec = importlib.util.spec_from_file_location(
            "dsv41_remote_probe_support",
            ROOT / "test/manual/remote_attn/dsv41_remote_probe_support.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_dsv41_remote_does_not_enable_qwen_prefill_overlap(self):
        f = self._dsv41_fixture()
        f.runner.model_config.hf_config.model_type = "qwen3"
        schedule = SimpleNamespace(
            disable_overlap_schedule=False, enable_mixed_chunk=False, page_size=1
        )
        disagg = SimpleNamespace(
            disaggregation_mode="prefill", enable_pd_role_switch=False
        )
        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(
                pp_size=1,
                dp_size=1,
                attn_cp_size=1,
                dcp_enabled=False,
                attn_dp_enabled=False,
            ),
            get_memory=f.context.get_memory,
            get_device=lambda: SimpleNamespace(device="cuda"),
            get_schedule=lambda: schedule,
            get_disagg=lambda: disagg,
            get_spec=f.context.get_spec,
            get_exec=lambda: SimpleNamespace(
                overlap=SimpleNamespace(enable_two_batch_overlap=False)
            ),
        )
        modules = {
            "sglang.srt.runtime_context": context,
            "sglang.srt.plugins.hook_registry": SimpleNamespace(
                HookRegistry=SimpleNamespace(_patched=set(self.hooks.HOOKS))
            ),
            "sglang.srt.mem_cache.memory_pool": SimpleNamespace(
                MHATokenToKVPool=object
            ),
            "mooncake.engine": SimpleNamespace(
                TransferEngine=SimpleNamespace(send_probe=lambda: 0)
            ),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"HANDOFF_PRE_LAYERWISE": "remote"}),
        ):
            with self.assertRaisesRegex(RuntimeError, "P overlap is supported only"):
                self.hooks._scheduler_init(lambda *a, **kw: None, f.scheduler)

    def test_dsv41_remote_startup_preserves_overlap_and_binds_only_native_backend(self):
        f = self._dsv41_fixture()
        layout = self._dsv41_remote_support().fixture(ROOT, "cpu").layout
        f.pool.kv_layout = layout.V41
        for handle in f.controller._layer_handles:
            handle.remove()
        f.controller.config = dataclasses.replace(f.controller.config, mode="remote")
        f.scheduler.disaggregation_mode = SimpleNamespace(value="prefill")
        f.scheduler.enable_overlap = True
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}):
            f.controller.validate()
        self.assertTrue(f.controller.dsv41_remote)
        self.assertTrue(f.scheduler.enable_overlap)
        for handle in f.controller._layer_handles:
            handle.remove()

        class NativeBackendBoundary:
            is_dsv41 = True
            trtllm_attn = mega_attn = False

            def _forward_attention(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                compress_ratio,
                save_kv_cache=True,
                attn_sink=None,
                cos_sin_cache=None,
                **kwargs,
            ):
                return "native-local-compute"

        backend = NativeBackendBoundary()
        f.runner.attn_backend = f.runner.decode_attn_backend = backend
        f.runner.decode_attn_backend_group = [backend]
        f.scheduler.disaggregation_mode = SimpleNamespace(value="decode")
        # Ring allocation has its own native CUDA/source-loaded coverage.
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}),
            patch("sglang_remote_attn.dsv41_receive.install") as receive,
        ):
            f.controller.validate()
        receive.assert_called_once_with(f.controller, f.manager, "decode")
        self.assertIs(backend._dsv41_remote, f.controller)
        self.assertEqual(
            backend._forward_attention(
                None,
                None,
                None,
                None,
                SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True)),
                0,
            ),
            "native-local-compute",
        )
        first = backend._forward_attention
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}):
            f.controller.validate()
        self.assertIs(first, backend._forward_attention)
        backend.mega_attn = True
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}):
            with self.assertRaisesRegex(ValueError, "without mega"):
                f.controller.validate()

    def test_dsv41_remote_reads_native_fp4_fp8_layouts_byte_exact(self):
        from sglang_remote_attn.dsv41_sparse_math import read_slots

        support = self._dsv41_remote_support()
        f = support.fixture(ROOT, "cpu")
        slots = torch.tensor([7, 1, 19, 0, 7], dtype=torch.int64)
        for layout in (
            f.layout.V41,
            f.layout.V41_NEW,
            f.layout.V41_FP4,
            f.layout.V41_FP4_NEW,
        ):
            with self.subTest(layout=layout.value):
                raw = support.encoded(layout.encoding, 5, 4, "cpu")
                oracle = (
                    f.oracle.dequantize_k_cache_v41_fp4
                    if layout.is_fp4
                    else f.oracle.dequantize_k_cache_v41
                )
                expected = oracle(raw, 4).reshape(-1, 512)[slots]
                if layout.is_interleaved:
                    raw = f.oracle.interleave_pages(
                        raw, 4, layout.data_bytes, layout.scale_bytes
                    )
                got = read_slots(raw, slots, 4, layout)
                self.assertTrue(
                    torch.equal(got.view(torch.int16), expected.view(torch.int16))
                )
                with self.assertRaisesRegex(ValueError, "outside"):
                    read_slots(raw, torch.tensor([20]), 4, layout)
                self.assertEqual(
                    read_slots(
                        raw, torch.tensor([], dtype=torch.int64), 4, layout
                    ).shape,
                    (0, 512),
                )

    def test_dsv41_remote_sparse_split_matches_union_and_keeps_odd_carry_local(self):
        from sglang_remote_attn import dsv41_sparse_math as math

        support = self._dsv41_remote_support()
        f = support.fixture(ROOT, "cpu")
        calls = []
        originals = [t.clone() for t in f.p_swa] + [
            p.kv_buffer[0].clone() for p in f.p_pool.kv_pools.values()
        ]
        index = f.core.c2_sparse_page_indices.clone()
        carry = torch.arange(512, dtype=torch.float32)
        before = carry.clone()

        def rpc(query, positions, layer, ratio, scale):
            calls.append((positions, f.p_pool.source_layer_of(layer)))
            cpp = f.page // ratio
            ids = torch.tensor([f.p_pages[p // cpp] * cpp + p % cpp for p in positions])
            keys = math.read_slots(
                f.p_pool.get_extra_key_buffer(layer), ids, cpp, f.layout.V41_FP4
            )
            return math.partial(query, keys, scale)

        f.controller.remote_targets[0] = SimpleNamespace(prefix_len=33, call=rpc)
        q, kv = f.query, torch.zeros(1, 512)
        got = f.remote.attention(
            None,
            f.backend,
            q,
            kv,
            kv,
            SimpleNamespace(layer_id=3),
            f.fb,
            2,
            attn_sink=f.sink,
        )
        swa = math.read_slots(
            f.d_pool.get_swa_key_buffer_radix(3),
            torch.tensor([13, 14]),
            8,
            f.layout.V41,
        )
        selected = index[0, :4]
        extra = math.read_slots(
            f.d_pool.get_extra_key_buffer(3), selected, 4, f.layout.V41_FP4
        )
        expected = math.partial(q[0], torch.cat((swa, extra)), 512**-0.5, f.sink)[0].to(
            q.dtype
        )
        torch.testing.assert_close(got[0], expected, atol=0.015625, rtol=0.01)
        self.assertEqual(calls, [([0, 15, 15], 2)])  # Consumer 3 reads source 2.
        self.assertEqual(f.stores, [3])
        self.assertTrue(torch.equal(index, f.core.c2_sparse_page_indices))
        self.assertTrue(torch.equal(carry, before))
        current = f.p_swa + [p.kv_buffer[0] for p in f.p_pool.kv_pools.values()]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(originals, current)))
        self.assertEqual(
            math.partition([28, 7, 24, -1], f.d_pages, 33, 8, 2), ([0, 15], [24])
        )
        # Mixed decode keeps ordinary rows bit-identical to native output and
        # must not store SWA twice while replacing only the Remote row.
        f.fb.req_pool_indices = torch.tensor([0, 1])
        f.fb.seq_lens = torch.tensor([34, 34])
        f.controller.scheduler.req_to_token_pool.req_to_token = f.d_row.repeat(2, 1)
        f.core.swa_page_indices = f.core.swa_page_indices.repeat(2, 1)
        f.core.swa_topk_lengths = f.core.swa_topk_lengths.repeat(2)
        f.core.c2_sparse_page_indices = f.core.c2_sparse_page_indices.repeat(2, 1)
        f.core.c2_sparse_topk_lengths = f.core.c2_sparse_topk_lengths.repeat(2)
        native = torch.arange(2048).reshape(2, 2, 512).to(q.dtype)
        f.stores.clear()

        def native_compute(backend, query, k, v, layer, fb, ratio, **kwargs):
            backend.store_cache(layer.layer_id, k, fb)
            return native

        mixed = f.remote.attention(
            native_compute,
            f.backend,
            q.repeat(2, 1, 1),
            kv,
            kv,
            SimpleNamespace(layer_id=3),
            f.fb,
            2,
            attn_sink=f.sink,
        )
        # Ordinary rows use the same eager union while a peer row is Remote;
        # native full-batch reads would touch its unfinished global pages.
        torch.testing.assert_close(mixed[1], expected, atol=0.015625, rtol=0.01)
        torch.testing.assert_close(mixed[0], expected, atol=0.015625, rtol=0.01)
        self.assertEqual(f.stores, [3])
        self.assertEqual(math.partition([24], f.d_pages, 34, 8, 2), ([16], []))
        with self.assertRaisesRegex(ValueError, "belong"):
            math.partition([0], f.d_pages, 33, 8, 2)
        empty = math.partial(q[0], torch.empty(0, 512), 512**-0.5)
        merged = math.merge(math.partial(q[0], swa, 512**-0.5, f.sink), empty)
        torch.testing.assert_close(
            merged, math.partial(q[0], swa, 512**-0.5, f.sink)[0]
        )

    def test_dsv41_remote_registry_tombstone_holds_cache_until_last_gpu_reader(self):
        from sglang_remote_attn import dsv41_remote as remote
        from sglang_remote_attn.source_cache import CacheGuard

        registry = self.runtime.HandoffRegistry()
        with tempfile.TemporaryDirectory() as directory:
            req = SimpleNamespace(
                cache_request_handle="attempt-1",
                lock="request-pin",
                last_node="node",
                kv=SimpleNamespace(cache_protected_len=8, req_pool_idx=0),
            )
            pins = set()
            cache = SimpleNamespace(
                checkpoint=lambda *a, **kw: None,
                finish=lambda *a: None,
                reset=lambda: None,
                lock=lambda node: pins.add(node) or node,
                unlock=pins.remove,
            )
            guard = CacheGuard(cache)
            source = SimpleNamespace(
                error=None,
                req=req,
                cache_lease=guard.hold(req),
                remote_closed=threading.Event(),
                path=Path(directory),
                generation="g",
                owner=SimpleNamespace(scheduler=None),
            )
            ctx = registry.register(
                "sparse",
                req,
                0,
                33,
                33,
                "token",
                on_release=lambda ctx: source.remote_closed.set(),
            )
            source.service = dict(handoff_id="sparse", epoch=ctx.owner_epoch)
            with patch.object(
                self.runtime,
                "get_remote_hub",
                return_value=SimpleNamespace(registry=registry),
            ):
                self.assertFalse(remote.source_ready(source))
                self.assertIs(registry.acquire("sparse", ctx.owner_epoch), ctx)
                self.hooks.pre_layerwise.write_json(
                    source.path / "remote-done.json", dict(generation="stale")
                )
                self.assertFalse(remote.source_ready(source))
                self.assertFalse(ctx.closed)
                self.hooks.pre_layerwise.write_json(
                    source.path / "remote-done.json", dict(generation="g")
                )
                self.assertFalse(remote.source_ready(source))
                self.assertIsNone(registry.acquire("sparse", ctx.owner_epoch))
                self.assertEqual(pins, {"node"})
                registry.release_ref("sparse", ctx.owner_epoch)
                self.assertTrue(remote.source_ready(source))
                self.assertEqual(pins, {"node"})  # TP/native release still required.
                source.cache_lease.retire(lambda: None)
                self.assertFalse(pins)
                req._handoff_transfer_unsafe = True
                with self.assertRaisesRegex(RuntimeError, "GPU read fence"):
                    remote.source_ready(source)

    def test_dsv41_remote_post_pd_commit_one_decode_then_native_graph(self):
        from sglang_remote_attn import dsv41_remote as remote

        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": SimpleNamespace(
                        get_parallel=lambda: SimpleNamespace(tp_size=1, tp_rank=0)
                    )
                },
            )
        )
        f = self._dsv41_remote_support().fixture(ROOT, "cpu")
        f.controller.graph_fast_path = True
        f.controller.plan = f.controller.executing = None
        f.controller.dsv41_layerwise = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            offer = SimpleNamespace(path=path, generation="g")
            dr = SimpleNamespace(req=f.req, _pre_layerwise_offer=offer)
            queue = SimpleNamespace(
                scheduler=SimpleNamespace(_pre_layerwise=f.controller)
            )
            self.hooks.pre_layerwise.write_json(
                path / "remote-service.json",
                dict(generation="g", prefix_len=33, route={}),
            )
            remote.commit(lambda *a, **kw: None, queue, dr)
            target = f.req._dsv41_remote
            f.req.kv.req_pool_idx = 99
            with self.assertRaisesRegex(RuntimeError, "attempt/row changed"):
                target.call(f.query[0], [0], 3, 2, 0.01)
            f.req.kv.req_pool_idx = 0
            runner = SimpleNamespace(
                model_runner=SimpleNamespace(_pre_layerwise=f.controller)
            )
            self.assertFalse(self.hooks._decode_graph(lambda *a: True, runner, f.fb))
            self.assertEqual(
                remote.forward(f.controller, lambda *a: "native-result", None, f.fb),
                "native-result",
            )
            self.assertTrue(target.closed)
            self.assertFalse(f.controller.remote_targets)
            self.assertTrue(self.hooks._decode_graph(lambda *a: True, runner, f.fb))
            result = json.loads((path / "remote-done.json").read_text())
            self.assertEqual(result["reason"], "first-post-PD-eager-decode-complete")
            self.assertEqual(result["generation"], "g")

    def _dsv41_receive_fixture(self, *, capacity=128):
        import struct

        from sglang_remote_attn import dsv41_receive as rx

        f = self._dsv41_fixture()
        f.stager, f.calls = self._dsv41_stage_fixture(f)
        r = self._native_ring_fixture(capacity=capacity)
        r.handler._wm_subscribers.clear()
        r.receiver.chunk_staging_infos.clear()
        r.receiver.chunk_staging_layers.clear()
        r.receiver.prefill_info.attn_tp_size = 1
        r.handler.decode_tp = 1
        r.manager.attn_tp_size = 1
        r.manager._dsv41_layerwise_schema = f.schema
        r.manager.kv_args = copy.copy(f.manager.kv_args)
        r.manager.kv_args.page_size = f.manager.kv_args.page_size = 8
        f.registration.dst_kv_item_len = f.manager.kv_args.kv_item_lens[0]
        r.manager.kv_args.kv_data_ptrs = f.registration.dst_kv_ptrs
        r.manager._staging_ctx.allocator = r.allocator
        r.manager._staging_handler = r.handler
        r.handler.kv_buffer_info = dict(
            k_buffers=f.destination, v_buffers=[], page_size=8
        )
        row = torch.cat([torch.arange(p * 8, p * 8 + 8) for p in (5, 6, 1, 2, 3)])
        r.handler.scheduler.req_to_token_pool.req_to_token = row.reshape(1, -1)
        r.dr.req.origin_input_ids = list(range(33))
        r.handler._scatter_region = lambda *a, **kw: rx.scatter(
            None, r.handler, *a, **kw
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.utils": SimpleNamespace(
                        create_device_stream=lambda device: SimpleNamespace(
                            wait_event=lambda event: None,
                            synchronize=lambda: [
                                setattr(e, "complete", True) for e in r.events
                            ],
                        ),
                        device_stream_context=lambda stream: nullcontext(),
                    ),
                    "sglang.srt.disaggregation.common.staging_handler": SimpleNamespace(
                        PrefillStagingContext=lambda: SimpleNamespace(
                            remote_watermarks={}, watermark_cv=threading.Condition()
                        )
                    ),
                },
            )
        )
        requests = []
        f.manager._staging_ctx = None
        f.manager.request_layer_staging = lambda room, span, groups: requests.extend(
            [
                b"STAGING_REQ",
                str(room).encode(),
                str(i).encode(),
                str(span.stop - span.start).encode(),
                b"peer:1",
                b"0",
                struct.pack(f"{len(indices)}Q", *indices),
                str(span.start).encode(),
            ]
            for i, (indices, _) in enumerate(groups)
        )
        f.registration.staging_base_ptr = r.allocator.get_base_ptr()
        f.registration.staging_total_size = r.allocator.total_size
        offer_path = f.controller.config.directory / "7/0/offer.json"
        offer = json.loads(offer_path.read_text())
        offer.update(native_staging=True, recv_host_staging=True, staging_wire=rx.WIRE)
        self.hooks.pre_layerwise.write_json(offer_path, offer)
        f.req.start_send_idx, f.sender.curr_idx = 16, 2
        f.req.extend_end, f.req.extend_len = 33, 17
        f.rx, f.ring, f.requests = rx, r, requests
        return f

    def test_dsv41_receive_native_ring_roundtrip_and_gpu_completion_barrier(self):
        f = self._dsv41_receive_fixture()
        r = f.ring
        with patch.dict(sys.modules, f.modules):
            source = f.controller.prepare(f.batch)[0]
        self.assertTrue(source.native_staging and source.recv_host_staging)
        self.assertEqual(len(f.requests), len(source.groups))
        source.enqueue(39, None, flush=True)
        # Allocate all groups first. Native overcommit and watermark prevent
        # a later round from overwriting an earlier unconsumed pinned interval.
        for msg in f.requests:
            f.rx.allocate(None, r.manager, msg)
        self.assertEqual(len(r.allocator.allocations), 4)
        strategy_cls = _source_class(
            "disaggregation/common/staging_handler.py",
            "PrefillStagingStrategy",
            {"check_ready"},
        )
        strategy = strategy_cls()
        strategy.kv_manager = f.manager
        watermark = _source_function(
            "disaggregation/common/staging_handler.py", "is_watermark_ready"
        )
        f.manager._is_watermark_ready = lambda session, rnd, end: watermark(
            f.manager._staging_ctx, session, rnd, end
        )
        f.manager._staging_ctx.remote_watermarks["peer:1"] = r.allocator.get_watermark()
        # Canonical Success may arrive before the last CHUNK_READY is consumed.
        r.handler.submit_last_scatter_async(7)
        for ordinal, (_, entries) in enumerate(source.groups):
            task = f.tasks.get_nowait()
            task.prefill_kv_indices = task.indices
            ready, _, offset, rnd, end = strategy.check_ready(
                SimpleNamespace(
                    staging=SimpleNamespace(
                        offsets=[i[1] for i in r.receiver.chunk_staging_infos],
                        rounds=[i[2] for i in r.receiver.chunk_staging_infos],
                        ends=[i[3] for i in r.receiver.chunk_staging_infos],
                    ),
                    mooncake_session_id="peer:1",
                ),
                2,
                3,
                chunk_idx=ordinal,
            )
            self.assertTrue(ready)
            self.assertEqual(task.staging_chunk_idx, ordinal)
            self.assertEqual(task.staging_layer_ids, entries)
            self.assertEqual(
                f.rx.send_group(
                    None,
                    f.manager,
                    SimpleNamespace(mooncake_session_id="peer:1"),
                    task,
                    f.registration,
                    offset,
                    None,
                ),
                0,
            )
            task.on_complete()
            self.assertTrue(r.handler.handle_chunk_arrived(7, ordinal, 2, 3, "peer:1"))
            r.handler.advance_scatter(r.dr)
            self.assertFalse(r.handler.is_done(r.dr))
            self.assertTrue(r.allocator.allocations)
            # Duplicate allocation after consumption cannot reassign a slot.
            before = dict(r.allocator.allocations)
            f.rx.allocate(None, r.manager, f.requests[ordinal])
            self.assertEqual(r.allocator.allocations, before)
            for e in r.events:
                e.complete = True
            r.handler.advance_scatter(r.dr)
            f.manager._staging_ctx.remote_watermarks["peer:1"] = (
                r.allocator.get_watermark()
            )
        self.assertTrue(r.handler.is_done(r.dr))
        self.assertFalse(r.allocator.allocations)
        self.assertEqual(len(source.completed), 12)
        for ptr, stride, dst in zip(
            f.manager.kv_args.kv_data_ptrs,
            f.manager.kv_args.kv_item_lens,
            f.destination,
        ):
            for sp, dp in zip((3, 4, 5), (1, 2, 3)):
                self.assertEqual(
                    bytes(dst[dp * stride : (dp + 1) * stride]),
                    ctypes.string_at(ptr + sp * stride, stride),
                )
            self.assertFalse(
                dst[5 * stride : 7 * stride].any()
            )  # Middle pages unchanged.
        self.assertEqual(f.manager.max_transfer_batch_indices, 2)
        self.assertGreater(r.allocator.get_watermark()[0], 0)

    def test_dsv41_receive_rejects_oversize_overlap_and_changed_envelopes(self):
        import struct

        f = self._dsv41_receive_fixture(capacity=16)
        with patch.dict(sys.modules, f.modules):
            source = f.controller.prepare(f.batch)[0]
        r = f.ring
        f.rx.allocate(None, r.manager, f.requests[0])
        self.assertEqual(r.receiver.chunk_staging_infos[0][1], -2)
        self.assertFalse(r.allocator.allocations)
        repeated = list(f.requests[0])
        repeated[3] = b"2"
        with self.assertRaisesRegex(ValueError, "envelope"):
            f.rx.allocate(None, r.manager, repeated)
        repeated = list(f.requests[0])
        repeated[2] = b"1"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            f.rx.allocate(None, r.manager, repeated)
        repeated = list(f.requests[0])
        repeated[6] = struct.pack("Q", 999)
        with self.assertRaisesRegex(ValueError, "entries"):
            f.rx.allocate(None, r.manager, repeated)
        repeated = list(f.requests[0])
        repeated[2] = b"9999999"
        with self.assertRaisesRegex(ValueError, "envelope"):
            f.rx.allocate(None, r.manager, repeated)
        self.assertFalse(r.allocator.allocations)
        from sglang_remote_attn.prefill_overlap import SourceLease

        lease = SourceLease.__new__(SourceLease)
        lease.check = lambda: None
        lease.chunk = SimpleNamespace(sender=f.sender)
        f.native_status[0] = f.polls.Failed
        with patch.dict(sys.modules, f.modules):
            with self.assertRaisesRegex(RuntimeError, "transport failed"):
                lease.transport_ready()
        # Fail before publication when a peer advertises unknown wire semantics.
        offer_path = f.controller.config.directory / "7/0/offer.json"
        offer = json.loads(offer_path.read_text())
        offer["staging_wire"] = "foreign"
        self.hooks.pre_layerwise.write_json(offer_path, offer)
        f.req._pre_layerwise_source = None
        with patch.dict(sys.modules, f.modules):
            self.assertEqual(f.controller.candidates(f.batch), [None])

    def test_dsv41_receive_cancel_drains_h2d_before_native_page_free(self):
        f = self._dsv41_receive_fixture()
        r = f.ring
        with patch.dict(sys.modules, f.modules):
            f.controller.prepare(f.batch)
        f.rx.allocate(None, r.manager, f.requests[0])
        r.handler.handle_chunk_arrived(7, 0, 2, 3, "peer:1")
        self.assertTrue(r.allocator.allocations)
        self.assertFalse(r.events[-1].complete)
        # The actual native release routine drains the stream before freeing
        # issued/unsent allocations. Simulate its successful GPU fence only.
        r.allocator._scatter_stream.synchronize = lambda: [
            setattr(e, "complete", True) for e in r.events
        ]
        r.handler.unregister_decode_req(7)
        self.assertTrue(r.events[-1].complete)
        self.assertFalse(r.allocator.allocations)
        self.assertFalse(r.dr._chunk_events)
        self.assertFalse(r.handler.is_staging_room(7))
        # Late messages cannot resurrect the old room or allocate more memory.
        f.rx.allocate(None, r.manager, f.requests[0])
        self.assertFalse(r.allocator.allocations)

    def test_dsv41_receive_install_offer_and_clean_teardown_use_native_lifetime(self):
        f = self._dsv41_receive_fixture()
        r = f.ring
        f.scheduler.disaggregation_mode = SimpleNamespace(value="decode")
        f.scheduler.schedule_stream = r.handler.scheduler.schedule_stream
        f.scheduler.disagg_decode_prealloc_queue.transfer_queue = SimpleNamespace()
        f.controller.config = dataclasses.replace(
            f.controller.config, host_staging=False, recv_host_staging=True
        )
        ctx = _source_class(
            "disaggregation/common/staging_handler.py", "DecodeStagingContext"
        )
        native = sys.modules["sglang.srt.disaggregation.common.staging_handler"]
        registered, deregistered = [], []
        f.manager._register_staging_memory = lambda pointer, size: registered.append(
            (pointer, size)
        )

        def allocate(register, kv_args, **kw):
            self.assertTrue(kw["pin_memory"])
            register(r.allocator.get_base_ptr(), r.allocator.total_size)
            return r.allocator

        f.manager._worker_threads = []
        f.manager.engine.batch_deregister = lambda pointers: (
            deregistered.extend(pointers) or 0
        )
        f.manager.get_session_id = lambda: "peer:1"
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}),
            patch.multiple(
                native,
                DecodeStagingContext=ctx,
                DecodeStagingHandler=type(r.handler),
                init_staging_allocator=allocate,
                create=True,
            ),
        ):
            f.controller.validate()
        queue = f.scheduler.disagg_decode_prealloc_queue.transfer_queue
        self.assertTrue(queue.enable_staging)
        self.assertTrue(f.manager.enable_staging)
        self.assertIs(queue.staging_handler, f.manager._staging_handler)
        self.assertEqual(
            registered, [(r.allocator.get_base_ptr(), r.allocator.total_size)]
        )
        r.receiver.kv_mgr = f.manager
        r.receiver.poll = lambda: f.polls.Transferring
        r.dr.is_rebootstrap = False
        queue._is_host_destination = lambda dr: False
        queue.queue = [r.dr]
        modules = dict(f.modules)
        modules["sglang.srt.disaggregation.mooncake.conn"] = SimpleNamespace(
            MooncakeKVReceiver=type(r.receiver)
        )
        with patch.dict(sys.modules, modules):
            admitted, offered = f.controller.early_targets(queue, None)
        self.assertEqual(admitted, [])
        self.assertTrue(offered[0].native_staging)
        self.assertTrue(r.receiver.require_staging)
        self.assertIsNone(r.dr.req._pre_layerwise_native_target)
        offer = json.loads(
            (f.controller.config.directory / "7/0/offer.json").read_text()
        )
        self.assertEqual(offer["staging_wire"], f.rx.WIRE)
        self.assertTrue(offer["recv_host_staging"])
        with self.assertRaisesRegex(RuntimeError, "still active"):
            f.rx.teardown(
                lambda manager: self.fail("unsafe native teardown"), f.manager
            )
        f.manager._staging_handler.unregister_decode_req(7)
        f.rx.teardown(lambda manager: None, f.manager)
        self.assertEqual(deregistered, [r.allocator.get_base_ptr()])
        self.assertEqual(f.manager.engine._dsv41_receive_owners, [])

    def test_dsv41_receive_failed_gpu_drain_quarantines_ring(self):
        f = self._dsv41_receive_fixture()
        r = f.ring
        with patch.dict(sys.modules, f.modules):
            f.controller.prepare(f.batch)
        f.rx.allocate(None, r.manager, f.requests[0])
        r.handler.handle_chunk_arrived(7, 0, 2, 3, "peer:1")

        def failed():
            raise RuntimeError("injected GPU fence failure")

        r.allocator._scatter_stream.synchronize = failed
        with self.assertRaisesRegex(RuntimeError, "GPU fence failure"):
            r.handler.unregister_decode_req(7)
        self.assertTrue(r.allocator.allocations)
        self.assertTrue(r.dr._chunk_events)
        self.assertFalse(r.events[-1].complete)

    def test_dsv41_host_staging_startup_roles_and_native_all_policy(self):
        f = self._dsv41_fixture()
        stager, _ = self._dsv41_stage_fixture(f)
        f.controller.config = dataclasses.replace(
            f.controller.config, host_staging=False
        )
        f.manager._host_staging_all = True
        f.scheduler.disaggregation_mode = SimpleNamespace(value="prefill")
        f.scheduler.enable_overlap = True
        for handle in f.controller._layer_handles:
            handle.remove()
        memory = f.context.get_memory()
        memory.disable_radix_cache = False
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}),
            patch.object(f.context, "get_memory", lambda: memory),
        ):
            f.controller.validate()
        self.assertIs(f.controller.stager, stager)
        self.assertTrue(f.scheduler.enable_overlap)
        self.assertFalse(memory.disable_radix_cache)
        f.scheduler.disaggregation_mode = SimpleNamespace(value="decode")
        memory.disable_radix_cache = True
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}):
            with self.assertRaisesRegex(ValueError, "P send / D receive roles"):
                f.controller.validate()
            f.manager._host_staging_all = False
            f.controller.config = dataclasses.replace(
                f.controller.config, host_staging=True
            )
            with self.assertRaisesRegex(ValueError, "P send / D receive roles"):
                f.controller.validate()
            f.scheduler.disaggregation_mode = SimpleNamespace(value="prefill")
            f.controller.config = dataclasses.replace(
                f.controller.config, recv_host_staging=True
            )
            with self.assertRaisesRegex(ValueError, "P send / D receive roles"):
                f.controller.validate()

    def test_dsv41_host_staging_layout_and_callback_fail_closed(self):
        f = self._dsv41_fixture()
        stager, calls = self._dsv41_stage_fixture(f)
        stage = f.ds.dsv41_staging
        f.manager._dsv41_layerwise_schema = f.schema
        original = f.manager.kv_args.kv_item_lens[0]
        f.manager.kv_args.kv_item_lens[0] = original + 1
        with self.assertRaisesRegex(ValueError, "tensor/stride"):
            stage.bind(f.pool, f.manager)
        f.manager.kv_args.kv_item_lens[0] = original
        f.manager.kv_args.kv_data_lens[0] -= 1
        with self.assertRaisesRegex(ValueError, "tensor/stride"):
            stage.bind(f.pool, f.manager)
        f.manager.kv_args.kv_data_lens[0] += 1
        for callback in (
            lambda *a: 0,
            functools.partial(f.manager._transfer_data, stage=False),
        ):
            with self.assertRaisesRegex(ValueError, "peer/layout changed"):
                f.ds.send_selected(
                    None,
                    f.manager,
                    "peer:1",
                    [],
                    f.registration.dst_kv_ptrs,
                    [],
                    None,
                    dst_attn_tp_size=1,
                    kv_buffer_indices=(0,),
                    transfer_fn=callback,
                )
        self.assertEqual(calls, [])
        # Native TE error poisons the shared slot. No direct-GPU retry.
        f.manager._dsv41_layerwise_schema = f.schema
        f.manager.engine.batch_transfer_sync = lambda *a: -1
        src = f.manager.kv_args.kv_data_ptrs[0]
        dst = f.registration.dst_kv_ptrs[0]
        self.assertEqual(
            f.manager._transfer_data("peer:1", [(src, dst, 8)], stage=True), -1
        )
        self.assertTrue(stager.poisoned)
        self.assertFalse(stager.close())
        self.assertIn(stager, stager.engine._core_staging_owners)
        with patch.dict(sys.modules, f.modules):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                f.controller.candidates(f.batch, allow_middle=True)
        from sglang_remote_attn.prefill_overlap import SourceLease

        lease = SourceLease.__new__(SourceLease)
        lease.check = lambda: None
        lease.chunk = SimpleNamespace(sender=f.sender)
        # Poison is observed before the outstanding-IO early exit; this method
        # is polled by ordered TP retirement, not by rank-local KV free hooks.
        with self.assertRaisesRegex(RuntimeError, "retaining KV and pinned slot"):
            lease.transport_ready()

    def test_dsv41_startup_rejects_invalid_mode_multimodal_speculation_and_unsafe_storage(
        self,
    ):
        f = self._dsv41_fixture()
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}):
            f.controller.config = dataclasses.replace(f.controller.config, mode="off")
            with self.assertRaisesRegex(ValueError, "layerwise or"):
                f.controller.validate()
            f.controller.config = dataclasses.replace(
                f.controller.config, mode="layerwise"
            )
            for method, value, reason in (
                (
                    "get_disagg",
                    SimpleNamespace(language_model_only=False),
                    "language-model-only",
                ),
                (
                    "get_spec",
                    SimpleNamespace(speculative_algorithm="DSPARK"),
                    "speculation disabled",
                ),
            ):
                with patch.object(f.context, method, lambda: value):
                    with self.assertRaisesRegex(ValueError, reason):
                        f.controller.validate()
        for handle in f.controller._layer_handles:
            handle.remove()
        f.scheduler.disaggregation_mode = SimpleNamespace(value="prefill")
        memory = f.context.get_memory()
        memory.disable_radix_cache = False
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}),
            patch.object(f.context, "get_memory", lambda: memory),
        ):
            f.controller.validate()  # P enabled cache is accepted; D remains gated.
        f.scheduler.disaggregation_mode = SimpleNamespace(value="decode")
        memory = f.context.get_memory()
        memory.disable_radix_cache = False
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}),
            patch.object(f.context, "get_memory", lambda: memory),
        ):
            with self.assertRaisesRegex(ValueError, "disable-radix-cache on D"):
                f.controller.validate()
        f.pool.uniform_fp8 = True
        with self.assertRaisesRegex(ValueError, "non-unified FlashMLA"):
            f.ds.manifest(f.pool, f.manager.kv_args, f.runner.model_config.hf_config)

    def test_dsv41_prefill_export_bypasses_graph_but_native_decode_keeps_graph(self):
        f = self._dsv41_fixture()
        f.controller.graph_fast_path = True
        runner = SimpleNamespace(model_runner=f.runner)
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}):
            f.controller.plan = [object()]
            self.assertFalse(
                self.hooks._prefill_graph(lambda *a: True, runner, f.batch)
            )
            f.controller.plan = None
            f.controller.executing = [object()]
            self.assertFalse(
                self.hooks._prefill_graph(lambda *a: True, runner, f.batch)
            )
            f.controller.executing = None
            with patch.object(
                self.runtime,
                "has_remote_work",
                side_effect=AssertionError("no legacy graph vote"),
            ):
                self.assertTrue(
                    self.hooks._decode_graph(lambda *a: True, runner, f.batch)
                )
                self.assertFalse(
                    self.hooks._decode_graph(lambda *a: False, runner, f.batch)
                )
                self.assertTrue(
                    self.hooks._prefill_graph(lambda *a: True, runner, f.batch)
                )

    def test_dsv41_forward_failure_retains_source_and_restores_native_model_state(self):
        f = self._dsv41_fixture()
        with patch.dict(sys.modules, f.modules):
            f.req.extend_end, f.req.extend_len = 33, 33
            jobs = f.controller.prepare(f.batch)
            f.controller.plan = jobs
            tail = f.runner.model.model.decoder_replay_tail

            def fail(*args):
                raise RuntimeError("kernel launch failed")

            with self.assertRaisesRegex(RuntimeError, "kernel launch failed"):
                f.controller.forward(fail, f.runner, f.batch)
            self.assertIs(f.runner.model.model.decoder_replay_tail, tail)
            self.assertIsNone(f.controller.executing)
            with self.assertRaisesRegex(RuntimeError, "retaining source KV"):
                self.hooks.pre_layerwise.defer_release(
                    f.req, lambda: self.fail("unsafe release")
                )
            self.assertFalse(f.req.kv.is_kv_released)
            self.assertFalse(jobs[0].finalized)

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
                from sglang_remote_attn.source_cache import CacheGuard

                cache_fixture = self._radix()
                req.cache_request_handle = cache_fixture.CacheRequestHandle("final", 0)
                req.kv.cache_protected_len = 0
                req.last_node = cache_fixture.cache.root_node
                req.lock = cache_fixture.cache.lock(req.last_node)
                controller.cache_guard = CacheGuard(cache_fixture.cache)
                req.session_id = "native-session"
                self.assertIsNone(controller.prepare(batch))
                self.assertEqual(controller.cache_guard.leases, {})
                req.session_id = None
                jobs = controller.prepare(batch)
                self.assertIs(jobs[0][0].cache_lease.req, req)
                self.assertEqual(len(controller.cache_guard.leases), 1)
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

    def _preflight_checkout(self, root):
        api = self.interfaces
        targets = (
            list(self.hooks.HOOKS)
            + list(api.CORE_CALLS)
            + list(api.CORE_FIELDS)
            + list(api.CORE_PROPERTIES)
            + list(api.CORE_STATE)
            + ["sglang.srt.environ.Envs"]
        )
        paths = set()
        for target in targets:
            resolver = api.SourceAPI(ROOT)
            resolver.resolve(target)
            paths.update(resolver.cache)
        for path in paths:
            destination = root / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(path.read_text())

    def test_prelayer_preflight_rejects_incomplete_core_keywords(self):
        """A valid hook target must not hide missing progressive call arguments."""
        cases = (
            (
                "sglang.srt.managers.utils.GenerationBatchResult.copy_to_cpu",
                "return_hidden_states",
            ),
            (
                "sglang.srt.managers.scheduler.Scheduler.is_disable_overlap_for_batch",
                "last_batch",
            ),
            ("sglang.srt.mem_cache.radix_cache.RadixCache.checkpoint", "up_to"),
            (
                "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.checkpoint",
                "up_to",
            ),
            ("sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.finish", "handle"),
            ("decode.DecodeTransferQueue._commit_transfer_to_req", "keep_receiver"),
            (
                "mooncake.conn.MooncakeKVManager.add_transfer_request",
                "kv_buffer_indices",
            ),
            ("mooncake.conn.MooncakeKVManager.add_transfer_request", "on_complete"),
            ("mooncake.conn.MooncakeKVManager.add_transfer_request", "transfer_fn"),
            (
                "mooncake.conn.MooncakeKVManager.add_transfer_request",
                "staging_chunk_idx",
            ),
            (
                "mooncake.conn.MooncakeKVManager.add_transfer_request",
                "staging_layer_ids",
            ),
            ("mooncake.conn.MooncakeKVManager._transfer_data", "stage"),
            ("common.staging_handler.init_staging_allocator", "pin_memory"),
            (
                "common.staging_handler.DecodeStagingHandler._scatter_region",
                "layer_ids",
            ),
            (
                "common.staging_handler.DecodeStagingHandler._scatter_region",
                "staging_bytes",
            ),
            ("common.staging_handler.PrefillStagingStrategy.check_ready", "chunk_idx"),
            (
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph.eager_on_graph",
                "capture_stub",
            ),
            (
                "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend.BreakableCudaGraphBackend.capture_one",
                "post_warmup_hook",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._preflight_checkout(root)
            for relative, keyword in cases:
                with self.subTest(interface=relative, keyword=keyword):
                    resolver = self.interfaces.SourceAPI(root)
                    node = resolver.resolve(
                        relative
                        if relative.startswith("sglang.")
                        else "sglang.srt.disaggregation." + relative
                    )
                    path, tree = next(iter(resolver.cache.items()))
                    original = path.read_text()
                    for arg in node.args.args + node.args.kwonlyargs:
                        if arg.arg == keyword:
                            arg.arg = "removed_" + keyword
                    path.write_text(ast.unparse(tree))
                    try:
                        with self.assertRaisesRegex(RuntimeError, keyword):
                            self.check.check_sources(root)
                    finally:
                        path.write_text(original)

    def test_prelayer_preflight_rejects_missing_task_state_and_environment(self):
        """Mixed core files must fail before a worker accesses absent fields."""
        cases = (
            ("sglang.srt.managers.utils.GenerationBatchResult", "delay_sample_func"),
            ("sglang.srt.managers.scheduler.Scheduler.run_batch", "copy_done"),
            (
                "sglang.srt.managers.async_d2h_copy_worker.HostCopyDone.__init__",
                "error",
            ),
            ("sglang.srt.disaggregation.common.utils.TransferKVChunk", "on_complete"),
            (
                "sglang.srt.disaggregation.common.utils.TransferKVChunk",
                "staging_layer_ids",
            ),
            (
                "sglang.srt.disaggregation.common.conn.CommonKVReceiver.send_metadata",
                "chunk_staging_layers",
            ),
            (
                "sglang.srt.disaggregation.common.staging_handler.DecodeStagingHandler.register_decode_req",
                "_staging_completed_layers",
            ),
            (
                "sglang.srt.disaggregation.common.staging_buffer.StagingBuffer.__init__",
                "gpu_buffer",
            ),
            ("sglang.srt.environ.Envs", "SGLANG_MOONCAKE_HOST_STAGING_ALL"),
            (
                "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.__init__",
                "_pool",
            ),
            (
                "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend.BreakableCudaGraphBackend.__init__",
                "_pool",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._preflight_checkout(root)
            for target, field in cases:
                with self.subTest(interface=target, field=field):
                    resolver = self.interfaces.SourceAPI(root)
                    node = resolver.resolve(target)
                    path, tree = next(iter(resolver.cache.items()))
                    original = path.read_text()
                    for child in ast.walk(node):
                        if isinstance(child, ast.Name) and child.id == field:
                            child.id = "removed_" + field
                        elif isinstance(child, ast.Attribute) and child.attr == field:
                            child.attr = "removed_" + field
                    path.write_text(ast.unparse(tree))
                    try:
                        with self.assertRaisesRegex(RuntimeError, field):
                            self.check.check_sources(root)
                    finally:
                        path.write_text(original)

    def test_prelayer_preflight_rejects_missing_staging_module(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._preflight_checkout(root)
            module = (
                root
                / "python/sglang/srt/disaggregation/mooncake/pinned_staging_core.py"
            )
            module.unlink()
            with self.assertRaisesRegex(RuntimeError, "pinned_staging_core"):
                self.check.check_sources(root)

    def test_prelayer_runtime_preflight_requires_explicit_retention_contract(self):
        """Installed checks must reject **kwargs and changed native defaults."""

        def discards_flag(self, request, **kwargs):
            pass

        def changes_native_default(self, request, *, keep_receiver=True):
            pass

        def positional_only(self, request, keep_receiver=False, /, **kwargs):
            pass

        for function in (discards_flag, changes_native_default, positional_only):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(RuntimeError, "keep_receiver"):
                    self.check.check_runtime_interfaces(resolve=lambda target: function)

    def test_prelayer_runtime_preflight_checks_loaded_fields_and_state(self):
        """Checking a good checkout must not mask an incompatible loaded module."""
        from sglang.srt.environ import envs

        api = self.interfaces
        source = api.SourceAPI(ROOT)
        loaded = {}
        for target in dict.fromkeys(
            [*api.CORE_CALLS, *api.CORE_STATE, *api.CORE_PROPERTIES]
        ):
            resolver = api.SourceAPI(ROOT)
            node = copy.deepcopy(resolver.resolve(target))
            path = next(iter(resolver.cache))
            node.decorator_list = []
            node.body = [ast.Pass()]
            # Keep call signatures without evaluating imports or constructing owners.
            for defaults in (node.args.defaults, node.args.kw_defaults):
                for index, value in enumerate(defaults):
                    if value is not None:
                        try:
                            ast.literal_eval(value)
                        except (ValueError, TypeError):
                            defaults[index] = ast.Constant(None)
            module = ast.Module(
                body=[
                    ast.ImportFrom(
                        module="__future__",
                        names=[ast.alias(name="annotations")],
                        level=0,
                    ),
                    node,
                ],
                type_ignores=[],
            )
            namespace = {}
            exec(
                compile(ast.fix_missing_locations(module), str(path), "exec"), namespace
            )
            loaded[target] = (
                property(namespace[node.name])
                if target in api.CORE_PROPERTIES
                else namespace[node.name]
            )
        for target in api.CORE_FIELDS:
            loaded[target] = dataclasses.make_dataclass(
                "LoadedTask",
                [
                    (name, object, None)
                    for name in api.declared_fields(source.resolve(target))
                ],
            )
        loaded["sglang.srt.environ.envs"] = envs
        self.check.check_runtime_interfaces(resolve=loaded.__getitem__)

        task = "sglang.srt.disaggregation.common.utils.TransferKVChunk"
        original = loaded[task]
        loaded[task] = dataclasses.make_dataclass(
            "OldTask",
            [
                (field.name, object, None)
                for field in dataclasses.fields(original)
                if field.name != "on_complete"
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "on_complete"):
            self.check.check_runtime_interfaces(resolve=loaded.__getitem__)
        loaded[task] = original

        def discards_ready_event(self, session, blocks, **ready_event):
            pass

        target = "sglang.srt.disaggregation.pinned_staging_buffer.PinnedKVStager.send"
        original = loaded[target]
        loaded[target] = discards_ready_event
        with self.assertRaisesRegex(RuntimeError, "ready_event"):
            self.check.check_runtime_interfaces(resolve=loaded.__getitem__)
        loaded[target] = original

        def missing_scatter_state(self, room, decode_req):
            pass

        target = "sglang.srt.disaggregation.common.staging_handler.DecodeStagingHandler.register_decode_req"
        loaded[target] = missing_scatter_state
        with self.assertRaisesRegex(RuntimeError, "_staging_layer_events"):
            self.check.check_runtime_interfaces(resolve=loaded.__getitem__)

    def test_dsv41_progressive_native_state_barrier_survives_reordered_index_groups(
        self,
    ):
        """Neither ordinal arrival nor finalization may resend D-owned state."""
        import numpy as np
        from sglang_remote_attn import dsv41_progressive as pg
        from sglang_remote_attn.dsv41_layerwise import DSV41Source

        f = self._dsv41_fixture()
        pending, states, aux = [], [], []
        schema = dict(entries=[["kv", 2, 2], ["index", 2, 2], ["index", 2, 2]])
        source = DSV41Source(
            f.controller,
            f.req,
            f.controller.config.directory / "ordered",
            "ordered",
            np.array([2]),
            slice(4, 5),
            ((f.manager.transfer_infos[7]["peer:1"], f.registration),),
            33,
            schema=schema,
            groups=pg.groups(schema, 1),
            final_event=object(),
        )
        f.controller.dsv41_remote = True
        f.manager._dsv41_progressive_sources = {7: source}
        f.registration.dst_aux_ptrs = []
        f.manager.maybe_send_extra = lambda peer, rows, ex, reg: (
            states.append(rows) or 0
        )
        f.manager.send_aux = lambda *a: aux.append(a) or 0

        def native(manager, room, indices, span, last, **kw):
            pending.append((indices, span, last, kw))

        f.manager.add_transfer_request = functools.partial(
            pg.canonical_track, native, f.manager
        )
        source.req.disagg_kv_sender.trace_ctx = SimpleNamespace(
            copy_for_thread=lambda: None
        )
        f.manager.add_transfer_request(
            7,
            np.array([], dtype=np.int32),
            slice(5, 5),
            True,
            aux_index=4,
            state_indices=["swa", "carry"],
        )
        self.assertEqual(len(pending), 2)
        pending[1][3]["on_complete"]()
        self.assertFalse(source.barrier_ready)
        self.assertFalse(states)
        with patch.dict(sys.modules, f.modules):
            pending[0][3]["on_complete"]()
        self.assertTrue(source.barrier_ready)
        self.assertEqual(states, [["swa", "carry"]])
        self.assertEqual(len(aux), 1)
        self.assertTrue(pending[-1][2])
        self.assertEqual(pending[-1][3]["state_indices"], [])
        self.assertEqual(pending[-2][3]["kv_buffer_indices"], (0,))
        self.assertIs(pending[-2][3]["wait_event"], source.final_event)
        self.assertEqual(
            pg.send_aux(
                lambda *a: self.fail("second aux DMA"),
                f.manager,
                f.manager.transfer_infos[7]["peer:1"],
                4,
                [],
            ),
            0,
        )
        # A duplicate native final is an error rather than another state write.
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            f.manager.add_transfer_request(
                7, np.array([]), slice(5, 5), True, aux_index=4
            )

    def test_dsv41_progressive_barrier_needs_index_scatter_state_metadata_and_attempt(
        self,
    ):
        """A P TE marker cannot replace D index/scales GPU completion."""
        from sglang_remote_attn import dsv41_progressive as pg

        f = self._dsv41_fixture()
        req = SimpleNamespace(
            origin_input_ids=list(range(33)), bootstrap_room=7, finished=lambda: False
        )
        dr = SimpleNamespace(
            req=req,
            metadata_buffer_index=0,
            _staging_completed_layers=[],
            kv_receiver=SimpleNamespace(kv_mgr=f.manager, poll=lambda: 3),
        )
        queue = SimpleNamespace(
            staging_handler=SimpleNamespace(
                advance_scatter=lambda dr: None, is_failed=lambda dr: False
            ),
            metadata_buffers=SimpleNamespace(bootstrap_room=torch.tensor([[0]])),
        )
        owner = SimpleNamespace(
            _schema=lambda m: dict(entries=[["kv"], ["index"], ["index"]])
        )
        offer = SimpleNamespace(
            path=f.controller.config.directory / "barrier",
            generation="g",
            decode_req=dr,
            queue=queue,
            owner=owner,
        )
        write = self.hooks.pre_layerwise.write_json
        write(offer.path / "sparse-ready.json", dict(generation="g", prefix_len=33))
        write(
            offer.path / "remote-service.json",
            dict(generation="g", prefix_len=33, route={}),
        )
        modules = {
            "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                KVPoll=SimpleNamespace(Failed=0)
            )
        }
        with patch.dict(sys.modules, modules):
            self.assertIsNone(pg.barrier_ready(offer))
            dr._staging_completed_layers = [((1,), 0, 5)]
            queue.metadata_buffers.bootstrap_room[0, 0] = 7
            self.assertIsNone(pg.barrier_ready(offer))
            dr._staging_completed_layers.append(((2,), 0, 5))
            self.assertIsNotNone(pg.barrier_ready(offer))  # Global entry 0 absent.
            dr.kv_receiver.poll = lambda: 0
            self.assertIsNone(pg.barrier_ready(offer))
            dr.kv_receiver.poll = lambda: 3
            write(
                offer.path / "sparse-ready.json", dict(generation="old", prefix_len=33)
            )
            self.assertIsNone(pg.barrier_ready(offer))

    def test_dsv41_progressive_scatter_preserves_new_rows_scales_and_page_padding(self):
        """Odd c2 carry and D new c1 rows must survive final-prefix DMA scatter."""
        from sglang_remote_attn.dsv41_receive import scatter_prefix

        support = self._dsv41_remote_support()
        f = support.fixture(ROOT, "cpu")
        for layout in (
            f.layout.V41,
            f.layout.V41_NEW,
            f.layout.V41_FP4,
            f.layout.V41_FP4_NEW,
        ):
            for ratio in (1, 2):
                for prefix in (32, 33, 34, 39):
                    cpp = 8 // ratio
                    src = support.encoded(layout.encoding, 3, cpp, "cpu")
                    if layout.is_interleaved:
                        src = f.oracle.interleave_pages(
                            src, cpp, layout.data_bytes, layout.scale_bytes
                        )
                    dst = torch.full((8, src.shape[1]), 199, dtype=torch.uint8)
                    before = dst.clone()
                    ids = torch.tensor([6, 2, 4])
                    scatter_prefix(dst, src, ids, 2, prefix, 8, ratio, layout)
                    # Independent per-byte mask from native address geometry.
                    mask = torch.zeros_like(dst, dtype=torch.bool)
                    ds, base, ss = layout.row_addressing(cpp)
                    for n, p in enumerate(ids.tolist()):
                        if (2 + n + 1) * 8 <= prefix:
                            mask[p] = True
                        for row in range(
                            min(cpp, max(0, prefix // ratio - (2 + n) * cpp))
                        ):
                            for col, size in (
                                (row * ds, layout.data_bytes),
                                (base + row * ss, layout.scale_bytes),
                            ):
                                mask[p, col : col + size] = True
                                self.assertTrue(
                                    torch.equal(
                                        dst[p, col : col + size],
                                        src[n, col : col + size],
                                    )
                                )
                    self.assertTrue(torch.equal(dst[~mask], before[~mask]))

    def test_dsv41_progressive_resource_switch_waits_ordered_vote_and_final_gpu_proof(
        self,
    ):
        """Rank-local scatter cannot switch; Local idle adds no retirement votes."""
        from sglang_remote_attn import dsv41_progressive as pg
        from sglang_remote_attn.pre_layerwise import Controller
        from sglang_remote_attn.retirement import Retirement

        with tempfile.TemporaryDirectory() as directory:
            owner = Controller.__new__(Controller)
            owner.rank, owner.tp_size = 0, 1
            owner.scheduler = SimpleNamespace()
            owner.config = SimpleNamespace(drain_timeout=0.05)
            votes = []
            owner.agree = lambda flags: votes.append(list(flags)) or flags
            dr = SimpleNamespace(
                req=SimpleNamespace(rid="r", bootstrap_room=7),
                _staging_completed_layers=[],
            )
            parent = SimpleNamespace(
                closed=False,
                decode_req=dr,
                generation="g",
                rpc=SimpleNamespace(local_entries=set()),
                path=Path(directory),
                queue=SimpleNamespace(
                    staging_handler=SimpleNamespace(
                        advance_scatter=lambda dr: None, is_failed=lambda dr: False
                    )
                ),
            )
            resources = [pg.Resource(parent, i, i + 1) for i in (0, 1)]
            owner.targets = list(resources)
            protocol = Retirement(owner)
            self.addCleanup(protocol.close)
            for resource in resources:
                protocol.register(resource)
            for _ in range(32):
                protocol.observe()
            self.assertEqual(votes, [])
            dr._staging_completed_layers = [((1,), 0, 5)]
            protocol.observe()
            self.assertFalse(parent.rpc.local_entries)
            command = protocol.poll_command()
            protocol.arm(command)
            protocol.finalize()
            self.assertEqual(parent.rpc.local_entries, {1})
            self.assertEqual(len(votes), 1)
            for _ in range(32):
                protocol.observe()
            self.assertEqual(len(votes), 1)
            dr._staging_completed_layers.append(((0,), 0, 5))
            protocol.observe()
            protocol.arm(protocol.poll_command())
            protocol.finalize()
            self.assertEqual(parent.rpc.local_entries, {0, 1})
            self.assertEqual(owner.targets, [])
            for _ in range(32):
                protocol.observe()
            self.assertEqual(len(votes), 2)

    def test_dsv41_progressive_eager_survives_multiple_tokens_and_mixed_unready_global(
        self,
    ):
        """No whole-batch native attention may read a still-missing global page."""
        from sglang_remote_attn import dsv41_remote as remote
        from sglang_remote_attn import dsv41_sparse_math as math

        f = self._dsv41_remote_support().fixture(ROOT, "cpu")
        schema = dict(entries=[["kv", 1, 1], ["kv", 2, 2]])
        f.controller._schema = lambda m: schema
        f.controller.scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            kv_manager=None
        )
        target = remote.Target(
            f.controller, f.req, Path("/tmp/unused"), "g", {}, 33, progressive=True
        )
        f.controller.remote_targets[0] = target

        def call(q, positions, layer, ratio, scale):
            target.calls += 1
            cpp = 8 // ratio
            ids = torch.tensor([f.p_pages[p // cpp] * cpp + p % cpp for p in positions])
            keys = math.read_slots(
                f.p_pool.get_extra_key_buffer(layer), ids, cpp, f.layout.V41_FP4
            )
            return math.partial(q, keys, scale)

        target.call = call
        # Keep the new odd carry row, poison all completed prefix rows on D.
        expected = f.d_pool.get_extra_key_buffer(3).clone()
        for p in f.d_pages[:4]:
            f.d_pool.get_extra_key_buffer(3)[p].fill_(0)
        kv = torch.zeros(1, 512)

        def native(*a, **kw):
            self.fail("native full-batch read of unready global")

        def decode(runner, fb):
            return remote.attention(
                native,
                f.backend,
                f.query,
                kv,
                kv,
                SimpleNamespace(layer_id=3),
                fb,
                2,
                attn_sink=f.sink,
            )

        a = remote.forward(f.controller, decode, None, f.fb)
        b = remote.forward(f.controller, decode, None, f.fb)
        self.assertFalse(target.closed)
        self.assertEqual(target.calls, 2)
        torch.testing.assert_close(a, b)
        # Per-resource Local is producer-based: consumer 3 shares producer 2.
        target.local_entries.add(1)
        f.d_pool.get_extra_key_buffer(3).copy_(expected)
        c = remote.forward(f.controller, decode, None, f.fb)
        torch.testing.assert_close(a, c, atol=0.015625, rtol=0.01)
        self.assertEqual(target.calls, 2)
        remote.release(f.req)  # No early detach/free while transport is live.
        self.assertFalse(target.closed)

    def test_dsv41_progressive_early_commit_retains_metadata_and_free_until_gpu_tp_retirement(
        self,
    ):
        """Decode can run before global Success; cancellation cannot free that row."""
        from sglang_remote_attn import dsv41_progressive as pg
        from sglang_remote_attn.decode_overlap import BatchProof, DecodeOverlap
        from sglang_remote_attn.pre_layerwise import Controller, write_json

        with tempfile.TemporaryDirectory() as directory:
            owner = Controller.__new__(Controller)
            owner.rank, owner.tp_size, owner.graph_fast_path = 0, 1, True
            owner.dsv41_remote = True
            owner.scheduler = SimpleNamespace(enable_overlap=True)
            owner.config = SimpleNamespace(drain_timeout=0.02)
            owner.targets, owner.remote_targets, owner.early_admissions = [], {}, 0
            owner.agree = lambda flags: flags
            owner._schema = lambda manager: dict(
                entries=[["kv", 1, 1], ["index", 2, 2], ["kv", 2, 2]]
            )
            overlap = DecodeOverlap(owner)
            owner.decode_overlap = overlap
            self.addCleanup(overlap.retirement.close)
            finished = []
            status = [3]
            receiver = SimpleNamespace(
                kv_mgr=None,
                poll=lambda: status[0],
                clear=lambda: finished.append("receiver"),
            )
            req = SimpleNamespace(
                rid="early",
                bootstrap_room=7,
                origin_input_ids=list(range(33)),
                kv=SimpleNamespace(req_pool_idx=0),
                finished=lambda: False,
            )
            dr = SimpleNamespace(
                req=req,
                kv_receiver=receiver,
                metadata_buffer_index=0,
                _staging_completed_layers=[((1,), 0, 5)],
            )
            done = [False]
            queue = SimpleNamespace(
                queue=[dr],
                staging_handler=SimpleNamespace(
                    advance_scatter=lambda dr: None,
                    is_failed=lambda dr: False,
                    is_done=lambda dr: done[0],
                    unregister_decode_req=lambda room: finished.append("ring"),
                ),
                metadata_buffers=SimpleNamespace(bootstrap_room=torch.tensor([[7]])),
                req_to_metadata_buffer_idx_allocator=SimpleNamespace(
                    free=lambda idx: finished.append("metadata")
                ),
            )
            commits = []
            queue._commit_transfer_to_req = lambda dr, **kw: commits.append(kw)
            path = Path(directory)
            offer = SimpleNamespace(
                owner=owner,
                queue=queue,
                decode_req=dr,
                path=path,
                generation="g",
                coverage=SimpleNamespace(prefix_len=33),
            )
            owner.offer = lambda queue, dr: offer
            write_json(
                path / "remote-service.json",
                dict(generation="g", prefix_len=33, route={}),
            )
            write_json(path / "sparse-ready.json", dict(generation="g", prefix_len=33))
            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                        KVPoll=SimpleNamespace(Success=4, Failed=0)
                    )
                },
            ):
                admitted, _ = pg.early_targets(owner, queue, None)
                self.assertEqual(admitted, [req])
                self.assertEqual(commits, [dict(keep_receiver=True)])
                self.assertEqual(queue.queue, [])
                self.assertIs(dr.kv_receiver, receiver)
                self.assertTrue(overlap.serial())
                target = req._pre_layerwise_target
                proof = BatchProof(
                    result=SimpleNamespace(
                        delay_sample_func=None,
                        copy_done=SimpleNamespace(query=lambda: gpu[0]),
                    )
                )
                gpu = [False]
                target.forward_proofs.append(proof)
                target.forward_count = 1
                freed = []
                self.assertTrue(
                    self.hooks.pre_layerwise.defer_release(
                        req, lambda: freed.append("KV")
                    )
                )
                self.assertFalse(target.rpc.closed)
                self.assertEqual(freed, [])
                self.assertEqual(overlap.retirement.final_votes, 0)
                status[0] = 4
                done[0] = True
                dr._staging_completed_layers += [((0,), 0, 5), ((2,), 0, 5)]
                protocol = overlap.retirement
                protocol.observe()
                protocol.arm(protocol.poll_command())
                protocol.finalize()
                self.assertEqual(freed, [])
                self.assertIs(dr.kv_receiver, receiver)
                gpu[0] = True
                protocol.observe()
                protocol.arm(protocol.poll_command())
                runtime = SimpleNamespace(_drop_target_state_for=lambda req: None)
                with patch(
                    "sglang_remote_attn.pre_layerwise._runtime", return_value=runtime
                ):
                    protocol.finalize()
                self.assertEqual(freed, ["KV"])
                self.assertEqual(finished, ["ring", "receiver", "metadata"])
                self.assertIsNone(dr.kv_receiver)
                self.assertEqual(owner.targets, [])
                self.assertEqual(owner.remote_targets, {})
                self.assertTrue(
                    overlap.serial()
                )  # Consume final serial result boundary.
                self.assertFalse(overlap.serial())
                for _ in range(32):
                    protocol.observe()
                self.assertEqual(protocol.final_votes, 2)

    def test_dsv41_progressive_http_snapshot_needs_all_tp_barriers_without_kv_mutation(
        self,
    ):
        """Returning P early cannot release the native request or skip a shard."""
        from sglang_remote_attn import dsv41_progressive as pg
        from sglang_remote_attn.pre_layerwise import write_json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            emitted = []
            req = SimpleNamespace(
                bootstrap_room=7,
                return_logprob=False,
                finished_reason=None,
                disagg_kv_sender=SimpleNamespace(poll=lambda: 3),
            )
            source = SimpleNamespace(
                req=req,
                generation="g0",
                path=root / "7/0/g0",
                prefix_len=33,
                barrier_ready=True,
                responded=False,
            )
            controller = SimpleNamespace(
                rank=0,
                tp_size=2,
                sources=[source],
                config=SimpleNamespace(directory=root),
                scheduler=SimpleNamespace(
                    output_streamer=SimpleNamespace(
                        stream_output=lambda rows, *a: emitted.extend(rows)
                    )
                ),
            )
            for rank in (0, 1):
                write_json(
                    root / f"7/{rank}/offer.json",
                    dict(generation=f"g{rank}", mode="remote"),
                )
            write_json(
                root / "7/0/g0/sparse-ready.json", dict(generation="g0", prefix_len=33)
            )
            modules = {
                "sglang.srt.managers.schedule_batch": SimpleNamespace(
                    FINISH_LENGTH=lambda **kw: kw
                )
            }
            with patch.dict(sys.modules, modules):
                pg.poll_responses(controller)
                self.assertFalse(emitted)
                write_json(
                    root / "7/1/g1/sparse-ready.json",
                    dict(generation="old", prefix_len=33),
                )
                pg.poll_responses(controller)
                self.assertFalse(emitted)
                write_json(
                    root / "7/1/g1/sparse-ready.json",
                    dict(generation="g1", prefix_len=33),
                )
                pg.poll_responses(controller)
                pg.poll_responses(controller)
            self.assertEqual(len(emitted), 1)
            self.assertIsNot(emitted[0], req)
            self.assertTrue(emitted[0]._pre_layerwise_response_snapshot)
            self.assertIsNone(req.finished_reason)
            self.assertEqual(req.disagg_kv_sender.poll(), 3)
            self.assertTrue(source.responded)

    def test_dsv41_progressive_state_transfer_failure_poison_retains_and_does_not_admit(
        self,
    ):
        """An uncertain sideband DMA cannot publish readiness or kill shared IO."""
        import numpy as np
        from sglang_remote_attn import dsv41_progressive as pg
        from sglang_remote_attn.dsv41_layerwise import DSV41Source

        f = self._dsv41_fixture()
        poison, failed = [], []
        f.manager._poison_unsettled_send = poison.append
        f.manager.conclude_failure = lambda **kw: failed.append(kw)
        source = DSV41Source(
            f.controller,
            f.req,
            f.controller.config.directory / "failed",
            "failed",
            np.array([]),
            slice(0, 0),
            (),
            33,
            final_payload=([], slice(0, 0), {}),
        )
        with patch.object(
            pg, "send_states", side_effect=RuntimeError("native DMA timeout")
        ):
            source._sideband()
        self.assertEqual(poison, [7])
        self.assertEqual(failed[0]["bootstrap_room"], 7)
        self.assertFalse(source.barrier_ready)
        self.assertFalse((source.path / "sparse-ready.json").exists())
        self.assertTrue((source.path / "error.json").exists())

    def test_dsv41_progressive_direct_reset_fences_before_attempt_mutation(self):
        """Native direct reset must not change row/attempt before the TP drain."""
        events = []
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=8))
        target = SimpleNamespace(owner=SimpleNamespace(dsv41_remote=True))
        req._pre_layerwise_target = target

        def drain(*, required):
            self.assertIs(required, target)
            self.assertEqual(req.kv.req_pool_idx, 8)
            events.append("gpu-io-tp-drained")
            req._pre_layerwise_target = None

        target.owner.drain_targets = drain

        def reset(req):
            self.assertEqual(events, ["gpu-io-tp-drained"])
            events.append("native-reset")
            req.kv.req_pool_idx = 9

        self.hooks._reset_retracted(reset, req)
        self.assertEqual(events, ["gpu-io-tp-drained", "native-reset"])

    def test_dsv41_progressive_oversized_entry_falls_back_before_reserving_native_pages(
        self,
    ):
        """A bounded receive ring must never force unsafe direct final-page DMA."""
        f = self._dsv41_receive_fixture()
        f.controller.dsv41_remote = True
        f.controller.config = dataclasses.replace(f.controller.config, mode="remote")
        path = f.controller.config.directory / "7/0/offer.json"
        offer = json.loads(path.read_text())
        offer.update(
            mode="remote",
            remote_boundary="progressive-sparse",
            recv_host_staging=False,
            progressive_ring_bytes=1,
        )
        self.hooks.pre_layerwise.write_json(path, offer)
        before = (f.req.start_send_idx, f.sender.curr_idx, f.tasks.qsize())
        with patch.dict(sys.modules, f.modules):
            self.assertEqual(f.controller.candidates(f.batch), [None])
            self.assertIsNone(f.controller.prepare(f.batch))
        self.assertEqual(
            (f.req.start_send_idx, f.sender.curr_idx, f.tasks.qsize()), before
        )
        self.assertEqual(f.requests, [])

    def test_dsv41_progressive_normal_scheduler_records_actual_gpu_stream_completion(
        self,
    ):
        """Normal mode has no overlap copy_done; never query an unrecorded event."""
        with tempfile.TemporaryDirectory() as directory:
            controller, target, _ = self._overlap_target(directory)
            scheduler = controller.scheduler
            scheduler.enable_overlap = False
            overlap = controller.decode_overlap
            batch = SimpleNamespace(
                reqs=[target.decode_req.req],
                forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
            )
            recorded = []
            stream = object()
            complete = [False]
            fence = SimpleNamespace(query=lambda: complete[0], record=recorded.append)
            result = SimpleNamespace(delay_sample_func=None, copy_done=None)
            with (
                patch.object(torch.cuda, "Event", return_value=fence),
                patch.object(torch.cuda, "current_stream", return_value=stream),
            ):
                self.assertIs(
                    overlap.run_batch(lambda *a: result, scheduler, batch), result
                )
            self.assertEqual(recorded, [stream])
            self.assertFalse(target.gpu_ready())
            complete[0] = True
            self.assertTrue(target.gpu_ready())

    def _dsv41_graph_fixture(self, variants=(None,)):
        from sglang_remote_attn import dsv41_graph as dg

        name = "dsv41_graph_probe_support"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "test/manual/remote_attn/dsv41_graph_probe_support.py"
        )
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        f = support.fixture()
        f.runner.attention_graph_variants = SimpleNamespace(
            capture_labels=variants, graph_limits=()
        )
        state = [None]

        def set_variant(value):
            state[0] = value

        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.model_executor.runner_utils.capture_mode": SimpleNamespace(
                        get_capture_attention_variant=lambda: state[0],
                        _set_capture_attention_variant=set_variant,
                    )
                },
            )
        )
        f.variant_state = state

        class Bank:
            def __init__(self):
                self.capturing = False
                self.keys = {}
                self.callbacks = {}
                self.known_keys = []

            def capture_session(self, stream):
                return nullcontext()

            def capture_one(self, key, fn, *args):
                self.capturing = True
                self.record = []
                try:
                    fn()
                    self.keys[key.size] = key
                    self.known_keys.append(key)
                    self.callbacks[key.size] = self.record
                finally:
                    self.capturing = False

            def can_run(self, fb, key):
                return key in self.known_keys

            def replay_session(self):
                return nullcontext()

            def replay(self, key, fb):
                for callback in self.callbacks[key.size]:
                    callback()
                return torch.ones(fb.batch_size, 4)

            def cleanup(self):
                self.keys.clear()

        full, remote = Bank(), Bank()

        def eager(fn):
            def wrapped(*a, **kw):
                result = fn(*a, **kw)
                if remote.capturing:
                    remote.record.append(lambda: fn(*a, **kw))
                return result

            return wrapped

        f.runner.device_module = SimpleNamespace(
            Event=lambda: SimpleNamespace(
                record=lambda stream: None, synchronize=lambda: None
            ),
            current_stream=lambda: None,
        )
        b = dg.DSV41GraphBackend(
            f.runner, full, remote, eager, lambda: remote.capturing, lambda: None
        )
        f.attach(b)
        with b.capture_session(None):
            for size in (8, 4, 2, 1):
                for variant in variants:
                    fb = f.batch(size, -100)
                    key = SimpleNamespace(
                        size=size,
                        stream_idx=None,
                        variant_label=None,
                        attention_variant=variant,
                    )
                    b.capture_one(
                        key,
                        lambda fb=fb: f.model(
                            torch.ones(fb.batch_size, 4), torch.zeros(fb.batch_size), fb
                        ),
                    )
        f.backend, f.full, f.remote = b, full, remote
        self.addCleanup(b.cleanup)
        return f

    def test_dsv41_segments_bind_live_metadata_prepare_and_attention_without_capture_rpc(
        self,
    ):
        """Captured dummy FB must not select replay sparse rows/carry or RPC route."""
        f = self._dsv41_graph_fixture()
        b = f.backend
        self.assertEqual(sorted(b.remote_keys), [1, 2, 4])
        self.assertEqual(sorted(f.full.keys), [1, 2, 4, 8])
        self.assertEqual(f.calls, [])
        self.assertEqual(f.prepared, [])
        self.assertIsNone(b.capture_batch)
        expected = (
            ("metadata", -1),
            ("state", 0),
            ("attention", 0),
            ("state", 1),
            ("attention", 1),
            ("state", 2),
            ("attention", 2),
        )
        self.assertEqual(b.manifest()[0], tuple((n, expected) for n in (1, 2, 4)))
        f.controller.targets = [object()]
        for n in (1, 2, 4):
            for marker in (3, 7):
                live = f.batch(n, marker)
                with b.execution(live), b.replay_session():
                    b.replay(b.remote_keys[n], live)
                self.assertEqual(f.calls[-3:], [(marker, i) for i in range(3)])
                self.assertEqual(f.prepared[-3:], [(marker, i) for i in range(3)])
                self.assertEqual(f.metadata[-1], marker)
        self.assertIsNone(b.active_batch)
        self.assertIsNone(b.capture_batch)

    def test_dsv41_segments_graph_policy_uses_shared_lease_and_never_pads_remote_rows(
        self,
    ):
        """Only TP-registered lifetimes enable Remote graph; Local uses native full."""
        f = self._dsv41_graph_fixture()
        f.controller.targets = [object()]
        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_size=1, tp_rank=0)
        )
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": context}):
            for size, expected in (
                (1, True),
                (2, True),
                (3, False),
                (4, True),
                (8, False),
            ):
                self.assertEqual(
                    self.hooks._decode_graph(lambda *a: True, f.runner, f.batch(size)),
                    expected,
                )
            # A compatibility/post-PD target has no shared progressive lease.
            f.controller.targets = []
            f.controller.remote_targets = {0: object()}
            self.assertFalse(
                self.hooks._decode_graph(lambda *a: True, f.runner, f.batch(1))
            )
            f.controller.remote_targets = {}
            for _ in range(32):
                self.assertTrue(
                    self.hooks._decode_graph(lambda *a: True, f.runner, f.batch(8))
                )

    def test_dsv41_segments_reject_mid_token_resource_mutation_and_wrong_bridge(self):
        """A resource state edge cannot leak into half of one segmented token."""
        f = self._dsv41_graph_fixture()
        b = f.backend
        f.controller.targets = [object()]
        target = SimpleNamespace(
            progressive=True, closed=False, check=lambda: None, local_entries=set()
        )
        f.controller.remote_targets[0] = target
        live = f.batch(1, 3)
        with b.execution(live):
            target.local_entries.add(1)
            with self.assertRaisesRegex(RuntimeError, "ownership changed"):
                b.replay(b.remote_keys[1], live)
        target.local_entries.clear()
        original = f.attn._dsv41_graph_originals["attention"]
        f.attn._dsv41_graph_originals["attention"] = lambda **kw: original(**kw)[:, :2]
        with self.assertRaisesRegex(RuntimeError, "graph bridge"):
            with b.execution(live):
                b.replay(b.remote_keys[1], live)
        self.assertIsNone(b.active_batch)

    def test_dsv41_segments_preparation_failure_retains_unjoined_gpu_storage(self):
        """A main-stream event cannot fence an unfinished native side stream."""
        f = self._dsv41_graph_fixture()
        b = f.backend
        f.controller.targets = [object()]

        def failed(**kw):
            raise RuntimeError("side-stream launch failed")

        f.layers[1].update_decode_state = failed
        with self.assertRaisesRegex(RuntimeError, "side-stream"):
            with b.execution(f.batch(1, 2)):
                b.replay(b.remote_keys[1], b.active_batch)
        self.assertTrue(b.unsafe)
        with self.assertRaisesRegex(RuntimeError, "completion is unproven"):
            b.cleanup()
        # Test-only CPU fake bank has no GPU storage; permit its cleanup fixture.
        b.unsafe = False

    def test_dsv41_segments_native_abi_and_boundary_completeness_fail_closed(self):
        """Missing prepare/metadata breaks cannot silently freeze native sparse state."""
        from sglang_remote_attn import dsv41_graph as dg

        f = self._dsv41_graph_fixture()
        self.assertTrue(dg.capture_signature(f.mr, f.runner)[0])
        f.backend.dynamic_trace.pop()
        with self.assertRaisesRegex(RuntimeError, "dynamic boundaries"):
            f.backend.validate_remote_capture(1)
        f.attn.trtllm_attn = True
        self.assertFalse(dg.capture_signature(f.mr, f.runner)[0])

    def test_dsv41_segments_native_candidate_variants_share_three_remote_captures(self):
        """Context-length variants must preserve native policy without multiplying Remote banks."""
        labels = (
            "candidate_all",
            "candidate_c2_all",
            "candidate_unfiltered",
            "candidate_filtered",
        )
        f = self._dsv41_graph_fixture(labels)
        b = f.backend
        self.assertEqual(len(f.full.known_keys), 16)
        self.assertEqual(len(f.remote.known_keys), 3)
        f.controller.targets = [object()]
        original = f.attn._dsv41_graph_originals["attention"]
        observed = []

        def attention(**kw):
            observed.append(f.variant_state[0])
            return original(**kw)

        f.attn._dsv41_graph_originals["attention"] = attention
        for label in labels:
            key = SimpleNamespace(
                size=2, stream_idx=None, variant_label=None, attention_variant=label
            )
            fb = f.batch(2, 3)
            self.assertTrue(b.can_run(fb, key))
            with b.execution(fb):
                b.replay(key, fb)
            self.assertEqual(observed[-3:], [label] * 3)
            self.assertIsNone(f.variant_state[0])
        f.controller.targets = []
        self.assertTrue(b.can_run(fb, key))

    def _dsv41_split_core(self, **options):
        spec = importlib.util.spec_from_file_location(
            "split_qkv_fixture",
            ROOT / "test/manual/remote_attn/dsv41_graph_probe_support.py",
        )
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        return support.fixture(**options)

    def test_dsv41_split_projection_state_matches_native_fusion_precision_and_aliases(
        self,
    ):
        """Projected-KV reuse cannot change precision mode or repeat the GEMM."""
        for fused in (False, True):
            for bf16 in (False, True):
                with self.subTest(fused=fused, bf16=bf16):
                    f = self._dsv41_split_core(
                        dtype=torch.bfloat16, fused=fused, bf16_source=bf16
                    )
                    layer = f.layers[1]
                    fb = f.batch(2, 3)
                    x = torch.arange(8).reshape(2, 4).to(torch.bfloat16) / 13
                    pos = torch.tensor([3, 7], dtype=torch.bfloat16)
                    f.attn.init_forward_metadata_in_graph(fb)
                    reference = layer._native_prepare(x, pos, fb, f.attn)[0].clone()
                    cache = f.cache[1].clone()
                    carry = f.carry[1].clone()
                    for buf in f.cache:
                        buf.fill_(-99)
                    for buf in f.carry:
                        buf.zero_()
                    f.projections.clear()
                    q, q_lora, kv = layer.project_decode_qkv(x, pos)
                    self.assertTrue(torch.equal(q, reference))
                    self.assertTrue(all(bool((b == -99).all()) for b in f.cache))
                    self.assertTrue(all(not bool(b.any()) for b in f.carry))
                    before = list(f.projections)
                    q_before = q.clone()
                    lora_before = q_lora.clone()
                    layer.update_decode_state(x, q_lora, kv, pos, fb, f.attn)
                    self.assertEqual(before, f.projections)
                    self.assertEqual(
                        [kind for kind, _ in before],
                        ["qkv", "qb"] if fused else ["qa", "qb", "kv"],
                    )
                    self.assertTrue(torch.equal(f.cache[1], cache))
                    self.assertTrue(torch.equal(f.carry[1], carry))
                    self.assertTrue(torch.equal(q, q_before))
                    self.assertTrue(torch.equal(q_lora, lora_before))

    def test_dsv41_split_qkv_keeps_quantized_wrappers_inside_projection_boundary(self):
        """Opaque quantized Q inputs must not escape to the eager state bridge."""
        f = self._dsv41_split_core()
        layer = f.layers[2]
        original = layer._compute_q_b
        layer._normalize_q_lora = lambda q: (q, SimpleNamespace(values=q, scale=2))
        layer._compute_q_b = lambda packed, pos, q_out=None: original(
            packed.values * packed.scale, pos, q_out
        )
        x = torch.ones(2, 4)
        pos = torch.tensor([1.0, 2.0])
        q, lora, kv = layer.project_decode_qkv(x, pos, x_quant=x * 3)
        self.assertTrue(torch.equal(q, x * 12 + pos[:, None]))
        self.assertTrue(torch.equal(lora, x * 3))
        self.assertTrue(torch.equal(kv, x * 9 + 0.125))
        self.assertTrue(all(torch.is_tensor(v) for v in (q, lora, kv)))

    def test_dsv41_native_prepare_router_preserves_all_original_dispatch_branches(self):
        """Extracted routing must retain native HIP/NPU/CUDA/low-ratio/serial calls."""
        f = self._dsv41_split_core()
        router = f.core.prepare_qkv
        values = []
        owner = SimpleNamespace()
        for tag, name in enumerate(
            (
                "_forward_prepare_multi_stream_hip",
                "_forward_prepare_multi_stream_npu",
                "_forward_prepare_multi_stream",
                "_forward_prepare_low_ratio_multi_stream",
                "_forward_prepare",
            )
        ):

            def native(*args, _tag=tag, **kw):
                values.append((_tag, args, kw))
                return (_tag, "kv") if _tag == 4 else _tag

            setattr(owner, name, native)
        for hip, npu, multi, low, expected in (
            (True, False, True, False, 0),
            (False, True, True, False, 1),
            (False, False, True, True, 2),
            (False, False, False, True, 3),
            (False, False, False, False, 4),
        ):
            router.__globals__.update(_is_hip=hip, _is_npu=npu)
            q, kv = router(
                owner,
                "x",
                "pos",
                "fb",
                "backend",
                q_out="q",
                x_quant="quant",
                q_rope_out="rope",
                k_nope_out="nope",
                k_rope_out="krope",
                enable_multi_stream=multi,
                low_ratio_multi_stream=low,
            )
            self.assertEqual(q, expected)
            self.assertEqual(kv, "kv" if expected == 4 else None)
            self.assertEqual(values[-1][1], ("x", "pos", "fb", "backend", "q"))
            self.assertEqual(values[-1][2]["x_quant"], "quant")
            if expected == 4:
                self.assertEqual(values[-1][2]["q_rope_out"], "rope")

    def test_dsv41_split_projection_rejects_unsupported_modes_before_state_mutation(
        self,
    ):
        """The narrow split contract must reject fused QK/mega/non-low-ratio use."""
        for flag, value in (
            ("use_fused_qk_norm_rope", True),
            ("use_mega_attention", True),
            ("compress_ratio", 4),
            ("is_dsv41", False),
        ):
            f = self._dsv41_split_core()
            setattr(f.layers[0], flag, value)
            with self.assertRaisesRegex(ValueError, "split decode QKV"):
                f.layers[0].project_decode_qkv(torch.ones(1, 4), torch.zeros(1))
            self.assertFalse(f.projections)
            self.assertTrue(all(bool((b == -99).all()) for b in f.cache))

    def _mesh_instances(self):
        from sglang_remote_attn.pd_mesh import Mesh

        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        meshes = {}
        for i, name in enumerate(("P0", "P1", "D0", "D1")):
            meshes[name] = Mesh(
                directory / name,
                name,
                uuid.uuid4().hex,
                0,
                "127.0.0.1",
                "127.0.0.1",
                ("127.0.0.1", 32000 + i),
            )
            self.addCleanup(meshes[name].close)
        return meshes

    def _mesh_wait(self, predicate):
        deadline = time.monotonic() + 4
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("PD journal propagation timed out")
            threading.Event().wait(0.005)

    def _mesh_link(self, p, d, room):
        from sglang_remote_attn.pd_mesh import JournalPath

        target = d.new_room(room, f"session-{d.instance}", p.source_endpoint)
        generation = uuid.uuid4().hex
        target.publish(
            "offer.json", dict(generation=generation, session=f"session-{d.instance}")
        )
        peer = SimpleNamespace(
            _remote_attn_journal=target.descriptor,
            room=room,
            mooncake_session_id=f"session-{d.instance}",
            is_dummy=False,
        )
        p.open_peer(peer)
        source = peer._remote_attn_room
        self._mesh_wait(lambda: (source.root / "offer.json").exists())
        return JournalPath(source) / generation, JournalPath(target) / generation, peer

    def test_pd_mesh_two_by_two_local_mirrors_isolate_requests_and_cancellation(self):
        """A single D must consume proofs from different P without cross-request release."""
        from sglang_remote_attn.pre_layerwise import read_json, write_json

        m = self._mesh_instances()
        links = [
            self._mesh_link(m[p], m[d], n)
            for p, d, n in (
                ("P0", "D0", 10),
                ("P1", "D0", 11),
                ("P0", "D1", 12),
                ("P1", "D1", 13),
            )
        ]
        for i, (p, d, _) in enumerate(links):
            self.assertNotEqual(p.local, d.local)
            write_json(p / "layer-0.json", dict(generation=p.name, layer=0, source=i))
        self._mesh_wait(lambda: all(read_json(d / "layer-0.json") for _, d, _ in links))
        self.assertEqual(
            [read_json(d / "layer-0.json")["source"] for _, d, _ in links],
            list(range(4)),
        )
        p, d, _ = links[0]
        write_json(
            d / "done.json", dict(generation=d.name, cancelled=True, gpu_proven=True)
        )
        self._mesh_wait(lambda: read_json(p / "done.json"))
        self.assertTrue(read_json(p / "done.json")["cancelled"])
        self.assertTrue(
            all(read_json(other / "done.json") is None for other, _, _ in links[1:])
        )
        p, d, _ = links[1]
        write_json(p / "resource-2.json", dict(generation=p.name, producer=2))
        self._mesh_wait(lambda: read_json(d / "resource-2.json"))

    def test_pd_mesh_disconnect_reconnect_replays_only_same_attempt_facts(self):
        """Network loss must retain proofs and never manufacture a release."""
        from sglang_remote_attn.pre_layerwise import read_json, write_json

        m = self._mesh_instances()
        p, d, _ = self._mesh_link(m["P0"], m["D0"], 7)
        import socket

        with p.room.condition:
            sockets = list(p.room.sockets)
        for sock in sockets:
            sock.shutdown(socket.SHUT_RDWR)
        write_json(p / "ready.json", dict(generation=p.name, prefix_len=33))
        self._mesh_wait(lambda: read_json(d / "ready.json"))
        self.assertIsNone(read_json(p / "done.json"))
        write_json(d / "done.json", dict(generation=d.name, gpu_proven=True))
        self._mesh_wait(lambda: read_json(p / "done.json"))
        self.assertTrue(read_json(p / "done.json")["gpu_proven"])

    def test_pd_mesh_capabilities_reject_forged_retirement_and_old_attempt_writes(self):
        """Neither a source nor a previous attempt can publish the new D's retirement."""
        from sglang_remote_attn.pre_layerwise import read_json, write_json

        m = self._mesh_instances()
        p, d, peer = self._mesh_link(m["P0"], m["D0"], 7)
        with self.assertRaises(ValueError):
            write_json(p / "done.json", dict(generation=p.name))
        with self.assertRaises(ValueError):
            write_json(p / "../done.json", dict(generation=p.name))
        p2, d2, _ = self._mesh_link(m["P0"], m["D0"], 7)
        self.assertNotEqual(p.local, p2.local)
        write_json(d / "done.json", dict(generation=d.name, old=True))
        self._mesh_wait(lambda: read_json(p / "done.json"))
        self.assertIsNone(read_json(p2 / "done.json"))
        bad = copy.copy(peer)
        bad._remote_attn_journal = {**peer._remote_attn_journal, "session": "wrong"}
        del bad._remote_attn_room
        m["P1"].open_peer(bad)
        self.assertFalse(hasattr(bad, "_remote_attn_room"))
        self.assertNotEqual(d.room.descriptor["token"], d2.room.descriptor["token"])
        from sglang_remote_attn.prefill_admission import fingerprint

        baseline = fingerprint(peer)
        changed = copy.copy(peer)
        changed._remote_attn_room = object()
        self.assertEqual(fingerprint(changed), baseline)
        changed._remote_attn_journal = {
            **peer._remote_attn_journal,
            "token": "new-attempt",
        }
        self.assertNotEqual(fingerprint(changed), baseline)

    def test_pd_mesh_native_bootstrap_extension_preserves_wire_and_rejects_invalid_cap(
        self,
    ):
        """Native request destinations remain authoritative; malformed extras fall back."""
        from sglang_remote_attn import pd_mesh

        m = self._mesh_instances()
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "mesh_native_defs",
            ROOT / "test/manual/remote_attn/dsv41_remote_probe_support.py",
        )
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        native = support.definitions(
            ROOT,
            "python/sglang/srt/disaggregation/mooncake/conn.py",
            owner="TransferInfo",
            members={"from_zmq"},
        )
        parse = native.from_zmq.__func__
        parse.__globals__.update(np=np, unpack_int_lists=lambda raw, fmt: [])
        frames = [
            b"7",
            b"127.0.0.1",
            b"32002",
            b"session-D0",
            np.array([3, 4], dtype=np.int32).tobytes(),
            b"0",
            b"",
            b"1",
            b"33",
            b"",
        ]
        receiver = SimpleNamespace(
            bootstrap_room=7,
            kv_mgr=SimpleNamespace(
                _remote_attn_mesh=m["D0"], get_session_id=lambda: "session-D0"
            ),
        )
        sent = []
        info = dict(rank_ip="127.0.0.1", rank_port=32000, is_dummy=False)
        pd_mesh.bootstrap(
            lambda receiver, **kw: sent.append(kw["message"]) or True,
            receiver,
            bootstrap_info=info,
            message=frames,
            operation="send_metadata",
        )
        self.assertEqual(sent[0][:10], frames)
        peer = pd_mesh.parse_transfer(lambda msg: parse(SimpleNamespace, msg), sent[0])
        self.assertEqual(peer.dst_kv_indices.tolist(), [3, 4])
        pd_mesh.resolve_peers(
            lambda *a: None,
            SimpleNamespace(_remote_attn_mesh=m["P0"]),
            {"session-D0": peer},
        )
        self.assertTrue(hasattr(peer, "_remote_attn_room"))
        malformed = pd_mesh.parse_transfer(
            lambda msg: parse(SimpleNamespace, msg),
            [*frames, pd_mesh.MAGIC + b'{"instance":"../bad"}'],
        )
        self.assertFalse(hasattr(malformed, "_remote_attn_journal"))

    def test_pd_mesh_startup_lock_rejects_duplicate_instance_and_restart_uses_new_epoch(
        self,
    ):
        """Two live instances must not share control commands or a restart's proofs."""
        from sglang_remote_attn import pd_mesh
        from sglang_remote_attn.pre_layerwise import Config

        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

        def controller():
            return SimpleNamespace(
                rank=0,
                tp_size=1,
                config=Config("remote", directory / "journal"),
                scheduler=SimpleNamespace(
                    disaggregation_mode=SimpleNamespace(value="prefill"),
                    disagg_prefill_bootstrap_queue=SimpleNamespace(
                        kv_manager=SimpleNamespace(
                            local_ip="127.0.0.1", rank_port=32000
                        )
                    ),
                ),
            )

        with patch.dict(
            os.environ,
            {
                "HANDOFF_PD_NETWORK": "1",
                "HANDOFF_INSTANCE_ID": "P0",
                "HANDOFF_CTL_DIR": str(directory / "control"),
            },
        ):
            a, b = controller(), controller()
            pd_mesh.install(a)
            self.addCleanup(a.mesh.close)
            with self.assertRaisesRegex(RuntimeError, "already serving"):
                pd_mesh.install(b)
            old = a.config.directory
            a.mesh.close()
            pd_mesh.install(b)
            self.addCleanup(b.mesh.close)
            self.assertNotEqual(old, b.config.directory)
            self.assertTrue((directory / "control/P0/instance.lock").exists())

    def _gpu_transport_fixture(self):
        """Real control sockets and tensor copies; only TE/CUDA boundaries are CPU."""
        from sglang_remote_attn import dsv41_remote
        from sglang_remote_attn import handoff_control as rt
        from sglang_remote_attn import remote_data_plane as dp

        self.enterContext(patch.dict(os.environ, {"HANDOFF_REMOTE_GPU_TRANSFER": "1"}))

        class Engine:
            peers = {}

            def __init__(self, name):
                self.name, self.regions, self.calls, self.fail = name, {}, [], False
                self.peers[name] = self

            def get_session_id(self):
                return self.name

            def batch_register(self, pointers, sizes):
                self.regions.update(zip(pointers, sizes))
                return 0

            def batch_deregister(self, pointers):
                for p in pointers:
                    self.regions.pop(p)
                return 0

            def batch_transfer_sync(self, session, sources, targets, sizes):
                self.calls.append(list(zip(sources, targets, sizes)))
                if self.fail:
                    return -1
                peer = self.peers[session]
                for a, b, n in zip(sources, targets, sizes):
                    assert any(
                        p <= a and a + n <= p + s for p, s in self.regions.items()
                    )
                    assert any(
                        p <= b and b + n <= p + s for p, s in peer.regions.items()
                    )
                    ctypes.memmove(b, a, n)
                return 0

        p, d = Engine("P"), Engine("D")
        pm, dm = SimpleNamespace(engine=p), SimpleNamespace(engine=d)
        pp, dd = dp.Plane(pm, "cpu"), dp.Plane(dm, "cpu")
        scheduler = SimpleNamespace(_pre_layerwise=SimpleNamespace(gpu_data_plane=pp))
        hub = rt.RemoteHub(scheduler, stream_factory=lambda n: [(None, None, None)] * n)
        self.addCleanup(hub.close)
        freed = []
        self.addCleanup(pp.close)
        self.addCleanup(dd.close)
        req = SimpleNamespace(
            _pre_layerwise_source=SimpleNamespace(
                peers=((SimpleNamespace(mooncake_session_id="D"), None),)
            )
        )
        ctx = hub.registry.register(
            "gpu", req, 0, 8, 8, "token", on_release=lambda ctx: freed.append(True)
        )
        route = dict(
            host=hub.host,
            port=hub.port,
            handoff_id="gpu",
            epoch=ctx.owner_epoch,
            token="token",
        )

        def compute(job, stream, cache):
            q = job.q_cpu
            return {}, q.float() * 3, torch.zeros(q.shape[0], dtype=torch.float32)

        self.enterContext(patch.object(dsv41_remote, "execute", compute))
        self.enterContext(
            patch.object(
                rt, "_send_tensor", side_effect=AssertionError("TCP tensor payload")
            )
        )
        return SimpleNamespace(
            dp=dp,
            rt=rt,
            dsv=dsv41_remote,
            p=p,
            d=d,
            pp=pp,
            dd=dd,
            hub=hub,
            source=req,
            ctx=ctx,
            route=route,
            freed=freed,
            compute=compute,
        )

    def test_gpu_transport_control_only_roundtrip_batches_and_slot_reuse(self):
        """Q/O/LSE must use registered regions; returned tensors cannot alias reuse."""
        f = self._gpu_transport_fixture()
        client = f.dd.client(f.route)
        address = client.slab.tensor.data_ptr()
        previous = []
        for size in (1, 2, 4, 1):
            q = (
                torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
                + size
            )
            reqs = [SimpleNamespace() for _ in range(size)]
            rows = [{**f.route, "dsv41_sparse": True} for _ in reqs]
            result, out, lse = client.call([q] * size, rows, reqs)
            self.assertTrue(torch.equal(out, torch.stack([q.float() * 3] * size)))
            self.assertEqual(tuple(lse.shape), (size, 4))
            self.assertEqual(address, client.slab.tensor.data_ptr())
            self.assertTrue(all(f.dp.ready(req) for req in reqs))
            previous.append((out, out.clone()))
        self.assertTrue(all(torch.equal(out, saved) for out, saved in previous))
        self.assertEqual([len(c) for c in f.d.calls], [1] * 4)
        self.assertEqual([len(c) for c in f.p.calls], [2] * 4)
        client.close()
        self.assertFalse(f.d.regions)
        self.assertFalse(f.p.regions)

    def test_gpu_transport_cancel_keeps_registry_until_reader_and_return_dma_finish(
        self,
    ):
        """A cancelled source stays pinned while the GPU RPC owns its reader reference."""
        f = self._gpu_transport_fixture()
        started, proceed = threading.Event(), threading.Event()

        def delayed(*args):
            started.set()
            if not proceed.wait(4):
                raise TimeoutError("reader gate")
            return f.compute(*args)

        self.enterContext(patch.object(f.dsv, "execute", delayed))
        req = SimpleNamespace()
        results, errors = [], []

        def run():
            try:
                results.append(
                    f.dd.client(f.route).call(
                        [torch.ones(2, 8, dtype=torch.bfloat16)],
                        [{**f.route, "dsv41_sparse": True}],
                        [req],
                    )
                )
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(started.wait(4))
            self.assertFalse(f.dp.ready(req))
            self.assertFalse(f.hub.registry.release("gpu", expected_req=f.source))
            self.assertFalse(f.freed)
        finally:
            proceed.set()
            thread.join(4)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self._mesh_wait(lambda: bool(f.freed))
        self.assertTrue(f.dp.ready(req))

    def test_gpu_transport_unknown_dma_quarantines_address_and_never_retries_request(
        self,
    ):
        """An old DMA may still arrive; only a new allocation can serve another request."""
        f = self._gpu_transport_fixture()
        client = f.dd.client(f.route)
        req = SimpleNamespace()
        f.d.fail = True
        with self.assertRaises(RuntimeError):
            client.call(
                [torch.ones(2, 8, dtype=torch.bfloat16)],
                [{**f.route, "dsv41_sparse": True}],
                [req],
            )
        old = client.slab
        self.assertTrue(old.quarantined)
        with self.assertRaises(RuntimeError):
            f.dp.ready(req)
        f.d.fail = False
        new = f.dd.client(f.route)
        self.assertNotEqual(new.slab.tensor.data_ptr(), old.tensor.data_ptr())
        _, out, _ = new.call(
            [torch.ones(2, 8, dtype=torch.bfloat16)],
            [{**f.route, "dsv41_sparse": True}],
            [SimpleNamespace()],
        )
        old.tensor.fill_(91)  # simulate a late writer to the quarantined region
        self.assertTrue(torch.equal(out, torch.full_like(out, 3)))
        self.assertEqual(len(f.d.calls), 2)  # failed operation was never retried
        new.close()
        self.assertIn(old.tensor.data_ptr(), f.d.regions)

    def test_gpu_transport_slot_requires_ack_and_consumer_proof_before_retirement(self):
        """Sequence changes cannot replace a transfer/consumer completion proof."""
        from sglang_remote_attn import remote_data_plane as dp

        pending = [True]
        io = dp.RequestIO()
        io.begin()
        self.assertFalse(io.ready())
        io.finish(SimpleNamespace(query=lambda: not pending[0]))
        self.assertFalse(io.ready())
        pending[0] = False
        self.assertTrue(io.ready())
        f = self._gpu_transport_fixture()
        lane = dp.Lane(f.pp, f.dd.client(f.route).slab.descriptor(), "D")
        self.addCleanup(lane.drop)
        lane.begin(1)
        with self.assertRaises(RuntimeError):
            lane.ack(1)
        with self.assertRaises(RuntimeError):
            lane.begin(2)
        lane.state = "returned"
        lane.ack(1)
        lane.begin(2)
        lane.drop()
        with self.assertRaises(RuntimeError):
            lane.begin(3)
        self.assertTrue(lane.slab.quarantined)

    def test_gpu_transport_mha_native_executor_returns_tensors_without_cpu_payload(
        self,
    ):
        """Dense MHA must preserve native execution while bypassing Q/O/LSE CPU copies."""
        f = self._gpu_transport_fixture()
        keys = torch.ones(8, 2, 8)
        pool = SimpleNamespace(
            get_key_buffer=lambda layer: keys, get_value_buffer=lambda layer: keys
        )
        f.source.kv = SimpleNamespace(req_pool_idx=0)
        f.hub.scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: pool
        )
        f.hub.scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(8).reshape(1, 8)
        )
        controller = SimpleNamespace(gpu_data_plane=f.dd)
        q = torch.ones(2, 8, dtype=torch.bfloat16)
        req = SimpleNamespace(_pre_layerwise_target=SimpleNamespace(owner=controller))
        state = SimpleNamespace(req=req, remote_service=f.route)
        prep = dict(
            state=state,
            q_row=q,
            row_header=dict(
                handoff_id="gpu", epoch=f.ctx.owner_epoch, layer=0, sm_scale=0.25
            ),
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "flashinfer": SimpleNamespace(
                        single_decode_with_kv_cache=lambda query, *a, **kw: (
                            query.float() * 3,
                            torch.zeros(query.shape[0]),
                        )
                    )
                },
            ),
            patch.object(f.rt, "PAGED_BATCH", False),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(synchronize=lambda: None),
            ),
            patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("tensor CPU payload")
            ),
        ):
            _, out, lse = f.dp.mha_call([prep])
        self.assertTrue(torch.equal(out, q.float().unsqueeze(0) * 3))
        self.assertEqual(tuple(lse.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
