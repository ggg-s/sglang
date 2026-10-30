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

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

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
        freed = []
        req = req or SimpleNamespace()
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
        )
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
            scheduler.process_batch_result = process
            scheduler.launch_batch_sample_if_needed = lambda *a: None
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
            controller.retire_targets()
            self.assertEqual(f.allocator.available_size(), 30)
            target.decode_req.kv_receiver.done = True
            controller.retire_targets()  # Native writes are complete; GPU is not.
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
            controller.retire_targets()
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
            with self.assertRaisesRegex(RuntimeError, "unproven"):
                controller.retire_targets()
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
                controller.retire_targets()
            with self.assertRaisesRegex(RuntimeError, "refusing retirement retry"):
                controller.retire_targets()
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
            with self.assertRaisesRegex(RuntimeError, "did not drain"):
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
                        if role == "decode" and mode != "off":
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
        self.assertTrue(rows[0]["leases"]["cache_mismatch_rejected"])
        self.assertTrue(rows[0]["leases"]["overlap_mismatch_rejected"])
        self.assertEqual(rows[0]["d_overlap"], rows[1]["d_overlap"])
        self.assertEqual(
            rows[0]["d_overlap"]["snapshots"],
            [[1, 0, True], [1, 0, True], [0, 1, True]],
        )
        self.assertEqual(rows[0]["d_overlap"]["steady"], [False] * 32)
        self.assertTrue(rows[0]["d_overlap"]["native_restriction"])
        self.assertTrue(rows[0]["d_overlap"]["reentry"])
        self.assertEqual(rows[0]["d_overlap"]["gpu_proven"], 1)
        self.assertEqual(rows[0]["d_overlap"]["retraction"], [True, 1])
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


if __name__ == "__main__":
    unittest.main()
