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
from collections import deque
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


def _request_list(items=(), *, kind="WAITING", req_of=lambda item: item):
    from sglang.srt.managers.scheduler_components.request_list import (
        IndexedReqList,
        ListKind,
    )

    result = IndexedReqList(ListKind[kind], req_of=req_of)
    result.join(items)
    return result


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
    from sglang.srt.managers.scheduler_components.request_list import (
        IndexedReqList,
        ListKind,
    )

    namespace = {
        "IndexedReqList": IndexedReqList,
        "ListKind": ListKind,
        "deque": deque,
        "pre_alloc_fill_len": _source_function(
            "disaggregation/utils.py", "pre_alloc_fill_len"
        ),
        "torch": torch,
        "dataclasses": dataclasses,
        "dataclass": dataclasses.dataclass,
        # These CPU fixtures model the supported unsharded deployment.
        "is_cp_sharded_padded": lambda: False,
        "_SCHEDULER_WINDOW": object(),
        "kv_storage_of": _native_storage_of,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[owner]


def _source_function(relative, name, *, namespace=None):
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
    namespace = {
        "is_cp_sharded_padded": lambda: False,
        "kv_storage_of": _native_storage_of,
        **(namespace or {}),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@functools.lru_cache(None)
def _native_storage_globals():
    # Execute production storage lifecycle methods; only disabled CP and GPU
    # model/type imports are omitted by the source loader.
    import msgspec

    ns = dict(
        torch=torch, msgspec=msgspec, page_interleave_shard_size=lambda allocator: 1
    )
    for relative, names in (
        ("mem_cache/kv/storage.py", {"ResumeUnits"}),
        (
            "mem_cache/static_kv_storage.py",
            {"StaticKVStorage", "kv_storage_of", "acquire_rows"},
        ),
    ):
        path = ROOT / "python/sglang/srt" / relative
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


def _native_storage_of(*args, **kwargs):
    return _native_storage_globals()["kv_storage_of"](*args, **kwargs)


def _bind_window_retirement(cache, advance):
    # This focused fixture supplies a storage boundary; retire_window is the
    # actual native dispatcher, and advance retains the test's physical ledger.
    cache.req_to_token_pool = getattr(cache, "req_to_token_pool", None)
    allocator = getattr(cache, "token_to_kv_pool_allocator", SimpleNamespace())
    allocator.storage = SimpleNamespace(
        advance=lambda req, frontier, **kw: advance(req, frontier)
    )
    cache.token_to_kv_pool_allocator = allocator
    cache.window_retention = lambda req: ((None, None, None),)
    cache.supports_prefix_sharing = lambda: True


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
        cls.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                        KVPoll=_source_class("disaggregation/base/conn.py", "KVPoll")
                    )
                },
            )
        )
        storage = _source_function("mem_cache/allocation.py", "_storage")
        retire = _source_function(
            "mem_cache/allocation.py", "retire_window", namespace={"_storage": storage}
        )
        cls.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.mem_cache.allocation": SimpleNamespace(
                        retire_window=retire
                    ),
                    "sglang.srt.mem_cache.static_kv_storage": SimpleNamespace(
                        kv_storage_of=_native_storage_of
                    ),
                    "sglang.srt.layers.cp.base": SimpleNamespace(
                        is_cp_sharded_padded=lambda: False
                    ),
                },
            )
        )
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
        allocator.storage = allocator.state = None
        allocator.has_window_pool = False
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
            is_cp_sharded_padded=lambda: False,
            kv_storage_of=_native_storage_of,
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
        cache.logical_page_size = 1
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

    def _destination_cache_target(self, directory):
        from sglang_remote_attn.destination_cache import Guard, key

        f = self._radix()
        warm, _ = f.request([1, 2], "warm-d")
        f.release_kv_cache(warm, f.cache, checkpoint=True)
        req, indices = f.request([1, 2, 3, 4], "decode-cache")
        req.bootstrap_room = 71
        req.finished = lambda: False
        req._remote_attn_decode_match = (
            req.cache_request_handle,
            2,
            key(req),
            req.lock,
        )
        guard = Guard(f.cache)
        lease = guard.hold(req)
        controller, target, freed = self._overlap_target(directory, req)
        controller.destination_cache = guard
        controller.scheduler.tree_cache = f.cache
        target.cache_lease = lease
        target.decode_prefix_len = 2
        target.coverage = self.hooks.pre_layerwise.LayerCoverage(4, 2)
        for layer in range(2):
            target.coverage.complete(layer, 0, 2)

        def complete():
            target.decode_req.kv_receiver.done = True
            for layer in range(2):
                self.hooks.pre_layerwise.write_json(
                    target.path / f"layer-{layer}.json",
                    dict(generation="g", layer=layer, start=2, end=4),
                )

        f.req, f.indices, f.guard, f.lease = req, indices, guard, lease
        f.controller, f.target, f.complete = controller, target, complete
        return f

    def test_d_cache_native_prefix_match_and_prealloc_use_only_incremental_indices(
        self,
    ):
        """Native cached pages stay shared; PD receives only fresh suffix indices."""
        from sglang_remote_attn.destination_cache import (
            Guard,
            match_prefix,
            send_metadata,
        )

        f = self._radix()
        warm, cached = f.request([1, 2], "warm-native-d")
        f.release_kv_cache(warm, f.cache, checkpoint=True)
        req = self._req()
        req.rid, req.bootstrap_room = "native-d", 91
        req.origin_input_ids, req.output_ids = array("q", [1, 2, 3, 4]), array("q")
        req.extra_key, req.cache_salt, req.priority = None, None, 0
        req.cache_request_handle = f.CacheRequestHandle(req.rid, 0)
        req.lock = None
        req.owned_kv_len = lambda: 4
        req.finished = lambda: False
        req._compute_max_prefix_len = lambda size: size - 1
        native_match = _source_function("mem_cache/common.py", "match_kv_cache")
        radix_key = _source_function("mem_cache/common.py", "_req_radix_key")
        radix_key.__globals__["RadixKey"] = f.RadixKey
        native_match.__globals__.update(
            MatchPrefixParams=f.MatchPrefixParams,
            _req_radix_key=radix_key,
            envs=SimpleNamespace(
                SGLANG_RADIX_FORCE_MISS=SimpleNamespace(get=lambda: False)
            ),
        )
        Queue = _source_class(
            "disaggregation/decode.py",
            "DecodePreallocQueue",
            {"_match_prefix_and_lock", "_pre_alloc", "_alloc_for_decode_prealloc"},
        )
        prefix = _source_class(
            "disaggregation/decode_hicache_mixin.py", "DecodePrefixMatch"
        )
        build = _source_class(
            "disaggregation/decode_hicache_mixin.py",
            "DecodeHiCachePreallocMixin",
            {"_build_decode_prefix_match"},
        )
        build._build_decode_prefix_match.__globals__["DecodePrefixMatch"] = prefix
        Queue._build_decode_prefix_match = build._build_decode_prefix_match
        alloc = _source_function(
            "disaggregation/decode.py",
            "alloc_for_decode_prealloc",
            namespace={
                "_evicted_by_caller": _source_function(
                    "disaggregation/decode.py", "_evicted_by_caller"
                )
            },
        )
        ns = Queue._pre_alloc.__globals__
        ns.update(
            match_kv_cache=native_match,
            alloc_for_decode_prealloc=alloc,
            get_disagg=lambda: SimpleNamespace(
                disaggregation_decode_enable_radix_cache=True
            ),
        )
        queue_ = Queue()
        guard = Guard(f.cache)
        queue_.scheduler = SimpleNamespace(
            _pre_layerwise=SimpleNamespace(destination_cache=guard, tp_size=1),
            enable_decode_hicache=False,
            enable_hisparse=False,
        )
        queue_.tree_cache = f.cache
        queue_.req_to_token_pool = f.cache.req_to_token_pool
        queue_.token_to_kv_pool_allocator = f.allocator
        queue_._uses_swa_tail_prealloc = lambda: False
        queue_._swa_tail_len = lambda fill: fill
        queue_._required_alloc_tokens = lambda **kw: kw["fill_len"] - kw["prefix_len"]
        queue_._radix_full_available = f.allocator.available_size
        result = match_prefix(Queue._match_prefix_and_lock, queue_, req)
        self.assertEqual((result.l1_prefix_len, result.decode_prefix_len), (2, 2))
        # Failed capacity admission unlocks the native request pin. The pending
        # certificate keeps this exact hit alive and retries without lookup/TP.
        for _ in range(8):
            f.cache.unlock(req.lock)
            req.lock = None
            self.assertIs(
                match_prefix(
                    lambda *a: self.fail("repeated native cache lookup"), queue_, req
                ),
                result,
            )
        before = f.allocator.available_size()
        delta = queue_._pre_alloc(req, result.prefix_indices, 2, 2)
        self.assertEqual(delta.numel(), 2)
        self.assertEqual(before - f.allocator.available_size(), 2)
        self.assertEqual(
            f.cache.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :2].tolist(),
            cached.tolist(),
        )
        self.assertTrue(set(delta.tolist()).isdisjoint(cached.tolist()))
        # Native pop_preallocated sets this after _pre_alloc, before metadata.
        req.kv.cache_protected_len = 2
        sent = []
        metadata = lambda *a, **kw: sent.append(
            (a[2].tolist(), kw["decode_prefix_len"])
        )
        with self.assertRaisesRegex(RuntimeError, "changed before"):
            send_metadata(
                metadata,
                queue_,
                SimpleNamespace(req=req),
                delta,
                1,
                decode_prefix_len=1,
            )
        self.assertEqual(sent, [])
        send_metadata(
            metadata, queue_, SimpleNamespace(req=req), delta, 1, decode_prefix_len=2
        )
        self.assertEqual(sent, [(delta.tolist(), 2)])
        self.assertEqual(guard.pending, {})
        lease = guard.hold(req)
        self.assertEqual(lease.prefix, 2)
        lease.abort = True
        req.skip_radix_cache_insert = False
        lease.retire(lambda: f.release_kv_cache(req, f.cache, checkpoint=False))
        pending_req = copy.copy(req)
        pending_req.kv, pending_req.lock = self.Kv(), None
        pending_req.cache_request_handle = f.CacheRequestHandle("cancel-pending", 0)
        pending_req.rid = "cancel-pending"
        match_prefix(Queue._match_prefix_and_lock, queue_, pending_req)
        self.assertEqual(len(guard.pending), 1)
        f.cache.finish(pending_req.cache_request_handle, f.CacheRequestOutcome.ABORT)
        self.assertEqual(guard.pending, {})
        self.assertIsNone(pending_req.lock)
        f.cache.evict(f.EvictParams(num_tokens=32))
        self.assertEqual(f.allocator.available_size(), 32)

    def test_d_cache_publish_waits_for_transport_gpu_and_ordered_retirement(self):
        """Half-local KV must not become cache-visible or be deduped under DMA."""
        from sglang_remote_attn.decode_overlap import BatchProof

        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            req, target = f.req, f.target
            # Prompt publication is armed by admission itself, not dependent
            # on a running decode request calling checkpoint before finish.
            self.assertEqual(f.lease.pending, 4)
            hit = lambda: f.cache.match_prefix(
                f.MatchPrefixParams(f.RadixKey(array("q", [1, 2, 3, 4])))
            )
            self.assertEqual(hit().device_prefix_len, 2)
            f.cache.evict(f.EvictParams(num_tokens=32))
            self.assertTrue(
                set(f.indices.tolist()).isdisjoint(f.allocator.free_pages.tolist())
            )
            with self.assertRaisesRegex(RuntimeError, "cannot reset"):
                f.cache.reset()
            # A concurrent complete request inserts canonical suffix pages.
            other, canonical = f.request([1, 2, 3, 4], "canonical-d")
            f.release_kv_cache(other, f.cache, checkpoint=True)
            self.assertNotEqual(f.indices[2:].tolist(), canonical[2:].tolist())
            ready = [False]
            target.forward_proofs.append(
                BatchProof(
                    result=SimpleNamespace(
                        delay_sample_func=None,
                        copy_done=SimpleNamespace(query=lambda: ready[0]),
                    )
                )
            )
            protocol = f.controller.decode_overlap.retirement
            for _ in range(16):
                self.assertIsNone(self._retire_step(f.controller))
            self.assertEqual(protocol.final_votes, 0)
            f.complete()
            self._retire_step(f.controller)
            self.assertFalse(target.closed)
            self.assertEqual(
                f.cache.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :4
                ].tolist(),
                f.indices.tolist(),
            )
            ready[0] = True
            self._retire_step(f.controller)
            self.assertTrue(target.closed and f.lease.closed)
            done = self.hooks.pre_layerwise.read_json(target.path / "done.json")
            self.assertTrue(
                done["d_cache_tp_committed"] and done["d_cache_guard_closed"]
            )
            self.assertEqual(
                (done["d_cached_prefix_tokens"], done["d_cache_retained_tokens"]),
                (2, 4),
            )
            self.assertEqual(
                f.cache.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :4
                ].tolist(),
                canonical.tolist(),
            )
            before = protocol.final_votes
            for _ in range(32):
                self._retire_step(f.controller)
            self.assertEqual(protocol.final_votes, before)
            f.release_kv_cache(req, f.cache, checkpoint=True)
            f.cache.evict(f.EvictParams(num_tokens=32))
            self.assertEqual(f.allocator.available_size(), 32)
            self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)

    def test_d_cache_cancel_is_sticky_and_native_release_frees_only_private_pages(self):
        """Cancel during Remote must drain writes and never publish its suffix."""
        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            f.cache.checkpoint(f.req, up_to=4)
            hit = f.cache.match_prefix(
                f.MatchPrefixParams(f.RadixKey(array("q", [1, 2, 3, 4])))
            )
            self.assertEqual(hit.device_prefix_len, 2)  # No publication under DMA.
            f.cache.finish(f.req.cache_request_handle, f.CacheRequestOutcome.ABORT)
            f.cache.finish(f.req.cache_request_handle, f.CacheRequestOutcome.SUCCESS)
            self.hooks._release_kv(f.release_kv_cache, f.req, f.cache, checkpoint=False)
            self.assertIsNone(self._retire_step(f.controller))
            self.assertFalse(f.req.kv.is_kv_released)
            f.complete()
            self._retire_step(f.controller)
            self.assertTrue(f.req.kv.is_kv_released and f.lease.closed)
            f.cache.checkpoint(
                f.req, up_to=4
            )  # Late checkpoint after ABORT is ignored.
            hit = f.cache.match_prefix(
                f.MatchPrefixParams(f.RadixKey(array("q", [1, 2, 3, 4])))
            )
            self.assertEqual(hit.device_prefix_len, 2)
            self.hooks._release_kv(f.release_kv_cache, f.req, f.cache, checkpoint=True)
            f.cache.evict(f.EvictParams(num_tokens=32))
            self.assertEqual(f.allocator.available_size(), 32)
            self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)

    def test_d_cache_stale_retirement_intent_retries_only_after_new_state(self):
        """A queued publish command cannot override a later cancellation."""
        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            f.cache.checkpoint(f.req, up_to=4)
            f.complete()
            command = self._retire_step(f.controller, finalize=False)
            self.assertIsNotNone(command)
            self.hooks._release_kv(f.release_kv_cache, f.req, f.cache, checkpoint=False)
            protocol = f.controller.decode_overlap.retirement
            protocol.finalize()
            self.assertFalse(f.target.closed)
            self.assertEqual(protocol.retries, 1)
            self._retire_step(f.controller)
            self.assertTrue(f.target.closed and f.req.kv.is_kv_released)
            self.assertEqual(protocol.final_votes, 2)
            f.cache.evict(f.EvictParams(num_tokens=32))
            self.assertEqual(f.allocator.available_size(), 32)

    def test_d_cache_checkpoint_failure_quarantines_pin_and_rejects_attempt_reuse(self):
        """Partial native cache mutation is not retryable or proof of release."""
        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            f.cache.checkpoint(f.req, up_to=4)
            f.complete()
            f.guard.checkpoint = lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("cache insert failed")
            )
            with self.assertRaisesRegex(RuntimeError, "cache insert failed"):
                self._retire_step(f.controller)
            self.assertTrue(f.lease.failed and f.target.release_failed)
            self.assertIsNotNone(f.lease.pin)
            self.assertEqual(len(f.guard.leases), 1)
            with self.assertRaisesRegex(RuntimeError, "retaining"):
                f.lease.retire(None)
            with self.assertRaisesRegex(RuntimeError, "quarantined"):
                f.cache.reset()

    def test_d_cache_retraction_drains_before_backup_and_invalidates_old_match(self):
        """Retraction cannot read partial KV or let an old attempt own new pages."""
        from sglang_remote_attn.decode_overlap import before_retract
        from sglang_remote_attn.destination_cache import checkpoint_allowed

        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            f.cache.checkpoint(f.req, up_to=4)
            f.complete()
            seen = []
            NativeReq = _source_class(
                "managers/schedule_batch.py",
                "Req",
                {"reset_for_retract", "advance_cache_request_handle"},
            )
            f.req.retraction_count, f.req.input_embeds = 0, None
            f.req.advance_cache_request_handle = lambda: (
                NativeReq.advance_cache_request_handle(f.req)
            )

            def backup_and_release(*, req):
                self.assertTrue(f.target.closed and f.lease.closed)
                self.assertFalse(checkpoint_allowed(req, True))
                seen.append(
                    f.cache.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, :4
                    ].clone()
                )
                f.release_kv_cache(req, f.cache, checkpoint=False)
                self.hooks._reset_retracted(NativeReq.reset_for_retract, req)
                return True

            self.assertTrue(before_retract(backup_and_release, req=f.req))
            self.assertTrue(torch.equal(seen[0], f.indices))
            self.assertIsNone(f.req._remote_attn_decode_match)
            self.assertTrue(checkpoint_allowed(f.req, True))
            f.cache.evict(f.EvictParams(num_tokens=32))
            self.assertEqual(f.allocator.available_size(), 32)

    def test_d_cache_incremental_coverage_separates_cached_prefix_from_wire_offsets(
        self,
    ):
        """Suffix-relative scatter must not mark a hole or rewrite shared prefix."""
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            f.target.native_staging = True
            f.target.decode_req._staging_completed_layers = [([0], 0, 2)]
            f.target.queue.staging_handler = SimpleNamespace(
                advance_scatter=lambda dr: None, is_failed=lambda dr: False
            )
            f.target.refresh()
            self.assertTrue(f.target.coverage.ready(0))
            self.assertFalse(f.target.coverage.ready(1))
            proof = pl.read_json(f.target.path / "layer-0.json")
            self.assertEqual((proof["start"], proof["end"]), (2, 4))
            f.req.kv.cache_protected_len = 3
            f.target.queue.staging_handler.advance_scatter = lambda dr: self.fail(
                "scatter ran after prefix changed"
            )
            with self.assertRaisesRegex(RuntimeError, "changed attempt, row or prefix"):
                f.target.refresh()
            f.req.kv.cache_protected_len = 2
            f.target.queue.staging_handler.advance_scatter = lambda dr: None
            f.target.decode_req._staging_completed_layers = [([1], 1, 2)]
            with self.assertRaisesRegex(RuntimeError, "does not cover"):
                f.target.refresh()
            source = pl.Source(
                None,
                f.req,
                Path(directory),
                "source",
                None,
                slice(0, 2),
                (),
                4,
                decode_prefix_len=2,
            )
            source.complete([0])
            self.assertEqual(pl.read_json(Path(directory) / "layer-0.json")["start"], 2)

    def test_d_cache_unknown_transport_completion_never_publishes_or_reuses_pages(self):
        """A failed transfer is not a cancellation ACK or permission to free."""
        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            self.hooks._release_kv(f.release_kv_cache, f.req, f.cache, checkpoint=False)
            self.hooks.pre_layerwise.write_json(
                f.target.path / "error.json",
                dict(generation="g", error="DMA completion unknown"),
            )
            with self.assertRaisesRegex(RuntimeError, "proof unproven"):
                self._retire_step(f.controller)
            self.assertFalse(f.req.kv.is_kv_released or f.lease.closed)
            self.assertIsNotNone(f.lease.pin)
            self.assertTrue(
                set(f.indices.tolist()).isdisjoint(f.allocator.free_pages.tolist())
            )
            with self.assertRaisesRegex(RuntimeError, "cannot reset"):
                f.cache.reset()

    def test_d_cache_without_overlap_still_requires_recorded_gpu_completion(self):
        """Turning scheduler overlap off must not bypass the publication barrier."""
        from sglang_remote_attn.decode_overlap import install

        native_receiver = _source_class(
            "managers/scheduler_components/request_receiver.py",
            "SchedulerRequestReceiver",
            strip_method_decorators=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            f = self._destination_cache_target(directory)
            scheduler = f.controller.scheduler
            scheduler.enable_overlap = False
            scheduler.request_receiver = native_receiver(
                **{field.name: None for field in dataclasses.fields(native_receiver)}
            )
            f.controller.decode_overlap.retirement.close()
            self.assertIsNone(install(f.controller))
            overlap = f.controller.decode_overlap = install(f.controller, force=True)
            self.addCleanup(overlap.retirement.close)
            overlap.retirement.register(f.target)
            completed, recorded, stream = [False], [], object()
            event = SimpleNamespace(query=lambda: completed[0], record=recorded.append)
            batch = SimpleNamespace(
                reqs=[f.req],
                forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
            )
            with (
                patch.object(torch.cuda, "Event", return_value=event),
                patch.object(torch.cuda, "current_stream", return_value=stream),
            ):
                overlap.run_batch(
                    lambda *a: SimpleNamespace(delay_sample_func=None, copy_done=None),
                    scheduler,
                    batch,
                )
            self.assertEqual(recorded, [stream])
            f.complete()
            self._retire_step(f.controller)
            self.assertFalse(f.lease.closed)
            completed[0] = True
            self._retire_step(f.controller)
            self.assertTrue(f.lease.closed)
            self.assertEqual(overlap.retirement.cache_votes, 1)
            f.release_kv_cache(f.req, f.cache, checkpoint=True)
            f.cache.evict(f.EvictParams(num_tokens=32))
            self.assertEqual(f.allocator.available_size(), 32)

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

        from sglang_remote_attn.source_resources import release_metadata

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
            req.metadata_buffer_index = 5
            metadata_frees = []
            release_metadata(
                _source_function(
                    "disaggregation/prefill.py", "maybe_release_metadata_buffer"
                ),
                req,
                SimpleNamespace(free=metadata_frees.append),
            )
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
                self.assertEqual(metadata_frees, [])
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
                self.assertEqual(metadata_frees, [])
                status.done = True
                self.assertTrue(source.poll())
                source.retire()
                source.retire()
                self.assertEqual(metadata_frees, [5])
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
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(
                storage=None, state_slots=SimpleNamespace(free=freed.append)
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
            root = Path(directory)
            for rank in (0, 1):
                for i in range(203):
                    (root / f"cmd.{rank}.123.{i}.done").touch()
            tombstone = root / "cancel.owner.epoch.json"
            tombstone.touch()
            path = root / "cmd.0.json"
            path.write_text(
                json.dumps({"cmd": "freeze", "rid": "one", "nonce": "first"})
            )
            sentinel = self.runtime.poll_broadcast_control()
            for rank in (0, 1):
                self.assertEqual(len(list(root.glob(f"cmd.{rank}.*.done"))), 200)
            self.assertTrue(tombstone.exists())
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
            self.runtime.enqueue_broadcast_control_once({"cmd": "resume"}, "test")
            self.assertEqual(self.runtime.poll_broadcast_control()["data"]["cmd"], "resume")

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
            (ROOT / "python/sglang/srt/model_executor/forward_mode.py").read_text()
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
        runner.uocr_greedy_graph = None
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

    def _prefix_stream_fixture(
        self, *, dsv=False, staged=False, enabled=True, inflight=1
    ):
        from collections import deque

        from sglang_remote_attn import prefix_stream as ps
        from sglang_remote_attn.source_cache import CacheGuard

        staged = staged or dsv
        f = self._dsv41_fixture(length=33 if dsv else 8)
        if dsv:
            c = self._dsv41_cache()
            req = f.req
            req.cache_request_handle = c.native.CacheRequestHandle(req.rid, 0)
            req.kv.cache_protected_len = 8
            req.last_node, req.lock = 1, c.cache.lock(1)
            cache, guard = c.cache, c.guard
            f.controller.dsv41_remote = True
            f.controller.schema_signature = f.schema
            f.manager._dsv41_layerwise_schema = f.schema
            f.manager._dsv41_receive_sender_ready = True
        else:
            c = self._radix()
            warm, _ = c.request(list(range(4)), "prefix-warm")
            c.release_kv_cache(warm, c.cache, checkpoint=True)
            req, _ = c.request(list(range(8)), "prefix-qwen")
            req.disagg_kv_sender = f.sender
            req.bootstrap_room = 7
            req.pending_bootstrap = False
            req.start_send_idx = req.disagg_decode_prefix_len = 0
            req.extend_end, req.extend_len = 8, 4
            req.inflight_steps = 0
            req.multimodal_inputs = req.input_embeds = None
            cache, guard = c.cache, CacheGuard(c.cache)
            controller = self.hooks.pre_layerwise.Controller.__new__(
                self.hooks.pre_layerwise.Controller
            )
            controller.scheduler = f.scheduler
            controller.rank, controller.tp_size, controller.layers = 0, 1, 2
            controller.config = dataclasses.replace(
                f.controller.config, mode="remote", group_size=1
            )
            controller.sources = []
            controller.batches = 0
            f.controller = controller
            f.manager.kv_args.kv_data_ptrs = f.manager.kv_args.kv_data_ptrs[:4]
            f.manager.kv_args.kv_item_lens = [8] * 4
            f.manager.kv_args.page_size = 1
            f.manager.kv_args.state_data_ptrs = []
            f.registration.dst_kv_ptrs = f.registration.dst_kv_ptrs[:4]
            f.scheduler.req_to_token_pool = c.cache.req_to_token_pool
            f.scheduler.token_to_kv_pool_allocator = SimpleNamespace(
                page_size=1, translate_kv_indices_for_transfer=lambda x: x
            )
            f.sender.num_kv_indices = 8
        req.prefix_len = 8 if dsv else 4
        req.extra_key = req.cache_salt = None
        req.host_hit_length = 0
        req.early_send_prefix_end = None
        f.req = req
        f.batch.reqs = [req]
        f.controller.cache_guard = guard
        f.controller.config = dataclasses.replace(f.controller.config, mode="remote")
        f.scheduler._pre_layerwise = f.controller
        f.scheduler.enable_overlap = False
        # Native computation/chunking is separately exercised by existing tests;
        # this fixture controls only the transport/GPU boundaries.
        f.scheduler.enable_staging = False
        f.manager.enable_staging = staged and not dsv
        f.manager.pp_rank = 0
        f.registration.dst_aux_ptrs = []
        f.registration.dst_aux_item_lens = []
        f.manager.kv_args.aux_data_ptrs = f.manager.kv_args.aux_item_lens = []
        f.registration.staging_base_ptr, f.registration.staging_total_size = 4096, 4096
        f.manager.kv_args.page_size = 8 if dsv else 1
        f.scheduler.disagg_metadata_buffers = SimpleNamespace(set_buf=lambda req: None)
        offer_path = f.controller.config.directory / "7/0/offer.json"
        offer = self.hooks.pre_layerwise.read_json(offer_path)
        offer.update(
            mode="remote",
            prefix_len=len(req.origin_input_ids),
            layers=f.controller.layers,
            range_stream=1,
            native_staging=staged,
            recv_host_staging=staged,
        )
        if dsv:
            offer.update(
                remote_boundary="progressive-sparse",
                staging_wire=self.hooks.dsv41_receive.WIRE,
                progressive_ring_bytes=4096,
            )
            f.registration.staging_base_ptr, f.registration.staging_total_size = (
                4096,
                4096,
            )
        self.hooks.pre_layerwise.write_json(offer_path, offer)
        wire, requests = deque(), []

        def raw(manager, room, indices, span, last, **kw):
            wire.append(
                SimpleNamespace(
                    room=room, indices=indices, index_slice=span, last=last, kwargs=kw
                )
            )

        f.manager.add_transfer_request = lambda *a, **kw: ps.track(
            raw, f.manager, *a, **kw
        )
        native_request = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"request_layer_staging"},
        ).request_layer_staging
        native_request.__globals__.update(
            NetworkAddress=lambda *a: SimpleNamespace(
                to_tcp=lambda: "native-endpoint", is_ipv6=False
            ),
            struct=__import__("struct"),
        )
        f.peer = next(iter(f.manager.transfer_infos[7].values()))
        f.peer.endpoint, f.peer.dst_port = "127.0.0.1", 31017
        f.manager._send_multipart_locked = lambda endpoint, msg, **kw: requests.append(
            msg
        )
        f.manager.request_layer_staging = functools.partial(native_request, f.manager)
        cached = _source_class(
            "disaggregation/prefill.py",
            "SchedulerDisaggregationPrefillMixin",
            {"maybe_send_cached_prefix_chunk"},
        ).maybe_send_cached_prefix_chunk
        cached.__globals__["envs"] = SimpleNamespace(
            SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX=SimpleNamespace(
                get=lambda: True
            )
        )
        f.cache, f.guard, f.cache_fixture, f.ps = (
            f.cache if hasattr(f, "cache") else cache,
            guard,
            c,
            ps,
        )
        f.wire, f.requests, f.raw, f.offer = wire, requests, raw, offer

        def activate():
            with (
                patch.dict(sys.modules, f.modules),
                patch.dict(
                    os.environ,
                    {"HANDOFF_PREFIX_REMOTE_PROGRESSIVE": "1" if enabled else "0"},
                ),
            ):
                return ps.cached_prefix(cached, f.scheduler, req)

        f.activate = activate
        with patch.dict(os.environ, {"HANDOFF_PREFIX_INFLIGHT": str(inflight)}):
            f.activate()
        f.plan = ps.stream_for(req)
        if enabled:
            self.assertIsNotNone(f.plan)
        return f

    def test_prefill_qualification_uses_batch_chunk_not_request_counter(self):
        """Merged Req lacks the old counter; only the batch identifies its middle chunk."""
        for dsv in (False, True):
            for parked in (False, True):
                with self.subTest(dsv=dsv, parked=parked):
                    f = self._prefix_stream_fixture(dsv=dsv)
                    req, batch = f.req, f.batch
                    self.assertFalse(hasattr(req, "inflight_middle_chunks"))
                    req.extend_end = len(req.origin_input_ids)
                    req.extend_len = req.extend_end - req.start_send_idx
                    req.inflight_steps = 0
                    # Even if the live Req has advanced, this batch is still
                    # a middle chunk; it must not consume the final sender tail.
                    batch.chunked_req = req
                    cursor = f.sender.curr_idx
                    with patch.dict(sys.modules, f.modules):
                        self.assertIsNone(f.controller.prepare(batch))
                        self.assertEqual(f.sender.curr_idx, cursor)
                        self.assertFalse(f.plan.bound)
                        if dsv:
                            self.assertIsNotNone(
                                f.controller.candidates(batch, allow_middle=True)[0]
                            )
                        # A previous launched step, or another parked request,
                        # cannot turn this final batch into a middle chunk.
                        req.inflight_steps = 1
                        batch.chunked_req = object() if parked else None
                        jobs = f.controller.prepare(batch)
                    self.assertTrue(jobs)
                    source = jobs[0] if dsv else jobs[0][0]
                    self.assertIs(source, f.plan.source)
                    self.assertTrue(f.plan.bound)
                    self.assertEqual(req.start_send_idx, len(req.origin_input_ids))

    def test_prefix_stream_off_preserves_native_cached_send(self):
        """The ablation must keep the native cached chunk without range ownership."""
        f = self._prefix_stream_fixture(enabled=False)
        self.assertIsNone(f.plan)
        self.assertFalse(f.controller.sources or f.guard.leases)
        self.assertEqual(len(f.wire), 1)
        task = f.wire.popleft()
        self.assertEqual(len(task.indices), 4)
        self.assertIsNone(task.kwargs.get("kv_buffer_indices"))
        self.assertFalse(hasattr(f.manager, "_prefix_stream_pump"))

    def test_prefix_stream_native_cached_send_has_bounded_ranges_and_staging_ids(self):
        """Logical cached-prefix send expands once, with unique native wire ids."""
        for dsv in (False, True):
            with self.subTest(dsv=dsv):
                f = self._prefix_stream_fixture(dsv=dsv, staged=True)
                self.assertEqual(f.req.start_send_idx, 8 if dsv else 4)
                self.assertEqual(len(f.controller.sources), 1)
                self.assertEqual(len(f.wire), 1)
                before = len(f.plan.pieces)
                f.activate()
                self.assertEqual(len(f.plan.pieces), before)
                completed = []
                while f.wire:
                    task = f.wire.popleft()
                    self.assertLessEqual(len(task.indices), 2)
                    self.assertFalse(task.last)
                    completed.append(task.kwargs["staging_chunk_idx"])
                    task.kwargs["on_complete"]()
                self.assertTrue(f.plan.drained())
                self.assertEqual(len(completed), before)
                self.assertEqual(len(set(completed)), before)
                self.assertEqual([int(m[2]) for m in f.requests], completed)
                self.assertTrue(f.guard.leases)  # IO completion alone never unpins.

    def test_prefix_stream_indexer_and_metadata_precede_pending_global(self):
        """Final indexer work must not wait behind an entire cached global prefix."""
        f = self._prefix_stream_fixture(dsv=True)
        first = f.wire.popleft()
        self.assertEqual(first.kwargs["kv_buffer_indices"], (0,))
        first.kwargs["on_complete"]()
        second = f.wire.popleft()
        self.assertEqual(
            f.schema["entries"][second.kwargs["kv_buffer_indices"][0]][0], "index"
        )
        # Drain urgent index work while leaving cached globals queued.
        second.kwargs["on_complete"]()
        while (
            f.wire
            and f.schema["entries"][f.wire[0].kwargs["kv_buffer_indices"][0]][0]
            == "index"
        ):
            f.wire.popleft().kwargs["on_complete"]()
        active = f.wire.popleft()
        with f.plan.priority():
            active.kwargs["on_complete"]()
            self.assertFalse(f.wire)
            self.assertTrue(f.plan.pump.bulk)
        self.assertEqual(len(f.wire), 1)
        self.assertFalse(f.plan.drained())

    def test_prefix_stream_qwen_final_binding_metadata_and_native_cache_reclaim(self):
        """Metadata can publish while cached KV remains queued; final waits for all ranges."""
        f = self._prefix_stream_fixture()
        f.req.extend_end, f.req.extend_len = 6, 2
        f.scheduler.send_kv_chunk(f.req, last_chunk=False, end_idx=6)
        f.req.extend_end, f.req.extend_len = 8, 2
        with patch.dict(sys.modules, f.modules):
            source = f.controller.prepare(f.batch)[0][0]
        self.assertIs(source, f.plan.source)
        self.assertEqual(len(f.controller.sources), 1)
        self.assertTrue(f.plan.bound)
        for layers, buffers in self.hooks.pre_layerwise.buffer_groups(
            [0, 1, 0, 1], 2, 1
        ):
            f.manager.add_transfer_request(
                7,
                source.indices,
                source.index_slice,
                False,
                kv_buffer_indices=buffers,
                on_complete=functools.partial(source.complete, layers),
            )
        metadata = []
        f.manager.send_aux = lambda *a: metadata.append("metadata") or 0
        event = SimpleNamespace(record=lambda *a: None, synchronize=lambda: None)
        hub = SimpleNamespace(register_handoff=lambda *a, **kw: dict(host="P", port=1))
        runtime = SimpleNamespace(get_remote_hub=lambda s: hub)
        with (
            patch("sglang_remote_attn.pre_layerwise._runtime", return_value=runtime),
            patch.object(torch.cuda, "Event", return_value=event),
            patch.object(torch.cuda, "current_stream", return_value=object()),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "set_device", return_value=None),
        ):
            source.finalize()
            source.thread.join(2)
        self.assertIsNone(source.error)
        self.assertEqual(metadata, ["metadata"])
        self.assertTrue(f.wire and not f.plan.drained())
        self.assertTrue(f.plan.sealed)
        self.assertFalse(any(t.last for t in f.wire))
        while f.wire:
            task = f.wire.popleft()
            if not task.last:
                task.kwargs["on_complete"]()
            else:
                self.assertTrue(f.plan.drained())
        source.release = lambda: f.cache_fixture.release_kv_cache(
            f.req, f.cache, checkpoint=False
        )
        source.retire()
        f.cache.evict(f.cache_fixture.EvictParams(num_tokens=32))
        self.assertEqual(f.cache_fixture.allocator.available_size(), 32)
        self.assertIsNone(f.ps.stream_for(f.req))

    def test_prefix_stream_range_gap_cannot_publish_layer_local(self):
        """Final-tail completion is not proof that the cached prefix has arrived."""
        f = self._prefix_stream_fixture(staged=True)
        with patch.dict(sys.modules, f.modules):
            source = f.controller.prepare(f.batch)[0][0]
        for layers, buffers in self.hooks.pre_layerwise.buffer_groups(
            [0, 1, 0, 1], 2, 1
        ):
            f.manager.add_transfer_request(
                7, source.indices, source.index_slice, False, kv_buffer_indices=buffers
            )
        pieces = f.plan.pieces
        dr = SimpleNamespace(req=f.req, _staging_completed_layers=[])
        # This is D's request ownership, independent from P's cached prefix.
        dr.req = copy.copy(f.req)
        dr.req.kv = SimpleNamespace(cache_protected_len=0)
        target = SimpleNamespace(
            path=f.plan.path,
            generation=f.plan.generation,
            decode_req=dr,
            owner=SimpleNamespace(),
            queue=SimpleNamespace(
                staging_handler=SimpleNamespace(
                    advance_scatter=lambda dr: None, is_failed=lambda dr: False
                )
            ),
        )
        dr._prefix_range_registry = {
            p["id"]: (tuple(p["resources"]), p["lo"], p["hi"] - p["lo"]) for p in pieces
        }
        tail = [p for p in pieces if p["lo"] >= 4]
        dr._staging_completed_layers = [
            (p["resources"], p["lo"], p["hi"] - p["lo"]) for p in tail
        ]
        coverage = f.ps.update(target)
        self.assertFalse(coverage.ready(0) or coverage.ready(1))
        head = [p for p in pieces if p["lo"] < 4]
        dr._staging_completed_layers += [
            (p["resources"], p["lo"], p["hi"] - p["lo"]) for p in head
        ]
        self.assertTrue(all(f.ps.update(target).ready(i) for i in (0, 1)))
        dr._staging_completed_layers.append(((0,), 3, 5))
        with self.assertRaisesRegex(RuntimeError, "no registered prefix range"):
            f.ps.update(target)

    def test_prefix_stream_dsv_native_final_barrier_bypasses_cached_globals(self):
        """Native final state/aux precedes the full cached global backlog."""
        f = self._prefix_stream_fixture(dsv=True)
        f.scheduler.send_kv_chunk(f.req, last_chunk=False, end_idx=16)
        f.req.extend_end, f.req.extend_len = 33, 17
        with patch.dict(sys.modules, f.modules):
            source = f.controller.prepare(f.batch)[0]
        self.assertIs(source, f.plan.source)
        states, aux = [], []
        f.manager.maybe_send_extra = lambda peer, rows, executor, registration: (
            states.append(rows) or 0
        )
        f.manager.send_aux = lambda *a: aux.append(a) or 0
        source.final_event = object()
        with (
            patch.dict(sys.modules, f.modules),
            patch(
                "sglang_remote_attn.dsv41_remote.register",
                side_effect=lambda s: setattr(s, "service", dict(host="P", port=1)),
            ),
        ):
            source.finalize_native(
                f.native_send, f.scheduler, f.req, last_chunk=True, end_idx=33
            )
        f.wire.popleft().kwargs["on_complete"]()
        with patch.dict(sys.modules, f.modules):
            while not source.barrier_ready:
                task = f.wire.popleft()
                self.assertTrue(
                    all(
                        f.schema["entries"][i][0] == "index"
                        for i in task.kwargs["kv_buffer_indices"]
                    )
                )
                task.kwargs["on_complete"]()
        self.assertEqual(len(states), 1)
        self.assertEqual(len(aux), 1)
        self.assertTrue(f.wire and not f.plan.drained())
        self.assertFalse(any(t.last for t in f.wire))
        with patch.dict(sys.modules, f.modules):
            while f.wire:
                task = f.wire.popleft()
                if not task.last:
                    task.kwargs["on_complete"]()
                else:
                    self.assertEqual(task.kwargs["state_indices"], [])
                    self.assertTrue(f.plan.drained())
        self.assertTrue(
            all(f.plan.sent.ready(i) for i in range(len(f.schema["entries"])))
        )

    def test_prefix_stream_qwen_native_ring_accepts_nonfinal_ranges(self):
        """Legacy final-only validation must not reject bounded cached-prefix IO."""
        import logging
        import struct

        from sglang_remote_attn import prefix_stream

        f = self._native_ring_fixture(capacity=32)
        f.dr.req.origin_input_ids = list(range(4))
        f.manager._staging_handler = f.handler
        f.manager._staging_ctx.allocator = f.allocator
        f.manager.kv_buffer_tensors = f.handler.kv_buffer_info
        target = SimpleNamespace(
            generation="prefix-attempt",
            path=Path(self.enterContext(tempfile.TemporaryDirectory())),
            decode_req=f.dr,
            queue=SimpleNamespace(staging_handler=f.handler),
        )
        f.dr._pre_layerwise_offer = target
        native = _source_function(
            "disaggregation/common/staging_handler.py", "handle_staging_req"
        )
        native.__globals__.update(struct=struct, logger=logging.getLogger(__name__))
        original = _source_class(
            "disaggregation/common/staging_handler.py",
            "StagingManagerMixin",
            {"_handle_staging_req"},
        )._handle_staging_req
        original.__globals__["handle_staging_req"] = native
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.common.staging_handler": SimpleNamespace(
                    handle_staging_req=native
                )
            },
        ):
            for index, start in enumerate((0, 2)):
                msg = [
                    b"STAGING_REQ",
                    b"7",
                    str(index).encode(),
                    b"2",
                    b"peer",
                    b"0",
                    struct.pack("Q", 0),
                    str(start).encode(),
                    b"prefix-attempt",
                ]
                self.hooks.dsv41_receive.allocate(original, f.manager, msg)
                slot = f.receiver.chunk_staging_infos[index]
                self.assertGreaterEqual(slot[0], 0)
                self.assertEqual(slot[3] - slot[1], 8)
                payload = torch.arange(8, dtype=torch.uint8) + index * 8
                f.allocator.buffer.buffer[slot[1] : slot[3]].copy_(payload)
                f.handler.handle_chunk_arrived(7, index, start, 2, "peer")
                coverage = prefix_stream.update(target)
                self.assertFalse(coverage.ready(0))
                for event in f.events:
                    event.complete = True
                coverage = prefix_stream.update(target)
                self.assertEqual(coverage.ready(0), index == 1)
                self.assertFalse(coverage.ready(1))
                before = dict(f.allocator.allocations)
                self.hooks.dsv41_receive.allocate(original, f.manager, msg)
                self.assertEqual(f.allocator.allocations, before)
                for buf, values in ((f.k[0], payload[:4]), (f.v[0], payload[4:])):
                    self.assertTrue(
                        torch.equal(
                            buf[f.rows[0, start : start + 2]], values.reshape(2, 1, 2)
                        )
                    )
                self.assertTrue(torch.all(f.k[1] == 255).item())

    def test_prefix_stream_dsv_native_ring_tracks_disjoint_entry_ranges(self):
        """Physical indexer entry completion requires both cached and tail scatters."""
        import struct

        from sglang_remote_attn import prefix_stream

        f = self._dsv41_receive_fixture()
        r = f.ring
        entry = next(i for i, e in enumerate(f.schema["entries"]) if e[0] == "index")
        target = SimpleNamespace(
            path=Path(self.enterContext(tempfile.TemporaryDirectory())),
            generation="dsv-prefix",
            decode_req=r.dr,
            queue=SimpleNamespace(staging_handler=r.handler),
        )
        r.dr._pre_layerwise_offer = target
        for index, start, pages in ((20, 0, 2), (21, 2, 3)):
            msg = [
                b"STAGING_REQ",
                b"7",
                str(index).encode(),
                str(pages).encode(),
                b"peer:1",
                b"0",
                struct.pack("Q", entry),
                str(start).encode(),
                b"dsv-prefix",
            ]
            f.rx.allocate(None, r.manager, msg)
            slot = r.receiver.chunk_staging_infos[index]
            r.allocator.buffer.buffer[slot[1] : slot[3]].fill_(index)
            r.handler.handle_chunk_arrived(7, index, start, pages, "peer:1")
            self.assertFalse(prefix_stream.update(target).ready(entry))
            for event in r.events:
                event.complete = True
            self.assertEqual(prefix_stream.update(target).ready(entry), index == 21)
            self.assertFalse(r.allocator.allocations)
            stride = r.manager.kv_args.kv_item_lens[entry]
            for dst in (5, 6) if index == 20 else (1, 2, 3):
                self.assertTrue(
                    torch.all(
                        f.destination[entry][dst * stride : (dst + 1) * stride] == index
                    ).item()
                )

    def test_prefix_stream_wire_registration_does_not_wait_for_remote_journal(self):
        """STAGING_REQ may precede its network journal without false completion."""
        for dsv in (False, True):
            f = self._prefix_stream_fixture(dsv=dsv, staged=True)
            req = SimpleNamespace(
                origin_input_ids=f.req.origin_input_ids,
                kv=SimpleNamespace(cache_protected_len=0),
            )
            dr = SimpleNamespace(req=req, _staging_completed_layers=[])
            target = SimpleNamespace(
                path=Path(f.controller.config.directory) / "delayed-mirror",
                generation=f.plan.generation,
                decode_req=dr,
                owner=SimpleNamespace(),
                queue=SimpleNamespace(
                    staging_handler=SimpleNamespace(
                        advance_scatter=lambda dr: None, is_failed=lambda dr: False
                    )
                ),
            )
            dr._pre_layerwise_offer = target
            manager = SimpleNamespace(
                kv_args=f.manager.kv_args,
                _dsv41_layerwise_schema=f.schema if dsv else None,
                _staging_handler=SimpleNamespace(_room_to_decode_req={7: dr}),
            )
            msg = f.requests[0]
            self.assertEqual(len(msg), 9)
            self.assertTrue(f.ps.register_range(manager, msg))
            self.assertTrue(f.ps.register_range(manager, msg))
            resources, lo, n = dr._prefix_range_registry[int(msg[2])]
            dr._staging_completed_layers = [(resources, lo, n)]
            self.assertFalse(
                any(
                    f.ps.update(target).ready(i)
                    for i in range(f.ps.header(target)["resources"])
                )
            )
            changed = list(msg)
            changed[2] = b"999"
            with self.assertRaisesRegex(RuntimeError, "overlapping"):
                f.ps.register_range(manager, changed)
            changed = list(msg)
            changed[8] = b"old-attempt"
            with self.assertRaisesRegex(RuntimeError, "stale"):
                f.ps.register_range(manager, changed)

    def test_prefix_stream_direct_completion_requires_gpu_visibility_and_event(self):
        """TE completion cannot publish Local before D's receive GPU proof."""
        f = self._prefix_stream_fixture()
        while f.wire:
            f.wire.popleft().kwargs["on_complete"]()
        dr = SimpleNamespace(req=copy.copy(f.req))
        dr.req.kv = SimpleNamespace(cache_protected_len=0)
        target = SimpleNamespace(
            path=f.plan.path,
            generation=f.plan.generation,
            decode_req=dr,
            owner=SimpleNamespace(),
            queue=SimpleNamespace(
                scheduler=SimpleNamespace(
                    req_to_token_pool=SimpleNamespace(req_to_token=torch.empty(1))
                )
            ),
        )
        from sglang_remote_attn.pd_window import WindowTrace

        target.pd_trace = WindowTrace({0: (0,), 1: (1,)})
        ready = [False]
        from sglang_remote_attn import remote_data_plane

        with (
            patch.dict(
                sys.modules, {"sglang_remote_attn.remote_data_plane": remote_data_plane}
            ),
            patch.object(
                remote_data_plane,
                "event",
                return_value=SimpleNamespace(query=lambda: ready[0]),
            ),
        ):
            coverage = f.ps.update(target)
            self.assertEqual(coverage.intervals, {})
            self.assertEqual(
                target.pd_trace.receipts.intervals, {0: [(0, 4)], 1: [(0, 4)]}
            )
            self.assertIsNone(target.pd_trace.snapshot()["layers"]["0"]["visible_ms"])
            ready[0] = True
            coverage = f.ps.update(target)
            self.assertEqual(coverage.intervals, {0: [(0, 4)], 1: [(0, 4)]})
        self.assertFalse(coverage.ready(0))

    def test_prefix_stream_abort_and_reset_keep_pages_until_native_io_drains(self):
        """P abort fences queued ranges before native sender.abort/free."""
        f = self._prefix_stream_fixture()
        with self.assertRaisesRegex(RuntimeError, "retire before request reset"):
            self.hooks._reset_retracted(lambda req: self.fail("reset live IO"), f.req)
        error = []
        mutation = []

        def abort():
            try:
                self.hooks.prefill_overlap.before_mutation(
                    lambda s, r: mutation.append("native-abort"), f.scheduler, f.req
                )
            except Exception as exc:
                error.append(exc)

        thread = threading.Thread(target=abort)
        thread.start()
        try:
            self.assertTrue(f.guard.leases)
            while f.wire:
                f.wire.popleft().kwargs["on_complete"]()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(error, [])
            self.assertEqual(mutation, ["native-abort"])
            self.assertFalse(f.guard.leases)
            self.assertIsNone(f.ps.stream_for(f.req))
            f.cache_fixture.release_kv_cache(f.req, f.cache, checkpoint=False)
            f.cache.evict(f.cache_fixture.EvictParams(num_tokens=32))
            self.assertEqual(f.cache_fixture.allocator.available_size(), 32)
        finally:
            thread.join(2)

    def test_prefix_stream_peer_change_quarantines_inflight_storage(self):
        """Unknown transfer ownership cannot be repaired by recycling cached pages."""
        f = self._prefix_stream_fixture()
        task = f.wire.popleft()
        f.manager.decode_kv_args_table["peer:1"] = copy.copy(f.registration)
        with self.assertRaisesRegex(RuntimeError, "registration changed"):
            task.kwargs["on_complete"]()
        self.assertTrue(f.plan.error and f.guard.leases)
        self.assertIsNotNone(f.plan.pump.active)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            f.plan.source.retire()

    def test_prefix_stream_native_staging_defaults_keep_legacy_wire(self):
        """The core extension must leave ordinary PD's existing frame unchanged."""
        f = self._prefix_stream_fixture(staged=True)
        f.requests.clear()
        f.manager.request_layer_staging(7, slice(0, 1), (((0,), None),))
        self.assertEqual((len(f.requests[0]), f.requests[0][2]), (8, b"0"))

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
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                self._check_decode_overlap_native_loop(with_cache=False, hybrid=hybrid)

    def test_d_cache_native_overlap_restores_pipeline_only_after_cache_commit(self):
        self._check_decode_overlap_native_loop(with_cache=True)

    def _check_decode_overlap_native_loop(self, *, with_cache, hybrid=False):
        from collections import deque

        from sglang_remote_attn.decode_overlap import process_last_batch_early

        native = _source_class(
            "disaggregation/decode.py",
            "SchedulerDisaggregationDecodeMixin",
            {"event_loop_overlap_disagg_decode"},
        )
        native.event_loop_overlap_disagg_decode.__wrapped__.__globals__.update(
            deque=deque,
            envs=SimpleNamespace(
                SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=SimpleNamespace(
                    get=lambda: False
                ),
                SGLANG_ENABLE_NV_MEGA_PDDEC_RUNTIME_OPTIMIZATION=SimpleNamespace(
                    get=lambda: False
                ),
            ),
        )
        predicate = _source_class(
            "managers/scheduler.py", "Scheduler", {"should_process_last_batch_early"}
        ).should_process_last_batch_early
        predicate.__globals__["envs"] = SimpleNamespace(
            SGLANG_DISABLE_OVERLAP_AFTER_PREFILL=SimpleNamespace(get=lambda: False)
        )
        with tempfile.TemporaryDirectory() as directory:
            if with_cache:
                f = self._destination_cache_target(directory)
                controller, target = f.controller, f.target
                freed = []
                target.decode_req.kv_receiver.clear = lambda: freed.append("receiver")
                target.queue.req_to_metadata_buffer_idx_allocator.free = freed.append
            else:
                controller, target, freed = self._overlap_target(directory)
            controller.targets = []
            controller.hybrid_gdn = hybrid
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
                    grammar_needs_sync=lambda: False,
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
                    if with_cache:
                        f.complete()
                    else:
                        target.decode_req.kv_receiver.done = True
                self._retire_step(controller, finalize=False)
                if iteration == len(batches) - 1:
                    scheduler.gracefully_exit = True

            def launch(scheduler, b):
                if with_cache and b.name != "local-0":
                    local = b.name.startswith("local")
                    hit = f.cache.match_prefix(
                        f.MatchPrefixParams(f.RadixKey(array("q", [1, 2, 3, 4])))
                    )
                    self.assertEqual(hit.device_prefix_len, 4 if local else 2)
                    self.assertEqual(f.lease.closed, local)
                    self.assertEqual(
                        controller.decode_overlap.retirement.cache_votes,
                        1 if local else 0,
                    )
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
            scheduler.process_decode_queue = lambda **kw: controller.retire_targets()
            scheduler.increment_scheduler_iterations = lambda: None
            scheduler._flush_staging_scatters = lambda: None
            scheduler.get_next_disagg_decode_batch_to_run = lambda **kw: (
                SimpleNamespace(running_batch=None, batch_to_run=batches[iteration])
            )
            scheduler.should_process_last_batch_early = lambda last_batch: (
                process_last_batch_early(predicate, scheduler, last_batch)
            )
            self.assertFalse(scheduler.should_process_last_batch_early(None))
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
                    ("remote-2", ["remote-1"] if hybrid else []),
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
                        process_last_batch_early(predicate, scheduler, batches[5])
                    )
                    scheduler.run_batch(batches[4])
            if with_cache:
                f.release_kv_cache(f.req, f.cache, checkpoint=True)
                f.cache.evict(f.EvictParams(num_tokens=32))
                self.assertEqual(f.allocator.available_size(), 32)
                self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)

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
            scheduler.tree_cache = f.cache
            scheduler.waiting_queue = _request_list()
            scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
                resume_retracted_reqs=lambda: [],
                retracted_queue=[object()],
                publish_pending_destinations=lambda: None,
            )
            queue = SimpleNamespace(scheduler=scheduler)
            scheduler.disagg_decode_transfer_queue = SimpleNamespace(
                resolve_deferred_releases=lambda: reap(lambda q: None, queue),
                publish_pending_metadata=lambda: None,
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
        retirer_cls = _source_class(
            "managers/scheduler_components/request_retirer.py",
            "RequestRetirer",
            {"retract"},
        )
        retirer = retirer_cls()
        retirer.tree_cache, retirer.hisparse_coordinator = f.cache, None
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
            retirer_cls.retract.__globals__.update(native_release.__globals__)
            with patch.object(
                self.hooks.pre_layerwise.time, "sleep", side_effect=device_progress
            ):
                self.assertTrue(
                    before_retract(
                        native_release,
                        req=req,
                        remaing_req_count=1,
                        tree_cache=f.cache,
                        request_retirer=retirer,
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

    def _segmented_graph_fixture(self, sizes=(32, 16, 8, 4, 2, 1)):
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
            capture_bs=sorted(sizes),
            attn_backend=SimpleNamespace(
                init_forward_metadata=lambda fb: trace.append(
                    ("metadata", fb.batch_size)
                )
            ),
        )

        class Bank:
            def __init__(self, name):
                self.name, self.keys, self.outputs, self.capturing = name, {}, {}, False

            def capture_session(self, stream):
                return nullcontext()

            def capture_one(self, key, fn, *a):
                fn()
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
                    q = torch.ones(key.size, 4)
                    for module in modules:
                        q = backend._attention(module, q, q, q, save_kv_cache=True)
                    self.outputs[key.size].copy_(q)
                return self.outputs[key.size]

            def cleanup(self):
                trace.append("clean_" + self.name)

        full, remote = Bank("full"), Bank("remote")

        def eager(fn, *, on_capture=None, output_factory=None):
            def wrapped(*a, **kw):
                if remote.capturing:
                    if on_capture is not None:
                        on_capture(*a, **kw)
                    return output_factory(*a, **kw) if output_factory else fn(*a, **kw)
                return fn(*a, **kw)

            return wrapped

        backend = sg.RemoteGraphBackend(
            runner,
            full,
            remote,
            eager,
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
                forward_mode=SimpleNamespace(
                    is_decode=lambda: True, is_idle=lambda: False
                ),
                needs_forward_metadata_init=lambda: True,
                marker=marker,
            )

        def attention(module, q, k, v, live, **kwargs):
            self.assertEqual(q.shape[0], live.batch_size)
            self.assertEqual(k.shape[0], live.batch_size)
            self.assertEqual(v.shape[0], live.batch_size)
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
            for size in sizes:
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

    def test_remote_segments_capture_five_buckets_and_use_live_requests(self):
        """Captured dummy requests must never leak into a later RPC callback."""
        f = self._segmented_graph_fixture()
        b = f.backend
        self.assertEqual(sorted(f.full.keys), [1, 2, 4, 8, 16, 32])
        self.assertEqual(sorted(f.remote.keys), [1, 2, 4, 8, 16])
        self.assertEqual(b.manifest(), tuple((n, (0, 1)) for n in (1, 2, 4, 8, 16)))
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
            for size in (1, 2, 4, 8, 16):
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
            for size in (17, 32):
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
            for size in (16, 8, 4, 2, 1, 32):
                live = f.batch(size, 999)
                self.assertTrue(
                    self.hooks._decode_graph(lambda *a: True, f.runner, live)
                )
                with b.execution(live), b.replay_session():
                    output = b.replay(f.full.keys[size], live)
                self.assertTrue(torch.equal(output, torch.full((size, 4), 4.0)))
            self.assertEqual(
                len(f.trace), before
            )  # No RPC or event on Local full replay.
        b.cleanup()
        self.assertEqual(f.trace[-3:], ["sync", "clean_remote", "clean_full"])
        self.assertEqual(b.replays, {"remote": 10, "full": 6})
        # A small native deployment keeps its own capture range; adding 8/16
        # support must not fabricate larger graphs or change its full bank.
        small = self._segmented_graph_fixture(sizes=(4, 2, 1))
        self.assertEqual(sorted(small.remote.keys), [1, 2, 4])
        self.assertEqual(sorted(small.full.keys), [1, 2, 4])
        self.assertFalse(small.backend.supports_remote(small.batch(8)))
        small.backend.cleanup()

    def test_remote_segments_pad_live_rows_and_restore_native_buckets(self):
        """RPC/KV sees real rows; every subsequent segment sees the full bridge."""
        f = self._segmented_graph_fixture(sizes=(32, 16, 14, 8, 6, 4, 3, 2, 1))
        b = f.backend
        f.controller.targets = [object()]
        native = f.runner.capture_bs
        with patch.object(
            self.hooks.segmented_graph, "_runtime", return_value=f.runtime
        ):
            for size in (13, 3, 15, 5, 7, 14, 6, 1, 16, 3):
                live = f.batch(size, size)
                key = b.remote_key(live)
                self.assertTrue(b.supports_remote(live))
                self.assertTrue(b.can_run(live, key))
                with b.execution(live), b.replay_session():
                    self.assertEqual(f.runner.capture_bs, [1, 2, 4, 8, 16])
                    output = b.replay(key, live)
                self.assertIs(f.runner.capture_bs, native)
                self.assertIsNone(b.active_key)
                torch.testing.assert_close(
                    output[:size], torch.full((size, 4), float(1 + 2 * size))
                )
                self.assertEqual(torch.count_nonzero(output[size:]).item(), 0)
            # Replaying a full-bank key (e.g. native 14) is forbidden when
            # remote eligibility selected 16; no graph/RPC may start.
            live = f.batch(13)
            with b.execution(live):
                with self.assertRaisesRegex(RuntimeError, "mismatched key"):
                    b.replay(f.full.keys[14], live)
            self.assertIs(f.runner.capture_bs, native)
            f.runner.disable_padding = True
            self.assertFalse(b.supports_remote(f.batch(3)))
            self.assertTrue(b.supports_remote(f.batch(4)))
            f.runner.disable_padding = False
            f.controller.targets.clear()
            with b.execution(f.batch(3)):
                self.assertIs(f.runner.capture_bs, native)
        b.cleanup()

    def test_remote_segments_startup_preserves_small_buckets_and_agrees_policy(self):
        """New buckets are optional; mixed plugin policies must fail before capture."""
        sg = self.hooks.segmented_graph
        f = self._segmented_graph_fixture(sizes=(4, 2, 1))
        self.addCleanup(f.backend.cleanup)
        f.runner.capture_bs = (1, 2, 4)
        f.runner.enable_torch_compile = False
        f.runner.device_module.graph_pool_handle = lambda: None
        f.runner.model_runner.model = type(
            "Qwen3ForCausalLM", (), {"modules": lambda self: []}
        )()
        f.runner.model_runner.model_config.hf_config = SimpleNamespace(
            model_type="qwen3"
        )
        f.runner.model_runner.lora_manager = None
        f.controller.sources = []
        config = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(backend="full")
                ),
                debug_cuda_graph=False,
            ),
            features=SimpleNamespace(enable_memory_saver=False),
        )
        context = SimpleNamespace(
            get_exec=lambda: config,
            get_disagg=lambda: SimpleNamespace(disaggregation_mode="decode"),
            get_parallel=lambda: SimpleNamespace(
                tp_size=2, tp_group=SimpleNamespace(cpu_group=None)
            ),
        )
        modules = {
            "sglang.srt.runtime_context": context,
            "sglang.srt.layers.radix_attention": SimpleNamespace(
                RadixAttention=type("RadixAttention", (), {})
            ),
            "sglang.srt.distributed.device_communicators.pynccl_allocator": SimpleNamespace(
                set_graph_pool_id=lambda *a: None
            ),
            "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend": SimpleNamespace(
                FullCudaGraphBackend=type(f.full)
            ),
            "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend": SimpleNamespace(
                BreakableCudaGraphBackend=lambda runner, **kw: SimpleNamespace(**kw)
            ),
            "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph": SimpleNamespace(
                eager_on_graph=lambda fn, **kw: fn
            ),
            "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph": SimpleNamespace(
                is_capturing=lambda: False
            ),
        }
        mismatch = [False]
        allocations = []

        def gather(reports, report, group):
            reports[0] = report
            reports[1] = tuple(
                (1, 2, 4) if mismatch[0] and value == (1, 2, 4, 8, 16) else value
                for value in report
            )

        def original(runner):
            allocations.append(True)
            return f.full

        with (
            patch.dict(sys.modules, modules),
            patch.dict(
                os.environ,
                {
                    "HANDOFF_PRE_LAYERWISE": "remote",
                    "HANDOFF_REMOTE_SEGMENTED_GRAPH": "1",
                },
            ),
            patch.object(
                sg,
                "_runtime",
                return_value=SimpleNamespace(has_target_states=lambda: False),
            ),
            patch.object(sg.importlib.util, "find_spec", return_value=object()),
            patch.object(torch.distributed, "all_gather_object", side_effect=gather),
        ):
            backend = sg.resolve_backend(original, f.runner)
            self.assertIsInstance(backend, sg.RemoteGraphBackend)
            self.assertIs(backend.full, f.full)
            self.assertEqual(backend.remote.capture_error_mode, "thread_local")
            self.assertTrue(backend.remote.preallocate_workspaces)
            f.runner.capture_bs = (1, 4, 8, 16)
            with self.assertRaisesRegex(ValueError, "must include 1, 2, 4"):
                sg.resolve_backend(original, f.runner)
            f.runner.capture_bs = (1, 2, 4)
            before = len(allocations)
            mismatch[0] = True
            with self.assertRaisesRegex(RuntimeError, "settings differ"):
                sg.resolve_backend(original, f.runner)
            self.assertEqual(len(allocations), before)

    def test_padded_remote_lease_rebuilds_metadata_for_wholly_local_layers(self):
        f = self._segmented_graph_fixture()
        b = f.backend
        self.addCleanup(b.cleanup)
        f.controller.targets = [object()]
        attn = f.runner.attn_backend
        attn.current = f.batch(4)  # Native staging prepared bucket-sized metadata.
        attn.init_forward_metadata = lambda fb: setattr(attn, "current", fb)

        def local(module, q, k, v, live, **kw):
            self.assertIs(attn.current, live)
            self.assertEqual(q.shape[0], attn.current.batch_size)
            return module._handoff_orig_forward(q, k, v, live, **kw)

        f.runtime._handoff_attention_forward = local
        live = f.batch(3)
        with patch.object(
            self.hooks.segmented_graph, "_runtime", return_value=f.runtime
        ):
            with b.execution(live), b.replay_session():
                output = b.replay(b.remote_key(live), live)
        torch.testing.assert_close(output[:3], torch.full((3, 4), 4.0))
        self.assertEqual(torch.count_nonzero(output[3:]).item(), 0)

    def test_remote_graph_stats_count_execution_not_eligibility(self):
        import bisect

        f = self._segmented_graph_fixture(sizes=(16, 14, 8, 6, 4, 3, 2, 1))
        b = f.backend
        self.addCleanup(b.cleanup)
        f.controller.targets = [object()]
        f.runtime.graph_step_begin = lambda fb: None
        pad = _source_class(
            "model_executor/runner/base_cuda_graph_runner.py",
            "BaseCudaGraphRunner",
            {"_pad_to_bucket"},
        )._pad_to_bucket
        pad.__globals__["bisect"] = bisect
        native = _source_class(
            "model_executor/runner/decode_cuda_graph_runner.py",
            "DecodeCudaGraphRunner",
            {
                "load_batch",
                "_with_logits_variant",
                "_resolve_logits_variant",
                "_replay_pads_rows",
            },
        )
        staged = []
        runner = f.runner
        runner.__dict__.update(
            dllm_uses_input_embeds=False,
            ragged_verify_mode=False,
            deepep_adapter=SimpleNamespace(replay=lambda: None),
            buffers=SimpleNamespace(),
            _validate_capture_hidden_mode=lambda fb: None,
            captured_req_width=1,
            require_mlp_tp_gather=False,
            _pad_to_bucket=pad,
            _capture_graph_size=lambda bs, num_tokens: bs,
            _resolve_attention_variant=lambda *a: None,
            _glue_key=lambda *a: None,
            _take_early_verify_metadata=lambda *a, **kw: False,
            buffer_registry=SimpleNamespace(
                fill_from=lambda fb, **kw: staged.append(kw)
            ),
            _skips_prefilled_inputs=False,
            _restage_skip=False,
            _verify_logits_processor=None,
            _with_logits_variant=native._with_logits_variant.__get__(runner),
            _resolve_logits_variant=native._resolve_logits_variant.__get__(runner),
            _replay_pads_rows=native._replay_pads_rows.__get__(runner),
            _set_dp_real_num_tokens=lambda n: None,
            enable_two_batch_overlap=False,
            _init_replay_attn_metadata=lambda *a, **kw: None,
            _draft_metadata_glue=False,
            _target_metadata_glue=False,
            _resolve_lora_variant=lambda fb: None,
            _early_verify_metadata=None,
            _make_graph_key=lambda size, *a: f.full.keys[size],
        )
        runner.model_runner.spec_algorithm = SimpleNamespace(
            is_dflash_family=lambda: False
        )
        runner.model_runner.hisparse_coordinator = None
        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(tp_rank=1, tp_size=2)
        )

        def execute(runner, live):
            native.load_batch(runner, live)
            with b.replay_session():
                return b.replay(runner._replay_graph_key, live)

        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": context}),
            patch.object(self.hooks, "_runtime", return_value=f.runtime),
            patch.object(
                self.hooks.segmented_graph, "_runtime", return_value=f.runtime
            ),
        ):
            live = f.batch(13, 3)
            for _ in range(20):
                self.assertTrue(
                    self.hooks._decode_graph(lambda *a: True, f.runner, live)
                )
            self.assertFalse(hasattr(f.runner.model_runner, "_remote_attn_graph_stats"))
            with self.assertLogs(
                self.hooks.segmented_graph.logger, level="INFO"
            ) as logs:
                from sglang.srt.environ import envs

                with envs.SGLANG_LOG_DECODE_GRAPH_KEY.override(True):
                    for _ in range(3):
                        self.hooks._graph_execute(execute, f.runner, live)
            self.assertEqual(
                sum("[remote-attn-graph-step]" in line for line in logs.output), 3
            )
            self.assertFalse(
                self.hooks._decode_graph(lambda *a: True, f.runner, f.batch(17))
            )
            eager = f.batch(17)
            self.hooks._decode_graph(lambda *a: True, f.runner, eager)
            for _ in range(2):
                self.hooks._eager_execute(lambda *a: "ok", f.runner, eager)
            f.controller.targets.clear()
            self.hooks._graph_execute(execute, f.runner, f.batch(3))

            # Failed executions are reported separately and must not inflate
            # successful step percentages or silently restart on eager.
            def fail(*args):
                raise RuntimeError("forward failed")

            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                self.hooks._eager_execute(fail, f.runner, eager)
        stats = f.runner.model_runner._remote_attn_graph_stats
        self.assertEqual(stats["rank"], 1)
        self.assertEqual(stats["steps"], 6)
        self.assertEqual(stats["paths"], {"remote-segmented": 3, "eager": 2, "full": 1})
        self.assertEqual(stats["batches"]["remote-segmented:13:16"], 3)
        self.assertEqual(stats["batches"]["full:3:3"], 1)
        self.assertEqual(stats["fallbacks"], {"remote_bucket_unavailable": 2})
        self.assertEqual(stats["failures"], {"eager": 1})
        self.assertEqual([item["padded_bs"] for item in staged], [16, 16, 16, 3])
        self.assertEqual([item["raw_bs"] for item in staged], [13, 13, 13, 3])

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
            for mode in ("remote", "layerwise"):
                with patch.dict(os.environ, {"HANDOFF_PRE_LAYERWISE": mode}):
                    disagg.disaggregation_mode = "decode"
                    disagg.disaggregation_decode_enable_radix_cache = True
                    self.hooks.validate_config()
            disagg.disaggregation_decode_enable_radix_cache = False
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
            if payload["cmd"] == "plan":
                return {
                    "plans": [
                        dict(rid=x["rid"], target_rid=x["target_rid"], upper=32895)
                        for x in payload["items"]
                    ]
                }
            if payload["cmd"] == "reservation_ready":
                return {"confirmed": True}
            raise AssertionError(f"unexpected source RPC {payload}")

        def target_send(payload, **kwargs):
            calls.append(("dst", copy.deepcopy(payload)))
            if payload["cmd"] == "status":
                return {"reqs": [], "max_running": 8}
            if payload["cmd"] == "reserve":
                return {
                    "errors": {},
                    "reservations": {x["target_rid"]: x for x in payload["plans"][0]},
                }
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
                self.assertFalse(
                    any(role == "src" and p["cmd"] == "submit" for role, p in calls)
                )
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
        self.assertFalse(
            any(role == "src" and p["cmd"] == "submit" for role, p in calls)
        )
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
            if payload["cmd"] == "reserve" and "plans" in payload:
                certs = {x["target_rid"]: x for x in payload["plans"][0]}
                payload = {
                    "cmd": "reserve",
                    "items": payload["items"],
                    "epoch": payload["epoch"],
                }
                requests.append(payload)
                return dict(
                    self.runtime._handle(scheduler, payload), reservations=certs
                )
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
        gateway.src = SimpleNamespace(
            send=lambda p, **kw: (
                {
                    "plans": [
                        dict(rid=x["rid"], target_rid=x["target_rid"], upper=5)
                        for x in p["items"]
                    ]
                }
                if p["cmd"] == "plan"
                else {"confirmed": True}
            )
        )
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
        self.assertEqual(rows[0]["segmented_startup"], [True, False, False, False])
        self.assertEqual(rows[0]["segmented_startup"], rows[1]["segmented_startup"])
        self.assertEqual(rows[0]["segmented_routing"], rows[1]["segmented_routing"])
        self.assertEqual(
            rows[0]["segmented_routing"],
            [[True] * 13, [True] * 12 + [False]] * 2,
        )
        self.assertEqual(rows[0]["segmented_buckets"], rows[1]["segmented_buckets"])
        self.assertEqual(
            rows[0]["segmented_buckets"],
            [[1, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, None]] * 4,
        )
        self.assertTrue(rows[0]["leases"]["mismatch_rejected"])
        self.assertTrue(rows[0]["leases"]["sparse_mode_mismatch_rejected"])
        self.assertEqual(rows[0]["leases"]["mesh_settings_rejected"], [True] * 4)
        self.assertEqual(rows[0]["leases"]["fast_settings_rejected"], [True, True])
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
                settings=[True] + [False] * 7,
                policies=[[True] * 6 + [False]] * 3 + [[True] * 7],
                per_decode_votes=0,
            ),
        )
        self.assertEqual(rows[0]["dsv41_segments"], rows[1]["dsv41_segments"])
        self.assertEqual(rows[0]["d_cache"], rows[1]["d_cache"])
        self.assertEqual(
            rows[0]["d_cache"],
            dict(
                match_rejections=[True, True],
                final_vote_shapes=[5, 5, 5, 1, 5, 1],
                released=1,
                published=0,
                local_votes=0,
                partial_apply_failure_stopped=True,
            ),
        )
        self.assertEqual(rows[0]["sparse_progressive"], rows[1]["sparse_progressive"])
        self.assertEqual(rows[0]["prefix_ranges"], rows[1]["prefix_ranges"])
        self.assertEqual(
            rows[0]["prefix_ranges"],
            [dict(votes=[5, 5, 5], local=[0, 1], closed=True, steady_votes=0)] * 2,
        )
        self.assertEqual(
            rows[0]["sparse_cached_progressive"], rows[1]["sparse_cached_progressive"]
        )
        self.assertEqual(
            rows[0]["sparse_cached_progressive"],
            dict(
                votes=[5, 5, 5, 5, 1],
                freed=["KV"],
                local_entries=[0, 1],
                retries=1,
                closed=1,
                stable_votes=0,
                cache_votes=1,
                pins_closed=True,
                published=[],
            ),
        )
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
        self.assertEqual(rows[0]["p_swa_quarantine"], rows[1]["p_swa_quarantine"])
        self.assertEqual(
            rows[0]["p_swa_quarantine"],
            dict(
                rank_states_agree=True,
                healthy_fully_reclaimed=True,
                quarantine_full_pages=32,
                quarantine_swa_pages=2,
            ),
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

    def _fast_sender_contract(
        self, failure=None, group=1, slots=1, reuse=False, async_=True
    ):
        device = threading.local()
        device.value = 1
        tensors = {}
        copied = []
        completed = []
        controls = []
        messages = queue.Queue()
        messages.put({"token": "regression"})
        messages.put(
            {
                "session": "peer",
                "addrs": list(range(100, 100 + 2 * slots)),
                "slots": slots,
            }
        )

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

        batches = []

        class Engine:
            def __init__(self):
                self.next = 0
                self.pending = {}
                self.available = []

            def batch_transfer_async_write(self, session, addresses, peer, sizes):
                if failure == "async":
                    raise RuntimeError("submission completion unknown")
                self.next += 1
                bid = self.available.pop() if reuse and self.available else self.next
                batches.append(len(addresses))
                self.pending[bid] = [
                    (tensors[a], tensors[a].clone()) for a in addresses
                ]
                return bid

            def batch_transfer_sync_write(self, session, addresses, peer, sizes):
                if failure == "sync":
                    return -1
                batches.append(len(addresses))
                return 0

            def get_batch_transfer_status(self, batches):
                for batch in batches:
                    for tensor, snapshot in self.pending.pop(batch):
                        if not torch.equal(tensor, snapshot):
                            raise AssertionError(
                                "source staging reused before TE completion"
                            )
                    completed.append(batch)
                    self.available.append(batch)
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
            # Frozen metadata wins even if the environment changes afterward.
            stack.enter_context(
                patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="invalid")
            )
            for name, value in {
                "FAST_BUFFERS": 3,
                "FAST_COPY_STREAMS": 2,
                "FAST_ASYNC": async_ and failure != "sync",
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
                scheduler,
                req,
                {"kv_committed_len": 3, "chunks": [[3]] * 5, "d2d_group_size": group},
                False,
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
            width = min(group, slots, max(3, group), 5)
            expected = [2 * min(width, 5 - start) for start in range(0, 5, width)]
            self.assertEqual(batches, expected)
            if async_:
                self.assertEqual(len(completed), len(expected))
            else:
                self.assertFalse(completed)
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

    def test_remote_metadata_bypasses_bulk_lock_and_admits_before_kv_success(self):
        from sglang_remote_attn.transfer_priority import TransferLock
        from sglang.srt.environ import envs

        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            lock = TransferLock()
            bulk_started, drain = threading.Event(), threading.Event()
            calls, frees = [], []
            room = torch.zeros((1, 1), dtype=torch.int64)
            manager = SimpleNamespace(_remote_attn_send_lock=lock)

            def aux(peer, idx, pointers, item_lens):
                def write(*args):
                    calls.append("metadata")
                    room[0, 0] = 7
                    return 0

                return self.hooks._transfer(write, manager, "peer", [(1, 2, 8)])

            manager.send_aux = aux
            sender = SimpleNamespace(
                kv_mgr=manager,
                aux_index=0,
                done=False,
                send=lambda indices: calls.append("canonical-enqueued"),
            )
            req = SimpleNamespace(
                rid="p", kv=SimpleNamespace(kv_committed_len=3), disagg_kv_sender=sender
            )
            owner = SimpleNamespace(
                config=pl.Config("remote", path),
                scheduler=SimpleNamespace(
                    disagg_metadata_buffers=SimpleNamespace(set_buf=lambda req: None)
                ),
            )
            source = pl.Source(
                owner,
                req,
                path,
                "g",
                [],
                slice(0, 3),
                ((object(), SimpleNamespace(dst_aux_ptrs=[2], dst_aux_item_lens=[8])),),
                3,
            )
            req._pre_layerwise_source = source
            registry = SimpleNamespace(
                register_handoff=lambda *a, **kw: {"handoff_id": "h", "epoch": 1}
            )

            def bulk():
                with lock:
                    bulk_started.set()
                    if not drain.wait(5):
                        raise TimeoutError("bulk test gate")
                    calls.append("kv-complete")

            def queued(which):
                deadline = time.monotonic() + 3
                with lock.condition:
                    while not getattr(lock, which):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"{which}: {source.error}")
                        lock.condition.wait(min(remaining, 0.01))

            event = SimpleNamespace(
                record=lambda stream: None, synchronize=lambda: None
            )
            lock.acquire()
            worker = threading.Thread(target=bulk)
            worker.start()
            with (
                envs.SGLANG_MOONCAKE_SEND_LOCK.override(True),
                patch.object(torch.cuda, "Event", lambda: event),
                patch.object(torch.cuda, "current_stream", lambda: None),
                patch.object(torch.cuda, "current_device", lambda: 0),
                patch.object(torch.cuda, "set_device", lambda device: None),
                patch.object(self.runtime, "get_remote_hub", return_value=registry),
                patch.object(
                    pl, "_poll_success", side_effect=lambda receiver: receiver.done
                ),
            ):
                try:
                    queued("bulk")
                    source.finalize()
                    queued("urgent")
                    lock.release()
                    source.thread.join(3)
                    self.assertFalse(source.thread.is_alive())
                    self.assertIsNone(source.error)
                    self.assertTrue(bulk_started.wait(3))
                    self.assertEqual(calls, ["metadata", "canonical-enqueued"])
                    self.assertTrue(pl.defer_release(req, lambda: frees.append("KV")))
                    self.assertTrue(
                        pl.defer_release(req, lambda: frees.append("duplicate"))
                    )
                    self.assertFalse(source.poll())
                    self.assertFalse(frees)

                    d = pl.Controller.__new__(pl.Controller)
                    d.config, d.targets, d.early_admissions = owner.config, [], 0
                    d.agree = lambda flags: flags
                    d.layers, d.rank, d.scheduler = 2, 0, object()
                    dreq = SimpleNamespace(
                        rid="d",
                        bootstrap_room=7,
                        kv=SimpleNamespace(req_pool_idx=0),
                        output_ids=[],
                        finished=lambda: False,
                    )
                    receiver = SimpleNamespace(done=False)
                    dr = SimpleNamespace(
                        req=dreq, kv_receiver=receiver, metadata_buffer_index=0
                    )
                    commits = []
                    queue = SimpleNamespace(
                        queue=[dr],
                        metadata_buffers=SimpleNamespace(bootstrap_room=room),
                        _commit_transfer_to_req=lambda dr, **kw: commits.append(kw),
                    )
                    target = pl.Target(d, queue, dr, path, "g", pl.LayerCoverage(3, 2))
                    d.offer = lambda queue, dr: target

                    def state(*args, **kw):
                        return SimpleNamespace(
                            events=args[2],
                            layer_ready_at=[None] * 2,
                            import_done_at=None,
                        )

                    with (
                        patch.object(
                            self.runtime, "_TargetHandoffState", side_effect=state
                        ),
                        patch.object(self.runtime, "_install_attention_wrappers"),
                    ):
                        admitted, _ = d.early_targets(queue, None)
                    self.assertEqual(admitted, [dreq])
                    self.assertEqual(commits, [{"keep_receiver": True}])
                    self.assertIs(dr.kv_receiver, receiver)
                    self.assertFalse(receiver.done)
                    self.assertNotIn("kv-complete", calls)
                finally:
                    if lock.busy and not bulk_started.is_set():
                        lock.release()
                    drain.set()
                    worker.join(3)
                    if source.thread is not None:
                        source.thread.join(3)
            self.assertEqual(calls[-1], "kv-complete")
            self.assertFalse(frees)

    def test_source_retirement_failure_after_free_is_never_retried(self):
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            freed = []
            req = SimpleNamespace()
            source = pl.Source(
                SimpleNamespace(config=pl.Config("remote", path)),
                req,
                path,
                "g",
                [],
                slice(0, 1),
                (),
                1,
            )
            source.release = lambda: freed.append("KV")
            with patch(
                "sglang_remote_attn.source_resources.retire_metadata",
                side_effect=RuntimeError("metadata allocator failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "allocator failed"):
                    source.retire()
                with self.assertRaisesRegex(RuntimeError, "refusing release retry"):
                    source.retire()
            self.assertEqual(freed, ["KV"])
            self.assertFalse(source.released)

    def test_target_metadata_partial_free_blocks_retirement_retry(self):
        pl = self.hooks.pre_layerwise
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            freed = []

            def free(idx):
                freed.append(idx)
                raise RuntimeError("partial allocator mutation")

            receiver = SimpleNamespace(clear=lambda: freed.append("receiver"))
            req = SimpleNamespace()
            dr = SimpleNamespace(req=req, kv_receiver=receiver, metadata_buffer_index=0)
            queue = SimpleNamespace(
                metadata_buffers=SimpleNamespace(bootstrap_room=torch.tensor([[7]])),
                req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=free),
            )
            target = pl.Target(
                SimpleNamespace(config=pl.Config("remote", path)),
                queue,
                dr,
                path,
                "g",
                pl.LayerCoverage(1, 1),
            )
            with (
                patch.object(target, "ready_to_finish", return_value=True),
                patch.object(target, "refresh"),
                patch.object(self.runtime, "_drop_target_state_for"),
            ):
                with self.assertRaisesRegex(RuntimeError, "allocator mutation"):
                    target.finish(wait=False)
                with self.assertRaisesRegex(RuntimeError, "refusing retirement retry"):
                    target.finish(wait=False)
            self.assertEqual(freed, ["receiver", 0])
            self.assertFalse(target.closed)

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
        manager_cls.customized_info_enabled = False
        manager_cls.validate_aux_buffers = _source_class(
            "disaggregation/common/conn.py", "CommonKVManager", {"validate_aux_buffers"}
        ).validate_aux_buffers
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
            "transport_error",
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
                manager._poison_unsettled_send = lambda room: events.append("uncertain")
                manager.decode_kv_args_table = {
                    session: SimpleNamespace(
                        requires_dcp_relayout=False,
                        dst_attn_tp_size=1,
                        dst_kv_ptrs=[30, 40],
                        dst_aux_ptrs=[],
                        dst_aux_item_lens=None,
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

                dropped = []

                def send(session, *args, **kwargs):
                    if outcome == "transport_error" and not dropped:
                        from sglang.srt.disaggregation.pinned_staging_buffer import (
                            StagingTransportError,
                        )

                        dropped.append(True)
                        raise StagingTransportError("injected TCP failure")
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
                if outcome == "transport_error":
                    manager.request_status[8] = polls.Transferring
                    manager.transfer_infos[8] = manager.transfer_infos[7]
                    healthy = copy.copy(chunk)
                    healthy.room, healthy.staging_counted = 8, False
                    healthy.on_complete = lambda: events.append("healthy-proof")
                    tasks.put(healthy)
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
                if outcome == "transport_error":
                    self.assertEqual(manager.request_status[7], polls.Failed)
                    self.assertEqual(manager.request_status[8], polls.Transferring)
                    self.assertIn("healthy-proof", events)
                    self.assertEqual(manager._staging_outstanding[7], 0)
                    self.assertTrue(tasks.empty())

    def test_staging_recovery_retains_failed_slots_and_rejects_exhausted_budget(self):
        from sglang.srt.disaggregation import pinned_staging_buffer as pinned

        spec = importlib.util.spec_from_file_location(
            "recovery_core",
            ROOT / "python/sglang/srt/disaggregation/mooncake/pinned_staging_core.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manager = _host_staging_manager()
        stager = module.CorePinnedStager(manager.engine, capacity=8)
        stager.resolve = lambda *a: SimpleNamespace(device=torch.device("cuda:0"))
        source, destination = (
            ctypes.create_string_buffer(4),
            ctypes.create_string_buffer(4),
        )
        blocks = [(ctypes.addressof(source), ctypes.addressof(destination), 4)]
        host = ctypes.create_string_buffer(b"H")
        mixed = [
            (ctypes.addressof(host), ctypes.addressof(destination), 1),
            (ctypes.addressof(source), ctypes.addressof(destination) + 1, 3),
        ]
        retired, calls = [], []

        def healthy(plan):
            calls.append(list(plan))
            for src, dst, n in plan:
                ctypes.memmove(dst, src, n)
            return 0

        factory = functools.partial(
            pinned.PinnedKVStager, slot_factory=_HostStagingSlot
        )
        with patch.object(module, "PinnedKVStager", factory):
            for index in range(8):
                source.raw = index.to_bytes(4, "little")
                attempts = []
                self.assertEqual(
                    stager.send(blocks, lambda plan: attempts.append(list(plan)) or -1),
                    -1,
                )
                self.assertEqual(len(attempts), 1)  # Unknown writes are never retried.
                old = stager.buffer.slot
                retired.append((old, source.raw))
                source.raw = b"next"
                if index == 7:
                    stager.resolve = lambda addr, n: (
                        None
                        if addr == ctypes.addressof(host)
                        else SimpleNamespace(device=torch.device("cuda:0"))
                    )
                if index == 7:
                    with self.assertRaisesRegex(
                        pinned.StagingTransportError, "budget exhausted"
                    ):
                        stager.send(mixed, healthy)
                else:
                    self.assertEqual(stager.send(blocks, healthy), 0)
                    self.assertEqual(destination.raw, b"next")
                for slot, content in retired:
                    self.assertFalse(slot.closed)
                    self.assertEqual(ctypes.string_at(slot.ptr, 4), content)
            self.assertTrue(stager.disabled)
            self.assertEqual(len(stager.retired), 8)
            self.assertEqual(len(manager.engine._pre_layerwise_staging_owners), 8)
            self.assertEqual(len(manager.registrations), 8)
            self.assertEqual(len(calls), 7)  # No host prefix or GPU-pointer fallback.
            with self.assertRaisesRegex(
                pinned.StagingTransportError, "budget exhausted"
            ):
                stager.send(blocks, healthy)
            self.assertEqual(len(calls), 7)
            self.assertEqual(len(manager.engine._pre_layerwise_staging_owners), 8)

    def test_tcp_descriptor_admission_is_bounded_and_failure_stops_tail(self):
        cls = _source_class(
            "disaggregation/mooncake/conn.py", "MooncakeKVManager", {"_transfer_data"}
        )
        manager = cls()
        manager._core_stager, manager._host_staging_all = None, False
        plans = []
        manager.engine = SimpleNamespace(
            batch_transfer_sync=lambda peer, src, dst, n: (
                plans.append(list(zip(src, dst, n))) or 0
            )
        )
        blocks = [(i * 8, 1000 + i * 8, 8) for i in range(11)]
        with patch.dict(
            os.environ,
            MC_FORCE_TCP="1",
            MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER="8",
            MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER="16",
        ):
            self.assertEqual(manager._transfer_data("peer", blocks), 0)
            self.assertEqual([len(p) for p in plans], [4, 4, 3])
            self.assertEqual(sum(plans, []), blocks)
            plans.clear()
            manager.engine.batch_transfer_sync = lambda peer, src, dst, n: (
                plans.append(list(zip(src, dst, n))) or (-1 if len(plans) == 2 else 0)
            )
            self.assertEqual(manager._transfer_data("peer", blocks), -1)
            self.assertEqual([len(p) for p in plans], [4, 4])

    def test_drained_cancel_is_not_classified_as_transport_failure(self):
        from sglang_remote_attn.prefill_overlap import SourceLease

        chunk = SimpleNamespace(
            req=SimpleNamespace(),
            check=lambda: None,
            sender=SimpleNamespace(kv_mgr=SimpleNamespace(), poll=lambda: 0),
        )
        lease = SourceLease(None, chunk, 1, "cancel", cancelling=True)
        lease.ledger.pending.add(1)
        self.assertFalse(lease.transport_ready())
        lease.ledger.pending.clear()
        self.assertTrue(lease.transport_ready())

    @staticmethod
    def _swa_scheduler():
        return SimpleNamespace(
            schedule_stream=SimpleNamespace(wait_stream=lambda stream: None),
            forward_stream=object(),
            device_module=SimpleNamespace(stream=lambda stream: nullcontext()),
        )

    def _native_source_swa(self):
        path = ROOT / "test/registered/unit/plugins/remote_attn_native_cache.py"
        spec = importlib.util.spec_from_file_location("remote_attn_native_cache", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.fixture(ROOT, self)

    def test_source_native_swa_merged_windows_publish_then_return_all_pages(self):
        from sglang_remote_attn.source_cache import CacheGuard

        for finished in (False, True):
            for prompt in (32772, 33028):
                with self.subTest(finished=finished, prompt=prompt):
                    f = self._native_source_swa()
                    guard = CacheGuard(
                        f.cache, allow_swa=True, scheduler=self._swa_scheduler()
                    )
                    req = f.request(prompt, finished=False)
                    lease = guard.hold(req)
                    for end in (8192, 16384, 24576, 32768, prompt):
                        f.extend(req, end)
                        f.cache.checkpoint(req, up_to=end)
                        f.ownership()
                        self.assertEqual(f.cache.tree_core.root_node.children, {})
                    cursor = req.kv.max_evicted_seqlen
                    req.finished = lambda: finished
                    lease.retire(lambda: None)
                    self.assertGreaterEqual(req.kv.max_evicted_seqlen, cursor)
                    f.ownership()
                    f.release(req)
                    identical = f.request(prompt, match=True)
                    self.assertEqual(identical.prefix_len, (prompt - 128) // 256 * 256)
                    f.extend(identical, prompt)
                    f.release(identical, checkpoint=True)
                    ids = list(range(24576)) + list(
                        range(100000, 100000 + prompt - 24576)
                    )
                    branch = f.request(prompt, tokens=ids, match=True)
                    self.assertEqual(branch.prefix_len, 24576)
                    f.extend(branch, prompt)
                    f.release(branch, checkpoint=True)
                    f.evict_all()

    def test_source_native_swa_abort_and_partial_replay_failure(self):
        from sglang_remote_attn.source_cache import CacheGuard

        for abort in (True, False):
            with self.subTest(abort=abort):
                f = self._native_source_swa()
                guard = CacheGuard(
                    f.cache, allow_swa=True, scheduler=self._swa_scheduler()
                )
                req = f.request(finished=False)
                lease = guard.hold(req)
                for end in (8192, 16384, 24576, 32768, 32772):
                    f.extend(req, end)
                    f.cache.checkpoint(req, up_to=end)
                if abort:
                    f.cache.finish(
                        req.cache_request_handle, f.ns["CacheRequestOutcome"].ABORT
                    )
                    lease.retire(lambda: f.release(req, checkpoint=True))
                    f.evict_all()
                else:
                    original = guard.checkpoint

                    def checkpoint(req, **kwargs):
                        if kwargs["up_to"] == 16384:
                            raise RuntimeError("injected second insert failure")
                        return original(req, **kwargs)

                    guard.checkpoint = checkpoint
                    cursor = req.kv.max_evicted_seqlen
                    with self.assertRaisesRegex(RuntimeError, "second insert failure"):
                        lease.retire(lambda: self.fail("released failed replay"))
                    self.assertTrue(lease.failed)
                    self.assertEqual(req.kv.cache_protected_len, 8192)
                    self.assertEqual(req.kv.max_evicted_seqlen, cursor)
                    f.ownership()
                before = (
                    set(f.allocator.full_attn_allocator.live),
                    set(f.allocator.swa_attn_allocator.live),
                )
                with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                    lease.retire(lambda: self.fail("retried release"))
                self.assertEqual(
                    before,
                    (
                        f.allocator.full_attn_allocator.live,
                        f.allocator.swa_attn_allocator.live,
                    ),
                )

    def test_source_native_swa_budget_trim_keeps_unmarked_live_native_tail(self):
        from sglang_remote_attn.source_cache import CacheGuard

        f = self._native_source_swa()
        guard = CacheGuard(f.cache, allow_swa=True, scheduler=self._swa_scheduler())
        # Reserve 33 pages in another owner, leaving one page in the global
        # 34-page budget. Its SWA is physically alive, not synthetic metadata.
        req = f.request(8192, finished=False)
        lease = guard.hold(req)
        other = f.request(8448, finished=False)
        held = guard.hold(other)
        f.extend(other, 8448)
        other.kv.swa_retained_ranges = [(0, 8448)]
        f.extend(req, 8192)
        f.cache.checkpoint(req, up_to=8192)
        self.assertEqual(req.kv.swa_retained_ranges, [(7936, 8192)])
        self.assertEqual(req.kv.max_evicted_seqlen, 7680)
        req.finished = lambda: True
        lease.retire(lambda: f.release(req, checkpoint=True))
        held.abort = True
        held.retire(lambda: f.release(other))
        f.evict_all()

    def test_source_native_swa_dedup_and_locked_full_repair_return_every_page(self):
        from sglang_remote_attn.source_cache import CacheGuard

        f = self._native_source_swa()
        guard = CacheGuard(f.cache, allow_swa=True, scheduler=self._swa_scheduler())

        def run():
            req = f.request(32772, finished=False)
            lease = guard.hold(req)
            for end in (8192, 16384, 24576, 32768, 32772):
                f.extend(req, end)
                f.cache.checkpoint(req, up_to=end)
            req.finished = lambda: True
            lease.retire(lambda: f.release(req, checkpoint=True))
            return req

        cold = run()
        baseline = set(f.allocator.full_attn_allocator.live)
        run()  # A pipelined miss computed a duplicate of an already inserted key.
        self.assertEqual(f.allocator.full_attn_allocator.live, baseline)
        lock = f.cache.lock(cold.last_node)
        f.cache.evict(f.ns["EvictParams"](swa_num_tokens=35072))
        f.ownership()
        run()  # Repair SWA tombstones without replacing another reader's FULL.
        self.assertEqual(f.allocator.full_attn_allocator.live, baseline)
        f.cache.unlock(lock)
        f.evict_all()

    def test_source_native_swa_explicit_checkpoint_rejects_omitted_live_pages(self):
        from sglang_remote_attn.source_cache import CacheGuard

        f = self._native_source_swa()
        guard = CacheGuard(f.cache, allow_swa=True, scheduler=self._swa_scheduler())
        req = f.request(8192, finished=False)
        lease = guard.hold(req)
        f.extend(req, 8192)
        f.cache.checkpoint(req, up_to=8192)
        before = (
            set(f.allocator.full_attn_allocator.live),
            set(f.allocator.swa_attn_allocator.live),
        )
        with self.assertRaisesRegex(ValueError, "every private live page"):
            guard.checkpoint(req, up_to=8192, swa_checkpoint_start=7936)
        self.assertEqual(
            before,
            (f.allocator.full_attn_allocator.live, f.allocator.swa_attn_allocator.live),
        )
        lease.abort = True
        lease.retire(lambda: f.release(req))
        f.evict_all()

    def test_source_native_swa_rank_local_quarantine_does_not_change_budget_or_tree(
        self,
    ):
        from sglang_remote_attn.source_cache import CacheGuard
        from sglang_remote_attn.transport_failure import retain_release

        states = []
        for rank in (0, 1):
            f = self._native_source_swa()
            guard = CacheGuard(f.cache, allow_swa=True, scheduler=self._swa_scheduler())
            a = f.request(8192, finished=False)
            la = guard.hold(a)
            f.extend(a, 8192)
            f.cache.checkpoint(a, up_to=8192)
            b = f.request(131332, finished=False)
            lb = guard.hold(b)
            for end in range(8192, 131073, 8192):
                f.extend(b, end)
                f.cache.checkpoint(b, up_to=end)
            a_pages = set(f.pool.req_to_token[a.kv.req_pool_idx, :8192].tolist())
            a_mapping = f.allocator.full_to_swa_index_mapping[
                list(sorted(a_pages))
            ].clone()
            if rank == 0:
                retain_release(a, la, lambda: None)
            f.extend(b, 131332)
            f.cache.checkpoint(b, up_to=131332)
            self.assertTrue(
                torch.equal(
                    a_mapping,
                    f.allocator.full_to_swa_index_mapping[list(sorted(a_pages))],
                )
            )
            states.append(
                (
                    list(a.kv.swa_retained_ranges),
                    list(b.kv.swa_retained_ranges),
                    f.allocator.swa_available_size(),
                )
            )
            b.finished = lambda: True
            lb.retire(lambda: f.release(b, checkpoint=True))
            f.ownership()
            tree = sorted(
                (
                    len(n.key),
                    None
                    if n.component_data[1].value is None
                    else len(n.component_data[1].value),
                )
                for n in f.cache.tree_core._node_arena.values()
                if n.parent is not None
            )
            states[-1] += (tree,)
            # A remains quarantined/held; do not invent a completion to release it.
            self.assertFalse(la.closed)
        self.assertEqual(states[0], states[1])

    def _source_swa_windows(self, prompt=32772, capacity=35072):
        """Real core eviction/accounting/row free; observe publication at retirement."""
        from sglang_remote_attn.source_cache import CacheGuard
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        ceil_align = lambda n, p: (n + p - 1) // p * p
        self.Kv.swa_dead_lo.__globals__["ceil_align"] = ceil_align
        self.Kv.swa_live_ranges.__globals__["ComponentType"] = ComponentType
        native_free = _source_function(
            "mem_cache/common.py",
            "free_swa_out_of_window_slots",
            namespace=dict(ComponentType=ComponentType, ceil_align=ceil_align),
        )
        row_free = _source_function("mem_cache/common.py", "free_kv_row_segments")
        footprint = _source_function(
            "managers/scheduler_components/pool_stats_observer.py",
            "kv_private_swa_tokens",
            namespace=dict(ComponentType=ComponentType, ceil_align=ceil_align),
        )
        allocated = ceil_align(prompt, 256)
        rows = torch.arange(allocated * 4).reshape(4, allocated)
        freed, full_freed, published = set(), set(), []

        def free_swa(indices, *, start_pos):
            self.assertEqual(start_pos % 256, 0)
            slots = set(indices.tolist())
            self.assertFalse(slots & freed, "double SWA free")
            freed.update(slots)

        def free_segments(segments, full_only=False):
            self.assertEqual([p for _, p in segments], sorted(p for _, p in segments))
            for indices, start in segments:
                slots = set(indices.tolist())
                self.assertFalse(slots & full_freed, "double FULL free")
                full_freed.update(slots)
                if not full_only:
                    free_swa(indices, start_pos=start)

        allocator = SimpleNamespace(
            page_size=256,
            swa_attn_allocator=SimpleNamespace(size=capacity),
            free_swa_segment=free_swa,
            free_segments=free_segments,
            free_full_segments=lambda segments: free_segments(segments, full_only=True),
        )
        cache = SimpleNamespace(
            page_size=256,
            tree_core=SimpleNamespace(is_eagle=False),
            components={
                ComponentType.SWA: SimpleNamespace(
                    sliding_window_size=128, gates_prefix_match=True
                )
            },
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(req_to_token=rows),
            finish=lambda *a: None,
            reset=lambda: None,
            lock=lambda n: object(),
            unlock=lambda pin: None,
            supports_swa=lambda: True,
        )

        def evict(req, pre_len):
            native_free(
                req,
                pre_len,
                sliding_window_size=128,
                page_size=256,
                req_to_token_pool=cache.req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
            )

        _bind_window_retirement(cache, evict)

        def checkpoint(req, *, up_to, swa_checkpoint_start=None):
            if swa_checkpoint_start is None:
                evict(req, up_to - 1)
            end = up_to // 256 * 256
            start = max(
                req.kv.cache_protected_len,
                req.kv.get_evicted_seqlen(ComponentType.SWA)
                if swa_checkpoint_start is None
                else swa_checkpoint_start,
            )
            live = rows[req.kv.req_pool_idx, start:end]
            self.assertFalse(set(live.tolist()) & freed, "publishing already freed SWA")
            if start < end:
                published.append((start, end))
            req.kv.cache_protected_len = end

        cache.checkpoint = checkpoint
        guard = CacheGuard(cache, allow_swa=True, scheduler=self._swa_scheduler())

        def request(index=0):
            req = SimpleNamespace(
                kv=self.Kv(req_pool_idx=index, kv_allocated_len=0),
                cache_request_handle=("windows", index),
                lock=None,
                last_node=0,
                origin_input_ids=range(prompt),
                finished=lambda: False,
            )
            return req, guard.hold(req)

        def release(req):
            end = ceil_align(req.kv.kv_allocated_len, 256)
            row_free(
                allocator,
                [
                    (
                        rows[req.kv.req_pool_idx, req.kv.cache_protected_len : end],
                        req.kv.cache_protected_len,
                    )
                ],
                swa_evicted_seqlen=req.kv.get_evicted_seqlen(ComponentType.SWA),
                swa_dead_lo=req.kv.swa_dead_lo(256),
                swa_retained_ranges=req.kv.swa_retained_ranges,
            )
            req.kv.mark_kv_released()

        return SimpleNamespace(**locals())

    def _match_swa_windows(self, ranges, *, shared=None, replay=True):
        """Run native key construction, tree walk and FULL/SWA validators."""
        r = self._radix()
        tree = _source_class(
            "mem_cache/unified_cache/unified_tree_core.py",
            "UnifiedTreeCore",
            {
                "match_prefix",
                "_create_match_validators",
                "_match_prefix_helper",
                "_match_post_processor",
                "_device_prefix_covered_by_match",
            },
        )()
        tree.match_prefix.__globals__.update(
            BASE_COMPONENT_TYPE=0,
            ComponentType=r.ComponentType,
            MatchResult=r.MatchResult,
            get_and_increase_time_counter=lambda: 1.0,
            LRURefreshPhase=SimpleNamespace(MATCH_END=1),
        )
        tokens = array("q", range(32772))

        def node(start, end, parent):
            live = any(lo <= start and end <= hi for lo, hi in ranges)
            return SimpleNamespace(
                id=end,
                key=r.RadixKey(tokens[start:end]),
                parent=parent,
                children={},
                evicted=False,
                backuped=False,
                component_data=[
                    SimpleNamespace(value=torch.arange(start, end), host_value=None),
                    SimpleNamespace(
                        value=torch.arange(start, end) if live else None,
                        host_value=None,
                    ),
                ],
            )

        tree.root_node = previous = node(0, 0, None)
        for start in range(0, 32768, 256):
            child = node(start, start + 256, previous)
            previous.children[child.key.child_key(256)] = child
            previous = child
        tree.is_eagle, tree.page_size, tree.enable_hicache = False, 256, False
        components = []
        for filename, name, ct in (
            ("full", "FullComponent", 0),
            ("swa", "SWAComponent", 1),
        ):
            comp = _source_class(
                f"mem_cache/unified_cache/components/{filename}.py",
                name,
                {"create_match_validator", "finalize_match_result_in_tree_core"},
            )()
            comp.finalize_match_result_in_tree_core.__globals__["ComponentType"] = (
                r.ComponentType
            )
            comp.component_type, comp.tree_core = ct, tree
            comp.sliding_window_size, comp.gates_prefix_match = 128, True
            comp.refresh_lru = lambda *a: None
            components.append(comp)
        tree.components = components
        tree.components_by_type = {c.component_type: c for c in components}
        key_fn = _source_function(
            "mem_cache/common.py", "_req_radix_key", namespace={"RadixKey": r.RadixKey}
        )
        ids = (
            tokens
            if shared is None
            else tokens[:shared]
            + array("q", range(100000, 100000 + len(tokens) - shared))
        )
        key = key_fn(
            SimpleNamespace(swa_reprefill_tail_tokens=lambda req: 128 if replay else 0),
            SimpleNamespace(extra_key=None, cache_salt=None),
            ids,
            max_prefix_len=len(ids) - 1,
        )
        return tree.match_prefix(SimpleNamespace(key=key))

    def test_source_swa_windows_restore_identical_and_shared_prefix_hits(self):
        f = self._source_swa_windows()
        req, lease = f.request()
        for end in (8192, 16384, 24576, 32768, 32772):
            req.kv.kv_allocated_len = end
            f.cache.checkpoint(req, up_to=end)
            self.assertEqual(f.published, [])
            self.assertEqual(
                f.footprint(req.kv, 256), f.ceil_align(end, 256) - len(f.freed)
            )
        self.assertEqual(
            req.kv.swa_retained_ranges,
            [(7680, 8192), (15872, 16384), (24064, 24576), (32256, 32768)],
        )
        lease.retire(lambda: f.cache.checkpoint(req, up_to=32772))
        self.assertEqual(
            f.published, [(7680, 8192), (15872, 16384), (24064, 24576), (32256, 32768)]
        )
        self.assertEqual(req.kv.swa_retained_ranges, [])
        self.assertEqual(self._match_swa_windows(f.published).device_prefix_len, 32512)
        self.assertEqual(
            self._match_swa_windows(f.published, shared=24576).device_prefix_len, 24576
        )
        # Reproduce the deployed failure with the same native matching code.
        self.assertEqual(self._match_swa_windows([(32512, 32768)]).device_prefix_len, 0)
        self.assertEqual(
            self._match_swa_windows([(32512, 32768)], replay=False).device_prefix_len,
            32768,
        )
        f.release(req)
        self.assertEqual(len(f.freed) + sum(b - a for a, b in f.published), f.allocated)

    def test_source_swa_window_abort_returns_every_private_page_once(self):
        f = self._source_swa_windows()
        req, lease = f.request()
        for end in (8192, 16384, 24576, 32768, 32772):
            req.kv.kv_allocated_len = end
            f.cache.checkpoint(req, up_to=end)
        lease.abort = True
        lease.retire(lambda: f.release(req))
        self.assertEqual(f.published, [])
        self.assertEqual(f.freed, set(range(f.allocated)))
        self.assertEqual(f.full_freed, set(range(f.allocated)))
        self.assertEqual(req.kv.swa_retained_ranges, [])
        with self.assertRaisesRegex(RuntimeError, "no longer usable"):
            lease.retire(lambda: f.release(req))

    def test_source_swa_window_budget_is_global_and_131k_stays_bounded(self):
        f = self._source_swa_windows(prompt=131076)
        owners = [f.request(i) for i in range(4)]
        for end in [*range(8192, 131073, 8192), 131076]:
            for req, lease in owners:
                req.kv.kv_allocated_len = end
                f.cache.checkpoint(req, up_to=end)
                retained = sum(
                    b - a
                    for request, _ in owners
                    for a, b in request.kv.swa_retained_ranges
                )
                self.assertLessEqual(retained, 35072 // 1024 * 256)
                actual = f.ceil_align(end, 256) - sum(
                    i // f.allocated == req.kv.req_pool_idx for i in f.freed
                )
                self.assertEqual(f.footprint(req.kv, 256), actual)
                self.assertLessEqual(actual + 8192, 35072)
        for req, lease in owners:
            lease.abort = True
            lease.retire(lambda req=req: f.release(req))
        self.assertEqual(len(f.freed), f.rows.numel())
        self.assertEqual(len(f.full_freed), f.rows.numel())

    def test_source_swa_budget_never_reclaims_quarantined_or_failed_owner(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                f = self._source_swa_windows(prompt=131076, capacity=8192)
                req, lease = f.request(0)
                for end in (8192, 16384, 24576, 32768):
                    req.kv.kv_allocated_len = end
                    f.cache.checkpoint(req, up_to=end)
                saved = list(req.kv.swa_retained_ranges)
                before = {i for i in f.freed if i < f.allocated}
                if failed:
                    lease.failed = True
                else:
                    req._remote_attn_transport_quarantine = {"reason": "unknown DMA"}
                other, _ = f.request(1)
                for end in range(8192, 131073, 8192):
                    other.kv.kv_allocated_len = end
                    f.cache.checkpoint(other, up_to=end)
                self.assertEqual(req.kv.swa_retained_ranges, saved)
                self.assertEqual({i for i in f.freed if i < f.allocated}, before)
                self.assertEqual(f.published, [])

    def test_source_swa_retained_segment_free_preserves_other_rows_and_shields(self):
        free = _source_function("mem_cache/common.py", "free_kv_row_segments")
        for dead_lo in (0, 4, 12):
            for dead_hi in (12, 24, 32):
                full, swa = set(), set()
                row = torch.arange(32)
                kept = [(8, 12), (16, 20)]
                segments = [(row[:8], 0), (row[12:28], 12), (row[28:], 28)]

                def apply(parts, full_only=False):
                    self.assertEqual([p for _, p in parts], sorted(p for _, p in parts))
                    for indices, start in parts:
                        self.assertFalse(full.intersection(indices.tolist()))
                        full.update(indices.tolist())
                        if not full_only:
                            self.assertFalse(swa.intersection(indices.tolist()))
                            swa.update(indices.tolist())

                allocator = SimpleNamespace(
                    page_size=4,
                    free_segments=apply,
                    free_full_segments=lambda parts: apply(parts, full_only=True),
                )
                free(
                    allocator,
                    segments,
                    swa_evicted_seqlen=dead_hi,
                    swa_dead_lo=dead_lo,
                    swa_retained_ranges=kept,
                )
                expected_full = set(range(32)) - set(range(8, 12))
                expected_swa = {
                    i
                    for i in expected_full
                    if not dead_lo <= i < dead_hi or any(a <= i < b for a, b in kept)
                }
                self.assertEqual(full, expected_full)
                self.assertEqual(swa, expected_swa)

    def test_source_swa_window_replay_failure_restores_quarantine_accounting(self):
        f = self._source_swa_windows()
        req, lease = f.request()
        for end in (8192, 16384, 24576, 32768, 32772):
            req.kv.kv_allocated_len = end
            f.cache.checkpoint(req, up_to=end)
        old_cursor = req.kv.max_evicted_seqlen
        native = f.guard.checkpoint

        def fail(req, *, up_to, **kwargs):
            if up_to == 16384:
                raise RuntimeError("insertion failed")
            return native(req, up_to=up_to, **kwargs)

        f.guard.checkpoint = fail
        with self.assertRaisesRegex(RuntimeError, "insertion failed"):
            lease.retire(lambda: self.fail("released failed publication"))
        self.assertTrue(lease.failed)
        self.assertEqual(req.kv.max_evicted_seqlen, old_cursor)
        actual = f.allocated - len(f.freed) - sum(b - a for a, b in f.published)
        self.assertEqual(f.footprint(req.kv, 256), actual)
        with self.assertRaisesRegex(RuntimeError, "no longer usable"):
            lease.retire(lambda: self.fail("retried failed publication"))

    def test_source_native_swa_windows_wait_for_overlap_dma_remote_and_retirement(self):
        from sglang_remote_attn.source_cache import CacheGuard

        f = self._native_source_swa()
        guard = CacheGuard(f.cache, allow_swa=True, scheduler=self._swa_scheduler())
        req = f.request(finished=False)
        cache_lease = guard.hold(req)
        for end in (8192, 16384, 24576, 32768, 32772):
            f.extend(req, end)
            f.cache.checkpoint(req, up_to=end)
        native = self._dsv41_overlap_fixture()
        req.rid, req.bootstrap_room = native.req.rid, native.req.bootstrap_room
        req.disagg_kv_sender = native.req.disagg_kv_sender
        gpu, remote = threading.Event(), threading.Event()
        source = SimpleNamespace(
            error=None,
            finalized=True,
            owner=SimpleNamespace(dsv41_remote=True),
            service=None,
            remote_closed=remote,
            indices_keepalive=object(),
        )
        source.retire = lambda: cache_lease.retire(source.release)
        lease = native.po.SourceLease(
            native.overlap,
            native.po.Chunk(
                req, req.disagg_kv_sender, req.cache_request_handle, 0, 0, 32772, True
            ),
            1,
            "swa-window-proof",
            source=source,
            cache_lease=cache_lease,
            finalized=True,
            release=lambda: f.release(req),
        )
        plan = SimpleNamespace(
            error=None,
            proof=SimpleNamespace(ready=lambda: True),
            forward_done=SimpleNamespace(query=gpu.is_set),
            indices_refs=[],
        )
        lease.plans.append(plan)
        lease.ledger.pending.add("dma")
        native.overlap.leases.append(lease)
        native.overlap.retirement.register(lease)
        native.native_status[0] = native.polls.Success

        def snapshot():
            f.ownership()
            return (
                set(f.allocator.full_attn_allocator.live),
                set(f.allocator.swa_attn_allocator.live),
            )

        before = snapshot()
        with patch.dict(sys.modules, native.modules):
            for complete in (lambda: None, gpu.set, lease.ledger.pending.clear):
                complete()
                self.assertFalse(lease.ready_to_finish())
                with self.assertRaisesRegex(RuntimeError, "proof is incomplete"):
                    lease.finish()
                self.assertEqual(snapshot(), before)
                self.assertEqual(f.cache.tree_core.root_node.children, {})
            remote.set()
            self.assertTrue(lease.ready_to_finish())
            self.assertEqual(f.cache.tree_core.root_node.children, {})
            native.overlap.poll()
            command = native.overlap.retirement.poll_command()
            self.assertIsNotNone(command)
            self.assertEqual(f.cache.tree_core.root_node.children, {})
            native.overlap.retirement.arm(command)
            native.overlap.retirement.finalize()
        self.assertTrue(cache_lease.closed)
        self.assertTrue(lease.closed)
        self.assertFalse(req.kv.holds_kv)
        f.ownership()
        # Repeated finalization cannot replay checkpoints or release twice.
        native.overlap.retirement.finalize()
        f.evict_all()

    def test_source_swa_windows_wait_for_overlap_dma_remote_and_retirement(self):
        f = self._source_swa_windows()
        req, cache_lease = f.request()
        for end in (8192, 16384, 24576, 32768, 32772):
            req.kv.kv_allocated_len = end
            f.cache.checkpoint(req, up_to=end)
        native = self._dsv41_overlap_fixture()
        req.rid, req.bootstrap_room = native.req.rid, native.req.bootstrap_room
        req.disagg_kv_sender = native.req.disagg_kv_sender
        gpu, remote = threading.Event(), threading.Event()
        source = SimpleNamespace(
            error=None,
            finalized=True,
            owner=SimpleNamespace(dsv41_remote=True),
            service=None,
            remote_closed=remote,
            indices_keepalive=object(),
        )
        source.retire = lambda: cache_lease.retire(source.release)
        lease = native.po.SourceLease(
            native.overlap,
            native.po.Chunk(
                req, req.disagg_kv_sender, req.cache_request_handle, 0, 0, 32772, True
            ),
            1,
            "swa-window-proof",
            source=source,
            cache_lease=cache_lease,
            finalized=True,
            release=lambda: f.release(req),
        )
        plan = SimpleNamespace(
            error=None,
            proof=SimpleNamespace(ready=lambda: True),
            forward_done=SimpleNamespace(query=gpu.is_set),
            indices_refs=[],
        )
        lease.plans.append(plan)
        lease.ledger.pending.add("dma")
        native.overlap.leases.append(lease)
        native.overlap.retirement.register(lease)
        native.native_status[0] = native.polls.Success
        before = set(f.freed)
        with patch.dict(sys.modules, native.modules):
            for complete in (lambda: None, gpu.set, lease.ledger.pending.clear):
                complete()
                self.assertFalse(lease.ready_to_finish())
                with self.assertRaisesRegex(RuntimeError, "proof is incomplete"):
                    lease.finish()
                self.assertEqual(f.freed, before)
                self.assertEqual(f.published, [])
            remote.set()
            self.assertTrue(lease.ready_to_finish())
            self.assertEqual(f.published, [])
            native.overlap.poll()
            command = native.overlap.retirement.poll_command()
            self.assertIsNotNone(command)
            self.assertEqual(f.published, [])
            native.overlap.retirement.arm(command)
            native.overlap.retirement.finalize()
        self.assertTrue(cache_lease.closed)
        self.assertTrue(lease.closed)
        self.assertEqual(len(f.published), 4)
        self.assertEqual(len(f.freed) + sum(b - a for a, b in f.published), f.allocated)

    def test_source_swa_window_declined_insert_releases_without_final_retry(self):
        f = self._source_swa_windows()
        req, lease = f.request()
        for end in (8192, 16384, 24576, 32768, 32772):
            req.kv.kv_allocated_len = end
            f.cache.checkpoint(req, up_to=end)
        calls = []
        f.guard.checkpoint = lambda req, **kw: calls.append(kw["up_to"])
        lease.retire(lambda: (f.cache.checkpoint(req, up_to=32772), f.release(req)))
        self.assertEqual(calls, [8192])
        self.assertTrue(lease.publication_declined)
        self.assertEqual(len(f.freed), f.allocated)
        self.assertEqual(len(f.full_freed), f.allocated)

    def test_source_replays_pending_checkpoint_before_row_release_and_warm_hit(self):
        from sglang_remote_attn.source_cache import CacheGuard

        for callback_checkpoint in (False, True):
            with self.subTest(callback_checkpoint=callback_checkpoint):
                f = self._radix()
                req, original = f.request([1, 2, 3, 4, 5, 6], "cold")
                calls = []
                native = f.cache.checkpoint
                f.cache.checkpoint = lambda req, **kw: (
                    calls.append(kw["up_to"]),
                    native(req, **kw),
                )[-1]
                lease = CacheGuard(f.cache).hold(req)
                for end in (2, 4, 6, 3):
                    f.cache.checkpoint(req, up_to=end)
                self.assertEqual(calls, [])
                self.assertEqual(f.cache.total_size(), 0)
                self.assertEqual(lease.pending_checkpoint[0], 6)
                lease.retire(
                    lambda: f.release_kv_cache(
                        req, f.cache, checkpoint=callback_checkpoint
                    )
                )
                self.assertEqual(calls, [6])
                self.assertTrue(lease.replayed_checkpoint)
                hit, indices = f.request([1, 2, 3, 4, 5, 6, 7], "warm")
                self.assertEqual(hit.kv.cache_protected_len, 6)
                self.assertEqual(indices[:6].tolist(), original.tolist())
                f.release_kv_cache(hit, f.cache, checkpoint=True)
                f.cache.evict(f.EvictParams(num_tokens=32))
                self.assertEqual(f.allocator.available_size(), 32)
                self.assertEqual(len(set(f.allocator.free_pages.tolist())), 32)
                with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                    lease.retire(lambda: self.fail("duplicate row free"))

    def test_source_pending_checkpoint_abort_and_publish_failure_never_free_twice(self):
        from sglang_remote_attn.source_cache import CacheGuard, checkpoint_allowed

        for abort in (False, True):
            f = self._radix()
            req, _ = f.request([1, 2, 3], "attempt")
            calls = []

            def fail(req, **kwargs):
                calls.append("checkpoint")
                raise RuntimeError("partial publication")

            if not abort:
                f.cache.checkpoint = fail
            lease = CacheGuard(f.cache).hold(req)
            req._pre_layerwise_source = SimpleNamespace(cache_lease=lease)
            f.cache.checkpoint(req, up_to=3)
            if abort:
                f.cache.finish(req.cache_request_handle, f.CacheRequestOutcome.ABORT)
                lease.retire(
                    lambda: f.release_kv_cache(
                        req, f.cache, checkpoint=checkpoint_allowed(req, True)
                    )
                )
                self.assertEqual(f.cache.total_size(), 0)
                self.assertEqual(f.allocator.available_size(), 32)
            else:
                with self.assertRaisesRegex(RuntimeError, "partial publication"):
                    lease.retire(lambda: self.fail("free after failed publication"))
                with self.assertRaisesRegex(RuntimeError, "no longer usable"):
                    lease.retire(lambda: self.fail("retry free"))
                self.assertEqual(calls, ["checkpoint"])
                self.assertEqual(f.allocator.available_size(), 29)

    def test_source_retirement_publishes_live_swa_tail_past_stale_branch(self):
        from sglang_remote_attn.source_cache import CacheGuard

        component = _source_class(
            "mem_cache/unified_cache/components/swa.py",
            "SWAComponent",
            {"prepare_for_caching_req", "_clip_to_window"},
        )()
        component.component_type = "swa"
        component.cache = SimpleNamespace(is_mamba_enabled=False)
        component.tree_core = SimpleNamespace(is_eagle=False, page_size=256)
        req = SimpleNamespace(
            cache_request_handle="swa-branch",
            last_node=0,
            lock=None,
            kv=SimpleNamespace(
                cache_protected_len=0, get_evicted_seqlen=lambda ct: 7424
            ),
            swa_branching_seqlen=2048,
        )
        published = []

        def checkpoint(req, *, up_to):
            cursors = {}
            params = SimpleNamespace(
                set_evicted_seqlen=lambda c, n: cursors.update({c: n})
            )
            shortened = component.prepare_for_caching_req(req, params, up_to, True)
            end = min(up_to, shortened) if shortened is not None else up_to
            node = SimpleNamespace(component_data={"swa": SimpleNamespace(value=None)})
            # Only the live final window is published. The old 2K branch's
            # SWA was reclaimed, so truncating here would cache no usable tail.
            component.tree_core._split_node = lambda *a: (None, None)
            node.key = range(end)
            live = component._clip_to_window(
                node=node,
                start=0,
                end=end,
                window_start=cursors["swa"],
                cache_actions=[],
            )
            published.append((end, live))

        cache = SimpleNamespace(
            checkpoint=checkpoint,
            finish=lambda *a: None,
            reset=lambda: None,
            lock=lambda node: object(),
            unlock=lambda pin: None,
        )
        guard = CacheGuard(cache, allow_swa=True)
        lease = guard.hold(req)
        cache.checkpoint(req, up_to=8192)
        self.assertEqual(published, [])
        lease.retire(lambda: cache.checkpoint(req, up_to=8192))
        self.assertEqual(published, [(8192, 7424)])
        self.assertIsNone(req.swa_branching_seqlen)

    def test_source_metadata_waits_overlap_gpu_io_and_ordered_retirement(self):
        from sglang_remote_attn.source_resources import release_metadata

        f, c = self._dsv41_overlap_fixture(), self._dsv41_cache()
        f.controller.cache_guard = c.guard
        f.req.cache_request_handle = c.native.CacheRequestHandle(f.req.rid, 0)
        f.req.kv.cache_protected_len = 8
        f.req.last_node, f.req.lock = 1, c.cache.lock(1)
        completed = [False]

        class Event:
            def record(self, stream):
                pass

            def query(self):
                return completed[0]

        def launch(scheduler, batch):
            f.controller.forward(lambda *a: None, f.runner, batch)
            return SimpleNamespace(delay_sample_func=None, copy_done=Event())

        native_meta = _source_function(
            "disaggregation/prefill.py", "maybe_release_metadata_buffer"
        )
        frees = []
        allocator = SimpleNamespace(free=frees.append)
        f.req.metadata_buffer_index = 7
        f.req.extend_end, f.req.extend_len = 33, 33
        with (
            patch.dict(sys.modules, f.modules),
            patch.object(torch.cuda, "Event", Event),
            patch.object(torch.cuda, "current_stream", lambda: None),
            patch.object(
                f.po,
                "page_indices",
                lambda raw, page: (
                    (raw[::page] // page).numpy().copy(),
                    Event(),
                    (raw,),
                ),
            ),
        ):
            result = f.overlap.run_batch(launch, f.scheduler, f.batch)
            lease = f.req._dsv41_p_lease
            f.po.after_result(
                lambda scheduler, *a: scheduler.send_kv_chunk(f.req, last_chunk=True),
                f.scheduler,
                f.batch,
                result,
            )
            release_metadata(native_meta, f.req, allocator)
            release_metadata(native_meta, f.req, allocator)
            f.overlap.defer_release(f.req, lambda: c.cache.unlock(f.req.lock))
            f.overlap.defer_release(f.req, lambda: self.fail("duplicate KV free"))
            f.native_status[0] = f.polls.Success
            self.assertFalse(lease.ready_to_finish())
            self.assertEqual(frees, [])
            completed[0] = True
            self.assertFalse(lease.ready_to_finish())  # transfer callbacks pending
            for _ in range(f.tasks.qsize()):
                f.tasks.get_nowait().on_complete()
            self.assertTrue(lease.ready_to_finish())
            f.overlap.poll()
            self.assertEqual(frees, [])  # local proof alone cannot release
            f.overlap.retirement.arm(f.overlap.retirement.poll_command())
            f.overlap.retirement.finalize()
            self.assertEqual(frees, [7])
            release_metadata(native_meta, f.req, allocator)
            self.assertEqual(frees, [7])

    def test_source_quarantine_failed_inflight_keeps_metadata_and_kv(self):
        from sglang_remote_attn.source_resources import (
            release_metadata,
            retire_metadata,
        )
        from sglang_remote_attn.transport_failure import quarantine
        from sglang_remote_attn.pre_layerwise import defer_release
        import logging

        native = _source_class(
            "disaggregation/prefill.py",
            "SchedulerDisaggregationPrefillMixin",
            {
                "process_disagg_prefill_transfer_queue",
                "handle_inflight_transfer_failure",
            },
            strip_method_decorators=True,
        )

        class Aborted:
            pass

        def failure():
            raise RuntimeError("unsettled DMA")

        freed, kv_freed = [], []

        class Request(SimpleNamespace):
            __hash__ = object.__hash__
            __eq__ = object.__eq__

        req = Request(
            rid="failed",
            bootstrap_room=9,
            metadata_buffer_index=7,
            cache_request_handle="attempt",
            return_logprob=False,
            pending_bootstrap=False,
            finished_reason=Aborted(),
            finished=lambda: True,
            time_stats=SimpleNamespace(
                set_completion_time=lambda: None,
                trace_ctx=SimpleNamespace(abort=lambda **kw: None),
            ),
            disagg_kv_sender=SimpleNamespace(failure_exception=failure),
        )
        req.kv = self.Kv(req_pool_idx=0, kv_allocated_len=8)
        source = SimpleNamespace(
            req=req,
            released=False,
            final_payload=([], slice(0, 0), {"aux_index": 7}),
            ledger=SimpleNamespace(pending={"aux"}),
            cache_lease=SimpleNamespace(
                abort=False,
                defer_checkpoint=lambda *a: self.fail(
                    "checkpoint on quarantined lease"
                ),
            ),
        )
        req._pre_layerwise_source = source
        scheduler = SimpleNamespace(
            chunked_req=None,
            disagg_prefill_transfer_queue=_request_list([req], kind="PD_SENDING"),
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            tree_cache=SimpleNamespace(finish=lambda *a: None),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            output_streamer=SimpleNamespace(stream_output=lambda *a: None),
            request_retirer=SimpleNamespace(
                release=lambda req, **kw: defer_release(
                    req, lambda: kv_freed.append(req), checkpoint=kw["checkpoint"]
                )
            ),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=freed.append),
        )
        scheduler.handle_inflight_transfer_failure = lambda req: (
            native.handle_inflight_transfer_failure(scheduler, req)
        )
        native_meta = _source_function(
            "disaggregation/prefill.py", "maybe_release_metadata_buffer"
        )
        native.handle_inflight_transfer_failure.__globals__.update(
            maybe_release_metadata_buffer=lambda req, alloc: release_metadata(
                native_meta, req, alloc
            ),
            FINISH_ABORT=Aborted,
            logger=logging.getLogger(__name__),
            get_parallel=lambda: SimpleNamespace(tp_rank=0),
            KVPoll=SimpleNamespace(
                Success="success",
                Failed="failed",
                WaitingForInput="waiting",
                Transferring="transferring",
            ),
            poll_and_all_reduce_attn_cp_tp_group=lambda *a: ["failed"],
            release_kv_cache=lambda req, cache, **kw: defer_release(
                req, lambda: kv_freed.append(req), checkpoint=kw["checkpoint"]
            ),
        )
        owner = SimpleNamespace(scheduler=scheduler, sources=[source], targets=[])
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.utils": SimpleNamespace(
                    prepare_abort=lambda *a, **kw: None
                )
            },
        ):
            quarantine(owner, source, "pending metadata DMA")
            native.process_disagg_prefill_transfer_queue(scheduler)
        retire_metadata(req)  # quarantine never supplies retirement proof
        release_metadata(
            native_meta, req, scheduler.req_to_metadata_buffer_idx_allocator
        )
        self.assertEqual(freed, [])
        self.assertEqual(kv_freed, [])
        self.assertEqual(req.metadata_buffer_index, 7)
        self.assertEqual(source.final_payload[2]["aux_index"], 7)

    def test_source_metadata_attempt_reuse_and_failed_free_are_not_retried(self):
        from sglang_remote_attn.source_resources import (
            release_metadata,
            retire_metadata,
        )

        native = _source_function(
            "disaggregation/prefill.py", "maybe_release_metadata_buffer"
        )
        frees = []
        allocator = SimpleNamespace(free=frees.append)
        req = SimpleNamespace(cache_request_handle="a", metadata_buffer_index=2)
        release_metadata(native, req, allocator)
        self.assertEqual(frees, [2])  # ordinary native path
        req._pre_layerwise_source = SimpleNamespace(released=False, cache_lease=None)
        for attempt in ("b", "c"):
            req.cache_request_handle, req.metadata_buffer_index = attempt, 3
            release_metadata(native, req, allocator)
            self.assertEqual(frees.count(3), int(attempt == "c"))
            retire_metadata(req)
            retire_metadata(req)
        self.assertEqual(frees, [2, 3, 3])  # two distinct allocations
        req.cache_request_handle, req.metadata_buffer_index = "d", 4

        def fail(index):
            frees.append(index)
            raise RuntimeError("free completion unknown")

        release_metadata(native, req, SimpleNamespace(free=fail))
        with self.assertRaisesRegex(RuntimeError, "completion unknown"):
            retire_metadata(req)
        with self.assertRaisesRegex(RuntimeError, "refusing retry"):
            retire_metadata(req)
        self.assertEqual(frees.count(4), 1)

    def test_source_external_owners_are_counted_without_hiding_actual_leaks(self):
        from sglang_remote_attn.source_resources import owned_requests
        from sglang.srt.environ import envs
        from types import MethodType

        names = {
            "_external_kv_owners",
            "_requests_owning_rows",
            "_get_total_uncached_sizes",
            "_check_all_pools",
            "_check_req_pool",
            "_check_pool_invariant",
        }
        cls = _source_class(
            "managers/scheduler_components/invariant_checker.py",
            "SchedulerInvariantChecker",
            names,
        )
        private = _source_function(
            "managers/scheduler_components/pool_stats_observer.py", "kv_private_tokens"
        )
        swa = _source_function(
            "managers/scheduler_components/pool_stats_observer.py",
            "kv_private_swa_tokens",
        )
        ceil_align = lambda n, p: (n + p - 1) // p * p
        private.__globals__["ceil_align"] = swa.__globals__["ceil_align"] = ceil_align
        swa.__globals__["ComponentType"] = SimpleNamespace(
            SWA="swa", AUXILIARY_SWA="auxiliary_swa"
        )
        self.Kv.swa_dead_lo.__globals__["ceil_align"] = ceil_align
        cls._get_total_uncached_sizes.__globals__.update(
            kv_private_tokens=private,
            kv_private_swa_tokens=swa,
            envs=envs,
            raise_error_or_warn=_source_function(
                "utils/common.py", "raise_error_or_warn"
            ),
        )
        kv = self.Kv(req_pool_idx=0, kv_allocated_len=16, cache_protected_len=4)
        kv.set_evicted_seqlen("swa", 12)
        req = SimpleNamespace(kv=kv, beam_group=None)
        source = SimpleNamespace(req=req)
        owner = SimpleNamespace(
            sources=[source],
            targets=[],
            transport_quarantine={1: {"req": req}},
            cache_guard=SimpleNamespace(leases={1: SimpleNamespace(req=req)}),
        )
        checker = SimpleNamespace(
            page_size=4,
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            tree_cache=SimpleNamespace(
                session_records=lambda: {}, swa_transient_size=lambda: 0
            ),
            get_external_kv_owners=lambda: owned_requests(owner),
            get_last_batch=lambda: SimpleNamespace(reqs=[req]),
            get_request_lists=lambda: {"running": [req]},
            get_chunked_req=lambda: req,
            req_to_token_pool=SimpleNamespace(size=2, pre_alloc_size=0, free_slots=[1]),
            pool_stats_observer=SimpleNamespace(session_held_req_count=lambda: 0),
        )
        for name in names - {"_check_pool_invariant"}:
            setattr(checker, name, MethodType(getattr(cls, name), checker))
        self.assertEqual(checker._get_total_uncached_sizes(True), (12, 4))
        ticket = SimpleNamespace(req=req, cancelled=False, consumed=False)
        owner.d2d_reservations = {"prepared": ticket}
        # A prepared ticket can be the only owner for an entire Prefill. It
        # must count exactly once even outside all native scheduling lists.
        owner.sources.clear()
        owner.cache_guard.leases.clear()
        owner.transport_quarantine.clear()
        checker.get_request_lists = lambda: {}
        self.assertEqual(checker._get_total_uncached_sizes(True), (12, 4))
        checker.get_last_batch, checker.get_chunked_req = lambda: None, lambda: None
        checker._check_full_pool = lambda ps, uncached=0: cls._check_pool_invariant(
            "full", ps.full_free, 0, 4, 0, 32, uncached
        )
        checker._check_swa_pool = lambda ps, uncached=0: cls._check_pool_invariant(
            "swa", ps.swa_free, 0, 4, 0, 32, uncached
        )
        stats = SimpleNamespace(
            full_free=16, swa_free=24, is_hybrid_swa=True, is_hybrid_ssm=False
        )
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True):
            self.assertFalse(checker._check_all_pools(stats)[0])
            checker._check_req_pool()
            stats.full_free -= 4
            self.assertTrue(checker._check_all_pools(stats)[0])
            checker.req_to_token_pool.free_slots.clear()
            with self.assertRaisesRegex(ValueError, "memory leak"):
                checker._check_req_pool()
        self.assertTrue(req.kv.holds_kv)
        self.assertEqual(kv.get_evicted_seqlen("swa"), 12)
        ticket.consumed = True
        owner.d2d_targets = {
            "prepared": SimpleNamespace(
                closed=False, decode_req=SimpleNamespace(req=SimpleNamespace(kv=kv))
            )
        }
        self.assertEqual(len(checker._external_kv_owners()), 1)
        owner.d2d_targets.clear()
        ticket.cancelled = True
        self.assertEqual(checker._external_kv_owners(), [])

    def test_source_deferred_checkpoint_reclaims_long_prompt_swa_only(self):
        from sglang_remote_attn.source_cache import CacheGuard

        tree = ast.parse((ROOT / "python/sglang/srt/mem_cache/common.py").read_text())
        node = next(
            n
            for n in tree.body
            if getattr(n, "name", None) == "free_swa_out_of_window_slots"
        )
        namespace = dict(
            ComponentType=SimpleNamespace(SWA="swa", AUXILIARY_SWA="auxiliary_swa"),
            ceil_align=lambda n, p: (n + p - 1) // p * p,
            is_cp_sharded_padded=lambda: False,
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
        exec(
            compile(ast.fix_missing_locations(module), "<native-swa-free>", "exec"),
            namespace,
        )
        native_free = namespace["free_swa_out_of_window_slots"]
        for protected in (0, 2048):
            with self.subTest(protected=protected):
                freed, published = [], []
                cursor = [protected]
                rows = torch.arange(131072).reshape(1, -1)
                kv = SimpleNamespace(
                    holds_kv=True,
                    cache_protected_len=protected,
                    req_pool_idx=0,
                    get_evicted_seqlen=lambda c: cursor[0],
                    set_evicted_seqlen=lambda c, n: cursor.__setitem__(0, n),
                    swa_dead_lo=lambda p: protected,
                )
                req = SimpleNamespace(
                    kv=kv,
                    origin_input_ids=range(131072),
                    cache_request_handle="attempt",
                    lock=object(),
                    last_node=object(),
                    finished=lambda: False,
                )
                cache = SimpleNamespace(
                    supports_swa=lambda: True,
                    checkpoint=lambda *a, **k: published.append(k),
                    finish=lambda *a: None,
                    reset=lambda: None,
                    lock=lambda n: object(),
                )
                advance = lambda req, end: native_free(
                    req,
                    end,
                    sliding_window_size=512,
                    page_size=128,
                    req_to_token_pool=SimpleNamespace(req_to_token=rows),
                    token_to_kv_pool_allocator=SimpleNamespace(
                        free_swa_segment=lambda indices, **kw: freed.extend(
                            indices.tolist()
                        )
                    ),
                )
                _bind_window_retirement(cache, advance)
                guard = CacheGuard(
                    cache, allow_swa=True, scheduler=self._swa_scheduler()
                )
                lease = guard.hold(req)
                for end in range(protected + 2048, 131073, 2048):
                    # Same checkpoint-before-next-chunk boundary as native P.
                    cache.checkpoint(req, up_to=end)
                    self.assertLessEqual(end - protected - len(freed), 640)
                    self.assertLessEqual(end - len(freed) + 2048, protected + 2688)
                before_final_dma = list(freed)
                cache.checkpoint(req, up_to=131072 + 2048)
                self.assertEqual(freed, before_final_dma)
                self.assertEqual(len(freed), len(set(freed)))
                self.assertTrue(all(protected <= i < 131072 - 512 for i in freed))
                self.assertEqual(published, [])
                self.assertFalse(lease.closed)
                self.assertTrue(torch.equal(rows, torch.arange(131072).reshape(1, -1)))

    def test_source_swa_131k_fits_35072_pool_with_native_accounting(self):
        from sglang_remote_attn.source_cache import CacheGuard

        # Source-load the real freeing and footprint code. The fake allocator
        # records physical returns separately, so merely advancing a cursor
        # without returning SWA pages cannot make this test pass.
        native_free = _source_function(
            "mem_cache/common.py",
            "free_swa_out_of_window_slots",
            namespace={
                "ComponentType": SimpleNamespace(
                    SWA="swa", AUXILIARY_SWA="auxiliary_swa"
                )
            },
        )
        footprint = _source_function(
            "managers/scheduler_components/pool_stats_observer.py",
            "kv_private_swa_tokens",
        )
        ceil_align = lambda n, p: (n + p - 1) // p * p
        component = SimpleNamespace(SWA="swa", AUXILIARY_SWA="auxiliary_swa")
        native_free.__globals__.update(ComponentType=component, ceil_align=ceil_align)
        footprint.__globals__.update(ComponentType=component, ceil_align=ceil_align)
        self.Kv.swa_dead_lo.__globals__["ceil_align"] = ceil_align
        prompt, page, capacity, chunk = 131076, 256, 35072, 8192
        rows = torch.arange(ceil_align(prompt, page)).reshape(1, -1)
        for window in (128, 512, 8192):
            for protected in (0, 30720):
                for shield in (0, 2048):
                    for reclaim in (False, True):
                        with self.subTest(
                            window=window,
                            protected=protected,
                            shield=shield,
                            reclaim=reclaim,
                        ):
                            freed, published = set(), []
                            kv = self.Kv(
                                req_pool_idx=0,
                                cache_protected_len=protected,
                                kv_allocated_len=protected,
                                swa_evict_floor=protected + shield,
                            )
                            req = SimpleNamespace(
                                kv=kv,
                                cache_request_handle="capacity-attempt",
                                origin_input_ids=range(prompt),
                                last_node=object(),
                                lock=object(),
                                finished=lambda: False,
                            )
                            cache = SimpleNamespace(
                                supports_swa=lambda: True,
                                checkpoint=lambda *a, **k: published.append(k),
                                finish=lambda *a: None,
                                reset=lambda: None,
                                lock=lambda node: object(),
                            )

                            def free_segment(indices, *, start_pos):
                                slots = set(indices.tolist())
                                self.assertFalse(slots & freed)
                                self.assertTrue(
                                    all(i >= kv.swa_dead_lo(page) for i in slots)
                                )
                                self.assertEqual(start_pos % page, 0)
                                freed.update(slots)

                            advance = lambda req, end: native_free(
                                req,
                                end,
                                sliding_window_size=window,
                                page_size=page,
                                req_to_token_pool=SimpleNamespace(req_to_token=rows),
                                token_to_kv_pool_allocator=SimpleNamespace(
                                    free_swa_segment=free_segment
                                ),
                            )
                            _bind_window_retirement(cache, advance)
                            guard = CacheGuard(
                                cache,
                                allow_swa=reclaim,
                                scheduler=self._swa_scheduler(),
                            )
                            lease = guard.hold(req)
                            # A warm hit owns only its resident tree SWA tail;
                            # its FULL prefix is still held for transfer/Remote.
                            cached_live = min(
                                protected, ceil_align(window, page) + page
                            )
                            end, peak = protected, cached_live
                            sizes = iter([chunk] * 4 + [2048])
                            while end < prompt:
                                take = min(next(sizes, chunk), prompt - end)
                                allocation = ceil_align(end + take, page) - protected
                                used = cached_live + allocation - len(freed)
                                if used > capacity:
                                    break
                                end += take
                                kv.kv_allocated_len = end
                                peak = max(peak, used)
                                cache.checkpoint(req, up_to=end)
                                private = ceil_align(end, page) - protected - len(freed)
                                self.assertEqual(footprint(kv, page), private)
                                if reclaim:
                                    self.assertLessEqual(
                                        private,
                                        shield + ceil_align(window, page) + 2 * page,
                                    )
                            if reclaim:
                                self.assertEqual(end, prompt)
                                self.assertLessEqual(peak, capacity)
                                tail_start = (prompt - window) // page * page
                                self.assertTrue(all(i < tail_start for i in freed))
                                self.assertGreater(
                                    kv.get_evicted_seqlen(component.SWA), protected
                                )
                            else:
                                self.assertLess(end, prompt)
                                self.assertEqual(freed, set())
                                if protected == 0:
                                    self.assertEqual(end, 34816)
                                    self.assertEqual(prompt - end, 96260)
                            self.assertFalse(lease.closed)
                            self.assertEqual(published, [])
                            self.assertTrue(
                                torch.equal(
                                    rows, torch.arange(rows.numel()).reshape(1, -1)
                                )
                            )

    def test_source_swa_reclaim_waits_for_forward_and_keeps_final_transfer_window(self):
        from sglang_remote_attn.source_cache import CacheGuard

        # Model device queues, not CPU completion: checkpoint must return while
        # the current chunk still reads its SWA, with the free queued behind it.
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                forward = SimpleNamespace(issued=1, completed=0)
                queue, reclaimed, active = [], [], []
                schedule = SimpleNamespace(
                    wait_stream=lambda stream: queue.append(
                        ("wait", stream, stream.issued)
                    )
                )

                @contextmanager
                def stream_context(stream):
                    active.append(stream)
                    try:
                        yield
                    finally:
                        active.pop()

                def evict(req, pre_len):
                    self.assertIs(active[-1], schedule)
                    queue.append(("free", pre_len))

                def drain():
                    while queue:
                        op = queue[0]
                        if op[0] == "wait" and op[1].completed < op[2]:
                            break
                        queue.pop(0)
                        if op[0] == "free":
                            reclaimed.append(op[1])

                scheduler = SimpleNamespace(
                    enable_overlap=overlap,
                    schedule_stream=schedule,
                    forward_stream=forward,
                    device_module=SimpleNamespace(stream=stream_context),
                )
                cache = SimpleNamespace(
                    checkpoint=lambda *a, **k: self.fail("published live FULL KV"),
                    finish=lambda *a: self.fail("finished live source"),
                    reset=lambda: None,
                    lock=lambda node: object(),
                    supports_swa=lambda: True,
                )
                req = SimpleNamespace(
                    cache_request_handle="attempt",
                    kv=SimpleNamespace(cache_protected_len=0),
                    last_node=object(),
                    origin_input_ids=range(8192),
                    finished=lambda: False,
                )
                _bind_window_retirement(cache, evict)
                guard = CacheGuard(cache, allow_swa=True, scheduler=scheduler)
                lease = guard.hold(req)
                cache.checkpoint(req, up_to=2048)
                drain()
                self.assertEqual(reclaimed, [])
                forward.completed = 1
                drain()
                self.assertEqual(reclaimed, [2047])

                # Final SWA DMA may still be running. The final prompt window
                # remains protected even if a caller supplies a larger frontier.
                forward.issued = 2
                cache.checkpoint(req, up_to=8192)
                cache.checkpoint(req, up_to=8192 + 2048)
                drain()
                self.assertEqual(reclaimed, [2047])
                forward.completed = 2
                drain()
                self.assertEqual(reclaimed, [2047, 8191, 8191])
                self.assertFalse(lease.closed)
                self.assertIsNotNone(lease.lock)

                # Unknown stream owner and aborted requests cannot reclaim.
                guard.scheduler = None
                cache.checkpoint(req, up_to=8192)
                guard.scheduler = scheduler
                req.finished = lambda: True
                cache.checkpoint(req, up_to=8192)
                self.assertEqual(queue, [])

    def test_quarantine_chunk_abort_and_admission_stop_preserve_pins(self):
        from sglang_remote_attn.transport_failure import quarantine, hold_release

        for already_finished in (False, True):
            state = [already_finished]
            req = SimpleNamespace(
                rid="failed-chunk", return_logprob=False, finished=lambda: state[0]
            )
            pin = object()
            cert = SimpleNamespace(done=False, cache_lease=pin, candidate=object())
            req._dsv41_p_certificate = cert
            scheduler = SimpleNamespace(
                chunked_req=req,
                _pending_chunked_abort_req=None,
                output_streamer=SimpleNamespace(stream_output=lambda *a: None),
            )
            owner = SimpleNamespace(scheduler=scheduler, sources=[], targets=[])
            item = SimpleNamespace(req=req)

            def abort(*args, **kwargs):
                state[0] = True

            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.utils": SimpleNamespace(
                        prepare_abort=abort
                    )
                },
            ):
                quarantine(owner, item, "timeout")
                quarantine(owner, item, "repeat")
            self.assertIs(scheduler._pending_chunked_abort_req, req)
            self.assertTrue(cert.done)
            self.assertIs(cert.cache_lease, pin)
            self.assertTrue(hold_release(req, lambda: self.fail("unsafe free")))
            # Admission observe skips a quarantined certificate before examining
            # failed peers or trying a revoke/drain with no completion proof.
            from sglang_remote_attn.prefill_admission import Admission

            admission = Admission.__new__(Admission)
            admission.records = {"failed": cert}
            admission.observe()

    def test_ordered_transport_failure_isolates_request_without_free_or_local_switch(
        self,
    ):
        from sglang_remote_attn.transport_failure import hold_release

        with tempfile.TemporaryDirectory() as directory:
            req = SimpleNamespace(
                rid="bad",
                bootstrap_room=7,
                return_logprob=False,
                finished=lambda: False,
            )
            owner, target, freed = self._overlap_target(directory, req)
            target.queue.queue = [target.decode_req]
            outputs = []
            owner.scheduler.output_streamer = SimpleNamespace(
                stream_output=lambda *args: outputs.append(args)
            )
            receiver = target.decode_req.kv_receiver
            receiver.poll = lambda: 0
            healthy = copy.copy(target)
            healthy.decode_req = SimpleNamespace(
                req=SimpleNamespace(rid="good", bootstrap_room=8)
            )
            healthy.admission_id, healthy.generation = 2, "healthy"
            healthy.gpu_ready = healthy.transport_ready = lambda: False
            owner.targets.append(healthy)
            owner.decode_overlap.retirement.register(healthy)
            utils = SimpleNamespace(
                prepare_abort=lambda req, reason, status_code: setattr(
                    req, "finished_reason", (reason, status_code)
                )
            )
            with patch.dict(sys.modules, {"sglang.srt.disaggregation.utils": utils}):
                command = self._retire_step(owner)
            self.assertEqual(command["kind"], "quarantine")
            self.assertEqual(owner.targets, [healthy])
            self.assertFalse(target.closed)
            self.assertEqual(freed, [])
            self.assertIs(target.decode_req.kv_receiver, receiver)
            self.assertEqual(target.queue.queue, [])
            self.assertEqual(req.finished_reason[1], 503)
            self.assertEqual(len(outputs), 1)
            self.assertTrue(hold_release(req, lambda: freed.append("unsafe")))
            self.assertEqual(freed, [])
            self.assertEqual(len(owner.transport_quarantine), 1)
            self.assertIsNone(owner.decode_overlap.retirement.poll_command())
            self.assertFalse(hasattr(owner, "transport_degraded"))
            self.assertEqual(owner.config.mode, "remote")

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
        flat.__globals__["StateType"] = SimpleNamespace(
            SWA="swa", AUXILIARY_SWA="auxiliary_swa"
        )
        manager_cls._send_kvcache_generic.__globals__.update(
            StateType=SimpleNamespace(
                SWA="swa", AUXILIARY_SWA="auxiliary_swa", DSA="dsa"
            ),
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
                    _pre_layerwise_source=SimpleNamespace(
                        stager=stager, error="unproven copy"
                    )
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

    def test_staging_grid_preserves_disabled_compute_chunking(self):
        grid = _source_function(
            "disaggregation/common/staging_buffer.py", "staging_grid_tokens"
        )
        for page in (1, 256, 300, 16384):
            for size in (None, -1, 0, -2):
                with self.subTest(page=page, size=size):
                    tokens = grid(size, page)
                    self.assertGreaterEqual(tokens, 8192)
                    self.assertEqual(tokens % page, 0)
                    self.assertLess(tokens - 8192, page)
            self.assertEqual(grid(4 * page, page), 4 * page)
        with self.assertRaisesRegex(RuntimeError, "multiple of page_size"):
            grid(8193, 256)
        with self.assertRaisesRegex(ValueError, "page_size must be positive"):
            grid(-1, 0)

    def test_staging_bootstrap_accepts_disabled_chunking(self):
        cls = _source_class(
            "disaggregation/prefill.py", "PrefillBootstrapQueue", {"__init__", "_abort"}
        )
        grid = _source_function(
            "disaggregation/common/staging_buffer.py", "staging_grid_tokens"
        )
        capacity = _source_function(
            "disaggregation/prefill.py", "max_bootstrap_input_len"
        )
        schedule = SimpleNamespace(chunked_prefill_size=-1)
        cls.__init__.__globals__.update(
            get_parallel=lambda: SimpleNamespace(tp_rank=0, pp_rank=0, pp_size=1),
            is_mla_backend=lambda pool: False,
            envs=SimpleNamespace(
                SGLANG_DISAGG_STAGING_BUFFER=SimpleNamespace(get=lambda: True)
            ),
            get_schedule=lambda: schedule,
            staging_grid_tokens=grid,
            max_bootstrap_input_len=capacity,
        )
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    effective_logical_max_total_num_tokens=65536
                )
            ),
            kv_shard_widening=1,
            token_to_kv_pool_allocator=SimpleNamespace(page_size=256),
        )

        class ManagerInitReached(Exception):
            pass

        def manager_init(self):
            raise ManagerInitReached

        cls._init_kv_manager = manager_init
        args = (None, None, None, None, 0, 8998, 65536, scheduler, None, None)
        for size in (-1, None, 0, 8192):
            with self.subTest(size=size):
                schedule.chunked_prefill_size = size
                with self.assertRaises(ManagerInitReached):
                    cls(*args)
                self.assertEqual(schedule.chunked_prefill_size, size)
        schedule.chunked_prefill_size = 8193
        with self.assertRaisesRegex(RuntimeError, "multiple of page_size"):
            cls(*args)

    def test_unchunked_staging_prefetch_buffer_and_send_use_same_grid(self):
        import numpy as np

        grid = _source_function(
            "disaggregation/common/staging_buffer.py", "staging_grid_tokens"
        )
        segments = _source_function(
            "disaggregation/common/staging_buffer.py", "compute_grid_segments"
        )
        init_buffers = _source_function(
            "disaggregation/common/staging_handler.py", "init_staging_buffers"
        )
        prefetch = _source_function(
            "disaggregation/common/staging_handler.py", "prefetch_staging_reqs"
        )
        strategy_cls = _source_class(
            "disaggregation/common/staging_handler.py", "PrefillStagingStrategy"
        )
        sender_cls = _source_class(
            "disaggregation/common/conn.py",
            "CommonKVSender",
            {
                "send",
                "_prepare_send_indices",
                "_record_transfer_indices",
                "should_send_kv_chunk",
            },
        )
        scheduler_cls = _source_class(
            "disaggregation/prefill.py",
            "SchedulerDisaggregationPrefillMixin",
            {"_send_kv_chunk"},
        )
        page_indices = _source_function("mem_cache/common.py", "kv_to_page_indices")
        page_indices.__globals__.update(torch=torch, np=np)
        state_types = SimpleNamespace(
            **{
                name: name
                for name in (
                    "MAMBA",
                    "QSA_PENDING",
                    "QSA_COMPRESSED",
                    "SWA",
                    "AUXILIARY_SWA",
                    "DSA",
                    "DSA_TAIL",
                    "MINIMAX_INDEX_K",
                    "MINIMAX_DENSE_KV",
                    "SWA_RING",
                    "DSV4_REQUEST_STATE",
                    "REQUEST_ROW",
                    "BLOCK_SCALE",
                    "BLOCK_SCALE_SWA",
                )
            }
        )
        schedule = SimpleNamespace(chunked_prefill_size=-1)
        scheduler_cls._send_kv_chunk.__globals__.update(
            get_schedule=lambda: schedule,
            staging_grid_tokens=grid,
            compute_grid_segments=segments,
            kv_to_page_indices=page_indices,
            torch=torch,
            np=np,
            StateType=state_types,
            _is_npu=False,
        )
        strategy_cls.__init__.__globals__["get_schedule"] = lambda: schedule
        polls = SimpleNamespace(
            Success="success", Failed="failed", Transferring="transferring"
        )
        sender_cls.send.__globals__.update(KVPoll=polls, time=time)

        class Buffer:
            def __init__(self, size, device, gpu_id, **kw):
                self.size = size

            def get_ptr(self):
                return id(self)

            def get_size(self):
                return self.size

        init_buffers.__globals__.update(
            _bare_device_type=lambda device: device,
            _get_custom_mem_pool=lambda device: (None, None),
        )
        modules = {
            "sglang.srt.disaggregation.common.staging_buffer": SimpleNamespace(
                StagingBuffer=Buffer, staging_grid_tokens=grid
            ),
            "sglang.srt.utils.network": SimpleNamespace(
                NetworkAddress=lambda *a: SimpleNamespace(to_tcp=lambda: "tcp://peer:1")
            ),
        }
        with patch.dict(sys.modules, modules):
            for page in (1, 256, 300):
                for size in (-1, None, 0, 4 * page):
                    with self.subTest(page=page, size=size):
                        schedule.chunked_prefill_size = size
                        width = grid(size, page)
                        # D already owns a nonzero prefix; the transfer spans
                        # two complete slots plus a partial final page.
                        prefix, count = 3 * page, 2 * width + page + 1
                        length = prefix + count
                        total_pages = (count + page - 1) // page
                        kv_args = SimpleNamespace(
                            page_size=page, gpu_id=0, kv_item_lens=[2 * page, 3 * page]
                        )
                        registrations = []
                        buffers = init_buffers(
                            lambda *a: registrations.append(a),
                            kv_args,
                            2,
                            size,
                            device_type="cuda",
                        )
                        self.assertEqual(
                            [b.get_size() for b in buffers], [5 * width] * 2
                        )
                        self.assertEqual(len(registrations), 2)
                        messages, requested = [], set()
                        info = SimpleNamespace(
                            is_dummy=False,
                            dst_kv_indices=range(total_pages),
                            endpoint="peer",
                            dst_port=1,
                        )
                        sockets = {
                            "tcp://peer:1": SimpleNamespace(
                                send_multipart=messages.append
                            )
                        }
                        for _ in range(2):
                            prefetch(
                                7,
                                {7: {"peer": info}},
                                {"page_size": page},
                                size,
                                requested,
                                sockets,
                            )
                        self.assertEqual([int(m[2]) for m in messages], [0, 1, 2])
                        expected_pages = [
                            width // page,
                            width // page,
                            total_pages - 2 * (width // page),
                        ]
                        self.assertEqual([int(m[3]) for m in messages], expected_pages)
                        manager = SimpleNamespace(
                            kv_buffer_tensors={"page_size": page},
                            enable_all_cp_ranks_for_transfer=False,
                            kv_shard_size=1,
                            is_dummy_cp_rank=False,
                            prepare_send_state_indices=lambda state: state,
                        )
                        strategy = strategy_cls(manager, None)
                        self.assertEqual(strategy.full_chunk_pages, width // page)
                        sends = []
                        manager.add_transfer_request = (
                            lambda room, indices, index_slice, last, **kw: sends.append(
                                (indices, index_slice, last, kw)
                            )
                        )
                        sender = sender_cls()
                        sender.kv_mgr, sender.bootstrap_room, sender.curr_idx = (
                            manager,
                            7,
                            0,
                        )
                        sender.num_kv_indices, sender.aux_index = total_pages, 9
                        sender._transfer_num_kv_indices = (
                            sender._transfer_num_state_indices
                        ) = 0
                        sender._transfer_start_time = sender._early_send_wait_event = (
                            sender.conclude_state
                        ) = None
                        sender.trace_ctx = SimpleNamespace(copy_for_thread=lambda: None)
                        sender.get_max_transfer_tokens = lambda: None
                        sender.poll = lambda: polls.Transferring
                        req = SimpleNamespace(
                            rid="unchunked",
                            start_send_idx=prefix,
                            disagg_decode_prefix_len=prefix,
                            origin_input_ids=range(length),
                            extend_end=length,
                            kv=SimpleNamespace(req_pool_idx=0),
                            disagg_kv_sender=sender,
                        )
                        scheduler = scheduler_cls()
                        scheduler.enable_staging = True
                        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
                            page_size=page,
                            translate_kv_indices_for_transfer=lambda x: x,
                        )
                        scheduler.req_to_token_pool = SimpleNamespace(
                            req_to_token=torch.arange(length).reshape(1, -1)
                        )
                        scheduler.disagg_metadata_buffers = SimpleNamespace(
                            set_buf=lambda r: None
                        )
                        scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(
                            kv_manager=SimpleNamespace(
                                kv_args=SimpleNamespace(
                                    state_types=[state_types.REQUEST_ROW]
                                )
                            )
                        )
                        scheduler.disagg_prefill_pending_chunk_rids = {req.rid}
                        scheduler._send_kv_chunk(req, last_chunk=True)
                        self.assertEqual([len(s[0]) for s in sends], expected_pages)
                        self.assertEqual([s[2] for s in sends], [False, False, True])
                        self.assertEqual(
                            [s[1].start // strategy.full_chunk_pages for s in sends],
                            [0, 1, 2],
                        )
                        for indices, _, last, extra in sends:
                            self.assertLessEqual(
                                len(indices) * sum(kv_args.kv_item_lens),
                                buffers[0].get_size(),
                            )
                            self.assertEqual(extra["aux_index"], 9 if last else None)
                            if last:
                                np.testing.assert_array_equal(
                                    extra["state_indices"][0], [0]
                                )
                            else:
                                self.assertIsNone(extra["state_indices"])
                        np.testing.assert_array_equal(
                            np.concatenate([s[0] for s in sends]),
                            np.arange(prefix // page, prefix // page + total_pages),
                        )
                        self.assertEqual(req.start_send_idx, length)
                        self.assertNotIn(
                            req.rid, scheduler.disagg_prefill_pending_chunk_rids
                        )

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
        handler._staging_lifecycle = threading.Condition(handler._scatter_lock)
        handler._room_tokens = {}
        handler._inflight_submits = {}
        handler._claimed_chunks = set()
        handler._closing_rooms = set()
        handler._defer_scatter = False
        handler._pending_scatters = deque()
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

    def _whole_prefill_fixture(self, mode):
        import numpy as np

        pl = self.hooks.pre_layerwise
        controller = pl.Controller.__new__(pl.Controller)
        controller.config = pl.Config(mode, Path("/tmp/unused"))
        trace, tasks = [], []

        class Event:
            def record(self, stream):
                trace.append("proof")

        def submit(*args, **kwargs):
            trace.append("send")
            tasks.append((args, kwargs))

        sender = SimpleNamespace(
            bootstrap_room=7,
            kv_mgr=SimpleNamespace(add_transfer_request=submit),
            trace_ctx=SimpleNamespace(copy_for_thread=lambda: "trace"),
        )
        source = SimpleNamespace(
            req=SimpleNamespace(disagg_kv_sender=sender),
            indices=np.array([3, 4]),
            index_slice=slice(2, 4),
            complete=lambda layers: trace.append(tuple(layers)),
            stager=None,
            native_staging=True,
        )
        controller.plan = [(source, [((0, 1), (0, 1, 4, 5)), ((2, 3), (2, 3, 6, 7))])]
        fb = SimpleNamespace(forward_mode="extend", split_index=17)
        self.enterContext(patch.object(torch.cuda, "Event", Event))
        self.enterContext(patch.object(torch.cuda, "current_stream", lambda: None))
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(
                        ForwardMode=SimpleNamespace(SPLIT_PREFILL="split")
                    )
                },
            )
        )
        return SimpleNamespace(controller=controller, fb=fb, trace=trace, tasks=tasks)

    def test_remote_final_chunk_runs_native_once_before_any_group_submission(self):
        f = self._whole_prefill_fixture("remote")
        result, marker = object(), object()

        def compute(runner, fb, positional, **kwargs):
            self.assertIs(positional, marker)
            self.assertEqual(kwargs, {"native_option": True})
            self.assertEqual((fb.forward_mode, fb.split_index), ("extend", 17))
            self.assertFalse(f.tasks)
            f.trace.extend(["layer0", "layer1", "layer2", "layer3", "forward-end"])
            return result

        self.assertIs(
            f.controller.forward(compute, None, f.fb, marker, native_option=True),
            result,
        )
        self.assertEqual(
            f.trace,
            [
                "layer0",
                "layer1",
                "layer2",
                "layer3",
                "forward-end",
                "proof",
                "send",
                "send",
            ],
        )
        self.assertIs(f.tasks[0][1]["wait_event"], f.tasks[1][1]["wait_event"])
        self.assertEqual([t[1]["staging_chunk_idx"] for t in f.tasks], [0, 1])
        self.assertEqual([t[1]["staging_layer_ids"] for t in f.tasks], [(0, 1), (2, 3)])
        self.assertEqual(
            [t[1]["kv_buffer_indices"] for t in f.tasks], [(0, 1, 4, 5), (2, 3, 6, 7)]
        )
        for _, kw in f.tasks:
            kw["on_complete"]()
        self.assertEqual(f.trace[-2:], [(0, 1), (2, 3)])
        self.assertIsNone(f.controller.plan)
        self.assertEqual((f.fb.forward_mode, f.fb.split_index), ("extend", 17))

    def test_remote_failed_final_forward_never_submits_partial_kv(self):
        f = self._whole_prefill_fixture("remote")

        def fail(runner, fb):
            f.trace.append("partial-forward")
            raise RuntimeError("compute failed")

        with self.assertRaisesRegex(RuntimeError, "compute failed"):
            f.controller.forward(fail, None, f.fb)
        self.assertEqual(f.trace, ["partial-forward"])
        self.assertFalse(f.tasks)
        self.assertEqual((f.fb.forward_mode, f.fb.split_index), ("extend", 17))

    def test_layerwise_mode_keeps_compute_send_interleaving(self):
        f = self._whole_prefill_fixture("layerwise")

        def compute(runner, fb, *, split_forward_count):
            self.assertEqual(fb.forward_mode, "split")
            self.assertEqual(split_forward_count, 2)
            f.trace.append("compute")
            return len(f.tasks)

        self.assertEqual(f.controller.forward(compute, None, f.fb), 1)
        self.assertEqual(
            f.trace, ["compute", "proof", "send", "compute", "proof", "send"]
        )
        self.assertIsNot(f.tasks[0][1]["wait_event"], f.tasks[1][1]["wait_event"])
        self.assertEqual(f.fb.forward_mode, "extend")

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
        pool.request_window = None
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
            aux_data_ptrs=[],
            aux_item_lens=[],
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
            dst_kv_item_lens=None,
            dst_aux_ptrs=[],
            dst_aux_item_lens=[],
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
            inflight_steps=0,
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
        scheduler.invariant_checker = SimpleNamespace()
        scheduler.kv_checksum_computer = None
        scheduler.enable_staging = False
        scheduler.sliding_window_size = 128
        scheduler.chunked_prefill_size = 16
        scheduler.disagg_prefill_pending_chunk_rids = set()
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(kv_manager=manager)
        scheduler.disaggregation_mode = "prefill"
        scheduler.req_to_token_pool = SimpleNamespace(
            req_to_token=(torch.arange(length, dtype=torch.int32) + 8).reshape(1, -1)
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            storage=None,
            state=None,
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
            chunked_req=None,
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
            "sglang.srt.disaggregation.mode": SimpleNamespace(
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
        slides = []
        cache.supports_swa = lambda: True
        _bind_window_retirement(cache, lambda req, end: slides.append(end))
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
            guard = install(cache, allow_swa=True, scheduler=self._swa_scheduler())
        return SimpleNamespace(
            cache=cache,
            guard=guard,
            tree=tree,
            leaf=leaf,
            node=node,
            components=components,
            operations=operations,
            slides=slides,
            native=r,
        )

    def _dsv41_destination_cache(self):
        from sglang_remote_attn.destination_cache import install, key

        f = self._dsv41_cache()
        f.cache.checkpoint, f.cache.finish, f.cache.reset = (
            f.guard.checkpoint,
            f.guard.finish,
            f.guard.reset,
        )
        f.cache.page_size = 8
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.radix_cache": SimpleNamespace(
                    RadixCache=f.native.RadixCache
                ),
                "sglang.srt.mem_cache.unified_radix_cache": SimpleNamespace(
                    UnifiedRadixCache=type(f.cache)
                ),
                "sglang.srt.mem_cache.unified_cache.component_type": SimpleNamespace(
                    ComponentType=f.native.ComponentType
                ),
            },
        ):
            f.guard = install(f.cache, allow_swa=True)
        req = SimpleNamespace(
            rid="dsv-cache",
            bootstrap_room=7,
            origin_input_ids=list(range(33)),
            extra_key=None,
            cache_salt=None,
            cache_request_handle=f.native.CacheRequestHandle("dsv-cache", 0),
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=8),
            last_node=1,
            lock=f.cache.lock(1, skip_lock_components=(f.native.ComponentType.SWA,)),
            refresh_fill_ids=lambda: None,
            owned_kv_len=lambda: 33,
        )
        req._remote_attn_decode_match = (
            req.cache_request_handle,
            8,
            key(req),
            req.lock,
        )
        f.req = req
        return f

    def test_dsv41_d_cache_pins_only_full_and_defers_abort_across_tree_split(self):
        """Global/index pages stay pinned; old SWA is evictable and ABORT never inserts."""
        f = self._dsv41_destination_cache()
        lease = f.guard.hold(f.req)
        f.cache.checkpoint(f.req, up_to=33)
        f.cache.finish(f.req.cache_request_handle, f.native.CacheRequestOutcome.ABORT)
        self.assertEqual(f.operations, [])
        parent = f.node(2, 4, f.tree.root_node)
        for component in f.components:
            component.redistribute_on_node_split(parent, f.leaf)
        f.leaf.key, f.leaf.parent = [4, 5, 6, 7], parent
        f.tree.nodes[2] = parent
        self.assertEqual(
            f.tree.component_protected_size_[f.native.ComponentType.FULL], 8
        )
        self.assertEqual(
            f.tree.component_protected_size_[f.native.ComponentType.SWA], 0
        )
        lease.retire(lambda: f.cache.unlock(f.req.lock))
        self.assertEqual(f.operations, ["finish"])
        self.assertTrue(lease.closed and lease.abort)
        for component in f.cache.tree_components:
            self.assertEqual(f.tree.component_protected_size_[component], 0)

    def test_dsv41_d_cache_initialization_binds_common_guard_and_ordered_receiver(self):
        """Native D startup must actually install the guard, including serial control mode."""
        from sglang_remote_attn import pd_mesh, remote_data_plane

        f, cached = self._dsv41_fixture(), self._dsv41_destination_cache()
        f.scheduler.disaggregation_mode = SimpleNamespace(value="decode")
        f.scheduler.tree_cache = cached.cache
        f.scheduler.enable_overlap = False
        receiver = _source_class(
            "managers/scheduler_components/request_receiver.py",
            "SchedulerRequestReceiver",
            strip_method_decorators=True,
        )
        f.scheduler.request_receiver = receiver(
            **{field.name: None for field in dataclasses.fields(receiver)}
        )
        disagg, memory = f.context.get_disagg(), f.context.get_memory()
        disagg.disaggregation_decode_enable_radix_cache = True
        memory.disable_radix_cache = False
        f.scheduler.enable_fpm = True
        control_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        with (
            patch.object(self.runtime, "_CTL_DIR", str(control_dir)),
            patch.object(
                self.runtime, "_create_hub_streams", lambda n: [(None, None, None)] * n
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": f.context,
                    "sglang.srt.mem_cache.radix_cache": SimpleNamespace(
                        RadixCache=cached.native.RadixCache
                    ),
                    "sglang.srt.mem_cache.unified_radix_cache": SimpleNamespace(
                        UnifiedRadixCache=type(cached.cache)
                    ),
                    "sglang.srt.mem_cache.unified_cache.component_type": SimpleNamespace(
                        ComponentType=cached.native.ComponentType
                    ),
                },
            ),
            patch.object(f.context, "get_disagg", lambda: disagg),
            patch.object(f.context, "get_memory", lambda: memory),
            patch.object(
                self.hooks.pre_layerwise.Config,
                "from_env",
                return_value=f.controller.config,
            ),
            patch.object(
                f.ds.DSV41Controller,
                "validate_graphs",
                lambda c: setattr(c, "graph_fast_path", True),
            ),
            patch.object(pd_mesh, "enabled", return_value=False),
            patch.object(remote_data_plane, "enabled", return_value=False),
        ):
            self.hooks.pre_layerwise.initialize(f.scheduler)
        self.addCleanup(self.runtime.close_handoff_hub, f.scheduler)
        ready = json.loads((control_dir / "ready.0.json").read_text())
        self.assertEqual((ready["rank"], ready["tp_size"]), (0, 1))
        controller = f.scheduler._pre_layerwise
        self.addCleanup(controller.decode_overlap.retirement.close)
        self.assertIs(controller.destination_cache, cached.guard)
        self.assertIs(f.runner._pre_layerwise, controller)
        self.assertIs(
            f.scheduler.request_receiver.retire_protocol,
            controller.decode_overlap.retirement,
        )
        self.assertFalse(controller.decode_overlap.serial())

    def test_dsv41_d_cache_pending_certificate_pins_full_without_retry_collectives(
        self,
    ):
        """Admission retries cannot reacquire a different H or hold obsolete SWA."""
        from sglang_remote_attn.destination_cache import match_prefix, send_metadata

        f = self._dsv41_destination_cache()
        f.cache.unlock(f.req.lock)
        f.req.lock, f.req.kv.req_pool_idx = None, None
        controller = SimpleNamespace(
            destination_cache=f.guard,
            tp_size=1,
            dsv41_layerwise=True,
            pool=SimpleNamespace(request_window=None),
        )
        queue = SimpleNamespace(scheduler=SimpleNamespace(_pre_layerwise=controller))

        def native(queue, req):
            req.lock = f.cache.lock(
                1, skip_lock_components=(f.native.ComponentType.SWA,)
            )
            return SimpleNamespace(l1_prefix_len=8, decode_prefix_len=8)

        def unaligned(queue, req):
            native(queue, req)
            return SimpleNamespace(l1_prefix_len=7, decode_prefix_len=7)

        with self.assertRaisesRegex(RuntimeError, "TP prefix match differs"):
            match_prefix(unaligned, queue, f.req)
        self.assertEqual(
            f.tree.component_protected_size_[f.native.ComponentType.FULL], 0
        )

        match_prefix(native, queue, f.req)
        with (
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=AssertionError("retry collective"),
            ),
            patch.object(
                torch.distributed,
                "all_gather_object",
                side_effect=AssertionError("retry gather"),
            ),
        ):
            for _ in range(8):
                f.cache.unlock(f.req.lock)
                f.req.lock = None
                self.assertEqual(
                    match_prefix(
                        lambda *a: self.fail("new lookup"), queue, f.req
                    ).l1_prefix_len,
                    8,
                )
        f.req.kv.req_pool_idx = 0
        sent = []
        send_metadata(
            lambda *a, **kw: sent.append(kw["decode_prefix_len"]),
            queue,
            SimpleNamespace(req=f.req),
            [],
            8,
            decode_prefix_len=8,
        )
        self.assertEqual(sent, [8])
        self.assertFalse(f.guard.pending)
        lease = f.guard.hold(f.req)
        self.assertNotIn(
            f.native.ComponentType.SWA, lease.pin.receipt.component_lock_uuids
        )
        lease.abort = True
        lease.retire(lambda: f.cache.unlock(f.req.lock))
        self.assertEqual(
            f.tree.component_protected_size_[f.native.ComponentType.FULL], 0
        )

    def test_dsv41_d_cache_incremental_sender_reserves_only_suffix_pages(self):
        """Native sender cursors are relative to H, including final odd compressed rows."""
        f = self._dsv41_receive_fixture()
        f.req.disagg_decode_prefix_len = 8
        f.sender.curr_idx, f.sender.num_kv_indices = 1, 4
        path = f.controller.config.directory / "7/0/offer.json"
        offer = self.hooks.pre_layerwise.read_json(path)
        offer["decode_prefix_len"] = 8
        self.hooks.pre_layerwise.write_json(path, offer)
        with patch.dict(sys.modules, f.modules):
            self.assertIsNotNone(f.controller.candidates(f.batch)[0])
            source = f.controller.prepare(f.batch)[0]
        self.assertEqual((source.index_slice.start, source.index_slice.stop), (1, 4))
        self.assertEqual(source.indices.tolist(), [3, 4, 5])
        self.assertEqual(source.decode_prefix_len, 8)
        self.assertTrue(all(msg[7] == b"1" and msg[3] == b"3" for msg in f.requests))

    def test_dsv41_d_cache_scatter_preserves_cached_global_indexer_and_mutable_tail(
        self,
    ):
        """Suffix-relative wire offsets must never overwrite shared global/index pages."""
        f = self._dsv41_receive_fixture(capacity=4096)
        r = f.ring
        r.dr.req.kv.cache_protected_len = 8
        r.dr.req.cache_request_handle = ("scatter", 0)
        r.dr.req._remote_attn_decode_match = (r.dr.req.cache_request_handle, 8)
        r.manager._dsv41_progressive_pool = f.pool
        f.pool.get_extra_key_layout = lambda layer: None
        # Only the final page is sent; its wire cursor is suffix-relative.
        entries = (0, 4)
        sizes = r.manager.kv_args.kv_item_lens
        required = sum(sizes[i] for i in entries)
        r.allocator.buffer.buffer[:required] = 211
        before = [t.clone() for t in f.destination]
        f.rx.scatter(
            None,
            r.handler,
            0,
            3,
            1,
            r.dr,
            r.receiver,
            layer_ids=entries,
            staging_bytes=required,
        )
        # Physical last page=3, cached prefix page=5; neither cached entry is touched.
        for i in entries:
            stride = sizes[i]
            self.assertTrue(
                torch.equal(
                    f.destination[i][5 * stride : 6 * stride],
                    before[i][5 * stride : 6 * stride],
                )
            )
        # N=33 has no completed ratio-2 row in the last page: carry/new rows stay D-owned.
        self.assertTrue(torch.equal(f.destination[0], before[0]))
        self.assertTrue(
            bool((f.destination[4][3 * sizes[4] : 4 * sizes[4]] == 211).all())
        )

        r.dr.req.kv.cache_protected_len = 16
        with self.assertRaisesRegex(RuntimeError, "changed after native metadata"):
            f.rx.scatter(
                None,
                r.handler,
                0,
                3,
                1,
                r.dr,
                r.receiver,
                layer_ids=entries,
                staging_bytes=required,
            )

    def test_dsv41_d_cache_native_match_keeps_complete_swa_tail_fresh(self):
        """A complete cached prompt must still allocate the live SWA window privately."""
        f = self._radix()
        warm, _ = f.request(list(range(16)), "warm-window")
        f.release_kv_cache(warm, f.cache, checkpoint=True)
        req = self._req()
        req.origin_input_ids, req.output_ids = array("q", range(17)), array("q")
        req.extra_key, req.cache_salt = None, None
        req._compute_max_prefix_len = lambda size: size - 1
        native = _source_function("mem_cache/common.py", "match_kv_cache")
        radix_key = _source_function("mem_cache/common.py", "_req_radix_key")
        radix_key.__globals__["RadixKey"] = f.RadixKey
        native.__globals__.update(
            MatchPrefixParams=f.MatchPrefixParams,
            _req_radix_key=radix_key,
            envs=SimpleNamespace(
                SGLANG_RADIX_FORCE_MISS=SimpleNamespace(get=lambda: False)
            ),
        )
        cls = _source_class(
            "disaggregation/decode.py",
            "DecodePreallocQueue",
            {"_match_prefix_and_lock", "_swa_tail_len"},
        )
        cls._swa_tail_len.__globals__["get_disagg"] = lambda: SimpleNamespace(
            disaggregation_decode_enable_radix_cache=True
        )
        cls._match_prefix_and_lock.__globals__["match_kv_cache"] = native
        queue = cls()
        queue._uses_swa_tail_prealloc = lambda: True
        queue.scheduler = SimpleNamespace(sliding_window_size=8)
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=8)
        queue.tree_cache = f.cache
        queue._build_decode_prefix_match = lambda req, result, **kw: result
        result = queue._match_prefix_and_lock(req)
        self.assertEqual(
            result.device_prefix_len, 8
        )  # Native caps a 16-token tree hit.
        self.assertEqual(queue._swa_tail_len(17), 9)
        f.cache.unlock(req.lock)

    def test_dsv41_d_cache_request_window_metadata_uses_native_request_row_indices(
        self,
    ):
        """SWA ring ownership cannot be inferred from the cached FULL page map."""
        from sglang_remote_attn.dsv41_receive import cache_metadata

        cls = _source_class(
            "mem_cache/dsv41_request_window.py",
            "RequestWindow",
            {"row_loc", "page_indices"},
        )
        cls.page_indices.__globals__["torch"] = torch
        window = cls()
        window.page_size, window.capacity = 8, 24
        pool = SimpleNamespace(request_window=window, page_size=8, sliding_window=16)
        controller = SimpleNamespace(dsv41_layerwise=True, pool=pool)
        queue = SimpleNamespace(
            scheduler=SimpleNamespace(_pre_layerwise=controller),
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=["swa", "carry"])
            ),
        )
        dr = SimpleNamespace(
            req=SimpleNamespace(
                origin_input_ids=list(range(41)),
                kv=SimpleNamespace(cache_protected_len=8, req_pool_idx=2),
            )
        )
        carry = object()
        native_payload = [object(), carry]
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                    StateType=SimpleNamespace(SWA="swa", AUXILIARY_SWA="auxiliary_swa")
                )
            },
        ):
            result = cache_metadata(queue, dr, native_payload)
        self.assertEqual(
            result[0].tolist(), [6, 7, 8]
        )  # slot2, positions24/32/40 wrap.
        self.assertIs(result[1], carry)
        self.assertIsNot(result, native_payload)

    def test_dsv41_d_cache_sparse_reads_cached_prefix_locally_before_global_switch(
        self,
    ):
        """A cached selected slot stays Local while only the missing suffix uses P."""
        from sglang_remote_attn import dsv41_sparse_math as math

        f = self._dsv41_remote_support().fixture(ROOT, "cpu")
        calls = []

        def rpc(query, positions, layer, ratio, scale):
            calls.append(positions)
            ids = torch.tensor([f.p_pages[p // 4] * 4 + p % 4 for p in positions])
            keys = math.read_slots(
                f.p_pool.get_extra_key_buffer(layer), ids, 4, f.layout.V41_FP4
            )
            return math.partial(query, keys, scale)

        target = SimpleNamespace(prefix_len=33, decode_prefix_len=8, call=rpc)
        f.controller.remote_targets[0] = target
        selected = f.core.c2_sparse_page_indices[0, :4]
        keys = math.read_slots(
            f.d_pool.get_extra_key_buffer(3), selected, 4, f.layout.V41_FP4
        )
        swa = math.read_slots(
            f.d_pool.get_swa_key_buffer_radix(3),
            torch.tensor([13, 14]),
            8,
            f.layout.V41,
        )
        expected = math.partial(f.query[0], torch.cat((swa, keys)), 512**-0.5, f.sink)[
            0
        ].to(f.query.dtype)
        # Corrupt P's cached page; reading it remotely would change the answer.
        f.p_pool.get_extra_key_buffer(3)[f.p_pages[0]].zero_()
        kv = torch.zeros(1, 512)
        got = f.remote.attention(
            None,
            f.backend,
            f.query,
            kv,
            kv,
            SimpleNamespace(layer_id=3),
            f.fb,
            2,
            attn_sink=f.sink,
        )
        self.assertEqual(calls, [[15, 15]])  # Preserve native repeated top-k entries.
        torch.testing.assert_close(got[0], expected, atol=0.015625, rtol=0.01)

    def test_dsv41_prefix_cache_pin_keeps_full_swa_receipt_valid_across_split(self):
        f = self._dsv41_cache()
        req = SimpleNamespace(
            finished=lambda: False,
            origin_input_ids=range(8),
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
        self.assertEqual(f.slides, [7])
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
            c.cache.checkpoint(f.req, up_to=33)
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
        abort_reason = _source_class("managers/req_types.py", "FINISH_ABORT")
        f.modules["sglang.srt.disaggregation.utils"].prepare_abort = native_abort
        f.modules["sglang.srt.managers.req_types"] = SimpleNamespace(
            FINISH_ABORT=abort_reason
        )
        f.req.finish_outside_step = lambda reason: setattr(
            f.req, "finished_reason", reason
        )
        f.manager._remote_attn_send_lock = threading.Lock()
        f.manager.engine = SimpleNamespace(send_probe=lambda *a: 0)
        native_publish = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"_publish_peer_registration"},
        )._publish_peer_registration
        f.manager.session_lock = threading.Lock()
        f.manager._kv_layout_mismatch = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"_kv_layout_mismatch"},
        )._kv_layout_mismatch.__get__(f.manager)
        f.manager.check_peer_aux_buffer_count = _source_class(
            "disaggregation/common/conn.py",
            "CommonKVManager",
            {"check_peer_aux_buffer_count"},
        ).check_peer_aux_buffer_count.__get__(f.manager)
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
        self.assertTrue(
            all(not layer._forward_hooks for layer in f.runner.model.model.layers)
        )
        trace = []
        source = SimpleNamespace(enqueue=lambda *a, **kw: trace.append((a, kw)))
        f.controller.plan = [source]
        tail = f.runner.model.model.decoder_replay_tail

        def full_forward(runner, fb):
            self.assertIs(runner.model.model.decoder_replay_tail, tail)
            for layer in runner.model.model.layers:
                layer(torch.zeros(1))
            self.assertFalse(trace)
            return "full-forward"

        event = SimpleNamespace(record=lambda stream: None)
        with (
            patch.object(torch.cuda, "Event", lambda: event),
            patch.object(torch.cuda, "current_stream", lambda: None),
        ):
            self.assertEqual(
                f.controller.forward(full_forward, f.runner, f.batch), "full-forward"
            )
        self.assertEqual(trace, [((f.controller.layers - 1, event), {"flush": True})])
        self.assertIs(f.runner.model.model.decoder_replay_tail, tail)
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
        f.runner.attn_backend = backend
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

    def _dsv41_layout_startup_fixture(self):
        layout = self._dsv41_remote_support().fixture(ROOT, "cpu").layout
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=SimpleNamespace(kv_layout=layout.V4),
                model=SimpleNamespace(),
                model_config=SimpleNamespace(
                    num_hidden_layers=40,
                    hf_config=SimpleNamespace(model_type="deepseek_v41"),
                ),
                lora_manager=None,
            ),
            capture_bs=(1, 2, 4),
            enable_torch_compile=False,
        )
        context = SimpleNamespace(
            get_exec=lambda: SimpleNamespace(
                graph=SimpleNamespace(
                    cuda_graph_config=SimpleNamespace(
                        decode=SimpleNamespace(backend="full")
                    ),
                    debug_cuda_graph=False,
                ),
                features=SimpleNamespace(enable_memory_saver=False),
            ),
            get_disagg=lambda: SimpleNamespace(disaggregation_mode="decode"),
            get_parallel=lambda: SimpleNamespace(
                tp_size=2, tp_group=SimpleNamespace(cpu_group=None)
            ),
        )
        stack = self.enterContext(ExitStack())
        stack.enter_context(
            patch.dict(sys.modules, {"sglang.srt.runtime_context": context})
        )
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "HANDOFF_PRE_LAYERWISE": "remote",
                    "HANDOFF_REMOTE_SEGMENTED_GRAPH": "1",
                },
            )
        )
        sg = self.hooks.segmented_graph
        stack.enter_context(
            patch.object(
                sg,
                "_runtime",
                return_value=SimpleNamespace(has_target_states=lambda: False),
            )
        )
        stack.enter_context(
            patch.object(sg.importlib.util, "find_spec", return_value=object())
        )
        from sglang_remote_attn import dsv41_graph

        stack.enter_context(
            patch.object(dsv41_graph, "capture_signature", return_value=(True, ()))
        )
        original = unittest.mock.Mock(return_value=object())
        return SimpleNamespace(sg=sg, runner=runner, layout=layout, original=original)

    def test_dsv41_remote_rejects_unsupported_layout_before_backend_creation_with_or_without_segments(
        self,
    ):
        f = self._dsv41_layout_startup_fixture()
        f.runner.model_runner.token_to_kv_pool.kv_layout = f.layout.V41_FP4
        with patch.object(f.sg, "agree_capture_settings") as agree:
            for option in ("1", "0"):
                with (
                    self.subTest(segmented_graph=option),
                    patch.dict(os.environ, {"HANDOFF_REMOTE_SEGMENTED_GRAPH": option}),
                ):
                    with self.assertRaisesRegex(ValueError, "got kv_layout=v41_fp4"):
                        f.sg.resolve_backend(f.original, f.runner)
                    self.assertEqual(agree.call_args.args[0][-1], "v41_fp4")
                    f.original.assert_not_called()
            # Both V41 arrangements pass the shared guard. Layerwise also keeps
            # the native V4 backend; it does not use the Remote sparse reader.
            for layout, mode in (
                (f.layout.V4, "remote"),
                (f.layout.V41, "remote"),
                (f.layout.V41_NEW, "remote"),
                (f.layout.V4, "layerwise"),
            ):
                with (
                    self.subTest(layout=layout, mode=mode),
                    patch.dict(
                        os.environ,
                        {
                            "HANDOFF_PRE_LAYERWISE": mode,
                            "HANDOFF_REMOTE_SEGMENTED_GRAPH": "0",
                        },
                    ),
                ):
                    f.runner.model_runner.token_to_kv_pool.kv_layout = layout
                    self.assertIs(
                        f.sg.resolve_backend(f.original, f.runner),
                        f.original.return_value,
                    )
            self.assertEqual(f.original.call_count, 4)

    def test_dsv41_remote_layout_disagreement_rejects_all_ranks_before_backend_creation(
        self,
    ):
        f = self._dsv41_layout_startup_fixture()

        def gather(reports, report, group):
            reports[0] = report
            reports[1] = (*report[:-1], "v41" if report[-1] == "v4" else "v4")

        with patch.object(torch.distributed, "all_gather_object", side_effect=gather):
            for layout in (f.layout.V4, f.layout.V41):
                with self.subTest(local_layout=layout):
                    f.runner.model_runner.token_to_kv_pool.kv_layout = layout
                    with self.assertRaisesRegex(RuntimeError, "settings differ"):
                        f.sg.resolve_backend(f.original, f.runner)
                    f.original.assert_not_called()

    def test_dsv41_v4_sparse_reader_matches_native_oracle_and_ignores_padding(self):
        from sglang_remote_attn.dsv41_sparse_math import read_slots

        support = self._dsv41_remote_support()
        f = support.fixture(ROOT, "cpu", kv_layout="v4")
        for page in (1, 3, 4, 64, 128):
            with self.subTest(page_size=page):
                raw = support.encoded(f.layout.V4, 3, page, "cpu")
                slots = torch.tensor([3 * page - 1, 0, page, 1, 0], dtype=torch.int64)
                expected = f.v4_oracle.dequantize_k_cache_paged_ref(
                    raw, slots, page
                ).reshape(-1, 512)
                for cache in (raw, raw.view(torch.bfloat16)):
                    got = read_slots(cache, slots, page, f.layout.V4)
                    self.assertTrue(
                        torch.equal(got.view(torch.int16), expected.view(torch.int16))
                    )
                self.assertEqual(
                    read_slots(
                        raw, torch.tensor([], dtype=torch.int64), page, f.layout.V4
                    ).shape,
                    (0, 512),
                )
                for bad in (-1, 3 * page):
                    with self.assertRaisesRegex(ValueError, "outside"):
                        read_slots(raw, torch.tensor([bad]), page, f.layout.V4)

        raw = support.encoded(f.layout.V4, 1, 4, "cpu")
        # UE8M0 byte 0 follows the native V4 FTZ rule. RoPE signed zero is
        # copied verbatim, and the eighth scale/padding byte is never decoded.
        raw[0, 4 * 576] = 0
        raw[0, 448:450] = torch.tensor([-0.0], dtype=torch.bfloat16).view(torch.uint8)
        got = read_slots(raw, torch.tensor([0]), 4, f.layout.V4)
        expected = f.v4_oracle.dequantize_k_cache_paged_ref(
            raw, torch.tensor([0]), 4
        ).reshape(-1, 512)
        self.assertTrue(torch.equal(got.view(torch.int16), expected.view(torch.int16)))
        self.assertTrue(torch.signbit(got[0, 448]))
        for column, value in ((0, 127), (4 * 576, 255), (448, 192)):
            invalid = raw.clone()
            invalid[0, column] = value
            if column == 448:
                invalid[0, 449] = 127  # BF16 NaN in the RoPE region.
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                read_slots(invalid, torch.tensor([0]), 4, f.layout.V4)
        for invalid in (raw[:, :16], raw.reshape(-1), raw[:, ::2]):
            with self.assertRaisesRegex(ValueError, "shape/stride"):
                read_slots(invalid, torch.tensor([0]), 4, f.layout.V4)

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
        self._check_dsv41_sparse_union("v41")

    def test_dsv41_v4_remote_sparse_union_matches_native_dequantization(self):
        self._check_dsv41_sparse_union("v4")

    def _check_dsv41_sparse_union(self, kv_layout):
        from sglang_remote_attn import dsv41_sparse_math as math

        support = self._dsv41_remote_support()
        f = support.fixture(ROOT, "cpu", kv_layout=kv_layout)
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
                f.p_pool.get_extra_key_buffer(layer), ids, cpp, f.extra_layout
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
            f.main_layout,
        )
        selected = index[0, :4]
        extra = math.read_slots(
            f.d_pool.get_extra_key_buffer(3), selected, 4, f.extra_layout
        )
        if kv_layout == "v4":
            # Expected keys come from the independent native V4 dequantizer,
            # never the plugin reader under test.
            swa = f.v4_oracle.dequantize_k_cache_paged_ref(
                f.d_pool.get_swa_key_buffer_radix(3), torch.tensor([13, 14]), 8
            ).reshape(-1, 512)
            extra = f.v4_oracle.dequantize_k_cache_paged_ref(
                f.d_pool.get_extra_key_buffer(3), selected, 4
            ).reshape(-1, 512)
        keys = torch.cat((swa, extra)).float()
        logits = q[0].float() @ keys.T * 512**-0.5
        logits = torch.cat((logits, f.sink[:, None]), dim=-1)
        expected = (
            torch.softmax(logits, dim=-1) @ torch.cat((keys, torch.zeros(1, 512)))
        ).to(q.dtype)
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

            self.hooks.pre_layerwise.write_json(
                path / "remote-service.json",
                dict(generation="g", prefix_len=33, route={}, boundary="progressive-sparse"),
            )
            remote.commit(lambda *a, **kw: None, queue, dr)
            self.assertFalse(f.controller.remote_targets)
            self.assertTrue(self.hooks._decode_graph(lambda *a: True, runner, f.fb))
            result = json.loads((path / "remote-done.json").read_text())
            self.assertEqual(result["reason"], "native-pd-complete")

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
        lease.chunk = SimpleNamespace(sender=f.sender, req=SimpleNamespace())
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

    def test_dsv41_decode_staging_does_not_require_prefill_scheduler_attribute(self):
        f = self._dsv41_fixture()
        f.scheduler.disaggregation_mode = SimpleNamespace(value="decode")
        del f.scheduler.enable_staging
        f.controller.config = dataclasses.replace(
            f.controller.config, recv_host_staging=True
        )
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}),
            patch("sglang_remote_attn.dsv41_receive.install") as receive,
        ):
            f.controller.validate()
            receive.assert_called_once_with(f.controller, f.manager, "decode")
            # A missing scheduler flag must not bypass the manager's native
            # MHA staging guard.
            f.controller.receive_installed = False
            f.manager.enable_staging = True
            with self.assertRaisesRegex(
                ValueError, "native MHA staging must be disabled"
            ):
                f.controller.validate()

    def test_dsv41_host_staging_mismatch_identifies_entry_and_failed_checks(self):
        f = self._dsv41_fixture()
        self._dsv41_stage_fixture(f)
        stage = f.ds.dsv41_staging
        args = f.manager.kv_args
        # Exercise an indexer entry, whose logical item spans physical rows.
        entry = next(
            i for i, item in enumerate(f.schema["entries"]) if item[0] == "index"
        )
        size, stride = args.kv_data_lens[entry], args.kv_item_lens[entry]
        for bad_size, bad_stride, reason in (
            (size, size + 1, "size_not_divisible_by_item_stride"),
            (size - 1, stride, "registered_size_mismatch"),
            (size, 0, "nonpositive_item_stride"),
            (size, -1, "nonpositive_item_stride"),
            (0, stride, "nonpositive_registered_size"),
        ):
            with self.subTest(size=bad_size, stride=bad_stride):
                args.kv_data_lens[entry], args.kv_item_lens[entry] = (
                    bad_size,
                    bad_stride,
                )
                with self.assertRaisesRegex(ValueError, reason) as error:
                    stage.buffers(f.pool, f.manager)
                message = str(error.exception)
                for text in (
                    f"entry={entry} kind=index ratio=",
                    "buffer_index=",
                    "pointer=0x",
                    "kv_layout=",
                    "page_size=8",
                    "shape=",
                    "dtype=",
                    "device=cuda:0",
                    "expected_device=cuda:0",
                    "tensor_stride=",
                    "contiguous=True",
                    f"tensor_nbytes={size}",
                    f"registered_size={bad_size}",
                    f"item_stride={bad_stride}",
                    f"size_mod_stride={bad_size % bad_stride if bad_stride > 0 else None}",
                ):
                    self.assertIn(text, message)
        args.kv_data_lens[entry], args.kv_item_lens[entry] = size, stride
        self.assertEqual(len(stage.buffers(f.pool, f.manager)), len(args.kv_data_ptrs))

    def test_dsv41_native_index_allocation_stages_complete_terminal_items(self):
        spec = importlib.util.spec_from_file_location(
            "staging_geometry",
            ROOT / "test/manual/remote_attn/dsv41_staging_probe_support.py",
        )
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        core_spec = importlib.util.spec_from_file_location(
            "staging_geometry_core",
            ROOT / "python/sglang/srt/disaggregation/mooncake/pinned_staging_core.py",
        )
        core = importlib.util.module_from_spec(core_spec)
        core_spec.loader.exec_module(core)

        class Boundary:
            device = torch.device("cuda:0")

            def __init__(self, tensor):
                self.tensor = tensor

            def __getattr__(self, name):
                return getattr(self.tensor, name)

            def __getitem__(self, key):
                return self.tensor[key]

            def view(self, *args):
                return Boundary(self.tensor.view(*args))

        manager_cls = _source_class(
            "disaggregation/mooncake/conn.py",
            "MooncakeKVManager",
            {"ensure_host_stager", "set_kv_buffer_tensors"},
        )
        for page_size, full_size in ((128, 640), (256, 619776), (512, 2560)):
            with self.subTest(page_size=page_size):
                pool = support.allocated_pool(
                    ROOT, page_size=page_size, full_size=full_size
                )
                for ratio, index in pool.index_pools.items():
                    self.assertEqual(index.global_page_size, page_size)
                    if page_size == 256 and ratio == 1:
                        legacy_bytes = ((index.size + 64 + 63) // 64) * 4352
                        self.assertEqual(legacy_bytes, 42166528)
                        self.assertEqual(legacy_bytes % 17408, 4352)
                    index.index_k_with_scale_buffer = [
                        Boundary(t) for t in index.index_k_with_scale_buffer
                    ]
                    kv = pool.kv_pools[ratio]
                    kv.kv_buffer = [Boundary(t) for t in kv.kv_buffer]
                ptrs, sizes, strides = pool.get_contiguous_buf_infos()
                manager = manager_cls()
                manager.kv_args = SimpleNamespace(
                    kv_data_ptrs=ptrs, kv_data_lens=sizes, kv_item_lens=strides
                )
                manager._host_staging_all = False
                stager = core.CorePinnedStager(SimpleNamespace(), capacity=257)
                manager._core_stager = stager
                stager.reserve = lambda device: None
                bound = self.hooks.dsv41_layerwise.dsv41_staging.bind(pool, manager)
                self.assertIs(bound, stager)
                for ptr, size, stride in zip(ptrs, sizes, strides):
                    self.assertEqual(size % stride, 0)
                    tail = stager.resolve(ptr + size - stride, stride)
                    self.assertIsNotNone(tail)
                    self.assertEqual(tail.numel(), stride)
                    self.assertIsNone(stager.resolve(ptr + size - stride, stride + 1))
                    if stager.resolve(ptr + size, stride) is not None:
                        self.assertTrue(
                            any(
                                base == ptr + size and capacity >= stride
                                for base, capacity, _, _ in stager.regions
                            )
                        )

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
            self.assertEqual(
                len(f.controller.candidates(f.batch, allow_middle=True)),
                len(f.batch.reqs),
            )
        from sglang_remote_attn.prefill_overlap import SourceLease

        lease = SourceLease.__new__(SourceLease)
        lease.check = lambda: None
        lease.chunk = SimpleNamespace(sender=f.sender, req=SimpleNamespace())
        f.sender.poll = lambda: 0
        # Poison is observed before the outstanding-IO early exit; this method
        # is polled by ordered TP retirement, not by rank-local KV free hooks.
        with self.assertRaisesRegex(RuntimeError, "transport failed"):
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
            f.controller.validate()  # P keeps its existing cache path.
        f.scheduler.disaggregation_mode = SimpleNamespace(value="decode")
        memory = f.context.get_memory()
        memory.disable_radix_cache = False
        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": f.context}),
            patch.object(f.context, "get_memory", lambda: memory),
        ):
            with self.assertRaisesRegex(ValueError, "native decode radix opt-in"):
                f.controller.validate()
            original = f.context.get_disagg
            with patch.object(
                f.context,
                "get_disagg",
                lambda: SimpleNamespace(
                    **vars(original()), disaggregation_decode_enable_radix_cache=True
                ),
            ):
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
            f.controller.hybrid_gdn = True
            f.controller.plan = [object()]
            self.assertTrue(
                self.hooks._prefill_graph(lambda *a: True, runner, f.batch)
            )
            f.controller.plan = None
            f.controller.hybrid_gdn = False
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
                inflight_steps=0,
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
                chunked_req=req,
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
                    "sglang.srt.disaggregation.mode": SimpleNamespace(
                        DisaggregationMode=SimpleNamespace(PREFILL="prefill")
                    ),
                },
            ):
                self.assertIsNone(controller.prepare(batch))
                self.assertEqual(sender.curr_idx, 4)
                batch.chunked_req = None
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

                # D already owns two prompt tokens. Sender cursors and staging
                # slices are relative to that immutable decode-prefix boundary.
                req._pre_layerwise_source = None
                req.disagg_decode_prefix_len = 2
                req.start_send_idx = 4
                sender.curr_idx, sender.num_kv_indices = 2, 7
                controller.cache_guard = None
                offer.update(generation="cached-d", decode_prefix_len=2)
                pl.write_json(offer_path, offer)
                delta = controller.prepare(batch)[0][0]
                self.assertEqual(delta.index_slice, slice(2, 7))
                self.assertEqual(delta.indices.tolist(), [104, 105, 106, 107, 108])
                self.assertEqual(delta.decode_prefix_len, 2)
                delta.complete([0])
                proof = pl.read_json(delta.path / "layer-0.json")
                self.assertEqual((proof["start"], proof["end"]), (2, 9))
                self.assertEqual(sender.curr_idx, 7)

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

    def test_preflight_compares_literal_defaults_by_value_and_type(self):
        import inspect

        kind = inspect.Parameter.KEYWORD_ONLY
        expected = "".join(("thread", "_local"))
        actual = "thread_local".encode().decode()
        self.interfaces.check_parameters(
            "capture", {"mode": (kind, actual)}, {"mode": expected}
        )
        for value in ("global", None, False):
            with self.assertRaisesRegex(RuntimeError, "unexpected default"):
                self.interfaces.check_parameters(
                    "capture", {"mode": (kind, value)}, {"mode": expected}
                )
        with self.assertRaisesRegex(RuntimeError, "unexpected default"):
            self.interfaces.check_parameters(
                "capture", {"flag": (kind, 0)}, {"flag": False}
            )

    def _preflight_checkout(self, root):
        api = self.interfaces
        targets = (
            list(self.hooks.HOOKS)
            + list(api.CORE_CALLS)
            + list(api.CORE_FIELDS)
            + list(api.CORE_ANNOTATIONS)
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
                "sglang.srt.managers.scheduler.Scheduler.should_process_last_batch_early",
                "last_batch",
            ),
            ("sglang.srt.mem_cache.radix_cache.RadixCache.checkpoint", "up_to"),
            (
                "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.checkpoint",
                "up_to",
            ),
            ("sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.finish", "handle"),
            (
                "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.checkpoint_into_tree",
                "swa_checkpoint_start",
            ),
            ("sglang.srt.managers.schedule_batch.ReqKvInfo.swa_live_ranges", "end"),
            ("decode.DecodeTransferQueue._commit_transfer_to_req", "keep_receiver"),
            ("decode.DecodePreallocQueue._match_prefix_and_lock", "req"),
            ("decode.DecodePreallocQueue._pre_alloc", "prefix_indices"),
            ("decode.DecodePreallocQueue._pre_alloc", "swa_tail_start"),
            (
                "mooncake.conn.MooncakeKVManager.build_receiver_metadata_message",
                "decode_prefix_len",
            ),
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
                "on_capture",
            ),
            (
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph.eager_on_graph",
                "output_factory",
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
            ("sglang.srt.managers.schedule_batch.Req.__init__", "pending_bootstrap"),
            ("sglang.srt.managers.schedule_batch.ReqKvInfo", "kv_committed_len"),
            ("sglang.srt.disaggregation.decode.DecodeRequest", "pending_metadata"),
            (
                "sglang.srt.disaggregation.mooncake.conn.KVArgsRegisterInfo",
                "dst_aux_item_lens",
            ),
            ("sglang.srt.disaggregation.base.conn.KVArgs", "state_item_lens"),
            ("sglang.srt.managers.schedule_batch.ScheduleBatch", "chunked_req"),
            ("sglang.srt.managers.schedule_batch.Req.__init__", "inflight_steps"),
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
                with (
                    patch.dict(
                        self.interfaces.CORE_CALLS,
                        {
                            "sglang.srt.disaggregation.decode.DecodeTransferQueue._commit_transfer_to_req": {
                                "keep_receiver": False
                            }
                        },
                        clear=True,
                    ),
                    self.assertRaisesRegex(RuntimeError, "keep_receiver"),
                ):
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
        for target in api.CORE_ANNOTATIONS:
            loaded[target] = type(
                "LoadedAnnotations",
                (),
                {
                    "__annotations__": {
                        name: object
                        for name in api.declared_fields(source.resolve(target))
                    }
                },
            )
        loaded["sglang.srt.environ.envs"] = envs
        self.check.check_runtime_interfaces(resolve=loaded.__getitem__)

        annotation = "sglang.srt.disaggregation.base.conn.KVArgs"
        original = loaded[annotation]
        loaded[annotation] = type(
            "OldKVArgs",
            (),
            {
                "__annotations__": {
                    k: v
                    for k, v in original.__annotations__.items()
                    if k != "state_item_lens"
                }
            },
        )
        with self.assertRaisesRegex(RuntimeError, "state_item_lens"):
            self.check.check_runtime_interfaces(resolve=loaded.__getitem__)
        loaded[annotation] = original

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
        f.registration.dst_aux_item_lens = []
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
            origin_input_ids=list(range(33)),
            bootstrap_room=7,
            finished=lambda: False,
            kv=SimpleNamespace(cache_protected_len=8),
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
        write(
            offer.path / "sparse-ready.json",
            dict(generation="g", prefix_len=33, decode_prefix_len=8),
        )
        write(
            offer.path / "remote-service.json",
            dict(generation="g", prefix_len=33, decode_prefix_len=8, route={}),
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
            f.layout.V4,
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

    def _hybrid_observation_fixture(self):
        from sglang_remote_attn import qwen_hybrid_pd as pd
        from sglang_remote_attn.pre_layerwise import LayerCoverage
        from sglang_remote_attn.retirement import Retirement

        state = SimpleNamespace(refreshes=0, error=None, gpu=True)
        parent = SimpleNamespace(
            closed=False,
            decode_req=SimpleNamespace(
                req=SimpleNamespace(rid="hybrid", bootstrap_room=7)
            ),
            generation="g",
            coverage=LayerCoverage(8, 2),
            rpc=SimpleNamespace(local_entries=set()),
            gpu_ready=lambda: state.gpu,
        )

        def refresh():
            # External receive boundary: receipts can fail between advisory
            # observation and the final retirement vote.
            state.refreshes += 1
            if state.error is not None:
                raise state.error

        parent.refresh = refresh
        resources = [pd.Resource(parent, i, i + 1) for i in range(2)]
        owner = SimpleNamespace(
            rank=0,
            tp_size=1,
            scheduler=SimpleNamespace(),
            config=SimpleNamespace(drain_timeout=0.01),
            targets=resources[:],
            agree=lambda flags: flags,
        )
        protocol = Retirement(owner)
        self.addCleanup(protocol.close)
        for resource in resources:
            protocol.register(resource)
        return state, parent, resources, protocol

    def test_hybrid_observe_snapshot_expires_and_final_gpu_proof_is_fresh(self):
        """Sibling K/V resources share a scan, never a retirement authorization."""
        state, parent, resources, protocol = self._hybrid_observation_fixture()
        protocol.observe()
        self.assertIsNone(protocol.poll_command())
        self.assertEqual(state.refreshes, 1)
        for entry in range(2):
            parent.coverage.complete(entry, 0, 8)
        protocol.observe()
        self.assertEqual(state.refreshes, 2)
        self.assertEqual(parent.rpc.local_entries, set())
        command = protocol.poll_command()
        self.assertEqual(command["kind"], "retire")
        state.gpu = False
        protocol.arm(command)
        protocol.finalize()
        self.assertEqual(parent.rpc.local_entries, set())
        self.assertEqual(protocol.retries, 1)
        state.gpu = True
        protocol.observe()
        protocol.arm(protocol.poll_command())
        before = state.refreshes
        protocol.finalize()
        self.assertGreater(state.refreshes, before)
        self.assertEqual(parent.rpc.local_entries, {0, 1})
        self.assertEqual(protocol.owner.targets, [])

    def test_hybrid_observe_shares_failure_but_does_not_cache_it_across_passes(self):
        """A failed shared refresh poisons all siblings, then can recover."""
        state, parent, resources, protocol = self._hybrid_observation_fixture()
        state.error = RuntimeError("receive failed")
        protocol.observe()
        self.assertEqual(state.refreshes, 1)
        for resource in resources:
            self.assertIn(
                "receive failed", protocol.reports[resource.admission_id][0]["error"]
            )
        state.error = None
        for entry in range(2):
            parent.coverage.complete(entry, 0, 8)
        protocol.observe()
        self.assertEqual(state.refreshes, 2)
        command = protocol.poll_command()
        self.assertEqual(command["kind"], "retire")
        # A new failure after observe must not be masked by its cached success.
        state.error = RuntimeError("late receive failure")
        protocol.arm(command)
        with self.assertRaisesRegex(RuntimeError, "retaining KV"):
            protocol.finalize()
        self.assertEqual(parent.rpc.local_entries, set())
        self.assertFalse(any(r.closed for r in resources))

    def test_hybrid_window_trace_requires_both_kv_and_keeps_bounded_step_history(self):
        """A receipt is not GPU visibility; partial K/V is not a local layer."""
        from sglang_remote_attn.pd_window import WindowTrace
        from sglang_remote_attn.pre_layerwise import LayerCoverage

        with patch(
            "sglang_remote_attn.pd_window.time.monotonic", return_value=10.0
        ) as clock:
            trace = WindowTrace({3: (0, 1), 7: (2, 3)})
            header = dict(length=8, resources=4, base=2, page=1)
            clock.return_value = 10.01
            trace.received(dict(resources=[0, 1], lo=2, hi=6), header)
            self.assertIsNone(trace.snapshot()["layers"]["3"]["receipt_ms"])
            trace.received(dict(resources=[0], lo=0, hi=2), header)
            self.assertIsNone(trace.snapshot()["layers"]["3"]["receipt_ms"])
            clock.return_value = 10.02
            trace.received(dict(resources=[1], lo=0, hi=2), header)
            record = trace.snapshot()["layers"]["3"]
            self.assertAlmostEqual(record["receipt_ms"], 20)
            self.assertIsNone(record["visible_ms"])
            coverage = LayerCoverage(8, 4)
            coverage.complete(0, 0, 8)
            trace.visible(coverage)
            trace.local(0)
            self.assertIsNone(trace.snapshot()["layers"]["3"]["local_ms"])
            clock.return_value = 10.03
            coverage.complete(1, 0, 8)
            trace.visible(coverage)
            clock.return_value = 10.05
            trace.local(1)
            # Repeated receipts/observations must not move the first timestamp.
            trace.received(dict(resources=[0, 1], lo=0, hi=6), header)
            trace.visible(coverage)
            trace.local(1)
            record = trace.snapshot()["layers"]["3"]
            self.assertAlmostEqual(record["receipt_to_visible_ms"], 10)
            self.assertAlmostEqual(record["visible_to_local_ms"], 20)
            for step in range(300):
                trace.layer(3, step < 5)
                trace.layer(7, step < 10)
            result = trace.snapshot()
            self.assertEqual(result["remote_steps"], 10)
            self.assertEqual(result["remote_layer_calls"], 15)
            self.assertEqual(result["layers"]["3"]["remote_after_visible"], 5)
            self.assertEqual(result["layers"]["7"]["remote_after_visible"], 0)
            self.assertEqual(result["observed_steps"], 300)
            self.assertEqual(result["dropped_steps"], 44)
            self.assertEqual(len(result["recent_steps"]), 256)
            self.assertEqual(result["recent_steps"][0]["step"], 45)
            self.assertEqual(result["recent_steps"][-1]["layers_seen"], [3, 7])

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
        self._check_dsv41_progressive_sparse("v41")

    def test_dsv41_v4_progressive_preserves_remote_until_resource_is_local(self):
        self._check_dsv41_progressive_sparse("v4")

    def _check_dsv41_progressive_sparse(self, kv_layout):
        """No whole-batch native attention may read a still-missing global page."""
        from sglang_remote_attn import dsv41_remote as remote
        from sglang_remote_attn import dsv41_sparse_math as math

        f = self._dsv41_remote_support().fixture(ROOT, "cpu", kv_layout=kv_layout)
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
                f.p_pool.get_extra_key_buffer(layer), ids, cpp, f.extra_layout
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
        self._check_dsv41_progressive_retirement(with_cache=False, cancel=True)

    def test_dsv41_d_cache_progressive_graph_overlap_publish_and_cancel_boundaries(
        self,
    ):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                self._check_dsv41_progressive_retirement(with_cache=True, cancel=cancel)

    def _check_dsv41_progressive_retirement(self, *, with_cache, cancel, flash=False):
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
                entries=(
                    [["kv", 4, 1], ["index", 4, 1], ["kv", 128, 2]]
                    if flash
                    else [["kv", 1, 1], ["index", 2, 2], ["kv", 2, 2]]
                )
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
            if with_cache:
                cached = self._dsv41_destination_cache()
                req = cached.req
                req.finished = lambda: False
                owner.destination_cache = cached.guard
            prefix = 8 if with_cache else 0
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
                dict(generation="g", prefix_len=33, decode_prefix_len=prefix, route={}),
            )
            write_json(
                path / "sparse-ready.json",
                dict(generation="g", prefix_len=33, decode_prefix_len=prefix),
            )
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
                if with_cache:
                    from sglang_remote_attn.segmented_graph import RemoteGraphBackend

                    backend = RemoteGraphBackend.__new__(RemoteGraphBackend)
                    backend.supports_remote = lambda batch: batch.bs in (1, 2, 4, 8, 16)
                    graph = SimpleNamespace(
                        model_runner=SimpleNamespace(_pre_layerwise=owner),
                        backend=backend,
                    )
                    context = SimpleNamespace(
                        get_parallel=lambda: SimpleNamespace(tp_rank=0, tp_size=1)
                    )

                    def graph_check(all_local):
                        with (
                            patch.dict(
                                sys.modules, {"sglang.srt.runtime_context": context}
                            ),
                            patch.object(
                                torch.distributed,
                                "all_reduce",
                                side_effect=AssertionError("graph steady-state vote"),
                            ),
                        ):
                            for bs in (1, 2, 4, 8, 16):
                                self.assertTrue(
                                    self.hooks._decode_graph(
                                        lambda *a: True, graph, SimpleNamespace(bs=bs)
                                    )
                                )
                                self.assertEqual(
                                    graph._remote_attn_graph_verdict, (True, all_local)
                                )

                    graph_check(False)
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
                if cancel:
                    if with_cache:
                        cached.cache.finish(
                            req.cache_request_handle,
                            cached.native.CacheRequestOutcome.ABORT,
                        )

                    def release():
                        freed.append("KV")
                        if with_cache:
                            cached.cache.unlock(req.lock)

                    self.assertTrue(
                        self.hooks.pre_layerwise.defer_release(req, release)
                    )
                if with_cache:
                    graph_check(False)
                    self.assertEqual(cached.operations, [])
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
                self.assertEqual(freed, ["KV"] if cancel else [])
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
                if with_cache:
                    self.assertTrue(target.cache_lease.closed)
                    graph_check(True)
                    self.assertEqual(protocol.cache_votes, 1)
                    self.assertEqual(
                        cached.operations, ["finish"] if cancel else ["checkpoint"]
                    )
                    if not cancel:
                        cached.cache.unlock(req.lock)
                    self.assertEqual(
                        cached.tree.component_protected_size_[
                            cached.native.ComponentType.FULL
                        ],
                        0,
                    )

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

    def _dsv41_graph_fixture(self, variants=(None,), **options):
        from sglang_remote_attn import dsv41_graph as dg

        name = "dsv41_graph_probe_support"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "test/manual/remote_attn/dsv41_graph_probe_support.py"
        )
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        f = support.fixture(**options)
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
                fn()  # Native capture_one warms eagerly before entering capture.
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

        def eager(fn, *, on_capture=None, output_factory=None):
            def wrapped(*a, **kw):
                if remote.capturing:
                    if on_capture is not None:
                        on_capture(*a, **kw)
                    result = (
                        output_factory(*a, **kw) if output_factory else fn(*a, **kw)
                    )
                    if output_factory and isinstance(result, torch.Tensor):
                        # CPU fake capture does not execute attention here.
                        # Later fake model ops must not consume empty() bytes.
                        result.zero_()
                    remote.record.append(lambda: fn(*a, **kw))
                    return result
                return fn(*a, **kw)

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
            for size in (32, 16, 8, 4, 2, 1):
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
        self.assertEqual(sorted(b.remote_keys), [1, 2, 4, 8, 16])
        self.assertEqual(sorted(f.full.keys), [1, 2, 4, 8, 16, 32])
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
        self.assertEqual(
            b.manifest()[0], tuple((n, expected) for n in (1, 2, 4, 8, 16))
        )
        f.controller.targets = [object()]
        for n in (1, 2, 4, 8, 16):
            for marker in (3, 7):
                live = f.batch(n, marker)
                with b.execution(live), b.replay_session():
                    b.replay(b.remote_keys[n], live)
                self.assertEqual(f.calls[-3:], [(marker, i) for i in range(3)])
                self.assertEqual(f.prepared[-3:], [(marker, i) for i in range(3)])
                self.assertEqual(f.metadata[-1], marker)
        self.assertIsNone(b.active_batch)
        self.assertIsNone(b.capture_batch)

    def test_dsv41_probe_native_logits_survive_bank_batch_and_sampler_changes(self):
        """Tensor-only probes missed BCG output/import failures and sampler leakage.

        Native capture/replay/output methods and DSV bank selection run on CPU.
        Graph launches are a boundary substitute executing the small CPU model;
        this is not CUDA capture or GPU numerical validation.
        """
        f = self._dsv41_split_core()
        output = f.model(torch.zeros(2, 4), torch.zeros(2), f.batch(2))
        self.assertTrue(dataclasses.is_dataclass(output))
        self.assertEqual(type(output).__name__, "LogitsProcessorOutput")
        self.assertEqual(output.next_token_logits.shape, (2, 4))
        manual = ROOT / "test/manual/remote_attn"
        with patch.object(sys, "path", [str(manual), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "dsv41_output_probe", manual / "dsv41_segmented_graph_probe.py"
            )
            probe = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(probe)
        from sglang_remote_attn.dsv41_graph import DSV41GraphBackend

        mode = type(sys)("sglang.srt.model_executor.runner_utils.capture_mode")
        mode.__dict__.update(
            contextmanager=contextmanager, _capture_attention_variant=None
        )
        probe.definitions(
            ROOT / "python/sglang/srt/model_executor/runner_utils/capture_mode.py",
            mode.__dict__,
        )
        previous_mode = sys.modules.get(mode.__name__)
        sys.modules[mode.__name__] = mode

        def restore_mode():
            if previous_mode is None:
                sys.modules.pop(mode.__name__, None)
            else:
                sys.modules[mode.__name__] = previous_mode

        self.addCleanup(restore_mode)
        capturing = [False]

        class Graph:
            def __init__(self, *args, **kwargs):
                self.work = lambda: None

            def replay(self):
                self.work()

            def reset(self):
                pass

        @contextmanager
        def capture(**kwargs):
            capturing[0] = True
            try:
                yield
            finally:
                capturing[0] = False

        device = SimpleNamespace(
            synchronize=lambda: None,
            graph=capture,
            Event=lambda: SimpleNamespace(
                record=lambda *a: None, synchronize=lambda: None
            ),
            current_stream=lambda: None,
        )

        def eager(fn, *, on_capture=None, output_factory=None):
            def wrapped(*a, **kw):
                if capturing[0]:
                    if on_capture is not None:
                        on_capture(*a, **kw)
                    return output_factory(*a, **kw) if output_factory else fn(*a, **kw)
                return fn(*a, **kw)

            return wrapped

        namespace = dict(
            torch=torch,
            contextmanager=contextmanager,
            BaseCudaGraphBackend=object,
            DedupedCudaGraphMixin=type(
                "Dedup",
                (),
                dict(
                    deduped_cuda_graph=None,
                    begin_cuda_graph_capture=lambda self: None,
                    end_cuda_graph_capture=lambda self: None,
                    close=lambda self: None,
                ),
            ),
            PPProxyTensors=type("Proxy", (), {}),
            GraphPoolPrecarve=lambda: SimpleNamespace(
                measure=nullcontext, mint=lambda: None
            ),
            TorchMemorySaverAdapter=SimpleNamespace(create=lambda **kw: None),
            get_parallel=lambda: SimpleNamespace(
                tp_group=SimpleNamespace(barrier=lambda: None)
            ),
            get_or_create_global_graph_memory_pool=lambda device: None,
            set_graph_pool_id=lambda *a: None,
            graph_pool_capture_scope=nullcontext,
            graph_pool_replay_scope=nullcontext,
            eager_on_graph=eager,
            BreakableCUDAGraph=Graph,
            BreakableCUDAGraphCapture=capture,
            enable_breakable_cuda_graph=nullcontext,
        )
        # Load through the actual CUDA probe entry: a separate test namespace
        # must not accidentally provide the missing dataclasses dependency.
        probe.load_native_backends(ROOT, namespace)
        f.runner.device_module = device
        full = namespace["FullCudaGraphBackend"](f.runner)
        remote = namespace["BreakableCudaGraphBackend"](f.runner)
        backend = DSV41GraphBackend(
            f.runner, full, remote, eager, lambda: capturing[0], lambda: None
        )
        f.attach(backend)
        self.addCleanup(backend.cleanup)
        ShapeKey = probe.load(
            "dsv_output_shape",
            ROOT / "python/sglang/srt/model_executor/runner/shape_key.py",
        ).ShapeKey
        inputs = {}
        with (
            patch.object(torch.cuda, "CUDAGraph", Graph),
            backend.capture_session(None),
        ):
            for size in (32, 16, 8, 4, 2, 1):
                inputs[size] = (torch.zeros(size, 4), torch.zeros(size))
                dummy = f.batch(size, -100)
                backend.capture_one(
                    ShapeKey(size),
                    lambda size=size, fb=dummy: f.model(*inputs[size], fb),
                )
        self.assertEqual(sorted(backend.remote_keys), [1, 2, 4, 8, 16])
        self.assertEqual(f.calls, [])

        def replay_body(bank, key):
            live = backend.active_batch
            x, positions = inputs[key.size]
            if bank is remote:
                # Execute graph projection segments around the real dynamic
                # breaks, preserving their bucket-sized output bridges.
                backend._dynamic("metadata", f.attn)
                for layer in f.layers:
                    q, q_lora, kv = layer.project_decode_qkv(x, positions)
                    backend._dynamic(
                        "state",
                        layer,
                        x=x,
                        q_lora=q_lora,
                        kv=kv,
                        positions=positions,
                        attn_backend=f.attn,
                    )
                    x = (
                        backend._dynamic(
                            "attention",
                            f.attn,
                            q=q,
                            k=q,
                            layer=layer,
                            save_kv_cache=False,
                            compress_ratio=layer.layer_id,
                            attn_sink=None,
                        )
                        + 1
                    )
                result = f.output_type(next_token_logits=x, hidden_states=x + 0.25)
            else:
                result = f.model(x, positions, live)
            # Substitute only the GPU work. The native backend still returns
            # its original capture buffers and the runner processes them.
            for name in ("next_token_logits", "hidden_states"):
                getattr(bank._outputs[key], name).copy_(getattr(result, name))

        for bank in (remote, full):
            for key, graph in bank._graphs.items():
                graph.work = functools.partial(replay_body, bank, key)
        previous = []
        for use_remote, sizes in (
            (True, (16, 3, 13, 5, 6, 7, 14, 15, 8, 4, 1, 2, 16, 3)),
            (False, (16, 8, 4, 1, 2, 16, 8, 32, 32)),
        ):
            f.controller.targets = [object()] if use_remote else []
            before = len(f.calls)
            for step, size in enumerate(sizes):
                live = f.batch(size, step + 1)
                key = backend.remote_key(live) if use_remote else ShapeKey(size)
                x, pos = inputs[key.size]
                x.fill_(step / 8)
                pos.fill_(step % 3)
                expected = x[:size].clone()
                for _ in range(3):
                    expected = expected * 2 + pos[:size, None] + 1
                    if use_remote:
                        expected += live.marker
                # Capture uses ShapeKey, while canonical remote keys preserve
                # native candidate policy through replay.
                with backend.execution(live):
                    output = f.decode_output(key, live)
                self.assertTrue(all(output is not prior for prior in previous))
                bank = remote if use_remote else full
                template = bank._outputs[key]
                self.assertIsInstance(output, f.output_type)
                self.assertIsNot(output, template)
                self.assertEqual(
                    output.next_token_logits.data_ptr(),
                    template.next_token_logits.data_ptr(),
                )
                f.check_output(output, expected)
                self.assertIsNone(template.next_token_logprobs)
                self.assertIsNone(template.sampling_mask_output)
                previous.append(output)
            self.assertEqual(len(f.calls) - before, 3 * len(sizes) if use_remote else 0)
        self.assertEqual(backend.replays, {"remote": 14, "full": 9})

    def test_dsv41_probe_rejects_dirty_sampler_output(self):
        """The probe must reject stale fields even when logits remain correct."""
        f = self._dsv41_split_core()
        expected = torch.ones(2, 4)
        for name in (
            "next_token_logprobs",
            "sampling_mask_output",
            "next_token_sampling_logprobs",
        ):
            output = f.output_type(expected, hidden_states=expected + 0.25)
            setattr(output, name, [object()])
            with self.assertRaisesRegex(AssertionError, "sampler field leaked"):
                f.check_output(output, expected)

    def test_dsv41_segments_graph_policy_uses_shared_lease_and_padded_buckets(
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
                (3, True),
                (4, True),
                (8, True),
                (16, True),
                (12, True),
                (32, False),
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

    def test_dsv41_padded_segments_never_write_dummy_kv_or_carry_rows(self):
        f = self._dsv41_graph_fixture()
        b = f.backend
        f.controller.targets = [object()]
        for size in (3, 5, 6, 7, 13, 14, 15, 3):
            live = f.batch(size, 7)
            for cache, carry in zip(f.cache, f.carry):
                cache.fill_(-99)
                carry.zero_()
            with b.execution(live):
                b.replay(b.remote_key(live), live)
            unused = torch.ones(32, dtype=torch.bool)
            unused[live.out_cache_loc] = False
            for i, (cache, carry) in enumerate(zip(f.cache, f.carry)):
                self.assertTrue(torch.all(cache[unused] == -99))
                self.assertTrue(torch.all(cache[live.out_cache_loc] != -99))
                self.assertTrue(torch.all(carry[unused] == 0))
                self.assertTrue(
                    torch.all(carry[live.out_cache_loc] > 0)
                    if i
                    else torch.all(carry == 0)
                )

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

    def test_dsv41_segments_native_candidate_variants_share_five_remote_captures(self):
        """Context-length variants must preserve native policy without multiplying Remote banks."""
        labels = (
            "candidate_all",
            "candidate_c2_all",
            "candidate_unfiltered",
            "candidate_filtered",
        )
        f = self._dsv41_graph_fixture(labels)
        b = f.backend
        self.assertEqual(len(f.full.known_keys), 24)
        self.assertEqual(len(f.remote.known_keys), 5)
        f.controller.targets = [object()]
        original = f.attn._dsv41_graph_originals["attention"]
        observed = []

        def attention(**kw):
            observed.append(f.variant_state[0])
            return original(**kw)

        f.attn._dsv41_graph_originals["attention"] = attention
        for size in (2, 8, 16):
            for label in labels:
                key = SimpleNamespace(
                    size=size,
                    stream_idx=None,
                    variant_label=None,
                    attention_variant=label,
                )
                fb = f.batch(size, 3)
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

    def test_dsv41_hopper_serial_split_matches_native_state_without_repeating_projection(
        self,
    ):
        # H20 uses the non-fused compressor and serial low-ratio prepare. Keep
        # the source-loaded native projection/state methods, with kernel calls
        # at the existing small fixture boundary.
        for bf16_source in (False, True):
            f = self._dsv41_split_core(
                kv_layout="v4", multi_stream=False, bf16_source=bf16_source
            )
            self.hooks.dsv41_remote.validate_layout(f.mr.token_to_kv_pool)
            for layer in f.layers:
                fb = f.batch(3, 5)
                x, positions = torch.ones(3, 4), torch.tensor([4.0, 5.0, 8.0])
                f.attn.init_forward_metadata_in_graph(fb)
                for buf in f.cache:
                    buf.fill_(-99)
                for buf in f.carry:
                    buf.zero_()
                expected_q = layer._native_prepare(x, positions, fb, f.attn)[0].clone()
                expected_cache, expected_carry = (
                    f.cache[layer.layer_id].clone(),
                    f.carry[layer.layer_id].clone(),
                )
                for buf in f.cache:
                    buf.fill_(-99)
                for buf in f.carry:
                    buf.zero_()
                f.projections.clear()
                q, lora, kv = layer.project_decode_qkv(x, positions)
                projections = list(f.projections)
                layer.update_decode_state(x, lora, kv, positions, fb, f.attn)
                self.assertEqual(f.projections, projections)
                self.assertTrue(torch.equal(q, expected_q))
                self.assertTrue(torch.equal(f.cache[layer.layer_id], expected_cache))
                self.assertTrue(torch.equal(f.carry[layer.layer_id], expected_carry))

    def test_dsv41_hopper_segments_keep_live_state_and_fingerprint_compressor_policy(
        self,
    ):
        from sglang_remote_attn import dsv41_graph as dg

        f = self._dsv41_graph_fixture(kv_layout="v4", multi_stream=False)
        self.hooks.dsv41_remote.validate_layout(f.mr.token_to_kv_pool)
        for layer in f.layers:
            layer._forward_prepare_multi_stream = lambda *a, **kw: self.fail(
                "Hopper entered multi-stream prepare"
            )
            layer._forward_prepare_low_ratio_multi_stream = (
                layer._forward_prepare_multi_stream
            )
            layer.compressor = (
                SimpleNamespace(use_fused_compress=False, use_fused_gate=False)
                if layer.compress_ratio
                else None
            )
        before = dg.capture_signature(f.mr, f.runner)
        self.assertTrue(before[0])
        f.layers[2].compressor.use_fused_compress = True
        self.assertNotEqual(before, dg.capture_signature(f.mr, f.runner))
        f.layers[2].compressor.use_fused_compress = False
        f.controller.targets = [object()]
        for size in (1, 3, 7):
            live = f.batch(size, 7)
            for cache, carry in zip(f.cache, f.carry):
                cache.fill_(-99)
                carry.zero_()
            with f.backend.execution(live):
                f.backend.replay(f.backend.remote_key(live), live)
            unused = torch.ones(32, dtype=torch.bool)
            unused[live.out_cache_loc] = False
            for cache, carry in zip(f.cache, f.carry):
                self.assertTrue(torch.all(cache[unused] == -99))
                self.assertTrue(torch.all(carry[unused] == 0))
            self.assertEqual(f.metadata[-1], 7)

    def test_dsv41_hopper_compressor_support_preserves_sm100_only_index_store(self):
        supported = _source_function(
            "layers/attention/dsv4/dsv41_sparse.py",
            "fused_low_ratio_compress_supported",
        )
        supported.__globals__.update(torch=torch, is_gfx95_supported=lambda: False)
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "hip", None),
        ):
            for major, expected in ((8, False), (9, True), (10, True)):
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=(major, 0)
                ):
                    self.assertEqual(supported(), expected)
        backend_cls = _source_class(
            "layers/attention/deepseek_v4_backend.py",
            "DeepseekV4AttnBackend",
            {"_low_ratio_compress_decode"},
        )
        backend_cls._low_ratio_compress_decode.__globals__.update(
            current_arch=lambda: SimpleNamespace(is_sm_at_least=lambda major: False)
        )
        calls = []
        backend = backend_cls()
        core = SimpleNamespace(
            c1_out_loc=torch.tensor([7, -1]),
            raw_out_loc=torch.tensor([9, 0]),
            c2_out_loc=torch.tensor([3, -1]),
        )
        backend.forward_metadata = SimpleNamespace(core_metadata=core)
        backend._low_ratio_compress_fused = lambda *a, **kw: self.fail(
            "SM100 fused compressor"
        )
        backend._low_ratio_write_group = lambda *a, **kw: calls.append((a, kw))
        kv, score = torch.ones(2, 512), torch.zeros(2, 512)
        compressor = SimpleNamespace(
            use_fused_compress=False, project=lambda x: (kv, score)
        )
        layer = SimpleNamespace(compress_ratio=1, compressor=compressor, layer_id=2)
        positions, req = torch.tensor([7, 8]), torch.tensor([1, 0])
        x = SimpleNamespace(is_cuda=True)
        state = SimpleNamespace(
            kv_score_buffer=SimpleNamespace(kv=kv, score=score, shape=(17, 1024)),
            ring_size=8,
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_attention_compress_states=lambda i: state
        )
        pool_kernel = unittest.mock.Mock(return_value=(kv, positions, core.c2_out_loc))
        with (
            patch.object(torch.version, "cuda", "13.0"),
            patch.dict(
                sys.modules,
                {
                    "sglang.kernels.ops.attention.dsv4.c2_decode_pool": SimpleNamespace(
                        c2_decode_pool=pool_kernel
                    )
                },
            ),
        ):
            backend._low_ratio_compress_decode(layer, x, req, positions)
            self.assertFalse(calls[-1][1]["fuse_index_store"])
            layer.compress_ratio = 2
            backend._low_ratio_compress_decode(layer, x, req, positions)
            self.assertEqual(pool_kernel.call_count, 1)
            self.assertEqual(pool_kernel.call_args.kwargs, {"ring_size": 8})
            self.assertFalse(calls[-1][1]["fuse_index_store"])

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
            if flag == "is_dsv41":
                f.layers[0].compress_ratio = 1  # V4 rejects V4.1-only C1.
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

    def test_dsv_d2d_bulk_uses_registered_mooncake_bytes_without_tcp_payload(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        envelopes = []
        client = d2d.remote.runtime().TargetClient
        call = client.call

        def capture(client, header, *args, **kwargs):
            result = call(client, header, *args, **kwargs)
            if header.get("dsv_d2d") in {"read", "read_group"}:
                response, out, lse = result
                envelopes.append((out.numel() + lse.numel(), response["bulk_bytes"]))
            return result

        with patch.object(client, "call", capture):
            meta = e.freeze()
            e.import_(meta)
            target = e.target.d2d_targets[e.epoch]
            self.assertTrue(e.blocked.wait(3))
            self.assertTrue(e.command(e.target, "d2d_commit")["admitted"])
            self.assertFalse(target.copied)
            e.gate.set()
            e.wait_copy()
        engine = d2d.manager(e.source).engine
        self.assertEqual(meta["bulk_transport"], "mooncake-host-v1")
        self.assertTrue(engine.calls)
        self.assertTrue(envelopes)
        self.assertTrue(all(n == 0 and size > 0 for n, size in envelopes))
        self.assertEqual(len(engine.calls), len(envelopes))
        self.assertTrue(
            all(size <= d2d.LIMIT + 32 for _, sizes in engine.calls for size in sizes)
        )

    def test_dsv_d2d_bulk_unknown_write_quarantines_both_registered_slots(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.import_(e.freeze())
        self.assertTrue(e.blocked.wait(3))
        source = e.source.d2d_sources[e.epoch]
        target = e.target.d2d_targets[e.epoch]
        engine = d2d.manager(e.source).engine
        transfer = engine.batch_transfer_sync

        def failed(*args):
            transfer(*args)  # A negative return does not imply nothing was written.
            return -1

        engine.batch_transfer_sync = failed
        e.gate.set()
        target.worker.join(5)
        self.assertFalse(target.worker.is_alive())
        self.assertIsNotNone(target.error)
        self.assertTrue(source.bulk_sender.unsafe)
        self.assertTrue(target.bulk_reader.bulk.unsafe)
        self.assertIn(source.bulk_sender, engine._d2d_bulk_owners)
        self.assertTrue(engine.regions)
        with self.assertRaises(d2d.TransportFailure):
            e.command(e.target, "d2d_abort")
        with self.assertRaises(ValueError):
            e.command(e.source, "d2d_resume")
        self.assertEqual(e.freed, [])
        self.assertTrue(target.decode_req.req.kv.holds_kv)

    def test_dsv_d2d_bulk_lost_ack_retains_receiver_even_after_successful_write(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.import_(e.freeze())
        self.assertTrue(e.blocked.wait(3))
        target = e.target.d2d_targets[e.epoch]
        call = target.bulk_reader.client.call

        def lost_ack(header, *args, **kwargs):
            result = call(header, *args, **kwargs)
            if header.get("dsv_d2d") in {"read", "read_group"}:
                raise TimeoutError("lost successful bulk ACK")
            return result

        target.bulk_reader.client.call = lost_ack
        e.gate.set()
        target.worker.join(5)
        self.assertFalse(target.worker.is_alive())
        slot = target.bulk_reader.bulk
        self.assertTrue(slot.unsafe)
        self.assertFalse(slot.closed)
        self.assertIn(slot, d2d.manager(e.target).engine._d2d_bulk_owners)
        with self.assertRaises(d2d.TransportFailure):
            e.command(e.target, "d2d_abort")
        self.assertEqual(e.freed, [])

    def test_dsv_d2d_bulk_host_arrival_and_sequence_gate_h2d(self):
        from sglang_remote_attn import d2d_bulk as bulk
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        req = SimpleNamespace()
        slot = bulk.Receiver(d2d.manager(e.target).engine, req, "cpu")
        self.addCleanup(slot.close)
        region = slot.prepare()
        header = dict(
            bulk_protocol=bulk.PROTOCOL,
            bulk_nonce=region["nonce"],
            bulk_sequence=slot.sequence,
            bytes=4,
        )
        slot.tensor.fill_(0)
        slot.tensor[:4] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
        # A successful native send/ACK with no arrival footer is insufficient.
        with self.assertRaisesRegex(TimeoutError, "arrival"):
            slot.consume(header, 4, 0)
        footer = bulk.marker(slot.nonce, slot.sequence, 4)
        slot.tensor[4 : 4 + bulk.FOOTER] = torch.tensor(list(footer), dtype=torch.uint8)
        self.assertEqual(slot.consume(header, 4, 1).tolist(), [1, 2, 3, 4])
        slot.prepare()
        with self.assertRaisesRegex(ValueError, "completion"):
            slot.consume(header, 4, 0)

    def test_dsv_d2d_bulk_rejects_old_wire_before_target_allocation(self):
        e = self._d2d_env()
        meta = e.freeze()
        meta["version"] = "dsv-d2d-v1"
        with self.assertRaisesRegex(ValueError, "epoch/TP"):
            e.import_(meta)
        self.assertFalse(e.target.d2d_targets)

    def test_dsv_d2d_bulk_rejects_replay_or_changed_peer_before_native_write(self):
        from sglang_remote_attn import d2d_bulk as bulk
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.freeze()
        sender = e.source.d2d_sources[e.epoch].bulk_sender
        engine = d2d.manager(e.target).engine
        receiver = bulk.Receiver(
            engine, SimpleNamespace(), "cpu", capacity=bulk.LEGACY_CAPACITY
        )
        self.addCleanup(receiver.close)
        region = receiver.prepare()
        value = torch.arange(16, dtype=torch.uint8)
        ready = d2d.dp.event("cpu")
        sender.write(value, ready, region, 1)
        with self.assertRaisesRegex(ValueError, "sequence"):
            sender.write(value, ready, region, 1)
        with self.assertRaisesRegex(ValueError, "region"):
            sender.write(value, ready, dict(region, base=region["base"] + 1), 2)
        with self.assertRaisesRegex(ValueError, "session"):
            sender.write(value, ready, dict(region, session="changed-peer"), 2)
        self.assertEqual(len(sender.engine.calls), 1)
        self.assertFalse(sender.unsafe)
        large = torch.ones(bulk.LEGACY_CAPACITY + 1, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "size"):
            sender.write(large, ready, region, 2)
        wide = bulk.Receiver(engine, SimpleNamespace(), "cpu")
        self.addCleanup(wide.close)
        wide_region = wide.prepare()
        wide_sender = bulk.Sender(d2d.manager(e.source).engine, SimpleNamespace(), "cpu")
        self.addCleanup(wide_sender.close)
        size = wide_sender.write(large, ready, wide_region, 1)
        received = wide.consume(
            dict(
                bytes=size, bulk_protocol=wide.protocol,
                bulk_nonce=wide.nonce, bulk_sequence=wide.sequence,
            ),
            size,
            1,
        )
        self.assertTrue(torch.equal(received, large))

    def test_dsv_d2d_bulk_failed_deregister_blocks_prepared_free(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.import_(e.freeze())
        self.assertTrue(e.blocked.wait(3))
        e.gate.set()
        target = e.wait_copy()
        slot = target.reader.bulk
        d2d.manager(e.target).engine.batch_deregister = lambda ptrs: -1
        with self.assertRaisesRegex(RuntimeError, "deregistration"):
            e.command(e.target, "d2d_abort")
        self.assertTrue(slot.unsafe)
        self.assertFalse(slot.closed)
        self.assertEqual(e.freed, [])
        with self.assertRaises(d2d.TransportFailure):
            e.command(e.target, "d2d_abort")

    def test_dsv_d2d_bulk_disconnect_cannot_retry_or_release_pending_destination(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.import_(e.freeze())
        self.assertTrue(e.blocked.wait(3))
        source = e.source.d2d_sources[e.epoch]
        target = e.target.d2d_targets[e.epoch]
        engine = d2d.manager(e.source).engine
        peer = d2d.manager(e.target).engine
        client = target.bulk_reader.client
        rt = d2d.remote.runtime()
        started, proceed, completed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original_transfer, original_wait = (
            engine.batch_transfer_sync,
            rt.TargetClient._wait,
        )
        attempts = []

        def transfer(*args):
            started.set()
            try:
                if not proceed.wait(8):
                    raise TimeoutError("test bulk gate")
                return original_transfer(*args)
            finally:
                completed.set()

        def wait(self, fut, deadline):
            if self is client:
                attempts.append(fut.rpc_id)
                if not started.wait(3):
                    raise TimeoutError("test native write did not start")
                raise rt._RemoteDisconnected("injected ambiguous bulk disconnect")
            return original_wait(self, fut, deadline)

        with (
            patch.object(engine, "batch_transfer_sync", transfer),
            patch.object(rt.TargetClient, "_wait", wait),
        ):
            try:
                e.gate.set()
                self.assertTrue(started.wait(3))
                target.worker.join(3)
                self.assertFalse(target.worker.is_alive())
                slot = target.bulk_reader.bulk
                self.assertEqual(len(attempts), 1)
                self.assertEqual(client.stats()["rpc_retried"], 0)
                self.assertTrue(slot.unsafe)
                self.assertFalse(slot.closed)
                self.assertIn(slot.tensor.data_ptr(), peer.regions)
                self.assertTrue(e.command(e.source, "d2d_resume")["pending_reader"])
                with self.assertRaises(d2d.TransportFailure):
                    e.command(e.target, "d2d_abort")
                self.assertEqual(e.freed, [])
                self.assertFalse(source.remote_closed.is_set())
            finally:
                proceed.set()
                self.assertTrue(completed.wait(5))

    def test_dsv_d2d_bulk_failed_copy_retains_gpu_input_owner(self):
        from sglang_remote_attn import d2d_bulk as bulk, dsv_d2d as d2d

        e = self._d2d_env()
        e.freeze()
        sender = e.source.d2d_sources[e.epoch].bulk_sender
        receiver = bulk.Receiver(d2d.manager(e.target).engine, SimpleNamespace(), "cpu")
        self.addCleanup(receiver.close)
        region = receiver.prepare()
        proof = SimpleNamespace()

        def invoke():
            data = torch.arange(32, dtype=torch.uint8)
            ref = weakref.ref(data)
            try:
                sender.write(data, proof, region, 1)
            except RuntimeError:
                pass
            return ref

        with patch.object(
            sender.slot,
            "copy",
            side_effect=RuntimeError("D2H submitted without completion proof"),
        ):
            ref = invoke()
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertIs(sender.copy_owner[0], ref())
        self.assertIs(sender.copy_owner[2], proof)
        self.assertTrue(sender.unsafe)
        self.assertIn(sender, sender.engine._d2d_bulk_owners)
        with self.assertRaises(RuntimeError):
            sender.close()
        self.assertIsNotNone(ref())

    def test_dsv_d2d_bulk_completed_copy_releases_input_owner(self):
        from sglang_remote_attn import d2d_bulk as bulk, dsv_d2d as d2d

        e = self._d2d_env()
        e.freeze()
        sender = e.source.d2d_sources[e.epoch].bulk_sender
        receiver = bulk.Receiver(d2d.manager(e.target).engine, SimpleNamespace(), "cpu")
        self.addCleanup(receiver.close)
        region = receiver.prepare()

        def invoke():
            data = torch.arange(32, dtype=torch.uint8)
            ref = weakref.ref(data)
            sender.write(data, d2d.dp.event("cpu"), region, 1)
            return ref

        ref = invoke()
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(sender.copy_owner)
        self.assertFalse(sender.unsafe)

    def _d2d_env(self, ratio=2, **kwargs):
        import importlib.util
        import sys

        folder = ROOT / "test/manual/remote_attn"
        self.enterContext(
            patch.dict(
                sys.modules,
                {"dsv41_remote_probe_support": self._dsv41_remote_support()},
            )
        )
        for name in ("flash0731_probe_support", "dsv_d2d_probe_support"):
            spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            self.enterContext(patch.dict(sys.modules, {name: module}))
            spec.loader.exec_module(module)
        env = module.Environment(ROOT, "cpu", ratio, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_dsv_d2d_remote_commit_before_bulk_and_source_released_only_after_readers(
        self,
    ):

        e = self._d2d_env()
        meta = e.freeze()
        self.assertEqual(meta["sampling"]["stop_token_ids"], [999])
        e.import_(meta)
        self.assertTrue(e.blocked.wait(3))
        source = e.source.d2d_sources[e.epoch]
        target = e.target.d2d_targets[e.epoch]
        self.assertFalse(target.copied)
        self.assertIsNot(target.bulk_reader, target.reader)
        self.assertFalse(source.poll())
        self.assertEqual(e.command(e.target, "d2d_commit")["admitted"], True)
        self.assertEqual(e.resumed[-1][1][0].output_ids, meta["output_ids"])
        self.assertTrue(source.committed)
        with self.assertRaisesRegex(ValueError, "owns generation"):
            e.command(e.source, "d2d_resume")
        self.assertFalse(e.freed)
        proof = [False]
        target.forward_proofs.append(SimpleNamespace(ready=lambda: proof[0]))
        e.gate.set()
        e.wait_copy()
        self.assertFalse(target.ready_to_finish())
        self.assertFalse(source.poll())
        proof[0] = True
        e.target.drain_targets(required=target)
        deadline = time.monotonic() + 3
        while not source.poll():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        self.assertFalse(e.freed)
        source.retire()
        source.retire()
        self.assertEqual(e.freed, ["source"])
        self.assertFalse(
            target.rpc._sparse_page_maps
            if hasattr(target.rpc, "_sparse_page_maps")
            else {}
        )

    def test_dsv_d2d_native_scheduler_poll_and_shutdown_reap_source_once(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.import_(e.freeze())
        target = e.target.d2d_targets[e.epoch]
        e.command(e.target, "d2d_commit")
        e.gate.set()
        e.wait_copy()
        e.target.drain_targets(required=target)
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.schedule_batch": SimpleNamespace(
                    FINISH_LENGTH=object
                )
            },
        ):
            d2d.drain_sources(e.source)
            d2d.drain_sources(e.source)
        self.assertEqual(e.freed, ["source"])
        self.assertFalse(e.source.sources)
        self.assertFalse(e.target.targets)
        self.assertFalse(e.target.decode_overlap.retirement.registered)

    def test_dsv_d2d_parent_retirement_supersedes_resource_events_safely(self):
        e = self._d2d_env()
        e.import_(e.freeze())
        target = e.target.d2d_targets[e.epoch]
        e.command(e.target, "d2d_commit")
        e.gate.set()
        e.wait_copy()
        # A canonical parent-only command can supersede resource notifications.
        e.target.decode_overlap.retirement.drain(target)
        self.assertTrue(target.closed)
        self.assertFalse(target.events)
        self.assertIsNone(target.copy_stream)
        for resource in target.resources:
            self.assertTrue(resource.ready_to_finish())
            e.target.decode_overlap.retirement.drain(resource)
        self.assertFalse(e.target.targets)
        self.assertFalse(e.target.decode_overlap.retirement.registered)

    def test_dsv_d2d_prepared_cancel_then_source_resume_never_double_frees(self):
        e = self._d2d_env()
        meta = e.freeze()
        e.import_(meta)
        target = e.target.d2d_targets[e.epoch]
        e.gate.set()
        e.wait_copy()
        result = e.command(e.target, "d2d_abort")
        self.assertTrue(result["aborted"])
        e.target.drain_targets(required=target)
        self.assertEqual(e.freed, ["target"])
        self.assertTrue(e.command(e.target, "d2d_abort")["closed"])
        deadline = time.monotonic() + 3
        while not e.command(e.source, "d2d_resume")["resumed"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        self.assertFalse(e.req._handoff_frozen)
        self.assertTrue(e.req.kv.holds_kv)
        self.assertEqual(e.freed, ["target"])
        with self.assertRaisesRegex(ValueError, "aborted"):
            e.command(e.target, "d2d_commit")

    def test_dsv_d2d_rejects_schema_or_epoch_before_allocating_target(self):
        e = self._d2d_env()
        meta = e.freeze()
        bad = copy.deepcopy(meta)
        bad["schema"]["page_size"] += 1
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            e.import_(bad)
        self.assertFalse(e.target.d2d_targets)
        bad = copy.deepcopy(meta)
        bad["epoch"] = "stale"
        with self.assertRaisesRegex(ValueError, "epoch/TP"):
            e.import_(bad)
        e.import_(meta)
        self.assertTrue(e.import_(meta)["duplicate"])

    def test_dsv_d2d_native_state_handoff_preserves_pending_blocks(self):
        from sglang_remote_attn.dsv_d2d import state_rows

        for ratio in (2, 4, 128):
            with self.subTest(ratio=ratio):
                e = self._d2d_env(ratio)
                meta = e.freeze()
                src = state_rows(e.source, e.req, e.prefix)
                e.import_(meta)
                target = e.target.d2d_targets[e.epoch]
                dst = state_rows(e.target, target.decode_req.req, e.prefix)
                self.assertEqual([d for d, _, _ in src], [d for d, _, _ in dst])
                for (_, a, ai), (_, b, bi) in zip(src, dst):
                    self.assertTrue(torch.equal(a[ai.long()], b[bi.long()]))
                self.assertTrue(e.blocked.wait(3))
                self.assertFalse(target.copied)
                self.assertTrue(e.command(e.target, "d2d_commit")["admitted"])
                e.gate.set()
                e.wait_copy()

    def test_dsv_d2d_raw_c128_ring_handoff_uses_native_request_page(self):
        from sglang_remote_attn.dsv_d2d import state_rows

        e = self._d2d_env(128, online_c128=False)
        meta = e.freeze()
        e.import_(meta)
        req = e.target.d2d_targets[e.epoch].decode_req.req
        src = [
            x
            for x in state_rows(e.source, e.req, e.prefix)
            if x[0]["kind"] == "request"
        ]
        dst = [
            x for x in state_rows(e.target, req, e.prefix) if x[0]["kind"] == "request"
        ]
        self.assertEqual(src[0][0]["stride"], 128 * 16 * 4)
        self.assertEqual(src[0][2].tolist(), [0])
        self.assertEqual(dst[0][2].tolist(), [1])
        self.assertTrue(torch.equal(src[0][1][0], dst[0][1][1]))

    def test_dsv_d2d_native_detokenize_header_keeps_frozen_prefix_context(self):
        from sglang.srt.managers.incremental_detokenize import build_detokenize_header

        e = self._d2d_env()
        frozen = list(e.req.output_ids)
        e.import_(e.freeze())
        req = e.target.d2d_targets[e.epoch].decode_req.req
        self.assertEqual(req.send_token_offset, len(frozen))
        self.assertFalse(req.detokenize_header_sent)
        req.origin_input_ids_unpadded = array("q", req.origin_input_ids)
        req.http_worker_ipc = None
        req.sampling_params = SimpleNamespace(
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )
        req.output_ids.append(34)
        header = build_detokenize_header(req, output_prefix_len=req.send_token_offset)
        self.assertEqual(
            list(header.prompt_tail_ids), [*req.origin_input_ids[-5:], *frozen]
        )
        due = _source_class(
            "managers/schedule_batch.py", "Req", {"take_detokenize_header_due"}
        ).take_detokenize_header_due
        self.assertTrue(due(req))
        self.assertFalse(due(req))
        # P4 status/SSE owns handoff output: native detokenizer never sees a
        # second cumulative stream with a fresh local detokenizer state.
        delivered = []
        self.hooks._stream_output(
            lambda owner, reqs: delivered.extend(reqs), None, [req]
        )
        self.assertEqual(delivered, [])
        self.assertEqual(list(req.output_ids[req.send_token_offset :]), [34])

    def test_dsv_d2d_native_prebuilt_replays_pending_token_without_resampling(self):
        e = self._d2d_env()
        e.import_(e.freeze())
        req = e.target.d2d_targets[e.epoch].decode_req.req
        req.skip_radix_cache_insert = False
        req.extend_end = e.prefix
        req.swa_branching_seqlen = None
        checkpoint = _source_function("mem_cache/common.py", "checkpoint_kv_cache")
        cls = _source_class(
            "disaggregation/decode_schedule_batch_mixin.py",
            "ScheduleBatchDisaggregationDecodeMixin",
            {"process_prebuilt"},
        )
        cls.process_prebuilt.__globals__.update(
            checkpoint_kv_cache=checkpoint,
            async_h2d=lambda data, **kw: torch.tensor(data, dtype=kw["dtype"]),
            RelayPayload=SimpleNamespace,
        )
        batch = cls()
        batch.reqs, batch.device = [req], "cpu"
        batch.req_pool_indices = torch.tensor([1])
        checkpoints, relays = [], []
        batch.tree_cache = SimpleNamespace(
            checkpoint=lambda r, **kw: checkpoints.append((r, kw["up_to"]))
        )
        batch.spec_algorithm = SimpleNamespace(build_disagg_draft_input=lambda *a: None)
        outputs = list(req.output_ids)
        batch.process_prebuilt(
            SimpleNamespace(stash=lambda rows, payload: relays.append((rows, payload)))
        )
        self.assertEqual(req.output_ids, outputs)
        self.assertEqual(checkpoints, [(req, e.prefix)])
        self.assertEqual(relays[0][1].bonus_tokens.tolist(), [outputs[-1]])
        self.assertEqual(req.prefix_len, e.prefix)
        self.assertIsNone(batch.input_ids)

    def test_dsv_d2d_quarantine_cannot_bypass_source_or_target_fences(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.import_(e.freeze())
        target = e.target.d2d_targets[e.epoch]
        e.req._remote_attn_transport_quarantine = {"items": []}
        target.decode_req.req._remote_attn_transport_quarantine = {"items": []}
        with self.assertRaisesRegex(d2d.TransportFailure, "quarantine"):
            e.command(e.target, "d2d_abort")
        with self.assertRaisesRegex(ValueError, "owns generation"):
            e.command(e.source, "d2d_resume")
        with self.assertRaises(d2d.TransportFailure):
            e.source.d2d_sources[e.epoch].poll()
        self.assertFalse(e.freed)
        self.assertTrue(e.req.kv.holds_kv)
        e.gate.set()

    def test_dsv_d2d_cancel_waits_for_source_reader_without_false_quarantine(self):
        e = self._d2d_env()
        meta = e.freeze()
        source = e.source.d2d_sources[e.epoch]
        ctx = e.hub.registry.acquire(
            meta["route"]["handoff_id"], meta["route"]["epoch"]
        )
        source.aborted = True
        e.hub.registry.release(ctx.handoff_id, expected_req=e.req)
        self.assertFalse(source.poll())
        self.assertFalse(e.freed)
        e.hub.registry.release_ref(ctx.handoff_id, ctx.owner_epoch)
        self.assertTrue(source.remote_closed.is_set())
        self.assertFalse(source.poll())
        e.req._handoff_transfer_unsafe = True
        from sglang_remote_attn import dsv_d2d as d2d

        with self.assertRaises(d2d.TransportFailure):
            source.poll()
        self.assertFalse(e.freed)

    def test_dsv_d2d_resume_and_cancel_wait_for_real_socket_copy_reader(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        meta = e.freeze()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        actual = d2d.execute

        def execute(job, stream, cache):
            if (
                job.header.get("dsv_d2d") == "read_group"
                or job.header.get("dsv_d2d") == "read"
                and job.header.get("kind") == "global"
                and job.ctx.dsv41_source.schema["entries"][job.header["entry"]][0]
                == "kv"
            ):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("actual D2D read gate")
            return actual(job, stream, cache)

        with patch.object(d2d, "execute", execute):
            e.import_(meta)
            target = e.target.d2d_targets[e.epoch]
            source = e.source.d2d_sources[e.epoch]
            e.gate.set()
            self.assertTrue(entered.wait(3))
            self.assertTrue(e.command(e.source, "d2d_resume")["pending_reader"])
            self.assertFalse(source.poll())
            self.assertTrue(e.command(e.target, "d2d_abort")["pending_reader"])
            self.assertFalse(e.freed)
            release.set()
            target.worker.join(3)
            self.assertFalse(target.worker.is_alive())
        self.assertTrue(e.command(e.target, "d2d_abort")["aborted"])
        e.target.drain_targets(required=target)
        self.assertEqual(e.freed, ["target"])
        self.assertTrue(e.command(e.source, "d2d_resume")["resumed"])
        self.assertTrue(e.req.kv.holds_kv)
        self.assertEqual(e.freed, ["target"])

    def test_dsv_d2d_failed_partial_free_cannot_be_retried_by_abort(self):
        e = self._d2d_env()
        e.import_(e.freeze())
        e.gate.set()
        target = e.wait_copy()
        e.command(e.target, "d2d_abort")
        calls = []

        def partial_free():
            calls.append(True)
            raise RuntimeError("partial native free")

        target.release = partial_free
        with self.assertRaisesRegex(RuntimeError, "partial native free"):
            target.finish()
        with self.assertRaisesRegex(RuntimeError, "refusing abort retry"):
            e.command(e.target, "d2d_abort")
        with self.assertRaisesRegex(RuntimeError, "unproven/failed"):
            target.finish()
        self.assertEqual(calls, [True])

    def test_dsv_d2d_finished_requests_cannot_resume_or_commit_prebuilt(self):
        e = self._d2d_env()
        e.import_(e.freeze())
        target = e.target.d2d_targets[e.epoch]
        e.req.finished = lambda: True
        target.decode_req.req.finished = lambda: True
        with self.assertRaisesRegex(ValueError, "owns generation"):
            e.command(e.source, "d2d_resume")
        with self.assertRaisesRegex(ValueError, "finished"):
            e.command(e.target, "d2d_commit")
        self.assertFalse(e.resumed)
        self.assertFalse(e.freed)
        self.assertFalse(target.admitted)

    def test_dsv_d2d_budget_rejection_has_no_allocation_or_invalid_free(self):
        e = self._d2d_env()
        meta = e.freeze()
        prealloc = e.target.scheduler.disagg_decode_prealloc_queue
        prealloc._prealloc_reservation_fits = lambda *a, **kw: False
        with self.assertRaisesRegex(ValueError, "budget unavailable"):
            e.import_(meta)
        target = e.target.d2d_targets[e.epoch]
        self.assertIsNone(target.decode_req.req.kv.req_pool_idx)
        self.assertFalse(target.allocation_complete)
        self.assertTrue(e.command(e.target, "d2d_abort")["aborted"])
        e.target.drain_targets(required=target)
        self.assertFalse(e.freed)

    def test_dsv_d2d_partial_allocator_failure_is_retained_not_freed_by_length(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        meta = e.freeze()

        def partial(req):
            req.kv.req_pool_idx = 1
            req.kv.kv_committed_len = e.prefix
            raise RuntimeError("native allocation failed before row publication")

        e.target.scheduler.disagg_decode_prealloc_queue._pre_alloc = partial
        with self.assertRaisesRegex(RuntimeError, "native allocation failed"):
            e.import_(meta)
        with self.assertRaisesRegex(d2d.TransportFailure, "partial native"):
            e.command(e.target, "d2d_abort")
        self.assertFalse(e.freed)

    def test_dsv_d2d_tp2_asymmetric_completion_and_quarantine(self):
        worker = ROOT / "test/manual/remote_attn/dsv_d2d_tp_worker.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
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
                outputs = [p.communicate(timeout=45)[0] for p in processes]
                for p in processes:
                    self.assertEqual(p.returncode, 0, "\n".join(outputs))
                rows = [
                    json.loads((root / f"rank{r}.json").read_text()) for r in range(2)
                ]
                self.assertEqual(
                    rows,
                    [
                        dict(
                            tp_partial_remote_waits_for_peer_gpu=True,
                            tp_partial_commit_never_admits=True,
                            reserved_before_freeze=True,
                            tp_finished_window_ranks=[0, 1],
                            tp_pre_sample_reservation=True,
                            tp_precopy_asymmetric_pages_and_exact_bytes=True,
                            tp_preflight_rejected_before_freeze=True,
                            asymmetric_gpu_blocks_both=True,
                            decode_before_copy=True,
                            source_free_count=1,
                            tp_quarantine_retains_both=True,
                        )
                    ]
                    * 2,
                )
            finally:
                for p in processes:
                    if p.poll() is None:
                        p.kill()
                        p.communicate(timeout=5)

    def test_dsv_d2d_freeze_drains_only_a_native_decode_result_once(self):
        from sglang_remote_attn.dsv_d2d import quiesce
        from collections import deque

        req = SimpleNamespace(
            origin_input_ids=[1, 2],
            output_ids=[3],
            kv=SimpleNamespace(kv_committed_len=2),
            finished=lambda: False,
        )
        batch = SimpleNamespace(
            reqs=[req], forward_mode=SimpleNamespace(is_decode=lambda: True)
        )
        calls = []

        def process(b, r):
            calls.append(r)
            req.output_ids.append(4)
            req.kv.kv_committed_len += 1

        scheduler = SimpleNamespace(
            result_queue=deque([(batch, "result")]),
            running_batch=batch,
            last_batch=batch,
            process_batch_result=process,
        )
        quiesce(scheduler, req)
        quiesce(scheduler, req)
        self.assertEqual(calls, ["result"])
        self.assertIsNone(scheduler.last_batch)
        batch.forward_mode.is_decode = lambda: False
        scheduler.result_queue.append((batch, "extend"))
        scheduler.last_batch = batch
        with self.assertRaisesRegex(RuntimeError, "native extend/prebuilt"):
            quiesce(scheduler, req)
        self.assertEqual(len(scheduler.result_queue), 1)

    def test_dsv_d2d_non_native_remote_defers_and_local_owner_retires_after_freeze(
        self,
    ):
        from sglang_remote_attn import dsv_d2d as d2d
        from sglang_remote_attn import dsv_gateway_control as gateway
        from sglang_remote_attn.dsv41_remote import Target

        e = self._d2d_env()
        rpc = Target(
            e.source, e.req, Path(e.tmp) / "previous", "previous", {}, e.prefix
        )
        rpc.progressive = True
        e.req._dsv41_remote = rpc
        e.source.remote_targets[rpc.row] = rpc
        kv_entries = {
            i
            for i, entry in enumerate(
                e.source._schema(d2d.manager(e.source))["entries"]
            )
            if entry[0] == "kv"
        }
        self.assertFalse(gateway.request_status(e.source, e.req)["migration_ready"])
        deferred = d2d.handle(
            e.source,
            dict(
                cmd="d2d_plan",
                rid=e.req.rid,
                destination="D1",
                target_rid="target",
                migration_epoch="not-local",
                growth_tokens=0,
            ),
        )
        self.assertTrue(deferred["deferred"])
        self.assertEqual(
            deferred["migration_deferred_reason"], "remote_entries_pending"
        )
        self.assertFalse(e.source.d2d_plans)
        rpc.local_entries.update(kv_entries)
        self.assertTrue(gateway.request_status(e.source, e.req)["migration_ready"])
        parent = SimpleNamespace(
            closed=False,
            release=None,
            release_failed=False,
            ready_to_finish=lambda: False,
            admission_id=1,
            generation="previous",
            decode_req=SimpleNamespace(req=e.req),
        )
        e.req._pre_layerwise_target = parent
        self.assertFalse(gateway.request_status(e.source, e.req)["migration_ready"])
        parent.ready_to_finish = lambda: True
        e.source.targets.append(parent)
        e.source.decode_overlap.retirement.register(parent)
        retired = []

        def finish(*, wait):
            self.assertTrue(e.req._handoff_frozen)
            rpc.close("previous-owner-retired")
            parent.closed = True
            e.req._pre_layerwise_target = None
            retired.append(True)

        parent.finish = finish
        parent.refresh = lambda: None
        _, reserved = self._d2d_reserve(e)
        data = dict(
            cmd="d2d_freeze",
            rid=e.req.rid,
            destination="D1",
            target_rid="target",
            migration_epoch=e.epoch,
            reservation=reserved["reservation"],
        )
        self.assertTrue(d2d.handle(e.source, data)["pending_retirement"])
        e.source.decode_overlap.retirement.drain(required=parent)
        self.assertIn("transfer", d2d.handle(e.source, data))
        self.assertEqual(retired, [True])
        self.assertTrue(e.req._handoff_frozen)
        self.assertIsNone(e.req._dsv41_remote)
        self.assertNotIn(rpc.row, e.source.remote_targets)

    def _partial_d2d_env(self, concurrency=4):
        # Reuse the established source-loaded native pool/module boundaries.
        bootstrap = self._d2d_env()
        bootstrap.gate.set()
        import importlib.util

        path = ROOT / "test/manual/remote_attn/dsv_d2d_partial_probe_support.py"
        spec = importlib.util.spec_from_file_location("partial_probe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = module.SharedEnvironment(ROOT, concurrency)
        self.addCleanup(env.close)
        return module, env

    def test_dsv_d2d_partial_remote_shared_worker_c4_c8(self):
        """Concurrent incomplete PD requests freeze and migrate before they finish."""
        for concurrency in (4, 8):
            with self.subTest(concurrency=concurrency):
                module, env = self._partial_d2d_env(concurrency)
                result = module.exercise_shared(env)
                self.assertEqual(result["migrations"], concurrency)
                self.assertEqual(result["source_frees"], concurrency)

    def test_dsv_d2d_partial_remote_resume_preserves_native_pd_owner(self):
        from sglang_remote_attn import dsv_d2d as d2d

        _, e = self._partial_d2d_env()
        data, reply = e.plan_freeze(0)
        self.assertTrue(reply["pending_retirement"])
        req, parent = e.requests[0], e.parents[0]
        old_rpc, old_row, history = (
            req._dsv41_remote,
            req.kv.req_pool_idx,
            list(req.output_ids),
        )
        source = e.source.d2d_sources[data["migration_epoch"]]
        result = d2d.handle(
            e.source, dict(cmd="d2d_resume", migration_epoch=source.epoch)
        )
        self.assertTrue(result["resumed"])
        self.assertFalse(req._handoff_frozen)
        self.assertIs(req._pre_layerwise_target, parent)
        self.assertIs(req._dsv41_remote, old_rpc)
        self.assertEqual(req.kv.req_pool_idx, old_row)
        self.assertEqual(req.output_ids, history)
        self.assertIsNone(source.service)
        self.assertFalse(e.freed)
        self.assertIn(req, e.source.scheduler.running_batch.reqs)
        for i in range(e.concurrency):
            e.arrive(i)
        e.complete_pd()
        self.assertFalse(e.freed)

    def test_dsv_d2d_partial_remote_failed_receive_retains_frozen_kv(self):
        from sglang_remote_attn import dsv_d2d as d2d

        _, e = self._partial_d2d_env()
        data, _ = e.plan_freeze(0)
        parent, req = e.parents[0], e.requests[0]
        parent.decode_req.kv_receiver.status = e.KVPoll.Failed
        with self.assertRaisesRegex(RuntimeError, "previous PD lifetime failed"):
            d2d.handle(e.source, data)
        source = e.source.d2d_sources[data["migration_epoch"]]
        with self.assertRaisesRegex(ValueError, "destination owns generation"):
            d2d.handle(e.source, dict(cmd="d2d_resume", migration_epoch=source.epoch))
        self.assertTrue(req._handoff_frozen and req.kv.holds_kv)
        self.assertIsNone(source.service)
        self.assertFalse(e.freed)

    def test_dsv_d2d_worker_status_recovers_only_released_submission_history(self):
        """The submit replay record can prove EOS after native tracking is gone."""
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.source.scheduler.handoff_tracked_reqs.clear()
        e.source.scheduler.running_batch.reqs.clear()
        e.source.d2d_submission_requests = {"submitted": e.req}
        lookup = dict(
            cmd="d2d_worker_status", migration_epoch="submitted", request_rid=e.req.rid
        )
        self.assertIsNone(d2d.handle(e.source, lookup)["request"])
        e.req.finished = lambda: True
        self.assertIsNone(d2d.handle(e.source, lookup)["request"])
        e.req.kv.holds_kv = False
        self.assertIsNone(
            d2d.handle(e.source, dict(lookup, request_rid="other"))["request"]
        )
        terminal = d2d.handle(e.source, lookup)["request"]
        self.assertTrue(terminal["finished"] and terminal["kv_released"])
        self.assertEqual(terminal["output_ids"], [31, 32, 33])

    def test_layerwise_d2d_control_publishes_unsupported_result(self):
        from sglang.srt.runtime_context import get_parallel

        e = self._d2d_env()
        e.source.dsv41_remote = False
        e.source.dsv41_layerwise = True
        control_dir = Path(e.tmp) / "control"
        control_dir.mkdir()
        command = dict(cmd="d2d_worker_status", migration_epoch="status", nonce="first")
        (control_dir / "cmd.0.json").write_text(json.dumps(command))
        with (
            patch.object(self.runtime, "_CTL_DIR", str(control_dir)),
            get_parallel().override(
                tp_rank=0, tp_size=1, pp_rank=0, attn_tp_rank=0, attn_cp_rank=0
            ),
        ):
            controls = self.hooks._broadcast(
                lambda receiver, reqs, local: reqs, None, []
            )
            self.hooks._process_requests(
                lambda scheduler, reqs: None, e.source.scheduler, controls
            )
        result = json.loads((control_dir / "result.0.json").read_text())
        self.assertFalse(result["ok"])
        self.assertEqual(result["nonce"], "first")
        self.assertIn("requires a remote Decode worker", result["error"])

    def test_flash0731_mixed_ratio_page_maps_keep_native_guard_capacities(self):
        from sglang_remote_attn import dsv41_paged as paged

        native = _source_function(
            "mem_cache/deepseek_v4_memory_pool.py", "_num_dsv4_physical_kv_pages"
        )
        native.__globals__["ceil_div"] = lambda a, b: (a + b - 1) // b
        capacities = {r: native(65536 // r, 256 // r, 256) for r in (1, 2, 4, 128)}
        self.assertNotEqual(capacities[4], capacities[128])
        table = torch.arange(256, 768).reshape(1, -1)
        target = SimpleNamespace(
            row=0,
            handle="mixed",
            prefix_len=383,
            decode_prefix_len=0,
            check=lambda: None,
        )
        maps = {}
        for ratio in (1, 2, 4, 128, 4, 128, 1, 2):
            mapping = paged.page_map(
                target, table, 256, capacities[ratio], 384, ratio=ratio
            )
            if ratio in maps:
                self.assertIs(mapping, maps[ratio])
            maps[ratio] = mapping
            ids = torch.tensor([256 // ratio, 2 * 256 // ratio])
            li, re = torch.empty_like(ids), torch.empty_like(ids)
            mapping.partition(ids, torch.tensor([2]), ratio, li, re)
            self.assertEqual(
                re.tolist(), [0, -1] if ratio == 128 else [0, 256 // ratio]
            )
            self.assertEqual(li.tolist(), [-1, 4] if ratio == 128 else [-1, -1])
        with self.assertRaisesRegex(RuntimeError, "storage changed"):
            paged.page_map(target, table, 256, capacities[4] + 1, 384, ratio=4)
        with self.assertRaisesRegex(RuntimeError, "storage changed"):
            paged.page_map(target, table.clone(), 256, capacities[128], 384, ratio=128)
        target.handle = "reused"
        with self.assertRaisesRegex(RuntimeError, "attempt/extent changed"):
            paged.page_map(target, table, 256, capacities[128], 384, ratio=128)

    def test_flash0731_startup_routes_native_backend_and_keeps_full_graph(self):
        from sglang_remote_attn import dsv41_remote as remote, dsv41_layerwise

        f = self._dsv41_layout_startup_fixture()
        cfg = f.runner.model_runner.model_config.hf_config
        cfg.model_type, cfg.compress_ratios = "deepseek_v4", [0, 0, 4, 128]
        self.assertTrue(dsv41_layerwise.is_dsv41(f.runner.model_runner))
        with (
            patch.object(f.sg, "agree_capture_settings") as agree,
            patch.dict(os.environ, {"HANDOFF_REMOTE_SEGMENTED_GRAPH": "0"}),
        ):
            self.assertIs(
                f.sg.resolve_backend(f.original, f.runner), f.original.return_value
            )
            report = agree.call_args.args[0]
            self.assertIn("deepseek_v4", report)
            self.assertIn((0, 0, 4, 128), report)
            self.assertEqual(report[-1], "v4")

        class DeepseekV4AttnBackend:
            is_dsv41 = trtllm_attn = mega_attn = False

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
                **kw,
            ):
                return "native"

        backend = DeepseekV4AttnBackend()
        mr = f.runner.model_runner
        mr.attn_backend = backend
        controller = SimpleNamespace(pool=mr.token_to_kv_pool, remote_targets={})
        remote.validate(controller, mr, "decode")
        self.assertIs(backend._dsv41_remote, controller)
        self.assertEqual(
            backend._forward_attention(
                None,
                None,
                None,
                None,
                SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: True)),
                4,
            ),
            "native",
        )
        backend.trtllm_attn = True
        with self.assertRaisesRegex(ValueError, "flashmla backend"):
            remote.validate(controller, mr, "decode")

    def test_flash0731_segmented_requires_split_abi_before_native_warmup(self):
        from sglang_remote_attn import dsv41_graph

        f = self._dsv41_layout_startup_fixture()
        cfg = f.runner.model_runner.model_config.hf_config
        cfg.model_type, cfg.compress_ratios = "deepseek_v4", [0, 0, 4, 128]
        with (
            patch.object(f.sg, "agree_capture_settings"),
            patch.dict(os.environ, {"HANDOFF_REMOTE_SEGMENTED_GRAPH": "1"}),
            patch.object(dsv41_graph, "capture_signature", return_value=(False, ())),
        ):
            with self.assertRaisesRegex(ValueError, "ABI is incompatible"):
                f.sg.resolve_backend(f.original, f.runner)
            f.original.assert_not_called()

    def test_flash0731_native_c4_and_c128_state_registration_and_pending_ring(self):
        import numpy as np

        f = self._dsv41_fixture()
        cls = type(f.pool.compress_state_pools[0])
        transfer = _source_function(
            "mem_cache/deepseek_v4_compress_state.py",
            "request_scoped_state_transfer_indices",
        )
        transfer.__globals__["np"] = np
        cls.transfer_indices.__globals__["request_scoped_state_transfer_indices"] = (
            transfer
        )
        c4 = SimpleNamespace(
            ratio=4,
            request_scoped=False,
            ring_size=8,
            kv_score_buffer=SimpleNamespace(kv_score=torch.zeros((32, 16))),
        )
        f.pool.indexer_compress_state_pools = [c4]
        for online in (False, True):
            c128 = cls()
            c128.ratio, c128.online, c128.request_scoped, c128.ring_size = (
                128,
                online,
                True,
                1 if online else 128,
            )
            c128.kv_score_buffer = SimpleNamespace(kv_score=torch.zeros((1024, 24)))
            f.pool.compress_state_pools = [c4, c128]
            with patch.dict(
                f.pool.get_request_state_buf_infos.__globals__, {"ONLINE_C128": online}
            ):
                ptrs, _, lens = f.pool.get_request_state_buf_infos()
            self.assertEqual(ptrs, [c128.kv_score_buffer.kv_score.data_ptr()])
            self.assertEqual(lens, [96 if online else 96 * 128])
            ptrs, _, lens = f.pool.get_state_buf_infos()
            self.assertNotIn(c128.kv_score_buffer.kv_score.data_ptr(), ptrs)
            self.assertEqual(ptrs.count(c4.kv_score_buffer.kv_score.data_ptr()), 2)
            self.assertEqual(lens[-2:], [16 * 4 * 8] * 2)
            for end in (0, 1, 127, 128, 129, 255, 256, 257):
                expected = [] if end % 128 == 0 else [3]
                self.assertEqual(
                    f.pool.request_state_transfer_indices(3, end).tolist(), expected
                )

    def test_flash0731_progressive_overlap_cancel_and_normal_retirement(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                self._check_dsv41_progressive_retirement(
                    with_cache=False, cancel=cancel, flash=True
                )

    def test_flash0731_partition_c4_c128_native_metadata_and_tail_ownership(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "flash_support", ROOT / "test/manual/remote_attn/flash0731_probe_support.py"
        )
        support = importlib.util.module_from_spec(spec)
        # The support module shares the source-loaded native metadata fixture.
        with patch.dict(
            sys.modules, {"dsv41_remote_probe_support": self._dsv41_remote_support()}
        ):
            spec.loader.exec_module(support)
        from sglang_remote_attn import dsv41_sparse_math as math, dsv41_paged as paged

        for ratio in (4, 128):
            f = support.flash_fixture(ROOT, "cpu", ratio)
            ids = f.core.sparse_page_indices(ratio)[0]
            length = f.core.sparse_topk_lengths(ratio)
            remote, local = math.partition(
                ids[: int(length[0])].tolist(),
                (f.d_row[0, ::256] // 256).tolist(),
                f.prefix,
                256,
                ratio,
            )
            self.assertEqual(remote, [p for p in f.positions if p < f.prefix // ratio])
            self.assertEqual(len(local), 1)
            if ratio == 128:
                self.assertGreater(len(remote), 512)
            target = SimpleNamespace(
                row=0,
                handle="a",
                prefix_len=f.prefix,
                decode_prefix_len=0,
                check=lambda: None,
            )
            mapping = paged.page_map(
                target,
                f.d_row,
                256,
                f.d_pool.get_extra_key_buffer(3).shape[0],
                f.prefix + ratio,
            )
            li, re = torch.empty_like(ids), torch.empty_like(ids)
            mapping.partition(ids, length, ratio, li, re)
            self.assertEqual(re[re >= 0].tolist(), remote)
            self.assertEqual(li[li >= 0].tolist(), local)

    def test_flash0731_manifest_has_owning_c4_c128_layers_and_no_c128_indexer(self):
        f = self._dsv41_fixture()
        # Reuse real native buffers/strides with the two high-ratio owners.
        f.controller.hf_config.model_type = "deepseek_v4"
        f.controller.hf_config.compress_ratios = [0] * 2 + [128] * 18 + [4] * 20
        f.controller.hf_config.kv_source_layer_ids = []
        f.pool.sources_by_ratio = {4: [20], 128: [2, 8, 14]}
        f.pool.kv_pools = {4: f.pool.kv_pools[1], 128: f.pool.kv_pools[2]}
        f.pool.index_pools = (
            {4: f.pool.index_pools[1]} if 1 in f.pool.index_pools else {}
        )
        # Native registration order includes only buffers actually present.
        tensors = []
        for ratio, pool in f.pool.kv_pools.items():
            tensors.extend(pool.kv_buffer)
            if ratio in f.pool.index_pools:
                tensors.extend(f.pool.index_pools[ratio].contiguous_page_row_buffers())
        f.manager.kv_args.kv_data_ptrs = [t.data_ptr() for t in tensors]
        f.manager.kv_args.kv_item_lens = [t[0].nbytes for t in tensors]
        f.pool.get_contiguous_buf_infos = lambda: (
            f.manager.kv_args.kv_data_ptrs,
            [t.nbytes for t in tensors],
            f.manager.kv_args.kv_item_lens,
        )
        schema = f.ds.manifest(f.pool, f.manager.kv_args, f.controller.hf_config)
        self.assertTrue(any(e[:3] == ["kv", 4, 20] for e in schema["entries"]))
        self.assertTrue(any(e[:3] == ["kv", 128, 2] for e in schema["entries"]))
        self.assertFalse(
            any(e[0] == "index" and e[1] == 128 for e in schema["entries"])
        )

    def test_fast_transfer_priority_is_bounded_and_keeps_native_exclusion(self):
        from sglang_remote_attn.transfer_priority import TransferLock

        lock = TransferLock(urgent_burst=2)
        order, busy = [], [0]
        lock.acquire()

        def run(name, urgent):
            with lock.priority() if urgent else lock:
                busy[0] += 1
                self.assertEqual(busy[0], 1)
                order.append(name)
                busy[0] -= 1

        threads = []
        for name, urgent in (("bulk", False), ("u0", True), ("u1", True), ("u2", True)):
            t = threading.Thread(target=run, args=(name, urgent))
            threads.append(t)
            t.start()
            self._mesh_wait(lambda: len(lock.bulk) + len(lock.urgent) == len(threads))
        lock.release()
        for t in threads:
            t.join(2)
            self.assertFalse(t.is_alive())
        self.assertEqual(order, ["u0", "u1", "bulk", "u2"])

    def test_fast_prefix_pipeline_is_bounded_handles_reordered_completions_and_priority(
        self,
    ):
        f = self._prefix_stream_fixture(dsv=True, inflight=2)
        self.assertEqual(len(f.wire), 2)
        peak, completed = 0, []
        with f.plan.priority():
            # Already submitted tasks can finish; queued bulk cannot replace them.
            while f.wire:
                peak = max(peak, len(f.plan.pump.active))
                task = f.wire.pop()
                completed.append(task.kwargs["staging_chunk_idx"])
                task.kwargs["on_complete"]()
            self.assertTrue(f.plan.pump.bulk)
            self.assertFalse(f.plan.drained())
        while f.wire:
            peak = max(peak, len(f.plan.pump.active))
            task = f.wire.pop()
            completed.append(task.kwargs["staging_chunk_idx"])
            task.kwargs["on_complete"]()
        self.assertLessEqual(peak, 2)
        self.assertTrue(f.plan.drained())
        self.assertEqual(len(set(completed)), len(f.plan.pieces))
        self.assertTrue(f.guard.leases)  # Neither DMA callback frees a read lease.

    def test_fast_prefix_pipeline_failure_does_not_recycle_active_tickets(self):
        f = self._prefix_stream_fixture(dsv=True, inflight=2)
        active = dict(f.plan.pump.active)
        task = f.wire.pop()
        f.req.cache_request_handle = "different-attempt"
        with self.assertRaises(RuntimeError):
            task.kwargs["on_complete"]()
        self.assertTrue(f.plan.pump.failed)
        self.assertEqual(f.plan.pump.active, active)
        self.assertTrue(f.guard.leases)
        f.plan.pump.drive()
        self.assertEqual(f.plan.pump.active, active)

    def test_fast_native_layer_requires_tp_local_and_preserves_shared_producer(self):
        from sglang_remote_attn import dsv41_fast as fast
        from sglang_remote_attn import dsv41_remote as remote
        from sglang_remote_attn import dsv41_paged as paged

        f = self._dsv41_remote_support().fixture(ROOT, "cpu", kv_layout="v4")
        f.controller._schema = lambda m: dict(entries=[["kv", 1, 1], ["kv", 2, 2]])
        f.controller.scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            kv_manager=None
        )
        target = remote.Target(
            f.controller, f.req, Path("/tmp/unused"), "g", {}, 33, progressive=True
        )
        f.controller.remote_targets[0] = target
        kv = torch.zeros(1, 512)
        calls = []

        def native(*args, **kwargs):
            calls.append(True)
            self.assertTrue(kwargs["save_kv_cache"])
            return f.query

        with patch.object(paged, "page_map", side_effect=RuntimeError("still remote")):
            with self.assertRaisesRegex(RuntimeError, "still remote"):
                fast.attention(
                    native,
                    f.backend,
                    f.query,
                    kv,
                    kv,
                    SimpleNamespace(layer_id=3),
                    f.fb,
                    2,
                    attn_sink=f.sink,
                )
            self.assertFalse(calls)
            target.local_entries.add(1)  # Consumer layer 3 shares producer layer 2.
            result = fast.attention(
                native,
                f.backend,
                f.query,
                kv,
                kv,
                SimpleNamespace(layer_id=3),
                f.fb,
                2,
                attn_sink=f.sink,
            )
            self.assertIs(result, f.query)
            self.assertEqual(calls, [True])

    def test_fast_page_map_extension_preserves_duplicates_cached_prefix_and_odd_tail(
        self,
    ):
        from sglang_remote_attn import dsv41_paged as paged

        table = torch.tensor([[16, 17, 18, 19, 4, 5, 6, 7, 24, 25, 26, 27]])
        target = SimpleNamespace(
            row=0, handle="a", prefix_len=9, decode_prefix_len=4, check=lambda: None
        )
        m = paged.page_map(target, table, 4, 8, 8)
        self.assertIs(paged.page_map(target, table, 4, 8, 10), m)
        ids = torch.tensor([8, 3, 3, 12, -1])
        local, remote = torch.empty_like(ids), torch.empty_like(ids)
        m.partition(ids, torch.tensor([5]), 2, local, remote)
        self.assertEqual(remote.tolist(), [-1, 3, 3, -1, -1])
        self.assertEqual(local.tolist(), [8, -1, -1, 12, -1])
        with self.assertRaisesRegex(RuntimeError, "extent changed"):
            m.extend(4)
        with self.assertRaisesRegex(ValueError, "outside request"):
            m.partition(torch.tensor([0]), torch.tensor([1]), 2, local[:1], remote[:1])
        target.handle = "b"
        with self.assertRaisesRegex(RuntimeError, "attempt/extent"):
            m.extend(10)

    def test_fast_batched_attention_matches_union_and_reduces_five_rpcs_to_two(self):
        from sglang_remote_attn import dsv41_fast as fast
        from sglang_remote_attn import dsv41_remote as remote
        from sglang_remote_attn import dsv41_paged as paged

        net = self._gpu_transport_fixture()
        f = self._dsv41_remote_support().fixture(ROOT, "cpu", kv_layout="v4")
        f.controller.gpu_data_plane = net.dd
        f.controller._schema = lambda m: dict(entries=[["kv", 1, 1], ["kv", 2, 2]])
        f.controller.scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            kv_manager=None
        )
        count = 5
        f.controller.scheduler.req_to_token_pool.req_to_token = f.d_row.repeat(count, 1)
        f.fb.req_pool_indices = torch.arange(count)
        f.fb.req_pool_indices_cpu = list(range(count))
        f.fb.seq_lens = torch.full((count,), 34)
        for name in ("swa_page_indices", "c2_sparse_page_indices"):
            setattr(f.core, name, getattr(f.core, name).repeat(count, 1))
        for name in ("swa_topk_lengths", "c2_sparse_topk_lengths"):
            setattr(f.core, name, getattr(f.core, name).repeat(count))
        for i in range(count):
            req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=i))
            f.controller.remote_targets[i] = remote.Target(
                f.controller,
                req,
                Path("/tmp/unused"),
                str(i),
                net.route,
                33,
                progressive=True,
            )
        query = f.query.repeat(count, 1, 1)
        kv = torch.zeros(count, 512)
        got = fast.attention(
            lambda *a, **k: self.fail("unready native read"),
            f.backend,
            query,
            kv,
            kv,
            SimpleNamespace(layer_id=3),
            f.fb,
            2,
            attn_sink=f.sink,
        )
        batches = f.controller.dsv41_rpc_batches
        self.addCleanup(batches.close)
        local = paged.partial(
            query[0],
            [
                (
                    f.d_pool.get_swa_key_buffer_radix(3),
                    torch.tensor([13, 14]),
                    8,
                    f.main_layout,
                ),
                (
                    f.d_pool.get_extra_key_buffer(3),
                    torch.tensor([24]),
                    4,
                    f.extra_layout,
                ),
            ],
            512**-0.5,
            f.sink,
        )
        expected = paged.merge(
            local[:2], (query[0].float() * 3, torch.zeros(query.shape[1])), query.dtype
        )
        torch.testing.assert_close(got, expected[None].repeat(count, 1, 1))
        self.assertEqual(batches.stats["batches"], 2)
        self.assertEqual(batches.stats["rows"], 5)
        self.assertEqual(len(net.d.calls), 2)
        self.assertEqual(len(net.p.calls), 2)
        self.assertTrue(
            all(net.dp.ready(t.req) for t in f.controller.remote_targets.values())
        )
        self.assertEqual(f.stores, [3])

    def test_fast_queued_rpc_holds_io_before_worker_and_unknown_dma_poison_is_sticky(
        self,
    ):
        from sglang_remote_attn import dsv41_rpc as rpc
        from sglang_remote_attn import dsv41_remote as remote

        f = self._gpu_transport_fixture()
        controller = SimpleNamespace(gpu_data_plane=f.dd, remote_targets={})
        batch = rpc.pool(controller)
        self.addCleanup(batch.close)
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))
        target = remote.Target(controller, req, Path("/tmp/unused"), "g", f.route, 8)
        row = rpc.Row(0, target, torch.ones(2, 8, dtype=torch.bfloat16), [0], 1, 1, 1.0)
        started, proceed = threading.Event(), threading.Event()

        def delayed(*args):
            started.set()
            if not proceed.wait(3):
                raise TimeoutError("compute gate")
            return f.compute(*args)

        with patch.object(f.dsv, "execute", delayed):
            pending = batch.submit([row])
            self.assertFalse(f.dp.ready(req))
            self.assertTrue(started.wait(2))
            with self.assertRaisesRegex(RuntimeError, "has not drained"):
                target.close("cancel")
            proceed.set()
            self.assertEqual(pending.result()[0][0], 0)
        self.assertTrue(f.dp.ready(req))
        f.d.fail = True
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            batch.submit([row]).result()
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            f.dp.ready(req)
        before = len(f.d.calls)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            batch.submit([row])
        self.assertEqual(len(f.d.calls), before)

    def _gpu_transport_fixture(self):
        """Real control sockets and tensor copies; only TE/CUDA boundaries are CPU."""
        from sglang_remote_attn import dsv41_remote
        from sglang_remote_attn import handoff_control as rt
        from sglang_remote_attn import remote_data_plane as dp

        self.enterContext(patch.dict(os.environ, {"HANDOFF_REMOTE_GPU_TRANSFER": "1"}))

        class Engine:
            peers = {}
            protocol = "tcp"
            d2d_protocol_verified = True

            def __init__(self, name):
                self.name, self.regions, self.calls, self.fail = name, {}, [], False
                self.peers[name] = self
                self.batches = {}
                self.next_batch = 0

            def get_session_id(self):
                return self.name

            def batch_register(self, pointers, sizes):
                self.regions.update(zip(pointers, sizes))
                return 0

            def batch_deregister(self, pointers):
                for p in pointers:
                    self.regions.pop(p)
                return 0

            def batch_transfer_sync(self, *args):
                raise AssertionError("Q/O must not use hidden-retry sync API")

            def batch_transfer_async_write(self, session, sources, targets, sizes):
                self.calls.append(list(zip(sources, targets, sizes)))
                self.next_batch += 1
                self.batches[self.next_batch] = -1 if self.fail else 0
                if self.fail:
                    return self.next_batch
                peer = self.peers[session]
                for a, b, n in zip(sources, targets, sizes):
                    assert any(
                        p <= a and a + n <= p + s for p, s in self.regions.items()
                    )
                    assert any(
                        p <= b and b + n <= p + s for p, s in peer.regions.items()
                    )
                    ctypes.memmove(b, a, n)
                return self.next_batch

            def get_batch_transfer_status(self, ids):
                return max((self.batches.pop(bid) for bid in ids), key=abs)

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
        client.connect()
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
        new.connect()
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
        self.assertIn(old.wire.data_ptr(), f.d.regions)

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
        client = f.dd.client(f.route)
        client.connect()
        lane = dp.Lane(f.pp, client.slab.descriptor(), "D")
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

    def _d2d_reserve(self, e, growth=0):
        from sglang_remote_attn import dsv_d2d as d2d

        e.epoch = "reserved-migration"
        plan = d2d.handle(
            e.source,
            dict(
                cmd="d2d_plan",
                rid="source",
                destination="D1",
                target_rid="target",
                migration_epoch=e.epoch,
                growth_tokens=growth,
            ),
        )["plan"]
        reserved = d2d.handle(
            e.target,
            dict(
                cmd="d2d_reserve",
                migration_epoch=e.epoch,
                plan=plan,
            ),
        )
        return plan, reserved

    def _d2d_reserved_freeze(self, e, reserved):
        from sglang_remote_attn import dsv_d2d as d2d

        return d2d.handle(
            e.source,
            dict(
                cmd="d2d_freeze",
                rid="source",
                destination="D1",
                target_rid="target",
                migration_epoch=e.epoch,
                reservation=reserved["reservation"],
            ),
        )["transfer"]

    def _precopy_support(self):
        # Sampler/allocator ABI fixtures; native pool layouts and actual byte TE.
        folder = str(ROOT / "test/manual/remote_attn")
        self.enterContext(patch.object(sys, "path", [folder] + sys.path))
        return importlib.import_module("d2d_precopy_probe_support")

    def test_dsv_d2d_precopy_live_prefix_sealed_tail_and_mutable_state(self):
        """Closed pages survive source continuation; freeze copies only the suffix."""
        from sglang_remote_attn import d2d_precopy as pc
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        p = self._precopy_support()
        common, plan = p.reserve(e)
        available, receipts = p.copy_available(e, "precopy", retry=False)
        self.assertEqual(available, [4] * len(plan["schema"]["entries"]))
        self.assertFalse(e.req._handoff_frozen)
        self.assertIn(e.req, e.source.scheduler.running_batch.reqs)
        # A following forward changes the open tail and mutable carry.
        e.req.output_ids.extend([34, 35])
        e.req.kv.kv_committed_len = e.req.kv.kv_allocated_len = 35
        for _, table, indices in d2d.state_rows(e.source, e.req, 35):
            table.index_fill_(0, indices.long(), 77)
        with patch.dict(os.environ, {"HANDOFF_D2D_TOKEN_RANGE": "16"}):
            meta = p.command(
                e.source,
                "precopy",
                "freeze",
                reservation=plan,
                precopy_receipts=[receipts],
                **common,
            )["transfer"]
        self.assertEqual(meta["precopy"]["prefix_pages"], available)
        target = None
        reads = []
        actual = d2d.Reader.call

        def read(reader, action, **kw):
            if action == "read" and kw.get("kind") == "global":
                reads.append((kw["entry"], kw["start"], kw["count"]))
            if action == "read_group":
                reads.extend(
                    (x["entry"], x["start"], x["count"]) for x in kw["segments"]
                )
            return actual(reader, action, **kw)

        with patch.object(d2d.Reader, "call", read):
            target = p.finish(e, "precopy", meta)
        self.assertTrue(reads)
        self.assertTrue(all(lo >= 4 for _, lo, _ in reads), reads)
        self.assertTrue(target.closed)
        self.assertFalse(
            any(not s.closed for s in pc.bulk.owners(d2d.manager(e.source).engine))
        )

    def test_dsv_d2d_precopy_packed_rdma_seals_exact_ranges_and_replays(self):
        """Distinct row widths share one transfer and seal with the frozen tail."""
        from contextlib import nullcontext

        from sglang_remote_attn import d2d_precopy as pc
        from sglang_remote_attn import dsv_d2d as d2d
        from sglang_remote_attn import mooncake_backend as transport

        e = self._d2d_env()
        p = self._precopy_support()
        common, plan = p.reserve(e)
        entries = plan["schema"]["entries"]
        first, second = next(
            (i, j)
            for i, a in enumerate(entries)
            for j, b in enumerate(entries)
            if i < j and a[-1] != b[-1]
        )
        parts = [
            dict(entry=first, start=0, count=4),
            dict(entry=second, start=0, count=2),
        ]
        frame = dict(chunk=uuid.uuid4().hex, segments=parts)
        engine = d2d.manager(e.source).engine
        stream = SimpleNamespace(wait_event=lambda _: None, synchronize=lambda: None)
        # Retain actual gather, scatter, region registration, and byte copies.
        with (
            patch.dict(os.environ, {"HANDOFF_D2D_BACKEND": "rdma"}),
            patch.object(transport, "gpu_capability", return_value=True),
            patch.object(torch.cuda, "Stream", return_value=stream),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
        ):
            invalid = dict(frame, segments=[parts[0], dict(parts[1], entry=first)])
            with self.assertRaisesRegex(ValueError, "bounded native layout"):
                p.command(
                    e.target, "precopy", "precopy_prepare", frame=invalid, rdma=True
                )
            receive = p.command(
                e.target, "precopy", "precopy_prepare", frame=frame, rdma=True
            )["receive"]
            changed = dict(frame, segments=[parts[0], dict(parts[1], count=3)])
            with self.assertRaisesRegex(ValueError, "certificate mismatch"):
                p.command(
                    e.source, "precopy", "precopy_copy", frame=changed, receive=receive
                )

            copied = p.command(
                e.source, "precopy", "precopy_copy", frame=frame, receive=receive
            )
            replay = p.command(
                e.source, "precopy", "precopy_copy", frame=frame, receive=receive
            )
            size = 4 * entries[first][-1] + 2 * entries[second][-1]
            self.assertEqual(engine.calls, [(receive["region"]["session"], [size])])
            self.assertEqual(copied["copied_payload_bytes"], size)
            self.assertEqual(replay["copied_payload_bytes"], 0)

            proof = copied["proof"]
            applied = p.command(
                e.target, "precopy", "precopy_apply", frame=frame, proof=proof
            )
            coverage = [0] * len(entries)
            coverage[first], coverage[second] = 4, 2
            self.assertEqual(applied["prefix_pages"], coverage)
            self.assertEqual(
                p.command(
                    e.target, "precopy", "precopy_apply", frame=frame, proof=proof
                ),
                applied,
            )

        # Verify the live prefix itself before the frozen copy can fill any holes.
        ticket = e.target.d2d_reservations["precopy"]
        source_tables, target_tables = pc.tables(e.source), pc.tables(e.target)
        for part in parts:
            i, count = part["entry"], part["count"]
            src = pc.page_ids(e.source, e.req, 0, count)
            dst = pc.page_ids(e.target, ticket.req, 0, count)
            self.assertTrue(torch.equal(source_tables[i][src], target_tables[i][dst]))

        transfer = p.command(
            e.source,
            "precopy",
            "freeze",
            reservation=plan,
            precopy_receipts=[[applied["receipt"]]],
            **common,
        )["transfer"]
        self.assertEqual(transfer["precopy"]["prefix_pages"], coverage)
        self.assertTrue(p.finish(e, "precopy", transfer).closed)

    def test_dsv_d2d_precopy_replay_reports_zero_new_payload_bytes(self):
        """Retrying an acknowledged chunk must not count another physical payload."""
        e = self._d2d_env()
        p = self._precopy_support()
        _, plan = p.reserve(e)
        frame = dict(entry=0, start=0, count=1, chunk=uuid.uuid4().hex)
        receive = p.command(e.target, "precopy", "precopy_prepare", frame=frame)[
            "receive"
        ]
        first = p.command(
            e.source, "precopy", "precopy_copy", frame=frame, receive=receive
        )
        replay = p.command(
            e.source, "precopy", "precopy_copy", frame=frame, receive=receive
        )
        self.assertFalse(first["replayed"])
        self.assertEqual(
            first["copied_payload_bytes"], plan["schema"]["entries"][0][-1]
        )
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["copied_payload_bytes"], 0)
        self.assertEqual(first["proof"], replay["proof"])
        one = p.command(
            e.target, "precopy", "precopy_apply", frame=frame, proof=first["proof"]
        )
        two = p.command(
            e.target, "precopy", "precopy_apply", frame=frame, proof=replay["proof"]
        )
        self.assertEqual(one["prefix_pages"], two["prefix_pages"])
        self.assertEqual(two["prefix_pages"][0], 1)

    def test_dsv_d2d_precopy_rejects_changed_mapping_before_freeze(self):
        """Retraction/page-map reuse invalidates a previously copied prefix."""
        e = self._d2d_env()
        p = self._precopy_support()
        common, plan = p.reserve(e, epoch="map")
        receipt = p.copy_frame(e, "map", 0, 0, 4)
        row = e.source.scheduler.req_to_token_pool.req_to_token[e.req.kv.req_pool_idx]
        row[0] += e.f.page
        with self.assertRaisesRegex(ValueError, "ownership changed"):
            p.command(
                e.source,
                "map",
                "freeze",
                reservation=plan,
                precopy_receipts=[[receipt]],
                **common,
            )
        self.assertFalse(e.req._handoff_frozen)

    def test_dsv_d2d_precopy_cancel_late_dma_isolated_from_reused_native_kv(self):
        """Unknown late TE owns private staging; canceled native rows cannot be touched."""
        from sglang_remote_attn import d2d_precopy as pc

        e = self._d2d_env()
        p = self._precopy_support()
        _, plan = p.reserve(e)
        f = dict(entry=0, start=0, count=1, chunk=uuid.uuid4().hex)
        receive = p.command(e.target, "precopy", "precopy_prepare", frame=f)["receive"]
        state = e.target.d2d_reservations["precopy"].precopy
        private = state.receiver.tensor
        p.command(e.target, "precopy", "reserve_cancel")
        self.assertTrue(state.receiver.unsafe)
        self.assertTrue(e.target.d2d_reservations["precopy"].cancelled)
        self.assertTrue(e.target.d2d_precopy_unsafe)
        native = pc.tables(e.target)[0]
        native.fill_(199)  # allocator has reused the canceled row
        # Deterministic delayed native write to the OLD published descriptor.
        proof = p.command(
            e.source, "precopy", "precopy_copy", frame=f, receive=receive
        )["proof"]
        self.assertTrue(torch.all(native == 199))
        self.assertTrue(private[: proof["bytes"]].any())
        with self.assertRaisesRegex(ValueError, "live unadmitted"):
            p.command(e.target, "precopy", "precopy_apply", frame=f, proof=proof)
        closes = p.command(e.source, "precopy", "precopy_close")
        self.assertTrue(closes["safe"])
        with self.assertRaisesRegex(ValueError, r"closed(?:/failed)?"):
            p.command(e.source, "precopy", "precopy_copy", frame=f, receive=receive)
        self.assertFalse(e.req._handoff_frozen)

    def test_dsv_d2d_precopy_cancel_sent_slot_waits_for_receive_visibility(self):
        """Producer stop is not peer arrival: cancellation must fence the last write."""
        e = self._d2d_env()
        p = self._precopy_support()
        p.reserve(e)
        frame = dict(entry=0, start=0, count=1, chunk=uuid.uuid4().hex)
        receive = p.command(e.target, "precopy", "precopy_prepare", frame=frame)[
            "receive"
        ]
        p.command(e.source, "precopy", "precopy_copy", frame=frame, receive=receive)
        stopped = p.command(
            e.source, "precopy", "precopy_close", pending_chunk=frame["chunk"]
        )
        ticket = e.target.d2d_reservations["precopy"]
        receiver = ticket.precopy.receiver
        actual = receiver.consume
        entered, visible = threading.Event(), threading.Event()
        errors = []

        def consume(*args):
            entered.set()
            if not visible.wait(5):
                raise TimeoutError("delayed peer arrival")
            return actual(*args)

        def cancel():
            try:
                p.command(
                    e.target, "precopy", "reserve_cancel", precopy_close=[stopped]
                )
            except BaseException as exc:
                errors.append(exc)

        with patch.object(receiver, "consume", consume):
            worker = threading.Thread(target=cancel)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(ticket.req.kv.holds_kv)
                self.assertFalse(receiver.closed)
            finally:
                visible.set()
                worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertTrue(ticket.cancelled and receiver.closed)
        self.assertFalse(ticket.req.kv.holds_kv)

    def test_dsv_d2d_precopy_seal_requires_destination_receipts(self):
        """Successful send is not permission to skip pages not applied on D1."""
        e = self._d2d_env()
        p = self._precopy_support()
        common, plan = p.reserve(e)
        receipts = [p.copy_frame(e, "precopy", 0, 0, 4)]
        meta = p.command(
            e.source,
            "precopy",
            "freeze",
            reservation=plan,
            precopy_receipts=[receipts],
            **common,
        )["transfer"]
        bad = copy.deepcopy(meta)
        bad["precopy"]["receipts"] = []
        with self.assertRaisesRegex(ValueError, "destination applied receipts"):
            e.epoch = "precopy"
            e.import_(bad)
        self.assertFalse(e.target.d2d_reservations["precopy"].consumed)
        self.assertTrue(e.req._handoff_frozen)

    def test_dsv_d2d_precopy_failed_gpu_read_or_scatter_keeps_native_owner(self):
        """A private DMA error may free native KV; an unproven GPU access may not."""
        from sglang_remote_attn import d2d_precopy as pc
        from sglang_remote_attn import dsv_d2d as d2d
        from sglang_remote_attn.pre_layerwise import defer_release

        for stage in ("read", "scatter"):
            with self.subTest(stage=stage):
                e = self._d2d_env()
                p = self._precopy_support()
                p.reserve(e)
                f = dict(entry=0, start=0, count=1, chunk=uuid.uuid4().hex)
                receive = p.command(e.target, "precopy", "precopy_prepare", frame=f)[
                    "receive"
                ]

                def fail():
                    raise RuntimeError("GPU completion unproven")

                proof = None
                if stage == "scatter":
                    proof = p.command(
                        e.source, "precopy", "precopy_copy", frame=f, receive=receive
                    )["proof"]
                boundary = SimpleNamespace(
                    prepare_abort=lambda req, *a, **kw: setattr(
                        req, "finished", lambda: True
                    )
                )
                with (
                    patch.object(
                        d2d.dp, "event", return_value=SimpleNamespace(synchronize=fail)
                    ),
                    patch.dict(
                        sys.modules, {"sglang.srt.disaggregation.utils": boundary}
                    ),
                ):
                    with self.assertRaisesRegex(
                        d2d.TransportFailure, "GPU completion unproven"
                    ):
                        if stage == "read":
                            p.command(
                                e.source,
                                "precopy",
                                "precopy_copy",
                                frame=f,
                                receive=receive,
                            )
                        else:
                            p.command(
                                e.target,
                                "precopy",
                                "precopy_apply",
                                frame=f,
                                proof=proof,
                            )
                if stage == "read":
                    freed = []
                    self.assertTrue(
                        defer_release(
                            e.req, lambda: freed.append(True), checkpoint=False
                        )
                    )
                    self.assertFalse(freed)
                    self.assertTrue(e.req.kv.holds_kv)
                    self.assertIsNotNone(pc.sources(e.source)["precopy"].gpu_hold)
                else:
                    with self.assertRaisesRegex(d2d.TransportFailure, "unproven"):
                        p.command(e.target, "precopy", "reserve_cancel")
                    self.assertTrue(
                        e.target.d2d_reservations["precopy"].req.kv.holds_kv
                    )
                self.assertFalse(e.freed)
                e.close()

    def test_dsv_d2d_reserve_before_freeze_consumes_native_row_once(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        alloc = e.target.scheduler.disagg_decode_prealloc_queue._pre_alloc
        with patch.object(
            e.target.scheduler.disagg_decode_prealloc_queue, "_pre_alloc", wraps=alloc
        ) as native:
            plan, reserved = self._d2d_reserve(e)
            self.assertTrue(reserved["reserved"])
            self.assertFalse(e.req._handoff_frozen)
            self.assertEqual(e.source.scheduler.running_batch.reqs, [e.req])
            ticket = e.target.d2d_reservations[e.epoch]
            duplicate = d2d.handle(
                e.target, dict(cmd="d2d_reserve", migration_epoch=e.epoch, plan=plan)
            )
            self.assertTrue(duplicate["duplicate"])
            e.import_(self._d2d_reserved_freeze(e, reserved))
            target = e.target.d2d_targets[e.epoch]
            self.assertIs(target.decode_req.req.kv, ticket.req.kv)
            self.assertTrue(ticket.consumed)
            self.assertEqual(native.call_count, 1)
            with self.assertRaisesRegex(ValueError, "consumed"):
                e.command(e.target, "d2d_reserve_cancel")
            e.command(e.target, "d2d_commit")
            e.gate.set()
            e.wait_copy()
            e.target.drain_targets(required=target)
            self.assertTrue(target.closed)
            self.assertTrue(e.source.d2d_sources[e.epoch].committed)

    def test_dsv_d2d_reserve_capacity_rejected_while_source_runs(self):
        e = self._d2d_env()
        e.target.scheduler.req_to_token_pool.available_size = lambda: 0
        with self.assertRaisesRegex(ValueError, "reserve preflight"):
            self._d2d_reserve(e)
        self.assertFalse(e.req._handoff_frozen)
        self.assertEqual(e.source.scheduler.running_batch.reqs, [e.req])
        self.assertFalse(e.target.d2d_reservations)
        self.assertFalse(e.freed)

    def test_dsv_d2d_reserve_cancel_is_idempotent_and_cannot_resurrect(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        plan, _ = self._d2d_reserve(e)
        self.assertTrue(e.command(e.target, "d2d_reserve_cancel")["cancelled"])
        self.assertTrue(e.command(e.target, "d2d_reserve_cancel")["duplicate"])
        self.assertEqual(e.freed, ["target"])
        with self.assertRaisesRegex(ValueError, "retired"):
            d2d.handle(
                e.target, dict(cmd="d2d_reserve", migration_epoch=e.epoch, plan=plan)
            )
        self.assertFalse(e.req._handoff_frozen)

    def test_dsv_d2d_reserve_cancel_before_rpc_leaves_tombstone(self):

        e = self._d2d_env()
        e.epoch = "reserved-migration"
        e.command(e.target, "d2d_reserve_cancel")
        with self.assertRaisesRegex(ValueError, "retired"):
            self._d2d_reserve(e)
        self.assertTrue(e.command(e.target, "d2d_status")["reservation"]["cancelled"])
        self.assertFalse(e.freed)

    def test_dsv_d2d_planned_freeze_requires_certificate_and_bound(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        plan, reserved = self._d2d_reserve(e)
        with self.assertRaisesRegex(ValueError, "requires target reservation"):
            d2d.handle(
                e.source,
                dict(
                    cmd="d2d_freeze",
                    rid="source",
                    destination="D1",
                    target_rid="target",
                    migration_epoch=e.epoch,
                ),
            )
        e.req.output_ids.append(34)
        e.req.kv.kv_committed_len += 1
        with self.assertRaisesRegex(ValueError, "outside reserved"):
            self._d2d_reserved_freeze(e, reserved)
        self.assertFalse(e.req._handoff_frozen)
        self.assertEqual(e.source.scheduler.running_batch.reqs, [e.req])
        self.assertFalse(e.source.d2d_sources)

    def test_dsv_d2d_reservation_failed_native_allocation_is_quarantined(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()

        def partial(req, **kwargs):
            req.kv.req_pool_idx = 1
            raise RuntimeError("allocation interrupted")

        e.target.scheduler.disagg_decode_prealloc_queue._pre_alloc = partial
        with self.assertRaisesRegex(d2d.TransportFailure, "allocation failed"):
            self._d2d_reserve(e)
        with self.assertRaisesRegex(d2d.TransportFailure, "unproven"):
            e.command(e.target, "d2d_reserve_cancel")
        self.assertFalse(e.freed)
        self.assertFalse(e.req._handoff_frozen)

    def test_dsv_d2d_reservation_cancel_cuda_proof_failure_retains_row(self):
        e = self._d2d_env()
        self._d2d_reserve(e)
        ticket = e.target.d2d_reservations[e.epoch]

        def fail():
            raise RuntimeError("CUDA proof failed")

        ticket.ready = SimpleNamespace(synchronize=fail)
        with self.assertRaisesRegex(RuntimeError, "CUDA proof failed"):
            e.command(e.target, "d2d_reserve_cancel")
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            e.command(e.target, "d2d_reserve_cancel")
        self.assertFalse(e.freed)
        self.assertTrue(ticket.req.kv.holds_kv)

    def test_dsv_d2d_reservation_import_mismatch_keeps_ticket_for_cancel(self):
        e = self._d2d_env()
        self._d2d_reserve(e)
        ticket = e.target.d2d_reservations[e.epoch]
        # A legacy freeze cannot change a planned source destination silently.
        plan = e.source.d2d_plans.pop(e.epoch)
        meta = e.freeze(e.epoch)
        meta["output_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "does not fit reservation"):
            e.import_(meta)
        self.assertFalse(ticket.consumed)
        self.assertTrue(ticket.req.kv.holds_kv)
        self.assertTrue(e.command(e.target, "d2d_reserve_cancel")["cancelled"])
        self.assertEqual(e.freed, ["target"])

    def test_dsv_d2d_reservation_union_window_budget_and_trim_pages(self):
        e = self._d2d_env()
        scheduler = e.target.scheduler
        for node in e.nodes:
            pool = node.scheduler.req_to_token_pool
            rows = pool.req_to_token
            pool.req_to_token = torch.cat((rows, rows[:, :24] + 100), dim=1)
        fits = Mock(return_value=True)
        queue = scheduler.disagg_decode_prealloc_queue
        queue._prealloc_reservation_fits = fits
        freed = []
        scheduler.token_to_kv_pool_allocator.free = lambda indices: freed.append(
            indices.clone()
        )
        plan, reserved = self._d2d_reserve(e, growth=15)
        # lower=33, upper=48, SWA window=8: reserve [24,48), not [40,48).
        self.assertEqual(plan["upper"], 48)
        self.assertEqual(fits.call_args.args[1], 24)
        meta = self._d2d_reserved_freeze(e, reserved)
        e.import_(meta)
        ticket = e.target.d2d_reservations[e.epoch]
        self.assertEqual(ticket.req.kv.kv_allocated_len, 33)
        self.assertEqual(len(freed), 1)
        # Preserve the partial page containing positions [32,40).
        self.assertTrue(
            torch.equal(freed[0], scheduler.req_to_token_pool.req_to_token[1, 40:48])
        )

    def test_dsv_d2d_reservation_peer_preflight_rejects_before_allocation(self):
        e = self._d2d_env()
        e.target.agree = lambda flags: [0]
        native = Mock()
        e.target.scheduler.disagg_decode_prealloc_queue._pre_alloc = native
        with self.assertRaisesRegex(ValueError, "peer rejected"):
            self._d2d_reserve(e)
        native.assert_not_called()
        self.assertFalse(e.req._handoff_frozen)
        self.assertFalse(e.freed)

    def test_dsv_d2d_reservation_completed_peer_cancels_after_peer_allocation_failure(
        self,
    ):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.target.agree = Mock(side_effect=[[1], [0]])
        with self.assertRaisesRegex(d2d.TransportFailure, "peer failed"):
            self._d2d_reserve(e)
        self.assertTrue(e.target.d2d_reservations[e.epoch].complete)
        self.assertTrue(e.command(e.target, "d2d_reserve_cancel")["cancelled"])
        self.assertEqual(e.freed, ["target"])
        self.assertFalse(e.req._handoff_frozen)

    def test_dsv_d2d_native_prealloc_binds_union_window_and_validates_before_alloc(
        self,
    ):
        cls = _source_class(
            "disaggregation/decode.py", "DecodePreallocQueue", {"_pre_alloc"}
        )
        cls._pre_alloc.__globals__.update(
            get_disagg=lambda: SimpleNamespace(
                disaggregation_decode_enable_radix_cache=False
            ),
            ceil_align=lambda n, page: (n + page - 1) // page * page,
            is_swa_req_ring=lambda allocator: False,
        )
        q = cls()
        cls._pre_alloc.__globals__["pre_alloc_fill_len"] = _source_function(
            "disaggregation/utils.py", "pre_alloc_fill_len"
        )
        q._swa_tail_len = lambda length: 8
        q._uses_swa_tail_prealloc = lambda: True
        q.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=8, state=None, storage=None
        )

        def alloc(reqs):
            reqs[0].kv.req_pool_idx = 1
            return [1]

        q.req_to_token_pool = SimpleNamespace(
            alloc=Mock(side_effect=alloc), write=Mock()
        )
        q._required_alloc_tokens = lambda **kw: 48
        q.scheduler = SimpleNamespace(enable_hisparse=False)
        q.tree_cache = SimpleNamespace(maybe_hand_to_session=lambda req: None)
        q._alloc_for_decode_prealloc = Mock(return_value=torch.arange(48))
        req = SimpleNamespace(
            kv=self.Kv(), origin_input_ids=list(range(48)), output_ids=[2]
        )
        with self.assertRaisesRegex(ValueError, "SWA window"):
            q._pre_alloc(req, swa_tail_start=25)
        q.req_to_token_pool.alloc.assert_not_called()
        q._pre_alloc(req, swa_tail_start=24)
        self.assertEqual(
            q._alloc_for_decode_prealloc.call_args.kwargs["swa_tail_len"], 24
        )
        self.assertEqual(req.kv.kv_committed_len, 48)

    def test_dsv_d2d_reserved_manifest_cannot_fallback_to_unreserved_import(self):
        e = self._d2d_env()
        _, reserved = self._d2d_reserve(e)
        meta = self._d2d_reserved_freeze(e, reserved)
        self.assertTrue(meta["reservation_required"])
        ticket = e.target.d2d_reservations.pop(e.epoch)
        native = e.target.scheduler.disagg_decode_prealloc_queue._pre_alloc
        with patch.object(
            e.target.scheduler.disagg_decode_prealloc_queue, "_pre_alloc", wraps=native
        ) as alloc:
            with self.assertRaisesRegex(
                ValueError, "requires a live native reservation"
            ):
                e.import_(meta)
            alloc.assert_not_called()
        self.assertFalse(ticket.consumed)
        self.assertFalse(e.freed)

    def test_dsv_d2d_unreserved_freeze_is_rejected_by_default(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        with self.assertRaisesRegex(ValueError, "requires target reservation"):
            d2d.handle(
                e.source,
                dict(
                    cmd="d2d_freeze",
                    rid="source",
                    destination="D1",
                    target_rid="target",
                    migration_epoch="unreserved",
                ),
            )
        self.assertFalse(e.req._handoff_frozen)
        self.assertEqual(e.source.scheduler.running_batch.reqs, [e.req])
        self.assertFalse(e.freed)

    def _p4_dsv_env(self, accept=True):
        import importlib.util

        e = self._d2d_env()
        path = ROOT / "test/manual/remote_attn/p4_dsv_probe_support.py"
        spec = importlib.util.spec_from_file_location("p4_dsv_probe_support", path)
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        folder = self.enterContext(tempfile.TemporaryDirectory())
        driver = support.Driver(e)
        gw = support.gateway(driver, ROOT, folder)
        self.addCleanup(gw.close)
        gw.tick()
        body = dict(
            rid="logical-request",
            prompt_ids=list(e.req.origin_input_ids),
            max_new_tokens=8,
            cut=2,
            growth_tokens=2,
        )
        rid = gw.accept(body) if accept else None
        return SimpleNamespace(
            e=e, driver=driver, gw=gw, rid=rid, folder=folder, support=support
        )

    def _p4_dsv_restart(self, f):
        f.gw.close()
        f.gw = f.support.gateway(f.driver, ROOT, f.folder)
        self.addCleanup(f.gw.close)
        return f.gw

    def test_p4_dsv_partial_remote_freeze_survives_restart_and_lost_reply(self):
        """A frozen partial PD owner must neither roll back nor freeze twice."""
        import importlib.util

        _, e = self._partial_d2d_env()
        path = ROOT / "test/manual/remote_attn/p4_dsv_probe_support.py"
        spec = importlib.util.spec_from_file_location("partial_gateway_support", path)
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        driver = support.Driver.__new__(support.Driver)
        driver.env = e
        driver.calls, driver.before, driver.after = [], {}, {}
        driver.controllers = dict(src=e.source, dst=e.target)
        folder = self.enterContext(tempfile.TemporaryDirectory())
        gw = support.gateway(driver, ROOT, folder)
        self.addCleanup(gw.close)
        gw.tick()
        rid = gw.accept(
            dict(
                rid="partial",
                prompt_ids=list(range(31)),
                max_new_tokens=64,
                cut=2,
                growth_tokens=0,
            )
        )
        # PD/model submission is the existing ABI boundary; adopt its active Req.
        gw._update(rid, phase="SOURCE", source_rid=e.req.rid)
        support.until(gw, rid, "FREEZE")
        first, _ = gw.events(rid)
        self.assertEqual([x["token_id"] for x in first], [31, 32, 33])
        self.assertEqual({x["owner"] for x in first}, {"src"})
        driver.after["d2d_freeze"] = [TimeoutError("lost freeze response")]
        gw.tick()
        entry = gw.entry(rid)
        epoch = entry["epoch"]
        self.assertEqual(entry["phase"], "FREEZE")
        self.assertTrue(e.req._handoff_frozen)
        source = e.source.d2d_sources[epoch]
        self.assertIsNone(source.service)
        gw.close()
        gw = support.gateway(driver, ROOT, folder)
        self.addCleanup(gw.close)
        gw.tick()
        entry = gw.entry(rid)
        self.assertEqual(entry["epoch"], epoch)
        self.assertTrue(entry["freeze_pending"] and entry["source_partial_remote"])
        self.assertEqual(entry["attempts"], 1)
        self.assertEqual(driver.count("d2d_import"), 0)
        for i in range(e.concurrency):
            e.arrive(i)
        e.complete_pd()
        support.until(gw, rid, "TARGET")
        self.assertIs(e.source.d2d_sources[epoch], source)
        self.assertFalse(gw.entry(rid)["freeze_pending"])
        self.assertEqual(gw.entry(rid)["migration_outcome"], "migrated")
        target = e.target.d2d_targets[epoch]
        target.decode_req.req.output_ids.append(34)
        target.decode_req.req.kv.kv_committed_len += 1
        target.decode_req.req.kv.kv_allocated_len += 1
        gw.tick()
        events, _ = gw.events(rid, after=2)
        self.assertEqual(
            [(x["seq"], x["token_id"], x["owner"]) for x in events],
            [(3, 34, "dst")],
        )
        self.assertFalse(e.freed)
        e.gate.set()
        target.worker.join(10)
        self.assertFalse(target.worker.is_alive())

    def test_p4_dsv_large_payloads_leave_hot_journal_and_survive_restart(self):
        """27 long prompts must not be re-encoded by each scalar intent update."""
        f = self._p4_dsv_env(accept=False)
        prompt = list(range(32768))
        rids = [
            f.gw.accept(
                dict(rid=f"long-{i}", prompt_ids=prompt, max_new_tokens=128, cut=32)
            )
            for i in range(27)
        ]
        self.assertLess(f.gw.path.stat().st_size, 65536)
        with patch.object(
            f.gw.payloads,
            "put",
            side_effect=AssertionError("hot update encoded payload"),
        ):
            f.gw._update(rids[0], migration_deferred=True)
        # Recover an old v1 journal atomically, then continue with small intents.
        legacy = copy.deepcopy(f.gw.journal)
        legacy["version"] = 1
        for rid in rids:
            entry = legacy["entries"][rid]
            entry["input_ids"] = prompt
            entry.pop("input_ids_ref")
        f.gw.close()
        f.gw.path.write_text(json.dumps(legacy))
        self._p4_dsv_restart(f)
        self.assertEqual(f.gw.journal["version"], 2)
        self.assertEqual(f.gw.entry(rids[-1])["input_ids"], prompt)
        self.assertTrue(f.gw.entry(rids[0])["migration_deferred"])
        self.assertLess(f.gw.path.stat().st_size, 65536)
        self.assertEqual(len(list(f.gw.payloads.root.glob("*.json"))), 1)

    def test_p4_dsv_admission_plan_does_not_wait_for_status_or_first_sample(self):
        """The submission RPC itself must create the plan before any Decode step."""
        f = self._p4_dsv_env(accept=False)
        f.driver.admission_plan = True
        enqueue = f.e.source.scheduler._add_request_to_queue

        def unsampled(req):
            enqueue(req)
            req.output_ids.clear()
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 0

        f.e.source.scheduler._add_request_to_queue = unsampled
        f.gw.tick()
        f.rid = f.gw.accept(
            dict(
                rid="admission",
                prompt_ids=list(f.e.req.origin_input_ids),
                max_new_tokens=128,
                cut=32,
            )
        )
        f.gw.tick()  # native submission, including its inline capacity plan
        entry = f.gw.entry(f.rid)
        plan = entry["admission_plans"][0]
        self.assertFalse(plan["output_ids"])
        self.assertIn(plan["epoch"], f.e.source.d2d_plans)
        f.support.until(f.gw, f.rid, "WAIT_CUT")
        entry = f.gw.entry(f.rid)
        self.assertTrue(entry["plan_before_first_token"])
        self.assertEqual(f.driver.count("d2d_plan"), 0)
        self.assertEqual(f.driver.count("d2d_reserve"), 1)
        self.assertFalse(f.e.req._handoff_frozen)

    def test_p4_dsv_prepares_before_first_token_waits_for_cut_and_recovers(self):
        """Plan/reserve cannot consume the short request's post-cut window."""
        f = self._p4_dsv_env(accept=False)
        f.rid = f.gw.accept(
            dict(
                rid="early",
                prompt_ids=list(f.e.req.origin_input_ids),
                max_new_tokens=8,
                cut=5,
                growth_tokens=0,
            )
        )
        f.support.until(f.gw, f.rid, "SOURCE")
        req = f.e.req
        req.output_ids[:] = []
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 0
        f.support.until(f.gw, f.rid, "WAIT_CUT")
        entry = f.gw.entry(f.rid)
        epoch = entry["epoch"]
        self.assertEqual(f.driver.count("d2d_plan"), 1)
        self.assertEqual(f.driver.count("d2d_reserve"), 1)
        self.assertEqual(f.driver.count("d2d_freeze"), 0)
        self.assertFalse(req._handoff_frozen)
        self.assertTrue(f.e.target.d2d_reservations[epoch].req.kv.holds_kv)
        self.assertEqual(
            f.gw.latency_metrics()["control_latency_ms"]["src/d2d_plan"]["count"], 1
        )
        self._p4_dsv_restart(f)
        for _ in range(3):
            f.gw.tick()
        self.assertEqual(f.gw.entry(f.rid)["phase"], "WAIT_CUT")
        self.assertEqual(f.driver.count("d2d_plan"), 1)
        self.assertTrue(entry["plan_before_first_token"])
        self.assertFalse(req.output_ids)
        plan = entry["plans"][0]
        self.assertEqual(plan["lower"], len(req.origin_input_ids))
        self.assertEqual(
            f.e.target.d2d_reservations[epoch].req.kv.kv_committed_len, plan["upper"]
        )
        req.output_ids.extend([31, 32, 33, 34, 35])
        req.kv.kv_committed_len = req.kv.kv_allocated_len = (
            len(req.origin_input_ids) + 4
        )
        f.support.until(f.gw, f.rid, "TARGET")
        moved = f.gw.entry(f.rid)
        self.assertEqual(moved["epoch"], epoch)
        self.assertEqual(moved["migration_outcome"], "migrated")
        self.assertEqual(f.driver.count("d2d_freeze"), 1)
        self.assertEqual(
            f.e.source.d2d_sources[epoch].frozen_output, tuple(req.output_ids)
        )
        self.assertEqual(f.gw.latency_metrics()["handoff_latency_ms"]["count"], 1)
        self.assertIsNotNone(moved["freeze_ack_latency_ms"])
        self.assertGreaterEqual(
            moved["handoff_latency_ms"], moved["freeze_ack_latency_ms"]
        )
        f.e.gate.set()

    def test_p4_dsv_cut_finishes_pending_frame_without_copying_to_snapshot(self):
        """A cut must freeze after the active frame, even with more stable KV."""
        f = self._p4_dsv_env()
        f.gw._update(f.rid, cut=7)
        f.support.until(f.gw, f.rid, "WAIT_CUT")
        f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["precopy_pending"]["stage"], "COPY")
        frame = entry["precopy_pending"]["frame"]
        req = f.e.req
        req.output_ids.extend([64] * (7 - len(req.output_ids)))
        req.kv.kv_committed_len = req.kv.kv_allocated_len = (
            len(req.origin_input_ids) + len(req.output_ids) - 1
        )
        self._p4_dsv_restart(f)
        f.support.until(f.gw, f.rid, "FREEZE")
        entry = f.gw.entry(f.rid)
        self.assertIsNone(entry["precopy_pending"])
        self.assertEqual(len(entry["precopy_receipts"][0]), 1)
        self.assertEqual(entry["precopy_pages"][frame["entry"]], frame["count"])
        self.assertEqual(sum(entry["precopy_pages"]), frame["count"])
        self.assertFalse(req._handoff_frozen)
        f.support.until(f.gw, f.rid, "TARGET")
        self.assertEqual(f.gw.entry(f.rid)["migration_outcome"], "migrated")

    def test_p4_dsv_source_finishes_between_status_and_precopy_copy(self):
        """A stale running snapshot cannot turn normal completion into an error."""
        from sglang_remote_attn import d2d_precopy

        f = self._p4_dsv_env()
        f.gw._update(f.rid, cut=7)
        f.support.until(f.gw, f.rid, "WAIT_CUT")
        f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["precopy_pending"]["stage"], "COPY")
        req = f.e.req
        req.finished = lambda: True
        d2d_precopy.release_source(req, lambda: None)
        # The previous status still described a runnable request.
        with patch.object(f.gw, "_refresh"):
            f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["phase"], "ROLLBACK")
        self.assertEqual(entry["migration_outcome"], "migration_cancelled_by_finish")
        self.assertIsNone(entry["error"])
        f.gw.tick()
        ticket = f.e.target.d2d_reservations[entry["epoch"]]
        self.assertTrue(ticket.cancelled)
        self.assertFalse(ticket.req.kv.holds_kv)
        self.assertTrue(req.kv.holds_kv)
        self.assertFalse(req._handoff_frozen)

    def test_p4_dsv_finished_window_is_classified_and_reservation_cancelled(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "FREEZE")
        epoch = f.gw.entry(f.rid)["epoch"]
        f.e.req.finished = lambda: True
        f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["phase"], "ROLLBACK")
        self.assertEqual(entry["migration_skip_reason"], "request_finished")
        self.assertEqual(entry["migration_outcome"], "window_missed_request_finished")
        self.assertNotIn(epoch, f.e.source.d2d_sources)
        f.gw.tick()
        self.assertTrue(f.e.target.d2d_reservations[epoch].cancelled)
        self.assertFalse(f.e.req._handoff_frozen)
        self.assertEqual(
            f.gw.latency_metrics()["migration_skip_reasons"], {"request_finished": 1}
        )
        self.assertEqual(f.gw.latency_metrics()["handoff_latency_ms"]["count"], 0)

    def test_p4_dsv_cancel_prepared_before_cut_never_freezes_source(self):
        f = self._p4_dsv_env(accept=False)
        f.rid = f.gw.accept(
            dict(
                rid="cancel-early",
                prompt_ids=list(f.e.req.origin_input_ids),
                max_new_tokens=8,
                cut=6,
                growth_tokens=0,
            )
        )
        f.support.until(f.gw, f.rid, "WAIT_CUT")
        epoch = f.gw.entry(f.rid)["epoch"]
        f.gw.cancel(f.rid)
        f.support.until(f.gw, f.rid, "CANCELLED")
        self.assertTrue(f.e.target.d2d_reservations[epoch].cancelled)
        self.assertEqual(f.driver.count("d2d_freeze"), 0)
        self.assertEqual(f.driver.count("d2d_commit"), 0)

    def test_dsv_pre_sample_plan_reserves_native_capacity_without_fake_source_output(
        self,
    ):
        from sglang_remote_attn import dsv_d2d as d2d
        from sglang_remote_attn.dsv_gateway_control import request_status

        for limit in (1, 8):
            with self.subTest(max_new_tokens=limit):
                e = self._d2d_env()
                e.req.output_ids.clear()
                e.req.sampling_params.max_new_tokens = limit
                e.req.kv.kv_committed_len = e.req.kv.kv_allocated_len = 0
                status = request_status(e.source, e.req)
                self.assertTrue(status["plan_ready"])
                self.assertFalse(status["migration_ready"])
                self.assertEqual(
                    status["migration_deferred_reason"], "sampled_output_pending"
                )
                plan, reserved = self._d2d_reserve(e, growth=limit)
                self.assertEqual(plan["lower"], len(e.req.origin_input_ids))
                self.assertEqual(plan["upper"], plan["lower"] + limit - 1)
                ticket = e.target.d2d_reservations[e.epoch]
                self.assertEqual(ticket.req.kv.kv_committed_len, plan["upper"])
                self.assertEqual(len(ticket.req.output_ids), limit)
                from sglang_remote_attn.source_resources import owned_requests

                self.assertIn(ticket.req, owned_requests(e.target))
                skipped = d2d.handle(
                    e.source,
                    dict(
                        cmd="d2d_freeze",
                        migration_epoch=e.epoch,
                        rid=e.req.rid,
                        destination="D1",
                        target_rid="target",
                        reservation=reserved["reservation"],
                    ),
                )
                self.assertEqual(
                    skipped["migration_deferred_reason"], "sampled_output_pending"
                )
                self.assertFalse(
                    e.req.output_ids
                    or e.req._handoff_frozen
                    or e.source.d2d_sources
                    or e.freed
                )
                e.command(e.target, "d2d_reserve_cancel")
                e.command(e.source, "d2d_resume")
                self.assertEqual(e.freed, ["target"])
                self.assertNotIn(ticket.req, owned_requests(e.target))
                self.assertNotIn(e.epoch, e.source.d2d_plans)
                with self.assertRaisesRegex(ValueError, "cancelled"):
                    d2d.handle(
                        e.source,
                        dict(
                            cmd="d2d_plan",
                            migration_epoch=e.epoch,
                            rid=e.req.rid,
                            destination="D1",
                            target_rid="target",
                            growth_tokens=limit,
                        ),
                    )

    def test_dsv_prepared_reservation_rejects_changed_source_attempt_or_history(self):
        for change in ("handle", "retraction", "prompt", "sampling"):
            with self.subTest(change=change):
                e = self._d2d_env()
                plan, reserved = self._d2d_reserve(e)
                if change == "handle":
                    e.req.cache_request_handle = (e.req.rid, 1)
                elif change == "retraction":
                    e.req.retraction_count = 1
                elif change == "prompt":
                    e.req.origin_input_ids[0] += 1
                else:
                    e.req.sampling_params.temperature = 0.5
                with self.assertRaisesRegex(ValueError, "attempt/history"):
                    self._d2d_reserved_freeze(e, reserved)
                self.assertFalse(
                    e.req._handoff_frozen or e.source.d2d_sources or e.freed
                )
                self.assertTrue(e.target.d2d_reservations[e.epoch].req.kv.holds_kv)
                e.command(e.target, "d2d_reserve_cancel")
                self.assertEqual(e.freed, ["target"])

    def test_p4_dsv_pre_sample_cancel_or_finish_returns_unused_reservation(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                f = self._p4_dsv_env()
                f.support.until(f.gw, f.rid, "SOURCE")
                f.e.req.output_ids.clear()
                f.e.req.kv.kv_committed_len = 0
                f.support.until(f.gw, f.rid, "WAIT_CUT")
                epoch = f.gw.entry(f.rid)["epoch"]
                if cancel:
                    f.gw.cancel(f.rid)
                else:
                    f.e.req.finished = lambda: True
                f.gw.tick()
                self.assertEqual(f.gw.entry(f.rid)["phase"], "ROLLBACK")
                f.gw.tick()
                self.assertTrue(f.e.target.d2d_reservations[epoch].cancelled)
                self.assertEqual(f.driver.count("d2d_freeze"), 0)
                self.assertNotIn(epoch, f.e.source.d2d_plans)
                self.assertFalse(f.e.req._handoff_frozen)

    def test_dsv_plan_can_prepare_before_decode_boundary_but_freeze_defers(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        e.source.scheduler.result_queue = [
            (
                SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: False)),
                None,
            )
        ]
        data = dict(
            migration_epoch="early",
            rid=e.req.rid,
            target_rid="early-target",
            destination="D1",
        )
        result = d2d.handle(e.source, dict(cmd="d2d_plan", growth_tokens=8, **data))
        self.assertIn("plan", result)
        deferred = d2d.handle(e.source, dict(cmd="d2d_freeze", **data))
        self.assertEqual(
            deferred["migration_deferred_reason"], "decode_boundary_pending"
        )
        self.assertFalse(e.req._handoff_frozen)
        e.source.scheduler.result_queue = []
        e.req.finished = lambda: True
        skipped = d2d.handle(e.source, dict(cmd="d2d_freeze", **data))
        self.assertEqual(skipped["migration_deferred_reason"], "request_finished")
        self.assertFalse(e.source.d2d_sources or e.freed)

    def test_p4_dsv_status_poll_yields_to_waiting_mutation_without_worker_error(self):
        f = self._p4_dsv_env()
        held, waiting = threading.Lock(), threading.Event()
        held.acquire()

        class GateLock:
            def acquire(self, *args, **kwargs):
                if kwargs.get("timeout"):
                    waiting.set()
                return held.acquire(*args, **kwargs)

            def release(self):
                held.release()

        f.gw.role_locks["src"] = GateLock()
        results = []
        t = threading.Thread(
            target=lambda: results.append(f.gw._call("src", "d2d_status", "priority"))
        )
        t.start()
        try:
            self.assertTrue(waiting.wait(2))
            polls = f.driver.count("d2d_worker_status")
            f.gw._refresh_role("src")
            self.assertEqual(f.driver.count("d2d_worker_status"), polls)
            self.assertNotIn("src", f.gw.fact_errors)
        finally:
            held.release()
            t.join(2)
        self.assertEqual(len(results), 1)

    def test_p4_dsv_orders_native_reserve_before_freeze_and_sse_owner_switch(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "SOURCE")
        f.gw.tick()
        first, _ = f.gw.events(f.rid)
        self.assertEqual([x["seq"] for x in first], [0, 1, 2])
        self.assertEqual({x["owner"] for x in first}, {"src"})
        f.support.until(f.gw, f.rid, "TARGET")
        commands = [p["cmd"] for _, p in f.driver.calls]
        self.assertLess(commands.index("d2d_reserve"), commands.index("d2d_freeze"))
        self.assertLess(commands.index("d2d_import"), commands.index("d2d_commit"))
        target = f.e.target.d2d_targets[f.gw.entry(f.rid)["epoch"]]
        self.assertFalse(target.copied)
        target.decode_req.req.output_ids.append(34)
        target.decode_req.req.kv.kv_committed_len += 1
        target.decode_req.req.kv.kv_allocated_len += 1
        f.gw.tick()
        events, _ = f.gw.events(f.rid, after=2)
        self.assertEqual(
            [(x["seq"], x["token_id"], x["owner"]) for x in events], [(3, 34, "dst")]
        )
        self.assertEqual(f.driver.count("d2d_submit_pd"), 1)
        self.assertTrue(f.e.target.d2d_reservations[target.generation].consumed)
        f.driver.finish_local()
        f.gw.tick()
        self.assertEqual(f.gw.entry(f.rid)["phase"], "LOCAL")

    def test_p4_dsv_waits_for_local_reports_deferral_once_and_recovers_after_restart(
        self,
    ):
        from sglang_remote_attn.dsv41_remote import Target

        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "SOURCE")
        rpc = Target(
            f.e.source, f.e.req, Path(f.folder) / "previous", "old", {}, f.e.prefix
        )
        rpc.progressive = True
        f.e.req._dsv41_remote = rpc
        f.e.source.remote_targets[rpc.row] = rpc
        metrics = Path(f.folder) / "metrics.jsonl"
        f.gw.out = metrics
        f.gw.start_drain()
        for _ in range(5):
            f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["phase"], "SOURCE")
        self.assertTrue(entry["migration_deferred"])
        self.assertEqual(entry["migration_deferred_reason"], "remote_entries_pending")
        self.assertEqual(entry["migration_deferred_count"], 1)
        self.assertEqual(entry["attempts"], 0)
        self.assertEqual(f.driver.count("d2d_plan"), 0)
        self.assertFalse(f.e.freed or f.e.req._handoff_frozen)
        with patch.object(
            time, "time", return_value=entry["migration_deferred_since"] + 1
        ):
            self.assertGreaterEqual(
                f.gw.entry(f.rid)["migration_deferred_wait_ms"], 1000
            )
        self.assertEqual(f.gw.drain_status()["migration_deferred"], 1)
        waiting = [json.loads(line) for line in metrics.read_text().splitlines()]
        self.assertEqual(len(waiting), 1)
        handler = SimpleNamespace(
            wfile=io.BytesIO(),
            send_response=lambda *args: None,
            send_header=lambda *args: None,
            end_headers=lambda: None,
        )
        f.gw.http_get(handler, SimpleNamespace(path="/v1/router/status", query=""))
        visible = json.loads(handler.wfile.getvalue())["migrations"][f.rid]
        self.assertTrue(visible["migration_deferred"])
        handler.wfile = io.BytesIO()
        with patch.object(f.gw, "events", return_value=([], True)):
            f.gw.stream(handler, f.rid)
        wire = handler.wfile.getvalue().decode()
        self.assertIn("event: status", wire)
        self.assertIn('"migration_deferred_reason": "remote_entries_pending"', wire)
        self.assertEqual(
            waiting[0]["migration_deferred_reason"], "remote_entries_pending"
        )
        self._p4_dsv_restart(f)
        self.assertEqual(f.gw.entry(f.rid)["migration_deferred_count"], 1)
        rpc.local_entries.update(
            i for i, e in enumerate(f.e.source._schema(None)["entries"]) if e[0] == "kv"
        )
        f.support.until(f.gw, f.rid, "TARGET")
        entry = f.gw.entry(f.rid)
        self.assertFalse(entry["migration_deferred"])
        self.assertEqual(entry["migration_deferred_count"], 1)
        self.assertEqual(entry["attempts"], 1)
        self.assertEqual(entry["migration_outcome"], "migrated")

    def test_p4_dsv_plan_readiness_race_defers_without_rollback_or_retry_budget(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "PLAN")
        f.e.req._dsv41_remote = SimpleNamespace(all_local=lambda: False)
        f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["phase"], "SOURCE")
        self.assertEqual(entry["attempts"], 0)
        self.assertFalse(entry["cut_attempted"])
        self.assertTrue(entry["migration_deferred"])
        self.assertEqual(entry["migration_deferred_reason"], "remote_entries_pending")
        self.assertFalse(f.e.source.d2d_plans)
        self.assertFalse(f.e.freed or f.e.req._handoff_frozen)
        self.assertEqual(f.driver.count("d2d_reserve"), 0)
        self.assertEqual(f.driver.count("d2d_abort"), 0)
        self.assertEqual(f.driver.count("d2d_resume"), 0)
        f.e.req._dsv41_remote = None
        with patch.object(time, "time", return_value=entry["retry_at"] + 1):
            f.support.until(f.gw, f.rid, "RESERVE")
        self.assertFalse(f.gw.entry(f.rid)["migration_deferred"])
        self.assertEqual(f.gw.entry(f.rid)["attempts"], 1)

    def test_p4_dsv_source_completion_records_missed_deferred_migration(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "SOURCE")
        f.e.req._dsv41_remote = SimpleNamespace(all_local=lambda: False)
        f.gw.tick()
        self.assertTrue(f.gw.entry(f.rid)["migration_deferred"])
        f.e.req.finished = lambda: True
        f.e.req.kv.holds_kv = False
        f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["phase"], "DONE")
        self.assertFalse(entry["migration_deferred"])
        self.assertEqual(
            entry["migration_outcome"], "completed_on_source_before_migration"
        )
        self.assertEqual(entry["migration_deferred_count"], 1)
        self.assertEqual(f.driver.count("d2d_plan"), 0)

    def test_p4_dsv_restart_reconciles_every_precommit_intent_and_commit(self):
        for phase in (
            "SUBMIT",
            "SOURCE",
            "PLAN",
            "RESERVE",
            "FREEZE",
            "IMPORT",
            "COMMIT",
            "TARGET",
        ):
            with self.subTest(phase=phase):
                f = self._p4_dsv_env()
                f.support.until(f.gw, f.rid, phase)
                self._p4_dsv_restart(f)
                f.support.until(f.gw, f.rid, "TARGET")
                self.assertEqual(f.gw.entry(f.rid)["owner"], "dst")
                self.assertEqual(f.driver.count("d2d_submit_pd"), 1)
                self.assertEqual(len(f.e.target.d2d_targets), 1)
                self.assertEqual(len(f.e.resumed), 1)
                self.assertFalse(
                    any(p["cmd"] == "d2d_resume" for _, p in f.driver.calls)
                )
                f.gw.close()
                f.e.close()

    def test_p4_dsv_lost_commit_reply_rolls_forward_after_restart_without_resampling(
        self,
    ):
        f = self._p4_dsv_env()
        f.driver.after["d2d_commit"] = [TimeoutError("lost commit reply")]
        f.support.until(f.gw, f.rid, "COMMIT")
        f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["phase"], "COMMIT")
        self.assertEqual(entry["owner"], "src")
        self.assertTrue(f.e.source.d2d_sources[entry["epoch"]].committed)
        self._p4_dsv_restart(f)
        f.support.until(f.gw, f.rid, "TARGET")
        self.assertEqual(len(f.e.resumed), 1)
        self.assertEqual(f.driver.count("d2d_resume"), 0)
        self.assertEqual(f.driver.count("d2d_abort"), 0)

    def test_p4_dsv_lost_submit_and_import_replies_do_not_allocate_twice(self):
        for command in (
            "d2d_submit_pd",
            "d2d_import",
            "d2d_freeze",
            "d2d_reserve",
            "d2d_precopy_prepare",
            "d2d_precopy_copy",
            "d2d_precopy_apply",
        ):
            with self.subTest(command=command):
                f = self._p4_dsv_env()
                precopy = command.startswith("d2d_precopy_")
                if precopy:
                    f.gw._update(f.rid, cut=7)
                f.driver.after[command] = [TimeoutError("reply lost")]
                for _ in range(40):
                    f.gw.tick()
                    if f.gw.entry(f.rid)["blocked"]:
                        break
                self.assertTrue(f.gw.entry(f.rid)["blocked"])
                if precopy:
                    req = f.e.req
                    req.output_ids.extend([64] * (7 - len(req.output_ids)))
                    req.kv.kv_committed_len = req.kv.kv_allocated_len = (
                        len(req.origin_input_ids) + len(req.output_ids) - 1
                    )
                self._p4_dsv_restart(f)
                f.support.until(f.gw, f.rid, "TARGET")
                self.assertEqual(len(f.e.target.d2d_targets), 1)
                self.assertEqual(len(f.e.resumed), 1)
                self.assertEqual(len(f.e.source.d2d_submissions), 1)
                f.gw.close()
                f.e.close()

    def test_p4_dsv_cancel_precommit_tombstones_late_import_and_freeze(self):
        from sglang_remote_attn import dsv_d2d as d2d

        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "IMPORT")
        frozen = f.gw.entry(f.rid)
        f.gw.cancel(f.rid)
        f.support.until(f.gw, f.rid, "SOURCE")
        self.assertFalse(f.e.req._handoff_frozen)
        self.assertTrue(f.e.req.kv.holds_kv)
        epoch = frozen["epoch"]
        with self.assertRaisesRegex(ValueError, "cancelled"):
            d2d.handle(
                f.e.target,
                dict(
                    cmd="d2d_import",
                    migration_epoch=epoch,
                    destination="D1",
                    transfers=frozen["transfers"],
                ),
            )
        with self.assertRaisesRegex(ValueError, "cancelled"):
            d2d.handle(
                f.e.source,
                dict(
                    cmd="d2d_freeze",
                    migration_epoch=epoch,
                    destination="D1",
                    target_rid=frozen["target_rid"],
                    rid=frozen["source_rid"],
                    reservations=frozen["reservations"],
                ),
            )
        f.support.until(f.gw, f.rid, "CANCELLED")
        self.assertEqual(len(f.driver.cancels), 1)

    def test_p4_dsv_cancel_at_commit_intent_never_resumes_source(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "COMMIT")
        f.gw.cancel(f.rid)
        f.support.until(f.gw, f.rid, "TARGET")
        f.gw.tick()
        self.assertEqual(f.driver.cancels[0][1], f.gw.entry(f.rid)["target_rid"])
        self.assertEqual(f.driver.count("d2d_resume"), 0)
        self.assertEqual(f.driver.count("d2d_abort"), 0)
        self.assertTrue(f.e.req._handoff_frozen)
        self.assertTrue(f.e.req.kv.holds_kv)
        self.assertFalse(f.e.target.d2d_targets[f.gw.entry(f.rid)["epoch"]].copied)
        f.driver.finish_local()
        f.gw.tick()
        self.assertEqual(f.gw.entry(f.rid)["phase"], "CANCELLED")

    def test_p4_dsv_worker_restart_and_missing_status_never_restore_source(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "COMMIT")
        f.e.target.d2d_instance_id = "replacement-worker"
        self._p4_dsv_restart(f)
        f.gw.tick()
        self.assertTrue(f.gw.entry(f.rid)["blocked"])
        self.assertEqual(f.gw.entry(f.rid)["phase"], "COMMIT")
        self.assertEqual(f.driver.count("d2d_commit"), 0)
        self.assertEqual(f.driver.count("d2d_resume"), 0)
        self.assertTrue(f.e.req._handoff_frozen)

    def test_p4_dsv_drain_fences_new_admission_and_waits_for_source_retirement(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "SOURCE")
        state = f.gw.start_drain()
        self.assertEqual(state["state"], "MIGRATING")
        second = f.gw.accept(
            dict(rid="new-target", prompt_ids=[1, 2], max_new_tokens=4)
        )
        self.assertEqual(f.gw.entry(second)["initial_owner"], "dst")
        # Do not submit the second native request into this single-row fixture.
        f.gw._update(second, phase="DONE")
        f.support.until(f.gw, f.rid, "TARGET")
        self.assertNotEqual(f.gw.drain_status()["state"], "SAFE_TO_STOP")
        f.driver.finish_local()
        f.gw.tick()
        self.assertEqual(f.gw.drain_status()["state"], "SAFE_TO_STOP")
        self._p4_dsv_restart(f)
        f.gw.tick()
        self.assertEqual(f.gw.drain_status()["state"], "SAFE_TO_STOP")
        third = f.gw.accept(dict(rid="after-restart", prompt_ids=[1], max_new_tokens=2))
        self.assertEqual(f.gw.entry(third)["owner"], "dst")

    def test_p4_dsv_commit_journal_failure_prevents_commit_rpc(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "IMPORT")
        persist = f.gw._persist

        def fail_commit():
            if f.gw.journal["entries"][f.rid]["phase"] == "COMMIT":
                raise OSError("journal fsync failed")
            persist()

        with patch.object(f.gw, "_persist", side_effect=fail_commit):
            f.gw.tick()
        self.assertEqual(f.driver.count("d2d_commit"), 0)
        self.assertFalse(f.e.source.d2d_sources[f.gw.entry(f.rid)["epoch"]].committed)

    def test_p4_dsv_journal_single_writer_and_corruption_fail_closed(self):
        f = self._p4_dsv_env()
        with self.assertRaises(BlockingIOError):
            f.support.gateway(f.driver, ROOT, f.folder)
        f.gw.close()
        Path(f.folder, "dsv_gateway_journal.json").write_text("{broken")
        with self.assertRaises(ValueError):
            f.support.gateway(f.driver, ROOT, f.folder)

    def test_p4_dsv_resume_before_late_freeze_leaves_source_tombstone(self):
        from sglang_remote_attn import dsv_d2d as d2d

        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "FREEZE")
        before = f.gw.entry(f.rid)
        f.gw.cancel(f.rid)
        f.support.until(f.gw, f.rid, "SOURCE")
        with self.assertRaisesRegex(ValueError, "cancelled"):
            d2d.handle(
                f.e.source,
                dict(
                    cmd="d2d_freeze",
                    migration_epoch=before["epoch"],
                    rid=before["source_rid"],
                    destination="D1",
                    target_rid=before["target_rid"],
                    reservations=before["reservations"],
                ),
            )
        self.assertFalse(f.e.req._handoff_frozen)

    def test_p4_dsv_native_reservation_prevents_idle_allocator_reset(self):
        from sglang_remote_attn.pre_layerwise import is_fully_idle

        e = self._d2d_env()
        self._d2d_reserve(e)
        original = Mock(return_value=True)
        self.assertFalse(is_fully_idle(original, e.target.scheduler))
        original.assert_not_called()
        e.command(e.target, "d2d_reserve_cancel")
        self.assertTrue(is_fully_idle(original, e.target.scheduler))

    def _p4_dsv_http_server(self, gw):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import parse_qs, urlparse

        path = PLUGIN / "control_plane/p4_gateway.py"
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "Handler"
        )
        ns = dict(
            BaseHTTPRequestHandler=BaseHTTPRequestHandler,
            Gateway=object,
            urlparse=urlparse,
            parse_qs=parse_qs,
            json=json,
            uuid=uuid,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(path),
                "exec",
            ),
            ns,
        )
        handler = ns["Handler"]
        handler.gateway = gw
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(2)

        self.addCleanup(close)
        return server

    def test_p4_dsv_real_http_sse_continues_across_cutover_without_duplicates(self):
        import http.client

        f = self._p4_dsv_env(accept=False)
        from sglang.srt.mem_cache.common import release_kv_cache

        server = self._p4_dsv_http_server(f.gw)
        seen, errors = [], []
        connected, initial, done = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def client():
            conn = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            try:
                conn.request(
                    "POST",
                    "/v1/completions",
                    json.dumps(
                        dict(
                            rid="http-request",
                            prompt_ids=list(f.e.req.origin_input_ids),
                            max_new_tokens=8,
                            cut=2,
                            growth_tokens=2,
                        )
                    ),
                    {"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                connected.set()
                while line := response.readline():
                    if line.startswith(b"data: "):
                        payload = line[6:].strip()
                        if payload == b"[DONE]":
                            done.set()
                            break
                        event = json.loads(payload)
                        seen.append(event)
                        if len(seen) == 3:
                            initial.set()
            except BaseException as exc:
                errors.append(repr(exc))
                connected.set()
            finally:
                conn.close()

        thread = threading.Thread(target=client, daemon=True)
        thread.start()
        self.assertTrue(connected.wait(5), errors)
        self.assertFalse(errors)
        f.rid = "http-request"
        f.support.until(f.gw, f.rid, "SOURCE")
        f.gw.tick()
        self.assertTrue(initial.wait(5), (seen, errors))
        f.support.until(f.gw, f.rid, "TARGET")
        req = f.e.target.d2d_targets[f.gw.entry(f.rid)["epoch"]].decode_req.req
        req.output_ids.append(34)
        req.kv.kv_committed_len += 1
        req.kv.kv_allocated_len += 1
        req.finished = lambda: True
        release_kv_cache(req, f.e.target.scheduler.tree_cache, checkpoint=False)
        f.gw.tick()
        self.assertTrue(done.wait(5), (seen, errors))
        thread.join(2)
        self.assertFalse(errors)
        self.assertEqual([e["seq"] for e in seen], [0, 1, 2, 3])
        self.assertEqual([e["token_id"] for e in seen], [31, 32, 33, 34])
        self.assertEqual(seen[-1]["owner"], "dst")
        self.assertTrue(f.e.req.kv.holds_kv)
        self.assertTrue(req.kv.holds_kv)
        f.driver.finish_local()
        f.gw.tick()
        self.assertEqual(f.gw.entry(f.rid)["phase"], "DONE")

    def test_p4_dsv_real_http_reconnect_uses_last_event_id_after_restart(self):
        import http.client

        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "SOURCE")
        f.gw.tick()
        first = self._p4_dsv_http_server(f.gw)
        conn = http.client.HTTPConnection("127.0.0.1", first.server_port, timeout=5)
        conn.request("GET", "/v1/attach?rid=" + f.rid)
        response = conn.getresponse()
        seen = []
        while len(seen) < 2:
            line = response.readline()
            if line.startswith(b"data: "):
                seen.append(json.loads(line[6:]))
        self.assertEqual([e["seq"] for e in seen], [0, 1])
        conn.close()
        self._p4_dsv_restart(f)
        f.support.until(f.gw, f.rid, "TARGET")
        req = f.e.target.d2d_targets[f.gw.entry(f.rid)["epoch"]].decode_req.req
        req.output_ids.append(34)
        req.finished = lambda: True
        f.gw.tick()
        second = self._p4_dsv_http_server(f.gw)
        conn = http.client.HTTPConnection("127.0.0.1", second.server_port, timeout=5)
        conn.request("GET", "/v1/attach?rid=" + f.rid, headers={"Last-Event-ID": "1"})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        resumed = []
        for line in response:
            if line.startswith(b"data: "):
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                resumed.append(json.loads(payload))
        conn.close()
        self.assertEqual([e["seq"] for e in resumed], [2, 3])
        self.assertEqual([e["token_id"] for e in resumed], [33, 34])
        self.assertEqual(f.driver.count("d2d_submit_pd"), 1)
        self.assertEqual(len(f.e.resumed), 1)

    def test_p4_dsv_backend_rejects_restart_between_status_and_mutation(self):
        f = self._p4_dsv_env()
        send = f.driver.send

        def replace(role, payload, timeout):
            if payload["cmd"] == "d2d_submit_pd":
                f.e.source.d2d_instance_id = "replacement-during-rpc"
            return send(role, payload, timeout)

        with patch.object(f.driver, "send", side_effect=replace):
            f.gw.tick()
        self.assertTrue(f.gw.entry(f.rid)["blocked"])
        self.assertFalse(hasattr(f.e.source, "d2d_submissions"))
        self.assertFalse(f.e.req._handoff_frozen)

    def test_p4_dsv_native_admission_exception_after_commit_cannot_resume_or_replay(
        self,
    ):
        from sglang_remote_attn import handoff_control as rt

        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "COMMIT")
        with patch.object(
            rt,
            "_flush_decode_resume",
            side_effect=RuntimeError("native admission interrupted"),
        ):
            f.gw.tick()
        entry = f.gw.entry(f.rid)
        self.assertEqual(entry["phase"], "COMMIT")
        self.assertTrue(entry["blocked"])
        self.assertTrue(f.e.source.d2d_sources[entry["epoch"]].committed)
        f.gw.cancel(f.rid)
        self._p4_dsv_restart(f)
        f.gw.tick()
        self.assertEqual(f.driver.count("d2d_resume"), 0)
        self.assertEqual(f.driver.count("d2d_abort"), 0)
        self.assertTrue(f.e.req.kv.holds_kv)
        self.assertTrue(
            f.e.target.d2d_targets[entry["epoch"]].decode_req.req.kv.holds_kv
        )

    def test_p4_dsv_duplicate_source_resume_does_not_enqueue_twice(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "IMPORT")
        entry = f.gw.entry(f.rid)
        f.gw.cancel(f.rid)
        f.support.until(f.gw, f.rid, "SOURCE")
        self.assertEqual(len(f.e.resumed), 1)
        result = f.driver.send(
            "src", dict(cmd="d2d_resume", migration_epoch=entry["epoch"])
        )
        self.assertTrue(result["resumed"])
        self.assertTrue(result["duplicate"])
        self.assertEqual(len(f.e.resumed), 1)

    def test_p4_dsv_waits_for_previous_native_pd_remote_before_planning(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "SOURCE")
        f.e.req._dsv41_remote = object()
        for _ in range(3):
            f.gw.tick()
        self.assertEqual(f.gw.entry(f.rid)["phase"], "SOURCE")
        self.assertEqual(f.driver.count("d2d_plan"), 0)
        f.e.req._dsv41_remote = None
        f.support.until(f.gw, f.rid, "TARGET")

    def test_p4_dsv_native_prefill_http_thread_survives_gateway_restart(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from sglang_remote_attn import dsv_gateway_control as control

        f = self._p4_dsv_env(accept=False)
        arrived, release = threading.Event(), threading.Event()
        bodies = []

        class Prefill(BaseHTTPRequestHandler):
            def do_POST(self):
                bodies.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                arrived.set()
                release.wait(5)
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Prefill)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(release.set)
        f.gw.prefill_url = f"http://127.0.0.1:{server.server_port}/generate"
        f.rid = f.gw.accept(
            dict(
                rid="native-p",
                prompt_ids=list(f.e.req.origin_input_ids),
                max_new_tokens=8,
            )
        )
        with patch.object(control, "prefill", side_effect=f.driver.native_prefill):
            f.gw.tick()
            self.assertTrue(arrived.wait(5))
            worker_thread = f.e.req._d2d_prefill_thread
            self._p4_dsv_restart(f)
            self.assertTrue(worker_thread.is_alive())
            f.gw.tick()
            self.assertEqual(f.driver.count("d2d_submit_pd"), 1)
            release.set()
            worker_thread.join(5)
        self.assertFalse(worker_thread.is_alive())
        self.assertFalse(getattr(f.e.req, "_d2d_submission_error", None))
        self.assertEqual(len(bodies), 1)
        self.assertEqual(
            bodies[0]["bootstrap_room"], f.gw.entry(f.rid)["bootstrap_room"]
        )
        self.assertEqual(bodies[0]["sampling_params"]["max_new_tokens"], 1)

    def test_p4_dsv_async_import_cancel_waits_for_target_producer(self):
        """A cancelled async import retains its native row while setup can write."""
        from sglang_remote_attn import dsv_d2d as d2d

        f = self._p4_dsv_env()
        entry = f.support.until(f.gw, f.rid, "IMPORT")
        f.gw.async_handoff = True
        f.gw.background = True
        entered, release = threading.Event(), threading.Event()
        call = d2d.Reader.call

        def delayed(reader, action, **fields):
            if action == "manifest":
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("test did not release import")
            return call(reader, action, **fields)

        def tick():
            f.gw._refresh()
            f.gw.tick()

        with patch.object(d2d.Reader, "call", delayed):
            try:
                tick()
                self.assertTrue(entered.wait(5))
                target = f.e.target.d2d_targets[entry["epoch"]]
                f.gw.cancel(f.rid)
                tick()
                tick()
                self.assertEqual(f.gw.entry(f.rid)["phase"], "ROLLBACK")
                self.assertTrue(target.decode_req.req.kv.holds_kv)
                self.assertFalse(target.admitted or f.e.resumed)
            finally:
                release.set()
            target.setup_worker.join(5)
            self.assertFalse(target.setup_worker.is_alive())
            tick()
            self.assertEqual(f.gw.entry(f.rid)["phase"], "SOURCE")
            f.e.target.drain_targets(required=target)
            self.assertFalse(target.decode_req.req.kv.holds_kv)
            self.assertTrue(f.e.req.kv.holds_kv)
            self.assertFalse(f.e.req._handoff_frozen)

    def test_p4_dsv_async_handoff_overlaps_epochs_and_recovers_commit(self):
        """Neither a blocked import nor a blocked commit may hold D1's command slot."""
        from sglang_remote_attn import dsv_d2d as d2d

        _, e = self._partial_d2d_env(2)
        with patch.object(
            sys, "path", [str(ROOT / "test/manual/remote_attn"), *sys.path]
        ):
            from p4_dsv_probe_support import Driver, gateway

        for i in range(2):
            e.arrive(i)
        e.complete_pd()
        e.gate.set()
        driver = Driver.__new__(Driver)
        driver.controllers = dict(src=e.source, dst=e.target)
        driver.calls, driver.before, driver.after = [], {}, {}
        folder = self.enterContext(tempfile.TemporaryDirectory())

        def start():
            gw = gateway(driver, ROOT, folder, async_handoff=True)
            self.addCleanup(gw.close)
            gw.background = True  # Run production reconciliation with explicit ticks.
            gw._refresh()
            return gw

        gw = start()
        rids = []
        for req in e.requests:
            rid = gw.accept(
                dict(
                    rid=req.rid,
                    prompt_ids=list(req.origin_input_ids),
                    cut=2,
                    max_new_tokens=8,
                )
            )
            gw._update(rid, phase="SOURCE", source_rid=req.rid)
            rids.append(rid)

        gates = {action: threading.Event() for action in ("manifest", "commit")}
        entered = {action: set() for action in gates}
        lock = threading.Lock()
        call = d2d.Reader.call

        def delayed(reader, action, **fields):
            if action in gates:
                with lock:
                    entered[action].add(reader.epoch)
                if not gates[action].wait(15):
                    raise TimeoutError(f"test did not release {action}")
            return call(reader, action, **fields)

        def until(predicate):
            deadline = time.monotonic() + 5
            while not predicate():
                phases = [gw.entry(r)["phase"] for r in rids]
                self.assertLess(time.monotonic(), deadline, phases)
                gw._refresh()
                gw.tick()
                time.sleep(0.001)

        with patch.object(d2d.Reader, "call", delayed):
            try:
                until(lambda: len(entered["manifest"]) == 2)
                self.assertTrue(all(gw.entry(r)["phase"] == "IMPORT" for r in rids))
                self.assertFalse(e.resumed)
                gates["manifest"].set()
                until(lambda: len(entered["commit"]) == 2)
                self.assertTrue(all(gw.entry(r)["phase"] == "COMMIT" for r in rids))
                self.assertFalse(e.resumed)
                epoch = gw.entry(rids[0])["epoch"]
                with self.assertRaisesRegex(ValueError, "pending commit"):
                    d2d.handle(e.target, dict(cmd="d2d_abort", migration_epoch=epoch))
                gw.close()
                gw = start()
                self.assertTrue(all(gw.entry(r)["owner"] == "src" for r in rids))
                gates["commit"].set()
                for target in e.target.d2d_targets.values():
                    target.commit_worker.join(5)
                    self.assertTrue(target.source_confirmed)
                    self.assertFalse(target.admitted)
                    self.assertIsNotNone(target.worker)
                    self.assertFalse(target.defer_bulk)  # The source has a separate bulk queue.
                from sglang_remote_attn import handoff_control as rt

                # The first COMMIT wave admits both requests, but its reply is
                # lost. Retrying the same epochs must not enqueue them twice.
                driver.after["d2d_control_wave"] = [TimeoutError("lost commit wave reply")]
                with patch.object(
                    rt, "_flush_decode_resume", wraps=rt._flush_decode_resume
                ) as admit:
                    until(lambda: all(gw.entry(r)["owner"] == "dst" for r in rids))
                self.assertEqual(admit.call_count, 1)
                self.assertEqual(len(admit.call_args.args[1]), 2)
                self.assertEqual(len(e.resumed), 1)
                self.assertEqual(len(e.resumed[0][1]), 2)
                self.assertEqual(len(e.target.d2d_targets), 2)
                self.assertFalse(any(p["cmd"] == "d2d_resume" for _, p in driver.calls))
            finally:
                for gate in gates.values():
                    gate.set()

    def test_p4_dsv_background_pollers_execute_native_migration(self):
        f = self._p4_dsv_env(accept=False)
        f.gw.close()
        f.gw = f.support.gateway(
            f.driver, ROOT, f.folder, autostart=True, async_handoff=True
        )
        self.addCleanup(f.gw.close)
        deadline = time.monotonic() + 5
        while len(f.gw.facts) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        f.rid = f.gw.accept(
            dict(
                rid="background",
                prompt_ids=list(f.e.req.origin_input_ids),
                max_new_tokens=8,
                cut=2,
                growth_tokens=0,
            )
        )
        while f.gw.entry(f.rid)["phase"] != "TARGET" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(f.gw.entry(f.rid)["phase"], "TARGET", f.gw.entry(f.rid))
        self.assertEqual(f.gw.entry(f.rid)["owner"], "dst")
        self.assertEqual(len(f.e.resumed), 1)

    def test_p4_dsv_drain_never_claims_safe_with_unmanaged_source_queue(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "TARGET")
        f.gw.start_drain()
        f.driver.finish_local()
        foreign = SimpleNamespace(
            rid="unmanaged",
            output_ids=[],
            kv=SimpleNamespace(kv_committed_len=0, holds_kv=False),
            finished=lambda: False,
        )
        f.e.source.scheduler.waiting_queue.append(foreign)
        f.gw.tick()
        self.assertEqual(f.gw.entry(f.rid)["phase"], "LOCAL")
        self.assertNotEqual(f.gw.drain_status()["state"], "SAFE_TO_STOP")
        f.e.source.scheduler.waiting_queue.clear()
        f.gw.tick()
        self.assertEqual(f.gw.drain_status()["state"], "SAFE_TO_STOP")

    def test_p4_dsv_stopped_source_is_not_polled_after_gateway_restart(self):
        import http.client

        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "TARGET")
        f.gw.start_drain()
        f.driver.finish_local()
        f.gw.tick()
        self.assertEqual(f.gw.drain_status()["state"], "SAFE_TO_STOP")
        server = self._p4_dsv_http_server(f.gw)
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request(
            "POST", "/v1/drain/stop", "{}", {"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read())["state"], "STOPPED")
        conn.close()
        self._p4_dsv_restart(f)
        send = f.driver.send

        def stopped(role, payload, timeout):
            if role == "src":
                raise AssertionError("stopped source polled")
            return send(role, payload, timeout)

        with patch.object(f.driver, "send", side_effect=stopped):
            f.gw.tick()
        self.assertNotIn("src", f.gw.fact_errors)
        self.assertEqual(f.gw.drain_status()["state"], "STOPPED")
        self.assertFalse(f.gw.entry(f.rid)["blocked"])

    def test_p4_dsv_finished_sse_history_survives_source_and_gateway_restart(self):
        f = self._p4_dsv_env()
        f.support.until(f.gw, f.rid, "SOURCE")
        from sglang.srt.mem_cache.common import release_kv_cache

        f.e.req.finished = lambda: True
        release_kv_cache(f.e.req, f.e.source.scheduler.tree_cache, checkpoint=False)
        f.gw.tick()
        self.assertEqual(f.gw.entry(f.rid)["phase"], "DONE")
        f.e.source.d2d_instance_id = "new-source-without-history"
        f.e.source.scheduler.handoff_tracked_reqs.clear()
        f.e.source.scheduler.running_batch.reqs.clear()
        self._p4_dsv_restart(f)
        f.gw.tick()
        events, finished = f.gw.events(f.rid, after=1)
        self.assertEqual([e["token_id"] for e in events], [33])
        self.assertTrue(finished)

    def test_p4_dsv_real_cli_sigkill_commit_recovery_and_sse_attach(self):
        import http.client
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        f = self._p4_dsv_env(accept=False)
        f.gw.close()
        commit_applied, release_reply = threading.Event(), threading.Event()
        role_locks = {role: threading.Lock() for role in ("src", "dst")}

        class Proxy(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                role = self.path.lstrip("/")
                with role_locks[role]:
                    try:
                        result = f.driver.send(role, payload)
                        if (
                            payload["cmd"] == "d2d_commit"
                            and not commit_applied.is_set()
                        ):
                            commit_applied.set()
                            release_reply.wait(10)
                        status = 200
                    except Exception as exc:
                        result, status = (
                            dict(error=repr(exc), type=type(exc).__name__),
                            500,
                        )
                    data = json.dumps(result).encode()
                    try:
                        self.send_response(status)
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

            def log_message(self, *args):
                pass

        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        proxy.daemon_threads = True
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(proxy.server_close)
        self.addCleanup(proxy.shutdown)
        self.addCleanup(release_reply.set)
        worker = ROOT / "test/manual/remote_attn/p4_dsv_gateway_worker.py"
        processes = []
        state_dir = Path(f.folder)

        def start(index):
            ready = state_dir / f"ready-{index}.json"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(worker),
                    "--source-root",
                    str(ROOT),
                    "--proxy-port",
                    str(proxy.server_port),
                    "--state",
                    str(state_dir),
                    "--ready",
                    str(ready),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )
            processes.append(process)
            deadline = time.monotonic() + 8
            while (
                not ready.exists()
                and process.poll() is None
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertTrue(
                ready.exists(),
                process.stderr.read()
                if process.poll() is not None
                else "startup timed out",
            )
            return process, json.loads(ready.read_text())["port"]

        try:
            process, port = start(1)
            deadline = time.monotonic() + 5
            while (
                f.driver.count("d2d_worker_status") < 2 and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            time.sleep(0.05)
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
            conn.request(
                "POST",
                "/v1/completions",
                json.dumps(
                    dict(
                        rid="sigkill",
                        prompt_ids=list(f.e.req.origin_input_ids),
                        max_new_tokens=8,
                        cut=2,
                        growth_tokens=0,
                    )
                ),
                {"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            first = []
            while len(first) < 2:
                line = response.readline()
                if line.startswith(b"data: "):
                    item = json.loads(line[6:])
                    if "seq" in item:
                        first.append(item["seq"])
            self.assertTrue(commit_applied.wait(5))
            self.assertEqual(first, [0, 1])
            journal = json.loads((state_dir / "dsv_gateway_journal.json").read_text())
            self.assertEqual(journal["entries"]["sigkill"]["phase"], "COMMIT")
            process.kill()
            process.wait(5)
            conn.close()
            release_reply.set()
            _, port = start(2)
            deadline = time.monotonic() + 8
            entry = None
            while time.monotonic() < deadline:
                entry = json.loads(
                    (state_dir / "dsv_gateway_journal.json").read_text()
                )["entries"]["sigkill"]
                if entry["phase"] == "TARGET":
                    break
                time.sleep(0.01)
            self.assertEqual(entry["phase"], "TARGET", entry)
            req = f.e.target.d2d_targets[entry["epoch"]].decode_req.req
            req.output_ids.append(34)
            req.kv.kv_committed_len += 1
            req.kv.kv_allocated_len += 1
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
            conn.request(
                "GET", "/v1/attach?rid=sigkill", headers={"Last-Event-ID": "1"}
            )
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            resumed = []
            while len(resumed) < 2:
                line = response.readline()
                if line.startswith(b"data: "):
                    item = json.loads(line[6:])
                    if "seq" in item:
                        resumed.append((item["seq"], item["token_id"]))
            conn.close()
            self.assertEqual(resumed, [(2, 33), (3, 34)])
            self.assertEqual(len(f.e.resumed), 1)
            self.assertEqual(f.driver.count("d2d_resume"), 0)
            self.assertEqual(f.driver.count("d2d_submit_pd"), 1)
            self.assertTrue(f.e.req.kv.holds_kv)
        finally:
            release_reply.set()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate(timeout=5)

    def test_d2d_group_width_validation_and_bounded_native_frames(self):
        from sglang_remote_attn import d2d_groups as groups

        for value in (0, -1, True, 1.5, "bad"):
            with self.assertRaises(ValueError):
                groups.size(value)
        entries = [
            ["kv", 2, 0, "v4", "u8", 4],
            ["index", 2, 0, "v4", "u8", 4],
            ["kv", 4, 1, "v4", "u8", 4],
            ["kv", 2, 4, "v4", "u8", 4],
        ]
        frames = list(groups.frames(entries, 2, 4, 12))
        self.assertEqual(
            frames,
            [
                [dict(entry=0, start=0, count=2), dict(entry=2, start=0, count=1)],
                [dict(entry=2, start=1, count=1)],
                [dict(entry=3, start=0, count=2)],
            ],
        )
        self.assertTrue(
            all(
                sum(p["count"] * entries[p["entry"]][-1] for p in frame) <= 12
                for frame in frames
            )
        )

    def test_qwen_grouped_fast_batches_preserve_slot_ack_and_handle_reuse(self):
        for slots in (1, 2, 3):
            with self.subTest(slots=slots):
                self._fast_sender_contract(group=2, slots=slots, reuse=True)
        self._fast_sender_contract(group=3, slots=3, async_=False)

    def test_qwen_grouped_fast_unknown_submit_retains_staging(self):
        for failure in ("sync", "async"):
            with self.subTest(failure=failure):
                self._fast_sender_contract(failure, group=2, slots=2)

    def test_dsv_grouped_bulk_packs_producers_and_preserves_forward_fence(self):
        e = self._d2d_env()
        with patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="4"):
            meta = e.reserve_and_freeze()
        e.import_(meta)  # Target follows the frozen source config, not its env.
        self.assertTrue(e.blocked.wait(3))
        target = e.target.d2d_targets[e.epoch]
        self.assertEqual(target.group_size, 4)
        e.command(e.target, "d2d_commit")
        pending = [False]
        target.forward_proofs.append(SimpleNamespace(ready=lambda: pending[0]))
        e.gate.set()
        e.wait_copy()
        self.assertEqual(target.group_frames, 1)
        self.assertEqual(target.grouped_entries, 2)
        self.assertEqual(len(target.events), 2)
        self.assertFalse(target.ready_to_finish())
        self.assertFalse(e.freed)
        pending[0] = True
        e.target.drain_targets(required=target)
        self.assertTrue(target.closed)

    def test_dsv_grouped_entry_can_become_local_before_its_group_peer(self):
        from sglang_remote_attn import dsv41_receive as receive

        e = self._d2d_env()
        e.gate.set()
        original = receive.scatter_prefix
        second, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        calls = []

        def scatter(*args, **kw):
            calls.append(True)
            if len(calls) == 2:
                second.set()
                self.assertTrue(release.wait(5))
            return original(*args, **kw)

        with patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="4"):
            meta = e.reserve_and_freeze()
        with patch.object(receive, "scatter_prefix", side_effect=scatter):
            e.import_(meta)
            target = e.target.d2d_targets[e.epoch]
            self.assertTrue(second.wait(3))
            e.command(e.target, "d2d_commit")
            first, last = target.resources
            self.assertTrue(first.ready_to_finish())
            self.assertFalse(last.ready_to_finish())
            e.target.decode_overlap.retirement.drain(first)
            self.assertIn(first.entry, target.rpc.local_entries)
            self.assertNotIn(last.entry, target.rpc.local_entries)
            self.assertFalse(target.closed)
            self.assertTrue(e.req.kv.holds_kv)
            release.set()
            e.wait_copy()
        e.target.drain_targets(required=target)
        self.assertTrue(target.closed)

    def test_dsv_grouped_transport_failure_cannot_abort_or_resume_storage(self):
        from sglang_remote_attn import dsv_d2d as d2d

        e = self._d2d_env()
        with patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="4"):
            meta = e.reserve_and_freeze()
        e.import_(meta)
        self.assertTrue(e.blocked.wait(3))
        engine = d2d.manager(e.source).engine
        with patch.object(engine, "batch_transfer_sync", return_value=-1):
            e.gate.set()
            target = e.target.d2d_targets[e.epoch]
            target.worker.join(5)
        self.assertTrue(target.error)
        self.assertTrue(target.bulk_reader.bulk.unsafe)
        self.assertFalse(target.events)
        with self.assertRaises(d2d.TransportFailure):
            e.command(e.target, "d2d_abort")
        with self.assertRaises(ValueError):
            e.command(e.source, "d2d_resume")
        self.assertTrue(e.req.kv.holds_kv)
        self.assertTrue(target.decode_req.req.kv.holds_kv)
        self.assertFalse(e.freed)

    def test_dsv_grouped_cancel_and_lost_commit_reply_preserve_p4_semantics(self):
        with patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="4"):
            self.test_p4_dsv_lost_commit_reply_rolls_forward_after_restart_without_resampling()
            self.test_p4_dsv_cancel_precommit_tombstones_late_import_and_freeze()

    def test_dsv_grouped_invalid_config_rejects_before_freeze(self):
        e = self._d2d_env()
        with patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="0"):
            with self.assertRaisesRegex(ValueError, "GROUP_SIZE"):
                e.freeze()
        self.assertFalse(e.req._handoff_frozen)
        self.assertEqual(e.source.scheduler.running_batch.reqs, [e.req])

    def test_dsv_grouped_source_validates_all_segments_before_any_native_write(self):
        from sglang_remote_attn import dsv_d2d as d2d, d2d_groups as groups

        e = self._d2d_env()
        with patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="4"):
            e.freeze()
        source = e.source.d2d_sources[e.epoch]
        entries = [
            i for i, entry in enumerate(source.schema["entries"]) if entry[0] == "kv"
        ]
        with self.assertRaisesRegex(ValueError, "budget"):
            groups.validate_segments(
                source, [dict(entry=entries[0], start=0, count=1)], 1
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            groups.validate_segments(
                source,
                [dict(entry=entries[0], start=0, count=source.pages.numel() + 1)],
                d2d.LIMIT,
            )
        source.group_size = 1
        with self.assertRaisesRegex(ValueError, "crosses"):
            groups.validate_segments(
                source, [dict(entry=i, start=0, count=1) for i in entries], d2d.LIMIT
            )

    def _qwen_native_fixture(self, page=4, capacity=64):
        path = ROOT / "test/manual/remote_attn/qwen_reservation_probe_support.py"
        spec = importlib.util.spec_from_file_location(
            "qwen_reservation_probe_support", path
        )
        module = importlib.util.module_from_spec(spec)
        self.enterContext(patch.dict(sys.modules, {spec.name: module}))
        spec.loader.exec_module(module)
        f = module.Fixture(ROOT, page=page, capacity=capacity)
        self.addCleanup(f.close)
        req = f.new(
            f.scheduler, "source", list(range(5)), [11, 12, 13], module.Sampling()
        )
        req.kv.req_pool_idx = 0
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 7
        f.scheduler.handoff_tracked_reqs[req.rid] = req
        return f, req

    def _qwen_native_reserve(self, f, req, growth=8):
        from sglang_remote_attn import qwen_reservation as reservation

        data = dict(
            epoch="range-epoch",
            items=[dict(rid=req.rid, target_rid="target", growth_tokens=growth)],
        )
        planned = reservation.plan(f.scheduler, data)["plans"]
        result = reservation.reserve(
            f.scheduler, dict(epoch=data["epoch"], plans=[planned])
        )
        reservation.confirm(
            f.scheduler,
            dict(epoch=data["epoch"], certificates=[result["reservations"]]),
        )
        return planned[0]

    def test_debug_queue_uses_native_extend_fields_and_live_held_lists(self):
        """Queue diagnostics must report real intervals/owners instead of deleted fields."""
        from sglang.srt.managers.scheduler_components.request_list import ListKind

        req = SimpleNamespace(
            rid="queued",
            kv=SimpleNamespace(kv_committed_len=3, req_pool_idx=1),
            seqlen=9,
            output_ids=[],
            prefix_len=3,
            extend_end=9,
            extend_len=6,
            inflight_steps=1,
            finished=lambda: False,
        )
        scheduler = SimpleNamespace(waiting_queue=[req], request_lists=lambda: {})
        for owner in ("none", "native", "plugin"):
            with self.subTest(owner=owner):
                scheduler.request_lists = lambda: (
                    {ListKind.PD_HELD: [req]} if owner == "native" else {}
                )
                scheduler._handoff_held_reqs = [req] if owner == "plugin" else []
                result = self.runtime._handle(scheduler, dict(cmd="debug_queue"))
                row = result["waiting"][0]
                self.assertEqual(row["extend_range"], dict(start=3, end=9, length=6))
                self.assertEqual(row["held"], owner != "none")
                self.assertEqual(row["inflight_steps"], 1)
        req.extend_end = None
        self.assertIsNone(
            self.runtime._handle(scheduler, dict(cmd="debug_queue"))["waiting"][0][
                "extend_range"
            ]
        )

    def test_qwen_native_reservation_reuses_row_and_preserves_partial_tail_page(self):
        from sglang_remote_attn import qwen_reservation as reservation

        f, req = self._qwen_native_fixture()
        before = f.allocator.available_size()
        with patch.object(f.queue, "_pre_alloc", wraps=f.queue._pre_alloc) as allocate:
            p = self._qwen_native_reserve(f, req)
            self.assertEqual(allocate.call_count, 1)
            self.assertEqual(p["upper"], 15)
            self.assertEqual(before - f.allocator.available_size(), 16)
            self.assertEqual(f.pool.available_size(), 2)
            reservation.reserve(f.scheduler, dict(epoch=p["epoch"], plans=[[p]]))
            self.assertEqual(allocate.call_count, 1)
            meta = dict(p, kv_committed_len=7)
            native, indices = reservation.consume(
                f.scheduler, "target", meta, p["epoch"]
            )
            self.assertEqual(before - f.allocator.available_size(), 8)
            self.assertEqual(indices.numel(), 7)
            with self.assertRaisesRegex(ValueError, "consumed"):
                self.runtime._handle(
                    f.scheduler,
                    dict(cmd="reserve_cancel", rids=["target"], epoch=p["epoch"]),
                )
            row = native.kv.req_pool_idx
            self.runtime.bind_decode_prefix(native, f.pool, indices, 7)
            self.assertEqual(native.kv.req_pool_idx, row)
            self.assertEqual(f.pool.available_size(), 2)
            self.assertEqual(allocate.call_count, 1)
            self.runtime._free_source_request(f.scheduler, native)
            self.runtime._free_source_request(f.scheduler, native)
            self.assertEqual(f.allocator.available_size(), before)
            self.assertEqual(f.pool.available_size(), 3)

    def test_qwen_native_reservation_rejects_overflow_before_source_freeze(self):
        from sglang_remote_attn import qwen_reservation as reservation

        f, req = self._qwen_native_fixture()
        with self.assertRaisesRegex(ValueError, "requires"):
            reservation.check_freeze(f.scheduler, req, "range-epoch")
        p = self._qwen_native_reserve(f, req, growth=1)
        req.kv.kv_committed_len = p["upper"] + 1
        with self.assertRaisesRegex(ValueError, "interval"):
            reservation.check_freeze(f.scheduler, req, p["epoch"])
        self.assertFalse(getattr(req, "_handoff_frozen", False))
        self.assertIsNone(f.scheduler._handoff_reserved["target"].get("consumed_at"))

    def test_qwen_native_reservation_failed_or_consumed_allocation_cannot_cancel(self):
        from sglang_remote_attn import qwen_reservation as reservation

        f, req = self._qwen_native_fixture()
        actual = f.queue._pre_alloc

        def fail(native):
            actual(native)
            raise RuntimeError("native allocation reply lost")

        with patch.object(f.queue, "_pre_alloc", side_effect=fail):
            with self.assertRaises(reservation.TransportFailure):
                self._qwen_native_reserve(f, req)
        rec = f.scheduler._handoff_reserved["target"]
        self.assertTrue(rec["native_req"].kv.holds_kv)
        with self.assertRaises(reservation.TransportFailure):
            reservation.cancel(f.scheduler, "target", rec)
        with self.assertRaises(ValueError):
            reservation.reserve(
                f.scheduler, dict(epoch="range-epoch", plans=[[rec["plan"]]])
            )

    def test_qwen_native_reservation_checks_real_paged_budget_before_allocating(self):
        f, req = self._qwen_native_fixture(capacity=8)
        before = f.pool.available_size()
        with patch.object(f.queue, "_pre_alloc", wraps=f.queue._pre_alloc) as allocate:
            with self.assertRaisesRegex(ValueError, "preflight"):
                self._qwen_native_reserve(f, req)
            self.assertEqual(allocate.call_count, 0)
            self.assertEqual(f.pool.available_size(), before)
            self.assertFalse(
                f.scheduler._handoff_reserved
                if hasattr(f.scheduler, "_handoff_reserved")
                else {}
            )

    def test_d2d_rectangles_preserve_group_boundaries_and_reject_incomplete_coverage(
        self,
    ):
        from sglang_remote_attn import d2d_ranges as ranges, qwen_ranges

        meta = dict(
            chunks=[[]] * 5, kv_committed_len=11, d2d_group_size=3, d2d_token_range=4
        )
        work = list(qwen_ranges.batches(meta, 2))
        self.assertEqual(
            work[:4], [((0, 1), 0, 4), ((2,), 0, 4), ((0, 1), 4, 8), ((2,), 4, 8)]
        )
        self.assertTrue(
            all(len({layer // 3 for layer in g}) == 1 for g, lo, hi in work)
        )
        coverage = ranges.Coverage(5, 11)
        for g, lo, hi in work:
            for layer in g:
                self.assertEqual(coverage.add(layer, lo, hi), hi == 11)
        with self.assertRaises(ValueError):
            coverage.add(0, 8, 11)
        with self.assertRaises(ValueError):
            ranges.Coverage(1, 11).add(0, 4, 8)
        for bad in (0, -1, True, 1.5, "oops"):
            with self.assertRaises(ValueError):
                ranges.tokens(bad)

    def test_dsv_rectangles_cover_every_entry_page_once_with_bounded_frames(self):
        from sglang_remote_attn import d2d_groups as groups, d2d_ranges as ranges

        entries = [
            ["kv", 2, 1, "v4", "uint8", 7],
            ["kv", 4, 2, "v4", "uint8", 13],
            ["kv", 4, 4, "v4", "uint8", 13],
        ]
        frames = list(groups.range_frames(entries, 19, 4, 3, 9, 32))
        coverage = ranges.Coverage(3, 5)
        for rect in frames:
            lo, hi = rect["token_range"]
            self.assertEqual(lo % 4, 0)
            self.assertLessEqual(
                sum(p["count"] * entries[p["entry"]][-1] for p in rect["segments"]), 32
            )
            for seg in rect["segments"]:
                self.assertEqual(entries[seg["entry"]][2] // 3, rect["layer_group"])
                self.assertTrue(
                    lo // 4
                    <= seg["start"]
                    < seg["start"] + seg["count"]
                    <= (hi + 3) // 4
                )
                coverage.add(seg["entry"], seg["start"], seg["start"] + seg["count"])
        self.assertEqual(coverage.ends, [5, 5, 5])

    def test_qwen_native_reservation_manifest_mismatch_does_not_consume_row(self):
        from sglang_remote_attn import qwen_reservation as reservation

        f, req = self._qwen_native_fixture()
        p = self._qwen_native_reserve(f, req)
        bad = dict(p, kv_committed_len=7, origin_input_ids=[99])
        with self.assertRaisesRegex(ValueError, "manifest|history"):
            reservation.consume(f.scheduler, "target", bad, p["epoch"])
        rec = f.scheduler._handoff_reserved["target"]
        self.assertFalse(rec["consumed"])
        wrong_geometry = dict(
            p,
            kv_committed_len=7,
            d2d_range_protocol="layer-token-range-v1",
            chunks=[[7]] * 2,
            kv_heads=1,
            head_dim=8,
            dtype="float16",
        )
        with self.assertRaisesRegex(ValueError, "geometry"):
            reservation.consume(f.scheduler, "target", wrong_geometry, p["epoch"])
        self.assertFalse(rec["consumed"])
        row = rec["native_req"].kv.req_pool_idx
        reservation.cancel(f.scheduler, "target", rec)
        self.assertIsNone(rec["native_req"].kv.req_pool_idx)
        self.assertNotIn("target", f.scheduler._handoff_reserved)
        self.assertIn(row, f.pool.free_slots)

    def test_qwen_native_cancel_before_reserve_leaves_epoch_tombstone(self):
        from sglang_remote_attn import qwen_reservation as reservation

        f, req = self._qwen_native_fixture()
        p = reservation.plan(
            f.scheduler,
            dict(
                epoch="range-epoch",
                items=[dict(rid="source", target_rid="target", growth_tokens=8)],
            ),
        )["plans"][0]
        self.runtime._handle(
            f.scheduler, dict(cmd="reserve_cancel", rids=["target"], epoch=p["epoch"])
        )
        with patch.object(f.queue, "_pre_alloc", wraps=f.queue._pre_alloc) as allocate:
            with self.assertRaises(ValueError):
                reservation.reserve(f.scheduler, dict(epoch=p["epoch"], plans=[[p]]))
            self.assertEqual(allocate.call_count, 0)

    def test_qwen_native_reservation_accounts_for_ordinary_leases(self):
        f, req = self._qwen_native_fixture()
        f.scheduler._handoff_ordinary_leases = {"ordinary": dict(tokens=60)}
        with patch.object(f.queue, "_pre_alloc", wraps=f.queue._pre_alloc) as allocate:
            with self.assertRaisesRegex(ValueError, "preflight"):
                self._qwen_native_reserve(f, req)
            self.assertEqual(allocate.call_count, 0)

    def test_qwen_native_freeze_rejects_reused_source_request_identity(self):
        from sglang_remote_attn import qwen_reservation as reservation

        f, req = self._qwen_native_fixture()
        p = self._qwen_native_reserve(f, req)
        replacement = copy.copy(req)
        with self.assertRaisesRegex(ValueError, "original source"):
            reservation.check_freeze(f.scheduler, replacement, p["epoch"])
        f.scheduler.handoff_tracked_reqs["source"] = replacement
        with self.assertRaisesRegex(ValueError, "identity"):
            reservation.plan(
                f.scheduler,
                dict(epoch=p["epoch"], items=[dict(rid="source", target_rid="target")]),
            )

    def test_qwen_native_tp2_rejects_partial_freeze_and_asymmetric_reservation(self):
        worker = ROOT / "test/manual/remote_attn/qwen_reservation_tp_worker.py"
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join(
                    [str(PLUGIN), os.environ.get("PYTHONPATH", "")]
                ),
                PYTHONDONTWRITEBYTECODE="1",
            )
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        str(worker),
                        str(rank),
                        str(ROOT),
                        str(folder / "rendezvous"),
                        str(folder / f"rank{rank}.json"),
                    ],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for rank in range(2)
            ]
            try:
                output = [p.communicate(timeout=40)[0] for p in processes]
                for p in processes:
                    self.assertEqual(p.returncode, 0, "\n".join(output))
                for rank in range(2):
                    row = json.loads((folder / f"rank{rank}.json").read_text())
                    self.assertTrue(
                        row["asymmetric_freeze_rejected"]
                        and row["asymmetric_ticket_rejected"]
                        and row["unproven_row_retained"]
                    )
            finally:
                for p in processes:
                    if p.poll() is None:
                        p.kill()
                        p.communicate(timeout=5)

    def test_qwen_range_registration_failure_keeps_host_backing_without_owner_handle(
        self,
    ):
        from sglang_remote_attn import qwen_ranges, mooncake_backend as mb

        runtime = self.runtime
        engine = SimpleNamespace(
            protocol="tcp",
            d2d_protocol_verified=True,
            get_session_id=lambda: "test-registration",
        )
        meta = dict(
            d2d_capabilities=mb.tcp_offer(),
            chunks=[[4]],
            kv_committed_len=4,
            kv_heads=1,
            head_dim=1,
            dtype="float16",
            d2d_group_size=1,
            d2d_token_range=4,
            d2d_range_protocol="layer-token-range-v1",
        )
        original = torch.empty

        def empty(*args, **kw):
            kw.pop("pin_memory", None)
            return original(*args, **kw)

        manager = SimpleNamespace(
            register=Mock(side_effect=RuntimeError("registration result lost")),
            release_safe=Mock(return_value=True),
        )
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(runtime, "_get_te", return_value=engine),
            patch.object(runtime, "_TRANSFER_REGISTRATIONS", {}),
            patch.object(runtime, "_get_registration_manager", return_value=manager),
            patch.object(runtime, "_CTL_DIR", ""),
            patch.object(runtime, "_TRANSFER_QUARANTINE", {}),
            patch.object(runtime, "_TRANSFER_RELEASES", {}),
            patch.object(torch, "empty", side_effect=empty),
            patch.object(qwen_ranges.secrets, "token_hex", return_value="reg-lost"),
        ):
            with self.assertRaisesRegex(RuntimeError, "registration"):
                qwen_ranges.source(None, SimpleNamespace(), meta, False)
            self.assertIn("reg-lost", runtime._TRANSFER_QUARANTINE)
            self.assertFalse(runtime._TRANSFER_RELEASES["reg-lost"]["gpu_safe"])
            with self.assertRaisesRegex(RuntimeError, "registration"):
                qwen_ranges.setup(None, meta, "target-lost")
            self.assertIn("target-lost", runtime._TRANSFER_QUARANTINE)
            self.assertFalse(runtime._TRANSFER_RELEASES["target-lost"]["gpu_safe"])
            runtime.reap_transfer_buffers()
            manager.release_safe.assert_not_called()

    def test_d2d_history_and_reservation_bounds_reject_before_native_allocation(self):
        from sglang_remote_attn import qwen_reservation as reservation

        e = self._d2d_env()
        meta = e.freeze()
        meta["output_ids"].append(99)
        with self.assertRaisesRegex(ValueError, "history"):
            e.import_(meta)
        self.assertFalse(e.target.d2d_targets)
        f, req = self._qwen_native_fixture()
        p = reservation.plan(
            f.scheduler,
            dict(
                epoch="bad-history",
                items=[dict(rid="source", target_rid="target", growth_tokens=8)],
            ),
        )["plans"][0]
        p["lower"] += 1
        with patch.object(f.queue, "_pre_alloc", wraps=f.queue._pre_alloc) as allocate:
            with self.assertRaisesRegex(ValueError, "preflight"):
                reservation.reserve(f.scheduler, dict(epoch=p["epoch"], plans=[[p]]))
            self.assertEqual(allocate.call_count, 0)
            self.assertEqual(f.pool.available_size(), 3)


if __name__ == "__main__":
    unittest.main()
