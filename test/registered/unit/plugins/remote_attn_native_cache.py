"""Source-load the real CPU-compatible cache implementation without CUDA imports.

Checkpoint, tree insertion/splitting, FULL/SWA components, dedup actions, locks,
matching and eviction are unmodified production methods. Only startup, disabled
session/event services and GPU memory allocation are replaced. The page ledger
rejects double/foreign frees; ownership assertions compare it with actual tree
values and request rows, independently of eviction cursors/retention metadata.
"""

import ast
import dataclasses
import enum
import heapq
import itertools
import logging
import sys
import time
import typing
from abc import ABC, abstractmethod
from array import array
from collections import defaultdict
from enum import Enum, IntFlag, auto
from types import SimpleNamespace as NS
from typing import NamedTuple, Protocol
from unittest.mock import patch

import msgspec
import numpy as np
import torch


class Pages:
    def __init__(self, size, page):
        self.size, self.page = size, page
        self.free = set(range(page, size + page, page))
        self.live = set()

    def alloc(self, n):
        n = (n + self.page - 1) // self.page
        assert len(self.free) >= n, "pool exhausted"
        bases = sorted(self.free)[:n]
        self.free.difference_update(bases)
        self.live.update(bases)
        return torch.tensor(
            [v for b in bases for v in range(b, b + self.page)], dtype=torch.int64
        )

    def release(self, ids):
        bases = [v // self.page * self.page for v in ids.tolist()[:: self.page]]
        assert len(set(bases)) == len(bases), "duplicate pages within free"
        assert set(bases) <= self.live, "double free or foreign page"
        self.live.difference_update(bases)
        self.free.update(bases)

    def free_segment(self, ids, *, start_pos):
        self.release(ids)

    def available_size(self):
        return len(self.free) * self.page


class Allocator:
    def __init__(self, full=300000, swa=35072, page=256):
        self.device = "cpu"
        self.page_size = page
        self.full_attn_allocator = Pages(full // page * page, page)
        self.swa_attn_allocator = Pages(swa // page * page, page)
        self.full_to_swa_index_mapping = torch.zeros(full + page, dtype=torch.int64)
        self._swa_req_ring = False

    def alloc(self, n):
        full = self.full_attn_allocator.alloc(n)
        swa = self.swa_attn_allocator.alloc(n)
        self.full_to_swa_index_mapping[full] = swa
        return full

    def translate_loc_from_full_to_swa(self, ids):
        out = self.full_to_swa_index_mapping[ids]
        assert torch.all(out > 0), "reading freed/unallocated SWA mapping"
        return out

    translate_swa_indices_for_transfer = translate_loc_from_full_to_swa

    def clear_full_to_swa_mapping(self, ids):
        self.full_to_swa_index_mapping[ids] = 0

    def set_full_to_swa_mapping(self, ids, swa):
        self.full_to_swa_index_mapping[ids] = swa

    def free_swa_segment(self, ids, *, start_pos):
        ids = self._expand(ids)
        assert start_pos % self.page_size == 0
        if not ids.numel():
            return
        idx = self.full_to_swa_index_mapping[ids]
        assert torch.all(idx > 0), "double SWA free"
        self.swa_attn_allocator.release(idx)
        self.full_to_swa_index_mapping[ids] = 0

    def _expand(self, ids):
        return torch.tensor(
            [
                v
                for b in ids.tolist()[:: self.page_size]
                for v in range(
                    b // self.page_size * self.page_size,
                    b // self.page_size * self.page_size + self.page_size,
                )
            ],
            dtype=torch.int64,
        )

    def free_full_segment(self, ids, *, start_pos):
        ids = self._expand(ids)
        assert not torch.any(self.full_to_swa_index_mapping[ids] > 0), (
            "leaking mapped SWA on FULL-only free"
        )
        self.full_attn_allocator.release(ids)

    def free_segment(self, ids, *, start_pos):
        self.free_swa_segment(ids, start_pos=start_pos)
        self.free_full_segment(ids, start_pos=start_pos)

    def free_segments(self, parts):
        for ids, pos in parts:
            self.free_segment(ids, start_pos=pos)

    def free_full_segments(self, parts):
        for ids, pos in parts:
            self.free_full_segment(ids, start_pos=pos)

    def swa_available_size(self):
        return self.swa_attn_allocator.available_size()

    def full_available_size(self):
        return self.full_attn_allocator.available_size()


def fixture(root, testcase):
    ns = dict(
        __name__=__name__,
        torch=torch,
        np=np,
        msgspec=msgspec,
        dataclasses=dataclasses,
        dataclass=dataclasses.dataclass,
        replace=dataclasses.replace,
        array=array,
        Enum=Enum,
        IntFlag=IntFlag,
        auto=auto,
        ABC=ABC,
        abstractmethod=abstractmethod,
        Protocol=Protocol,
        NamedTuple=NamedTuple,
        defaultdict=defaultdict,
        heapq=heapq,
        time=time,
        sys=sys,
        logger=logging.getLogger(__name__),
        float64=np.float64,
        PoolName=NS(KV="kv", SWA="swa"),
        rank_consensus=lambda func=None, **kw: (
            func if func is not None else lambda f: f
        ),
        ceil_align=lambda n, p: (n + p - 1) // p * p,
    )
    ns.update(logging=logging, enum=enum)
    ns.update({k: v for k, v in vars(typing).items() if not k.startswith("__")})
    ns["__name__"] = __name__

    def load(path, names=None, assign=False):
        tree = ast.parse((root / "python/sglang/srt" / path).read_text())
        nodes = [
            n
            for n in tree.body
            if (
                isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and (names is None or n.name in names)
            )
            or (assign and isinstance(n, (ast.Assign, ast.AnnAssign)))
        ]
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
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
                ),
                str(root / "python/sglang/srt" / path),
                "exec",
            ),
            ns,
        )

    load("mem_cache/unified_cache/component_type.py", assign=True)
    load("disaggregation/kv_events.py", {"StorageMedium"})
    load("mem_cache/base_prefix_cache.py")
    load("mem_cache/evict_policy.py", {"EvictionStrategy", "LRUStrategy"})
    load("mem_cache/utils.py", {"split_node_hash_value"})
    load("mem_cache/radix_cache.py", {"RadixKey"})
    load("mem_cache/unified_cache/cache_action.py", assign=True)
    load("mem_cache/unified_cache/components/base.py", assign=True)
    load("mem_cache/unified_cache/unified_tree_core_interface.py", assign=True)
    load("mem_cache/unified_cache/unified_tree_core.py", assign=True)
    load("mem_cache/unified_cache/components/full.py")
    load("mem_cache/unified_cache/components/swa.py")
    load("mem_cache/unified_radix_cache.py", {"UnifiedRadixCache"})
    load("managers/schedule_batch.py", {"ReqKvInfo"})
    load(
        "mem_cache/common.py",
        {
            "free_swa_out_of_window_slots",
            "coalesce_ranges",
            "free_kv_row_segments",
            "release_kv_cache",
            "_release_overallocated_kv_indices",
            "_req_radix_key",
        },
    )
    from sglang.srt.environ import envs

    ns.update(
        envs=envs,
        get_eviction_strategy=lambda *a: ns["LRUStrategy"](),
        get_spec=lambda: NS(speculative_algorithm=None),
        get_serving=lambda: NS(strip_thinking_cache=False),
        is_virtual_node=lambda node: False,
    )
    events = NS(
        enabled=False,
        record_all_cleared=lambda: None,
        record_store=lambda *a, **kw: None,
        record_remove=lambda *a, **kw: None,
    )
    ns["KVCacheEventRecorder"] = lambda **kw: events
    cache = ns["UnifiedRadixCache"].__new__(ns["UnifiedRadixCache"])
    allocator = Allocator()
    pool = NS(req_to_token=torch.zeros((8, 150000), dtype=torch.int64))
    pool.write = lambda where, ids: pool.req_to_token.__setitem__(where, ids)
    pool.free = lambda req: setattr(req.kv, "req_pool_idx", None)
    cache.req_to_token_pool = pool
    cache.token_to_kv_pool_allocator = allocator
    cache.disable = False
    cache.enable_session_radix_cache = False
    cache.is_mamba_enabled = False
    cache.is_swa_enabled = True
    cache._sliding_window_size = 128
    cache.draft_swa_window = 0
    cache.decoder_swa_bounded_replay = True
    cache.buffer_pipeline = cache.linker = cache.cache_controller = None
    cache.host_memory_mode = None
    cache._storage_prefetch_wait = {}
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache.ongoing_prefetch = {}
    cache.storage_prefetch_retries = NS(cancel=lambda rid: None)
    cache.session = NS(
        try_checkpoint=lambda *a, **kw: False,
        try_match_prefix=lambda p: None,
        will_lend_record=lambda req: False,
        try_cache_finished_req=lambda req: False,
    )
    ct = ns["ComponentType"]
    cache.tree_components = (ct.FULL, ct.SWA)
    full = ns["FullComponent"](cache, NS())
    swa = ns["SWAComponent"].__new__(ns["SWAComponent"])
    ns["TreeComponent"].__init__(swa, cache, NS())
    swa._session_leaf_covered_len = {}
    swa.sliding_window_size = 128
    swa.gates_prefix_match = True
    swa.full_window_pages = 1
    swa._swa_kv_pool_host = None
    cache.components = {ct.FULL: full, ct.SWA: swa}
    cache._components_tuple = (full, swa)
    params = NS(
        page_size=256,
        is_eagle=False,
        enable_session_radix_cache=False,
        eviction_policy="lru",
        eviction_policy_config=None,
        token_to_kv_pool_allocator=allocator,
        enable_kv_cache_events=False,
        dynamo_kv_event_format=False,
        kv_event_lora_names=None,
    )
    cache.tree_core = ns["UnifiedTreeCore"](params, cache.components)
    # Only GPU-independent import boundaries, not checkpoint/tree logic.
    common = NS(
        **{
            n: ns[n]
            for n in (
                "free_swa_out_of_window_slots",
                "coalesce_ranges",
                "free_kv_row_segments",
            )
        }
    )
    testcase.enterContext(
        patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.common": common,
                "sglang.srt.mem_cache.unified_cache.component_type": NS(
                    ComponentType=ct
                ),
                "sglang.srt.layers.attention.dsv4.unified_kv_gate": NS(
                    is_unified_kv_triton=lambda: False
                ),
                "sglang.srt.mem_cache.allocator.unified_hybrid_swa": NS(
                    UnifiedSWAAllocatorBase=type("UnifiedSWAAllocatorBase", (), {})
                ),
            },
        )
    )
    request_count = itertools.count()
    requests = []

    def request(n=32772, tokens=None, finished=True, match=False):
        idx = next(request_count)
        tokens = array("q", range(n)) if tokens is None else array("q", tokens)
        req = NS(
            kv=ns["ReqKvInfo"](req_pool_idx=idx),
            rid=str(idx),
            cache_request_handle=ns["CacheRequestHandle"](str(idx), 0),
            origin_input_ids=tokens,
            full_untruncated_fill_ids=tokens,
            extra_key=None,
            cache_salt=None,
            priority=0,
            session_id=None,
            kv_rotation_base=None,
            swa_branching_seqlen=None,
            finished=lambda: finished,
            skip_radix_cache_insert=False,
            finished_len=None,
            refresh_fill_ids=lambda: None,
            owned_kv_len=lambda: len(tokens),
        )
        prefix = 0
        req.last_node = cache.tree_core.root_node.id
        if match:
            key = ns["_req_radix_key"](
                cache, req, tokens, max_prefix_len=len(tokens) - 1
            )
            hit = cache.match_prefix(ns["MatchPrefixParams"](key=key))
            prefix = hit.device_prefix_len
            req.last_node = hit.last_device_node
            req.swa_branching_seqlen = hit.swa_branching_seqlen
            pool.req_to_token[idx, :prefix] = cache.path_device_indices(req.last_node)
        req.lock = cache.lock(req.last_node)
        req.prefix_len = prefix
        req.kv.cache_protected_len = prefix
        req.kv.kv_allocated_len = req.kv.kv_committed_len = prefix
        requests.append(req)
        return req

    def extend(req, end):
        old = (req.kv.kv_allocated_len + 255) // 256 * 256
        new = (end + 255) // 256 * 256
        if new > old:
            pool.req_to_token[req.kv.req_pool_idx, old:new] = allocator.alloc(new - old)
        req.kv.kv_allocated_len = req.kv.kv_committed_len = end

    def ownership():
        owned_full, owned_swa = set(), set()

        def add(target, ids):
            pages = {i // 256 * 256 for i in ids.tolist() if i > 0}
            assert not target.intersection(pages), "page has two independent owners"
            target.update(pages)

        for node in cache.tree_core._node_arena.values():
            if node is cache.tree_core.root_node:
                continue
            for ct_, target in ((ct.FULL, owned_full), (ct.SWA, owned_swa)):
                values = node.component_data[ct_].value
                if values is not None:
                    add(target, values)
        for req in requests:
            if not req.kv.holds_kv:
                continue
            end = (req.kv.kv_allocated_len + 255) // 256 * 256
            ids = pool.req_to_token[
                req.kv.req_pool_idx, req.kv.cache_protected_len : end
            ]
            add(owned_full, ids)
            add(owned_swa, allocator.full_to_swa_index_mapping[ids])
        testcase.assertEqual(allocator.full_attn_allocator.live, owned_full)
        testcase.assertEqual(allocator.swa_attn_allocator.live, owned_swa)

    def release(req, checkpoint=False):
        ns["release_kv_cache"](req, cache, checkpoint=checkpoint)
        ownership()

    def evict_all():
        cache.evict(ns["EvictParams"](num_tokens=300000, swa_num_tokens=35072))
        ownership()
        testcase.assertEqual(allocator.full_attn_allocator.live, set())
        testcase.assertEqual(allocator.swa_attn_allocator.live, set())

    return NS(
        cache=cache,
        allocator=allocator,
        pool=pool,
        request=request,
        extend=extend,
        ns=ns,
        ownership=ownership,
        release=release,
        evict_all=evict_all,
    )
