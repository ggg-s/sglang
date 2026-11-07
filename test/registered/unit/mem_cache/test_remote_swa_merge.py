"""Merged native CP release and Remote retained-owner contracts, on CPU.

Load production functions without importing GPU-only attention kernels. The CP
coordinate conversion helpers are also production code, with a two-rank config.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def load(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    definitions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            definitions.append(node)
    assert {node.name for node in definitions} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)


def runtime(cp=True):
    namespace = dict(
        torch=torch,
        ComponentType=NS(SWA="swa"),
        ceil_align=lambda value, page: (value + page - 1) // page * page,
        is_cp_sharded_padded=lambda: cp,
        get_cp_strategy=lambda: NS(cp_size=2),
        get_schedule=lambda: NS(page_size=2),
    )
    load(
        "layers/cp/utils.py",
        {
            "get_cp_sharded_padded",
            "from_cp_sharded",
            "get_cp_sharded_padded_cache_indices",
            "get_cp_sharded_padded_cache_transfer_indices",
            "from_cp_sharded_expanded_indices",
        },
        namespace,
    )
    load(
        "mem_cache/common.py",
        {
            "free_swa_out_of_window_slots",
            "free_kv_row_segments",
            "get_cp_sharded_padded_kv_segments",
        },
        namespace,
    )
    return namespace


def kv(*, held=True, allocated=40, evicted=0, retained=()):
    record = NS(
        holds_kv=held,
        req_pool_idx=0,
        cache_protected_len=0,
        kv_allocated_len=allocated,
        evicted=evicted,
        swa_retained_ranges=list(retained),
        swa_dead_lo=lambda page: 0,
    )
    record.get_evicted_seqlen = lambda kind: record.evicted
    record.set_evicted_seqlen = lambda kind, value: setattr(record, "evicted", value)
    return record


class TestRemoteSwaMerge(CustomTestCase):
    def test_eviction_keeps_retained_logical_ranges_and_converts_each_free(self):
        for cp in (False, True):
            with self.subTest(cp=cp):
                ns = runtime(cp)
                local = torch.arange(2, 34, dtype=torch.int64)
                row = ns["from_cp_sharded_expanded_indices"](local) if cp else local
                record = kv(retained=((8, 16), (24, 32)))
                calls = []
                allocator = NS(
                    free_swa_segment=lambda indices, *, start_pos: calls.append(
                        (indices.tolist(), start_pos)
                    )
                )
                ns["free_swa_out_of_window_slots"](
                    NS(kv=record),
                    44 if cp else 36,
                    sliding_window_size=4,
                    page_size=2,
                    req_to_token_pool=NS(req_to_token=row.unsqueeze(0)),
                    token_to_kv_pool_allocator=allocator,
                    supports_prefix_sharing=False,
                )
                expected = (
                    [
                        (local[0:4].tolist(), 0),
                        (local[8:12].tolist(), 8),
                        (local[16:20].tolist(), 16),
                    ]
                    if cp
                    else [(local[0:8].tolist(), 0), (local[16:24].tolist(), 16)]
                )
                self.assertEqual(calls, expected)
                self.assertEqual(record.evicted, 40 if cp else 32)
                self.assertEqual(record.swa_retained_ranges, [(8, 16), (24, 32)])

    def test_retire_window_uses_native_storage_and_preserves_held_dma_pages(self):
        ns = runtime(cp=False)
        load("mem_cache/static_kv_storage.py", {"__init__", "advance"}, ns)
        ns["StaticKVStorage"] = type(
            "StaticKVStorage", (), {name: ns[name] for name in ("__init__", "advance")}
        )
        load("mem_cache/static_kv_storage.py", {"kv_storage_of"}, ns)
        load("mem_cache/allocation.py", {"_storage", "retire_window"}, ns)
        freed = []
        row = torch.arange(2, 66, dtype=torch.int64)
        allocator = NS(
            storage=None,
            page_size=2,
            free_swa_segment=lambda ids, *, start_pos: freed.extend(ids.tolist()),
        )
        cache = NS(
            req_to_token_pool=NS(req_to_token=row.unsqueeze(0)),
            token_to_kv_pool_allocator=allocator,
            window_retention=lambda req: [("swa", 4, None)],
            supports_prefix_sharing=lambda: False,
        )
        record = kv(retained=[(8, 16), (24, 32)])
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.mem_cache.common": NS(
                    free_swa_out_of_window_slots=ns["free_swa_out_of_window_slots"]
                ),
                "sglang.srt.mem_cache.unified_cache.component_type": NS(
                    ComponentType=ns["ComponentType"]
                ),
            },
        ):
            ns["retire_window"](NS(kv=record), 44, cache)
        self.assertEqual(freed, torch.cat((row[:8], row[16:24], row[32:40])).tolist())
        self.assertFalse(set(freed) & set(torch.cat((row[8:16], row[24:32])).tolist()))
        self.assertEqual(record.evicted, 40)

    def test_final_cp_release_frees_retained_swa_once_and_keeps_order(self):
        ns = runtime()
        local = torch.arange(2, 22, dtype=torch.int64)
        row = ns["from_cp_sharded_expanded_indices"](local)
        full, both = [], []
        allocator = NS(
            page_size=2,
            free_full_segments=lambda segments: full.extend(segments),
            free_segments=lambda segments: both.extend(segments),
        )
        ns["free_kv_row_segments"](
            allocator,
            [(row, 0)],
            swa_evicted_seqlen=32,
            swa_dead_lo=4,
            swa_retained_ranges=[(8, 16), (24, 32)],
        )
        self.assertEqual([start for _, start in full], [2, 8])
        self.assertEqual([start for _, start in both], [0, 4, 12, 16])
        full_ids = torch.cat([ids for ids, _ in full]).tolist()
        both_ids = torch.cat([ids for ids, _ in both]).tolist()
        self.assertEqual(full_ids, [4, 5, 10, 11, 12, 13])
        self.assertEqual(sorted(full_ids + both_ids), local.tolist())
        self.assertEqual(len(set(full_ids + both_ids)), len(local))

    def test_native_request_lists_and_external_owners_deduplicate_kv(self):
        namespace = {}
        load(
            "managers/scheduler_components/invariant_checker.py",
            {"_external_kv_owners", "_requests_owning_rows"},
            namespace,
        )
        active, remote, session, released = kv(), kv(), kv(), kv(held=False)
        req = lambda record: NS(kv=record)
        checker = NS(
            get_request_lists=lambda: {
                "running": [req(active)],
                "transfer": [req(active)],
            },
            get_external_kv_owners=lambda: [
                req(remote),
                req(active),
                req(remote),
                req(session),
                req(released),
            ],
            tree_cache=NS(session_records=lambda: {"session": session}),
        )
        checker._external_kv_owners = MethodType(
            namespace["_external_kv_owners"], checker
        )
        owners = namespace["_requests_owning_rows"](checker)
        self.assertEqual([id(owner.kv) for owner in owners], [id(active), id(remote)])

    def test_external_held_swa_uses_current_pool_stats_capability(self):
        namespace = dict(
            ComponentType=NS(SWA="swa"), ceil_align=lambda x, p: (x + p - 1) // p * p
        )
        load(
            "managers/scheduler_components/pool_stats_observer.py",
            {"kv_private_tokens", "kv_private_swa_tokens"},
            namespace,
        )
        load(
            "managers/scheduler_components/invariant_checker.py",
            {"_check_all_pools"},
            namespace,
        )
        calls = []

        def check(kind, ps, *, uncached=0):
            calls.append((kind, uncached))
            return False, kind

        checker = NS(
            page_size=2,
            _external_kv_owners=lambda: [NS(kv=kv(evicted=32, retained=[(8, 16)]))],
            _check_full_pool=lambda ps, **kw: check("full", ps, **kw),
            _check_swa_pool=lambda ps, **kw: check("swa", ps, **kw),
        )
        namespace["_check_all_pools"](
            checker, NS(is_hybrid_swa=True, is_hybrid_ssm=False)
        )
        self.assertEqual(calls, [("full", 40), ("swa", 16)])


if __name__ == "__main__":
    unittest.main()
