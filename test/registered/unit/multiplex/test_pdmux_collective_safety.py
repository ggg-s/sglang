"""Regressions for collectives behind local readiness and shared workspaces."""

from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_pdmux_sxf_port import load_methods, register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CACHES = (
    ("mem_cache/hiradix_cache.py", "HiRadixCache"),
    ("mem_cache/unified_radix_cache.py", "UnifiedRadixCache"),
)


def local_poll(path, owner):
    return load_methods(path, owner, ["is_load_back_event_done"])[
        "is_load_back_event_done"
    ]


class TestHiCacheLocalPolling(unittest.TestCase):
    def test_readiness_never_starts_a_collective(self):
        for path, owner in CACHES:
            for ready in (False, True):
                with self.subTest(cache=owner, ready=ready):
                    event = NS(query=Mock(return_value=ready))
                    cache = NS(
                        cache_controller=NS(
                            layer_done_counter=NS(events=[NS(finish_event=event)])
                        ),
                        loading_check=Mock(
                            side_effect=AssertionError(
                                "rank-local poll entered a collective"
                            )
                        ),
                    )
                    self.assertIs(local_poll(path, owner)(cache, 0), ready)
                    cache.loading_check.assert_not_called()
                    self.assertTrue(local_poll(path, owner)(cache, -1))
                    event.query.assert_called_once()

    def test_unified_without_a_controller_is_already_ready(self):
        self.assertTrue(local_poll(*CACHES[1])(NS(cache_controller=None), 0))


class TestFusionWorkspaceSafety(unittest.TestCase):
    def fusion(self, pdmux):
        manager = Mock(
            side_effect=AssertionError("shared fusion workspace accessed under PDMux")
        )
        functions = load_methods(
            "layers/flashinfer_comm_fusion.py",
            None,
            [
                "is_flashinfer_allreduce_unavailable",
                "resolve_flashinfer_allreduce_fusion_backend",
                "ensure_workspace_initialized",
                "can_use_flashinfer_allreduce",
                "flashinfer_allreduce_residual_rmsnorm",
            ],
            _flashinfer_allreduce_unavailable=False,
            _flashinfer_comm=object(),
            is_pdmux_enabled=lambda: pdmux,
            get_exec=lambda: NS(comm=NS(flashinfer_allreduce_fusion_backend="auto")),
            get_parallel=lambda: NS(nnodes=1, attn_tp_size=2),
            _resolve_backend=lambda backend, multi_node: "trtllm",
            _get_workspace_manager=manager,
            is_flashinfer_available=lambda: True,
            logger=Mock(),
        )
        return NS(**functions), manager

    def test_pdmux_rejects_both_fusion_entry_points_before_workspace_access(self):
        fusion, manager = self.fusion(True)
        self.assertTrue(fusion.is_flashinfer_allreduce_unavailable())
        self.assertIsNone(fusion.resolve_flashinfer_allreduce_fusion_backend())
        self.assertFalse(fusion.ensure_workspace_initialized())
        tensor = NS(shape=(1, 8), is_contiguous=lambda: True)
        self.assertFalse(
            fusion.can_use_flashinfer_allreduce(
                tensor,
                use_attn_tp_group=True,
                expected_world_size=2,
                expected_group_key=(object(), object()),
            )
        )
        self.assertEqual(
            fusion.flashinfer_allreduce_residual_rmsnorm(tensor, tensor, tensor),
            (None, None),
        )
        manager.assert_not_called()

    def test_non_pdmux_keeps_the_requested_fusion_backend(self):
        fusion, _ = self.fusion(False)
        self.assertFalse(fusion.is_flashinfer_allreduce_unavailable())
        self.assertEqual(fusion.resolve_flashinfer_allreduce_fusion_backend(), "trtllm")

    def test_equal_width_different_groups_or_ranks_cannot_reuse_a_workspace(self):
        initialize = load_methods(
            "layers/flashinfer_comm_fusion.py",
            "FlashInferWorkspaceManager",
            ["initialize"],
            torch=NS(bfloat16="bf16"),
            logger=Mock(),
            _flashinfer_comm=None,
            _create_allreduce_fusion_workspace=None,
        )["initialize"]
        device_group, cpu_group = object(), object()
        for changes in (
            {},
            {"device_group": object()},
            {"cpu_group": object()},
            {"rank": 1},
            {"world_size": 4},
        ):
            with self.subTest(changes=changes):
                manager = NS(
                    _max_token_num_seen=128,
                    _max_hidden_dim_seen=256,
                    initialized=True,
                    world_size=2,
                    rank=0,
                    group=(device_group, cpu_group),
                    is_buffer_size_sufficient=Mock(return_value=True),
                    cleanup=Mock(),
                )
                kwargs = dict(
                    world_size=2,
                    rank=0,
                    max_token_num=128,
                    hidden_dim=256,
                    device_group=device_group,
                    cpu_group=cpu_group,
                )
                kwargs.update(changes)
                initialize(manager, **kwargs)
                self.assertEqual(manager.cleanup.call_count, int(bool(changes)))

    def test_legacy_rendezvous_group_is_part_of_workspace_identity(self):
        initialize = load_methods(
            "layers/flashinfer_comm_fusion.py",
            "FlashInferWorkspaceManager",
            ["initialize"],
            torch=NS(bfloat16="bf16"),
            logger=Mock(),
            _flashinfer_comm=None,
            _create_allreduce_fusion_workspace=None,
        )["initialize"]
        old_group = object()
        for group, cleanups in ((old_group, 0), (object(), 1)):
            with self.subTest(same_group=group is old_group):
                manager = NS(
                    _max_token_num_seen=128,
                    _max_hidden_dim_seen=256,
                    initialized=True,
                    world_size=2,
                    rank=0,
                    group=(None, old_group),
                    is_buffer_size_sufficient=Mock(return_value=True),
                    cleanup=Mock(),
                )
                initialize(
                    manager,
                    world_size=2,
                    rank=0,
                    max_token_num=128,
                    hidden_dim=256,
                    group=group,
                )
                self.assertEqual(manager.cleanup.call_count, cleanups)


def _polling_rank(rank, store_path):
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        for path, owner in CACHES:
            poll = local_poll(path, owner)
            for tick in range(32):
                # One rank can have completed and stopped polling while its
                # peer still waits for DMA. None of these local polls may
                # consume the next scheduler collective's sequence number.
                ready = (tick + rank) % 3 == 0
                cache = NS(
                    cache_controller=NS(
                        layer_done_counter=NS(
                            events=[NS(finish_event=NS(query=lambda: ready))]
                        )
                    ),
                    loading_check=lambda: dist.all_reduce(
                        torch.ones(1, dtype=torch.int64)
                    ),
                )
                for _ in range(rank + 1):
                    assert poll(cache, 0) == ready
                control = torch.tensor([tick + 1], dtype=torch.int64)
                dist.all_reduce(control)
                assert control.item() == 2 * (tick + 1)
    finally:
        dist.destroy_process_group()


class TestHiCachePollingGloo(unittest.TestCase):
    def test_unequal_readiness_and_poll_counts_preserve_collective_order(self):
        try:
            import torch.distributed as dist
        except ImportError:
            self.skipTest("PyTorch required for the two-rank Gloo test")
        if not dist.is_gloo_available():
            self.skipTest("Gloo unavailable")
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="hicache-poll-gloo-") as tmp:
            processes = [
                ctx.Process(target=_polling_rank, args=(rank, str(Path(tmp) / "store")))
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            try:
                for process in processes:
                    process.join(timeout=25)
                    self.assertFalse(
                        process.is_alive(),
                        "HiCache polling desynchronized the TP ranks",
                    )
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
