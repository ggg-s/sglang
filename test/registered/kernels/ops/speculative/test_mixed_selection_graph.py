"""Real CUDA capture/replay and asynchronous route feedback contracts."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.mixed_selection import (
    ACTIONS,
    OfflineMixedSelector,
    RouteFeatures,
)
from sglang.srt.speculative.mixed_selection_graph import (
    GraphEntry,
    OfflineAdaptiveGraphs,
    RouteMailbox,
    route_features,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def ids(n, modulus=256):
    return (
        (torch.arange(n, device="cuda")[:, None] * 13 + torch.arange(8, device="cuda"))
        % modulus
    ).to(torch.int32)


class TestMixedSelectionGraph(unittest.TestCase):
    def test_histogram_matches_reference_and_replays_new_ids(self):
        for p in (23, 65, 257, 512):
            left, right = ids(p), ids(24)
            route_features(left, right)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                stats = route_features(left, right)
            for modulus in (256, 64, 32):
                left.copy_(ids(p, modulus))
                right.copy_(ids(24, modulus))
                graph.replay()
                hp = torch.bincount(left.flatten().long(), minlength=256)
                hd = torch.bincount(right.flatten().long(), minlength=256)
                h = hp + hd
                values = stats.cpu().tolist()
                mass = (p + 24) * 8
                effective = min(
                    256, mass * (mass - 8) / max(int((h * h).sum()) - mass, 1)
                )
                self.assertAlmostEqual(values[3], effective, delta=0.0001)
                self.assertEqual(values[0], mass)
                self.assertEqual(values[1], 0)
                self.assertEqual(values[2], int((h > 0).sum()))
                for index, counts in ((8, hp), (11, hd), (14, h)):
                    for j, block in enumerate((8, 16, 32)):
                        self.assertEqual(
                            values[index + j],
                            int(((counts + block - 1) // block).sum()),
                        )
                RouteFeatures(tuple(values))
            left.fill_(-1)
            graph.replay()
            with self.assertRaises(ValueError):
                RouteFeatures(tuple(stats.cpu().tolist()))

    def test_mailbox_ring_never_overwrites_pending_host_reads(self):
        box = RouteMailbox()
        stats = route_features(ids(32), ids(24))
        self.assertTrue(box.submit(stats, step=0, context=1))
        self.assertTrue(box.submit(stats, step=1, context=1))
        self.assertFalse(box.submit(stats, step=2, context=1))
        torch.cuda.synchronize()
        latest = box.poll(step=2, context=1, max_age=2)
        self.assertEqual(latest[0], 1)
        self.assertEqual(box.completed, 2)
        self.assertIsNone(box.poll(step=3, context=2, max_age=2))

    def test_replay_switches_prebuilt_graphs_without_sync_item_or_capture(self):
        p, d = ids(32), ids(24)
        x = torch.randn(128, device="cuda")
        entries = {}
        for name in ACTIONS:
            route_features(p, d)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = x * 2
                stats = route_features(p, d)
            entries[name] = GraphEntry(graph, output, stats, (p, d, x), validated=True)
        samples = []
        for effective, costs in (
            (32, (1000, 800, 1000, 700, 1000, 750)),
            (256, (1000, 800, 1000, 700, 1000, 600)),
        ):
            vector = ((32 + 24) * 8, 0, effective, effective, 0.1, 32, 24, 20) + (
                10,
            ) * 9
            samples.append(
                dict(
                    operator="toy",
                    P=32,
                    capacity=32,
                    features=vector,
                    costs_us=dict(zip(ACTIONS, costs)),
                    correctness_passed=True,
                )
            )
        policy = OfflineMixedSelector(samples, context={})
        runner = OfflineAdaptiveGraphs(
            entries,
            policy,
            operator="toy",
            p=32,
            capacity=32,
            sample_every=1,
            max_age=2,
        )
        for step in range(4):
            if step == 2:
                p.copy_(ids(32, 32))
                d.copy_(ids(24, 32))
                x.mul_(0.5)
            with (
                patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("GPU item")
                ),
                patch("torch.cuda.synchronize", side_effect=AssertionError("sync")),
                patch("torch.cuda.graph", side_effect=AssertionError("recapture")),
            ):
                choice = runner.replay(step=step, context=1)
            torch.cuda.synchronize()
            torch.testing.assert_close(entries[choice.action].outputs, x * 2)
        self.assertGreaterEqual(runner.transitions, 2)
        prior = policy.choose("toy", 32, 32).predicted_us
        choice = runner.replay(step=100, context=1)
        self.assertEqual(choice.predicted_us, prior)
        choice = runner.replay(step=0, context=2)
        self.assertEqual(choice.predicted_us, prior)
        with self.assertRaises(ValueError):
            OfflineAdaptiveGraphs({}, policy, operator="toy", p=32, capacity=32)


if __name__ == "__main__":
    unittest.main()
