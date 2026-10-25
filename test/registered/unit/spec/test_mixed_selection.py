"""GPU-free cost-table validation and geometry/routing decisions."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ci_spec = importlib.util.spec_from_file_location(
    "_mixed_selection_ci", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(ci_spec)
sys.modules[ci_spec.name] = ci
ci_spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "_offline_mixed_selection",
    ROOT / "python/sglang/srt/speculative/mixed_selection.py",
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
ACTIONS, RouteFeatures, Policy = (
    module.ACTIONS,
    module.RouteFeatures,
    module.OfflineMixedSelector,
)


def features(p, effective):
    return (
        (p + 24) * 8,
        0,
        min(effective, (p + 24) * 8),
        effective,
        0.1,
        20,
        20,
        10,
    ) + (10,) * 9


def sample(p, effective, costs, capacity=512):
    return dict(
        operator="gdn",
        P=p,
        capacity=capacity,
        features=features(p, effective),
        costs_us=dict(zip(ACTIONS, costs)),
        correctness_passed=True,
    )


class TestMixedSelection(unittest.TestCase):
    def policy(self):
        return Policy(
            [
                sample(257, 32, (9, 20, 8, 19, 10, 18)),
                sample(257, 256, (20, 22, 18, 16, 21, 15)),
                sample(512, 32, (20, 19, 18, 16, 20, 17)),
                sample(512, 256, (30, 29, 25, 22, 27, 18)),
            ],
            context={"device": "test"},
            switch_us=0,
        )

    def test_padding_and_routing_choose_different_graphs(self):
        policy = self.policy()
        self.assertEqual(
            policy.choose("gdn", 257, 512, RouteFeatures(features(257, 32))).action,
            "m16_separate",
        )
        self.assertEqual(
            policy.choose("gdn", 257, 512, RouteFeatures(features(257, 256))).action,
            "m32_mixed",
        )
        self.assertEqual(
            policy.choose("gdn", 512, 512, RouteFeatures(features(512, 32))).action,
            "m16_mixed",
        )
        self.assertEqual(
            policy.choose("gdn", 512, 512, RouteFeatures(features(512, 256))).action,
            "m32_mixed",
        )

    def test_interpolation_cache_and_shape_prior(self):
        policy = self.policy()
        route = RouteFeatures(features(384, 128))
        first = policy.choose("gdn", 384, 512, route)
        self.assertIs(first, policy.choose("gdn", 384, 512, route))
        self.assertEqual(len(first.predicted_us), 6)
        self.assertEqual(policy.choose("gdn", 384, 512).reason, "shape_prior")
        self.assertEqual(policy.choose("unknown", 384, 512).action, "legacy_separate")

    def test_hysteresis_keeps_nearly_equal_graph(self):
        data = [sample(512, 128, (100, 100, 100, 100, 100, 99))]
        policy = Policy(data, context={})
        choice = policy.choose(
            "gdn", 512, 512, RouteFeatures(features(512, 128)), current="m16_mixed"
        )
        self.assertEqual(choice.action, "m16_mixed")
        self.assertEqual(choice.reason, "switch_margin")

    def test_device_source_schema_and_cost_gate(self):
        policy = self.policy()
        encoded = policy.to_dict()
        restored = Policy.from_dict(encoded, expected_context={"device": "test"})
        self.assertEqual(restored.to_dict(), encoded)
        with self.assertRaises(ValueError):
            Policy.from_dict(encoded, expected_context={"device": "A100"})
        for bad in (0, -1, math.nan, math.inf):
            row = sample(512, 128, (bad,) * 6)
            with self.assertRaises(ValueError):
                Policy([row], context={})
        row = sample(512, 128, (1,) * 6)
        row["correctness_passed"] = False
        with self.assertRaises(ValueError):
            Policy([row], context={})

    def test_invalid_or_foreign_route_summaries_are_rejected(self):
        for bad in ((1,), (math.nan,) * 17, (0,) * 17):
            with self.assertRaises(ValueError):
                RouteFeatures(bad)
        invalid = list(features(512, 128))
        invalid[1] = 1
        with self.assertRaises(ValueError):
            RouteFeatures(tuple(invalid))
        with self.assertRaises(ValueError):
            self.policy().choose("gdn", 257, 512, RouteFeatures(features(512, 128)))
        with self.assertRaises(ValueError):
            self.policy().choose("gdn", 513, 512)


if __name__ == "__main__":
    unittest.main()
