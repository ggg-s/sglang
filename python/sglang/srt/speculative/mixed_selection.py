"""CPU-only, opt-in cost table for offline repaired graph selection.

No serving imports or global settings. Calibrate on the actual device first;
the table is a local operator model, not an A100/full-model performance claim.
"""

from __future__ import annotations

import bisect
import copy
import math
from dataclasses import dataclass

ACTIONS = tuple(
    f"{variant}_{layout}"
    for variant in ("legacy", "m16", "m32")
    for layout in ("separate", "mixed")
)
FEATURE_COUNT = 17


@dataclass(frozen=True)
class RouteFeatures:
    values: tuple[float, ...]

    def __post_init__(self):
        if len(self.values) != FEATURE_COUNT or not all(
            math.isfinite(v) and v >= 0 for v in self.values
        ):
            raise ValueError("Invalid route feature vector")
        if self.values[0] <= 0 or self.values[1] != 0 or not 1 <= self.values[3] <= 256:
            raise ValueError("Missing or invalid expert routes")

    @property
    def effective_experts(self):
        return self.values[3]

    @property
    def bucket(self):
        # Quantization only affects the cost lookup/cache, never model routing.
        return min(256, max(8, int(self.effective_experts / 8 + 0.5) * 8))


@dataclass(frozen=True)
class Selection:
    action: str
    predicted_us: tuple[float, ...]
    reason: str


def _bracket(values, target):
    upper = bisect.bisect_left(values, target)
    if upper == 0:
        return 0, 0, 0.0
    if upper == len(values):
        return upper - 1, upper - 1, 0.0
    lower = upper - 1
    return lower, upper, (target - values[lower]) / (values[upper] - values[lower])


def _interpolate(left, right, alpha):
    return tuple(a + (b - a) * alpha for a, b in zip(left, right))


class OfflineMixedSelector:
    """Interpolate measured costs in P/padding and routing concentration.

    Capacity is an exact graph bucket. Within it, interpolate P and observed
    effective experts. Use a shape prior while async route feedback is absent.
    Current actions within 2% (or 3 us) of best keep their graph to avoid noise
    driven switching. Evaluation never updates these frozen calibration costs.
    """

    def __init__(self, samples, *, context, switch_fraction=0.02, switch_us=3.0):
        self.context = copy.deepcopy(context)
        self.switch_fraction, self.switch_us = switch_fraction, switch_us
        if (
            not 0 <= switch_fraction < 1
            or not math.isfinite(switch_us)
            or switch_us < 0
        ):
            raise ValueError("Invalid switch margin")
        self.samples = tuple(copy.deepcopy(samples))
        grouped = {}
        for sample in self.samples:
            p, capacity = sample["P"], sample["capacity"]
            costs = tuple(float(sample["costs_us"][a]) for a in ACTIONS)
            route = RouteFeatures(tuple(sample["features"]))
            if (
                not sample.get("correctness_passed")
                or not 0 < p <= capacity
                or not all(math.isfinite(v) and v > 0 for v in costs)
            ):
                raise ValueError("Calibration requires validated positive costs")
            if route.values[0] != (p + 24) * 8:
                raise ValueError("Calibration does not match P + D3x8")
            grouped.setdefault((sample["operator"], capacity), {}).setdefault(
                p, []
            ).append((route.effective_experts, costs))
        self.curves = {
            key: (
                tuple(sorted(rows)),
                tuple(tuple(sorted(rows[p])) for p in sorted(rows)),
            )
            for key, rows in grouped.items()
        }
        self.cache = {}

    def _costs(self, operator, p, capacity, effective):
        curve = self.curves.get((operator, capacity))
        if curve is None:
            return None
        positions, route_curves = curve
        lower, upper, alpha = _bracket(positions, p)

        def at(index):
            rows = route_curves[index]
            if effective is None:
                # Median measured routing regime provides a shape-only prior.
                return tuple(
                    sorted(r[1][i] for r in rows)[len(rows) // 2]
                    for i in range(len(ACTIONS))
                )
            lo, hi, mix = _bracket(tuple(r[0] for r in rows), effective)
            return _interpolate(rows[lo][1], rows[hi][1], mix)

        return _interpolate(at(lower), at(upper), alpha)

    def choose(self, operator, p, capacity, route=None, *, current=None):
        if not 0 < p <= capacity:
            raise ValueError("P must fit its captured capacity")
        if route is not None and route.values[0] != (p + 24) * 8:
            raise ValueError("Route summary belongs to another shape")
        if current is not None and current not in ACTIONS:
            raise ValueError("Unknown current action")
        key = (operator, p, capacity, route.bucket if route else None)
        cached = self.cache.get(key)
        if cached is None:
            costs = self._costs(operator, p, capacity, key[-1])
            if costs is None:
                cached = Selection("legacy_separate", (), "uncalibrated")
            else:
                best = min(range(len(ACTIONS)), key=costs.__getitem__)
                cached = Selection(
                    ACTIONS[best], costs, "route_feedback" if route else "shape_prior"
                )
            self.cache[key] = cached
        if current and cached.predicted_us:
            costs = cached.predicted_us
            old = costs[ACTIONS.index(current)]
            best = min(costs)
            if old - best <= max(self.switch_us, old * self.switch_fraction):
                if current == cached.action:
                    return cached
                return Selection(current, costs, "switch_margin")
        return cached

    def to_dict(self):
        return dict(
            schema=1,
            context=copy.deepcopy(self.context),
            samples=copy.deepcopy(list(self.samples)),
            switch_fraction=self.switch_fraction,
            switch_us=self.switch_us,
        )

    @classmethod
    def from_dict(cls, data, *, expected_context):
        if data.get("schema") != 1 or data.get("context") != expected_context:
            raise ValueError("Calibration device/model/source context mismatch")
        return cls(
            data["samples"],
            context=data["context"],
            switch_fraction=data["switch_fraction"],
            switch_us=data["switch_us"],
        )
