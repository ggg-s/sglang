"""Estimates for the experimental split prefill feedback loop."""

import math
import os
from dataclasses import dataclass, field

MIN_SUFFIX_TOKENS = 2048
RATE_LIMIT = 10_000_000
SUFFIX_CAP_GRANULARITY = 512


def read_budget(name, default, *, allow_zero=False):
    message = (
        f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} number"
    )
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(message)
    return value


def ema(previous, sample):
    return sample if previous is None else 0.7 * previous + 0.3 * sample


@dataclass
class DecodeBudget:
    target: float
    stall_ms: float
    patience_ms: float
    pinned_chunk: bool
    max_cap: int
    suffix_rate: float | None = None
    decode_ms: float | None = None
    ledger: object = field(default=None, compare=False, repr=False)
    worker: str = field(default="0", compare=False, repr=False)

    def observe(self, tokens, elapsed_ms, *, suffix, rank):
        if suffix:
            self.suffix_rate = ema(self.suffix_rate, tokens * 1000 / elapsed_ms)
        else:
            self.decode_ms = ema(self.decode_ms, elapsed_ms)

    def offer(self, chunk, max_prefill_tokens, length):
        from .split_budget_ledger import ALPHA_SCALE

        rate = min(RATE_LIMIT, int(self.suffix_rate or 0))
        if rate and not self.pinned_chunk:
            chunk = int(
                min(
                    max_prefill_tokens,
                    max(MIN_SUFFIX_TOKENS // 2, self.stall_ms / 1000 * rate),
                )
            )
        step_us = math.ceil(min(2**31 - 1, (self.decode_ms or 0) * 1000))
        alpha = self.ledger.snapshot()["alpha"] if self.ledger is not None else 1.0
        return (
            min(self.max_cap, length // 2),
            chunk,
            rate,
            -step_us,
            int(alpha * ALPHA_SCALE),
        )

    def patience_steps(self, step_us, fallback_ticks):
        if step_us <= 0:
            return fallback_ticks
        # Scheduler ticks approximate decode steps in this path.
        step_ms = step_us / 1000
        patience = int(min(2**31 - 1, self.patience_ms / step_ms))
        return max(1, patience) if self.patience_ms else 0
