"""Estimates for the experimental split prefill feedback loop."""

import logging
import math
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_SUFFIX_TOKENS = 2048
RATE_LIMIT = 10_000_000
SUFFIX_CAP_GRANULARITY = 512
MIN_CAP_SHARE = 0.05
MAX_CAP_SHARE = 0.50


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
    share: float = 0.25
    min_cap: int = MIN_SUFFIX_TOKENS
    coverage: float = 1.0
    suffix_rate: float | None = None
    decode_ms: float | None = None
    forward_ms: float = 0.0
    suffix_ms: float = 0.0
    suffix_samples: int = 0
    windows: int = 0

    def startup_signature(self):
        # Only configuration participates in the TP/CP startup agreement.
        return (
            self.target,
            self.stall_ms,
            self.patience_ms,
            self.pinned_chunk,
            self.max_cap,
            self.min_cap,
        )

    def observe(self, tokens, elapsed_ms, *, suffix, rank):
        if suffix:
            self.suffix_rate = ema(self.suffix_rate, tokens * 1000 / elapsed_ms)
            self.suffix_ms += elapsed_ms
            self.suffix_samples += 1
        else:
            self.decode_ms = ema(self.decode_ms, elapsed_ms)
        self.forward_ms += elapsed_ms
        if self.forward_ms < 1500:
            return
        if not self.suffix_samples:
            self.forward_ms = 0.0
            return
        if self.suffix_samples < 8:
            return

        delta = self.suffix_ms / self.forward_ms
        self.forward_ms = self.suffix_ms = 0.0
        self.suffix_samples = 0
        if self.target == 0:
            return
        factor = min(1.10, max(0.90, 1 + 0.25 * (self.target - delta) / self.target))
        if self.share <= MIN_CAP_SHARE and (self.coverage < 1 or delta > self.target):
            # Delta already reflects the current admission rate. Scale that
            # rate, retaining a probe share when the target is nonzero.
            self.coverage = min(1.0, max(0.05, self.coverage * factor))
        else:
            self.share = min(MAX_CAP_SHARE, max(MIN_CAP_SHARE, self.share * factor))
        self.windows += 1
        if rank == 0 and self.windows % 3 == 0:
            logger.info(
                "[handoff-split-budget] delta=%.3f target=%.3f share=%.3f coverage=%.2f",
                delta,
                self.target,
                self.share,
                self.coverage,
            )

    def offer(self, chunk, max_prefill_tokens, length):
        cap = int(self.share * length)
        cap -= cap % SUFFIX_CAP_GRANULARITY
        cap = min(self.max_cap, max(self.min_cap, cap))
        rate = min(RATE_LIMIT, int(self.suffix_rate or 0))
        if rate and not self.pinned_chunk:
            chunk = int(
                min(
                    max_prefill_tokens,
                    max(MIN_SUFFIX_TOKENS // 2, self.stall_ms / 1000 * rate),
                )
            )
        # The TP MIN vote encodes the decode step as a negated duration.
        step_us = math.ceil(min(2**31 - 1, (self.decode_ms or 0) * 1000))
        coverage = int(self.coverage * 1000) if self.target else 0
        return cap, chunk, rate, -step_us, coverage

    def patience_steps(self, step_us, fallback_ticks):
        if step_us <= 0:
            return fallback_ticks
        # Scheduler ticks approximate decode steps in this path.
        step_ms = step_us / 1000
        patience = int(min(2**31 - 1, self.patience_ms / step_ms))
        return max(1, patience) if self.patience_ms else 0
