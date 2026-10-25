"""Throughput model for single-rank full-device temporal PDMux."""

from __future__ import annotations

import math


class TemporalThroughputPolicy:
    """Choose P service by predicted useful-token completion throughput.

    The horizon contains current D output AND the output P will admit. P
    residence and intervening D rounds both cost time. Rewarding a decode-only
    window would instead encourage starving P. Measurements use completed CPU
    output, not speculative proposals. Four layers ensure progress each turn;
    a 32-layer / estimated 64 ms P-service exploration cap limits interruptions.
    This is not a latency SLA: first-call setup and the unchanged final draft-KV
    tail can exceed it.
    """

    def __init__(self, base_layers: int = 8):
        self.base_layers = base_layers
        self.decode_ms = 14.0
        self.decode_ms_by_bs: dict[int, float] = {}
        self.tokens_by_bs: dict[int, float] = {}
        self.tokens_per_req = 2.5
        self.layer_ms: dict[int, float] = {}
        self.useful_tokens = 0
        self.last_scores: dict[int, float] = {}

    @staticmethod
    def token_bucket(tokens: int) -> int:
        return 0 if tokens <= 256 else 1 if tokens <= 4096 else 2

    def observe_decode(self, cpu_ms: float, gpu_ms: float, batch_size: int) -> None:
        cost = max(cpu_ms, gpu_ms)
        if batch_size > 0 and math.isfinite(cost) and cost > 0:
            cost = min(cost, 128.0)
            self.decode_ms = 0.8 * self.decode_ms + 0.2 * cost
            previous = self.decode_ms_by_bs.get(batch_size, cost)
            self.decode_ms_by_bs[batch_size] = 0.8 * previous + 0.2 * cost

    def observe_output(self, batch_size: int, useful_tokens: int) -> None:
        if batch_size <= 0 or useful_tokens < 0:
            return
        self.useful_tokens += useful_tokens
        previous = self.tokens_by_bs.get(batch_size, float(useful_tokens))
        self.tokens_by_bs[batch_size] = 0.8 * previous + 0.2 * useful_tokens
        self.tokens_per_req = (
            0.8 * self.tokens_per_req + 0.2 * useful_tokens / batch_size
        )

    def observe_prefill(
        self,
        *,
        tokens: int,
        layers: int,
        start: int,
        final: bool,
        cpu_ms: float,
        gpu_ms: float,
    ) -> None:
        if start == 0 or final or layers <= 0:
            return
        cost = max(cpu_ms, gpu_ms) / layers
        if math.isfinite(cost) and cost > 0:
            bucket = self.token_bucket(tokens)
            previous = self.layer_ms.get(bucket, cost)
            self.layer_ms[bucket] = 0.5 * previous + 0.5 * cost

    def decode_cost(self, batch_size: int) -> float:
        if batch_size in self.decode_ms_by_bs:
            return self.decode_ms_by_bs[batch_size]
        if self.decode_ms_by_bs:
            nearest = min(self.decode_ms_by_bs, key=lambda n: abs(n - batch_size))
            return self.decode_ms_by_bs[nearest] * (0.8 + 0.2 * batch_size / nearest)
        return self.decode_ms

    def round_tokens(self, batch_size: int) -> float:
        return max(
            0.1, self.tokens_by_bs.get(batch_size, batch_size * self.tokens_per_req)
        )

    def decode_first(
        self,
        *,
        tokens: int,
        quantum: int,
        current_remaining: list[int],
        new_remaining: list[int],
        since_decode_ms: float,
    ) -> bool:
        current = [n for n in current_remaining if n > 0]
        new = [n for n in new_remaining if n > 0]
        if not current:
            return False
        if not new:
            return True
        bs = len(current)
        expanded = bs + len(new)
        current_rate = self.round_tokens(bs) / self.decode_cost(bs)
        expanded_rate = self.round_tokens(expanded) / self.decode_cost(expanded)
        if expanded_rate <= current_rate * 1.02:
            return True
        cost = self.layer_ms.get(self.token_bucket(tokens), 3.0)
        # Bound deferral while filling D, without preparing/allocating a D
        # batch merely to decide later that it should not run.
        return since_decode_ms + quantum * cost > 64.0

    def choose(
        self,
        *,
        remaining: int,
        tokens: int,
        current_remaining: list[int],
        new_remaining: list[int],
    ) -> int:
        self.last_scores = {}
        if remaining <= 0:
            return 0
        current = [n for n in current_remaining if n > 0]
        new = [n for n in new_remaining if n > 0]
        if not current:
            return remaining
        layer_cost = self.layer_ms.get(self.token_bucket(tokens))
        if layer_cost is None:
            return min(remaining, self.base_layers)
        cap = min(32, max(4, int(64.0 / layer_cost)))
        candidates = {min(remaining, q) for q in (4, 8, 16, 32) if q <= cap}
        if remaining <= cap:
            candidates.add(remaining)
        bs = len(current)
        d_ms = self.decode_cost(bs)
        tokens_per_req = self.round_tokens(bs) / bs
        total_tokens = sum(current) + sum(new)
        for quantum in sorted(candidates):
            # This iteration's D is already submitted, common to all choices.
            turns = math.ceil(remaining / quantum) - 1
            turns = min(turns, math.ceil(max(current) / tokens_per_req))
            residual = [max(0.0, n - turns * tokens_per_req) for n in current]
            after_bs = sum(n > 0 for n in residual) + len(new)
            after_tokens = sum(residual) + sum(new)
            after_ms = 0.0
            if after_bs and after_tokens:
                rate = self.round_tokens(after_bs) / self.decode_cost(after_bs)
                after_ms = after_tokens / rate
            completion_ms = remaining * layer_cost + turns * d_ms + after_ms
            self.last_scores[quantum] = total_tokens / max(completion_ms, 0.001)
        best = max(self.last_scores.values())
        # Prefer fewer scheduling boundaries when predictions are within 2%.
        # This is controller hysteresis, not a statistical confidence bound.
        return max(q for q, score in self.last_scores.items() if score >= best * 0.98)
