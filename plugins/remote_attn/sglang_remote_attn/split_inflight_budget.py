"""Bound outstanding suffix GPU time while leaving share feedback in charge."""

from dataclasses import dataclass

from .split_decode_budget import MIN_SUFFIX_TOKENS

MICROSECONDS_PER_SECOND = 1_000_000


class InflightSuffixBudget:
    def __init__(self, limit_ms, *, store=None, rank=0):
        self.limit_us = round(limit_ms * 1000)
        self.store = store
        self.rank = rank
        self.used_us = 0
        if store is not None:
            config = str(self.limit_us).encode()
            if store.compare_set("limit_us", b"", config) != config:
                raise RuntimeError("D attention groups disagree on split inflight limit")
            store.compare_set("used_us", b"", b"0:0:0")

    def _update(self, change):
        if self.store is None:
            delta, result = change(self.used_us)
            self.used_us += delta
            return result

        previous = self.store.get("used_us")
        while True:
            used, revision, _ = map(int, previous.split(b":"))
            delta, result = change(used)
            if not delta:
                return result
            # Include the writer so concurrent equal debits cannot both treat
            # compare_set returning the same value as a successful reservation.
            desired = f"{used + delta}:{revision + 1}:{self.rank}".encode()
            current = self.store.compare_set("used_us", previous, desired)
            if current == desired:
                return result
            previous = current

    def reserve(self, cap, *, length, page, rate):
        def charge(used):
            tokens = cap
            if used:
                available = max(0, self.limit_us - used)
                tokens = min(tokens, available * rate // MICROSECONDS_PER_SECOND)
            # The cut, not the suffix length, must be page aligned.
            tokens -= (tokens - length) % page
            if tokens < MIN_SUFFIX_TOKENS:
                return 0, None
            return _cost_us(tokens, rate), SuffixReservation(self, rate, tokens)

        return self._update(charge)


def _cost_us(tokens, rate):
    return (tokens * MICROSECONDS_PER_SECOND + rate - 1) // rate


@dataclass
class SuffixReservation:
    budget: InflightSuffixBudget
    rate: int
    tokens: int

    def shrink(self, tokens):
        tokens = min(self.tokens, tokens)
        refund = _cost_us(self.tokens, self.rate) - _cost_us(tokens, self.rate)
        if refund:
            self.budget._update(lambda _: (-refund, None))
        self.tokens = tokens
