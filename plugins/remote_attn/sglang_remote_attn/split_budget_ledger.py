"""D-instance time credits shared by attention groups through the native Store.

Store CAS is independent of model collectives, so an idle DP rank need not
participate. A reservation belongs to one native request attempt, not a P worker.
"""

import json
import math
import threading
import time

from .split_decode_budget import ema

ALPHA_SCALE = 1_000_000


class LocalStore:
    """The single-process equivalent of the distributed Store CAS interface."""

    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def compare_set(self, key, expected, desired):
        with self.lock:
            current = self.values.get(key, b"")
            if current == expected:
                current = self.values[key] = desired
            return current

    def get(self, key):
        with self.lock:
            return self.values[key]


def native_store(scheduler):
    import torch.distributed as dist

    if not dist.is_initialized():
        return LocalStore()
    # Attention-DP shards share the model TP group. Other model replicas in
    # the same launch have a different group Store and independent credits.
    from torch.distributed.distributed_c10d import _get_process_group_store

    store = _get_process_group_store(scheduler.tp_cpu_group)
    return dist.PrefixStore("remote-attn-split-budget/", store)


class BudgetLedger:
    def __init__(self, store, *, g, stall_ms, clock=time.time):
        self.store, self.g, self.stall_ms, self.clock = store, g, stall_ms, clock
        self.target = g / (1 + g)
        initial = dict(
            config=[g, stall_ms],
            alpha=1.0,
            delta=None,
            credit=stall_ms,
            capacity=stall_ms,
            handoff_ms=None,
            refill_rate=None,
            refill_ms=0.0,
            refill_start=clock(),
            clock=clock(),
            decode_end=None,
            decode_step=-1,
            decode_step_ms=0.0,
            reservations={},
            workers={},
            suffix_ms=0.0,
            decode_ms=0.0,
            controlled=0,
            windows=0,
        )
        self.store.compare_set("ledger", b"", self._encode(initial))
        if json.loads(self.store.get("ledger"))["config"] != [g, stall_ms]:
            raise RuntimeError("D attention groups disagree on split budget g/stall_ms")

    @staticmethod
    def _encode(state):
        return json.dumps(state, allow_nan=False, separators=(",", ":")).encode()

    def _update(self, operation):
        previous = self.store.get("ledger")
        while True:
            state = json.loads(previous)
            result = operation(state)
            encoded = self._encode(state)
            current = self.store.compare_set("ledger", previous, encoded)
            if current == encoded:
                return result
            previous = current

    @staticmethod
    def _reserved(state):
        return sum(r["ms"] for r in state["reservations"].values())

    def _limit(self, state):
        ceiling = max(0.0, state["capacity"] - self._reserved(state))
        state["credit"] = min(state["credit"], ceiling)

    def _advance(self, state, now):
        elapsed = max(0.0, (now - state["clock"]) * 1000)
        idle = not any(w["decode"] or w["busy"] for w in state["workers"].values())
        if idle:
            state["credit"] += elapsed
            state["refill_ms"] += elapsed
        state["clock"] = max(now, state["clock"])
        self._capacity(state, now)
        self._limit(state)

    def _capacity(self, state, now):
        interval = max(0.0, (now - state["refill_start"]) * 1000)
        if interval >= self.stall_ms:
            state["refill_rate"] = ema(
                state["refill_rate"], min(1.0, state["refill_ms"] / interval)
            )
            state.update(refill_ms=0.0, refill_start=now)
        probe = max(
            (w.get("probe_ms", 0) for w in state["workers"].values()), default=0
        )
        state["capacity"] = max(self.stall_ms, probe) + (
            (state["refill_rate"] or 0) * (state["handoff_ms"] or 0)
        )

    def snapshot(self):
        state = json.loads(self.store.get("ledger"))
        return dict(
            alpha=state["alpha"],
            delta=state["delta"],
            target=self.target,
            credit_ms=state["credit"],
            reserved_ms=self._reserved(state),
            capacity_ms=state["capacity"],
            outstanding=len(state["reservations"]),
            windows=state["windows"],
        )

    def demand(self, worker, decode):
        now = self.clock()

        def update(state):
            self._advance(state, now)
            current = state["workers"].setdefault(
                worker, dict(decode=False, busy=False)
            )
            current["decode"] = bool(decode)

        self._update(update)

    def reserve(self, key, *, worker, cap, minimum, rate, limit, page=1, length=0):
        now = self.clock()

        def update(state):
            current = state["workers"].setdefault(
                worker, dict(decode=False, busy=False)
            )
            legal_minimum = minimum + (length - minimum) % page
            current["probe_ms"] = legal_minimum * 1000 / rate
            self._capacity(state, now)
            self._advance(state, now)
            old = state["reservations"].get(key)
            if old is not None:
                return (0 if old["closing"] else old["offered"]), old["alpha"]
            if not self.g or len(state["reservations"]) >= limit:
                return 0, int(state["alpha"] * ALPHA_SCALE)
            tokens = min(cap, math.floor(max(0, state["credit"]) * rate / 1000))
            tokens -= (tokens - length) % page
            if tokens < minimum:
                return 0, int(state["alpha"] * ALPHA_SCALE)
            cost = tokens * 1000 / rate
            state["credit"] -= cost
            state["reservations"][key] = dict(
                worker=worker,
                tokens=tokens,
                offered=tokens,
                alpha=int(state["alpha"] * ALPHA_SCALE),
                rate=rate,
                ms=cost,
                created=now,
                active=0,
                active_ms=0.0,
                closing=False,
                controlled=False,
            )
            return tokens, int(state["alpha"] * ALPHA_SCALE)

        return self._update(update)

    def commit(self, key, tokens, *, controlled):
        now = self.clock()

        def update(state):
            self._advance(state, now)
            r = state["reservations"].get(key)
            if r is None or r["closing"]:
                return False
            if "committed" in r:
                return r["committed"] == tokens
            if tokens > r["tokens"] or tokens < 0:
                return False
            cost = tokens * 1000 / r["rate"]
            state["credit"] += r["ms"] - cost
            r.update(tokens=tokens, ms=cost, committed=tokens, controlled=controlled)
            state["controlled"] += int(controlled)
            state["handoff_ms"] = ema(
                state["handoff_ms"], max(0, (now - r["created"]) * 1000)
            )
            self._capacity(state, now)
            self._limit(state)
            return True

        return self._update(update)

    def begin(self, worker, key=None, tokens=0):
        now = self.clock()

        def update(state):
            self._advance(state, now)
            competing = any(w["decode"] for w in state["workers"].values())
            w = state["workers"].setdefault(worker, dict(decode=False, busy=False))
            w["busy"] += 1
            if key is not None:
                r = state["reservations"].get(key)
                if r is None or r["active"] or r["closing"] or tokens > r["tokens"]:
                    raise RuntimeError("suffix forward has no live budget reservation")
                r.update(active=tokens, active_ms=tokens * 1000 / r["rate"])
            return competing

        return self._update(update)

    def complete(self, worker, elapsed_ms, *, key=None, competing=False, step=None):
        now = self.clock()

        def update(state):
            self._advance(state, now)
            previous_windows = state["windows"]
            state["workers"][worker]["busy"] -= 1
            if key is None:
                # Native forward_iter includes idle passes, so DP leaders
                # identify the same model forward without another collective.
                if step is None:
                    last = state["decode_end"]
                    duration = (
                        elapsed_ms
                        if last is None
                        else min(elapsed_ms, max(0, (now - last) * 1000))
                    )
                elif step < state["decode_step"]:
                    duration = 0.0
                elif step == state["decode_step"]:
                    duration = max(0.0, elapsed_ms - state["decode_step_ms"])
                    state["decode_step_ms"] = max(elapsed_ms, state["decode_step_ms"])
                else:
                    duration = elapsed_ms
                    state.update(decode_step=step, decode_step_ms=elapsed_ms)
                state["decode_end"] = now
                state["credit"] += self.g * duration
                state["refill_ms"] += self.g * duration
                state["decode_ms"] += duration
            else:
                r = state["reservations"].get(key)
                if r is None or not r["active"]:
                    raise RuntimeError("suffix result has no active budget reservation")
                estimated = r["active_ms"]
                completed_tokens = r["active"]
                # Cancellation releases future work immediately, but keeps this
                # in-flight estimate until its GPU result has been consumed.
                r["tokens"] -= r["active"]
                r["ms"] = max(0.0, r["ms"] - estimated)
                r.update(active=0, active_ms=0.0)
                state["credit"] += estimated - elapsed_ms
                # Reprice work still queued for this request with the latest
                # sample, so slow chunks immediately increase outstanding debt.
                remaining_estimate = r["ms"]
                r["rate"] = ema(r["rate"], completed_tokens * 1000 / elapsed_ms)
                r["ms"] = r["tokens"] * 1000 / r["rate"]
                state["credit"] += remaining_estimate - r["ms"]
                if competing:
                    state["suffix_ms"] += elapsed_ms
                    state["controlled"] += int(r["controlled"])
                if r["closing"] or not r["tokens"]:
                    del state["reservations"][key]
            self._capacity(state, now)
            total = state["suffix_ms"] + state["decode_ms"]
            # Occupancy needs a time window, independent of rate EMA samples.
            # Decode-only windows can recover alpha after it suppresses splits.
            if total >= self.stall_ms:
                delta = state["suffix_ms"] / total
                state["delta"] = delta
                if self.target and state["controlled"]:
                    factor = min(
                        1.10, max(0.90, 1 + 0.25 * (self.target - delta) / self.target)
                    )
                    state["alpha"] = min(1.0, state["alpha"] * factor)
                state.update(suffix_ms=0.0, decode_ms=0.0, controlled=0)
                state["windows"] += 1
            self._limit(state)
            return state["windows"] != previous_windows

        return self._update(update)

    def release(self, key):
        now = self.clock()

        def update(state):
            self._advance(state, now)
            r = state["reservations"].get(key)
            if r is None or r["closing"]:
                return
            state["credit"] += max(0.0, r["ms"] - r["active_ms"])
            if r["active"]:
                r.update(tokens=r["active"], ms=r["active_ms"], closing=True)
            else:
                del state["reservations"][key]
            self._limit(state)

        self._update(update)
