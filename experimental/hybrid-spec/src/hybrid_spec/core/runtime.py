from __future__ import annotations

import heapq

import torch

from hybrid_spec.core.contracts import HistoryDelta, RequestKey, RequestSnapshot
from hybrid_spec.core.request_state import RequestState
from hybrid_spec.selection.policy import select_requests


class HybridRuntime:
    """One worker, one stream; committed history only, no shared-request cache."""

    def __init__(self, config, backend, vocab_size):
        self.config = config
        self.backend = backend
        self.vocab_size = vocab_size
        self.states = {}
        self.free_slots = list(range(config.max_requests))
        self.generation = 0

    def close_request(self, rid):
        state = self.states.pop(rid, None)
        if state is not None:
            # reset/append/query share the worker forward stream. Reuse is
            # ordered after all previous consumers, including graph replay.
            heapq.heappush(self.free_slots, state.slot)

    def reset(self):
        self.states.clear()
        self.free_slots = list(range(self.config.max_requests))
        # Never reset generation: old request keys remain invalid.

    def reconcile(self, rid, history):
        history = tuple(history)
        if any(
            type(token) is not int or not 0 <= token < self.vocab_size
            for token in history
        ):
            raise ValueError("history must use the target vocabulary")
        state = self.states.get(rid)
        if state is not None and history[: len(state.history)] != state.history:
            self.close_request(rid)
            state = None
        if len(history) > self.config.max_history_tokens:
            self.close_request(rid)
            return None  # Explicit whole-request neural fallback.
        if state is None:
            if not self.free_slots:
                return None
            self.generation += 1
            slot = heapq.heappop(self.free_slots)
            state = RequestState(RequestKey(rid, self.generation), slot)
            self.backend.reset_slot(slot)
            self.states[rid] = state
        delta = HistoryDelta(
            state.key, len(state.history), history[len(state.history) :]
        )
        if delta.accept_tokens:
            self.backend.append(state.slot, delta)
            state.commit(delta)
        return state

    def prepare(self, requests):
        """requests = [(rid, confirmed target-token history, remaining budget)]."""
        keys, slots, versions, remaining = [], [], [], []
        for rid, history, budget in requests:
            state = self.reconcile(rid, history)
            keys.append(state.key if state else RequestKey(rid, -1))
            slots.append(state.slot if state else -1)
            versions.append(len(history))
            remaining.append(budget)
        snapshot = RequestSnapshot(
            tuple(keys), tuple(slots), tuple(versions), tuple(remaining)
        )
        candidates = self.backend.query(snapshot, self.config.num_drafts)
        view = select_requests(
            candidates,
            torch.tensor(versions, device=self.backend.device, dtype=torch.int64),
            torch.tensor(remaining, device=self.backend.device, dtype=torch.int64),
            self.config.min_match_len,
            self.vocab_size,
            self.config.partial_drafts,
        )
        view.num_neural_gate_tokens = self.config.num_neural_gate_tokens
        return view
