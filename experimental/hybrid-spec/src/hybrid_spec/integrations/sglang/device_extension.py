"""GPU feedback and graph-resident SA updates; CPU only manages request leases."""

from __future__ import annotations

import heapq
import threading
import weakref
from contextlib import contextmanager

import torch
from hybrid_spec.core.contracts import DraftOverrideView, HistoryDelta, RequestKey
from hybrid_spec.observability.metrics import Metrics


class DeviceOverride(DraftOverrideView):
    def __init__(self, backend, slots, generations):
        self.backend = backend
        self.slots = slots
        self.generations = generations
        super().__init__(
            torch.zeros(
                (len(slots), backend.config.num_drafts),
                dtype=torch.int64,
                device=backend.device,
            ),
            torch.zeros(len(slots), dtype=torch.bool, device=backend.device),
        )

    def prepare(self, seq_lens):
        self.backend.consume_and_query(
            self.slots, self.generations, seq_lens, self.tokens, self.use_retrieval
        )


class DeviceGraphBuffers:
    def __init__(self, backend, max_bs):
        self.view = DeviceOverride(
            backend,
            torch.full((max_bs,), -1, dtype=torch.int64, device=backend.device),
            torch.zeros(max_bs, dtype=torch.int64, device=backend.device),
        )

    def bind_capture_view(self, batch_size):
        # Views share storage with staging; the query writes outputs in graph.
        view = object.__new__(DeviceOverride)
        view.backend = self.view.backend
        for name in ("slots", "generations", "tokens", "use_retrieval"):
            setattr(view, name, getattr(self.view, name)[:batch_size])
        return view

    def stage(self, view, raw_bs, padded_bs):
        if not 0 <= raw_bs <= padded_bs <= len(self.view.slots):
            raise ValueError("invalid graph batch sizes")
        if view.slots.shape != (raw_bs,) or view.generations.shape != (raw_bs,):
            raise ValueError("override does not match the captured batch")
        self.view.slots.fill_(-1)
        self.view.slots[:raw_bs].copy_(view.slots)
        self.view.generations[:raw_bs].copy_(view.generations)
        # Feedback consumes the actual captured query's selection mask.
        view.use_retrieval = self.view.use_retrieval[:raw_bs]


class DeviceDraftExtension:
    def __init__(self, config, backend, vocab_size, tp_sync=None):
        if tp_sync is not None:
            raise ValueError("device history currently requires TP=1")
        self.config = config
        self.backend = backend
        self.vocab_size = vocab_size
        self.metrics = Metrics()
        self.states = {}
        self.free_slots = list(range(config.max_requests))
        self.generation = 0
        self.lock = threading.RLock()
        self.pending_view = None
        self.last_binding = None

    def create_graph_buffers(self, max_bs):
        return DeviceGraphBuffers(self.backend, max_bs)

    def close_request(self, rid):
        # Finish callbacks may run on the scheduler thread. Never touch GPU
        # state there: the next forward initializes a new generation in stream.
        with self.lock:
            state = self.states.pop(rid, None)
            if state is not None:
                heapq.heappush(self.free_slots, state[1])

    def reset(self):
        with self.lock:
            self.states.clear()
            self.free_slots = list(range(self.config.max_requests))
            # Retain the generation counter across flush, fencing old feedback.

    def on_prefill(self, batch, result):
        if batch.forward_mode.is_idle():
            return
        slots, generations = [], []
        with self.lock:
            for row, req in enumerate(batch.reqs):
                self.close_request(req.rid)
                length = int(batch.seq_lens_cpu[row])
                history = tuple(req.get_fill_ids()[:length])
                if (
                    not self.free_slots
                    or len(history) != length
                    or length >= self.config.max_history_tokens
                ):
                    slots.append(-1)
                    generations.append(-1)
                    continue
                if any(not 0 <= token < self.vocab_size for token in history):
                    raise ValueError("history must use the target vocabulary")
                slot = heapq.heappop(self.free_slots)
                self.generation += 1
                generation = self.generation
                self.backend.initialize_device_slot(
                    slot,
                    generation,
                    len(req.origin_input_ids) + req.sampling_params.max_new_tokens,
                )
                self.backend.append(
                    slot, HistoryDelta(RequestKey(req.rid, generation), 0, history)
                )
                self.states[req.rid] = (weakref.ref(req), slot, generation)
                slots.append(slot)
                generations.append(generation)
        view = self._view(slots, generations)
        bs = len(slots)
        # Prefill sampled one GPU bonus token. It is appended by the first
        # draft graph, along with all later verify feedback, exactly once.
        self.backend.stage_feedback(
            view.slots,
            view.generations,
            result.next_token_ids.to(torch.int32),
            torch.arange(bs, dtype=torch.int32, device=self.backend.device)[:, None],
            torch.ones(bs, dtype=torch.int32, device=self.backend.device),
            batch.seq_lens,
            False,
            self.vocab_size,
        )

    def _view(self, slots, generations):
        return DeviceOverride(
            self.backend,
            torch.tensor(slots, dtype=torch.int64, device=self.backend.device),
            torch.tensor(generations, dtype=torch.int64, device=self.backend.device),
        )

    @contextmanager
    def bind(self, batch, draft_input):
        if batch.is_extend_in_batch or batch.has_grammar:
            raise ValueError("hybrid-spec does not support mixed prefill or grammar")
        if not batch.sampling_info.is_all_greedy:
            raise ValueError("hybrid-spec supports greedy requests only")
        slots, generations = [], []
        with self.lock:
            for req in batch.reqs:
                if req.multimodal_inputs is not None:
                    raise ValueError("hybrid-spec does not index multimodal inputs")
                state = self.states.get(req.rid)
                if state is None or state[0]() is not req:
                    slots.append(-1)
                    generations.append(-1)
                else:
                    slots.append(state[1])
                    generations.append(state[2])
        binding_key = (tuple(slots), tuple(generations))
        if self.last_binding is None or self.last_binding[0] != binding_key:
            self.last_binding = (binding_key, self._view(slots, generations))
        view = self.last_binding[1]
        self.pending_view = view
        draft_input.draft_override = view
        try:
            yield view
        finally:
            draft_input.draft_override = None

    def on_device_verify(self, result, batch):
        view, self.pending_view = self.pending_view, None
        if view is None or batch.forward_mode.is_idle():
            return
        self.metrics.record(view.use_retrieval, result.accept_lens)
        self.backend.stage_feedback(
            view.slots,
            view.generations,
            result.next_token_ids,
            result.accept_index,
            result.accept_lens,
            batch.seq_lens,
            True,
            self.vocab_size,
        )
