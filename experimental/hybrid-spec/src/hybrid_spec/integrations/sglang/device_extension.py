"""GPU feedback and graph-resident SA updates; CPU only manages request leases."""

from __future__ import annotations

import heapq
import threading
import weakref
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from hybrid_spec.core.contracts import DraftOverrideView
from hybrid_spec.observability.metrics import Metrics


class DeviceOverride(DraftOverrideView):
    def __init__(self, backend, slots, generations, tp_sync=None):
        self.backend = backend
        self.slots = slots
        self.generations = generations
        self.tp_sync = tp_sync
        # Only worker-created views provide this immutable host metadata key.
        # Manually constructed views may mutate their tensors between stages.
        self.mapping_key = None
        if backend.config.enable_global_pool:
            self.shared_slots = torch.full(
                (backend.config.num_sa_slots,),
                -1,
                dtype=torch.int64,
                device=backend.device,
            )
            self.shared_generations = torch.zeros_like(self.shared_slots)
            # -1 means a live source. Completed sources carry the CPU-confirmed
            # final length, excluding any verify tail discarded at EOS/stop.
            self.shared_lengths = torch.full_like(self.shared_slots, -1)
        super().__init__(
            torch.zeros(
                (len(slots), backend.config.num_drafts),
                dtype=torch.int64,
                device=backend.device,
            ),
            torch.zeros(len(slots), dtype=torch.bool, device=backend.device),
            backend.config.num_neural_gate_tokens,
        )

    def prepare(self, seq_lens):
        shared = (
            {
                "shared_slots": self.shared_slots,
                "shared_generations": self.shared_generations,
                "shared_lengths": self.shared_lengths,
            }
            if self.backend.config.enable_global_pool
            else {}
        )
        self.backend.consume_and_query(
            self.slots,
            self.generations,
            seq_lens,
            self.tokens,
            self.use_retrieval,
            **shared,
        )
        if self.tp_sync is not None:
            agreed = self.tp_sync.synchronize(self)
            # Preserve graph storage and the mask alias used by verify metrics.
            self.tokens.copy_(agreed.tokens)
            self.use_retrieval.copy_(agreed.use_retrieval)


class DeviceGraphBuffers:
    def __init__(self, backend, max_bs, tp_sync=None):
        self.staged_mapping = None
        self.view = DeviceOverride(
            backend,
            torch.full((max_bs,), -1, dtype=torch.int64, device=backend.device),
            torch.zeros(max_bs, dtype=torch.int64, device=backend.device),
            tp_sync,
        )

    def bind_capture_view(self, batch_size):
        # Views share storage with staging; the query writes outputs in graph.
        view = object.__new__(DeviceOverride)
        view.backend = self.view.backend
        view.tp_sync = self.view.tp_sync
        view.mapping_key = None
        view.num_neural_gate_tokens = self.view.num_neural_gate_tokens
        for name in ("slots", "generations", "tokens", "use_retrieval"):
            setattr(view, name, getattr(self.view, name)[:batch_size])
        if self.view.backend.config.enable_global_pool:
            view.shared_slots = self.view.shared_slots
            view.shared_generations = self.view.shared_generations
            view.shared_lengths = self.view.shared_lengths
        return view

    def stage(self, view, raw_bs, padded_bs):
        if not 0 <= raw_bs <= padded_bs <= len(self.view.slots):
            raise ValueError("invalid graph batch sizes")
        if view.slots.shape != (raw_bs,) or view.generations.shape != (raw_bs,):
            raise ValueError("override does not match the captured batch")
        mapping = (view.mapping_key, raw_bs, padded_bs)
        if view.mapping_key is None or mapping != self.staged_mapping:
            # Graphs only read this mapping. Unchanged batches need no GPU
            # staging kernels; changing shape must still clear active padding.
            self.view.slots[raw_bs:padded_bs].fill_(-1)
            self.view.slots[:raw_bs].copy_(view.slots)
            self.view.generations[:raw_bs].copy_(view.generations)
            if self.view.backend.config.enable_global_pool:
                self.view.shared_slots.copy_(view.shared_slots)
                self.view.shared_generations.copy_(view.shared_generations)
                self.view.shared_lengths.copy_(view.shared_lengths)
            self.staged_mapping = mapping
        # Feedback consumes the actual captured query's selection mask.
        view.use_retrieval = self.view.use_retrieval[:raw_bs]


@dataclass
class PrefillLease:
    rid: str
    request: weakref.ReferenceType
    slot: int
    generation: int
    limit: int
    build: Future | None
    device_initialized: bool = False


class DeviceDraftExtension:
    def wrap_graph_backend(self, backend, runner):
        if self.config.draft_graph_mode == "full":
            return backend
        from hybrid_spec.integrations.sglang.split_graph import SplitDraftGraphBackend

        return SplitDraftGraphBackend(backend, runner, self.config.draft_graph_mode)

    def __init__(self, config, backend, vocab_size, tp_sync=None):
        if (
            tp_sync is not None
            and backend.device.type == "cuda"
            and tp_sync.communicator is None
        ):
            raise ValueError(
                "device history TP requires a graph-capable PyNccl communicator"
            )
        self.config = config
        self.backend = backend
        self.vocab_size = vocab_size
        self.tp_sync = tp_sync
        self.metrics = Metrics()
        self.states = {}
        self.free_slots = list(range(config.num_sa_slots))
        # Completed requests keep their own automata. Never concatenate texts;
        # global query only reads after all per-slot updates have completed.
        self.shared = deque()
        self.generation = 0
        self.lock = threading.RLock()
        self.pending_view = None
        self.last_binding = None
        self.builder_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hybrid-sa-prefill"
        )

    def create_graph_buffers(self, max_bs):
        return DeviceGraphBuffers(self.backend, max_bs, self.tp_sync)

    def close_request(self, rid, *, retain=False):
        # Finish callbacks may run on the scheduler thread. Never touch GPU
        # state there: the next forward initializes a new generation in stream.
        with self.lock:
            state = self.states.pop(rid, None)
            if state is not None:
                req = state.request()
                if (
                    retain
                    and self.config.enable_global_pool
                    and self.config.max_shared_requests
                    and req is not None
                    and state.device_initialized
                    and state.build is None
                ):
                    length = len(req.origin_input_ids) + len(req.output_ids)
                    self.shared.append((state.slot, state.generation, length))
                    if len(self.shared) > self.config.max_shared_requests:
                        slot, _, _ = self.shared.popleft()
                        heapq.heappush(self.free_slots, slot)
                    return
                if state.build is not None:
                    state.build.cancel()
                heapq.heappush(self.free_slots, state.slot)

    def reset(self):
        with self.lock:
            for state in self.states.values():
                if state.build is not None:
                    state.build.cancel()
            self.states.clear()
            self.shared.clear()
            self.free_slots = list(range(self.config.num_sa_slots))
            # Retain the generation counter across flush, fencing old feedback.

    def shutdown(self):
        self.reset()
        self.builder_pool.shutdown(wait=True, cancel_futures=True)

    def cancel_prefill(self, ticket):
        with self.lock:
            for state in ticket:
                if state is not None and self.states.get(state.rid) is state:
                    self.close_request(state.rid)

    def begin_prefill(self, batch):
        """Build full histories on final prefill chunks, without waiting."""
        if batch.forward_mode.is_idle():
            return ()
        ticket = []
        try:
            with self.lock:
                for row, req in enumerate(batch.reqs):
                    self.close_request(req.rid)
                    # A middle chunk's sampled token is discarded by SGLang.
                    # It must not seed SA feedback or reserve a slot. Use this
                    # batch's marker, not req.inflight_middle_chunks, which
                    # overlap result processing can change independently.
                    # Closing first also invalidates pre-retraction history.
                    if req is getattr(batch, "chunked_req", None):
                        ticket.append(None)
                        continue
                    length = int(batch.seq_lens_cpu[row])
                    if (
                        len(self.states) >= self.config.max_requests
                        or not self.free_slots
                        or length >= self.config.max_history_tokens
                    ):
                        ticket.append(None)
                        continue
                    history = tuple(req.get_fill_ids()[:length])
                    if len(history) != length:
                        ticket.append(None)
                        continue
                    request_ref = weakref.ref(req)
                    limit = (
                        len(req.origin_input_ids) + req.sampling_params.max_new_tokens
                    )
                    host = self.backend.prepare_host(history)
                    future = self.builder_pool.submit(
                        self.backend.build_host, host, self.vocab_size
                    )
                    slot = heapq.heappop(self.free_slots)
                    self.generation += 1
                    state = PrefillLease(
                        req.rid,
                        request_ref,
                        slot,
                        self.generation,
                        limit,
                        future,
                    )
                    self.states[req.rid] = state
                    ticket.append(state)
        except BaseException:
            self.cancel_prefill(ticket)
            raise
        return tuple(ticket)

    def on_prefill(self, batch, result, ticket):
        if batch.forward_mode.is_idle():
            return
        slots, generations = [], []
        with self.lock:
            for req, state in zip(batch.reqs, ticket, strict=True):
                if (
                    state is None
                    or self.states.get(req.rid) is not state
                    or state.request() is not req
                ):
                    slots.append(-1)
                    generations.append(-1)
                    continue
                # Do not join the CPU build here. GPU prefill/draft-extend is
                # still running; import only at the first draft consumer.
                self.backend.initialize_device_slot(
                    state.slot, state.generation, state.limit
                )
                state.device_initialized = True
                slots.append(state.slot)
                generations.append(state.generation)
        if not any(slot >= 0 for slot in slots):
            return
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

    def _install_prefill(self, state):
        future = state.build
        if future is None:
            return
        try:
            host = future.result()
        except Exception:
            with self.lock:
                if self.states.get(state.rid) is not state:
                    return  # Cancelled/superseded work cannot fail another request.
                self.close_request(state.rid)
            raise
        with self.lock:
            # Finish/flush/retract can run while result() releases the GIL.
            if self.states.get(state.rid) is state:
                try:
                    self.backend.import_host(state.slot, host)
                except BaseException:
                    self.close_request(state.rid)
                    raise
                state.build = None

    def _view(self, slots, generations, shared=()):
        view = DeviceOverride(
            self.backend,
            torch.tensor(slots, dtype=torch.int64, device=self.backend.device),
            torch.tensor(generations, dtype=torch.int64, device=self.backend.device),
            self.tp_sync,
        )
        if self.config.enable_global_pool and shared:
            # Unused entries remain -1. Published order breaks exact donor ties.
            view.shared_slots[: len(shared)] = torch.tensor(
                [slot for slot, _, _ in shared],
                dtype=torch.int64,
                device=self.backend.device,
            )
            view.shared_generations[: len(shared)] = torch.tensor(
                [generation for _, generation, _ in shared],
                dtype=torch.int64,
                device=self.backend.device,
            )
            view.shared_lengths[: len(shared)] = torch.tensor(
                [length for _, _, length in shared],
                dtype=torch.int64,
                device=self.backend.device,
            )
        view.mapping_key = (tuple(slots), tuple(generations), shared)
        return view

    @contextmanager
    def bind(self, batch, draft_input):
        if batch.is_extend_in_batch or batch.has_grammar:
            raise ValueError("hybrid-spec does not support mixed prefill or grammar")
        if not batch.sampling_info.is_all_greedy:
            raise ValueError("hybrid-spec supports greedy requests only")
        slots, generations = [], []
        with self.lock:
            states = [self.states.get(req.rid) for req in batch.reqs]
        for req, state in zip(batch.reqs, states, strict=True):
            if (
                state is not None
                and state.request() is req
                and state.device_initialized
            ):
                self._install_prefill(state)
        with self.lock:
            for req in batch.reqs:
                if req.multimodal_inputs is not None:
                    raise ValueError("hybrid-spec does not index multimodal inputs")
                state = self.states.get(req.rid)
                if (
                    state is None
                    or state.request() is not req
                    or not state.device_initialized
                ):
                    slots.append(-1)
                    generations.append(-1)
                else:
                    slots.append(state.slot)
                    generations.append(state.generation)
            shared = ()
            if self.config.enable_global_pool:
                live = [
                    (state.slot, state.generation, -1)
                    for state in self.states.values()
                    if state.device_initialized and state.build is None
                ]
                shared = tuple(
                    sorted(
                        live + list(self.shared),
                        key=lambda entry: entry[1],
                        reverse=True,
                    )
                )
        binding_key = (tuple(slots), tuple(generations), shared)
        if self.last_binding is None or self.last_binding[0] != binding_key:
            self.last_binding = (binding_key, self._view(slots, generations, shared))
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
        self.backend.stage_feedback(
            view.slots,
            view.generations,
            result.next_token_ids,
            result.accept_index,
            result.accept_lens,
            batch.seq_lens,
            True,
            self.vocab_size,
            use_retrieval=view.use_retrieval,
            counts=self.metrics.ensure_buffer(self.backend.device),
        )
