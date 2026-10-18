"""CPU suffix queries with GPU verify feedback and compact split graphs."""

import weakref
from dataclasses import dataclass

import torch

from hybrid_spec.core.runtime import HybridRuntime
from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
from hybrid_spec.integrations.sglang.gated_override import GatedOverride
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers


@dataclass
class Feedback:
    host: torch.Tensor
    event: object = None

    def row(self, index):
        # A CPU suffix query needs these tokens. Wait only at the next consumer,
        # after the copy was enqueued behind verify, before draft-extend.
        if self.event is not None:
            self.event.synchronize()
        return self.host[index].tolist()


@dataclass
class ArcticHistory:
    request: object
    tokens: tuple[int, ...]
    pending: object = None

    def consume(self):
        if self.pending is not None:
            feedback, row = self.pending
            values = feedback.row(row)
            self.pending = None
            count, version, healthy = values[-3:]
            if not healthy or len(self.tokens) != version:
                return False
            self.tokens += tuple(values[:count])
        return True


class ArcticRuntime(HybridRuntime):
    def close_request(self, rid, *, retain=False):
        state = self.states.get(rid)
        if state is not None:
            self.backend.close_slot(state.slot, retain=retain)
        super().close_request(rid)

    def reset(self):
        self.backend.reset()
        super().reset()


class ArcticDraftExtension(DraftExtension):
    def __init__(self, config, backend, vocab_size, tp_sync=None):
        if vocab_size > 0x80000000:
            raise ValueError("Arctic suffix tokens must fit signed int32")
        super().__init__(config, backend, vocab_size, tp_sync)
        self.runtime = ArcticRuntime(config, backend, vocab_size)
        self.histories = {}

    def _prepare_request(self, req):
        self.runtime.backend.set_prompt(req.rid, req.origin_input_ids)
        self.runtime.backend.set_confirmed(req.rid, req.output_ids)

    def _request_history(self, req):
        committed = tuple(req.origin_input_ids + req.output_ids)
        state = self.histories.get(req.rid)
        if state is not None and state.request() is req:
            if state.consume() and len(state.tokens) <= self.config.max_history_tokens:
                # CPU stop handling may have trimmed the last GPU accept path.
                # Such requests are closed before another draft. For a live
                # rewrite/retraction discard the private tree and use CPU data.
                if state.tokens[: len(committed)] == committed:
                    return list(state.tokens)
                if committed[: len(state.tokens)] != state.tokens:
                    self.histories.pop(req.rid, None)
                    return list(committed)
            else:
                self.histories.pop(req.rid, None)
                return list(committed)
        self.histories.pop(req.rid, None)
        if (
            len(committed) <= self.config.max_history_tokens
            and len(self.histories) < self.config.max_requests
        ):
            self.histories[req.rid] = ArcticHistory(weakref.ref(req), committed)
        return list(committed)

    def begin_prefill(self, batch):
        if batch.forward_mode.is_idle():
            return ()
        for rid, ref in list(self.identities.items()):
            req = ref()
            if req is None or req.finished():
                self.close_request(rid, retain=req is not None and req.finished())
        ticket = []
        try:
            for row, req in enumerate(batch.reqs):
                # Also invalidate pre-retraction feedback on intermediate chunks.
                self.close_request(req.rid)
                if req is getattr(batch, "chunked_req", None):
                    ticket.append(None)
                    continue
                length = int(batch.seq_lens_cpu[row])
                history = tuple(req.get_fill_ids()[:length])
                if (
                    len(history) != length
                    or length >= self.config.max_history_tokens
                    or len(self.histories) >= self.config.max_requests
                ):
                    ticket.append(None)
                    continue
                state = ArcticHistory(weakref.ref(req), history)
                self.histories[req.rid] = state
                self.identities[req.rid] = state.request
                ticket.append(state)
        except BaseException:
            self.cancel_prefill(ticket)
            raise
        return tuple(ticket)

    def cancel_prefill(self, ticket):
        for state in ticket:
            if state is not None:
                req = state.request()
                if req is not None and self.histories.get(req.rid) is state:
                    self.close_request(req.rid)

    def _stage_feedback(self, tokens, counts, versions, healthy):
        packed = torch.cat(
            (tokens, counts[:, None], versions[:, None], healthy[:, None]), dim=1
        ).to(torch.int64)
        if packed.is_cuda:
            host = torch.empty_like(packed, device="cpu", pin_memory=True)
            host.copy_(packed, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            return Feedback(host, event)
        return Feedback(packed.cpu())

    def on_prefill(self, batch, result, ticket):
        if not ticket or not any(state is not None for state in ticket):
            return
        bs = len(ticket)
        feedback = self._stage_feedback(
            result.next_token_ids.reshape(bs, 1),
            torch.ones(bs, dtype=torch.int64, device=self.device),
            batch.seq_lens,
            (result.next_token_ids >= 0)
            & (result.next_token_ids < self.runtime.vocab_size),
        )
        for row, (req, state) in enumerate(zip(batch.reqs, ticket, strict=True)):
            if state is not None and self.histories.get(req.rid) is state:
                state.pending = (feedback, row)

    def on_device_verify(self, result, batch):
        if self.pending_view is None or batch.forward_mode.is_idle():
            return
        self.on_verify_result(result)
        # topk=1: accept_index is a row of global node indices, padded with -1.
        indices = result.accept_index.to(torch.int64)
        source = result.next_token_ids.reshape(-1)
        tokens = source[indices.clamp(0, source.numel() - 1)]
        positions = torch.arange(indices.shape[1], device=indices.device)
        valid = positions[None, :] < result.accept_lens[:, None]
        healthy = (
            (
                (
                    (indices >= 0)
                    & (indices < source.numel())
                    & (tokens >= 0)
                    & (tokens < self.runtime.vocab_size)
                )
                | ~valid
            ).all(dim=1)
            & (result.accept_lens >= 1)
            & (result.accept_lens <= indices.shape[1])
        )
        feedback = self._stage_feedback(
            tokens, result.accept_lens, batch.seq_lens + 1, healthy
        )
        for row, req in enumerate(batch.reqs):
            state = self.histories.get(req.rid)
            if state is not None and state.request() is req:
                state.pending = (feedback, row)

    def reset(self):
        super().reset()
        self.histories.clear()

    def _wrap_view(self, view):
        return GatedOverride(
            view.tokens,
            view.use_retrieval,
            self.config.num_neural_gate_tokens,
            tp_sync=self.tp_sync,
            defer_prefix=self.config.draft_graph_mode != "full",
            query_metadata=view.query_metadata,
        )

    def create_graph_buffers(self, max_bs):
        return GraphBuffers(
            max_bs,
            self.config.num_drafts,
            self.device,
            self.tp_sync,
            self.config.num_neural_gate_tokens,
            defer_prefix=self.config.draft_graph_mode != "full",
        )

    def wrap_graph_backend(self, backend, runner):
        if self.config.draft_graph_mode == "full":
            return backend
        from hybrid_spec.integrations.sglang.split_graph import SplitDraftGraphBackend

        return SplitDraftGraphBackend(backend, runner, self.config.draft_graph_mode)

    def close_request(self, rid, *, retain=False):
        ref = self.identities.get(rid)
        req = ref() if ref is not None else None
        # The last verify runs before the scheduler commits/trims CPU output.
        # Reconcile the final delivered history before retaining a global donor.
        if retain and req is not None and req.finished():
            self._prepare_request(req)
            self.runtime.reconcile(rid, req.origin_input_ids + req.output_ids)
        else:
            retain = False
        self.runtime.close_request(rid, retain=retain)
        self.runtime.backend.prompts.pop(rid, None)
        self.runtime.backend.confirmed.pop(rid, None)
        self.histories.pop(rid, None)
        self.identities.pop(rid, None)
