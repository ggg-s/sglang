from __future__ import annotations

import weakref
from contextlib import contextmanager

import torch

from hybrid_spec.core.runtime import HybridRuntime
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers
from hybrid_spec.integrations.sglang.tp_sync import TPOverride
from hybrid_spec.observability.metrics import Metrics


class DraftExtension:
    def __init__(self, config, backend, vocab_size, tp_sync=None):
        if (
            tp_sync is not None
            and backend.device.type == "cuda"
            and tp_sync.communicator is None
        ):
            raise ValueError(
                "CPU history TP requires a graph-capable PyNccl communicator"
            )
        self.config = config
        self.runtime = HybridRuntime(config, backend, vocab_size)
        self.device = backend.device
        self.identities = {}
        self.metrics = Metrics()
        self.pending_view = None
        self.tp_sync = tp_sync

    def create_graph_buffers(self, max_bs):
        return GraphBuffers(
            max_bs,
            self.config.num_drafts,
            self.device,
            self.tp_sync,
            self.config.num_neural_gate_tokens,
        )

    @contextmanager
    def bind(self, batch, draft_input):
        if batch.is_extend_in_batch or batch.has_grammar:
            raise ValueError("hybrid-spec does not support mixed prefill or grammar")
        if not batch.sampling_info.is_all_greedy:
            raise ValueError("hybrid-spec supports greedy requests only")
        for rid, ref in list(self.identities.items()):
            req = ref()
            if req is None or req.finished():
                self.close_request(rid, retain=req is not None and req.finished())
        requests = []
        for req in batch.reqs:
            ref = self.identities.get(req.rid)
            if ref is None or ref() is not req:
                self.runtime.close_request(req.rid)
                self.identities[req.rid] = weakref.ref(req)
            if req.multimodal_inputs is not None:
                raise ValueError("hybrid-spec does not index multimodal inputs")
            self._prepare_request(req)
            history = self._request_history(req)
            requests.append(
                (
                    req.rid,
                    history,
                    req.sampling_params.max_new_tokens
                    - (len(history) - len(req.origin_input_ids)),
                )
            )
        view = self.runtime.prepare(requests)
        # The native seq_lens excludes the current anchor. A mismatch must
        # fall back, never apply candidates to another history version.
        expected = torch.tensor(
            [len(history) for _, history, _ in requests],
            dtype=torch.int64,
            device=self.device,
        )
        view.use_retrieval &= batch.seq_lens + 1 == expected
        view = self._wrap_view(view)
        self.pending_view = view
        draft_input.draft_override = view
        try:
            yield view
        finally:
            draft_input.draft_override = None

    def _request_history(self, req):
        if not req.output_ids:
            raise ValueError("hybrid-spec requires the committed prefill anchor")
        return req.origin_input_ids + req.output_ids

    def _prepare_request(self, req):
        pass

    def _wrap_view(self, view):
        if self.tp_sync is not None:
            return TPOverride(
                view.tokens,
                view.use_retrieval,
                self.tp_sync,
                self.config.num_neural_gate_tokens,
                query_metadata=view.query_metadata,
            )
        return view

    def close_request(self, rid, *, retain=False):
        self.runtime.close_request(rid)
        self.identities.pop(rid, None)

    def reset(self):
        self.runtime.reset()
        self.identities.clear()
        self.pending_view = None

    def on_verify_result(self, result):
        if self.pending_view is not None:
            self.metrics.record(self.pending_view.use_retrieval, result.accept_lens)
            self.pending_view = None
