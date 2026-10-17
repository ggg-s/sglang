"""Committed CPU history with the shared GPU gate and compact split graphs."""

from hybrid_spec.core.runtime import HybridRuntime
from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
from hybrid_spec.integrations.sglang.gated_override import GatedOverride
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers


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

    def _prepare_request(self, req):
        self.runtime.backend.set_prompt(req.rid, req.origin_input_ids)

    def _wrap_view(self, view):
        return GatedOverride(
            view.tokens,
            view.use_retrieval,
            self.config.num_neural_gate_tokens,
            tp_sync=self.tp_sync,
            defer_prefix=self.config.draft_graph_mode != "full",
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
        self.identities.pop(rid, None)
