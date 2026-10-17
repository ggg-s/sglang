import torch

from hybrid_spec.core.contracts import DraftOverrideView
from hybrid_spec.integrations.sglang.gated_override import GatedOverride
from hybrid_spec.integrations.sglang.tp_sync import TPOverride


class GraphBuffers:
    def __init__(
        self,
        max_bs,
        num_drafts,
        device,
        tp_sync=None,
        num_neural_gate_tokens=0,
        *,
        defer_prefix=False,
    ):
        self.defer_prefix = defer_prefix
        self.neural_prefix = (
            torch.empty(
                (max_bs, max(0, num_neural_gate_tokens - 1)),
                device=device,
                dtype=torch.int64,
            )
            if defer_prefix
            else None
        )
        self.tp_sync = tp_sync
        self.num_neural_gate_tokens = num_neural_gate_tokens
        self.tokens = torch.zeros(
            (max_bs, num_drafts), device=device, dtype=torch.int64
        )
        self.use_retrieval = torch.zeros(max_bs, device=device, dtype=torch.bool)

    def bind_capture_view(self, batch_size):
        if self.defer_prefix:
            return GatedOverride(
                self.tokens[:batch_size],
                self.use_retrieval[:batch_size],
                self.num_neural_gate_tokens,
                tp_sync=self.tp_sync,
                defer_prefix=True,
                neural_prefix=self.neural_prefix[:batch_size],
            )
        if self.tp_sync is not None:
            return TPOverride(
                self.tokens[:batch_size],
                self.use_retrieval[:batch_size],
                self.tp_sync,
                self.num_neural_gate_tokens,
            )
        return DraftOverrideView(
            self.tokens[:batch_size],
            self.use_retrieval[:batch_size],
            self.num_neural_gate_tokens,
        )

    def stage(self, view, raw_bs, padded_bs):
        if not 0 <= raw_bs <= padded_bs <= self.tokens.shape[0]:
            raise ValueError("invalid graph batch sizes")
        if view.tokens.shape != (raw_bs, self.tokens.shape[1]):
            raise ValueError("override does not match the captured draft width")
        # Clear all padding even if the preceding replay used a larger bucket.
        self.use_retrieval[raw_bs:].zero_()
        self.tokens[:raw_bs].copy_(view.tokens)
        self.use_retrieval[:raw_bs].copy_(view.use_retrieval)
        # prepare() updates this mask in graph, after staging local eligibility.
        view.use_retrieval = self.use_retrieval[:raw_bs]
