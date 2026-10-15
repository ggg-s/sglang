import torch
from hybrid_spec.core.contracts import DraftOverrideView


class GraphBuffers:
    def __init__(self, max_bs, num_drafts, device):
        self.tokens = torch.zeros(
            (max_bs, num_drafts), device=device, dtype=torch.int64
        )
        self.use_retrieval = torch.zeros(max_bs, device=device, dtype=torch.bool)

    def bind_capture_view(self, batch_size):
        return DraftOverrideView(
            self.tokens[:batch_size], self.use_retrieval[:batch_size]
        )

    def stage(self, view, raw_bs, padded_bs):
        if not 0 <= raw_bs <= padded_bs <= self.tokens.shape[0]:
            raise ValueError("invalid graph batch sizes")
        if view.tokens.shape != (raw_bs, self.tokens.shape[1]):
            raise ValueError("override does not match the captured draft width")
        # Clear all padding even if the preceding replay used a larger bucket.
        self.use_retrieval.zero_()
        self.tokens[:raw_bs].copy_(view.tokens)
        self.use_retrieval[:raw_bs].copy_(view.use_retrieval)
