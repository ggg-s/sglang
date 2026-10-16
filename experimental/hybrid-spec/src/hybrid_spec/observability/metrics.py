import torch


class Metrics:
    """Device counters. snapshot() is an explicit diagnostic synchronization."""

    names = (
        "retrieval_requests",
        "neural_requests",
        "retrieval_correct_drafts",
        "neural_correct_drafts",
        "retrieval_accept_tokens",
        "neural_accept_tokens",
    )

    def __init__(self):
        self.counts = None

    def ensure_buffer(self, device):
        # Allocate on the first consuming forward stream, not the construction
        # stream. The fused feedback kernel updates these same counters.
        if self.counts is None:
            self.counts = torch.zeros(len(self.names), dtype=torch.int64, device=device)
        return self.counts

    def record(self, use_retrieval, num_accept_tokens):
        if use_retrieval.shape != num_accept_tokens.shape:
            raise ValueError("feedback rows differ from the draft selection")
        num_correct_drafts = num_accept_tokens - 1
        mask = use_retrieval.to(torch.int64)
        values = torch.stack(
            (
                mask.sum(),
                (1 - mask).sum(),
                (mask * num_correct_drafts).sum(),
                ((1 - mask) * num_correct_drafts).sum(),
                (mask * num_accept_tokens).sum(),
                ((1 - mask) * num_accept_tokens).sum(),
            )
        )
        if self.counts is None:
            self.counts = values
        else:
            self.counts.add_(values)

    def snapshot(self):
        values = (
            [0] * len(self.names) if self.counts is None else self.counts.cpu().tolist()
        )
        return dict(zip(self.names, values))
