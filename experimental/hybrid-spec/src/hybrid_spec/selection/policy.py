import torch
from hybrid_spec.core.contracts import DraftOverrideView, RetrievalCandidates


def select_requests(
    candidates: RetrievalCandidates,
    versions: torch.Tensor,
    remaining: torch.Tensor,
    min_match_len: int,
    vocab_size: int,
    partial_drafts: str = "fallback",
) -> DraftOverrideView:
    num_drafts = candidates.tokens.shape[1]
    mask = (
        candidates.healthy
        & (candidates.versions == versions)
        & (candidates.match_lens >= min_match_len)
        & (
            candidates.num_valid_drafts
            >= (1 if partial_drafts == "zero" else num_drafts)
        )
        & (remaining >= num_drafts + 1)
        & ((candidates.tokens >= 0) & (candidates.tokens < vocab_size)).all(dim=1)
    )
    return DraftOverrideView(candidates.tokens, mask)
