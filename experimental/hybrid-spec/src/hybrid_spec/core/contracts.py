from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class RequestKey:
    rid: str
    generation: int


@dataclass(frozen=True)
class HistoryDelta:
    key: RequestKey
    start_version: int
    accept_tokens: tuple[int, ...]

    @property
    def end_version(self):
        return self.start_version + len(self.accept_tokens)


@dataclass(frozen=True)
class RequestSnapshot:
    keys: tuple[RequestKey, ...]
    slots: tuple[int, ...]
    versions: tuple[int, ...]
    remaining: tuple[int, ...]


@dataclass
class RetrievalCandidates:
    tokens: torch.Tensor
    num_valid_drafts: torch.Tensor
    match_lens: torch.Tensor
    versions: torch.Tensor
    healthy: torch.Tensor


@dataclass
class DraftOverrideView:
    tokens: torch.Tensor
    use_retrieval: torch.Tensor

    def select_step(self, neural_tokens: torch.Tensor, step: int):
        # topk=1 has exactly one token per row. Preserve [B] or [B,1], never
        # accidentally broadcast [B] against [B,1] into [B,B].
        if neural_tokens.shape not in (
            (self.tokens.shape[0],),
            (self.tokens.shape[0], 1),
        ):
            raise ValueError("hybrid selection requires one neural token per row")
        return torch.where(
            self.use_retrieval.reshape(neural_tokens.shape),
            self.tokens[:, step].reshape(neural_tokens.shape),
            neural_tokens,
        )


class RetrievalBackend(Protocol):
    device: torch.device

    def reset_slot(self, slot: int) -> None: ...

    def append(self, slot: int, delta: HistoryDelta) -> None: ...

    def query(
        self, snapshot: RequestSnapshot, num_drafts: int
    ) -> RetrievalCandidates: ...
