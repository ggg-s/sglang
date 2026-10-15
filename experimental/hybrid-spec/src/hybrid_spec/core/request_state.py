from __future__ import annotations

from dataclasses import dataclass

from hybrid_spec.core.contracts import HistoryDelta, RequestKey


@dataclass
class RequestState:
    key: RequestKey
    slot: int
    history: tuple[int, ...] = ()

    def commit(self, delta: HistoryDelta) -> bool:
        if delta.key != self.key:
            raise ValueError("stale request generation")
        if delta.start_version < 0:
            raise ValueError("negative history version")
        if delta.end_version <= len(self.history):
            if (
                self.history[delta.start_version : delta.end_version]
                == delta.accept_tokens
            ):
                return False
            raise ValueError("conflicting history replay")
        if delta.start_version != len(self.history):
            raise ValueError("history gap or partial overlap")
        self.history += delta.accept_tokens
        return True
