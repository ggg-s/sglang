from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HybridConfig:
    num_drafts: int
    min_match_len: int
    max_requests: int
    max_history_tokens: int
    backend: str = "gpu_sa"
    history_mode: str = "cpu"
    partial_drafts: str = "fallback"

    def __post_init__(self):
        for name in (
            "num_drafts",
            "min_match_len",
            "max_requests",
            "max_history_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_drafts >= self.max_history_tokens:
            raise ValueError("num_drafts must be smaller than max_history_tokens")
        if self.backend not in ("gpu_sa", "reference"):
            raise ValueError("backend must be gpu_sa or reference")
        if self.history_mode not in ("cpu", "device"):
            raise ValueError("history_mode must be cpu or device")
        if self.partial_drafts not in ("fallback", "zero"):
            raise ValueError("partial_drafts must be fallback or zero")

    @classmethod
    def from_file(cls, path: str):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
