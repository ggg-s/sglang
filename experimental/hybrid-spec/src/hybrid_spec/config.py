from __future__ import annotations

import json
import math
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
    num_neural_gate_tokens: int = 0
    draft_graph_mode: str = "full"
    enable_global_pool: bool = False
    max_shared_requests: int = 0
    shared_context_tokens: int = 128
    arctic_max_tree_depth: int = 64
    arctic_max_spec_factor: float = 1.0
    arctic_max_spec_offset: float = 0.0
    arctic_min_token_prob: float = 0.1

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
        if self.backend not in ("gpu_sa", "reference", "arctic"):
            raise ValueError("backend must be gpu_sa, arctic or reference")
        if self.history_mode not in ("cpu", "device"):
            raise ValueError("history_mode must be cpu or device")
        if self.partial_drafts not in ("fallback", "zero"):
            raise ValueError("partial_drafts must be fallback or zero")
        if (
            type(self.num_neural_gate_tokens) is not int
            or not 0 <= self.num_neural_gate_tokens <= self.num_drafts
        ):
            raise ValueError("num_neural_gate_tokens must be between 0 and num_drafts")
        if self.draft_graph_mode not in ("full", "split", "split_no_skip"):
            raise ValueError("draft_graph_mode must be full, split or split_no_skip")
        if self.backend == "arctic" and self.history_mode != "cpu":
            raise ValueError("Arctic suffix requires history_mode=cpu")
        if (
            self.draft_graph_mode != "full"
            and self.history_mode != "device"
            and self.backend != "arctic"
        ):
            raise ValueError("split draft graphs require device history")
        if type(self.max_shared_requests) is not int or self.max_shared_requests < 0:
            raise ValueError("max_shared_requests must be a nonnegative integer")
        if (
            type(self.shared_context_tokens) is not int
            or self.shared_context_tokens < 1
        ):
            raise ValueError("shared_context_tokens must be a positive integer")
        if type(self.enable_global_pool) is not bool:
            raise ValueError("enable_global_pool must be a boolean")
        if self.max_shared_requests and not self.enable_global_pool:
            raise ValueError("max_shared_requests requires enable_global_pool=true")
        if (
            self.enable_global_pool
            and self.history_mode != "device"
            and self.backend != "arctic"
        ):
            raise ValueError("shared history requires history_mode=device")

        if (
            type(self.arctic_max_tree_depth) is not int
            or self.arctic_max_tree_depth < 1
        ):
            raise ValueError("arctic_max_tree_depth must be a positive integer")
        for name in (
            "arctic_max_spec_factor",
            "arctic_max_spec_offset",
            "arctic_min_token_prob",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.arctic_max_spec_factor < 0:
            raise ValueError("arctic_max_spec_factor must be nonnegative")
        if not 0 <= self.arctic_min_token_prob <= 1:
            raise ValueError("arctic_min_token_prob must be between 0 and 1")
        if self.backend == "arctic" and self.num_sa_slots > 0x7FFFFFFF:
            raise ValueError("Arctic request capacity must fit int32")

    @property
    def num_sa_slots(self):
        return self.max_requests + self.max_shared_requests

    @classmethod
    def from_file(cls, path: str):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
