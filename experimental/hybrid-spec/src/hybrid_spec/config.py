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
    num_neural_drafts: int | None = None
    verify_mode: str = "static"
    verify_token_buckets: tuple[int, ...] = ()
    dynamic_verify: bool = False
    dynamic_verify_min_samples: int = 32
    dynamic_verify_explore_interval: int = 32
    dynamic_verify_timing_interval: int = 16
    dynamic_verify_profile: bool = False
    dynamic_verify_prior_strength: float = 4.0
    dynamic_verify_prior_rate: float = 0.8
    dynamic_verify_min_drafts: int = 1
    dynamic_verify_cost_table_path: str | None = None
    conditional_draft_graph: bool = False
    conditional_draft_graph_dispatch: str = "if"
    conditional_draft_graph_joined: bool = False

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
        if type(self.conditional_draft_graph) is not bool or (
            self.conditional_draft_graph and self.draft_graph_mode != "split"
        ):
            raise ValueError("conditional_draft_graph requires draft_graph_mode=split")
        if self.conditional_draft_graph_dispatch not in ("if", "switch") or (
            self.conditional_draft_graph_dispatch != "if"
            and not self.conditional_draft_graph
        ):
            raise ValueError(
                "conditional_draft_graph_dispatch requires if or switch "
                "and switch requires conditional_draft_graph=true"
            )
        if type(self.conditional_draft_graph_joined) is not bool or (
            self.conditional_draft_graph_joined
            and self.conditional_draft_graph_dispatch != "switch"
        ):
            raise ValueError("conditional_draft_graph_joined requires SWITCH dispatch")
        if self.verify_mode not in ("static", "padded", "compact"):
            raise ValueError("verify_mode must be static, padded or compact")
        neural = self.neural_drafts
        if type(neural) is not int or not 1 <= neural <= self.num_drafts:
            raise ValueError("num_neural_drafts must be between 1 and num_drafts")
        if neural != self.num_drafts and (
            self.verify_mode == "static" or self.draft_graph_mode != "split"
        ):
            raise ValueError(
                "different draft lengths require split and padded/compact verify"
            )
        if not isinstance(self.verify_token_buckets, (tuple, list)) or any(
            type(n) is not int or n < 1 for n in self.verify_token_buckets
        ):
            raise ValueError("verify_token_buckets must contain positive integers")
        if self.verify_token_buckets and self.verify_mode != "compact":
            raise ValueError("verify_token_buckets requires compact verify")
        if type(self.dynamic_verify) is not bool:
            raise ValueError("dynamic_verify must be a boolean")
        if type(self.dynamic_verify_profile) is not bool or (
            self.dynamic_verify_profile
            and (not self.dynamic_verify or self.dynamic_verify_cost_table_path)
        ):
            raise ValueError(
                "dynamic_verify_profile requires dynamic_verify and no seed table"
            )
        if self.dynamic_verify_cost_table_path is not None and (
            not isinstance(self.dynamic_verify_cost_table_path, str)
            or not self.dynamic_verify_cost_table_path.strip()
            or not self.dynamic_verify
        ):
            raise ValueError(
                "dynamic_verify_cost_table_path requires dynamic_verify and a path"
            )
        for name in (
            "dynamic_verify_min_samples",
            "dynamic_verify_explore_interval",
            "dynamic_verify_timing_interval",
            "dynamic_verify_min_drafts",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dynamic_verify_prior_strength", "dynamic_verify_prior_rate"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.dynamic_verify_prior_rate >= 1:
            raise ValueError("dynamic_verify_prior_rate must be smaller than one")
        if not 1e-30 <= self.dynamic_verify_prior_strength <= 1e30:
            raise ValueError(
                "dynamic_verify_prior_strength must fit finite float32 arithmetic"
            )
        if self.dynamic_verify_min_samples > 0x7FFFFFFF:
            raise ValueError("dynamic_verify_min_samples must fit int32")
        if self.dynamic_verify_min_drafts > neural:
            raise ValueError("dynamic_verify_min_drafts exceeds neural draft budget")
        if self.dynamic_verify and (
            self.verify_mode != "compact" or self.partial_drafts != "fallback"
        ):
            raise ValueError(
                "dynamic_verify requires compact verify and full candidates"
            )
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
    def neural_drafts(self):
        return (
            self.num_drafts
            if self.num_neural_drafts is None
            else self.num_neural_drafts
        )

    @property
    def num_sa_slots(self):
        return self.max_requests + self.max_shared_requests

    @classmethod
    def from_file(cls, path: str):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
