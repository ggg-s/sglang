"""Instance-owned CUDA suffix automata; compilation is deferred to construction."""

from __future__ import annotations

from pathlib import Path

import torch
from hybrid_spec.core.contracts import RetrievalCandidates


class GPUSuffixAutomaton:
    @staticmethod
    def estimate_memory(config):
        h, n = config.max_history_tokens, config.max_requests
        return n * (h + (2 * h + 1) * 4 + (3 * h + 1) * 3 + 5) * 4

    def __init__(self, config, device):
        from torch.utils.cpp_extension import load

        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.version.hip:
            raise ValueError("gpu_sa requires NVIDIA CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.module = load(
            name="hybrid_spec_gpu_sa_v1",
            sources=[str(Path(__file__).with_name("suffix_automaton.cu"))],
            extra_cuda_cflags=["-O3"],
        )
        n, h = config.max_requests, config.max_history_tokens
        self.history = torch.empty((n, h), dtype=torch.int32, device=self.device)
        self.states = torch.empty(
            (n, 2 * h + 1, 4), dtype=torch.int32, device=self.device
        )
        self.edges = torch.empty(
            (n, 3 * h + 1, 3), dtype=torch.int32, device=self.device
        )
        self.meta = torch.zeros((n, 5), dtype=torch.int32, device=self.device)
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(self.device))
        self.stream = None

    def _check_stream(self):
        if self.stream is None:
            self.stream = torch.cuda.current_stream(self.device)
            self.stream.wait_event(self.ready)
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("gpu_sa must be used on its owning forward stream")

    def reset_slot(self, slot):
        self._check_stream()
        self.module.reset(self.states, self.meta, slot)

    def append(self, slot, delta):
        self._check_stream()
        tokens = torch.tensor(
            delta.accept_tokens, dtype=torch.int64, device=self.device
        )
        self.module.append(
            self.history,
            self.states,
            self.edges,
            self.meta,
            tokens,
            slot,
            delta.start_version,
        )

    def query(self, snapshot, num_drafts):
        self._check_stream()
        slots = torch.tensor(snapshot.slots, dtype=torch.int64, device=self.device)
        outputs = self.module.query(
            self.history, self.states, self.edges, self.meta, slots, num_drafts
        )
        return RetrievalCandidates(*outputs)
