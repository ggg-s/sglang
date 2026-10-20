"""Instance-owned CUDA suffix automata; compilation is deferred to construction."""

from __future__ import annotations

from pathlib import Path

import torch

from hybrid_spec.backends.host_sa import NativeHostBuilder
from hybrid_spec.core.contracts import RetrievalCandidates


class GPUSuffixAutomaton:
    @staticmethod
    def estimate_memory(config):
        h, n = config.max_history_tokens, config.num_sa_slots
        return n * (
            (h + (2 * h + 1) * 5 + (h + 1) * 6 + 7) * 4
            + 40
            + 4 * (config.num_drafts + 1)
        )

    @staticmethod
    def load_module():
        from torch.utils.cpp_extension import load

        return load(
            name="hybrid_spec_gpu_sa_v4",
            sources=[
                str(Path(__file__).with_name("suffix_automaton.cu")),
                str(Path(__file__).with_name("host_builder.cpp")),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
        )

    def __init__(self, config, device):
        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.version.hip:
            raise ValueError("gpu_sa requires NVIDIA CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.module = self.load_module()
        self.host_builder = NativeHostBuilder(self.module.__file__)
        self.host_transfers = []
        n, h = config.num_sa_slots, config.max_history_tokens
        self.history = torch.empty((n, h), dtype=torch.int32, device=self.device)
        self.states = torch.empty(
            (n, 2 * h + 1, 5), dtype=torch.int32, device=self.device
        )
        self.edges = torch.empty((n, h + 1, 6), dtype=torch.int32, device=self.device)
        self.meta = torch.zeros((n, 7), dtype=torch.int32, device=self.device)
        self.owners = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.limits = torch.zeros(n, dtype=torch.int64, device=self.device)
        self.pending = torch.empty(
            (n, config.num_drafts + 1), dtype=torch.int32, device=self.device
        )
        self.pending_meta = torch.zeros((n, 3), dtype=torch.int64, device=self.device)
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

    def initialize_device_slot(self, slot, generation, limit):
        self.reset_slot(slot)
        self.owners[slot].fill_(generation)
        self.limits[slot].fill_(limit)
        self.pending_meta[slot].zero_()

    def prepare_host(self, history):
        return self.host_builder.prepare(history)

    def build_host(self, host, vocab_size):
        return self.host_builder.build(host, vocab_size)

    def import_host(self, slot, host):
        self._check_stream()
        length, num_states, num_edges, _, healthy, hash_base, num_indexed = (
            host.meta.tolist()
        )
        h = self.config.max_history_tokens
        if not (
            0 <= slot < self.config.num_sa_slots
            and 0 <= length < h
            and 1 <= num_states <= min(host.states.shape[0], 2 * h + 1)
            and 0 <= num_edges <= min(host.edges.shape[0], h + 1)
            and healthy == 1
            and hash_base > 0
            and hash_base & (hash_base - 1) == 0
            and hash_base <= max(1, num_indexed) < 2 * hash_base
            and 0 <= num_indexed <= num_edges
        ):
            raise ValueError("invalid completed host automaton")
        # Only used prefixes are copied; metadata/bonus feedback and ownership
        # live in separate buffers. All copies precede draft replay in stream.
        self.history[slot, :length].copy_(host.history[:length], non_blocking=True)
        self.states[slot, :num_states].copy_(
            host.states[:num_states], non_blocking=True
        )
        self.edges[slot, :num_edges].copy_(host.edges[:num_edges], non_blocking=True)
        self.meta[slot].copy_(host.meta, non_blocking=True)
        # Explicitly retain pinned sources until their asynchronous H2D reads
        # finish, even if abort/flush closes the request immediately afterwards.
        self.host_transfers = [
            transfer for transfer in self.host_transfers if not transfer[0].query()
        ]
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        self.host_transfers.append((done, host))

    def stage_feedback(
        self,
        slots,
        generations,
        predict,
        indices,
        lengths,
        seq_lens,
        has_anchor,
        vocab,
        *,
        use_retrieval=None,
        counts=None,
    ):
        self._check_stream()
        self.module.stage_feedback(
            self.meta,
            self.owners,
            self.pending,
            self.pending_meta,
            slots,
            generations,
            predict,
            indices,
            lengths,
            seq_lens,
            has_anchor,
            vocab,
            use_retrieval,
            counts,
        )

    def consume_and_query(
        self,
        slots,
        generations,
        seq_lens,
        tokens,
        mask,
        *,
        shared_slots=None,
        shared_generations=None,
        shared_lengths=None,
        query_metadata=None,
    ):
        # Called on the native capture stream during capture and on its forward
        # stream during replay. No host readback, allocation or stream binding.
        self.module.consume_and_query(
            self.history,
            self.states,
            self.edges,
            self.meta,
            self.owners,
            self.limits,
            self.pending,
            self.pending_meta,
            slots,
            generations,
            seq_lens,
            tokens,
            mask,
            self.config.min_match_len,
            self.config.partial_drafts == "zero",
            shared_slots,
            shared_generations,
            shared_lengths,
            self.config.shared_context_tokens,
            query_metadata,
            self.config.sa_require_unique_continuation,
        )
