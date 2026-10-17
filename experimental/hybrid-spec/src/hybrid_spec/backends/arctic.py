"""Optional official ArcticInference CPU suffix-tree backend.

Only committed histories enter this cache. Drafting and CPU->GPU staging happen
outside graphs; the shared gate/TP/split consumer reads fixed GPU buffers.
"""

from collections import deque
from dataclasses import dataclass

import torch

from hybrid_spec.core.contracts import RetrievalCandidates


@dataclass
class ArcticSlot:
    key: object
    history: tuple[int, ...]


class ArcticBackend:
    def __init__(self, config, device="cuda"):
        try:
            from arctic_inference.suffix_decoding import SuffixDecodingCache
        except ImportError as exc:
            raise ImportError(
                "backend=arctic requires ArcticInference with its native suffix "
                "extension; install hybrid-spec[arctic] or the pinned upstream "
                "suffix-only build described in validation/arctic-suffix.md"
            ) from exc
        self.config = config
        self.device = torch.device(device)
        self.cache_type = SuffixDecodingCache
        self.prompts = {}
        self.reset()

    def reset(self):
        # Reserve enough global slots for live sources plus retained responses.
        # A separate FIFO below bounds completed donors, even at low concurrency.
        capacity = self.config.num_sa_slots if self.config.enable_global_pool else 0
        self.cache = self.cache_type(
            max_tree_depth=self.config.arctic_max_tree_depth,
            max_cached_requests=capacity,
        )
        self.slots = {}
        self.completed = deque()
        self.prompts.clear()

    def set_prompt(self, rid, tokens):
        self.prompts[rid] = tuple(tokens)

    def reset_slot(self, slot):
        self.close_slot(slot)

    def close_slot(self, slot, *, retain=False):
        state = self.slots.pop(slot, None)
        if state is None:
            return
        self.cache.stop_request(state.key)
        if (
            retain
            and self.config.enable_global_pool
            and self.config.max_shared_requests
        ):
            self.completed.append(state.key)
        elif state.key in self.cache.cached_requests:
            self.cache.evict_cached_response(state.key)
        while len(self.completed) > self.config.max_shared_requests:
            key = self.completed.popleft()
            if key in self.cache.cached_requests:
                self.cache.evict_cached_response(key)

    def append(self, slot, delta):
        state = self.slots.get(slot)
        if state is None:
            prompt = self.prompts[delta.key.rid]
            if delta.start_version != 0 or delta.accept_tokens[: len(prompt)] != prompt:
                raise ValueError("Arctic initial history must contain its prompt")
            self.cache.start_request(delta.key, prompt)
            state = ArcticSlot(delta.key, ())
            self.slots[slot] = state
            output = delta.accept_tokens[len(prompt) :]
        else:
            if state.key != delta.key or len(state.history) != delta.start_version:
                raise ValueError("Arctic history generation/version mismatch")
            output = delta.accept_tokens
        if output:
            self.cache.add_active_response(state.key, output)
        state.history += delta.accept_tokens

    def query(self, snapshot, num_drafts):
        bs = len(snapshot.slots)
        tokens = torch.zeros((bs, num_drafts), dtype=torch.int64)
        valid = torch.zeros(bs, dtype=torch.int64)
        matches = torch.zeros_like(valid)
        versions = torch.zeros_like(valid)
        healthy = torch.zeros(bs, dtype=torch.bool)
        for row, slot in enumerate(snapshot.slots):
            state = self.slots.get(slot)
            if state is None or state.key != snapshot.keys[row]:
                continue
            versions[row] = len(state.history)
            if len(state.history) != snapshot.versions[row]:
                continue
            draft = self.cache.speculate(
                state.key,
                state.history[-self.config.arctic_max_tree_depth :],
                max_spec_tokens=num_drafts,
                max_spec_factor=self.config.arctic_max_spec_factor,
                max_spec_offset=self.config.arctic_max_spec_offset,
                min_token_prob=self.config.arctic_min_token_prob,
                use_tree_spec=False,
            )
            count = len(draft.token_ids)
            if count > num_drafts or draft.parents != list(range(-1, count - 1)):
                raise ValueError("Arctic must return at most K single-chain candidates")
            if count:
                tokens[row, :count] = torch.tensor(draft.token_ids, dtype=torch.int64)
            valid[row] = count
            matches[row] = draft.match_len
            healthy[row] = True
        # Bulk staging on the worker stream precedes graph replay. These copies
        # finish before the CPU tensors go out of scope (non_blocking=False).
        return RetrievalCandidates(
            *(x.to(self.device) for x in (tokens, valid, matches, versions, healthy))
        )
