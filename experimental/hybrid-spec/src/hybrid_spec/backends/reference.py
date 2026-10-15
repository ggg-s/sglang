"""CPU suffix automaton used for differential tests, never enabled in serving."""

from __future__ import annotations

import torch
from hybrid_spec.core.contracts import RetrievalCandidates


class SuffixAutomaton:
    def __init__(self):
        self.history = []
        self.lengths = [0]
        self.links = [-1]
        self.first_pos = [-1]
        self.edges = [{}]
        self.last = 0

    def append(self, tokens):
        for token in tokens:
            self.history.append(token)
            current = len(self.links)
            self.links.append(0)
            self.lengths.append(self.lengths[self.last] + 1)
            self.first_pos.append(len(self.history) - 1)
            self.edges.append({})
            p = self.last
            while p >= 0 and token not in self.edges[p]:
                self.edges[p][token] = current
                p = self.links[p]
            if p >= 0:
                q = self.edges[p][token]
                if self.lengths[p] + 1 == self.lengths[q]:
                    self.links[current] = q
                else:
                    clone = len(self.links)
                    self.links.append(self.links[q])
                    self.lengths.append(self.lengths[p] + 1)
                    self.first_pos.append(self.first_pos[q])
                    self.edges.append(self.edges[q].copy())
                    while p >= 0 and self.edges[p].get(token) == q:
                        self.edges[p][token] = clone
                        p = self.links[p]
                    self.links[q] = self.links[current] = clone
            self.last = current

    def query(self, num_drafts):
        # The suffix link of last is the longest repeated suffix. first_pos
        # chooses its earliest occurrence, giving the longest available tail.
        state = self.links[self.last]
        if state <= 0:
            return [], 0
        start = self.first_pos[state] + 1
        return self.history[start : start + num_drafts], self.lengths[state]


class ReferenceBackend:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.slots = {}

    def reset_slot(self, slot):
        self.slots[slot] = SuffixAutomaton()

    def append(self, slot, delta):
        sa = self.slots[slot]
        if len(sa.history) != delta.start_version:
            raise ValueError("backend history version mismatch")
        sa.append(delta.accept_tokens)

    def query(self, snapshot, num_drafts):
        bs = len(snapshot.slots)
        tokens = torch.zeros((bs, num_drafts), dtype=torch.int64, device=self.device)
        valid = torch.zeros(bs, dtype=torch.int64, device=self.device)
        matches = torch.zeros_like(valid)
        versions = torch.zeros_like(valid)
        healthy = torch.zeros(bs, dtype=torch.bool, device=self.device)
        for row, slot in enumerate(snapshot.slots):
            if slot < 0:
                continue
            sa = self.slots[slot]
            result, match = sa.query(num_drafts)
            tokens[row, : len(result)] = torch.tensor(result, device=self.device)
            valid[row] = len(result)
            matches[row] = match
            versions[row] = len(sa.history)
            healthy[row] = True
        return RetrievalCandidates(tokens, valid, matches, versions, healthy)
