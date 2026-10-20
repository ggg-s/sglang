"""Global search must preserve serial selection across lanes and strided scans."""

import random
from dataclasses import replace

import pytest
import torch
from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
from hybrid_spec.backends.reference import SuffixAutomaton
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import HistoryDelta, RequestKey

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA"
)


def _serial_candidates(histories, sources, num_drafts, min_match, context, partial):
    """Independent text search oracle, including eligibility before ranking."""
    outputs, masks, metadata = [], [], []
    for slot in range(4):
        own = histories[slot]
        sa = SuffixAutomaton()
        sa.append(own)
        own_tokens, own_match = sa.query(num_drafts)
        valid = own_match >= min_match and (
            len(own_tokens) == num_drafts or partial == "zero"
        )
        own_valid = valid
        best = (own_match if valid else 0, 0, 0)
        tokens = own_tokens if valid else []
        shared = False
        suffix = own[-context:]
        for index, donor in enumerate(sources):
            if donor < 0 or donor == slot:
                continue
            history = histories[donor]
            # An end-only occurrence has no continuation: try a shorter suffix.
            for length in range(len(suffix), 0, -1):
                needle = suffix[-length:]
                start = next(
                    (
                        pos + length
                        for pos in range(len(history) - length)
                        if history[pos : pos + length] == needle
                    ),
                    None,
                )
                if start is not None:
                    break
            else:
                continue
            continuation = len(history) - start
            if length < min_match or (continuation < num_drafts and partial != "zero"):
                continue
            # Equal own matches win regardless of continuation or source order.
            if own_valid and length == own_match:
                continue
            rank = (length, continuation, -index)
            if rank > best:
                best = rank
                tokens = history[start : start + num_drafts]
                valid = True
                shared = True
        outputs.append((tokens + [0] * num_drafts)[:num_drafts])
        masks.append(valid)
        metadata.append([best[0], int(shared)] if valid else [0, 0])
    return outputs + [[0] * num_drafts], masks + [False], metadata + [[0, 0]]


@pytest.mark.parametrize("num_sources", [20, 64, 257])
@pytest.mark.parametrize("partial", ["fallback", "zero"])
def test_shared_parallel_matches_text_oracle_in_graph(num_sources, partial):
    """Covers multiple warps, >128 sources, stale donors and publication changes."""
    config = HybridConfig(
        4,
        2,
        4,
        256,
        history_mode="device",
        enable_global_pool=True,
        max_shared_requests=num_sources - 4,
        shared_context_tokens=13,
        partial_drafts=partial,
    )
    backend = GPUSuffixAutomaton(config, "cuda")
    device = {"device": "cuda", "dtype": torch.int64}
    slots = torch.tensor([0, 1, 2, 3, -1], **device)
    generations = torch.tensor([1, 2, 3, 4, -1], **device)
    seq_lens = torch.zeros(5, **device)
    sources = torch.arange(num_sources, **device)
    source_generations = sources + 1
    source_lengths = torch.full_like(sources, -1)
    tokens = torch.zeros((5, 4), **device)
    mask = torch.zeros(5, device="cuda", dtype=torch.bool)
    metadata = torch.zeros((5, 2), **device)

    def query():
        backend.consume_and_query(
            slots,
            generations,
            seq_lens,
            tokens,
            mask,
            shared_slots=sources,
            shared_generations=source_generations,
            shared_lengths=source_lengths,
            query_metadata=metadata,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query()
    for seed in range(6):
        rng = random.Random(8137 + seed)
        histories = [
            [rng.randrange(7) for _ in range(rng.randrange(16, 90))]
            for _ in range(num_sources)
        ]
        # Long matches can be in any lane; some sources only offer partial tails.
        for donor in range(4, num_sources, 3):
            own = histories[donor % 4]
            histories[donor] = (
                histories[donor][:12]
                + own[-rng.randrange(2, 14) :]
                + [1000 + donor] * rng.randrange(1, 9)
            )
        for slot, history in enumerate(histories):
            backend.initialize_device_slot(slot, slot + 1, 256)
            backend.append(
                slot, HistoryDelta(RequestKey(str(slot), slot + 1), 0, tuple(history))
            )
        seq_lens[:4].copy_(torch.tensor([len(h) - 1 for h in histories[:4]], **device))
        for _ in range(2):
            published = list(range(num_sources))
            rng.shuffle(published)
            expected_sources = published.copy()
            source_gens = [s + 1 for s in published]
            # Invalid generation, unhealthy source and padding cannot compete.
            for i in range(0, num_sources, 17):
                if published[i] >= 4:
                    source_gens[i] += num_sources
                    expected_sources[i] = -1
            for i in range(5, num_sources, 23):
                if published[i] >= 4:
                    backend.meta[published[i], 4] = 0
            healthy = backend.meta[:, 4].tolist()
            expected_sources = [
                s if s < 0 or healthy[s] else -1 for s in expected_sources
            ]
            sources.copy_(torch.tensor(published, **device))
            source_generations.copy_(torch.tensor(source_gens, **device))
            expected_tokens, expected_mask, expected_metadata = _serial_candidates(
                histories, expected_sources, 4, 2, 13, partial
            )
            for _ in range(2):
                graph.replay()
                assert mask.tolist() == expected_mask
                assert tokens.tolist() == expected_tokens
                assert metadata.tolist() == expected_metadata


def test_shared_parallel_ties_across_lanes_and_strides():
    """Equal tails must use publication order, even across a strided scan."""
    config = HybridConfig(
        4,
        2,
        1,
        64,
        history_mode="device",
        enable_global_pool=True,
        max_shared_requests=257,
    )
    backend = GPUSuffixAutomaton(config, "cuda")
    device = {"device": "cuda", "dtype": torch.int64}
    histories = {
        0: [9, 1, 2],
        1: [1, 2, 11, 12, 13, 14],
        2: [1, 2, 21, 22, 23, 24],
        3: [1, 2, 31, 32, 33, 34],
    }
    for slot, history in histories.items():
        backend.initialize_device_slot(slot, slot + 1, 100)
        backend.append(
            slot, HistoryDelta(RequestKey(str(slot), slot + 1), 0, tuple(history))
        )
    sources = torch.full((258,), -1, **device)
    generations = torch.zeros_like(sources)
    tokens = torch.zeros((1, 4), **device)
    mask = torch.zeros(1, device="cuda", dtype=torch.bool)
    for positions in ((16, 128, 256), (128, 16, 256), (256, 128, 16)):
        sources.fill_(-1)
        generations.zero_()
        for donor, position in enumerate(positions, 1):
            sources[position] = donor
            generations[position] = donor + 1
        backend.consume_and_query(
            torch.tensor([0], **device),
            torch.tensor([1], **device),
            torch.tensor([2], **device),
            tokens,
            mask,
            shared_slots=sources,
            shared_generations=generations,
            shared_lengths=torch.full_like(sources, -1),
        )
        winner = positions.index(min(positions)) + 1
        assert mask.item()
        assert tokens.tolist() == [histories[winner][2:]]


def test_unique_continuation_local_branch_and_graph_reuse():
    """Matching the first next token cannot hide a branch at token two."""
    backend = GPUSuffixAutomaton(
        HybridConfig(3, 2, 1, 128, history_mode="device"), "cuda"
    )
    device = {"device": "cuda", "dtype": torch.int64}
    slots, generations = torch.tensor([0], **device), torch.tensor([1], **device)
    lengths = torch.zeros(1, **device)
    tokens = torch.zeros((1, 3), **device)
    mask = torch.zeros(1, device="cuda", dtype=torch.bool)
    # Same matched suffix, same first candidate, differing second candidate.
    ambiguous = [1, 2, 10, 11, 12, 90, 1, 2, 10, 21, 22, 91, 1, 2]
    unique = [1, 2, 10, 11, 12, 90, 1, 2]
    backend.config = replace(backend.config, sa_require_unique_continuation=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.consume_and_query(slots, generations, lengths, tokens, mask)
    for history, expected in ((ambiguous, False), (unique, True), (ambiguous, False)):
        backend.initialize_device_slot(0, 1, 128)
        backend.append(0, HistoryDelta(RequestKey("local", 1), 0, tuple(history)))
        lengths.fill_(len(history) - 1)
        for _ in range(2):
            graph.replay()
            assert mask.item() == expected
        backend.config = replace(backend.config, sa_require_unique_continuation=False)
        backend.consume_and_query(slots, generations, lengths, tokens, mask)
        assert mask.item() and tokens.tolist() == [[10, 11, 12]]
        backend.config = replace(backend.config, sa_require_unique_continuation=True)


@pytest.mark.parametrize("own_match", [False, True])
def test_unique_continuation_pool_conflicts_and_specificity(own_match):
    """Pool ties, strided scans and stale donors cannot bypass ambiguity veto."""
    backend = GPUSuffixAutomaton(
        HybridConfig(
            3,
            2,
            1,
            128,
            history_mode="device",
            enable_global_pool=True,
            max_shared_requests=257,
            sa_require_unique_continuation=True,
        ),
        "cuda",
    )
    device = {"device": "cuda", "dtype": torch.int64}
    own = [1, 2, 10, 11, 12, 90, 1, 2] if own_match else [90, 1, 2]
    histories = {
        0: own,
        1: [1, 2, 10, 11, 12],
        2: [1, 2, 10, 21, 22],
        3: [90, 1, 2, 30, 31, 32],  # Longer, more specific context wins.
    }
    for slot, history in histories.items():
        backend.initialize_device_slot(slot, slot + 1, 128)
        backend.append(
            slot, HistoryDelta(RequestKey(str(slot), slot + 1), 0, tuple(history))
        )
    slots = torch.tensor([0, -1], **device)
    generations = torch.tensor([1, -1], **device)
    lengths = torch.tensor([len(own) - 1, 0], **device)
    sources = torch.full((258,), -1, **device)
    source_generations = torch.zeros_like(sources)
    source_lengths = torch.full_like(sources, -1)
    tokens = torch.zeros((2, 3), **device)
    mask = torch.zeros(2, device="cuda", dtype=torch.bool)

    def query():
        backend.consume_and_query(
            slots,
            generations,
            lengths,
            tokens,
            mask,
            shared_slots=sources,
            shared_generations=source_generations,
            shared_lengths=source_lengths,
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query()
    for left, right in ((16, 128), (128, 16), (16, 256)):
        sources.fill_(-1)
        source_generations.zero_()
        sources[left], source_generations[left] = 1, 2
        sources[right], source_generations[right] = 2, 3
        graph.replay()
        assert mask.tolist() == [False, False]
        # Invalidating the conflicting generation must remove its veto.
        source_generations[right] = 99
        graph.replay()
        assert mask.tolist() == [True, False]
        assert tokens[0].tolist() == [10, 11, 12]
        # A more specific context must beat both ambiguous shorter donors.
        source_generations[right] = 3
        sources[64], source_generations[64] = 3, 4
        graph.replay()
        assert mask.tolist() == [True, False]
        assert tokens[0].tolist() == [30, 31, 32]
