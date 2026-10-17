"""Global search must preserve serial selection across lanes and strided scans."""

import random

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
    outputs, masks = [], []
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
        outputs.append((tokens + [0] * num_drafts)[:num_drafts])
        masks.append(valid)
    return outputs + [[0] * num_drafts], masks + [False]


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
            expected_tokens, expected_mask = _serial_candidates(
                histories, expected_sources, 4, 2, 13, partial
            )
            for _ in range(2):
                graph.replay()
                assert mask.tolist() == expected_mask
                assert tokens.tolist() == expected_tokens


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
