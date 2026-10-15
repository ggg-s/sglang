"""Requires NVIDIA CUDA and nvcc; never silently counted as a CPU pass."""

import itertools

import pytest
import torch
from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
from hybrid_spec.backends.reference import SuffixAutomaton
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import (
    DraftOverrideView,
    HistoryDelta,
    RequestKey,
    RequestSnapshot,
)
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA"
)


def test_cuda_sa_differential_capacity_reorder_and_slot_reuse():
    config = HybridConfig(3, 1, 16, 128)
    backend = GPUSuffixAutomaton(config, "cuda")
    # Include clone-heavy ternary histories and very long single-token suffixes.
    histories = [
        list(s) * 3 for s in itertools.islice(itertools.product(range(3), repeat=5), 15)
    ]
    histories.append([1] * 128)
    for slot, history in enumerate(histories):
        backend.reset_slot(slot)
        key = RequestKey(str(slot), 1)
        for start in range(0, len(history), 7):
            backend.append(
                slot, HistoryDelta(key, start, tuple(history[start : start + 7]))
            )
    slots = list(reversed(range(16))) + [-1]
    snapshot = RequestSnapshot((), tuple(slots), (), ())
    result = backend.query(snapshot, 3)
    torch.cuda.synchronize()
    for row, slot in enumerate(slots):
        if slot < 0:
            assert not result.healthy[row].item()
            continue
        oracle = SuffixAutomaton()
        oracle.append(histories[slot])
        tokens, match = oracle.query(3)
        assert result.healthy[row].item()
        assert result.versions[row].item() == len(histories[slot])
        assert result.match_lens[row].item() == match
        assert result.num_valid_drafts[row].item() == len(tokens)
        assert result.tokens[row, : len(tokens)].tolist() == tokens
    backend.reset_slot(0)
    backend.append(0, HistoryDelta(RequestKey("new", 2), 0, (9, 8, 9)))
    result = backend.query(RequestSnapshot((), (0,), (), ()), 3)
    assert result.tokens[0, :2].tolist() == [8, 9]
    # Stale updates invalidate the slot; they must never append silently.
    backend.append(0, HistoryDelta(RequestKey("new", 2), 0, (7,)))
    assert not backend.query(RequestSnapshot((), (0,), (), ()), 3).healthy.item()


@pytest.mark.parametrize("k", [1, 4])
def test_cuda_graph_replay_reads_live_mask_and_padding(k):
    buffers = GraphBuffers(4, k, "cuda")
    view = buffers.bind_capture_view(4)
    neural = torch.full((4, 1), 99, dtype=torch.int64, device="cuda")
    # Warm up on a side stream, then capture while all masks are false.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = torch.cat(
                [view.select_step(neural, step) for step in range(k)], dim=1
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = torch.cat([view.select_step(neural, step) for step in range(k)], dim=1)
    for mask in ([True] * 4, [False] * 4, [True, False, True, False]):
        values = torch.arange(4 * k, dtype=torch.int64, device="cuda").view(4, k)
        mask_tensor = torch.tensor(mask, device="cuda")
        buffers.stage(DraftOverrideView(values, mask_tensor), 4, 4)
        graph.replay()
        assert torch.equal(output, torch.where(mask_tensor[:, None], values, neural))
    buffers.stage(DraftOverrideView(values[:1], mask_tensor[:1]), 1, 4)
    graph.replay()
    assert torch.equal(output[1:], neural[1:].expand(-1, k))
