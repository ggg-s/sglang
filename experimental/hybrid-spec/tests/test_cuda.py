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
from hybrid_spec.integrations.sglang.device_extension import (
    DeviceGraphBuffers,
    DeviceOverride,
)
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers
from hybrid_spec.observability.metrics import Metrics

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


@pytest.mark.parametrize("partial", ["fallback", "zero"])
def test_device_feedback_graph_reorder_generation_and_replay(partial):
    config = HybridConfig(3, 2, 2, 64, history_mode="device", partial_drafts=partial)
    backend = GPUSuffixAutomaton(config, "cuda")
    # Slot 0 has only two continuation tokens. Slot 1 has a complete chain.
    histories = [[1, 2, 1], [4, 5, 6, 7, 4]]
    for slot, history in enumerate(histories):
        backend.initialize_device_slot(slot, slot + 10, 64)
        backend.append(
            slot, HistoryDelta(RequestKey(str(slot), slot + 10), 0, tuple(history))
        )
    buffers = DeviceGraphBuffers(backend, 4)
    capture_view = buffers.bind_capture_view(4)
    seq_lens = torch.ones(4, dtype=torch.int64, device="cuda")
    # Capturing/warming empty slots must not mutate live automata.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        capture_view.prepare(seq_lens)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capture_view.prepare(seq_lens)

    def stage(slots, generations, lengths):
        view = DeviceOverride(
            backend,
            torch.tensor(slots, device="cuda"),
            torch.tensor(generations, device="cuda"),
        )
        buffers.stage(view, len(slots), 4)
        seq_lens[: len(slots)].copy_(torch.tensor(lengths, device="cuda"))
        return view

    # Prefill bonus is appended exactly once inside the replay, not on the host.
    slots = torch.tensor([0, 1], device="cuda")
    generations = torch.tensor([10, 11], device="cuda")
    backend.stage_feedback(
        slots,
        generations,
        torch.tensor([2, 5], dtype=torch.int32, device="cuda"),
        torch.tensor([[0], [1]], dtype=torch.int32, device="cuda"),
        torch.ones(2, dtype=torch.int32, device="cuda"),
        torch.tensor([3, 5], device="cuda"),
        False,
        100,
    )
    view = stage([1, 0], [11, 10], [5, 3])
    graph.replay()
    assert view.use_retrieval.tolist() == [True, partial == "zero"]
    assert capture_view.tokens[0].tolist() == [6, 7, 4]
    if partial == "zero":
        assert capture_view.tokens[1].tolist() == [1, 2, 0]
    assert not capture_view.use_retrieval[2:].any()
    versions = backend.meta[:, 0].clone()
    graph.replay()
    assert torch.equal(backend.meta[:, 0], versions)

    # Verify indices are global and may be non-contiguous; -1 padding is unread.
    backend.stage_feedback(
        slots[:1],
        generations[:1],
        torch.tensor([8, 9, 3, 4], dtype=torch.int32, device="cuda"),
        torch.tensor([[2, 0, -1, -1]], dtype=torch.int32, device="cuda"),
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        torch.tensor([3], device="cuda"),
        True,
        100,
    )
    stage([0], [10], [5])
    graph.replay()
    assert backend.history[0, :6].tolist() == [1, 2, 1, 2, 3, 8]

    # Old in-flight feedback/query must never touch a reassigned slot.
    backend.initialize_device_slot(0, 20, 64)
    backend.append(0, HistoryDelta(RequestKey("replacement", 20), 0, (9, 8, 9)))
    backend.stage_feedback(
        slots[:1],
        generations[:1],
        torch.tensor([7], dtype=torch.int32, device="cuda"),
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        torch.tensor([3], device="cuda"),
        True,
        100,
    )
    stage([0], [10], [3])
    graph.replay()
    assert not capture_view.use_retrieval.any()
    assert backend.meta[0, 0].item() == 3
    assert backend.meta[0, 4].item() == 1


def test_host_prefill_import_preserves_gpu_bonus_feedback():
    config = HybridConfig(3, 2, 1, 64, history_mode="device")
    backend = GPUSuffixAutomaton(config, "cuda")
    prompt = (4, 5, 6, 7, 4)
    host = backend.build_host(backend.prepare_host(prompt), 100)
    assert all(
        t.is_pinned() for t in (host.history, host.states, host.edges, host.meta)
    )
    backend.initialize_device_slot(0, 9, 64)
    slots = torch.tensor([0], device="cuda")
    generations = torch.tensor([9], device="cuda")
    seq_lens = torch.tensor([len(prompt)], device="cuda")
    backend.stage_feedback(
        slots,
        generations,
        torch.tensor([5], dtype=torch.int32, device="cuda"),
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.ones(1, dtype=torch.int32, device="cuda"),
        seq_lens,
        False,
        100,
    )
    backend.import_host(0, host)
    view = DeviceOverride(backend, slots, generations)
    view.prepare(seq_lens)
    assert view.use_retrieval.tolist() == [True]
    assert view.tokens.tolist() == [[6, 7, 4]]
    assert backend.history[0, :6].tolist() == [4, 5, 6, 7, 4, 5]


def test_feedback_fused_metrics_includes_neural_rows_without_slots():
    """Fusing statistics must retain rows rejected by SA slot/generation checks."""
    config = HybridConfig(3, 2, 2, 64, history_mode="device")
    backend = GPUSuffixAutomaton(config, "cuda")
    backend.initialize_device_slot(0, 10, 64)
    backend.initialize_device_slot(1, 11, 64)
    slots = torch.tensor([0, 1, -1], device="cuda")
    generations = torch.tensor([10, 11, -1], device="cuda")
    predict = torch.arange(12, dtype=torch.int32, device="cuda")
    indices = predict.reshape(3, 4)
    seq_lens = torch.full((3,), 5, device="cuda")
    metrics = Metrics()
    counts = metrics.ensure_buffer("cuda")
    reference = Metrics()
    for mask, lengths in (
        ([False, True, False], [1, 2, 4]),
        ([True, False, False], [3, 1, 2]),
    ):
        backend.pending_meta.zero_()
        mask = torch.tensor(mask, device="cuda")
        lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
        backend.stage_feedback(
            slots,
            generations,
            predict,
            indices,
            lengths,
            seq_lens,
            True,
            100,
            use_retrieval=mask,
            counts=counts,
        )
        reference.record(mask, lengths)
        assert metrics.snapshot() == reference.snapshot()
    assert counts.tolist() == [2, 4, 3, 4, 5, 8]
    # A stale generation still contributes to neural stats, without modifying
    # the replacement request's feedback. Prefill must not count as a verify.
    pending = backend.pending_meta.clone()
    generations[0] = 99
    mask.zero_()
    backend.stage_feedback(
        slots,
        generations,
        predict,
        indices,
        lengths,
        seq_lens,
        True,
        100,
        use_retrieval=mask,
        counts=counts,
    )
    reference.record(mask, lengths)
    assert torch.equal(backend.pending_meta[0], pending[0])
    assert metrics.snapshot() == reference.snapshot()
    backend.stage_feedback(
        slots, generations, predict, indices, lengths, seq_lens, False, 100
    )
    assert metrics.snapshot() == reference.snapshot()
