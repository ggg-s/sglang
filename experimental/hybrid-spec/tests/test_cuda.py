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
    DeviceDraftExtension,
    DeviceGraphBuffers,
    DeviceOverride,
)
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers
from hybrid_spec.observability.metrics import Metrics

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires NVIDIA CUDA"
)


@pytest.mark.parametrize("dynamic", [False, True])
def test_split_query_stream_graph_feedback_publication_and_gate(monkeypatch, dynamic):
    """A delayed native query must see fresh feedback and join before gating.

    Publish a donor after capture, mutate the gate between replays, then fence
    its generation. The auxiliary stream executes actual SA kernels.
    """
    config = HybridConfig(
        8,
        2,
        1,
        128,
        history_mode="device",
        draft_graph_mode="split",
        num_neural_gate_tokens=2,
        enable_global_pool=True,
        max_shared_requests=2,
        verify_mode="compact" if dynamic else "static",
        dynamic_verify=dynamic,
    )
    backend = GPUSuffixAutomaton(config, "cuda")
    for slot, prompt in enumerate(([99, 1], list(range(1, 11)))):
        backend.initialize_device_slot(slot, 10 + slot, 128)
        backend.import_host(slot, backend.build_host(backend.prepare_host(prompt), 200))
    native_query = backend.consume_and_query

    def delayed_query(*args, **kwargs):
        torch.cuda._sleep(1_000_000)
        native_query(*args, **kwargs)

    monkeypatch.setattr(backend, "consume_and_query", delayed_query)
    query_stream = torch.cuda.Stream()
    buffers = DeviceGraphBuffers(backend, 1, query_stream=query_stream)
    captured = buffers.bind_capture_view(1)
    slots = torch.tensor([0], dtype=torch.int64, device="cuda")
    generations = torch.tensor([10], dtype=torch.int64, device="cuda")
    view = DeviceOverride(backend, slots, generations)
    buffers.stage(view, 1, 1)
    seq_lens = torch.tensor([2], dtype=torch.int64, device="cuda")
    neural = torch.tensor([[3, 4, 80, 81, 82, 83, 84, 85]], device="cuda")

    def run():
        captured.prepare(seq_lens)
        return torch.cat(
            [captured.select_step(neural[:, i : i + 1] + 0, i) for i in range(8)],
            dim=1,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    # Feed the anchor and publish a completed donor after capture.
    backend.stage_feedback(
        slots,
        generations,
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        seq_lens,
        False,
        200,
    )
    view.shared_slots[0] = 1
    view.shared_generations[0] = 11
    view.shared_lengths[0] = 10
    buffers.stage(view, 1, 1)
    for disagree in (False, True, False):
        neural[0, 1] = 70 if disagree else 4
        graph.replay()
        assert view.use_retrieval.tolist() == [not disagree]
        expected = neural if disagree else torch.arange(3, 11, device="cuda")[None, :]
        assert torch.equal(output, expected)
        assert backend.meta[0, 0].item() == 3  # feedback is consumed exactly once
        if dynamic:
            assert view.query_metadata.tolist() == [[2, 1]]
            assert view.query_metadata.data_ptr() == captured.query_metadata.data_ptr()
    backend.owners[1].add_(1)
    graph.replay()
    assert not view.use_retrieval.any().item()
    assert torch.equal(output, neural)
    if dynamic:
        assert view.query_metadata.tolist() == [[0, 0]]


@pytest.mark.parametrize("partial", ["fallback", "zero"])
def test_shared_history_graph_final_feedback_boundaries_and_generation(partial):
    """Final verify tokens must survive finish; stale/reused donors must not."""
    config = HybridConfig(
        2,
        2,
        1,
        64,
        history_mode="device",
        max_shared_requests=2,
        partial_drafts=partial,
        enable_global_pool=True,
    )
    backend = GPUSuffixAutomaton(config, "cuda")

    def install(slot, generation, history):
        backend.initialize_device_slot(slot, generation, 100)
        backend.append(
            slot, HistoryDelta(RequestKey(str(slot), generation), 0, tuple(history))
        )

    install(0, 10, [1, 2])
    install(1, 11, [5, 6, 7, 8])
    install(2, 12, [9, 1, 2])
    final = [3, 4] if partial == "fallback" else [3]
    device = {"device": "cuda", "dtype": torch.int64}
    backend.stage_feedback(
        torch.tensor([0], **device),
        torch.tensor([10], **device),
        torch.tensor(final, device="cuda", dtype=torch.int32),
        torch.tensor([list(range(len(final)))], device="cuda", dtype=torch.int32),
        torch.tensor([len(final)], device="cuda", dtype=torch.int32),
        torch.tensor([1], **device),
        True,
        100,
    )
    view = DeviceOverride(
        backend, torch.tensor([2], **device), torch.tensor([12], **device)
    )
    buffers = DeviceGraphBuffers(backend, 2)
    captured = buffers.bind_capture_view(2)
    buffers.stage(view, 1, 2)
    seq_lens = torch.tensor([2, 0], **device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured.prepare(seq_lens)
    # Publication occurs after capture, with an unchanged active batch.
    assert backend.pending_meta[0, 2].item() == len(final)
    view.shared_slots.copy_(torch.tensor([0, 1, -1], **device))
    view.shared_generations.copy_(torch.tensor([10, 11, 0], **device))
    view.shared_lengths.copy_(torch.tensor([2 + len(final), 4, -1], **device))
    buffers.stage(view, 1, 2)
    for _ in range(2):
        graph.replay()
        assert captured.tokens[0].tolist() == (final + [0])[:2]
        assert captured.use_retrieval.tolist() == [True, False]
        assert backend.meta[0, 0].item() == 2 + len(final)
        assert backend.pending_meta[0, 2].item() == 0
    # A suffix spanning two donor documents must never be synthesized.
    install(2, 12, [4, 5])
    seq_lens[0] = 1
    graph.replay()
    assert not captured.use_retrieval.any().item()
    install(2, 12, [9, 1, 2])
    seq_lens[0] = 2
    # A slot reused after FIFO eviction is fenced even if an old mapping replays.
    install(0, 20, [1, 2, 8, 9])
    graph.replay()
    assert not captured.use_retrieval.any().item()
    # Publishing the new generation makes it eligible, without recapture.
    view.shared_generations[0] = 20
    view.shared_lengths[0] = 4
    buffers.stage(view, 1, 2)
    graph.replay()
    assert captured.tokens[0].tolist() == [8, 9]
    assert captured.use_retrieval.tolist() == [True, False]
    # Among foreign sources, a longer continuation beats publication order.
    install(1, 11, [1, 2, 6, 7, 8, 9])
    view.shared_lengths[1] = 6
    buffers.stage(view, 1, 2)
    graph.replay()
    assert captured.tokens[0].tolist() == [6, 7]
    # Longest foreign match [5,6,1,2] ends at the donor's boundary; the
    # earlier [1,2] occurrence still supplies a valid [3,4] continuation.
    install(1, 11, [1, 2, 3, 4, 5, 6, 1, 2])
    install(2, 12, [9, 5, 6, 1, 2])
    seq_lens[0] = 4
    view.shared_lengths[1] = 8
    buffers.stage(view, 1, 2)
    graph.replay()
    assert captured.tokens[0].tolist() == [3, 4]
    view.shared_slots.fill_(-1)
    buffers.stage(view, 1, 2)
    graph.replay()
    assert not captured.use_retrieval.any().item()


def test_global_pool_reads_other_active_rows_after_all_updates():
    """A row must see another row's current feedback, independent of row order."""
    config = HybridConfig(2, 2, 2, 64, history_mode="device", enable_global_pool=True)
    backend = GPUSuffixAutomaton(config, "cuda")
    device = {"device": "cuda", "dtype": torch.int64}
    for slot, history in enumerate(([1, 2], [9, 1, 2])):
        backend.initialize_device_slot(slot, slot + 1, 100)
        backend.append(
            slot, HistoryDelta(RequestKey(str(slot), slot + 1), 0, tuple(history))
        )
    backend.stage_feedback(
        torch.tensor([0], **device),
        torch.tensor([1], **device),
        torch.tensor([3, 4], device="cuda", dtype=torch.int32),
        torch.tensor([[0, 1]], device="cuda", dtype=torch.int32),
        torch.tensor([2], device="cuda", dtype=torch.int32),
        torch.tensor([1], **device),
        True,
        100,
    )
    view = DeviceOverride(
        backend, torch.tensor([1, 0, -1], **device), torch.tensor([2, 1, -1], **device)
    )
    view.shared_slots.copy_(torch.tensor([0, 1], **device))
    view.shared_generations.copy_(torch.tensor([1, 2], **device))
    seq_lens = torch.tensor([2, 3, 0], **device)
    for _ in range(2):
        view.prepare(seq_lens)
        assert view.tokens[0].tolist() == [3, 4]
        assert view.use_retrieval.tolist() == [True, False, False]
        assert backend.meta[0, 0].item() == 4
    # Local history wins equal-length matches even with a longer foreign tail.
    backend.initialize_device_slot(1, 2, 100)
    backend.append(1, HistoryDelta(RequestKey("1", 2), 0, (1, 2, 7, 8, 1, 2)))
    seq_lens[0] = 5
    view.prepare(seq_lens)
    assert view.tokens[0].tolist() == [7, 8]


def test_completed_pool_trims_uncommitted_stop_tail():
    """CPU stop truncation may discard part of the last target-verified delta."""
    config = HybridConfig(
        2,
        2,
        1,
        64,
        history_mode="device",
        enable_global_pool=True,
        max_shared_requests=1,
        partial_drafts="zero",
    )
    backend = GPUSuffixAutomaton(config, "cuda")
    device = {"device": "cuda", "dtype": torch.int64}
    for slot, history in enumerate(([1, 2], [9, 1, 2])):
        backend.initialize_device_slot(slot, slot + 1, 100)
        backend.append(
            slot, HistoryDelta(RequestKey(str(slot), slot + 1), 0, tuple(history))
        )
    backend.stage_feedback(
        torch.tensor([0], **device),
        torch.tensor([1], **device),
        torch.tensor([3, 99, 98], device="cuda", dtype=torch.int32),
        torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32),
        torch.tensor([3], device="cuda", dtype=torch.int32),
        torch.tensor([1], **device),
        True,
        100,
    )
    view = DeviceOverride(
        backend, torch.tensor([1], **device), torch.tensor([2], **device)
    )
    view.shared_slots.copy_(torch.tensor([0, -1], **device))
    view.shared_generations.copy_(torch.tensor([1, 0], **device))
    view.shared_lengths.copy_(torch.tensor([3, -1], **device))
    seq_lens = torch.tensor([2], **device)
    view.prepare(seq_lens)
    assert view.use_retrieval.item()
    assert view.tokens.tolist() == [[3, 0]]
    assert backend.meta[0, 0].item() == 3
    # A pre-existing discarded tail cannot be truncated in-place in an SA.
    backend.append(0, HistoryDelta(RequestKey("0", 1), 3, (99,)))
    view.prepare(seq_lens)
    assert not view.use_retrieval.item()
    assert not backend.meta[0, 4].item()


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


def test_cuda_graph_neural_gate_replay_resets_final_selection_mask():
    """Graph replay must not reuse an earlier batch's rejected retrieval mask."""
    buffers = GraphBuffers(3, 3, "cuda", num_neural_gate_tokens=2)
    view = buffers.bind_capture_view(3)
    neural = [
        torch.full((3, 1), t, device="cuda", dtype=torch.int64) for t in (10, 20, 30)
    ]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = torch.cat(
            [view.select_step(t, i) for i, t in enumerate(neural)], dim=1
        )
    candidates = torch.tensor([[10, 20, 99], [11, 20, 98], [10, 21, 97]], device="cuda")
    eligibility = torch.ones(3, dtype=torch.bool, device="cuda")
    buffers.stage(DraftOverrideView(candidates, eligibility), 3, 3)
    graph.replay()
    assert output.tolist() == [[10, 20, 99], [10, 20, 30], [10, 20, 30]]
    assert view.use_retrieval.tolist() == [True, False, False]
    candidates[:, :2] = torch.tensor([10, 20], device="cuda")
    buffers.stage(DraftOverrideView(candidates, eligibility), 3, 3)
    graph.replay()
    assert torch.equal(output, candidates)
    assert view.use_retrieval.tolist() == [True, True, True]


def native_triton_metadata_methods(monkeypatch):
    """Load unchanged multi-step planning and its real Triton KV index kernel.

    Child attention execution and the model remain synthetic. This exercises
    the native request/step read-table math inside each compact tail graph.
    """
    import importlib.util
    import sys
    from types import ModuleType
    from types import SimpleNamespace as NS
    from typing import Optional

    import triton
    from test_draft_recurrence import ROOT, load_functions

    utils = ModuleType("sglang.srt.utils")
    for name in ("cpu", "hip", "musa", "npu", "xpu"):
        setattr(utils, "is_" + name, lambda: False)
    utils.is_cuda = lambda: True
    utils.next_power_of_2 = triton.next_power_of_2
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    name = "sglang.kernels.ops.speculative.cache_locs"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "python/sglang/kernels/ops/speculative/cache_locs.py"
    )
    kernels = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, kernels)
    spec.loader.exec_module(kernels)
    ns = dict(
        torch=torch,
        ForwardBatch=object,
        ForwardMode=NS(DECODE=object()),
        Optional=Optional,
        next_power_of_2=triton.next_power_of_2,
        generate_draft_decode_kv_indices=kernels.generate_draft_decode_kv_indices,
    )
    load_functions(
        "python/sglang/srt/speculative/spec_utils.py",
        ["draft_kv_indices_buffer_width", "draft_kv_indices_used_len"],
        ns,
    )
    load_functions(
        "python/sglang/srt/model_executor/forward_batch_info.py",
        ["build_inner_fb_view"],
        ns,
    )
    module = ModuleType("sglang.srt.model_executor.forward_batch_info")
    module.build_inner_fb_view = ns["build_inner_fb_view"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    names = [
        "common_template",
        "init_cuda_graph_state",
        "init_forward_metadata_out_graph",
        "init_forward_metadata_in_graph",
    ]
    load_functions(
        "python/sglang/srt/layers/attention/triton_backend.py",
        names,
        ns,
        "TritonMultiStepDraftBackend",
    )
    return {name: ns[name] for name in names}


@pytest.mark.parametrize(
    "kind",
    [
        "triton",
        "dsv4",
        "eagle3",
        "eagle3_hot",
        "standalone",
        "arctic_triton",
        "arctic_dsv4",
        "arctic_eagle3_hot",
        "arctic_standalone",
    ],
)
@pytest.mark.parametrize(
    "steps,gate,batch_size,neural,mode,conditional",
    [
        (k, g, b, None, mode, False)
        for k, g in (
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (2, 2),
            (3, 3),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 8),
        )
        for b in (1, 8)
        for mode in ("split", "split_no_skip")
    ]
    + [(8, 3, 16, None, mode, False) for mode in ("split", "split_no_skip")]
    + [
        (8, 3, 8, 2, "split", False),
        (8, 1, 8, 3, "split", False),
        (8, 8, 8, 1, "split", False),
        (8, 3, 1, 2, "split", False),
        pytest.param(8, 0, 1, None, "split", True, id="conditional-b1"),
        pytest.param(8, 1, 8, None, "split", True, id="conditional-b8"),
        pytest.param(8, 2, 8, 3, "split", True, id="conditional-source-caps"),
        pytest.param(8, 8, 8, None, "split", True, id="conditional-no-tail"),
        pytest.param(8, 0, 1, None, "split", "switch", id="switch-b1"),
        pytest.param(8, 1, 8, None, "split", "switch", id="switch-b8"),
        pytest.param(8, 2, 3, 3, "split", "switch", id="switch-source-caps-b3"),
        pytest.param(8, 8, 8, None, "split", "switch", id="switch-no-tail"),
    ],
)
def test_compact_tail_graph_reorders_rows_without_computing_sa_requests(
    monkeypatch, mode, kind, steps, gate, batch_size, neural, conditional
):
    """Run the actual worker body, topk, compaction and CUDA graph scheduling.

    Model and child attention are synthetic. EAGLE3/STANDALONE use the native
    multi-step metadata methods and KV index kernel. Recurrence, prefix boundary,
    GPU row selection, original-row KV mapping and private pools are real.
    """
    import contextlib
    from dataclasses import replace
    from types import SimpleNamespace as NS
    from typing import List, Optional

    from hybrid_spec.integrations.sglang.split_graph import SplitDraftGraphBackend
    from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
    from sglang.srt.layers.attention import triton_backend
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
        get_attn_backend,
    )
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )
    from test_draft_recurrence import load_functions

    if conditional:
        from hybrid_spec.integrations.sglang.conditional_graph import (
            require_conditional_graph,
        )

        require_conditional_graph("switch" if conditional == "switch" else "if")

    arctic = kind.startswith("arctic_")
    kind = kind.removeprefix("arctic_")
    native_metadata = kind in ("eagle3", "eagle3_hot", "standalone")

    class Child:
        max_kv_splits = 1

        def init_cuda_graph_state(self, *args, **kwargs):
            self.cuda_graph_num_kv_splits = kwargs["cuda_graph_num_kv_splits_buf"]

        def init_forward_metadata_out_graph(self, batch, in_capture):
            self.indptr = batch.spec_info.kv_indptr
            self.indices = batch.spec_info.kv_indices

        def init_forward_metadata_in_graph(self, batch):
            pass

        def get_num_kv_splits(self, output, seq_lens):
            output.fill_(1)

    class Attention:
        def __init__(self, model=None, topk=1, speculative_num_steps=steps):
            self.attn_backends = [None] * steps
            if native_metadata:
                self.attn_backends = [Child() for _ in range(steps - 1)]
                self.topk = topk
                self.speculative_num_steps = steps
                self.device = "cuda"
                self.max_context_len = 256
                self.kv_indptr = torch.zeros(
                    (steps, batch_size + 1), dtype=torch.int32, device="cuda"
                )
                self.kv_index_translator = translator
                self.page_size = 1
                self.draft_window_size = self.draft_sink_size = 0

        def init_cuda_graph_state(self, *args):
            pass

        def init_forward_metadata_out_graph(self, *args, **kwargs):
            pass

        def init_forward_metadata_in_graph(self, *args):
            pass

    if native_metadata:
        for name, method in native_triton_metadata_methods(monkeypatch).items():
            setattr(Attention, name, method)
    monkeypatch.setattr(triton_backend, "TritonMultiStepDraftBackend", Attention)

    class Dsv4Child:
        def __init__(self, step):
            self.step = step

        def init_forward_metadata_out_graph(self, batch, in_capture):
            assert batch.seq_lens_cpu is None
            assert batch.out_cache_loc.shape == (batch.batch_size,)
            if in_capture:
                self.req = torch.empty_like(batch.req_pool_indices)
                self.seq = torch.empty_like(batch.seq_lens)
                self.raw = torch.empty_like(batch.out_cache_loc)
                self.swa = torch.empty_like(self.raw)
            self.req.copy_(batch.req_pool_indices)
            self.seq.copy_(batch.seq_lens)
            self.raw.copy_(batch.out_cache_loc)

        def init_forward_metadata_in_graph(self, batch):
            assert batch.out_cache_loc.shape == (batch.batch_size * steps,)
            self.swa.copy_(
                batch.out_cache_loc.view(batch.batch_size, steps)[:, self.step]
            )

    class Dsv4Attention(Attention):
        decode_attention_backend_str = "dsv4"

        def __init__(self, *args):
            self.attn_backends = [Dsv4Child(i) for i in range(steps)]

        def on_after_cuda_graph_warmup(self):
            pass

    class DerivedDsv4Attention(Dsv4Attention):
        pass

    device = "cuda"
    width = max(1, gate)
    prefix_steps = width - 1
    tail_steps = max(0, (neural or steps) - width)
    standalone = kind == "standalone"
    eagle3 = kind.startswith("eagle3")
    hot_tokens = (
        torch.arange(128, device=device).flip(0) * 2 + 1
        if kind == "eagle3_hot"
        else None
    )
    hidden_width = 4 if kind == "dsv4" or eagle3 else 1
    # EAGLE3 can begin with concatenated target features, then recur with
    # the draft hidden width. STANDALONE carries no recurrent features.
    seed_width = 3 * hidden_width if eagle3 else hidden_width
    seed = torch.arange(batch_size, device=device) * 2 + 2
    initial_hidden = (
        None
        if standalone
        else (seed[:, None] - 1).float().expand(-1, seed_width).clone()
    )
    base_positions = torch.arange(batch_size, device=device) * 7 + 11
    reqs = torch.arange(batch_size, device=device).roll(1).flip(0)
    cache = reqs[None, :] * 100 + torch.arange(steps, device=device)[:, None] + 40
    flat_cache = cache.T.contiguous().flatten()
    table = (
        torch.arange(batch_size, device=device)[:, None] * 1000
        + torch.arange(256, device=device)[None, :]
    )
    for row in range(batch_size):
        table[reqs[row], base_positions[row] + torch.arange(steps, device=device)] = (
            cache[:, row]
        )

    def read_source(plan, req_pool_indices, bs):
        assert plan.rows is req_pool_indices
        return NS(
            row_ids=req_pool_indices,
            ids=table,
            row_stride=256,
            entry_page_size=1,
            v2p=None,
        )

    translator = NS(is_translating=False, read_source=read_source)
    work = torch.zeros(batch_size, dtype=torch.int64, device=device)
    eligible = torch.ones(batch_size, dtype=torch.bool, device=device)
    candidates = torch.zeros((batch_size, steps), dtype=torch.int64, device=device)
    config = HybridConfig(
        steps,
        1,
        batch_size,
        128,
        history_mode="device",
        draft_graph_mode=mode,
        conditional_draft_graph=bool(conditional),
        conditional_draft_graph_dispatch="switch" if conditional == "switch" else "if",
        num_neural_gate_tokens=gate,
        num_neural_drafts=neural,
        verify_mode="padded" if neural is not None else "static",
    )

    def query(slots, generations, seq_lens, tokens, mask):
        tokens.copy_(candidates)
        mask.copy_(eligible)

    extension = buffers = None
    if arctic:
        from hybrid_spec.backends.arctic import ArcticBackend
        from hybrid_spec.integrations.sglang.arctic_extension import (
            ArcticDraftExtension,
        )
        from hybrid_spec.integrations.sglang.gated_override import GatedOverride
        from test_arctic import Request

        config = replace(
            config,
            backend="arctic",
            history_mode="cpu",
            min_match_len=1,
            arctic_max_spec_factor=0.0,
            arctic_max_spec_offset=float(steps),
        )
        extension = ArcticDraftExtension(config, ArcticBackend(config, device), 8192)
        buffers = extension.create_graph_buffers(batch_size)
        view = buffers.bind_capture_view(batch_size)
        cpu_requests = [Request(f"r{row}", [], []) for row in range(batch_size)]
    else:
        view = DeviceOverride(
            NS(config=config, device=torch.device(device), consume_and_query=query),
            reqs,
            reqs + 1,
        )
    fb = NS(
        batch_size=batch_size,
        positions=base_positions.clone(),
        mrope_positions=base_positions.expand(3, -1).clone()
        if kind == "triton"
        else None,
        seq_lens=base_positions.clone(),
        req_pool_indices=reqs,
        extend_seq_lens=torch.ones(batch_size, dtype=torch.int32, device=device),
        spec_info=NS(
            draft_override=view,
            topk_p=torch.ones((batch_size, 1), device=device),
            topk_index=seed[:, None],
            hidden_states=initial_hidden,
        ),
        forward_mode=NS(is_idle=lambda: False),
        sampling_info=NS(is_all_greedy=True),
        mark_forward_metadata_ready=lambda: None,
    )

    def model_forward(batch):
        work.index_add_(
            0, batch.req_pool_indices, torch.ones_like(batch.req_pool_indices)
        )
        if kind == "dsv4" and batch is not fb:
            child = get_attn_backend()
            assert batch.kv_loc_plan.rows is batch.req_pool_indices
            req = child.req
            rotary_pos = child.seq + child.step
            write = (
                child.raw
                + child.swa
                + batch.kv_loc_plan.write.view(batch.batch_size, steps)[:, child.step]
            )
        else:
            req = batch.req_pool_indices
            rotary_pos = (
                batch.mrope_positions[0]
                if kind == "triton"
                else batch.seq_lens
                if kind == "dsv4"
                else batch.positions
            )
            # V4's synthetic native metadata includes its step in rotary_pos.
            if kind == "dsv4":
                rotary_pos = rotary_pos + batch.positions - base_positions
            write = batch.out_cache_loc * (3 if kind == "dsv4" else 1)
        read_checksum = 0
        if native_metadata:
            offsets = torch.arange(256, device=device)[None, :]
            if batch is fb:
                # Independent dense reference for the full batch's KV list.
                read_checksum = torch.where(
                    offsets <= batch.positions[:, None],
                    table[batch.req_pool_indices],
                    0,
                ).sum(dim=1)
            else:
                child = get_attn_backend()
                lengths = child.indptr[1:] - child.indptr[:-1]
                addresses = child.indptr[:-1, None] + offsets
                # Masked gather for the ragged native read lists.
                read_checksum = torch.where(
                    offsets < lengths[:, None],
                    child.indices[addresses.clamp(max=child.indices.numel() - 1)],
                    0,
                ).sum(dim=1)
        token = (
            read_checksum
            + batch.input_ids
            + (
                0
                if batch.spec_info.hidden_states is None
                else batch.spec_info.hidden_states.long().sum(dim=1)
            )
            + batch.positions
            + rotary_pos
            + req
            + write
        ) % 128
        logits = torch.full((batch.batch_size, 128), -100.0, device=device)
        logits.scatter_(1, token[:, None], 1.0)
        return NS(
            logits_output=NS(
                next_token_logits=logits,
                hidden_states=(
                    None
                    if standalone
                    else batch.input_ids[:, None]
                    .float()
                    .expand(-1, hidden_width)
                    .clone()
                ),
            )
        )

    def bind_slots(batch):
        batch.kv_loc_plan = NS(rows=batch.req_pool_indices, write=batch.out_cache_loc)

    translator.bind_runner_slots = bind_slots
    model = NS(
        kv_index_translator=translator,
        canary_manager=None,
        model_config=NS(
            model_is_mrope=kind == "triton",
            hf_config=NS(
                architectures=[
                    "Qwen3_5ForCausalLMMTP"
                    if kind == "triton"
                    else "DeepseekV4ForCausalLMNextN"
                    if kind == "dsv4"
                    else "LlamaForCausalLMEagle3"
                    if eagle3
                    else "LlamaForCausalLM"
                ]
            ),
        ),
        forward=model_forward,
    )
    attention = DerivedDsv4Attention() if kind == "dsv4" else Attention()
    worker = NS(
        speculative_num_steps=steps,
        hot_token_id=hot_tokens,
        seed_dsa_topk_from_draft_extend=False,
        index_share_for_mtp_iteration=False,
        draft_extension=extension if arctic else NS(config=config),
        topk=1,
        draft_attn_backend=attention,
        draft_runner=model,
        _topk1_parents_prealloc=torch.zeros(
            (batch_size, max(0, steps - 1)), device=device
        ),
        _topk1_score_indices_prealloc=torch.zeros((batch_size, steps), device=device),
    )
    runner = NS(
        draft_attn_backend=attention,
        model_runner=model,
        speculative_num_steps=steps,
        eagle_worker=worker,
        topk=1,
        max_bs=batch_size,
        captured_req_width=1,
        raw_bs=batch_size,
    )
    ns = dict(
        torch=torch,
        contextlib=contextlib,
        List=List,
        Optional=Optional,
        ForwardBatch=object,
        EagleDraftInput=object,
        _is_cuda=True,
        _is_hip=False,
        get_spec=lambda: NS(speculative_use_rejection_sampling=False),
        maybe_detect_nan=lambda *a: None,
        maybe_detect_inf=lambda *a: None,
        maybe_detect_oob=lambda *a: None,
        draft_topk1_postprocess=draft_topk1_postprocess,
        fast_topk=torch.topk,
        forward_context=forward_context,
        ForwardContext=ForwardContext,
        IndexTopKShareState=NS(mtp_iteration=lambda *a, **kw: contextlib.nullcontext()),
    )
    load_functions(
        "python/sglang/srt/speculative/eagle_utils.py",
        ["per_step_draft_out_cache_loc"],
        ns,
    )
    load_functions(
        "python/sglang/srt/speculative/eagle_worker_v2.py",
        ["draft_forward"],
        ns,
        "EagleDraftWorker",
    )

    load_functions(
        "python/sglang/srt/speculative/spec_utils.py",
        [
            "select_top_k_tokens",
            "_select_top_k_tokens_first",
            "_select_top_k_tokens_later",
        ],
        ns,
    )
    full_view = (
        None
        if arctic
        else DeviceOverride(
            NS(
                config=replace(
                    config,
                    draft_graph_mode="full",
                    conditional_draft_graph=False,
                    conditional_draft_graph_dispatch="if",
                    num_neural_drafts=None,
                    verify_mode="static",
                ),
                device=torch.device(device),
                consume_and_query=query,
            ),
            reqs,
            reqs + 1,
        )
    )

    def prefix():
        fb.spec_info.hidden_states = initial_hidden
        fb.out_cache_loc = flat_cache
        if fb.mrope_positions is not None:
            fb.mrope_positions.copy_(base_positions.expand(3, -1))
        result = ns["draft_forward"](worker, fb)
        fb.positions.sub_(steps - 1)  # Actual runner's run_once restoration.
        return result

    delegate = object.__new__(FullCudaGraphBackend)
    backend = (
        extension.wrap_graph_backend(delegate, runner)
        if arctic
        else SplitDraftGraphBackend(delegate, runner, mode)
    )
    backend.stream = torch.cuda.Stream()
    backend.capture_one(batch_size, prefix, capture_inputs=fb)
    assert backend.can_run(NS(batch_size=batch_size), batch_size)
    record = backend.records[batch_size]
    assert len(record.tails) == (batch_size if tail_steps else 0)
    assert all(
        t.rows.data_ptr() == record.rows.data_ptr() for t in record.tails.values()
    )
    poison = [
        torch.full((size, 8), 47, dtype=dtype, device=device)
        for size in range(1, batch_size + 1)
        for _ in range(8)
        for dtype in (torch.int64, torch.float32, torch.bool)
    ]
    native = []
    for row in range(batch_size):
        token = int(seed[row] if hot_tokens is None else hot_tokens[seed[row]])
        hidden = 0 if standalone else int(initial_hidden[row].sum())
        chain = [token]
        for step in range(steps - 1):
            read_checksum = (
                int(table[reqs[row], : int(base_positions[row]) + step + 1].sum())
                if native_metadata
                else 0
            )
            nxt = (
                read_checksum
                + token
                + hidden
                + int(base_positions[row]) * 2
                + step * 2
                + int(reqs[row])
                + int(cache[step, row]) * (3 if kind == "dsv4" else 1)
            ) % 128
            nxt = nxt if hot_tokens is None else int(hot_tokens[nxt])
            hidden, token = 0 if standalone else token * hidden_width, nxt
            chain.append(token)
        native.append(chain)
    for raw_bs in sorted(
        {batch_size, max(1, batch_size - 1), 1, min(3, batch_size)}, reverse=True
    ):
        bits_values = (
            range(1 << raw_bs)
            if raw_bs <= 3
            else [0, (1 << raw_bs) - 1, 1, 2, 0x55, 0xAA, 0x33]
        )
        for iteration, bits in enumerate(bits_values):
            expected = [chain.copy() for chain in native]
            hit = [i < raw_bs and bool(bits & (1 << i)) for i in range(batch_size)]
            candidate = torch.tensor(native, device=device)
            if gate == 0:
                candidate.fill_(99 + iteration)
                eligible.copy_(torch.tensor(hit, device=device))
            else:
                eligible.fill_(True)
                candidate[:, gate:] = 99 + iteration
                for row in range(batch_size):
                    if not hit[row]:
                        candidate[row, gate - 1] = (candidate[row, gate - 1] + 1) % 128
            for row in range(batch_size):
                if hit[row]:
                    expected[row][gate:] = [99 + iteration] * (steps - gate)
            candidates.copy_(candidate)
            if arctic:
                for row, req in enumerate(cpu_requests[:raw_bs]):
                    anchor = 2048 + row
                    start = anchor if gate or hit[row] else anchor + 1000
                    chain = candidate[row].tolist()
                    req.origin_input_ids = (
                        [start] + chain + [0] * (int(base_positions[row]) - steps - 1)
                    )
                    req.output_ids = [anchor]
                cpu_batch = NS(
                    reqs=cpu_requests[:raw_bs],
                    is_extend_in_batch=False,
                    has_grammar=False,
                    sampling_info=NS(is_all_greedy=True),
                    seq_lens=base_positions[:raw_bs],
                )
                with extension.bind(cpu_batch, NS(draft_override=None)) as staged:
                    buffers.stage(staged, raw_bs, batch_size)
                    full_view = GatedOverride(
                        buffers.tokens.clone(),
                        buffers.use_retrieval.clone(),
                        gate,
                    )
                # This path uses the actual official CPU suffix tree, not query().
                assert view.tokens[:raw_bs][staged.use_retrieval].tolist() == [
                    candidate[row].tolist()
                    for row in range(raw_bs)
                    if staged.use_retrieval[row]
                ]
            fb.positions.copy_(base_positions)
            work.zero_()
            runner.raw_bs = raw_bs
            output = backend.replay(batch_size, fb)[2]
            for row in range(raw_bs):
                limit = steps if hit[row] else (neural or steps)
                assert output[row, :limit].tolist() == expected[row][:limit]
            counts = work[reqs].tolist()
            assert counts == [
                prefix_steps
                + (
                    tail_steps
                    if i < raw_bs and (mode == "split_no_skip" or not hit[i])
                    else 0
                )
                for i in range(batch_size)
            ]
            assert torch.equal(fb.positions, base_positions)
            if tail_steps:
                chosen = [
                    i for i in range(raw_bs) if mode == "split_no_skip" or not hit[i]
                ]
                assert record.rows[: len(chosen)].tolist() == chosen
            # Execute the un-split worker with the same candidates and gate.
            # Verify every live row, including mapped vocab and hidden=None.
            fb.spec_info.draft_override = full_view
            try:
                full_output = prefix()[2]
                for row in range(raw_bs):
                    limit = steps if hit[row] else (neural or steps)
                    assert torch.equal(output[row, :limit], full_output[row, :limit])
            finally:
                fb.spec_info.draft_override = view
    assert poison
    counters = backend.snapshot()
    assert (
        counters["request_ct"]
        == counters["skip_request_ct"] + counters["tail_request_ct"]
    )
    assert counters["prefix_ct"] == counters["skip_ct"] + counters["tail_ct"]
    assert counters["prefix_ct"] == sum(counters["tail_batch_histogram"].values())
    assert counters["tail_request_ct"] == sum(
        int(k) * v for k, v in counters["tail_batch_histogram"].items()
    )


@pytest.mark.parametrize("size", [1, 8, 33, 257, 4097])
@pytest.mark.parametrize("no_skip", [False, True])
@pytest.mark.parametrize("write_live_bs", [False, True])
def test_gpu_dynamic_rows_stable_scan_excludes_padding(size, no_skip, write_live_bs):
    from hybrid_spec.backends.gpu_sa.row_compaction import compact_neural_rows

    rows = torch.empty(size, dtype=torch.int64, device="cuda")
    count = torch.empty((), dtype=torch.int32, device="cuda")
    live_bs = torch.empty_like(count) if write_live_bs else None
    index = torch.arange(size, device="cuda")
    for raw_bs in (size, size // 2, 0, min(size, 1), size):
        for mask in (index % 3 == 0, index >= 0, index < 0):
            rows.fill_(-1)
            if live_bs is not None:
                live_bs.fill_(-99)
            compact_neural_rows(mask, rows, count, raw_bs, no_skip, live_bs=live_bs)
            if live_bs is not None:
                assert int(live_bs) == raw_bs
            expected = [i for i in range(raw_bs) if no_skip or not bool(mask[i])]
            assert int(count) == len(expected)
            assert rows[: len(expected)].tolist() == expected
            assert rows[len(expected) :].eq(-1).all()


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


@pytest.mark.parametrize("max_requests", [1, 4])
@pytest.mark.parametrize("global_pool", [False, True])
def test_finished_prefill_leases_recover_sa_through_graph_replay(
    max_requests, global_pool
):
    """Short-request turnover must not starve subsequent captured SA queries."""
    from types import SimpleNamespace as NS

    class Request:
        def __init__(self, rid):
            self.rid = rid
            self.origin_input_ids = [4, 5, 6, 7, 4]
            self.output_ids = [5]
            self.sampling_params = NS(max_new_tokens=32)
            self.multimodal_inputs = None
            self.done = False

        def get_fill_ids(self):
            return self.origin_input_ids

        def finished(self):
            return self.done

    config = HybridConfig(
        3,
        2,
        max_requests,
        64,
        history_mode="device",
        enable_global_pool=global_pool,
        max_shared_requests=2 if global_pool else 0,
    )
    backend = GPUSuffixAutomaton(config, "cuda")
    extension = DeviceDraftExtension(config, backend, 100)
    buffers = extension.create_graph_buffers(max_requests)
    captured = buffers.bind_capture_view(max_requests)
    seq_lens = torch.full((max_requests,), 5, device="cuda")
    batch = NS(
        seq_lens_cpu=seq_lens.cpu(),
        seq_lens=seq_lens,
        forward_mode=NS(is_idle=lambda: False),
        is_extend_in_batch=False,
        has_grammar=False,
        sampling_info=NS(is_all_greedy=True),
    )
    result = NS(next_token_ids=torch.full((max_requests,), 5, device="cuda"))
    completed = []
    graph = None
    try:
        for wave in range(6):
            batch.reqs = [Request(f"{wave}-{row}") for row in range(max_requests)]
            ticket = extension.begin_prefill(batch)
            assert all(state is not None for state in ticket)
            extension.on_prefill(batch, result, ticket)
            with extension.bind(batch, NS(draft_override=None)) as view:
                buffers.stage(view, max_requests, max_requests)
                if graph is None:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            captured.prepare(seq_lens)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured.prepare(seq_lens)
                graph.replay()
                assert captured.use_retrieval.tolist() == [True] * max_requests
                assert captured.tokens.tolist() == [[6, 7, 4]] * max_requests
            for req in batch.reqs:
                req.done = True  # Deliberately omit the decode finish callback.
            completed.extend(batch.reqs)
        assert len(extension.states) == max_requests
        assert len(extension.shared) == (2 if global_pool else 0)
    finally:
        torch.cuda.synchronize()
        extension.shutdown()


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


@pytest.mark.parametrize("alphabet", [1, 2, 4096])
def test_long_history_hash_import_and_captured_splits(alphabet):
    """Imported hash buckets must grow correctly inside the production graph."""
    config = HybridConfig(8, 4, 1, 32768, history_mode="device", partial_drafts="zero")
    backend = GPUSuffixAutomaton(config, "cuda")
    prompt = list(range(alphabet)) * (8192 // alphabet) + list(range(min(alphabet, 4)))
    backend.initialize_device_slot(0, 1, config.max_history_tokens)
    backend.import_host(0, backend.build_host(backend.prepare_host(prompt), 100000))
    slots = torch.tensor([0], device="cuda")
    generations = torch.tensor([1], device="cuda")
    seq_lens = torch.tensor([len(prompt)], device="cuda")
    predict = torch.zeros(9, dtype=torch.int32, device="cuda")
    indices = torch.arange(9, dtype=torch.int32, device="cuda").view(1, 9)
    lengths = torch.tensor([9], dtype=torch.int32, device="cuda")
    view = DeviceOverride(backend, slots, generations)
    # Capture with no pending update, then stage real feedback on each replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        view.prepare(seq_lens)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        view.prepare(seq_lens)
    oracle = SuffixAutomaton()
    oracle.append(prompt)
    history = list(prompt)
    for step in range(32):
        delta = [5000 + step * 9 + j for j in range(9)]
        # Mix new high-fanout transitions with a long repeated suffix.
        if step % 4 == 3:
            delta = list(range(9))
        predict.copy_(torch.tensor(delta, dtype=torch.int32, device="cuda"))
        seq_lens.fill_(len(history))
        backend.stage_feedback(
            slots, generations, predict, indices, lengths, seq_lens, False, 100000
        )
        history.extend(delta)
        oracle.append(delta)
        seq_lens.fill_(len(history) - 1)
        graph.replay()
        tokens, match = oracle.query(8)
        eligible = match >= 4
        assert view.use_retrieval.item() == eligible
        assert view.tokens.tolist() == [
            tokens + [0] * (8 - len(tokens)) if eligible else [0] * 8
        ]
        assert backend.meta[0, 0].item() == len(history)
        assert backend.meta[0, 4].item() == 1
    # Compare every semantic state and edge with an independent CPU automaton.
    num_states = backend.meta[0, 1].item()
    assert num_states == len(oracle.links)
    states = backend.states[0, :num_states].cpu().tolist()
    edges = backend.edges[0, : backend.meta[0, 2].item(), :3].cpu().tolist()
    for p, (depth, link, pos, inline, head) in enumerate(states):
        assert (depth, link, pos) == (
            oracle.lengths[p],
            oracle.links[p],
            oracle.first_pos[p],
        )
        transitions = {} if inline == -1 else {history[pos + 1]: inline}
        if head < -1:
            target = ~head
            transitions[history[states[target][2]]] = target
            head = -1
        while head != -1:
            token, target, head = edges[head]
            transitions[token] = target
        assert transitions == oracle.edges[p]
