"""Frequency routing against text counts, including device feedback and replay."""

from collections import Counter
from pathlib import Path

import pytest
import torch
from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
from hybrid_spec.backends.reference import SuffixAutomaton
from hybrid_spec.backends.sa_layout import unpack_states
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import HistoryDelta, RequestKey, RequestSnapshot


def text_draft(history, context, k, policy):
    for length in range(min(len(history) - 1, len(context)), 0, -1):
        pattern = context[-length:]
        positions = [
            start + length
            for start in range(len(history) - length)
            if history[start : start + length] == pattern
        ]
        if positions:
            break
    else:
        return [], 0
    if policy == "first":
        return history[positions[0] : positions[0] + k], length
    result = []
    for _ in range(k):
        counts = Counter(
            history[start + len(pattern)]
            for start in range(len(history) - len(pattern))
            if history[start : start + len(pattern)] == pattern
        )
        if not counts:
            break
        token = min(counts, key=lambda token: (-counts[token], token))
        result.append(token)
        pattern = pattern + [token]
    return result, length


def test_default_configuration_and_model_templates_use_fixed_compact_split_gate_one():
    config = HybridConfig(3, 2, 4, 128)
    assert (config.history_mode, config.draft_graph_mode) == ("device", "split")
    assert config.num_neural_gate_tokens == 1
    assert config.sa_continuation_policy == "frequency"
    assert (config.verify_mode, config.dynamic_verify) == ("compact", False)
    directory = Path(__file__).resolve().parents[1] / "configs"
    for path in directory.glob("*.json"):
        config = HybridConfig.from_file(path)
        if path.name in (
            "device-sa-spec.json",  # Explicit upstream comparison policy.
            "device-dspark-sa.json",  # DSPARK only supports gate=0 split.
            "device-dspark-sa-split.json",
        ):
            continue
        assert (config.draft_graph_mode, config.num_neural_gate_tokens) == ("split", 1)
        if path.name == "device-dsv4-mtp-padded-verify.json":
            assert config.verify_mode == "padded"
        else:
            assert config.verify_mode == "compact"
        assert config.dynamic_verify == (
            path.name == "device-dsv4-mtp-dynamic-verify.json"
        )
    with pytest.raises(ValueError, match="sa_continuation_policy"):
        HybridConfig(3, 2, 4, 128, sa_continuation_policy="unknown")


def test_frequency_selects_common_tail_instead_of_first_occurrence():
    history = [1, 2, 9, 9, 9, 40, 1, 2, 7, 7, 7, 41, 1, 2, 7, 7, 7, 42, 1, 2]
    sa = SuffixAutomaton()
    sa.append(history)
    assert sa.query(3, "first") == ([9, 9, 9], 2)
    assert sa.query(3, "frequency") == ([7, 7, 7], 2)
    # A new complete occurrence changes the majority; no stale prompt snapshot.
    for separator in (43, 44):
        delta = [9, 9, 9, separator, 1, 2]
        history.extend(delta)
        sa.append(delta)
        assert sa.query(3, "frequency") == text_draft(history, history, 3, "frequency")
    assert sa.query(3, "frequency") == ([9, 9, 9], 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("policy", ["first", "frequency"])
@pytest.mark.parametrize("graph", [False, True])
def test_device_frequency_feedback_shared_donor_and_graph(shared, policy, graph):
    config = HybridConfig(
        3, 2, 2, 256, sa_continuation_policy=policy, enable_global_pool=shared
    )
    backend = GPUSuffixAutomaton(config, "cuda")
    history = [1, 2, 9, 9, 9, 40, 1, 2, 7, 7, 7, 41, 1, 2, 7, 7, 7, 42, 1, 2]
    query_history = [99, 1, 2] if shared else history
    donor = 1 if shared else 0
    for slot, text in ((0, query_history), (1, history)):
        backend.initialize_device_slot(slot, slot + 1, 256)
        backend.import_host(slot, backend.build_host(backend.prepare_host(text), 100))
    kwargs = dict(device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, -1], **kwargs)
    generations = torch.tensor([1, -1], **kwargs)
    seq_lens = torch.tensor([len(query_history) - 1, 0], **kwargs)
    tokens = torch.empty((2, 3), **kwargs)
    mask = torch.empty(2, device="cuda", dtype=torch.bool)
    sources = (
        dict(
            shared_slots=torch.tensor([0, 1], **kwargs),
            shared_generations=torch.tensor([1, 2], **kwargs),
            shared_lengths=torch.tensor([-1, -1], **kwargs),
        )
        if shared
        else {}
    )

    def query():
        backend.consume_and_query(slots, generations, seq_lens, tokens, mask, **sources)

    query()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            query()
        query = capture.replay
    for delta in ([], [9, 9, 9, 43], [1, 2], [9, 9, 9, 44], [1, 2]):
        if delta:
            # A rejected trailing token is present in predict but not its path.
            predict = torch.tensor(delta + [98], device="cuda", dtype=torch.int32)
            backend.stage_feedback(
                torch.tensor([donor], **kwargs),
                torch.tensor([donor + 1], **kwargs),
                predict,
                torch.arange(len(delta), device="cuda", dtype=torch.int32)[None],
                torch.tensor([len(delta)], device="cuda", dtype=torch.int32),
                torch.tensor([len(history)], **kwargs),
                False,
                100,
            )
            history.extend(delta)
            if not shared:
                seq_lens[0] = len(history) - 1
        query()
        expected, length = text_draft(
            history, query_history if shared else history, 3, policy
        )
        eligible = length >= 2 and len(expected) == 3
        assert mask.tolist() == [eligible, False]
        if eligible:
            assert tokens[0].tolist() == expected
        assert tokens[1].tolist() == [0, 0, 0]
        before = backend.states.clone()
        query()
        assert torch.equal(
            before, backend.states
        )  # No duplicate feedback or donor writes.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("reason", ["short", "ambiguous"])
def test_rejected_frequency_draft_clears_candidate_buffer(reason):
    config = HybridConfig(
        3, 2, 1, 128, sa_require_unique_continuation=reason == "ambiguous"
    )
    backend = GPUSuffixAutomaton(config, "cuda")
    history = (
        [1, 2, 1, 2]
        if reason == "short"
        else [1, 2, 10, 11, 12, 90, 1, 2, 10, 21, 22, 91, 1, 2]
    )
    backend.initialize_device_slot(0, 1, 128)
    backend.append(0, HistoryDelta(RequestKey("a", 1), 0, tuple(history)))
    tokens = torch.full((1, 3), 99, device="cuda", dtype=torch.int64)
    mask = torch.ones(1, device="cuda", dtype=torch.bool)
    backend.consume_and_query(
        torch.tensor([0], device="cuda"),
        torch.tensor([1], device="cuda"),
        torch.tensor([len(history) - 1], device="cuda"),
        tokens,
        mask,
    )
    assert not mask.item()
    assert tokens.tolist() == [[0, 0, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("capacity", [16384, 32767, 32768, 65536])
def test_compact_layout_memory_import_feedback_and_graph(capacity):
    """Both destination layouts must import a short prompt without conversion."""
    config = HybridConfig(3, 2, 1, capacity)
    backend = GPUSuffixAutomaton(config, "cuda")
    compact = capacity <= 32767
    assert backend.states.dtype == (torch.uint16 if compact else torch.int32)
    assert backend.states.shape[-1] == (11 if compact else 10)
    tensors = (
        backend.history,
        backend.states,
        backend.edges,
        backend.meta,
        backend.owners,
        backend.limits,
        backend.pending,
        backend.pending_meta,
    )
    assert sum(
        t.numel() * t.element_size() for t in tensors
    ) == backend.estimate_memory(config)
    history = [
        100001,
        100002,
        100009,
        100009,
        100009,
        100040,
        100001,
        100002,
        100007,
        100007,
        100007,
        100041,
        100001,
        100002,
        100007,
        100007,
        100007,
        100042,
        100001,
        100002,
    ]
    host = backend.build_host(backend.prepare_host(history), 200000)
    assert host.states.dtype == backend.states.dtype
    assert host.states.shape[-1] == backend.states.shape[-1]
    backend.initialize_device_slot(0, 1, capacity)
    wrong = backend.build_host(
        backend.host_builder.prepare(history, compact=not compact), 200000
    )
    with pytest.raises(ValueError, match="invalid completed host automaton"):
        backend.import_host(0, wrong)
    backend.import_host(0, host)
    ns = host.meta[1].item()
    assert torch.equal(
        unpack_states(backend.states[0, :ns].cpu()), unpack_states(host.states[:ns])
    )
    # Exercise the eager query entry point as well as graph consume-and-query.
    candidates = backend.query(RequestSnapshot((), (0, -1), (), ()), 3)
    expected, _ = text_draft(history, history, 3, "frequency")
    assert candidates.tokens.tolist() == [expected, [0, 0, 0]]
    slots = torch.tensor([0, -1], device="cuda")
    generations = torch.tensor([1, -1], device="cuda")
    seq_lens = torch.tensor([len(history) - 1, 0], device="cuda")
    tokens = torch.empty((2, 3), dtype=torch.int64, device="cuda")
    mask = torch.empty(2, dtype=torch.bool, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.consume_and_query(slots, generations, seq_lens, tokens, mask)
    for delta in ([], [100009, 100009, 100009, 100043], [100001, 100002]):
        if delta:
            backend.stage_feedback(
                slots[:1],
                generations[:1],
                torch.tensor(delta + [199999], dtype=torch.int32, device="cuda"),
                torch.arange(len(delta), dtype=torch.int32, device="cuda")[None],
                torch.tensor([len(delta)], dtype=torch.int32, device="cuda"),
                torch.tensor([len(history)], device="cuda"),
                False,
                200000,
            )
            history.extend(delta)
            seq_lens[0] = len(history) - 1
        graph.replay()
        expected, length = text_draft(history, history, 3, "frequency")
        eligible = length >= 2 and len(expected) == 3
        assert mask.tolist() == [eligible, False]
        assert tokens.tolist() == [expected if eligible else [0, 0, 0], [0, 0, 0]]
        used = backend.meta[0, 1].item()
        before = backend.states[0, :used].clone()
        graph.replay()
        assert torch.equal(before, backend.states[0, :used])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_compact_large_state_ids_match_wide_through_graph_updates():
    import random

    rng = random.Random(71)
    history = [100000 + rng.randrange(2) for _ in range(32710)]
    oracle = SuffixAutomaton()
    oracle.append(history)
    backends, graphs, outputs = [], [], []
    slots = torch.tensor([0], device="cuda")
    generations = torch.tensor([1], device="cuda")
    seq_lens = torch.tensor([len(history) - 1], device="cuda")
    for capacity in (32767, 32768):
        backend = GPUSuffixAutomaton(HybridConfig(3, 2, 1, capacity), "cuda")
        backend.initialize_device_slot(0, 1, 100000)
        backend.import_host(
            0, backend.build_host(backend.prepare_host(history), 200000)
        )
        tokens = torch.empty((1, 3), dtype=torch.int64, device="cuda")
        mask = torch.empty(1, dtype=torch.bool, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            backend.consume_and_query(slots, generations, seq_lens, tokens, mask)
        backends.append(backend)
        graphs.append(graph)
        outputs.append((tokens, mask))
    for _ in range(4):
        delta = [100000 + rng.randrange(2) for _ in range(4)]
        for backend in backends:
            backend.stage_feedback(
                slots,
                generations,
                torch.tensor(delta, dtype=torch.int32, device="cuda"),
                torch.arange(4, dtype=torch.int32, device="cuda")[None],
                torch.tensor([4], dtype=torch.int32, device="cuda"),
                torch.tensor([len(history)], device="cuda"),
                False,
                200000,
            )
        history.extend(delta)
        oracle.append(delta)
        seq_lens[0] = len(history) - 1
        for graph in graphs:
            graph.replay()
        assert torch.equal(backends[0].meta, backends[1].meta)
        ns = backends[0].meta[0, 1].item()
        assert ns > 32768
        assert torch.equal(
            unpack_states(backends[0].states[0, :ns].cpu()),
            unpack_states(backends[1].states[0, :ns].cpu()),
        )
        expected, length = oracle.query(3, "frequency")
        eligible = length >= 2 and len(expected) == 3
        for tokens, mask in outputs:
            assert mask.item() == eligible
            assert tokens.tolist() == [expected if eligible else [0, 0, 0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_wide_frequency_above_uint16_survives_graph_feedback():
    """A long slot must preserve actual counts above 65535, not just its dtype."""
    backend = GPUSuffixAutomaton(HybridConfig(8, 2, 1, 131072), "cuda")
    history = [100001] * 65530
    backend.initialize_device_slot(0, 1, 131072)
    backend.import_host(0, backend.build_host(backend.prepare_host(history), 200000))
    slots = torch.tensor([0], device="cuda")
    generations = torch.tensor([1], device="cuda")
    seq_lens = torch.tensor([len(history) - 1], device="cuda")
    tokens = torch.empty((1, 8), dtype=torch.int64, device="cuda")
    mask = torch.empty(1, dtype=torch.bool, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.consume_and_query(slots, generations, seq_lens, tokens, mask)
    backend.stage_feedback(
        slots,
        generations,
        torch.full((9,), 100001, dtype=torch.int32, device="cuda"),
        torch.arange(9, dtype=torch.int32, device="cuda")[None],
        torch.tensor([9], dtype=torch.int32, device="cuda"),
        torch.tensor([len(history)], device="cuda"),
        False,
        200000,
    )
    seq_lens[0] += 9
    graph.replay()
    assert backend.states.dtype == torch.int32
    assert backend.meta[0, 0].item() == 65539
    assert backend.meta[0, 4].item() == 1
    states = backend.states[0, : backend.meta[0, 1].item()].cpu().tolist()
    for node in (0, 1):
        count = states[node][9]
        for _ in range(len(states)):
            parent = states[node][7]
            if parent < 0 or node not in states[parent][5:7]:
                break
            node = parent
            count += states[node][8]
        else:
            pytest.fail("cycle in frequency auxiliary tree")
        assert count == 65539
