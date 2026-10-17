import importlib
import itertools
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from hybrid_spec.backends.reference import ReferenceBackend, SuffixAutomaton
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import (
    DraftOverrideView,
    HistoryDelta,
    RequestKey,
    RetrievalCandidates,
)
from hybrid_spec.core.request_state import RequestState
from hybrid_spec.core.runtime import HybridRuntime
from hybrid_spec.integrations.sglang.compatibility import (
    validate_config,
    validate_model,
)
from hybrid_spec.integrations.sglang.device_extension import (
    DeviceDraftExtension,
    DeviceOverride,
)
from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers
from hybrid_spec.observability.metrics import Metrics
from hybrid_spec.plugin import (
    _diagnostics,
    _prefill,
    _validate_inputs,
    _validate_sampling,
)
from hybrid_spec.selection.policy import select_requests


def sequence_oracle(history, k):
    # Independent exhaustive substring search, not an automaton traversal.
    for length in range(len(history) - 1, 0, -1):
        suffix = history[-length:]
        for start in range(len(history) - length):
            if history[start : start + length] == suffix:
                end = start + length
                return history[end : end + k], length
    return [], 0


def test_automaton_matches_exhaustive_oracle_after_every_append():
    for width in range(1, 10):
        for sequence in itertools.product(range(2), repeat=width):
            sa = SuffixAutomaton()
            history = []
            for token in sequence:
                history.append(token)
                sa.append([token])
                for k in (1, 3, 5):
                    assert sa.query(k) == sequence_oracle(history, k)


def test_history_replay_conflict_gap_and_generation():
    key = RequestKey("a", 1)
    state = RequestState(key, 0)
    delta = HistoryDelta(key, 0, (1, 2, 3))
    assert state.commit(delta)
    assert not state.commit(delta)
    for invalid in (
        HistoryDelta(key, 0, (7, 2, 3)),
        HistoryDelta(key, 4, (4,)),
        HistoryDelta(key, 2, (3, 4)),
        HistoryDelta(RequestKey("a", 2), 3, (4,)),
    ):
        with pytest.raises(ValueError):
            state.commit(invalid)
    assert state.history == (1, 2, 3)


def test_runtime_reorder_rebuild_capacity_and_id_reuse():
    config = HybridConfig(2, 2, 2, 32, "reference")
    backend = ReferenceBackend(config)
    runtime = HybridRuntime(config, backend, 100)
    history = [1, 2, 3, 4, 1, 2]
    first = runtime.prepare([("a", history, 20), ("b", [8, 9], 20)])
    assert first.use_retrieval.tolist() == [True, False]
    assert first.tokens[0].tolist() == [3, 4]
    key = runtime.states["a"].key
    repeated = runtime.prepare([("b", [8, 9], 20), ("a", history, 20)])
    assert repeated.use_retrieval.tolist() == [False, True]
    assert backend.slots[runtime.states["a"].slot].history == history
    assert runtime.prepare([("c", history, 20)]).use_retrieval.tolist() == [False]
    # Leaving a batch did not close either request.
    assert set(runtime.states) == {"a", "b"}
    runtime.close_request("a")
    assert runtime.prepare([("c", history, 20)]).use_retrieval.tolist() == [True]
    runtime.close_request("c")
    runtime.prepare([("a", [7, 7, 7], 20)])
    assert runtime.states["a"].key != key
    # Retract/rebuild with changed history invalidates the old generation.
    old = runtime.states["a"].key
    runtime.prepare([("a", history, 20)])
    assert runtime.states["a"].key != old
    assert not runtime.prepare([("a", history * 6, 20)]).use_retrieval.item()
    runtime.reset()
    runtime.prepare([("a", history, 20)])
    assert runtime.states["a"].key.generation > old.generation


@pytest.mark.parametrize(
    "field,value",
    [
        ("healthy", False),
        ("versions", 7),
        ("num_valid_drafts", 1),
        ("match_lens", 1),
        ("remaining", 2),
        ("tokens", 100),
        ("tokens", -1),
    ],
)
def test_whole_chain_fallback(field, value):
    candidates = RetrievalCandidates(
        torch.tensor([[3, 4], [5, 6]]),
        torch.tensor([2, 2]),
        torch.tensor([3, 3]),
        torch.tensor([6, 6]),
        torch.tensor([True, True]),
    )
    remaining = torch.tensor([20, 20])
    if field == "remaining":
        remaining[0] = value
    else:
        getattr(candidates, field)[0] = value
    view = select_requests(candidates, torch.tensor([6, 6]), remaining, 2, 100)
    assert view.use_retrieval.tolist() == [False, True]
    for shape in ((2,), (2, 1)):
        chosen = view.select_step(torch.tensor([90, 91]).reshape(shape), 1)
        assert chosen.shape == shape
        assert chosen.flatten().tolist() == [90, 6]


def test_graph_storage_switches_sources_and_clears_padding():
    buffers = GraphBuffers(4, 2, "cpu")
    bound = buffers.bind_capture_view(4)
    address = bound.tokens.data_ptr()
    for mask in ([True, True, True, True], [False] * 4, [True, False, True, False]):
        buffers.stage(
            DraftOverrideView(torch.arange(8).view(4, 2), torch.tensor(mask)), 4, 4
        )
        assert bound.select_step(torch.full((4, 1), 99), 0).flatten().tolist() == [
            2 * row if enabled else 99 for row, enabled in enumerate(mask)
        ]
    buffers.stage(DraftOverrideView(torch.tensor([[8, 9]]), torch.tensor([True])), 1, 4)
    assert bound.tokens.data_ptr() == address
    assert bound.use_retrieval.tolist() == [True, False, False, False]


class Request:
    def __init__(self, rid="a"):
        self.rid = rid
        self.origin_input_ids = [1, 2, 3, 4, 1]
        self.output_ids = [2]
        self.multimodal_inputs = None
        self.sampling_params = SimpleNamespace(max_new_tokens=30)
        self.done = False

    def finished(self):
        return self.done


def test_bridge_anchor_reconcile_cleanup_and_failure_clear():
    config = HybridConfig(2, 2, 2, 32, "reference")
    extension = DraftExtension(config, ReferenceBackend(config), 100)
    req = Request()
    batch = SimpleNamespace(
        reqs=[req],
        is_extend_in_batch=False,
        has_grammar=False,
        sampling_info=SimpleNamespace(is_all_greedy=True),
        seq_lens=torch.tensor([5]),
    )
    draft_input = SimpleNamespace(draft_override=None)
    with extension.bind(batch, draft_input) as view:
        assert view.tokens.tolist() == [[3, 4]]
        assert view.use_retrieval.tolist() == [True]
    assert draft_input.draft_override is None
    key = extension.runtime.states["a"].key
    batch.seq_lens += 1
    with pytest.raises(RuntimeError), extension.bind(batch, draft_input) as view:
        assert not view.use_retrieval.item()
        raise RuntimeError("test model failure")
    assert draft_input.draft_override is None
    # Same RID with a different request must not inherit the first generation.
    batch.reqs = [Request()]
    with extension.bind(batch, draft_input):
        assert extension.runtime.states["a"].key != key
    batch.reqs[0].done = True
    batch.reqs = []
    batch.seq_lens = torch.tensor([], dtype=torch.int64)
    with extension.bind(batch, draft_input):
        assert not extension.runtime.states


def valid_config():
    return SimpleNamespace(
        device="cuda",
        tp_size=1,
        dp_size=1,
        pp_size=1,
        ep_size=1,
        attn_dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        moe_dp_size=1,
        nnodes=1,
        enable_torch_compile=False,
        speculative_eagle_topk=1,
        speculative_num_steps=2,
        speculative_num_draft_tokens=3,
        disable_overlap_schedule=True,
        enable_mixed_chunk=False,
        enable_multi_layer_eagle=False,
        speculative_adaptive=False,
        enable_lora=False,
        disaggregation_mode="null",
        decoupled_spec_role="null",
        chunked_prefill_size=-1,
        speculative_algorithm="STANDALONE",
    )


def test_configuration_fails_closed():
    cfg = valid_config()
    config = HybridConfig(2, 2, 2, 32)
    validate_config(config, cfg)
    device_config = HybridConfig(2, 2, 2, 32, history_mode="device")
    overlap_cfg = SimpleNamespace(**(vars(cfg) | {"disable_overlap_schedule": False}))
    validate_config(device_config, overlap_cfg)
    for algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
        for tp_size in (1, 2, 4, 8):
            for overlap, chunk_size in itertools.product((False, True), (-1, 2048)):
                validate_config(
                    device_config,
                    SimpleNamespace(
                        **(
                            vars(cfg)
                            | {
                                "tp_size": tp_size,
                                "speculative_algorithm": algorithm,
                                "disable_overlap_schedule": not overlap,
                                "chunked_prefill_size": chunk_size,
                            }
                        )
                    ),
                )
    # ServerArgs resolves an unspecified, disabled LoRA option to None.
    validate_config(config, SimpleNamespace(**(vars(cfg) | {"enable_lora": None})))
    for tp_size in (2, 4, 8):
        validate_config(config, SimpleNamespace(**(vars(cfg) | {"tp_size": tp_size})))
    for field, value in (
        ("device", "cpu"),
        ("tp_size", 0),
        ("tp_size", -1),
        ("tp_size", True),
        ("tp_size", None),
        ("dp_size", 2),
        ("pp_size", 2),
        ("attn_dp_size", 2),
        ("attn_cp_size", 2),
        ("dcp_size", 2),
        ("moe_dp_size", 2),
        ("enable_torch_compile", True),
        ("enable_lora", True),
        ("lora_paths", ["adapter"]),
        ("speculative_eagle_topk", 2),
        ("speculative_num_steps", 0),
        ("disable_overlap_schedule", False),
        ("speculative_adaptive", True),
        ("enable_multi_layer_eagle", True),
        ("chunked_prefill_size", 2048),
        ("enable_mixed_chunk", True),
        ("speculative_algorithm", "FROZEN_KV_MTP"),
    ):
        changed = SimpleNamespace(**vars(cfg))
        setattr(changed, field, value)
        with pytest.raises(ValueError):
            validate_config(config, changed)
    with pytest.raises(ValueError, match="enable_mixed_chunk"):
        validate_config(
            device_config,
            SimpleNamespace(**(vars(overlap_cfg) | {"enable_mixed_chunk": True})),
        )


def test_real_lora_cli_default_none_is_disabled_for_all_spec_algorithms():
    """Load actual field/CLI code, without model imports or hand-written defaults."""
    root = Path(__file__).resolve().parents[3] / "python"
    # Only unused adapter metadata and GPU-dependent constants are boundaries.
    # Field defaults, msgspec records, argparse generation and path action are real.
    modules = {}
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.arg_groups",
        "sglang.srt.arg_groups.fields",
        "sglang.srt.lora",
        "sglang.srt.utils",
    ):
        module = ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        modules[name] = module
    registry = ModuleType("sglang.srt.lora.lora_registry")
    registry.LoRARef = object
    modules[registry.__name__] = registry
    common = ModuleType("sglang.srt.utils.common")
    common.LORA_TARGET_ALL_MODULES = "all"
    common.SUPPORTED_LORA_TARGET_MODULES = []
    modules[common.__name__] = common
    # Restore newly imported child modules too, not just the injected parents.
    with patch.dict(sys.modules, modules):
        utils = importlib.import_module("sglang.srt.arg_groups.arg_utils")
        fields = importlib.import_module("sglang.srt.arg_groups.fields.lora")
        import argparse

        parser = argparse.ArgumentParser()
        utils.add_cli_args_from_dataclass(parser, fields.Lora)
        assert fields.Lora().enable_lora is None
        config = HybridConfig(3, 4, 4, 8192)
        for argv, disabled in (
            ([], True),
            (["--lora-paths"], True),
            (["--enable-lora"], False),
            (["--lora-paths", "/models/adapter"], False),
        ):
            parsed = parser.parse_args(argv)
            for algorithm in ("STANDALONE", "EAGLE", "EAGLE3"):
                for tp_size in (1, 2, 4, 8):
                    cfg = valid_config()
                    cfg.speculative_algorithm, cfg.tp_size = algorithm, tp_size
                    cfg.speculative_num_steps, cfg.speculative_num_draft_tokens = 3, 4
                    cfg.enable_lora, cfg.lora_paths = (
                        parsed.enable_lora,
                        parsed.lora_paths,
                    )
                    if disabled:
                        validate_config(config, cfg)
                    else:
                        with pytest.raises(ValueError, match="LoRA must be disabled"):
                            validate_config(config, cfg)


def test_feedback_attributes_bonus_separately_from_correct_drafts():
    metrics = Metrics()
    metrics.record(torch.tensor([True, False, True]), torch.tensor([3, 1, 2]))
    metrics.record(torch.tensor([False]), torch.tensor([4]))
    assert metrics.snapshot() == dict(
        retrieval_requests=2,
        neural_requests=2,
        retrieval_correct_drafts=3,
        neural_correct_drafts=3,
        retrieval_accept_tokens=5,
        neural_accept_tokens=5,
    )


def test_device_bridge_keeps_feedback_on_device_and_ignores_cpu_decode_history():
    """CPU output lags under overlap; feedback must use verify tensors instead."""
    config = HybridConfig(2, 2, 2, 32, history_mode="device")

    class Backend:
        device = torch.device("cpu")

        def __init__(self):
            self.config = config
            self.feedback = []

        def initialize_device_slot(self, *args):
            self.initialization = args

        def append(self, slot, delta):
            pytest.fail("prefill must not build the prompt SA on GPU")

        def prepare_host(self, history):
            return history

        def build_host(self, host, vocab_size):
            return host

        def import_host(self, slot, host):
            self.prompt = host

        def stage_feedback(self, *args, use_retrieval=None, counts=None):
            self.feedback.append(args)
            if counts is not None:
                reference = Metrics()
                reference.record(use_retrieval, args[4])
                counts.add_(reference.counts)

        def consume_and_query(self, slots, generations, seq_lens, tokens, mask):
            tokens.fill_(7)
            mask.copy_(slots >= 0)

    backend = Backend()
    extension = DeviceDraftExtension(config, backend, 100)
    req = Request()
    req.get_fill_ids = lambda: req.origin_input_ids
    batch = SimpleNamespace(
        reqs=[req],
        seq_lens_cpu=torch.tensor([5]),
        seq_lens=torch.tensor([5]),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        is_extend_in_batch=False,
        has_grammar=False,
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    ticket = extension.begin_prefill(batch)
    extension.on_prefill(
        batch, SimpleNamespace(next_token_ids=torch.tensor([2])), ticket
    )
    assert not hasattr(backend, "prompt")
    assert backend.initialization == (0, 1, 35)
    assert backend.feedback[-1][-2] is False

    class UnavailableHistory:
        def __len__(self):
            raise AssertionError("decode read stale CPU history")

        def __add__(self, other):
            raise AssertionError("decode reconstructed CPU history")

    req.origin_input_ids = req.output_ids = UnavailableHistory()
    req.get_fill_ids = lambda: pytest.fail("decode read CPU fill IDs")
    draft = SimpleNamespace(draft_override=None)
    buffers = extension.create_graph_buffers(4)
    captured = buffers.bind_capture_view(4)
    result = SimpleNamespace(
        next_token_ids=torch.tensor([3, 4, 9], dtype=torch.int32),
        accept_index=torch.tensor([[0, 1, -1]], dtype=torch.int32),
        accept_lens=torch.tensor([2], dtype=torch.int32),
    )
    with extension.bind(batch, draft) as view:
        assert backend.prompt == (1, 2, 3, 4, 1)
        buffers.stage(view, 1, 4)
        captured.prepare(torch.tensor([5, 1, 1, 1]))
        # The replay-written mask, including cleared padding, drives metrics.
        assert view.use_retrieval.tolist() == [True]
        assert captured.use_retrieval.tolist() == [True, False, False, False]
    extension.on_device_verify(result, batch)
    feedback = backend.feedback[-1]
    assert feedback[2] is result.next_token_ids
    assert feedback[3] is result.accept_index
    assert feedback[4] is result.accept_lens
    assert feedback[5] is batch.seq_lens
    assert feedback[-2] is True
    assert extension.metrics.snapshot()["retrieval_accept_tokens"] == 2
    assert draft.draft_override is None
    extension.close_request(req.rid)
    with extension.bind(batch, draft) as view:
        assert view.slots.tolist() == [-1]
    extension.shutdown()


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_shared_requests": -1},
        {"max_shared_requests": True},
        {"shared_context_tokens": 0},
        {"shared_context_tokens": True},
        {"max_shared_requests": 1},
        {"enable_global_pool": 1},
        {"enable_global_pool": True, "history_mode": "cpu"},
    ],
)
def test_shared_history_rejects_invalid_capacity_and_cpu_mode(overrides):
    with pytest.raises(ValueError):
        HybridConfig(2, 2, 1, 32, **overrides)


def test_shared_history_finish_eviction_graph_mapping_and_flush(host_prefill_case):
    """A finished donor changes graph inputs even when active rows stay fixed."""
    extension, backend, batch, _, result = host_prefill_case
    config = replace(
        extension.config, max_requests=2, max_shared_requests=1, enable_global_pool=True
    )
    extension.config = backend.config = config
    extension.free_slots = list(range(config.num_sa_slots))
    a = batch.reqs[0]
    b = Request("b")
    b.get_fill_ids = lambda: b.origin_input_ids
    batch.reqs = [a, b]
    batch.seq_lens_cpu = batch.seq_lens = torch.tensor([5, 5])
    result.next_token_ids = torch.tensor([2, 2])
    ticket = extension.begin_prefill(batch)
    extension.on_prefill(batch, result, ticket)
    draft = SimpleNamespace(draft_override=None)
    # Install both prompts before finishing a. Pending prefill-only builds must
    # not be retained because their device SA has not yet been imported.
    with extension.bind(batch, draft):
        pass
    a_slot = extension.states["a"].slot
    a_generation = extension.states["a"].generation
    batch.reqs = [b]
    batch.seq_lens_cpu = batch.seq_lens = torch.tensor([5])
    buffers = extension.create_graph_buffers(2)
    capture = buffers.bind_capture_view(2)
    with extension.bind(batch, draft) as before:
        buffers.stage(before, 1, 2)
        assert capture.shared_slots.tolist() == [extension.states["b"].slot, a_slot, -1]
        assert capture.shared_lengths.tolist() == [-1, -1, -1]
    # Scheduler finish is CPU-only and may happen on another thread.
    thread = threading.Thread(
        target=extension.close_request, args=("a",), kwargs={"retain": True}
    )
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    extension.close_request("a", retain=True)  # Duplicate delivery is harmless.
    with extension.bind(batch, draft) as after:
        assert after is not before
        buffers.stage(after, 1, 2)
        assert capture.shared_slots.tolist() == [extension.states["b"].slot, a_slot, -1]
        assert capture.shared_generations.tolist() == [
            a_generation + 1,
            a_generation,
            0,
        ]
        assert capture.shared_lengths.tolist() == [-1, 6, -1]
        assert capture.slots[1].item() == -1
    b_slot = extension.states["b"].slot
    extension.close_request("b", retain=True)
    assert list(extension.shared) == [(b_slot, a_generation + 1, 6)]
    assert a_slot in extension.free_slots and b_slot not in extension.free_slots
    extension.reset()
    with extension.bind(batch, draft) as cleared:
        buffers.stage(cleared, 1, 2)
        assert capture.shared_slots.tolist() == [-1, -1, -1]
        assert capture.slots.tolist() == [-1, -1]
    # Aborts/retractions use the ordinary close path and do not publish a donor.
    batch.reqs = [a]
    ticket = extension.begin_prefill(batch)
    result.next_token_ids = torch.tensor([2])
    extension.on_prefill(batch, result, ticket)
    extension.close_request("a", retain=True)
    assert not extension.shared  # Host build not installed.
    assert sorted(extension.free_slots) == list(range(config.num_sa_slots))


@pytest.mark.parametrize("extension_type", [DraftExtension, DeviceDraftExtension])
def test_tp_rejects_process_group_only_communication(extension_type):
    """Both history modes require a communicator usable by the draft graph."""
    config = HybridConfig(2, 2, 1, 32)
    with pytest.raises(ValueError, match="graph-capable PyNccl"):
        extension_type(
            config,
            SimpleNamespace(device=torch.device("cuda")),
            100,
            tp_sync=SimpleNamespace(communicator=None),
        )


def test_device_graph_mapping_cache_preserves_padding_and_generation():
    """Skipping stable mappings must still fence shape changes and slot reuse."""
    config = HybridConfig(3, 2, 4, 32, history_mode="device")
    backend = SimpleNamespace(config=config, device=torch.device("cpu"))
    extension = DeviceDraftExtension(config, backend, 100)
    buffers = extension.create_graph_buffers(4)
    try:

        def stage(slots, generations, padded):
            view = extension._view(slots, generations)
            buffers.stage(view, len(slots), padded)
            assert buffers.view.slots[:padded].tolist() == slots + [-1] * (
                padded - len(slots)
            )
            assert buffers.view.generations[: len(slots)].tolist() == generations
            assert (
                view.use_retrieval.data_ptr() == buffers.view.use_retrieval.data_ptr()
            )
            return view

        view = stage([0, 1], [10, 11], 4)
        versions = (buffers.view.slots._version, buffers.view.generations._version)
        buffers.view.use_retrieval[:2].fill_(True)
        # Tensor version counters prove no input-buffer writes are enqueued.
        # Selection outputs remain live: they are not part of the cached input.
        for _ in range(2):
            buffers.stage(view, 2, 4)
            assert (
                buffers.view.slots._version,
                buffers.view.generations._version,
            ) == versions
            assert view.use_retrieval.tolist() == [True, True]
        stage([0], [10], 1)
        stage([0], [10], 4)  # Growing the graph must clear previously active rows.
        stage([0], [12], 4)  # Same slot, new request generation.
        stage([1, 0], [11, 12], 4)  # Reordered batch.

        # Views without an immutable host key retain the mutable tensor API.
        mutable = DeviceOverride(backend, torch.tensor([1]), torch.tensor([11]))
        buffers.stage(mutable, 1, 4)
        mutable.slots[0] = 0
        mutable.generations[0] = 13
        buffers.stage(mutable, 1, 4)
        assert buffers.view.slots.tolist() == [0, -1, -1, -1]
        assert buffers.view.generations[0].item() == 13
    finally:
        extension.shutdown()


@pytest.fixture
def host_prefill_case():
    config = HybridConfig(2, 2, 1, 32, history_mode="device")

    class Backend:
        device = torch.device("cpu")

        def __init__(self):
            self.config = config
            self.started, self.release = threading.Event(), threading.Event()
            self.release.set()
            self.owner_thread = threading.get_ident()
            self.imports, self.feedback = [], []
            self.fail_build = False

        def prepare_host(self, history):
            return history

        def build_host(self, host, vocab_size):
            assert threading.get_ident() != self.owner_thread
            self.started.set()
            assert self.release.wait(5), "test failed to release native build boundary"
            if self.fail_build:
                raise ValueError("invalid host history")
            return host

        def initialize_device_slot(self, slot, generation, limit):
            assert threading.get_ident() == self.owner_thread
            self.generation = generation

        def import_host(self, slot, host):
            assert threading.get_ident() == self.owner_thread
            self.imports.append((slot, host))

        def stage_feedback(self, *args):
            assert threading.get_ident() == self.owner_thread
            self.feedback.append(args)

    backend = Backend()
    extension = DeviceDraftExtension(config, backend, 100)
    req = Request()
    req.get_fill_ids = lambda: req.origin_input_ids
    batch = SimpleNamespace(
        reqs=[req],
        seq_lens_cpu=torch.tensor([5]),
        seq_lens=torch.tensor([5]),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        is_extend_in_batch=False,
        has_grammar=False,
        sampling_info=SimpleNamespace(is_all_greedy=True),
    )
    worker = SimpleNamespace(draft_worker=SimpleNamespace(draft_extension=extension))
    result = SimpleNamespace(next_token_ids=torch.tensor([2]))
    try:
        yield extension, backend, batch, worker, result
    finally:
        backend.release.set()
        extension.shutdown()


@pytest.mark.parametrize("max_requests", [1, 4])
@pytest.mark.parametrize("global_pool", [False, True])
@pytest.mark.parametrize("installed", [False, True])
def test_prefill_reclaims_finished_leases_before_capacity_check(
    host_prefill_case, max_requests, global_pool, installed
):
    """Prefill-only finishes bypass the decode finish hook, even at c=4."""
    extension, backend, batch, _, result = host_prefill_case
    config = replace(
        extension.config,
        max_requests=max_requests,
        enable_global_pool=global_pool,
        max_shared_requests=int(global_pool),
    )
    extension.config = backend.config = config
    extension.free_slots = list(range(config.num_sa_slots))

    def request(rid):
        req = Request(rid)
        req.get_fill_ids = lambda: [1, 2, 3, 4, 1]
        return req

    def prefill(reqs):
        batch.reqs = reqs
        batch.seq_lens_cpu = batch.seq_lens = torch.full((len(reqs),), 5)
        result.next_token_ids = torch.full((len(reqs),), 2)
        ticket = extension.begin_prefill(batch)
        extension.on_prefill(batch, result, ticket)
        if installed:
            with extension.bind(batch, SimpleNamespace(draft_override=None)):
                pass
        return ticket

    active = [request(f"initial-{i}") for i in range(max_requests)]
    assert all(state is not None for state in prefill(active))
    # Shared-pool spare slots must not increase active-request capacity.
    assert prefill([request("overflow")]) == (None,)
    for iteration in range(6):
        old = active.pop(0)
        state = extension.states[old.rid]
        survivors = dict(extension.states)
        del survivors[old.rid]
        old.done = True  # No explicit close_request / decode finish callback.
        new = request(f"replacement-{iteration}")
        ticket = prefill([new])
        assert ticket[0] is not None, "finished lease blocked the next SA history"
        assert old.rid not in extension.states
        assert ticket[0].generation > state.generation
        assert len(extension.states) == max_requests
        for rid, survivor in survivors.items():
            assert extension.states[rid] is survivor
        expected_shared = (
            [(state.slot, state.generation, 6)] if global_pool and installed else []
        )
        assert list(extension.shared) == expected_shared
        occupied = [s.slot for s in extension.states.values()]
        occupied += [slot for slot, _, _ in extension.shared]
        assert sorted(occupied + extension.free_slots) == list(
            range(config.num_sa_slots)
        )
        active.append(new)


@pytest.mark.parametrize("global_pool", [False, True])
def test_prefill_reclaims_dead_request_without_finish_callback(
    host_prefill_case, global_pool
):
    extension, backend, batch, _, result = host_prefill_case
    config = replace(
        extension.config,
        enable_global_pool=global_pool,
        max_shared_requests=int(global_pool),
    )
    extension.config = backend.config = config
    extension.free_slots = list(range(config.num_sa_slots))
    orphan = Request("orphan")
    orphan.get_fill_ids = lambda: [1, 2, 3, 4, 1]
    batch.reqs = [orphan]
    ticket = extension.begin_prefill(batch)
    extension.on_prefill(batch, result, ticket)
    state = ticket[0]
    batch.reqs = [Request("replacement")]
    batch.reqs[0].get_fill_ids = lambda: [1, 2, 3, 4, 1]
    del orphan
    assert state.request() is None
    replacement = extension.begin_prefill(batch)[0]
    assert replacement is not None, "dead request leaked its SA lease"
    assert replacement.slot == state.slot
    assert replacement.generation > state.generation
    assert not extension.shared  # No confirmed final length for a dead request.


@pytest.mark.parametrize("chunked", [False, True])
def test_host_build_overlaps_prefill_and_imports_only_at_first_draft(
    host_prefill_case, chunked
):
    """A blocked builder must not block prefill launch or its completion hook."""
    extension, backend, batch, worker, result = host_prefill_case
    backend.release.clear()
    if chunked:
        # The batch marker is immutable for this forward, unlike the request's
        # middle-chunk counter, which overlap result processing can change.
        req = batch.reqs[0]
        req.inflight_middle_chunks = 0
        middle = SimpleNamespace(
            **(
                vars(batch)
                | {
                    "chunked_req": req,
                    "seq_lens_cpu": torch.tensor([2]),
                    "seq_lens": torch.tensor([2]),
                }
            )
        )
        # SGLang can skip sampling entirely for an all-middle prefill batch.
        _prefill(
            lambda worker, batch: SimpleNamespace(next_token_ids=None), worker, middle
        )
        assert not extension.states and extension.free_slots == [0]
        assert not backend.started.is_set()
        assert backend.feedback == []
        # The final batch has no middle row even if prior results are pending.
        req.inflight_middle_chunks = 1
        batch.chunked_req = None

    def prefill(worker, batch):
        assert backend.started.wait(5)
        assert not extension.states["a"].build.done()
        batch.reqs[0].origin_input_ids[:] = [8] * 5
        return result

    assert _prefill(prefill, worker, batch) is result
    assert not extension.states["a"].build.done()
    assert backend.imports == []
    backend.release.set()
    draft = SimpleNamespace(draft_override=None)
    for _ in range(2):
        with extension.bind(batch, draft) as view:
            assert view.slots.tolist() == [0]
    assert backend.imports == [(0, (1, 2, 3, 4, 1))]


def test_chunked_prefill_keeps_final_rows_and_discards_middle_bonus(host_prefill_case):
    """A middle chunk must neither occupy a slot nor steal a final row's bonus."""
    extension, backend, batch, worker, result = host_prefill_case
    middle_req = batch.reqs[0]
    final_req = Request("b")
    final_req.origin_input_ids = [6, 7, 8, 6, 7]
    final_req.get_fill_ids = lambda: final_req.origin_input_ids
    batch.reqs = [middle_req, final_req]
    batch.chunked_req = middle_req
    batch.contains_last_prefill_chunk = True
    batch.seq_lens_cpu = torch.tensor([2, 5])
    batch.seq_lens = torch.tensor([2, 5])
    result.next_token_ids = torch.tensor([99, 8])
    _prefill(lambda worker, batch: result, worker, batch)
    with extension.bind(batch, SimpleNamespace(draft_override=None)) as view:
        assert view.slots.tolist() == [-1, 0]
    assert backend.imports == [(0, (6, 7, 8, 6, 7))]
    feedback = backend.feedback[-1]
    assert feedback[0].tolist() == [-1, 0]
    assert feedback[2].tolist() == [99, 8]
    assert feedback[3].tolist() == [[0], [1]]

    # After the other request finishes, the final chunk indexes the full
    # prefix (including tokens processed/cached in earlier chunks).
    extension.close_request("b")
    batch.reqs = [middle_req]
    batch.chunked_req = None
    batch.seq_lens_cpu = torch.tensor([5])
    batch.seq_lens = torch.tensor([5])
    result.next_token_ids = torch.tensor([2])
    _prefill(lambda worker, batch: result, worker, batch)
    with extension.bind(batch, SimpleNamespace(draft_override=None)) as view:
        assert view.slots.tolist() == [0]
    assert backend.imports[-1] == (0, (1, 2, 3, 4, 1))
    assert backend.feedback[-1][2].tolist() == [2]


def test_prefill_exception_cancels_reserved_host_build(host_prefill_case):
    extension, backend, batch, worker, _ = host_prefill_case
    backend.release.clear()

    def prefill(worker, batch):
        assert backend.started.wait(5)
        raise RuntimeError("prefill failed")

    with pytest.raises(RuntimeError, match="prefill failed"):
        _prefill(prefill, worker, batch)
    assert extension.states == {} and extension.free_slots == [0]
    assert backend.imports == []


@pytest.mark.parametrize("finish", ["close", "reset", "chunk"])
def test_late_host_build_cannot_overwrite_reused_slot(host_prefill_case, finish):
    extension, backend, batch, worker, result = host_prefill_case
    backend.release.clear()
    old_req = batch.reqs[0]
    old_ticket = extension.begin_prefill(batch)
    assert backend.started.wait(5)
    extension.on_prefill(batch, result, old_ticket)
    if finish == "close":
        extension.close_request(old_req.rid)
    elif finish == "chunk":
        # Retraction re-prefills a live request. A new middle chunk must
        # invalidate its old build even before the final chunk is scheduled.
        middle = SimpleNamespace(**(vars(batch) | {"chunked_req": old_req}))
        ticket = extension.begin_prefill(middle)
        extension.on_prefill(middle, result, ticket)
        assert ticket == (None,)
        assert not extension.states and extension.free_slots == [0]
    else:
        extension.reset()
    replacement = Request("a")
    replacement.origin_input_ids = [7, 8, 7, 8, 7]
    replacement.get_fill_ids = lambda: replacement.origin_input_ids
    new_batch = SimpleNamespace(**(vars(batch) | {"reqs": [replacement]}))
    _prefill(lambda worker, batch: result, worker, new_batch)
    assert backend.generation == 2
    extension.on_prefill(batch, result, old_ticket)
    extension.cancel_prefill(old_ticket)
    assert backend.generation == 2
    with extension.bind(batch, SimpleNamespace(draft_override=None)) as view:
        assert view.slots.tolist() == [-1]
    backend.release.set()
    with extension.bind(new_batch, SimpleNamespace(draft_override=None)) as view:
        assert view.generations.tolist() == [2]
    assert backend.imports == [(0, (7, 8, 7, 8, 7))]


def test_failed_host_build_releases_slot_before_propagating(host_prefill_case):
    extension, backend, batch, worker, result = host_prefill_case
    backend.fail_build = True
    _prefill(lambda worker, batch: result, worker, batch)
    with pytest.raises(ValueError, match="invalid host history"):
        with extension.bind(batch, SimpleNamespace(draft_override=None)):
            pytest.fail("failed host build must not reach draft execution")
    assert extension.states == {} and extension.free_slots == [0]


@pytest.mark.parametrize("moe", [False, True])
def test_qwen_linear_attention_requires_text_only_mtp_pair(moe):
    target = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[
                "Qwen3_5MoeForConditionalGeneration"
                if moe
                else "Qwen3_5ForConditionalGeneration"
            ],
            language_only=True,
        ),
        is_multimodal=True,
        model_is_mrope=True,
        is_hybrid_swa=False,
    )
    draft = SimpleNamespace(
        **(
            vars(target)
            | {
                "hf_config": SimpleNamespace(
                    architectures=["Qwen3_5ForCausalLMMTP"], language_only=True
                )
            }
        )
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model_config=target)
        ),
        draft_runner=SimpleNamespace(model_config=draft),
    )
    validate_model(worker)
    # Do not turn this exception into permission for arbitrary hybrid targets,
    # external draft models, multimodal serving, or sliding-window pools.
    for obj, field, value in (
        (target.hf_config, "architectures", ["Qwen3NextForCausalLM"]),
        (draft.hf_config, "architectures", ["Qwen3ForCausalLM"]),
        (target.hf_config, "language_only", False),
        (draft.hf_config, "language_only", False),
        (target, "is_hybrid_swa", True),
        (draft, "is_hybrid_swa", True),
    ):
        original = getattr(obj, field)
        setattr(obj, field, value)
        with pytest.raises(ValueError):
            validate_model(worker)
        setattr(obj, field, original)


def dsv4_worker():
    def config(architecture):
        return SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[architecture]),
            is_hybrid_swa=True,
            is_multimodal=False,
            model_is_mrope=False,
        )

    return SimpleNamespace(
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model_config=config("DeepseekV4ForCausalLM"))
        ),
        draft_runner=SimpleNamespace(model_config=config("DeepseekV4ForCausalLMNextN")),
    )


@pytest.mark.parametrize("swa", [True, False])
def test_dsv4_native_mtp_sa_is_not_rejected_as_generic_hybrid_swa(swa):
    # The target and draft both advertise hybrid SWA even though the native
    # V4 MTP path already owns its per-step KV/window metadata.
    worker = dsv4_worker()
    for config in (
        worker.target_worker.model_runner.model_config,
        worker.draft_runner.model_config,
    ):
        config.is_hybrid_swa = swa
    validate_model(worker)


def test_dsv4_sa_exception_requires_mtp_pair_and_text_only():
    worker = dsv4_worker()
    target = worker.target_worker.model_runner.model_config
    draft = worker.draft_runner.model_config
    for obj, field, value in (
        (target.hf_config, "architectures", ["LlamaForCausalLM"]),
        (draft.hf_config, "architectures", ["DeepseekV4ForCausalLMDSpark"]),
        (draft.hf_config, "architectures", ["Qwen3ForCausalLM"]),
        (target, "is_multimodal", True),
        (draft, "model_is_mrope", True),
    ):
        original = getattr(obj, field)
        setattr(obj, field, value)
        with pytest.raises(ValueError, match="DeepSeek V4"):
            validate_model(worker)
        setattr(obj, field, original)
    # Text-only use of a vision-capable V4 checkpoint remains explicit.
    target.is_multimodal = True
    target.hf_config.language_only = True
    validate_model(worker)
    # An unrelated hybrid SWA model does not inherit the V4 exception.
    target.hf_config.architectures = ["Step3p5ForCausalLM"]
    draft.hf_config.architectures = ["Step3p5MTP"]
    with pytest.raises(ValueError, match="hybrid sliding-window"):
        validate_model(worker)


def native_split_backend(monkeypatch, mode, attention="dsv4"):
    # Replace only the CUDA backend type boundary. Dispatch and counter/position
    # bookkeeping use the production split implementation on real CPU tensors.
    from hybrid_spec.integrations.sglang.split_graph import SplitDraftGraphBackend

    name = "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend"
    module = ModuleType(name)
    module.FullCudaGraphBackend = type("FullCudaGraphBackend", (), {})
    monkeypatch.setitem(sys.modules, name, module)
    worker = dsv4_worker()
    worker.hot_token_id = None
    worker.seed_dsa_topk_from_draft_extend = False
    worker.draft_extension = SimpleNamespace()
    runner = SimpleNamespace(
        topk=1,
        max_bs=4,
        captured_req_width=1,
        eagle_worker=worker,
        model_runner=worker.draft_runner,
        raw_bs=4,
        draft_attn_backend=SimpleNamespace(decode_attention_backend_str=attention),
    )
    return SplitDraftGraphBackend(module.FullCudaGraphBackend(), runner, mode)


def test_dsv4_split_rejects_qwen_triton_attention(monkeypatch):
    with pytest.raises(ValueError, match="native dsv4 draft attention"):
        native_split_backend(monkeypatch, "split", attention="triton")


@pytest.mark.parametrize("k", [3, 8])
def test_dsv4_compact_metadata_preserves_request_major_step_writes(k):
    from hybrid_spec.integrations.sglang.compact_graph import _plan_dsv4_metadata

    requests = torch.tensor([31, 7, 42, 13])
    lengths = torch.tensor([151, 257, 389, 517])
    # Independent ordinal encoding makes wrong row/step strides observable.
    full_cache = requests[None, :] * 1000 + torch.arange(k)[:, None]

    class Child:
        def __init__(self, step, size):
            self.step = step
            self.req = torch.empty(size, dtype=torch.int64)
            self.lens = torch.empty_like(self.req)
            self.raw_write = torch.empty_like(self.req)
            self.swa_write = torch.empty_like(self.req)

        def init_forward_metadata_out_graph(self, fb, in_capture):
            assert fb.seq_lens_cpu is None
            self.req.copy_(fb.req_pool_indices)
            self.lens.copy_(fb.seq_lens)
            self.raw_write.copy_(fb.out_cache_loc)

        def init_forward_metadata_in_graph(self, fb):
            # Native SWA refresh reads the request-major full window by step.
            assert fb.out_cache_loc.shape == (fb.batch_size * k,)
            self.swa_write.copy_(fb.out_cache_loc.view(fb.batch_size, k)[:, self.step])

    for size in range(1, 5):
        children = [Child(step, size) for step in range(k)]
        attention = SimpleNamespace(attn_backends=children)
        cache_window = torch.empty((size, k), dtype=torch.int64)
        cache = cache_window.T
        flat = cache_window.flatten()
        fb = SimpleNamespace(
            batch_size=size,
            req_pool_indices=torch.empty(size, dtype=torch.int64),
            seq_lens=torch.empty(size, dtype=torch.int64),
            seq_lens_cpu=None,
            out_cache_loc=flat,
        )
        for selection in itertools.combinations(range(4), size):
            # Also vary the row order: the request id is not the compact row id.
            rows = torch.tensor(selection[::-1])
            fb.req_pool_indices.copy_(requests[rows])
            fb.seq_lens.copy_(lengths[rows])
            torch.index_select(full_cache, 1, rows, out=cache)
            _plan_dsv4_metadata(
                attention, fb, cache, flat, start_step=1, in_capture=True
            )
            _plan_dsv4_metadata(
                attention, fb, cache, flat, start_step=1, in_capture=False
            )
            assert fb.out_cache_loc is flat
            for step in range(1, k - 1):
                child = children[step]
                actual = (
                    child.raw_write + 2 * child.swa_write + 7 * child.req + child.lens
                )
                expected = torch.tensor(
                    [
                        3 * (int(requests[row]) * 1000 + step)
                        + 7 * int(requests[row])
                        + int(lengths[row])
                        for row in selection[::-1]
                    ]
                )
                assert torch.equal(actual, expected)


def test_diagnostic_readback_preserves_native_state_and_counts():
    metrics = Metrics()
    metrics.record(torch.tensor([True, False]), torch.tensor([2, 1]))
    controller = SimpleNamespace(
        scheduler=SimpleNamespace(
            draft_worker=SimpleNamespace(
                draft_worker=SimpleNamespace(
                    draft_extension=SimpleNamespace(metrics=metrics)
                )
            )
        )
    )
    result = SimpleNamespace(internal_state={"avg_spec_accept_length": 1.5})
    _diagnostics(result, controller)
    assert result.internal_state["avg_spec_accept_length"] == 1.5
    assert result.internal_state["hybrid_spec"]["retrieval_correct_drafts"] == 1
    assert result.internal_state["hybrid_spec"]["neural_accept_tokens"] == 1
    # Repeated reads must not consume counters or change subsequent results.
    _diagnostics(result, controller)
    assert result.internal_state["hybrid_spec"]["retrieval_accept_tokens"] == 2


def test_unsupported_requests_rejected_before_scheduling():
    params = dict(
        top_k=1,
        json_schema=None,
        regex=None,
        ebnf=None,
        structural_tag=None,
        beam_width=None,
        watermark=None,
    )
    _validate_sampling(SimpleNamespace(**params), 100)
    for field, value in (
        ("top_k", 50),
        ("regex", "a.*"),
        ("beam_width", 2),
        ("watermark", {"enabled": True}),
    ):
        with pytest.raises(ValueError):
            _validate_sampling(SimpleNamespace(**(params | {field: value})), 100)


def test_media_and_embeddings_rejected_before_gpu_scheduling():
    _validate_inputs(
        SimpleNamespace(contains_mm_input=lambda: False, input_embeds=None)
    )
    for media, embeds in ((True, None), (False, [[0.5]])):
        with pytest.raises(ValueError, match="text or token IDs"):
            _validate_inputs(
                SimpleNamespace(contains_mm_input=lambda: media, input_embeds=embeds)
            )


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_drafts", 0),
        ("num_drafts", True),
        ("min_match_len", -1),
        ("max_requests", 0),
        ("max_history_tokens", 2),
        ("backend", "unknown"),
    ],
)
def test_invalid_plugin_configuration(field, value):
    values = dict(num_drafts=2, min_match_len=2, max_requests=2, max_history_tokens=32)
    values[field] = value
    with pytest.raises(ValueError):
        HybridConfig(**values)


@pytest.mark.parametrize("value", [-1, 3, True, 1.5])
def test_invalid_neural_gate_width(value):
    with pytest.raises(ValueError, match="num_neural_gate_tokens"):
        HybridConfig(2, 2, 2, 32, num_neural_gate_tokens=value)


@pytest.mark.parametrize(
    "change",
    [
        {"draft_graph_mode": "unknown"},
        {"history_mode": "cpu"},
    ],
)
def test_split_graph_configuration_rejects_unsupported_state(change):
    values = dict(
        num_drafts=8,
        min_match_len=4,
        max_requests=1,
        max_history_tokens=128,
        history_mode="device",
        num_neural_gate_tokens=2,
        draft_graph_mode="split",
    )
    with pytest.raises(ValueError):
        HybridConfig(**(values | change))


def test_split_graph_runtime_configuration_fails_closed():
    config = HybridConfig(
        8,
        4,
        1,
        128,
        history_mode="device",
        num_neural_gate_tokens=2,
        draft_graph_mode="split",
    )
    values = vars(valid_config()) | {
        "speculative_num_steps": 8,
        "speculative_algorithm": "EAGLE",
        "speculative_num_draft_tokens": 9,
        "max_running_requests": 1,
        "speculative_use_rejection_sampling": False,
        "disable_cuda_graph": False,
        "enable_memory_saver": False,
    }
    for algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
        for tp_size in (1, 2, 4, 8):
            validate_config(
                config,
                SimpleNamespace(
                    **(
                        values
                        | {"tp_size": tp_size, "speculative_algorithm": algorithm}
                    )
                ),
            )
    for field, value in (
        ("max_running_requests", 2),
        ("speculative_use_rejection_sampling", True),
        ("disable_cuda_graph", True),
        ("enable_memory_saver", True),
    ):
        with pytest.raises(ValueError, match="split draft graphs require"):
            validate_config(config, SimpleNamespace(**(values | {field: value})))


def test_compact_dispatch_uses_raw_count_before_runner_restores_padding(monkeypatch):
    """The runner temporarily sets batch_size=4 for three live requests.

    Dispatch must use raw_bs or a padded sink row enters the neural tail and
    inflates request counters. The compact gather itself is tested on CUDA.
    """
    from hybrid_spec.integrations.sglang import compact_graph
    from hybrid_spec.integrations.sglang.split_graph import SplitDraftGraphBackend

    backend = object.__new__(SplitDraftGraphBackend)
    backend.records = {4: compact_graph.CompactCapture(None, None, None, None, ())}
    backend.mode = "split"
    backend.counters = {}
    backend.runner = SimpleNamespace(raw_bs=3)
    monkeypatch.setattr(
        compact_graph, "replay_compact", lambda record, counters, raw_bs: raw_bs
    )
    assert backend.replay(4, SimpleNamespace(batch_size=4)) == 3


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("mode", ["split", "split_no_skip"])
def test_dynamic_compact_dispatch_keeps_live_rows_and_buffer_addresses(
    monkeypatch, mode, batch_size
):
    from hybrid_spec.integrations.sglang.compact_graph import (
        CompactCapture,
        prepare_compaction,
        replay_compact,
    )

    mask = torch.zeros(batch_size, dtype=torch.bool)
    output = torch.zeros(batch_size, dtype=torch.int64)
    record = CompactCapture(
        SimpleNamespace(replay=lambda: output.zero_()),
        output,
        SimpleNamespace(batch_size=batch_size),
        SimpleNamespace(use_retrieval=mask),
        (),
    )
    prepare_compaction(record, mode)
    for size in range(1, batch_size + 1):
        rows = record.rows[:size]
        record.tails[size] = SimpleNamespace(
            rows=rows,
            graph=SimpleNamespace(
                replay=lambda rows=rows: output.index_fill_(0, rows, 7)
            ),
        )

    def compact(mask, rows, count, raw_bs, no_skip):
        chosen = (
            torch.arange(raw_bs) if no_skip else (~mask[:raw_bs]).nonzero().flatten()
        )
        rows[: len(chosen)].copy_(chosen)
        count.fill_(len(chosen))

    from hybrid_spec.integrations.sglang import compact_graph

    monkeypatch.setattr(compact_graph, "compact_neural_rows", compact)
    addresses = {key: tail.rows.data_ptr() for key, tail in record.tails.items()}
    counters = dict.fromkeys(
        (
            "prefix_ct",
            "skip_ct",
            "tail_ct",
            "decision_wait_ns",
            "request_ct",
            "skip_request_ct",
            "tail_request_ct",
            "mixed_ct",
        ),
        0,
    )
    counters["tail_batch_histogram"] = {str(i): 0 for i in range(batch_size + 1)}
    # Repeat changing masks/sizes: stale row vectors and padded rows must never
    # execute, including split_no_skip's all-live-rows path.
    for raw_bs in (batch_size, max(1, batch_size - 1), 1, min(3, batch_size)):
        bits_values = (
            range(1 << raw_bs) if raw_bs <= 3 else (0, (1 << raw_bs) - 1, 1, 0x55, 0xAA)
        )
        for bits in bits_values:
            mask.zero_()
            for i in range(raw_bs):
                mask[i] = bool(bits & (1 << i))
            replay_compact(record, counters, raw_bs)
            expected = [
                7
                if i < raw_bs and (mode == "split_no_skip" or not bits & (1 << i))
                else 0
                for i in range(batch_size)
            ]
            assert output.tolist() == expected
            assert addresses == {
                key: tail.rows.data_ptr() for key, tail in record.tails.items()
            }


def test_neural_gate_falls_back_before_divergence_and_resets_next_batch():
    """A four-token suffix alone can replace a correct MTP prefix with a wrong one.

    Gating must preserve the neural recurrence after either early disagreement,
    keep the metrics mask alias, and allow retrieval again on the next batch.
    """
    buffers = GraphBuffers(3, 3, "cpu", num_neural_gate_tokens=2)
    view = buffers.bind_capture_view(3)
    mask = view.use_retrieval
    candidates = torch.tensor([[10, 20, 99], [11, 20, 98], [10, 21, 97]])
    buffers.stage(DraftOverrideView(candidates, torch.ones(3, dtype=torch.bool)), 3, 3)
    outputs = [
        view.select_step(torch.full((3, 1), t), i) for i, t in enumerate((10, 20, 30))
    ]
    assert torch.cat(outputs, dim=1).tolist() == [
        [10, 20, 99],
        [10, 20, 30],
        [10, 20, 30],
    ]
    assert view.use_retrieval is mask and mask.tolist() == [True, False, False]
    candidates[:, :2] = torch.tensor([10, 20])
    buffers.stage(DraftOverrideView(candidates, torch.ones(3, dtype=torch.bool)), 3, 3)
    outputs = [
        view.select_step(torch.full((3, 1), t), i) for i, t in enumerate((10, 20, 30))
    ]
    assert torch.equal(torch.cat(outputs, dim=1), candidates)
    assert mask.tolist() == [True, True, True]


def test_neural_gate_propagates_through_history_and_capture_views():
    config = HybridConfig(3, 2, 1, 32, "reference", num_neural_gate_tokens=2)
    runtime = HybridRuntime(config, ReferenceBackend(config), 100)
    assert runtime.prepare([("a", [1, 2, 3, 4, 1, 2], 20)]).num_neural_gate_tokens == 2
    backend = SimpleNamespace(config=config, device=torch.device("cpu"))
    from hybrid_spec.integrations.sglang.device_extension import DeviceGraphBuffers

    assert (
        DeviceGraphBuffers(backend, 3).bind_capture_view(3).num_neural_gate_tokens == 2
    )


@pytest.mark.parametrize(
    "k,gate", [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (8, 3), (8, 8)]
)
def test_split_gate_and_large_batch_configuration(k, gate):
    config = HybridConfig(
        k,
        1,
        32,
        128,
        history_mode="device",
        num_neural_gate_tokens=gate,
        draft_graph_mode="split",
    )
    cfg = valid_config()
    cfg.speculative_algorithm = "EAGLE"
    cfg.speculative_num_steps = k
    cfg.speculative_num_draft_tokens = k + 1
    cfg.max_running_requests = 32
    cfg.speculative_use_rejection_sampling = False
    cfg.disable_cuda_graph = False
    cfg.enable_memory_saver = False
    validate_config(config, cfg)


@pytest.mark.parametrize("gate", [3, -1, True, 1.5])
def test_split_rejects_gate_outside_draft_budget(gate):
    with pytest.raises(ValueError, match="num_neural_gate_tokens"):
        HybridConfig(
            2,
            1,
            8,
            128,
            history_mode="device",
            num_neural_gate_tokens=gate,
            draft_graph_mode="split",
        )


def test_arctic_split_requires_committed_cpu_history_configuration():
    config = HybridConfig(
        2,
        1,
        8,
        128,
        backend="arctic",
        history_mode="cpu",
        draft_graph_mode="split",
        num_neural_gate_tokens=2,
        enable_global_pool=True,
        max_shared_requests=16,
    )
    values = vars(valid_config()) | {
        "max_running_requests": 8,
        "speculative_use_rejection_sampling": False,
        "disable_cuda_graph": False,
        "enable_memory_saver": False,
    }
    for algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
        validate_config(
            config, SimpleNamespace(**(values | {"speculative_algorithm": algorithm}))
        )
    for change in ({"disable_overlap_schedule": False}, {"chunked_prefill_size": 2048}):
        with pytest.raises(ValueError, match="CPU history"):
            validate_config(config, SimpleNamespace(**(values | change)))
    with pytest.raises(ValueError, match="history_mode=cpu"):
        HybridConfig(2, 1, 8, 128, backend="arctic", history_mode="device")


@pytest.mark.parametrize(
    "field,value",
    [
        ("arctic_max_tree_depth", 0),
        ("arctic_max_tree_depth", True),
        ("arctic_max_spec_factor", -1),
        ("arctic_max_spec_factor", float("nan")),
        ("arctic_max_spec_offset", float("inf")),
        ("arctic_max_spec_offset", True),
        ("arctic_min_token_prob", -0.1),
        ("arctic_min_token_prob", 1.1),
    ],
)
def test_arctic_parameters_reject_invalid_probabilities_and_bounds(field, value):
    with pytest.raises(ValueError, match="arctic_"):
        HybridConfig(2, 1, 8, 128, backend="arctic", **{field: value})
