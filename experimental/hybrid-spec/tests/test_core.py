import importlib
import itertools
import sys
import threading
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
