import itertools
from types import SimpleNamespace

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
from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers
from hybrid_spec.observability.metrics import Metrics
from hybrid_spec.plugin import _diagnostics, _validate_inputs, _validate_sampling
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


def test_configuration_fails_closed():
    cfg = SimpleNamespace(
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
    config = HybridConfig(2, 2, 2, 32)
    validate_config(config, cfg)
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
        ("speculative_eagle_topk", 2),
        ("speculative_num_steps", 0),
        ("disable_overlap_schedule", False),
        ("speculative_adaptive", True),
        ("enable_multi_layer_eagle", True),
        ("chunked_prefill_size", 2048),
        ("speculative_algorithm", "FROZEN_KV_MTP"),
    ):
        changed = SimpleNamespace(**vars(cfg))
        setattr(changed, field, value)
        with pytest.raises(ValueError):
            validate_config(config, changed)


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
