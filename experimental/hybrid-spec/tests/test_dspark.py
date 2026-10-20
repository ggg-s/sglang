"""DSPARK SA bridge contracts; model/accept kernels are explicit test boundaries."""

import ast
import contextlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import torch
from hybrid_spec.config import HybridConfig
from hybrid_spec.integrations.sglang.compatibility import (
    validate_config,
    validate_dspark_model,
)
from hybrid_spec.integrations.sglang.dspark import (
    decode,
    splice_proposal,
    stage_feedback,
)
from hybrid_spec.integrations.sglang.gated_override import GatedOverride
from hybrid_spec.observability.metrics import Metrics
from hybrid_spec.plugin import _diagnostics, _finished, _prefill, _reset
from test_core import valid_config
from test_draft_recurrence import ROOT, load_functions


def proposal_types():
    path = ROOT / "python/sglang/srt/speculative/dspark_components/dspark_draft.py"
    tree = ast.parse(path.read_text())
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("DraftBlockResult", "DraftProposal")
    ]
    namespace = dict(
        torch=torch, msgspec=msgspec, Optional=__import__("typing").Optional
    )
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["DraftBlockResult"], namespace["DraftProposal"]


Block, Proposal = proposal_types()


def config(**kwargs):
    return HybridConfig(3, 2, 4, 64, history_mode="device", **kwargs)


def make_proposal(tokens, folded=True):
    bs = len(tokens)
    return Proposal(
        draft_block_ids=torch.full((bs, 4), 42, device=tokens.device),
        draft_block=Block(
            tokens,
            torch.ones(1),
            torch.ones(bs, dtype=torch.bool),
            torch.ones(bs),
            False,
        ),
        confidence=torch.ones_like(tokens, dtype=torch.float32),
        confidence_raw=torch.ones_like(tokens, dtype=torch.float32),
        folded=folded,
    )


@pytest.mark.parametrize("gate", [0, 1, 2, 3])
@pytest.mark.parametrize("folded", [False, True])
def test_gate_miss_and_folded_storage(gate, folded):
    neural = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    view = GatedOverride(
        torch.tensor([[1, 2, 30], [4, 50, 60], [7, 80, 90]]),
        torch.tensor([True, True, False]),
    )
    original = make_proposal(neural, folded)
    pointer = neural.data_ptr()
    extension = SimpleNamespace(
        config=config(num_neural_gate_tokens=gate), pending_view=view
    )
    result = splice_proposal(
        original, SimpleNamespace(_hybrid_spec_extension=extension)
    )
    assert result.draft_block.draft_tokens.data_ptr() == pointer
    assert result.folded == folded
    assert result.draft_block_ids is original.draft_block_ids
    expected = [
        [1, 2, 30] if gate <= 2 else [1, 2, 3],
        [4, 50, 60] if gate <= 1 else [4, 5, 6],
        [7, 8, 9],
    ]
    assert result.draft_block.draft_tokens.tolist() == expected
    assert result.confidence is result.confidence_raw is None
    assert result.draft_block.corrected_logits is None
    assert result.draft_block.greedy_draft


def test_tp_consensus_runs_after_gate_and_can_veto():
    seen = []

    def synchronize(view):
        seen.append(view.use_retrieval.tolist())
        return SimpleNamespace(
            tokens=view.tokens, use_retrieval=torch.tensor([False, False])
        )

    view = GatedOverride(
        torch.tensor([[1, 9, 9], [8, 8, 8]]),
        torch.ones(2, dtype=torch.bool),
        tp_sync=SimpleNamespace(synchronize=synchronize),
    )
    extension = SimpleNamespace(
        config=config(num_neural_gate_tokens=1), pending_view=view
    )
    original = make_proposal(torch.tensor([[1, 2, 3], [4, 5, 6]]))
    result = splice_proposal(
        original, SimpleNamespace(_hybrid_spec_extension=extension)
    )
    assert seen == [[True, False]]
    assert result.draft_block.draft_tokens.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_configuration_and_example_fail_closed():
    cfg = valid_config()
    cfg.speculative_algorithm = "DSPARK"
    cfg.speculative_num_steps = 1  # DSPARK uses gamma, not recurrent depth.
    cfg.speculative_num_draft_tokens = 4
    validate_config(config(), cfg)
    for invalid in (
        replace(config(), history_mode="cpu"),
        replace(config(), backend="arctic", history_mode="cpu"),
        replace(config(), draft_graph_mode="split"),
        replace(config(), verify_mode="padded"),
        replace(config(), verify_mode="compact", dynamic_verify=True),
    ):
        with pytest.raises(ValueError, match="DSPARK SA requires"):
            validate_config(invalid, cfg)
    example = HybridConfig.from_file(
        str(ROOT / "experimental/hybrid-spec/configs/device-dspark-sa.json")
    )
    cfg.speculative_num_draft_tokens = example.num_drafts + 1
    validate_config(example, cfg)


def test_model_guards_and_native_confidence_contract(monkeypatch):
    module = ModuleType("sglang.srt.configs.hybrid_arch")
    module.mambaish_config = lambda model: getattr(model, "mamba", None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        model_is_mrope=False,
        is_multimodal=False,
        is_hybrid_swa=True,
    )
    worker = SimpleNamespace(
        gamma=3,
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(model_config=model)),
        draft_model_runner=SimpleNamespace(model_config=model),
        _verify_planner=SimpleNamespace(mode_value="static", carries_confidence=True),
        _hosts_draft=True,
        draft_extension=None,
    )
    validate_dspark_model(worker, config())
    draft = SimpleNamespace(**vars(model))
    draft.hf_config = SimpleNamespace(architectures=["DeepseekV4ForCausalLMDSpark"])
    worker.draft_model_runner.model_config = draft
    validate_dspark_model(worker, config())
    namespace = {}
    load_functions(
        "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
        ["carries_confidence"],
        namespace,
        "DSparkWorkerV2",
    )
    assert namespace["carries_confidence"](worker)
    del worker.draft_extension
    assert namespace["carries_confidence"](worker)
    worker.draft_extension = object()
    assert not namespace["carries_confidence"](worker)
    worker._verify_planner.mode_value = "compact"
    with pytest.raises(ValueError, match="RAGGED_VERIFY_MODE=static"):
        validate_dspark_model(worker, config())
    worker._verify_planner.mode_value = "static"
    worker.gamma = 2
    with pytest.raises(ValueError, match="gamma"):
        validate_dspark_model(worker, config())
    worker.gamma = 3
    model.is_multimodal = True
    with pytest.raises(ValueError, match="text-only"):
        validate_dspark_model(worker, config())
    model.hf_config.language_only = True
    validate_dspark_model(worker, config())
    model.mamba = object()
    with pytest.raises(ValueError, match="text-only"):
        validate_dspark_model(worker, config())


class TestExtension:
    __test__ = False

    def __init__(self, view):
        self.config = config(num_neural_gate_tokens=1)
        self.pending_view = None
        self.view = view
        self.vocab_size = 100
        self.metrics = Metrics()
        self.feedback = []
        self.backend = SimpleNamespace(
            device="cpu", stage_feedback=lambda *a, **kw: self.feedback.append((a, kw))
        )
        self.close_request = Mock()
        self.reset = Mock()

    @contextlib.contextmanager
    def bind(self, batch, draft_input):
        self.pending_view = self.view
        yield self.view


def test_native_decode_uses_sa_and_preserves_verify_commit_order(monkeypatch):
    """Execute DSPARK's actual decode body, substituting only model/KV kernels."""

    class DraftInput:
        pass

    namespace = dict(
        torch=torch,
        ScheduleBatch=object,
        GenerationBatchResult=SimpleNamespace,
        DFlashDraftInputV2=DraftInput,
        get_parallel=lambda: SimpleNamespace(attn_dp_enabled=False),
        InfoSegment=SimpleNamespace(DRAFT="draft", TARGET_VERIFY="verify"),
        alloc_verify_window=lambda **kw: object(),
        verify_logits_adjustments_are_noop=lambda sampling: True,
        prepare_mamba_track_for_verify=lambda batch: None,
        SpeculativeSamplingMaskCapture=SimpleNamespace(
            from_logits=lambda *a, **kw: None
        ),
        make_next_draft_input=lambda **kw: SimpleNamespace(**kw),
    )
    load_functions(
        "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py",
        ["_forward_decode"],
        namespace,
        "DSparkWorkerV2",
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda *a: SimpleNamespace(current_stream=lambda: None),
    )
    view = GatedOverride(
        torch.tensor([[1, 2, 3], [9, 9, 9]]), torch.ones(2, dtype=torch.bool)
    )
    view.slots = torch.tensor([3, 1])
    view.generations = torch.tensor([8, 9])
    view._query = Mock()
    extension = TestExtension(view)
    proposer = SimpleNamespace(_hybrid_spec_extension=extension)
    proposer.propose = lambda **kw: splice_proposal(
        make_proposal(torch.tensor([[1, 8, 8], [4, 5, 6]]), folded=False), proposer
    )
    planner = SimpleNamespace(
        mode_value="static",
        resolve_verify_token_budget=lambda **kw: None,
        schedule_layout=lambda **kw: None,
        should_run_compact=lambda **kw: False,
    )
    observed = {}

    def run_verify(**kw):
        observed["verify"] = kw["verify_ids_2d"].clone()
        return SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=torch.empty(8, 100), hidden_states=torch.empty(8, 2)
            ),
            can_run_cuda_graph=False,
        )

    def accept(**kw):
        # Target oracle accepts SA row fully; neural row fails after one draft.
        assert kw["draft_tokens"].tolist() == [[1, 2, 3], [4, 5, 6]]
        observed["accept"] = True
        return SimpleNamespace(
            out_tokens=torch.tensor([[1, 2, 3, 10], [4, 11, -1, -1]]),
            commit_lens=torch.tensor([4, 2], dtype=torch.int32),
            correct_len=torch.tensor([3, 1]),
            bonus=torch.tensor([10, 11]),
            new_seq_lens=torch.tensor([14, 22]),
            cap_trim_lens=torch.zeros(2, dtype=torch.int32),
        )

    def commit(**kw):
        assert observed["accept"]
        observed["commit"] = kw["commit_lens"].clone()

    noop = lambda *a, **kw: None
    worker = SimpleNamespace(
        draft_extension=extension,
        device="cpu",
        _draft_is_moe=False,
        _simulate_acc_greedy=False,
        _observers=SimpleNamespace(
            begin_step=noop,
            segment=lambda *a: contextlib.nullcontext(),
            observe_verify_step=lambda **kw: observed.update(
                confidence=kw["confidence"]
            ),
        ),
        _draft_context=contextlib.nullcontext,
        _block_pos_offsets=None,
        verify_num_draft_tokens=4,
        _proposer=proposer,
        _verify_planner=planner,
        _dp_verify_tier_num_tokens=lambda batch: 0,
        _verify_epilogue=None,
        _verify_executor=SimpleNamespace(
            verify_epilogue=None,
            begin_simulated_step=noop,
            run_non_compact=run_verify,
            accept_and_finalize=accept,
            commit_hidden=commit,
        ),
        model_runner=SimpleNamespace(
            ngram_embedding_manager=SimpleNamespace(update_after_verify=noop)
        ),
        target_worker=SimpleNamespace(model_runner=SimpleNamespace(model=object())),
        _commit_target_mamba_states_after_verify=noop,
    )
    batch = SimpleNamespace(
        spec_info=DraftInput(),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([10, 20]),
        seq_lens_cpu=torch.tensor([10, 20]),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        has_grammar=False,
        req_pool_indices=torch.tensor([3, 1]),
        reqs=[],
        forward_iter=1,
        spec_verify_tier_num_tokens=0,
        return_logprob=False,
    )
    published = []
    result = decode(
        namespace["_forward_decode"], worker, batch, lambda lens: published.append(lens)
    )
    assert observed["verify"].tolist() == [[42, 1, 2, 3], [42, 4, 5, 6]]
    assert observed["commit"].tolist() == [4, 2]
    assert observed["confidence"] is None
    assert published[0] is result.new_seq_lens
    args, kw = extension.feedback[0]
    assert args[2].tolist() == [1, 2, 3, 10, 4, 11, -1, -1]
    assert args[3].tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert args[4].tolist() == [4, 2]
    assert args[5].tolist() == [10, 20]
    assert args[6] is True
    assert kw["use_retrieval"].tolist() == [True, False]
    assert extension.pending_view is None


def test_decode_exception_idle_and_lifecycle():
    view = SimpleNamespace(_query=Mock())
    extension = TestExtension(view)
    worker = SimpleNamespace(draft_extension=extension)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False), seq_lens=torch.tensor([10])
    )

    def fail(*a, **kw):
        raise RuntimeError("model failed")

    with pytest.raises(RuntimeError, match="model failed"):
        decode(fail, worker, batch)
    assert extension.pending_view is None
    assert not extension.feedback
    batch.forward_mode.is_idle = lambda: True
    assert decode(lambda *a: "idle", worker, batch) == "idle"
    _finished(None, worker, rid="a", natural_stop=True)
    extension.close_request.assert_called_once_with("a", retain=True)
    _reset(None, worker)
    extension.reset.assert_called_once()
    result = SimpleNamespace(internal_state={})
    _diagnostics(
        result, SimpleNamespace(scheduler=SimpleNamespace(draft_worker=worker))
    )
    assert "hybrid_spec" in result.internal_state
    extension.begin_prefill = Mock(return_value="ticket")
    extension.on_prefill = Mock()
    extension.cancel_prefill = Mock()
    assert _prefill(lambda *a: "result", worker, batch) == "result"
    extension.on_prefill.assert_called_once_with(batch, "result", "ticket")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("accept_len", [1, 2, 4])
def test_cuda_dense_feedback_round_trip(accept_len):
    from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
    from hybrid_spec.integrations.sglang.device_extension import DeviceOverride

    cfg = config(num_neural_gate_tokens=0)
    backend = GPUSuffixAutomaton(cfg, "cuda:0")
    # History includes the current anchor; feedback starts after that anchor.
    history = [1, 2, 3, 4, 5, 1, 2]
    host = backend.build_host(backend.prepare_host(history), 100)
    backend.initialize_device_slot(0, 1, 60)
    backend.import_host(0, host)
    view = DeviceOverride(
        backend, torch.tensor([0], device="cuda"), torch.tensor([1], device="cuda")
    )
    prefix = torch.tensor([6], device="cuda")
    view.prepare(prefix)
    assert view.tokens.tolist() == [[3, 4, 5]]
    assert view.use_retrieval.tolist() == [True]
    extension = SimpleNamespace(
        config=cfg, backend=backend, vocab_size=100, metrics=Metrics()
    )
    tokens = [3, 4, 5, 1][:accept_len]
    output = tokens + [-1] * (4 - accept_len)
    stage_feedback(
        extension,
        view,
        SimpleNamespace(
            next_token_ids=torch.tensor(output, device="cuda"),
            accept_lens=torch.tensor([accept_len], dtype=torch.int32, device="cuda"),
        ),
        prefix,
    )
    # Only the accept prefix (including bonus) enters SA, never rejected padding.
    view.prepare(torch.tensor([6 + accept_len], device="cuda"))
    assert backend.history[0, : 7 + accept_len].tolist() == history + tokens
    assert extension.metrics.snapshot()["retrieval_correct_drafts"] == accept_len - 1
