"""DSPARK SA bridge contracts; model/accept kernels are explicit test boundaries."""

import ast
import contextlib
import copy
import enum
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
    _SplitProposalBuffers,
    decode,
    propose,
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


class Window(msgspec.Struct):
    positions_2d: torch.Tensor
    verify_cache_loc: torch.Tensor
    verify_cache_loc_2d: torch.Tensor
    kv_loc_plan: object


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
    cfg.disable_cuda_graph = False
    cfg.enable_memory_saver = False
    cfg.speculative_use_rejection_sampling = False
    cfg.max_running_requests = 4
    validate_config(config(draft_graph_mode="split"), cfg)
    for invalid in (
        replace(config(), history_mode="cpu"),
        replace(config(), backend="arctic", history_mode="cpu"),
        replace(config(), draft_graph_mode="split", num_neural_gate_tokens=1),
        replace(config(), draft_graph_mode="split_no_skip"),
        replace(config(), draft_graph_mode="split", conditional_draft_graph=True),
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
    example = HybridConfig.from_file(
        str(ROOT / "experimental/hybrid-spec/configs/device-dspark-sa-split.json")
    )
    cfg.speculative_num_draft_tokens = example.num_drafts + 1
    validate_config(example, cfg)


def native_kv_types():
    """Load the actual plan class without importing serving/model dependencies."""
    path = ROOT / "python/sglang/srt/mem_cache/kv_loc_plan.py"
    tree = ast.parse(path.read_text())
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in ("IdSpaceKind", "TokenSpan", "KVLocPlan")
    ]
    namespace = dict(torch=torch, enum=enum, msgspec=msgspec, copy=copy.copy)
    import __future__

    exec(
        compile(
            ast.Module(body=classes, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["KVLocPlan"], namespace["IdSpaceKind"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("sample_from_anchor", [False, True])
@pytest.mark.parametrize("folded", [False, True])
def test_split_runs_only_neural_rows_through_native_proposer(
    monkeypatch, device, sample_from_anchor, folded
):
    """Repeated shape/mask changes must retain original KV ids and row ownership.

    Actual native propose/_run_forward and KVLocPlan execute. Only the model,
    Markov kernel, and ForwardBatch construction are lightweight boundaries.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    Plan, _ = native_kv_types()
    module_name = "sglang.srt.speculative.dspark_components.dspark_draft"
    dto_module = ModuleType(module_name)
    dto_module.DraftBlockResult, dto_module.DraftProposal = Block, Proposal
    monkeypatch.setitem(sys.modules, module_name, dto_module)

    gamma, bs = 3, 4
    q = gamma if sample_from_anchor else gamma + 1
    request_ids = torch.tensor([5, 1, 7, 3], device=device)
    seq_lens = torch.tensor([11, 23, 17, 29], device=device)
    virtual = torch.arange(bs * 4, device=device).reshape(bs, 4) + 20
    physical = virtual + 100
    source = SimpleNamespace(
        req_to_token=object(),
        space=lambda kind: SimpleNamespace(key="full", write=lambda ids: ids + 100),
    )
    plan = Plan(
        source=source,
        req_pool_indices=request_ids,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        write_virtual=virtual.flatten(),
        read_extent=4,
    )
    # A translated SWA cache and a stale full-batch read table must not leak
    # into a subset. The parent must remain reusable for target verification.
    plan._write_ids["swa"] = virtual.flatten() + 200
    plan._read_tables["full"] = object()
    selected = plan.select_rows(
        torch.tensor([3, 1], device=device), torch.tensor([3, 1])
    )
    assert selected.req_pool_indices.tolist() == [3, 1]
    assert selected.write_physical.tolist() == physical[[3, 1]].flatten().tolist()
    assert (
        selected._write_ids["swa"].tolist()
        == (virtual[[3, 1]] + 200).flatten().tolist()
    )
    assert selected._read_tables == {} and plan._read_tables

    window = Window(
        torch.arange(4, device=device)[None, :] + seq_lens[:, None],
        virtual.flatten(),
        virtual,
        plan,
    )
    batch = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        req_pool_indices=request_ids,
        reqs=["a", "b", "c", "d"],
    )
    draft_input = SimpleNamespace(
        bonus_tokens=torch.tensor([10, 20, 30, 40], device=device),
        nxt_kv_lens_cpu=seq_lens.cpu() + q,
        nxt_kv_lens_sum=int((seq_lens.cpu() + q).sum()),
    )
    sampler = SimpleNamespace(
        out=torch.empty(bs * gamma, dtype=torch.int64, device=device),
        params=SimpleNamespace(
            greedy_mask=torch.ones(bs, device=device, dtype=torch.bool),
            temperatures=torch.ones(bs, device=device),
        ),
        confidence_out=None,
        stage_sampling_params=lambda **kw: None,
    )
    cache = torch.full((200,), -1, dtype=torch.int64, device=device)
    calls = []
    graphs = []

    def model_forward(fb):
        ids = fb.req_pool_indices
        calls.append(ids.tolist())
        expected_rows = [
            batch.reqs.index({5: "a", 1: "b", 7: "c", 3: "d"}[i]) for i in ids.tolist()
        ]
        assert (
            fb.positions.reshape(-1, q).tolist()
            == window.positions_2d[expected_rows, :q].tolist()
        )
        assert (
            fb.out_cache_loc.tolist() == physical[expected_rows, :q].flatten().tolist()
        )
        assert (
            fb.kv_loc_plan.seq_lens_cpu.tolist()
            == seq_lens.cpu()[expected_rows].tolist()
        )
        assert fb.kv_loc_plan._read_tables == {}

        def calculate():
            values = ids[:, None] * 10 + torch.arange(q, device=device)[None, :]
            cache[fb.out_cache_loc] = values.flatten()
            start = 0 if sample_from_anchor else 1
            sampler.out[: len(ids) * gamma].copy_(values[:, start:].flatten())
            return values.float().reshape(-1, 1)

        if folded and device == "cuda":
            calculate()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                raw = calculate()
            graph.replay()
            graphs.append(graph)
        else:
            raw = calculate()
        return SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=raw), can_run_graph=folded
        )

    namespace = dict(
        torch=torch,
        ScheduleBatch=object,
        DFlashDraftInputV2=object,
        VerifyWindow=object,
        DraftProposal=Proposal,
        DraftBlockResult=Block,
        DraftForwardResult=SimpleNamespace,
        ForwardBatch=SimpleNamespace,
        ForwardMode=SimpleNamespace(TARGET_VERIFY="verify"),
        CaptureHiddenMode=SimpleNamespace(NULL="null"),
        SpeculativeAlgorithm=SimpleNamespace(DSPARK="DSPARK"),
        Optional=__import__("typing").Optional,
        unwrap_lora_layer=lambda x: x,
        _make_num_token_non_padded=lambda *a: None,
        envs=SimpleNamespace(
            SGLANG_DSPARK_FOLDED_PROPOSAL=SimpleNamespace(get=lambda: folded),
            SGLANG_DSPARK_EMBED_IN_GRAPH=SimpleNamespace(get=lambda: False),
        ),
    )
    load_functions(
        "python/sglang/srt/speculative/dspark_components/dspark_draft.py",
        ["select_draft_hidden_without_anchor"],
        namespace,
    )
    load_functions(
        "python/sglang/srt/speculative/dspark_components/dspark_draft.py",
        ["propose", "_run_forward"],
        namespace,
        "DraftBlockProposer",
    )
    namespace["sample_draft_block"] = lambda **kw: Block(
        kw["base_logits"][:, :, 0].long(),
        None,
        torch.ones(
            kw["base_logits"].shape[0],
            dtype=torch.bool,
            device=device,
        ),
        torch.ones(kw["base_logits"].shape[0], device=device),
        True,
    )
    view = GatedOverride(
        torch.tensor([[81, 82, 83]] * bs, device=device),
        torch.zeros(bs, dtype=torch.bool, device=device),
    )
    counters = dict.fromkeys(
        (
            "proposal_ct",
            "request_ct",
            "neural_request_ct",
            "skip_request_ct",
            "all_sa_ct",
            "mixed_ct",
        ),
        0,
    )
    extension = SimpleNamespace(
        config=config(draft_graph_mode="split"),
        pending_view=view,
        dspark_draft_counters=counters,
        dspark_split_buffers=_SplitProposalBuffers(bs, gamma, device, 0),
    )
    proposer = SimpleNamespace(
        _hybrid_spec_extension=extension,
        _mask_token_id=0,
        gamma=gamma,
        query_token_num=q,
        sample_from_anchor=sample_from_anchor,
        _draft_sampler=sampler,
        _draft_block_ids_buf=None,
        _draft_block_spec_info=object(),
        _greedy_draft=True,
        _noise=None,
        _tp_sync=None,
        _fill_dp_moe_sync_metadata=lambda *a: None,
        _base_logits_context=contextlib.nullcontext,
        draft_model=SimpleNamespace(
            embed_tokens=lambda x: x.float().unsqueeze(-1),
            compute_base_logits=lambda x: (x, None),
            markov_head=None,
            confidence_head=None,
        ),
        draft_model_runner=SimpleNamespace(
            forward=model_forward,
            decode_cuda_graph_runner=SimpleNamespace(can_run_graph=lambda fb: folded),
            kv_index_translator=source,
        ),
    )
    proposer._run_forward = lambda **kw: namespace["_run_forward"](proposer, **kw)
    kwargs = dict(
        batch=batch,
        draft_input=draft_input,
        verify_window=window,
        bs=bs,
        device=device,
        target_model=SimpleNamespace(
            get_input_embeddings=lambda: proposer.draft_model.embed_tokens
        ),
        sampling_info=None,
    )
    for step, mask in enumerate(
        (
            [True] * bs,
            [False] * bs,
            [True, False, True, False],
            [False, True, True, True],
            [True] * bs,
        )
    ):
        cache.fill_(-1)
        calls.clear()
        view.use_retrieval.copy_(torch.tensor(mask, device=device))
        if step >= 3:
            batch.seq_lens_cpu = None  # overlap's native nxt_kv_lens_cpu fallback
        if step == 4:

            def veto(view):
                assert view.use_retrieval.tolist() == [True] * bs
                agreed = view.use_retrieval.clone()
                agreed[-1] = False
                return SimpleNamespace(tokens=view.tokens, use_retrieval=agreed)

            view.tp_sync = SimpleNamespace(synchronize=veto)
            mask[-1] = False
        # No-miss path uses the parent's plan unchanged, including its cache.
        plan._read_tables.clear()
        result = propose(namespace["propose"], proposer, **kwargs)
        misses = [i for i, hit in enumerate(mask) if not hit]
        expected = view.tokens.clone()
        start = 0 if sample_from_anchor else 1
        for i in misses:
            expected[i] = request_ids[i] * 10 + torch.arange(
                start, start + gamma, device=device
            )
        assert torch.equal(result.draft_block.draft_tokens, expected)
        assert calls == ([request_ids[misses].tolist()] if misses else [])
        assert result.folded == (folded and len(misses) == bs)
        assert result.draft_block_ids[:, 0].tolist() == [10, 20, 30, 40]
        assert result.confidence is None and result.draft_block.corrected_logits is None
        written = physical[misses, :q].flatten()
        assert int((cache != -1).sum()) == len(misses) * q
        assert bool((cache[written] != -1).all())
        assert batch.req_pool_indices.tolist() == [5, 1, 7, 3]
        assert draft_input.bonus_tokens.tolist() == [10, 20, 30, 40]
        assert plan.write_virtual.tolist() == virtual.flatten().tolist()
    assert counters == dict(
        proposal_ct=5,
        request_ct=20,
        neural_request_ct=8,
        skip_request_ct=12,
        all_sa_ct=1,
        mixed_ct=3,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_single_request_split_reuses_buffers_across_hit_miss_transitions(
    monkeypatch, device
):
    """Row 0 is a real miss; scratch reuse must not retain stale SA/anchor data."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    module_name = "sglang.srt.speculative.dspark_components.dspark_draft"
    dto_module = ModuleType(module_name)
    dto_module.DraftBlockResult, dto_module.DraftProposal = Block, Proposal
    monkeypatch.setitem(sys.modules, module_name, dto_module)
    buffers = _SplitProposalBuffers(4, 3, device, 0)
    view = GatedOverride(
        torch.empty((1, 3), dtype=torch.int64, device=device),
        torch.empty(1, dtype=torch.bool, device=device),
    )
    counters = dict.fromkeys(
        (
            "proposal_ct",
            "request_ct",
            "neural_request_ct",
            "skip_request_ct",
            "all_sa_ct",
            "mixed_ct",
        ),
        0,
    )
    extension = SimpleNamespace(
        config=config(draft_graph_mode="split"),
        pending_view=view,
        dspark_draft_counters=counters,
        dspark_split_buffers=buffers,
    )
    proposer = SimpleNamespace(_hybrid_spec_extension=extension)
    neural_calls = []

    def original(proposer, **kwargs):
        neural_calls.append(kwargs["bs"])
        return make_proposal(torch.tensor([[1, 2, 3]], device=device))

    for step, hit in enumerate([True, False, True, False, True]):
        view.use_retrieval.fill_(hit)
        view.tokens.fill_(step + 10)
        before = view.tokens.clone()
        result = propose(
            original,
            proposer,
            bs=1,
            draft_input=SimpleNamespace(
                bonus_tokens=torch.tensor([step], device=device)
            ),
        )
        expected = [[step + 10] * 3] if hit else [[1, 2, 3]]
        assert result.draft_block.draft_tokens.tolist() == expected
        assert result.folded == (not hit)
        assert torch.equal(view.tokens, before)
        if hit:
            assert result.draft_block.draft_tokens.data_ptr() == view.tokens.data_ptr()
            assert result.draft_block_ids.data_ptr() == buffers.block_ids.data_ptr()
            assert (
                result.draft_block.greedy_mask.data_ptr()
                == buffers.greedy_mask.data_ptr()
            )
            assert (
                result.draft_block.temperatures.data_ptr()
                == buffers.temperatures.data_ptr()
            )
            assert result.draft_block_ids.tolist() == [[step, 0, 0]]
    assert neural_calls == [1, 1]
    assert counters == dict(
        proposal_ct=5,
        request_ct=5,
        neural_request_ct=2,
        skip_request_ct=3,
        all_sa_ct=3,
        mixed_ct=0,
    )


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


@pytest.mark.parametrize("mode", ["full", "split", "all_sa_split"])
def test_native_decode_uses_sa_and_preserves_verify_commit_order(monkeypatch, mode):
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
        alloc_verify_window=None,
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
    if mode != "full":
        module_name = "sglang.srt.speculative.dspark_components.dspark_draft"
        dto_module = ModuleType(module_name)
        dto_module.DraftBlockResult, dto_module.DraftProposal = Block, Proposal
        monkeypatch.setitem(sys.modules, module_name, dto_module)
        extension.config = config(draft_graph_mode="split")
        extension.dspark_split_buffers = _SplitProposalBuffers(2, 3, "cpu", 0)
        extension.dspark_draft_counters = dict.fromkeys(
            (
                "proposal_ct",
                "request_ct",
                "neural_request_ct",
                "skip_request_ct",
                "all_sa_ct",
                "mixed_ct",
            ),
            0,
        )
    if mode == "all_sa_split":
        view.tokens[1] = torch.tensor([4, 5, 6])
    elif mode == "split":
        view.use_retrieval[1] = False
    proposer = SimpleNamespace(_hybrid_spec_extension=extension, _mask_token_id=0)

    def model_proposal(proposer, **kw):
        ids = kw["batch"].req_pool_indices
        return make_proposal(
            torch.stack(
                [
                    torch.tensor([1, 8, 8]) if int(i) == 3 else torch.tensor([4, 5, 6])
                    for i in ids
                ]
            ),
            folded=False,
        )

    proposer.propose = lambda **kw: propose(model_proposal, proposer, **kw)
    Plan, _ = native_kv_types()

    def allocate(**kw):
        prefix = kw["batch"].seq_lens
        positions = prefix[:, None] + torch.arange(4)[None, :]
        cache = torch.arange(8).reshape(2, 4)
        source = SimpleNamespace(
            space=lambda kind: SimpleNamespace(key="full", write=None)
        )
        plan = Plan(
            source=source,
            req_pool_indices=kw["batch"].req_pool_indices,
            seq_lens=prefix,
            seq_lens_cpu=prefix.clone(),
            write_virtual=cache.flatten(),
        )
        return Window(positions, cache.flatten(), cache, plan)

    namespace["alloc_verify_window"] = allocate
    draft_input = DraftInput()
    draft_input.bonus_tokens = torch.tensor([42, 42])
    draft_input.nxt_kv_lens_cpu = None
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
        spec_info=draft_input,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([10, 20]),
        seq_lens_cpu=torch.tensor([10, 20]),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        has_grammar=False,
        req_pool_indices=torch.tensor([3, 1]),
        reqs=["a", "b"],
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
    assert kw["use_retrieval"].tolist() == [True, mode == "all_sa_split"]
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
