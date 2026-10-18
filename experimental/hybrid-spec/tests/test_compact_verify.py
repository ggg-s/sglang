"""Hybrid verify contracts; CUDA cases use native packing/accept/commit kernels.

Serving dependencies are isolated at import/forward-batch boundaries. Neural
weights and a full scheduler are not exercised by this component suite.
"""

import ast
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from typing import Optional

import pytest
import torch
from hybrid_spec.config import HybridConfig
from hybrid_spec.integrations.sglang.compact_verify import token_buckets
from test_draft_recurrence import load_functions

ROOT = Path(__file__).resolve().parents[3]


def dynamic_config(**kwargs):
    return HybridConfig(
        8,
        2,
        16,
        128,
        history_mode="device",
        draft_graph_mode="split",
        num_neural_drafts=3,
        verify_mode="compact",
        dynamic_verify=True,
        **kwargs,
    )


def test_online_feedback_censors_tail_and_separates_sources():
    """Neither a cropped chain nor descendants of a mismatch are failures."""
    from hybrid_spec.selection.online_verify import OnlineAcceptance

    model = OnlineAcceptance(dynamic_config(), "cpu")
    view = NS(
        use_retrieval=torch.tensor([False, True, True, True]),
        query_metadata=torch.tensor([[99, 1], [4, 0], [4, 1], [17, 0]]),
    )
    buckets = model.buckets(view)
    assert buckets.tolist() == [0, 1, 6, 4]
    model.update(buckets, torch.tensor([3, 8, 8, 2]), torch.tensor([4, 4, 1, 3]))
    assert model.check_ct[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert model.correct_ct[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert model.check_ct[1].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert model.correct_ct[1].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert model.check_ct[6].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert not model.correct_ct[6].any()
    assert model.check_ct[4].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    before = model.check_ct.clone()
    model.update(buckets, torch.ones(4), torch.tensor([0, 3, -1, 999]))
    assert torch.equal(model.check_ct, before)
    scores = model.survival(buckets)
    assert (scores[:, 1:] <= scores[:, :-1]).all()
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("aliased_capacity", [False, True])
def test_dynamic_lengths_match_exhaustive_prefix_cost_oracle(aliased_capacity):
    """Global allocation must maximise utility over real physical graph costs."""
    import itertools

    from hybrid_spec.selection.online_verify import DynamicVerify

    chooser = DynamicVerify(dynamic_config(), "cpu")
    view = NS(
        use_retrieval=torch.tensor([True, False, True]),
        query_metadata=torch.tensor([[4, 0], [0, 0], [4, 1]]),
    )
    chooser.model.check_ct.fill_(1000)
    chooser.model.correct_ct.fill_(950)
    chooser.model.correct_ct[0].fill_(200)
    chooser.model.correct_ct[6].fill_(600)
    key = (3, 512)
    costs = (
        {6: 1.0, 9: 4.0, 12: 0.001} if aliased_capacity else {6: 1.0, 9: 1.1, 12: 4.0}
    )
    for n, cost in costs.items():
        chooser.observe_cost(key, n, 0.4, cost)
    caps = torch.tensor([3, 1, 2] if aliased_capacity else [3, 3, 3])
    lens = chooser.select(view, caps, list(costs), key) - 1
    scores = chooser.model.survival(chooser.model.buckets(view))

    def objective(lengths):
        physical = next(n for n in costs if n >= 3 + sum(lengths))
        utility = 3 + sum(scores[i, :n].sum().item() for i, n in enumerate(lengths))
        return utility / (0.4 + costs[physical])

    best = max(
        objective(x)
        for x in itertools.product(*(range(1, n + 1) for n in caps.tolist()))
    )
    assert objective(lens.tolist()) == pytest.approx(best)
    if aliased_capacity:
        # The apparently cheap 12-token graph cannot actually be selected:
        # all available candidates fit in 9, which is the expensive tier.
        assert lens.tolist() == [1, 1, 1]
    else:
        assert lens[0] > lens[1]
        # Extra logical tokens within the same physical graph should be used.
        assert sum(lens).item() + 3 == 9
    chooser.feedback(lens + 1)
    before = chooser.model.check_ct.clone()
    chooser.feedback(lens + 1)  # A duplicated/idle callback cannot label twice.
    assert torch.equal(before, chooser.model.check_ct)


def test_dynamic_cold_start_exploration_and_abort_reset():
    from hybrid_spec.selection.online_verify import DynamicVerify

    chooser = DynamicVerify(dynamic_config(dynamic_verify_explore_interval=2), "cpu")
    view = NS(
        use_retrieval=torch.tensor([True, False]),
        query_metadata=torch.zeros((2, 2), dtype=torch.long),
    )
    caps = torch.tensor([8, 3])
    key = (2, 512)
    assert chooser.select(view, caps, [4, 8, 12, 18], key).tolist() == [9, 4]
    # Cropping cannot start just because a small graph has a cheap cost sample.
    chooser.observe_cost(key, 4, 0, 0.01)
    assert chooser.select(view, caps, [4, 8, 12, 18], key).tolist() == [9, 4]
    chooser.model.check_ct.fill_(100)
    chooser.model.correct_ct.zero_()
    chooser.observe_cost(key, 18, 0, 100)
    seen = []
    for _ in range(20):
        lengths = chooser.select(view, caps, [4, 8, 12, 18], key) - 1
        seen.append(lengths.tolist())
    assert [8, 3] in seen and [1, 1] in seen
    before = chooser.model.check_ct.clone()
    chooser.begin(NS(reqs=[]))  # Failed iteration must not train a new batch.
    chooser.feedback(torch.tensor([1, 1]))
    assert torch.equal(before, chooser.model.check_ct)


def test_cuda_dynamic_plugin_labels_before_history_consumer_and_flushes(cuda_modules):
    from hybrid_spec import plugin

    config = dynamic_config()
    extension = NS(config=config, backend=NS(device="cuda"), tp_sync=None)
    extension.pending_view = NS(
        tokens=torch.zeros((2, 8), dtype=torch.long, device="cuda"),
        use_retrieval=torch.tensor([True, False], device="cuda"),
        query_metadata=torch.tensor([[8, 0], [0, 0]], device="cuda"),
    )
    extension.on_device_verify = lambda result, batch: setattr(
        extension, "pending_view", None
    )
    extension.reset = lambda: setattr(extension, "pending_view", None)
    worker = NS(
        draft_extension=extension,
        target_worker=NS(model_runner=NS(decode_cuda_graph_runner=None)),
    )
    batch = NS(
        reqs=[NS(origin_input_ids=[1], output_ids=[2])] * 2,
        seq_lens=torch.tensor([1, 1], device="cuda"),
        forward_mode=NS(is_idle=lambda: False),
    )
    plugin._begin_draft(worker, batch)
    vi = NS(
        draft_token_num=9,
        retrieve_next_token=torch.arange(1, 10, device="cuda").repeat(2, 1),
    )
    plugin._configure_verify(vi, worker, batch)
    controller = extension.dynamic_verify
    plugin._verify(
        NS(can_run_cuda_graph=False, accept_lens=torch.tensor([3, 2], device="cuda")),
        NS(draft_worker=worker, target_worker=worker.target_worker),
        batch,
    )
    assert controller.model.correct_ct.sum().item() == 3
    assert controller.model.check_ct.sum().item() == 5
    assert extension.pending_view is None
    # Native draft-extend runs after _verify. Its time belongs to this cycle;
    # only the next draft boundary closes the sample.
    assert not controller.timings
    plugin._begin_draft(worker, batch)
    torch.cuda.synchronize()
    assert controller.snapshot()["cost_sample_ct"] == 1
    controller.completed = (
        controller.context_key,
        13,
        controller.start,
        controller.start,
    )
    idle = NS(forward_mode=NS(is_idle=lambda: True))
    plugin._begin_draft(worker, idle)
    assert controller.completed is None
    controller.completed = (controller.context_key, 13, None, None)
    plugin._prefill(
        lambda *a, **kw: None,
        NS(draft_worker=NS(draft_extension=NS(dynamic_verify=controller))),
        batch,
    )
    assert controller.completed is None
    plugin._reset(None, NS(draft_worker=worker))
    assert extension.dynamic_verify is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dynamic_verify": 1},
        {"dynamic_verify_min_samples": 0},
        {"dynamic_verify_explore_interval": 0},
        {"dynamic_verify_prior_strength": float("nan")},
        {"dynamic_verify_prior_strength": 1e300},
        {"dynamic_verify_prior_rate": 1},
        {"dynamic_verify_min_drafts": 4},
        {"partial_drafts": "zero"},
    ],
)
def test_dynamic_invalid_config(kwargs):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(dynamic_config(), **kwargs)


@pytest.mark.parametrize("bs", [1, 4, 16, 33])
def test_cuda_fused_online_policy_and_feedback_match_cpu_reference(cuda_modules, bs):
    """One-block policy and >256-position fallback must share the same labels."""
    from dataclasses import replace

    from hybrid_spec.selection.online_verify import DynamicVerify

    config = replace(dynamic_config(dynamic_verify_explore_interval=2), max_requests=64)
    cpu = DynamicVerify(config, "cpu")
    gpu = DynamicVerify(config, "cuda")
    rng = torch.Generator().manual_seed(7301 + bs)
    mask = torch.arange(bs) % 3 != 0
    metadata = torch.tensor([[i % 40, i % 2] for i in range(bs)])
    caps = torch.where(mask, 8, 3).int()
    key = (bs, 512)
    grid = token_buckets(config, [bs * 9], bs)
    choices = [n for n in grid if bs * 2 <= n <= bs * 9]
    for iteration in range(8):
        check = torch.randint(0, 100, cpu.model.check_ct.shape, generator=rng)
        correct = (check * torch.rand(check.shape, generator=rng)).long()
        cpu.model.check_ct.copy_(check)
        cpu.model.correct_ct.copy_(correct)
        gpu.model.check_ct.copy_(check)
        gpu.model.correct_ct.copy_(correct)
        if iteration:
            for n in choices:
                cost = 1 + ((n * 7) % 23) / 10
                cpu.observe_cost(key, n, 0.2, cost)
                gpu.observe_cost(key, n, 0.2, cost)

        def view(device):
            return NS(
                tokens=torch.zeros((bs, 8), dtype=torch.long, device=device),
                use_retrieval=mask.to(device),
                query_metadata=metadata.to(device),
            )

        expected = cpu.select(view("cpu"), caps, choices, key)
        actual = gpu.select(view("cuda"), caps.cuda(), choices, key)
        assert actual.cpu().tolist() == expected.tolist()
        num_accept = torch.minimum(expected, torch.arange(bs).int() % 5 + 1)
        cpu.feedback(num_accept)
        gpu.feedback(num_accept.cuda())
        assert torch.equal(gpu.model.check_ct.cpu(), cpu.model.check_ct)
        assert torch.equal(gpu.model.correct_ct.cpu(), cpu.model.correct_ct)


def load_module(monkeypatch, name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def compact_module(monkeypatch):
    return load_module(
        monkeypatch,
        "sglang.srt.speculative.eagle_compact_verify",
        "python/sglang/srt/speculative/eagle_compact_verify.py",
    )


@pytest.mark.parametrize("neural,gate", [(1, 0), (2, 3), (3, 1), (8, 8)])
def test_source_lengths_config_and_token_grid(neural, gate):
    config = HybridConfig(
        8,
        2,
        16,
        128,
        history_mode="device",
        draft_graph_mode="split",
        num_neural_drafts=neural,
        num_neural_gate_tokens=gate,
        verify_mode="compact",
    )
    for custom in ((), (16, 32)):
        from dataclasses import replace

        cfg = replace(config, verify_token_buckets=custom)
        grid = token_buckets(cfg, [9, 18, 36], 4)
        for bs in range(1, 5):
            for num_sa in range(bs + 1):
                total = num_sa * 9 + (bs - num_sa) * (neural + 1)
                selected = next(n for n in grid if n >= total)
                assert total <= selected <= bs * 9
        if not custom:
            assert neural + 1 in grid and 9 in grid
            assert all(
                bs * (neural + 1) + n * (8 - neural) in grid
                for bs in range(1, 5)
                for n in range(bs + 1)
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_neural_drafts": 0},
        {"num_neural_drafts": 9},
        {"num_neural_drafts": True},
        {"verify_mode": "unknown"},
        {"verify_token_buckets": [0]},
        {"num_neural_drafts": 3},
        {"verify_token_buckets": [4]},
    ],
)
def test_invalid_budgets_fail_before_cuda(kwargs):
    with pytest.raises(ValueError):
        HybridConfig(8, 2, 4, 128, **kwargs)


def test_cutoff_does_not_accept_zero_filled_tail(compact_module):
    vi = NS(
        draft_token_num=9,
        retrieve_next_token=torch.tensor([list(range(1, 9)) + [-1]] * 2),
    )
    compact_module.cap_verify_chain(vi, torch.tensor([3, 9]))
    assert vi.retrieve_next_token[0].tolist() == [1, 2] + [-1] * 7
    assert vi.retrieve_next_token[1].tolist() == list(range(1, 9)) + [-1]


def test_capture_loop_uses_token_tiers_instead_of_multiplying_by_width():
    seen = []
    ns = dict(
        torch=torch,
        Optional=Optional,
        get_parallel=lambda: NS(tp_rank=1, tp_group=None),
        get_available_gpu_memory=lambda *a, **kw: 8,
        _set_capture_lora_variant=lambda x: None,
        _set_capture_attention_variant=lambda x: None,
        torch_compile_decoration=NS(patch_model=lambda *a, **kw: nullcontext(None)),
    )
    load_functions(
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
        ["_capture_one_stream"],
        ns,
        "DecodeCudaGraphRunner",
    )
    runner = NS(
        model_runner=NS(device="cuda", gpu_id=0, model=None),
        ragged_verify_mode=True,
        capture_bs=[1, 2, 4],
        capture_num_tokens=[4, 9, 13, 18, 36],
        compile_bs=[],
        captured_req_width=9,
        record_nolora_graph=False,
        attention_graph_variants=None,
        _set_profile_trace_tag=lambda *a: None,
        capture_one_shape=lambda size, *a: seen.append(size),
    )
    ns["_capture_one_stream"](runner)
    assert seen == [36, 18, 13, 9, 4]


@pytest.fixture
def cuda_modules(monkeypatch, compact_module):
    if not torch.cuda.is_available():
        pytest.skip("requires NVIDIA CUDA")
    packages = (
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.speculative",
        "sglang.kernels.ops.speculative.dspark",
        "sglang.srt",
        "sglang.srt.speculative",
        "sglang.srt.managers",
        "sglang.srt.mem_cache",
        "sglang.srt.model_executor",
    )
    for name in packages:
        module = ModuleType(name)
        module.__path__ = [str(ROOT / "python" / Path(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, module)
    utils = ModuleType("sglang.srt.utils")
    for name, value in dict(
        is_cpu=False,
        is_cuda=True,
        is_hip=False,
        is_musa=False,
        is_npu=False,
        is_xpu=False,
    ).items():
        setattr(utils, name, lambda value=value: value)
    utils.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    environ = ModuleType("sglang.srt.environ")
    environ.envs = NS(SGLANG_RAGGED_VERIFY_MODE=NS(get=lambda: "compact"))
    monkeypatch.setitem(sys.modules, environ.__name__, environ)
    schedule = ModuleType("sglang.srt.managers.schedule_batch")
    schedule.ScheduleBatch = object
    monkeypatch.setitem(sys.modules, schedule.__name__, schedule)
    for name in (
        "cache_locs",
        "ragged_verify_kernels",
        "dspark.dispatch",
        "dspark.dspark_verify_window",
        "spec_tree",
        "eagle",
    ):
        full = "sglang.kernels.ops.speculative." + name
        load_module(monkeypatch, full, "python/" + full.replace(".", "/") + ".py")
    return compact_module


@pytest.mark.parametrize("neural,gate", [(3, 1), (2, 3), (1, 8), (8, 8)])
@pytest.mark.parametrize("coarse", [False, True])
@pytest.mark.parametrize("alignment", [1, 8])
@pytest.mark.parametrize("dynamic", [False, True])
def test_cuda_mixed_verify_graph_tier_switches_and_native_commit(
    monkeypatch, cuda_modules, neural, gate, coarse, alignment, dynamic
):
    from hybrid_spec.integrations.sglang.compact_verify import configure_verify
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        BuildRaggedVerifyWindow,
    )
    from sglang.kernels.ops.speculative.eagle import prepare_verify_commit_outputs
    from sglang.kernels.ops.speculative.spec_tree import (
        verify_tree_greedy_kernel_triton,
    )
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    config = HybridConfig(
        8,
        2,
        16,
        128,
        history_mode="device",
        draft_graph_mode="split",
        num_neural_drafts=neural,
        num_neural_gate_tokens=gate,
        verify_mode="compact",
        verify_token_buckets=(16, 32, 144) if coarse else (),
        dynamic_verify=dynamic,
        dynamic_verify_explore_interval=2,
    )
    grid = token_buckets(config, [9, 18, 36, 72, 144], 16, alignment)
    cache_table = torch.arange(32 * 128, device="cuda").view(32, 128)
    platform = ModuleType("sglang.srt.platforms")
    platform.current_platform = NS(is_out_of_tree=lambda: False)
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    # Execute the actual runner's in-place refresh, not a replacement thereof.
    ns = {"torch": torch}
    load_functions(
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
        ["_stage_ragged_verify_layout"],
        ns,
        "DecodeCudaGraphRunner",
    )
    graph_runner = NS(
        capture_num_tokens=grid, _captured_ragged_layouts={}, captured_req_width=9
    )
    worker = NS(
        draft_extension=NS(config=config),
        target_worker=NS(model_runner=NS(decode_cuda_graph_runner=graph_runner)),
    )
    graphs = {}
    visited = []
    baseline_total = 0
    patterns = [
        (1, 0),
        (1, 1),
        (4, 0),
        (4, 2),
        (2, 1),
        (8, 3),
        (16, 7),
        (4, 4),
        (1, 0),
        (16, 0),
        (2, 0),
        (1, 1),
    ]
    for iteration, (bs, num_sa) in enumerate(patterns):
        mask = torch.tensor([i < num_sa for i in range(bs)], device="cuda").roll(
            iteration % bs
        )
        worker.draft_extension.pending_view = NS(
            use_retrieval=mask,
            tokens=torch.zeros((bs, 8), device="cuda", dtype=torch.long),
            query_metadata=torch.tensor([[4, i % 2] for i in range(bs)], device="cuda"),
        )
        req = torch.arange(bs, device="cuda").roll(iteration % bs)
        prefix = torch.arange(bs, device="cuda", dtype=torch.int32) + 5 + iteration
        dense = (
            (req[:, None] * 11 + torch.arange(9, device="cuda") + iteration) % 127
        ).long()
        # Force a mismatch on alternate steps; other steps must reach the cap.
        if iteration % 2:
            dense[0, 2] = (dense[0, 2] + 7) % 127
        next_node = torch.arange(1, 10, device="cuda").expand(bs, -1).clone()
        next_node[:, -1] = -1
        vi = NS(
            draft_token_num=9,
            draft_token=dense.flatten(),
            retrieve_next_token=next_node,
        )
        batch = NS(
            seq_lens=prefix,
            req_pool_indices=req,
            device="cuda",
            forward_mode=NS(is_idle=lambda: False),
            reqs=[NS(origin_input_ids=[1] * 16, output_ids=[]) for _ in range(bs)],
        )
        if dynamic:
            from hybrid_spec.integrations.sglang.compact_verify import (
                get_dynamic_verify,
            )

            chooser = get_dynamic_verify(worker.draft_extension, "cuda")
            chooser.begin(batch)
            # Different sources have different target correctness. The native
            # verifier below supplies the next feedback; model calls are real.
            if iteration == 0:
                chooser.model.check_ct.fill_(100)
                chooser.model.correct_ct.fill_(95)
                chooser.model.correct_ct[0].fill_(40)
            for n in grid:
                chooser.observe_cost(chooser.context_key, n, 0.2, 0.2 + n * 0.1)
        configure_verify(vi, worker, batch)
        layout = vi.compact_verify_layout
        bucket = layout.graph_num_tokens
        visited.append(bucket)
        baseline_total += ((bs * 9 + alignment - 1) // alignment) * alignment
        window = BuildRaggedVerifyWindow.execute(
            batch=batch,
            layout=layout,
            draft_block_ids=dense[:, :-1].contiguous(),
            draft_tokens=dense[:, 1:].contiguous(),
            bs=bs,
            device="cuda",
            verify_num_draft_tokens=9,
            model_runner=NS(req_to_token_pool=NS(req_to_token=cache_table)),
        )
        if bucket not in graphs:
            slots = min(bucket, 16)
            cap_layout = RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=torch.ones(slots, dtype=torch.int32, device="cuda"),
                graph_num_tokens=bucket,
            )
            graph_runner._captured_ragged_layouts[bucket] = cap_layout
            ids = torch.zeros(bucket, dtype=torch.long, device="cuda")
            positions = torch.zeros_like(ids)
            kv = torch.zeros_like(ids)

            # Each captured graph reads the stable row boundaries. A stale
            # layout must change logits/hidden and fail the independent oracle.
            def target(ids=ids, positions=positions, kv=kv, cap_layout=cap_layout):
                row = torch.bucketize(
                    torch.arange(ids.numel(), device="cuda"),
                    cap_layout.qo_indptr_device[1:],
                    right=True,
                )
                hidden = torch.stack((ids, positions, kv, row), dim=1).float()
                logits = torch.nn.functional.one_hot((ids + 1) % 127, 127).float()
                return NS(next_token_logits=logits, hidden_states=hidden)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    target()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = target()
            graphs[bucket] = (
                graph,
                ids,
                positions,
                kv,
                out,
                cap_layout.verify_lens.data_ptr(),
            )
        graph, ids, pos, kv, out, ptr = graphs[bucket]
        ns["_stage_ragged_verify_layout"](graph_runner, layout, bucket)
        assert (
            graph_runner._captured_ragged_layouts[bucket].verify_lens.data_ptr() == ptr
        )
        ids.copy_(window.verify_ids)
        pos.copy_(window.positions)
        kv.copy_(window.verify_cache_loc)
        graph.replay()
        # Preserve graph-owned outputs; scatter mutates the wrapper only.
        logits = NS(
            next_token_logits=out.next_token_logits, hidden_states=out.hidden_states
        )
        cuda_modules.scatter_verify_output(vi, logits)
        lens = vi.verify_lens.tolist()
        for row, width in enumerate(lens):
            start = row * 9
            assert (
                logits.hidden_states[start : start + width, 0].long().tolist()
                == dense[row, :width].tolist()
            )
            assert logits.hidden_states[
                start : start + width, 1
            ].long().tolist() == list(range(int(prefix[row]), int(prefix[row]) + width))
            assert (
                logits.hidden_states[start : start + width, 2].long().tolist()
                == cache_table[req[row], prefix[row] : prefix[row] + width].tolist()
            )
            assert (
                logits.hidden_states[start : start + width, 3].long().tolist()
                == [row] * width
            )
        retrieve = torch.arange(bs * 9, device="cuda").view(bs, 9)
        predict = torch.zeros(bs * 9, device="cuda", dtype=torch.int32)
        indices = torch.full((bs, 9), -1, device="cuda", dtype=torch.int32)
        correct = torch.empty(bs, device="cuda", dtype=torch.int32)
        verify_tree_greedy_kernel_triton[(bs,)](
            predict,
            indices,
            correct,
            dense,
            retrieve,
            vi.retrieve_next_token,
            torch.full_like(retrieve, -1),
            logits.next_token_logits.argmax(-1).view(bs, 9),
            batch_size=bs,
            num_speculative_tokens=9,
            num_draft_tokens=9,
        )
        accept = correct + 1
        if dynamic:
            before = chooser.model.check_ct.sum().item()
            chooser.feedback(accept)
            assert chooser.model.check_ct.sum().item() > before
        new_seq, bonus, _, draft_inputs = prepare_verify_commit_outputs(
            predict,
            indices,
            accept,
            prefix,
            num_draft_tokens=9,
        )
        select = draft_inputs[1]
        for row, cap in enumerate(lens):
            count = 1
            while (
                count < cap and dense[row, count] == (dense[row, count - 1] + 1) % 127
            ):
                count += 1
            assert accept[row].item() == count
            assert indices[row, :count].tolist() == list(
                range(row * 9, row * 9 + count)
            )
            assert (indices[row, count:] == -1).all()
            assert bonus[row].item() == (dense[row, count - 1].item() + 1) % 127
            assert new_seq[row].item() == prefix[row].item() + count
            assert select[row].item() == row * 9 + count - 1
    assert len(set(visited)) >= 3
    assert (
        worker.draft_extension.compact_verify_counters["logical_token_ct"]
        <= worker.draft_extension.compact_verify_counters["target_token_ct"]
    )
    assert (
        worker.draft_extension.compact_verify_counters["target_token_ct"]
        <= baseline_total
    )


@pytest.mark.parametrize(
    "standalone,with_graph", [(False, False), (False, True), (True, True)]
)
def test_cuda_prepare_packs_native_kv_window_without_changing_commit_layout(
    monkeypatch, cuda_modules, standalone, with_graph
):
    from enum import Enum

    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    class Mode(Enum):
        TARGET_VERIFY = 1
        IDLE = 2

        def is_idle(self):
            return self is Mode.IDLE

    class Hidden(Enum):
        NULL = 0
        FULL = 2

    class Spec:
        def __init__(self, spec_input_type):
            self.spec_input_type = spec_input_type

    prepared = []

    class ForwardBatch:
        @staticmethod
        def init_new(batch, runner, **kwargs):
            plan = kwargs["kv_loc_plan"]
            cols = kwargs["write_cols"]
            fb = NS(
                input_ids=batch.input_ids,
                positions=batch.spec_info.positions,
                spec_info=batch.spec_info,
                req_pool_indices=batch.req_pool_indices,
                seq_lens=batch.seq_lens,
                kv_loc_plan=plan,
                kv_loc_cols=cols,
                out_cache_loc=torch.where(
                    cols >= 0, plan.write_virtual[cols.clamp_min(0)], 0
                ),
                capture_hidden_mode=kwargs["capture_hidden_mode"],
                ready=False,
            )
            fb.mark_forward_metadata_ready = lambda: setattr(fb, "ready", True)
            prepared.append(fb)
            return fb

    dependencies = {
        "sglang.srt.mem_cache.kv_index_translator": dict(KVIndexTranslator=object),
        "sglang.srt.mem_cache.kv_loc_plan": dict(
            Cols=object, IdSpaceKind=NS(FULL=0), KVLocPlan=object
        ),
        "sglang.srt.model_executor.forward_batch_info": dict(
            CaptureHiddenMode=Hidden, ForwardBatch=ForwardBatch, ForwardMode=Mode
        ),
        "sglang.srt.speculative.spec_info": dict(
            SpecInput=Spec, SpecInputType=NS(DFLASH_VERIFY=0)
        ),
        "sglang.srt.speculative.spec_utils": dict(
            prepare_mamba_track_for_verify=lambda batch: None
        ),
        "sglang.srt.platforms": dict(current_platform=NS(is_out_of_tree=lambda: False)),
    }
    for name, attrs in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    load_module(
        monkeypatch,
        "sglang.srt.speculative.dflash_info",
        "python/sglang/srt/speculative/dflash_info.py",
    )
    table = torch.arange(8 * 128, device="cuda").view(8, 128)
    req = torch.tensor([4, 1], device="cuda")
    prefix = torch.tensor([5, 8], device="cuda", dtype=torch.int32)
    plan = NS(write_virtual=torch.cat((table[4, 5:14], table[1, 8:17])))
    dense = torch.arange(18, device="cuda").view(2, 9)
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor([9, 3], device="cuda", dtype=torch.int32),
        graph_num_tokens=13,
    )
    vi = NS(
        topk=1,
        kv_loc_plan=plan,
        draft_token_num=9,
        draft_token=dense.flatten(),
        compact_verify_layout=layout,
    )
    batch = NS(
        input_ids=None,
        req_pool_indices=req,
        seq_lens=prefix,
        seq_lens_cpu=None,
        seq_lens_sum=None,
        mamba_track_indices=None,
        hisparse_coordinator=None,
        forward_mode=Mode.TARGET_VERIFY,
        device="cuda",
        spec_info=vi,
    )
    seen = []
    graph = (
        NS(can_run_graph=lambda fb: True, load_batch=lambda fb: seen.append(fb))
        if with_graph
        else None
    )
    runner = NS(
        req_to_token_pool=NS(req_to_token=table),
        spec_algorithm=NS(is_standalone=lambda: standalone),
        decode_cuda_graph_runner=graph,
        attn_backend=NS(init_forward_metadata=lambda fb: seen.append(fb)),
    )
    fb, can_graph = cuda_modules.prepare_compact_verify(
        vi, batch, NS(model_runner=runner)
    )
    assert can_graph == with_graph and fb.ready == with_graph
    assert fb.capture_hidden_mode == (Hidden.NULL if standalone else Hidden.FULL)
    assert fb.input_ids.tolist() == list(range(9)) + [9, 10, 11, 0]
    assert fb.positions.tolist() == list(range(5, 14)) + [8, 9, 10, 0]
    assert fb.out_cache_loc.tolist() == plan.write_virtual[:12].tolist() + [0]
    assert fb.kv_loc_cols.tolist() == list(range(12)) + [-1]
    assert batch.spec_info is vi and batch.input_ids is vi.draft_token
    assert batch.out_cache_loc is plan.write_virtual
    assert fb.kv_loc_plan is plan
    assert len(seen) == 1 and seen[0] is fb


def test_cuda_ragged_triton_attention_matches_independent_torch_reference(
    monkeypatch, cuda_modules
):
    """Replay real attention with changed row boundaries, padding and graph tiers."""
    from types import MethodType
    from typing import Optional

    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    utils = sys.modules["sglang.srt.utils"]
    utils.is_gfx95_supported = lambda: False
    utils.is_gfx1250_supported = lambda: False
    package = ModuleType("sglang.kernels.ops.attention")
    package.__path__ = [str(ROOT / "python/sglang/kernels/ops/attention")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    stride_ns = {}
    load_functions(
        "python/sglang/kernels/ops/attention/decode_attention.py",
        ["_extract_kv_strides"],
        stride_ns,
    )
    for name, attrs in {
        "sglang.kernels.ops.attention.decode_attention": stride_ns,
        "sglang.kernels.ops.attention.prefill_attention": {
            "context_attention_fwd": None
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    load_module(
        monkeypatch,
        "sglang.kernels.ops.attention.score_mod",
        "python/sglang/kernels/ops/attention/score_mod.py",
    )
    attention = load_module(
        monkeypatch,
        "sglang.kernels.ops.attention.extend_attention",
        "python/sglang/kernels/ops/attention/extend_attention.py",
    )
    ns = dict(torch=torch, Optional=Optional, SpecInput=object, KVLocPlan=object)
    load_functions(
        "python/sglang/srt/layers/attention/triton_backend.py",
        ["_target_verify_num_tokens_per_req", "_update_target_verify_buffers"],
        ns,
        "TritonAttnBackend",
    )
    torch.manual_seed(20261018)
    prefix_k = torch.randn(128, 2, 32, device="cuda", dtype=torch.float16)
    prefix_v = torch.randn_like(prefix_k)
    graphs = {}
    patterns = [
        ([9, 3], 13),
        ([3, 9], 13),
        ([3], 4),
        ([9, 9], 18),
        ([3, 3, 9, 3], 18),
        ([3, 9], 13),
        ([3], 4),
        ([9], 16),
    ]
    for iteration, (lens, tier) in enumerate(patterns):
        bs = len(lens)
        slots = min(tier, 4)
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor(lens, device="cuda", dtype=torch.int32),
            graph_num_tokens=tier,
        )
        prefix_lens_cpu = [3 + iteration + row for row in range(bs)] + [0] * (
            slots - bs
        )
        prefix_lens = torch.tensor(prefix_lens_cpu, device="cuda", dtype=torch.int32)
        indptr = torch.tensor(
            [0] + list(torch.tensor(prefix_lens_cpu).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
        if tier not in graphs:
            backend = NS(
                device="cuda",
                qo_indptr=torch.zeros(slots + 1, device="cuda", dtype=torch.int32),
                mask_indptr=torch.zeros(slots + 1, device="cuda", dtype=torch.int32),
                window_kv_indptr=None,
                sliding_window_size=None,
                _verify_mask=None,
                cuda_graph_kv_indices=torch.zeros(
                    128, device="cuda", dtype=torch.int64
                ),
            )
            backend._target_verify_num_tokens_per_req = MethodType(
                ns["_target_verify_num_tokens_per_req"], backend
            )
            q, k, v = [
                torch.empty(tier, 2, 32, device="cuda", dtype=torch.float16)
                for _ in range(3)
            ]
            out = torch.zeros_like(q)
            kv_ptr = torch.zeros(slots + 1, device="cuda", dtype=torch.int32)
            kv_indices = torch.arange(128, device="cuda")

            def forward(
                q=q,
                k=k,
                v=v,
                out=out,
                backend=backend,
                kv_ptr=kv_ptr,
                kv_indices=kv_indices,
            ):
                attention.extend_attention_fwd(
                    q,
                    k,
                    v,
                    out,
                    prefix_k,
                    prefix_v,
                    backend.qo_indptr,
                    kv_ptr,
                    kv_indices,
                    None,
                    True,
                    None,
                    9,
                    1.0,
                    1.0,
                )

            # Initialize valid layouts before warmup/capture.
            backend.qo_indptr.copy_(
                layout.padded_to_bucket(padded_bs=slots, cap=9).qo_indptr_device
            )
            kv_ptr.copy_(indptr)
            for x in (q, k, v):
                x.normal_()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                forward()
            graphs[tier] = backend, q, k, v, out, kv_ptr, graph, kv_indices
        backend, q, k, v, out, kv_ptr, graph, kv_indices = graphs[tier]
        for x in (q, k, v):
            x.normal_()
        kv_ptr.copy_(indptr)
        backend._fill_kv_indptr_and_indices = lambda *args: kv_ptr
        metadata = ns["_update_target_verify_buffers"](
            backend,
            None,
            slots,
            prefix_lens,
            NS(num_tokens_per_req=9, ragged_verify_layout=layout, custom_mask=None),
            torch.arange(slots, device="cuda"),
        )
        assert metadata[0] is not None
        graph.replay()
        start = 0
        prefix_start = 0
        for row, width in enumerate(lens):
            qs = q[start : start + width].float().transpose(0, 1)
            ks = (
                torch.cat(
                    (
                        prefix_k[prefix_start : prefix_start + prefix_lens_cpu[row]],
                        k[start : start + width],
                    )
                )
                .float()
                .transpose(0, 1)
            )
            vs = (
                torch.cat(
                    (
                        prefix_v[prefix_start : prefix_start + prefix_lens_cpu[row]],
                        v[start : start + width],
                    )
                )
                .float()
                .transpose(0, 1)
            )
            scores = qs @ ks.transpose(1, 2) / (32**0.5)
            visible = torch.arange(ks.shape[1], device="cuda")[None, :] <= (
                prefix_lens_cpu[row] + torch.arange(width, device="cuda")[:, None]
            )
            reference = (
                scores.masked_fill(~visible[None, :, :], -float("inf")).softmax(-1) @ vs
            ).transpose(0, 1)
            torch.testing.assert_close(
                out[start : start + width].float(), reference, atol=3e-3, rtol=3e-3
            )
            start += width
            prefix_start += prefix_lens_cpu[row]


@pytest.mark.parametrize("compact", [False, True])
def test_final_counters_include_real_graph_padding(compact):
    from hybrid_spec.integrations.sglang.compact_verify import finalize_verify_counters

    counters = {"target_token_ct": 12, "graph_token_histogram": {"12": 1}}
    extension = NS(pending_verify_tokens=12, compact_verify_counters=counters)
    finalize_verify_counters(
        extension,
        NS(can_run_cuda_graph=True),
        NS(
            ragged_verify_mode=compact,
            _ragged_graph_size=13,
            bs=2,
            captured_req_width=9,
        ),
    )
    expected = 13 if compact else 18
    assert counters == {
        "target_token_ct": expected,
        "graph_token_histogram": {str(expected): 1},
    }
    assert extension.pending_verify_tokens is None
    finalize_verify_counters(extension, NS(can_run_cuda_graph=False), None)
    assert counters["target_token_ct"] == expected


def test_cuda_source_budget_reuses_split_host_count(monkeypatch, cuda_modules):
    from hybrid_spec.integrations.sglang.compact_verify import configure_verify
    from hybrid_spec.plugin import _begin_draft

    config = HybridConfig(
        8,
        2,
        4,
        128,
        history_mode="device",
        draft_graph_mode="split",
        num_neural_drafts=3,
        verify_mode="compact",
    )
    extension = NS(
        config=config,
        compact_num_neural=1,
        pending_view=NS(use_retrieval=torch.tensor([True, False], device="cuda")),
    )
    worker = NS(
        draft_extension=extension,
        target_worker=NS(
            model_runner=NS(
                decode_cuda_graph_runner=NS(capture_num_tokens=[4, 9, 13, 18])
            )
        ),
    )
    vi = NS(
        draft_token_num=9,
        retrieve_next_token=torch.tensor([list(range(1, 9)) + [-1]] * 2, device="cuda"),
    )
    batch = NS(
        seq_lens=torch.tensor([10, 12], device="cuda"),
        forward_mode=NS(is_idle=lambda: False),
    )

    def fail_readback(*args, **kwargs):
        raise AssertionError("extra scalar D2H after split")

    with monkeypatch.context() as scope:
        scope.setattr(torch.Tensor, "item", fail_readback)
        configure_verify(vi, worker, batch)
    assert vi.compact_verify_layout.graph_num_tokens == 13
    assert vi.verify_lens.tolist() == [9, 4]
    _begin_draft(worker, batch)
    assert (
        extension.compact_num_neural is None
    )  # eager/next-batch cannot reuse a stale count


def test_cuda_zero_filled_tail_cannot_extend_acceptance(cuda_modules):
    from sglang.kernels.ops.speculative.spec_tree import (
        verify_tree_greedy_kernel_triton,
    )

    # After 126 the true target predicts zero. Padded logits also argmax to zero,
    # so without a cutoff the synthetic zero tail would appear entirely correct.
    dense = torch.tensor([[124, 125, 126, 0, 0, 0, 0, 0, 0]], device="cuda")
    target = torch.tensor([[125, 126, 0, 0, 0, 0, 0, 0, 0]], device="cuda")
    links = torch.tensor([list(range(1, 9)) + [-1]], device="cuda")
    vi = NS(draft_token_num=9, retrieve_next_token=links)
    cuda_modules.cap_verify_chain(vi, torch.tensor([3], device="cuda"))
    for next_node, expected in ((links, 8), (vi.retrieve_next_token, 2)):
        correct = torch.empty(1, device="cuda", dtype=torch.int32)
        indices = torch.full((1, 9), -1, device="cuda", dtype=torch.int32)
        predict = torch.zeros(9, device="cuda", dtype=torch.int32)
        verify_tree_greedy_kernel_triton[(1,)](
            predict,
            indices,
            correct,
            dense,
            torch.arange(9, device="cuda")[None, :],
            next_node,
            torch.full_like(links, -1),
            target,
            batch_size=1,
            num_speculative_tokens=9,
            num_draft_tokens=9,
        )
        assert correct.item() == expected


def test_capture_shape_preserves_nonmultiple_token_tiers():
    from typing import Callable

    class Prepared(Exception):
        pass

    seen = []

    def prepare(bs, *, num_tokens, **kwargs):
        seen.append((bs, num_tokens))
        raise Prepared

    ns = dict(
        Optional=Optional,
        Callable=Callable,
        get_exec=lambda: NS(graph=NS(debug_cuda_graph=False)),
    )
    load_functions(
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
        ["capture_one_shape"],
        ns,
        "DecodeCudaGraphRunner",
    )
    runner = NS(
        ragged_verify_mode=True,
        captured_req_width=9,
        capture_prepare=prepare,
        _ragged_capture_slots=lambda tokens: min(tokens, 4),
    )
    for tier in (4, 9, 13):
        with pytest.raises(Prepared):
            ns["capture_one_shape"](runner, tier, None)
    assert seen == [(4, 4), (4, 9), (4, 13)]


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("ragged", [False, True])
def test_inherited_capture_loop_without_target_initialization(tp_rank, ragged):
    """Draft subclasses skip target __init__ but execute its capture loop."""
    path = ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
    definition = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DecodeCudaGraphRunner"
    )
    definition.bases = []
    definition.body = [
        n
        for n in definition.body
        if isinstance(n, ast.Assign)
        or isinstance(n, ast.FunctionDef)
        and n.name == "_capture_one_stream"
    ]
    patched, captured = [], []

    def patch(model, compile_enabled, *, num_tokens, tp_group):
        patched.append((compile_enabled, num_tokens))
        return nullcontext(model)

    class Progress(list):
        def set_description(self, text):
            pass

    ns = dict(
        Optional=Optional,
        get_available_gpu_memory=lambda *a, **k: 1,
        get_parallel=lambda: NS(tp_rank=tp_rank, tp_group=None),
        tqdm=NS(tqdm=Progress),
        torch_compile_decoration=NS(patch_model=patch),
        _set_capture_lora_variant=lambda *a: None,
        _set_capture_attention_variant=lambda *a: None,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), ns)

    class DraftRunner(ns["DecodeCudaGraphRunner"]):
        def __init__(self):
            self.model_runner = NS(device="cuda", gpu_id=0, model=object())
            self.capture_bs = [1, 4]
            self.captured_req_width = 3
            self.compile_bs = [4]
            self.record_nolora_graph = False
            self.attention_graph_variants = None
            if ragged:
                self.ragged_verify_mode = True
                self.capture_num_tokens = [4, 12]

        def _set_profile_trace_tag(self, *args):
            pass

        def capture_one_shape(self, size, *args):
            captured.append(size)

    DraftRunner()._capture_one_stream()
    assert captured == ([12, 4] if ragged else [4, 1])
    assert patched == [(True, 12), (False, 4 if ragged else 3)]


@pytest.mark.parametrize("supported", [False, True])
def test_compact_backend_validation_waits_for_attention_initialization(
    monkeypatch, supported
):
    from hybrid_spec import plugin
    from hybrid_spec.integrations.sglang.compatibility import validate_model

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs.hybrid_arch",
        NS(mambaish_config=lambda config: None),
    )
    model = NS(
        hf_config=NS(architectures=["Qwen3ForCausalLM"]),
        model_is_mrope=False,
        is_multimodal=False,
        is_hybrid_swa=False,
    )
    config = dynamic_config()
    # Real worker constructors attach the extension before initializing any
    # target attention backend. Architecture validation must work at this point.
    validate_model(
        NS(
            target_worker=NS(model_runner=NS(model_config=model)),
            draft_runner=NS(model_config=model),
        ),
        config,
    )
    monkeypatch.setattr(plugin, "_config", lambda: config)
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.utils",
        NS(get_cuda_graph_batch_size_alignment=lambda **kwargs: 1),
    )
    runner = NS(
        model_runner=NS(is_draft_worker=False),
        attn_backend=NS(supports_ragged_verify_graph=supported),
        max_bs=4,
        require_gathered_buffer=False,
    )
    if supported:
        assert plugin._verify_token_buckets([9, 36], runner) == token_buckets(
            config, [9, 36], 4
        )
    else:
        with pytest.raises(ValueError, match="ragged-capable"):
            plugin._verify_token_buckets([9, 36], runner)


@pytest.mark.parametrize("alignment", [1, 2, 8, 16])
def test_token_tiers_obey_native_tp_sp_alignment(alignment):
    config = HybridConfig(
        8,
        2,
        16,
        128,
        history_mode="device",
        draft_graph_mode="split",
        num_neural_drafts=3,
        verify_mode="compact",
    )
    grid = token_buckets(config, [9, 18, 36, 72, 144], 16, alignment)
    assert all(n % alignment == 0 for n in grid)
    for bs in range(1, 17):
        full_aligned = ((bs * 9 + alignment - 1) // alignment) * alignment
        for num_sa in range(bs + 1):
            logical = bs * 4 + num_sa * 5
            selected = next(n for n in grid if n >= logical)
            assert logical <= selected <= full_aligned
