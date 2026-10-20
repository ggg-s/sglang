"""Bounded regression of device tier selection and compact target dispatch."""

import dataclasses
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from types import MethodType, SimpleNamespace as NS
from typing import Optional, Union

import pytest
import torch
from hybrid_spec.integrations.sglang.conditional_verify import (
    ConditionalVerifyBackend,
    DeviceVerifyCounters,
    DeviceVerifyPlan,
    packed_indices,
    padded_query_lens,
)
from torch.utils._python_dispatch import TorchDispatchMode

from test_compact_verify import compact_module as compact_module
from test_compact_verify import cuda_modules as cuda_modules
from test_compact_verify import dynamic_config
from test_draft_recurrence import load_functions


class NoScalarRead(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("device scalar escaped to the host")
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("floor", [0, 16])
def test_device_plan_rounds_after_consensus_and_counts_without_scalar_read(floor):
    grid = [4, 9, 12, 16, 18, 27, 36]
    counters = DeviceVerifyCounters(grid, "cpu")
    lens = torch.tensor([4, 3, 2], dtype=torch.int32)
    with NoScalarRead():
        plan = DeviceVerifyPlan.create(lens, torch.tensor(grid), 27, floor)
        counters.add(plan, 27)
    expected = max(9, floor)
    assert plan.graph_tokens.item() == expected
    assert counters.snapshot() == {
        "verify_ct": 1,
        "logical_token_ct": 9,
        "target_token_ct": expected,
        "dense_token_ct": 27,
        "graph_token_histogram": {str(expected): 1},
        "conditional_verify_graph": True,
    }


def test_pack_map_preserves_request_order_and_marks_padding():
    with NoScalarRead():
        indices, rows, columns, valid = packed_indices(
            torch.tensor([4, 0, 2, 0], dtype=torch.int32), 9, 9
        )
    assert indices[valid].tolist() == [0, 1, 2, 3, 18, 19]
    assert rows[valid].tolist() == [0, 0, 0, 0, 2, 2]
    assert columns[valid].tolist() == [0, 1, 2, 3, 0, 1]
    assert valid.tolist() == [True] * 6 + [False] * 3


def test_sparse_grid_distributes_query_padding_only_after_live_requests():
    with NoScalarRead():
        padded = padded_query_lens(torch.tensor([9, 0, 0, 0]), 27, 9)
    assert padded.tolist() == [9, 6, 6, 6]


def test_prices_reuse_storage_until_that_context_changes():
    from hybrid_spec.selection.online_verify import DynamicVerify

    chooser = DynamicVerify(dynamic_config(), "cpu")
    key = (2, 512)
    grid, before = chooser.device_prices(key, [4, 9, 18])
    assert torch.isinf(before).all()
    assert chooser.device_prices(key, [4, 9, 18])[1] is before
    chooser.observe_cost((1, 512), 9, 1, 2)
    assert chooser.device_prices(key, [4, 9, 18])[1] is before
    chooser.observe_cost(key, 9, 1, 2)
    after_grid, after = chooser.device_prices(key, [4, 9, 18])
    assert after_grid is grid
    assert after.tolist() == [float("inf"), 3, float("inf")]
    assert chooser.device_prices(key, [4, 9, 18])[1] is after
    # A new observation refreshes every grid in the cell's shared fixed term.
    chooser.observe_cost(key, 18, 2, 4)
    assert chooser.device_prices(key, [9, 18])[1].tolist() == pytest.approx([3.1, 5.1])


def test_tensor_policy_fallback_does_not_extract_selected_budget():
    from hybrid_spec.selection.online_verify import DynamicVerify

    chooser = DynamicVerify(dynamic_config(), "cpu")
    view = NS(
        use_retrieval=torch.tensor([True, False]),
        query_metadata=torch.tensor([[4, 0], [0, 0]]),
    )
    with NoScalarRead():
        lens = chooser.select(view, torch.tensor([8, 3]), [4, 9, 12, 18], (2, 512))
    assert lens.tolist() == [9, 4]


def test_conditional_verify_is_opt_in_and_has_distinct_cost_fingerprint():
    from hybrid_spec.selection.verify_costs import cost_profile, parse_cost_profile

    config = dynamic_config()
    old = cost_profile(config, {}, {})
    assert "conditional_verify_graph" not in old["config"]
    with pytest.raises(ValueError, match="conditional_verify_graph requires"):
        replace(config, conditional_verify_graph=True)
    enabled = replace(
        config, conditional_draft_graph=True, conditional_verify_graph=True
    )
    with pytest.raises(ValueError, match="does not match hybrid config"):
        parse_cost_profile(old, enabled)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_timing_retains_async_tier_label_until_completion():
    from hybrid_spec.selection.online_verify import DynamicVerify

    chooser = DynamicVerify(dynamic_config(), "cpu")
    chooser.model.device = torch.device("cuda")
    chooser.start = torch.cuda.Event(enable_timing=True)
    chooser.start.record()
    tier = torch.tensor(9, device="cuda", dtype=torch.int64)
    with NoScalarRead():
        chooser.mark_verify(tier)
    label = chooser.graph_tokens
    assert label.device.type == "cpu" and label.is_pinned()
    tier.fill_(18)
    end = torch.cuda.Event(enable_timing=True)
    # A little GPU work keeps the measured interval positive on fast devices.
    torch.arange(1024, device="cuda").square_()
    end.record()
    chooser.timings.append(
        ((1, 512), label, chooser.start, chooser.verify_start, end, 1, end)
    )
    torch.cuda.synchronize()
    chooser.poll_timings()
    assert label.item() == 9
    assert 9 in chooser.costs[(1, 512)]
    assert 18 not in chooser.costs[(1, 512)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_target_dispatch_runs_only_selected_physical_tier(cuda_modules):
    """Real SWITCH and native ragged/scatter kernels; no weights or server.

    The model and KV translator are small deterministic GPU boundaries. The
    production backend captures and replays all packing/prep/forward/scatter
    work. Descending tiers and shrinking batches detect stale shared metadata.
    """
    from hybrid_spec.integrations.sglang.conditional_graph import (
        _switch_adapter,
        require_conditional_graph,
    )
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyLayout,
        build_capture_verify_lens,
    )

    require_conditional_graph("switch")
    backend = ConditionalVerifyBackend.__new__(ConditionalVerifyBackend)
    device, max_bs, width = "cuda", 4, 9
    grid = [4, 9, 12, 18, 27, 36]
    seq = torch.zeros(max_bs, dtype=torch.int64, device=device)
    req = torch.zeros_like(seq)
    target_rows = torch.zeros((), dtype=torch.int64, device=device)
    cache = torch.zeros(256, dtype=torch.float32, device=device)
    metadata = torch.zeros((), dtype=torch.int64, device=device)

    class Translator:
        def plan(self, **kwargs):
            def bind(fb, translator, cols):
                # A nonidentity v2p catches accidental packing of a stale
                # physical window. The cache store below uses these addresses.
                fb.out_cache_loc = torch.where(
                    cols >= 0, kwargs["write_virtual"][cols.clamp(min=0)] + 7, 0
                )

            return NS(bind=bind)

    layouts = {}
    namespace = dict()
    load_functions(
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
        ["_stage_ragged_verify_layout"],
        namespace,
        "DecodeCudaGraphRunner",
    )

    def prepare_metadata(fb, **kwargs):
        metadata.copy_(fb.seq_lens.sum() + fb.req_pool_indices.sum())

    runner = NS(
        max_bs=max_bs,
        captured_req_width=width,
        seq_len_fill_value=1,
        _captured_ragged_layouts=layouts,
        device_module=torch.cuda,
        model_runner=NS(kv_index_translator=Translator()),
        attn_backend=NS(init_forward_metadata_out_graph=prepare_metadata),
    )
    runner._stage_ragged_verify_layout = MethodType(
        namespace["_stage_ragged_verify_layout"], runner
    )
    backend.runner = runner
    backend.online_c128 = None
    backend.adapter = _switch_adapter()
    backend.grid = tuple(grid)
    backend.capacities = torch.tensor(grid, device=device)
    backend.counters = DeviceVerifyCounters(grid, device)
    backend.leaves, backend.outputs, backend.owners = {}, {}, []
    backend.pool = torch.cuda.graph_pool_handle()
    backend.groups = ()
    backend.parent = None
    backend.drafts = torch.zeros(max_bs * width, device=device, dtype=torch.long)
    backend.cache_locs = torch.zeros_like(backend.drafts)
    backend.lens = torch.zeros(max_bs, device=device, dtype=torch.int32)
    backend.seq_lens, backend.req_indices = seq, req
    backend.branch = torch.ones((), device=device, dtype=torch.int32)
    backend.delegate = NS(cleanup=lambda: None)
    stream = backend.stream = torch.cuda.Stream()

    @dataclass
    class Output:
        next_token_logits: torch.Tensor
        hidden_states: torch.Tensor
        full_logits: Optional[torch.Tensor] = None

    class ReadEnds(Enum):
        PRE_REPLAY = 0
        IN_REPLAY = 1
        POST_REPLAY = 2

    events = []
    runner.backend = backend
    runner.is_dllm = False
    runner._replay_attn_backend = lambda: runner.attn_backend
    runner._resolve_shared_read_ends = lambda *args: ReadEnds.IN_REPLAY
    runner.model_runner.device_timer = None
    runner._publish_read_done = lambda in_graph: events.append(in_graph)
    runner._process_output_after_replay = lambda output, fb: output

    def reject_host_load(*args):
        raise AssertionError("device-selected tier went through CPU load_batch")

    runner.load_batch = reject_host_load
    backend.delegate.replay_session = nullcontext
    ns = dict(
        Optional=Optional,
        Union=Union,
        ForwardBatch=object,
        PPProxyTensors=object,
        LogitsProcessorOutput=Output,
        dataclasses=dataclasses,
        device_timer_ctx=lambda *args: nullcontext(),
        SharedReadEnds=ReadEnds,
    )
    load_functions(
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
        ["execute"],
        ns,
        "DecodeCudaGraphRunner",
    )
    execute = MethodType(ns["execute"], runner)

    # capture_session's communicator scope is covered separately by TP tests.
    # Here only the unavailable distributed service boundary is replaced.
    with torch.cuda.stream(stream):
        for capacity in reversed(grid):
            slots = min(max_bs, capacity)
            layout = RaggedVerifyLayout.from_verify_lens(
                verify_lens_cpu=build_capture_verify_lens(
                    num_tokens=capacity, num_slots=slots, num_draft_tokens=width
                ),
                device=device,
                grid=grid,
            )
            layouts[capacity] = layout
            fb = NS(
                batch_size=slots,
                spec_info=NS(ragged_verify_layout=layout),
                input_ids=torch.zeros(capacity, dtype=torch.long, device=device),
                positions=torch.zeros(capacity, dtype=torch.long, device=device),
                out_cache_loc=torch.zeros(capacity, dtype=torch.long, device=device),
                seq_lens=torch.zeros(slots, dtype=torch.long, device=device),
                req_pool_indices=torch.zeros(slots, dtype=torch.long, device=device),
            )

            def forward(fb=fb, capacity=capacity):
                target_rows.add_(capacity)
                logits = fb.input_ids.float() + fb.positions + metadata
                cache.index_copy_(0, fb.out_cache_loc, logits)
                return Output(logits[:, None], (logits * 2)[:, None])

            backend.capture_one(NS(size=capacity), forward, capture_inputs=fb)
        backend.parent = torch.cuda.CUDAGraph()
        with torch.cuda.graph(backend.parent, stream=stream):
            backend.adapter.capture_switch(
                backend.branch.data_ptr(),
                [backend.leaves[n].raw_cuda_graph() for n in grid],
                stream.cuda_stream,
            )
    torch.cuda.current_stream().wait_stream(stream)
    target_rows.zero_()
    expected_rows, expected_logical, expected_dense = 0, 0, 0
    cases = [[9, 4, 9, 4], [4], [2, 3], [4, 4, 4], [9, 9, 9, 9], [2]]
    try:
        for lengths in cases:
            bs = len(lengths)
            lens = torch.tensor(lengths, dtype=torch.int32, device=device)
            with NoScalarRead():
                plan = DeviceVerifyPlan.create(lens, backend.capacities, bs * width)
            inputs = torch.arange(bs * width, device=device) + 20
            prefixes = torch.arange(bs, device=device) + 10
            request_indices = torch.arange(bs, device=device) + 30
            virtual = torch.arange(bs * width, device=device) + 1
            fb = NS(
                conditional_verify_plan=plan,
                conditional_verify_write_virtual=virtual,
                batch_size=bs,
                seq_lens=prefixes,
                req_pool_indices=request_indices,
                input_ids=inputs,
                forward_mode=NS(name="TARGET_VERIFY"),
            )
            with NoScalarRead():
                out = execute(fb)
            physical = next(n for n in grid if n >= sum(lengths))
            expected_rows += physical
            expected_logical += sum(lengths)
            expected_dense += bs * width
            slots = min(max_bs, physical)
            meta = sum(range(10, 10 + bs)) + (slots - bs) + sum(range(30, 30 + bs))
            expected = torch.zeros(bs, width, device=device)
            for row, length in enumerate(lengths):
                expected[row, :length] = (
                    inputs.view(bs, width)[row, :length]
                    + 10
                    + row
                    + torch.arange(length, device=device)
                    + meta
                )
            torch.testing.assert_close(
                out.next_token_logits.flatten(), expected.flatten()
            )
            torch.testing.assert_close(
                out.hidden_states.flatten(), expected.flatten() * 2
            )
            valid = torch.arange(width, device=device)[None, :] < lens[:, None]
            torch.testing.assert_close(
                cache[(virtual + 7).view(bs, width)[valid]], expected[valid]
            )
        assert target_rows.item() == expected_rows
        stats = backend.counters.snapshot()
        assert stats["target_token_ct"] == expected_rows
        assert stats["logical_token_ct"] == expected_logical
        assert stats["dense_token_ct"] == expected_dense
        assert expected_rows < expected_dense
        assert events == [False] * len(cases)
    finally:
        torch.cuda.synchronize()
        backend.cleanup()
