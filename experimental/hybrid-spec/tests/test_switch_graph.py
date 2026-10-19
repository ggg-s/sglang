"""Native SWITCH case indexing and graph/pool lifetime on real CUDA."""

from types import SimpleNamespace as NS

import pytest
import torch
from hybrid_spec.integrations.sglang.conditional_graph import (
    ConditionalTailGraph,
    require_conditional_graph,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("batch_size", [1, 3, 8, 32])
@pytest.mark.parametrize("batch_prepared", [False, True])
def test_switch_replay_selects_exact_tail_and_preserves_private_allocations(
    batch_size, batch_prepared
):
    """Catch off-by-one case selection, stale handles and recycled leaf buffers."""
    require_conditional_graph("switch")
    count = torch.zeros((), dtype=torch.int32, device="cuda")
    output = torch.full((batch_size,), -1, dtype=torch.int64, device="cuda")
    source = torch.arange(batch_size, device="cuda") + 17
    record = NS(inputs=NS(batch_size=batch_size), num_neural=count, tails={})
    for size in range(1, batch_size + 1):

        def tail(size=size):
            # A temporary allocation must remain protected by its leaf's pool.
            intermediate = source[:size] * size
            output[:size].copy_(intermediate)

        record.tails[size] = NS(forward=tail)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for tail in record.tails.values():
            tail.forward()
    torch.cuda.synchronize()
    dispatcher = ConditionalTailGraph(record, stream, dispatch="switch")
    from hybrid_spec.backends.gpu_sa.row_compaction import compact_neural_rows

    rows = torch.empty(batch_size, dtype=torch.int64, device="cuda")
    index = torch.arange(batch_size, device="cuda")
    expected_histogram = [0] * (batch_size + 1)
    mixed = 0
    for raw_bs in sorted({1, max(1, batch_size - 1), batch_size}):
        for size in [*range(raw_bs + 1), *range(raw_bs, -1, -1)]:
            count.fill_(size)
            if batch_prepared:
                # Poison the previous batch to detect a missing fused store.
                dispatcher.live_bs.fill_(-99)
                compact_neural_rows(
                    index >= size,
                    rows,
                    count,
                    raw_bs,
                    False,
                    live_bs=dispatcher.live_bs,
                )
            output.fill_(-1)
            # Allocate/release outside the graph to exercise pool retention.
            poison = torch.full((4096,), 999, dtype=torch.int64, device="cuda")
            dispatcher.replay(raw_bs, batch_prepared=batch_prepared)
            assert output.cpu().tolist() == [
                (row + 17) * size if row < size else -1 for row in range(batch_size)
            ]
            del poison
            expected_histogram[size] += 1
            mixed += 0 < size < raw_bs
    assert dispatcher.snapshot() == (expected_histogram, mixed)
    torch.cuda.synchronize()
    dispatcher.reset()


def test_switch_cost_profile_requires_matching_dispatch():
    """An IF seed table cannot silently price a different dispatcher."""
    from dataclasses import replace

    from hybrid_spec.config import HybridConfig
    from hybrid_spec.selection.verify_costs import (
        cost_profile,
        parse_cost_profile,
    )

    config = HybridConfig(8, 1, 8, 128, draft_graph_mode="split", history_mode="device")
    legacy = cost_profile(config, {}, {})
    assert "conditional_draft_graph_dispatch" not in legacy["config"]
    enabled = replace(config, conditional_draft_graph=True)
    profile = cost_profile(enabled, {}, {})
    switch = replace(enabled, conditional_draft_graph_dispatch="switch")
    with pytest.raises(ValueError, match="does not match hybrid config"):
        parse_cost_profile(profile, switch)
    costs, fixed = {(1, 512): {9: (0.2, 0.4)}}, {(1, 512): 0.2}
    fused = cost_profile(switch, costs, fixed)
    parse_cost_profile(fused, switch)
    joined = replace(switch, conditional_draft_graph_joined=True)
    with pytest.raises(ValueError, match="does not match hybrid config"):
        parse_cost_profile(fused, joined)
    parse_cost_profile(cost_profile(joined, costs, fixed), joined)
    fused["config"].pop("conditional_draft_graph_revision")
    with pytest.raises(ValueError, match="does not match hybrid config"):
        parse_cost_profile(fused, switch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("batch_size", [1, 3, 8, 32, 257])
def test_joined_replay_queues_distinct_raw_batches_without_host_scalar_race(batch_size):
    """Queued launches must retain their own raw_bs, including graph padding.

    A reused pinned scalar would make queued steps see the last host value.
    Accumulate per-row execution and histogram to catch that without fencing
    between successive launches. Prefix must run exactly once per replay.
    """
    from hybrid_spec.integrations.sglang.compact_graph import (
        CompactCapture,
        _capture_graph,
        prepare_compaction,
        replay_compact,
    )

    stream = torch.cuda.Stream()
    mask = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
    output = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    prefix_ct = torch.zeros((), dtype=torch.int64, device="cuda")

    def prefix_body():
        prefix_ct.add_(1)

    prefix, _ = _capture_graph(prefix_body, stream, keep_graph=True)
    record = CompactCapture(
        prefix, output, NS(batch_size=batch_size), NS(use_retrieval=mask), ()
    )
    prepare_compaction(record, "split")
    for size in range(1, batch_size + 1):

        def tail(size=size):
            output.index_add_(
                0,
                record.rows[:size],
                torch.ones(size, dtype=torch.int64, device="cuda"),
            )

        record.tails[size] = NS(forward=tail)
    with torch.cuda.stream(stream):
        for tail in record.tails.values():
            tail.forward()
    torch.cuda.synchronize()
    dispatcher = record.conditional = ConditionalTailGraph(
        record, stream, dispatch="switch", joined=True
    )
    output.zero_()
    prefix_ct.zero_()
    counters = {"prefix_ct": 0, "request_ct": 0}
    batches = [
        batch_size,
        1,
        max(1, batch_size - 1),
        min(batch_size, 3),
        1,
        batch_size,
    ] * 3
    expected = [sum(row < raw for raw in batches) for row in range(batch_size)]
    torch.cuda._sleep(20_000_000)
    for raw_bs in batches:
        replay_compact(record, counters, raw_bs)
    assert output.cpu().tolist() == expected
    assert prefix_ct.item() == len(batches)
    histogram, mixed = dispatcher.snapshot()
    assert histogram == [batches.count(n) for n in range(batch_size + 1)]
    assert mixed == 0
    assert counters == {"prefix_ct": len(batches), "request_ct": sum(batches)}
    torch.cuda.synchronize()
    dispatcher.reset()
    prefix.reset()
