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
def test_switch_replay_selects_exact_tail_and_preserves_private_allocations(batch_size):
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
    expected_histogram = [0] * (batch_size + 1)
    mixed = 0
    for raw_bs in sorted({1, max(1, batch_size - 1), batch_size}):
        for size in [*range(raw_bs + 1), *range(raw_bs, -1, -1)]:
            count.fill_(size)
            output.fill_(-1)
            # Allocate/release outside the graph to exercise pool retention.
            poison = torch.full((4096,), 999, dtype=torch.int64, device="cuda")
            dispatcher.replay(raw_bs)
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
    parse_cost_profile(cost_profile(switch, costs, fixed), switch)
