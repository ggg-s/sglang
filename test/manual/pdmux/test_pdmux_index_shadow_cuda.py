"""CUDA observation checks; injected device assertions run in child processes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.multiplex.pdmux_index_shadow import INPUT_FIELDS, IndexShadow

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_readonly_shadows_with_prefill_and_decode_graph_replays():
    context = torch.cuda.GreenContext.create(num_sms=8)
    prefill, decode = context.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(prefill):
        inputs = NS(
            out_cache_loc=torch.arange(16384, device="cuda"),
            req_pool_indices=torch.arange(8, device="cuda"),
        )
        persistent = IndexShadow("target/P tp=0 split-prefill")
        persistent.capture(inputs, INPUT_FIELDS, prefix="batch")
    with torch.cuda.stream(decode):
        buffers = NS(
            out_cache_loc=torch.ones(64, device="cuda", dtype=torch.int64),
            req_pool_indices=torch.ones(8, device="cuda", dtype=torch.int64),
        )
        output = torch.empty_like(buffers.out_cache_loc)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=decode):
            output.copy_(buffers.out_cache_loc + buffers.req_pool_indices[0])
    torch.cuda.synchronize()
    for step in range(32):
        with torch.cuda.stream(prefill):
            persistent.check(f"before-segment-{step}")
        with torch.cuda.stream(decode):
            buffers.out_cache_loc.fill_(step)
            shadow = IndexShadow("target/D tp=0 graph")
            shadow.capture(buffers, INPUT_FIELDS, prefix="graph")
            graph.replay()
            shadow.check("after-graph-replay")
        with torch.cuda.stream(prefill):
            persistent.check(f"after-segment-{step}")
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 32))


def inject_corruption(field):
    owner, writer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(owner):
        inputs = NS(
            out_cache_loc=torch.arange(16, device="cuda"),
            req_pool_indices=torch.arange(2, device="cuda"),
        )
        shadow = IndexShadow("target/P tp=0 injected")
        shadow.capture(inputs, INPUT_FIELDS, prefix="batch")
        captured = owner.record_event()
    with torch.cuda.stream(writer):
        # Deliberate ordering for deterministic test injection only. The
        # production diagnostic never creates a cross-lane dependency.
        writer.wait_event(captured)
        getattr(inputs, field).view(-1).fill_(999999)
        changed = writer.record_event()
    with torch.cuda.stream(owner):
        owner.wait_event(changed)
        shadow.check("before-segment-1")
    torch.cuda.synchronize()


@pytest.mark.parametrize("field", INPUT_FIELDS)
def test_cross_lane_mutation_reports_the_field_in_an_async_assert(field):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), field],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0, result.stdout
    assert "PDMux index mutated" in result.stdout, result.stdout
    assert f"batch.{field}" in result.stdout, result.stdout


if __name__ == "__main__":
    inject_corruption(sys.argv[1])
