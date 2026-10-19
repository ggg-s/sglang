"""Standalone CUDA dispatch probe, with no serving/attention import substitutes.

The tail is a small GEMM plus gather/scatter, not an LLM. Timing is only a dispatch
microbenchmark, with a completion fence after each round in both modes.
"""

import json
import statistics
import time
from types import SimpleNamespace as NS

import torch
from hybrid_spec.integrations.sglang.compact_graph import (
    CompactCapture,
    _capture_graph,
    prepare_compaction,
    replay_compact,
)
from hybrid_spec.integrations.sglang.conditional_graph import ConditionalTailGraph


def make_record(batch_size, conditional):
    stream = torch.cuda.Stream()
    mask = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
    output = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    source = torch.arange(batch_size, device="cuda") * 3 + 7
    features = source[:, None].to(torch.float16).expand(-1, 256).contiguous()
    weight = torch.eye(256, dtype=torch.float16, device="cuda")
    prefix, _ = _capture_graph(lambda: output.fill_(-1), stream)
    record = CompactCapture(
        prefix,
        output,
        NS(batch_size=batch_size),
        NS(use_retrieval=mask),
        (source, features, weight),
    )
    prepare_compaction(record, "split")
    for size in range(1, batch_size + 1):

        def tail(size=size):
            rows = record.rows[:size]
            result = features.index_select(0, rows) @ weight
            output.index_copy_(0, rows, result[:, 0].long())

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            tail()
            tail()
        torch.cuda.synchronize()
        if conditional:
            record.tails[size] = NS(forward=tail, graph=None)
        else:
            graph, _ = _capture_graph(tail, stream)
            record.tails[size] = NS(forward=None, graph=graph)
    if conditional:
        record.conditional = ConditionalTailGraph(record, stream)
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
    return record, counters


def probe(batch_size, conditional):
    record, counters = make_record(batch_size, conditional)
    expected_histogram = [0] * (batch_size + 1)
    expected_mixed = 0
    checked = 0
    for raw_bs in sorted({1, batch_size, max(1, batch_size - 1)}):
        # Every possible tail size, with both leading and trailing neural rows.
        for size in range(raw_bs + 1):
            for reverse in (False, True):
                chosen = (
                    list(range(size))
                    if not reverse
                    else list(range(raw_bs - size, raw_bs))
                )
                record.override.use_retrieval.fill_(True)
                record.override.use_retrieval[chosen] = False
                replay_compact(record, counters, raw_bs)
                assert record.output.cpu().tolist() == [
                    i * 3 + 7 if i in chosen else -1 for i in range(batch_size)
                ]
                expected_histogram[size] += 1
                expected_mixed += int(0 < size < raw_bs)
                checked += 1
    if conditional:
        assert record.conditional.snapshot() == (expected_histogram, expected_mixed)
    # Compare a half-neural batch with the same final synchronization in both
    # modes. This is not a serving throughput or model latency measurement.
    record.override.use_retrieval.copy_(
        torch.arange(batch_size, device="cuda") % 2 == 0
    )
    for _ in range(20):
        replay_compact(record, counters, batch_size)
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(100):
            replay_compact(record, counters, batch_size)
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 100)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        for _ in range(10):
            replay_compact(record, counters, batch_size)
        torch.cuda.synchronize()
    item_ct = sum(
        event.count for event in profile.key_averages() if event.key == "aten::item"
    )
    assert item_ct == (0 if conditional else 10)
    result = {
        "batch_size": batch_size,
        "conditional": conditional,
        "checked_rounds": checked,
        "item_ct_in_10_rounds": item_ct,
        "dispatch_us_median": statistics.median(samples),
        "dispatch_us_samples": samples,
    }
    if record.conditional is not None:
        record.conditional.reset()
    record.prefix.reset()
    for tail in record.tails.values():
        if tail.graph is not None:
            tail.graph.reset()
        tail.forward = None
    record.tails.clear()
    return result


if __name__ == "__main__":
    results = [
        probe(bs, conditional) for bs in (1, 4, 16, 32) for conditional in (False, True)
    ]
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "results": results,
            },
            indent=2,
        )
    )
