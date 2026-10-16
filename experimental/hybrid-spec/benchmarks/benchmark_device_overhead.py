"""Isolate hybrid feedback/metrics and stable graph-mapping staging on CUDA.

This is a component microbenchmark, not model throughput or TPOT. Feedback
measurements include the same pending-state reset in both implementations.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
from hybrid_spec.config import HybridConfig
from hybrid_spec.integrations.sglang.device_extension import DeviceDraftExtension
from hybrid_spec.observability.metrics import Metrics


def measure(step, iterations):
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    end.synchronize()
    return {
        "stream_us": start.elapsed_time(end) * 1000 / iterations,
        "wall_us": (time.perf_counter() - wall_start) * 1e6 / iterations,
    }


def compare(before, after, iterations, repeats):
    for _ in range(10):
        before()
        after()
    samples = {"before": [], "after": []}
    for repeat in range(repeats):
        order = (("before", before), ("after", after))
        for name, step in order if repeat % 2 == 0 else reversed(order):
            samples[name].append(measure(step, iterations))
    return {
        name: {
            "median": {
                metric: statistics.median(row[metric] for row in rows)
                for metric in ("stream_us", "wall_us")
            },
            "samples": rows,
        }
        for name, rows in samples.items()
    }


def benchmark(batch_size, k, iterations, repeats):
    config = HybridConfig(k, 4, batch_size, max(64, k + 2), history_mode="device")
    backend = GPUSuffixAutomaton(config, "cuda")
    for slot in range(batch_size):
        backend.initialize_device_slot(slot, slot + 1, config.max_history_tokens)
    slots = torch.arange(batch_size, device="cuda")
    generations = slots + 1
    predict = torch.arange(batch_size * (k + 1), dtype=torch.int32, device="cuda") % 100
    indices = torch.arange(predict.numel(), dtype=torch.int32, device="cuda").view(
        batch_size, k + 1
    )
    lengths = (slots % (k + 1) + 1).to(torch.int32)
    seq_lens = torch.ones(batch_size, dtype=torch.int64, device="cuda")
    mask = slots % 2 == 0
    metrics = Metrics()
    counts = metrics.ensure_buffer("cuda")

    def feedback(fused):
        # Each invocation represents a fresh verify delivery. Reset the queue
        # instead of benchmarking a full SA update/query loop here.
        backend.pending_meta.zero_()
        if not fused:
            metrics.record(mask, lengths)
        backend.stage_feedback(
            slots,
            generations,
            predict,
            indices,
            lengths,
            seq_lens,
            True,
            100,
            use_retrieval=mask if fused else None,
            counts=counts if fused else None,
        )

    result = {
        "batch_size": batch_size,
        "num_drafts": k,
        "feedback_and_metrics": compare(
            lambda: feedback(False), lambda: feedback(True), iterations, repeats
        ),
    }
    extension = DeviceDraftExtension(config, backend, 100)
    try:
        buffers = extension.create_graph_buffers(batch_size + 1)
        view = extension._view(list(range(batch_size)), list(range(1, batch_size + 1)))

        def old_stage():
            buffers.view.slots.fill_(-1)
            buffers.view.slots[:batch_size].copy_(view.slots)
            buffers.view.generations[:batch_size].copy_(view.generations)
            view.use_retrieval = buffers.view.use_retrieval[:batch_size]

        result["stable_mapping_stage"] = compare(
            old_stage,
            lambda: buffers.stage(view, batch_size, batch_size + 1),
            iterations,
            repeats,
        )
    finally:
        extension.shutdown()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 32])
    parser.add_argument("--num-drafts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.batch_sizes, args.num_drafts, args.iterations, args.repeats) < 1:
        parser.error(
            "batch sizes, draft length, iterations and repeats must be positive"
        )
    if not torch.cuda.is_available() or torch.version.hip:
        parser.error("requires NVIDIA CUDA and a CUDA extension compiler")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "scope": "eager component overhead, includes host enqueue gaps; not model speedup",
        "results": [
            benchmark(bs, args.num_drafts, args.iterations, args.repeats)
            for bs in args.batch_sizes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
