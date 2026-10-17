"""Measure identical serial/parallel SA inputs; run each source tree separately."""

import argparse
import hashlib
import json
import random
import statistics
from functools import partial
from pathlib import Path

import torch
from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import HistoryDelta, RequestKey


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for num_sources in (20, 64, 257):
        for batch in (1, 4):
            for workload in ("different", "matching"):
                rng = random.Random(20261017)
                context = [rng.randrange(32) for _ in range(128)]
                histories = [
                    [rng.randrange(32) for _ in range(384)] for _ in range(num_sources)
                ]
                for slot in range(batch):
                    histories[slot][-128:] = context
                if workload == "matching":
                    for slot in range(batch, num_sources):
                        histories[slot][224:352] = context
                        histories[slot][352:] = [100 + slot] * 32
                config = HybridConfig(
                    8,
                    4,
                    4,
                    1024,
                    history_mode="device",
                    enable_global_pool=True,
                    max_shared_requests=num_sources - 4,
                )
                backend = GPUSuffixAutomaton(config, "cuda")
                for slot, history in enumerate(histories):
                    backend.initialize_device_slot(slot, slot + 1, 1024)
                    backend.append(
                        slot,
                        HistoryDelta(
                            RequestKey(str(slot), slot + 1), 0, tuple(history)
                        ),
                    )
                device = {"device": "cuda", "dtype": torch.int64}
                slots = torch.arange(batch, **device)
                generations = slots + 1
                seq_lens = torch.full_like(slots, 383)
                sources = torch.arange(num_sources, **device)
                source_generations = sources + 1
                source_lengths = torch.full_like(sources, -1)
                tokens = torch.zeros((batch, 8), **device)
                mask = torch.zeros(batch, device="cuda", dtype=torch.bool)

                query = partial(
                    backend.consume_and_query,
                    slots,
                    generations,
                    seq_lens,
                    tokens,
                    mask,
                    shared_slots=sources,
                    shared_generations=source_generations,
                    shared_lengths=source_lengths,
                )

                for _ in range(8):
                    query()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(8):
                        query()
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                samples = []
                for _ in range(5):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(25):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / (25 * 8))
                stem = f"s{num_sources}-b{batch}-{workload}"
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize()
                trace = out / f"{stem}-trace.json"
                prof.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
                kernel_times = [
                    e["dur"]
                    for e in events
                    if e.get("cat") == "kernel"
                    and "shared_query_kernel" in e.get("name", "")
                ]
                assert len(kernel_times) == 40, (stem, len(kernel_times))
                row = {
                    "sources": num_sources,
                    "batch": batch,
                    "workload": workload,
                    "total_query_us_samples": samples,
                    "total_query_us_median": statistics.median(samples),
                    "shared_kernel_us_median": statistics.median(kernel_times),
                    "shared_kernel_us_min": min(kernel_times),
                    "shared_kernel_us_max": max(kernel_times),
                    "tokens": tokens.tolist(),
                    "mask": mask.tolist(),
                    "inputs_sha256": hashlib.sha256(
                        json.dumps(histories).encode()
                    ).hexdigest(),
                }
                results.append(row)
                (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(
                    json.dumps(
                        {k: v for k, v in row.items() if k not in ("tokens", "mask")}
                    ),
                    flush=True,
                )
                del prof, graph, query, backend
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
