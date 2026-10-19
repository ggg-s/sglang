"""Diagnostic controls for compact dispatch; never alters serving code.

The direct and single-IF controls know the selected shape in advance, so they
are lower bounds, not semantics-preserving replacements for dynamic dispatch.
"""

import argparse
import gc
import gzip
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from hybrid_spec.integrations.sglang.compact_graph import (
    compact_neural_rows,
    replay_compact,
)


def summarize_trace(path, steps):
    events = json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    counts = Counter(e["name"] for e in kernels)
    times = defaultdict(float)
    for event in kernels:
        times[event["name"]] += event["dur"]
    return {
        "kernels_per_step": len(kernels) / steps,
        "set_handle_kernels_per_step": sum(
            v for k, v in counts.items() if "conditional_handle" in k
        )
        / steps,
        "kernel_us_per_step": sum(times.values()) / steps,
        "kernels": [
            {
                "name": k,
                "count_per_step": counts[k] / steps,
                "us_per_step": times[k] / steps,
            }
            for k in sorted(counts, key=times.get, reverse=True)
        ],
        "item_ct": sum(e.get("name") == "aten::item" for e in events),
        "graph_launch_ct": sum(e.get("name") == "cudaGraphLaunch" for e in events),
    }


class Case:
    def __init__(self, batch_size, selected, make_record):
        self.batch_size, self.selected = batch_size, selected
        self.record, self.counters = make_record(batch_size, True)
        self.host_record = replace(self.record, conditional=None)
        self.record.override.use_retrieval.fill_(True)
        self.record.override.use_retrieval[:selected] = False
        self.prepare()
        count = self.record.num_neural
        self.graphs = {}
        self.keep = []
        stream = torch.cuda.Stream()

        def selected_tail():
            if selected:
                self.record.tails[selected].forward()

        def statistics_body():
            prod = self.record.conditional
            prod.stats[:-1].scatter_add_(0, count.reshape(1).long(), self.one)
            prod.stats[-1:].add_(((count > 0) & (count < prod.live_bs)).reshape(1))

        self.one = torch.ones(1, dtype=torch.int64, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            selected_tail()
            statistics_body()
        torch.cuda.synchronize()

        for kind in ("direct", "single_if", "tree", "direct_stats"):
            graph = torch.cuda.CUDAGraph()

            def branch(predicate, body, graph=graph):
                graph.begin_capture_to_if_node(predicate)
                try:
                    body()
                finally:
                    graph.end_capture_to_conditional_node()

            def tree(first, last):
                if first == last:
                    self.record.tails[first].forward()
                else:
                    middle = (first + last) // 2
                    branch(count <= middle, lambda: tree(first, middle))
                    branch(count > middle, lambda: tree(middle + 1, last))

            with torch.cuda.graph(
                graph, pool=torch.cuda.graph_pool_handle(), stream=stream
            ):
                if kind in ("direct", "direct_stats"):
                    selected_tail()
                    if kind == "direct_stats":
                        statistics_body()
                elif kind == "single_if":
                    branch(
                        count > 0, lambda: self.record.tails[max(1, selected)].forward()
                    )
                else:
                    branch(count > 0, lambda: tree(1, batch_size))
            self.graphs[kind] = graph
        if selected:
            self.record.tails[selected].graph = self.graphs["direct"]

        self.dispatch = {k: graph.replay for k, graph in self.graphs.items()}
        # Same production graph, only omit the per-replay live_bs.fill_.
        self.dispatch["tree_stats"] = self.record.conditional.graph.replay
        self.dispatch["production"] = lambda: self.record.conditional.replay(batch_size)
        self.modes = ["host", "host_no_metrics", *self.dispatch]

        for mode in self.modes:
            self.run(mode)
            assert self.record.output.cpu().tolist() == [
                row * 3 + 7 if row < selected else -1 for row in range(batch_size)
            ], (batch_size, selected, mode)

    def prepare(self):
        r = self.record
        compact_neural_rows(
            r.override.use_retrieval, r.rows, r.num_neural, self.batch_size, False
        )

    def run(self, mode):
        if mode == "host":
            replay_compact(self.host_record, self.counters, self.batch_size)
        elif mode == "production":
            replay_compact(self.record, self.counters, self.batch_size)
        else:
            self.record.prefix.replay()
            self.prepare()
            if mode == "host_no_metrics":
                size = int(self.record.num_neural.item())
                if size:
                    self.graphs["direct"].replay()
            else:
                self.dispatch[mode]()

    def measure(self):
        rng = random.Random(187)
        for _ in range(50):
            for mode in self.modes:
                self.run(mode)
        torch.cuda.synchronize()
        wall = {mode: [] for mode in self.modes}
        for _ in range(7):
            order = self.modes.copy()
            rng.shuffle(order)
            for mode in order:
                start = time.perf_counter_ns()
                for _ in range(100):
                    self.run(mode)
                    torch.cuda.synchronize()
                wall[mode].append((time.perf_counter_ns() - start) / 100_000)
        # Delay the GPU ahead of each event pair so CPU enqueueing completes
        # before the start event executes. Times exclude common prefix/scan.
        event_times = {mode: [] for mode in self.dispatch}
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(31):
            order = list(self.dispatch)
            rng.shuffle(order)
            for mode in order:
                torch.cuda._sleep(2_000_000)
                start.record()
                self.dispatch[mode]()
                end.record()
                end.synchronize()
                event_times[mode].append(start.elapsed_time(end) * 1000)
        return {
            mode: {
                "wall_us_median": statistics.median(wall[mode]),
                "wall_us_samples": wall[mode],
                "dispatch_gpu_us_median": statistics.median(event_times[mode])
                if mode in event_times
                else None,
                "dispatch_gpu_us_samples": event_times.get(mode),
            }
            for mode in self.modes
        }

    def profile(self, mode, directory):
        for _ in range(20):
            self.run(mode)
        torch.cuda.synchronize()
        steps = 5
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(steps):
                with torch.profiler.record_function(f"dispatch::{mode}"):
                    self.run(mode)
                    torch.cuda.synchronize()
        raw = directory / f"b{self.batch_size}-n{self.selected}-{mode}.trace.json"
        prof.export_chrome_trace(str(raw))
        target = raw.with_suffix(raw.suffix + ".gz")
        target.write_bytes(gzip.compress(raw.read_bytes(), mtime=0))
        raw.unlink()
        return {"trace": target.name, **summarize_trace(target, steps)}

    def close(self):
        self.record.conditional.reset()
        self.record.prefix.reset()
        for graph in self.graphs.values():
            graph.reset()
        for tail in self.record.tails.values():
            tail.forward = None
            tail.graph = None
        self.record.tails.clear()
        self.dispatch.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.probe_dir)
    from dispatch_probe import make_record

    directory = Path(args.output_dir)
    directory.mkdir(exist_ok=True)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
    }
    for batch_size, selected in ((1, 0), (1, 1), (4, 2), (16, 8), (32, 16)):
        gc.collect()
        gc.disable()
        case = Case(batch_size, selected, make_record)
        gc.enable()
        row = {
            "batch_size": batch_size,
            "selected": selected,
            "timings": case.measure(),
        }
        if batch_size in (1, 32):
            row["profiles"] = {
                mode: case.profile(mode, directory)
                for mode in (
                    "host",
                    "direct",
                    "single_if",
                    "tree",
                    "tree_stats",
                    "production",
                )
            }
        report["cases"].append(row)
        (directory / "ablation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            batch_size,
            selected,
            {
                mode: round(data["wall_us_median"], 2)
                for mode, data in row["timings"].items()
            },
            flush=True,
        )
        case.close()
        del case
    torch.cuda.synchronize()
