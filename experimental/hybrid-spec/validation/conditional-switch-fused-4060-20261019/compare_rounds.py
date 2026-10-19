"""Paired full-round wall comparison: original item, SWITCH, fused SWITCH.

Only the baseline row-kernel import is relocated, so old and new code can run
in one process. The prefix/tails/buffers are shared; captured pools stay owned.
"""

import argparse
import gc
import importlib.util
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from hybrid_spec.integrations.sglang.compact_graph import _capture_graph, replay_compact
from hybrid_spec.integrations.sglang.conditional_graph import ConditionalTailGraph


def load_baseline(name, path, *, relocate_rows=False):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source = path.read_text()
    if relocate_rows:
        source = source.replace(
            "from hybrid_spec.backends.gpu_sa.row_compaction import",
            "from hybrid_spec.backends.gpu_sa._baseline_row_compaction import",
        )
    # Trusted baseline Git source; only relocate the row-kernel import above.
    exec(compile(source, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-src", required=True)
    parser.add_argument("--helpers", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=327)
    args = parser.parse_args()
    sys.path.insert(0, args.helpers)
    sys.path.insert(0, args.probe_dir)
    from ablate_dispatch import Case
    from dispatch_probe import make_record

    baseline = Path(args.baseline_src) / "hybrid_spec"
    load_baseline(
        "hybrid_spec.backends.gpu_sa._baseline_row_compaction",
        baseline / "backends/gpu_sa/row_compaction.py",
    )
    old_compact = load_baseline(
        "hybrid_spec.integrations.sglang._baseline_compact_graph",
        baseline / "integrations/sglang/compact_graph.py",
        relocate_rows=True,
    )
    old_conditional = load_baseline(
        "hybrid_spec.integrations.sglang._baseline_conditional_graph",
        baseline / "integrations/sglang/conditional_graph.py",
    )

    class RoundCase(Case):
        def __init__(self, b, n):
            super().__init__(b, n, make_record)
            stream = torch.cuda.Stream()
            # Case's direct control only captures the timed size. Validation
            # switches sizes, so the original host path needs all its graphs.
            for size, tail in self.record.tails.items():
                if tail.graph is None:
                    graph, _ = _capture_graph(tail.forward, stream)
                    tail.graph = graph
                    self.graphs[f"host_tail_{size}"] = graph
            self.old_switch = old_conditional.ConditionalTailGraph(
                self.record, stream, dispatch="switch"
            )
            self.fused = ConditionalTailGraph(self.record, stream, dispatch="switch")
            self.old_record = replace(self.record, conditional=self.old_switch)
            self.fused_record = replace(self.record, conditional=self.fused)
            self.modes = ["host", "switch", "fused"]
            self.dispatch = {
                "switch": lambda: self.old_switch.replay(b),
                "fused": lambda: self.fused.replay(b, batch_prepared=True),
            }

        def run(self, mode, raw_bs=None):
            b = self.batch_size if raw_bs is None else raw_bs
            if mode == "host":
                old_compact.replay_compact(self.host_record, self.counters, b)
            elif mode == "switch":
                old_compact.replay_compact(self.old_record, self.counters, b)
            elif mode == "fused":
                replay_compact(self.fused_record, self.counters, b)
            else:
                super().run(mode)

        def validate(self):
            before = {
                m: d.snapshot()
                for m, d in (("switch", self.old_switch), ("fused", self.fused))
            }
            histogram = [0] * (self.batch_size + 1)
            mixed, checked = 0, 0
            for raw_bs in sorted({1, max(1, self.batch_size - 1), self.batch_size}):
                for size in range(raw_bs + 1):
                    for reverse in (False, True):
                        rows = (
                            list(range(raw_bs - size, raw_bs))
                            if reverse
                            else list(range(size))
                        )
                        self.record.override.use_retrieval.fill_(True)
                        self.record.override.use_retrieval[rows] = False
                        expected = [
                            row * 3 + 7 if row in rows else -1
                            for row in range(self.batch_size)
                        ]
                        for mode in self.modes:
                            self.run(mode, raw_bs)
                            assert self.record.output.cpu().tolist() == expected, (
                                mode,
                                raw_bs,
                                size,
                            )
                        histogram[size] += 1
                        mixed += 0 < size < raw_bs
                        checked += 1
            for mode, dispatcher in (
                ("switch", self.old_switch),
                ("fused", self.fused),
            ):
                after, after_mixed = dispatcher.snapshot()
                assert [a - z for a, z in zip(after, before[mode][0])] == histogram
                assert after_mixed - before[mode][1] == mixed
            self.record.override.use_retrieval.fill_(True)
            self.record.override.use_retrieval[: self.selected] = False
            self.run("fused")
            self.run("switch")
            return checked

        def measure_paired(self, seed):
            rng = random.Random(seed)
            for _ in range(100):
                for mode in self.modes:
                    self.run(mode)
            torch.cuda.synchronize()
            wall = {m: [] for m in self.modes}
            # Shuffle all three modes within EVERY iteration, not just blocks.
            # Block means preserve samples without over-weighting outliers.
            for _ in range(15):
                totals = dict.fromkeys(self.modes, 0)
                for _ in range(100):
                    order = self.modes.copy()
                    rng.shuffle(order)
                    for mode in order:
                        start = time.perf_counter_ns()
                        self.run(mode)
                        torch.cuda.synchronize()
                        totals[mode] += time.perf_counter_ns() - start
                for mode in self.modes:
                    wall[mode].append(totals[mode] / 100_000)
            events = {m: [] for m in self.dispatch}
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(31):
                order = list(self.dispatch)
                rng.shuffle(order)
                for mode in order:
                    torch.cuda._sleep(2_000_000)
                    start.record()
                    self.dispatch[mode]()
                    end.record()
                    end.synchronize()
                    events[mode].append(start.elapsed_time(end) * 1000)
            return {
                m: {
                    "wall_us_samples": wall[m],
                    "wall_us_median": statistics.median(wall[m]),
                    "dispatch_gpu_us_samples": events.get(m),
                    "dispatch_gpu_us_median": statistics.median(events[m])
                    if m in events
                    else None,
                }
                for m in self.modes
            }

        def close(self):
            self.old_switch.reset()
            self.fused.reset()
            super().close()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)
    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "seed": args.seed,
        "cases": [],
    }
    for b, n in ((1, 0), (1, 1), (3, 1), (4, 2), (8, 4), (16, 8), (32, 16)):
        gc.collect()
        gc.disable()
        case = RoundCase(b, n)
        checked = case.validate()
        row = {
            "batch_size": b,
            "selected": n,
            "checked_rounds_per_mode": checked,
            "timings": case.measure_paired(args.seed),
        }
        if b in (1, 4, 8, 16, 32):
            row["profiles"] = {mode: case.profile(mode, out) for mode in case.modes}
        result["cases"].append(row)
        (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            b,
            n,
            {m: round(t["wall_us_median"], 2) for m, t in row["timings"].items()},
            flush=True,
        )
        case.close()
        del case
        gc.enable()
    torch.cuda.synchronize()
