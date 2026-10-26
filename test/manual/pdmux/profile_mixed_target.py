"""Opt-in, one-shape target diagnostic; never a serving throughput benchmark."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch

_completed = False


def profile_target(count, before, native, separate, mixed):
    global _completed
    destination = os.environ.get("MIXED_TARGET_PROFILE_DIR")
    minimum = int(os.environ.get("MIXED_TARGET_PROFILE_MIN_TOKENS", "1000"))
    if not destination or _completed or count < minimum:
        return
    _completed = True
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    functions = dict(native=native, repaired_separate=separate, mixed=mixed)
    result = dict(tokens=count, scope="target_stage_only", samples={}, profiles={})
    # Rotate order; restore/synchronize outside every measured interval.
    samples = {key: [] for key in functions}
    for trial in range(3):
        order = list(functions)
        order = order[trial:] + order[:trial]
        for name in order:
            before.restore()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            functions[name]()
            end.record()
            end.synchronize()
            samples[name].append(dict(gpu_ms=start.elapsed_time(end), wall_ms=(time.perf_counter()-wall)*1000))
    for name, function in functions.items():
        result["samples"][name] = dict(
            samples=samples[name],
            median_gpu_ms=statistics.median(row["gpu_ms"] for row in samples[name]),
            median_wall_ms=statistics.median(row["wall_ms"] for row in samples[name]),
        )
        before.restore()
        torch.cuda.synchronize()
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                with torch.profiler.record_function(name):
                    function()
                    torch.cuda.synchronize()
            trace = root / f"{name}-trace.json"
            prof.export_chrome_trace(str(trace))
            result["profiles"][name] = dict(trace=str(trace), table=prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
        except RuntimeError as exc:
            result["profiles"][name] = dict(error=str(exc))
    before.restore()
    (root / "target-profile.json").write_text(json.dumps(result, indent=2) + "\n")
