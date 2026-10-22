"""Attribute GPU work through runtime correlations and diagnostic CPU scopes.

GPU sums, GPU interval unions, CPU waits and idle intervals are separate metrics.
This never treats the CPU's wait for already-running GPU work as additive cost.
"""

import collections
import gzip
import heapq
import json
import re
import statistics
import sys
from pathlib import Path


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def overlap(start, end, intervals):
    return sum(max(0, min(end, right) - max(start, left)) for left, right in intervals)


def contexts(events, runtimes):
    scopes, queries = collections.defaultdict(list), collections.defaultdict(list)
    for e in events:
        if e.get("cat") == "user_annotation" and e.get("ph") == "X":
            scopes[e["tid"]].append(e)
    for e in runtimes:
        queries[e["tid"]].append(e)
    result = {}
    for tid, queries_for_thread in queries.items():
        scopes_for_thread = sorted(scopes[tid], key=lambda e: e["ts"])
        index, active, ends = 0, {}, []
        for runtime in sorted(queries_for_thread, key=lambda e: e["ts"]):
            timestamp = runtime["ts"]
            while (
                index < len(scopes_for_thread)
                and scopes_for_thread[index]["ts"] <= timestamp
            ):
                scope = scopes_for_thread[index]
                active[index] = scope
                heapq.heappush(ends, (scope["ts"] + scope["dur"], index))
                index += 1
            while ends and ends[0][0] < timestamp:
                _, key = heapq.heappop(ends)
                active.pop(key, None)
            result[runtime["args"]["correlation"]] = sorted(
                active.values(), key=lambda e: e["dur"]
            )
    return result


def role(names):
    for label in [
        "compact_pack",
        "compact_scatter",
        "verify_plan",
        "split_replay",
        "draft_extend",
        "verify",
        "draft",
        "prefill",
    ]:
        if "diag/" + label in names:
            return label
    return "outside_scopes"


def analyze(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    gpu = [
        e
        for e in events
        if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"] and e.get("ph") == "X"
    ]
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    assert kernels, path
    runtimes = [
        e
        for e in events
        if e.get("cat") in ["cuda_runtime", "cuda_driver"]
        and "correlation" in e.get("args", {})
    ]
    ctx = contexts(events, runtimes)
    roles = {
        correlation: role([e["name"] for e in scopes])
        for correlation, scopes in ctx.items()
    }
    graph_kernels = collections.defaultdict(list)
    for e in kernels:
        if e.get("args", {}).get("graph id"):
            graph_kernels[e["args"].get("correlation")].append(e)
    graphs = []
    for correlation, group in graph_kernels.items():
        label = roles.get(correlation, "outside_scopes")
        if label == "split_replay":
            label = (
                "sa_prefix"
                if any("consume_query_kernel" in e["name"] for e in group)
                else "neural_tail"
            )
        roles[correlation] = label
        graphs.append(
            {
                "role": label,
                "correlation": correlation,
                "start_us": min(e["ts"] for e in group),
                "end_us": max(e["ts"] + e["dur"] for e in group),
                "kernel_sum_us": sum(e["dur"] for e in group),
                "kernels": len(group),
                "raw_batch": next(
                    (
                        int(match.group(1))
                        for scope in ctx.get(correlation, [])
                        if (match := re.search(r"step\[VERIFY bs=(\d+)", scope["name"]))
                    ),
                    None,
                ),
            }
        )
    anchors = sorted(g["start_us"] for g in graphs if g["role"] == "verify")
    decode_only = "DECODE" in str(path)
    if decode_only:
        assert len(anchors) >= 3, (path, collections.Counter(g["role"] for g in graphs))
        start, end = anchors[0], anchors[-1]
        cycles = len(anchors) - 1
        expected_batch = int(re.search(r"DECODE-c(\d+)", str(path)).group(1))
        assert all(
            graph["raw_batch"] == expected_batch
            for graph in graphs
            if graph["role"] == "verify"
        ), (path, [(g["role"], g["raw_batch"]) for g in graphs])
    else:
        start, end = min(e["ts"] for e in gpu), max(e["ts"] + e["dur"] for e in gpu)
        cycles = None
    clip = lambda e: max(0, min(end, e["ts"] + e["dur"]) - max(start, e["ts"]))
    busy = union(
        [(max(start, e["ts"]), min(end, e["ts"] + e["dur"])) for e in gpu if clip(e)]
    )
    phase_rows = collections.defaultdict(
        lambda: {"kernel_ct": 0, "kernel_sum_us": 0, "intervals": []}
    )
    top = collections.defaultdict(lambda: {"ct": 0, "sum_us": 0})
    sa_sum, split_copy_sum = 0, 0
    for e in kernels:
        cost = clip(e)
        if not cost:
            continue
        label = roles.get(e.get("args", {}).get("correlation"), "outside_scopes")
        row = phase_rows[label]
        row["kernel_ct"] += 1
        row["kernel_sum_us"] += cost
        row["intervals"].append((max(start, e["ts"]), min(end, e["ts"] + e["dur"])))
        top[e["name"]]["ct"] += 1
        top[e["name"]]["sum_us"] += cost
        if "consume_query_kernel" in e["name"] or "feedback_kernel(" in e["name"]:
            sa_sum += cost
        if any(
            name in e["name"]
            for name in [
                "_compact_neural_rows",
                "_gather_rows",
                "_scatter_rows",
                "_gather_cache",
            ]
        ):
            split_copy_sum += cost
    for row in phase_rows.values():
        row["kernel_union_us"] = duration(union(row.pop("intervals")))
        if cycles:
            row["kernel_ms_per_cycle"] = row["kernel_sum_us"] / cycles / 1000
    first_gpu, last_gpu = (
        min(e["ts"] for e in gpu),
        max(e["ts"] + e["dur"] for e in gpu),
    )
    all_busy = union([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    waits = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("name") == "hybrid_prefix_decision"
        and first_gpu <= e["ts"]
        and e["ts"] + e["dur"] <= last_gpu
    ]
    wait_summary = {
        "complete_span_ct": len(waits),
        "host_ms_mean": statistics.mean(e["dur"] for e in waits) / 1000
        if waits
        else None,
        "no_gpu_activity_ms_mean": statistics.mean(
            max(0, e["dur"] - overlap(e["ts"], e["ts"] + e["dur"], all_busy))
            for e in waits
        )
        / 1000
        if waits
        else None,
        "note": "No-GPU portion is observed idle, not a proven removable dependency-free speedup.",
    }
    return {
        "trace": str(path),
        "kernel_ct": len(kernels),
        "decode_cycles": cycles,
        "cycle_scope": "first target-verify GPU start to last target-verify GPU start"
        if cycles
        else "full GPU trace",
        "window_ms": (end - start) / 1000,
        "cycle_ms": (end - start) / cycles / 1000 if cycles else None,
        "gpu_busy_ms": duration(busy) / 1000,
        "gpu_idle_ms": (end - start - duration(busy)) / 1000,
        "gpu_busy_rate": duration(busy) / (end - start),
        "phases": dict(phase_rows),
        "prefix_decision": wait_summary,
        "sa_consume_and_feedback_kernel_sum_us": sa_sum,
        "split_copy_named_kernel_sum_us": split_copy_sum,
        "top_kernels": sorted(
            ({"name": k, **v} for k, v in top.items()), key=lambda e: -e["sum_us"]
        )[:20],
        "graphs": graphs,
        "scope": "Diagnostic profiling only; steady decode holds batch with ignore_eos and forced continuation, not the formal natural-EOS throughput workload.",
    }


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    rows = [analyze(p) for p in sorted((root / "traces").rglob("*.trace.json.gz"))]
    (root / "profile-summary.json").write_text(
        json.dumps({"rows": rows}, indent=2) + "\n"
    )
    for row in rows:
        print(
            row["trace"], row["cycle_ms"], row["gpu_busy_rate"], row["prefix_decision"]
        )


if __name__ == "__main__":
    main()
