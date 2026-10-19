"""Fixed-batch/Graph steady-median calibration, following DSpark's methodology.

Use a dedicated dynamic_verify_profile=true server with SGLANG_SIMULATE_ACC_LEN=1.
We retain CUDA Event stage timings and actual Graph token counts; production
uses the resulting median table as the seed for its low-frequency online EMA.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from hybrid_spec.config import HybridConfig
from hybrid_spec.selection.verify_costs import cost_config, cost_profile


def snapshots_in(value):
    if isinstance(value, dict):
        if "hybrid_spec_dynamic_verify" in value:
            yield value["hybrid_spec_dynamic_verify"]
        else:
            for child in value.values():
                yield from snapshots_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from snapshots_in(child)


def aligned_rows(snapshots, bs, graph_tokens):
    """Join ranks by step ID, retaining the requested batch and physical tier."""
    ranks = [{r["seq"]: r for r in s["profile_records"]} for s in snapshots]
    if not ranks:
        return [], 0.0
    common = set.intersection(*(set(rank) for rank in ranks))
    rows = []
    for seq in sorted(common):
        samples = [rank[seq] for rank in ranks]
        if any(r["bs"] != bs or r["graph_tokens"] != graph_tokens for r in samples):
            continue
        if len({r["context"] for r in samples}) != 1:
            continue
        if any(
            not math.isfinite(r[k]) or r[k] < 0
            for r in samples
            for k in ("fixed_ms", "verify_ms")
        ):
            continue
        if any(r["verify_ms"] == 0 for r in samples):
            continue
        rows.append(
            {
                "seq": seq,
                "bs": bs,
                "context": samples[0]["context"],
                "graph_tokens": graph_tokens,
                "fixed_ms": max(r["fixed_ms"] for r in samples),
                "verify_ms": max(r["verify_ms"] for r in samples),
            }
        )
    window = [
        seq for seq in common if rows and rows[0]["seq"] <= seq <= rows[-1]["seq"]
    ]
    return rows, len(rows) / len(window) if window else 0.0


def summarize_cell(snapshots, bs, graph_tokens, warmup_steps, min_steps):
    rows, match = aligned_rows(snapshots, bs, graph_tokens)
    if len(rows) < warmup_steps + min_steps or match < 0.5:
        raise ValueError(
            "profile cell never stabilized at the requested batch/Graph tier"
        )
    groups = defaultdict(list)
    for row in rows[warmup_steps:]:
        groups[row["context"]].append(row)
    summaries = []
    for ctx, samples in sorted(groups.items()):
        if len(samples) >= min_steps:
            summaries.append(
                {
                    "bs": bs,
                    "context": ctx,
                    "graph_tokens": graph_tokens,
                    "fixed_ms": statistics.median(r["fixed_ms"] for r in samples),
                    "verify_ms": statistics.median(r["verify_ms"] for r in samples),
                    "num_steps": len(samples),
                    "match_fraction": match,
                }
            )
    if not summaries:
        raise ValueError(
            "no context bucket has enough steady samples; increase round length"
        )
    return summaries


def build_profile(summaries, config, runtime):
    """Median across repeats, without importing acceptance labels or explored EMA."""
    fixed, verify = defaultdict(list), defaultdict(list)
    for row in summaries:
        key = (row["bs"], row["context"])
        fixed[key].append(row["fixed_ms"])
        verify[(*key, row["graph_tokens"])].append(row["verify_ms"])
    if not verify:
        raise ValueError("no steady Graph cost samples")
    drafts = {key: statistics.median(values) for key, values in fixed.items()}
    costs = {}
    for (bs, ctx, n), values in verify.items():
        costs.setdefault((bs, ctx), {})[n] = (
            drafts[(bs, ctx)],
            statistics.median(values),
        )
    return cost_profile(config, costs, drafts, runtime)


def request(url, path, payload=None, *, timeout=30):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        if response.headers.get_content_type() == "application/json":
            return json.loads(body)
        return body.decode()


def pin_cell(url, bs, n):
    result = request(
        url,
        "/set_internal_state",
        {
            "server_args": {
                "hybrid_spec_profile_batch_size": bs,
                "hybrid_spec_profile_graph_tokens": n,
            }
        },
    )
    results = result if isinstance(result, list) else [result]
    if not results or not all(
        isinstance(r, dict) and r.get("updated") for r in results
    ):
        raise RuntimeError(f"server rejected profile cell ({bs}, {n}): {result}")


def generate(url, prompts, bs, max_tokens):
    return request(
        url,
        "/generate",
        {
            "text": [prompts[i % len(prompts)] for i in range(bs)],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
        },
        timeout=1800,
    )


def run_cell(args, prompts, bs, n):
    # Flush resets worker-local statistics and pins, so it must precede pinning.
    flushed = request(args.url, "/flush_cache?timeout=30", {})
    if isinstance(flushed, dict) and flushed.get("success") is False:
        raise RuntimeError("profile server is busy; use a dedicated idle server")
    pin_cell(args.url, bs, n)
    errors = []

    def load():
        try:
            generate(args.url, prompts, bs, args.max_tokens)
        except (OSError, ValueError) as exc:
            errors.append(exc)

    thread = threading.Thread(target=load, daemon=True)
    thread.start()
    first_steady = None
    deadline = time.monotonic() + args.round_timeout
    try:
        while time.monotonic() < deadline:
            time.sleep(1)
            snapshots = list(snapshots_in(request(args.url, "/server_info")))
            rows, _ = aligned_rows(snapshots, bs, n)
            if len(rows) >= args.warmup_steps and first_steady is None:
                first_steady = time.monotonic()
            if (
                len(rows) >= args.warmup_steps + args.min_steady_steps
                and first_steady is not None
                and time.monotonic() - first_steady >= args.min_steady_seconds
            ):
                return snapshots, summarize_cell(
                    snapshots, bs, n, args.warmup_steps, args.min_steady_steps
                )
            if not thread.is_alive():
                raise RuntimeError(
                    f"load ended before steady gates; increase --max-tokens: {errors}"
                )
        raise RuntimeError(f"profile cell ({bs}, {n}) timed out before steady gates")
    finally:
        if thread.is_alive():
            request(args.url, "/abort_request", {"abort_all": True})
        thread.join(timeout=60)
        if thread.is_alive():
            raise RuntimeError("profile load did not drain after abort")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON list of strings"
    )
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--min-steady-steps", type=int, default=32)
    parser.add_argument("--min-steady-seconds", type=float, default=10)
    parser.add_argument("--round-timeout", type=float, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = HybridConfig.from_file(str(args.config))
    if not config.dynamic_verify_profile:
        parser.error("server/config must enable dynamic_verify_profile")
    prompts = json.loads(args.prompts.read_text())
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) for p in prompts)
    ):
        parser.error("prompts must be a nonempty JSON list of strings")
    if (
        min(args.batch_size + [args.repeats, args.min_steady_steps, args.max_tokens])
        < 1
    ):
        parser.error(
            "batch-size, repeats, min-steady-steps and max-tokens must be positive"
        )
    if args.warmup_steps < 0 or not 0 <= args.min_steady_seconds < args.round_timeout:
        parser.error("invalid warmup/steady time gates")
    if any(not math.isfinite(f) or not 0 < f <= 1 for f in args.fracs):
        parser.error("fracs must be finite and in (0, 1]")
    if max(args.batch_size) > config.max_requests:
        parser.error("batch-size exceeds max_requests")
    generate(args.url, prompts, 1, 16)
    initial = list(snapshots_in(request(args.url, "/server_info")))
    if not initial or any("profile_records" not in s for s in initial):
        parser.error("running server does not expose profile mode")
    if any(s["cost_profile"]["config"] != cost_config(config) for s in initial):
        parser.error("running server's hybrid config does not match --config")
    runtime = initial[0]["cost_profile"]["runtime"]
    if any(s["cost_profile"]["runtime"] != runtime for s in initial):
        parser.error("ranks disagree on model/GPU/TP/capture metadata")
    grid, cells = runtime["capture_token_grid"], []
    for bs in sorted(set(args.batch_size)):
        for frac in sorted(set(args.fracs)):
            logical = max(
                bs * (config.dynamic_verify_min_drafts + 1),
                bs + int(frac * bs * config.num_drafts),
            )
            n = next((n for n in grid if n >= logical), None)
            if n is None or bs > runtime["capture_max_bs"]:
                parser.error("profile sweep exceeds captured CUDA Graph capacity")
            if (bs, n) not in cells:
                cells.append((bs, n))
    summaries = []
    try:
        with (
            args.output.with_suffix(".records.jsonl").open("w") as records,
            args.output.with_suffix(".rounds.jsonl").open("w") as rounds,
        ):
            for repeat in range(args.repeats):
                for bs, n in cells:
                    snapshots, rows = run_cell(args, prompts, bs, n)
                    records.write(
                        json.dumps(
                            {
                                "repeat": repeat,
                                "bs": bs,
                                "graph_tokens": n,
                                "ranks": [s["profile_records"] for s in snapshots],
                            }
                        )
                        + "\n"
                    )
                    for row in rows:
                        row["repeat"] = repeat
                        rounds.write(json.dumps(row) + "\n")
                    records.flush()
                    rounds.flush()
                    summaries.extend(rows)
                    print(f"Measured batch={bs}, graph_tokens={n}, repeat={repeat + 1}")
    finally:
        pin_cell(args.url, 0, 0)
    args.output.write_text(
        json.dumps(build_profile(summaries, config, runtime), indent=2) + "\n"
    )
    manifest = {
        "method": "fixed_graph_steady_median",
        "clock": "cuda_events",
        "cells": cells,
        "warmup_steps": args.warmup_steps,
        "min_steady_steps": args.min_steady_steps,
        "min_steady_seconds": args.min_steady_seconds,
        "repeats": args.repeats,
        "runtime": runtime,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Saved steady median cost table to {args.output}")


if __name__ == "__main__":
    main()
