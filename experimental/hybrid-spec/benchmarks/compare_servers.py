"""Compare target-only / native-spec / hybrid endpoints under identical requests.

Servers must be started separately with deterministic greedy configuration.
This measures end-to-end request latency and output throughput, not decode TPOT.
"""

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path


def generate(url, prompt, max_tokens):
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    meta = result["meta_info"]
    tokens = result.get("output_ids")
    if tokens is None:
        tokens = [entry[1] for entry in meta["output_token_logprobs"]]
    return {"tokens": tokens, "latency_s": elapsed, "meta_info": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline")
    source.add_argument("--record", help="Record one server for sequential GPU runs")
    parser.add_argument("--hybrid")
    parser.add_argument(
        "--reference", type=Path, help="Earlier --record JSON to compare"
    )
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON list of strings"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmup-all", action="store_true")
    parser.add_argument("--flush-cache-before-run", action="store_true")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not prompts or not all(isinstance(p, str) for p in prompts):
        parser.error("prompts must be a nonempty JSON list of strings")
    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("concurrency and max-tokens must be positive")
    if (args.baseline and not args.hybrid) or (args.record and args.hybrid):
        parser.error(
            "--hybrid is required with --baseline and incompatible with --record"
        )
    if args.reference and not args.record:
        parser.error("--reference requires --record")
    report = {
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "prompts_sha256": hashlib.sha256(
            json.dumps(prompts, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    reference = None
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if "record" not in reference or any(
            reference.get(key) != value for key, value in report.items()
        ):
            parser.error("reference prompts, concurrency, or max-tokens do not match")
    endpoints = (
        (("record", args.record),)
        if args.record
        else (("baseline", args.baseline), ("hybrid", args.hybrid))
    )
    for label, url in endpoints:
        # Warm-up is intentionally excluded from the measured run.
        # Warm the measured concurrency, including batch-size-specific kernels.
        # A single request leaves the concurrent run's first batch cold.
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            list(
                pool.map(
                    lambda p: generate(url, p, args.max_tokens),
                    prompts if args.warmup_all else prompts[: args.concurrency],
                )
            )
        if args.flush_cache_before_run:
            with urllib.request.urlopen(
                url.rstrip("/") + "/flush_cache?timeout=30", timeout=35
            ) as response:
                if response.status != 200:
                    raise RuntimeError("Cache flush failed before measurement")
        with urllib.request.urlopen(
            url.rstrip("/") + "/get_server_info", timeout=30
        ) as response:
            before = json.load(response)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            rows = list(pool.map(lambda p: generate(url, p, args.max_tokens), prompts))
        wall = time.perf_counter() - start
        with urllib.request.urlopen(
            url.rstrip("/") + "/get_server_info", timeout=30
        ) as response:
            after = json.load(response)
        report[label] = {
            "url": url,
            "warmup_requests": (
                len(prompts) if args.warmup_all else min(len(prompts), args.concurrency)
            ),
            "cache_flushed_before_run": args.flush_cache_before_run,
            "wall_s": wall,
            "mean_latency_s": statistics.mean(r["latency_s"] for r in rows),
            "output_tokens_per_s": sum(len(r["tokens"]) for r in rows) / wall,
            "requests": rows,
            "server_info_before": before,
            "server_info_after": after,
        }
    baseline = reference["record"] if reference else report.get("baseline")
    hybrid = report["record"] if args.record else report["hybrid"]
    if baseline and len(baseline["requests"]) != len(hybrid["requests"]):
        raise ValueError("Reference request count differs from this run")
    mismatches = (
        [
            i
            for i, (left, right) in enumerate(
                zip(baseline["requests"], hybrid["requests"])
            )
            if left["tokens"] != right["tokens"]
        ]
        if baseline
        else None
    )
    report["token_mismatch_indices"] = mismatches
    if args.reference:
        report["reference"] = str(args.reference)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if mismatches:
        raise SystemExit(
            f"Greedy token mismatch at requests {mismatches}; see {args.output}"
        )
    if baseline:
        print(f"All {len(prompts)} token sequences match; report: {args.output}")
    else:
        print(f"Recorded {len(prompts)} token sequences; report: {args.output}")


if __name__ == "__main__":
    main()
