"""Compare target-only / native-spec / hybrid endpoints under identical requests.

Servers must be started separately with deterministic greedy configuration.
This measures end-to-end request latency and output throughput, not decode TPOT.
"""

import argparse
import concurrent.futures
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
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument(
        "--prompts", type=Path, required=True, help="JSON list of strings"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not prompts or not all(isinstance(p, str) for p in prompts):
        parser.error("prompts must be a nonempty JSON list of strings")
    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("concurrency and max-tokens must be positive")
    report = {"concurrency": args.concurrency, "max_tokens": args.max_tokens}
    for label, url in (("baseline", args.baseline), ("hybrid", args.hybrid)):
        # Warm-up is intentionally excluded from the measured run.
        generate(url, prompts[0], args.max_tokens)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            rows = list(pool.map(lambda p: generate(url, p, args.max_tokens), prompts))
        wall = time.perf_counter() - start
        report[label] = {
            "url": url,
            "wall_s": wall,
            "mean_latency_s": statistics.mean(r["latency_s"] for r in rows),
            "output_tokens_per_s": sum(len(r["tokens"]) for r in rows) / wall,
            "requests": rows,
        }
    mismatches = [
        i
        for i, (baseline, hybrid) in enumerate(
            zip(report["baseline"]["requests"], report["hybrid"]["requests"])
        )
        if baseline["tokens"] != hybrid["tokens"]
    ]
    report["token_mismatch_indices"] = mismatches
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if mismatches:
        raise SystemExit(
            f"Greedy token mismatch at requests {mismatches}; see {args.output}"
        )
    print(f"All {len(prompts)} token sequences match; report: {args.output}")


if __name__ == "__main__":
    main()
