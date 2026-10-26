"""Closed-loop HTTP workload: three decoders plus sequential long prefills.

Use identical fixture JSON for every server. All request/token counts and raw
token IDs are retained. Cache is flushed between rounds; salts prevent the
two Prefill requests from reusing each other's prompt. Warmup is unmeasured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def generate(url, ids, count, *, rid, ready=None):
    started = time.perf_counter()
    first = None
    output_ids, meta = [], {}
    try:
        with requests.post(
            url + "/generate",
            json=dict(
                rid=rid,
                input_ids=ids,
                cache_salt=rid,
                sampling_params=dict(
                    temperature=0, top_k=1, max_new_tokens=count, ignore_eos=True
                ),
                stream=True,
            ),
            stream=True,
            timeout=(10, 600),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=None):
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    break
                data = json.loads(payload)
                if "error" in data:
                    raise RuntimeError(data["error"])
                meta = data["meta_info"]
                completed = meta["completion_tokens"]
                current = data.get("output_ids", [])
                if len(current) == completed:
                    output_ids = current
                elif len(output_ids) + len(current) == completed:
                    output_ids.extend(current)
                elif completed:
                    raise RuntimeError("Streaming token IDs do not match token counts")
                if completed and first is None:
                    first = time.perf_counter()
                    if ready is not None:
                        ready.set()
        ended = time.perf_counter()
        if first is None or len(output_ids) != count:
            raise RuntimeError(f"{rid}: expected {count} tokens, got {len(output_ids)}")
        return dict(
            rid=rid,
            prompt_tokens=len(ids),
            completion_tokens=len(output_ids),
            output_ids=output_ids,
            ttft_ms=(first - started) * 1000,
            tpot_ms=(ended - first) * 1000 / max(1, count - 1),
            latency_ms=(ended - started) * 1000,
            finish_reason=meta.get("finish_reason"),
        )
    finally:
        if ready is not None:
            ready.set()


def run_round(url, fixture, *, label, seed_output=512, prefill_output=64, waves=2):
    requests.post(url + "/flush_cache", params=dict(timeout=30), timeout=40).raise_for_status()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        events = [threading.Event() for _ in range(3)]
        seeds = [
            pool.submit(
                generate,
                url,
                ids,
                seed_output,
                rid=f"{label}-decode-{index}",
                ready=events[index],
            )
            for index, ids in enumerate(fixture["seed_input_ids"])
        ]
        for event, future in zip(events, seeds):
            if not event.wait(timeout=180):
                raise TimeoutError("Seed decoders did not emit their first tokens")
            if future.done():
                future.result()
        prefills = [
            pool.submit(
                generate,
                url,
                fixture["prefill_input_ids"],
                prefill_output,
                rid=f"{label}-prefill-{index}",
            ).result()
            for index in range(waves)
        ]
        results = [future.result() for future in seeds] + prefills
    elapsed = time.perf_counter() - started
    return dict(
        label=label,
        wall_s=elapsed,
        output_tok_s=sum(row["completion_tokens"] for row in results) / elapsed,
        request_s=len(results) / elapsed,
        requests=results,
    )


def fixture_for_model(model_path, tokens):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = (
        "def solve_problem(values):\n    result = []\n    for value in values:\n"
        "        if value > 0:\n            result.append(value * 2)\n"
        "    return result\n\n"
    )
    ids = tokenizer.encode(text * (tokens // 20 + 50), add_special_tokens=False)
    if len(ids) < tokens:
        raise ValueError("Fixture text is shorter than the requested prompt")
    return dict(
        seed_input_ids=[ids[offset : offset + 64] for offset in (0, 7, 19)],
        prefill_input_ids=ids[:tokens],
        workload="synthetic_code_P_plus_D3_closed_loop",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:39928")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed-output", type=int, default=512)
    parser.add_argument("--prefill-output", type=int, default=64)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--label-prefix", default="joint")
    args = parser.parse_args()
    if args.fixture.exists():
        fixture = json.loads(args.fixture.read_text())
    else:
        if not args.model_path:
            parser.error("--model-path is required to create a new fixture")
        fixture = fixture_for_model(args.model_path, args.prefill_tokens)
        args.fixture.write_text(json.dumps(fixture))
    digest = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    rounds = []
    for trial in range(-1, args.trials):
        suffix = "warmup" if trial == -1 else f"trial-{trial}"
        label = f"{args.label_prefix}-{suffix}"
        row = run_round(
            args.url,
            fixture,
            label=label,
            seed_output=args.seed_output,
            prefill_output=args.prefill_output,
            waves=args.waves,
        )
        print(label, row["output_tok_s"], flush=True)
        if trial >= 0:
            rounds.append(row)
        args.output.write_text(
            json.dumps(
                dict(
                    workload=fixture["workload"],
                    fixture_sha256=digest,
                    warmup_excluded=True,
                    trials=rounds,
                    median_output_tok_s=(
                        statistics.median(r["output_tok_s"] for r in rounds)
                        if rounds
                        else None
                    ),
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
