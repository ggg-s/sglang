"""Measure actual P4 DSV ownership cutover; natural completion is not migration."""

import argparse
import concurrent.futures
import json
import time
import uuid
import urllib.request
from pathlib import Path


def measure(url, prompt, max_new, cut, timeout):
    rid = "bench-" + uuid.uuid4().hex
    body = dict(
        rid=rid,
        prompt_ids=prompt,
        max_new_tokens=max_new,
        cut=cut,
        growth_tokens=max_new,
    )
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    tokens, statuses = [], []
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            elapsed = (time.monotonic() - started) * 1000
            if "token_id" in event:
                tokens.append(dict(event, elapsed_ms=elapsed))
            else:
                statuses.append(event)
    seq = [t["seq"] for t in tokens]
    contiguous = seq == list(range(len(seq)))
    owners = [t["owner"] for t in tokens]
    migrated = "src" in owners and "dst" in owners
    monotonic_owner = not migrated or "src" not in owners[owners.index("dst") :]
    first_target = next((t for t in tokens if t["owner"] == "dst"), None)
    return dict(
        rid=rid,
        complete=done,
        migrated=migrated,
        tokens=len(tokens),
        contiguous=contiguous,
        monotonic_owner=monotonic_owner,
        ttft_ms=tokens[0]["elapsed_ms"] if tokens else None,
        e2el_ms=tokens[-1]["elapsed_ms"] if tokens else None,
        first_target_ms=first_target["elapsed_ms"] if first_target else None,
        max_token_gap_ms=max(
            (b["elapsed_ms"] - a["elapsed_ms"] for a, b in zip(tokens, tokens[1:])),
            default=None,
        ),
        statuses=statuses,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", default="http://127.0.0.1:31080")
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--cut", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency <= 0 or not 0 < args.cut < args.max_new_tokens:
        parser.error("require concurrency > 0 and 0 < cut < max-new-tokens")
    prompt = json.loads(args.prompt_ids.read_text())
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in prompt)
    ):
        parser.error("prompt-ids must contain a nonempty JSON list of token ids")

    def run(_):
        try:
            return measure(
                args.gateway, prompt, args.max_new_tokens, args.cut, args.timeout
            )
        except Exception as exc:
            return dict(complete=False, migrated=False, error=repr(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(run, range(args.concurrency)))
    passed = all(
        r.get("complete")
        and r.get("migrated")
        and r.get("contiguous")
        and r.get("monotonic_owner")
        for r in rows
    )
    result = dict(
        protocol="p4-dsv-sse",
        input_tokens=len(prompt),
        concurrency=args.concurrency,
        passed=passed,
        requests=rows,
        scope="Client SSE timings and actual src-to-dst ownership; not bulk-copy duration or SAFE_TO_STOP proof",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
