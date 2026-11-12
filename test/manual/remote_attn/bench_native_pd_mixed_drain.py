"""Measure a mixed short/long drain through a native-PD SMG test deployment."""

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from bench_p4_dsv_drain import rpc, sse_events


def completion(url, body, first_content, timeout):
    request = Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    completion_id, finish_reason = None, None
    chunks, last_chunk, max_gap_ms = 0, None, None
    try:
        with urlopen(request, timeout=timeout) as response:
            for kind, data in sse_events(response):
                if kind == "error":
                    raise RuntimeError(data)
                if data == "[DONE]":
                    if not finish_reason or not chunks:
                        raise RuntimeError(
                            "SSE completed without content or finish_reason"
                        )
                    return {
                        "rid": completion_id.removeprefix("cmpl-"),
                        "finish_reason": finish_reason,
                        "content_chunks": chunks,
                        "max_chunk_gap_ms": max_gap_ms,
                    }
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                if completion_id is None:
                    completion_id = event["id"]
                    if not completion_id.startswith("cmpl-pd-"):
                        raise RuntimeError(
                            f"unexpected native PD completion id: {completion_id}"
                        )
                elif event["id"] != completion_id:
                    raise RuntimeError("completion id changed during SSE")
                choice = event["choices"][0]
                if choice["text"]:
                    now = time.monotonic()
                    if last_chunk is not None:
                        max_gap_ms = max(max_gap_ms or 0, (now - last_chunk) * 1000)
                    last_chunk = now
                    chunks += 1
                    if not first_content.done():
                        first_content.set_result(None)
                if choice["finish_reason"] is not None:
                    if finish_reason is not None:
                        raise RuntimeError("duplicate finish_reason")
                    finish_reason = choice["finish_reason"]
        raise ConnectionError("SSE EOF before [DONE]")
    except Exception as exc:
        if not first_content.done():
            first_content.set_exception(exc)
        raise


def run(args, prompt):
    status_path = "/v1/drain/status"
    if rpc(args.admin, status_path, timeout=5)["state"] != "READY":
        raise RuntimeError("use a fresh controller state directory for each run")
    with ThreadPoolExecutor(max_workers=2 * args.per_group) as pool:
        jobs = []
        for group, limit in (("long", args.long_tokens), ("short", args.short_tokens)):
            for _ in range(args.per_group):
                first_content = Future()
                body = {
                    "model": args.model,
                    "prompt": prompt,
                    "stream": True,
                    "max_tokens": limit,
                    "temperature": 0,
                }
                stream = pool.submit(
                    completion, args.smg, body, first_content, args.timeout
                )
                jobs.append((group, first_content, stream))
        deadline = time.monotonic() + args.timeout
        for _, first_content, _ in jobs:
            first_content.result(timeout=max(0, deadline - time.monotonic()))
        drain_started = time.monotonic()
        drain = rpc(
            args.admin, "/v1/drain", {"src": args.src, "dst": args.dst}, timeout=5
        )
        drain_id = drain["id"]
        while drain["state"] != "SAFE_TO_STOP":
            if time.monotonic() >= deadline:
                raise TimeoutError(f"source did not reach SAFE_TO_STOP: {drain}")
            time.sleep(0.02)
            drain = rpc(args.admin, status_path, timeout=5)
            if drain["id"] != drain_id:
                raise RuntimeError("drain changed during measurement")
        safe_ms = (time.monotonic() - drain_started) * 1000
        requests = []
        for group, _, stream in jobs:
            row = stream.result(timeout=max(0, deadline - time.monotonic()))
            member = drain["members_state"].get(row["rid"])
            requests.append(dict(group=group, **row, drain_member=member))

    members = [row for row in requests if row["drain_member"] is not None]
    moved = [
        row
        for row in members
        if row["drain_member"]["owner"] == args.dst
        and row["drain_member"]["migration_outcome"] == "migrated"
    ]
    short_natural = [
        row
        for row in members
        if row["group"] == "short"
        and row["drain_member"]["owner"] == args.src
        and row["drain_member"]["phase"] == "DONE"
        and row["drain_member"]["migration_attempts"] == 0
    ]
    return {
        "source_safe_to_stop_ms": safe_ms,
        "input_tokens": len(prompt),
        "short_tokens": args.short_tokens,
        "long_tokens": args.long_tokens,
        "moved": len(moved),
        "short_natural": len(short_natural),
        "precopy_bytes": sum(row["drain_member"]["precopy_bytes"] for row in members),
        "moved_precopy_bytes": sum(
            row["drain_member"]["precopy_bytes"] for row in moved
        ),
        "valid_mixed_run": (
            len(members) == len(drain["members"])
            and bool(short_natural)
            and any(row["group"] == "long" for row in moved)
        ),
        "drain": drain,
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smg", default="http://127.0.0.1:32000")
    parser.add_argument("--admin", default="http://127.0.0.1:31081")
    parser.add_argument("--model", required=True)
    parser.add_argument("--src", default="D1")
    parser.add_argument("--dst", default="D2")
    parser.add_argument("--prompt-ids", required=True, type=Path)
    parser.add_argument("--short-tokens", type=int, default=192)
    parser.add_argument("--long-tokens", type=int, default=1024)
    parser.add_argument("--per-group", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prompt = json.loads(args.prompt_ids.read_text())
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(token) is not int or token < 0 for token in prompt)
    ):
        parser.error("prompt-ids must be a nonempty JSON token-id list")
    if not 0 < args.short_tokens < args.long_tokens or args.per_group < 1:
        parser.error("require 0 < short-tokens < long-tokens and per-group >= 1")
    result = run(args, prompt)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "valid_mixed_run",
                    "source_safe_to_stop_ms",
                    "moved",
                    "short_natural",
                    "precopy_bytes",
                )
            }
        )
    )
    raise SystemExit(0 if result["valid_mixed_run"] else 1)


if __name__ == "__main__":
    main()
