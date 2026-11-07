"""Common-origin natural/D2D drain measurement against a real P4 gateway.

Run each mode on an otherwise idle, freshly started gateway/source pair, with
identical P prefix-cache warmup, prompts, concurrency, output and trigger token.
The clock starts BEFORE the drain HTTP request, including pre-copy cost.
"""

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path


def rpc(url, path, body=None, timeout=600):
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def stream(url, body, row, lock, timeout):
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                with lock:
                    if data == b"[DONE]":
                        row["done"] = True
                        return
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(event["error"])
                    if "token_id" in event:
                        row["tokens"].append(dict(event, at=time.monotonic()))
    except Exception as exc:
        with lock:
            row["error"] = repr(exc)


def run(url, prompt, concurrency, max_new, trigger, mode, timeout):
    lock = threading.Lock()
    if rpc(url, "/v1/drain/status").get("active"):
        raise ValueError("use a fresh gateway/source for each comparison run")
    # Identical complete warmup before either measured mode (no cut migration).
    warm_rid = "warm-" + uuid.uuid4().hex
    warm = dict(tokens=[], done=False)
    stream(
        url,
        dict(rid=warm_rid, prompt_ids=prompt, max_new_tokens=1, cut=1),
        warm,
        lock,
        timeout,
    )
    if not warm["done"] or warm.get("error"):
        raise RuntimeError(f"warmup failed: {warm}")
    warm_deadline = time.monotonic() + timeout
    while rpc(url, "/v1/router/status")["migrations"][warm_rid]["phase"] != "DONE":
        if time.monotonic() > warm_deadline:
            raise TimeoutError("warmup KV was not retired")
        time.sleep(0.005)
    rows = [
        dict(rid="drain-" + uuid.uuid4().hex, tokens=[], done=False)
        for _ in range(concurrency)
    ]
    deadline = time.monotonic() + timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                stream,
                url,
                dict(
                    rid=r["rid"],
                    prompt_ids=prompt,
                    max_new_tokens=max_new,
                    cut=max_new,
                    growth_tokens=max_new,
                ),
                r,
                lock,
                timeout,
            )
            for r in rows
        ]
        while True:
            with lock:
                if any(r.get("error") or r["done"] for r in rows):
                    raise RuntimeError(
                        "request finished/failed before common drain trigger"
                    )
                ready = all(len(r["tokens"]) >= trigger for r in rows)
            if ready:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("common drain trigger missed")
            time.sleep(0.001)
        t0 = time.monotonic()
        snapshot = rpc(url, "/v1/drain", dict(mode=mode), timeout)
        if set(snapshot["members"]) != {r["rid"] for r in rows}:
            raise RuntimeError(
                "drain contains other workload or a request already finished"
            )
        while snapshot["state"] not in {"SAFE_TO_STOP", "STOPPED"}:
            if time.monotonic() > deadline:
                raise TimeoutError(f"source did not release safely: {snapshot}")
            time.sleep(0.005)
            snapshot = rpc(url, "/v1/drain/status", timeout=timeout)
        source_release_ms = (time.monotonic() - t0) * 1000
        for future in futures:
            future.result(timeout=max(1, deadline - time.monotonic()))
    result = []
    for row in rows:
        tokens = row["tokens"]
        owners = [x["owner"] for x in tokens]
        first = next((x for x in tokens if x["owner"] == "dst"), None)
        last = next((x for x in reversed(tokens) if x["owner"] == "src"), None)
        result.append(
            dict(
                rid=row["rid"],
                done=row["done"],
                error=row.get("error"),
                tokens=len(tokens),
                contiguous=[x["seq"] for x in tokens] == list(range(len(tokens))),
                owner_monotonic="dst" not in owners
                or "src" not in owners[owners.index("dst") :],
                migrated=first is not None,
                first_target_from_drain_ms=None
                if first is None
                else (first["at"] - t0) * 1000,
                last_source_to_first_target_ms=None
                if first is None or last is None
                else (first["at"] - last["at"]) * 1000,
                max_token_gap_ms=max(
                    ((b["at"] - a["at"]) * 1000 for a, b in zip(tokens, tokens[1:])),
                    default=None,
                ),
            )
        )
    return dict(
        mode=mode,
        concurrency=concurrency,
        input_tokens=len(prompt),
        max_new_tokens=max_new,
        trigger_token=trigger,
        warmed=True,
        origin="before POST /v1/drain; includes all plan/reserve/precopy/freeze/import/retirement",
        source_safe_to_stop_ms=source_release_ms,
        drain=snapshot,
        router=rpc(url, "/v1/router/status"),
        requests=result,
        passed=all(
            r["done"]
            and not r["error"]
            and r["contiguous"]
            and r["owner_monotonic"]
            and (r["migrated"] if mode == "d2d" else not r["migrated"])
            for r in result
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--mode", choices=("natural", "d2d"), required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 8), default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--trigger-token", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompt = json.loads(args.prompt_ids.read_text())
    if (
        not prompt
        or not isinstance(prompt, list)
        or any(type(x) is not int or x < 0 for x in prompt)
    ):
        parser.error("prompt-ids must be a nonempty token-id JSON list")
    if not 1 <= args.trigger_token < args.max_new_tokens:
        parser.error("require 1 <= trigger-token < max-new-tokens")
    result = run(
        args.gateway,
        prompt,
        args.concurrency,
        args.max_new_tokens,
        args.trigger_token,
        args.mode,
        args.timeout,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
