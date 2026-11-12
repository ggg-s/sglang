"""Measure all-long and mixed drains through a native-PD SMG test deployment."""

import argparse
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from bench_p4_dsv_drain import rpc, sse_events


def stream_generate(url, body, first_content, timeout):
    request = Request(
        url.rstrip("/") + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    rid, finish_reason = None, None
    output_ids = []
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
                        "rid": rid,
                        "finish_reason": finish_reason,
                        "content_chunks": chunks,
                        "completion_tokens": len(output_ids),
                        "max_chunk_gap_ms": max_gap_ms,
                    }
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])
                meta = event["meta_info"]
                if rid is None:
                    rid = meta["id"]
                    if not rid.startswith("pd-"):
                        raise RuntimeError(f"unexpected native PD request id: {rid}")
                elif meta["id"] != rid:
                    raise RuntimeError("request id changed during SSE")
                current_ids = event["output_ids"]
                if current_ids[: len(output_ids)] != output_ids:
                    raise RuntimeError("generated token prefix changed during SSE")
                if len(current_ids) > len(output_ids):
                    output_ids = current_ids
                    now = time.monotonic()
                    if last_chunk is not None:
                        max_gap_ms = max(max_gap_ms or 0, (now - last_chunk) * 1000)
                    last_chunk = now
                    chunks += 1
                    if not first_content.done():
                        first_content.set_result(None)
                if meta["finish_reason"] is not None:
                    if finish_reason is not None:
                        raise RuntimeError("duplicate finish_reason")
                    finish_reason = meta["finish_reason"]["type"]
                    if finish_reason not in {"stop", "length"}:
                        raise RuntimeError(meta["finish_reason"])
        raise ConnectionError("SSE EOF before [DONE]")
    except Exception as exc:
        if not first_content.done():
            first_content.set_exception(exc)
        raise


def peak_cutovers(moved):
    events = []
    for row in moved:
        timing = row["drain_member"]["handoff_timing"]
        begin = timing["freeze_requested_at"]
        end = timing["target_admitted_at"]
        if begin is not None and end is not None and begin < end:
            events.extend(((begin, 1), (end, -1)))
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def run(args, prompt):
    status_path = "/v1/drain/status"
    if rpc(args.admin, status_path, timeout=5)["state"] != "READY":
        raise RuntimeError("use a fresh controller state directory for each run")
    groups = [("long", args.long_tokens)]
    if args.mode == "mixed":
        groups.append(("short", args.short_tokens))
    with ThreadPoolExecutor(max_workers=len(groups) * args.per_group) as pool:
        jobs = []
        for group, limit in groups:
            for _ in range(args.per_group):
                first_content = Future()
                body = {
                    "model": args.model,
                    "input_ids": prompt,
                    "stream": True,
                    "sampling_params": {
                        "max_new_tokens": limit,
                        "temperature": 0,
                        "ignore_eos": args.mode == "all-long",
                    },
                }
                stream = pool.submit(
                    stream_generate, args.smg, body, first_content, args.timeout
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
            if drain["state"] in {"ABORTING", "ABORTED"}:
                raise RuntimeError(f"drain aborted during measurement: {drain}")
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

    if args.journal is not None:
        entries = json.loads(args.journal.read_text())["entries"]
        for row in requests:
            row["initial_decode"] = entries[row["rid"]]["workers"]["src"]

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
    peak = peak_cutovers(moved)
    waves = drain["control_wave_sizes"]
    matched = len(members) == len(drain["members"])
    valid_mixed = (
        matched and bool(short_natural) and any(row["group"] == "long" for row in moved)
    )
    valid_all_long = (
        matched
        and len(members) == len(moved) == 4
        and peak == drain["migration_limits"]["cutover"]
        and all(waves.get(worker) for worker in (args.src, args.dst))
        and (
            args.journal is None
            or sum(row["initial_decode"] == args.src for row in requests) == 4
        )
    )
    return {
        "mode": args.mode,
        "source_safe_to_stop_ms": safe_ms,
        "activate_to_safe_to_stop_ms": (
            drain["source_released_at"] - drain["activated_at"]
        )
        * 1000,
        "peak_cutovers": peak,
        "migration_limits": drain["migration_limits"],
        "control_wave_sizes": waves,
        "input_tokens": len(prompt),
        "short_tokens": args.short_tokens,
        "long_tokens": args.long_tokens,
        "moved": len(moved),
        "short_natural": len(short_natural),
        "precopy_bytes": sum(row["drain_member"]["precopy_bytes"] for row in members),
        "moved_precopy_bytes": sum(
            row["drain_member"]["precopy_bytes"] for row in moved
        ),
        "valid_run": valid_mixed if args.mode == "mixed" else valid_all_long,
        "valid_mixed_run": valid_mixed,
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
    parser.add_argument("--mode", choices=("all-long", "mixed"), default="mixed")
    parser.add_argument("--prompt-ids", required=True, type=Path)
    parser.add_argument(
        "--journal",
        type=Path,
        help="Controller journal used to label initial Decode workers",
    )
    parser.add_argument("--short-tokens", type=int, default=192)
    parser.add_argument("--long-tokens", type=int, default=1024)
    parser.add_argument("--per-group", type=int)
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
    if args.per_group is None:
        args.per_group = 6 if args.mode == "mixed" else 12
    if args.per_group < 1 or args.long_tokens < 1:
        parser.error("per-group and long-tokens must be positive")
    if args.mode == "mixed" and not 0 < args.short_tokens < args.long_tokens:
        parser.error("mixed mode requires 0 < short-tokens < long-tokens")
    result = run(args, prompt)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "mode",
                    "valid_run",
                    "source_safe_to_stop_ms",
                    "activate_to_safe_to_stop_ms",
                    "peak_cutovers",
                    "moved",
                    "short_natural",
                    "precopy_bytes",
                )
            }
        )
    )
    raise SystemExit(0 if result["valid_run"] else 1)


if __name__ == "__main__":
    main()
