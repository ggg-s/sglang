"""Replay tokenized LongBench requests against /generate with streaming timings.

Input: a JSON list, or {"records": [...], "provenance": ...}; each record has
input_ids, id, task, and prompt_sha256. No tokenizer or private server hook is
required. This is a performance replay, not a LongBench answer-quality score.

Each trial flushes the cache, runs every record cold, and immediately replays
every record without another flush. Repeated means a cache-reuse opportunity,
not a guarantee that every prefix remains resident. Warmup is stored separately.

Example:
  python bench_longbench_pdmux.py --workload workload.json --output results.json

Requires aiohttp. Use a dedicated, otherwise idle server: this client flushes
its prefix cache. No generation request is retried automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import hashlib
import json
import math
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


class SSEDecoder:
    """Incrementally decode UTF-8 and SSE frames, independent of HTTP chunks."""

    def __init__(self):
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""
        self._data = []

    def _line(self, line):
        if not line:
            if self._data:
                event = "\n".join(self._data)
                self._data.clear()
                return event
            return None
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if field == "data":
            self._data.append(value[1:] if value.startswith(" ") else value)
        return None

    def feed(self, raw, *, final=False):
        self._buffer += self._decoder.decode(raw, final=final)
        events = []
        # LF, CRLF, and CR are all valid SSE line endings. Keep a trailing CR
        # until the next chunk so a split CRLF cannot create a spurious blank.
        while True:
            lf, cr = self._buffer.find("\n"), self._buffer.find("\r")
            positions = [p for p in (lf, cr) if p >= 0]
            if not positions:
                break
            end = min(positions)
            if self._buffer[end] == "\r" and end + 1 == len(self._buffer) and not final:
                break
            width = 2 if self._buffer[end : end + 2] == "\r\n" else 1
            event = self._line(self._buffer[:end])
            self._buffer = self._buffer[end + width :]
            if event is not None:
                events.append(event)
        if final:
            if self._buffer:
                self._line(self._buffer)
                self._buffer = ""
            event = self._line("")
            if event is not None:
                events.append(event)
        return events


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def load_workload(path, context_length, output_tokens):
    raw = path.read_bytes()
    workload = json.loads(raw)
    rows = workload if isinstance(workload, list) else workload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Workload must be a nonempty list or an object with records")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(
            name not in row for name in ("id", "task", "prompt_sha256", "input_ids")
        ):
            raise ValueError(f"Record {index} lacks required fields")
        ids = row["input_ids"]
        if not isinstance(ids, list) or not ids or any(
            type(token) is not int or token < 0 for token in ids
        ):
            raise ValueError(f"Record {index} has invalid input_ids")
        if len(ids) + output_tokens > context_length:
            raise ValueError(
                f"Record {index}: {len(ids)} input + {output_tokens} output exceeds "
                f"context length {context_length}; prepare/truncate the workload first"
            )
        identity = (str(row["task"]), str(row["id"]))
        if identity in seen:
            raise ValueError(f"Duplicate task/id: {identity}")
        seen.add(identity)
        digest = row["prompt_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in digest
        ):
            raise ValueError(f"Record {index} has invalid prompt_sha256")
    return rows, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(path.resolve()),
        "provenance": workload.get("provenance") if isinstance(workload, dict) else None,
        "requests": len(rows),
        "prompt_tokens": sum(len(row["input_ids"]) for row in rows),
    }


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


async def generate(session, url, row, *, output_tokens, request_id):
    started = time.perf_counter()
    tokens, pushes, count, meta = [], [], 0, {}
    done = False
    result = {
        "id": row["id"],
        "task": row["task"],
        "prompt_sha256": row["prompt_sha256"],
        "prompt_tokens": len(row["input_ids"]),
        "request_id": request_id,
        "success": False,
    }

    def consume(payload):
        nonlocal count, meta, tokens, done
        if payload.strip() == "[DONE]":
            done = True
            return
        event = json.loads(payload)
        if not isinstance(event, dict):
            raise ValueError("SSE payload must be a JSON object")
        if "error" in event:
            raise RuntimeError(f"Server stream error: {event['error']}")
        new_meta = event.get("meta_info")
        if not isinstance(new_meta, dict):
            raise ValueError("SSE event lacks meta_info")
        meta.update(new_meta)
        current = meta.get("completion_tokens")
        if type(current) is not int or current < count:
            raise ValueError(f"Invalid completion_tokens: {current!r}, previous={count}")
        chunk = event.get("output_ids", [])
        if not isinstance(chunk, list) or any(type(token) is not int for token in chunk):
            raise ValueError("Invalid output_ids")
        if len(chunk) == current:
            if chunk[:count] != tokens:
                raise ValueError("Cumulative output_ids changed an already emitted token")
            tokens = list(chunk)
        elif len(chunk) == current - count:
            tokens.extend(chunk)
        else:
            raise ValueError(
                f"Streaming token accounting mismatch: ids={len(chunk)}, "
                f"completion_tokens={current}, previous={count}"
            )
        if current > count:
            elapsed_ms = (time.perf_counter() - started) * 1000
            gap = elapsed_ms - pushes[-1]["elapsed_ms"] if pushes else None
            pushes.append(
                {
                    "elapsed_ms": elapsed_ms,
                    "completion_tokens": current,
                    "new_tokens": current - count,
                    "gap_ms": gap,
                    "normalized_gap_ms_per_token": gap / (current - count)
                    if gap is not None
                    else None,
                }
            )
        count = current

    try:
        async with session.post(
            url + "/generate",
            json={
                "rid": request_id,
                "input_ids": row["input_ids"],
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "top_k": 1,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                },
            },
        ) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:4000]}")
            parser = SSEDecoder()
            async for raw in response.content.iter_chunked(65536):
                for payload in parser.feed(raw):
                    consume(payload)
                    if done:
                        break
                if done:
                    break
            if not done:
                for payload in parser.feed(b"", final=True):
                    consume(payload)
                    if done:
                        break
            if not done:
                raise RuntimeError("Stream ended without [DONE]")
            if count != output_tokens or len(tokens) != count:
                raise RuntimeError(
                    f"Incomplete output: expected={output_tokens}, completion_tokens={count}, "
                    f"output_ids={len(tokens)}"
                )
            if not pushes:
                raise RuntimeError("Stream produced no tokens")
            finish = meta.get("finish_reason")
            if isinstance(finish, dict) and finish.get("type") in ("abort", "error"):
                raise RuntimeError(f"Unsuccessful finish_reason: {finish}")
            result["success"] = True
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        first_count = pushes[0]["completion_tokens"] if pushes else 0
        denominator = count - first_count
        result.update(
            output_ids=tokens,
            completion_tokens=count,
            meta=meta,
            cache={key: value for key, value in meta.items() if "cache" in key.lower()},
            stream_done=done,
            e2e_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=pushes[0]["elapsed_ms"] if pushes else None,
            last_token_ms=pushes[-1]["elapsed_ms"] if pushes else None,
            first_push_tokens=first_count,
            tpot_denominator_tokens=denominator,
            tpot_ms=(pushes[-1]["elapsed_ms"] - pushes[0]["elapsed_ms"]) / denominator
            if denominator > 0
            else None,
            pushes=pushes,
            push_gaps_ms=[push["gap_ms"] for push in pushes[1:]],
            normalized_push_gaps_ms_per_token=[
                push["normalized_gap_ms_per_token"] for push in pushes[1:]
            ],
        )
    return result


async def snapshot(session, args, directory, label):
    output = {}
    endpoints = []
    if args.capture_server_info:
        endpoints.append(("server_info", "/get_server_info"))
    if args.capture_metrics:
        endpoints.append(("metrics", "/metrics"))
    for name, endpoint in endpoints:
        try:
            async with session.get(args.url + endpoint, timeout=30) as response:
                raw = await response.read()
                path = directory / f"{label}.{name}.txt"
                path.write_bytes(raw)
                output[name] = {
                    "status": response.status,
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
        except Exception as error:
            output[name] = {"error": f"{type(error).__name__}: {error}"}
    return output


async def flush_cache(session, url):
    started = time.perf_counter()
    try:
        async with session.post(url + "/flush_cache?timeout=30", timeout=45) as response:
            body = await response.text()
            return {
                "success": response.status == 200,
                "http_status": response.status,
                "body": body,
                "wall_s": time.perf_counter() - started,
            }
    except Exception as error:
        return {"success": False, "error": f"{type(error).__name__}: {error}"}


def summarize(records, wall_s, *, cache_precondition_valid):
    successful = [row for row in records if row["success"]]
    tokens = sum(row["completion_tokens"] for row in successful)
    result = {
        "success": len(successful) == len(records) and cache_precondition_valid,
        "cache_precondition_valid": cache_precondition_valid,
        "requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "successful_output_tokens": tokens,
        "wall_s": wall_s,
        "successful_output_tok_s": tokens / wall_s if wall_s else None,
        "successful_request_s": len(successful) / wall_s if wall_s else None,
    }
    result["valid_output_tok_s"] = result["successful_output_tok_s"] if result["success"] else None
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [row[metric] for row in successful if row[metric] is not None]
        result[metric] = {
            "n": len(values),
            "mean": sum(values) / len(values) if values else None,
            **{f"p{p}": percentile(values, p) for p in (50, 95, 99)},
        }
    result["cached_tokens"] = sum(
        row["meta"].get("cached_tokens", 0)
        for row in successful
        if isinstance(row["meta"].get("cached_tokens", 0), (int, float))
    )
    return result


async def run_phase(session, args, rows, directory, label, run_id, *, cache_precondition_valid):
    before = await snapshot(session, args, directory, label + ".before")
    semaphore = asyncio.Semaphore(args.concurrency)
    jsonl_path = directory / f"{label}.requests.jsonl"
    started = time.perf_counter()
    with jsonl_path.open("w") as jsonl:

        async def one(index, row):
            async with semaphore:
                dispatched = time.perf_counter()
                record = await generate(
                    session,
                    args.url,
                    row,
                    output_tokens=args.output_tokens,
                    request_id=f"longbench-{run_id}-{label}-{index}",
                )
                record.update(index=index, client_queue_wait_ms=(dispatched - started) * 1000)
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()
                return record

        records = await asyncio.gather(*(one(index, row) for index, row in enumerate(rows)))
    wall_s = time.perf_counter() - started
    result = summarize(records, wall_s, cache_precondition_valid=cache_precondition_valid)
    result.update(
        phase=label,
        records=records,
        records_jsonl=str(jsonl_path.resolve()),
        snapshots={"before": before, "after": await snapshot(session, args, directory, label + ".after")},
    )
    print(label, json.dumps({key: value for key, value in result.items() if key not in ("records", "snapshots")}), flush=True)
    return result


async def main(args):
    import aiohttp

    rows, workload_info = load_workload(args.workload, args.context_length, args.output_tokens)
    args.url = args.url.rstrip("/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    directory = args.output.parent / f"{args.output.stem}.artifacts-{run_id}"
    directory.mkdir()
    results = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "workload": workload_info,
        "server_url": args.url,
        "python": platform.python_version(),
        "aiohttp": aiohttp.__version__,
        "concurrency": args.concurrency,
        "output_tokens_per_request": args.output_tokens,
        "context_length": args.context_length,
        "requested_trials": args.trials,
        "sampling": {"temperature": 0, "top_k": 1, "ignore_eos": True},
        "performance_only": True,
        "timing_definitions": {
            "ttft_ms": "Client dispatch to first SSE push containing output tokens; excludes semaphore queue time.",
            "e2e_ms": "Client dispatch to [DONE] or terminal failure; excludes semaphore queue time.",
            "tpot_ms": "(last token push - first token push) / (completion_tokens - first_push_tokens); null if denominator is zero.",
            "push_gaps_ms": "Time between successive pushes containing new tokens, excluding the first push.",
            "normalized_push_gaps_ms_per_token": "Each push gap divided by tokens newly delivered in that push; an aggregate, not observed per-token ITL.",
            "wall_s": "Full phase client makespan including concurrency queueing; excludes flush, snapshots, and warmup.",
        },
        "trials": [],
        "success": False,
    }
    save_json(args.output, results)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.warmup_requests:
            warmup_flush = await flush_cache(session, args.url)
            results["warmup_flush"] = warmup_flush
            if not warmup_flush["success"]:
                results["error"] = "Cache flush before warmup failed; measured phases were not run"
                save_json(args.output, results)
                return 1
            results["warmup"] = await run_phase(
                session, args, rows[: args.warmup_requests], directory, "warmup", run_id,
                cache_precondition_valid=True,
            )
            results["warmup"]["excluded_from_measurements"] = True
            save_json(args.output, results)
            if not results["warmup"]["success"]:
                results["error"] = "Warmup failed; measured phases were not run"
                save_json(args.output, results)
                return 1
        for trial_number in range(1, args.trials + 1):
            trial = {"trial": trial_number, "flush": await flush_cache(session, args.url)}
            results["trials"].append(trial)
            save_json(args.output, results)
            if not trial["flush"]["success"]:
                trial["success"] = False
                trial["error"] = "Cache flush failed; cold and repeated phases were not run"
                save_json(args.output, results)
                break
            for phase in ("cold", "repeated"):
                trial[phase] = await run_phase(
                    session, args, rows, directory, f"trial_{trial_number:02d}_{phase}", run_id,
                    cache_precondition_valid=phase == "cold" or trial["cold"]["success"],
                )
                save_json(args.output, results)
            trial["success"] = trial["cold"]["success"] and trial["repeated"]["success"]
            save_json(args.output, results)
    results["success"] = len(results["trials"]) == args.trials and all(
        trial.get("success", False) for trial in results["trials"]
    )
    results["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(args.output, results)
    return 0 if results["success"] else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:39928")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--capture-server-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-metrics", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    for name in ("concurrency", "output_tokens", "context_length", "trials", "request_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be nonnegative")
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
