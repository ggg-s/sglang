"""Separate real-prompt prefill and warmed steady decode diagnostic traces."""

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import bench_helpers as h

ROOT = Path(__file__).resolve().parent
MODE = sys.argv[1]


async def post(session, path, payload):
    async with session.post(h.URL + path, json=payload) as response:
        response.raise_for_status()
        raw = await response.text()
        try:
            return json.loads(raw)
        except ValueError:
            return raw


def wait_trace(folder):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        files = list(folder.glob("*.trace.json.gz"))
        if files and all(path.stat().st_size > 1000 for path in files):
            return files
        time.sleep(1)
    raise TimeoutError(f"Trace not flushed: {folder}")


def profile_config(folder, steps=None):
    return {
        "output_dir": str(folder),
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": False,
        "profile_by_stage": False,
        "profile_id": folder.name,
        "num_steps": steps,
        "detailed_annotations": True,
    }


async def prefill(prompts):
    folder = ROOT / "traces" / MODE / "PREFILL-c1"
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("*.trace.json.gz")) and (folder / "response.json").exists():
        return
    h.request("/flush_cache?timeout=120", {}, timeout=150)
    cfg = profile_config(folder)
    h.save(folder / "profile-request.json", cfg)
    # Unique, complete real Agent contexts; one output token prevents decode.
    payload = {
        "text": prompts[:1],
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        h.save(
            folder / "start-response.json", await post(session, "/start_profile", cfg)
        )
        h.save(folder / "response.json", await post(session, "/generate", payload))
        h.save(folder / "stop-response.json", await post(session, "/stop_profile", {}))
    wait_trace(folder)


async def decode(prompts, concurrency):
    folder = ROOT / "traces" / MODE / f"DECODE-c{concurrency}"
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("*.trace.json.gz")) and (folder / "responses.json").exists():
        return
    h.request("/flush_cache?timeout=120", {}, timeout=150)
    selected = prompts[:concurrency]
    # Cold long prompts can enter decode at different times. Populate all prompt
    # cache entries first, then drain this workload before starting live rows.
    h.save(
        folder / "prefill-warmup-responses.json",
        h.request(
            "/generate",
            {
                "text": selected,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
            timeout=600,
        ),
    )
    latest, progress = [None] * concurrency, [0] * concurrency
    ready = asyncio.Event()
    cfg = profile_config(folder, steps=11)
    payload = {
        "text": selected,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1024,
            "ignore_eos": True,
        },
    }
    h.save(folder / "profile-request.json", cfg)
    h.save(
        folder / "workload.json",
        {
            "prompt_indices": list(range(concurrency)),
            "max_new_tokens": 1024,
            "ignore_eos": True,
            "warmup_min_completion_tokens": 192,
            "prompt_cache_warmed_and_drained": True,
            "requested_steps_including_guard": 11,
            "note": "Forced continuation holds batch shape for diagnostic decode only; excluded from serving throughput.",
        },
    )
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=900)
    ) as session:

        async def generate():
            async with session.post(h.URL + "/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.content:
                    raw = line.strip()
                    if not raw.startswith(b"data:") or raw[5:].strip() == b"[DONE]":
                        continue
                    value = json.loads(raw[5:].strip())
                    index = value.get("index", 0)
                    latest[index] = value
                    progress[index] = value.get("meta_info", {}).get(
                        "completion_tokens", 0
                    )
                    if min(progress) >= 192:
                        ready.set()

        task = asyncio.create_task(generate())
        try:
            await asyncio.wait_for(ready.wait(), timeout=300)
            if task.done():
                await task
                raise RuntimeError("Decode workload finished before profiling")
            assert max(progress) < 800, progress
            h.save(
                folder / "capture-start.json", {"completion_tokens": progress.copy()}
            )
            result = await post(session, "/start_profile", cfg)
            h.save(folder / "start-response.json", result)
            await task
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert all(
        value and value["meta_info"]["completion_tokens"] == 1024 for value in latest
    )
    h.save(folder / "responses.json", latest)
    h.save(folder / "after-server-info.json", h.request("/get_server_info"))
    wait_trace(folder)


async def mixed(prompts):
    # A small natural-EOS replay checks that the steady trace's limits are visible.
    folder = ROOT / "traces" / MODE / "MIXED-c8"
    folder.mkdir(parents=True, exist_ok=True)
    if list(folder.glob("*.trace.json.gz")) and (folder / "responses.json").exists():
        return
    h.request("/flush_cache?timeout=120", {}, timeout=150)
    cfg = profile_config(folder, steps=24)
    h.save(folder / "profile-request.json", cfg)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:
        h.save(
            folder / "start-response.json", await post(session, "/start_profile", cfg)
        )
        result = await post(
            session,
            "/generate",
            {
                "text": prompts[:16],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 2048,
                    "ignore_eos": False,
                },
            },
        )
        h.save(folder / "responses.json", result)
        # The API rejects an explicit stop after num_steps already exported.
        if list(folder.glob("*.trace.json.gz")):
            h.save(folder / "stop-response.json", {"automatic_stop": True})
        else:
            h.save(
                folder / "stop-response.json", await post(session, "/stop_profile", {})
            )
    wait_trace(folder)


async def main():
    prompts = [
        json.loads(line)["conversations"][0]["value"]
        for line in (ROOT / "workload.jsonl").read_text().splitlines()
    ]
    await prefill(prompts)
    await decode(prompts, 1)


if __name__ == "__main__":
    asyncio.run(main())
