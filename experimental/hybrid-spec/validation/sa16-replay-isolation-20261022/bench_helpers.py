"""SA16 ReplaySSM isolation on the authorized A100 host."""

import concurrent.futures
import hashlib
import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

OUT = Path(__file__).resolve().parent
SOURCE = OUT / "source"
MODEL = "/root/rivermind-data/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
URL = "http://127.0.0.1:39922"


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def event(kind, **kw):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **kw}
    with (OUT / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def request(path, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        if path.startswith("/flush_cache"):
            return body.decode()
        return json.loads(body) if body else None


def wait_ready(pid):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            raise RuntimeError(f"Server exited: {pid}")
        try:
            request("/model_info", timeout=2)
            return
        except (OSError, json.JSONDecodeError):
            time.sleep(2)
    raise TimeoutError("Server readiness deadline exceeded")


def stop(pid):
    if pid is None:
        return
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    # Terminate only the server launched for this matrix and its children.
    for proc in reversed(processes):
        try:
            proc.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=20)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)
    time.sleep(3)


def prepare():
    provenance = json.loads((OUT / "workload-provenance.json").read_text())
    raw = (OUT / "workload.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == provenance["selected_jsonl_sha256"]
    prompts = [
        json.loads(line)["conversations"][0]["value"] for line in raw.splitlines()
    ]
    assert len(prompts) == 50
    hashes = json.loads((OUT / "source-manifest.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, (
            name
        )
    save(OUT / "source-hashes.json", hashes)
    return prompts


def preheat(prompts, name):
    # /model_info is available before the server's own startup probe finishes.
    # Wait for that probe to release its request slot before clearing the cache.
    request("/flush_cache?timeout=120", {}, timeout=150)

    def generate(prompt):
        return request(
            "/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 256,
                    "ignore_eos": False,
                },
            },
            timeout=600,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        outputs = list(pool.map(generate, prompts[:8]))
    save(OUT / f"{name}-preheat.json", outputs)


def benchmark(mode, round_id, concurrency):
    name = f"{mode}-r{round_id}-c{concurrency}"
    output = OUT / f"{name}.jsonl"
    if output.exists():
        raise FileExistsError(output)
    argv = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        URL,
        "--model",
        MODEL,
        "--tokenizer",
        MODEL,
        "--dataset-name",
        "custom",
        "--dataset-path",
        str(OUT / "workload.jsonl"),
        "--num-prompts",
        "50",
        "--sharegpt-output-len",
        "2048",
        "--sharegpt-context-len",
        "16384",
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--seed",
        "42",
        "--temperature",
        "0",
        "--disable-ignore-eos",
        "--warmup-requests",
        "0",
        "--flush-cache",
        "--output-details",
        "--output-file",
        str(output),
    ]
    save(OUT / f"{name}-command.json", argv)
    save(OUT / f"{name}-before.json", request("/get_server_info"))
    event("benchmark_start", mode=mode, round=round_id, concurrency=concurrency)
    with (OUT / f"{name}-client.log").open("w") as log:
        subprocess.run(
            argv,
            cwd=SOURCE,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    save(OUT / f"{name}-after.json", request("/get_server_info"))
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    if len(rows) != 1 or rows[0]["completed"] != 50 or any(rows[0]["errors"]):
        raise RuntimeError(f"Incomplete benchmark {name}")
    result = rows[0]
    event(
        "benchmark_done",
        mode=mode,
        round=round_id,
        concurrency=concurrency,
        output_tokens=result["total_output_tokens"],
        throughput=result["output_throughput"],
        tpot_ms=result["mean_tpot_ms"],
        accept_length=result["accept_length"],
    )
