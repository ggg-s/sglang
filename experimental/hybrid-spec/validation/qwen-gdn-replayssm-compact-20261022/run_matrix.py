"""Matched serving comparison; run only on the authorized full-model host."""

import concurrent.futures
import json
import subprocess
from pathlib import Path

import bench_helpers as h

ROOT = Path(__file__).resolve().parent
MODES = ("native-k8", "native-k8-replay", "sa16-n8-replay")


def warm(prompts, concurrency):
    h.request("/flush_cache?timeout=120", {}, timeout=150)

    def generate(prompt):
        return h.request(
            "/generate",
            {
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=300,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(generate, prompts[:8]))
    h.request("/flush_cache?timeout=120", {}, timeout=150)


def sanity(mode, prompts):
    directory = ROOT / (mode + "-sanity")
    argv = [
        "/root/rivermind-data/envs/sgl-eval-a100/bin/sgl-eval",
        "run",
        "gsm8k",
        "--base-url",
        h.URL + "/v1",
        "--num-examples",
        "20",
        "--from-dataset",
        str(ROOT / "gsm8k-20.jsonl"),
        "--num-threads",
        "8",
        "--temperature",
        "0",
        "--seed",
        "42",
        "--max-tokens",
        "2048",
        "--chat-template-kwarg",
        "enable_thinking=false",
        "--prompt",
        str(ROOT / "sanity-cot.yaml"),
        "--out-dir",
        str(directory),
    ]
    h.save(ROOT / (mode + "-sanity-command.json"), argv)
    with (ROOT / (mode + "-sanity.log")).open("w") as log:
        subprocess.run(
            argv, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True
        )
    metrics_paths = list(directory.rglob("metrics.json"))
    assert len(metrics_paths) == 1
    metrics = json.loads(metrics_paths[0].read_text())
    assert metrics["aggregate"]["score"] > 0.8, metrics
    # Repeated long-prefix generation exercises real radix checkpoint recovery.
    h.request("/flush_cache?timeout=120", {}, timeout=150)
    results = [
        h.request(
            "/generate",
            {
                "text": prompts[0],
                "sampling_params": {"temperature": 0, "max_new_tokens": 160},
            },
            timeout=600,
        )
        for _ in range(2)
    ]
    h.save(ROOT / (mode + "-cache-recovery.json"), results)
    assert all(result.get("text") for result in results)
    assert results[1]["meta_info"].get("cached_tokens", 0) > 0
    h.save(
        ROOT / (mode + "-sanity-ok.json"),
        {"gsm8k": metrics["aggregate"], "cache_restore": True},
    )


def main():
    cuda = json.loads((ROOT / "cuda-summary.json").read_text())
    assert cuda["gdn_replay"]["failures"] == 0
    assert cuda["gdn_replay"]["errors"] == 0
    assert cuda["gdn_replay"]["skipped"] == 0
    prompts = h.prepare()
    pid = None
    try:
        for mode in MODES:
            with (ROOT / (mode + "-server.log")).open("w") as log:
                proc = subprocess.Popen(
                    ["bash", str(ROOT / "launch.sh"), mode],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            pid = proc.pid
            (ROOT / (mode + "-server.pid")).write_text(str(pid) + "\n")
            h.wait_ready(pid)
            info = h.request("/get_server_info")
            h.save(ROOT / (mode + "-ready.json"), info)
            state = info["internal_states"][0]
            assert state["effective_max_running_requests_per_dp"] == 8
            assert state["memory_usage"]["token_capacity"] == 131072
            sanity(mode, prompts)
            h.preheat(prompts, mode)
            for round_id, concurrencies in ((1, (1, 8)), (2, (8, 1))):
                for concurrency in concurrencies:
                    warm(prompts, concurrency)
                    h.benchmark(mode, round_id, concurrency)
            h.stop(pid)
            pid = None
        h.save(
            ROOT / "performance-status.json",
            {"status": "complete", "measured_requests": 600},
        )
    finally:
        h.stop(pid)


if __name__ == "__main__":
    main()
