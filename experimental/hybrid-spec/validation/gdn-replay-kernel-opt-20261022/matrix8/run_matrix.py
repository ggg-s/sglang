"""Fresh ABBA comparison of old/new Replay kernels at capacity eight."""

import hashlib
import json
import socket
import subprocess
import time
from pathlib import Path

import bench_helpers as h
import checks

ROOT = Path(__file__).resolve().parent
ORDER = [
    ("baseline-replay", 1),
    ("optimized-replay", 1),
    ("optimized-replay", 2),
    ("baseline-replay", 2),
]


def main():
    deadline = time.monotonic() + 1200
    gate = ROOT.parent / "optimized" / "optimized.exit"
    while not gate.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Optimized C1 validation did not finish")
        time.sleep(2)
    assert gate.read_text().strip() == "0"
    prompts = h.prepare()
    hashes = json.loads((ROOT / "source-opt-manifest.json").read_text())
    for name, expected in hashes.items():
        assert (
            hashlib.sha256((ROOT / "source-opt" / name).read_bytes()).hexdigest()
            == expected
        ), name
    pid = None
    try:
        for mode, repeat in ORDER:
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", 39924)) != 0
            label = f"{mode}-boot{repeat}"
            with (ROOT / (label + "-server.log")).open("w") as log:
                proc = subprocess.Popen(
                    ["bash", str(ROOT / "launch.sh"), mode],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            pid = proc.pid
            (ROOT / (label + ".pid")).write_text(str(pid) + "\n")
            h.event("launch", mode=mode, round=repeat)
            h.wait_ready(pid)
            info = h.request("/get_server_info")
            h.save(ROOT / (label + "-ready.json"), info)
            state = info["internal_states"][0]
            assert state["effective_max_running_requests_per_dp"] == 8
            assert state["memory_usage"]["token_capacity"] == 131072
            assert info["mamba_ssm_dtype"] == "float32"
            assert info["enable_linear_replayssm_spec"]
            checks.sanity(label, prompts)
            h.preheat(prompts, label)
            for concurrency in (1, 8) if repeat == 1 else (8, 1):
                checks.warm(prompts, concurrency)
                h.benchmark(mode, repeat, concurrency)
            h.stop(pid)
            pid = None
        h.save(
            ROOT / "performance-status.json",
            {"status": "complete", "measured_requests": 400, "order": ORDER},
        )
    finally:
        h.stop(pid)


if __name__ == "__main__":
    main()
