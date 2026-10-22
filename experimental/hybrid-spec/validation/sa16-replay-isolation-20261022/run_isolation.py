"""ABBA serving runs; only ReplaySSM differs between the two arms."""

import json
import socket
import subprocess
from pathlib import Path

import bench_helpers as h
import checks

ROOT = Path(__file__).resolve().parent
ORDER = [("sa16-snapshot", 1), ("sa16-replay", 1), ("sa16-replay", 2), ("sa16-snapshot", 2)]


def main():
    prompts = h.prepare()
    pid = None
    try:
        for mode, repeat in ORDER:
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", 39922)) != 0, "Port already in use"
            label = f"{mode}-boot{repeat}"
            with (ROOT / (label + "-server.log")).open("w") as log:
                proc = subprocess.Popen(
                    ["bash", str(ROOT / "launch.sh"), mode],
                    stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
            pid = proc.pid
            (ROOT / (label + ".pid")).write_text(str(pid) + "\n")
            h.event("launch", mode=mode, round=repeat, concurrency=1)
            h.wait_ready(pid)
            info = h.request("/get_server_info")
            h.save(ROOT / (label + "-ready.json"), info)
            state = info["internal_states"][0]
            assert state["effective_max_running_requests_per_dp"] == 1
            assert state["memory_usage"]["token_capacity"] == 131072
            assert info["mamba_ssm_dtype"] == "float32"
            assert info["cuda_graph_max_bs_decode"] == 1
            assert info["enable_linear_replayssm_spec"] == (mode == "sa16-replay")
            checks.sanity(label, prompts)
            h.preheat(prompts, label)
            checks.warm(prompts, 1)
            h.benchmark(mode, repeat, 1)
            h.stop(pid)
            pid = None
        h.save(ROOT / "performance-status.json", {
            "status": "complete", "measured_requests": 200,
            "order": ORDER, "source_commit": "50084814319d2e61eee6499500e9fe1904a2bc8b",
        })
    finally:
        h.stop(pid)


if __name__ == "__main__":
    main()
