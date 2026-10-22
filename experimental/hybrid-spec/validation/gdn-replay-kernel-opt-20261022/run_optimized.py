"""Validate optimized full-model C1 serving, then capture diagnostic profiles."""

import subprocess
import sys
import time
from pathlib import Path

import bench_helpers as h
import checks

ROOT = Path(__file__).resolve().parent


def main():
    deadline = time.monotonic() + 900
    while not (ROOT / "components.exit").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Numerical regression did not finish")
        time.sleep(2)
    assert (ROOT / "registered.exit").read_text().strip() == "0"
    assert (ROOT / "components.exit").read_text().strip() == "0"
    prompts = h.prepare()
    pid = None
    try:
        with (ROOT / "sa16-replay-opt-server.log").open("w") as log:
            proc = subprocess.Popen(
                ["bash", str(ROOT / "launch.sh"), "sa16-replay"],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        pid = proc.pid
        (ROOT / "sa16-replay-opt-server.pid").write_text(str(pid) + "\n")
        h.event("launch", mode="sa16-replay-opt")
        h.wait_ready(pid)
        info = h.request("/get_server_info")
        h.save(ROOT / "sa16-replay-opt-ready.json", info)
        assert info["mamba_ssm_dtype"] == "float32"
        assert info["internal_states"][0]["memory_usage"]["token_capacity"] == 131072
        checks.sanity("sa16-replay-opt", prompts)
        h.preheat(prompts, "sa16-replay-opt")
        for repeat in (1, 2):
            checks.warm(prompts, 1)
            h.benchmark("sa16-replay-opt", repeat, 1)
        with (ROOT / "sa16-replay-opt-capture.log").open("w") as log:
            subprocess.run(
                [sys.executable, str(ROOT / "capture_profiles.py"), "sa16-replay-opt"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=900,
            )
        h.save(
            ROOT / "optimized-status.json",
            {"status": "complete", "measured_requests": 100},
        )
    finally:
        h.stop(pid)


if __name__ == "__main__":
    main()
