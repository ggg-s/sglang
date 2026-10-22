"""Capture matched FP32 snapshot/Replay profiles before changing kernels."""

import subprocess
import sys
from pathlib import Path

import bench_helpers as h
import checks

ROOT = Path(__file__).resolve().parent


def main():
    prompts = h.prepare()
    pid = None
    try:
        for mode in ("sa16-snapshot", "sa16-replay"):
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
            h.event("launch", mode=mode)
            h.wait_ready(pid)
            info = h.request("/get_server_info")
            h.save(ROOT / (mode + "-ready.json"), info)
            assert info["mamba_ssm_dtype"] == "float32"
            assert (
                info["internal_states"][0]["memory_usage"]["token_capacity"] == 131072
            )
            checks.sanity(mode, prompts)
            h.preheat(prompts, mode)
            with (ROOT / (mode + "-capture.log")).open("w") as log:
                subprocess.run(
                    [sys.executable, str(ROOT / "capture_profiles.py"), mode],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=900,
                )
            h.event("profile_complete", mode=mode)
            h.stop(pid)
            pid = None
        h.save(ROOT / "profile-status.json", {"status": "complete"})
    finally:
        h.stop(pid)


if __name__ == "__main__":
    main()
