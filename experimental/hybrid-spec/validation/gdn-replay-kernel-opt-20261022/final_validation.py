"""Run final-source correctness and compiler-resource checks after all timing."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(argv, label, source, cwd=None):
    env = dict(
        os.environ,
        PYTHONPATH=str(source / "python")
        + ":"
        + str(source / "experimental/hybrid-spec/src")
        + ":"
        + str(ROOT),
    )
    (ROOT / (label + "-command.json")).write_text(json.dumps(argv, indent=2) + "\n")
    with (ROOT / (label + ".log")).open("w") as log:
        value = subprocess.run(
            argv,
            cwd=cwd or source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    (ROOT / (label + ".exit")).write_text(str(value.returncode) + "\n")
    assert value.returncode == 0, label


def main():
    deadline = time.monotonic() + 2400
    while not (ROOT / "matrix8/matrix.exit").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Formal comparisons did not finish")
        time.sleep(2)
    assert (ROOT / "matrix8/matrix.exit").read_text().strip() == "0"
    source = ROOT / "optimized/source"
    expected = json.loads((ROOT / "final-production-sha256.json").read_text())
    for name, sha in expected.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == sha
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "test_gdn_replayssm_final.py"),
            "-q",
            "--tb=short",
            "--maxfail=2",
            "--junitxml=" + str(ROOT / "final-registered.xml"),
        ],
        "final-registered",
        source,
    )
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "experimental/hybrid-spec/tests/test_gdn_compact.py",
            "-q",
            "--tb=short",
            "--maxfail=2",
            "--junitxml=" + str(ROOT / "final-components.xml"),
        ],
        "final-components",
        source,
    )
    for label, src in [("baseline", ROOT / "source"), ("optimized", source)]:
        folder = ROOT / (label + "-resources")
        folder.mkdir(exist_ok=True)
        run(
            [sys.executable, str(ROOT / "probe_resources.py")],
            label + "-resources",
            src,
            cwd=folder,
        )
    for port in (39923, 39924):
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    )
    assert "0 MiB" in gpu
    (ROOT / "final-validation.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "gpu_after": gpu.strip(),
                "ports_closed": [39923, 39924],
                "production_hashes_match": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
