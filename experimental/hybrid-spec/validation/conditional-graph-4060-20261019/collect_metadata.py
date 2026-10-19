"""Run beside the remote repo and raw validation outputs after testing."""

import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import torch
import triton

root = Path(__file__).resolve().parent
expected = json.loads((root / "expected-source-hashes.json").read_text())
actual = {
    path: hashlib.sha256((root / "repo" / path).read_bytes()).hexdigest()
    for path in expected["sha256"]
}
assert actual == expected["sha256"], "remote source differs from tested commit"


def junit(name):
    suite = next(ET.parse(root / f"{name}.xml").getroot().iter("testsuite"))
    result = {
        key: int(suite.attrib.get(key, 0))
        for key in ("tests", "failures", "errors", "skipped")
    }
    result["passed"] = (
        result["tests"] - result["failures"] - result["errors"] - result["skipped"]
    )
    result["seconds"] = float(suite.attrib["time"])
    return result


metadata = {
    "recorded_at_utc": datetime.now(UTC).isoformat(),
    "server": "ai-workstation",
    "remote_directory": str(root),
    "source_commit": expected["commit"],
    "archive_sha256": expected["archive_sha256"],
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "triton": triton.__version__,
    "gpu": torch.cuda.get_device_name(),
    "gpu_driver": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).strip(),
    "conditional_bindings": hasattr(torch.cuda.CUDAGraph, "begin_capture_to_if_node"),
    "results": {
        name: junit(name)
        for name in (
            "legacy-torch28",
            "focused",
            "cuda-regression-v2",
            "component-regression-v2",
        )
    },
    "initial_attempts": {
        name: junit(name) for name in ("cuda-regression", "component-regression")
    },
    "source_sha256": actual,
    "helper_sha256": hashlib.sha256(
        (root / "lean_cuda_plugin.py").read_bytes()
    ).hexdigest(),
    "probe_sha256": hashlib.sha256(
        (root / "dispatch_probe.py").read_bytes()
    ).hexdigest(),
    "gc_helper_sha256": hashlib.sha256(
        (root / "capture_gc_plugin.py").read_bytes()
    ).hexdigest(),
    "post_test_gpu": (root / "post-test-gpu.txt").read_text().strip(),
    "scope": "Real single-GPU conditional capture/replay; model and child attention "
    "are synthetic. Native topk/KV metadata/row compaction are retained. The "
    "separate dispatch probe uses real FP16 GEMM and gather/scatter. No TP2 NCCL, "
    "Arctic, full-weight serving, model accuracy or serving throughput validation.",
}
(root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata["results"], indent=2))
