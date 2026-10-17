"""Keep the original three-mode run intact and combine the extra split control."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    control = root / "split-off-control"
    original = json.loads((root / "results.json").read_text())
    extra = json.loads((control / "results.json").read_text())
    assert len(original) == 54 and len(extra) == 18
    assert (root / "driver.exit").read_text().strip() == "0"
    assert (root / "control.exit").read_text().strip() == "0"
    rows = original + extra
    assert all(r["status"] == "ok" and r["completed"] == 80 for r in rows)
    assert all(r["total_output_tokens"] == 10240 for r in rows)
    assert (
        len({(r["mode"], r["scenario"], r["concurrency"], r["round"]) for r in rows})
        == 72
    )
    for name in ("provenance.json", "unique.jsonl", "repeat.jsonl"):
        assert (root / name).read_bytes() == (control / name).read_bytes(), name
    combined = root / "combined"
    combined.mkdir(exist_ok=False)
    for directory in (root, control):
        for path in directory.iterdir():
            if path.is_file() and not (combined / path.name).exists():
                shutil.copy2(path, combined / path.name)
    shutil.copy2(root / "plan.json", combined / "main-plan.json")
    shutil.copy2(control / "plan.json", combined / "control-plan.json")
    (combined / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (combined / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    plan = json.loads((root / "plan.json").read_text())
    plan["additional_control"] = {
        "mode": "split-off",
        "reason": "Separate the effect of shared history from skipping MTP tails",
        "completed_after_main_matrix": True,
        "rounds": 3,
        "same_dataset_files_verified": True,
    }
    if (root / "interruption.json").exists():
        plan["interruption"] = json.loads((root / "interruption.json").read_text())
    (combined / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    subprocess.run(
        [sys.executable, str(combined / "analyze.py"), str(combined)], check=True
    )
    subprocess.run(
        [sys.executable, str(combined / "normalize.py"), str(combined)], check=True
    )
    environment = json.loads((root / "environment.json").read_text())
    runtime = Path(environment["runtime_root"])
    source = json.loads((root / "source-hashes.json").read_text())
    mismatches = [
        p
        for p, digest in source.items()
        if hashlib.sha256((runtime / p).read_bytes()).hexdigest() != digest
    ]
    assert not mismatches
    memory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    original_status = subprocess.check_output(
        [
            "git",
            "-C",
            "/root/rivermind-data/sglang",
            "status",
            "--short",
        ],
        text=True,
    )
    verification = {
        "measured_cases": len(rows),
        "measured_requests": sum(r["completed"] for r in rows),
        "measured_output_tokens": sum(r["total_output_tokens"] for r in rows),
        "all_measurements_ok": True,
        "source_match": {"files": len(source), "mismatches": mismatches},
        "final_gpu_memory_mib": memory,
        "original_remote_checkout_status": original_status,
        "main_driver_sha256": hashlib.sha256(
            (root / "run_benchmark.py").read_bytes()
        ).hexdigest(),
        "control_driver_sha256": hashlib.sha256(
            (root / "run_control.py").read_bytes()
        ).hexdigest(),
    }
    (combined / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    archive = root.parent / (root.name + "-evidence.tar.gz")
    subprocess.run(
        [
            "tar",
            "--exclude=__pycache__",
            "-czf",
            str(archive),
            "-C",
            str(root.parent),
            root.name,
        ],
        check=True,
    )
    finalized = {
        "archive": str(archive),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "archive_bytes": archive.stat().st_size,
        "verification": verification,
    }
    (root / "finalized.json").write_text(json.dumps(finalized, indent=2) + "\n")
    print(json.dumps(finalized, indent=2), flush=True)


if __name__ == "__main__":
    main()
