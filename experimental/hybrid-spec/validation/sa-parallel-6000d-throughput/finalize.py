"""Verify completed evidence and preserve the exact parallel runtime sources."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    assert (root / "comparison.exit").read_text().strip() == "0"
    assert (root / "analysis.exit").read_text().strip() == "0"
    plan = json.loads((root / "comparison-plan.json").read_text())
    source_checks = {}
    for variant in ("serial", "parallel"):
        source = Path(plan[f"{variant}_root"])
        hashes = json.loads((root / f"{variant}-source-hashes.json").read_text())
        mismatches = [
            p
            for p, digest in hashes.items()
            if hashlib.sha256((source / p).read_bytes()).hexdigest() != digest
        ]
        assert not mismatches, mismatches
        source_checks[variant] = {"files": len(hashes), "mismatches": mismatches}
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    assert memory == "0", memory
    checkout = subprocess.check_output(
        ["git", "-C", "/root/rivermind-data/sglang", "status", "--short"], text=True
    )
    assert not checkout, checkout
    environment = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv",
        ],
        text=True,
    )
    verification = json.loads((root / "measurement-verification.json").read_text())
    verification.update(
        source_checks=source_checks,
        final_gpu_memory_mib=memory,
        original_remote_checkout_status=checkout,
        gpu=environment,
        tests=(root / "tests.log").read_text().splitlines()[-3:],
    )
    (root / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    for variant in ("serial", "parallel"):
        source = Path(plan[f"{variant}_root"])
        shutil.copy2(
            source
            / "experimental/hybrid-spec/src/hybrid_spec/backends/gpu_sa/suffix_automaton.cu",
            root / f"{variant}-suffix_automaton.cu",
        )
        subprocess.run(
            [
                "tar",
                "--exclude=__pycache__",
                "--exclude=.pytest_cache",
                "-czf",
                str(root / f"{variant}-source.tar.gz"),
                "-C",
                str(source),
                "python",
                "experimental/hybrid-spec/src",
                "experimental/hybrid-spec/tests",
            ],
            check=True,
        )
    evidence = []
    for path in sorted(root.rglob("*")):
        if (
            path.is_file()
            and "serial-extensions" not in path.parts
            and path.name != "evidence-hashes.json"
        ):
            evidence.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    (root / "evidence-hashes.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
