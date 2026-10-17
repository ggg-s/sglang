"""Same-day serial/parallel full/split comparison on the reserved single GPU."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((args.serial / "source-hashes.json").read_text())
    source_changes = []
    for variant, source in (("serial", args.serial), ("parallel", args.parallel)):
        hashes = {}
        for path, digest in baseline.items():
            actual = hashlib.sha256((source / path).read_bytes()).hexdigest()
            hashes[path] = actual
            if actual != digest:
                assert variant == "parallel" and path.endswith(
                    "/gpu_sa/suffix_automaton.cu"
                ), path
                source_changes.append(path)
        (root / f"{variant}-source-hashes.json").write_text(
            json.dumps(hashes, indent=2) + "\n"
        )
    assert source_changes == [
        "experimental/hybrid-spec/src/hybrid_spec/backends/gpu_sa/suffix_automaton.cu"
    ]
    micro_serial = json.loads((root / "micro-serial/results.json").read_text())
    micro_parallel = json.loads((root / "micro-parallel/results.json").read_text())
    assert len(micro_serial) == len(micro_parallel) == 12
    for serial, parallel in zip(micro_serial, micro_parallel):
        for key in ("sources", "batch", "workload", "inputs_sha256", "tokens", "mask"):
            assert serial[key] == parallel[key], (key, serial, parallel)
    assert (root / "tests.exit").read_text().strip() == "0"
    order = [
        ("serial", "full"),
        ("parallel", "full"),
        ("parallel", "split"),
        ("serial", "split"),
    ]
    plan = {
        "order": order,
        "scenarios": ["unique-cold", "repeat-warm"],
        "concurrencies": [1, 4],
        "rounds": 3,
        "requests_per_case": 80,
        "output_tokens_per_request": 128,
        "source_changes": source_changes,
        "micro_output_parity": "12/12 exact tokens, masks and input hashes",
        "serial_root": str(args.serial),
        "parallel_root": str(args.parallel),
        "control": "Only production source difference is shared_query_kernel and its launch block size; same model, data, config and benchmark client",
    }
    (root / "comparison-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for variant, graph in order:
        source = args.serial if variant == "serial" else args.parallel
        label = f"{variant}-{graph}"
        output = root / label
        env = os.environ.copy()
        env.update(
            PYTHONPATH=f"{source}/experimental/hybrid-spec/src:{source}/python",
            TORCH_EXTENSIONS_DIR=str(
                root / "serial-extensions"
                if variant == "serial"
                else args.parallel / "torch-extensions"
            ),
            TORCH_CUDA_ARCH_LIST="12.0",
            MAX_JOBS="2",
            SGLANG_IS_IN_CI="0",
        )
        command = [
            sys.executable,
            str(args.driver),
            "--workdir",
            str(source),
            "--output-dir",
            str(output),
            "--model",
            args.model,
            "--dataset",
            args.dataset,
            "--modes",
            f"{graph}-global",
            "--scenarios",
            "unique-cold",
            "repeat-warm",
        ]
        (root / f"{label}-driver-command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        print(f"START {label}", flush=True)
        with (root / f"{label}-driver.log").open("w") as log:
            result = subprocess.run(
                command,
                env=env,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        (root / f"{label}.exit").write_text(str(result.returncode) + "\n")
        assert result.returncode == 0, label
        rows = json.loads((output / "results.json").read_text())
        assert len(rows) == 12 and all(r["status"] == "ok" for r in rows), label
        print(f"COMPLETE {label}", flush=True)
    print("COMPLETE 48 cases", flush=True)


if __name__ == "__main__":
    main()
