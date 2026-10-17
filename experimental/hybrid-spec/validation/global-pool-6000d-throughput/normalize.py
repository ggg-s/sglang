"""Export the fixed-config experiment using the serving benchmark result schema."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    plan = json.loads((root / "plan.json").read_text())
    rows = json.loads((root / "results.json").read_text())
    gpu = next(csv.DictReader(plan["gpu"].splitlines(), skipinitialspace=True))
    with (root / "normalized.jsonl").open("w") as stream:
        for row in rows:
            raw = Path(row["official_result"]).name
            stem = raw.removesuffix(".jsonl")
            command = json.loads((root / f"{row['mode']}-command.json").read_text())
            benchmark = json.loads((root / f"{stem}-command.json").read_text())
            config = json.loads((root / f"{row['mode']}-config.json").read_text())
            record = {
                "framework": "sglang",
                "framework_version": "0.0.0.dev1+g1b0f58030 (installed metadata; source overridden)",
                "framework_commit": plan["source_baseline"],
                "candidate_id": f"{row['mode']}-r{row['round']}",
                "model": plan["model"],
                "status": row["status"],
                "failure_reason": row.get("failure_reason", ""),
                "hardware": {
                    "gpu_model": gpu["name"],
                    "gpu_count": 1,
                    "visible_devices": "0",
                    "uuid": gpu["uuid"],
                    "driver_version": gpu["driver_version"],
                    "cuda": plan["cuda"],
                    "torch": plan["torch"],
                    "nccl": "2.30.7",
                },
                "workload": {
                    "kind": "custom",
                    "scenario": row["scenario"],
                    "dataset_path": "unique.jsonl"
                    if row["scenario"] == "unique-cold"
                    else "repeat.jsonl",
                    "input_len_p50": row["input_len_p50"],
                    "input_len_p95": row["input_len_p95"],
                    "output_len": plan["max_tokens"],
                    "output_len_p50": row["output_len_p50"],
                    "output_len_p95": row["output_len_p95"],
                    "num_prompts": plan["num_prompts"],
                    "request_rate": "inf",
                    "max_concurrency": row["concurrency"],
                    "endpoint": "/generate",
                    "ignore_eos": plan["ignore_eos"],
                },
                "sla": None,
                "metrics": {
                    "request_throughput": row["request_throughput"],
                    "output_token_throughput": row["output_throughput"],
                    "mean_ttft_ms": row["mean_ttft_ms"],
                    "p99_ttft_ms": row["p99_ttft_ms"],
                    "mean_tpot_ms": row["mean_tpot_ms"],
                    "p99_tpot_ms": row["p99_tpot_ms"],
                    "mean_e2e_ms": row["mean_e2e_latency_ms"],
                    "p99_e2e_ms": row["p99_e2e_latency_ms"],
                    "success_rate": row["completed"] / plan["num_prompts"],
                    "sa_fraction": row["sa_fraction"],
                },
                "server_command": command,
                "benchmark_command": benchmark,
                "hybrid_config": config,
                "seed_cost": row["seed"],
                "artifacts": {
                    "server_log": f"{row['mode']}-server.log",
                    "raw_result": raw,
                    "server_help": "server-help.txt",
                    "benchmark_help": "benchmark-help.txt",
                    "diagnostics": f"{stem}-diagnostics.json",
                },
            }
            stream.write(json.dumps(record) + "\n")
    print(f"Exported {len(rows)} normalized rows; no SLA threshold was specified")


if __name__ == "__main__":
    main()
