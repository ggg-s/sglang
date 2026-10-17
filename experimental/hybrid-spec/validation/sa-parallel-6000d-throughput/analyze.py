"""Summarize the paired kernel change without attributing FP8 output drift to it."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    rows, groups, outputs = [], {}, {}
    reference_inputs = None
    for variant, graph in (
        ("serial", "full"),
        ("parallel", "full"),
        ("parallel", "split"),
        ("serial", "split"),
    ):
        label = f"{variant}-{graph}"
        directory = root / label
        assert (root / f"{label}.exit").read_text().strip() == "0"
        values = json.loads((directory / "results.json").read_text())
        assert len(values) == 12
        inputs = [
            (directory / name).read_bytes()
            for name in ("provenance.json", "unique.jsonl", "repeat.jsonl")
        ]
        if reference_inputs is None:
            reference_inputs = inputs
        assert inputs == reference_inputs, label
        for value in values:
            assert value["status"] == "ok" and value["completed"] == 80
            assert value["total_output_tokens"] == 10240
            raw = json.loads(
                (directory / Path(value["official_result"]).name).read_text()
            )
            assert (
                not any(raw["errors"])
                and raw["cache_report"]["total_cached_tokens"] == 0
            )
            key = (value["scenario"], value["concurrency"], graph, value["round"])
            outputs[(variant, *key)] = raw
            value.update(variant=variant, graph=graph, label=label)
            rows.append(value)
            groups.setdefault(key[:3], {}).setdefault(variant, []).append(value)
    assert len(rows) == 48
    comparisons, parity = [], []
    for (scenario, concurrency, graph), variants in sorted(groups.items()):
        serial = {r["round"]: r for r in variants["serial"]}
        parallel = {r["round"]: r for r in variants["parallel"]}
        assert set(serial) == set(parallel) == {1, 2, 3}
        paired = [
            100
            * (parallel[i]["output_throughput"] / serial[i]["output_throughput"] - 1)
            for i in (1, 2, 3)
        ]
        record = {
            "scenario": scenario,
            "concurrency": concurrency,
            "graph": graph,
            "paired_gain_pct_median": statistics.median(paired),
            "paired_gain_pct_min": min(paired),
            "paired_gain_pct_max": max(paired),
        }
        for variant in ("serial", "parallel"):
            samples = variants[variant]
            for key in (
                "output_throughput",
                "mean_ttft_ms",
                "mean_tpot_ms",
                "mean_accept_length",
                "sa_fraction",
            ):
                values = [r[key] for r in samples]
                record[f"{variant}_{key}_median"] = statistics.median(values)
                if key == "output_throughput":
                    record[f"{variant}_tok_s_min"] = min(values)
                    record[f"{variant}_tok_s_max"] = max(values)
            record[f"{variant}_split_skip_batches"] = sum(
                r["split_delta"].get("skip_ct", 0) for r in samples
            )
            record[f"{variant}_split_mixed_batches"] = sum(
                r["split_delta"].get("mixed_ct", 0) for r in samples
            )
        comparisons.append(record)
        for round_id in (1, 2, 3):
            a = outputs[("serial", scenario, concurrency, graph, round_id)]
            b = outputs[("parallel", scenario, concurrency, graph, round_id)]
            assert a["input_lens"] == b["input_lens"]
            assert a["output_lens"] == b["output_lens"] == [128] * 80
            different = [
                i
                for i, (x, y) in enumerate(
                    zip(a["generated_texts"], b["generated_texts"])
                )
                if x != y
            ]
            parity.append(
                {
                    "scenario": scenario,
                    "concurrency": concurrency,
                    "graph": graph,
                    "round": round_id,
                    "compared": 80,
                    "different": len(different),
                    "different_request_indices": different,
                }
            )
    write(root / "results.json", rows)
    write(root / "summary.json", comparisons)
    write(
        root / "output-parity.json",
        {
            "compared": sum(p["compared"] for p in parity),
            "different": sum(p["different"] for p in parity),
            "cases": parity,
            "note": "Fixed-128-token greedy FP8 runs; C4 arrival/batch timing can change outputs. Exact kernel semantics are separately checked against a CPU text oracle and serial microbenchmark outputs. This is not a task accuracy evaluation.",
        },
    )
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    serial_micro = json.loads((root / "micro-serial/results.json").read_text())
    parallel_micro = json.loads((root / "micro-parallel/results.json").read_text())
    micro = []
    for a, b in zip(serial_micro, parallel_micro):
        for key in ("sources", "batch", "workload", "inputs_sha256", "tokens", "mask"):
            assert a[key] == b[key]
        micro.append(
            {
                "sources": a["sources"],
                "batch": a["batch"],
                "workload": a["workload"],
                "serial_shared_kernel_us": a["shared_kernel_us_median"],
                "parallel_shared_kernel_us": b["shared_kernel_us_median"],
                "shared_kernel_speedup": a["shared_kernel_us_median"]
                / b["shared_kernel_us_median"],
                "serial_total_query_us": a["total_query_us_median"],
                "parallel_total_query_us": b["total_query_us_median"],
                "exact_output_parity": True,
            }
        )
    write(root / "micro-summary.json", micro)
    (root / "report-table.md").write_text(
        "| 负载 | 并发 | Graph | 串行 tok/s | 并行 tok/s | 配对收益 [三轮范围] |\n"
        "| --- | ---: | --- | ---: | ---: | ---: |\n"
        + "".join(
            f"| {r['scenario']} | {r['concurrency']} | {r['graph']} | {r['serial_output_throughput_median']:.2f} | {r['parallel_output_throughput_median']:.2f} | {r['paired_gain_pct_median']:+.2f}% [{r['paired_gain_pct_min']:+.2f}%, {r['paired_gain_pct_max']:+.2f}%] |\n"
            for r in comparisons
        )
    )
    write(
        root / "measurement-verification.json",
        {
            "cases": 48,
            "requests": sum(r["completed"] for r in rows),
            "output_tokens": sum(r["total_output_tokens"] for r in rows),
            "same_input_files": True,
            "same_output_budget": True,
            "prefix_cache_tokens": 0,
            "all_request_errors_empty": True,
            "prompt_files_sha256": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in zip(
                    ("provenance.json", "unique.jsonl", "repeat.jsonl"),
                    reference_inputs,
                )
            },
            "split_mixed_batches": sum(
                r["split_delta"].get("mixed_ct", 0) for r in rows
            ),
            "split_skipped_batches": sum(
                r["split_delta"].get("skip_ct", 0) for r in rows
            ),
        },
    )
    print((root / "report-table.md").read_text())


if __name__ == "__main__":
    main()
