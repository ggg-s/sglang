"""Run several fixed workloads on each explicitly configured test server."""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from run_model_matrix import check_text_only_boundary, find_metrics, wait_ready


def official_command(plan, server, workload, concurrency, result_path):
    # Keep all request generation, streaming and latency statistics upstream.
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        server["url"],
        "--model",
        plan["model"],
        "--tokenizer",
        plan["model"],
        "--dataset-name",
        "custom",
        "--dataset-path",
        workload["dataset"],
        "--num-prompts",
        str(workload["num_prompts"]),
        "--sharegpt-output-len",
        str(workload["max_tokens"]),
        "--sharegpt-context-len",
        str(workload["context_length"]),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        "inf",
        "--seed",
        str(plan.get("seed", 42)),
        "--temperature",
        "0",
        "--disable-ignore-eos",
        "--warmup-requests",
        str(concurrency),
        "--flush-cache",
        "--output-details",
        "--output-file",
        str(result_path),
    ]


def server_info(url):
    with urllib.request.urlopen(url.rstrip("/") + "/get_server_info", timeout=30) as r:
        return json.load(r)


def run_official(plan, server, workload, concurrency, out):
    stem = f"{server['name']}-{workload['name']}-c{concurrency}"
    result_path = out / f"{stem}.jsonl"
    # Official bench appends JSONL. Never silently merge a previous experiment.
    if result_path.exists():
        raise FileExistsError(
            f"Use a new output directory; already exists: {result_path}"
        )
    command = official_command(plan, server, workload, concurrency, result_path)
    (out / f"{stem}-command.json").write_text(json.dumps(command, indent=2))
    before = server_info(server["url"])
    hybrid = "hybrid_config" in server
    if hybrid != (find_metrics(before) is not None):
        raise RuntimeError("Server plugin diagnostics do not match the benchmark plan")
    print(f"BENCH {stem}: sglang.benchmark.serving (streaming)", flush=True)
    with (out / f"{stem}-bench.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    after = server_info(server["url"])
    (out / f"{stem}-diagnostics.json").write_text(
        json.dumps(
            dict(
                scope="Includes official warmup and measured requests",
                before=before,
                after=after,
            ),
            indent=2,
        )
    )
    lines = result_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"Expected one official benchmark result: {result_path}")
    result = json.loads(lines[0])
    if result["completed"] != workload["num_prompts"] or any(result["errors"]):
        raise RuntimeError(
            f"Benchmark dropped or failed requests; inspect {result_path} and bench log"
        )
    before_counts, after_counts = find_metrics(before), find_metrics(after)
    if hybrid and after_counts is None:
        raise RuntimeError("Hybrid diagnostics missing after official benchmark")
    delta = (
        {key: after_counts[key] - before_counts[key] for key in before_counts}
        if before_counts is not None and after_counts is not None
        else None
    )
    # Copy upstream statistics verbatim; do not reconstruct TPOT from wall time.
    row = {
        key: result[key]
        for key in (
            "completed",
            "total_output_tokens",
            "output_throughput",
            "mean_e2e_latency_ms",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "median_tpot_ms",
            "p95_tpot_ms",
            "p99_tpot_ms",
            "accept_length",
        )
    }
    row.update(
        server=server["name"],
        workload=workload["name"],
        concurrency=concurrency,
        benchmark="sglang.benchmark.serving",
        max_tokens=workload["max_tokens"],
        hybrid_spec_delta=delta,
        hybrid_spec_delta_scope="official_warmup_and_measurement",
        official_result=str(result_path),
    )
    if delta is not None:
        retrieval, neural = delta["retrieval_requests"], delta["neural_requests"]
        row.update(
            sa_fraction=retrieval / (retrieval + neural)
            if retrieval + neural
            else None,
            sa_mean_accept_length=(
                delta["retrieval_accept_tokens"] / retrieval if retrieval else None
            ),
            neural_mean_accept_length=(
                delta["neural_accept_tokens"] / neural if neural else None
            ),
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from sglang.test.test_utils import terminate_and_kill_process_tree

    plan = json.loads(args.plan.read_text())
    official = plan.get("benchmark") == "sglang"
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    benchmark = Path(__file__).with_name("compare_servers.py")
    root = Path(__file__).resolve().parents[3]
    summary = []
    for server in plan["servers"]:
        name = server["name"]
        if not name.replace("-", "").isalnum():
            raise ValueError("Invalid server name")
        env = os.environ.copy()
        env.pop("HYBRID_SPEC_CONFIG", None)
        hybrid = "hybrid_config" in server
        if hybrid:
            config = out / f"{name}-config.json"
            config.write_text(json.dumps(server["hybrid_config"], indent=2))
            env["HYBRID_SPEC_CONFIG"] = str(config)
            # An inherited whitelist must not silently disable the experiment.
            if "SGLANG_PLUGINS" in env:
                plugins = env["SGLANG_PLUGINS"].split(",")
                if "hybrid_spec" not in plugins:
                    plugins.append("hybrid_spec")
                env["SGLANG_PLUGINS"] = ",".join(plugins)
        print(f"START {name}", flush=True)
        with (out / f"{name}-server.log").open("w") as log:
            process = subprocess.Popen(
                server["command"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                url = server["url"]
                wait_ready(url, process, plan.get("startup_timeout", 1800))
                print(f"READY {name}", flush=True)
                if hybrid and not official:
                    check_text_only_boundary(url)
                for workload in plan["workloads"]:
                    label = workload["name"]
                    if not label.replace("-", "").isalnum():
                        raise ValueError("Invalid workload name")
                    for concurrency in plan.get("concurrencies", [1, 4]):
                        if official:
                            row = run_official(plan, server, workload, concurrency, out)
                            summary.append(row)
                            (out / "summary.json").write_text(
                                json.dumps(summary, indent=2)
                            )
                            print(f"DONE {row}", flush=True)
                            continue
                        result_path = out / f"{name}-{label}-c{concurrency}.json"
                        run = [
                            sys.executable,
                            str(benchmark),
                            "--record",
                            url,
                            "--prompts",
                            workload["prompts"],
                            "--output",
                            str(result_path),
                            "--max-tokens",
                            str(workload["max_tokens"]),
                            "--concurrency",
                            str(concurrency),
                            "--warmup-all",
                            "--flush-cache-before-run",
                        ]
                        subprocess.run(run, check=True)
                        report = json.loads(result_path.read_text())
                        record = report["record"]
                        before = find_metrics(record["server_info_before"])
                        after = find_metrics(record["server_info_after"])
                        if hybrid and (before is None or after is None):
                            raise RuntimeError(
                                "Hybrid diagnostics missing: the plugin was not "
                                "loaded or the serving code is outdated"
                            )
                        delta = (
                            {k: after[k] - before[k] for k in before}
                            if before is not None and after is not None
                            else None
                        )
                        row = dict(
                            server=name,
                            workload=label,
                            concurrency=concurrency,
                            max_tokens=workload["max_tokens"],
                            output_tokens_per_s=record["output_tokens_per_s"],
                            mean_latency_s=record["mean_latency_s"],
                            hybrid_spec_delta=delta,
                            mean_accept_length=sum(
                                r["meta_info"]["completion_tokens"]
                                for r in record["requests"]
                            )
                            / sum(
                                r["meta_info"]["spec_verify_ct"]
                                for r in record["requests"]
                            ),
                        )
                        if delta is not None:
                            retrieval = delta["retrieval_requests"]
                            neural = delta["neural_requests"]
                            row.update(
                                sa_fraction=(
                                    retrieval / (retrieval + neural)
                                    if retrieval + neural
                                    else None
                                ),
                                sa_mean_accept_length=(
                                    delta["retrieval_accept_tokens"] / retrieval
                                    if retrieval
                                    else None
                                ),
                                neural_mean_accept_length=(
                                    delta["neural_accept_tokens"] / neural
                                    if neural
                                    else None
                                ),
                            )
                        summary.append(row)
                        (out / "summary.json").write_text(json.dumps(summary, indent=2))
                        print(f"DONE {row}", flush=True)
            finally:
                terminate_and_kill_process_tree(process)
    print("WORKLOAD_PLAN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
