"""Run several fixed workloads on each explicitly configured test server."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_model_matrix import check_text_only_boundary, find_metrics, wait_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from sglang.test.test_utils import terminate_and_kill_process_tree

    plan = json.loads(args.plan.read_text())
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
                if hybrid:
                    check_text_only_boundary(url)
                for workload in plan["workloads"]:
                    label = workload["name"]
                    if not label.replace("-", "").isalnum():
                        raise ValueError("Invalid workload name")
                    for concurrency in plan.get("concurrencies", [1, 4]):
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
                        summary.append(row)
                        (out / "summary.json").write_text(json.dumps(summary, indent=2))
                        print(f"DONE {row}", flush=True)
            finally:
                terminate_and_kill_process_tree(process)
    print("WORKLOAD_PLAN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
