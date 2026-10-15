"""Sequential single-GPU Qwen3.5/3.6 MTP + SA acceptance runs.

Launches and shuts down its own servers; artifacts include exact commands, logs,
token sequences, diagnostics and mismatches. Performance is end-to-end, not TPOT.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def wait_ready(url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(5)
    raise TimeoutError(f"Server not ready within {timeout} seconds")


def find_metrics(value):
    if isinstance(value, dict):
        if "hybrid_spec" in value:
            return value["hybrid_spec"]
        for child in value.values():
            result = find_metrics(child)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = find_metrics(child)
            if result is not None:
                return result
    return None


def check_text_only_boundary(url):
    for payload in (
        {"text": "test", "image_data": "unsupported-test-image"},
        {"input_embeds": [[0.5]]},
    ):
        payload["sampling_params"] = {"temperature": 0, "max_new_tokens": 1}
        request = urllib.request.Request(
            url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as error:
            body = error.read().decode()
            if (
                error.code != 400
                or "hybrid-spec requires text or token IDs" not in body
            ):
                raise RuntimeError(f"Unexpected request rejection: {body}") from error
        else:
            raise RuntimeError("Unsupported input was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "native-eager",
            "hybrid-eager",
            "native-graph",
            "hybrid-graph",
            "target-eager",
            "native-k1-graph",
            "hybrid-k1-graph",
        ],
    )
    parser.add_argument("--eval-examples", type=int, default=0)
    args = parser.parse_args()
    from sglang.test.test_utils import terminate_and_kill_process_tree

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    benchmark = Path(__file__).with_name("compare_servers.py")
    prompts = args.prompts or Path(__file__).with_name("prompts.json")
    url = f"http://127.0.0.1:{args.port}"
    summary = {}
    for mode in args.modes:
        if mode not in {
            "native-eager",
            "hybrid-eager",
            "native-graph",
            "hybrid-graph",
            "target-eager",
            "native-k1-graph",
            "hybrid-k1-graph",
        }:
            parser.error(f"Unknown mode: {mode}")
    for mode in args.modes:
        k = 1 if "k1" in mode else 3
        hybrid = mode.startswith("hybrid")
        graph = mode.endswith("graph")
        env = os.environ.copy()
        env.pop("HYBRID_SPEC_CONFIG", None)
        if hybrid:
            config = out / f"{mode}-config.json"
            config.write_text(
                json.dumps(
                    dict(
                        num_drafts=k,
                        min_match_len=2,
                        max_requests=4,
                        max_history_tokens=8192,
                        backend="gpu_sa",
                    )
                )
            )
            env["HYBRID_SPEC_CONFIG"] = str(config)
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model_path,
            "--language-only",
            "--disable-overlap-schedule",
            "--chunked-prefill-size",
            "-1",
            "--context-length",
            "8192",
            "--max-running-requests",
            "4",
            "--mem-fraction-static",
            "0.8",
            "--random-seed",
            "42",
            "--watchdog-timeout",
            "1800",
            "--cuda-graph-backend-decode",
            "full" if graph else "disabled",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs-decode",
            "4",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ]
        if not mode.startswith("target"):
            command += [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                str(k),
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                str(k + 1),
            ]
        (out / f"{mode}-command.json").write_text(json.dumps(command, indent=2))
        print(f"START {mode}", flush=True)
        with (out / f"{mode}-server.log").open("w") as log:
            process = subprocess.Popen(
                command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                wait_ready(url, process, args.startup_timeout)
                print(f"READY {mode}", flush=True)
                summary[mode] = {}
                if hybrid:
                    check_text_only_boundary(url)
                for concurrency in (1, 4):
                    result_path = out / f"{mode}-c{concurrency}.json"
                    run = [
                        sys.executable,
                        str(benchmark),
                        "--record",
                        url,
                        "--prompts",
                        str(prompts),
                        "--output",
                        str(result_path),
                        "--concurrency",
                        str(concurrency),
                        "--max-tokens",
                        str(args.max_tokens),
                    ]
                    reference_mode = mode.replace("hybrid", "native")
                    if mode.startswith("target"):
                        reference_mode = "native-eager"
                    reference = out / f"{reference_mode}-c{concurrency}.json"
                    if reference_mode != mode and reference.exists():
                        run += ["--reference", str(reference)]
                    result = subprocess.run(run, check=False)
                    if not result_path.exists():
                        raise RuntimeError(
                            f"Benchmark failed before saving {result_path}"
                        )
                    report = json.loads(result_path.read_text())
                    record = report["record"]
                    before = find_metrics(record["server_info_before"])
                    after = find_metrics(record["server_info_after"])
                    delta = None
                    if before is not None and after is not None:
                        delta = {key: after[key] - before[key] for key in before}
                    summary[mode][str(concurrency)] = dict(
                        exit_code=result.returncode,
                        token_mismatch_indices=report["token_mismatch_indices"],
                        output_tokens_per_s=record["output_tokens_per_s"],
                        mean_latency_s=record["mean_latency_s"],
                        hybrid_spec_delta=delta,
                    )
                    if hybrid and (delta is None or not delta["retrieval_requests"]):
                        raise RuntimeError(
                            "SA usage was not observed; cannot validate hybrid"
                        )
                    (out / "summary.json").write_text(json.dumps(summary, indent=2))
                if args.eval_examples and mode in ("native-graph", "hybrid-graph"):
                    from types import SimpleNamespace
                    from sglang.test.sgl_eval_utils import run_sgl_eval

                    metrics = run_sgl_eval(
                        SimpleNamespace(
                            eval_name="gsm8k",
                            base_url=url,
                            model=args.model_path,
                            num_examples=args.eval_examples,
                            num_threads=4,
                            max_tokens=2048,
                            temperature=0,
                            seed=42,
                            sgl_eval_thinking=False,
                            sgl_eval_out_dir=str(out / mode),
                        )
                    )
                    summary[mode]["gsm8k"] = metrics
                    (out / "summary.json").write_text(json.dumps(summary, indent=2))
                print(f"DONE {mode}: {summary[mode]}", flush=True)
            finally:
                terminate_and_kill_process_tree(process)
    print(f"MATRIX_COMPLETE {out / 'summary.json'}", flush=True)
    if any(
        result.get("exit_code", 0) != 0
        for results in summary.values()
        for result in results.values()
    ):
        raise SystemExit("Token mismatches detected; inspect summary.json")


if __name__ == "__main__":
    main()
