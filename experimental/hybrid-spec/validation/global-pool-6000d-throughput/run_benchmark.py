"""Controlled single-GPU full/off, full/global and split/global throughput matrix.

Uses the repository's unchanged official streaming benchmark client in process.
Pre-seeding always generates real requests; dataset reference solutions are unused.
"""

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import requests


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-prompts", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--modes", nargs="+", default=["full-off", "full-global", "split-global"]
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=["unique-cold", "repeat-cold", "repeat-warm"],
        default=["unique-cold", "repeat-cold", "repeat-warm"],
    )
    opts = parser.parse_args()
    out = opts.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "results.json").exists(), "Use a new result directory"
    assert opts.num_prompts >= 80 and opts.num_prompts % 16 == 0
    import torch
    from sglang.benchmark import serving
    from transformers import AutoTokenizer

    os.environ["SGLANG_IS_IN_CI"] = "0"
    os.environ.pop("HYBRID_SPEC_CONFIG", None)
    port = 39982
    url = f"http://127.0.0.1:{port}"
    probe = socket.socket()
    assert probe.connect_ex(("127.0.0.1", port)) != 0, "Benchmark port occupied"
    probe.close()

    def call(method, path, **kwargs):
        r = requests.request(method, url + path, timeout=600, **kwargs)
        r.raise_for_status()
        return (
            r.json()
            if "application/json" in r.headers.get("Content-Type", "")
            else r.text
        )

    def info():
        return call("GET", "/server_info")["internal_states"][0]

    def telemetry():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.used,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        ).strip()

    raw = Path(opts.dataset).read_bytes()
    source = json.loads(raw)
    tokenizer = AutoTokenizer.from_pretrained(opts.model, local_files_only=True)
    prompts, provenance, seen = [], [], set()
    for i, row in enumerate(source):
        # Never read the reference answer from the dataset.
        code, edit = row["code"], row["edit"]
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a coding assistant."},
                {
                    "role": "user",
                    "content": f"Edit request:\n{edit}\n\nOriginal code:\n```\n{code}\n```\n\nReturn the complete updated source code. Preserve unaffected code. Return code only, without an explanation.",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        length = len(tokenizer.encode(prompt))
        code_len = len(tokenizer.encode(code, add_special_tokens=False))
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if not 192 <= code_len or length > 3584 or digest in seen:
            continue
        prompts.append(prompt)
        seen.add(digest)
        provenance.append(
            {
                "row_index": i,
                "prompt_tokens": length,
                "source_tokens": code_len,
                "prompt_sha256": digest,
            }
        )
        if len(prompts) == opts.num_prompts:
            break
    assert len(prompts) == opts.num_prompts
    datasets = {"unique": prompts, "repeat": prompts[:16] * (opts.num_prompts // 16)}
    for name, values in datasets.items():
        with (out / f"{name}.jsonl").open("w") as f:
            for prompt in values:
                f.write(
                    json.dumps(
                        {
                            "conversations": [
                                {"from": "human", "value": prompt},
                                {"from": "gpt", "value": ""},
                            ]
                        }
                    )
                    + "\n"
                )
    write(
        out / "provenance.json",
        {
            "dataset": "glaiveai/code_edits_sample",
            "revision": "dfdacc8eb57d576d0662645144bdba4cdba6a106",
            "source_file_sha256": hashlib.sha256(raw).hexdigest(),
            "selection": "First 80 distinct prompts in source order with source >=192 tokens and prompt <=3584; no truncation; code+edit only; chat template enable_thinking=False",
            "reference_solution_in_prompt": False,
            "prompts": provenance,
        },
    )

    common = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        opts.model,
        "--language-only",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "8",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "9",
        "--speculative-moe-runner-backend",
        "triton",
        "--linear-attn-backend",
        "triton",
        "--speculative-draft-attention-backend",
        "triton",
        "--tp-size",
        "1",
        "--context-length",
        "4096",
        "--max-running-requests",
        "4",
        "--max-mamba-cache-size",
        "20",
        "--mem-fraction-static",
        "0.8",
        "--random-seed",
        "42",
        "--watchdog-timeout",
        "300",
        "--cuda-graph-max-bs-decode",
        "4",
        "--disable-radix-cache",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    plan = {
        "model": opts.model,
        "source_baseline": "2d53a3d2d2+uncommitted-global-pool",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": telemetry(),
        "endpoint": "/generate streaming",
        "concurrencies": [1, 4],
        "rounds": opts.rounds,
        "num_prompts": opts.num_prompts,
        "max_tokens": opts.max_tokens,
        "ignore_eos": True,
        "prefix_cache": False,
        "neural_gate": 2,
        "num_drafts": 8,
        "min_match_len": 4,
        "shared_context_tokens": 128,
        "completed_slots": 16,
        "context_length": 4096,
        "hist_tokens": 4096,
        "server_command": common,
        "scenarios": {
            "unique-cold": "80 distinct real code-edit prompts; flush before each run",
            "repeat-cold": "16 real prompts repeated 5 times; flush before each run",
            "repeat-warm": "16 real prompts repeated 5 times; flush then generate 16 seed requests; seed cost recorded separately",
        },
        "sla": None,
        "client": "Unmodified sglang.benchmark.serving.cli_main; --warmup-requests 0, warmup controlled externally",
        "order": f"Mode order {opts.modes}; each has {opts.rounds} rounds, shuffled scenario/concurrency cases with seed=20261017+round",
        "selected_scenarios": opts.scenarios,
    }
    write(out / "plan.json", plan)
    rows = []

    def generate(prompt):
        return call(
            "POST",
            "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": opts.max_tokens,
                    "ignore_eos": True,
                },
            },
        )

    def batch(values, concurrency):
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(concurrency) as executor:
            responses = list(executor.map(generate, values))
        return {
            "wall_s": time.perf_counter() - start,
            "completion_tokens": sum(
                r["meta_info"]["completion_tokens"] for r in responses
            ),
            "response_text_sha256": [
                hashlib.sha256(r["text"].encode()).hexdigest() for r in responses
            ],
        }

    for mode in opts.modes:
        if mode not in ("full-off", "full-global", "split-global", "split-off"):
            raise ValueError(mode)
        config = {
            "num_drafts": 8,
            "min_match_len": 4,
            "max_requests": 4,
            "max_history_tokens": 4096,
            "backend": "gpu_sa",
            "history_mode": "device",
            "partial_drafts": "fallback",
            "num_neural_gate_tokens": 2,
            "draft_graph_mode": "split" if mode.startswith("split-") else "full",
            "enable_global_pool": mode.endswith("-global"),
            "max_shared_requests": 16 if mode.endswith("-global") else 0,
            "shared_context_tokens": 128,
        }
        cfgpath = out / f"{mode}-config.json"
        write(cfgpath, config)
        write(out / f"{mode}-command.json", common)
        env = os.environ.copy()
        env["HYBRID_SPEC_CONFIG"] = str(cfgpath)
        print(f"START {mode}", flush=True)
        with (out / f"{mode}-server.log").open("w") as server_log:
            process = subprocess.Popen(
                common,
                cwd=opts.workdir,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 900
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{mode} server exited {process.returncode}")
                    try:
                        if requests.get(url + "/health", timeout=2).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{mode} server startup timed out")
                    time.sleep(2)
                call("POST", "/flush_cache")
                # Warm both requested concurrency levels before any measurement.
                warmup = [batch(prompts[:8], 1), batch(prompts[:16], 4)]
                write(out / f"{mode}-warmup.json", warmup)
                call("POST", "/flush_cache")
                print(f"READY {mode}", flush=True)
                for round_id in range(1, opts.rounds + 1):
                    import random

                    cases = [
                        (scenario, c) for scenario in opts.scenarios for c in (1, 4)
                    ]
                    random.Random(20261017 + round_id).shuffle(cases)
                    for scenario, concurrency in cases:
                        stem = f"{mode}-{scenario}-c{concurrency}-r{round_id}"
                        result_path = out / f"{stem}.jsonl"
                        assert not result_path.exists()
                        call("POST", "/flush_cache")
                        seed_result = (
                            batch(prompts[:16], concurrency)
                            if scenario == "repeat-warm"
                            else None
                        )
                        if seed_result:
                            write(out / f"{stem}-seed.json", seed_result)
                        before = info()
                        telemetry_before = telemetry()
                        command = [
                            sys.executable,
                            "-m",
                            "sglang.benchmark.serving",
                            "--backend",
                            "sglang",
                            "--base-url",
                            url,
                            "--model",
                            opts.model,
                            "--tokenizer",
                            opts.model,
                            "--dataset-name",
                            "custom",
                            "--dataset-path",
                            str(
                                out
                                / (
                                    "unique.jsonl"
                                    if scenario == "unique-cold"
                                    else "repeat.jsonl"
                                )
                            ),
                            "--num-prompts",
                            str(opts.num_prompts),
                            "--sharegpt-output-len",
                            str(opts.max_tokens),
                            "--sharegpt-context-len",
                            "4096",
                            "--max-concurrency",
                            str(concurrency),
                            "--request-rate",
                            "inf",
                            "--seed",
                            str(42 + round_id),
                            "--temperature",
                            "0",
                            "--warmup-requests",
                            "0",
                            "--output-details",
                            "--cache-report",
                            "--disable-tqdm",
                            "--output-file",
                            str(result_path),
                        ]
                        write(out / f"{stem}-command.json", command)
                        print(f"BENCH {stem}", flush=True)
                        oldargv = sys.argv
                        with (
                            (out / f"{stem}-bench.log").open("w") as client_log,
                            contextlib.redirect_stdout(client_log),
                            contextlib.redirect_stderr(client_log),
                        ):
                            try:
                                sys.argv = command[2:]
                                serving.cli_main()
                            finally:
                                sys.argv = oldargv
                        lines = result_path.read_text().splitlines()
                        assert len(lines) == 1
                        result = json.loads(lines[0])
                        after = info()
                        write(
                            out / f"{stem}-diagnostics.json",
                            {
                                "before": before,
                                "after": after,
                                "telemetry_before": telemetry_before,
                                "telemetry_after": telemetry(),
                            },
                        )
                        delta = {
                            k: after["hybrid_spec"][k] - v
                            for k, v in before["hybrid_spec"].items()
                        }
                        split_delta = {
                            k: after["hybrid_spec_split_graph"][k] - v
                            for k, v in before.get(
                                "hybrid_spec_split_graph", {}
                            ).items()
                            if isinstance(v, int)
                        }
                        retrieval = delta["retrieval_requests"]
                        neural = delta["neural_requests"]
                        row = {
                            k: result[k]
                            for k in (
                                "completed",
                                "total_input_tokens",
                                "total_output_tokens",
                                "duration",
                                "request_throughput",
                                "output_throughput",
                                "mean_e2e_latency_ms",
                                "p99_e2e_latency_ms",
                                "mean_ttft_ms",
                                "p99_ttft_ms",
                                "mean_tpot_ms",
                                "p99_tpot_ms",
                            )
                        }
                        row.update(
                            mode=mode,
                            scenario=scenario,
                            concurrency=concurrency,
                            round=round_id,
                            status="ok",
                            hybrid_spec_delta=delta,
                            sa_fraction=retrieval / (retrieval + neural)
                            if retrieval + neural
                            else 0,
                            sa_mean_accept_length=delta["retrieval_accept_tokens"]
                            / retrieval
                            if retrieval
                            else None,
                            mean_accept_length=(
                                delta["retrieval_accept_tokens"]
                                + delta["neural_accept_tokens"]
                            )
                            / (retrieval + neural),
                            split_delta=split_delta,
                            seed=seed_result,
                            official_result=str(result_path),
                            input_len_p50=float(
                                statistics.median(result["input_lens"])
                            ),
                            input_len_p95=sorted(result["input_lens"])[
                                int(0.95 * (len(result["input_lens"]) - 1))
                            ],
                            output_len_p50=float(
                                statistics.median(result["output_lens"])
                            ),
                            output_len_p95=sorted(result["output_lens"])[
                                int(0.95 * (len(result["output_lens"]) - 1))
                            ],
                            cache_report=result.get("cache_report"),
                        )
                        if (
                            result["completed"] != opts.num_prompts
                            or any(result["errors"])
                            or result["total_output_tokens"]
                            != opts.num_prompts * opts.max_tokens
                        ):
                            row["status"] = "failed"
                            row["failure_reason"] = (
                                "Request failure or unequal token budget"
                            )
                        if (
                            result.get("cache_report", {}).get("total_cached_tokens", 0)
                            != 0
                        ):
                            row["status"] = "failed"
                            row["failure_reason"] = "Unexpected KV prefix cache reuse"
                        rows.append(row)
                        write(out / "results.json", rows)
                        with (out / "results.jsonl").open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(
                            f"DONE {stem}: {row['output_throughput']:.2f} tok/s; SA={row['sa_fraction']:.3f}; skip={split_delta.get('skip_ct', 0)}; {row['status']}",
                            flush=True,
                        )
                        assert row["status"] == "ok", row.get("failure_reason")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                print(f"STOP {mode}", flush=True)
    print(f"COMPLETE {len(rows)} cases", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
