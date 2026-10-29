"""Full-model Qwen3.6 MTP/PDMux acceptance on one CUDA GPU.

Run against a local checkpoint with its bundled MTP head. The baseline and
PDMux variants use the same greedy prompts, chunks and cache policy. Results
include token IDs; runtime completion and token parity are reported separately.
Example:
    python test/manual/pdmux/validate_qwen36_mtp_pdmux.py \
        --model-path /models/Qwen3.6-35B-A3B-FP8 --output-dir /tmp/mtp-check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer


def post(url, path, body):
    response = requests.post(url + path, json=body, timeout=600)
    response.raise_for_status()
    return response.json()


def workload(tokenizer):
    records = []
    filler = tokenizer.encode(
        "The engineering team measured request latency, memory usage and cache "
        "hits. Results were checked against a control before deployment. "
    )
    for index, length in enumerate((128, 513, 2304, 6144, 64, 4096, 768, 3072)):
        head = tokenizer.encode(f"Validation record {index}. Read this report:\n")
        tail = tokenizer.encode(
            "\nSummarize the report and list five useful validation steps.\nAssistant:"
        )
        count = length - len(head) - len(tail)
        prompt = head + (filler * (count // len(filler) + 1))[:count] + tail
        records.append((f"prompt-{index}", prompt))
    return records


def exercise(url, records, output):
    results = []

    def generate(item, phase, max_tokens, delay=0):
        time.sleep(delay)
        key, prompt = item
        start = time.monotonic()
        data = post(
            url,
            "/generate",
            {
                "input_ids": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
        )
        ids = data["output_ids"]
        assert len(ids) == max_tokens, (phase, key, len(ids))
        meta = data["meta_info"]
        assert meta.get("spec_verify_ct", 0) > 0, meta
        return {
            "phase": phase,
            "key": key,
            "prompt_tokens": len(prompt),
            "output_ids": ids,
            "seconds": time.monotonic() - start,
            "spec_verify_ct": meta["spec_verify_ct"],
            "spec_accept_length": meta.get("spec_accept_length"),
        }

    # Serial outputs are the same-shape control; later phases force live P/D
    # batches of different sizes and repeatedly reuse tracked prefix states.
    for item in records:
        results.append(generate(item, "serial", 64))
    for round_index in range(3):
        phase = f"concurrent-{round_index}"
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(generate, item, phase, 256, index * 0.025)
                for index, item in enumerate(records)
            ]
            results.extend(future.result() for future in futures)
        output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def run_variant(args, variant, records):
    out = args.output_dir
    log_path = out / f"{variant}.server.log"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--language-only",
        "--trust-remote-code",
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "flashinfer",
        "--linear-attn-prefill-backend",
        "triton",
        "--linear-attn-decode-backend",
        "triton",
        "--context-length",
        "8192",
        "--chunked-prefill-size",
        "2048",
        "--max-prefill-tokens",
        "8192",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "32768",
        "--mem-fraction-static",
        "0.82",
        "--max-mamba-cache-size",
        "64",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "disabled" if variant.endswith("eager") else "full",
        "--cuda-graph-max-bs-decode",
        "8",
        "--random-seed",
        "42",
        "--watchdog-timeout",
        "300",
        "--enable-metrics",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    is_baseline = variant.startswith("baseline")
    if not is_baseline:
        mode = "standard" if variant.startswith("standard") else "layer_split"
        command += [
            "--enable-pdmux",
            "--pdmux-prefill-mode",
            mode,
            "--sm-group-num",
            "3",
            "--pdmux-config-path",
            str(out / "pdmux.yaml"),
        ]
    (out / f"{variant}.command.json").write_text(json.dumps(command, indent=2))
    url = f"http://127.0.0.1:{args.port}"
    summary = {"variant": variant, "passed": False}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=log, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited: {process.returncode}; see {log_path}"
                    )
                try:
                    if requests.get(url + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("server startup")
            info = requests.get(url + "/get_server_info", timeout=30).json()
            assert info["speculative_algorithm"] == "EAGLE"
            assert info["enable_pdmux"] == (not is_baseline)
            assert not info["disable_overlap_schedule"]
            assert info["cuda_graph_config"]["decode"]["backend"] == (
                "disabled" if variant.endswith("eager") else "full"
            )
            (out / f"{variant}.server_info.json").write_text(json.dumps(info, indent=2))
            start = time.monotonic()
            output = out / f"{variant}.outputs.json"
            results = exercise(url, records, output)
            summary.update(
                passed=True,
                requests=len(results),
                output_tokens=sum(len(r["output_ids"]) for r in results),
                workload_seconds=time.monotonic() - start,
            )
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    text = log_path.read_text()
    summary["prefills_with_running_decode"] = len(
        re.findall(r"Prefill batch,.*#running-req: [1-9]", text)
    )
    summary["draft_graph_replay"] = (
        "PDMux draft_decode CUDA graph replay active" in text
    )
    summary["scatter_length_mismatch"] = "indices length mismatch" in text
    if not is_baseline and summary["passed"]:
        baseline_name = "baseline_eager" if variant.endswith("eager") else "baseline"
        baseline_path = out / f"{baseline_name}.outputs.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text())
            summary["baseline_token_matches"] = sum(
                a["output_ids"] == b["output_ids"] for a, b in zip(results, baseline)
            )
            summary["serial_baseline_token_matches"] = sum(
                a["output_ids"] == b["output_ids"]
                for a, b in zip(results[: len(records)], baseline[: len(records)])
            )
            summary["token_parity_passed"] = (
                len(results) == len(baseline)
                and summary["baseline_token_matches"] == len(results)
            )
    (out / f"{variant}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=39935)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline",
            "layer_split_graphs",
            "standard_graphs",
        ],
        choices=[
            "baseline",
            "baseline_eager",
            "layer_split_graphs",
            "layer_split_eager",
            "standard_graphs",
            "standard_eager",
        ],
    )
    parser.add_argument("--layout", choices=["fixed", "full"], default="fixed")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = torch.cuda.get_device_properties(0).multi_processor_count
    prefill_sms = total // 2 // 8 * 8
    decode_sms = (total - prefill_sms) // 8 * 8
    body = "sm_group_num: 3\nsplit_forward_token_budget: 8192\nsplit_forward_max_layers: 4\n"
    if args.layout == "full":
        body += "prefill_full_sm: true\n"
    else:
        body += f"manual_divisions:\n  - [{prefill_sms}, {decode_sms}, 1]\n"
    (args.output_dir / "pdmux.yaml").write_text(body)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    records = workload(tokenizer)
    summaries = [run_variant(args, variant, records) for variant in args.variants]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    if not all(item["passed"] for item in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
