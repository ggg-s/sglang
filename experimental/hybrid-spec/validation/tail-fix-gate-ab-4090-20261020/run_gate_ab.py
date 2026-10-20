"""Bounded gate A/B: same fixed requests, C=4, two rounds, no audit hooks."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from run_audit import DRAFT, MODEL, REPO, ROOT, URL, api


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_arm(name, unique, prompts, command):
    from sglang.test.test_utils import terminate_and_kill_process_tree

    out = ROOT / name
    out.mkdir()
    config = {
        "num_drafts": 8,
        "min_match_len": 4,
        "max_requests": 4,
        "max_history_tokens": 8192,
        "backend": "gpu_sa",
        "history_mode": "device",
        "num_neural_gate_tokens": 1,
        "draft_graph_mode": "split",
        "verify_mode": "static",
        "dynamic_verify": False,
        "enable_global_pool": True,
        "max_shared_requests": 16,
        "shared_context_tokens": 128,
        "sa_require_unique_continuation": unique,
    }
    save(out / "config.json", config)
    env = os.environ | {
        "PYTHONPATH": ":".join(
            map(str, [ROOT, REPO / "python", REPO / "experimental/hybrid-spec/src"])
        ),
        "SGLANG_PLUGINS": "hybrid_spec",
        "HYBRID_SPEC_CONFIG": str(out / "config.json"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MAX_JOBS": "4",
    }
    process = None
    try:
        with (out / "server.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        save(out / "process.json", {"pid": process.pid})
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{name}: server exited {process.returncode}")
            try:
                if requests.get(URL + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"{name}: server readiness timeout")
        params = {"temperature": 0, "max_new_tokens": 192, "ignore_eos": True}
        api("/generate", {"text": prompts[:4], "sampling_params": params})
        rounds = []
        for index in range(2):
            api("/flush_cache", {})
            before = api("/get_server_info")
            responses = []
            start = time.perf_counter()
            for offset in range(0, len(prompts), 4):
                responses.extend(
                    api(
                        "/generate",
                        {
                            "text": prompts[offset : offset + 4],
                            "sampling_params": params,
                        },
                    )
                )
            elapsed = time.perf_counter() - start
            after = api("/get_server_info")
            assert len(responses) == len(prompts)
            assert all(r["meta_info"]["completion_tokens"] == 192 for r in responses)
            output = sum(r["meta_info"]["completion_tokens"] for r in responses)
            verifies = sum(r["meta_info"]["spec_verify_ct"] for r in responses)
            correct = sum(r["meta_info"]["spec_num_correct_drafts"] for r in responses)
            row = {
                "round": index,
                "elapsed_s": elapsed,
                "output_tokens": output,
                "verify_requests": verifies,
                "correct_drafts": correct,
                "throughput": output / elapsed,
                "accept_length": output / verifies,
                "correct_per_verify": correct / verifies,
                "text_sha256": [
                    hashlib.sha256(r["text"].encode()).hexdigest() for r in responses
                ],
            }
            rounds.append(row)
            save(out / f"round{index}-responses.json", responses)
            save(out / f"round{index}-before.json", before)
            save(out / f"round{index}-after.json", after)
            save(out / "summary.json", rounds)
            print(
                name,
                json.dumps({k: v for k, v in row.items() if k != "text_sha256"}),
                flush=True,
            )
        return rounds
    finally:
        if process is not None:
            terminate_and_kill_process_tree(process, terminate_timeout=15)


def run():
    # Reuse exactly the audited command; no diagnostic plugin is enabled.
    command = json.loads((ROOT / "command.json").read_text())
    assert command[0] == sys.executable
    assert MODEL in command and DRAFT in command
    dataset = json.loads(
        Path(
            "/root/rivermind-data/benchmarks/qwen3-8b-06b-compact-ab-20261019/stable-dataset.json"
        ).read_text()
    )
    prompts = [row["conversations"][0]["value"] for row in dataset[:24]]
    save(ROOT / "ab-prompts.json", prompts)
    results = {}
    for name, unique in (("old-gate", False), ("new-gate", True)):
        results[name] = run_arm(name, unique, prompts, command)
        save(ROOT / "ab-summary.json", results)
    save(
        ROOT / "ab-final-gpu.json",
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"],
            text=True,
        ),
    )


if __name__ == "__main__":
    run()
