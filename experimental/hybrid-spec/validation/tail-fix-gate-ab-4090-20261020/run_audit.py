"""One server, existing public weights, 4 serial + 16 concurrent edit requests."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REPO = (ROOT / "source").resolve()
MODEL = "/root/rivermind-data/models/Qwen/Qwen3-8B"
DRAFT = "/root/rivermind-data/models/Qwen/Qwen3-0.6B"
URL = "http://127.0.0.1:39935"


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def api(route, data=None):
    response = (
        requests.get(URL + route, timeout=30)
        if data is None
        else requests.post(URL + route, json=data, timeout=180)
    )
    response.raise_for_status()
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        if route == "/flush_cache":
            return response.text
        raise


def run():
    from sglang.test.test_utils import terminate_and_kill_process_tree
    import torch

    assert not (ROOT / "audit.jsonl").exists(), "Use a fresh audit directory"
    config = dict(
        num_drafts=8,
        min_match_len=4,
        max_requests=4,
        max_history_tokens=8192,
        backend="gpu_sa",
        history_mode="device",
        num_neural_gate_tokens=1,
        draft_graph_mode="split",
        verify_mode="static",
        dynamic_verify=False,
        enable_global_pool=True,
        max_shared_requests=16,
        shared_context_tokens=128,
    )
    save("config.json", config)
    dist = ROOT / "split_audit-0.0.dist-info"
    dist.mkdir(exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: split-audit\nVersion: 0.0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[sglang.srt.plugins]\nsplit_audit = split_audit:register\nhybrid_spec = hybrid_spec.plugin:register\n"
    )
    env = os.environ.copy()
    env.update(
        PYTHONPATH=":".join(
            map(str, [ROOT, REPO / "python", REPO / "experimental/hybrid-spec/src"])
        ),
        SGLANG_PLUGINS="hybrid_spec,split_audit",
        HYBRID_SPEC_CONFIG=str(ROOT / "config.json"),
        SPLIT_AUDIT_ENABLE=str(ROOT / "enable-audit"),
        SPLIT_AUDIT_OUTPUT=str(ROOT / "audit.jsonl"),
        SPLIT_AUDIT_LIMIT="32",
        SGLANG_RAGGED_VERIFY_MODE="static",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MAX_JOBS="4",
    )
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--speculative-draft-model-path",
        DRAFT,
        "--speculative-algorithm",
        "STANDALONE",
        "--speculative-num-steps",
        "8",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "9",
        "--attention-backend",
        "triton",
        "--speculative-draft-attention-backend",
        "triton",
        "--dtype",
        "float16",
        "--context-length",
        "2048",
        "--max-running-requests",
        "4",
        "--cuda-graph-max-bs-decode",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--max-total-tokens",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        "39935",
        "--random-seed",
        "42",
        "--watchdog-timeout",
        "180",
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "-1",
    ]
    save("command.json", command)
    save(
        "manifest.json",
        dict(
            source_commit="a66968470a653ccaecab1c7c6a8d7473b31bcb3b",
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            model=MODEL,
            draft=DRAFT,
            config=config,
            protocol="Intrusive fixed-state draft audit; not end-to-end throughput. Raw neural columns2/3 copied in captured child graphs in both modes.",
        ),
    )
    process = None
    try:
        with (ROOT / "server.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        save("process.json", dict(pid=process.pid))
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited {process.returncode}; inspect server.log"
                )
            try:
                if requests.get(URL + "/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("server readiness exceeded six-minute bound")
        print("READY", flush=True)
        params = dict(temperature=0, max_new_tokens=128, ignore_eos=True)
        api(
            "/generate",
            dict(text=["Continue counting: 1,2,3,4,"] * 4, sampling_params=params),
        )
        api("/flush_cache", {})
        (ROOT / "enable-audit").touch()
        dataset = json.loads(
            Path(
                "/root/rivermind-data/benchmarks/qwen3-8b-06b-compact-ab-20261019/stable-dataset.json"
            ).read_text()
        )
        prompts = [row["conversations"][0]["value"] for row in dataset[:20]]
        save("prompts.json", prompts)
        responses = []
        for prompt in prompts[:4]:
            responses.append(
                api("/generate", dict(text=prompt, sampling_params=params))
            )
        print("SERIAL DONE", flush=True)
        request_count = 8 if os.environ.get("SPLIT_AUDIT_REPRO") == "1" else 20
        for start in range(4, request_count, 4):
            responses.extend(
                api(
                    "/generate",
                    dict(text=prompts[start : start + 4], sampling_params=params),
                )
            )
        save("responses.json", responses)
        assert len(responses) == request_count
        assert all(row["meta_info"]["completion_tokens"] == 128 for row in responses)
        save("server-info.json", api("/get_server_info"))
        print("AUDIT DONE", flush=True)
    finally:
        if process is not None:
            terminate_and_kill_process_tree(process, terminate_timeout=15)
        save(
            "final-gpu.json",
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv,noheader"],
                text=True,
            ),
        )


if __name__ == "__main__":
    run()
