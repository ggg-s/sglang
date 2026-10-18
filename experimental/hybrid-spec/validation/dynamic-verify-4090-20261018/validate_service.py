import json
import os
import pathlib
import subprocess
import time
import traceback

import psutil
import requests

ROOT = pathlib.Path("/root/rivermind-data/benchmarks/dynamic-verify-4090-20261018")
REPO = ROOT / "repo"
PY = "/root/rivermind-data/envs/sglang/bin/python"
MODEL = "/root/rivermind-data/models/Qwen/Qwen3-0.6B"
URL = "http://127.0.0.1:39923"
RECEIPT = pathlib.Path("/root/rivermind-data/sglang-current-service.json")


def save(p, x):
    p.write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n")


def api(url, route, body=None):
    r = (
        requests.get(url + route, timeout=120)
        if body is None
        else requests.post(url + route, json=body, timeout=600)
    )
    r.raise_for_status()
    return r.json()


def stop(p):
    try:
        procs = p.children(recursive=True) + [p]
    except psutil.NoSuchProcess:
        return
    for c in procs:
        try:
            c.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=10)
    for c in alive:
        try:
            c.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def ready(url, p, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.poll() is not None:
            raise RuntimeError("server exit " + str(p.returncode))
        try:
            if requests.get(url + "/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError("health timeout")


def launch(case, config):
    folder = ROOT / case
    folder.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.pop("HYBRID_SPEC_CONFIG", None)
    env["PYTHONPATH"] = (
        str(REPO / "python") + ":" + str(REPO / "experimental/hybrid-spec/src")
    )
    env["TORCH_EXTENSIONS_DIR"] = str(ROOT / "torch-extensions")
    env["MAX_JOBS"] = "4"
    env["TORCH_CUDA_ARCH_LIST"] = "8.9"
    env["SGLANG_PROFILE_V2"] = "0"
    if config is not None:
        save(folder / "config.json", config)
        env["HYBRID_SPEC_CONFIG"] = str(folder / "config.json")
    cmd = [
        PY,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--speculative-draft-model-path",
        MODEL,
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
        "16",
        "--cuda-graph-max-bs-decode",
        "16",
        "--mem-fraction-static",
        "0.4",
        "--max-total-tokens",
        "8192",
        "--host",
        "127.0.0.1",
        "--port",
        "39923",
        "--random-seed",
        "42",
        "--watchdog-timeout",
        "600",
        "--disable-overlap-schedule",
        "--chunked-prefill-size",
        "-1",
    ]
    save(folder / "command.json", cmd)
    p = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=env,
        stdout=(folder / "server.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    save(
        folder / "process.json",
        dict(pid=p.pid, create_time=psutil.Process(p.pid).create_time()),
    )
    try:
        ready(URL, p)
    except BaseException:
        try:
            stop(psutil.Process(p.pid))
        except psutil.NoSuchProcess:
            pass
        raise
    save(folder / "initial-info.json", api(URL, "/get_server_info"))
    print("READY", case, flush=True)
    return p, folder


prompts = [
    (
        "Copy the following pattern and continue it exactly: "
        + (" alpha beta gamma delta epsilon zeta. " * 14)
        + "\nalpha beta gamma"
    )
    if i % 2 == 0
    else (
        "Write a Python function to "
        + [
            "sort a list",
            "compute Fibonacci numbers",
            "find a prime",
            "merge dictionaries",
            "count words",
            "reverse a string",
            "traverse a tree",
            "parse a CSV",
        ][i // 2]
        + ". Include examples and comments.\n"
    )
    for i in range(16)
]


def payload(indices, lengths):
    return dict(
        text=[prompts[i] for i in indices],
        sampling_params=[
            dict(temperature=0, max_new_tokens=n, ignore_eos=True) for n in lengths
        ],
        return_logprob=True,
        logprob_start_len=-1,
        top_logprobs_num=0,
    )


def exercise(folder):
    responses = []
    for c in (1, 4, 16):
        for wave in range(3 if c == 16 else 1):
            lengths = [128 if c < 16 else [96, 128, 160, 192][i % 4] for i in range(c)]
            before = time.perf_counter()
            out = api(URL, "/generate", payload(range(c), lengths))
            elapsed = time.perf_counter() - before
            if isinstance(out, dict):
                out = [out]
            assert len(out) == c
            assert [x["meta_info"]["completion_tokens"] for x in out] == lengths
            save(
                folder / f"c{c}-wave{wave}.json",
                dict(elapsed_s=elapsed, lengths=lengths, responses=out),
            )
            responses.append(
                dict(
                    c=c,
                    wave=wave,
                    elapsed_s=elapsed,
                    completed=len(out),
                    output_tokens=sum(lengths),
                )
            )
            print("GENERATED", folder.name, c, wave, elapsed, flush=True)
    info = api(URL, "/get_server_info")
    save(folder / "final-info.json", info)
    save(folder / "runs.json", responses)
    return info


current = None
try:
    p, folder = launch("native", None)
    current = psutil.Process(p.pid)
    exercise(folder)
    stop(current)
    current = None
    cfg = dict(
        num_drafts=8,
        num_neural_drafts=3,
        min_match_len=2,
        max_requests=16,
        max_history_tokens=2048,
        backend="gpu_sa",
        history_mode="device",
        partial_drafts="fallback",
        draft_graph_mode="split",
        num_neural_gate_tokens=1,
        verify_mode="compact",
        enable_global_pool=True,
        max_shared_requests=16,
    )
    for case, config in [
        ("fixed", cfg),
        ("dynamic", dict(cfg, dynamic_verify=True)),
        (
            "gate3-neural2",
            dict(
                cfg,
                num_neural_drafts=2,
                num_neural_gate_tokens=3,
                dynamic_verify=True,
                dynamic_verify_min_samples=8,
                dynamic_verify_explore_interval=8,
            ),
        ),
        (
            "gate8-neural1",
            dict(
                cfg,
                num_neural_drafts=1,
                num_neural_gate_tokens=8,
                dynamic_verify=True,
                dynamic_verify_min_samples=8,
                dynamic_verify_explore_interval=8,
            ),
        ),
    ]:
        p, folder = launch(case, config)
        current = psutil.Process(p.pid)
        exercise(folder)
        stop(current)
        current = None
    (ROOT / "validation.done").write_text("ok\n")
    print("VALIDATED", flush=True)
except BaseException as e:
    save(ROOT / "failure.json", dict(error=repr(e), traceback=traceback.format_exc()))
    print(traceback.format_exc(), flush=True)
    raise
finally:
    if current is not None:
        stop(current)
    (ROOT / "gpu-released.done").write_text("ok\n")
