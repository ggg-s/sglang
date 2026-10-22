"""Check matched, capacity-one SA16 snapshot/Replay runs."""

import csv
import datetime as dt
import hashlib
import json
import re
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    assert (ROOT / "matrix.exit").read_text().strip() == "0"
    assert read("performance-status.json")["measured_requests"] == 200
    summary = read("summary.json")
    assert summary["complete"] and summary["inputs_identical"]
    with socket.socket() as probe:
        assert probe.connect_ex(("127.0.0.1", 39922)) != 0
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"
    ], text=True)
    assert "0 MiB" in gpu
    events = [json.loads(line) for line in (ROOT / "progress.jsonl").read_text().splitlines()]
    with (ROOT / "gpu-telemetry.csv").open() as stream:
        samples = [
            (dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), int(row[3].strip().split()[0]))
            for row in list(csv.reader(stream))[1:] if len(row) == 7
        ]
    result = {
        "source": read("a100-source-ok.json"),
        "workload_sha256": hashlib.sha256((ROOT / "workload.jsonl").read_bytes()).hexdigest(),
        "gpu_after": gpu.strip(), "port_closed": True, "cases": [],
    }
    texts = {}
    for row in summary["rows"]:
        mode, repeat = row["mode"], row["round"]
        boot = f"{mode}-boot{repeat}"
        info = read(boot + "-ready.json")
        state = info["internal_states"][0]
        assert state["effective_max_running_requests_per_dp"] == 1
        assert state["memory_usage"]["token_capacity"] == 131072
        assert info["mamba_ssm_dtype"] == "float32"
        assert info["cuda_graph_max_bs_decode"] == 1
        assert info["enable_linear_replayssm_spec"] == (mode == "sa16-replay")
        sanity = read(boot + "-sanity-ok.json")
        assert sanity["gsm8k"]["score"] > 0.8
        assert sanity["gsm8k"]["error_rate"] == sanity["gsm8k"]["truncated_rate"] == 0
        recovered = read(boot + "-cache-recovery.json")
        assert recovered[1]["meta_info"]["cached_tokens"] > 0
        lines = (ROOT / (boot + "-server.log")).read_text().splitlines()
        assert not any(re.search(r"Traceback \(most recent call last\)|CUDA error:|CUDA out of memory", line) for line in lines)
        values = read(f"{mode}-r{repeat}-c1.jsonl")
        assert values["completed"] == 50 and not any(values["errors"])
        texts[mode, repeat] = values["generated_texts"]
        assert all(s.strip() for s in values["generated_texts"])
        matched = [e for e in events if e["mode"] == mode and e["round"] == repeat and e["event"].startswith("benchmark_")]
        assert [e["event"] for e in matched] == ["benchmark_start", "benchmark_done"]
        start, end = [dt.datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) for e in matched]
        observed = [memory for time, memory in samples if start <= time <= end]
        assert observed
        result["cases"].append({
            "mode": mode, "round": repeat, "memory_usage": state["memory_usage"],
            "observed_peak_gpu_mib": max(observed), "telemetry_samples": len(observed),
            "pool_log_lines": [line for line in lines if "Mamba Cache is allocated" in line or "GDN ReplaySSM ring buffers allocated" in line],
            "sanity": sanity, "cache_recovery_cached_tokens": recovered[1]["meta_info"]["cached_tokens"],
            "cache_recovery_text_equal": recovered[0]["text"] == recovered[1]["text"],
        })
    result["exact_repeat_text_matches_out_of_50"] = {
        mode: sum(a == b for a, b in zip(texts[mode, 1], texts[mode, 2]))
        for mode in ("sa16-snapshot", "sa16-replay")
    }
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
