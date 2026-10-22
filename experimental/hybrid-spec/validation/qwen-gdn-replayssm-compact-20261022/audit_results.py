"""Audit completed A100 measurements and retain memory/output evidence."""

import csv
import datetime as dt
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ("native-k8", "native-k8-replay", "sa16-n8-replay")


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    assert (ROOT / "matrix.exit").read_text().strip() == "0"
    assert read("performance-status.json")["measured_requests"] == 600
    summary = read("summary.json")
    assert summary["complete"] and summary["inputs_identical"]
    events = [json.loads(line) for line in (ROOT / "progress.jsonl").read_text().splitlines()]
    with (ROOT / "gpu-telemetry.csv").open() as stream:
        samples = [
            (
                dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                int(row[3].strip().split()[0]),
            )
            for row in list(csv.reader(stream))[1:]
            if len(row) == 7
        ]
    result = {"source": read("a100-source-ok.json"), "modes": {}}
    for mode in MODES:
        info = read(mode + "-ready.json")
        state = info["internal_states"][0]
        assert state["effective_max_running_requests_per_dp"] == 8
        assert state["memory_usage"]["token_capacity"] == 131072
        assert info["mamba_ssm_dtype"] == "float32"
        sanity = read(mode + "-sanity-ok.json")
        assert sanity["gsm8k"]["error_rate"] == 0
        assert sanity["gsm8k"]["truncated_rate"] == 0
        recovered = read(mode + "-cache-recovery.json")
        cached_tokens = recovered[1]["meta_info"]["cached_tokens"]
        assert cached_tokens > 0
        logs = (ROOT / (mode + "-server.log")).read_text().splitlines()
        memory_lines = [
            line for line in logs
            if "Mamba Cache is allocated" in line or "GDN ReplaySSM ring buffers allocated" in line
        ]
        assert memory_lines
        assert not any(re.search(r"Traceback \(most recent call last\)|CUDA error:|CUDA out of memory", line) for line in logs)
        cases = []
        texts = {}
        for concurrency in (1, 8):
            for repeat in (1, 2):
                value = read(f"{mode}-r{repeat}-c{concurrency}.jsonl")
                assert value["completed"] == 50 and not any(value["errors"])
                assert len(value["generated_texts"]) == 50
                texts[concurrency, repeat] = value["generated_texts"]
                matched = [e for e in events if e["mode"] == mode and e["round"] == repeat and e["concurrency"] == concurrency]
                assert [e["event"] for e in matched] == ["benchmark_start", "benchmark_done"]
                start, end = [dt.datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%S%z").replace(tzinfo=None) for e in matched]
                observed = [memory for time, memory in samples if start <= time <= end]
                assert observed
                cases.append({
                    "concurrency": concurrency,
                    "round": repeat,
                    "observed_peak_gpu_mib": max(observed),
                    "telemetry_sample_count": len(observed),
                    "empty_outputs": sum(not s.strip() for s in value["generated_texts"]),
                })
        result["modes"][mode] = {
            "memory_usage": state["memory_usage"],
            "pool_log_lines": memory_lines,
            "sanity": sanity,
            "cache_recovery_cached_tokens": cached_tokens,
            "cache_recovery_text_equal": recovered[0]["text"] == recovered[1]["text"],
            "exact_repeat_text_matches_out_of_50": {
                str(c): sum(a == b for a, b in zip(texts[c, 1], texts[c, 2])) for c in (1, 8)
            },
            "cases": cases,
        }
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
