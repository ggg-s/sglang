"""Audit fresh capacity-eight timing and preserve per-case GPU observations."""

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "matrix8"


def read(path):
    return json.loads(path.read_text())


def main():
    assert (MATRIX / "matrix.exit").read_text().strip() == "0"
    assert read(MATRIX / "summary.json")["complete"]
    assert read(ROOT / "final-validation.json")["status"] == "complete"
    events = [
        json.loads(line)
        for line in (MATRIX / "progress.jsonl").read_text().splitlines()
    ]
    with (MATRIX / "gpu-telemetry.csv").open() as stream:
        samples = [
            (
                dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"),
                int(row[3].strip().split()[0]),
            )
            for row in list(csv.reader(stream))[1:]
            if len(row) == 7
        ]
    keys = [
        "mamba_ssm_dtype",
        "dtype",
        "max_running_requests",
        "max_mamba_cache_size",
        "max_total_tokens",
        "context_length",
        "cuda_graph_max_bs_decode",
        "speculative_num_steps",
        "speculative_num_draft_tokens",
        "enable_linear_replayssm_spec",
        "linear_replayssm_cache_len",
        "attention_backend",
        "linear_attn_backend",
        "linear_attn_verify_backend",
        "random_seed",
        "disable_overlap_schedule",
        "mem_fraction_static",
    ]
    reference = None
    result = {
        "cases": [],
        "boots": [],
        "final_validation": read(ROOT / "final-validation.json"),
    }
    for mode in ("baseline-replay", "optimized-replay"):
        for repeat in (1, 2):
            label = f"{mode}-boot{repeat}"
            info = read(MATRIX / (label + "-ready.json"))
            params = {key: info[key] for key in keys}
            if reference is None:
                reference = params
            assert params == reference
            state = info["internal_states"][0]
            assert state["effective_max_running_requests_per_dp"] == 8
            assert state["memory_usage"]["token_capacity"] == 131072
            assert info["mamba_ssm_dtype"] == "float32"
            assert info["enable_linear_replayssm_spec"]
            sanity = read(MATRIX / (label + "-sanity-ok.json"))
            assert sanity["gsm8k"]["score"] > 0.8
            assert (
                sanity["gsm8k"]["error_rate"] == sanity["gsm8k"]["truncated_rate"] == 0
            )
            recovered = read(MATRIX / (label + "-cache-recovery.json"))
            assert recovered[1]["meta_info"]["cached_tokens"] > 0
            lines = (MATRIX / (label + "-server.log")).read_text().splitlines()
            pools = [
                line
                for line in lines
                if "Mamba Cache is allocated" in line
                or "GDN ReplaySSM ring buffers allocated" in line
            ]
            assert any(
                "intermediate_ssm_state_cache size: 0.00GB" in line for line in pools
            )
            result["boots"].append(
                {
                    "mode": mode,
                    "round": repeat,
                    "memory_usage": state["memory_usage"],
                    "sanity": sanity,
                    "cache_recovery_cached_tokens": recovered[1]["meta_info"][
                        "cached_tokens"
                    ],
                    "cache_recovery_text_equal": recovered[0]["text"]
                    == recovered[1]["text"],
                    "pool_log_lines": pools,
                }
            )
            for c in (1, 8):
                value = read(MATRIX / f"{mode}-r{repeat}-c{c}.jsonl")
                assert value["completed"] == 50 and not any(value["errors"])
                assert len(value["generated_texts"]) == 50 and all(
                    s.strip() for s in value["generated_texts"]
                )
                matched = [
                    e
                    for e in events
                    if e["mode"] == mode
                    and e.get("round") == repeat
                    and e.get("concurrency") == c
                    and e["event"].startswith("benchmark_")
                ]
                assert [e["event"] for e in matched] == [
                    "benchmark_start",
                    "benchmark_done",
                ]
                start, end = [
                    dt.datetime.strptime(e["time"], "%Y-%m-%dT%H:%M:%S%z").replace(
                        tzinfo=None
                    )
                    for e in matched
                ]
                observed = [memory for time, memory in samples if start <= time <= end]
                assert observed
                result["cases"].append(
                    {
                        "mode": mode,
                        "round": repeat,
                        "concurrency": c,
                        "observed_peak_gpu_mib": max(observed),
                        "telemetry_samples": len(observed),
                        "output_tokens": value["total_output_tokens"],
                    }
                )
    result["matched_server_parameters"] = reference
    result["workload_sha256"] = hashlib.sha256(
        (MATRIX / "workload.jsonl").read_bytes()
    ).hexdigest()
    assert (
        result["workload_sha256"]
        == "14fa95c7d417ae66a0e59dfeb7e9aeea4013334ba474d2a19ea4109f69f004c5"
    )
    for mode in ("baseline-replay", "optimized-replay"):
        result.setdefault("exact_repeat_text_matches_out_of_50", {})[mode] = {
            str(c): sum(
                a == b
                for a, b in zip(
                    read(MATRIX / f"{mode}-r1-c{c}.jsonl")["generated_texts"],
                    read(MATRIX / f"{mode}-r2-c{c}.jsonl")["generated_texts"],
                )
            )
            for c in (1, 8)
        }
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
