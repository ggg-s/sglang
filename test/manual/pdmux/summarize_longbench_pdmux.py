"""Validate and summarize a completed two-trial LongBench DFlash/PDMux run.

Writes JSON to --output and a compact comparison table to stdout. All input
artifacts are read-only. Trial quantiles are never pooled across trials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


GROUPS = ("dflash", "pdmux")
PHASES = ("cold", "repeated")
METRICS = {
    "output_tok_s": ("successful_output_tok_s",),
    "ttft_p50_ms": ("ttft_ms", "p50"),
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "tpot_mean_ms": ("tpot_ms", "mean"),
    "tpot_p95_ms": ("tpot_ms", "p95"),
    "e2e_p50_ms": ("e2e_ms", "p50"),
    "cached_tokens": ("cached_tokens",),
}
EXPECTED_PARAMETERS = {
    "concurrency": 4,
    "output_tokens_per_request": 128,
    "context_length": 16384,
    "requested_trials": 2,
    "sampling": {"temperature": 0, "top_k": 1, "ignore_eos": True},
}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def nested(value, path):
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def median(values):
    # Preserve missing measurements instead of silently taking fewer trials.
    return statistics.median(values) if values and all(map(finite, values)) else None


def compare(baseline, candidate):
    if not finite(baseline) or not finite(candidate):
        return {"delta": None, "relative_change_pct": None}
    return {
        "delta": candidate - baseline,
        "relative_change_pct": (candidate / baseline - 1) * 100 if baseline else None,
    }


def summarize(run_dir):
    errors = []
    result = {
        "schema_version": 1,
        "success": False,
        "run_dir": str(run_dir),
        "aggregation": "Median of two per-trial metrics, separately for cold and repeated; warmup excluded.",
        "relative_change_definition": "100 * (PDMux / DFlash - 1); null when baseline is zero or a measurement is missing.",
        "spec_metric_definition": (
            "For requests exposing meta.spec_verify_ct, sum(completion_tokens) / "
            "sum(spec_verify_ct), then median across trials. The numerator includes "
            "all delivered output tokens, including any initial prefill or bonus "
            "tokens; this is output tokens per verify, not draft acceptance rate. "
            "Coverage is recorded and no missing counters are imputed."
        ),
        "inputs": {},
        "parameters": EXPECTED_PARAMETERS,
        "groups": {},
        "comparison": {},
        "validation_errors": errors,
    }

    def require(condition, message):
        if not condition:
            errors.append(message)

    def read(relative):
        path = run_dir / relative
        raw = path.read_bytes()
        result["inputs"][relative] = {
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return json.loads(raw)

    try:
        status = read("status.json")
        source = read("source-selection.json")
        results = {group: read(f"{group}/results.json") for group in GROUPS}
        require(status.get("success") is True and status.get("state") == "complete",
                "Driver status is not a completed successful run")
        status_groups = status.get("groups", [])
        require(len(status_groups) == 2 and
                {item.get("group") for item in status_groups} == set(GROUPS),
                "Driver must contain exactly dflash and pdmux groups")
        for item in status_groups:
            require(item.get("success") is True and item.get("state") == "complete",
                    f"Driver group {item.get('group')} failed or is unfinished")
        require(status.get("trials_per_group") == 2, "Driver trial count must be two")
        manifest_hash = nested(source, ("workload", "sha256"))
        require(bool(manifest_hash), "Source manifest lacks workload SHA256")
        require(source.get("request_count") == 32, "Manifest must contain 32 requests")
        result["workload_sha256"] = manifest_hash
        result["source_selection"] = source
        reference_requests = None
        for group in GROUPS:
            data = results[group]
            require(data.get("success") is True, f"{group}: benchmark failed")
            require(nested(data, ("workload", "sha256")) == manifest_hash,
                    f"{group}: workload SHA256 differs from source manifest")
            require(nested(data, ("workload", "requests")) == 32,
                    f"{group}: workload request count differs from 32")
            for key, expected in EXPECTED_PARAMETERS.items():
                require(data.get(key) == expected, f"{group}: parameter {key} differs")
            trials = data.get("trials", [])
            require(len(trials) == 2 and {t.get("trial") for t in trials} == {1, 2},
                    f"{group}: expected trials 1 and 2")
            group_result = {phase: {"trials": []} for phase in PHASES}
            result["groups"][group] = group_result
            for trial in trials:
                label = f"{group}/trial_{trial.get('trial')}"
                require(trial.get("success") is True, f"{label}: trial failed")
                require(nested(trial, ("flush", "success")) is True,
                        f"{label}: cold cache flush did not succeed")
                for phase in PHASES:
                    payload = trial.get(phase, {})
                    tag = f"{label}/{phase}"
                    require(payload.get("success") is True, f"{tag}: phase failed")
                    require(payload.get("cache_precondition_valid") is True,
                            f"{tag}: cache precondition invalid")
                    for key, expected in {
                        "requests": 32, "successful_requests": 32,
                        "failed_requests": 0, "successful_output_tokens": 4096,
                    }.items():
                        require(payload.get(key) == expected, f"{tag}: {key} != {expected}")
                    records = payload.get("records", [])
                    require(len(records) == 32, f"{tag}: expected 32 request records")
                    identities, spec_records = [], []
                    for row in records:
                        identity = (
                            row.get("index"), row.get("task"), row.get("id"),
                            row.get("prompt_sha256"), row.get("prompt_tokens"),
                        )
                        identities.append(identity)
                        request_tag = f"{tag}/request_{row.get('index')}"
                        require(row.get("success") is True and not row.get("error"),
                                f"{request_tag}: request failed")
                        require(row.get("stream_done") is True,
                                f"{request_tag}: missing stream terminal marker")
                        ids = row.get("output_ids", [])
                        require(row.get("completion_tokens") == 128 and
                                isinstance(ids, list) and len(ids) == 128 and
                                all(type(token) is int for token in ids),
                                f"{request_tag}: expected 128 output token IDs")
                        meta = row.get("meta", {})
                        if "spec_verify_ct" in meta:
                            count = meta["spec_verify_ct"]
                            require(type(count) is int and count >= 0,
                                    f"{request_tag}: invalid spec_verify_ct")
                            if type(count) is int and count >= 0:
                                spec_records.append((row["completion_tokens"], count))
                    require(len({str(item[:3]) for item in identities}) == 32,
                            f"{tag}: duplicate request identities")
                    if reference_requests is None:
                        reference_requests = identities
                    require(identities == reference_requests,
                            f"{tag}: prompt identity/order differs from the first phase")
                    metrics = {key: nested(payload, path) for key, path in METRICS.items()}
                    require(finite(metrics["output_tok_s"]) and metrics["output_tok_s"] > 0,
                            f"{tag}: invalid throughput")
                    wall = payload.get("wall_s")
                    require(finite(wall) and wall > 0 and
                            finite(metrics["output_tok_s"]) and
                            math.isclose(metrics["output_tok_s"], 4096 / wall, rel_tol=1e-9),
                            f"{tag}: throughput does not equal 4096 / wall_s")
                    saved = {"trial": trial["trial"], "wall_s": wall, **metrics}
                    if spec_records:
                        output_tokens = sum(item[0] for item in spec_records)
                        verify_ct = sum(item[1] for item in spec_records)
                        saved["spec"] = {
                            "requests_with_counter": len(spec_records),
                            "requests_total": len(records),
                            "completion_tokens_with_counter": output_tokens,
                            "spec_verify_ct": verify_ct,
                            "output_tokens_per_verify": output_tokens / verify_ct
                            if verify_ct else None,
                        }
                    group_result[phase]["trials"].append(saved)
            for phase in PHASES:
                phase_result = group_result[phase]
                phase_result["median_of_trial_metrics"] = {
                    key: median([trial[key] for trial in phase_result["trials"]])
                    for key in METRICS
                }
                spec_values = [nested(trial, ("spec", "output_tokens_per_verify"))
                               for trial in phase_result["trials"]]
                if any(finite(value) for value in spec_values):
                    phase_result["median_of_trial_metrics"]["output_tokens_per_verify"] = median(spec_values)
        require(results["dflash"].get("timing_definitions") ==
                results["pdmux"].get("timing_definitions"),
                "Timing definitions differ between groups")
        for phase in PHASES:
            baseline = result["groups"]["dflash"][phase]["median_of_trial_metrics"]
            candidate = result["groups"]["pdmux"][phase]["median_of_trial_metrics"]
            result["comparison"][phase] = {
                key: {"dflash": baseline.get(key), "pdmux": candidate.get(key),
                      **compare(baseline.get(key), candidate.get(key))}
                for key in dict.fromkeys([*baseline, *candidate])
            }
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["success"] = not errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, output = args.run_dir.resolve(), args.output.resolve()
    inputs = [run_dir / name for name in (
        "status.json", "source-selection.json", "dflash/results.json", "pdmux/results.json",
    )]
    if output in inputs:
        parser.error("--output must not overwrite an input artifact")
    result = summarize(run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    if not result["success"]:
        print("Validation failed; comparison is not a valid completed benchmark.", file=sys.stderr)
        for error in result["validation_errors"]:
            print("- " + error, file=sys.stderr)
        return 1
    print(f"{'phase':<9} {'metric':<25} {'DFlash':>12} {'PDMux':>12} {'change':>10}")
    for phase, metrics in result["comparison"].items():
        for key, values in metrics.items():
            baseline = f"{values['dflash']:.3f}" if finite(values["dflash"]) else "n/a"
            candidate = f"{values['pdmux']:.3f}" if finite(values["pdmux"]) else "n/a"
            change = values["relative_change_pct"]
            delta = f"{change:+.2f}%" if finite(change) else "n/a"
            print(f"{phase:<9} {key:<25} {baseline:>12} {candidate:>12} {delta:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
