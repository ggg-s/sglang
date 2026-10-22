"""Summarize matched timing cases; device deltas exclude all warmup requests."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODES = ["baseline-replay", "optimized-replay"]


def read(path):
    return json.loads(path.read_text())


def delta(before, after):
    return {
        key: delta(before.get(key, {}), value)
        if isinstance(value, dict)
        else value - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, (int, float, dict))
    }


def main():
    rows = []
    inputs = None
    for mode in MODES:
        for c in [1, 8]:
            for r in [1, 2]:
                stem = f"{mode}-r{r}-c{c}"
                path = ROOT / f"{stem}.jsonl"
                after_path = ROOT / f"{stem}-after.json"
                if not path.exists() or not after_path.exists():
                    continue
                value = read(path)
                assert value["completed"] == 50 and not any(value["errors"]), stem
                argv = read(ROOT / f"{stem}-command.json")
                assert argv[argv.index("--warmup-requests") + 1] == "0"
                if inputs is None:
                    inputs = value["input_lens"]
                assert value["input_lens"] == inputs, stem
                before = read(ROOT / f"{stem}-before.json")["internal_states"][0]
                after = read(after_path)["internal_states"][0]
                row = {
                    "mode": mode,
                    "concurrency": c,
                    "round": r,
                    **{
                        key: value[key]
                        for key in [
                            "output_throughput",
                            "mean_tpot_ms",
                            "mean_ttft_ms",
                            "duration",
                            "total_output_tokens",
                        ]
                    },
                    "diagnostics_scope": "50 measured requests only; manual warmup excluded",
                    "output_cap_hits": sum(n == 2048 for n in value["output_lens"]),
                    "complete_tool_markers": sum(
                        "<tool_call>" in text and "</tool_call>" in text
                        for text in value["generated_texts"]
                    ),
                    "diagnostics": {
                        key: delta(before.get(key, {}), after[key])
                        for key in [
                            "hybrid_spec",
                            "hybrid_spec_compact_verify",
                            "hybrid_spec_split_graph",
                        ]
                        if key in after
                    },
                }
                diagnostics = row["diagnostics"]
                if "hybrid_spec" in diagnostics:
                    config = read(ROOT / "sa16-n8.json")
                    metrics = diagnostics["hybrid_spec"]
                    sa, neural = (
                        metrics["retrieval_requests"],
                        metrics["neural_requests"],
                    )
                    total = (
                        metrics["retrieval_accept_tokens"]
                        + metrics["neural_accept_tokens"]
                    )
                    compact = diagnostics["hybrid_spec_compact_verify"]
                    row.update(
                        sa_selection_rate=sa / (sa + neural),
                        overall_accept_length=total / (sa + neural),
                        sa_accept_length=metrics["retrieval_accept_tokens"] / sa
                        if sa
                        else None,
                        sa_accept_rate=metrics["retrieval_correct_drafts"]
                        / (sa * config["num_drafts"])
                        if sa
                        else None,
                        neural_accept_length=metrics["neural_accept_tokens"] / neural
                        if neural
                        else None,
                        verify_rows_per_1k_accept=1000 * (sa + neural) / total,
                        target_positions_per_1k_accept=1000
                        * compact["target_token_ct"]
                        / total,
                        target_position_reduction=1
                        - compact["target_token_ct"] / compact["dense_token_ct"],
                        verify_rows_per_1k_output=1000
                        * (sa + neural)
                        / value["total_output_tokens"],
                        target_positions_per_1k_output=1000
                        * compact["target_token_ct"]
                        / value["total_output_tokens"],
                    )
                    if "hybrid_spec_split_graph" in diagnostics:
                        split = diagnostics["hybrid_spec_split_graph"]
                        row.update(
                            prefix_ct=split["prefix_ct"],
                            prefix_per_1k_accept=1000 * split["prefix_ct"] / total,
                            tail_rows_per_1k_accept=1000
                            * split["tail_request_ct"]
                            / total,
                            host_decision_wait_ms_per_prefix=split["decision_wait_ns"]
                            / split["prefix_ct"]
                            / 1e6,
                            prefix_per_1k_output=1000
                            * split["prefix_ct"]
                            / value["total_output_tokens"],
                            tail_rows_per_1k_output=1000
                            * split["tail_request_ct"]
                            / value["total_output_tokens"],
                        )
                rows.append(row)
    aggregate = []
    for mode in MODES:
        for concurrency in [1, 8]:
            selected = [
                r for r in rows if r["mode"] == mode and r["concurrency"] == concurrency
            ]
            if not selected:
                continue
            result = {"mode": mode, "concurrency": concurrency, "rounds": len(selected)}
            for key in selected[0]:
                if key in ["mode", "concurrency", "round"]:
                    continue
                values = [
                    r[key] for r in selected if isinstance(r.get(key), (int, float))
                ]
                if values:
                    result[key] = {"mean": statistics.mean(values), "by_round": values}
            for baseline in ["baseline-replay"]:
                reference = {
                    r["round"]: r
                    for r in rows
                    if r["mode"] == baseline and r["concurrency"] == concurrency
                }
                paired = [r for r in selected if r["round"] in reference]
                if paired:
                    result[f"vs_{baseline}"] = {
                        "throughput_gain_pct": statistics.mean(
                            [
                                (
                                    r["output_throughput"]
                                    / reference[r["round"]]["output_throughput"]
                                    - 1
                                )
                                * 100
                                for r in paired
                            ]
                        ),
                        "tpot_reduction_pct": statistics.mean(
                            [
                                (
                                    1
                                    - r["mean_tpot_ms"]
                                    / reference[r["round"]]["mean_tpot_ms"]
                                )
                                * 100
                                for r in paired
                            ]
                        ),
                    }
            aggregate.append(result)
    summary = {
        "complete": len(rows) == 8,
        "benchmark_count": len(rows),
        "measured_requests": len(rows) * 50,
        "inputs_identical": True,
        "rows": rows,
        "aggregate": aggregate,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in rows:
        print(
            r["mode"],
            r["concurrency"],
            r["round"],
            round(r["output_throughput"], 3),
            round(r["mean_tpot_ms"], 3),
            r.get("overall_accept_length"),
        )


if __name__ == "__main__":
    main()
