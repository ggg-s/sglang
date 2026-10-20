"""Summarize paired, bounded runs from original responses and counters."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            result = find(value, key)
            if result is not None:
                return result
    elif isinstance(node, list):
        for value in node:
            result = find(value, key)
            if result is not None:
                return result
    return None


def summarize(root=ROOT):
    if not (root / "ab-summary.json").exists():
        root = root / "evidence"
    results = json.loads((root / "ab-summary.json").read_text())
    report = {}
    for name, rounds in results.items():
        output = sum(r["output_tokens"] for r in rounds)
        verifies = sum(r["verify_requests"] for r in rounds)
        elapsed = sum(r["elapsed_s"] for r in rounds)
        counters = []
        for index in range(len(rounds)):
            before = json.loads((root / name / f"round{index}-before.json").read_text())
            after = json.loads((root / name / f"round{index}-after.json").read_text())
            row = {}
            for key in ("hybrid_spec", "hybrid_spec_split_graph"):
                a, b = find(before, key), find(after, key)
                if a is not None and b is not None:
                    row[key] = {k: b[k] - v for k, v in a.items() if type(v) is int}
                    assert all(v >= 0 for v in row[key].values()), (name, index, key)
            counters.append(row)
        report[name] = {
            "output_tokens": output,
            "elapsed_s": elapsed,
            "throughput": output / elapsed,
            "verify_requests": verifies,
            "accept_length": output / verifies,
            "correct_per_verify": sum(r["correct_drafts"] for r in rounds) / verifies,
            "round_throughput": [r["throughput"] for r in rounds],
            "round_accept_length": [r["accept_length"] for r in rounds],
            "repeat_text_equal": rounds[0]["text_sha256"] == rounds[1]["text_sha256"],
            "counter_deltas": counters,
        }
    old, new = report["old-gate"], report["new-gate"]
    report["change_pct"] = {
        key: (new[key] / old[key] - 1) * 100
        for key in ("throughput", "accept_length", "correct_per_verify")
    }
    report["paired_round_change_pct"] = [
        {
            key: (b[key] / a[key] - 1) * 100
            for key in ("throughput", "accept_length", "correct_per_verify")
        }
        for a, b in zip(results["old-gate"], results["new-gate"])
    ]
    report["cross_arm_equal_text_requests"] = [
        sum(a == b for a, b in zip(x["text_sha256"], y["text_sha256"]))
        for x, y in zip(results["old-gate"], results["new-gate"])
    ]
    (root / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
