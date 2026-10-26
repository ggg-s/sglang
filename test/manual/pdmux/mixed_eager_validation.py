"""GPU-free contracts for the private mixed validation driver and worker.

Historical reports remain evidence. Only complete, freshly source-matched
stable Eager gates may authorize a new offline Mixed Graph capture.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

TOLERANCES = dict(rtol=0.02, atol=0.002)
EAGER_CHUNKS = (23, 32, 40, 64, 65, 128, 257, 430, 512)
EAGER_CAPACITIES = (32, 32, 64, 64, 128, 128, 512, 512, 512)
GRAPH_CHUNKS = (32, 32, 32, 128, 128, 512, 512, 23, 23)
GRAPH_CAPACITIES = (32, 32, 32, 128, 128, 512, 512, 32, 32)
NUMERICAL_SOURCES = (
    "python/sglang/srt/speculative/mixed_eager.py",
    "python/sglang/srt/speculative/mixed_marlin.py",
    "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py",
    "test/manual/pdmux/mixed_eager_harness.py",
    "test/manual/pdmux/mixed_eager_validation.py",
    "test/manual/pdmux/bench_mixed_full_model.py",
)
DIAGNOSTIC_FLAGS = (
    "numerical_probe",
    "freeze_token_local_diagnostic",
    "fixed_reduction_diagnostic",
    "fp32_router_diagnostic",
)


def write_json(path, value):
    """Publish complete reports/requests; readers never observe partial JSON."""
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    pending.replace(path)


def source_hashes(repo, names=NUMERICAL_SOURCES):
    return {
        name: hashlib.sha256((Path(repo) / name).read_bytes()).hexdigest()
        for name in names
    }


def proposals_digest(cases):
    """Digest only proposals, independently of chunk geometry or file paths."""
    values = [case["proposal_ids"] for case in cases]
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


def validate_proposals(report, *, rounds):
    cases = report.get("cases", [])
    if len(cases) < rounds:
        raise ValueError("Proposal source has fewer cases than this run")
    selected = cases[:rounds]
    for case in selected:
        ids = case.get("proposal_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != 3
            or any(
                not isinstance(row, list)
                or len(row) != 8
                or any(type(token) is not int or token < 0 for token in row)
                for row in ids
            )
        ):
            raise ValueError("Proposal source must contain full D3 x block8 IDs")
    return proposals_digest(selected)


def _state_matches(metrics):
    return (
        isinstance(metrics, dict)
        and bool(metrics)
        and all(
            isinstance(row, dict)
            and row.get("mismatch") == 0
            and type(row.get("elements")) is int
            and row["elements"] > 0
            and all(
                isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                for key in ("max_abs", "rel_l2")
                if key in row
            )
            for row in metrics.values()
        )
    )


def validate_eager_gate(report, *, expected_sources, rejection):
    """Raise on incomplete, diagnostic, stale, or numerically failed gates."""
    config = report.get("config", {})
    if (
        report.get("stage") != "completed"
        or report.get("gate_passed") is not True
        or report.get("pools_flushed") is not True
        or report.get("formal_serving_changed") is not False
        or report.get("tolerances") != TOLERANCES
        or report.get("numerical_source_sha256") != expected_sources
        or config.get("stable_marlin_candidate") is not True
        or config.get("repeat_control") is not True
        or bool(config.get("rejection_fixture")) != rejection
        or config.get("mixed_graph_candidate")
        or any(config.get(flag) for flag in DIAGNOSTIC_FLAGS)
        or config.get("rounds") != 9
        or config.get("chunks") != list(EAGER_CHUNKS)
        or config.get("capacities") != list(EAGER_CAPACITIES)
    ):
        raise ValueError("Full source-matched stable Eager gate is required")
    cases = report.get("cases", [])
    hits = report.get("draft_graph_hits", [])
    if len(cases) != 9 or len(hits) != 9 or any(hit is not True for hit in hits):
        raise ValueError("Full nine-case native Draft Graph coverage is required")
    for turn, case in enumerate(cases):
        if (
            case.get("turn") != turn
            or case.get("p_tokens") != EAGER_CHUNKS[turn]
            or case.get("p_capacity") != EAGER_CAPACITIES[turn]
            or any(
                case.get(key) is not True
                for key in (
                    "acceptance_equal",
                    "native_acceptance_equal",
                    "native_verify_graph",
                    "safety_passed",
                )
            )
            or case.get("state_mismatches") != {}
            or not _state_matches(case.get("state_comparison"))
            or not _state_matches(case.get("repeat_control_metrics"))
            or not case.get("safety_checks")
            or not all(
                row and all(value is True for value in row.values())
                for row in case["safety_checks"]
            )
        ):
            raise ValueError(f"Eager case {turn} has incomplete or failed checks")
    digest = validate_proposals(report, rounds=9)
    expected = config.get("proposals_sha256")
    if expected is not None and digest != expected:
        raise ValueError("Eager gate proposals differ from its prepared input")
    return digest


def build_jobs(*, smoke, numerics, mixed_graph):
    if mixed_graph and (smoke or numerics != "stable"):
        raise ValueError("Mixed Graph requires full stable Eager validation")
    jobs = []
    for graph in [False, True] if mixed_graph else [False]:
        for reject in (False, True):
            jobs.append(
                dict(
                    nonce=("graph" if graph else "native")
                    + ("-rejection" if reject else "-natural"),
                    chunks=list(
                        GRAPH_CHUNKS
                        if graph
                        else EAGER_CHUNKS[:1]
                        if smoke
                        else EAGER_CHUNKS
                    ),
                    capacities=list(
                        GRAPH_CAPACITIES
                        if graph
                        else EAGER_CAPACITIES[:1]
                        if smoke
                        else EAGER_CAPACITIES
                    ),
                    rejection_fixture=reject,
                    freeze_token_local_diagnostic=False,
                    stable_marlin_candidate=numerics == "stable",
                    repeat_control=not graph,
                    mixed_graph_candidate=graph,
                )
            )
    if not mixed_graph:
        jobs.append(
            jobs[1]
            | dict(
                nonce="state-routing-reference",
                freeze_token_local_diagnostic=True,
                repeat_control=False,
            )
        )
    for job in jobs:
        job.update(
            rounds=len(job["chunks"]), rotate_rounds=[2, 6], continue_on_mismatch=True
        )
    return jobs
