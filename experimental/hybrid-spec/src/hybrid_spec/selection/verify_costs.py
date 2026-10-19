"""Portable cost-table format shared by offline calibration and online verify.

Only CUDA Graph verify-stage costs and a shared fixed term are persisted.
Acceptance labels remain worker-local. Profiles must come from the same model,
GPU and TP setup.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path


def cost_config(config):
    # Scheduling/learning knobs may change between calibration and serving.
    values = {
        k: v
        for k, v in asdict(config).items()
        if not k.startswith("dynamic_verify")
        # Preserve tables produced before this opt-in was introduced.
        and (k != "conditional_draft_graph" or v)
        and (k != "conditional_draft_graph_dispatch" or v != "if")
        and (k != "conditional_draft_graph_joined" or v)
    }
    if config.conditional_draft_graph_dispatch == "switch":
        # The fused setter and compaction remove per-round kernels. A table
        # collected with the original SWITCH must not seed these fixed costs.
        values["conditional_draft_graph_revision"] = 2
    return json.loads(json.dumps(values))


def cost_profile(config, costs, draft_costs, runtime=None):
    return {
        "version": 2,
        "scope": "cuda_graph_verify_stage",
        "runtime": runtime or {},
        "config": cost_config(config),
        "fixed_ms": {f"{bs}:{ctx}": v for (bs, ctx), v in draft_costs.items()},
        "verify_ms": {
            f"{bs}:{ctx}": {str(n): v[1] for n, v in rows.items()}
            for (bs, ctx), rows in costs.items()
        },
    }


def parse_cost_profile(profile, config, runtime=None):
    if profile.get("version") != 2 or profile.get("scope") != "cuda_graph_verify_stage":
        raise ValueError("unsupported dynamic verify cost profile version/scope")
    if profile.get("config") != cost_config(config):
        raise ValueError("dynamic verify cost profile does not match hybrid config")
    if profile.get("runtime") != (runtime or {}):
        raise ValueError(
            "dynamic verify cost profile does not match model/GPU/TP/backend"
        )

    def milliseconds(value, *, positive=False):
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            or (positive and value == 0)
            or value > 1e30
        ):
            raise ValueError("dynamic verify costs must be finite milliseconds")
        return float(value)

    costs, drafts = {}, {}
    for shape, rows in profile["verify_ms"].items():
        bs, ctx = (int(v) for v in shape.split(":"))
        if bs < 1 or ctx < 512 or ctx & (ctx - 1) or not rows:
            raise ValueError("invalid dynamic verify batch/context bucket")
        key = (bs, ctx)
        drafts[key] = milliseconds(profile["fixed_ms"][shape])
        costs[key] = {}
        for size, verify_ms in rows.items():
            n = int(size)
            if n < bs:
                raise ValueError("invalid dynamic verify physical graph cost")
            costs[key][n] = (
                drafts[key],
                milliseconds(verify_ms, positive=True),
            )
    if not costs:
        raise ValueError("dynamic verify cost profile is empty")
    return costs, drafts


def load_cost_profile(path, config, runtime=None):
    try:
        return parse_cost_profile(json.loads(Path(path).read_text()), config, runtime)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"invalid dynamic verify cost profile {path}: {exc}") from exc
