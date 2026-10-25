"""Offline fixed-reduction tile sweep; real CUDA operators, synthetic weights.

References the previous repaired M16/N32/K64 FP16 and full-K Marlin M8 path.
Every variant captures Separate and packed Mixed graphs, compares both to the
old repaired reference, and checks integer router IDs exactly. No model load.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
from functools import partial
from pathlib import Path

import bench_mixed_operators as bench
import torch
import triton

from sglang.srt.speculative.mixed_marlin import fixed_linear

OLD_TILE = (16, 32, 4)
LEGACY_LINEAR = fixed_linear


@contextlib.contextmanager
def variant(tile, block):
    original_arithmetic = bench.arithmetic
    original_linear = bench.fixed_linear

    @contextlib.contextmanager
    def arithmetic(stable):
        original_fused = bench.MARLIN.fused_marlin_moe
        with original_arithmetic(stable):
            if stable:
                bench.MARLIN.fused_marlin_moe = partial(
                    original_fused, deterministic_block_size_m=block
                )
            try:
                yield
            finally:
                bench.MARLIN.fused_marlin_moe = original_fused

    bench.arithmetic = arithmetic
    bench.fixed_linear = (
        LEGACY_LINEAR if tile == OLD_TILE else partial(fixed_linear, tile=tile)
    )
    try:
        yield
    finally:
        bench.arithmetic = original_arithmetic
        bench.fixed_linear = original_linear


def capture_pair(operators, op, p, d, capacity, stable):
    separate = bench.Captured(
        [lambda: (operators.run(op, p, stable), operators.run(op, d, stable))],
        stable,
    )
    mixed = bench.Captured(
        [
            lambda: bench.split_output(
                operators.run(op, bench.pack(p, d, capacity, op), stable),
                len(p["hidden"]),
                capacity,
            )
        ],
        stable,
    )
    return separate, mixed


def measure(lanes, repeats, trials, seed):
    old = bench.LANES
    bench.LANES = tuple(lanes)
    try:
        return bench.measure(lanes, repeats, trials, seed)
    finally:
        bench.LANES = old


@torch.no_grad()
def run(args):
    global LEGACY_LINEAR
    if args.reference_linear:
        spec = importlib.util.spec_from_file_location(
            "frozen_mixed_marlin", args.reference_linear
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        LEGACY_LINEAR = module.fixed_linear
    raw_config = json.loads(args.config.read_text())
    config = raw_config["text_config"]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    operators = bench.Operators(config, args.stage != "linear")
    tile = (
        "projection"
        if args.linear_tile == "projection"
        else tuple(map(int, args.linear_tile.split(",")))
    )
    if args.stage == "linear":
        variants = [
            (f"fp16_{m}_{n}_{w}", (m, n, w), 8)
            for m, n, w in (
                (16, 64, 4),
                (16, 128, 4),
                (32, 64, 4),
                (32, 128, 4),
                (64, 64, 4),
                (64, 128, 4),
                (32, 128, 8),
            )
        ]
    elif args.stage == "moe":
        variants = [(f"marlin_m{m}", OLD_TILE, m) for m in (16, 32, 48, 64)]
    else:
        variants = [
            (
                (f"combined_m{args.control_block}_control", tile, args.control_block)
                if args.control_block
                else ("fp16_only", tile, 8)
            ),
            ("marlin_only", OLD_TILE, args.moe_block),
            ("optimized", tile, args.moe_block),
        ]
    cases = [tuple(map(int, pair.split(":"))) for pair in args.cases.split(",")]
    ops = args.operators.split(",")
    root = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__).resolve(),
        Path(bench.__file__),
        root / "python/sglang/srt/speculative/mixed_marlin.py",
        root / "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py",
        root / "python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h",
        root / "python/sglang/kernels/jit/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh",
    ]
    result = dict(
        stage="running",
        experiment=args.stage,
        seed=args.seed,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        triton=triton.__version__,
        gpu=torch.cuda.get_device_name(0),
        trials=args.trials,
        repeats=args.repeats,
        tolerances=bench.TOLERANCES,
        variants=variants,
        rows=[],
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        synthetic_weights=True,
        full_model_tested=False,
        reference_linear_sha256=(
            hashlib.sha256(args.reference_linear.read_bytes()).hexdigest()
            if args.reference_linear
            else None
        ),
        note="All lanes one CUDA Graph; Mixed includes packing/padding. CUDA p95 across trial means, not token output intervals. Old repaired is numerical reference; native is performance control. Native Mixed is not a supported candidate.",
    )
    for op in ops:
        profiles = ["uniform"] if args.stage == "linear" else args.profiles.split(",")
        for profile in profiles:
            d = bench.make_inputs(operators, 24, profile)
            for n, capacity in cases:
                print("START", args.stage, op, profile, n, capacity, flush=True)
                p = bench.make_inputs(operators, n, profile)
                with variant(OLD_TILE, 8):
                    old_sep, old_mix = capture_pair(operators, op, p, d, capacity, True)
                    native_sep = bench.Captured(
                        [
                            lambda op=op, p=p, d=d: (
                                operators.run(op, p, False),
                                operators.run(op, d, False),
                            )
                        ],
                        False,
                    )
                lanes = dict(
                    native_separate=native_sep, old_separate=old_sep, old_mixed=old_mix
                )
                for name, cfg_tile, block in variants:
                    with variant(cfg_tile, block):
                        separate, mixed = capture_pair(
                            operators, op, p, d, capacity, True
                        )
                    lanes[name + "_separate"] = separate
                    lanes[name + "_mixed"] = mixed
                for lane in lanes.values():
                    lane.replay()
                torch.cuda.synchronize()
                reference = old_sep.outputs[0]
                checks = {
                    name: bench.compare(reference, lane.outputs[0])
                    for name, lane in lanes.items()
                }
                cross = {
                    name: bench.compare(
                        lanes[name + "_separate"].outputs[0],
                        lanes[name + "_mixed"].outputs[0],
                    )
                    for name, _, _ in variants
                }
                audits = {}
                if args.stage != "linear":
                    for name, cfg_tile, block in [("old", OLD_TILE, 8)] + variants:
                        with variant(cfg_tile, block):
                            audits[name] = bench.audit_marlin(
                                operators, op, p, d, capacity
                            )
                passed = {
                    name: checks[name + "_separate"]["passed"]
                    and checks[name + "_mixed"]["passed"]
                    and cross[name]["passed"]
                    for name, _, _ in variants
                }
                timings = measure(
                    lanes, args.repeats, args.trials, args.seed + len(result["rows"])
                )
                row = dict(
                    operator=op,
                    profile=profile,
                    P=n,
                    P_capacity=capacity,
                    checks=checks,
                    separate_vs_mixed=cross,
                    variant_passed=passed,
                    timings=timings,
                    marlin_dispatch=audits,
                    gpu_after=bench.gpu_status(),
                )
                result["rows"].append(row)
                bench.save(args.output / "results.json", result)
                print(
                    "DONE",
                    op,
                    profile,
                    n,
                    "passed",
                    passed,
                    "us",
                    {name: round(t["cuda_p50_us"], 2) for name, t in timings.items()},
                    flush=True,
                )
                del lanes, old_sep, old_mix, native_sep, separate, mixed, reference, p
                gc.collect()
                torch.cuda.empty_cache()
    result["stage"] = "completed"
    result["old_mixed_checks_passed"] = all(
        row["checks"]["old_mixed"]["passed"] for row in result["rows"]
    )
    result["optimized_checks_passed"] = (
        all(row["variant_passed"]["optimized"] for row in result["rows"])
        if args.stage == "validate"
        else None
    )
    bench.save(args.output / "results.json", result)
    if (
        not result["old_mixed_checks_passed"]
        or result["optimized_checks_passed"] is False
    ):
        raise RuntimeError("Fixed-reduction validation failed; inspect results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("linear", "moe", "validate"), required=True)
    parser.add_argument("--cases", default="512:512")
    parser.add_argument(
        "--operators", default="linear_qkvz,linear_out,linear_router,shared_expert"
    )
    parser.add_argument("--profiles", default="uniform,hot32")
    parser.add_argument("--linear-tile", default="32,128,4")
    parser.add_argument("--moe-block", type=int, default=32)
    parser.add_argument("--control-block", type=int, choices=(8, 16, 32, 48, 64))
    parser.add_argument("--reference-linear", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    assert args.trials >= 6 and args.repeats >= 16
    run(args)


if __name__ == "__main__":
    main()
