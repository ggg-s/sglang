"""Operator A/B for native Separate, repaired Separate and packed Mixed Graph.

Uses real checkpoint dimensions with synthetic FP16/uint4 weights and inputs.
No model load, Attention/GDN recurrence, draft/acceptance, logits or scheduler.
All primary lanes use one CUDA Graph: launch-count savings are a separate
two-graph control. Packing is included in the primary mixed lane.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.srt.layers.moe.topk import fused_topk
from sglang.srt.speculative.mixed_marlin import fixed_linear
from sglang.test.test_marlin_utils import awq_marlin_quantize

MARLIN = importlib.import_module(
    "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
)
TOLERANCES = dict(rtol=0.02, atol=0.002)
LANES = (
    "native_separate_one_graph",
    "repaired_separate_one_graph",
    "mixed_graph_with_pack",
    "native_separate_two_graphs",
    "repaired_separate_two_graphs",
    "mixed_graph_prepacked",
)


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def gpu_status():
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,clocks.sm,power.draw,memory.used",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()


@contextlib.contextmanager
def arithmetic(stable):
    old = MARLIN.is_batch_invariant_mode_enabled
    MARLIN.is_batch_invariant_mode_enabled = lambda: stable
    try:
        yield
    finally:
        MARLIN.is_batch_invariant_mode_enabled = old


def quantized_experts(config):
    """Same AWQ packing helper as native tests; setup is never timed."""
    from sgl_kernel.scalar_type import scalar_types

    experts, k, n = (
        config["num_experts"],
        config["hidden_size"],
        config["moe_intermediate_size"],
    )
    banks = []
    for input_size, output_size in ((k, 2 * n), (n, k)):
        packed, scales, zeros = [], [], []
        for _ in range(experts):
            weight = (
                torch.randn(input_size, output_size, device="cuda", dtype=torch.float16)
                * 0.02
            )
            reference, q, s, z = awq_marlin_quantize(weight, scalar_types.uint4, 128)
            packed.append(q)
            scales.append(s)
            zeros.append(z)
            del reference, weight
        banks.append(
            tuple(
                torch.stack(values).contiguous() for values in (packed, scales, zeros)
            )
        )
    return banks


class Operators:
    def __init__(self, config, need_moe):
        self.config = config
        k, n = config["hidden_size"], config["shared_expert_intermediate_size"]
        v = config["linear_num_value_heads"] * config["linear_value_head_dim"]
        key = config["linear_num_key_heads"] * config["linear_key_head_dim"]
        q = config["num_attention_heads"] * config["head_dim"]
        kv = config["num_key_value_heads"] * config["head_dim"]
        self.shapes = dict(
            qkvz=(k, 2 * key + 2 * v),
            ba=(k, 2 * config["linear_num_value_heads"]),
            out=(v, k),
            qkv=(k, 2 * q + 2 * kv),
            attn_out=(q, k),
            router=(k, config["num_experts"]),
            shared_up=(k, 2 * n),
            shared_down=(n, k),
            shared_gate=(k, 1),
        )
        self.weights = {
            name: torch.randn(o, i, device="cuda", dtype=torch.float16) * 0.02
            for name, (i, o) in self.shapes.items()
        }
        self.banks = quantized_experts(config) if need_moe else None

    def linear(self, name, x, stable):
        return (
            fixed_linear(x, self.weights[name])
            if stable
            else F.linear(x, self.weights[name])
        )

    def run(self, op, inputs, stable):
        x = inputs.get("hidden")
        if op.startswith("linear_"):
            name = op.removeprefix("linear_")
            return (self.linear(name, inputs["out"] if name == "out" else x, stable),)
        if op == "shared_expert":
            activated = silu_and_mul(self.linear("shared_up", x, stable))
            # This nn.Linear gate is unchanged by the offline numerical adapter.
            gate = F.linear(x, self.weights["shared_gate"]).sigmoid()
            return (self.linear("shared_down", activated, stable) * gate,)
        if op == "routed_moe":
            logits, ids, weights = inputs["logits"], inputs["ids"], inputs["scores"]
        else:
            logits = self.linear("router", x, stable) + inputs["router_bias"]
            weights, ids = fused_topk(
                x, logits, self.config["num_experts_per_tok"], True
            )
        (w1, s1, z1), (w2, s2, z2) = self.banks
        routed = MARLIN.fused_marlin_moe(
            x,
            w1,
            w2,
            s1,
            s2,
            logits,
            weights,
            ids,
            w1_zeros=z1,
            w2_zeros=z2,
            num_bits=4,
        )
        if op == "routed_moe":
            return (routed,)
        shared = self.run("shared_expert", inputs, stable)[0]
        names = ("qkvz", "ba") if op == "gdn_token_local_bundle" else ("qkv",)
        outputs = tuple(self.linear(name, x, stable) for name in names)
        out_name = "out" if op == "gdn_token_local_bundle" else "attn_out"
        return outputs + (
            self.linear(out_name, inputs["out"], stable),
            routed + shared,
            ids,
            weights,
        )


def make_inputs(operators, n, profile):
    c = operators.config
    inputs = dict(
        hidden=torch.randn(n, c["hidden_size"], device="cuda", dtype=torch.float16),
        out=torch.randn(
            n,
            c["linear_num_value_heads"] * c["linear_value_head_dim"],
            device="cuda",
            dtype=torch.float16,
        ),
    )
    bias = torch.zeros(1, c["num_experts"], device="cuda", dtype=torch.float16)
    if profile == "hot32":
        bias[:, :32] = 8
    logits = F.linear(inputs["hidden"], operators.weights["router"]) + bias
    scores, ids = fused_topk(inputs["hidden"], logits, c["num_experts_per_tok"], True)
    return inputs | dict(logits=logits, ids=ids, scores=scores, router_bias=bias)


def pack(p, d, capacity, op):
    padding = capacity - len(p["hidden"])
    joined = {}
    names = (
        ("out",)
        if op == "linear_out"
        else ("hidden",)
        if op.startswith("linear_") or op == "shared_expert"
        else ("hidden", "logits", "ids", "scores")
        if op == "routed_moe"
        else ("hidden", "out", "router_bias")
    )
    for name, value in p.items():
        if name not in names:
            continue
        if name == "router_bias":
            joined[name] = value
        else:
            pad = value.new_zeros((padding,) + value.shape[1:])
            joined[name] = torch.cat((value, pad, d[name]))
    return joined


class Captured:
    def __init__(self, functions, stable):
        self.graphs, self.outputs = [], []
        self.functions = functions  # Retain external graph input storage.
        with arithmetic(stable):
            # Initialize JIT and cuBLAS before capturing real operations.
            for fn in functions:
                for _ in range(3):
                    fn()
            torch.cuda.synchronize()
            for fn in functions:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn()
                self.graphs.append(graph)
                self.outputs.append(output)

    def replay(self):
        for graph in self.graphs:
            graph.replay()


def flatten_outputs(outputs):
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    return [tensor for item in outputs for tensor in flatten_outputs(item)]


def compare(reference, candidate):
    a, b = flatten_outputs(reference), flatten_outputs(candidate)
    assert len(a) == len(b)
    rows = []
    for expected, actual in zip(a, b):
        assert expected.shape == actual.shape
        floating = expected.is_floating_point()
        mismatch = (
            int((~torch.isclose(expected, actual, **TOLERANCES)).sum())
            if floating
            else int((expected != actual).sum())
        )
        rows.append(
            dict(
                shape=list(expected.shape),
                mismatch=mismatch,
                max_abs=float((expected.float() - actual.float()).abs().max()),
                floating=floating,
            )
        )
    return dict(passed=all(r["mismatch"] == 0 for r in rows), fields=rows)


def split_output(output, n, capacity):
    return (tuple(t[:n] for t in output), tuple(t[capacity:] for t in output))


def capture_lanes(operators, op, p, d, capacity):
    joined = pack(p, d, capacity, op)

    def separate(stable):
        return lambda: (operators.run(op, p, stable), operators.run(op, d, stable))

    def mixed(with_pack):
        return lambda: split_output(
            operators.run(op, pack(p, d, capacity, op) if with_pack else joined, True),
            len(p["hidden"]),
            capacity,
        )

    lanes = {}
    for stable, name in (
        (False, "native_separate_one_graph"),
        (True, "repaired_separate_one_graph"),
    ):
        lanes[name] = Captured([separate(stable)], stable)
    lanes["mixed_graph_with_pack"] = Captured([mixed(True)], True)
    for stable, name in (
        (False, "native_separate_two_graphs"),
        (True, "repaired_separate_two_graphs"),
    ):
        lanes[name] = Captured(
            [
                lambda stable=stable: operators.run(op, p, stable),
                lambda stable=stable: operators.run(op, d, stable),
            ],
            stable,
        )
    lanes["mixed_graph_prepacked"] = Captured([mixed(False)], True)
    for lane in lanes.values():
        lane.replay()
    torch.cuda.synchronize()
    reference = lanes["repaired_separate_one_graph"].outputs[0]
    checks = {}
    for name, lane in lanes.items():
        output = lane.outputs if len(lane.graphs) == 2 else lane.outputs[0]
        checks[name] = compare(reference, output)
    assert all(checks[n]["passed"] for n in LANES if not n.startswith("native")), checks
    return lanes, checks


def audit_marlin(operators, op, p, d, capacity):
    calls = []
    original = MARLIN.moe_wna16_marlin_gemm

    def audited(*args, **kwargs):
        calls.append(
            {
                key: kwargs[key]
                for key in (
                    "size_m",
                    "size_n",
                    "size_k",
                    "moe_block_size",
                    "use_atomic_add",
                    "use_full_k",
                    "use_fp32_reduce",
                )
            }
        )
        return original(*args, **kwargs)

    MARLIN.moe_wna16_marlin_gemm = audited
    try:
        result = {}
        for stable, name in (
            (False, "native_separate"),
            (True, "repaired_separate"),
            (True, "mixed"),
        ):
            calls.clear()
            with arithmetic(stable):
                if name == "mixed":
                    operators.run(op, pack(p, d, capacity, op), stable)
                else:
                    operators.run(op, p, stable)
                    operators.run(op, d, stable)
            torch.cuda.synchronize()
            result[name] = list(calls)
        return result
    finally:
        MARLIN.moe_wna16_marlin_gemm = original


def percentile(values, q):
    ordered = sorted(values)
    i = (len(ordered) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (i - lo)


def measure(lanes, repeats, trials, seed):
    rng = random.Random(seed)
    base = list(LANES)
    rng.shuffle(base)
    samples = {name: [] for name in LANES}
    # Hold the GPU busy before sampling so the first lane does not own clock ramp-up.
    start = time.perf_counter()
    while time.perf_counter() - start < 0.3:
        for lane in lanes.values():
            for _ in range(3):
                lane.replay()
        torch.cuda.synchronize()
    for trial in range(trials):
        order = base[trial % len(base) :] + base[: trial % len(base)]
        if (trial // len(base)) % 2:
            order.reverse()
        for name in order:
            lane = lanes[name]
            for _ in range(4):
                lane.replay()
            torch.cuda.synchronize()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start = time.perf_counter()
            begin.record()
            submit_start = time.perf_counter()
            for _ in range(repeats):
                lane.replay()
            submit_end = time.perf_counter()
            end.record()
            end.synchronize()
            samples[name].append(
                dict(
                    trial=trial,
                    cuda_us=begin.elapsed_time(end) * 1000 / repeats,
                    wall_us=(time.perf_counter() - start) * 1e6 / repeats,
                    submit_us=(submit_end - submit_start) * 1e6 / repeats,
                )
            )
    return {
        name: dict(
            cuda_p50_us=statistics.median(r["cuda_us"] for r in rows),
            cuda_p95_trial_mean_us=percentile([r["cuda_us"] for r in rows], 0.95),
            wall_p50_us=statistics.median(r["wall_us"] for r in rows),
            submit_p50_us=statistics.median(r["submit_us"] for r in rows),
            samples=rows,
        )
        for name, rows in samples.items()
    }


def route_summary(p, d):
    left = set(p["ids"].flatten().cpu().tolist())
    right = set(d["ids"].flatten().cpu().tolist())
    return dict(
        P_active_experts=len(left),
        D_active_experts=len(right),
        shared_experts=len(left & right),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="23:32,32:32,65:128,128:128,512:512")
    parser.add_argument(
        "--operators",
        default="linear_qkvz,linear_qkv,linear_out,linear_router,shared_expert,routed_moe,gdn_token_local_bundle,attn_token_local_bundle",
    )
    parser.add_argument("--profiles", default="uniform,hot32")
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    assert args.trials >= 6 and args.repeats >= 16
    raw = json.loads(args.config.read_text())
    config = raw["text_config"]
    assert (
        config["hidden_size"] == 2048
        and config["num_experts"] == 256
        and config["num_experts_per_tok"] == 8
    )
    assert (
        raw["quantization_config"]["bits"] == 4
        and raw["quantization_config"]["group_size"] == 128
    )
    args.output.mkdir(parents=True, exist_ok=False)
    operators_to_run = args.operators.split(",")
    cases = [tuple(map(int, x.split(":"))) for x in args.cases.split(",")]
    assert all(
        0 < n <= capacity and capacity in (32, 128, 512) for n, capacity in cases
    )
    torch.manual_seed(42)
    operators = Operators(
        config,
        any(
            op in ("routed_moe", "gdn_token_local_bundle", "attn_token_local_bundle")
            for op in operators_to_run
        ),
    )
    root = Path(__file__).resolve().parents[3]
    sources = [
        Path(__file__).resolve(),
        root / "python/sglang/srt/speculative/mixed_marlin.py",
        root / "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py",
        root / "python/sglang/test/test_marlin_utils.py",
    ]
    result = dict(
        stage="running",
        scope="Synthetic operator microbenchmark, not complete MixedTargetGraph or model throughput.",
        model_config=raw,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        tolerances=TOLERANCES,
        seed=42,
        repeats=args.repeats,
        trials=args.trials,
        synthetic_weights=True,
        checkpoint_weights_loaded=False,
        rows=[],
        lanes=list(LANES),
        timing_note="p95 is across trial mean times, not request/token p95. Includes allocations recorded into graph and primary mixed packing. Excludes capture/JIT/quantization and input refresh; weights/inputs are shared across lanes.",
        exclusions=[
            "Attention/GDN stateful computation and state commits",
            "DFlash draft/acceptance/logits",
            "norms/embedding/scheduler/CPU planning",
            "first unquantized MoE layer",
            "real checkpoint routing distributions",
        ],
    )
    save(args.output / "results.json", result)
    for op in operators_to_run:
        profiles = (
            args.profiles.split(",")
            if op in ("routed_moe", "gdn_token_local_bundle", "attn_token_local_bundle")
            else ["uniform"]
        )
        for profile in profiles:
            # Reuse the same D inputs across P sizes in this operator/profile.
            d = make_inputs(operators, 24, profile)
            for n, capacity in cases:
                print("START", op, profile, n, capacity, flush=True)
                p = make_inputs(operators, n, profile)
                lanes, checks = capture_lanes(operators, op, p, d, capacity)
                audit = (
                    audit_marlin(operators, op, p, d, capacity)
                    if op
                    in (
                        "routed_moe",
                        "gdn_token_local_bundle",
                        "attn_token_local_bundle",
                    )
                    else None
                )
                status_before = gpu_status()
                timings = measure(
                    lanes, args.repeats, args.trials, len(result["rows"]) + 42
                )
                native = timings[LANES[0]]["cuda_p50_us"]
                stable = timings[LANES[1]]["cuda_p50_us"]
                mixed = timings[LANES[2]]["cuda_p50_us"]
                penalty, saving = stable - native, stable - mixed
                row = dict(
                    operator=op,
                    profile=profile,
                    P=n,
                    P_capacity=capacity,
                    D_requests=3,
                    block=8,
                    timings=timings,
                    correctness=checks,
                    marlin_dispatch=audit,
                    route_summary=route_summary(p, d),
                    gpu_before=status_before,
                    gpu_after=gpu_status(),
                    fixed_reduction_penalty_us=penalty,
                    mixed_saving_vs_repaired_us=saving,
                    repaired_over_native_pct=100 * (stable / native - 1),
                    mixed_over_native_pct=100 * (mixed / native - 1),
                    penalty_recovery_fraction=saving / penalty if penalty > 0 else None,
                    penalty_offset=mixed <= native,
                )
                result["rows"].append(row)
                save(args.output / "results.json", result)
                print(
                    "DONE",
                    op,
                    profile,
                    n,
                    dict(
                        native_us=round(native, 2),
                        repaired_us=round(stable, 2),
                        mixed_us=round(mixed, 2),
                        net_pct=round(row["mixed_over_native_pct"], 2),
                    ),
                    flush=True,
                )
                del lanes, p
                gc.collect()
                torch.cuda.empty_cache()
    result["stage"] = "completed"
    save(args.output / "results.json", result)


if __name__ == "__main__":
    main()
