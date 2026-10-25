"""Offline adaptive CUDA Graph selection on synthetic Qwen3.5 AWQ operators.

Train local costs, freeze them, then evaluate independent shapes/inputs/routes.
All real AWQ/FP16 candidates are checked against frozen repaired Separate.
No serving integration, checkpoint weights, acceptance or SSM state claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import bench_mixed_operators as base
import bench_mixed_reduction_tuning as tuning
import torch

from sglang.srt.speculative.mixed_selection import (
    ACTIONS,
    OfflineMixedSelector,
    RouteFeatures,
)
from sglang.srt.speculative.mixed_selection_graph import (
    GraphEntry,
    OfflineAdaptiveGraphs,
    route_features,
)

CONFIGS = dict(
    legacy=(tuning.OLD_TILE, 8), m16=("projection", 16), m32=("projection", 32)
)
OPERATORS = ("gdn_token_local_bundle", "attn_token_local_bundle")
TRAIN_CASES = (
    (23, 32),
    (32, 32),
    (65, 128),
    (128, 128),
    (257, 512),
    (430, 512),
    (512, 512),
)
TEST_CASES = (
    (16, 32),
    (31, 32),
    (96, 128),
    (127, 128),
    (300, 512),
    (448, 512),
    (511, 512),
    (512, 512),
)


class DirectReplay:
    def __init__(self, entry):
        self.entry = entry

    def replay(self):
        self.entry.graph.replay()


class AdaptiveReplay:
    def __init__(self, runner, context):
        self.runner, self.context = runner, context

    def replay(self):
        self.runner.replay(step=1, context=self.context)


def inputs(operators, n, profile):
    result = base.make_inputs(operators, n, "uniform")
    if profile.startswith("hot"):
        result["router_bias"][:, : int(profile[3:])] = 8
    return result


class Bank:
    def __init__(self, operators, operator, p, capacity):
        self.operators, self.operator, self.n, self.capacity = (
            operators,
            operator,
            p,
            capacity,
        )
        self.p, self.d = (
            inputs(operators, p, "uniform"),
            inputs(operators, 24, "uniform"),
        )
        self.entries = {action: self.capture(action) for action in ACTIONS}
        self.captures = len(self.entries)

    def update(self, profile, seed):
        torch.manual_seed(seed)
        for target, n in ((self.p, self.n), (self.d, 24)):
            source = inputs(self.operators, n, profile)
            for name in ("hidden", "out", "router_bias"):
                target[name].copy_(source[name])

    def capture(self, action, with_stats=True):
        name, layout = action.split("_")
        tile, block = CONFIGS[name]
        op, operators, p, d, n, capacity = (
            self.operator,
            self.operators,
            self.p,
            self.d,
            self.n,
            self.capacity,
        )

        def body():
            if layout == "separate":
                left, right = operators.run(op, p, True), operators.run(op, d, True)
                outputs = (left, right)
                stats = route_features(left[-2], right[-2]) if with_stats else None
            else:
                joined = operators.run(op, base.pack(p, d, capacity, op), True)
                outputs = base.split_output(joined, n, capacity)
                stats = (
                    route_features(joined[-2][:n], joined[-2][capacity:])
                    if with_stats
                    else None
                )
            return outputs, stats

        with tuning.variant(tile, block):
            holder = base.Captured([body], True)
        outputs, stats = holder.outputs[0]
        return GraphEntry(holder.graphs[0], outputs, stats, holder)

    def validate(self):
        for entry in self.entries.values():
            entry.graph.replay()
        torch.cuda.synchronize()
        reference = self.entries["legacy_separate"].outputs
        checks = {
            name: base.compare(reference, entry.outputs)
            for name, entry in self.entries.items()
        }
        stats = self.entries["legacy_separate"].stats.cpu().tolist()
        RouteFeatures(tuple(stats))
        assert all(c["passed"] for c in checks.values()), checks
        assert all(
            torch.equal(e.stats, self.entries["legacy_separate"].stats)
            for e in self.entries.values()
        )
        for entry in self.entries.values():
            entry.validated = True
        return checks, stats

    def lanes(self):
        return {name: DirectReplay(entry) for name, entry in self.entries.items()}


def cpu_overhead(policy, operator, p, capacity, features):
    route = RouteFeatures(tuple(features))
    policy.choose(operator, p, capacity, route)
    hot, cold = [], []
    for _ in range(12):
        start = time.perf_counter_ns()
        for _ in range(2000):
            policy.choose(operator, p, capacity, route)
        hot.append((time.perf_counter_ns() - start) / 2000 / 1000)
    for _ in range(100):
        policy.cache.clear()
        start = time.perf_counter_ns()
        policy.choose(operator, p, capacity, route)
        cold.append((time.perf_counter_ns() - start) / 1000)
    return dict(
        hot_us=statistics.median(hot),
        cold_p50_us=statistics.median(cold),
        cold_p95_us=base.percentile(cold, 0.95),
    )


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    spec = importlib.util.spec_from_file_location(
        "frozen_mixed_marlin", args.reference_linear
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tuning.LEGACY_LINEAR = module.fixed_linear
    torch.manual_seed(args.seed)
    operators = base.Operators(config["text_config"], True)
    root = Path(__file__).resolve().parents[3]
    names = [
        "python/sglang/srt/speculative/mixed_selection.py",
        "python/sglang/srt/speculative/mixed_selection_graph.py",
        "python/sglang/srt/speculative/mixed_marlin.py",
        "python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py",
        "test/manual/pdmux/bench_mixed_selection.py",
        "test/manual/pdmux/bench_mixed_operators.py",
        "test/manual/pdmux/bench_mixed_reduction_tuning.py",
    ]
    sources = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }
    context = dict(
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
        torch=torch.__version__,
        model="QuantTrio/Qwen3.5-35B-A3B-AWQ",
        tp=1,
        verify="D3xblock8",
        experts=256,
        topk=8,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        operator_source_sha256={name: sources[name] for name in names[:4]},
    )
    result = dict(
        stage="calibrating",
        scope="Offline synthetic operator graph selection; no complete model or native serving changes.",
        context=context,
        source_sha256=sources,
        seed=args.seed,
        tolerances=base.TOLERANCES,
        reference_linear_sha256=hashlib.sha256(
            args.reference_linear.read_bytes()
        ).hexdigest(),
        train=[],
        holdout=[],
        transitions=[],
        policy=None,
        captures=0,
    )
    path = args.output / "results.json"
    samples = []
    train_cases = ((32, 32), (512, 512)) if args.smoke else TRAIN_CASES
    test_cases = ((31, 32), (448, 512)) if args.smoke else TEST_CASES
    ops = OPERATORS[:1] if args.smoke else OPERATORS
    profiles = ("uniform", "hot32") if args.smoke else ("uniform", "hot32", "hot128")
    for op in ops:
        for p, capacity in train_cases:
            bank = Bank(operators, op, p, capacity)
            result["captures"] += bank.captures
            for profile in profiles:
                seed = args.seed + 1000 + len(samples)
                bank.update(profile, seed)
                checks, features = bank.validate()
                timings = tuning.measure(bank.lanes(), args.repeats, args.trials, seed)
                costs = {name: t["cuda_p50_us"] for name, t in timings.items()}
                samples.append(
                    dict(
                        operator=op,
                        P=p,
                        capacity=capacity,
                        features=features,
                        costs_us=costs,
                        correctness_passed=True,
                    )
                )
                result["train"].append(
                    dict(
                        operator=op,
                        P=p,
                        capacity=capacity,
                        profile=profile,
                        seed=seed,
                        features=features,
                        checks=checks,
                        timings=timings,
                    )
                )
                base.save(path, result)
                print(
                    "TRAIN",
                    op,
                    p,
                    capacity,
                    profile,
                    "effective",
                    round(features[3], 1),
                    "best",
                    min(costs, key=costs.get),
                    flush=True,
                )
            del bank
            gc.collect()
            torch.cuda.empty_cache()
    policy = OfflineMixedSelector(samples, context=context)
    encoded = policy.to_dict()
    base.save(args.output / "calibration.json", encoded)
    frozen = hashlib.sha256(json.dumps(encoded, sort_keys=True).encode()).hexdigest()
    policy = OfflineMixedSelector.from_dict(encoded, expected_context=context)
    result.update(stage="holdout", policy=encoded, calibration_digest=frozen)
    test_profiles = (
        ("uniform", "hot32") if args.smoke else ("uniform", "hot32", "hot64")
    )
    for op in ops:
        for p, capacity in test_cases:
            bank = Bank(operators, op, p, capacity)
            result["captures"] += bank.captures
            bank.validate()
            runner = OfflineAdaptiveGraphs(
                bank.entries, policy, operator=op, p=p, capacity=capacity
            )
            for profile in test_profiles:
                seed = args.seed + 10000 + len(result["holdout"])
                bank.update(profile, seed)
                checks, features = bank.validate()
                context_id = 100 + len(result["holdout"])
                cold = runner.replay(step=0, context=context_id)
                torch.cuda.synchronize()
                decision = runner.replay(step=1, context=context_id)
                torch.cuda.synchronize()
                adaptive_check = base.compare(
                    bank.entries["legacy_separate"].outputs,
                    bank.entries[decision.action].outputs,
                )
                assert adaptive_check["passed"]
                lanes = bank.lanes() | dict(adaptive=AdaptiveReplay(runner, context_id))
                timings = tuning.measure(lanes, args.repeats, args.eval_trials, seed)
                assert runner.current == decision.action
                costs = {name: t["cuda_p50_us"] for name, t in timings.items()}
                oracle = min(ACTIONS, key=costs.get)
                row = dict(
                    operator=op,
                    P=p,
                    capacity=capacity,
                    profile=profile,
                    seed=seed,
                    features=features,
                    checks=checks,
                    cold_action=cold.action,
                    selected=decision.action,
                    decision_reason=decision.reason,
                    adaptive_check=adaptive_check,
                    timings=timings,
                    oracle=oracle,
                    adaptive_over_oracle_pct=100
                    * (costs["adaptive"] / costs[oracle] - 1),
                    selected_over_oracle_pct=100
                    * (costs[decision.action] / costs[oracle] - 1),
                    cpu_selection=cpu_overhead(policy, op, p, capacity, features),
                    feedback=dict(
                        submitted=runner.mailbox.submitted,
                        completed=runner.mailbox.completed,
                        dropped=runner.mailbox.dropped,
                        invalid=runner.mailbox.invalid,
                    ),
                )
                if p in (31, 127, 512) and profile in ("uniform", "hot32"):
                    bare = bank.capture(decision.action, with_stats=False)
                    overhead_lanes = dict(
                        bare=DirectReplay(bare),
                        stats=DirectReplay(bank.entries[decision.action]),
                        adaptive=AdaptiveReplay(runner, context_id),
                    )
                    row["overhead_control"] = tuning.measure(
                        overhead_lanes, args.repeats, 6, seed + 1
                    )
                result["holdout"].append(row)
                assert (
                    hashlib.sha256(
                        json.dumps(policy.to_dict(), sort_keys=True).encode()
                    ).hexdigest()
                    == frozen
                )
                base.save(path, result)
                print(
                    "TEST",
                    op,
                    p,
                    capacity,
                    profile,
                    "selected",
                    decision.action,
                    "oracle",
                    oracle,
                    "regret",
                    round(row["adaptive_over_oracle_pct"], 2),
                    flush=True,
                )
            del runner, bank
            gc.collect()
            torch.cuda.empty_cache()
    # Real routing transitions use lagged feedback, not profile labels.
    for op in ops:
        bank = Bank(operators, op, 257, 512)
        result["captures"] += bank.captures
        bank.update("uniform", args.seed + 99999)
        bank.validate()
        runner = OfflineAdaptiveGraphs(
            bank.entries,
            policy,
            operator=op,
            p=257,
            capacity=512,
            sample_every=1,
            max_age=2,
        )
        for step, profile in enumerate(
            (
                "uniform",
                "uniform",
                "hot32",
                "hot32",
                "hot64",
                "hot64",
                "uniform",
                "uniform",
            )
        ):
            bank.update(profile, args.seed + 20000 + step)
            choice = runner.replay(step=step, context=1)
            torch.cuda.synchronize()
            bank.entries["legacy_separate"].graph.replay()
            torch.cuda.synchronize()
            check = base.compare(
                bank.entries["legacy_separate"].outputs,
                bank.entries[choice.action].outputs,
            )
            assert check["passed"]
            result["transitions"].append(
                dict(
                    operator=op,
                    step=step,
                    profile=profile,
                    selected=choice.action,
                    reason=choice.reason,
                    check=check,
                )
            )
        del runner, bank
        gc.collect()
        torch.cuda.empty_cache()
    result["stage"] = "completed"
    result["evaluation_costs_updated_policy"] = False
    base.save(path, result)
    print(
        "COMPLETE",
        len(result["train"]),
        len(result["holdout"]),
        len(result["transitions"]),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-linear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--eval-trials", type=int, default=14)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.repeats >= 16 and args.trials >= 6 and args.eval_trials >= 7
    run(args)


if __name__ == "__main__":
    main()
