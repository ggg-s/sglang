"""Offline identical-input AWQ/Marlin and MoE routing diagnostics.

No baseline flag is changed. Explicit fixed-reduction microbenchmarks are
diagnostic controls, never model correctness or graph eligibility evidence.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager

import torch


def cpu(x):
    return x.detach().to("cpu", copy=True)


@contextmanager
def audit_marlin():
    module = importlib.import_module(
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
    )
    original = module.moe_wna16_marlin_gemm
    calls = []

    def wrapped(*args, **kwargs):
        calls.append(
            {
                k: kwargs.get(k)
                for k in [
                    "size_m",
                    "size_n",
                    "size_k",
                    "moe_block_size",
                    "use_atomic_add",
                    "use_fp32_reduce",
                    "use_full_k",
                ]
            }
        )
        return original(*args, **kwargs)

    module.moe_wna16_marlin_gemm = wrapped
    try:
        yield calls
    finally:
        module.moe_wna16_marlin_gemm = original


def metrics(a, b):
    if not a.is_floating_point():
        row = dict(mismatch=int((a != b).sum()), elements=a.numel())
        if a.ndim == 2:
            row["different_expert_set_rows"] = int(
                (a.sort(-1).values != b.sort(-1).values).any(-1).sum()
            )
        return row
    e = (a.float() - b.float()).abs()
    return dict(
        max_abs=float(e.max()),
        rel_l2=float(
            torch.linalg.vector_norm(e)
            / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
        ),
        mismatch=int((~torch.isclose(a, b, rtol=0.02, atol=0.002)).sum()),
        elements=a.numel(),
        bit_differences=int((a != b).sum()),
    )


@contextmanager
def capture(model, layers):
    values, handles = {}, []
    suffixes = (
        ".mlp",
        ".mlp.gate",
        ".mlp.topk",
        ".mlp.experts",
        ".in_proj_qkvz",
        ".in_proj_ba",
        ".qkv_proj",
        ".out_proj",
        ".o_proj",
    )
    for name, module in model.named_modules():
        if not name.endswith(suffixes) or not any(
            f".layers.{i}." in name for i in layers
        ):
            continue

        def hook(owner, args, output, name=name):
            row = dict(input=cpu(args[0]))
            if hasattr(output, "topk_ids"):
                row.update(
                    ids=cpu(output.topk_ids),
                    weights=cpu(output.topk_weights),
                    logits=cpu(output.router_logits),
                )
            else:
                value = output if isinstance(output, torch.Tensor) else output[0]
                row["output"] = cpu(value)
            values.setdefault(name, []).append(row)

        handles.append(module.register_forward_hook(hook))
    try:
        yield values
    finally:
        for h in handles:
            h.remove()


def compare(first, second, capacity=None):
    result = {}
    for name, rows in first.items():
        other = second[name]
        regions = []
        if capacity is None:
            regions = [("P", rows[0], other[0]), ("V", rows[1], other[1])]
        else:
            for label, a, offset in [("P", rows[0], 0), ("V", rows[1], capacity)]:
                regions.append(
                    (
                        label,
                        a,
                        {
                            k: v[offset : offset + len(a[k])]
                            for k, v in other[0].items()
                        },
                    )
                )
        result[name] = {
            label: {k: metrics(a[k], b[k]) for k in a} for label, a, b in regions
        }
    return result


@contextmanager
def fp32_router_diagnostic(model):
    hooks = []
    for name, module in model.named_modules():
        if not name.endswith(".mlp.gate"):
            continue

        def hook(module, args, output):
            y = torch.nn.functional.linear(args[0].float(), module.weight.float())
            return (y, None)

        hooks.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def linear_micro(model, reference, capacity, fixed_context):
    from contextlib import nullcontext

    modules = dict(model.named_modules())
    results = {}
    for name, rows in reference.items():
        if ".mlp" in name or len(rows) != 2:
            continue
        module = modules[name]
        p, v = [r["input"].cuda() for r in rows]
        joint = torch.cat(
            [
                p,
                torch.zeros(
                    (capacity - len(p), p.shape[1]), dtype=p.dtype, device=p.device
                ),
                v,
            ]
        )
        lanes = {}
        for fixed in [False, True]:
            with fixed_context() if fixed else nullcontext():
                parts, joined = [], []
                for _ in range(3):
                    current = []
                    for x in [p, v]:
                        y = module(x)
                        current.append(
                            dict(output=cpu(y if isinstance(y, torch.Tensor) else y[0]))
                        )
                    parts.append(current)
                    y = module(joint)
                    joined.append(
                        [dict(output=cpu(y if isinstance(y, torch.Tensor) else y[0]))]
                    )
                lanes["fixed_reductions" if fixed else "original"] = dict(
                    repeat=pair_metrics(parts[0], parts[1], len(p)),
                    mixed=pair_metrics(parts[0], joined[0], len(p), capacity),
                )
        results[name] = lanes
    return results


@contextmanager
def audit_moe(mlp):
    values = {}
    hooks = []
    for name in ["gate", "topk", "experts", "shared_expert"]:
        module = getattr(mlp, name, None)
        if module is None:
            continue

        def hook(owner, args, output, name=name):
            if hasattr(output, "topk_ids"):
                values["ids"], values["weights"] = (
                    cpu(output.topk_ids),
                    cpu(output.topk_weights),
                )
            else:
                value = output if isinstance(output, torch.Tensor) else output[0]
                values[name] = cpu(value)

        hooks.append(module.register_forward_hook(hook))
    try:
        yield values
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def frozen_routes(mlp, reference, p, capacity):
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    def hook(module, args, out):
        ids, weights = out.topk_ids.clone(), out.topk_weights.clone()
        for source, start, n in [(reference[0], 0, p), (reference[1], capacity, 24)]:
            ids[start : start + n] = source["ids"].to(ids.device)
            weights[start : start + n] = source["weights"].to(weights.device)
        return StandardTopKOutput(
            topk_ids=ids, topk_weights=weights, router_logits=out.router_logits
        )

    h = mlp.topk.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def pair_metrics(a, b, p, capacity=None):
    result = {}
    for label, i, offset in [("P", 0, 0), ("V", 1, capacity)]:
        left = a[i]
        right = (
            b[i]
            if capacity is None
            else {k: v[offset : offset + len(left[k])] for k, v in b[0].items()}
        )
        result[label] = {k: metrics(left[k], right[k]) for k in left}
    return result


@torch.no_grad()
def micro(model, reference, capacity, fixed_context):
    modules = dict(model.named_modules())
    result = {}
    for name, rows in reference.items():
        if not name.endswith(".mlp") or len(rows) != 2:
            continue
        mlp = modules[name]
        p, v = [r["input"].cuda() for r in rows]
        joint = torch.cat(
            [
                p,
                torch.zeros(
                    (capacity - len(p), p.shape[1]), dtype=p.dtype, device=p.device
                ),
                v,
            ]
        )
        lanes = {}
        for fixed in [False, True]:
            from contextlib import nullcontext

            with fixed_context() if fixed else nullcontext():
                separate, mixed = [], []
                for _ in range(3):
                    current = []
                    for x in [p, v]:
                        with audit_moe(mlp) as trace:
                            output = mlp(x.clone())
                            trace["output"] = cpu(output)
                        current.append(trace)
                    separate.append(current)
                    with audit_moe(mlp) as trace:
                        output = mlp(joint.clone())
                        trace["output"] = cpu(output)
                    mixed.append([trace])
                with frozen_routes(mlp, separate[0], len(p), capacity):
                    with audit_moe(mlp) as trace:
                        output = mlp(joint.clone())
                        trace["output"] = cpu(output)
                lanes["fixed_reductions" if fixed else "original"] = dict(
                    self_repeat=[
                        pair_metrics(separate[0], x, len(p)) for x in separate[1:]
                    ],
                    mixed_vs_separate=[
                        pair_metrics(separate[0], x, len(p), capacity) for x in mixed
                    ],
                    frozen_routes=pair_metrics(separate[0], [trace], len(p), capacity),
                )
        result[name] = dict(gate_dtype=str(mlp.gate.weight.dtype), lanes=lanes)
    return result


@torch.no_grad()
def run(scheduler, config):
    harness = importlib.import_module("mixed_eager_harness")
    importlib.reload(harness)
    original = harness._models
    model = scheduler.tp_worker.model_runner.model
    traces = []
    backend_modes = []
    layers = config.get("probe_layers", [0, 3, 7, 15, 31, 39])

    def wrapped(*args, **kwargs):
        backend_modes.append(
            {
                label: dict(
                    use_ragged=getattr(
                        backend.full_attn_backend.forward_metadata, "use_ragged", None
                    ),
                    extend_no_prefix=getattr(
                        backend.full_attn_backend.forward_metadata,
                        "extend_no_prefix",
                        None,
                    ),
                    is_multimodal=backend.full_attn_backend.is_multimodal,
                )
                for label, backend in [("P", args[5]), ("V", args[6])]
            }
        )
        with capture(model, layers) as data:
            value = original(*args, **kwargs)
        traces.append(data)
        return value

    harness._models = wrapped
    try:
        with audit_marlin() as calls:
            result = harness.run(
                scheduler,
                config
                | dict(
                    rounds=1,
                    repeat_control=True,
                    continue_on_mismatch=True,
                    numerical_probe=False,
                ),
            )
    finally:
        harness._models = original
    capacity = config["capacities"][0]
    result["layer_repeat"] = compare(traces[0], traces[1])
    result["layer_mixed"] = compare(traces[0], traces[2], capacity)
    result["backend_modes"] = backend_modes
    result["marlin_kernel_calls"] = calls
    result["moe_quant_methods"] = {
        name: type(module.quant_method).__name__
        for name, module in model.named_modules()
        if name.endswith(".mlp.experts")
    }
    if not config.get("skip_micro"):
        result["identical_input_moe"] = micro(
            model, traces[0], capacity, harness._fixed_reduction_diagnostic
        )
        result["identical_input_linear"] = linear_micro(
            model, traces[0], capacity, harness._fixed_reduction_diagnostic
        )
    result["gate_passed"] = False
    result["diagnostic_only"] = True
    return result
