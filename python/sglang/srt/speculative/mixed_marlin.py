"""Opt-in offline AWQ numerical contract for mixed target validation.

Keep this checkpoint's AWQ routed experts and its excluded FP16 projections.
Use full-K Marlin experts and fixed-tile FP16 Linear/router GEMMs with FP32
accumulation and FP16 output. Normal serving does not import this module.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from functools import partial

import torch
import triton
import triton.language as tl


@triton.jit
def _router(
    x,
    w,
    y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 64)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(K, 64)):
        kk = start * 64 + k
        a = tl.load(
            x + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0
        )
        b = tl.load(
            w + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0
        )
        acc = tl.dot(a, b, acc)
    tl.store(y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def projection_tile(weight):
    """Offline candidate selected only by weight shape, never batch size.

    Retain the small tile for routers/BA. Wider projections reuse weight
    traffic across more rows/columns. K64 and its accumulation order stay
    fixed. This RTX4060 candidate still needs full-model A100 validation.
    """
    n = weight.shape[0]
    if n >= 8192:
        return (32, 128, 8)
    if n >= 1024:
        return (32, 64, 4)
    return (16, 32, 4)


def fixed_linear(x, weight, *, tile=(16, 32, 4)):
    assert x.dtype == weight.dtype == torch.float16
    assert x.is_contiguous() and weight.is_contiguous()
    m, k = x.shape
    n = weight.shape[0]
    if tile == "projection":
        tile = projection_tile(weight)
    if len(tile) == 3:
        bm, bn, warps = tile
        stages = 2
    elif len(tile) == 4:
        bm, bn, warps, stages = tile
    else:
        raise ValueError(f"Unsupported fixed linear tile: {tile}")
    if (
        bm not in (16, 32, 64, 128)
        or bn not in (32, 64, 128)
        or warps not in (4, 8)
        or stages not in (2, 3)
    ):
        raise ValueError(f"Unsupported fixed linear tile: {tile}")
    y = x.new_empty((m, n))
    _router[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, weight, y, m, n, k, bm, bn, num_warps=warps, num_stages=stages
    )
    return y


@contextmanager
def stable_marlin(
    model,
    *,
    full_k_moe=True,
    linear_tile=(16, 32, 4),
    moe_block_size=8,
):
    """Apply to this target's layers, restoring all methods on exit.

    Defaults retain the validated legacy arithmetic. `linear_tile="projection"`
    and a constant larger `moe_block_size` are opt-in operator-level candidates;
    they do not change native serving or establish full-model correctness.
    """
    moe = importlib.import_module(
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
    )
    triton_moe = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    )
    old_mode = moe.is_batch_invariant_mode_enabled
    old_fused = moe.fused_marlin_moe
    old_triton_mode = triton_moe.is_batch_invariant_mode_enabled
    changes = []
    counts = dict(fp16_linear=0, full_k_moe=full_k_moe)
    seen = set()
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if (
            type(method).__name__ == "UnquantizedLinearMethod"
            and type(module).__name__ != "ParallelLMHead"
            and id(method) not in seen
        ):
            seen.add(id(method))
            changes.append((method, "apply", method.__dict__.get("apply")))

            def apply_fp16(layer, x, bias=None):
                counts["fp16_linear"] += 1
                y = fixed_linear(
                    x.reshape(-1, x.shape[-1]).contiguous(),
                    layer.weight,
                    tile=linear_tile,
                )
                if bias is not None:
                    y.add_(bias)
                return y.reshape(x.shape[:-1] + (layer.weight.shape[0],))

            method.apply = apply_fp16
    if full_k_moe:
        moe.is_batch_invariant_mode_enabled = lambda: True
        triton_moe.is_batch_invariant_mode_enabled = lambda: True
        moe.fused_marlin_moe = partial(
            old_fused,
            deterministic_block_size_m=moe_block_size,
        )
    try:
        yield counts
    finally:
        moe.is_batch_invariant_mode_enabled = old_mode
        moe.fused_marlin_moe = old_fused
        triton_moe.is_batch_invariant_mode_enabled = old_triton_mode
        for obj, name, value in reversed(changes):
            if value is None:
                delattr(obj, name)
            else:
                setattr(obj, name, value)
