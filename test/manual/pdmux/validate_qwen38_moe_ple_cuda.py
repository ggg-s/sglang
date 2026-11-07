"""Real Qwen MoE/PLE CUDA regression with explicit single-rank metadata.

No replacement expert outputs: use production routing, expert orchestration,
alignment, GEMM, activation, reduction, PLE hashing/gather and state updates.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import functools
import hashlib
import json
import logging
import math
import sys
import traceback
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

import qwen38_small_decoder as small
import validate_qwen38_real_ops_cuda as ops

NS = types.SimpleNamespace


def all_functions(rel):
    return tuple(
        n.name
        for n in ast.parse((ops.ROOT / rel).read_text()).body
        if isinstance(n, ast.FunctionDef)
    )


def jit_bootstrap():
    """Build existing JIT wrappers with TVM-FFI; bypass server build imports."""
    from tvm_ffi.cpp import load_inline

    cpp = ops.load(
        "moe_ple_cpp_args", "python/sglang/kernels/jit/utils/compile/cpp_args.py"
    )

    @functools.lru_cache(None)
    def build(name, args, cuda_files, wrappers, flags):
        source = [
            "#include <tvm/ffi/container/tensor.h>",
            "#include <tvm/ffi/dtype.h>",
            "#include <tvm/ffi/error.h>",
            "#include <tvm/ffi/extra/c_env_api.h>",
            "#include <tvm/ffi/function.h>",
        ]
        for rel in cuda_files:
            p = ops.ROOT / "python/sglang/kernels/jit/csrc" / rel
            ops.SOURCES[str(p.relative_to(ops.ROOT))] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            source.append(f'#include "{p}"')
        source.append("namespace sglang {")
        source += [f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({n}, ({fn}));" for n, fn in wrappers]
        source.append("}")
        return load_inline(
            "qwen38_moe_ple_" + name,
            cuda_sources="\n".join(source),
            extra_include_paths=[str(ops.ROOT / "python/sglang/kernels/jit/include")],
            extra_cuda_cflags=[
                "-DSGL_CUDA_ARCH=890",
                "-std=c++20",
                "-O3",
                "-gencode=arch=compute_89,code=sm_89",
                *flags,
            ],
        )

    def load_jit(name, *args, cuda_files, cuda_wrappers, extra_cuda_cflags=()):
        return build(
            name,
            tuple(args),
            tuple(cuda_files),
            tuple(cuda_wrappers),
            tuple(extra_cuda_cflags),
        )

    return dict(
        cache_once=functools.lru_cache(None),
        make_cpp_args=cpp.make_cpp_args,
        load_jit=load_jit,
        is_arch_support_pdl=lambda: False,
        get_activation_cuda_cflags=lambda: ["--use_fast_math"],
    )


def production_quant():
    return small.selected(
        "moe_ple_quant",
        "python/sglang/kernels/ops/quantization/fp8_kernel.py",
        {},
        dict(
            torch=torch,
            triton=triton,
            tl=triton.language,
            _is_hip=False,
            fp8_dtype=torch.float8_e4m3fn,
        ),
        functions=(
            "_per_token_group_quant_8bit",
            "_per_token_group_quant_8bit_colmajor",
            "_per_token_group_quant_8bit_raw",
            "create_per_token_group_quant_fp8_output_scale",
        ),
    )


def load_moe():
    ops.bootstrap_fla()
    jit = jit_bootstrap()
    quant = production_quant()
    runtime = NS(
        deterministic=NS(enable_deterministic_inference=False),
        moe=NS(enable_fused_moe_sum_all_reduce=False),
    )
    envs = NS(
        SGLANG_USE_AITER=NS(get=lambda: False),
        SGLANG_EXPERIMENTAL_LORA_OPTI=NS(get=lambda: False),
    )
    common = dict(
        is_cuda=lambda: True,
        is_hip=lambda: False,
        is_sm90_supported=lambda: False,
        is_arch_support_pdl=lambda: False,
        is_batch_invariant_mode_enabled=lambda: False,
        get_moe_padding_size=lambda *a: 0,
        envs=envs,
    )
    kernel = ops.load(
        "moe_ple_kernels",
        "python/sglang/kernels/ops/moe/fused_moe_triton_kernels.py",
        **common,
        per_token_group_quant_fp8=quant._per_token_group_quant_8bit_raw,
        sglang_per_token_group_quant_fp8=quant._per_token_group_quant_8bit_raw,
    )
    activation = ops.load(
        "moe_ple_activation",
        "python/sglang/kernels/ops/activation/activation.py",
        **jit,
        register_custom_op=lambda **kw: lambda fn: fn,
    )
    align_jit = ops.load(
        "sglang.kernels.ops.moe.moe_align",
        "python/sglang/kernels/ops/moe/moe_align.py",
        **jit,
    )
    # This supported production option selects the in-tree JIT provider. The
    # cached sgl_kernel wheel on this host has an incompatible Torch ABI.
    align = small.selected(
        "moe_ple_align",
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py",
        {},
        dict(
            torch=torch,
            triton=triton,
            _is_cuda=True,
            _is_hip=False,
            _is_musa=False,
            _is_xpu=False,
            _SGLANG_EXPERIMENTAL_LORA_OPTI=False,
            _CUDA_SMALL_BATCH_MAX_BUCKETS=64,
            SMALL_NUMEL_LIMIT=64,
        ),
        functions=("moe_align_block_size",),
    )
    # The nested import resolves the exact production Python wrapper above.
    assert sys.modules["sglang.kernels.ops.moe.moe_align"] is align_jit
    config = ops.load(
        "moe_ple_config",
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py",
        get_exec=lambda: runtime,
        is_hip=lambda: False,
        get_device_name=lambda: torch.cuda.get_device_name(),
    )
    config.logger.warning_once = config.logger.warning
    config_ns = types.ModuleType("sglang.srt.layers.moe.moe_runner.triton_utils")
    for name in (
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.moe_runner",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(ops.ROOT / "python" / name.replace(".", "/"))]
        sys.modules[name] = package
    config_ns.__path__ = [
        str(ops.ROOT / "python/sglang/srt/layers/moe/moe_runner/triton_utils")
    ]
    config_ns.get_config = lambda: None
    # Only used by the production config selector's lazy get_config import.
    sys.modules[config_ns.__name__] = config_ns
    tail = ops.load(
        "moe_ple_silu_fp8",
        "python/sglang/kernels/ops/activation/silu_and_mul_block_quant.py",
    )
    path = "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
    impl = small.selected(
        "moe_ple_impl",
        path,
        {},
        dict(
            torch=torch,
            F=F,
            functools=functools,
            tl=triton.language,
            logger=logging.getLogger("moe_ple_impl"),
            padding_size=0,
            _is_cuda=True,
            _is_hip=False,
            _is_xpu=False,
            _is_musa=False,
            _use_aiter=False,
            is_arch_support_pdl=lambda: False,
            is_batch_invariant_mode_enabled=lambda: False,
            get_exec=lambda: runtime,
            get_parallel=lambda: NS(tp_group=None),
            is_allocation_symmetric=lambda: False,
            use_symmetric_memory=lambda *a, **kw: contextlib.nullcontext(),
            invoke_fused_moe_kernel=kernel.invoke_fused_moe_kernel,
            support_tensor_descriptor=kernel.support_tensor_descriptor,
            act_and_mul_triton=kernel.act_and_mul_triton,
            moe_sum_reduce_triton=kernel.moe_sum_reduce_triton,
            silu_and_mul=activation.silu_and_mul,
            silu_and_mul_block_quant_two_group=tail.silu_and_mul_block_quant_two_group,
            register_custom_op=lambda **kw: lambda fn: fn,
            get_config_dtype_str=config.get_config_dtype_str,
            try_get_optimal_moe_config=config.try_get_optimal_moe_config,
            moe_align_block_size=functools.partial(
                align.moe_align_block_size, pad_expert_ids=True
            ),
        ),
        functions=all_functions(path),
    )
    topk = small.selected(
        "moe_ple_topk",
        "python/sglang/srt/layers/moe/topk.py",
        {},
        dict(torch=torch, F=F, _RENORMALIZE_SUM_EPSILON=1e-20),
        functions=("fused_topk_torch_native",),
    )
    block = small.selected(
        "moe_ple_block",
        "python/sglang/srt/models/qwen2_moe.py",
        {
            "Qwen2MoeSparseMoeBlock": (
                ["nn.Module"],
                {"forward", "_forward_router_experts", "_forward_shared_experts"},
            )
        },
        dict(
            torch=torch,
            nn=nn,
            _use_aiter=False,
            _is_hip=False,
            get_moe_a2a_backend=lambda: NS(
                is_deepep=lambda: False,
                is_deepep_v2=lambda: False,
                is_mori=lambda: False,
                is_ascend_fuseep=lambda: False,
            ),
            use_intel_amx_backend=lambda *a: False,
            is_npu=lambda: False,
        ),
    )
    return NS(impl=impl, topk=topk, block=block, quant=quant, activation=activation)


def quantize_weight(weight):
    e, n, k = weight.shape
    padded = F.pad(weight.float(), (0, (-k) % 128, 0, (-n) % 128))
    scales = (
        padded.view(e, padded.shape[1] // 128, 128, padded.shape[2] // 128, 128)
        .abs()
        .amax((2, 4))
        .clamp_min(1e-10)
        / 448
    )
    expanded = scales.repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :n, :k]
    return (weight.float() / expanded).to(torch.float8_e4m3fn), scales


def torch_group_quant(x):
    width = x.shape[-1]
    padded = F.pad(x.float(), (0, (-width) % 128))
    grouped = padded.view(*x.shape[:-1], -1, 128)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    q = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * scale).flatten(-2)[..., :width]


class Experts(nn.Module):
    def __init__(self, m, fp8, intermediate=256):
        super().__init__()
        self.m, self.fp8 = m, fp8
        self.w1 = (
            torch.randn(4, intermediate * 2, 256, device="cuda", dtype=torch.bfloat16)
            / 16
        )
        self.w2 = torch.randn(
            4, 256, intermediate, device="cuda", dtype=torch.bfloat16
        ) / math.sqrt(intermediate)
        self.s1 = self.s2 = None
        if fp8:
            self.w1, self.s1 = quantize_weight(self.w1)
            self.w2, self.s2 = quantize_weight(self.w2)

    def forward(self, x, topk):
        return self.m.impl.fused_experts_impl(
            x.contiguous(),
            self.w1,
            self.w2,
            topk.topk_weights,
            topk.topk_ids,
            use_fp8_w8a8=self.fp8,
            w1_scale=self.s1,
            w2_scale=self.s2,
            block_shape=[128, 128] if self.fp8 else None,
            filter_expert=False,
        )

    def reference(self, x, topk):
        w1, w2 = self.w1.float(), self.w2.float()
        if self.fp8:
            w1 = (
                w1
                * self.s1.repeat_interleave(128, 1).repeat_interleave(128, 2)[
                    :, : w1.shape[1], : w1.shape[2]
                ]
            )
            w2 = (
                w2
                * self.s2.repeat_interleave(128, 1).repeat_interleave(128, 2)[
                    :, : w2.shape[1], : w2.shape[2]
                ]
            )
        inp = torch_group_quant(x) if self.fp8 else x.float()
        routes = []
        for choice in range(2):
            ids = topk.topk_ids[:, choice].long()
            up = torch.bmm(w1[ids], inp.unsqueeze(-1)).squeeze(-1).bfloat16().float()
            gate, value = up.chunk(2, -1)
            act = (F.silu(gate) * value).bfloat16()
            act = torch_group_quant(act) if self.fp8 else act.float()
            down = torch.bmm(w2[ids], act.unsqueeze(-1)).squeeze(-1)
            routes.append((down * topk.topk_weights[:, choice, None]).bfloat16())
        return routes[0] + routes[1]


class Router(nn.Module):
    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, hidden, logits):
        weights, ids = self.topk.fused_topk_torch_native(hidden, logits, 2, True)
        self.last = NS(topk_weights=weights, topk_ids=ids.int(), router_logits=logits)
        return self.last


def make_moe(m, fp8, intermediate=256):
    block = m.block.Qwen2MoeSparseMoeBlock()
    block.gate = small.Linear(256, 4)
    block.topk = Router(m.topk)
    block.experts = Experts(m, fp8, intermediate)
    block.shared_expert = block.shared_expert_gate = None
    block.alt_stream = None
    block._use_mega_moe = block.enable_shared_expert_fusion = block.reduce_results = (
        False
    )
    return block


def moe_case(m, fp8, intermediate):
    block = make_moe(m, fp8, intermediate)
    funcs, refs = [], []
    for rows in (65, 1):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        topk = block.topk(x, block.gate(x)[0])
        refs.append(block.experts.reference(x, topk))
        funcs.append(
            lambda x=x: (
                block(x),
                block.topk.last.topk_ids.clone(),
                block.topk.last.topk_weights.clone(),
            )
        )

    def references(p, d):
        pairs = []
        for actual, ref in ((p[0], refs[0]), (d[0], refs[1])):
            nrmse = (actual.float() - ref.float()).norm() / ref.float().norm()
            assert nrmse < 0.02, nrmse.item()
            pairs.append((actual, ref, 0.02))
        return pairs

    return *funcs, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument(
        "--only", choices=("moe", "ple", "integrated", "all"), default="all"
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    torch.manual_seed(613)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("Explicit SM89 test bootstrap")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": [],
        "errors": [],
    }
    if args.only in ("moe", "all"):
        m = load_moe()
        for fp8, intermediate in ((False, 256), (True, 256), (True, 192)):
            name = f"real_moe_{'fp8' if fp8 else 'bf16'}_I{intermediate}"
            try:
                result = ops.pair_test(
                    name, *moe_case(m, fp8, intermediate), args.rounds
                )
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                traceback.print_exc()
                report["errors"].append({"name": name, "error": str(exc)})
    report["source_sha256"] = ops.SOURCES
    report["passed"] = not report["errors"] and bool(report["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
