"""Real Qwen MoE/PLE CUDA regression with explicit single-rank metadata.

No replacement expert outputs: use production routing, expert orchestration,
alignment, GEMM, activation, reduction, PLE hashing/gather and state updates.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import contextvars
import functools
import hashlib
import json
import logging
import math
import sys
import traceback
import types
import typing
from pathlib import Path

import qwen38_small_decoder as small
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
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
                "--expt-relaxed-constexpr",
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
        functions=tuple(n for n in all_functions(path) if n != "fused_moe"),
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


def effective_moe_reference(block, x):
    """Check independent matmul math on the exact FP8 operands that were used.

    A separately rounded ideal quantizer can cross FP8 midpoint boundaries;
    preserve that comparison as a second metric, rather than conflating it
    with the expert GEMMs' arithmetic error.
    """
    impl = block.experts.m.impl
    invoke = impl.invoke_fused_moe_kernel
    namespace = invoke.__globals__
    quant = namespace["sglang_per_token_group_quant_fp8"]
    tail = impl.silu_and_mul_block_quant_two_group
    operands = []
    gemms = []

    def capture_quant(a, g):
        q, s = quant(a, g)
        operands.append((a.clone(), q.clone(), s.clone()))
        return q, s

    def capture_invoke(*a, **kw):
        invoke(*a, **kw)
        gemms.append((a[0].clone(), a[3].clone()))

    def capture_tail(*a, **kw):
        q, s = tail(*a, **kw)
        gate, up = a[0].float().chunk(2, -1)
        operands.append(((F.silu(gate) * up).bfloat16(), q.clone(), s.clone()))
        return q, s

    namespace["sglang_per_token_group_quant_fp8"] = capture_quant
    impl.invoke_fused_moe_kernel = capture_invoke
    impl.silu_and_mul_block_quant_two_group = capture_tail
    try:
        actual = block(x)
    finally:
        namespace["sglang_per_token_group_quant_fp8"] = quant
        impl.invoke_fused_moe_kernel = invoke
        impl.silu_and_mul_block_quant_two_group = tail
    topk = block.topk.last
    ideal = block.experts.reference(x, topk)
    ideal_nrmse = (
        (actual.float() - ideal.float()).norm() / ideal.float().norm()
    ).item()
    assert ideal_nrmse < 0.02, ideal_nrmse
    info = {
        "ideal_quantizer_nrmse": ideal_nrmse,
        "ideal_quantizer_max_abs": (actual.float() - ideal.float()).abs().max().item(),
        "effective_operand_gemm_nrmse": [],
        "quantizer_different_elements": [],
        "quantizer_midpoint_fraction": [],
    }
    assert len(operands) == len(gemms) == 2
    stage_refs = []
    for i, ((a, q, s), (_, out), w, scale) in enumerate(
        zip(
            operands,
            gemms,
            (block.experts.w1, block.experts.w2),
            (block.experts.s1, block.experts.s2),
        )
    ):
        expanded = s.repeat_interleave(128, -1)[:, : q.shape[-1]]
        dequant = q.float() * expanded
        # E4M3 round-to-nearest error: <=6.25% for normals plus half a
        # subnormal step. The small margin covers approximate reciprocals.
        bound = a.float().abs() * 0.063 + expanded * 0.0011
        assert ((dequant - a.float()).abs() <= bound).all().item()
        info["quantizer_different_elements"].append(
            int((torch_group_quant(a) != dequant).sum().item())
        )
        ideal_q = (
            (a.float() / expanded).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        )
        different = q.float() != ideal_q
        fraction = 0.0
        if different.any().item():
            midpoint = (q.float() + ideal_q) * 0.5
            step = (q.float() - ideal_q).abs().clamp_min(1e-10)
            fraction = (
                ((a.float() / expanded - midpoint).abs() / step)[different].max().item()
            )
        info["quantizer_midpoint_fraction"].append(fraction)
        weights = (
            w.float()
            * scale.repeat_interleave(128, 1).repeat_interleave(128, 2)[
                :, : w.shape[1], : w.shape[2]
            ]
        )
        values = []
        for choice in range(2):
            ids = topk.topk_ids[:, choice].long()
            rows = dequant if i == 0 else dequant.view(x.shape[0], 2, -1)[:, choice]
            value = torch.bmm(weights[ids], rows.unsqueeze(-1)).squeeze(-1)
            if i == 1:
                value = value * topk.topk_weights[:, choice, None]
            values.append(value.bfloat16())
        reference = torch.stack(values, 1).reshape_as(out)
        nrmse = (
            (out.float() - reference.float()).norm() / reference.float().norm()
        ).item()
        assert nrmse < 0.003, nrmse
        torch.testing.assert_close(out, reference, atol=0.02, rtol=0.02)
        info["effective_operand_gemm_nrmse"].append(nrmse)
        stage_refs.append(reference)
    # Independently check the activation before its second quantization.
    gate, up = gemms[0][1].float().chunk(2, -1)
    torch.testing.assert_close(
        operands[1][0], (F.silu(gate) * up).bfloat16(), atol=0.002, rtol=0.008
    )
    result = stage_refs[1][:, 0] + stage_refs[1][:, 1]
    return result, info


def moe_case(m, fp8, intermediate):
    block = make_moe(m, fp8, intermediate)
    funcs, refs, diagnostics = [], [], []
    for rows in (65, 1):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        topk = block.topk(x, block.gate(x)[0])
        if fp8:
            reference, info = effective_moe_reference(block, x)
            diagnostics.append(info)
            refs.append(reference)
        else:
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

    return (*funcs, references), diagnostics


def enable_fp8_linears(module):
    quant = production_quant()
    gemm = ops.load(
        "moe_ple_fp8_gemm",
        "python/sglang/kernels/ops/gemm/fp8_kernel.py",
        is_cuda=lambda: True,
        is_gfx1250_supported=lambda: False,
        register_custom_op=lambda **kw: lambda fn: fn,
        get_device_name=torch.cuda.get_device_name,
        get_platform=lambda: NS(is_sm90=False),
        log_info_on_rank0=lambda logger, message: logger.info(message),
    )
    count = 0
    for linear in module.modules():
        if isinstance(linear, small.Linear) and all(
            n % 128 == 0 for n in linear.weight.shape
        ):
            if linear.fp8_ops is None:
                linear.enable_fp8(
                    quant._per_token_group_quant_8bit_raw,
                    gemm.w8a8_block_fp8_matmul_triton,
                )
            count += 1
    return count


class VocabMetadata(nn.Module):
    """Only the sharding/weight-loading boundary; actual lookups use pinned PLE."""

    def __init__(self, rows, dim, params_dtype, output_dtype, use_attn_tp_group):
        super().__init__()
        self.quant_method = UnquantizedEmbeddingMetadata()
        self.weight = nn.Parameter(
            torch.empty(rows, dim, dtype=params_dtype), requires_grad=False
        )
        self.shard_indices = NS(org_vocab_start_index=0, org_vocab_end_index=rows)
        self.embedding_dim, self.tp_size, self.num_added_embeddings = dim, 1, 0
        self.quant_config = None
        self.enable_tp = self.use_attn_tp_group = self.use_presharded_weights = False
        self.padding_size = 0
        for key in (
            "num_embeddings",
            "org_vocab_size",
            "org_vocab_size_padded",
            "num_embeddings_padded",
            "num_embeddings_per_partition",
            "num_org_embeddings_per_partition",
        ):
            setattr(self, key, rows)
        self.num_added_embeddings_per_partition = 0

    def weight_loader(self, param, loaded):
        param.copy_(loaded)


class UnquantizedEmbeddingMetadata:
    pass


def load_ple(fp8):
    import msgspec
    import sympy

    if "sglang" not in sys.modules:
        ops.bootstrap_fla()
    current = contextvars.ContextVar("ple_current_batch")
    req_pool = NS(
        ple_window_cache=None,
        size=8,
        state_sub_pool="ple",
        get_mamba_indices=lambda indices: indices,
    )
    pools = small.selected(
        "ple_real_pools",
        "python/sglang/srt/mem_cache/ple_state_pool.py",
        {
            "NGramPool": ([], {"get_context", "set_context"}),
            "ShortConvPool": ([], {"layer_cache", "layer_intermediate_cache"}),
        },
        dict(torch=torch),
    )
    ngram = pools.NGramPool()
    ngram.context = torch.randint(1, 256, (8, 2), device="cuda")
    ngram.context[0] = 0
    ngram.context[1, 0] = 0
    ngram.context[2, 1] = 0
    conv = pools.ShortConvPool()
    conv.layer_map = {0: 0, 1: 1}
    conv.conv_state = (
        torch.randn(2, 8, 1024, 9, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    conv.conv_state[:, 0] = 0
    conv.intermediate_conv_state = None
    router = NS(
        part_view=lambda part: ngram,
        wait_for_part=lambda part: None,
        read=lambda layer, part: NS(view=conv, view_layer_id=layer),
    )
    token_pool = NS(layer_router=router)

    class Backend:
        runner_ids = NS(state_ids=lambda pool: NS(ids=lambda indices: indices))

        @property
        def linear_attn_backend(self):
            return NS(state_tracking_indices=current.get().track_indices)

    backend = Backend()
    envs = NS(
        SGLANG_ENABLE_QWEN4_PLE_FUSION=NS(get=lambda: True),
        SGLANG_USE_ATTN_TP_NGRAM=NS(get=lambda: False),
    )
    host = ops.load(
        "ple_real_host", "python/sglang/srt/models/qwen4_exp_ple_table.py", envs=envs
    )
    numerics = ops.load("ple_real_numerics", "python/sglang/kernels/numerics.py")
    for name, rel, kw in (
        (
            "sglang.kernels.ops.embeddings.qwen4_ngram",
            "python/sglang/kernels/ops/embeddings/qwen4_ngram.py",
            {},
        ),
        (
            "sglang.kernels.ops.elementwise.qwen4_gate",
            "python/sglang/kernels/ops/elementwise/qwen4_gate.py",
            {"round_bf16_to_fp32": numerics.round_bf16_to_fp32},
        ),
        (
            "sglang.kernels.ops.mamba.qwen4_short_conv",
            "python/sglang/kernels/ops/mamba/qwen4_short_conv.py",
            {"round_bf16_to_fp32": numerics.round_bf16_to_fp32},
        ),
    ):
        ops.load(name, rel, **kw)

    class Projection(small.Linear):
        def __init__(self, i, o, **kwargs):
            super().__init__(i, o)

    path = "python/sglang/srt/models/qwen4_exp.py"
    src = small.selected(
        "ple_real_model",
        path,
        {
            "_PLEBatch": (["msgspec.Struct"], None),
            "Qwen4ExpPLEGroupedNorm": (["nn.Module"], None),
            "Qwen4ExpNGramEmbedding": (["nn.Module"], None),
            "Qwen4ExpPinnedHostEmbedding": (["VocabParallelEmbedding"], None),
            "Qwen4ExpPLELayer": (["nn.Module"], None),
        },
        dict(
            torch=torch,
            nn=nn,
            F=F,
            triton=triton,
            tl=triton.language,
            math=math,
            msgspec=msgspec,
            Optional=typing.Optional,
            ForwardMode=small.Mode,
            sympy=sympy,
            nullcontext=contextlib.nullcontext,
            envs=envs,
            get_disagg=lambda: NS(enable_pdmux=True),
            get_parallel=lambda: NS(attn_dp_size=1),
            is_dp_attention_enabled=lambda: False,
            get_is_capture_mode=torch.cuda.is_current_stream_capturing,
            is_in_breakable_cuda_graph=lambda: False,
            get_req_to_token_pool=lambda: req_pool,
            get_token_to_kv_pool=lambda: token_pool,
            get_attn_backend=lambda: backend,
            SHORT_CONV_PART="short_conv",
            NGRAM_PART="ngram",
            ReplicatedLinear=Projection,
            VocabParallelEmbedding=VocabMetadata,
            UnquantizedEmbeddingMethod=UnquantizedEmbeddingMetadata,
            allocate_ple_host_table=host.allocate_ple_host_table,
            make_ple_file_prefetcher=host.make_ple_file_prefetcher,
            make_ple_file_rss_trimmer=host.make_ple_file_rss_trimmer,
        ),
        functions=(
            "_ple_table_is_fp8",
            "_use_attn_tp_ngram",
            "_get_ple_forward_mode",
            "_get_processed_token_count",
            "_prepare_ple_batch",
            "_commit_ple_batch",
            "_ple_track_targets",
            "_pad_token_rows",
            "_gather_ple_embedding_from_pinned_kernel",
        ),
    )
    cfg = NS(
        hidden_size=256,
        ple_embed_dim=256,
        ple_conv_kernel_size=4,
        hc_count=4,
        ngram_size=3,
        heads_per_ngram=8,
        vocab_size=256,
        ngram_vocab_size_base=101,
        make_ngram_vocab_size_divisible_by=128,
        eos_token_id=0,
        seed=1234,
        ple_embedding_dtype="float8_e4m3fn" if fp8 else None,
        ple_offload_embedding=True,
        ple_offload_backend="pinned",
        rms_norm_eps=1e-6,
    )
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            layers = nn.ModuleList(
                [
                    src.Qwen4ExpPLELayer(cfg, layer_id=i, ple_layer_index=i)
                    for i in range(2)
                ]
            )
    finally:
        torch.set_default_dtype(old_dtype)
    for layer in layers:
        table = layer.ple_embedding.ngram_embedding.weight
        table.copy_((torch.randn(table.shape, device="cpu") * 0.1).to(table.dtype))
        layer.ple_embedding.ngram_embedding.weight_scale.fill_(0.75)
        layer.conv1d.weight.copy_(torch.randn_like(layer.conv1d.weight) * 0.1)
        assert table.is_pinned() and table.device.type == "cpu"
    if fp8:
        assert enable_fp8_linears(layers) == 4
    initial = ngram.context.clone(), conv.conv_state.clone()
    return NS(
        src=src,
        layers=layers,
        ngram=ngram,
        conv=conv,
        req_pool=req_pool,
        current=current,
        initial=initial,
        cfg=cfg,
    )


def ple_batch(tokens, decode=False, slots=None):
    lengths = (
        [1] if decode else ([33, 32] if tokens == 65 and slots is None else [tokens])
    )
    slots = slots or ([2] if decode else ([1, 3] if len(lengths) > 1 else [1]))
    physical = tokens if decode else tokens + 5
    b = NS(
        forward_mode=small.Mode(decode),
        _original_forward_mode=None,
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=None,
        global_num_token_non_padded_cpu=tokens,
        extend_seq_lens_cpu=lengths,
        extend_seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        req_pool_indices=torch.tensor(slots, device="cuda", dtype=torch.long),
        out_cache_loc=torch.ones(physical, device="cuda", dtype=torch.long),
        mamba_track_indices=torch.tensor(
            [6] if decode else [4, 5][: len(slots)], device="cuda"
        ),
        mamba_track_mask=torch.ones(len(slots), device="cuda", dtype=torch.bool),
    )
    b.track_indices = b.mamba_track_indices
    aligned = torch.tensor(
        [1] if decode else [min(16, n) for n in lengths], device="cuda"
    )
    b.mamba_track_aligned_lens = lambda: aligned
    b.input_ids = torch.randint(1, 256, (physical,), device="cuda")
    if not decode:
        b.input_ids[10] = 0
        if tokens > 40:
            b.input_ids[40] = 0
    return b


def reset_ple(p, b):
    indices = torch.cat([b.req_pool_indices, b.track_indices])
    p.ngram.context[indices] = p.initial[0][indices]
    p.conv.conv_state[:, indices] = p.initial[1][:, indices]


def scalar_ngram(layer, contexts):
    rows = contexts.cpu().tolist()
    multipliers = layer.layer_multipliers.cpu().tolist()
    sizes = layer.ngram_heads_vocab_sizes.cpu().tolist()
    offsets = layer.ngram_heads_offsets.cpu().tolist()
    ids = []
    for old, previous, token in rows:
        a = token * multipliers[0]
        b = previous * multipliers[1]
        c = (old if old != 0 and previous != 0 else 0) * multipliers[2]
        ids.append(
            [
                ((a ^ b) if h < 8 else (a ^ b ^ c)) % sizes[h] + offsets[h]
                for h in range(16)
            ]
        )
    return torch.tensor(ids, device=contexts.device, dtype=torch.long)


def ple_case(p):
    funcs, refs = [], []
    for lane, (tokens, decode) in enumerate(((65, False), (1, True))):
        b = ple_batch(tokens, decode)
        x = (
            torch.randn(b.input_ids.numel(), 1024, device="cuda", dtype=torch.bfloat16)
            * 0.1
        )
        p.current.set(b)
        reset_ple(p, b)
        batch = p.src._prepare_ple_batch(
            b.input_ids, b, ngram_size=3, ngram_eos_token_id=0
        )
        contexts = (
            batch.ngram_context
            if decode
            else batch.ngram_context.unfold(1, 3, 1)[
                batch.req_indices, batch.token_offsets
            ]
        )
        expected_ids = scalar_ngram(p.layers[-1].ple_embedding, contexts)
        table = p.layers[-1].ple_embedding.ngram_embedding
        expected_embed = (
            table.weight.float()[expected_ids.cpu()].bfloat16().to("cuda").flatten(1)
            * table.weight_scale
        )
        expected_context = batch.ngram_context.gather(
            1, batch.lengths[:, None] + torch.arange(2, device="cuda")
        )
        aligned = b.mamba_track_aligned_lens()
        expected_track = batch.ngram_context.gather(
            1, aligned[:, None] + torch.arange(2, device="cuda")
        )
        refs.append(
            (
                expected_ids,
                expected_embed,
                expected_context,
                expected_track,
                x[-5:].clone() if not decode else None,
            )
        )

        def fn(b=b, x=x):
            p.current.set(b)
            reset_ple(p, b)
            batch = p.src._prepare_ple_batch(
                b.input_ids, b, ngram_size=3, ngram_eos_token_id=0
            )
            p.layers[1].start_prefetch(batch, b)
            assert p.layers[1]._prefetch_state is None
            out = x
            for layer in p.layers:
                out = out + layer(out, b, batch)
            ids = p.layers[-1].ple_embedding.compute_ngram_ids(batch)
            embedding = p.layers[-1].ple_embedding(batch, b)
            p.src._commit_ple_batch(batch, b)
            return (
                out,
                ids.clone(),
                embedding.clone(),
                p.ngram.context[b.req_pool_indices].clone(),
                p.conv.conv_state[:, b.req_pool_indices].clone(),
                p.ngram.context[b.track_indices].clone(),
                p.conv.conv_state[:, b.track_indices].clone(),
            )

        funcs.append(fn)
    sentinel = p.ngram.context[7].clone(), p.conv.conv_state[:, 7].clone()

    def references(a, b):
        torch.testing.assert_close(p.ngram.context[7], sentinel[0], rtol=0, atol=0)
        torch.testing.assert_close(p.conv.conv_state[:, 7], sentinel[1], rtol=0, atol=0)
        torch.testing.assert_close(a[0][-5:], refs[0][4], rtol=0, atol=0)
        for output in (a[0], b[0]):
            assert torch.isfinite(output).all().item()
        return [
            (actual[i], expected[j], 0)
            for actual, expected in zip((a, b), refs)
            for i, j in ((1, 0), (2, 1), (3, 2), (5, 3))
        ]

    return *funcs, references


def integrated_run(rounds, fp8):
    stack, gdn, qsa = small.build_stack()
    qsa.canonical_topk = True
    moe = load_moe()
    ple = load_ple(fp8)
    globals_ = stack.forward.__func__.__globals__
    globals_.update(
        _prepare_ple_batch=ple.src._prepare_ple_batch,
        _commit_ple_batch=ple.src._commit_ple_batch,
        Qwen2MoeSparseMoeBlock=moe.block.Qwen2MoeSparseMoeBlock,
    )
    stack.has_ple = True
    stack.ple_ngram_size, stack.ple_ngram_eos_token_id = 3, 0
    for i, layer in enumerate(stack.layers):
        shared = layer.mlp
        layer.mlp = make_moe(moe, fp8)
        layer.mlp.shared_expert = shared
        layer.mlp.shared_expert_gate = nn.Linear(
            256, 1, bias=False, device="cuda", dtype=torch.bfloat16
        )
        layer.ple = ple.layers[i]
    moe.block.__dict__["fused_gate_sigmoid_mul_add"] = (
        qsa.m.elem.fused_gate_sigmoid_mul_add
    )
    fp8_count = enable_fp8_linears(stack) if fp8 else 0
    batches = []
    inputs = []
    for lane, tokens in enumerate((65, 1)):
        b = small.prepare_batch(tokens, lane)
        side = ple_batch(tokens, tokens == 1, slots=[lane + 1])
        b.__dict__.update(side.__dict__)
        b.indices = b.req_pool_indices.int()
        b.input_ids = b.input_ids[:tokens].contiguous()
        b.out_cache_loc = b.out_cache_loc[:tokens]
        batches.append(b)
        inputs.append(
            torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16) * 0.1
        )

    def reset(lane):
        gdn.reset(lane + 1)
        qsa.reset(lane)
        reset_ple(ple, batches[lane])

    def forward(lane, interval=None):
        b = batches[lane]
        ple.current.set(b)
        return stack.forward(
            b.input_ids,
            b.positions,
            b,
            inputs_embeds=inputs[lane],
            split_interval=interval,
        )

    def snapshot(lane, output):
        b = batches[lane]
        values = [
            *output,
            gdn.state[lane + 1],
            gdn.conv_state[lane + 1],
            qsa.k[lane],
            qsa.v[lane],
            b.layer_outputs[0],
            b.selected_tokens,
            ple.ngram.context[b.req_pool_indices],
            ple.ngram.context[b.track_indices],
            ple.conv.conv_state[:, b.req_pool_indices],
            ple.conv.conv_state[:, b.track_indices],
        ]
        for layer in stack.layers:
            values += [layer.mlp.topk.last.topk_ids, layer.mlp.topk.last.topk_weights]
        return tuple(t.clone() for t in values)

    controls = []
    for lane in (0, 1):
        reset(lane)
        controls.append(snapshot(lane, forward(lane)))
    torch.cuda.synchronize()
    sentinel = (
        ple.ngram.context[7].clone(),
        ple.conv.conv_state[:, 7].clone(),
        gdn.state[0].clone(),
        gdn.conv_state[0].clone(),
        qsa.k[2].clone(),
        qsa.v[2].clone(),
    )
    ready = torch.cuda.current_stream().record_event()
    ps, ds = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=-1)
    ps.wait_event(ready)
    ds.wait_event(ready)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=ds):
        reset(1)
        graph_outputs = snapshot(1, forward(1))
    ds.synchronize()
    records = []
    for _ in range(rounds):
        with torch.cuda.stream(ps):
            reset(0)
            assert forward(0, (0, 1)) is None
        with torch.cuda.stream(ds):
            graph.replay()
            d = tuple(t.clone() for t in graph_outputs)
            torch.empty((128, 128), device="cuda").fill_(3)
        with torch.cuda.stream(ps):
            p = snapshot(0, forward(0, (1, 2)))
            assert not batches[0].model_specific_states
            assert all(layer.ple._prefetch_state is None for layer in stack.layers)
        records.append((p, d))
    ps.synchronize()
    ds.synchronize()
    errors = [0.0] * len(controls[0])
    for pair in records:
        for outputs, control in zip(pair, controls):
            for i, (a, b) in enumerate(zip(outputs, control)):
                assert torch.isfinite(a).all().item()
                errors[i] = max(errors[i], (a.float() - b.float()).abs().max().item())
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    actual = (
        ple.ngram.context[7],
        ple.conv.conv_state[:, 7],
        gdn.state[0],
        gdn.conv_state[0],
        qsa.k[2],
        qsa.v[2],
    )
    for a, b in zip(actual, sentinel):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    return dict(
        name=f"real_gdn_qsa_moe_ple_{'fp8' if fp8 else 'bf16'}",
        rounds=rounds,
        decode_cuda_graph=True,
        decode_stream_priority=-1,
        split_at_layer=1,
        canonical_qsa_tokens=True,
        experts=4,
        topk=2,
        shared_expert=True,
        fp8_linear_count=fp8_count,
        ple_table_dtype=str(ple.layers[0].ple_embedding.ngram_embedding.weight.dtype),
        max_abs=errors,
        untouched_slots_exact=True,
        prefetch_state_isolated=True,
        scope="production decoder/MoE/PLE forwards; real kernels, TP1 metadata adapters and fixed QSA prefix",
    )


def ple_progressive_run(rounds):
    """Graph replay must consume the history committed by its previous step."""
    p = load_ple(True)
    pb, db = ple_batch(65), ple_batch(1, True)
    px = torch.randn(70, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    dx = torch.empty(1, 1024, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(rounds, 1, 1024, device="cuda", dtype=torch.bfloat16) * 0.1
    tokens = torch.randint(1, 256, (rounds,), device="cuda")
    if rounds > 5:
        tokens[5] = 0
    if rounds > 17:
        tokens[17] = 0

    def encode(b, x):
        p.current.set(b)
        batch = p.src._prepare_ple_batch(
            b.input_ids, b, ngram_size=3, ngram_eos_token_id=0
        )
        p.layers[1].start_prefetch(batch, b)
        out = x
        for layer in p.layers:
            out = out + layer(out, b, batch)
        ids = p.layers[-1].ple_embedding.compute_ngram_ids(batch)
        p.src._commit_ple_batch(batch, b)
        return tuple(
            t.clone()
            for t in (
                out,
                ids,
                p.ngram.context[b.req_pool_indices],
                p.ngram.context[b.track_indices],
                p.conv.conv_state[:, b.req_pool_indices],
                p.conv.conv_state[:, b.track_indices],
            )
        )

    reset_ple(p, pb)
    prefill_control = encode(pb, px)
    reset_ple(p, db)
    serial = []
    for i in range(rounds):
        dx.copy_(inputs[i])
        db.input_ids.copy_(tokens[i : i + 1])
        serial.append(encode(db, dx))
    torch.cuda.synchronize()
    sentinel = p.ngram.context[7].clone(), p.conv.conv_state[:, 7].clone()
    ready = torch.cuda.current_stream().record_event()
    ps, ds = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
    ps.wait_event(ready)
    ds.wait_event(ready)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=ds):
        graph_outputs = encode(db, dx)
    ds.synchronize()
    with torch.cuda.stream(ds):
        reset_ple(p, db)
    records = []
    for i in range(rounds):
        with torch.cuda.stream(ps):
            reset_ple(p, pb)
            prefill = encode(pb, px)
        with torch.cuda.stream(ds):
            dx.copy_(inputs[i])
            db.input_ids.copy_(tokens[i : i + 1])
            graph.replay()
            decode = tuple(t.clone() for t in graph_outputs)
            torch.empty((128, 128), device="cuda").fill_(3)
        records.append((prefill, decode))
    ps.synchronize()
    ds.synchronize()
    for i, (prefill, decode) in enumerate(records):
        for got, expected in ((prefill, prefill_control), (decode, serial[i])):
            for a, b in zip(got, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
    if rounds >= 2:
        torch.testing.assert_close(p.ngram.context[2], tokens[-2:], atol=0, rtol=0)
        torch.testing.assert_close(p.ngram.context[6], tokens[-2:], atol=0, rtol=0)
    for a, b in zip((p.ngram.context[7], p.conv.conv_state[:, 7]), sentinel):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    return dict(
        name="real_ple_progressive_fp8",
        rounds=rounds,
        decode_cuda_graph=True,
        reset_between_decode_steps=False,
        changing_token_ids_and_hidden_states=True,
        history_and_conv_exact=True,
        outputs_exact=True,
        untouched_slots_exact=True,
    )


@torch.inference_mode()
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
                case, diagnostics = moe_case(m, fp8, intermediate)
                result = ops.pair_test(name, *case, args.rounds)
                result["reference_diagnostics"] = diagnostics
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                traceback.print_exc()
                report["errors"].append({"name": name, "error": str(exc)})
    if args.only in ("ple", "all"):
        for fp8 in (False, True):
            name = f"real_ple_{'fp8' if fp8 else 'bf16'}_pinned_table_and_state"
            try:
                result = ops.pair_test(name, *ple_case(load_ple(fp8)), args.rounds)
                report["cases"].append(result)
                print(json.dumps(result), flush=True)
            except Exception as exc:
                traceback.print_exc()
                report["errors"].append({"name": name, "error": str(exc)})
        try:
            result = ple_progressive_run(args.rounds)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            traceback.print_exc()
            report["errors"].append(
                {"name": "real_ple_progressive_fp8", "error": str(exc)}
            )
    if args.only in ("integrated", "all"):
        for fp8 in (False, True):
            name = f"integrated_{'fp8' if fp8 else 'bf16'}"
            try:
                result = integrated_run(args.rounds, fp8)
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
