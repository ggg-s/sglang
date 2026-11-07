"""Production decoder forwards with real CUDA math and a single-rank harness.

Only constructors/server dispatch are bypassed. Selected method bodies are
compiled verbatim from the checkout. Test adapters own request metadata and
cache slots; they do not substitute simulated attention or FFN layers.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import math
import sys
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton

import validate_qwen38_real_ops_cuda as ops

NS = types.SimpleNamespace


def selected(name, rel, classes, env, functions=()):
    """Keep requested production methods intact, excluding unrelated imports."""
    path = ops.ROOT / rel
    ops.SOURCES[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(path.read_text())
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            bases, methods = classes[node.name]
            node.bases = [ast.parse(b, mode="eval").body for b in bases]
            node.body = [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AnnAssign))
                and (
                    methods is None
                    or getattr(n, "name", None) in methods
                    or isinstance(n, ast.AnnAssign)
                )
            ]
            if not node.body:
                node.body = [ast.Pass()]
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            body.append(node)
    module = types.ModuleType(name)
    module.__dict__.update(env)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        module.__dict__,
    )
    return module


class Mode:
    def __init__(self, decode):
        self.decode = decode

    def is_idle(self):
        return False

    def is_decode(self):
        return self.decode

    def is_extend(self):
        return not self.decode

    def is_target_verify(self):
        return False

    def is_extend_without_speculative(self):
        return not self.decode


def batch(tokens, lane):
    mode = Mode(tokens == 1)
    positions = torch.arange(4096 - tokens, 4096, device="cuda")
    return NS(
        forward_mode=mode,
        logical_forward_mode=mode,
        positions=positions,
        model_specific_states=None,
        residuals={},
        lane=lane,
        indices=torch.tensor([lane], device="cuda", dtype=torch.int32),
        cu=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
        prefix=torch.tensor([True], device="cuda"),
        mm_input_embeds=None,
        capture_hidden_mode=None,
    )


class Linear(nn.Module):
    """Unsharded BF16 projection with actual learned tensors (no fake outputs)."""

    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(o, i, device="cuda", dtype=torch.bfloat16) / math.sqrt(i)
        )

    def forward(self, x, **kwargs):
        return F.linear(x, self.weight), None


class Boundary:
    """TP=PP=1 transport adapter; all HC reads/updates call production methods."""

    def __init__(self, layer, kind, layer_id):
        self.layer, self.kind, self.key = layer, kind, (layer_id, kind)

    def prepare(self, x, b):
        x = self.layer._widen_streams(x)
        read = self.layer._attn_mix if self.kind == "attn" else self.layer._ffn_mix
        x, residual = read(x)
        b.residuals[self.key] = residual
        return x

    def finish(self, x, b):
        write = (
            self.layer._attn_combine if self.kind == "attn" else self.layer._ffn_combine
        )
        output = write(x, b.residuals.pop(self.key))
        if self.kind == "ffn":
            b.layer_outputs[self.key[0]] = output.clone()
        return output

    @contextlib.contextmanager
    def exit(self, b):
        yield NS(defer_moe_finalize=False, finish=lambda x: self.finish(x, b))


class ResidualBatch:
    @staticmethod
    def start(b):
        b.residuals = {}
        b.layer_outputs = {}

    @staticmethod
    def final_norm(x, b, mixer):
        assert not b.residuals
        return mixer(x)


def modules():
    chunk = ops.bootstrap_fla()
    runtime = types.ModuleType("sglang.srt.runtime_context")
    runtime.get_exec = lambda: NS(
        deterministic=NS(enable_deterministic_inference=False, rl_on_policy_target=None)
    )
    sys.modules[runtime.__name__] = runtime
    arch = lambda: NS(supports_pdl=False)
    hm = ops.load(
        "small_hc_mix", "python/sglang/kernels/ops/gemm/hc_mix.py", current_arch=arch
    )
    fly = ops.load("small_hc_fly", "python/sglang/srt/layers/hc_mix_flydsl.py")
    hc = ops.load(
        "small_hc",
        "python/sglang/srt/layers/hyperconnection.py",
        fused_hc_mix=hm.fused_hc_mix,
        fused_hc_mix_supported=hm.fused_hc_mix_supported,
        flydsl_hc_mix_supported=fly.flydsl_hc_mix_supported,
        flydsl_hc_mix_weights_supported=fly.flydsl_hc_mix_weights_supported,
        weight_cache_key=fly.weight_cache_key,
    )
    common = dict(
        is_arch_support_pdl=lambda: False,
        is_hip=lambda: False,
        is_batch_invariant_mode_enabled=lambda: False,
    )
    conv = ops.load(
        "small_conv",
        "python/sglang/kernels/ops/mamba/causal_conv1d_triton.py",
        is_arch_support_pdl=lambda: False,
    )
    split = ops.load(
        "small_gdn_split",
        "python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py",
        is_hip=lambda: False,
        envs=NS(SGLANG_USE_AITER=NS(get=lambda: False)),
    )
    gating = ops.load(
        "small_gdn_gating",
        "python/sglang/kernels/ops/attention/linear/fused_gdn_gating.py",
    )
    recurrent = ops.load(
        "small_gdn_recurrent",
        "python/sglang/kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py",
        is_cuda=lambda: True,
        is_hip=lambda: False,
        is_gfx95_supported=lambda: False,
        is_sm90_supported=lambda: False,
        get_device_sm=lambda: 89,
        is_arch_support_pdl=lambda: False,
    )
    norm = ops.load(
        "small_gdn_norm",
        "python/sglang/kernels/ops/attention/fla/layernorm_gated.py",
        **common,
        is_cpu=lambda: False,
        is_npu=lambda: False,
        cpu_has_amx_support=lambda: False,
        cdiv=triton.cdiv,
        next_power_of_2=triton.next_power_of_2,
        check_cuda_graph_backend=lambda *a: False,
        Phase=NS(PREFILL=0),
        Backend=NS(TC_PIECEWISE=0),
        device_context=lambda d: torch.cuda.device(d),
    )
    rope = ops.load(
        "small_qk_rope",
        "python/sglang/kernels/ops/attention/fused_qk_rmsnorm_rope_gate.py",
        current_arch=arch,
    )
    elem = ops.load(
        "small_gate", "python/sglang/kernels/ops/elementwise/elementwise.py", **common
    )
    meta = ops.load(
        "small_qsa_meta", "python/sglang/srt/layers/attention/qsa/kernel.py", **common
    )
    sparse = ops.load(
        "small_qsa_sparse",
        "python/sglang/srt/layers/attention/qsa/sparse_attn.py",
        **common,
    )
    mqa = ops.load(
        "small_qsa_mqa",
        "python/sglang/srt/layers/attention/qsa/mqa.py",
        is_hip=lambda: False,
    )
    if not mqa.HAS_TILELANG:
        raise RuntimeError("Real TileLang MQA is required")
    env = dict(
        torch=torch,
        nn=nn,
        triton=triton,
        contextmanager=contextlib.contextmanager,
        dataclass=dataclass,
        _use_aiter=False,
        _is_cuda=True,
        _is_cpu=False,
        _is_npu=False,
        _is_xpu=False,
        _is_hip=False,
        _gdn_decode_fused_proj_conv=False,
        _GDN_FUSED_QKVZBA_RATIOS=(1, 2, 3, 4),
        _fp8_tuple_input=lambda *a: False,
        _gdn_use_alt_stream=False,
        get_is_capture_mode=torch.cuda.is_current_stream_capturing,
        check_cuda_graph_backend=lambda *a: False,
        Phase=NS(PREFILL=0),
        Backend=NS(TC_PIECEWISE=0),
        fused_qkvzba_split_reshape_cat_contiguous=split.fused_qkvzba_split_reshape_cat_contiguous,
        qwen3_5_gdn_prefill_projection_views=split.qwen3_5_gdn_prefill_projection_views,
        fused_qk_gemma_rmsnorm_rope_gate=rope.fused_qk_gemma_rmsnorm_rope_gate,
    )
    base = selected(
        "small_qwen35",
        "python/sglang/srt/models/qwen3_5.py",
        {
            "Qwen3_5GatedDeltaNet": (["nn.Module"], {"forward", "_forward_input_proj"}),
            "Qwen3_5AttentionDecoderLayer": (
                ["nn.Module"],
                {"forward_prepare_cuda_fused", "_prepare_qkv_gate"},
            ),
            "Qwen3_5LinearDecoderLayer": (["nn.Module"], set()),
        },
        env,
    )
    return NS(
        chunk=chunk,
        hc=hc,
        conv=conv,
        split=split,
        gating=gating,
        recurrent=recurrent,
        norm=norm,
        rope=rope,
        elem=elem,
        meta=meta,
        sparse=sparse,
        mqa=mqa,
        base=base,
        env=env,
    )


class GDNCore(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.conv_weight = nn.Parameter(
            torch.randn(1280, 4, device="cuda", dtype=torch.bfloat16) / 4
        )
        self.A = torch.randn(6, device="cuda") * 0.1
        self.bias = torch.randn(6, device="cuda") * 0.1
        self.state = torch.randn(3, 2, 6, 128, 128, device="cuda")[:, 0] * 0.01
        # Keep the envelope slot stride, including after scaling.
        self.state = torch.stack([self.state, self.state], dim=1)[:, 0]
        self.conv_state = (
            torch.randn(3, 2, 4, 1280, device="cuda", dtype=torch.bfloat16).transpose(
                -1, -2
            )[:, 0]
            * 0.01
        )
        self.conv_state = torch.stack([self.conv_state, self.conv_state], dim=1)[:, 0]
        self.initial = self.state.clone(), self.conv_state.clone()

    def reset(self, lane):
        self.state[lane].copy_(self.initial[0][lane])
        self.conv_state[lane].copy_(self.initial[1][lane])

    def forward(self, b, mixed_qkv, a, b_gate):
        if b.forward_mode.is_decode():
            mixed = self.m.conv.causal_conv1d_update(
                mixed_qkv,
                self.conv_state,
                self.conv_weight,
                activation="silu",
                conv_state_indices=b.indices,
            )
        else:
            mixed = self.m.conv.causal_conv1d_fn(
                mixed_qkv.T,
                self.conv_weight,
                None,
                self.conv_state,
                b.cu,
                [mixed_qkv.shape[0]],
                b.indices,
                b.prefix,
                "silu",
            ).T
        q, k, v = self.m.split.fused_qkv_split_gdn_prefill(
            mixed, 2, 2, 6, 128, 128, 128
        )
        if b.forward_mode.is_decode():
            return self.m.recurrent.fused_sigmoid_gating_delta_rule_update(
                self.A,
                a,
                self.bias,
                1.0,
                20.0,
                q,
                k,
                v,
                b_gate,
                self.state,
                b.indices,
                use_qk_l2norm_in_kernel=True,
            )
        g, beta = self.m.gating.fused_gdn_gating(self.A, a, b_gate, self.bias)
        return self.m.chunk(
            q,
            k,
            v,
            g,
            beta,
            initial_state=self.state,
            initial_state_indices=b.indices,
            use_qk_l2norm_in_kernel=True,
        )[0]


def make_gdn(m):
    layer = m.base.Qwen3_5GatedDeltaNet()
    layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim = (
        2,
        6,
        128,
        128,
    )
    layer.attn_tp_size = 1
    layer.alt_stream = None
    layer._fused_in_proj_weight = None
    layer._fused_input_proj_cpu_enabled = NS(value=False)
    layer.in_proj_qkvz = Linear(256, 2048)
    layer.in_proj_ba = Linear(256, 12)
    layer.out_proj = Linear(768, 256)
    layer.norm = m.norm.RMSNorm(
        128, eps=1e-6, norm_before_gate=True, dtype=torch.bfloat16, device="cuda"
    )
    core = GDNCore(m)
    # Parameter name `b` is part of the production attention interface.
    layer.attn = lambda forward_batch, mixed_qkv, a, b: core(
        forward_batch, mixed_qkv, a, b
    )
    return layer, core


def native_topk():
    from functools import lru_cache
    from tvm_ffi.cpp import load_inline

    header = ops.ROOT / "python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh"
    ops.SOURCES[str(header.relative_to(ops.ROOT))] = hashlib.sha256(
        header.read_bytes()
    ).hexdigest()
    source = "\n".join(
        [
            "#include <tvm/ffi/container/tensor.h>",
            "#include <tvm/ffi/dtype.h>",
            "#include <tvm/ffi/error.h>",
            "#include <tvm/ffi/extra/c_env_api.h>",
            "#include <tvm/ffi/function.h>",
            f'#include "{header}"',
            "namespace sglang {",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_topk, (FastTopKKernel<512, false>::run));",
            "}",
        ]
    )
    compiled = load_inline(
        "qwen38_real_topk512_sm89",
        cuda_sources=source,
        extra_cuda_cflags=[
            "-DSGL_CUDA_ARCH=890",
            "-std=c++20",
            "-O3",
            "-allow-unsupported-compiler",
            "-gencode=arch=compute_89,code=sm_89",
        ],
        extra_include_paths=[str(ops.ROOT / "python/sglang/kernels/jit/include")],
    )
    ops.load(
        "sglang.kernels.ops.attention.fast_topk",
        "python/sglang/kernels/ops/attention/fast_topk.py",
        cache_once=lru_cache(None),
        is_arch_support_pdl=lambda: False,
        make_cpp_args=lambda *a: "512, false",
        load_jit=lambda *a, **kw: compiled,
    )


class QSACore(nn.Module):
    """Request-owned cache/metadata adapter; scoring/selection/attention are real."""

    def __init__(self, m, indexer):
        super().__init__()
        self.m, self.indexer = m, indexer
        self.k = torch.randn(3, 4096, 1, 256, device="cuda", dtype=torch.bfloat16) * 0.1
        self.v = torch.randn_like(self.k) * 0.1
        self.compressed = (
            torch.randn(3, 1024, 1, 128, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        self.initial = self.k.clone(), self.v.clone()
        self.index_proj = Linear(256, 8 * 128)

    def reset(self, lane):
        self.k[lane].copy_(self.initial[0][lane])
        self.v[lane].copy_(self.initial[1][lane])

    def select(self, hidden, positions, b):
        q = self.index_proj(hidden)[0].view(-1, 8, 128)
        if b.forward_mode.is_decode():
            result = self.indexer.select_decode_tokens(
                q,
                self.compressed.reshape(-1, 16, 1, 128),
                b.page_table,
                b.compressed_lens,
                1024,
                positions,
                b.seq_lens,
            )
        else:
            result = self.indexer.select_prefill_tokens(
                q,
                self.compressed[b.lane],
                b.row_starts,
                b.row_ends,
                positions,
                b.row_seq_lens,
                max_row_keys=1024,
            )
        b.selected_tokens = result.sort(dim=-1).values
        return result

    def forward(self, q, k, v, b, topk_indices):
        rows = q.shape[0]
        self.k[b.lane, -rows:].copy_(k.view(rows, 1, 256))
        self.v[b.lane, -rows:].copy_(v.view(rows, 1, 256))
        args = (
            q.view(rows, 6, 256),
            self.k[b.lane],
            self.v[b.lane],
            topk_indices,
            b.cu,
            b.kcu,
            b.seq_lens,
            256**-0.5,
        )
        if b.forward_mode.is_decode():
            output = self.m.sparse.sparse_gqa_packed_decode_triton(*args)
        else:
            output = self.m.sparse.sparse_gqa_fwd_interface_triton_ck(*args, max_q=rows)
        return output.reshape(rows, -1)


def build_stack():
    m = modules()
    native_topk()
    index_module = selected(
        "small_qsa_indexer",
        "python/sglang/srt/layers/attention/qsa/qsa_indexer.py",
        {
            "QSAIndexer": (
                ["nn.Module"],
                {"select_prefill_tokens", "select_decode_tokens"},
            )
        },
        dict(
            nn=nn,
            torch=torch,
            HAS_TILELANG=True,
            _QSA_PREFILL_LOGITS_BUDGET_BYTES=64 * 1024 * 1024,
            is_batch_invariant_mode_enabled=lambda: False,
            qsa_mqa_prefill=m.mqa.qsa_mqa_prefill,
            qsa_mqa_prefill_relative=m.mqa.qsa_mqa_prefill_relative,
            qsa_mqa_decode=m.mqa.qsa_mqa_decode,
            qsa_fast_topk=m.meta.qsa_fast_topk,
            expand_qsa_block_indices=m.meta.expand_qsa_block_indices,
        ),
        functions=("_qsa_prefill_row_chunk_size",),
    )
    indexer = index_module.QSAIndexer()
    indexer.token_topk, indexer.block_topk, indexer.compress_ratio = 2048, 512, 4
    qsa_core = QSACore(m, indexer)
    parallel = NS(attn_dp_size=1, attn_cp_size=1, moe_ep_size=1)

    class SparseMoEMarker(nn.Module):
        pass

    env = dict(
        m.env,
        Qwen3_5LinearDecoderLayer=m.base.Qwen3_5LinearDecoderLayer,
        Qwen3_5AttentionDecoderLayer=m.base.Qwen3_5AttentionDecoderLayer,
        Qwen3_5ForCausalLM=nn.Module,
        _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD=0,
        get_disagg=lambda: NS(enable_pdmux=True),
        get_parallel=lambda: parallel,
        get_attn_tp_context=lambda: NS(input_scattered=False),
        get_moe_a2a_backend=lambda: NS(is_none=lambda: True),
        Qwen2MoeSparseMoeBlock=SparseMoEMarker,
        _shards_rows_over_cp=lambda b: False,
        is_in_breakable_cuda_graph=lambda: False,
        get_global_expert_distribution_recorder=lambda: NS(
            with_current_layer=lambda i: contextlib.nullcontext()
        ),
        residual_batch=ResidualBatch,
        AuxHiddenStateList=list,
        _commit_ple_batch=lambda **kw: None,
        fused_sigmoid_mul=m.elem.fused_sigmoid_mul,
    )
    prod = selected(
        "small_qwen4",
        "python/sglang/srt/models/qwen4_exp.py",
        {
            "_MixStreams": ([], None),
            "_HCNorms": ([], None),
            "Qwen4ExpLayerExtensionMixin": (
                [],
                {
                    "hc_norms",
                    "_widen_streams",
                    "_attn_mix",
                    "_attn_combine",
                    "_ffn_mix",
                    "_ffn_combine",
                    "_prepare_attn_stage",
                    "_run_ffn_stage",
                },
            ),
            "Qwen4ExpLinearDecoderLayer": (
                ["Qwen4ExpLayerExtensionMixin", "Qwen3_5LinearDecoderLayer"],
                {"forward"},
            ),
            "Qwen4ExpAttentionDecoderLayer": (
                ["Qwen4ExpLayerExtensionMixin", "Qwen3_5AttentionDecoderLayer"],
                {"forward", "self_attention"},
            ),
            "Qwen4ExpModel": (["nn.Module"], {"forward"}),
        },
        env,
    )
    # Dense FFN's real Qwen2 production orchestration and real PyTorch activation.
    mlp_mod = selected(
        "small_qwen2",
        "python/sglang/srt/models/qwen2.py",
        {"Qwen2MLP": (["nn.Module"], {"forward"})},
        dict(
            nn=nn,
            torch=torch,
            get_exec=lambda: NS(deterministic=NS(rl_on_policy_target=None)),
            can_run_silu_fp4_quant_fusion=lambda *a: False,
        ),
    )
    activation = selected(
        "small_silu",
        "python/sglang/srt/layers/activation.py",
        {"SiluAndMul": (["nn.Module"], {"forward_native"})},
        dict(nn=nn, torch=torch, F=F),
    )
    act = activation.SiluAndMul()
    layers = []
    gdn, gdn_core = make_gdn(m)
    for i, cls in enumerate(
        (prod.Qwen4ExpLinearDecoderLayer, prod.Qwen4ExpAttentionDecoderLayer)
    ):
        layer = cls()
        layer.hc_count, layer.hidden_size, layer.ple = 4, 256, None
        layer._hc_norms, layer._decode_moe_comm = None, None
        cfg = m.hc.HyperConnectionConfig(
            hc_count=4, hidden_size=256, hc_lowrank=16, hc_per_branch_norm=True
        )
        layer.attn_hyper_connection = m.hc.GatedResidual(cfg).to(
            device="cuda", dtype=torch.bfloat16
        )
        layer.mlp_hyper_connection = m.hc.GatedResidual(cfg).to(
            device="cuda", dtype=torch.bfloat16
        )
        layer.attn_boundary, layer.ffn_boundary = (
            Boundary(layer, "attn", i),
            Boundary(layer, "ffn", i),
        )
        mlp = mlp_mod.Qwen2MLP()
        mlp.dual_gemm = NS(can_run=lambda *a: False)
        mlp.gate_up_proj, mlp.down_proj = Linear(256, 1024), Linear(512, 256)
        mlp.act_fn = act.forward_native
        layer.mlp = mlp
        if i == 0:
            layer.linear_attn = gdn
        else:
            layer.is_qsa, layer.alt_stream, layer.attn_output_gate = True, None, True
            layer.num_heads, layer.num_kv_heads, layer.head_dim = 6, 1, 256
            layer.q_size, layer.kv_size = 1536, 256
            layer.qkv_proj, layer.o_proj = Linear(256, 3584), Linear(1536, 256)
            layer.q_norm = NS(
                weight=torch.zeros(256, device="cuda", dtype=torch.bfloat16),
                variance_epsilon=1e-6,
            )
            layer.k_norm = NS(weight=torch.zeros_like(layer.q_norm.weight))
            angle = torch.outer(
                torch.arange(4096, device="cuda"),
                10000 ** (-torch.arange(64, device="cuda") / 64),
            )
            layer.rotary_emb = NS(
                rotary_dim=128,
                cos_sin_cache=torch.cat([angle.cos(), angle.sin()], dim=-1),
            )
            layer._compute_qsa_topk_indices = qsa_core.select
            layer.attn = qsa_core
        layers.append(layer)
    stack = prod.Qwen4ExpModel()
    stack.layers = nn.ModuleList(layers)
    stack.start_layer, stack.end_layer, stack.has_ple = 0, 2, False
    stack.pp_group = NS(is_first_rank=True, is_last_rank=True)
    stack.hyper_connection_mixer = m.hc.GatedResidual(cfg, use_combine=False).to(
        device="cuda", dtype=torch.bfloat16
    )
    return stack, gdn_core, qsa_core


def prepare_batch(tokens, lane):
    b = batch(tokens, lane)
    b.page_table = torch.arange(
        lane * 64, (lane + 1) * 64, device="cuda", dtype=torch.int32
    ).view(1, 64)
    b.compressed_lens = torch.tensor([1024], device="cuda", dtype=torch.int32)
    b.seq_lens = torch.tensor([4096], device="cuda", dtype=torch.int32)
    b.kcu = torch.tensor([0, 4096], device="cuda", dtype=torch.int32)
    b.row_starts = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    b.row_ends = ((b.positions + 1) // 4).int()
    b.row_seq_lens = torch.full((tokens,), 4096, device="cuda", dtype=torch.int32)
    return b


@torch.inference_mode()
def run_decoder(rounds):
    stack, gdn, qsa = build_stack()
    batches = [prepare_batch(t, lane) for lane, t in enumerate((65, 1))]
    inputs = [
        torch.randn(t, 256, device="cuda", dtype=torch.bfloat16) * 0.1 for t in (65, 1)
    ]

    def reset(lane):
        gdn.reset(lane)
        qsa.reset(lane)

    def forward(lane, interval=None):
        b = batches[lane]
        return stack.forward(
            None, b.positions, b, inputs_embeds=inputs[lane], split_interval=interval
        )

    def snapshot(lane, output):
        return tuple(
            t.clone()
            for t in (
                *output,
                gdn.state[lane],
                gdn.conv_state[lane],
                qsa.k[lane],
                qsa.v[lane],
                batches[lane].layer_outputs[0],
                batches[lane].selected_tokens,
            )
        )

    controls = []
    for lane in (0, 1):
        reset(lane)
        controls.append(snapshot(lane, forward(lane)))
    torch.cuda.synchronize()
    untouched = (
        gdn.state[2].clone(),
        gdn.conv_state[2].clone(),
        qsa.k[2].clone(),
        qsa.v[2].clone(),
    )
    ready = torch.cuda.current_stream().record_event()
    ps, ds = torch.cuda.Stream(), torch.cuda.Stream()
    ps.wait_event(ready)
    ds.wait_event(ready)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=ds):
        reset(1)
        dout = forward(1)
    ds.synchronize()
    names = (
        "terminal_mix",
        "hc_output",
        "gdn_state",
        "conv_state",
        "qsa_k",
        "qsa_v",
        "gdn_layer",
        "selected_tokens",
    )
    max_abs = [0.0] * len(names)
    records = []
    for _ in range(rounds):
        with torch.cuda.stream(ps):
            reset(0)
            assert forward(0, (0, 1)) is None
        # Deterministic Python interleaving: decode is submitted between real
        # layers, while the prefill batch retains its own continuation state.
        with torch.cuda.stream(ds):
            graph.replay()
            d = snapshot(1, dout)
            torch.empty(128, 128, device="cuda").fill_(3)
        with torch.cuda.stream(ps):
            p = snapshot(0, forward(0, (1, 2)))
            assert not batches[0].model_specific_states
        records.append((p, d))
    ps.synchronize()
    ds.synchronize()
    failures = {}
    for p, d in records:
        for lane, (actual, control) in enumerate(zip((p, d), controls)):
            for i, (a, c) in enumerate(zip(actual, control)):
                assert torch.isfinite(a).all()
                max_abs[i] = max(max_abs[i], (a.float() - c.float()).abs().max().item())
                # Native top-k output order is unspecified; sparse reductions
                # may differ by BF16 rounding, while state writes must be exact.
                try:
                    torch.testing.assert_close(
                        a, c, atol=0.002 if i < 2 else 0, rtol=0.002 if i < 2 else 0
                    )
                except AssertionError:
                    failures.setdefault(
                        f"lane{lane}.{names[i]}",
                        {
                            "max_abs": max_abs[i],
                            "unequal": int((a != c).sum().item()),
                            "total": a.numel(),
                        },
                    )
    for a, c in zip((gdn.state[2], gdn.conv_state[2], qsa.k[2], qsa.v[2]), untouched):
        torch.testing.assert_close(a, c, atol=0, rtol=0)
    if failures:
        raise AssertionError(f"decoder parity diagnostics: {failures}")
    return dict(
        name="production_gdn_qsa_decoder_stack",
        rounds=rounds,
        decode_cuda_graph=True,
        split_at_layer=1,
        max_abs=max_abs,
        untouched_slots_exact=True,
        hidden_size=256,
        hc_count=4,
        gdn_heads=[2, 6],
        gdn_dim=128,
        qsa_heads=[6, 1],
        qsa_dim=256,
        token_topk=2048,
        context=4096,
        scope="production forwards, real HC/dense FFN/GDN conv+norm/QSA kernels; TP1 transport and fixed compressed-prefix adapters",
    )
