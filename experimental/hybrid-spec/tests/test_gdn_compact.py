"""Run production GDN bodies with real CUDA kernels and synthetic model inputs.

Only serving imports and the model/pool objects are isolated. These component
checks do not substitute for a Qwen checkpoint/server acceptance run.
"""

import logging
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import MethodType, ModuleType
from types import SimpleNamespace as NS
from typing import Optional, Union

import pytest
import torch
from test_compact_verify import load_module
from test_draft_recurrence import load_functions

GDN = "python/sglang/srt/layers/attention/linear/gdn_backend.py"


@pytest.mark.parametrize("ring_len", [0, 16, 32, 48])
def test_replay_ring_rejects_invalid_sa16_capacity_before_allocation(ring_len):
    ns = {"Optional": Optional, "List": list, "BaseLinearStateParams": object}
    load_functions(
        "python/sglang/srt/mem_cache/memory_pool.py", ["__init__"], ns, "MambaPool"
    )
    # No cache/model/device object is needed: rejection must precede allocation.
    with pytest.raises(ValueError, match="power of two|ring too small"):
        ns["__init__"](
            NS(),
            size=8,
            spec_state_size=8,
            cache_params=None,
            mamba_layer_ids=[],
            device="cpu",
            speculative_num_draft_tokens=17,
            enable_linear_replayssm_spec=True,
            linear_replayssm_cache_len=ring_len,
        )


@pytest.mark.parametrize("mode", ["compact", "cap-accept"])
@pytest.mark.parametrize(
    "gdn,algorithm,verify,topk,allowed",
    [
        (True, "EAGLE", "triton", 1, True),
        (True, "EAGLE", None, 1, True),
        (True, "EAGLE", "flashinfer", 1, False),
        (True, "EAGLE", "triton", 2, False),
        (False, "EAGLE", "triton", 1, False),
        (False, "DSPARK", "triton", 1, True),
        (False, "DSPARK", "flashinfer", 1, False),
    ],
)
def test_replay_ragged_validation(
    monkeypatch, mode, gdn, algorithm, verify, topk, allowed
):
    # Model/config discovery is the external boundary; execute the complete
    # production validation body, including the existing negative guards.
    class Mode(str, Enum):
        STATIC = "static"
        COMPACT = "compact"
        CAP_ACCEPT = "cap-accept"

    for name, values in {
        "sglang.srt.speculative.ragged_verify": {
            "RaggedVerifyMode": Mode,
            "read_ragged_verify_mode": lambda: Mode(mode),
        },
        "sglang.srt.configs.hybrid_arch": {
            "hybrid_gdn_config": lambda _: object() if gdn else None
        },
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
    ns = {
        "Any": object,
        "resolving_view": lambda args: args,
        "get_platform": lambda: NS(is_sm100=False, is_cuda=False),
        "pp_spec_stable_rows_enabled": lambda: False,
        "model_config_of": lambda args: args,
        "uses_kda_attention": lambda _: not gdn,
        "logger": logging.getLogger(__name__),
    }
    load_functions(
        "python/sglang/srt/arg_groups/attention_hook.py",
        ["handle_linear_attn_backend"],
        ns,
    )
    cfg = NS(
        linear_attn_backend="triton",
        linear_attn_decode_backend=None,
        linear_attn_verify_backend=verify,
        linear_attn_prefill_backend=None,
        mamba_ssm_dtype="float32",
        enable_linear_replayssm=False,
        enable_linear_replayssm_spec=True,
        speculative_eagle_topk=topk,
        speculative_algorithm=algorithm,
        disaggregation_mode="null",
    )
    if allowed:
        ns["handle_linear_attn_backend"](cfg)
        assert cfg.mamba_ssm_dtype == "float32"
    else:
        with pytest.raises(ValueError, match="linear draft chain|requires GDN"):
            ns["handle_linear_attn_backend"](cfg)


def test_replay_initializes_compact_graph_slack(monkeypatch):
    name = "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode"
    module = ModuleType(name)

    def kernel(**kwargs):
        # GPU replay only visits intervals covered by query_start_loc. Poison
        # the allocator separately so an unwritten graph tail fails reliably.
        kwargs["out"][: kwargs["query_start_loc"][-1]].fill_(1)

    module.gdn_replayssm_spec_decode = kernel
    monkeypatch.setitem(sys.modules, name, module)
    ns = {
        "torch": torch,
        "RadixLinearAttention": object,
        "MambaPool": NS(SpeculativeState=object),
    }
    load_functions(GDN, ["_replayssm_target_verify"], ns, "GDNAttnBackend")
    monkeypatch.setattr(
        torch.Tensor,
        "new_empty",
        lambda tensor, *shape, **kwargs: torch.full(
            shape, float("nan"), dtype=tensor.dtype
        ),
    )
    result = ns["_replayssm_target_verify"](
        NS(),
        layer=NS(
            num_k_heads=1,
            head_k_dim=2,
            num_v_heads=1,
            head_v_dim=2,
            A_log=None,
            dt_bias=None,
        ),
        query=torch.zeros(8, 1, 2),
        key=torch.zeros(8, 1, 2),
        value=torch.zeros(8, 1, 2),
        a=torch.zeros(8, 1),
        b=torch.zeros(8, 1),
        mamba_pool=NS(
            replayssm_spec_write_pos=None,
            replayssm_cache_base=None,
            replayssm_is_flush=None,
        ),
        layer_cache=NS(
            temporal=None,
            replayssm_d=torch.zeros(2, 1, 64, 2),
            replayssm_k=None,
            replayssm_g=None,
            replayssm_rawv=None,
            replayssm_rawk=None,
            replayssm_beta=None,
        ),
        cache_indices=None,
        replay_indices=None,
        query_start_loc=torch.tensor([0, 3, 4]),
        draft_token_num=17,
        ragged_verify=True,
    )
    assert torch.equal(result[:4], torch.ones_like(result[:4]))
    assert torch.equal(result[4:], torch.zeros_like(result[4:]))


@pytest.mark.parametrize("num_tokens", [0, 13, 16, 18])
def test_text_mrope_preserves_packed_positions(num_tokens):
    ns = {"torch": torch, "ModelRunner": object, "ScheduleBatch": object}
    load_functions(
        "python/sglang/srt/model_executor/forward_batch_info.py",
        ["compute_spec_mrope_positions"],
        ns,
        "ForwardBatch",
    )
    positions = torch.arange(num_tokens, dtype=torch.int64) + 37
    fb = NS(seq_lens=torch.tensor([37, 90]))
    batch = NS(multimodal_inputs=[None, None], spec_info=NS(positions=positions))
    ns["compute_spec_mrope_positions"](fb, NS(device="cpu"), batch)
    torch.testing.assert_close(fb.mrope_positions, positions.repeat(3, 1))


def test_ragged_gdn_disables_uniform_qkv_routing():
    ns = {"ForwardBatch": object}
    load_functions(GDN, ["_init_target_verify_qkv_routing"], ns, "GDNAttnBackend")
    backend = NS(_use_strided_target_verify_qkv=True)
    batch = NS(
        forward_mode=NS(is_target_verify=lambda: True),
        spec_info=NS(ragged_verify_layout=object()),
    )
    ns["_init_target_verify_qkv_routing"](backend, batch)
    assert backend._use_strided_target_verify_qkv is False


CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")


@pytest.fixture
def gdn_cuda_modules(monkeypatch):
    """Isolate serving imports; load the production kernels without alteration."""
    root = Path(__file__).resolve().parents[3] / "python"
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.fla",
        "sglang.kernels.ops.mamba",
        "sglang.kernels.ops.memory",
        "sglang.kernels.jit",
    ):
        module = ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, module)
    sm = torch.cuda.get_device_capability()
    utils = ModuleType("sglang.srt.utils")
    for name, value in {
        "is_cuda": True,
        "is_hip": False,
        "is_gfx95_supported": False,
        "is_sm90_supported": sm[0] == 9,
        "get_device_sm": sm[0] * 10 + sm[1],
        "get_bool_env_var": False,
    }.items():
        setattr(utils, name, lambda *args, value=value: value)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    common = ModuleType("sglang.srt.utils.common")
    common.torch_release = tuple(
        int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
    )
    monkeypatch.setitem(sys.modules, common.__name__, common)
    jit = ModuleType("sglang.kernels.jit.utils")
    jit.is_arch_support_pdl = lambda: sm[0] >= 9
    monkeypatch.setitem(sys.modules, jit.__name__, jit)
    for name in (
        "attention.fla.fused_sigmoid_gating_recurrent",
        "attention.fla.gdn_replayssm_spec_fold",
        "attention.fla.gdn_replayssm_spec_decode",
        "attention.triton_gdn_fused_proj",
        "memory.ptr_table",
        "mamba.causal_conv1d_triton",
        "mamba.mamba_state_scatter_triton",
    ):
        full = "sglang.kernels.ops." + name
        load_module(monkeypatch, full, "python/" + full.replace(".", "/") + ".py")


@CUDA
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_gdn_replay_variable_windows(gdn_cuda_modules, graph, dtype):
    """Run the registered kernel test with the lean serving-import fixture.

    Both the registered test body and backend method are unchanged; only the
    full serving imports are isolated on component-only CUDA installations.
    """
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
        commit_gdn_replayssm_circular,
        commit_gdn_replayssm_spec,
    )

    ns = {
        "torch": torch,
        "RadixLinearAttention": object,
        "MambaPool": NS(SpeculativeState=object),
        "SimpleNamespace": NS,
        "fused_sigmoid_gating_delta_rule_update": fused_sigmoid_gating_delta_rule_update,
        "commit_gdn_replayssm_circular": commit_gdn_replayssm_circular,
        "commit_gdn_replayssm_spec": commit_gdn_replayssm_spec,
    }
    load_functions(GDN, ["_replayssm_target_verify"], ns, "GDNAttnBackend")
    ns["GDNAttnBackend"] = type(
        "Backend", (), {"_replayssm_target_verify": ns["_replayssm_target_verify"]}
    )
    load_functions(
        "test/registered/kernels/ops/attention/test_gdn_replayssm_compact.py",
        ["_run_windows"],
        ns,
        "TestGdnReplayssmCompact",
    )
    ns["_run_windows"](unittest.TestCase(), dtype, graph)


@CUDA
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dedup", [False, True])
@pytest.mark.parametrize("fold", [False, True])
def test_cuda_gdn_packed_verify_commit_and_next_round(
    gdn_cuda_modules, graph, dtype, dedup, fold
):
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
        commit_gdn_replayssm_fold_all_layers,
    )
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        scatter_mamba_states_after_mtp_verify,
    )

    torch.manual_seed(73)
    bs, width, capacity, slots = 4, 9, 16, 8
    heads, value_heads, head_dim, conv_width = 1, 2, 64, 4
    dim = (2 * heads + value_heads) * head_dim

    class Cache(NS):
        pass

    def recurrence(**kwargs):
        kwargs["initial_state_source"] = kwargs.pop("ssm_states")
        kwargs["initial_state_indices"] = kwargs.pop("cache_indices")
        kwargs["cu_seqlens"] = kwargs.pop("query_start_loc")
        kwargs.pop("stable_rows", None)
        return fused_sigmoid_gating_delta_rule_update(
            **kwargs,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

    ns = {
        "torch": torch,
        "Optional": Optional,
        "Tuple": tuple,
        "Union": Union,
        "ForwardBatch": object,
        "RadixLinearAttention": object,
        "MambaPool": NS(SpeculativeState=Cache),
        "MAX_FUSED_QKV_SPLIT_DIM": 8192,
        "_is_hip": False,
        "is_cuda": lambda: True,
        "is_hip": lambda: False,
        "is_xpu": lambda: False,
        "is_cpu": lambda: False,
        "causal_conv1d_update": causal_conv1d_update,
        "fused_qkv_split_gdn_prefill": fused_qkv_split_gdn_prefill,
        # PP is disabled; the production selector returns these same rows.
        "select_verify_intermediate_state_indices": lambda rows, *args: rows,
        "pp_spec_stable_rows_enabled": lambda: False,
    }
    load_functions(
        GDN,
        ["_convolve_target_verify", "forward_extend", "_replayssm_fold_target_verify"],
        ns,
        "GDNAttnBackend",
    )
    layer = NS(
        layer_id=0,
        q_dim=heads * head_dim,
        k_dim=heads * head_dim,
        v_dim=value_heads * head_dim,
        num_q_heads=heads,
        num_k_heads=heads,
        num_v_heads=value_heads,
        head_q_dim=head_dim,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        conv_weights=torch.randn(dim, conv_width, device="cuda") * 0.1,
        bias=torch.randn(dim, device="cuda") * 0.1,
        activation="silu",
        A_log=torch.zeros(value_heads, device="cuda"),
        dt_bias=torch.zeros(value_heads, device="cuda"),
    )
    initial_conv = torch.randn(slots, conv_width - 1, dim, dtype=dtype, device="cuda")
    initial_ssm = (
        torch.randn(slots, value_heads, head_dim, head_dim, device="cuda") * 0.1
    )

    def make_backend(ragged):
        if dedup:
            storage = torch.zeros(
                bs + 1, width + conv_width - 2, dim, device="cuda", dtype=dtype
            )
            windows = storage.as_strided(
                (bs + 1, width, dim, conv_width - 1),
                (storage.stride(0), storage.stride(1), 1, storage.stride(1)),
            )
        else:
            windows = torch.zeros(
                bs + 1, width, conv_width - 1, dim, device="cuda", dtype=dtype
            ).transpose(-1, -2)
        cache = Cache(
            conv=[initial_conv.clone().transpose(-1, -2)],
            temporal=initial_ssm.clone(),
            intermediate_ssm=torch.zeros(
                bs + 1, width, value_heads, head_dim, head_dim, device="cuda"
            ),
            intermediate_conv_window=[windows],
            replayssm_rawv=torch.zeros(
                slots, value_heads, width, head_dim, device="cuda", dtype=dtype
            )
            if fold
            else None,
            replayssm_rawk=torch.zeros(
                slots, heads, width, head_dim, device="cuda", dtype=dtype
            ),
            replayssm_g=torch.zeros(slots, value_heads, width, device="cuda"),
            replayssm_beta=torch.zeros(slots, value_heads, width, device="cuda"),
            replayssm_d=None,
        )
        qsl = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
        indices = torch.full((bs,), -1, dtype=torch.int32, device="cuda")
        backend = NS(
            mis_metadata=None,
            requires_contiguous_prefill_state=True,
            _use_strided_target_verify_qkv=False,
            verify_intermediate_state_indices=torch.arange(
                bs, device="cuda", dtype=torch.int32
            ),
            req_to_token_pool=NS(
                size=slots,
                mamba_pool=NS(replayssm_spec_fold=fold),
                mamba2_layer_cache=lambda _: cache,
            ),
            forward_metadata=NS(
                query_start_loc=qsl,
                mamba_cache_indices=indices,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                retrieve_parent_token=None,
            ),
            kernel_dispatcher=NS(
                tree_verify_kernel=NS(target_verify=recurrence),
                target_verify=recurrence,
            ),
        )
        backend._convolve_target_verify = MethodType(
            ns["_convolve_target_verify"], backend
        )
        backend._replayssm_fold_target_verify = MethodType(
            ns["_replayssm_fold_target_verify"], backend
        )
        backend._replayssm_fold_uses_cutedsl = lambda *args: False
        fb = NS(
            forward_mode=NS(is_target_verify=lambda: True),
            spec_info=NS(
                draft_token_num=width, ragged_verify_layout=object() if ragged else None
            ),
            req_pool_indices=torch.arange(bs, device="cuda"),
        )
        return backend, fb, cache

    packed, packed_fb, packed_cache = make_backend(True)
    dense, dense_fb, dense_cache = make_backend(False)
    dense.forward_metadata.query_start_loc.copy_(
        torch.arange(0, (bs + 1) * width, width, device="cuda")
    )
    x = torch.zeros(capacity, dim, dtype=dtype, device="cuda")
    a = torch.zeros(capacity, value_heads, dtype=dtype, device="cuda")
    b = torch.zeros_like(a)

    def run():
        return ns["forward_extend"](packed, layer, packed_fb, x, a, b)

    # One graph, stable buffers; later replays change lengths and request order.
    packed.forward_metadata.query_start_loc.copy_(
        torch.tensor([0, 9, 13, 13, 13], device="cuda")
    )
    packed.forward_metadata.mamba_cache_indices.copy_(
        torch.tensor([3, 1, -1, -1], device="cuda")
    )
    run()
    if graph:
        torch.cuda.synchronize()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            out = run()

    for lens, mapping in (
        ([9, 4, 0, 0], [3, 1, -1, -1]),
        ([4, 9, 0, 0], [1, 3, -1, -1]),
        ([8, 3, 1, 0], [3, 1, 5, -1]),
    ):
        # Continue from the previous partial commit; stale scratch is deliberate.
        dense_cache.conv[0].copy_(packed_cache.conv[0])
        dense_cache.temporal.copy_(packed_cache.temporal)
        before_ssm = packed_cache.temporal.clone()
        dense_x = torch.randn(bs, width, dim, device="cuda", dtype=dtype)
        dense_a = torch.randn(bs, width, value_heads, device="cuda", dtype=dtype)
        dense_b = torch.randn_like(dense_a)
        x.fill_(float("nan"))
        a.zero_()
        b.zero_()
        offset = 0
        for row, length in enumerate(lens):
            x[offset : offset + length] = dense_x[row, :length]
            a[offset : offset + length] = dense_a[row, :length]
            b[offset : offset + length] = dense_b[row, :length]
            offset += length
        qsl = torch.tensor(
            [0] + list(torch.tensor(lens).cumsum(0).tolist()),
            device="cuda",
            dtype=torch.int32,
        )
        indices = torch.tensor(mapping, device="cuda", dtype=torch.int32)
        packed.forward_metadata.query_start_loc.copy_(qsl)
        packed.forward_metadata.mamba_cache_indices.copy_(indices)
        dense.forward_metadata.mamba_cache_indices.copy_(indices)
        if graph:
            capture.replay()
        else:
            out = run()
        expected = ns["forward_extend"](
            dense,
            layer,
            dense_fb,
            dense_x.flatten(0, 1),
            dense_a.flatten(0, 1),
            dense_b.flatten(0, 1),
        ).view(bs, width, value_heads, head_dim)
        torch.testing.assert_close(packed_cache.temporal, before_ssm, rtol=0, atol=0)
        offset = 0
        for row, length in enumerate(lens):
            if not length:
                continue
            torch.testing.assert_close(
                out[0, offset : offset + length], expected[row, :length], rtol=0, atol=0
            )
            if fold:
                for name in (
                    "replayssm_rawv",
                    "replayssm_rawk",
                    "replayssm_g",
                    "replayssm_beta",
                ):
                    torch.testing.assert_close(
                        getattr(packed_cache, name)[mapping[row], ..., :length, :]
                        if name in ("replayssm_rawv", "replayssm_rawk")
                        else getattr(packed_cache, name)[mapping[row], ..., :length],
                        getattr(dense_cache, name)[mapping[row], ..., :length, :]
                        if name in ("replayssm_rawv", "replayssm_rawk")
                        else getattr(dense_cache, name)[mapping[row], ..., :length],
                        rtol=0,
                        atol=0,
                    )
            else:
                torch.testing.assert_close(
                    packed_cache.intermediate_ssm[row, :length],
                    dense_cache.intermediate_ssm[row, :length],
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                packed_cache.intermediate_conv_window[0][row, :length],
                dense_cache.intermediate_conv_window[0][row, :length],
                rtol=0,
                atol=0,
            )
            offset += length
        # Commit a short prefix and a separate radix tracking checkpoint using
        # the production scatter; following rounds must start at those states.
        steps = torch.tensor(
            [max(0, n // 2) if n else -1 for n in lens],
            device="cuda",
            dtype=torch.int32,
        )
        tracking = torch.tensor([6, 7, -1, -1], device="cuda", dtype=torch.int32)
        track_steps = torch.tensor([0, 1, -1, -1], device="cuda", dtype=torch.int32)
        for cache in (packed_cache, dense_cache):
            if fold:
                commit_gdn_replayssm_fold_all_layers(
                    checkpoint_state=cache.temporal.unsqueeze(0),
                    rawv_cache=cache.replayssm_rawv.unsqueeze(0),
                    rawk_cache=cache.replayssm_rawk.unsqueeze(0),
                    g_cache=cache.replayssm_g.unsqueeze(0),
                    beta_cache=cache.replayssm_beta.unsqueeze(0),
                    ssm_state_indices=indices,
                    accept_lens=steps + 1,
                    max_cache_len=width,
                    num_k_heads=heads,
                    mamba_track_indices=tracking,
                    mamba_steps_to_track=track_steps,
                    null_block_id=-1,
                )
            scatter_mamba_states_after_mtp_verify(
                NS(
                    temporal=cache.temporal.new_empty(1, 0)
                    if fold
                    else cache.temporal.unsqueeze(0),
                    intermediate_ssm=cache.intermediate_ssm.unsqueeze(0),
                    conv=[cache.conv[0].transpose(-1, -2).unsqueeze(0)],
                    intermediate_conv_window=[
                        cache.intermediate_conv_window[0].transpose(-1, -2).unsqueeze(0)
                    ],
                ),
                indices,
                steps,
                tracking,
                track_steps,
            )
        torch.testing.assert_close(
            packed_cache.temporal, dense_cache.temporal, rtol=0, atol=0
        )
        torch.testing.assert_close(
            packed_cache.conv[0], dense_cache.conv[0], rtol=0, atol=0
        )
