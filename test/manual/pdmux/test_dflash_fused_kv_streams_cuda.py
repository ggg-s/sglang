"""Real DFlash fused KV projection/norm/RoPE with concurrent lane consumers."""

from types import SimpleNamespace as NS

import pytest
import torch

from sglang.kernels.ops.speculative.fused_kv_materialize import FusedKVMaterializeHelper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_materialization_scratch_is_private_to_each_cuda_stream():
    torch.manual_seed(71)
    tokens, hidden, head_dim = 512, 64, 64
    device = torch.device("cuda", 0)
    rope = NS(
        rotary_dim=head_dim,
        is_neox_style=True,
        cos_sin_cache=torch.cat(
            [
                torch.ones(tokens, head_dim // 2, device=device),
                torch.zeros(tokens, head_dim // 2, device=device),
            ],
            dim=1,
        ),
    )
    layers = [
        NS(
            self_attn=NS(
                num_kv_heads=1,
                head_dim=head_dim,
                q_size=head_dim,
                kv_size=head_dim,
                rotary_emb=rope,
                qkv_proj=NS(
                    weight=torch.randn(
                        3 * head_dim, hidden, device=device, dtype=torch.float16
                    )
                ),
                k_norm=NS(
                    weight=torch.ones(head_dim, device=device, dtype=torch.float16),
                    variance_epsilon=1e-6,
                ),
            )
        )
        for _ in range(2)
    ]
    helper = FusedKVMaterializeHelper(
        layers, rope, 1, head_dim, device, max_position_hint=tokens
    )
    a = torch.randn(tokens, hidden, device=device, dtype=torch.float16)
    b = -3 * a
    positions = torch.arange(tokens, device=device)
    context = torch.cuda.GreenContext.create(num_sms=8)
    prefill, decode = context.Stream(), torch.cuda.Stream()
    torch.cuda.synchronize()

    def materialize(stream, values, *, delayed=False):
        outputs = []

        def store(layer_id, k, v):
            if delayed:
                # Make the other lane's overwrite happen before this queued
                # consumer. Real KV writes also consume these scratch views.
                torch.cuda._sleep(300_000_000)
            outputs.append((k.clone(), v.clone()))

        with torch.cuda.stream(stream):
            helper.materialize(values, positions, store)
        return outputs

    expected_a = materialize(prefill, a)
    torch.cuda.synchronize()
    expected_b = materialize(decode, b)
    torch.cuda.synchronize()
    for _ in range(3):
        actual_a = materialize(prefill, a, delayed=True)
        actual_b = materialize(decode, b)
        torch.cuda.synchronize()
        for actual, expected in ((actual_a, expected_a), (actual_b, expected_b)):
            for (k, v), (ref_k, ref_v) in zip(actual, expected):
                torch.testing.assert_close(k, ref_k, rtol=0, atol=0)
                torch.testing.assert_close(v, ref_v, rtol=0, atol=0)
