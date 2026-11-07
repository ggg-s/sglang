"""4060 hybrid D2D: native slot allocation, Mooncake TCP and GDN continuation.

Uses small native pool layouts, not the 27B checkpoint. No real RDMA HCA/CQ is
present on this GPU; shared RDMA fault probes are run separately with a fake NIC.
"""

import argparse
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch


def compute(root):
    import flashinfer
    from qwen_hybrid_probe_support import HybridEnvironment
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang_remote_attn import qwen_hybrid as hybrid

    f = HybridEnvironment(root, device="cuda")
    try:
        torch.manual_seed(20261107)
        p, d = f.source, f.target
        m = p.pool.mamba_pool
        # Write on an independent non-default producer stream. Freeze must
        # establish an explicit dependency on this stream before exporting.
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        p.scheduler.tp_worker.model_runner.forward_stream = producer
        with torch.cuda.stream(producer):
            torch.cuda._sleep(300_000_000)
            m.mamba_cache.conv[0].normal_(std=0.1)
            m.mamba_cache.temporal.normal_(std=0.1)
            for tensor in hybrid.kv_buffers(p.pool):
                tensor.normal_(std=0.2)
        tr = f.reserve_and_freeze(growth=2)
        f.import_(tr)
        target = d.d2d_targets[f.epoch]
        req = target.decode_req.req
        assert f.blocked.wait(5) and not target.copied
        assert req.kv.mamba_needs_clear is False
        si = f.req.kv.mamba_pool_idx.reshape(1).int()
        di = req.kv.mamba_pool_idx.reshape(1).int()
        # Run the actual native Conv and packed GDN decode kernels after a
        # state-only import while all full-attention prefix KV remains Remote.
        x = torch.randn((1, 640), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((640, 4), device="cuda", dtype=torch.bfloat16) * 0.1
        a = torch.randn((1, 3), device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = torch.randn(3, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(3, device="cuda", dtype=torch.bfloat16)
        errors = []
        # Compute the uninterrupted reference on private copies: frozen source
        # state itself must stay immutable until the remote read lease closes.
        ref_conv = m.mamba_cache.conv[0].clone()
        ref_ssm = m.mamba_cache.temporal.clone()
        dm = d.pool.mamba_pool
        for li in range(6):
            expected = causal_conv1d_update(
                x.clone(),
                ref_conv[li],
                weight,
                activation="silu",
                conv_state_indices=si,
            )
            actual = causal_conv1d_update(
                x.clone(),
                dm.mamba_cache.conv[0][li],
                weight,
                activation="silu",
                conv_state_indices=di,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            outputs = []
            for mixed, state, indices in (
                (expected, ref_ssm[li], si),
                (actual, dm.mamba_cache.temporal[li], di),
            ):
                out = mixed.new_empty((1, 1, 3, 128))
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed,
                    a=a,
                    b=b,
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=128**-0.5,
                    initial_state=state,
                    out=out,
                    ssm_state_indices=indices,
                    use_qk_l2norm_in_kernel=True,
                )
                outputs.append(out)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            torch.testing.assert_close(
                ref_ssm[li, si.long()],
                dm.mamba_cache.temporal[li, di.long()],
                rtol=0,
                atol=0,
            )
            errors.append((outputs[0] - outputs[1]).abs().max().item())
        # GQA remote prefix + newly decoded local suffix must equal a full KV
        # decode, both before and after the per-entry Local certificate.
        assert f.command(d, "d2d_commit")["admitted"]
        row = req.kv.req_pool_idx
        location = f.native[1].allocator.alloc(1)
        f.native[1].pool.write((row, slice(f.prefix, f.prefix + 1)), location.int())
        req.kv.kv_allocated_len += 1
        req.kv.kv_committed_len += 1
        q = torch.randn((1, 6 * 256), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, 1, 256), device="cuda", dtype=torch.bfloat16)
        k.mul_(0.2)
        v = torch.randn_like(k) * 0.2
        module = NS(
            layer_id=3,
            tp_q_head_num=6,
            tp_k_head_num=1,
            tp_v_head_num=1,
            qk_head_dim=256,
            v_head_dim=256,
            scaling=256**-0.5,
        )
        source_indices = p.scheduler.req_to_token_pool.req_to_token[
            f.req.kv.req_pool_idx, : f.prefix
        ].long()
        full_k = torch.cat([p.pool.get_key_buffer(3)[source_indices], k])
        full_v = torch.cat([p.pool.get_value_buffer(3)[source_indices], v])
        expected, _ = flashinfer.single_decode_with_kv_cache(
            q.reshape(6, 256),
            full_k,
            full_v,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
            sm_scale=256**-0.5,
            return_lse=True,
        )
        repeated_k = full_k.float().repeat_interleave(6, dim=1)
        repeated_v = full_v.float().repeat_interleave(6, dim=1)
        logits = (
            torch.einsum("hd,nhd->hn", q.reshape(6, 256).float(), repeated_k)
            * 256**-0.5
        )
        oracle = torch.einsum("hn,nhd->hd", logits.softmax(-1), repeated_v)
        # Native writer is tested by the common Qwen CUDA probes; here pin down
        # the hybrid prefix/suffix merge and logical full-layer lookup.
        d.pool.get_key_buffer(3)[location.long()] = k
        d.pool.get_value_buffer(3)[location.long()] = v
        fb = NS(
            req_pool_indices=torch.tensor([row], device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor([f.prefix + 1], device="cuda", dtype=torch.int32),
            forward_mode=NS(is_decode=lambda: True),
            mha_return_lse=False,
        )

        def native(*args, **kwargs):
            ids = d.scheduler.req_to_token_pool.req_to_token[row, : f.prefix + 1].long()
            out, _ = flashinfer.single_decode_with_kv_cache(
                q.reshape(6, 256),
                d.pool.get_key_buffer(3)[ids],
                d.pool.get_value_buffer(3)[ids],
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                sm_scale=256**-0.5,
                return_lse=True,
            )
            return out.reshape(1, -1)

        remote_o = hybrid.attention(d, native, module, q, k, v, fb, save_kv_cache=False)
        torch.testing.assert_close(
            remote_o.reshape_as(expected), expected, rtol=0.02, atol=0.001
        )
        torch.testing.assert_close(
            remote_o.reshape_as(oracle).float(), oracle, rtol=0.02, atol=0.001
        )
        assert target.rpc.calls == 1
        f.gate.set()
        f.wait_copy()
        assert target.group_frames == 4 and target.grouped_entries == 4
        for resource in target.resources:
            resource.finish()
        local_o = hybrid.attention(d, native, module, q, k, v, fb, save_kv_cache=False)
        torch.testing.assert_close(
            local_o.reshape_as(expected), expected, rtol=0, atol=0
        )
        assert target.rpc.calls == 1
        assert torch.equal(d.pool.get_key_buffer(3)[location.long()], k)
        target.finish()
        source = p.d2d_sources[f.epoch]
        assert source.poll()
        source.retire()
        assert not f.req.kv.holds_mamba and req.kv.holds_mamba
        return {
            "gdn_layers": 6,
            "conv_max_error": 0.0,
            "ssm_output_max_error": max(errors),
            "gqa_remote_max_error": (remote_o.reshape_as(expected) - expected)
            .abs()
            .max()
            .item(),
            "gqa_local_max_error": (local_o.reshape_as(expected) - expected)
            .abs()
            .max()
            .item(),
            "gqa_fp32_oracle_max_error": (remote_o.reshape_as(oracle).float() - oracle)
            .abs()
            .max()
            .item(),
            "native_mooncake_submissions": len(f.transport.calls),
            "distinct_source_target_slots": [int(si.item()), int(di.item())],
            "layer_group_size": target.group_size,
            "token_range": target.token_range,
            "bulk_group_frames": target.group_frames,
            "max_entries_per_frame": target.grouped_entries,
            "producer_stream_fenced": True,
            "tail_preserved": True,
        }
    finally:
        f.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    torch.cuda.set_device(0)
    sys.path[:0] = [
        str(root / "python"),
        str(root / "plugins/remote_attn"),
        str(root / "test/manual/remote_attn"),
    ]
    # Isolate the optional CLI/testing dependency surface, as other manual
    # source-layout probes do. Production kernel and transport bodies stay real.
    pkg = types.ModuleType("sglang")
    pkg.__path__ = [str(root / "python/sglang")]
    with patch.dict(sys.modules, {"sglang": pkg}):
        result = compute(root)
        stub = types.ModuleType("sglang.test.test_utils")
        stub.CustomTestCase = unittest.TestCase
        with patch.dict(sys.modules, {stub.__name__: stub}):
            spec = importlib.util.spec_from_file_location(
                "hybrid_contracts",
                root / "test/registered/unit/plugins/test_remote_main_contracts.py",
            )
            tests = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tests)
            tests.TestQwenHybridD2D.device = "cuda"
            run = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(
                    tests.TestQwenHybridD2D
                )
            )
            if not run.wasSuccessful():
                raise AssertionError("CUDA hybrid lifecycle regression failed")
    result.update(
        status="PASS",
        device=torch.cuda.get_device_name(),
        lifecycle_tests=run.testsRun,
        full_checkpoint=False,
        real_rdma=False,
        scope=__doc__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
