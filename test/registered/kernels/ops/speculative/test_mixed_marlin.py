"""Shape-stable FP16 projections for the offline Qwen3.5 AWQ mixed path."""

import importlib
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.speculative.mixed_marlin import fixed_linear, stable_marlin
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=200, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestMixedMarlin(unittest.TestCase):
    def test_graph_buffer_removal_and_dtype_change_are_rejected(self):
        from sglang.srt.speculative.mixed_graph import (
            MixedGraphLayoutChanged,
            _copy_batch,
        )

        target = SimpleNamespace(ids=torch.zeros(3, dtype=torch.int64))
        for source in [
            SimpleNamespace(ids=None),
            SimpleNamespace(ids=torch.zeros(3, dtype=torch.int32)),
        ]:
            with self.assertRaises(MixedGraphLayoutChanged):
                _copy_batch(target, source)
        self.assertEqual(target.ids.dtype, torch.int64)

    def test_graph_rejects_host_branch_and_nested_buffer_changes_before_copy(self):
        from sglang.srt.speculative.mixed_graph import (
            MixedGraphLayoutChanged,
            _copy_batch,
        )

        target = SimpleNamespace(
            ids=torch.zeros(3, dtype=torch.int64),
            return_logprob=False,
            spec_info=SimpleNamespace(topk=1, positions=torch.zeros(3)),
        )
        changed = [
            SimpleNamespace(
                ids=torch.ones(3, dtype=torch.int64),
                return_logprob=True,
                spec_info=target.spec_info,
            ),
            SimpleNamespace(
                ids=torch.ones(3, dtype=torch.int64),
                return_logprob=False,
                spec_info=SimpleNamespace(topk=2, positions=torch.zeros(3)),
            ),
            SimpleNamespace(
                ids=torch.ones(3, dtype=torch.int64),
                return_logprob=False,
                spec_info=SimpleNamespace(topk=1, positions=None),
            ),
            SimpleNamespace(return_logprob=False, spec_info=target.spec_info),
        ]
        for source in changed:
            with self.assertRaises(MixedGraphLayoutChanged):
                _copy_batch(target, source)
            self.assertEqual(target.ids.tolist(), [0, 0, 0])
            self.assertEqual(target.spec_info.topk, 1)

    def test_fp16_projection_is_invariant_to_prefill_padding_and_joint_rows(self):
        torch.manual_seed(42)
        w = torch.randn(256, 2048, device="cuda", dtype=torch.float16) * 0.02
        d = torch.randn(24, 2048, device="cuda", dtype=torch.float16)
        separate = fixed_linear(d, w)
        for n, capacity in [(23, 32), (65, 128), (257, 512), (512, 512)]:
            p = torch.randn(n, 2048, device="cuda", dtype=torch.float16)
            joined = torch.cat([p, p.new_zeros(capacity - n, 2048), d])
            combined = fixed_linear(joined, w)
            self.assertTrue(torch.equal(combined[:n], fixed_linear(p, w)))
            self.assertTrue(torch.equal(combined[capacity:], separate))
            torch.testing.assert_close(
                combined, torch.nn.functional.linear(joined, w), rtol=0.02, atol=0.002
            )

    def test_graph_replay_reads_new_inputs_without_changing_reduction(self):
        x = torch.randn(56, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(64, 128, device="cuda", dtype=torch.float16) * 0.02
        fixed_linear(x, w)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fixed_linear(x, w)
        for scale in [0.5, 1.0, 2.0]:
            x.mul_(scale)
            graph.replay()
            self.assertTrue(torch.equal(output, fixed_linear(x, w)))

    def test_projection_tiles_match_legacy_across_padding_and_changed_graph_inputs(
        self,
    ):
        torch.manual_seed(43)
        # Exercise wide, medium and router tiles, including non-tile-aligned P.
        for width in (9216, 2048, 256):
            w = torch.randn(width, 128, device="cuda", dtype=torch.float16) * 0.02
            for n, capacity in ((23, 32), (65, 128), (257, 512), (511, 512)):
                p = torch.randn(n, 128, device="cuda", dtype=torch.float16)
                d = torch.randn(24, 128, device="cuda", dtype=torch.float16)

                def mixed():
                    joined = torch.cat((p, p.new_zeros(capacity - n, 128), d))
                    return fixed_linear(joined, w, tile="projection")

                mixed()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = mixed()
                for scale in (1.0, 0.5):
                    p.mul_(scale)
                    d.mul_(scale)
                    graph.replay()
                    self.assertTrue(torch.equal(output[:n], fixed_linear(p, w)))
                    self.assertTrue(torch.equal(output[capacity:], fixed_linear(d, w)))
                    self.assertTrue(
                        torch.equal(
                            fixed_linear(p, w, tile="projection"), fixed_linear(p, w)
                        )
                    )

    def test_full_k_larger_blocks_preserve_graph_outputs_and_adapter_scope(self):
        from sgl_kernel.scalar_type import scalar_types

        from sglang.srt.layers.moe.topk import fused_topk
        from sglang.test.test_marlin_utils import awq_marlin_quantize

        torch.manual_seed(44)
        marlin = importlib.import_module(
            "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
        )
        banks = []
        for k, out in ((256, 256), (128, 256)):
            rows = [
                awq_marlin_quantize(
                    torch.randn(k, out, device="cuda", dtype=torch.float16) * 0.02,
                    scalar_types.uint4,
                    128,
                )[1:]
                for _ in range(8)
            ]
            banks.append(tuple(torch.stack(items).contiguous() for items in zip(*rows)))
        (w1, s1, z1), (w2, s2, z2) = banks
        p = torch.randn(65, 256, device="cuda", dtype=torch.float16)
        d = torch.randn(24, 256, device="cuda", dtype=torch.float16)
        router = torch.randn(8, 256, device="cuda", dtype=torch.float16) * 0.02
        model = torch.nn.Module()
        original = marlin.fused_marlin_moe

        def run(x):
            logits = fixed_linear(x, router)
            scores, ids = fused_topk(x, logits, 2, True)
            return marlin.fused_marlin_moe(
                x,
                w1,
                w2,
                s1,
                s2,
                logits,
                scores,
                ids,
                w1_zeros=z1,
                w2_zeros=z2,
                num_bits=4,
            )

        for block in (16, 32):
            with stable_marlin(model, moe_block_size=block):

                def mixed():
                    return run(torch.cat((p, p.new_zeros(63, 256), d)))

                mixed()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = mixed()
            self.assertIs(marlin.fused_marlin_moe, original)
            for scale in (1.0, 0.5, 2.0):
                p.mul_(scale)
                d.mul_(scale)
                graph.replay()
                with stable_marlin(model):
                    expected_p, expected_d = run(p), run(d)
                with stable_marlin(model, moe_block_size=block):
                    separate_p, separate_d = run(p), run(d)
                for actual, expected in (
                    (output[:65], expected_p),
                    (output[128:], expected_d),
                    (separate_p, expected_p),
                    (separate_d, expected_d),
                ):
                    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)

    def test_target_only_adapter_restores_methods_on_failure(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(1024, 128, device="cuda", dtype=torch.float16)
                )
                self.quant_method = UnquantizedLinearMethod()

            def forward(self, x):
                return self.quant_method.apply(self, x)

        layer = Linear()
        original = layer.quant_method.apply
        pointer = layer.weight.data_ptr()
        marlin = importlib.import_module(
            "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
        )
        triton_moe = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
        )
        old_marlin, old_triton = (
            marlin.is_batch_invariant_mode_enabled,
            triton_moe.is_batch_invariant_mode_enabled,
        )
        old_fused = marlin.fused_marlin_moe
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with stable_marlin(
                layer, linear_tile="projection", moe_block_size=16
            ) as counts:
                layer(torch.ones(3, 128, device="cuda", dtype=torch.float16))
                self.assertEqual(counts["fp16_linear"], 1)
                raise RuntimeError("stop")
        self.assertEqual(layer.quant_method.apply, original)
        self.assertIs(marlin.is_batch_invariant_mode_enabled, old_marlin)
        self.assertIs(marlin.fused_marlin_moe, old_fused)
        self.assertIs(triton_moe.is_batch_invariant_mode_enabled, old_triton)
        self.assertEqual(layer.weight.data_ptr(), pointer)
        self.assertEqual(layer.weight.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
