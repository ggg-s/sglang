"""Compact online C128 plan ABI, prefix banks and conditional replay state."""

import __future__
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, ModuleType
from typing import NamedTuple
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.online_c128_mtp import OnlineC128MTPController
from sglang.kernels.ops.attention.dsv4.online_c128_ragged import (
    commit_ragged_pending,
    plan_online_verify,
    write_ragged_prefix_states,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestOnlineC128Ragged(CustomTestCase):
    def test_device_metadata_bypasses_cpu_lengths_and_masks_query_padding(self):
        root = Path(__file__).resolve().parents[5]
        tree = ast.parse(
            (
                root / "python/sglang/srt/layers/attention/deepseek_v4_backend.py"
            ).read_text()
        )
        raw = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DSV4RawVerifyMetadata"
        )
        backend = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepseekV4AttnBackend"
        )
        methods = [
            n
            for n in backend.body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in (
                "init_forward_metadata_target_verify",
                "_make_target_verify_c128_metadata",
            )
        ]
        plan_tree = ast.parse(
            (root / "python/sglang/kernels/ops/attention/dsv4/compress.py").read_text()
        )
        plan_class = next(
            n
            for n in plan_tree.body
            if isinstance(n, ast.ClassDef) and n.name == "CompressorPrefillPlan"
        )
        ns = dict(torch=torch, dataclass=dataclass, NamedTuple=NamedTuple)
        exec(
            compile(
                ast.Module(body=[plan_class, raw, *methods], type_ignores=[]),
                "<dsv4-metadata>",
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            ns,
        )
        module = ModuleType("sglang.kernels.ops.attention.dsv4.compress")
        module.CompressorPrefillPlan = ns["CompressorPrefillPlan"]
        seq, lens, req, _, _, _ = self.inputs()
        target = NS(
            needs_cpu_seq_lens=True,
            has_c128=True,
            speculative_num_draft_tokens=9,
            online_c128_mtp=NS(enabled=lambda: True),
            extend_seq_lens_buffer=torch.empty(4, dtype=torch.int32),
            extend_start_loc_buffer=torch.empty(4, dtype=torch.int32),
        )
        target._make_target_verify_c128_metadata = MethodType(
            ns["_make_target_verify_c128_metadata"], target
        )
        layout = NS(
            verify_lens=torch.tensor([4, 9, 2, 3]),
            extend_start_loc=torch.tensor([0, 4, 13, 15]),
            graph_num_tokens=18,
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            metadata = ns["init_forward_metadata_target_verify"](
                target,
                512,
                req,
                seq,
                seq_lens_cpu=None,
                out_cache_loc=torch.zeros(18),
                ragged_layout=layout,
                online_verify_lens=lens,
                online_c128_state_slot_offset=5,
            )
        self.assertIsNone(metadata.seq_lens_cpu)
        self.assertEqual(metadata.total_verify_tokens, 18)
        self.assertEqual(metadata.verify_lens.tolist(), [4, 9, 2, 3])
        actual = metadata.c128_compress_metadata.plan_w.view(torch.int32)
        self.assertEqual(
            actual[actual[:, 0] >= 0, 1].tolist(),
            [3 | (2 << 16), 12 | (9 << 16), 14 | (1 << 16)],
        )

    def inputs(self, dtype=torch.float32):
        torch.manual_seed(312)
        # Request zero is live; padded rows must not overwrite its pending slot.
        seq = torch.tensor([126, 128, 255, 1], dtype=torch.int64)
        lens = torch.tensor([4, 9, 2, 0], dtype=torch.int32)
        req = torch.tensor([0, 3, 1, -1], dtype=torch.int64)
        state = torch.randn(5 * 10, 3 * 512).to(dtype)
        state[:, 512:1024].abs_().add_(1)
        kv = torch.randn(18, 1024)
        ape = torch.randn(128, 512)
        return seq, lens, req, state, kv, ape

    def test_plan_abi_cpu(self):
        seq, lens, req, _, _, _ = self.inputs()
        c, w = plan_online_verify(seq, lens, req, 18, 5)
        c, w = c.view(torch.int32), w.view(torch.int32)
        self.assertEqual(
            c[c[:, 0] >= 0].tolist(),
            [[128, 1 | (2 << 16), 5, 0], [256, 13 | (1 << 16), 6, 1]],
        )
        self.assertEqual(
            w[w[:, 0] >= 0].tolist(),
            [
                [130, 3 | (2 << 16), 5, 0],
                [137, 12 | (9 << 16), 8, 3],
                [257, 14 | (1 << 16), 6, 1],
            ],
        )

    def test_cuda_plans_and_state_banks_match_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("requires NVIDIA CUDA")
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                seq, lens, req, state, kv, ape = self.inputs(dtype)
                gpu = [x.cuda() for x in (seq, lens, req, state, kv, ape)]
                gs, gl, gr, gst, gkv, ga = gpu
                expected_plan = plan_online_verify(seq, lens, req, 18, 5)
                actual_plan = plan_online_verify(gs, gl, gr, 18, 5)
                for expected, actual in zip(expected_plan, actual_plan):
                    torch.testing.assert_close(actual.cpu(), expected)
                expected = state.clone()
                write_ragged_prefix_states(kv, seq, req, lens, ape, expected, 5)
                write_ragged_prefix_states(gkv, gs, gr, gl, ga, gst, 5)
                torch.testing.assert_close(
                    gst.cpu().float(),
                    expected.float(),
                    atol=0.025 if dtype == torch.bfloat16 else 2e-5,
                    rtol=0.01 if dtype == torch.bfloat16 else 2e-5,
                )
                pending = torch.tensor([126, 255, -1, 128, -1])
                current = seq + torch.tensor([1, 5, 2, 0])
                commit_ragged_pending(current, req, pending, expected, 9, 5)
                commit_ragged_pending(current.cuda(), gr, pending.cuda(), gst, 9, 5)
                torch.testing.assert_close(
                    gst.cpu().float(),
                    expected.float(),
                    atol=0.025 if dtype == torch.bfloat16 else 2e-5,
                    rtol=0.01 if dtype == torch.bfloat16 else 2e-5,
                )

    def controller(self, device):
        seq, lens, req, state, kv, ape = [x.to(device) for x in self.inputs()]
        pending = torch.full((5,), 111, dtype=torch.int64, device=device)
        pool = NS(
            get_online_c128_mtp_pending_seq_lens=lambda: pending,
            get_online_c128_mtp_max_draft_tokens=lambda: 9,
            get_online_c128_mtp_state_slot_offset=lambda: 5,
            get_attention_compress_states=lambda layer: NS(
                kv_score_buffer=NS(kv_score=state), online_mtp_state_slot_offset=5
            ),
        )
        backend = NS(
            mtp_enabled=True,
            speculative_num_draft_tokens=9,
            token_to_kv_pool=pool,
            req_to_token=None,
        )
        controller = OnlineC128MTPController(backend)
        controller.enabled = lambda: True
        controller._layer_runtimes = [
            NS(
                head_dim=512,
                main_state=state,
                state_dtype=state.dtype,
                state_slot_offset=5,
            )
        ]
        controller.enable_ragged_verify()
        return controller, seq, lens, req, state, kv, ape, pending

    def test_replay_commit_gate_consumes_pending_once(self):
        controller, seq, lens, req, state, kv, ape, pending = self.controller("cpu")
        for _ in range(3):
            controller.prepare_forward(
                NS(is_idle=lambda: False, is_target_verify=lambda: True),
                req,
                seq,
                verify_lens=lens,
            )
            self.assertEqual(pending.tolist(), [126, 255, -1, 128, -1])
            write_ragged_prefix_states(kv, seq, req, lens, ape, state, 5)
            controller.arm_ragged_verify(req, seq, lens)
            current = seq + torch.tensor([1, 5, 2, 0])
            controller.commit_pending(req_pool_indices=req, seq_lens=current)
            self.assertTrue((pending == -1).all())
            saved = state.clone()
            controller.commit_pending(req_pool_indices=req, seq_lens=current + 1)
            torch.testing.assert_close(state, saved)

    def test_cuda_controller_capture_replays_changed_lengths(self):
        if not torch.cuda.is_available():
            self.skipTest("requires NVIDIA CUDA")
        controller, seq, lens, req, state, kv, ape, pending = self.controller("cuda")
        mode = NS(is_idle=lambda: False, is_target_verify=lambda: True)

        def body():
            controller.prepare_forward(mode, req, seq, verify_lens=lens)
            controller.write_prefix_states(
                layer_id=0,
                compressor=NS(is_in_indexer=False, ratio=128, head_dim=512, ape=ape),
                kv_score_input=kv,
                logical_forward_mode=mode,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
        try:
            for lengths in ([4, 9, 2, 0], [1, 3, 1, 0], [9, 2, 0, 0]):
                lens.copy_(torch.tensor(lengths, device="cuda"))
                controller.clear()
                initial = state.cpu()
                expected = initial.clone()
                write_ragged_prefix_states(
                    kv.cpu(), seq.cpu(), req.cpu(), lens.cpu(), ape.cpu(), expected, 5
                )
                controller.arm_ragged_verify(req, seq, lens)
                graph.replay()
                current = seq + lens.clamp(max=2)
                cpu_pending = torch.full((5,), -1, dtype=torch.int64)
                for r, s, length in zip(
                    req.cpu().tolist(), seq.cpu().tolist(), lengths
                ):
                    if r >= 0 and length:
                        cpu_pending[r] = s
                commit_ragged_pending(
                    current.cpu(), req.cpu(), cpu_pending, expected, 9, 5
                )
                controller.commit_pending(req_pool_indices=req, seq_lens=current)
                torch.testing.assert_close(state.cpu(), expected, atol=2e-5, rtol=2e-5)
                self.assertTrue((pending.cpu() == -1).all())
        finally:
            torch.cuda.synchronize()
            graph.reset()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__]))
