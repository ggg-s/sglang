"""DFlash's direct commit must keep request rings and physical state slots distinct."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")


class TestGdnReplayssmDirectCommit(unittest.TestCase):
    def test_capacity_fold_track_and_conv_rollback(self):
        torch.manual_seed(42)
        layers, slots, requests, hv, h, dim, length, drafts = 2, 8, 4, 4, 2, 128, 32, 8

        def rand(*shape):
            return torch.randn(*shape, device="cuda") * 0.02

        def ints(values):
            return torch.tensor(values, dtype=torch.int32, device="cuda")

        # Nonidentity state translation and ring wrap reproduce the PDMux contract.
        state_ids, replay_ids = ints([5, 2]), ints([1, 3])
        track_virtual, track_physical = ints([5, 6]), [6, 7]
        last_steps, track_steps = ints([0, 7]), ints([0, 2])
        initial = rand(layers, slots, hv, dim, dim)
        full_d = rand(layers, requests, hv, length, dim)
        full_k = rand(layers, requests, h, length, dim)
        high_d, high_k = full_d.half(), full_k.half()
        low_d = (full_d - high_d.float()).half()
        low_k = (full_k - high_k.float()).half()
        gates = -torch.rand(layers, requests, hv, length, device="cuda") * 0.03
        conv = rand(layers, slots, 16, 3).half()
        windows = rand(layers, slots, drafts, 16, 3).half()
        spec_state = SimpleNamespace(
            temporal=initial.clone(),
            replayssm_d=high_d,
            replayssm_k=high_k,
            replayssm_rawv=low_d,
            replayssm_rawk=low_k,
            replayssm_g=gates,
            conv=[conv.clone()],
            intermediate_conv_window=[windows],
        )
        write_pos, base = ints([0, 5, 0, 16]), ints([0, 29, 0, 21])
        pool = SimpleNamespace(
            replayssm_cache_base=base,
            replayssm_spec_write_pos=write_pos,
            replayssm_is_flush=torch.zeros(requests, dtype=torch.int8, device="cuda"),
            replayssm_is_kda=False,
        )
        req_pool = SimpleNamespace(
            mamba_pool=pool,
            get_speculative_mamba2_params_all_layers=lambda: spec_state,
        )
        backend = SimpleNamespace(
            linear_attn_backend=SimpleNamespace(
                req_to_token_pool=req_pool,
                forward_metadata=SimpleNamespace(mamba_cache_indices=state_ids),
                _translate_mamba_indices=lambda ids: ids + 1,
            )
        )

        def materialize(row, count):
            sid, rid = state_ids[row].item(), replay_ids[row].item()
            start = base[rid].item()
            result = initial[:, sid].double()
            for t in range(count):
                pos = (start + t) % length
                key = high_k[:, rid, :, pos].float() + low_k[:, rid, :, pos].float()
                key = key.repeat_interleave(hv // h, dim=1).double()
                delta = (
                    high_d[:, rid, :, pos].float() + low_d[:, rid, :, pos].float()
                ).double()
                decay = gates[:, rid, :, pos].double().exp()
                result = (
                    result * decay[..., None, None]
                    + delta[..., :, None] * key[..., None, :]
                )
            return result.float()

        expected = initial.clone()
        expected[:, 2] = materialize(1, 24)  # Capacity fold; first row stays deferred.
        expected[:, track_physical[0]] = materialize(0, 6)
        expected[:, track_physical[1]] = materialize(1, 19)
        expected_conv = conv.clone()
        for row, sid in enumerate([5, 2]):
            expected_conv[:, sid] = windows[:, row, last_steps[row]]
            expected_conv[:, track_physical[row]] = windows[:, row, track_steps[row]]

        HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
            backend,
            last_correct_step_indices=last_steps,
            mamba_track_indices=track_virtual,
            mamba_steps_to_track=track_steps,
            model=None,
            req_pool_indices=replay_ids,
        )
        torch.testing.assert_close(spec_state.temporal, expected, rtol=0, atol=2e-7)
        self.assertTrue(torch.equal(spec_state.conv[0], expected_conv))
        self.assertEqual(write_pos.tolist(), [0, 6, 0, 0])
        self.assertEqual(base.tolist(), [0, 29, 0, 13])
        self.assertEqual(pool.replayssm_is_flush.tolist(), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
