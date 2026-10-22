"""Packed GDN replay must preserve request ownership across graph replays.

Exercise unequal windows, graph-tier slack, partial acceptance, capacity folds,
and radix checkpoints against recurrent per-token snapshots. Request slots and
physical state slots deliberately differ and change row order every iteration.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
    commit_gdn_replayssm_circular,
    commit_gdn_replayssm_spec,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestGdnReplayssmCompact(CustomTestCase):
    def _run_windows(
        self, dtype, use_graph, h=2, hv=4, capacity=64, windows=(17, 9, 1)
    ):
        torch.manual_seed(20261022)
        width, ring_len = 17, 64
        dim, slots, replay_slots = 128, 12, 9
        device = "cuda"
        state = torch.randn(slots, hv, dim, dim, device=device) * 0.05
        baseline = state.clone()
        snapshots = torch.zeros(slots, width, hv, dim, dim, device=device)
        cache = SimpleNamespace(
            temporal=state,
            replayssm_d=torch.zeros(
                replay_slots, hv, ring_len, dim, device=device, dtype=dtype
            ),
            replayssm_k=torch.zeros(
                replay_slots, h, ring_len, dim, device=device, dtype=dtype
            ),
            replayssm_g=torch.zeros(replay_slots, hv, ring_len, device=device),
            replayssm_rawv=torch.zeros(
                replay_slots, hv, ring_len, dim, device=device, dtype=dtype
            ),
            replayssm_rawk=torch.zeros(
                replay_slots, h, ring_len, dim, device=device, dtype=dtype
            ),
            replayssm_beta=None,
        )
        pool = SimpleNamespace(
            replayssm_spec_write_pos=torch.zeros(
                replay_slots, device=device, dtype=torch.int32
            ),
            replayssm_cache_base=torch.zeros(
                replay_slots, device=device, dtype=torch.int32
            ),
            replayssm_is_flush=torch.zeros(
                replay_slots, device=device, dtype=torch.int8
            ),
        )
        layer = SimpleNamespace(
            num_k_heads=h,
            head_k_dim=dim,
            num_v_heads=hv,
            head_v_dim=dim,
            A_log=torch.zeros(hv, device=device),
            dt_bias=torch.zeros(hv, device=device),
        )
        q = torch.zeros(capacity, h, dim, device=device, dtype=dtype)
        k = torch.zeros_like(q)
        v = torch.zeros(capacity, hv, dim, device=device, dtype=dtype)
        a = torch.zeros(capacity, hv, device=device, dtype=dtype)
        b = torch.zeros_like(a)
        owners = len(windows)
        physical_slots = [1, 4, 6][:owners]
        active_slots = [5, 2, 7][:owners]
        initial_ends = [0] + list(torch.tensor(list(windows) + [0]).cumsum(0))
        qsl = torch.tensor(initial_ends, device=device, dtype=torch.int32)
        state_indices = torch.tensor(
            physical_slots + [-1], device=device, dtype=torch.int32
        )
        replay_indices = torch.tensor(
            active_slots + [-1], device=device, dtype=torch.int32
        )
        backend = object.__new__(GDNAttnBackend)

        def verify():
            return backend._replayssm_target_verify(
                layer=layer,
                query=q,
                key=k,
                value=v,
                a=a,
                b=b,
                mamba_pool=pool,
                layer_cache=cache,
                cache_indices=state_indices,
                replay_indices=replay_indices,
                query_start_loc=qsl,
                draft_token_num=width,
                ragged_verify=True,
            )

        def fold(checkpoint, write_pos, cache_base, flags, **kwargs):
            commit_gdn_replayssm_circular(
                checkpoint_state=checkpoint.unsqueeze(0),
                d_cache=cache.replayssm_d.unsqueeze(0),
                k_cache=cache.replayssm_k.unsqueeze(0),
                g_cache=cache.replayssm_g.unsqueeze(0),
                d_residual_cache=cache.replayssm_rawv.unsqueeze(0),
                k_residual_cache=cache.replayssm_rawk.unsqueeze(0),
                state_batch_indices=state_indices,
                replay_indices=replay_indices,
                write_pos=write_pos,
                cache_base=cache_base,
                is_flush=flags,
                accept_lens=accept_lens,
                null_block_id=-1,
                **kwargs,
            )

        verify()
        graph = None
        if use_graph:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = verify()

        fold_ct = 0
        for step in range(64):
            # The same request keeps its replay/state slot as its batch row moves.
            order = [(step + i) % owners for i in range(owners)]
            lens = [windows[(owner + step // owners) % owners] for owner in order] + [0]
            mapping = [physical_slots[owner] for owner in order] + [-1]
            replay_mapping = [active_slots[owner] for owner in order] + [-1]
            state_indices.copy_(torch.tensor(mapping, device=device))
            replay_indices.copy_(torch.tensor(replay_mapping, device=device))
            qsl.copy_(
                torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=device)
            )
            total = sum(lens)
            for tensor in (q, k, v, a, b):
                tensor.fill_(float("nan"))
                tensor[:total].normal_()
            a[:total].sub_(2)
            before = state.clone()
            if graph is None:
                output = verify()
            else:
                graph.replay()
            self.assertTrue(torch.equal(state, before), "verify changed the checkpoint")
            self.assertTrue(torch.isfinite(output).all().item())
            self.assertEqual(torch.count_nonzero(output[total:]).item(), 0)

            expected = fused_sigmoid_gating_delta_rule_update(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q.unsqueeze(0),
                k=k.unsqueeze(0),
                v=v.unsqueeze(0),
                a=a,
                b=b,
                initial_state_source=baseline,
                initial_state_indices=state_indices,
                cu_seqlens=qsl,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                is_kda=False,
                intermediate_states_buffer=snapshots,
                intermediate_state_indices=state_indices,
                cache_steps=width,
            ).reshape_as(output)
            # Output reconstruction uses the production TF32 path; committed
            # FP32 state is checked separately at a substantially tighter bound.
            torch.testing.assert_close(
                output[:total],
                expected[:total],
                rtol=0.02,
                atol=0.003,
            )
            counts = [1 + step % n if n else 0 for n in lens]
            accept_lens = torch.tensor(counts, device=device, dtype=torch.int32)
            track_indices = torch.tensor(
                list(range(8, 8 + owners)) + [-1], device=device, dtype=torch.int32
            )
            track_steps = torch.tensor(
                [n // 2 if n else -1 for n in counts], device=device, dtype=torch.int32
            )
            commit_gdn_replayssm_spec(
                write_pos=pool.replayssm_spec_write_pos,
                cache_base=pool.replayssm_cache_base,
                is_flush=pool.replayssm_is_flush,
                num_accepted=accept_lens,
                replay_indices=replay_indices,
                max_cache_len=ring_len,
                max_spec_len=width,
                null_block_id=-1,
            )
            fold_ct += torch.count_nonzero(pool.replayssm_is_flush).item()
            fold(
                state,
                pool.replayssm_spec_write_pos,
                pool.replayssm_cache_base,
                pool.replayssm_is_flush,
                mamba_track_indices=track_indices,
                mamba_steps_to_track=track_steps,
            )
            for row, slot in enumerate(mapping[:-1]):
                baseline[slot].copy_(snapshots[slot, counts[row] - 1])
                torch.testing.assert_close(
                    state[8 + row],
                    snapshots[slot, counts[row] // 2],
                    rtol=0.002,
                    atol=0.0002,
                )
            # Materialize a COPY to check the effective state even when the
            # live implementation has intentionally deferred its capacity fold.
            materialized = state.clone()
            fold(
                materialized,
                pool.replayssm_spec_write_pos.clone(),
                pool.replayssm_cache_base.clone(),
                torch.ones_like(pool.replayssm_is_flush),
            )
            for slot in mapping[:-1]:
                torch.testing.assert_close(
                    materialized[slot],
                    baseline[slot],
                    rtol=0.002,
                    atol=0.0002,
                )
        self.assertGreater(fold_ct, 6, "test did not exercise repeated capacity folds")

    def test_fp32_checkpoint_across_variable_windows(self):
        for dtype in (torch.float16, torch.bfloat16):
            for graph in (False, True):
                with self.subTest(dtype=dtype, graph=graph):
                    self._run_windows(dtype, graph)

    def test_model_shape_fp32_checkpoint_graph(self):
        # Exercise the SM80 compact launch shape against recurrent snapshots,
        # including folds, tracked states and changing request/state ownership.
        self._run_windows(torch.float16, True, h=16, hv=32)

    def test_narrow_packed_graph_with_wider_draft_maximum(self):
        # A 9-token graph tier still advertises a 17-token draft maximum.
        # Bounding its tile by packed capacity must preserve delayed history
        # folds and cache-tracked FP32 states across repeated Graph replays.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run_windows(dtype, True, h=16, hv=32, capacity=9, windows=(9,))


if __name__ == "__main__":
    unittest.main()
