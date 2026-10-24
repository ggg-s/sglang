import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    make_gdn_cases,
    run_gdn_attention_case,
)
from sglang.test.kits.attention_unittest.runner_modes.cuda_graph_decode_runner import (
    run_gdn_cuda_graph_decode_case,
)
from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
    run_gdn_eagle_verify_case,
    run_gdn_eagle_verify_cuda_graph_case,
)
from sglang.test.kits.attention_unittest.runner_modes.split_op_runner import (
    run_gdn_split_op_extend_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=13, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-large-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestTritonGDNBackendCorrectness(CustomTestCase):
    @torch.inference_mode()
    def test_verify_matches_packed_decode_state(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        for dtype in (torch.float16, torch.bfloat16):
            for batch_size in (1, 4):
                with self.subTest(dtype=dtype, batch_size=batch_size):
                    torch.manual_seed(42)
                    heads, value_heads, dim, steps = 16, 32, 128, 8
                    mixed = torch.randn(
                        batch_size,
                        steps,
                        (2 * heads + value_heads) * dim,
                        device="cuda",
                        dtype=dtype,
                    )
                    a = torch.randn(
                        batch_size, steps, value_heads, device="cuda", dtype=dtype
                    )
                    b = torch.randn_like(a)
                    a_log = torch.randn(value_heads, device="cuda")
                    bias = torch.randn_like(a_log)
                    state = (
                        torch.randn(batch_size, value_heads, dim, dim, device="cuda")
                        / 100
                    )
                    indices = torch.arange(
                        batch_size - 1, -1, -1, device="cuda", dtype=torch.int32
                    )
                    intermediate = torch.empty(
                        batch_size, steps, value_heads, dim, dim, device="cuda"
                    )
                    q, k, v = mixed.reshape(1, batch_size * steps, -1).split(
                        [heads * dim, heads * dim, value_heads * dim], -1
                    )
                    kernel = TritonGDNKernel()
                    verified = kernel.target_verify(
                        A_log=a_log,
                        dt_bias=bias,
                        q=q.reshape(1, batch_size * steps, heads, dim),
                        k=k.reshape(1, batch_size * steps, heads, dim),
                        v=v.reshape(1, batch_size * steps, value_heads, dim),
                        a=a.reshape(batch_size * steps, value_heads),
                        b=b.reshape(batch_size * steps, value_heads),
                        ssm_states=state.clone(),
                        cache_indices=indices,
                        query_start_loc=torch.arange(
                            batch_size + 1, device="cuda", dtype=torch.int32
                        )
                        * steps,
                        intermediate_states_buffer=intermediate,
                        intermediate_state_indices=indices,
                        cache_steps=steps,
                        retrieve_parent_token=None,
                    ).reshape(batch_size, steps, value_heads, dim)
                    for step in range(steps):
                        decoded = kernel.packed_decode(
                            mixed[:, step].contiguous(),
                            a[:, step].contiguous(),
                            b[:, step].contiguous(),
                            A_log=a_log,
                            dt_bias=bias,
                            scale=dim**-0.5,
                            ssm_states=state,
                            cache_indices=indices,
                            num_v_heads=value_heads,
                            head_v_dim=dim,
                        ).reshape(batch_size, value_heads, dim)
                        torch.testing.assert_close(
                            decoded, verified[:, step], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            state, intermediate[:, step], rtol=0, atol=0
                        )

    @torch.inference_mode()
    def test_prefix_track_state_keeps_fp32_precision(self):
        """A cache hit must restore the accumulator, not an activation-rounded h."""
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)
                lengths, boundary = (197, 128), 192
                total, heads, value_heads, dim = sum(lengths), 2, 4, 128
                q = torch.randn(1, total, heads, dim, device="cuda", dtype=dtype)
                k = torch.randn_like(q)
                v = torch.randn(1, total, value_heads, dim, device="cuda", dtype=dtype)
                g = torch.nn.functional.logsigmoid(
                    torch.randn(1, total, value_heads, device="cuda")
                )
                beta = torch.rand(1, total, value_heads, device="cuda", dtype=dtype)
                initial = torch.randn(2, value_heads, dim, dim, device="cuda") / 100

                def run(start, end, state, lengths, **kwargs):
                    return chunk_gated_delta_rule(
                        q[:, start:end],
                        k[:, start:end],
                        v[:, start:end],
                        g[:, start:end],
                        beta[:, start:end],
                        initial_state=state,
                        initial_state_indices=torch.arange(
                            len(lengths), device="cuda", dtype=torch.int32
                        ),
                        cu_seqlens=torch.tensor(
                            [0, *torch.tensor(lengths).cumsum(0).tolist()],
                            device="cuda",
                            dtype=torch.int32,
                        ),
                        use_qk_l2norm_in_kernel=True,
                        **kwargs,
                    )

                snapshot = torch.full_like(initial, float("nan"))
                full_state = initial.clone()
                full_output, _, h = run(
                    0,
                    total,
                    full_state,
                    lengths,
                    track_state=snapshot,
                    track_chunk_idx=torch.tensor(
                        [3, -1], device="cuda", dtype=torch.int32
                    ),
                )
                reference = initial[:1].clone()
                run(0, boundary, reference, (boundary,))
                torch.testing.assert_close(snapshot[:1], reference, rtol=0, atol=0)
                self.assertTrue(torch.isnan(snapshot[1]).all().item())
                torch.testing.assert_close(
                    h[0, 3], snapshot[0].to(dtype), rtol=0, atol=0
                )
                self.assertFalse(
                    torch.equal(snapshot[0], snapshot[0].to(dtype).float())
                )

                restored = snapshot[:1].clone()
                suffix_output, _, _ = run(
                    boundary, lengths[0], restored, (lengths[0] - boundary,)
                )
                torch.testing.assert_close(restored, full_state[:1], rtol=0, atol=0)
                torch.testing.assert_close(
                    suffix_output, full_output[:, boundary : lengths[0]], rtol=0, atol=0
                )
                rounded = snapshot[:1].to(dtype).float()
                run(boundary, lengths[0], rounded, (lengths[0] - boundary,))
                self.assertFalse(torch.equal(rounded, full_state[:1]))

    CASES = make_gdn_cases("triton")
    CUDA_GRAPH_CASES = (
        GDNAttentionCase(
            name="runner_cuda_graph_gdn_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )
    SPLIT_OP_CASES = (
        (
            GDNAttentionCase(
                name="runner_split_op_gdn_extend_ragged_page_boundary",
                backend="triton",
                forward_mode=ForwardMode.EXTEND,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(0, 8, 16),
                extend_lens=(15, 8, 1),
            ),
            32,
        ),
    )
    # GDN verify covers EAGLE chain/tree plus the non-EAGLE chain spec
    # kinds (frozen_kv_mtp, dflash, ngram). All three pass against the
    # pure-PyTorch gated-delta recurrence reference; the GDN backend
    # treats them uniformly via the spec_info custom/tree mask.
    EAGLE_VERIFY_CASES = (
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_eagle_verify_gdn_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_frozen_kv_mtp_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_dflash_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_ngram_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )
    EAGLE_VERIFY_CUDA_GRAPH_CASES = (
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_eagle_verify_gdn_tree",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(5, 6),
                extend_lens=(3, 3),
            ),
            2,
            "eagle",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_frozen_kv_mtp_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "frozen_kv_mtp",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_dflash_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "dflash",
        ),
        (
            GDNAttentionCase(
                name="runner_cuda_graph_ngram_verify_gdn_chain",
                backend="triton",
                forward_mode=ForwardMode.TARGET_VERIFY,
                num_k_heads=2,
                num_v_heads=2,
                page_size=16,
                prefix_lens=(4, 7),
                extend_lens=(3, 3),
            ),
            1,
            "ngram",
        ),
    )

    def test_projected_gdn_attention_cases(self):
        for case in self.CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_attention_case(self, case)

    def test_multi_item_scoring_mixed_batch_with_empty_query(self):
        case = GDNAttentionCase(
            name="gdn_mis_mixed_batch_empty_query",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=2,
            num_v_heads=2,
            page_size=1,
            prefix_lens=(0, 0),
            extend_lens=(9, 7),
            mis_delimiter_indices=((0, 3, 8), (4, 6)),
            conv_history_weight=0.25,
        )
        run_gdn_attention_case(self, case)

    def test_multi_item_scoring_crosses_chunk_boundaries(self):
        case = GDNAttentionCase(
            name="gdn_mis_chunk_boundaries",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=2,
            num_v_heads=2,
            page_size=1,
            prefix_lens=(0,),
            extend_lens=(198,),
            mis_delimiter_indices=((5, 68, 132, 197),),
            conv_history_weight=0.25,
        )
        run_gdn_attention_case(self, case, max_context_len=256)

    # Layout-robustness. See dense/test_triton.py for the rationale.
    # shuffled_pages is the default for all tests; this method opts
    # into the more aggressive interleaved_pages + non_monotonic_extend.
    # GDN Triton handles all non-tidy layouts cleanly.
    LAYOUT_ROBUSTNESS_CASES = (
        GDNAttentionCase(
            name="layout_gdn_extend_two_request",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(16, 16),
        ),
        GDNAttentionCase(
            name="layout_gdn_decode_page_boundary",
            backend="triton",
            forward_mode=ForwardMode.DECODE,
            num_k_heads=4,
            num_v_heads=4,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
    )

    def test_layout_robustness_cases(self):
        for case in self.LAYOUT_ROBUSTNESS_CASES:
            for layout in ("interleaved_pages", "non_monotonic_extend"):
                if layout == "non_monotonic_extend" and case.forward_mode.is_decode():
                    continue
                with self.subTest(case=case.name, layout=layout):
                    run_gdn_attention_case(self, case, loc_layout=layout)

    def test_runner_mode_cuda_graph_decode_cases(self):
        for case in self.CUDA_GRAPH_CASES:
            with self.subTest(case=case.name, backend=case.backend):
                run_gdn_cuda_graph_decode_case(self, case)

    @unittest.skipIf(
        is_hip(),
        "split-op extend runner exercises the piecewise-CUDA-graph path "
        "(TcPiecewiseForwardContext.num_tokens), which is not wired on ROCm.",
    )
    def test_runner_mode_split_op_extend_cases(self):
        for case, static_num_tokens in self.SPLIT_OP_CASES:
            for breakable in (False, True):
                runner = "bcg" if breakable else "pcg"
                with self.subTest(
                    case=case.name,
                    backend=case.backend,
                    runner=runner,
                ):
                    run_gdn_split_op_extend_case(
                        self,
                        case,
                        breakable=breakable,
                        static_num_tokens=static_num_tokens,
                    )

    def test_runner_mode_eagle_verify_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_case(self, case, topk=topk, spec_kind=spec_kind)

    def test_runner_mode_eagle_verify_cuda_graph_cases(self):
        for case, topk, spec_kind in self.EAGLE_VERIFY_CUDA_GRAPH_CASES:
            with self.subTest(
                case=case.name,
                backend=case.backend,
                topk=topk,
                spec_kind=spec_kind,
            ):
                run_gdn_eagle_verify_cuda_graph_case(
                    self, case, topk=topk, spec_kind=spec_kind
                )

    # Spy directly on each sub-backend's `init_forward_metadata*` so
    # dispatch-layer slice mutations show up as a missing call, which
    # forward-output assertions can miss when the fixture happens to
    # use identical capture/replay metadata.

    def _make_dispatch_spy_backend(self):
        full_attn_backend = MagicMock(name="full_attn_backend")
        # `HybridLinearAttnBackend.__init__` aliases these buffer refs.
        full_attn_backend.token_to_kv_pool = object()
        full_attn_backend.req_to_token_pool = object()

        linear_attn_backend = MagicMock(
            spec=MambaAttnBackendBase, name="linear_attn_backend"
        )

        backend = HybridLinearAttnBackend(
            full_attn_backend,
            linear_attn_backend,
            full_attn_layers=[],
        )
        return backend, full_attn_backend, linear_attn_backend

    @staticmethod
    def _assert_fanout_forwarded(method_mock, *sentinels):
        """Assert `method_mock` was called exactly once and that each sentinel
        object identity is present in the call's positional or keyword args.
        Tolerates production switching between positional / keyword arg
        forwarding (the previous `assert_called_once_with(*positional)` form
        would silently break on such a refactor)."""
        method_mock.assert_called_once()
        call = method_mock.call_args
        forwarded = list(call.args) + list(call.kwargs.values())
        for sentinel in sentinels:
            if not any(v is sentinel for v in forwarded):
                raise AssertionError(
                    f"sentinel {sentinel!r} not forwarded by "
                    f"{method_mock._mock_name or method_mock}; call_args={call}"
                )

    def test_hybrid_dispatch_eager_init_forward_metadata_fan_out(self):
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )
        # Sentinel exposes the attribute production reads at the dispatch
        # gate (`forward_mode.is_draft_extend_v2()`); returns False so the
        # fan-out path that delegates to both children is exercised, which
        # is what these spy tests assert.
        sentinel_forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_draft_extend_v2=lambda: False)
        )
        backend.init_forward_metadata(sentinel_forward_batch)
        self._assert_fanout_forwarded(
            full_attn_backend.init_forward_metadata, sentinel_forward_batch
        )
        self._assert_fanout_forwarded(
            linear_attn_backend.init_forward_metadata, sentinel_forward_batch
        )

    def _make_sentinel_fb(self):
        return SimpleNamespace(
            batch_size=3,
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=object(),
            seq_lens=object(),
            seq_lens_cpu=object(),
            seq_lens_sum=42,
            spec_info=object(),
            encoder_lens=None,
            positions=object(),
            input_ids=object(),
            out_cache_loc=None,
        )

    def test_hybrid_dispatch_replay_init_forward_metadata_fan_out(self):
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )

        fb = self._make_sentinel_fb()
        backend.init_forward_metadata_out_graph(fb)

        # We assert sentinel identity rather than exact (args, kwargs) shape
        # so a positional↔keyword refactor inside `HybridLinearAttnBackend`
        # doesn't trip the test as long as the fb still flows through.
        for sub_backend in (full_attn_backend, linear_attn_backend):
            self._assert_fanout_forwarded(
                sub_backend.init_forward_metadata_out_graph, fb
            )

    def test_hybrid_dispatch_capture_init_forward_metadata_fan_out(self):
        # Capture mirrors the eager/replay loop shape; a slice mutation
        # there would silently miss without a spy.
        backend, full_attn_backend, linear_attn_backend = (
            self._make_dispatch_spy_backend()
        )
        fb = self._make_sentinel_fb()
        backend.init_forward_metadata_out_graph(fb, in_capture=True)

        for sub_backend in (full_attn_backend, linear_attn_backend):
            self._assert_fanout_forwarded(
                sub_backend.init_forward_metadata_out_graph, fb
            )


if __name__ == "__main__":
    unittest.main()
