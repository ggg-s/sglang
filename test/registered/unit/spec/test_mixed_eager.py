"""Offline P+verify layout, metadata ownership and padding contracts."""

import unittest
from types import SimpleNamespace as NS

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
)
from sglang.srt.speculative.mixed_eager import (
    MixedAttentionRouter,
    MixedTokenLayout,
    run_mixed_eager,
    validate_mixed_layout,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Backend:
    def __init__(self, shared, offset):
        self.req_to_token_pool = shared
        self.token_to_kv_pool = shared
        self.offset = offset
        self.counts = []

    def forward(self, *, q=None, mixed_qkv=None, **kwargs):
        assert get_attn_backend() is self
        value = q if q is not None else mixed_qkv
        self.counts.append(len(value))
        value = value + self.offset
        return value if q is not None else value.reshape(1, len(value), 2, 4)


def batches():
    def batch(ids, reqs, mode):
        n = len(ids)
        return NS(
            batch_size=len(reqs),
            input_ids=torch.tensor(ids),
            req_pool_indices=torch.tensor(reqs),
            seq_lens=torch.tensor([n // len(reqs)] * len(reqs)),
            seq_lens_cpu=torch.tensor([n // len(reqs)] * len(reqs)),
            seq_lens_sum=n,
            positions=torch.arange(n),
            out_cache_loc=torch.arange(n),
            forward_mode=mode,
            spec_info=NS(draft_token_num=8),
        )

    return batch(list(range(5)), [4], ForwardMode.EXTEND), batch(
        list(range(24)), [3, 1, 2], ForwardMode.TARGET_VERIFY
    )


class TestMixedEager(unittest.TestCase):
    def test_layout_does_not_shorten_verify(self):
        layout = MixedTokenLayout(5, 32)
        self.assertEqual(layout.total_tokens, 56)
        self.assertEqual(layout.verify_tokens, 24)
        for args in [(0, 32), (33, 32), (5, 32, 2, 8), (5, 32, 3, 4)]:
            with self.assertRaises(ValueError):
                MixedTokenLayout(*args)

    def test_native_backends_receive_only_valid_rows_and_restore_context(self):
        p, v = batches()
        shared = object()
        pb, vb = Backend(shared, 100), Backend(shared, 200)
        layout = MixedTokenLayout(5, 32)
        router = MixedAttentionRouter(layout, p, v, pb, vb)
        values = torch.arange(56 * 8, dtype=torch.float32).reshape(56, 8)
        with forward_context(ForwardContext(attn_backend=router)):
            result = router.forward(q=values, layer=NS(layer_id=3))
            self.assertIs(get_attn_backend(), router)
        torch.testing.assert_close(result[:5], values[:5] + 100)
        torch.testing.assert_close(result[32:], values[32:] + 200)
        self.assertEqual(int(torch.count_nonzero(result[5:32])), 0)
        self.assertEqual((pb.counts, vb.counts), ([5], [24]))

    def test_gdn_token_axis_and_padding(self):
        p, v = batches()
        shared = object()
        pb, vb = Backend(shared, 10), Backend(shared, 20)
        router = MixedAttentionRouter(MixedTokenLayout(5, 32), p, v, pb, vb)
        values = torch.randn(56, 8)
        result = router.forward(
            mixed_qkv=values,
            a=torch.ones(1, 56, 2),
            b=torch.ones(56, 2),
            layer=NS(layer_id=0),
        )
        self.assertEqual(result.shape, (1, 56, 2, 4))
        self.assertEqual(int(torch.count_nonzero(result[:, 5:32])), 0)
        torch.testing.assert_close(result[:, 32:].reshape(24, 8), values[32:] + 20)

    def test_shared_backend_metadata_is_rejected(self):
        p, v = batches()
        backend = Backend(object(), 0)
        with self.assertRaisesRegex(ValueError, "independent"):
            MixedAttentionRouter(MixedTokenLayout(5, 32), p, v, backend, backend)

    def test_combines_token_local_work_once_and_preserves_inputs(self):
        p, v = batches()
        shared = object()
        pb, vb = Backend(shared, 10), Backend(shared, 20)
        calls = []

        class Body:
            layers = [object()]

            def __call__(self, ids, positions, batch):
                calls.append(len(ids))
                hidden = ids.float().unsqueeze(1).expand(-1, 8)
                return get_attn_backend().forward(q=hidden, layer=NS(layer_id=0))

        head = NS(
            model=Body(),
            lm_head=None,
            logits_processor=lambda ids, hidden, lm, batch, aux: hidden,
        )
        old_p, old_v = p.input_ids.clone(), v.input_ids.clone()
        pout, vout, _ = run_mixed_eager(head, p, v, pb, vb, prefill_capacity=32)
        self.assertEqual(calls, [56])
        torch.testing.assert_close(pout[:, 0], old_p.float() + 10)
        torch.testing.assert_close(vout[:, 0], old_v.float() + 20)
        self.assertTrue(torch.equal(p.input_ids, old_p))
        self.assertTrue(torch.equal(v.input_ids, old_v))
        self.assertEqual(v.batch_size, 3)
        self.assertEqual(p.forward_mode, ForwardMode.EXTEND)

    def test_request_row_alias_is_rejected_before_stateful_work(self):
        p, v = batches()
        p.req_pool_indices = v.req_pool_indices[:1]
        shared = object()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run_mixed_eager(
                None, p, v, Backend(shared, 0), Backend(shared, 0), prefill_capacity=32
            )

    def test_non_linear_verify_and_embedding_overrides_are_rejected(self):
        for field, value in (("topk", 2), ("ragged_verify_layout", object())):
            p, v = batches()
            setattr(v.spec_info, field, value)
            with self.assertRaisesRegex(ValueError, "linear DFlash"):
                validate_mixed_layout(p, v, 32)
        for name in ("input_embeds", "replace_embeds", "mm_inputs", "lora_ids"):
            p, v = batches()
            setattr(p, name, object())
            with self.assertRaisesRegex(ValueError, "Unsupported offline mixed input"):
                validate_mixed_layout(p, v, 32)
        p, v = batches()
        p.mm_inputs, v.lora_ids = [None], [None, None, None]
        self.assertEqual(validate_mixed_layout(p, v, 32).verify_tokens, 24)


if __name__ == "__main__":
    unittest.main()
