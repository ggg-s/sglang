"""Unit tests for the ForwardBatch attention plan marker / plan record.

Covers the contract behind ``skip_attn_backend_init`` deprecation:
  * fresh batches need planning; marked batches don't
  * the plan record (planned bs / num tokens) snapshots mark-time shapes
  * reshape after marking triggers a re-plan only for sites that opted
    into ``replan_equivalent``; re-marking re-records the new shapes
  * the deprecated kwarg shim maps explicit values onto the marker
    (mapped, not ignored) and warns once per process

CPU only; the eager decode ownership regression uses an in-memory model and
shared-buffer staging boundary.
"""

import unittest
import warnings
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.model_executor.forward_batch_info as fbi
import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_batch(bs: int = 2, num_tokens: int = 2) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=bs,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        req_pool_indices=torch.zeros(bs, dtype=torch.long),
        seq_lens=torch.ones(bs, dtype=torch.long),
        out_cache_loc=torch.zeros(bs, dtype=torch.long),
        seq_lens_sum=bs,
    )


class TestForwardMetadataPlanRecord(CustomTestCase):
    def test_preplanned_decode_preserves_outer_graph_request_mapping(self):
        """Compact inner decode must not stage into its outer graph's buffers.

        The eager registry coalesces inputs with the draft graph. Copying row
        three into its first lane corrupts the next full-batch KV lookup, even
        though the inner model received the correct row on this call.
        """
        from sglang.srt.model_executor.runner import eager_runner

        for planning in ("ready", "fresh", "stale"):
            with self.subTest(planning=planning):
                outer_seq = torch.tensor([792, 655, 408, 295])
                outer_reqs = torch.tensor([0, 1, 2, 3])
                original_seq = outer_seq.clone()
                original_reqs = outer_reqs.clone()
                fb = _make_batch(bs=1, num_tokens=1)
                fb.seq_lens = outer_seq[3:].clone()
                fb.req_pool_indices = outer_reqs[3:].clone()
                if planning != "fresh":
                    fb.mark_forward_metadata_ready(replan_equivalent=True)
                if planning == "stale":
                    fb.forward_metadata_planned_bs = 2

                def stage(
                    batch, pp_proxy_tensors, outer_seq=outer_seq, outer_reqs=outer_reqs
                ):
                    outer_seq[:1].copy_(batch.seq_lens)
                    outer_reqs[:1].copy_(batch.req_pool_indices)
                    return replace(
                        batch, seq_lens=outer_seq[:1], req_pool_indices=outer_reqs[:1]
                    )

                planned = []
                backend = SimpleNamespace(
                    init_forward_metadata=lambda batch, planned=planned: planned.append(
                        batch.seq_lens.clone()
                    )
                )
                runner = SimpleNamespace(
                    enable_pdmux=False,
                    load_batch=stage,
                    _resolve_decode_pdmux=lambda backend=backend: (
                        backend,
                        nullcontext(),
                    ),
                    model_runner=SimpleNamespace(
                        model=SimpleNamespace(
                            forward=lambda ids, pos, batch, **kw: (
                                batch.seq_lens + batch.req_pool_indices
                            )
                        ),
                        device_timer=None,
                        _pp_kwargs=lambda pp: {},
                    ),
                )
                with patch.object(
                    eager_runner, "device_timer_ctx", lambda *a: nullcontext()
                ):
                    output = eager_runner.EagerRunner._execute_decode(runner, fb)
                self.assertEqual(output.tolist(), [298])
                if planning == "ready":
                    self.assertTrue(torch.equal(outer_seq, original_seq))
                    self.assertTrue(torch.equal(outer_reqs, original_reqs))
                    self.assertEqual(planned, [])
                else:
                    self.assertEqual(outer_seq.tolist(), [295, 655, 408, 295])
                    self.assertEqual(outer_reqs.tolist(), [3, 1, 2, 3])
                    self.assertEqual(len(planned), 1)

    def test_fresh_batch_needs_planning(self):
        fb = _make_batch()
        self.assertTrue(fb.needs_forward_metadata_init())
        self.assertFalse(fb.forward_metadata_ready)

    def test_mark_records_shapes_and_skips_planning(self):
        fb = _make_batch(bs=3, num_tokens=7)
        fb.mark_forward_metadata_ready()
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(fb.forward_metadata_planned_bs, 3)
        self.assertEqual(fb.forward_metadata_planned_num_tokens, 7)

    def test_reshape_without_opt_in_keeps_skipping(self):
        # Wrapper regimes must never auto-re-plan (would clobber per-step metadata).
        fb = _make_batch(bs=2)
        fb.mark_forward_metadata_ready()
        fb.batch_size = 4  # DP padding reshapes the batch
        self.assertFalse(fb.needs_forward_metadata_init())

    def test_reshape_with_opt_in_replans(self):
        fb = _make_batch(bs=2, num_tokens=2)
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        self.assertFalse(fb.needs_forward_metadata_init())

        fb.batch_size = 4  # bs drift (prepare_mlp_sync_batch decode pad)
        self.assertTrue(fb.needs_forward_metadata_init())

        fb.batch_size = 2
        fb.input_ids = torch.zeros(6, dtype=torch.long)  # token drift
        self.assertTrue(fb.needs_forward_metadata_init())

    def test_remark_re_records_padded_shapes(self):
        # Per-step loops re-mark each plan; the re-mark must snapshot padded shapes.
        fb = _make_batch(bs=2)
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        fb.batch_size = 4
        self.assertTrue(fb.needs_forward_metadata_init())
        fb.mark_forward_metadata_ready(replan_equivalent=True)
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(fb.forward_metadata_planned_bs, 4)


class TestDeprecatedSkipKwargShim(CustomTestCase):
    def setUp(self):
        self._saved_warned = fbi._skip_attn_backend_init_warned
        fbi._skip_attn_backend_init_warned = False

    def tearDown(self):
        fbi._skip_attn_backend_init_warned = self._saved_warned

    def test_none_is_a_silent_no_op(self):
        fb = _make_batch()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fb.apply_deprecated_skip_attn_backend_init(None)
        self.assertTrue(fb.needs_forward_metadata_init())

    def test_true_maps_onto_marker_and_warns(self):
        # Mapped, not ignored: a no-op would silently re-plan multi-step metadata.
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(True)
        self.assertFalse(fb.needs_forward_metadata_init())
        self.assertEqual(len(caught), 1)
        self.assertTrue(issubclass(caught[0].category, DeprecationWarning))

    def test_false_warns_but_does_not_mark(self):
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(False)
        self.assertTrue(fb.needs_forward_metadata_init())
        self.assertEqual(len(caught), 1)

    def test_warns_once_per_process(self):
        # Hot-loop guard: per-forward callers must not pay warnings.warn repeatedly.
        fb = _make_batch()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fb.apply_deprecated_skip_attn_backend_init(True)
            fb.apply_deprecated_skip_attn_backend_init(True)
            _make_batch().apply_deprecated_skip_attn_backend_init(False)
        self.assertEqual(len(caught), 1)


if __name__ == "__main__":
    unittest.main()
