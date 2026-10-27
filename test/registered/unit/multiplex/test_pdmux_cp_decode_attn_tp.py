"""Numerical and lane-isolation tests for DSV4's persistent decode views.

Loads the production CP context and DSV4 scope without importing CUDA-only
model dependencies. Linears and attention are small PyTorch fixtures; these
tests do not run FlashMLA or a full model.
"""

from __future__ import annotations

import ast
import contextlib
import logging
import sys
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_pdmux_prefill_cp import load_definitions
from test_pdmux_sxf_port import SRT, load_methods, register_cpu_ci

try:
    import torch
except ImportError:
    torch = None

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def fixture(rank=0, size=2, *, draft=False, quantized=False, pdmux=True):
    class Linear(torch.nn.Module):
        def __init__(self, rows, cols):
            super().__init__()
            values = torch.sin(torch.arange(rows * cols, dtype=torch.float64))
            values = values.reshape(rows, cols) / 4
            if quantized:
                values = values.to(torch.float8_e4m3fn)
            self.weight = torch.nn.Parameter(values, requires_grad=False)
            if quantized:
                self.weight_scale_inv = torch.nn.Parameter(
                    torch.linspace(0.5, 1, rows * cols // 4).reshape(
                        rows // 2, cols // 2
                    ),
                    requires_grad=False,
                )
                self.weight_scale_inv.format_ue8m0 = True
            else:
                self.register_parameter("weight_scale_inv", None)

        def effective_weight(self):
            weight = self.weight.to(torch.float64)
            if self.weight_scale_inv is not None:
                scale = self.weight_scale_inv.repeat_interleave(2, 0).repeat_interleave(
                    2, 1
                )
                weight = weight * scale
            return weight

        def forward(self, x):
            return torch.nn.functional.linear(x, self.effective_weight())

    class Column(Linear):
        def __init__(self, rows, cols):
            super().__init__(rows, cols)
            self.output_size_per_partition = rows

    class Row(Linear):
        def __init__(self, rows, cols):
            super().__init__(rows, cols)
            self.input_size_per_partition = cols
            self.use_decode_attn_tp = False

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.n_heads = self.n_local_heads = 8
            self.n_groups = self.n_local_groups = 4
            self.attn_tp_size = 1
            self.attn_tp_rank = 0
            self.o_lora_rank = 4
            self.wq_b = Column(16, 8)
            self.wo_a = Column(16, 4)
            self.wo_b = Row(8, 16)
            self.wkv = Column(2, 8)  # Replicated injection path stays shared.
            self.attn_sink = torch.nn.Parameter(torch.arange(8.0), requires_grad=False)
            self._attn_sink_local = None
            self.radix_name = "attn" if draft else "attn_mqa"
            radix = torch.nn.Module()
            radix.tp_q_head_num = 8
            setattr(self, self.radix_name, radix)
            self.pool = NS(wait_layer_transfer=lambda layer: None)

        def forward(self, x, *args, **kwargs):
            if args:
                x = args[0]  # DSpark passes (positions, hidden, batch).
            self.pool.wait_layer_transfer(0)
            assert getattr(self, self.radix_name).tp_q_head_num == self.n_local_heads
            q = self.wq_b(x).reshape(x.shape[0], self.n_local_groups, 4)
            weight = self.wo_a.effective_weight().reshape(self.n_local_groups, 4, 4)
            o = torch.einsum("ngd,grd->ngr", q, weight)
            return self.wo_b(o.flatten(1))

    class MQA(Attention):
        pass

    linear_module = types.ModuleType("sglang.srt.layers.linear")
    linear_module.ColumnParallelLinear = Column
    linear_module.RowParallelLinear = Row
    config = NS(enable_cp_decode_attn_tp=True, attn_cp_size=size, attn_cp_rank=rank)
    definitions = load_definitions(
        "layers/cp/cp_decode_attn_tp.py",
        ["CpDecodeAttnTpContext"],
        copy=copy,
        torch=torch,
        logger=logging.getLogger(__name__),
        contextmanager=contextlib.contextmanager,
        get_parallel=lambda: config,
        is_cp_active=lambda batch: batch.cp_active,
        dsa_use_prefill_cp=lambda batch: getattr(batch, "dsa_cp", False),
    )
    ctx = definitions.CpDecodeAttnTpContext()
    methods = load_methods(
        "models/deepseek_v4.py",
        "MqaAttentionBase",
        [
            "maybe_use_decode_attn_tp",
            "refresh_pdmux_decode_attn_tp_view",
            "_local_attn_sink",
        ],
        get_cp_decode_attn_tp_ctx=lambda: ctx,
        is_pdmux_enabled=lambda: pdmux,
        MQALayer=MQA,
    )
    for cls in (Attention, MQA):
        cls.maybe_use_decode_attn_tp = contextlib.contextmanager(
            methods["maybe_use_decode_attn_tp"]
        )
        cls.refresh_pdmux_decode_attn_tp_view = methods[
            "refresh_pdmux_decode_attn_tp_view"
        ]
        cls._local_attn_sink = methods["_local_attn_sink"]
    return (Attention if draft else MQA)(), ctx, linear_module


@unittest.skipIf(torch is None, "PyTorch required for numerical CP decode TP tests")
class TestPdmuxCPDecodeAttentionTP(unittest.TestCase):
    def test_target_and_dspark_call_the_selected_attention_module(self):
        x = torch.ones((2, 8), dtype=torch.float64)
        for draft in (False, True):
            with self.subTest(draft=draft):
                model, _, linear = fixture(draft=draft)
                with patch.dict(sys.modules, {linear.__name__: linear}):
                    batch = NS(cp_active=False)
                    with model.maybe_use_decode_attn_tp(batch) as view:
                        expected = view(x)
                    layer = NS(self_attn=model)
                    if draft:
                        layer.hc_attn_fn = layer.hc_attn_scale = layer.hc_attn_base = (
                            None
                        )
                        layer.hc_ffn_fn = layer.hc_ffn_scale = layer.hc_ffn_base = None
                        layer._hc_pre_block = lambda x, *args: (x, None, None)
                        layer._hc_post_block = lambda x, *args: x
                        layer.input_layernorm = layer.post_attention_layernorm = (
                            lambda x: x
                        )
                        layer._run_ffn = lambda x, batch: x
                        forward = load_methods(
                            "models/deepseek_v4_dspark.py", "DSparkV4Stage", ["forward"]
                        )["forward"]
                        actual = forward(layer, None, x, batch)
                    else:
                        # Execute the actual decoder's attention dispatch block;
                        # its surrounding mHC/MoE kernels need CUDA dependencies.
                        tree = ast.parse((SRT / "models/deepseek_v4.py").read_text())
                        decoder = next(
                            node
                            for node in tree.body
                            if isinstance(node, ast.ClassDef)
                            and node.name == "DeepseekV4DecoderLayer"
                        )
                        block = next(
                            node
                            for node in ast.walk(decoder)
                            if isinstance(node, ast.With)
                            and any(
                                isinstance(item.context_expr, ast.Call)
                                and isinstance(item.context_expr.func, ast.Attribute)
                                and item.context_expr.func.attr
                                == "maybe_use_decode_attn_tp"
                                for item in node.items
                            )
                        )
                        ns = dict(
                            self=layer,
                            hidden_states=x,
                            positions=None,
                            forward_batch=batch,
                            x_quant=None,
                        )
                        exec(
                            compile(
                                ast.Module(body=[block], type_ignores=[]),
                                "dispatch",
                                "exec",
                            ),
                            ns,
                        )
                        actual = ns["hidden_states"]
                    torch.testing.assert_close(actual, expected)

    def test_sharded_projections_sum_to_full_prefill_for_target_and_draft(self):
        for draft in (False, True):
            for size in (2, 4):
                for quantized in (False, True):
                    with self.subTest(draft=draft, size=size, quantized=quantized):
                        x = torch.arange(24, dtype=torch.float64).reshape(3, 8) / 16
                        outputs = []
                        for rank in range(size):
                            model, _, linear = fixture(
                                rank, size, draft=draft, quantized=quantized
                            )
                            keys = tuple(model.state_dict())
                            full = model(x)
                            with patch.dict(sys.modules, {linear.__name__: linear}):
                                with model.maybe_use_decode_attn_tp(
                                    NS(cp_active=False)
                                ) as view:
                                    outputs.append(view(x))
                            self.assertEqual(tuple(model.state_dict()), keys)
                            self.assertEqual(model.n_local_heads, 8)
                            self.assertEqual(model.wo_b.weight.shape, (8, 16))
                            self.assertIs(view.wkv, model.wkv)
                            self.assertIs(view.pool, model.pool)
                        torch.testing.assert_close(sum(outputs), full)

    def test_prefill_does_not_observe_decode_weights_or_metadata_on_another_thread(
        self,
    ):
        for draft in (False, True):
            with self.subTest(draft=draft):
                model, ctx, linear = fixture(draft=draft, quantized=True)
                original = model.wq_b.weight
                ready, release = threading.Event(), threading.Event()
                x = torch.ones((2, 8), dtype=torch.float64)
                expected = model(x)

                def decode():
                    with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as view:
                        ready.set()
                        self.assertTrue(release.wait(5))
                        return view.wq_b.weight.shape, view._local_attn_sink().clone()

                with patch.dict(sys.modules, {linear.__name__: linear}):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pending = pool.submit(decode)
                        try:
                            self.assertTrue(ready.wait(5))
                            with model.maybe_use_decode_attn_tp(
                                NS(cp_active=True)
                            ) as prefill:
                                self.assertIs(prefill, model)
                                self.assertIs(prefill.wq_b.weight, original)
                                torch.testing.assert_close(prefill(x), expected)
                                self.assertEqual(prefill.n_local_groups, 4)
                                self.assertEqual(
                                    getattr(prefill, prefill.radix_name).tp_q_head_num,
                                    8,
                                )
                                self.assertFalse(prefill.wo_b.use_decode_attn_tp)
                                self.assertFalse(ctx.use_decode_attn_tp)
                        finally:
                            release.set()
                        shape, sink = pending.result()
                self.assertEqual(shape, (8, 8))
                torch.testing.assert_close(sink[:4], torch.arange(4.0))
                self.assertEqual(sink[4:].count_nonzero().item(), 0)
                self.assertIsNone(model._attn_sink_local)

    def test_cached_views_keep_addresses_and_fp8_scale_metadata(self):
        model, ctx, linear = fixture(quantized=True)
        with patch.dict(sys.modules, {linear.__name__: linear}):
            # A first eager forward runs inside inference_mode in the server.
            with torch.inference_mode():
                with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as first:
                    pointer = first.wo_b.weight.data_ptr()
                    self.assertTrue(first.wq_b.weight_scale_inv.format_ue8m0)
                    self.assertFalse(torch.is_inference(first.wo_b.weight))
            for _ in range(16):
                with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as again:
                    self.assertIs(again, first)
                    self.assertEqual(again.wo_b.weight.data_ptr(), pointer)
            self.assertEqual(ctx._slice_cache, {})

    def test_processed_wo_a_and_custom_scale_parameters_keep_their_layout(self):
        class ScaleParameter(torch.nn.Parameter):
            def __new__(cls, data, loader):
                return super().__new__(cls, data, requires_grad=False)

            def __init__(self, data, loader):
                self._loader = loader

            @property
            def format_ue8m0(self):
                return self._loader == "ue8m0"

        model, _, linear = fixture(rank=1)
        model.wo_a.weight = torch.nn.Parameter(
            model.wo_a.weight.reshape(4, 4, 4), requires_grad=False
        )
        model.wo_a.weight_scale_inv = ScaleParameter(torch.ones(4, 2, 2), "ue8m0")
        with patch.dict(sys.modules, {linear.__name__: linear}):
            with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as view:
                self.assertEqual(view.wo_a.weight.shape, (2, 4, 4))
                torch.testing.assert_close(view.wo_a.weight, model.wo_a.weight[2:])
                self.assertIsInstance(view.wo_a.weight_scale_inv, ScaleParameter)
                self.assertTrue(view.wo_a.weight_scale_inv.format_ue8m0)
                self.assertEqual(view.wo_a.weight_scale_inv.shape, (2, 2, 2))
        self.assertEqual(model.wo_a.weight_scale_inv.shape, (4, 2, 2))

    def test_nested_prefill_and_decode_exception_leave_original_intact(self):
        model, ctx, linear = fixture()
        original = model.wo_b.weight
        with patch.dict(sys.modules, {linear.__name__: linear}):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as view:
                    with model.maybe_use_decode_attn_tp(NS(cp_active=True)) as prefill:
                        self.assertIs(prefill, model)
                    self.assertEqual(view.n_local_heads, 4)
                    raise RuntimeError("injected")
        self.assertIs(model.wo_b.weight, original)
        self.assertEqual(model.wo_b.input_size_per_partition, 16)
        self.assertFalse(ctx.use_decode_attn_tp)

    def test_refit_refreshes_contiguous_shards_and_sink_without_new_addresses(self):
        model, _, linear = fixture()
        with patch.dict(sys.modules, {linear.__name__: linear}):
            with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as view:
                pointer = view.wo_b.weight.data_ptr()
                q_pointer = view.wq_b.weight.data_ptr()
                with torch.inference_mode():
                    sink = view._local_attn_sink()
            with torch.no_grad():
                model.wo_b.weight.add_(3)
                model.attn_sink.add_(2)
                model.wq_b.weight = torch.nn.Parameter(
                    model.wq_b.weight.clone() + 1, requires_grad=False
                )
            model.refresh_pdmux_decode_attn_tp_view()
            self.assertEqual(view.wo_b.weight.data_ptr(), pointer)
            self.assertEqual(view.wq_b.weight.data_ptr(), q_pointer)
            torch.testing.assert_close(view.wq_b.weight, model.wq_b.weight[:8])
            torch.testing.assert_close(view.wo_b.weight, model.wo_b.weight[:, :8])
            torch.testing.assert_close(sink[:4], model.attn_sink[:4])

    def test_legacy_non_pdmux_scope_still_restores_on_exception(self):
        model, _, linear = fixture(pdmux=False)
        shape = model.wo_b.weight.shape
        with patch.dict(sys.modules, {linear.__name__: linear}):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as selected:
                    self.assertIs(selected, model)
                    self.assertEqual(model.n_local_heads, 4)
                    raise RuntimeError("injected")
        self.assertEqual(model.n_local_heads, 8)
        self.assertEqual(model.wo_b.weight.shape, shape)
        self.assertFalse(model.wo_b.use_decode_attn_tp)

    def test_tiny_prefill_uses_decode_view_but_dsa_prefill_keeps_full_heads(self):
        model, _, linear = fixture()
        with patch.dict(sys.modules, {linear.__name__: linear}):
            with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as tiny:
                self.assertEqual(tiny.n_local_heads, 4)
            with model.maybe_use_decode_attn_tp(
                NS(cp_active=False, dsa_cp=True)
            ) as prefill:
                self.assertIs(prefill, model)

    @unittest.skipUnless(
        torch is not None and torch.cuda.is_available(), "CUDA required"
    )
    def test_decode_graph_replay_overlaps_prefill_and_keeps_weight_addresses(self):
        models, views = [], []
        for rank in range(2):
            model, _, linear = fixture(rank=rank)
            model = model.to("cuda")
            with patch.dict(sys.modules, {linear.__name__: linear}):
                with model.maybe_use_decode_attn_tp(NS(cp_active=False)) as view:
                    views.append(view)
            models.append(model)

        p_stream, d_stream = torch.cuda.Stream(), torch.cuda.Stream()
        x = torch.ones((4, 8), dtype=torch.float64, device="cuda")
        d_stream.wait_stream(torch.cuda.current_stream())
        p_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(d_stream):
            for _ in range(3):
                _ = sum(view(x) for view in views)
        d_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=d_stream):
            output = sum(view(x) for view in views)

        pointers = [view.wo_b.weight.data_ptr() for view in views]
        results = []
        for _ in range(32):
            with torch.cuda.stream(d_stream):
                graph.replay()
                decoded = output.clone()
            with torch.cuda.stream(p_stream):
                prefilled = models[0](x)
            results.append((decoded, prefilled))
        torch.cuda.synchronize()
        for decoded, prefilled in results:
            torch.testing.assert_close(decoded, prefilled)
        self.assertEqual(pointers, [view.wo_b.weight.data_ptr() for view in views])


if __name__ == "__main__":
    unittest.main()
