"""Segmented Qwen3.8 prefill must match full execution across decode yields.

Use the production model loop with CPU tensor collaborators in place of GPU
layers. Losing the per-batch HC norm/PLE state, committing each segment, or
prefetching into a shared PLE slot breaks the interleaved execution contract.
"""

import ast
import contextlib
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
SOURCE = Path(__file__).resolve().parents[4] / "python/sglang/srt/models/qwen4_exp.py"


def load_body(owner, name, **namespace):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    method = next(n for n in cls.body if getattr(n, "name", None) == name)
    method.decorator_list = []
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace[name]


class TestQwen38Pdmux(CustomTestCase):
    def fixture(self):
        journal = []
        pending_norm = [None]

        @contextlib.contextmanager
        def norms(normalized_input, next_norm):
            slot = NS(attn_input=normalized_input, output=None)
            pending_norm[0] = slot
            try:
                yield slot
            finally:
                pending_norm[0] = None

        class Layer:
            def __init__(self, index):
                self.index = index
                self.ple = NS(start_prefetch=lambda **kw: None) if index == 2 else None
                self.attn_hyper_connection = NS(hc_norm=object())
                self._is_layer_to_capture = True
                self.hc_norms = norms

            def __call__(
                self,
                *,
                positions,
                hidden_states,
                forward_batch,
                ple_batch,
                captured_last_layer_outputs,
            ):
                # PLE and the prepared norm both affect the result. Decode uses
                # the same modules but a different batch and token layout.
                slot = pending_norm[0]
                value = (
                    hidden_states.repeat(1, 4)
                    if hidden_states.shape[-1] == 2
                    else hidden_states
                )
                if slot.attn_input is not None:
                    value = value + slot.attn_input * 0.03
                if self.ple is not None:
                    value = value + ple_batch.context
                previous = forward_batch.residual_stream.value
                value = torch.tanh(
                    value * (self.index + 1)
                    + positions[:, None].to(value.dtype) * 0.01
                    + previous
                )
                forward_batch.residual_stream.value = value * 0.1
                slot.output = value * 0.2
                captured_last_layer_outputs.append(value.clone())
                return value

        def start(batch):
            batch.residual_stream = NS(value=0)

        def finish(hidden, batch, mixer):
            output = mixer(hidden + batch.residual_stream.value)
            batch.residual_stream = None
            return output

        def prepare(*, input_ids, forward_batch, **kwargs):
            journal.append((forward_batch.tag, "prepare"))
            return NS(context=input_ids[:, None].to(torch.get_default_dtype()) * 0.01)

        def commit(*, batch, forward_batch):
            journal.append((forward_batch.tag, "commit"))

        class Mix:
            def __init__(self, mixer, normalized_input):
                self.normalized = normalized_input

            def __call__(self, hidden):
                if self.normalized is not None:
                    hidden = hidden + self.normalized * 0.04
                return hidden.reshape(hidden.shape[0], 4, 2).sum(1)

        forward = load_body(
            "Qwen4ExpModel",
            "forward",
            torch=torch,
            residual_batch=NS(start=start, final_norm=finish),
            _shards_rows_over_cp=lambda batch: False,
            is_in_breakable_cuda_graph=lambda: False,
            _prepare_ple_batch=prepare,
            _commit_ple_batch=commit,
            AuxHiddenStateList=list,
            _MixStreams=Mix,
            get_global_expert_distribution_recorder=lambda: NS(
                with_current_layer=lambda i: contextlib.nullcontext()
            ),
        )
        model = NS(
            pp_group=NS(is_first_rank=True, is_last_rank=True),
            start_layer=0,
            end_layer=4,
            has_ple=True,
            ple_ngram_size=3,
            ple_ngram_eos_token_id=0,
            embed_tokens=lambda ids: torch.stack(
                (
                    ids.to(torch.get_default_dtype()),
                    ids.to(torch.get_default_dtype()) + 1,
                ),
                dim=1,
            ),
            layers=[Layer(i) for i in range(4)],
            hyper_connection_mixer=NS(hc_norm=object()),
        )
        return model, forward, journal

    @staticmethod
    def batch(tag):
        return NS(
            tag=tag,
            model_specific_states=None,
            residual_stream=None,
            forward_mode=NS(is_extend=lambda: True, is_idle=lambda: False),
        )

    def test_split_matches_full_with_decode_between_every_segment(self):
        for segments in [[(0, 1), (1, 2), (2, 3), (3, 4)], [(0, 2), (2, 4)], [(0, 4)]]:
            with self.subTest(segments=segments):
                model, forward, journal = self.fixture()
                ids, pos = torch.arange(5), torch.arange(5)
                expected = forward(model, ids, pos, self.batch("baseline"))
                prefill = self.batch("prefill")
                for start, end in segments:
                    result = forward(
                        model, ids, pos, prefill, split_interval=(start, end)
                    )
                    if end < 4:
                        self.assertIsNone(result)
                        self.assertIsNotNone(prefill.residual_stream)
                    # Force a complete decode through the same model before
                    # continuing; its terminal mix must not replace prefill's.
                    forward(
                        model,
                        torch.tensor([17]),
                        torch.tensor([100]),
                        self.batch("decode"),
                    )
                for actual, reference in zip(result, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                self.assertIsNone(prefill.residual_stream)
                self.assertNotIn("qwen4_pdmux", prefill.model_specific_states or {})
                self.assertEqual(
                    [event for tag, event in journal if tag == "prefill"],
                    ["prepare", "commit"],
                )

    def test_fp8_backend_policy_overrides_auto_and_rejects_shared_scratch(self):
        source = SOURCE.parents[1] / "arg_groups/model_overrides/qwen4_exp.py"
        tree = ast.parse(source.read_text())
        method = next(
            n for n in tree.body if getattr(n, "name", None) == "_qwen4_pdmux_overrides"
        )
        namespace = {"Any": object}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        policy = namespace["_qwen4_pdmux_overrides"]
        config = NS(
            enable_pdmux=True,
            quantization=None,
            moe_runner_backend="auto",
            speculative_algorithm="EAGLE",
            speculative_moe_runner_backend=None,
        )
        checkpoint = NS(
            quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}
        )
        self.assertEqual(
            policy(config, checkpoint),
            {
                "moe_runner_backend": "triton",
                "speculative_moe_runner_backend": "triton",
            },
        )
        config.moe_runner_backend = "cutlass"
        with self.assertRaisesRegex(ValueError, "moe-runner-backend triton"):
            policy(config, checkpoint)
        config.enable_pdmux = False
        self.assertEqual(policy(config, checkpoint), {})

    def test_ple_prefetch_does_not_touch_another_lanes_pending_state(self):
        pending = object()
        ple = NS(_prefetch_stream=object(), _prefetch_state=pending)
        start = load_body(
            "Qwen4ExpPLELayer",
            "start_prefetch",
            get_disagg=lambda: NS(enable_pdmux=True),
        )
        start(ple, batch=object(), forward_batch=object())
        self.assertIs(ple._prefetch_state, pending)

    def test_split_wrapper_preserves_mtp_hyperconnection_hidden_states(self):
        hc = torch.randn(3, 8)
        hidden = torch.randn(3, 2)
        outputs = iter([None, hidden])
        model = NS(forward=lambda **kw: next(outputs), last_hc_hidden_states=hc)
        wrapper = NS(
            pp_group=NS(is_first_rank=True, is_last_rank=True),
            is_mrope_enabled=False,
            model=model,
            lm_head=None,
            logits_processor=lambda *args: NS(hidden_states=hidden),
        )
        batch = NS(mm_inputs=None, spec_info=None)
        split = load_body("Qwen4ExpForConditionalGeneration", "forward_split_prefill")
        self.assertIsNone(split(wrapper, None, None, batch, (0, 2)))
        self.assertIs(split(wrapper, None, None, batch, (2, 4)).hidden_states, hc)
        batch.mm_inputs = [NS(contains_mm_input=lambda: True)]
        with self.assertRaisesRegex(ValueError, "text only"):
            split(wrapper, None, None, batch, (0, 2))


if __name__ == "__main__":
    unittest.main()
