"""Exercise the scheduler split branch through the real EAGLE split wrapper."""

import ast
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from test_pdmux_sxf_port import SRT, load_methods, register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Mode:
    def __init__(self, idle=False):
        self.idle = idle

    def is_idle(self):
        return self.idle


def load_scheduler_split_branch(resolve):
    path = SRT / "managers/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scheduler = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
    )
    run_batch = next(
        n
        for n in scheduler.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_batch"
    )
    branch = next(
        n
        for n in ast.walk(run_batch)
        if isinstance(n, ast.If)
        and any(
            isinstance(part, ast.Attribute) and part.attr == "split_prefill_batch"
            for part in ast.walk(n.test)
        )
    )
    wrapper = ast.parse("def run(self, batch):\n    pass\n").body[0]
    wrapper.body = branch.body + [
        ast.Return(value=ast.Name(id="batch_result", ctx=ast.Load()))
    ]
    namespace = {"resolve_forward_inputs": resolve}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["run"]


class TestSplitInputLifetime(unittest.TestCase):
    def test_inputs_survive_all_segments_and_release_after_draft(self):
        extend_mode = Mode()
        split_forward = load_methods(
            "speculative/eagle_worker_v2.py",
            "EAGLEWorkerV2",
            ["forward_batch_split_prefill"],
            CaptureHiddenMode=SimpleNamespace(FULL="full"),
            ForwardMode=SimpleNamespace(EXTEND=extend_mode),
            speculative_moe_backend_context=contextlib.nullcontext,
            speculative_moe_a2a_backend_context=contextlib.nullcontext,
            spec_stage_span=lambda *args: contextlib.nullcontext(),
        )["forward_batch_split_prefill"]
        for idle in (False, True):
            for segments in (1, 3):
                with self.subTest(idle=idle, segments=segments):
                    batch = SimpleNamespace(
                        forward_mode=Mode(idle),
                        input_ids=None,
                        seq_lens=[],
                        seq_lens_cpu=None,
                        req_pool_indices=[],
                        split_index=0,
                        split_forward_count=1,
                    )
                    materialized = []

                    def resolve(batch, future_map):
                        self.assertIsNone(batch.input_ids)
                        batch.input_ids = (
                            [] if idle else [11 + chunk, 21 + chunk, 31 + chunk]
                        )
                        materialized.append(batch.input_ids)

                    run_split = load_scheduler_split_branch(Mock(side_effect=resolve))
                    draft_calls = []

                    def target_forward(batch, **kwargs):
                        if batch.split_index == 0:
                            # The target's persistent FB can hold DP padding;
                            # MTP must instead retain the original input tensor.
                            batch.split_forward_batch = SimpleNamespace(
                                input_ids=batch.input_ids + [-1, -1]
                            )
                        final = batch.split_index + 1 == segments
                        return SimpleNamespace(
                            logits_output=(
                                SimpleNamespace(
                                    hidden_states="hidden", mm_input_embeds=None
                                )
                                if final and not idle
                                else None
                            ),
                            next_token_ids=None if idle else "sampled",
                            next_draft_input=None,
                            has_sampled_token_ids=final and not idle,
                        )

                    def draft_extend(batch, *args):
                        self.assertIs(batch.input_ids, materialized[-1])
                        self.assertNotIn(-1, batch.input_ids)
                        self.assertIs(
                            batch.forward_mode, original_mode if idle else extend_mode
                        )
                        draft_calls.append((chunk, list(batch.input_ids)))
                        # The real draft rotate rebinds input_ids; final cleanup
                        # must release this tensor as well as the original.
                        batch.input_ids = list(batch.input_ids)
                        return None if idle else "draft_state"

                    worker = SimpleNamespace(
                        target_worker=SimpleNamespace(
                            model_config=SimpleNamespace(num_hidden_layers=segments),
                            forward_batch_split_prefill=target_forward,
                        ),
                        draft_worker=SimpleNamespace(
                            draft_runner=SimpleNamespace(tp_group=None),
                            draft_tp_context=lambda *args: contextlib.nullcontext(),
                            prefill_lane_draft_extend_backend=contextlib.nullcontext,
                            _draft_extend_for_prefill=draft_extend,
                        ),
                    )
                    worker.forward_batch_split_prefill = lambda batch: split_forward(
                        worker, batch
                    )
                    scheduler = SimpleNamespace(
                        model_worker=worker,
                        tp_worker=worker,
                        future_map=None,
                        model_config=SimpleNamespace(num_hidden_layers=segments),
                        _relay_forward_payload=Mock(),
                    )
                    original_mode = batch.forward_mode
                    # Reuse the batch across chunks: inputs must rematerialize
                    # exactly once per chunk and cannot reuse the previous one.
                    for chunk in range(2):
                        for segment in range(segments):
                            batch.split_index = segment
                            run_split(scheduler, batch)
                            self.assertIs(batch.forward_mode, original_mode)
                            if segment + 1 < segments:
                                self.assertIs(batch.input_ids, materialized[-1])
                                self.assertEqual(len(draft_calls), chunk)
                            else:
                                self.assertIsNone(batch.input_ids)
                                self.assertEqual(len(draft_calls), chunk + 1)
                    self.assertEqual(len(materialized), 2)


if __name__ == "__main__":
    unittest.main()
