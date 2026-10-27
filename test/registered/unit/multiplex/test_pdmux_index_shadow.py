"""Mutation injection and hook placement for optional PDMux diagnostics."""

import contextlib
import os
import subprocess
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from test_pdmux_sxf_port import load_methods

from sglang.srt.multiplex.pdmux_index_shadow import (
    INPUT_FIELDS,
    IndexShadow,
    split_prefill_shadow,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def fixture():
    metadata = NS(
        query_start_loc=torch.tensor([0, 2]), mamba_cache_indices=torch.tensor([1])
    )
    runner = NS(
        is_draft_worker=False,
        ps=NS(tp_rank=1),
        forward_pass_id=17,
        attn_backend=NS(forward_metadata=metadata, init_forward_metadata=Mock()),
        _pdmux_index_shadow_enabled=True,
        model_config=NS(num_hidden_layers=3),
        device_timer=None,
    )

    class Fields(NS):
        pass

    batch = Fields(
        split_index=0,
        input_ids=torch.tensor([10, 11]),
        positions=torch.tensor([0, 1]),
        out_cache_loc=torch.tensor([8, 9]),
        req_pool_indices=torch.tensor([1]),
        seq_lens=torch.tensor([2]),
    )
    return runner, batch


class TestIndexShadow(unittest.TestCase):
    def test_shadow_does_not_keep_its_forward_batch_alive(self):
        runner, batch = fixture()
        shadow = split_prefill_shadow(runner, batch)
        owner = weakref.ref(batch)
        del batch
        self.assertIsNone(owner())
        self.assertTrue(shadow.tensors)

    def test_production_forward_hook_checks_the_actual_worker_inputs(self):
        method = load_methods(
            "model_executor/model_runner.py",
            "ModelRunner",
            ["forward"],
            contextlib=contextlib,
            profile_range=lambda *args: contextlib.nullcontext(),
            build_step_span_name=lambda batch: "forward",
            get_global_expert_distribution_recorder=lambda: NS(
                with_forward_pass=lambda *args: contextlib.nullcontext({})
            ),
            get_schedule=lambda: NS(disable_overlap_schedule=False),
            get_global_experts_capturer=lambda: None,
            get_global_indexer_capturer=lambda: None,
            dumper=NS(may_enable=False),
            get_exec=lambda: NS(moe=NS(elastic_ep_backend=None)),
        )["forward"]
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                runner, batch = fixture()
                runner._pdmux_index_shadow_enabled = enabled
                runner.msprobe_debugger = None
                runner.canary_manager = None
                runner.enable_elastic_ep = False
                runner.eplb_manager = None
                batch.apply_deprecated_skip_attn_backend_init = Mock()
                batch.forward_mode = NS(is_split_prefill=lambda: False)

                def raw(*args):
                    batch.out_cache_loc.fill_(999)
                    return NS()

                runner._forward_raw = raw
                if enabled:
                    with self.assertRaisesRegex(
                        RuntimeError, "forward.out_cache_loc at after-forward"
                    ):
                        method(runner, batch)
                else:
                    method(runner, batch)

    def test_alias_mutations_of_both_input_fields_are_detected(self):
        for name in INPUT_FIELDS:
            with self.subTest(name=name):
                _, batch = fixture()
                shadow = IndexShadow("target/P tp=1")
                shadow.capture(batch, INPUT_FIELDS, prefix="batch")
                getattr(batch, name).view(-1).add_(100)
                with self.assertRaisesRegex(RuntimeError, f"batch.{name} at after"):
                    shadow.check("after")

    def test_forward_snapshot_allows_rebinding_without_mutating_old_inputs(self):
        _, batch = fixture()
        shadow = IndexShadow("forward")
        shadow.capture(batch, INPUT_FIELDS, prefix="batch")
        batch.out_cache_loc = torch.tensor([20, 21])
        shadow.check("after-forward")

    def test_split_metadata_identity_and_values_survive_segments(self):
        runner, batch = fixture()
        shadow = split_prefill_shadow(runner, batch)
        shadow.check("segment-0")
        batch.split_index = 1
        self.assertIs(split_prefill_shadow(runner, batch), shadow)
        runner.attn_backend.forward_metadata.query_start_loc[1] = 100
        with self.assertRaisesRegex(RuntimeError, "linear.query_start_loc"):
            shadow.check("before-segment-1")

    def test_replaced_prefill_backend_metadata_is_reported_before_forward(self):
        runner, batch = fixture()
        shadow = split_prefill_shadow(runner, batch)
        runner.attn_backend.forward_metadata = object()
        with self.assertRaisesRegex(RuntimeError, "binding changed.*linear.metadata"):
            shadow.check("before-segment-1")

    def test_explicit_metadata_reinitialization_resets_the_snapshot(self):
        runner, batch = fixture()
        previous = split_prefill_shadow(runner, batch)
        batch.split_index = 1
        batch.req_pool_indices = torch.tensor([2])
        current = split_prefill_shadow(runner, batch, reset=True)
        self.assertIsNot(current, previous)
        current.check("reinitialized")

    def test_production_split_hook_checks_between_segments_and_after_kernel(self):
        method = load_methods(
            "model_executor/model_runner.py",
            "ModelRunner",
            ["forward_split_prefill"],
            device_timer_ctx=lambda *args: contextlib.nullcontext(),
        )["forward_split_prefill"]
        for during_forward in (False, True):
            with self.subTest(during_forward=during_forward):
                runner, batch = fixture()

                def model_forward(*args):
                    if during_forward:
                        batch.out_cache_loc.add_(100)

                runner.model = NS(forward_split_prefill=Mock(side_effect=model_forward))
                cp = NS(is_cp_active=lambda batch: False, prepare_cp_forward=Mock())
                with patch.dict(sys.modules, {"sglang.srt.layers.cp.utils": cp}):
                    if during_forward:
                        with self.assertRaisesRegex(RuntimeError, "after-segment-0"):
                            method(runner, batch)
                    else:
                        method(runner, batch)
                        batch.req_pool_indices.add_(10)
                        with self.assertRaisesRegex(RuntimeError, "before-segment-1"):
                            method(runner, batch)
                        self.assertEqual(
                            runner.model.forward_split_prefill.call_count, 1
                        )

    def test_graph_hook_observes_buffers_after_load_and_before_replay(self):
        class Output(NS):
            pass

        method = load_methods(
            "model_executor/runner/decode_cuda_graph_runner.py",
            "DecodeCudaGraphRunner",
            ["execute"],
            device_timer_ctx=lambda *args: contextlib.nullcontext(),
            envs=NS(SGLANG_LOG_DECODE_GRAPH_KEY=NS(get=lambda: False)),
            SharedReadEnds=NS(PRE_REPLAY="pre", IN_REPLAY="in", POST_REPLAY="post"),
            LogitsProcessorOutput=Output,
        )["execute"]
        for enabled, corrupt in ((True, False), (True, True), (False, True)):
            with self.subTest(enabled=enabled, corrupt=corrupt):
                model_runner, _ = fixture()
                model_runner.device_timer = None
                model_runner._pdmux_index_shadow_enabled = enabled
                buffers = NS(
                    out_cache_loc=torch.tensor([0]), req_pool_indices=torch.tensor([0])
                )

                def load(*args):
                    buffers.out_cache_loc.fill_(1)

                def replay(*args):
                    if corrupt:
                        buffers.out_cache_loc.fill_(99)
                    return Output(
                        next_token_logits=torch.ones(1, 2),
                        hidden_states=None,
                        customized_info=None,
                    )

                runner = NS(
                    model_runner=model_runner,
                    buffers=buffers,
                    _replay_attn_backend=lambda: None,
                    _resolve_shared_read_ends=lambda *args: None,
                    load_batch=load,
                    _replay_graph_key=None,
                    backend=NS(replay_session=contextlib.nullcontext, replay=replay),
                    is_dllm=False,
                    raw_num_token=1,
                )
                batch = NS(forward_mode=NS(name="TARGET_VERIFY"))
                if enabled and corrupt:
                    with self.assertRaisesRegex(
                        RuntimeError, "graph.out_cache_loc at after-graph-replay"
                    ):
                        method(runner, batch)
                else:
                    result = method(runner, batch)
                    torch.testing.assert_close(
                        result.next_token_logits, torch.ones(1, 2)
                    )


class TestDiagnosticLauncher(unittest.TestCase):
    def test_shadow_mode_preserves_arguments_and_failure_exit_status(self):
        script = (
            Path(__file__).resolve().parents[3]
            / "manual/pdmux/run_overlap_race_diagnostics.sh"
        )
        with tempfile.TemporaryDirectory(prefix="pdmux diag ") as folder:
            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    "shadow",
                    "--",
                    sys.executable,
                    "-c",
                    "import os,sys; print(os.environ['SGLANG_PDMUX_INDEX_SHADOW'], sys.argv[1]); sys.exit(7)",
                    "argument with spaces",
                ],
                env={**os.environ, "PDMUX_RACE_LOG_DIR": folder},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(result.returncode, 7, result.stdout)
            self.assertIn(
                "1 argument with spaces", Path(folder, "server.log").read_text()
            )
            self.assertEqual(Path(folder, "exit-code.txt").read_text().strip(), "7")

    def test_memcheck_mode_reports_a_missing_tool(self):
        script = (
            Path(__file__).resolve().parents[3]
            / "manual/pdmux/run_overlap_race_diagnostics.sh"
        )
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    "memcheck",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(0)",
                ],
                env={
                    **os.environ,
                    "PDMUX_RACE_LOG_DIR": folder,
                    "COMPUTE_SANITIZER": "/nonexistent/pdmux-sanitizer",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(result.returncode, 127, result.stdout)
            self.assertIn("compute-sanitizer not found", result.stdout)


if __name__ == "__main__":
    unittest.main()
