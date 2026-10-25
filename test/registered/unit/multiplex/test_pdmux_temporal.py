"""Exercise temporal scheduling credit and layer boundaries without CUDA imports."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import test_pdmux_decode_ready as decode_ready
from test_pdmux_main_compat import load_method, register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
MIXIN = "python/sglang/srt/multiplex/multiplexing_mixin.py"


class TestTemporalPdmux(unittest.TestCase):
    def test_real_loop_alternates_whole_decode_and_fixed_prefill_quanta(self):
        for quantum in (2, 4, 8):
            trace = decode_ready.DispatcherTests.run_loop(
                self,
                enabled=False,
                temporal_layers=quantum,
                layers=40,
                ticks=3,
            )
            work = [row for row in trace if row[0] in ("P", "D")]
            self.assertEqual(
                work,
                [
                    item
                    for start in range(0, quantum * 3, quantum)
                    for item in [("D",), ("P", start, quantum)]
                ],
            )
            self.assertEqual(trace.count(("prepare",)), 3)
            self.assertEqual(trace.count(("process",)), 2)

    def test_real_loop_finalizes_last_partial_quantum_once(self):
        trace = decode_ready.DispatcherTests.run_loop(
            self,
            enabled=False,
            temporal_layers=4,
            layers=6,
            ticks=2,
        )
        self.assertEqual(
            [row for row in trace if row[0] in ("P", "D")],
            [("D",), ("P", 0, 4), ("D",), ("P", 4, 2)],
        )
        self.assertEqual(trace.count(("merge",)), 1)
        self.assertEqual(trace.count(("process",)), 2)

    def test_full_device_stream_never_creates_green_context(self):
        # Execute the actual initializer. Importing spatial on this branch
        # would fail: this test intentionally has no sgl_kernel dependency.
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/multiplex/pdmux_context.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(
            n
            for n in tree.body
            if getattr(n, "name", None) == "initialize_stream_groups"
        )
        stream = object()
        idx = Mock()
        ns = dict(
            envs=NS(SGLANG_PDMUX_TEMPORAL_LAYERS=NS(get=lambda: 4)),
            torch=NS(
                cuda=NS(
                    Stream=Mock(return_value=stream),
                    get_device_properties=lambda _: NS(multi_processor_count=108),
                )
            ),
            CURRENT_STREAM_IDX=idx,
        )
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
        ns["initialize_stream_groups"](0, None)
        self.assertEqual(ns["STREAM_GROUPS"], [(stream, stream)])
        self.assertEqual(ns["SM_COUNTS"], [(108, 108)])
        idx.set.assert_called_once_with(0)

    def test_layer_quanta_preserve_prefill_progress_and_final_boundary(self):
        count = load_method(
            MIXIN, "SchedulerMultiplexMixin", "_get_split_forward_count"
        )
        for quantum in (2, 4, 8):
            with self.subTest(quantum=quantum):
                scheduler = NS(
                    pdmux_temporal_layers=quantum,
                    model_config=NS(num_hidden_layers=40),
                    running_batch=NS(is_empty=lambda: False),
                    split_prefill_batch=NS(split_index=0),
                )
                steps = []
                while scheduler.split_prefill_batch.split_index < 40:
                    n = count(scheduler)
                    steps.append(n)
                    scheduler.split_prefill_batch.split_index += n
                self.assertEqual(sum(steps), 40)
                self.assertTrue(all(0 < n <= quantum for n in steps))
                scheduler.split_prefill_batch.split_index = 39
                self.assertEqual(count(scheduler), 1)
                scheduler.split_prefill_batch.split_index = 3
                scheduler.running_batch = None
                self.assertEqual(count(scheduler), 37)

    def test_credit_processes_result_once_and_never_submits_gpu_work(self):
        actions = []
        event = NS(query=Mock(side_effect=[False, False, True]))
        result = (object(), object())
        scheduler = NS(
            _temporal_prefill_done=event,
            process_batch_result=Mock(
                side_effect=lambda *args: actions.append("result")
            ),
        )
        wait = load_method(
            MIXIN,
            "SchedulerMultiplexMixin",
            "_wait_temporal_prefill_credit",
            time=NS(sleep=lambda _: actions.append("poll")),
            set_pdmux_status=lambda status: actions.append(status),
        )
        self.assertIsNone(wait(scheduler, result))
        scheduler.process_batch_result.assert_called_once_with(*result)
        self.assertEqual(actions, [False, "result", True, "poll"])
        self.assertIsNone(scheduler._temporal_prefill_done)

    def test_ready_credit_keeps_decode_result_pending(self):
        wait = load_method(
            MIXIN, "SchedulerMultiplexMixin", "_wait_temporal_prefill_credit"
        )
        result = (object(), object())
        for event in (None, NS(query=lambda: True)):
            scheduler = NS(_temporal_prefill_done=event)
            self.assertIs(wait(scheduler, result), result)
            self.assertIsNone(scheduler._temporal_prefill_done)

    def test_temporal_stream_group_never_changes_with_batch_size(self):
        select = load_method(
            MIXIN, "SchedulerMultiplexMixin", "_select_pdmux_stream_idx"
        )
        for has_prefill in (False, True):
            for bs in (0, 1, 4, 32):
                self.assertEqual(
                    select(
                        NS(pdmux_temporal_layers=4),
                        has_prefill=has_prefill,
                        decode_bs=bs,
                    ),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
