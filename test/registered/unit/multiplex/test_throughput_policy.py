"""Useful-token throughput decisions and completed-output reward accounting."""

from __future__ import annotations

import importlib.util
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_pdmux_main_compat import load_method, register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
MIXIN = "python/sglang/srt/multiplex/multiplexing_mixin.py"
spec = importlib.util.spec_from_file_location(
    "_throughput_policy", ROOT / "python/sglang/srt/multiplex/throughput_policy.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Policy = module.TemporalThroughputPolicy


class TestThroughputPolicy(unittest.TestCase):
    def learned(self, cost=2):
        p = Policy()
        p.observe_prefill(
            tokens=48, layers=8, start=8, final=False, cpu_ms=cost * 8, gpu_ms=4
        )
        return p

    def test_bootstrap_and_no_decode(self):
        p = Policy()
        kw = dict(remaining=40, tokens=48, current_remaining=[128], new_remaining=[128])
        self.assertEqual(p.choose(**kw), 8)
        kw["current_remaining"] = []
        self.assertEqual(p.choose(**kw), 40)

    def test_fill_decode_batch_when_marginal_throughput_improves(self):
        p = self.learned()
        p.decode_ms_by_bs = {1: 12, 4: 16}
        p.tokens_by_bs = {1: 3, 4: 12}
        q = p.choose(
            remaining=40,
            tokens=48,
            current_remaining=[128],
            new_remaining=[128, 128, 128],
        )
        self.assertEqual(q, 32)
        self.assertGreater(p.last_scores[32], p.last_scores[8])

    def test_keep_efficient_current_decode_when_expansion_is_unproductive(self):
        p = self.learned()
        p.decode_ms_by_bs = {1: 10, 2: 20}
        p.tokens_by_bs = {1: 6, 2: 2}
        self.assertEqual(
            p.choose(
                remaining=40, tokens=48, current_remaining=[128], new_remaining=[128]
            ),
            4,
        )

    def test_feedback_uses_actual_tokens_including_zero_useful_rounds(self):
        p = Policy()
        p.observe_output(2, 6)
        self.assertAlmostEqual(p.tokens_by_bs[2], 6)
        p.observe_output(2, 0)
        self.assertAlmostEqual(p.tokens_by_bs[2], 4.8)
        self.assertEqual(p.useful_tokens, 6)

    def test_action_fills_batch_but_bounds_decode_deferral(self):
        p = self.learned()
        p.decode_ms_by_bs = {1: 12, 4: 16}
        p.tokens_by_bs = {1: 3, 4: 12}
        kw = dict(
            tokens=48,
            quantum=16,
            current_remaining=[128],
            new_remaining=[128, 128, 128],
        )
        self.assertFalse(p.decode_first(**kw, since_decode_ms=10))
        self.assertTrue(p.decode_first(**kw, since_decode_ms=40))
        p.tokens_by_bs[4] = 2
        self.assertTrue(p.decode_first(**kw, since_decode_ms=0))

    def test_action_does_not_wait_for_ineligible_prefill(self):
        p = self.learned()
        kw = dict(tokens=48, quantum=8, since_decode_ms=0)
        self.assertTrue(p.decode_first(**kw, current_remaining=[128], new_remaining=[]))
        self.assertFalse(
            p.decode_first(**kw, current_remaining=[], new_remaining=[128])
        )

    def test_deferred_decode_does_not_allocate_or_advance_counters(self):
        fn = load_method(
            MIXIN, "SchedulerMultiplexMixin", "_advance_temporal_running_batch"
        )
        batch = NS(filter_batch=Mock())
        s = NS(
            update_running_batch=Mock(return_value=batch),
            _filter_temporal_running_batch=lambda b: b.filter_batch(),
        )
        self.assertIs(fn(s, batch, False), batch)
        batch.filter_batch.assert_called_once_with()
        s.update_running_batch.assert_not_called()
        self.assertIs(fn(s, batch, True), batch)
        s.update_running_batch.assert_called_once_with(batch)

    def test_early_filter_reopens_prefill_admission_when_requests_finish(self):
        fn = load_method(
            MIXIN, "SchedulerMultiplexMixin", "_filter_temporal_running_batch"
        )
        batch = NS(
            batch_size=Mock(side_effect=[4, 3]), filter_batch=Mock(), batch_is_full=True
        )
        fn(NS(), batch)
        self.assertFalse(batch.batch_is_full)
        batch.batch_size = Mock(side_effect=[3, 3])
        batch.batch_is_full = True
        fn(NS(), batch)
        self.assertTrue(batch.batch_is_full)

    def test_estimated_service_cap_and_one_layer_remainder(self):
        p = self.learned(20)
        kw = dict(tokens=48, current_remaining=[128], new_remaining=[128])
        self.assertEqual(p.choose(remaining=40, **kw), 4)
        self.assertEqual(p.choose(remaining=1, **kw), 1)

    def test_no_short_tail_estimate_for_long_prompt(self):
        p = self.learned()
        self.assertEqual(
            p.choose(
                remaining=40, tokens=9000, current_remaining=[128], new_remaining=[128]
            ),
            8,
        )

    def test_first_setup_and_final_tail_excluded_from_layer_cost(self):
        p = self.learned()
        for start, final in [(0, False), (32, True)]:
            p.observe_prefill(
                tokens=48, layers=8, start=start, final=final, cpu_ms=900, gpu_ms=800
            )
        self.assertEqual(p.layer_ms[0], 2)

    def test_cpu_gpu_overlap_not_double_counted_and_nonfinite_ignored(self):
        p = self.learned()
        p.observe_prefill(
            tokens=48, layers=8, start=8, final=False, cpu_ms=16, gpu_ms=12
        )
        p.observe_decode(float("nan"), 2, 1)
        self.assertEqual(p.layer_ms[0], 2)
        self.assertFalse(p.decode_ms_by_bs)

    def test_real_result_hook_counts_only_output_through_stop(self):
        through_stop = load_method(
            "python/sglang/srt/managers/schedule_batch.py",
            "Req",
            "output_ids_through_stop",
        )
        Req = type("Req", (), {"output_ids_through_stop": property(through_stop)})
        req = Req()
        req.output_ids = [1, 2]
        req.finished_len = None

        def process(*args):
            req.output_ids.extend([3, 4, 5, 6])
            req.finished_len = 5

        mode = NS(is_decode=lambda: True, is_extend=lambda: False)
        batch = NS(reqs=[req], forward_mode=mode)
        scheduler = NS(
            pdmux_temporal_adaptive=True,
            _temporal_policy=NS(observe_output=Mock()),
            publish_load_snapshot=Mock(),
            load_publisher=Mock(),
            load_inquirer=Mock(),
            batch_result_processor=NS(process_batch_result_decode=process),
            _record_step_counters=Mock(),
            metrics_reporter=Mock(),
            enable_fpm=False,
            _maybe_clear_mm_inputs=Mock(),
            maybe_send_health_check_signal=Mock(),
        )
        fn = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "process_batch_result",
            flush_trace_batch=Mock(),
        )
        fn(scheduler, batch, object())
        scheduler._temporal_policy.observe_output.assert_called_once_with(1, 3)

    def test_ready_result_consumed_only_after_complete_copy(self):
        fn = load_method(
            MIXIN,
            "SchedulerMultiplexMixin",
            "_consume_ready_temporal_decode",
            set_pdmux_status=Mock(),
        )
        s = NS(process_batch_result=Mock())
        for ready in (False, True):
            r = (object(), NS(copy_done=NS(query=lambda: ready)))
            self.assertIs(fn(s, r), None if ready else r)
        self.assertEqual(s.process_batch_result.call_count, 1)

    def test_previous_result_before_ready_prefill_credit(self):
        fn = load_method(
            MIXIN,
            "SchedulerMultiplexMixin",
            "_wait_temporal_prefill_credit",
            set_pdmux_status=Mock(),
        )
        s = NS(
            pdmux_temporal_adaptive=True,
            _temporal_prefill_done=NS(query=lambda: True),
            process_batch_result=Mock(),
        )
        r = (object(), object())
        self.assertIsNone(fn(s, r))
        s.process_batch_result.assert_called_once_with(*r)

    def test_timing_poll_nonblocking(self):
        policy = NS(observe_decode=Mock(), observe_prefill=Mock())
        query = Mock(return_value=False)
        pair = (NS(elapsed_time=lambda _: 12), NS(query=query), 16, dict(decode_bs=2))
        s = NS(_temporal_timings=deque([pair]), _temporal_policy=policy)
        fn = load_method(MIXIN, "SchedulerMultiplexMixin", "_poll_temporal_timings")
        fn(s)
        policy.observe_decode.assert_not_called()
        query.return_value = True
        fn(s)
        policy.observe_decode.assert_called_once_with(16, 12, 2)
        self.assertFalse(s._temporal_timings)


if __name__ == "__main__":
    unittest.main()
