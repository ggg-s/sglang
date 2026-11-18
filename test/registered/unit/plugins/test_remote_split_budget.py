"""Split time credits, concurrent offers, and per-candidate alpha control."""

import json
import os
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
ROOT = Path(__file__).resolve().parents[4]
with patch.object(sys, "path", [str(ROOT / "plugins/remote_attn"), *sys.path]):
    from sglang_remote_attn import pd_mesh
    from sglang_remote_attn import qwen_split_prefill as split
    from sglang_remote_attn.split_budget_ledger import (
        ALPHA_SCALE,
        BudgetLedger,
        LocalStore,
    )
    from sglang_remote_attn.split_decode_budget import DecodeBudget


class TestSplitBudget(CustomTestCase):
    def ledger(self, *, g=1, stall=600, store=None):
        self.now = 0.0
        return BudgetLedger(
            store or LocalStore(), g=g, stall_ms=stall, clock=lambda: self.now
        )

    def reserve(self, ledger, key, **kwargs):
        return ledger.reserve(
            key, worker="0", cap=6000, minimum=1, rate=10000, limit=16, **kwargs
        )[0]

    def test_concurrent_prefill_offers_share_one_reservation_pool(self):
        store = dist.HashStore()
        ledgers = [self.ledger(store=store) for _ in range(16)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            grants = list(
                pool.map(
                    lambda item: self.reserve(item[1], str(item[0])), enumerate(ledgers)
                )
            )
        self.assertEqual(sum(grants), 6000)
        state = ledgers[0].snapshot()
        self.assertAlmostEqual(state["reserved_ms"], 600)
        self.assertEqual(state["credit_ms"], 0)

    def test_separate_decode_instances_have_independent_budgets(self):
        store = dist.HashStore()
        first = self.ledger(store=dist.PrefixStore("D1/", store))
        second = self.ledger(store=dist.PrefixStore("D2/", store))
        self.assertEqual(self.reserve(first, "same-room"), 6000)
        self.assertEqual(self.reserve(second, "same-room"), 6000)

    def test_repeated_offer_and_commit_are_idempotent(self):
        ledger = self.ledger()
        self.assertEqual(self.reserve(ledger, "a"), 6000)
        self.assertEqual(self.reserve(ledger, "a"), 6000)
        self.assertTrue(ledger.commit("a", 4000, controlled=True))
        before = ledger.snapshot()
        self.assertTrue(ledger.commit("a", 4000, controlled=True))
        self.assertEqual(ledger.snapshot(), before)
        self.assertFalse(ledger.commit("a", 5000, controlled=True))
        ledger.release("a")
        ledger.release("a")
        self.assertEqual(ledger.snapshot()["credit_ms"], 600)

    def test_actual_suffix_refunds_unused_offer_and_charges_overrun(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        ledger.commit("a", 4000, controlled=True)
        self.assertEqual(ledger.snapshot()["credit_ms"], 200)
        ledger.begin("0", "a", 2000)
        self.now += 0.25
        ledger.complete("0", 250, key="a")
        state = ledger.snapshot()
        self.assertGreater(state["reserved_ms"], 200)
        self.assertAlmostEqual(state["credit_ms"] + state["reserved_ms"], 350)

    def test_retry_preserves_offer_and_commit_after_partial_execution(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        ledger.commit("a", 4000, controlled=True)
        ledger.begin("0", "a", 1000)
        self.now += 0.1
        ledger.complete("0", 100, key="a")
        before = ledger.snapshot()
        self.assertEqual(self.reserve(ledger, "a"), 6000)
        self.assertTrue(ledger.commit("a", 4000, controlled=True))
        self.assertEqual(ledger.snapshot(), before)

    def test_cancel_keeps_inflight_credit_until_cuda_result(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        ledger.commit("a", 6000, controlled=True)
        ledger.begin("0", "a", 2000)
        ledger.release("a")
        self.assertEqual(ledger.snapshot()["reserved_ms"], 200)
        self.assertEqual(ledger.snapshot()["credit_ms"], 400)
        self.now += 0.3
        ledger.complete("0", 300, key="a")
        self.assertEqual(ledger.snapshot()["reserved_ms"], 0)
        self.assertEqual(ledger.snapshot()["credit_ms"], 300)

    def test_unfinished_work_survives_budget_windows(self):
        ledger = self.ledger(stall=2000)
        self.reserve(ledger, "a")
        ledger.demand("0", True)
        for _ in range(4):
            ledger.begin("0")
            self.now += 0.7
            ledger.complete("0", 700)
        self.assertEqual(ledger.snapshot()["reserved_ms"], 600)
        self.assertEqual(ledger.snapshot()["outstanding"], 1)

    def test_decode_refill_uses_g_and_deduplicates_simultaneous_dp_results(self):
        ledger = self.ledger(g=2)
        self.reserve(ledger, "a")
        ledger.commit("a", 6000, controlled=False)
        ledger.begin("0", "a", 6000)
        self.now += 0.6
        ledger.complete("0", 600, key="a")
        ledger.demand("0", True)
        ledger.begin("0")
        ledger.begin("1")
        self.now += 0.1
        ledger.complete("0", 100)
        ledger.complete("1", 100)
        self.assertAlmostEqual(ledger.snapshot()["credit_ms"], 200)

    def test_idle_refill_stops_at_capacity_and_repays_debt(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        ledger.commit("a", 6000, controlled=True)
        ledger.begin("0", "a", 6000)
        self.now += 0.8
        ledger.complete("0", 800, key="a")
        self.assertEqual(ledger.snapshot()["credit_ms"], -200)
        self.now += 0.1
        self.reserve(ledger, "probe")
        self.assertAlmostEqual(ledger.snapshot()["credit_ms"], -100)
        self.now += 10
        ledger.demand("0", False)
        self.assertEqual(ledger.snapshot()["credit_ms"], 600)

    def test_native_forward_iteration_deduplicates_delayed_dp_reports(self):
        ledger = self.ledger(g=2)
        self.reserve(ledger, "a")
        ledger.commit("a", 6000, controlled=False)
        ledger.begin("0", "a", 6000)
        self.now += 0.6
        ledger.complete("0", 600, key="a")
        ledger.demand("0", True)
        for worker, step, duration in (
            ("0", 7, 100),
            ("1", 7, 150),
            ("0", 8, 20),
            ("1", 7, 300),
        ):
            ledger.begin(worker)
            self.now += 0.4
            ledger.complete(worker, duration, step=step)
        self.assertEqual(ledger.snapshot()["credit_ms"], 340)

    def test_capacity_learns_offer_to_kv_time_without_reclaiming_reservations(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        self.now = 2
        ledger.commit("a", 4000, controlled=True)
        state = ledger.snapshot()
        self.assertEqual(state["capacity_ms"], 2600)
        self.assertEqual(state["reserved_ms"], 400)

    def test_alpha_feedback_affects_candidate_and_freezes_when_budget_limited(self):
        for controlled in (False, True):
            ledger = self.ledger(stall=600)
            ledger.demand("0", True)
            self.reserve(ledger, "a")
            ledger.commit("a", 6000, controlled=controlled)
            ledger.begin("0", "a", 6000)
            self.now += 0.6
            ledger.complete("0", 600, key="a", competing=True)
            self.assertAlmostEqual(ledger.snapshot()["alpha"], 0.9 if controlled else 1)

    def test_decode_only_window_recovers_suppressed_alpha(self):
        ledger = self.ledger()
        ledger.demand("0", True)
        self.reserve(ledger, "a")
        ledger.commit("a", 6000, controlled=True)
        ledger.begin("0", "a", 6000)
        self.now += 0.6
        ledger.complete("0", 600, key="a", competing=True)
        # A P that declined because alpha put it below the suffix floor reports
        # its candidate in the ordinary native-PD completion metadata.
        self.now += 0.6
        ledger.begin("0")
        ledger.complete("0", 600)
        self.reserve(ledger, "probe")
        ledger.commit("probe", 0, controlled=True)
        ledger.release("probe")
        ledger.begin("0")
        self.now += 0.61
        ledger.complete("0", 600)
        self.assertAlmostEqual(ledger.snapshot()["alpha"], 0.99)

    def test_idle_suffix_does_not_reduce_alpha(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        ledger.commit("a", 6000, controlled=True)
        ledger.begin("0", "a", 6000)
        self.now += 0.6
        ledger.complete("0", 600, key="a", competing=False)
        self.assertEqual(ledger.snapshot()["alpha"], 1)

    def test_zero_g_declines_offers(self):
        ledger = self.ledger(g=0)
        self.assertEqual(self.reserve(ledger, "a"), 0)

    def test_slow_suffix_can_accumulate_a_minimum_legal_probe(self):
        ledger = self.ledger()

        def offer():
            return ledger.reserve(
                "slow", worker="0", cap=6000, minimum=2048, rate=1000, limit=16
            )[0]

        self.assertEqual(offer(), 0)
        self.now = 3
        self.assertEqual(offer(), 2048)
        self.assertEqual(ledger.snapshot()["capacity_ms"], 2048)

    def test_pending_cuda_event_retains_cancelled_inflight_reservation(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        ledger.commit("a", 2000, controlled=True)
        ledger.begin("0", "a", 2000)
        budget = DecodeBudget(0.5, 600, 200, False, 65536, ledger=ledger)
        controller = NS(
            split_prefill_adaptive=True, split_prefill_decode_budget=budget, rank=0
        )
        start, end = Mock(), Mock()
        start.elapsed_time.return_value = 250
        end.query.side_effect = [False, True]
        result = NS(
            _split_suffix_timing=(2000, start, end, "suffix"),
            _split_budget_timing=((budget, "a"), False, 1),
        )
        split.record_completed_suffix(controller, result)
        ledger.release("a")
        self.assertEqual(ledger.snapshot()["reserved_ms"], 200)
        self.now += 0.25
        split.record_completed_suffix(controller, None)
        self.assertEqual(ledger.snapshot()["reserved_ms"], 0)
        self.assertEqual(ledger.snapshot()["credit_ms"], 350)
        self.assertEqual(controller._split_pending_timings, [])

    def test_finished_decode_publishes_idle_before_scheduler_waits_for_requests(self):
        ledger = self.ledger()
        self.reserve(ledger, "a")
        ledger.commit("a", 6000, controlled=False)
        ledger.begin("0", "a", 6000)
        self.now += 0.6
        ledger.complete("0", 600, key="a")
        ledger.demand("0", True)
        budget = DecodeBudget(0.5, 600, 200, False, 65536, ledger=ledger)
        running = NS(reqs=[NS(finished=lambda: True)])
        controller = NS(
            split_prefill_decode_budget=budget, scheduler=NS(running_batch=running)
        )
        split.record_completed_suffix(controller, None)
        self.now += 1
        self.assertEqual(self.reserve(ledger, "next"), 6000)

    def test_candidate_reuses_completion_metadata_even_when_alpha_suppresses_split(
        self,
    ):
        buffers = NS(output_ids=torch.zeros((1, 16), dtype=torch.int32))
        req = NS(
            metadata_buffer_index=0,
            _hybrid_split_at=0,
            _hybrid_split_candidate_tokens=2048,
        )
        split.set_metadata(lambda *_: None, buffers, req)
        self.assertEqual(buffers.output_ids[0, split.CUT_SLOT].item(), 0)
        self.assertEqual(buffers.output_ids[0, split.CANDIDATE_SLOT].item(), 2048)
        split.set_metadata(lambda *_: None, buffers, NS(metadata_buffer_index=0))
        self.assertEqual(buffers.output_ids[0, split.CANDIDATE_SLOT].item(), 0)

    def test_alpha_scales_candidate_before_nonbinding_offer_cap(self):
        controller = NS(
            split_prefill_adaptive=True,
            split_prefill_cut_ratio=0.75,
            split_prefill_cut_basis="miss",
            split_prefill_suffix_tokens=0,
            split_prefill_suffix_cap=None,
            split_prefill_page_size=1,
        )
        for alpha, suffix in ((ALPHA_SCALE, 8000), (900000, 7200)):
            cut, reason = split.select_cut(
                controller, 65536, 33536, (32768, 4096, 32768), alpha=alpha
            )
            self.assertEqual(reason, "selected")
            self.assertEqual(65536 - cut, suffix)

    def test_alpha_offer_uses_existing_bootstrap_message_and_rejects_invalid_alpha(
        self,
    ):
        for alpha in (0, 900000, ALPHA_SCALE):
            receiver = NS(
                _hybrid_split_limits=(16384, 4096),
                _hybrid_split_suffix_cap=16384,
                _hybrid_split_alpha=alpha,
                kv_mgr=NS(),
            )
            message = pd_mesh.bootstrap(
                lambda _, **kw: kw["message"],
                receiver,
                bootstrap_info={"is_dummy": False},
                message=[b"native"] * 10,
                operation="send_metadata",
            )
            peer = pd_mesh.parse_transfer(lambda **_: NS(), msg=message)
            self.assertEqual(peer._hybrid_split_alpha, alpha)
            bad = pd_mesh.parse_transfer(
                lambda **_: NS(), msg=[*message, split.ALPHA_WIRE + b"900000"]
            )
            self.assertFalse(hasattr(bad, "_hybrid_split_limits"))

    def test_tcp_store_serializes_offers_from_independent_processes(self):
        store = dist.TCPStore(
            "127.0.0.1", 0, is_master=True, timeout=timedelta(seconds=20)
        )
        ledger = BudgetLedger(store, g=1, stall_ms=600)
        code = """
import json, sys
import torch.distributed as dist
from sglang_remote_attn.split_budget_ledger import BudgetLedger
s = dist.TCPStore('127.0.0.1', int(sys.argv[1]), is_master=False)
l = BudgetLedger(s, g=1, stall_ms=600)
print(json.dumps(l.reserve(sys.argv[2], worker=sys.argv[2], cap=6000,
    minimum=2048, rate=10000, limit=16)))
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT / "plugins/remote_attn"), environment.get("PYTHONPATH", "")]
        )
        processes = []
        try:
            for i in range(2):
                processes.append(
                    subprocess.Popen(
                        [sys.executable, "-c", code, str(store.port), str(i)],
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            grants = []
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                self.assertEqual(process.returncode, 0, stderr)
                grants.append(json.loads(stdout)[0])
            self.assertLessEqual(sum(grants), 6000)
            self.assertGreater(sum(grants), 0)
            self.assertLessEqual(ledger.snapshot()["reserved_ms"], 600)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
