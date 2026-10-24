"""HiCache event-pump coverage across the PDMux scheduler loop's states.

The pump (`Scheduler.check_hicache_events_if_enabled`) drains HiCache
transfer acks, releases host-side write locks, and advances storage prefetch
progress. In the normal event loop it rides along with batch formation, which
runs every iteration. PDMux forms a batch only when no split prefill is in
flight, so the pump needs explicit coverage for the iterations where formation
is skipped.

Both halves of the call pattern matter:

- Too few pumps starve HiCache for the whole duration of a long split prefill.
- Too many desync the pump's collective all-reduces across TP ranks, which
  deadlocks rather than degrades.
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_LAYERS = 3
CONSUMER_INDEX = 7


class _LoopFinished(Exception):
    """Breaks out of `event_loop_pdmux`, which otherwise never returns."""


class _Event:
    def __init__(self, query_results):
        self._query_results = query_results

    def query(self):
        # Default to ready once the script runs out.
        return self._query_results.pop(0) if self._query_results else True


class _Stream:
    def __init__(self, query_results):
        self._query_results = query_results

    def record_event(self):
        return _Event(self._query_results)

    def wait_event(self, event):
        pass

    def synchronize(self):
        pass


class _DecodeBatch:
    """A non-empty decode batch, so split prefill advances one layer at a time."""

    batch_is_full = False
    sampling_info = None
    spec_info = None
    enable_overlap = False
    has_grammar = False

    def is_empty(self):
        return False

    def batch_size(self):
        return 1

    def check_decode_mem(self):
        return True

    def merge_batch(self, other):
        pass

    def copy(self):
        return self


class _SplitBatch:
    def __init__(self):
        self.split_index = 0
        self.extend_num_tokens = 1000
        self.split_forward_count = 0
        self.split_prefill_finished = False
        self.chunked_req = None
        self.forward_mode = SimpleNamespace(is_idle=lambda: False)
        self.hicache_consumer_index = CONSUMER_INDEX

    def is_empty(self):
        return False

    def batch_size(self):
        return 1

    def filter_batch(self, chunked_req_to_exclude=None):
        pass


class _FakeScheduler(SchedulerMultiplexMixin):
    """Drives the real `event_loop_pdmux` over stubbed collaborators."""

    def __init__(self, *, max_iterations, query_results, pump_interval=1):
        self.max_iterations = max_iterations
        self.enable_overlap = False
        self.operations = []
        self.iteration = -1
        self.pumps = []
        self.split_forward_consumer_indices = []
        self.HICACHE_PUMP_INTERVAL = pump_interval

        self.model_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
        self.pdmux_config = SimpleNamespace(split_forward_token_budget=1000)
        self.ps = SimpleNamespace(tp_size=1, attn_dp_size=1)
        self.tp_cpu_group = SimpleNamespace(
            allreduce=lambda tensor, op: SimpleNamespace(wait=lambda: None)
        )
        self.tree_cache = Mock()
        self.pdmux_standard = False
        self.draft_worker = None
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.chunked_req = None
        self.split_prefill_batch = None
        self.running_batch = _DecodeBatch()
        self.pending_split_batch = _SplitBatch()

        stream = _Stream(query_results)
        self.stream_groups = [(stream, stream)]
        self.sm_counts = [(1, 1)]

        self.request_receiver = SimpleNamespace(recv_requests=self._recv_requests)
        self.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch: (
                batch if batch is not None and not batch.is_empty() else None
            )
        )

    # --- collaborators the loop drives -------------------------------------

    def _recv_requests(self):
        self.iteration += 1
        if self.iteration >= self.max_iterations:
            raise _LoopFinished
        return []

    def process_input_requests(self, recv_reqs):
        pass

    def ingest_requests(self):
        self.process_input_requests(self.request_receiver.recv_requests())

    def process_pending_chunked_abort(self):
        pass

    def get_new_batch_prefill(self, running_batch):
        # Stands in for `_get_new_batch_prefill_raw`, whose first act is to pump
        # HiCache events. Formation admits one request, then finds none.
        self.check_hicache_events_if_enabled()
        batch, self.pending_split_batch = self.pending_split_batch, None
        return SimpleNamespace(batch_to_run=batch, running_batch=running_batch)

    def update_running_batch(self, running_batch):
        return running_batch

    def on_idle(self):
        pass

    def adjust_stream_groups(self, running_batch, has_prefill):
        return 0, self.stream_groups[0]

    def run_batch(self, batch):
        self.operations.append(("submit", batch, self.iteration))
        if batch is self.split_prefill_batch:
            self.split_forward_consumer_indices.append(batch.hicache_consumer_index)
        return object()

    def process_batch_result(self, batch, result):
        self.operations.append(("commit", batch, self.iteration))

    def check_hicache_events_if_enabled(self):
        self.pumps.append(self.iteration)
        # Host-only ack drain: no device work, so no dependency to publish.
        return False


@contextlib.contextmanager
def _stubbed_cuda():
    with (
        patch("torch.cuda.stream", lambda _stream: contextlib.nullcontext()),
        patch("torch.cuda.empty_cache", lambda: None),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.set_pdmux_status",
            lambda _enabled: None,
        ),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.get_current_stream_idx",
            lambda: 0,
        ),
    ):
        yield


def _run_loop(*, max_iterations, query_results, pump_interval=1):
    scheduler = _FakeScheduler(
        max_iterations=max_iterations,
        query_results=query_results,
        pump_interval=pump_interval,
    )
    with _stubbed_cuda():
        try:
            scheduler.event_loop_pdmux()
        except _LoopFinished:
            pass
    return scheduler


class TestPDMuxHiCacheEvents(CustomTestCase):
    def test_spec_overlap_submits_next_decode_before_committing_previous(self):
        """DFlash must not drain N-1 before issuing N while a split is active.

        Grammar still needs its previous CPU result before preparing N. The
        non-overlap speculative path retains that same synchronous boundary.
        """
        for overlap, grammar in ((True, False), (False, False), (True, True)):
            with self.subTest(overlap=overlap, grammar=grammar):
                scheduler = _FakeScheduler(max_iterations=2, query_results=[])
                scheduler.enable_overlap = overlap
                scheduler.running_batch.has_grammar = grammar
                scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False)
                with _stubbed_cuda(), self.assertRaises(_LoopFinished):
                    scheduler.event_loop_pdmux()
                ops = scheduler.operations
                next_submit = next(
                    i
                    for i, (op, batch, step) in enumerate(ops)
                    if op == "submit" and batch is scheduler.running_batch and step == 1
                )
                first_commit = next(
                    i for i, row in enumerate(ops) if row[0] == "commit"
                )
                if overlap and not grammar:
                    self.assertLess(next_submit, first_commit)
                    self.assertIs(
                        scheduler.forward_stream, scheduler.stream_groups[0][1]
                    )
                else:
                    self.assertLess(first_commit, next_submit)
                self.assertEqual(scheduler.pumps, [0, 1])

    def test_every_iteration_pumps_hicache_events_exactly_once(self):
        """At interval 1, one pump per iteration across all formation states.

        With three layers and a busy decode batch the loop walks: iteration 0
        forms the batch (pump rides along with formation), iterations 1-2 have a
        split prefill in flight (formation is skipped), and iteration 3 waits for
        the prefill kernel to retire (formation is not even attempted). Only the
        first is covered by the formation path.
        """
        # The finish event reports not-ready once, so the loop spends iteration 3
        # in `wait_prefill_kernel_done` before merging.
        scheduler = _run_loop(max_iterations=4, query_results=[False])

        self.assertEqual(scheduler.pumps, [0, 1, 2, 3])

    def test_split_prefill_forwards_keep_one_hicache_consumer_index(self):
        """Every layer segment must still carry the batch's consumer index.

        The index selects which layer-transfer event set the model waits on
        before reading loaded-back KV. This case only guards the ScheduleBatch
        field surviving across segments; the worker actually installing it per
        segment (`set_hicache_consumer`, which decode resets to -1 in between)
        is guarded by test_pdmux_scheduler.py's
        test_split_prefill_forward_installs_hicache_consumer_first.
        """
        scheduler = _run_loop(max_iterations=4, query_results=[False])

        self.assertEqual(
            scheduler.split_forward_consumer_indices,
            [CONSUMER_INDEX] * NUM_LAYERS,
        )

    def test_pump_still_runs_when_prefill_finishes_without_waiting(self):
        """The finish event may already be ready when the merge check runs, so
        the loop never enters `wait_prefill_kernel_done`. The in-flight
        iterations still need their pump."""
        scheduler = _run_loop(max_iterations=3, query_results=[])

        self.assertEqual(scheduler.pumps, [0, 1, 2])

    def test_pump_decimation_skips_off_interval_iterations(self):
        """With an interval of 2, only every second ELIGIBLE iteration pumps.

        The tick advances on eligible iterations only (formation iterations
        pump through the formation path instead), and it is a deterministic
        function of the loop state, so every TP rank pumps on the same
        iterations -- the pump's collectives stay aligned. Iteration 0 forms
        (formation-path pump); the in-flight/wait iterations 1-3 tick 1, 2, 3,
        pumping only on tick 2 (iteration 2).
        """
        scheduler = _run_loop(max_iterations=4, query_results=[False], pump_interval=2)

        self.assertEqual(scheduler.pumps, [0, 2])


if __name__ == "__main__":
    unittest.main()
