"""Exercise real ticket and dispatcher methods without importing CUDA runtime."""

from __future__ import annotations

import ast
import contextlib
import types
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

from test_pdmux_main_compat import load_method, register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
NS = types.SimpleNamespace
ROOT = Path(__file__).resolve().parents[4]
RELAY = "python/sglang/srt/managers/overlap_utils.py"
MIXIN = "python/sglang/srt/multiplex/multiplexing_mixin.py"


class Stream:
    def __init__(self, device):
        self.device = device
        self.queue = []
        self.position = 0
        self.host_syncs = 0

    def enqueue(self, action):
        self.queue.append(action)

    def finish(self, end=None):
        end = len(self.queue) if end is None else end
        while self.position < end:
            action = self.queue[self.position]
            self.position += 1
            action()

    def wait_event(self, event):
        self.enqueue(event.complete)

    def synchronize(self):
        self.host_syncs += 1
        self.finish()

    def record_event(self):
        event = Event(self.device)
        event.record(self)
        return event


class Event:
    def __init__(self, device):
        self.device = device
        self.host_syncs = 0

    def record(self, stream=None):
        self.stream = stream or self.device.current
        self.end = len(self.stream.queue)

    def query(self):
        return self.stream.position >= self.end

    def complete(self):
        self.stream.finish(self.end)

    def synchronize(self):
        self.host_syncs += 1
        self.complete()

    def wait(self):
        self.device.current.wait_event(self)


class Tensor:
    def __init__(self, device, values, gpu=False):
        self.device = device
        self.values = list(values)
        self.gpu = gpu
        self.dtype = "int64"

    @property
    def shape(self):
        return (len(self.values),)

    def clone(self):
        result = Tensor(self.device, self.values, self.gpu)
        if self.gpu:
            result.copy_(self)
        return result

    def copy_(self, source, non_blocking=False):
        def copy():
            self.values[:] = source.values

        if self.gpu or source.gpu:
            self.device.current.enqueue(copy)
        else:
            copy()
        return self

    def __getitem__(self, indices):
        result = Tensor(self.device, [0] * len(indices.values), self.gpu)

        def gather():
            result.values[:] = [self.values[i] for i in indices.values]

        if self.gpu:
            self.device.current.enqueue(gather)
        else:
            gather()
        return result

    def __setitem__(self, indices, source):
        def scatter():
            for i, value in zip(indices.values, source.values):
                self.values[i] = value

        self.device.current.enqueue(scatter)

    def to(self, dtype):
        return self

    def sum(self):
        return sum(self.values)

    def cpu(self):
        self.device.current.synchronize()
        return Tensor(self.device, self.values)


class Device:
    def __init__(self):
        self.current = Stream(self)

    def current_stream(self):
        return self.current

    def Event(self):
        return Event(self)

    @contextlib.contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


class TicketTests(unittest.TestCase):
    def setUp(self):
        self.device = d = Device()
        tree = ast.parse((ROOT / RELAY).read_text(encoding="utf-8"))
        ticket_node = next(
            n for n in tree.body if getattr(n, "name", "") == "SeqLensCpuTicket"
        )
        namespace = dict(dataclass=dataclass, torch=NS(Tensor=Tensor), Any=object)
        exec(
            compile(ast.Module(body=[ticket_node], type_ignores=[]), RELAY, "exec"),
            namespace,
        )
        fake_torch = NS(
            get_device_module=lambda _: d,
            equal=lambda a, b: a.values == b.values,
            empty_like=lambda t, **_: Tensor(d, [-1] * t.shape[0], t.gpu),
        )
        names = [
            "enable_eager_seq_lens_cpu",
            "_issue_seq_lens_cpu_ticket",
            "get_seq_lens_cpu_ticket",
            "seq_lens_cpu_ticket_ready",
            "resolve_seq_lens_cpu",
            "publish",
        ]
        methods = {
            name: load_method(
                RELAY,
                "FutureMap",
                name,
                torch=fake_torch,
                SeqLensCpuTicket=namespace["SeqLensCpuTicket"],
                _is_cuda=True,
                _is_hip=False,
                _DEBUG_ASSERT=False,
            )
            for name in names
        }
        self.relay = r = type("Relay", (), methods)()
        r.device = d
        r.needs_cpu_seq_lens = True
        r.needs_confidence_relay = False
        r.new_seq_lens_buf = Tensor(d, [0, 0, 0, 0], True)
        r.new_seq_lens_cpu_pinned = Tensor(d, [0, 0, 0, 0])
        r.fwd_prepare_d2h_stream = Stream(d)
        r._req_pool = NS(req_generation=Tensor(d, [0, 1, 1, 1]))
        r._seq_lens_cpu_slots = None
        r._seq_lens_cpu_generation = 0
        r._seq_lens_cpu_ticket = None
        r.publish_ready = None
        r.spec_algo = NS(is_some=lambda: True)
        self.batch = NS(
            spec_info=NS(future_indices=Tensor(d, [1, 3])),
            req_pool_indices_cpu=Tensor(d, [1, 3]),
        )

    def publish(self, values, indices=(1, 3)):
        self.relay.publish(
            Tensor(self.device, indices, True), Tensor(self.device, values, True)
        )
        return self.relay.get_seq_lens_cpu_ticket(self.batch)

    def test_publish_is_nonblocking_and_resolve_waits_only_for_ticket(self):
        r = self.relay
        r.enable_eager_seq_lens_cpu()
        ticket = self.publish([11, 31])
        self.assertFalse(r.seq_lens_cpu_ticket_ready(ticket))
        self.assertEqual(r.fwd_prepare_d2h_stream.host_syncs, 0)
        r.resolve_seq_lens_cpu(self.batch)
        self.assertEqual(self.batch.seq_lens_cpu.values, [11, 31])
        self.assertEqual(self.batch.seq_lens_sum, 42)
        self.assertEqual(ticket.done.host_syncs, 1)
        self.assertEqual(r.fwd_prepare_d2h_stream.host_syncs, 0)

    def test_ready_ticket_never_host_synchronizes(self):
        self.relay.enable_eager_seq_lens_cpu()
        ticket = self.publish([11, 31])
        ticket.done.complete()
        self.relay.resolve_seq_lens_cpu(self.batch)
        self.assertEqual(ticket.done.host_syncs, 0)

    def test_later_relay_writes_do_not_change_snapshot(self):
        r = self.relay
        r.enable_eager_seq_lens_cpu()
        ticket = self.publish([11, 31])
        r.new_seq_lens_buf[Tensor(self.device, [1], True)] = Tensor(
            self.device, [99], True
        )
        self.device.current.finish()
        ticket.done.complete()
        self.assertEqual(ticket.cpu_lens.values, [0, 11, 0, 31])
        self.assertEqual(r.new_seq_lens_buf.values[1], 99)

    def test_ring_reuse_fences_old_d2h_and_rejects_superseded_ticket(self):
        r = self.relay
        r.enable_eager_seq_lens_cpu()
        first = self.publish([11, 31])
        second = self.publish([12, 32])
        third = self.publish([13, 33])
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            r.seq_lens_cpu_ticket_ready(first)
        third.done.complete()
        self.assertTrue(second.done.query())
        r.resolve_seq_lens_cpu(self.batch)
        self.assertEqual(self.batch.seq_lens_cpu.values, [13, 33])

    def test_consumed_cpu_lengths_survive_slot_reuse(self):
        self.relay.enable_eager_seq_lens_cpu()
        self.publish([11, 31])
        self.relay.resolve_seq_lens_cpu(self.batch)
        consumed = self.batch.seq_lens_cpu
        self.publish([12, 32])
        self.publish([13, 33]).done.complete()
        self.assertEqual(consumed.values, [11, 31])

    def test_recycled_pool_row_is_rejected(self):
        self.relay.enable_eager_seq_lens_cpu()
        self.publish([11, 31])
        self.relay._req_pool.req_generation.values[1] += 1
        with self.assertRaisesRegex(RuntimeError, "recycled"):
            self.relay.resolve_seq_lens_cpu(self.batch)

    def test_merge_snapshot_preserves_existing_decode_rows_and_reordered_filter(self):
        self.relay.enable_eager_seq_lens_cpu()
        self.publish([11, 31])
        self.publish([22], [2])
        self.batch.req_pool_indices_cpu = Tensor(self.device, [3, 2, 1])
        self.relay.resolve_seq_lens_cpu(self.batch)
        self.assertEqual(self.batch.seq_lens_cpu.values, [31, 22, 11])

    def test_default_keeps_lazy_mirror(self):
        self.assertIsNone(self.publish([11, 31]))
        self.relay.resolve_seq_lens_cpu(self.batch)
        self.assertEqual(self.batch.seq_lens_cpu.values, [11, 31])
        self.assertEqual(self.relay.fwd_prepare_d2h_stream.host_syncs, 1)

    def test_empty_batch_has_no_ticket_and_publish_does_not_advance_generation(self):
        self.relay.enable_eager_seq_lens_cpu()
        self.publish([], [])
        self.batch.spec_info.future_indices = Tensor(self.device, [])
        self.assertIsNone(self.relay.get_seq_lens_cpu_ticket(self.batch))
        self.assertEqual(self.relay._seq_lens_cpu_generation, 0)


class ModeTests(unittest.TestCase):
    def configure(self, mode, **overrides):
        path = "python/sglang/srt/managers/scheduler.py"
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        owner = next(n for n in tree.body if getattr(n, "name", "") == "Scheduler")
        init = next(n for n in owner.body if getattr(n, "name", "") == "init_overlap")
        start = next(
            i
            for i, n in enumerate(init.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "decode_ready_mode"
        )
        wrapper = ast.parse("def configure(self):\n    pass\n").body[0]
        wrapper.body = init.body[start : start + 4]
        namespace = dict(
            envs=NS(SGLANG_PDMUX_DECODE_READY=NS(get=lambda: mode)),
            is_cuda=lambda: overrides.pop("cuda", True),
            logger=Mock(),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
                path,
                "exec",
            ),
            namespace,
        )
        scheduler = NS(
            enable_overlap=True,
            enable_pdmux=True,
            pdmux_standard=False,
            spec_algorithm=NS(is_dflash=lambda: True),
            ps=NS(tp_size=1, pp_size=1, attn_dp_size=1),
            future_map=NS(enable_eager_seq_lens_cpu=Mock()),
        )
        for name, value in overrides.items():
            if name != "cuda":
                setattr(scheduler, name, value)
        namespace["configure"](scheduler)
        return scheduler

    def test_off_keeps_unsupported_platforms_unchanged(self):
        scheduler = self.configure("off", cuda=False, enable_pdmux=False)
        scheduler.future_map.enable_eager_seq_lens_cpu.assert_not_called()
        self.assertFalse(scheduler._pdmux_decode_ready_dispatch)

    def test_experiment_modes_enable_only_the_requested_increment(self):
        for mode in ("mirror", "dispatch"):
            scheduler = self.configure(mode)
            scheduler.future_map.enable_eager_seq_lens_cpu.assert_called_once()
            self.assertEqual(scheduler._pdmux_decode_ready_dispatch, mode == "dispatch")

    def test_invalid_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "off, mirror or dispatch"):
            self.configure("typo")

    def test_unsupported_experiments_fail_before_running(self):
        for overrides in (
            dict(cuda=False),
            dict(enable_overlap=False),
            dict(enable_pdmux=False),
            dict(pdmux_standard=True),
            dict(spec_algorithm=NS(is_dflash=lambda: False)),
            dict(ps=NS(tp_size=2, pp_size=1, attn_dp_size=1)),
            dict(ps=NS(tp_size=1, pp_size=2, attn_dp_size=1)),
            dict(ps=NS(tp_size=1, pp_size=1, attn_dp_size=2)),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "TP=1"),
            ):
                self.configure("dispatch", **overrides)


class StopLoop(Exception):
    pass


class DispatcherTests(unittest.TestCase):
    def run_loop(
        self,
        *,
        before=0,
        after=2,
        budget=4,
        layers=20,
        ticks=1,
        enabled=True,
        temporal_layers=0,
    ):
        trace = []
        d = Device()
        prefill_stream, decode_stream = Stream(d), Stream(d)
        if temporal_layers:
            prefill_stream = decode_stream
        # Completion votes in these tests are immediately ready. Ticket
        # readiness below is controlled independently by completed host submits.
        prefill_stream.record_event = lambda: NS(query=lambda: True)
        decode_stream.record_event = lambda: NS(query=lambda: True)
        torch = NS(
            cuda=NS(stream=d.stream, empty_cache=lambda: None),
            ones=lambda *a, **kw: NS(item=lambda: 1),
            int32=int,
        )
        namespace = dict(
            torch=torch,
            deque=deque,
            logger=Mock(),
            time=NS(perf_counter=lambda: 0),
            NVTX_SCHEDULER_ENABLED=False,
            profile_range=lambda *a, **kw: contextlib.nullcontext(),
            get_current_stream_idx=lambda: 0 if temporal_layers else 1,
            set_pdmux_status=lambda _: None,
            publish_carried_tensors=lambda *a: None,
            dist=NS(ReduceOp=NS(SUM="sum")),
        )
        methods = {
            name: load_method(MIXIN, "SchedulerMultiplexMixin", name, **namespace)
            for name in (
                "event_loop_pdmux_layer_split",
                "_submit_pdmux_prefill_layer",
                "_wait_temporal_prefill_credit",
            )
        }
        s = type("Scheduler", (), methods)()
        batch = NS(
            is_empty=lambda: False,
            has_grammar=False,
            sampling_info=None,
            batch_size=lambda: 1,
            check_decode_mem=lambda: True,
            copy=lambda: object(),
        )
        s.running_batch = batch
        s.split_prefill_batch = NS(
            split_index=0, extend_num_tokens=2048, split_prefill_finished=False
        )
        s._pdmux_decode_ready_dispatch = enabled
        s.pdmux_temporal_layers = temporal_layers
        s._temporal_prefill_done = None
        s.enable_overlap = True
        s.spec_algorithm = NS(is_none=lambda: False)
        s.ps = NS(attn_dp_size=1, tp_size=1)
        s.model_config = NS(num_hidden_layers=layers)
        s.stream_groups = [(prefill_stream, decode_stream)] * 3
        s.sm_counts = [(108, 0), (88, 108), (0, 108)]
        s._check_pdmux_dp_graph_capability = lambda: None
        s.HICACHE_PUMP_INTERVAL = 16
        s.check_hicache_events_if_enabled = lambda: False
        s.update_split_prefill_batch = lambda *a, **kw: (False, batch)
        state = dict(tick=0, prefill=0, ready_at=before)

        def ingest():
            if state["tick"] == ticks:
                raise StopLoop()
            state["tick"] += 1
            trace.append(("ingest",))

        def prepare(batch):
            trace.append(("prepare",))
            return batch

        def run(batch_to_run):
            if batch_to_run is s.split_prefill_batch:
                trace.append(
                    ("P", batch_to_run.split_index, batch_to_run.split_forward_count)
                )
                state["prefill"] += batch_to_run.split_forward_count
            else:
                trace.append(("D",))
                state["ready_at"] = state["prefill"] + after
            return object()

        s.ingest_requests = ingest
        s.update_running_batch = prepare
        s.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=lambda b: b)
        s._get_split_forward_count = lambda _: min(
            temporal_layers or budget, layers - s.split_prefill_batch.split_index
        )
        s.run_batch = run
        s.process_batch_result = lambda *a: trace.append(("process",))
        s.tp_cpu_group = NS(allreduce=lambda *a: NS(wait=lambda: None))
        s.future_map = NS(
            get_seq_lens_cpu_ticket=lambda _: object(),
            seq_lens_cpu_ticket_ready=lambda _: state["prefill"] >= state["ready_at"],
        )

        def merge(*a):
            trace.append(("merge",))
            s.split_prefill_batch = None
            return batch

        s._merge_finished_prefill_batch = merge
        with self.assertRaises(StopLoop):
            s.event_loop_pdmux_layer_split()
        return trace

    def work(self, trace):
        return [row for row in trace if row[0] in ("P", "D")]

    def test_pending_ticket_uses_budget_before_decode_then_remainder_after(self):
        trace = self.run_loop(before=2, after=2)
        self.assertEqual(
            self.work(trace),
            [("P", 0, 1), ("P", 1, 1), ("D",), ("P", 2, 1), ("P", 3, 1)],
        )
        self.assertEqual(trace.count(("prepare",)), 1)

    def test_ready_during_prefill_yields_at_next_layer(self):
        self.assertEqual(
            self.work(self.run_loop(after=2)), [("D",), ("P", 0, 1), ("P", 1, 1)]
        )

    def test_pending_forever_cannot_exceed_original_round_budget(self):
        trace = self.run_loop(before=100, after=100, budget=3)
        self.assertEqual(
            self.work(trace), [("P", 0, 1), ("P", 1, 1), ("P", 2, 1), ("D",)]
        )

    def test_fast_decode_still_gives_prefill_one_layer(self):
        self.assertEqual(self.work(self.run_loop(after=0)), [("D",), ("P", 0, 1)])

    def test_final_layer_before_decode_finalizes_and_merges_once(self):
        trace = self.run_loop(before=100, layers=2)
        self.assertEqual(self.work(trace), [("P", 0, 1), ("P", 1, 1), ("D",)])
        self.assertEqual(trace.count(("merge",)), 1)
        self.assertEqual(trace.count(("process",)), 1)

    def test_result_and_decode_preparation_are_not_repeated_by_polling(self):
        trace = self.run_loop(before=2, after=2, ticks=3)
        self.assertEqual(trace.count(("prepare",)), 3)
        self.assertEqual(trace.count(("D",)), 3)
        self.assertEqual(trace.count(("process",)), 2)

    def test_disabled_dispatch_keeps_original_interval(self):
        self.assertEqual(self.work(self.run_loop(enabled=False)), [("D",), ("P", 0, 4)])


if __name__ == "__main__":
    unittest.main()
