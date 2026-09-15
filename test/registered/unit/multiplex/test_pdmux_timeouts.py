"""Run production timeout methods with CPU-only scheduler collaborators."""

import contextlib
import logging
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

from test_pdmux_sxf_port import bind, load_methods, register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FinishAbort:
    def __init__(self, message, status_code):
        self.message = message
        self.status_code = status_code

    def to_json(self):
        return dict(type="abort", message=self.message, status_code=self.status_code)


def request(rid, *, waiting=0, running=0, finished=False):
    return SimpleNamespace(
        rid=rid,
        time_stats=SimpleNamespace(
            wait_queue_entry_time=waiting,
            forward_entry_time=running,
            trace_ctx=SimpleNamespace(abort=Mock()),
        ),
        finished=lambda: finished,
        to_finish=None,
    )


def batch(*reqs):
    return SimpleNamespace(reqs=list(reqs))


class TestPdmuxTimeouts(unittest.TestCase):
    def scheduler(self, waiting=10, running=10):
        self.clock = Mock(return_value=100)
        self.broadcast = Mock(side_effect=lambda value, *args, **kwargs: value)
        methods = load_methods(
            "multiplex/multiplexing_mixin.py",
            "SchedulerMultiplexMixin",
            ["_check_pdmux_timeouts", "_extra_inflight_batches"],
            envs=SimpleNamespace(
                SGLANG_REQ_WAITING_TIMEOUT=SimpleNamespace(get=lambda: waiting),
                SGLANG_REQ_RUNNING_TIMEOUT=SimpleNamespace(get=lambda: running),
            ),
            time=SimpleNamespace(perf_counter=self.clock),
            broadcast_pyobj=self.broadcast,
            FINISH_ABORT=FinishAbort,
            HTTPStatus=HTTPStatus,
            AbortReq=SimpleNamespace,
        )
        return bind(
            SimpleNamespace(
                pdmux_standard=True,
                running_batch=batch(),
                last_batch=None,
                split_prefill_batch=None,
                _pdmux_prefill_pending=None,
                _pdmux_prefill_inflight=None,
                _pending_chunked_abort_req=None,
                chunked_req=None,
                waiting_queue=[],
                ps=SimpleNamespace(
                    attn_tp_rank=0, attn_cp_rank=0, attn_tp_size=1, attn_cp_size=1
                ),
                enable_hicache_storage=True,
                tree_cache=Mock(),
                ipc_channels=SimpleNamespace(send_to_tokenizer=Mock()),
            ),
            methods,
        )

    def test_disabled_does_not_inspect_batches_or_sync(self):
        scheduler = self.scheduler(waiting=-1, running=-1)
        del scheduler.running_batch
        scheduler._check_pdmux_timeouts()
        self.clock.assert_not_called()
        self.broadcast.assert_not_called()

    def test_waiting_timeout_uses_exact_id_and_releases_prefetch(self):
        scheduler = self.scheduler(running=-1)
        expired = request("req", waiting=1)
        survivors = [request("req-longer", waiting=95), request("unset")]
        scheduler.waiting_queue = [expired, *survivors]
        scheduler._check_pdmux_timeouts()
        self.assertEqual(scheduler.waiting_queue, survivors)
        scheduler.tree_cache.release_aborted_request.assert_called_once_with("req")
        output, req = (
            scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args
        )
        self.assertIs(req, expired)
        self.assertEqual(output.finished_reason["status_code"], 503)
        self.assertEqual(
            output.finished_reason["message"], "Request waiting timeout reached."
        )

    def test_running_timeout_covers_both_lanes_and_parked_chunk(self):
        for standard in (False, True):
            with self.subTest(standard=standard):
                scheduler = self.scheduler(waiting=-1)
                scheduler.pdmux_standard = standard
                decode, prefill, chunk = [
                    request(rid, running=1) for rid in ("d", "p", "c")
                ]
                scheduler.running_batch = batch(decode)
                scheduler.last_batch = batch(decode)  # Shared ownership, one decision.
                scheduler.chunked_req = chunk
                if standard:
                    scheduler._pdmux_prefill_pending = batch(prefill)
                    scheduler._pdmux_prefill_inflight = SimpleNamespace(
                        batch=batch(prefill)
                    )
                else:
                    scheduler.split_prefill_batch = batch(prefill)
                scheduler._check_pdmux_timeouts()
                for req in (decode, prefill, chunk):
                    self.assertEqual(req.to_finish.status_code, 503)
                self.assertIs(scheduler._pending_chunked_abort_req, chunk)
                # No KV or request-slot teardown while a lane can still read it.
                self.assertEqual(scheduler.tree_cache.mock_calls, [])
                scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_finished_unstarted_recent_and_already_aborting_are_untouched(self):
        scheduler = self.scheduler()
        finished = request("finished", running=1, finished=True)
        aborting = request("aborting", running=1)
        previous_reason = aborting.to_finish = object()
        reqs = [finished, aborting, request("unstarted"), request("recent", running=95)]
        scheduler.running_batch = batch(*reqs)
        scheduler._check_pdmux_timeouts()
        self.assertIs(aborting.to_finish, previous_reason)
        self.assertTrue(all(req.to_finish is None for req in (reqs[0], *reqs[2:])))

    def test_peer_uses_leaders_ids_without_reading_local_clock(self):
        scheduler = self.scheduler()
        scheduler.ps.attn_tp_rank = 1
        scheduler.ps.attn_tp_size = 2
        scheduler.attn_tp_group = SimpleNamespace(rank=1, ranks=[4, 5])
        scheduler.attn_tp_cpu_group = "dp1-attn-tp"
        # Peer timestamps differ; only the leader's expiry decision is used.
        req = request("peer-request", running=9999)
        scheduler.running_batch = batch(req)
        self.broadcast.side_effect = None
        self.broadcast.return_value = [(req.rid, "running")]
        scheduler._check_pdmux_timeouts()
        self.clock.assert_not_called()
        self.broadcast.assert_called_once_with(None, 1, "dp1-attn-tp", src=4)
        self.assertEqual(req.to_finish.status_code, 503)

    def test_leader_broadcasts_deduplicated_ids_within_attention_group(self):
        scheduler = self.scheduler()
        scheduler.ps.attn_tp_size = 2
        scheduler.attn_tp_group = SimpleNamespace(rank=0, ranks=[4, 5])
        scheduler.attn_tp_cpu_group = "dp1-attn-tp"
        req = request("local-dp-request", running=1)
        scheduler.running_batch = scheduler.last_batch = batch(req)
        scheduler._check_pdmux_timeouts()
        self.broadcast.assert_called_once_with(
            [(req.rid, "running")], 0, "dp1-attn-tp", src=4
        )

    def test_chunk_cleanup_preserves_timeout_reason_at_safe_point(self):
        scheduler = self.scheduler()
        req = request("chunk", running=1)
        scheduler.chunked_req = req
        scheduler._check_pdmux_timeouts()
        release = Mock()
        prepare = Mock()
        disagg = SimpleNamespace(PREFILL="prefill")
        scheduler.disaggregation_mode = None
        method = load_methods(
            "managers/scheduler.py",
            "Scheduler",
            ["process_pending_chunked_abort"],
            FINISH_ABORT=FinishAbort,
            prepare_abort=prepare,
            release_kv_cache=release,
            DisaggregationMode=disagg,
            AbortReq=SimpleNamespace,
            logger=logging.getLogger(__name__),
        )["process_pending_chunked_abort"]
        method(scheduler)
        prepare.assert_called_once_with(req, "Request running timeout reached.", 503)
        release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        output, _ = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args
        self.assertEqual(output.finished_reason["status_code"], 503)

    def test_both_loops_check_timeouts_before_launching_work(self):
        class Checked(Exception):
            pass

        stream = SimpleNamespace()
        for name in ("event_loop_pdmux_standard", "event_loop_pdmux_layer_split"):
            with self.subTest(loop=name):
                check = Mock(side_effect=Checked)
                scheduler = SimpleNamespace(
                    stream_groups=[(stream, stream)],
                    tp_cpu_group=None,
                    request_receiver=SimpleNamespace(
                        recv_requests=Mock(return_value=[])
                    ),
                    process_input_requests=Mock(),
                    _check_pdmux_timeouts=check,
                )
                method = load_methods(
                    "multiplex/multiplexing_mixin.py",
                    "SchedulerMultiplexMixin",
                    [name],
                    torch=SimpleNamespace(
                        cuda=SimpleNamespace(
                            stream=lambda _: contextlib.nullcontext(),
                            empty_cache=lambda: None,
                        )
                    ),
                    dist=SimpleNamespace(get_world_size=lambda group: 1),
                    get_current_stream_idx=lambda: 0,
                    set_pdmux_status=lambda _: None,
                    logger=logging.getLogger(__name__),
                )[name]
                with self.assertRaises(Checked):
                    method(scheduler)
                check.assert_called_once_with()
                scheduler.process_input_requests.assert_called_once_with([])


if __name__ == "__main__":
    unittest.main()
