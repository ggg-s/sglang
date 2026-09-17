"""Exercise PDMux integration contracts without importing the GPU runtime.

Load the actual method bodies, replacing only their collaborators. This keeps
the merge regressions runnable on hosts without Linux or CUDA dependencies.
"""

from __future__ import annotations

import ast
import contextlib
import enum
import importlib.util
import sys
import unittest
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]

# Avoid importing sglang.__init__, which loads the platform runtime.
_ci_spec = importlib.util.spec_from_file_location(
    "_pdmux_compat_ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
_ci = importlib.util.module_from_spec(_ci_spec)
sys.modules[_ci_spec.name] = _ci
_ci_spec.loader.exec_module(_ci)
register_cpu_ci = _ci.register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def load_method(path, owner, name, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if getattr(node, "name", None) == owner)
    method = next(node for node in cls.body if getattr(node, "name", None) == name)
    method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


class TestPdmuxMainCompat(unittest.TestCase):
    def test_standard_loop_broadcasts_timeout_aborts_through_ingest(self):
        class InputProcessed(Exception):
            pass

        ingest = load_method(
            "python/sglang/srt/managers/scheduler.py", "Scheduler", "ingest_requests"
        )
        loop = load_method(
            "python/sglang/srt/multiplex/multiplexing_mixin.py",
            "SchedulerMultiplexMixin",
            "event_loop_pdmux_standard",
            dist=SimpleNamespace(get_world_size=lambda **kwargs: 2),
            get_current_stream_idx=lambda: 0,
            torch=SimpleNamespace(
                cuda=SimpleNamespace(
                    empty_cache=lambda: None,
                    stream=lambda stream: contextlib.nullcontext(),
                )
            ),
            set_pdmux_status=Mock(),
            logger=Mock(),
        )
        abort = SimpleNamespace(
            rid="expired", abort_message="Request running timeout reached."
        )
        for ranks in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            with self.subTest(ranks=ranks):
                scheduler = SimpleNamespace(
                    ps=SimpleNamespace(
                        pp_rank=ranks[0], attn_tp_rank=ranks[1], attn_cp_rank=ranks[2]
                    ),
                    tp_cpu_group=object(),
                    stream_groups=[(object(), object())],
                    _poll_timeout_aborts=Mock(return_value=[abort]),
                    request_receiver=SimpleNamespace(
                        recv_requests=Mock(return_value=[abort])
                    ),
                    metrics_reporter=Mock(),
                    process_input_requests=Mock(side_effect=InputProcessed),
                )
                scheduler.ingest_requests = lambda: ingest(scheduler)
                # Stop after dispatch: this drives the real loop's input path
                # without submitting model work or requiring a CUDA device.
                with self.assertRaises(InputProcessed):
                    loop(scheduler)
                expected_local = [abort] if ranks == (0, 0, 0) else []
                scheduler.request_receiver.recv_requests.assert_called_once_with(
                    local_reqs=expected_local
                )
                self.assertEqual(
                    scheduler._poll_timeout_aborts.call_count, int(ranks == (0, 0, 0))
                )
                scheduler.process_input_requests.assert_called_once_with([abort])
                scheduler.metrics_reporter.record_scheduler_active.assert_called_once()

    def test_running_timeout_covers_pending_and_inflight_prefills(self):
        poll = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_poll_timeout_aborts",
            envs=SimpleNamespace(
                SGLANG_REQ_WAITING_TIMEOUT=SimpleNamespace(get=lambda: 0),
                SGLANG_REQ_RUNNING_TIMEOUT=SimpleNamespace(get=lambda: 10),
            ),
            time=SimpleNamespace(perf_counter=lambda: 100),
            AbortReq=SimpleNamespace,
            HTTPStatus=HTTPStatus,
        )

        def req(rid, entry=1, done=False):
            return SimpleNamespace(
                rid=rid,
                finished=lambda: done,
                time_stats=SimpleNamespace(forward_entry_time=entry),
            )

        pending, inflight = req("pending"), req("inflight")
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_size=1),
            running_batch=SimpleNamespace(reqs=[req("decode")]),
            last_batch=None,
            _extra_inflight_batches=lambda: [
                SimpleNamespace(reqs=[pending, req("unstamped", entry=0)]),
                SimpleNamespace(
                    reqs=[inflight, req("fresh", entry=99), req("done", done=True)]
                ),
                SimpleNamespace(reqs=[inflight]),
            ],
        )
        aborts = poll(scheduler)
        self.assertCountEqual(
            [abort.rid for abort in aborts], ["decode", "pending", "inflight"]
        )
        self.assertTrue(
            all(abort.finished_reason["status_code"] == 503 for abort in aborts)
        )

    def test_split_mode_participates_in_cp(self):
        mode = enum.Enum("Mode", "EXTEND MIXED SPLIT_PREFILL DRAFT_EXTEND_V2 DECODE")
        active = load_method(
            "python/sglang/srt/model_executor/forward_batch_info.py",
            "ForwardMode",
            "is_context_parallel_extend",
            ForwardMode=mode,
        )
        self.assertTrue(active(mode.SPLIT_PREFILL))
        self.assertFalse(active(mode.DECODE))
        self.assertFalse(active(mode.DRAFT_EXTEND_V2))

    def test_split_cp_is_planned_before_attention_and_only_once(self):
        journal = []
        cp = SimpleNamespace(
            is_cp_active=lambda batch: True,
            prepare_cp_forward=lambda batch: journal.append("cp"),
        )
        forward = load_method(
            "python/sglang/srt/model_executor/model_runner.py",
            "ModelRunner",
            "forward_split_prefill",
            device_timer_ctx=lambda *args: contextlib.nullcontext(),
        )
        runner = SimpleNamespace(
            attn_backend=SimpleNamespace(
                init_forward_metadata=lambda batch: journal.append("metadata")
            ),
            model_config=SimpleNamespace(num_hidden_layers=2),
            device_timer=None,
            model=SimpleNamespace(
                forward_split_prefill=Mock(side_effect=[None, "logits"])
            ),
        )
        batch = SimpleNamespace(split_index=0, input_ids=object(), positions=object())
        with patch.dict(sys.modules, {"sglang.srt.layers.cp.utils": cp}):
            self.assertIsNone(forward(runner, batch))
            self.assertEqual(forward(runner, batch), "logits")
        self.assertEqual(journal, ["cp", "metadata"])
        self.assertEqual(batch.split_index, 2)
        self.assertEqual(runner.model.forward_split_prefill.call_args.args[-1], (1, 2))

    def test_abort_collection_includes_pending_and_inflight_prefills(self):
        collect = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "collect_inflight_reqs",
        )
        decode, pending, inflight = object(), object(), object()
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_size=1),
            running_batch=SimpleNamespace(reqs=[decode]),
            last_batch=None,
            _extra_inflight_batches=lambda: [
                SimpleNamespace(reqs=[pending]),
                SimpleNamespace(reqs=[inflight, decode]),
            ],
        )
        self.assertEqual(collect(scheduler), {decode, pending, inflight})

    def test_standard_prefill_relays_with_batch_and_pins_worker_refs(self):
        submit = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_run_pdmux_standard_prefill",
            resolve_forward_inputs=Mock(),
        )
        worker_ref = object()
        result = SimpleNamespace(extra_keep_alive_refs=[worker_ref])
        batch = SimpleNamespace(
            return_logprob=False,
            return_hidden_states=False,
            req_pool_indices=object(),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        scheduler = SimpleNamespace(
            pdmux_prefill_stream=object(),
            future_map=object(),
            _isolate_forward_fields=lambda batch: contextlib.nullcontext(),
            _snapshot_batch_attrs=lambda batch: [batch.req_pool_indices],
            model_worker=SimpleNamespace(forward_batch_generation=lambda batch: result),
            _relay_forward_payload=Mock(),
            device_module=SimpleNamespace(Event=object),
            _launch_result_copy=Mock(),
            pdmux_prefill_copy_stream=object(),
            pdmux_prefill_copy_stream_ctx=contextlib.nullcontext(),
            update_cache_from_scheduler=Mock(),
        )
        self.assertIs(submit(scheduler, batch), result)
        scheduler._relay_forward_payload.assert_called_once_with(
            batch, batch.req_pool_indices, result
        )
        self.assertIs(result.extra_keep_alive_refs[0], batch)
        self.assertIn(worker_ref, result.extra_keep_alive_refs)
        self.assertIsNone(batch.input_ids)
        scheduler._launch_result_copy.assert_called_once()

    def test_external_linker_events_reach_the_scheduler(self):
        pump = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "check_hicache_events_if_enabled",
            get_memory=lambda: SimpleNamespace(enable_flexkv=False),
        )
        scheduler = SimpleNamespace(
            enable_hierarchical_cache=False,
            enable_unified_cache_external_linker=True,
            tree_cache=SimpleNamespace(check_hicache_events=Mock(return_value=True)),
        )
        self.assertTrue(pump(scheduler))
        scheduler.tree_cache.check_hicache_events.assert_called_once()

    def test_cache_ack_only_does_not_block_decode_but_writes_publish_dependency(self):
        pump = load_method(
            "python/sglang/srt/mem_cache/unified_radix_cache.py",
            "UnifiedRadixCache",
            "check_hicache_events",
        )
        for policy, writes, pipeline, expected in [
            ("write_through", 2, None, False),
            ("write_back", 0, None, False),
            ("write_back", 1, None, True),
            ("write_through", 0, SimpleNamespace(flush_pending_writes=Mock()), True),
        ]:
            with self.subTest(policy=policy, writes=writes, pipeline=pipeline):
                cache = SimpleNamespace(
                    linker=None,
                    _drain_async_work=Mock(),
                    cache_controller=SimpleNamespace(write_policy=policy),
                    _sync_hicache_ready_counts=lambda: (writes, 0, [], []),
                    writing_check=Mock(),
                    loading_check=Mock(),
                    enable_storage=False,
                    buffer_pipeline=pipeline,
                    enable_storage_metrics=False,
                )
                self.assertIs(pump(cache), expected)
                cache.writing_check.assert_called_once_with(finish_count=writes)
                if pipeline is not None:
                    pipeline.flush_pending_writes.assert_called_once()


if __name__ == "__main__":
    unittest.main()
