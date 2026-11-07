"""Exercise PDMux integration contracts without importing the GPU runtime.

Load the actual method bodies, replacing only their collaborators. This keeps
the merge regressions runnable on hosts without Linux or CUDA dependencies.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import enum
import importlib.util
import sys
import time
import unittest
from collections import deque
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
    cls = (
        tree
        if owner is None
        else next(node for node in tree.body if getattr(node, "name", None) == owner)
    )
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
    def test_active_split_graph_pins_its_lane_until_final_interval(self):
        """Repartitioning midway would select another graph's activations."""
        selected, trace = [0], []
        activate = load_method(
            "python/sglang/srt/multiplex/multiplexing_mixin.py",
            "SchedulerMultiplexMixin",
            "_activate_pdmux_stream_group",
            get_current_stream_idx=lambda: selected[0],
            set_current_stream_idx=lambda index: selected.__setitem__(0, index),
        )
        executor = SimpleNamespace(owner=object())
        stream = SimpleNamespace(synchronize=lambda: trace.append("drain"))
        groups = [(stream, stream), (object(), object())]
        scheduler = SimpleNamespace(
            stream_groups=groups,
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=SimpleNamespace(
                        pdmux_split_executor=executor
                    )
                )
            ),
            _update_decode_attn_backends=lambda index: trace.append(index),
        )
        self.assertEqual(activate(scheduler, 1), (0, groups[0]))
        self.assertEqual(trace, [])
        executor.owner = None
        self.assertEqual(activate(scheduler, 1), (1, groups[1]))
        self.assertEqual(trace, ["drain", "drain", 1])

    def test_fresh_forward_batch_can_enter_split_without_adapter_fields(self):
        """The real worker creates a fresh FB; it must own its initial cursor.

        Keep the production dataclass fields/defaults and omit GPU methods so
        this reproduces the worker boundary without importing CUDA packages.
        """
        path = ROOT / "python/sglang/srt/model_executor/forward_batch_info.py"
        cls = next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "ForwardBatch"
        )
        cls.bases = []
        cls.body = [n for n in cls.body if isinstance(n, ast.AnnAssign)]
        ns = {"dataclass": dataclasses.dataclass, "__name__": __name__}
        body = [ast.parse("from __future__ import annotations").body[0], cls]
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                str(path),
                "exec",
            ),
            ns,
        )
        frame_type = ns["ForwardBatch"]
        required = {
            f.name: None
            for f in dataclasses.fields(frame_type)
            if f.default is dataclasses.MISSING
        }
        first, second = frame_type(**required), frame_type(**required)
        run = load_method(
            "python/sglang/srt/model_executor/runner_utils/pdmux.py",
            None,
            "run_split_prefill",
            device_timer_ctx=lambda *a: contextlib.nullcontext(),
        )

        def eager(ids, positions, batch, interval):
            if interval[0] == 0:
                self.assertIsNone(batch.model_specific_states)
                batch.model_specific_states = {"saved": interval[1]}
                return None
            return batch.model_specific_states.pop("saved")

        runner = SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=2),
            device_timer=None,
            attn_backend=SimpleNamespace(init_forward_metadata=lambda batch: None),
            model=SimpleNamespace(forward_split_prefill=eager),
        )
        cp = SimpleNamespace(
            is_cp_active=lambda batch: False, prepare_cp_forward=lambda batch: None
        )
        with patch.dict(sys.modules, {"sglang.srt.layers.cp.utils": cp}):
            self.assertIsNone(run(runner, first))
            self.assertIsNone(second.model_specific_states)
            self.assertEqual(run(runner, first), 1)
            self.assertEqual(first.split_index, 2)
            self.assertFalse(first.split_prefill_can_run_graph)
            runner.prefill_cuda_graph_runner = SimpleNamespace(
                can_run_split=lambda *args: True,
                execute_split=lambda *args: "graph-logits",
            )
            self.assertEqual(run(runner, second, forward_count=2), "graph-logits")
            self.assertTrue(second.split_prefill_can_run_graph)
            self.assertEqual(second.split_index, 2)

    def test_split_bcg_interval_is_not_forwarded_to_the_model_tail(self):
        """Replay controls must not escape into arbitrary model.forward APIs."""
        execute = load_method(
            "python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py",
            "PrefillCudaGraphRunner",
            "_execute_body_capture",
        )
        calls = []
        original = lambda *args: "eager"
        layer = SimpleNamespace(forward=original)

        def model_forward(ids, positions, batch):
            return layer.forward(ids, positions, batch) + "-logits"

        def replay(key, batch, **kwargs):
            calls.append(kwargs["pdmux_layer_interval"])
            return "hidden"

        runner = SimpleNamespace(
            _is_full_backend=False,
            is_dsv41_model=False,
            layer_model=layer,
            model_runner=SimpleNamespace(
                model=SimpleNamespace(forward=model_forward),
                pp_group=SimpleNamespace(is_first_rank=False),
            ),
            backend=SimpleNamespace(replay=replay),
            _restore_qwen_bcg_hc_output=lambda out: out,
            _use_draft_input_embeds=False,
            _prefill_forward_context=lambda *a, **kw: contextlib.nullcontext(),
        )
        batch = SimpleNamespace(input_ids=[1], positions=[0], mm_input_embeds=None)
        self.assertEqual(
            execute(runner, batch, batch, 1, 1, "key", pdmux_layer_interval=(1, 2)),
            "hidden-logits",
        )
        self.assertEqual(calls, [(1, 2)])
        self.assertIs(layer.forward, original)

    def test_prefill_bcg_does_not_borrow_decode_graph_storage(self):
        """Concurrent lane graphs need distinct pools even with private inputs."""
        global_pool, private_pool = object(), object()
        borrowing = [True]
        factory = load_method(
            "python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py",
            "BreakableCudaGraphBackend",
            "capture_session",
            get_or_create_global_graph_memory_pool=lambda device: global_pool,
            disable_graph_pool_borrow=lambda reason: borrowing.__setitem__(0, False),
            set_graph_pool_id=lambda pool: None,
        )
        backend = SimpleNamespace(
            _pool=None,
            _cuda_graph_runner=SimpleNamespace(enable_pdmux=True),
            _device_module=SimpleNamespace(graph_pool_handle=lambda: private_pool),
            begin_cuda_graph_capture=lambda: None,
            end_cuda_graph_capture=lambda: None,
            replay_session=contextlib.nullcontext,
        )
        with contextlib.contextmanager(factory)(backend, "prefill-stream"):
            self.assertIs(backend._pool, private_pool)
            self.assertIsNot(backend._pool, global_pool)
            self.assertFalse(borrowing[0])
        self.assertIsNone(backend._capture_stream)

    def test_prefill_bcg_admits_plain_tp_and_rejects_unisolated_profiles(self):
        """BCG must not accidentally admit shared speculative/DP/EP state."""
        check = load_method(
            "python/sglang/srt/arg_groups/validation_hook.py",
            None,
            "check_pdmux_prefill_bcg",
        )
        valid = dict(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="breakable")
            ),
            speculative_algorithm=None,
            tp_size=8,
            pp_size=1,
            ep_size=1,
            attn_cp_size=1,
            dcp_size=1,
            enable_dp_attention=False,
        )
        check(SimpleNamespace(**valid))
        for field, value in (
            ("speculative_algorithm", "EAGLE"),
            ("enable_dp_attention", True),
            ("ep_size", 2),
            ("attn_cp_size", 2),
            ("dcp_size", 2),
            ("pp_size", 2),
            ("enable_prefill_cp", True),
        ):
            with self.subTest(field=field):
                with self.assertRaises(AssertionError):
                    check(SimpleNamespace(**(valid | {field: value})))

    @staticmethod
    def bind_inflight_batches(scheduler):
        method = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "_collect_inflight_batches",
            get_parallel=lambda: scheduler.ps,
        )
        scheduler.enable_continuous_input_polling = False
        scheduler.enable_pdmux = True
        scheduler.split_prefill_batch = None
        scheduler._collect_inflight_batches = lambda: method(scheduler)

    def test_standard_loop_broadcasts_timeout_aborts_through_ingest(self):
        class InputProcessed(Exception):
            pass

        ingest = load_method(
            "python/sglang/srt/managers/scheduler.py",
            "Scheduler",
            "ingest_requests",
            time=time,
            get_parallel=lambda: scheduler.ps,
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
            deque=deque,
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
                    uocr_runtime=None,
                    _deferred_input_requests=[],
                    tp_cpu_group=object(),
                    stream_groups=[(object(), object())],
                    enable_overlap=False,
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
                to_finish=None,
                time_stats=SimpleNamespace(forward_entry_time=entry),
            )

        pending, inflight = req("pending"), req("inflight")
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(pp_size=1),
            enable_pdmux=True,
            split_prefill_batch=None,
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
        self.bind_inflight_batches(scheduler)
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
            "python/sglang/srt/model_executor/forward_mode.py",
            "ForwardMode",
            "is_context_parallel_extend",
            ForwardMode=mode,
        )
        self.assertTrue(active(mode.SPLIT_PREFILL))
        self.assertFalse(active(mode.DECODE))
        self.assertFalse(active(mode.DRAFT_EXTEND_V2))

    def test_split_cp_restores_padding_each_segment_and_plans_attention_once(self):
        journal = []
        cp = SimpleNamespace(
            is_cp_active=lambda batch: True,
            prepare_cp_forward=lambda batch: journal.append("cp"),
        )
        forward = load_method(
            "python/sglang/srt/model_executor/runner_utils/pdmux.py",
            None,
            "run_split_prefill",
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
        self.assertEqual(journal, ["cp", "metadata", "cp"])
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
        self.bind_inflight_batches(scheduler)
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
                    flush_pending_backups=Mock(),
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
