"""TP overlap admission and lane ownership, without importing the GPU runtime."""

from __future__ import annotations

import ast
import contextlib
import multiprocessing
import tempfile
import threading
import unittest
from collections import deque
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_pdmux_sxf_port import SRT, load_methods, register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def lane_namespace(decode, prefill, *, attention=None, moe=None):
    namespace = dict(
        _TP=decode,
        _ATTN_TP=decode if attention is None else attention,
        _ATTN_CP=decode,
        _MOE_DP=decode,
        _MOE_TP=decode if moe is None else moe,
        _PDMUX_PREFILL_TP_GROUP=prefill,
        _ENABLE_PDMUX_P_TP=ContextVar("test_prefill_lane", default=False),
    )
    functions = load_methods(
        "distributed/parallel_state.py",
        None,
        [
            "set_pdmux_status",
            "get_tp_group",
            "get_attn_tp_group",
            "get_attn_cp_group",
            "get_moe_dp_group",
            "get_moe_tp_group",
        ],
        **namespace,
    )
    # load_methods gives every function the same globals dict.
    return NS(**functions)


def load_constructor(path, name, namespace, *, methods=()):
    tree = ast.parse((SRT / path).read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in ("__init__", *methods)
    ]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = dict(namespace, nn=NS(Module=object))
    exec(compile(ast.fix_missing_locations(module), str(SRT / path), "exec"), namespace)
    return namespace[name]


class TestPdmuxTPGroups(unittest.TestCase):
    def test_deepseek_v4_target_and_dspark_helpers_stay_on_pdmux_streams(self):
        for path, name in (
            ("models/deepseek_v4.py", "use_stream_pool"),
            ("models/deepseek_v4_dspark.py", "use_multi_stream"),
        ):
            tree = ast.parse((SRT / path).read_text())
            assignment = next(
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            )
            expr = compile(ast.Expression(assignment.value), path, "eval")
            for enabled in (False, True):
                with self.subTest(model=path, pdmux=enabled):
                    self.assertEqual(
                        eval(
                            expr,
                            dict(
                                _is_cuda=True,
                                _is_hip=False,
                                _is_npu=False,
                                torch=NS(cuda=NS(is_available=lambda: True)),
                                envs=NS(
                                    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=NS(
                                        get=lambda: True
                                    ),
                                    SGLANG_DSPARK_ENABLE_MULTI_STREAM=NS(
                                        get=lambda: True
                                    ),
                                ),
                                is_pdmux_enabled=lambda: enabled,
                            ),
                        ),
                        not enabled,
                    )

    def test_full_width_attention_and_moe_follow_each_lane(self):
        decode, prefill = object(), object()
        lanes = lane_namespace(decode, prefill)
        for chunk in range(1100):
            for is_prefill, expected in (
                (False, decode),
                (True, prefill),
                (False, decode),
            ):
                lanes.set_pdmux_status(is_prefill)
                self.assertIs(lanes.get_tp_group(), expected)
                self.assertIs(lanes.get_attn_tp_group(), expected)
                self.assertIs(lanes.get_moe_tp_group(), expected)
                self.assertIs(lanes.get_attn_cp_group(), expected)
                self.assertIs(lanes.get_moe_dp_group(), expected)

    def test_real_subgroups_keep_their_membership(self):
        attention, moe = object(), object()
        lanes = lane_namespace(object(), object(), attention=attention, moe=moe)
        for is_prefill in (True, False):
            lanes.set_pdmux_status(is_prefill)
            self.assertIs(lanes.get_attn_tp_group(), attention)
            self.assertIs(lanes.get_moe_tp_group(), moe)

    def test_prefill_group_selection_is_thread_local_for_aliases(self):
        decode, prefill = object(), object()
        lanes = lane_namespace(decode, prefill)
        observed = []

        def submit_prefill():
            lanes.set_pdmux_status(True)
            observed.extend([lanes.get_attn_tp_group(), lanes.get_moe_tp_group()])

        thread = threading.Thread(target=submit_prefill)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observed, [prefill, prefill])
        self.assertIs(lanes.get_attn_tp_group(), decode)
        self.assertIs(lanes.get_moe_tp_group(), decode)


class TestPdmuxTPAdmission(unittest.TestCase):
    def check(self, **overrides):
        cfg = NS(
            pdmux_prefill_mode="layer_split",
            speculative_algorithm="DFLASH",
            tp_size=2,
            enable_dp_attention=False,
            ep_size=1,
            moe_dp_size=1,
            attn_cp_size=1,
            dcp_size=1,
        )
        vars(cfg).update(overrides)
        check = load_methods(
            "arg_groups/validation_hook.py", None, ["check_pdmux_overlap"]
        )
        check["check_pdmux_overlap"](cfg)

    def test_plain_tp_and_hicache_are_admitted(self):
        for algorithm in ("DFLASH", "DSPARK"):
            for size in (1, 2, 4, 8):
                with self.subTest(algorithm=algorithm, tp_size=size):
                    self.check(
                        speculative_algorithm=algorithm,
                        tp_size=size,
                        enable_hierarchical_cache=True,
                    )

    def test_other_prefill_modes_and_algorithms_still_fail(self):
        for overrides in (
            dict(pdmux_prefill_mode="standard"),
            dict(speculative_algorithm="EAGLE"),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(
                    AssertionError, "layer_split prefill and DFLASH"
                ),
            ):
                self.check(**overrides)

    def test_subgroup_parallelism_is_not_silently_enabled(self):
        for algorithm in ("DFLASH", "DSPARK"):
            for name in (
                "enable_dp_attention",
                "ep_size",
                "moe_dp_size",
                "attn_cp_size",
                "dcp_size",
            ):
                with (
                    self.subTest(algorithm=algorithm, name=name),
                    self.assertRaisesRegex(AssertionError, "TP>1"),
                ):
                    self.check(
                        speculative_algorithm=algorithm,
                        **{name: True if name == "enable_dp_attention" else 2},
                    )


class TestPdmuxTPSampling(unittest.TestCase):
    @staticmethod
    def exec_config():
        return NS(
            deterministic=NS(
                rl_on_policy_target=None, enable_deterministic_inference=False
            ),
            kernel=NS(sampling_backend="pytorch"),
            features=NS(
                sampling_mask_max_tokens=0, enable_fp32_lm_head=False, enable_mis=False
            ),
        )

    def test_sampler_sync_uses_each_lanes_group_in_both_modes(self):
        decode, prefill = NS(device_group=object()), NS(device_group=object())
        lanes = lane_namespace(decode, prefill)
        for mode in ("standard", "layer_split"):
            with self.subTest(mode=mode):
                sync = Mock()
                sampler_cls = load_constructor(
                    "layers/sampler.py",
                    "Sampler",
                    dict(
                        get_tp_group=lanes.get_tp_group,
                        get_disagg=lambda: NS(
                            enable_pdmux=True, pdmux_prefill_mode=mode
                        ),
                        is_dp_attention_enabled=lambda: False,
                        get_exec=self.exec_config,
                        OutputLogprobProcessor=object,
                        SYNC_TOKEN_IDS_ACROSS_TP=False,
                        torch=NS(distributed=NS(all_reduce=sync)),
                        dist=NS(ReduceOp=NS(MIN="min")),
                    ),
                    methods=("_sync_token_ids_across_tp",),
                )
                lanes.set_pdmux_status(False)
                sampler = sampler_cls()
                tokens = object()
                for is_prefill, expected in (
                    (True, prefill),
                    (False, decode),
                    (True, prefill),
                ):
                    lanes.set_pdmux_status(is_prefill)
                    sampler._sync_token_ids_across_tp(tokens, NS(grammars=[object()]))
                    self.assertIs(sync.call_args.kwargs["group"], expected.device_group)

    def test_dp_sampler_keeps_its_attention_subgroup(self):
        attention = NS(device_group=object())
        sampler_cls = load_constructor(
            "layers/sampler.py",
            "Sampler",
            dict(
                get_tp_group=lambda: NS(device_group=object()),
                get_parallel=lambda: NS(
                    attn_tp_group=attention, attn_cp_group=NS(device_group=object())
                ),
                get_disagg=lambda: NS(enable_pdmux=True),
                is_dp_attention_enabled=lambda: True,
                get_exec=self.exec_config,
                OutputLogprobProcessor=object,
            ),
        )
        sampler = sampler_cls()
        self.assertFalse(sampler._resolve_tp_sync_group_per_call)
        self.assertIs(sampler.tp_sync_group, attention.device_group)

    def test_pdmux_disables_the_single_shared_symmetric_logits_buffer(self):
        for pdmux, mode, enabled in (
            (False, "layer_split", True),
            (True, "layer_split", False),
            (True, "standard", False),
        ):
            with self.subTest(pdmux=pdmux, mode=mode):
                gatherer = Mock()
                processor_cls = load_constructor(
                    "layers/logits_processor.py",
                    "LogitsProcessor",
                    dict(
                        get_parallel=lambda: NS(
                            enable_dp_lm_head=False,
                            enable_tp_lm_head_all_to_all=False,
                            tp_size=2,
                            attn_dp_size=1,
                        ),
                        get_disagg=lambda: NS(
                            enable_pdmux=pdmux, pdmux_prefill_mode=mode
                        ),
                        get_exec=self.exec_config,
                        triton_symm_mem_ag=NS(
                            MultimemAllGatherer=gatherer,
                            recommended_max_tokens=lambda **kw: 128,
                        ),
                        InputLogprobProcessor=object,
                    ),
                )
                processor_cls(NS(vocab_size=128))
                self.assertIs(gatherer.call_args.kwargs["enabled"], enabled)


def _exercise_chunked_scheduler(rank, group, *, overlap):
    import torch
    import torch.distributed as dist

    class Event:
        def __init__(self):
            self.polls = 0

        def query(self):
            self.polls += 1
            return self.polls > rank + 1

    stream = NS(
        record_event=Event, synchronize=lambda: None, wait_event=lambda event: None
    )
    methods = load_methods(
        "multiplex/multiplexing_mixin.py",
        "SchedulerMultiplexMixin",
        [
            "event_loop_pdmux_layer_split",
            "update_split_prefill_batch",
            "_get_split_forward_count",
        ],
        torch=NS(
            cuda=NS(
                stream=lambda stream: contextlib.nullcontext(), empty_cache=lambda: None
            ),
            ones=torch.ones,
            zeros=torch.zeros,
            int32=torch.int32,
        ),
        dist=dist,
        deque=deque,
        logger=Mock(),
        ForwardMode=NS(SPLIT_PREFILL=object()),
        get_current_stream_idx=lambda: 0,
        set_pdmux_status=lambda value: None,
        publish_carried_tensors=lambda *args: None,
        profile_range=lambda *args, **kw: contextlib.nullcontext(),
        NVTX_SCHEDULER_ENABLED=False,
        time=NS(perf_counter=lambda: 0),
    )
    scheduler = type("Scheduler", (), methods)()
    trace, merged = [], []
    batch = NS(
        is_empty=lambda: False,
        has_grammar=False,
        sampling_info=None,
        batch_size=lambda: 1,
        check_decode_mem=lambda: True,
    )
    batch.copy = lambda: batch
    scheduler.running_batch = batch
    scheduler.split_prefill_batch = None
    scheduler.ps = NS(tp_size=2, attn_dp_size=1)
    scheduler.model_config = NS(num_hidden_layers=3)
    scheduler.pdmux_config = NS(split_forward_token_budget=1000)
    scheduler.spec_algorithm = NS(is_none=lambda: False)
    scheduler.enable_overlap = overlap
    scheduler.stream_groups = [(stream, stream)]
    scheduler.sm_counts = [(8, 8)]
    scheduler.tp_cpu_group = group
    scheduler.HICACHE_PUMP_INTERVAL = 2
    scheduler._check_pdmux_dp_graph_capability = lambda: None
    scheduler.process_pending_chunked_abort = lambda: None
    scheduler.update_running_batch = lambda running: running
    scheduler.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=lambda batch: batch)
    scheduler.adjust_stream_groups = lambda **kw: (0, (stream, stream))
    scheduler.tick = 0

    def ingest():
        if len(merged) == 3:
            raise StopIteration
        scheduler.tick += 1
        assert scheduler.tick < 60, "chunked prefill never merged"

    def pump():
        # A real collective on the same CPU group catches an extra or missing
        # HiCache pump on either rank, including completion-poll iterations.
        flag = torch.ones(1, dtype=torch.int32)
        group.allreduce(flag, dist.ReduceOp.SUM).wait()
        assert flag.item() == 2
        trace.append(("pump", scheduler.tick))
        return False

    def admit(running):
        pump()
        return NS(
            running_batch=running,
            batch_to_run=NS(
                chunk=len(merged),
                forward_mode=NS(is_idle=lambda: False),
                extend_num_tokens=1000,
            ),
        )

    def run(current):
        if current is scheduler.split_prefill_batch:
            trace.append(("prefill", current.chunk, current.split_index))
        else:
            trace.append(("decode", scheduler.tick))
        return scheduler.tick

    def merge(*args):
        current = scheduler.split_prefill_batch
        assert current.split_index == 3
        merged.append(current.chunk)
        scheduler.split_prefill_batch = None
        trace.append(("merge", scheduler.tick))
        return batch

    scheduler.ingest_requests = ingest
    scheduler.check_hicache_events_if_enabled = pump
    scheduler.get_new_batch_prefill = admit
    scheduler.run_batch = run
    scheduler._merge_finished_prefill_batch = merge
    scheduler.process_batch_result = lambda batch, result: trace.append(
        ("commit", scheduler.tick, result)
    )
    try:
        scheduler.event_loop_pdmux_layer_split()
    except StopIteration:
        pass
    assert merged == [0, 1, 2]
    # The overlap path really submits N before committing N-1. The synchronous
    # path must retain the reverse order while exercising the same TP votes.
    second_decode = trace.index(("decode", 2))
    first_commit = next(
        i for i, row in enumerate(trace) if row[0] == "commit" and row[2] == 1
    )
    assert (second_decode < first_commit) == overlap
    traces = [None, None]
    dist.all_gather_object(traces, trace)
    assert traces[0] == traces[1], "TP ranks took different chunk/pump/merge paths"


def _gloo_lane_rank(rank, store_path):
    import torch
    import torch.distributed as dist

    class Group:
        def __init__(self, process_group):
            self.process_group = process_group
            self.reductions = 0

        def all_reduce(self, tensor):
            dist.all_reduce(tensor, group=self.process_group)
            self.reductions += 1
            return tensor

        def broadcast(self, tensor, src=0):
            self.broadcasts = getattr(self, "broadcasts", 0) + 1
            dist.broadcast(tensor, src=src, group=self.process_group)

    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        decode = Group(dist.group.WORLD)
        prefill = Group(dist.new_group([0, 1], backend="gloo"))
        lanes = lane_namespace(decode, prefill)
        # Unequal P/D tensor sizes expose a wrong-lane collective sequence.
        for chunk in range(32):
            for segment in range(3):
                lanes.set_pdmux_status(True)
                for get_group in (
                    lanes.get_attn_tp_group,
                    lanes.get_moe_tp_group,
                    lanes.get_attn_cp_group,
                    lanes.get_moe_dp_group,
                ):
                    p = torch.full((17,), rank + chunk + segment, dtype=torch.int64)
                    get_group().all_reduce(p)
                    torch.testing.assert_close(
                        p, torch.full_like(p, 1 + 2 * (chunk + segment))
                    )
                lanes.set_pdmux_status(False)
                d = torch.tensor([rank + segment], dtype=torch.int64)
                lanes.get_attn_tp_group().all_reduce(d)
                torch.testing.assert_close(d, torch.tensor([1 + 2 * segment]))
        assert prefill.reductions == 32 * 3 * 4
        assert decode.reductions == 32 * 3
        # Exercise the production prefill finalizer and SpecTpSync with two
        # real groups. A cached decode group must not receive DSPARK_TARGET.
        sync_method = load_methods(
            "speculative/spec_tp_sync.py", "SpecTpSync", ["sync"]
        )["sync"]
        sync = NS(_sites={"prefill", "plan"}, _tp_group=decode)
        sync.sync = lambda *args, **kwargs: sync_method(sync, *args, **kwargs)
        finalize = load_methods(
            "speculative/dspark_components/dspark_worker_v2.py",
            "DSparkWorkerV2",
            ["_finalize_prefill"],
            SpecTpSyncSite=NS(DSPARK_TARGET="prefill"),
            get_parallel=lambda: NS(enable_dp_attention=True, attn_cp_size=2),
            get_tp_group=lanes.get_tp_group,
        )["_finalize_prefill"]
        worker = NS(_target_hidden_projection_enabled=False, _tp_sync=sync)
        for chunk in range(32):
            lanes.set_pdmux_status(True)
            output = NS(
                logits_output=NS(hidden_states=None),
                next_token_ids=torch.tensor([chunk + rank]),
            )
            try:
                finalize(worker, NS(seq_lens=None), output, None)
            except RuntimeError as exc:
                assert "aux hidden capture" in str(exc)
            assert output.next_token_ids.item() == chunk
            lanes.set_pdmux_status(False)
            plan = torch.tensor([chunk + rank, chunk + rank])
            sync.sync("plan", plan)
            assert plan.tolist() == [chunk, chunk]
        assert prefill.broadcasts == decode.broadcasts == 32
        for overlap in (False, True):
            _exercise_chunked_scheduler(rank, decode.process_group, overlap=overlap)
    finally:
        dist.destroy_process_group()


class TestPdmuxTPGloo(unittest.TestCase):
    def test_two_ranks_keep_prefill_and_decode_collectives_separate(self):
        try:
            import torch.distributed as dist
        except ImportError:
            self.skipTest("PyTorch is required for the two-rank Gloo test")
        if not dist.is_gloo_available():
            self.skipTest("Gloo is unavailable")
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory(prefix="pdmux-tp-gloo-") as tmp:
            processes = [
                ctx.Process(
                    target=_gloo_lane_rank, args=(rank, str(Path(tmp) / "store"))
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            try:
                for process in processes:
                    process.join(timeout=30)
                    self.assertFalse(process.is_alive(), "TP lane collective hung")
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
