"""Run PDMux's real host control flow without importing the GPU runtime.

CUDA streams/model execution are stand-ins; scheduling methods and forward
dispatch are loaded unchanged from their AST. These tests do not prove GPU
numerics or NCCL progress.
"""

from __future__ import annotations

import ast
import contextlib
import enum
import multiprocessing
import runpy
import sys
import tempfile
import unittest
from collections import namedtuple
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def methods(path, names, namespace):
    tree = ast.parse((SRT / path).read_text(encoding="utf-8"))
    nodes = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in nodes} == set(names)
    # inference_mode decorators do not affect the host scheduling decisions.
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class Mode(enum.Enum):
    DECODE = 1
    IDLE = 2
    SPLIT_PREFILL = 3

    def is_idle(self):
        return self is Mode.IDLE

    def is_decode(self):
        return self is Mode.DECODE

    def is_split_prefill(self):
        return self is Mode.SPLIT_PREFILL

    def is_cuda_graph(self):
        return self in (Mode.IDLE, Mode.DECODE)


class Batch:
    def __init__(self, rows=1, mode=Mode.DECODE):
        self.rows = rows
        self.forward_mode = mode
        self.input_ids = torch.arange(rows)
        self.replace_embeds = None
        self.encoder_lens = None
        self.global_num_tokens = None
        self.split_index = 0

    def batch_size(self):
        return self.rows

    def is_empty(self):
        return not self.rows


def scheduler_class(extra=None):
    namespace = dict(torch=torch, dist=dist, **(extra or {}))
    names = [
        "_select_pdmux_stream_idx",
        "_get_pdmux_dp_stream_idx",
        "_get_split_forward_count",
    ]
    return type(
        "Scheduler", (), methods("multiplex/multiplexing_mixin.py", names, namespace)
    )


def scheduler_config(scheduler):
    scheduler.ps = NS(attn_dp_size=8, tp_size=8)
    scheduler.real_sm_group_num = 3
    scheduler.pdmux_config = NS(
        manual_divisions=[[104, 0, 1]],
        decode_bs_divisor=36,
        split_forward_token_budget=65536,
    )
    scheduler.tp_cpu_group = Mock()
    scheduler.split_prefill_batch = NS(split_index=0)
    scheduler.running_batch = Batch()
    return scheduler


class TestStreamAgreement(unittest.TestCase):
    def test_uneven_ranks_select_one_group_without_an_extra_collective(self):
        for local_rows in (0, 1, 7):
            scheduler = scheduler_config(scheduler_class()())
            scheduler.running_batch = Batch(local_rows)
            decode = NS(global_num_tokens=[0, 1, 0, 7, 0, 0, 0, 0])
            self.assertEqual(scheduler._get_pdmux_dp_stream_idx(decode, 0), 1)
            scheduler.tp_cpu_group.allreduce.assert_not_called()

    def test_group_is_frozen_until_prefill_finishes(self):
        scheduler = scheduler_config(scheduler_class()())
        scheduler.split_prefill_batch.split_index = 4
        for decode in (None, NS(global_num_tokens=[99] * 8)):
            self.assertEqual(scheduler._get_pdmux_dp_stream_idx(decode, 1), 1)
        scheduler.split_prefill_batch = None
        self.assertEqual(scheduler._get_pdmux_dp_stream_idx(None, 1), 0)
        self.assertEqual(
            scheduler._get_pdmux_dp_stream_idx(NS(global_num_tokens=[1] * 8), 1), 2
        )
        scheduler.tp_cpu_group.allreduce.assert_not_called()

    def test_local_only_metadata_uses_global_max(self):
        scheduler = scheduler_config(scheduler_class()())
        scheduler.running_batch = Batch(0)

        def reduce(tensor, op):
            self.assertEqual(op, dist.ReduceOp.MAX)
            return NS(wait=lambda: tensor.fill_(7))

        scheduler.tp_cpu_group.allreduce.side_effect = reduce
        self.assertEqual(
            scheduler._get_pdmux_dp_stream_idx(NS(global_num_tokens=[0]), 0), 1
        )


def graph_runner():
    namespace = dict(torch=torch, get_current_stream_idx=lambda: 1)
    names = [
        "can_replay_batch_locally",
        "can_run_pdmux_decode_batch",
        "can_run_graph",
        "pdmux_graph_capability",
    ]
    cls = type(
        "Runner",
        (),
        methods("model_executor/runner/decode_cuda_graph_runner.py", names, namespace),
    )
    runner = cls()
    runner.model_runner = NS(
        spec_algorithm=NS(is_none=lambda: True, is_ngram=lambda: False),
        lora_manager=None,
    )
    runner.attention_graph_variants = None
    runner.is_encoder_decoder = False
    runner.enable_two_batch_overlap = False
    runner.captured_req_width = 1
    runner.ragged_verify_mode = False
    runner.require_mlp_tp_gather = True
    runner.require_mlp_sync = True
    runner.enable_pdmux = True
    runner.dp_size = 8
    runner.disable_padding = False
    runner.capture_bs = [1, 2, 4, 8]
    runner.max_bs = 8
    runner._max_dp_batch_size = lambda batch: max(batch.original_global_num_tokens_cpu)
    runner._make_graph_key = (
        lambda size, stream_idx=None, variant_label=None, attention_variant=None: (
            size,
            stream_idx,
        )
    )
    runner._resolve_lora_variant = lambda batch: None
    runner._resolve_attention_variant = lambda batch: None
    runner._pad_to_bucket = lambda size, buckets: next(b for b in buckets if b >= size)
    runner.backend = NS(
        has_captured_key=lambda key: key[0] in runner.capture_bs,
        can_run=lambda batch, key: key[0] in runner.capture_bs,
    )
    return runner


class TestGraphAgreement(unittest.TestCase):
    def test_embedding_override_veto_is_shared_with_actual_replay(self):
        runner = graph_runner()
        batch = Batch()
        self.assertTrue(runner.can_run_pdmux_decode_batch(batch))
        batch.replace_embeds = torch.ones(1)
        self.assertFalse(runner.can_run_pdmux_decode_batch(batch))
        self.assertFalse(runner.can_run_graph(batch))
        self.assertTrue(runner.can_run_pdmux_decode_batch(None))
        self.assertTrue(runner.can_run_pdmux_decode_batch(Batch(0, Mode.IDLE)))

    def test_global_veto_and_exact_bucket_are_honored(self):
        runner = graph_runner()
        batch = NS(
            replace_embeds=None,
            spec_info=None,
            batch_size=1,
            input_ids=torch.zeros(1),
            encoder_lens=None,
            can_run_tbo=False,
            can_run_decode_cuda_graph=True,
            original_global_num_tokens_cpu=[3, 0],
        )
        self.assertTrue(runner.can_run_graph(batch))
        runner.disable_padding = True
        self.assertFalse(runner.can_run_graph(batch))
        runner.disable_padding = False
        runner.backend.has_captured_key = lambda key: key != (4, 1)
        self.assertFalse(runner.can_run_graph(batch))
        runner.backend.has_captured_key = lambda key: True
        batch.can_run_decode_cuda_graph = False
        self.assertFalse(runner.can_run_graph(batch))

    def test_capability_includes_missing_keys_and_complex_variants_use_eager(self):
        runner = graph_runner()
        before = runner.pdmux_graph_capability(3)
        runner.backend.has_captured_key = lambda key: key != (4, 1)
        self.assertNotEqual(before, runner.pdmux_graph_capability(3))
        runner.model_runner.lora_manager = object()
        self.assertFalse(runner.can_run_pdmux_decode_batch(Batch()))
        self.assertFalse(runner.can_run_pdmux_decode_batch(None))


class TestSplitContinuation(unittest.TestCase):
    def test_padded_batch_survives_intermediate_segments_and_unpads_once(self):
        state = {}
        namespace = dict(
            contextlib=contextlib,
            torch=torch,
            has_forward_context=lambda: True,
            get_global_dwdp_manager=lambda: None,
            device_timer_ctx=lambda *a: contextlib.nullcontext(),
            ModelRunnerOutput=lambda **kw: NS(**kw),
            is_dsa_enable_prefill_cp=lambda: False,
            is_mla_cp_enabled=lambda: False,
        )
        cls = type(
            "Runner",
            (),
            methods(
                "model_executor/model_runner.py",
                [
                    "_forward_raw",
                    "_prepare_eager_forward_batch",
                    "forward_split_prefill",
                ],
                namespace,
            ),
        )
        runner = cls()
        runner.device = "cuda"
        runner.device_timer = None
        runner.model_config = NS(num_hidden_layers=3)
        runner.pp_group = NS(is_last_rank=True)
        runner.hisparse_coordinator = None
        runner.attn_backend = NS(init_forward_metadata=Mock())
        runner.decode_cuda_graph_runner = NS(can_run_graph=Mock(return_value=True))
        runner._maybe_execute_deferred_mamba_cow_and_clear = Mock()
        # IDLE is a graph-capable mode, but its split participant must never
        # enter the decode graph even when that runner would accept it.
        batch = NS(
            forward_mode=Mode.IDLE,
            split_index=0,
            global_num_tokens_cpu=[4, 0],
            global_num_token_non_padded=None,
            input_ids=torch.zeros(0),
            positions=torch.zeros(0),
            batch_size=0,
            is_extend_in_batch=True,
        )
        state_ns = dict(
            set_dp_buffer_len=lambda *a: state.update(dp=a),
            set_is_extend_in_batch=lambda v: state.update(extend=v),
        )
        republish = methods(
            "model_executor/forward_batch_info.py", ["republish_dp_state"], state_ns
        )["republish_dp_state"]
        batch.republish_dp_state = lambda: republish(batch)

        def prepare(_runner):
            batch.positions = torch.zeros(4)
            batch.batch_size = 4
            batch._original_batch_size = 0
            batch._pdmux_dp_state = (8, 4, True, [4, 4], None)
            batch.republish_dp_state()

        batch.prepare_mlp_sync_batch = Mock(side_effect=prepare)

        def unpad(_ret):
            batch.positions = batch.positions[:0]
            batch.batch_size = batch._original_batch_size

        batch.post_forward_mlp_sync_batch = Mock(side_effect=unpad)
        intervals = []

        def model_forward(ids, positions, fb, interval):
            self.assertEqual(positions.numel(), 4)
            self.assertEqual(fb.batch_size, 4)
            self.assertEqual(state["dp"][:3], (8, 4, True))
            self.assertTrue(state["extend"])
            intervals.append(interval)
            return None

        runner.model = NS(forward_split_prefill=model_forward)
        cp_utils = NS(is_cp_active=Mock(return_value=False), prepare_cp_forward=Mock())
        with patch.dict(sys.modules, {"sglang.srt.layers.cp.utils": cp_utils}):
            for i in range(3):
                state.update(dp=(999,), extend=False)  # an intervening decode
                runner._forward_raw(batch, None, split_forward_count=1)
                self.assertEqual(
                    batch.post_forward_mlp_sync_batch.call_count, int(i == 2)
                )
        self.assertEqual(intervals, [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(batch.positions.numel(), 0)
        batch.prepare_mlp_sync_batch.assert_called_once()
        runner.decode_cuda_graph_runner.can_run_graph.assert_not_called()


class TestCarriedTensorLifetime(unittest.TestCase):
    def test_masks_penalizers_and_batch_tensors_follow_the_consumer_stream(self):
        module = runpy.run_path(str(SRT / "multiplex/pdmux_tensor_lifetime.py"))

        class Tensor:
            def __init__(self):
                self.record_stream = Mock()

        publish = module["publish_carried_tensors"]
        publish.__globals__["_is_carried_device_tensor"] = lambda x: isinstance(
            x, Tensor
        )
        direct, grammar, processor, penalty, spec, excluded = [
            Tensor() for _ in range(6)
        ]
        batch = NS(
            seq_lens=direct,
            sampling_info=NS(
                grammar_mask=namedtuple("GrammarMask", "vocab_mask grammar")(
                    grammar, object()
                ),
                custom_logit_processor={"p": (object(), processor)},
                penalizer_orchestrator=NS(penalizers={"p": NS(counts=penalty)}),
            ),
            spec_info=NS(hidden_states=spec),
            tree_cache=NS(storage=excluded),
            reqs=[NS(tensor=excluded)],
        )
        first, switched = object(), object()
        publish(batch, first)
        publish(batch, switched)
        for value in (direct, grammar, processor, penalty, spec):
            self.assertEqual(value.record_stream.call_count, 2)
            value.record_stream.assert_called_with(switched)
        excluded.record_stream.assert_not_called()


def _gloo_agreement_rank(rank, path, queue):
    try:
        store = dist.FileStore(path, 2)
        dist.init_process_group(
            "gloo", store=store, rank=rank, world_size=2, timeout=timedelta(seconds=20)
        )
        runner = graph_runner()
        scheduler = scheduler_config(scheduler_class()())
        scheduler.ps.attn_dp_size = 2
        scheduler.tp_cpu_group = dist.group.WORLD
        scheduler.tp_worker = NS(model_runner=NS(decode_cuda_graph_runner=runner))
        check = methods(
            "multiplex/multiplexing_mixin.py",
            ["_check_pdmux_dp_graph_capability"],
            dict(dist=dist),
        )["_check_pdmux_dp_graph_capability"]
        check(scheduler)
        batch = Batch(3 if rank == 0 else 0)
        batch.forward_mode = Mode.DECODE if rank == 0 else Mode.IDLE
        if rank == 0:
            batch.replace_embeds = torch.ones(1)
        # Same MIN vote and gathered counts used by MLPSyncBatchInfo.
        vote = torch.tensor([int(runner.can_run_pdmux_decode_batch(batch))])
        dist.all_reduce(vote, op=dist.ReduceOp.MIN)
        rows = [torch.zeros(1, dtype=torch.int64) for _ in range(2)]
        dist.all_gather(rows, torch.tensor([batch.batch_size()]))
        group = scheduler._get_pdmux_dp_stream_idx(
            NS(global_num_tokens=[int(v.item()) for v in rows]), 0
        )
        if rank == 1:
            runner.backend.has_captured_key = lambda key: key != (4, 1)
        mismatch = False
        try:
            check(scheduler)
        except RuntimeError as exc:
            mismatch = "capabilities" in str(exc)
        queue.put((rank, int(vote.item()), group, mismatch))
    except Exception as exc:
        queue.put((rank, repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestGlooAgreement(unittest.TestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "Gloo unavailable")
    def test_real_two_rank_veto_selection_and_capability_check(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            queue = context.Queue()
            processes = [
                context.Process(
                    target=_gloo_agreement_rank,
                    args=(rank, str(Path(directory) / "store"), queue),
                )
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                results = [queue.get(timeout=30) for _ in processes]
                for process in processes:
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual(sorted(results), [(0, 0, 1, True), (1, 0, 1, True)])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)
                queue.close()


class StopLoop(Exception):
    pass


class TestActualLoop(unittest.TestCase):
    def run_rank(self, rank):
        trace, current = [], {"idx": 0, "lane": None}

        class Stream:
            def __init__(self, lane):
                self.lane = lane

            def record_event(self):
                event = NS(query=lambda: True, name=(self.lane, len(trace)))
                trace.append(("record", event.name))
                return event

            def wait_event(self, event):
                trace.append(("wait", self.lane, event.name))

            def synchronize(self):
                trace.append(("sync", self.lane))

        @contextlib.contextmanager
        def stream_context(stream):
            old = current["lane"]
            current["lane"] = stream.lane
            try:
                yield
            finally:
                current["lane"] = old

        namespace = dict(
            torch=NS(
                cuda=NS(stream=stream_context, empty_cache=lambda: None),
                ones=torch.ones,
                zeros=torch.zeros,
                int32=torch.int32,
            ),
            dist=dist,
            get_current_stream_idx=lambda: current["idx"],
            set_current_stream_idx=lambda i: current.update(idx=i),
            set_pdmux_status=lambda v: None,
            logger=Mock(),
            publish_carried_tensors=lambda b, s: trace.append(("publish", s.lane)),
            ForwardMode=Mode,
        )
        names = [
            "event_loop_pdmux_layer_split",
            "_select_pdmux_stream_idx",
            "_get_pdmux_dp_stream_idx",
            "_get_split_forward_count",
            "update_split_prefill_batch",
            "_merge_finished_prefill_batch",
        ]
        cls = type(
            "Scheduler",
            (),
            methods("multiplex/multiplexing_mixin.py", names, namespace),
        )
        s = scheduler_config(cls())
        s.ps = NS(attn_dp_size=2, tp_size=2)
        s.model_config = NS(num_hidden_layers=3)
        s.pdmux_config.split_forward_token_budget = 2048
        s.split_prefill_batch = None
        s.HICACHE_PUMP_INTERVAL = 16
        s.running_batch = Batch(1 if rank == 1 else 0)
        s.stream_groups = [
            (Stream("p" + str(i)), Stream("d" + str(i))) for i in range(3)
        ]
        s.sm_counts = [(132, 0), (104, 132), (0, 132)]
        s._check_pdmux_dp_graph_capability = Mock()
        s._update_decode_attn_backends = lambda i: trace.append(("group", i))
        s.process_pending_chunked_abort = lambda: None
        s.check_hicache_events_if_enabled = lambda: False
        s.on_idle = Mock()
        tick = {"n": -1}

        def ingest():
            tick["n"] += 1
            if tick["n"] == 2:
                raise StopLoop
            trace.append(("ingest", tick["n"]))

        s.ingest_requests = ingest
        prefill = Batch(1, Mode.SPLIT_PREFILL) if rank == 0 else None
        s.get_new_batch_prefill = lambda b: NS(batch_to_run=prefill, running_batch=b)

        def sync_batch(batch):
            if current["lane"].startswith("p"):
                batch = batch or Batch(0, Mode.IDLE)
                batch.global_num_tokens = [2048, 0]
                batch.extend_num_tokens = 2048 if rank == 0 else 0
                return batch
            if tick["n"] > 0:
                return None
            batch = batch or Batch(0, Mode.IDLE)
            batch.global_num_tokens = [0, 1]
            return batch

        s.dp_attn_adapter = NS(maybe_prepare_mlp_sync_batch=sync_batch)
        s.update_running_batch = lambda b: b

        def run(batch):
            trace.append(
                (
                    "prefill" if batch is s.split_prefill_batch else "decode",
                    current["idx"],
                    batch.split_index,
                    getattr(batch, "split_forward_count", None),
                )
            )
            return object()

        s.run_batch = run

        def process(batch, result):
            trace.append(("process_decode",))
            s.running_batch.rows = 0

        s.process_batch_result = process

        def merge(**kw):
            trace.append(("merge",))
            return kw["running_batch"]

        s._merge_completed_prefill_batch = merge

        def reduce(flags, op):
            self.assertEqual(op, dist.ReduceOp.SUM)
            trace.append(("completion_vote",))
            return NS(wait=lambda: flags.fill_(2))

        s.tp_cpu_group = NS(allreduce=reduce)
        with self.assertRaises(StopLoop):
            s.event_loop_pdmux_layer_split()
        return trace

    def test_uneven_ranks_freeze_group_and_finalize_with_ordered_events(self):
        traces = [self.run_rank(rank) for rank in range(2)]
        for trace in traces:
            self.assertEqual([x for x in trace if x[0] == "group"], [("group", 1)])
            self.assertEqual(
                [x for x in trace if x[0] == "prefill"],
                [("prefill", 1, 0, 1), ("prefill", 1, 1, 2)],
            )
            merge = trace.index(("merge",))
            # Finalize waits for the event recorded after decode processing.
            wait = next(i for i in range(merge - 1, -1, -1) if trace[i][0] == "wait")
            self.assertEqual(trace[wait][1], "p1")
            event_name = trace[wait][2]
            record = trace.index(("record", event_name))
            self.assertLess(trace.index(("process_decode",)), record)
            self.assertLess(record, wait)
            self.assertEqual(sum(x[0] == "completion_vote" for x in trace), 1)
            # Admission consumes the input event, before the first model work.
            first_wait = next(x for x in trace if x[0] == "wait")
            self.assertEqual(first_wait[1], "p0")
            self.assertEqual(first_wait[2][0], "d0")


if __name__ == "__main__":
    unittest.main()
