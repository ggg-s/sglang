"""Experimental BS=1 split draft graph; the CPU decision intentionally syncs.

Prefix and tail use separate private pools: skipping a tail must not let a
subsequent graph overwrite the prefix's hidden state or its complete SA chain.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class SplitCapture:
    prefix: object
    tail: object
    output: object
    inputs: object
    override: object


class SplitDraftGraphBackend:
    def __init__(self, delegate, runner, mode):
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        if type(delegate) is not FullCudaGraphBackend:
            raise ValueError("split draft graphs require the full CUDA graph backend")
        if (
            runner.topk != 1
            or runner.max_bs != 1
            or runner.captured_req_width != 1
            or runner.eagle_worker.hot_token_id is not None
            or runner.eagle_worker.seed_dsa_topk_from_draft_extend
            or runner.model_runner.model_config.hf_config.architectures
            != ["Qwen3_5ForCausalLMMTP"]
        ):
            raise ValueError("split draft graphs support only BS=1 Qwen3.5 MTP")
        self.delegate = delegate
        self.runner = runner
        self.mode = mode
        self.records = {}
        self.stream = None
        self.counters = {
            "prefix_ct": 0,
            "skip_ct": 0,
            "tail_ct": 0,
            "decision_wait_ns": 0,
        }
        # Expose actual branch counts without reading GPU counters in metrics.
        runner.eagle_worker.draft_extension.split_graph_counters = self.counters

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @contextmanager
    def capture_session(self, stream):
        self.stream = stream
        try:
            with self.delegate.capture_session(stream):
                yield
        finally:
            self.stream = None

    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        batch = capture_inputs
        view = batch.spec_info.draft_override
        for _ in range(2):
            torch.cuda.synchronize()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()
        torch.cuda.synchronize()

        prefix, tail = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        # Keep both pools alive through the graph objects. They are deliberately
        # distinct from each other and from target/extend's shared pool.
        prefix_ctx = torch.cuda.graph(
            prefix, pool=torch.cuda.graph_pool_handle(), stream=self.stream
        )
        tail_ctx = torch.cuda.graph(
            tail, pool=torch.cuda.graph_pool_handle(), stream=self.stream
        )
        active = prefix_ctx
        switched = False

        def prefix_ready():
            nonlocal active, switched
            if switched:
                raise RuntimeError("split draft prefix boundary reached twice")
            active.__exit__(None, None, None)
            active = None
            tail_ctx.__enter__()
            active = tail_ctx
            switched = True

        view.on_prefix_ready = prefix_ready
        try:
            prefix_ctx.__enter__()
            output = forward_fn()
        except BaseException:
            if active is not None:
                active.__exit__(*sys.exc_info())
                active = None
            raise
        finally:
            del view.on_prefix_ready
        if active is not None:
            active.__exit__(None, None, None)
        if not switched:
            raise RuntimeError("draft worker did not reach the split prefix boundary")
        self.records[shape_key] = SplitCapture(prefix, tail, output, batch, view)

    def can_run(self, forward_batch, shape_key):
        return forward_batch.batch_size == 1 and shape_key in self.records

    def replay(self, shape_key, static_forward_batch, **kwargs):
        record = self.records[shape_key]
        record.prefix.replay()
        begin = time.perf_counter_ns()
        # This includes unfinished prefix GPU work, not just the incremental
        # cost of synchronizing. Compare split_no_skip against full for cost.
        with torch.profiler.record_function("hybrid_prefix_decision"):
            use_sa = bool(record.override.use_retrieval.item())
        self.counters["decision_wait_ns"] += time.perf_counter_ns() - begin
        self.counters["prefix_ct"] += 1
        if use_sa and self.mode == "split":
            self.counters["skip_ct"] += 1
            # run_once normally restores positions after all seven forwards;
            # the prefix ran only one. All other input buffers are restaged.
            record.inputs.positions.sub_(1)
        else:
            self.counters["tail_ct"] += 1
            record.tail.replay()
        return record.output

    def cleanup(self):
        for record in self.records.values():
            record.prefix.reset()
            record.tail.reset()
        self.records.clear()
        self.delegate.cleanup()
