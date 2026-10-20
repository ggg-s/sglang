"""Split draft graphs with GPU row compaction; the CPU reads one count.

Prefix and tail use separate private pools: skipping a tail must not let a
subsequent graph overwrite the prefix's hidden state or its complete SA chain.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext


class SplitDraftGraphBackend:
    def __init__(self, delegate, runner, mode):
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        if type(delegate) is not FullCudaGraphBackend:
            raise ValueError("split draft graphs require the full CUDA graph backend")
        architecture = runner.model_runner.model_config.hf_config.architectures
        if (
            runner.topk != 1
            or runner.captured_req_width != 1
            or runner.eagle_worker.seed_dsa_topk_from_draft_extend
        ):
            raise ValueError(
                "split draft graphs require topk=1, one token per request "
                "and no draft-extend DSA seed"
            )
        if (
            architecture == ["DeepseekV4ForCausalLMNextN"]
            and getattr(runner.draft_attn_backend, "decode_attention_backend_str", None)
            != "dsv4"
        ):
            raise ValueError(
                "DeepSeek V4 split requires native dsv4 draft attention; "
                "use --speculative-draft-attention-backend dsv4"
            )
        self.delegate = delegate
        self.runner = runner
        self.mode = mode
        config = getattr(runner.eagle_worker.draft_extension, "config", None)
        self.conditional = getattr(config, "conditional_draft_graph", False)
        self.conditional_dispatch = getattr(
            config, "conditional_draft_graph_dispatch", "if"
        )
        self.conditional_joined = getattr(
            config, "conditional_draft_graph_joined", False
        )
        if self.conditional:
            from .conditional_graph import require_conditional_graph

            require_conditional_graph(self.conditional_dispatch)
        self.records = {}
        self._conditional_groups = ()
        self.stream = None
        self.counters = {
            "prefix_ct": 0,
            "skip_ct": 0,
            "tail_ct": 0,
            "decision_wait_ns": 0,
            "request_ct": 0,
            "skip_request_ct": 0,
            "tail_request_ct": 0,
            "mixed_ct": 0,
            "tail_batch_histogram": {str(n): 0 for n in range(runner.max_bs + 1)},
        }
        # Expose actual branch counts without reading GPU counters in metrics.
        runner.eagle_worker.draft_extension.split_graph_counters = self.counters
        if self.conditional:
            runner.eagle_worker.draft_extension.split_graph_snapshot = self.snapshot

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @contextmanager
    def capture_session(self, stream):
        self.stream = stream
        try:
            scope = nullcontext(())
            if self.conditional:
                from sglang.srt.distributed.parallel_state import (
                    conditional_graph_capture_scope,
                )

                scope = conditional_graph_capture_scope()
            with scope as groups, self.delegate.capture_session(stream):
                self._conditional_groups = groups
                yield
        finally:
            self.stream = None

    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        from .compact_graph import capture_compact

        self.records[shape_key] = capture_compact(
            self.runner,
            capture_inputs,
            forward_fn,
            self.stream,
            post_warmup_hook,
            self.mode,
        )

    def can_run(self, forward_batch, shape_key):
        return (
            1 <= forward_batch.batch_size <= self.runner.max_bs
            and shape_key in self.records
        )

    def replay(self, shape_key, static_forward_batch, **kwargs):
        from .compact_graph import replay_compact

        record = self.records[shape_key]
        if self.conditional and self._conditional_groups:
            import torch

            stream = torch.cuda.current_stream()
            for group in self._conditional_groups:
                group.check_conditional_graph_replay_stream(stream)
        output = replay_compact(record, self.counters, self.runner.raw_bs)
        self.runner.eagle_worker.draft_extension.compact_num_neural = (
            record.runtime_num_neural
        )
        return output

    def cleanup(self):
        for record in self.records.values():
            if record.conditional is not None:
                record.conditional.reset()
            record.prefix.reset()
            for tail in record.tails.values():
                if tail.graph is not None:
                    tail.graph.reset()
                tail.forward = None
        self.records.clear()
        self._conditional_groups = ()
        self.delegate.cleanup()

    def snapshot(self):
        """Merge asynchronous GPU branch counters only when requested."""
        result = dict(self.counters)
        histogram = result["tail_batch_histogram"] = dict(
            self.counters["tail_batch_histogram"]
        )
        result["conditional_draft_graph"] = self.conditional
        if self.conditional:
            result["conditional_draft_graph_dispatch"] = self.conditional_dispatch
            result["conditional_draft_graph_joined"] = self.conditional_joined
        for record in self.records.values():
            if record.conditional is None:
                continue
            counts, mixed = record.conditional.snapshot()
            result["skip_ct"] += counts[0]
            result["tail_ct"] += sum(counts[1:])
            result["mixed_ct"] += mixed
            tail_requests = sum(size * count for size, count in enumerate(counts))
            result["tail_request_ct"] += tail_requests
            for size, count in enumerate(counts):
                histogram[str(size)] += count
        result["skip_request_ct"] = result["request_ct"] - result["tail_request_ct"]
        return result
