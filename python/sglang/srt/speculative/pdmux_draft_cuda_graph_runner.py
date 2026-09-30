"""Stream-bound speculative graphs for the PDMux decode lane."""

import logging

import torch

from sglang.srt.distributed.parallel_state import (
    is_pdmux_prefill_enabled,
    set_pdmux_status,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_stream_groups,
    set_current_stream_idx,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory

logger = logging.getLogger(__name__)


class PDMuxDraftCudaGraphRunner:
    """Dispatch complete graph runners, including metadata, by stream group.

    A graph retains the CUDA resource context of its capture stream. Backends
    also retain graph buffers indexed by batch size, so sharing one backend
    between groups would silently replace an earlier group's replay metadata.
    """

    def __init__(self, worker, runner_cls, *, draft_extend=False):
        self.runners = []
        self.phase = "draft_extend" if draft_extend else "draft_decode"
        self._reported_replay = set()
        self._reported_fallback = set()
        groups = get_stream_groups()
        if not groups:
            raise RuntimeError("PDMux draft graph capture requires stream groups")
        factory = DraftBackendFactory(
            worker.server_args,
            worker.draft_runner,
            worker.topk,
            worker.speculative_num_steps,
            seed_dsa_topk_from_draft_extend=worker.seed_dsa_topk_from_draft_extend,
        )
        # Decode phases run sequentially. Share their graph pool, but never the
        # pool used by the concurrently executing prefill lane.
        if not hasattr(worker, "_pdmux_draft_graph_memory_pool"):
            worker._pdmux_draft_graph_memory_pool = torch.cuda.graph_pool_handle()
        previous_pool = get_global_graph_memory_pool()
        previous_idx = get_current_stream_idx()
        previous_prefill = is_pdmux_prefill_enabled()
        previous_backend = worker.draft_attn_backend
        try:
            set_global_graph_memory_pool(worker._pdmux_draft_graph_memory_pool)
            set_pdmux_status(False)
            for idx, (_, decode_stream) in enumerate(groups):
                set_current_stream_idx(idx)
                if draft_extend:
                    backend = factory.create_draft_extend_backend()
                    kwargs = {"draft_extend_attn_backend": backend}
                else:
                    backend = factory.create_decode_backend()
                    # draft_forward publishes the worker's per-step backends.
                    worker.draft_attn_backend = backend
                    kwargs = {"draft_attn_backend": backend}
                runner = runner_cls(worker, capture_stream=decode_stream, **kwargs)
                self.runners.append(runner)
                logger.info(
                    "PDMux %s CUDA graphs captured: stream_group=%d, bs=%s",
                    self.phase,
                    idx,
                    runner.capture_bs,
                )
        finally:
            worker.draft_attn_backend = previous_backend
            set_current_stream_idx(previous_idx)
            set_pdmux_status(previous_prefill)
            set_global_graph_memory_pool(previous_pool)

    def __getattr__(self, name):
        # Includes buffers (DSA seed output), width and batch-size capabilities.
        return getattr(self.runners[get_current_stream_idx()], name)

    def can_run_graph(self, forward_batch):
        idx = get_current_stream_idx()
        runner = self.runners[idx]
        eligible = runner.can_run_graph(forward_batch)
        if not eligible and idx not in self._reported_fallback:
            self._reported_fallback.add(idx)
            logger.info(
                "PDMux %s CUDA graph fallback: stream_group=%d, bs=%s, "
                "max_bs=%s, dp_graph=%s",
                self.phase,
                idx,
                forward_batch.batch_size,
                runner.max_bs,
                forward_batch.can_run_dp_cuda_graph,
            )
        return eligible

    def execute(self, forward_batch, *args, **kwargs):
        idx = get_current_stream_idx()
        result = self.runners[idx].execute(forward_batch, *args, **kwargs)
        if idx not in self._reported_replay:
            self._reported_replay.add(idx)
            logger.info(
                "PDMux %s CUDA graph replay active: stream_group=%d",
                self.phase,
                idx,
            )
        return result
