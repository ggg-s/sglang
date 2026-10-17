"""Compact the unmatched rows before replaying the Qwen3.5 MTP tail.

Each tail size owns its attention metadata, buffers and private graph pool.
The row mapping is dynamic; no model forward is replayed for selected SA rows.
"""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import torch


@dataclass
class CompactTail:
    graph: object
    rows: torch.Tensor
    batch: object
    attention: object
    # Graphs retain addresses, not ownership of tensors allocated outside
    # capture. Keep every staging buffer alive after the run closure is gone.
    staging: tuple


@dataclass
class CompactCapture:
    prefix: object
    output: object
    inputs: object
    override: object
    state: tuple
    tails: dict = field(default_factory=dict)
    dispatch: dict = field(default_factory=dict)


def capture_compact(runner, batch, forward_fn, stream, post_warmup_hook, mode="split"):
    from sglang.srt.layers.attention.triton_backend import TritonMultiStepDraftBackend

    if type(runner.draft_attn_backend) is not TritonMultiStepDraftBackend:
        raise ValueError("multi-request split graphs require Triton draft attention")
    if runner.model_runner.kv_index_translator.is_translating:
        raise ValueError("compact tails do not support translated KV pools")
    if runner.model_runner.canary_manager is not None:
        raise ValueError("compact tails do not support KV canary instrumentation")
    view = batch.spec_info.draft_override
    state = None
    worker = runner.eagle_worker

    def prefix_ready(fb, tokens, hidden, cache, drafts):
        nonlocal state
        state = (tokens, hidden, cache, drafts)
        # run_once restores K-1 positions even though we return after one
        # inner forward. Leave the prefix inputs at their original positions.
        fb.positions.add_(worker.speculative_num_steps - 2)
        bs = drafts.shape[0]
        return (
            worker._topk1_parents_prealloc[:bs],
            worker._topk1_score_indices_prealloc[:bs],
            drafts,
            None,
        )

    view.on_prefix_ready = prefix_ready
    try:
        for _ in range(2):
            torch.cuda.synchronize()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()
        torch.cuda.synchronize()
        prefix = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            prefix, pool=torch.cuda.graph_pool_handle(), stream=stream
        ):
            output = forward_fn()
    finally:
        del view.on_prefix_ready
    record = CompactCapture(prefix, output, batch, view, state)
    # Capturing records kernels; it does not initialize newly allocated output
    # tensors. Tail warmup must consume a real prefix, never uninitialized IDs.
    prefix.replay()
    torch.cuda.synchronize()
    for size in range(1, batch.batch_size + 1):
        record.tails[size] = _capture_tail(runner, record, size, stream)
    prepare_dispatch(record, mode)
    return record


def prepare_dispatch(record, mode):
    # At most four rows: precompute every raw-batch mask and retain its device
    # row vector. Replay performs no tensor allocation or row-index H2D upload.
    # Keys include raw size through tuple length, excluding graph padding.
    device_rows = {}
    for raw_bs in range(1, record.inputs.batch_size + 1):
        for mask in itertools.product((False, True), repeat=raw_bs):
            rows = tuple(
                i for i, hit in enumerate(mask) if not hit or mode == "split_no_skip"
            )
            size = len(rows)
            if size and rows not in device_rows:
                device_rows[rows] = torch.tensor(
                    rows, dtype=torch.int64, device=record.tails[size].rows.device
                )
            record.dispatch[mask] = (
                size,
                record.tails[size] if size else None,
                device_rows.get(rows),
            )


def _capture_tail(runner, record, size, stream):
    from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
    from sglang.srt.layers.attention.triton_backend import TritonMultiStepDraftBackend
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    original = record.inputs
    tokens, hidden, cache, drafts = record.state
    steps = runner.speculative_num_steps
    rows = torch.arange(size, device=tokens.device)
    fb = copy.copy(original)
    fb.spec_info = copy.copy(original.spec_info)
    fb.spec_info.draft_override = None
    fb.batch_size = size
    fb.seq_lens_sum = None  # use the backend's static safe upper bound
    fb.seq_lens_cpu = None
    fb.positions = torch.empty_like(original.positions[:size])
    fb.mrope_positions = torch.empty_like(original.mrope_positions[:, :size])
    fb.seq_lens = torch.empty_like(original.seq_lens[:size])
    fb.req_pool_indices = torch.empty_like(original.req_pool_indices[:size])
    fb.extend_seq_lens = torch.ones_like(original.extend_seq_lens[:size])
    fb.extend_seq_lens_cpu = [1] * size
    fb.input_ids = torch.empty(size, device=tokens.device, dtype=tokens.dtype)
    fb.spec_info.hidden_states = torch.empty_like(hidden[:size])
    compact_drafts = torch.empty_like(drafts[:size])
    compact_cache = torch.empty_like(cache[:, :size])
    mask = torch.empty(size, device=tokens.device, dtype=torch.bool)
    sa = torch.empty_like(drafts[:size])
    attention = TritonMultiStepDraftBackend(runner.model_runner, 1, steps)
    attention.init_cuda_graph_state(size, size)

    def stage():
        torch.index_select(original.positions, 0, rows, out=fb.positions)
        torch.index_select(original.mrope_positions, 1, rows, out=fb.mrope_positions)
        torch.index_select(original.seq_lens, 0, rows, out=fb.seq_lens)
        torch.index_select(original.req_pool_indices, 0, rows, out=fb.req_pool_indices)
        torch.index_select(cache, 1, rows, out=compact_cache)
        torch.index_select(hidden, 0, rows, out=fb.spec_info.hidden_states)
        torch.index_select(drafts, 0, rows, out=compact_drafts)
        torch.index_select(record.override.use_retrieval, 0, rows, out=mask)
        torch.index_select(record.override.tokens, 0, rows, out=sa)
        fb.input_ids.copy_(tokens.index_select(0, rows).flatten())

    # Metadata is planned with the original positions: its step i already
    # includes i draft KV slots. The model then resumes at position +1.
    stage()
    attention.init_forward_metadata_out_graph(fb, in_capture=True)
    fb.mark_forward_metadata_ready()
    seed_hidden = fb.spec_info.hidden_states
    seed_tokens = fb.input_ids

    def run():
        fb.spec_info.hidden_states = seed_hidden
        fb.input_ids = seed_tokens
        stage()
        attention.init_forward_metadata_out_graph(fb, in_capture=False)
        attention.init_forward_metadata_in_graph(fb)
        fb.positions.add_(1)
        for i in range(1, steps - 1):
            fb.out_cache_loc = compact_cache[i]
            with forward_context(
                ForwardContext(attn_backend=attention.attn_backends[i])
            ):
                logits = runner.model_runner.forward(fb).logits_output
            _, topk = draft_topk1_postprocess(
                logits.next_token_logits, fb.positions, compact_drafts, i + 1
            )
            topk = torch.where(mask[:, None], sa[:, i + 1 : i + 2], topk)
            compact_drafts[:, i + 1 : i + 2].copy_(topk)
            fb.input_ids = topk.flatten()
            fb.spec_info.hidden_states = logits.hidden_states
            if runner.model_runner.model_config.model_is_mrope:
                fb.mrope_positions.add_(1)
        drafts.index_copy_(0, rows, compact_drafts)

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
        run()
    return CompactTail(
        graph,
        rows,
        fb,
        attention,
        (seed_hidden, seed_tokens, compact_drafts, compact_cache, mask, sa),
    )


def replay_compact(record, counters, raw_bs):
    record.prefix.replay()
    begin = time.perf_counter_ns()
    with torch.profiler.record_function("hybrid_prefix_decision"):
        mask = record.override.use_retrieval[:raw_bs].tolist()
    counters["decision_wait_ns"] += time.perf_counter_ns() - begin
    size, tail, rows = record.dispatch[tuple(mask)]
    counters["prefix_ct"] += 1
    counters["request_ct"] += raw_bs
    counters["skip_request_ct"] += raw_bs - size
    counters["tail_request_ct"] += size
    counters["mixed_ct"] += int(0 < size < raw_bs)
    counters["tail_batch_histogram"][str(size)] += 1
    if size:
        counters["tail_ct"] += 1
        tail.rows.copy_(rows)
        tail.graph.replay()
    else:
        counters["skip_ct"] += 1
    return record.output
