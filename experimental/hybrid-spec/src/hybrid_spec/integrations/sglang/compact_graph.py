"""Compact unmatched rows before replaying an EAGLE/MTP or standalone tail.

Each tail size owns its attention metadata, buffers and private graph pool.
The row mapping is dynamic; no model forward is replayed for selected SA rows.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
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
    forward: Callable | None = None


@dataclass
class CompactCapture:
    prefix: object
    output: object
    inputs: object
    override: object
    state: tuple
    tails: dict = field(default_factory=dict)
    prefix_width: int = 1
    rows: torch.Tensor | None = None
    num_neural: torch.Tensor | None = None
    no_skip: bool = False
    runtime_num_neural: int | None = None
    conditional: object | None = None


def compact_neural_rows(mask, rows, count, raw_bs, no_skip, *, live_bs=None):
    # Keep imports CPU-only until the CUDA runtime actually dispatches.
    from hybrid_spec.backends.gpu_sa.row_compaction import compact_neural_rows as run

    run(mask, rows, count, raw_bs, no_skip, live_bs=live_bs)


def capture_compact(runner, batch, forward_fn, stream, post_warmup_hook, mode="split"):
    from sglang.srt.layers.attention.triton_backend import TritonMultiStepDraftBackend

    dsv4 = _is_dsv4(runner)
    if not dsv4 and type(runner.draft_attn_backend) is not TritonMultiStepDraftBackend:
        raise ValueError(
            "compact split graphs require Triton or native dsv4 draft attention"
        )
    if runner.model_runner.kv_index_translator.is_translating:
        raise ValueError("compact tails do not support translated KV pools")
    if runner.model_runner.canary_manager is not None:
        raise ValueError("compact tails do not support KV canary instrumentation")
    view = batch.spec_info.draft_override
    state = None
    worker = runner.eagle_worker
    prefix_width = max(1, view.num_neural_gate_tokens)
    config = getattr(getattr(worker, "draft_extension", None), "config", None)
    neural_steps = (
        config.neural_drafts if config is not None else runner.speculative_num_steps
    )
    conditional = getattr(config, "conditional_draft_graph", False)

    def prefix_ready(fb, tokens, hidden, cache, drafts):
        nonlocal state
        state = (tokens, hidden, cache, drafts)
        # run_once restores K-1 positions. We executed prefix_width-1 inner
        # forwards, so compensate before its unconditional restoration.
        fb.positions.add_(worker.speculative_num_steps - prefix_width)
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
        prefix, output = _capture_graph(
            forward_fn,
            stream,
            keep_graph=getattr(config, "conditional_draft_graph_joined", False),
        )
    finally:
        del view.on_prefix_ready
    record = CompactCapture(
        prefix, output, batch, view, state, prefix_width=prefix_width
    )
    prepare_compaction(record, mode)
    # Capturing records kernels; it does not initialize newly allocated output
    # tensors. Tail warmup must consume a real prefix, never uninitialized IDs.
    prefix.replay()
    torch.cuda.synchronize()
    if prefix_width < neural_steps:
        for size in range(1, batch.batch_size + 1):
            record.tails[size] = _capture_tail(
                runner, record, size, stream, capture=not conditional
            )
        if conditional:
            from .conditional_graph import ConditionalTailGraph

            record.conditional = ConditionalTailGraph(
                record,
                stream,
                dispatch=getattr(config, "conditional_draft_graph_dispatch", "if"),
                joined=getattr(config, "conditional_draft_graph_joined", False),
            )
    return record


def prepare_compaction(record, mode):
    # All tail sizes alias one fixed-address row buffer. Runtime GPU scan
    # writes sorted original row ids; only its scalar count reaches the CPU.
    device = record.override.use_retrieval.device
    record.rows = torch.arange(record.inputs.batch_size, device=device)
    record.num_neural = torch.empty((), dtype=torch.int32, device=device)
    record.no_skip = mode == "split_no_skip"


def _capture_tail(runner, record, size, stream, *, capture=True):
    from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
    from sglang.srt.layers.attention.triton_backend import TritonMultiStepDraftBackend
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    original = record.inputs
    dsv4 = _is_dsv4(runner)
    tokens, hidden, cache, drafts = record.state
    steps = runner.speculative_num_steps
    config = getattr(
        getattr(runner.eagle_worker, "draft_extension", None), "config", None
    )
    neural_steps = config.neural_drafts if config is not None else steps
    start_step = record.prefix_width - 1
    rows = record.rows[:size]
    fb = copy.copy(original)
    fb.spec_info = copy.copy(original.spec_info)
    fb.spec_info.draft_override = None
    fb.batch_size = size
    fb.seq_lens_sum = None  # use the backend's static safe upper bound
    fb.seq_lens_cpu = None
    fb.positions = torch.empty_like(original.positions[:size])
    fb.mrope_positions = (
        None
        if original.mrope_positions is None
        else torch.empty_like(original.mrope_positions[:, :size])
    )
    fb.seq_lens = torch.empty_like(original.seq_lens[:size])
    fb.req_pool_indices = torch.empty_like(original.req_pool_indices[:size])
    fb.extend_seq_lens = torch.ones_like(original.extend_seq_lens[:size])
    fb.extend_seq_lens_cpu = [1] * size
    fb.input_ids = torch.empty(size, device=tokens.device, dtype=tokens.dtype)
    fb.spec_info.hidden_states = (
        torch.empty_like(hidden[:size]) if hidden is not None else None
    )
    compact_drafts = torch.empty_like(drafts[:size])
    # Native V4 metadata consumes a request-major, flattened K-step window.
    # Keep its per-step view as an alias, never a separately uploaded copy.
    cache_window = torch.empty((size, steps), dtype=cache.dtype, device=cache.device)
    compact_cache = cache_window.T
    flat_cache = cache_window.view(-1)
    fb.out_cache_loc = flat_cache
    mask = torch.empty(size, device=tokens.device, dtype=torch.bool)
    sa = torch.empty_like(drafts[:size])
    attention_type = (
        type(runner.draft_attn_backend) if dsv4 else TritonMultiStepDraftBackend
    )
    attention = attention_type(runner.model_runner, 1, steps)
    attention.init_cuda_graph_state(size, size)

    def stage():
        torch.index_select(original.positions, 0, rows, out=fb.positions)
        if fb.mrope_positions is not None:
            torch.index_select(
                original.mrope_positions, 1, rows, out=fb.mrope_positions
            )
        torch.index_select(original.seq_lens, 0, rows, out=fb.seq_lens)
        torch.index_select(original.req_pool_indices, 0, rows, out=fb.req_pool_indices)
        torch.index_select(cache, 1, rows, out=compact_cache)
        if hidden is not None:
            torch.index_select(hidden, 0, rows, out=fb.spec_info.hidden_states)
        torch.index_select(drafts, 0, rows, out=compact_drafts)
        torch.index_select(record.override.use_retrieval, 0, rows, out=mask)
        torch.index_select(record.override.tokens, 0, rows, out=sa)
        fb.input_ids.copy_(tokens.index_select(0, rows).flatten())

    # Metadata is planned with the original positions: its step i already
    # includes i draft KV slots. Resume at the prefix's final token position.
    stage()
    # A copied ForwardBatch must not retain the full batch's KV plan/tables.
    runner.model_runner.kv_index_translator.bind_runner_slots(fb)
    if dsv4:
        _plan_dsv4_metadata(
            attention,
            fb,
            compact_cache,
            flat_cache,
            start_step=start_step,
            stop_step=neural_steps - 1,
            in_capture=True,
        )
    else:
        attention.init_forward_metadata_out_graph(fb, in_capture=True)
    fb.mark_forward_metadata_ready()
    seed_hidden = fb.spec_info.hidden_states
    seed_tokens = fb.input_ids

    def run():
        fb.spec_info.hidden_states = seed_hidden
        fb.input_ids = seed_tokens
        stage()
        # The previous replay leaves a single-step write view on fb. Restore
        # the complete window before planning/capturing the next metadata set.
        fb.out_cache_loc = flat_cache
        if dsv4:
            _plan_dsv4_metadata(
                attention,
                fb,
                compact_cache,
                flat_cache,
                start_step=start_step,
                stop_step=neural_steps - 1,
                in_capture=False,
            )
        else:
            attention.init_forward_metadata_out_graph(fb, in_capture=False)
            attention.init_forward_metadata_in_graph(fb)
        fb.positions.add_(start_step)
        for i in range(start_step, neural_steps - 1):
            fb.out_cache_loc = compact_cache[i]
            if runner.model_runner.model_config.hf_config.architectures == [
                "Qwen3MoeForCausalLMMTP"
            ]:
                # Match native draft_forward's fused RoPE/KV-store contract.
                fb.out_cache_loc = fb.out_cache_loc.contiguous()
            with forward_context(
                ForwardContext(attn_backend=attention.attn_backends[i])
            ):
                logits = runner.model_runner.forward(fb).logits_output
            _, topk = draft_topk1_postprocess(
                logits.next_token_logits, fb.positions, compact_drafts, i + 1
            )
            if runner.eagle_worker.hot_token_id is not None:
                topk = runner.eagle_worker.hot_token_id[topk]
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
            if dsv4:
                # V4 upgrades raw -> full metadata during warmup; reset its raw
                # capture inputs so those GPU builders are recorded in graph.
                attention.on_after_cuda_graph_warmup()
    torch.cuda.synchronize()
    graph = _capture_graph(run, stream)[0] if capture else None
    return CompactTail(
        graph,
        rows,
        fb,
        attention,
        (seed_hidden, seed_tokens, compact_drafts, compact_cache, flat_cache, mask, sa),
        run if not capture else None,
    )


def _capture_graph(forward_fn, stream, *, keep_graph=False):
    graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
    with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
        output = forward_fn()
    if keep_graph:
        graph.instantiate()
    return graph, output


def _is_dsv4(runner):
    return (
        getattr(runner.draft_attn_backend, "decode_attention_backend_str", None)
        == "dsv4"
    )


def _plan_dsv4_metadata(
    attention, batch, cache, flat_cache, *, start_step, in_capture, stop_step=None
):
    # Keep native step ids: earlier steps ran in the prefix. Each child
    # owns its compact metadata and gets this request's actual write target.
    # Unlike native full-batch replay, do not copy child zero's raw metadata to
    # every step: raw_out_loc is also a per-step KV address.
    try:
        for step in range(
            start_step, cache.shape[0] - 1 if stop_step is None else stop_step
        ):
            child = attention.attn_backends[step]
            batch.out_cache_loc = cache[step]
            child.init_forward_metadata_out_graph(batch, in_capture=in_capture)
            if not in_capture:
                # Native in-graph SWA refresh indexes the full request-major
                # window by speculative_step_id, not a single-step vector.
                batch.out_cache_loc = flat_cache
                child.init_forward_metadata_in_graph(batch)
    finally:
        batch.out_cache_loc = flat_cache


def replay_compact(record, counters, raw_bs):
    if not 0 < raw_bs <= record.inputs.batch_size:
        raise ValueError("raw batch must fit the captured prefix")
    if record.conditional is not None and getattr(record.conditional, "joined", False):
        record.conditional.replay(raw_bs)
        record.runtime_num_neural = None
        counters["prefix_ct"] += 1
        counters["request_ct"] += raw_bs
        return record.output
    record.prefix.replay()
    if record.conditional is not None:
        if getattr(record.conditional, "dispatch", "if") == "switch":
            compact_neural_rows(
                record.override.use_retrieval,
                record.rows,
                record.num_neural,
                raw_bs,
                False,
                live_bs=record.conditional.live_bs,
            )
            record.conditional.replay(raw_bs, batch_prepared=True)
        else:
            compact_neural_rows(
                record.override.use_retrieval,
                record.rows,
                record.num_neural,
                raw_bs,
                False,
            )
            record.conditional.replay(raw_bs)
        # Verify may need its own host-visible token budget. Do not turn the
        # intermediate draft decision back into a blocking read for that reuse.
        record.runtime_num_neural = None
        counters["prefix_ct"] += 1
        counters["request_ct"] += raw_bs
        return record.output
    size = 0
    if record.tails:
        compact_neural_rows(
            record.override.use_retrieval,
            record.rows,
            record.num_neural,
            raw_bs,
            record.no_skip,
        )
        begin = time.perf_counter_ns()
        with torch.profiler.record_function("hybrid_prefix_decision"):
            size = int(record.num_neural.item())
        counters["decision_wait_ns"] += time.perf_counter_ns() - begin
    record.runtime_num_neural = size if record.tails else None
    counters["prefix_ct"] += 1
    counters["request_ct"] += raw_bs
    counters["skip_request_ct"] += raw_bs - size
    counters["tail_request_ct"] += size
    counters["mixed_ct"] += int(0 < size < raw_bs)
    histogram = counters["tail_batch_histogram"]
    histogram[str(size)] = histogram.get(str(size), 0) + 1
    if size:
        counters["tail_ct"] += 1
        record.tails[size].graph.replay()
    else:
        counters["skip_ct"] += 1
    return record.output
