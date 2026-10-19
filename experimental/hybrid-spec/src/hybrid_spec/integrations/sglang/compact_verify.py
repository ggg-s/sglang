"""Source-dependent fixed draft budgets and target token graph tiers."""

from __future__ import annotations

import torch


def token_buckets(config, base, max_bs, alignment=1):
    """Exact small-batch mixes and sparse dynamic tiers at geometric anchors."""
    width = config.num_drafts + 1
    neural_width = config.neural_drafts + 1
    maximum = max_bs * width
    full_tiers = {bs * width for bs in range(1, min(max_bs, config.max_requests) + 1)}
    if config.verify_token_buckets:
        buckets = set(config.verify_token_buckets)
        if max(buckets) > maximum:
            raise ValueError("verify_token_buckets exceeds captured token capacity")
        # Full-width tiers bound rounding by each live batch's planned KV window.
        buckets.update(full_tiers)
        buckets.add(maximum)
    else:
        buckets = set(base) | full_tiers
        # Capture every possible mix up to eight rows. This includes the c=1
        # neural tier even when max_bs itself is a large graph padding bucket.
        for bs in range(1, min(max_bs, config.max_requests, 8) + 1):
            for num_sa in range(bs + 1):
                buckets.add(bs * neural_width + num_sa * (width - neural_width))
        live_max = min(max_bs, config.max_requests)
        if config.dynamic_verify:
            # Add dynamic capacities around geometric batch anchors, rather
            # than 25/50/75% for every possible live batch. Keep the full-width
            # safety tiers above: rounding must fit each batch's KV window.
            batch_anchors = {live_max}
            bs = 1
            while bs <= live_max:
                batch_anchors.add(bs)
                bs *= 2
        else:
            batch_anchors = range(1, live_max + 1)
        for bs in batch_anchors:
            buckets.add(bs * neural_width)
            if config.dynamic_verify:
                # A bounded capacity grid, independent of the number of possible
                # per-row length combinations. Native alignment is applied below.
                buckets.add(bs * (config.dynamic_verify_min_drafts + 1))
                buckets.update((bs * width * q + 3) // 4 for q in (1, 2, 3))
    # Retain the runner's native TP/SP token alignment. New compact tiers
    # bypass the request-count filter used to build the original capture grid.
    if maximum % alignment:
        raise ValueError("captured token capacity must satisfy native graph alignment")
    return sorted(
        {
            ((n + alignment - 1) // alignment) * alignment
            for n in buckets
            if 0 < n <= maximum
        }
    )


def configure_verify(result, worker, batch):
    config = worker.draft_extension.config
    if config.verify_mode == "static" or batch.forward_mode.is_idle():
        return
    from msgspec.structs import replace
    from sglang.srt.speculative.eagle_compact_verify import cap_verify_chain
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, round_up_grid

    view = worker.draft_extension.pending_view
    if view is None:
        raise RuntimeError("hybrid verify requires the completed draft selection")
    verify_lens = torch.where(
        view.use_retrieval, config.num_drafts + 1, config.neural_drafts + 1
    ).to(torch.int32)
    dynamic = None
    if config.dynamic_verify:
        dynamic = get_dynamic_verify(worker.draft_extension, view.tokens.device)
        graph = worker.target_worker.model_runner.decode_cuda_graph_runner
        dense = len(batch.seq_lens) * (config.num_drafts + 1)
        grid = (
            graph.capture_num_tokens
            if graph
            else token_buckets(config, [dense], len(batch.seq_lens))
        )
        full = round_up_grid(dense, grid)
        choices = [
            n
            for n in grid
            if len(batch.seq_lens) * (config.dynamic_verify_min_drafts + 1) <= n <= full
        ]
        if graph:
            verify_lens = dynamic.select(
                view, verify_lens - 1, choices, dynamic.context_key
            )
        else:
            # Graph prices cannot predict eager execution. Preserve source caps
            # while still collecting online acceptance feedback.
            verify_lens = dynamic.finish_selection(
                dynamic.minimum(dynamic.model.buckets(view)), verify_lens - 1
            )
    cap_verify_chain(result, verify_lens)
    counters = getattr(worker.draft_extension, "compact_verify_counters", None)
    if counters is None:
        counters = worker.draft_extension.compact_verify_counters = {
            "verify_ct": 0,
            "logical_token_ct": 0,
            "target_token_ct": 0,
            "dense_token_ct": 0,
            "graph_token_histogram": {},
        }
    # The source mask reached TP consensus before dispatch. Reuse split's
    # host count when available; otherwise read one sum, never per-row masks.
    num_neural = getattr(worker.draft_extension, "compact_num_neural", None)
    dense_tokens = len(batch.seq_lens) * (config.num_drafts + 1)
    if dynamic is not None:
        logical = int(verify_lens.sum().item())
    elif config.num_drafts == config.neural_drafts:
        logical = dense_tokens
    elif num_neural is None:
        logical = int(verify_lens.sum().item())
    else:
        logical = dense_tokens - num_neural * (config.num_drafts - config.neural_drafts)
    target_tokens = dense_tokens
    if config.verify_mode == "compact":
        graph = worker.target_worker.model_runner.decode_cuda_graph_runner
        target_tokens = (
            round_up_grid(logical, graph.capture_num_tokens) if graph else logical
        )
        if dynamic is not None and dynamic.profile_batch_size == len(batch.seq_lens):
            # Keep the physical tier pinned even if source caps leave unused
            # capacity. Profile the graph that actually executes, not a logical M.
            target_tokens = max(target_tokens, dynamic.profile_graph_tokens)
        result.compact_verify_layout = replace(
            RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=verify_lens,
                graph_num_tokens=target_tokens,
            ),
            total_verify_tokens=logical,
        )
    worker.draft_extension.pending_verify_tokens = target_tokens
    counters["verify_ct"] += 1
    counters["logical_token_ct"] += logical
    counters["target_token_ct"] += target_tokens
    counters["dense_token_ct"] += dense_tokens
    histogram = counters["graph_token_histogram"]
    histogram[str(target_tokens)] = histogram.get(str(target_tokens), 0) + 1
    if dynamic is not None:
        dynamic.mark_verify(target_tokens)


def get_dynamic_verify(extension, device):
    if getattr(extension, "dynamic_verify", None) is None:
        from hybrid_spec.selection.online_verify import DynamicVerify

        extension.dynamic_verify = DynamicVerify(
            extension.config,
            device,
            getattr(extension, "tp_sync", None),
            getattr(extension, "cost_runtime", None),
        )
    return extension.dynamic_verify


def finalize_verify_counters(extension, result, graph):
    """Account for batch padding on the padded control as well as compact tiers."""
    requested = getattr(extension, "pending_verify_tokens", None)
    if requested is None:
        return
    extension.pending_verify_tokens = None
    if not result.can_run_cuda_graph:
        return
    actual = (
        graph._ragged_graph_size
        if graph.ragged_verify_mode
        else graph.bs * graph.captured_req_width
    )
    counters = extension.compact_verify_counters
    counters["target_token_ct"] += actual - requested
    histogram = counters["graph_token_histogram"]
    histogram[str(requested)] -= 1
    if not histogram[str(requested)]:
        del histogram[str(requested)]
    histogram[str(actual)] = histogram.get(str(actual), 0) + 1
