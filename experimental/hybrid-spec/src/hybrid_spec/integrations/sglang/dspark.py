"""Equal-length SA proposals around DSPARK's native block verify and KV commit.

Full mode preserves folded sampler storage. Gate=0 split executes the neural
proposal only for miss rows, then uses the native non-folded accept/commit.
"""

from __future__ import annotations

from copy import copy
from types import SimpleNamespace

import msgspec
import torch


class _SplitProposalBuffers:
    """Worker-owned scratch; static verification consumes it before the next step."""

    def __init__(self, max_bs, num_drafts, device, mask_token_id):
        device = torch.device(device)
        self.use_retrieval_cpu = torch.empty(
            max_bs, dtype=torch.bool, pin_memory=device.type == "cuda"
        )
        self.tokens = torch.empty(
            (max_bs, num_drafts), dtype=torch.int64, device=device
        )
        self.block_ids = torch.full_like(self.tokens, mask_token_id)
        self.greedy_mask = torch.ones(max_bs, dtype=torch.bool, device=device)
        self.temperatures = torch.ones(max_bs, device=device)


def attach(result, worker, *args, **kwargs):
    from sglang.srt.distributed.parallel_state import get_tp_group

    from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
    from hybrid_spec.integrations.sglang.compatibility import validate_dspark_model
    from hybrid_spec.integrations.sglang.device_extension import DeviceDraftExtension
    from hybrid_spec.integrations.sglang.tp_sync import TPSynchronizer
    from hybrid_spec.plugin import _config, _validate

    _validate(worker.server_args)
    config = _config()
    validate_dspark_model(worker, config)
    group = get_tp_group()
    sync = None
    if group.world_size > 1:
        comm = group.pynccl_comm
        if comm is None or not comm.available:
            raise ValueError("DSPARK SA TP requires an available PyNccl communicator")
        sync = TPSynchronizer(group.device_group, comm)
    worker.draft_extension = DeviceDraftExtension(
        config,
        GPUSuffixAutomaton(config, f"cuda:{worker.gpu_id}"),
        worker.target_worker.model_runner.model_config.vocab_size,
        tp_sync=sync,
    )
    worker._proposer._hybrid_spec_extension = worker.draft_extension
    if config.draft_graph_mode == "split":
        worker.draft_extension.dspark_split_buffers = _SplitProposalBuffers(
            config.max_requests,
            config.num_drafts,
            worker.draft_extension.backend.device,
            worker._proposer._mask_token_id,
        )
        worker.draft_extension.dspark_draft_counters = {
            "proposal_ct": 0,
            "request_ct": 0,
            "neural_request_ct": 0,
            "skip_request_ct": 0,
            "all_sa_ct": 0,
            "mixed_ct": 0,
        }


def propose(original, proposer, **kwargs):
    extension = getattr(proposer, "_hybrid_spec_extension", None)
    if extension is None or extension.pending_view is None:
        return original(proposer, **kwargs)
    if extension.config.draft_graph_mode == "full":
        return splice_proposal(original(proposer, **kwargs), proposer)

    # All TP ranks agree on candidates before any rank skips a collective.
    view = extension.pending_view
    view._synchronize()
    buffers = extension.dspark_split_buffers
    bs = kwargs["bs"]
    # CUDA nonzero first synchronizes to size its output, then rows.cpu() must
    # synchronize again for nonempty rows. Read one fixed-size mask instead;
    # host dispatch and native CPU length metadata still require this one wait.
    use_retrieval_cpu = buffers.use_retrieval_cpu[:bs]
    use_retrieval_cpu.copy_(view.use_retrieval)
    neural_rows = [i for i, hit in enumerate(use_retrieval_cpu.tolist()) if not hit]
    num_neural = len(neural_rows)
    counters = extension.dspark_draft_counters
    counters["proposal_ct"] += 1
    counters["request_ct"] += bs
    counters["neural_request_ct"] += num_neural
    counters["skip_request_ct"] += bs - num_neural
    counters["all_sa_ct"] += int(num_neural == 0)
    counters["mixed_ct"] += int(0 < num_neural < bs)
    if num_neural == bs:
        # No selection change: retain the original folded output alias.
        return _without_confidence(original(proposer, **kwargs))

    from sglang.srt.speculative.dspark_components.dspark_draft import (
        DraftBlockResult,
        DraftProposal,
    )

    # All-SA output can borrow the query buffer: native static verification
    # reads it without modification, and feedback finishes before the next query.
    tokens = view.tokens
    if num_neural:
        # Only a mixed batch needs indices or any proposal-row packing.
        rows_cpu = torch.tensor(neural_rows, dtype=torch.int64)
        rows = rows_cpu.to(tokens.device, non_blocking=True)
        tokens = buffers.tokens[:bs]
        tokens.copy_(view.tokens)
        compact = _select_proposal_rows(kwargs, rows, rows_cpu)
        neural = original(proposer, **compact)
        tokens.index_copy_(0, rows, neural.draft_block.draft_tokens)
    block_ids = buffers.block_ids[:bs]
    block_ids[:, 0].copy_(kwargs["draft_input"].bonus_tokens.reshape(-1))
    return DraftProposal(
        draft_block_ids=block_ids,
        draft_block=DraftBlockResult(
            draft_tokens=tokens,
            corrected_logits=None,
            greedy_mask=buffers.greedy_mask[:bs],
            temperatures=buffers.temperatures[:bs],
            greedy_draft=True,
        ),
        # A compact folded sampler writes its first N rows, not the original
        # full-batch slots. Native non-folded accept/commit consumes our merged
        # buffer and still commits target hidden/KV for every original row.
        folded=False,
    )


def _select_proposal_rows(kwargs, rows, rows_cpu):
    batch = copy(kwargs["batch"])
    draft_input = copy(kwargs["draft_input"])
    window = kwargs["verify_window"]
    batch.seq_lens = batch.seq_lens.index_select(0, rows)
    batch.req_pool_indices = batch.req_pool_indices.index_select(0, rows)
    if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = batch.seq_lens_cpu.index_select(0, rows_cpu)
    batch.reqs = [batch.reqs[i] for i in rows_cpu.tolist()]
    draft_input.bonus_tokens = draft_input.bonus_tokens.index_select(0, rows)
    if draft_input.nxt_kv_lens_cpu is not None:
        draft_input.nxt_kv_lens_cpu = draft_input.nxt_kv_lens_cpu.index_select(
            0, rows_cpu
        )
        draft_input.nxt_kv_lens_sum = int(draft_input.nxt_kv_lens_cpu.sum())
    positions = window.positions_2d.index_select(0, rows)
    cache = window.verify_cache_loc_2d.index_select(0, rows)
    window = msgspec.structs.replace(
        window,
        positions_2d=positions,
        verify_cache_loc=cache.reshape(-1),
        verify_cache_loc_2d=cache,
        kv_loc_plan=window.kv_loc_plan.select_rows(rows, rows_cpu),
    )
    # SA serving is greedy-only. None requests the native argmax draft params
    # without mutating/filtering the target's SamplingBatchInfo or penalizers.
    return dict(
        kwargs,
        batch=batch,
        draft_input=draft_input,
        verify_window=window,
        bs=rows.numel(),
        sampling_info=None,
    )


def _without_confidence(result):
    return msgspec.structs.replace(
        result,
        draft_block=msgspec.structs.replace(
            result.draft_block, corrected_logits=None, greedy_draft=True
        ),
        confidence=None,
        confidence_raw=None,
    )


def splice_proposal(result, proposer, **kwargs):
    extension = getattr(proposer, "_hybrid_spec_extension", None)
    if extension is None or extension.pending_view is None:
        return result
    view = extension.pending_view
    block = result.draft_block
    neural = block.draft_tokens
    if neural.shape != view.tokens.shape:
        raise ValueError("DSPARK and SA must propose the same block shape")
    gate = extension.config.num_neural_gate_tokens
    if gate:
        view.use_retrieval.logical_and_(
            (view.tokens[:, :gate] == neural[:, :gate]).all(dim=1)
        )
    # One consensus after both query and gate. DSPARK has already generated its
    # complete neural block, so there is no earlier per-token selection consumer.
    view._synchronize()
    neural.copy_(torch.where(view.use_retrieval[:, None], view.tokens, neural))
    # A neural confidence head does not score SA's replacement tokens.
    return _without_confidence(result)


def stage_feedback(extension, view, result, prefix_lens):
    width = extension.config.num_drafts + 1
    bs = len(prefix_lens)
    # DSPARK emits a dense row: accepted drafts followed by one bonus, then
    # unused tail slots. accept_lens already includes that bonus.
    indices = torch.arange(
        bs * width, dtype=torch.int32, device=prefix_lens.device
    ).view(bs, width)
    extension.backend.stage_feedback(
        view.slots,
        view.generations,
        result.next_token_ids.to(torch.int32),
        indices,
        result.accept_lens.to(torch.int32),
        prefix_lens,
        True,
        extension.vocab_size,
        use_retrieval=view.use_retrieval,
        counts=extension.metrics.ensure_buffer(extension.backend.device),
    )


def decode(original, worker, batch, *args, **kwargs):
    extension = worker.draft_extension
    if extension is None or batch.forward_mode.is_idle():
        return original(worker, batch, *args, **kwargs)
    # Keep the pre-verify length even if a native path updates its input buffer.
    prefix_lens = batch.seq_lens.clone()
    try:
        # Binding only manages SA request leases. DSPARK's DFlash spec state
        # remains native, including overlap FutureMap and KV reservation fields.
        with extension.bind(batch, SimpleNamespace()) as view:
            view._query(prefix_lens)
            result = original(worker, batch, *args, **kwargs)
            stage_feedback(extension, view, result, prefix_lens)
            return result
    finally:
        extension.pending_view = None
