"""Equal-length SA proposals around DSPARK's native block verify and KV commit.

The neural block still runs. Replacing tokens in place preserves its folded
sampler/verify buffer alias; native code owns every model and KV operation.
"""

from __future__ import annotations

from types import SimpleNamespace

import msgspec
import torch


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
    return msgspec.structs.replace(
        result,
        draft_block=msgspec.structs.replace(
            block, corrected_logits=None, greedy_draft=True
        ),
        # A neural confidence head does not score SA's replacement tokens.
        confidence=None,
        confidence_raw=None,
    )


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
