"""Pack chain verification while preserving EAGLE's dense accept/extend ABI."""

from __future__ import annotations

import copy

import torch


def cap_verify_chain(verify_input, verify_lens):
    """End every chain at its last computed logit, which produces the bonus."""
    columns = torch.arange(verify_input.draft_token_num, device=verify_lens.device)
    verify_input.retrieve_next_token = torch.where(
        columns[None, :] + 1 < verify_lens[:, None],
        verify_input.retrieve_next_token,
        -1,
    )
    verify_input.verify_lens = verify_lens


def prepare_compact_verify(verify_input, batch, target_worker):
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        BuildRaggedVerifyWindow,
    )
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    if verify_input.topk != 1 or verify_input.kv_loc_plan is None:
        raise ValueError("compact EAGLE verify requires a planned topk=1 chain")
    if batch.mamba_track_indices is not None or batch.hisparse_coordinator is not None:
        raise ValueError("compact EAGLE verify does not support Mamba or HiSparse")
    runner = target_worker.model_runner
    dense = verify_input.draft_token.view(-1, verify_input.draft_token_num)
    layout = verify_input.compact_verify_layout
    window = BuildRaggedVerifyWindow.execute(
        batch=batch,
        layout=layout,
        draft_block_ids=dense[:, :-1].contiguous(),
        draft_tokens=dense[:, 1:].contiguous(),
        bs=dense.shape[0],
        device=batch.device,
        verify_num_draft_tokens=verify_input.draft_token_num,
        model_runner=runner,
    )
    # The scheduler and native commit/draft-extend retain the full planned KV
    # window. Only this forward writes a packed selection of its columns.
    batch.input_ids = verify_input.draft_token
    batch.out_cache_loc = verify_input.kv_loc_plan.write_virtual
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    packed_batch = copy.copy(batch)
    packed_batch.out_cache_loc = window.verify_cache_loc
    packed = DFlashVerifyInput(
        draft_token=window.verify_ids,
        positions=window.positions,
        draft_token_num=verify_input.draft_token_num,
        custom_mask=None,
        capture_hidden_mode=(
            CaptureHiddenMode.NULL
            if runner.spec_algorithm.is_standalone()
            else CaptureHiddenMode.FULL
        ),
        ragged_verify_layout=layout,
        live_seq_lens_cpu=batch.seq_lens_cpu,
        kv_loc_plan=verify_input.kv_loc_plan,
        kv_loc_cols=window.window_index,
    )
    forward_batch, can_run_graph = packed.prepare_for_verify(
        packed_batch, target_worker
    )
    if can_run_graph:
        forward_batch.mark_forward_metadata_ready()
    return forward_batch, can_run_graph


def scatter_verify_output(verify_input, logits_output):
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        ScatterCompactToStrided,
    )

    # Acceptance and draft-extend continue using request-major node indices.
    # Invalid columns are harmless only because cap_verify_chain cut the links.
    for name in ("next_token_logits", "hidden_states"):
        value = getattr(logits_output, name)
        if value is not None:
            setattr(
                logits_output,
                name,
                ScatterCompactToStrided.execute(
                    compact=value,
                    layout=verify_input.compact_verify_layout,
                    fill_value=0.0,
                    verify_num_draft_tokens=verify_input.draft_token_num,
                ),
            )
