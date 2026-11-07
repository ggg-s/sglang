"""PDMux split execution and whole-prefill graph eligibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.utils.device_timer import device_timer_ctx

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput


def _can_replay_whole_split_prefill(
    forward_batch: ForwardBatch,
    forward_count: int,
    num_hidden_layers: int,
    prefill_graph,
    *,
    allows_context_parallel: bool = True,
) -> bool:
    """A full-model BCG is valid only before the first split layer runs."""
    return bool(
        forward_batch.forward_mode.is_split_prefill()
        and forward_batch.split_index == 0
        and forward_count >= num_hidden_layers
        and prefill_graph is not None
        and getattr(prefill_graph, "pdmux_split", False)
        and prefill_graph.can_run_graph(forward_batch)
        and allows_context_parallel
    )


def run_split_prefill(
    model_runner,
    forward_batch: ForwardBatch,
    reinit_attn_backend: bool = False,
    forward_count: int = 1,
) -> Optional[LogitsProcessorOutput]:
    from sglang.srt.layers.cp.utils import is_cp_active, prepare_cp_forward

    graph = getattr(model_runner, "prefill_cuda_graph_runner", None)
    can_run_split = getattr(graph, "can_run_split", None)
    if can_run_split is not None and can_run_split(forward_batch, forward_count):
        output = graph.execute_split(forward_batch, forward_count)
        forward_batch.split_index = min(
            forward_batch.split_index + forward_count,
            model_runner.model_config.num_hidden_layers,
        )
        return output

    # A continuation republishes the original DP sizing before reaching
    # this method. Restore CP's padded physical length on every segment,
    # including when the logical token count is not divisible by CP size.
    if is_cp_active(forward_batch):
        prepare_cp_forward(forward_batch)
    if forward_batch.split_index == 0 or reinit_attn_backend:
        model_runner.attn_backend.init_forward_metadata(forward_batch)
    shadow = None
    if getattr(model_runner, "_pdmux_index_shadow_enabled", False):
        from sglang.srt.multiplex.pdmux_index_shadow import split_prefill_shadow

        shadow = split_prefill_shadow(
            model_runner, forward_batch, reset=reinit_attn_backend
        )
        shadow.check(f"before-segment-{forward_batch.split_index}")
    next_split_index = min(
        forward_batch.split_index + forward_count,
        model_runner.model_config.num_hidden_layers,
    )
    with device_timer_ctx(model_runner.device_timer, "split_prefill"):
        ret = model_runner.model.forward_split_prefill(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            (forward_batch.split_index, next_split_index),
        )
    if shadow is not None:
        shadow.check(f"after-segment-{forward_batch.split_index}")
    forward_batch.split_index = next_split_index
    return ret
