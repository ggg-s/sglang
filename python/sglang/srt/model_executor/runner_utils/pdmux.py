"""Eligibility checks for PDMux whole-prefill graph replay."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


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
