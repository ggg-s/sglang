"""Opt-in observations of PDMux index mutation; no cross-lane synchronization.

These checks hold references and enqueue copies/comparisons, so enabling them
can change allocator reuse and timing. A clean diagnostic run is not a race
freedom proof. They neither repair indices nor identify the writing kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
from weakref import ref

import torch

from sglang.srt.distributed.parallel_state import is_pdmux_prefill_enabled
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx

INPUT_FIELDS = ("out_cache_loc", "req_pool_indices")
PREFILL_FIELDS = INPUT_FIELDS + (
    "seq_lens",
    "extend_seq_lens",
    "extend_prefix_lens",
    "extend_start_loc",
)
LINEAR_FIELDS = ("query_start_loc", "mamba_cache_indices")


@dataclass
class IndexShadow:
    description: str
    tensors: list[tuple[str, torch.Tensor, torch.Tensor, int]] = field(
        default_factory=list
    )
    bindings: list[tuple[Callable, str, object, str]] = field(default_factory=list)

    def bind(self, owner, name, expected, label):
        # The persistent shadow is attached to the ForwardBatch. A strong
        # back-reference would retain its GPU activations until cyclic GC.
        try:
            owner_ref = ref(owner)
        except TypeError:
            owner_ref = lambda: owner
        self.bindings.append((owner_ref, name, expected, label))

    def capture(self, owner, names, *, prefix, check_binding=False):
        for name in names:
            source = getattr(owner, name, None)
            if not torch.is_tensor(source):
                continue
            # Register diagnostic readers as well as retaining the references.
            # Both are needed if the observed tensor came from another stream.
            if source.is_cuda:
                source.record_stream(torch.cuda.current_stream(source.device))
            self.tensors.append(
                (f"{prefix}.{name}", source, source.clone(), source.data_ptr())
            )
            if check_binding:
                self.bind(owner, name, source, f"{prefix}.{name}")

    def check(self, stage):
        for owner_ref, name, expected, label in self.bindings:
            if getattr(owner_ref(), name, None) is not expected:
                raise RuntimeError(
                    f"PDMux index binding changed [{self.description}] "
                    f"{label} at {stage}"
                )
        for label, source, snapshot, captured_ptr in self.tensors:
            if source.shape != snapshot.shape or source.data_ptr() != captured_ptr:
                raise RuntimeError(
                    f"PDMux index storage changed [{self.description}] {label} at {stage}"
                )
            if source.is_cuda:
                stream = torch.cuda.current_stream(source.device)
                source.record_stream(stream)
                snapshot.record_stream(stream)
            torch._assert_async(
                (source == snapshot).all(),
                f"PDMux index mutated [{self.description}] {label} at {stage} "
                f"ptr={captured_ptr:#x} shape={tuple(source.shape)}",
            )


def describe(runner, site):
    role = "draft" if runner.is_draft_worker else "target"
    lane = "P" if is_pdmux_prefill_enabled() else "D"
    return (
        f"{role}/{lane} tp={runner.ps.tp_rank} "
        f"pass={runner.forward_pass_id} group={get_current_stream_idx()} {site}"
    )


def capture_inputs(runner, inputs, *, site):
    shadow = IndexShadow(describe(runner, site))
    shadow.capture(inputs, INPUT_FIELDS, prefix=site)
    return shadow


def split_prefill_shadow(runner, batch, *, reset=False):
    shadow = getattr(batch, "_pdmux_index_shadow", None)
    if shadow is not None and batch.split_index > 0 and not reset:
        return shadow

    shadow = IndexShadow(describe(runner, "split-prefill"))
    shadow.capture(batch, PREFILL_FIELDS, prefix="batch", check_binding=True)
    backend = runner.attn_backend
    linear = getattr(backend, "linear_attn_backend", backend)
    metadata = getattr(linear, "forward_metadata", None)
    if metadata is not None:
        shadow.bind(linear, "forward_metadata", metadata, "linear.metadata")
        shadow.capture(metadata, LINEAR_FIELDS, prefix="linear", check_binding=True)
    batch._pdmux_index_shadow = shadow
    return shadow
