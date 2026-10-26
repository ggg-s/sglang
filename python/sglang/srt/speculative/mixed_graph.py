"""Offline fixed-layout mixed target graph with independent P/V metadata.

No scheduler registration or eager verify fallback. Exact P valid length and
capacity are part of eligibility; callers may retain the native verify graph
for unsupported layouts. Mutable linear-attention metadata is refreshed into
pinned storage, while FlashInfer plans use its native capture-stable wrappers.
"""

from __future__ import annotations

import copy
import dataclasses
from contextlib import nullcontext

import torch

from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    get_attention_backend,
)
from sglang.srt.speculative.mixed_eager import run_mixed_eager, validate_mixed_layout
from sglang.srt.speculative.mixed_marlin import stable_marlin


class MixedGraphLayoutChanged(ValueError):
    pass


SUPPORTED_PREFILL_CAPACITIES = (32, 128, 512)
_CAPTURE_BRANCH_FIELDS = (
    "forward_mode",
    "batch_size",
    "capture_hidden_mode",
    "return_hidden_states_before_norm",
    "return_logprob",
    "is_prefill_only",
    "return_pooled_hidden_states",
    "is_extend_in_batch",
    "spec_algorithm",
    "attn_tp_sequence_sharded",
    "extend_num_tokens",
    "extend_seq_lens_cpu",
    "split_index",
    "dp_padding_mode",
    "global_forward_mode",
)
_SPEC_BRANCH_FIELDS = (
    "draft_token_num",
    "topk",
    "capture_hidden_mode",
    "num_tokens_per_req",
    "num_tokens_for_logprob_per_req",
    "spec_input_type",
)


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        new = copy.copy(value)
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            if isinstance(item, torch.Tensor):
                setattr(new, field.name, item.clone())
        return new
    return copy.copy(value) if isinstance(value, (list, dict)) else value


def _validate_buffers(target, source, *, label):
    for name, old in vars(target).items():
        if isinstance(old, torch.Tensor) and not isinstance(
            getattr(source, name, None), torch.Tensor
        ):
            raise MixedGraphLayoutChanged(f"{label} buffer removed: {name}")
    for name, item in vars(source).items():
        old = getattr(target, name, None)
        if isinstance(item, torch.Tensor):
            if (
                not isinstance(old, torch.Tensor)
                or old.shape != item.shape
                or old.dtype != item.dtype
                or old.device != item.device
            ):
                raise MixedGraphLayoutChanged(f"{label} buffer changed: {name}")


def _validate_batch(target, source):
    for name in _CAPTURE_BRANCH_FIELDS:
        if getattr(target, name, None) != getattr(source, name, None):
            raise MixedGraphLayoutChanged(f"Captured host branch changed: {name}")
    old_spec, new_spec = (
        getattr(target, "spec_info", None),
        getattr(source, "spec_info", None),
    )
    if type(old_spec) is not type(new_spec):
        raise MixedGraphLayoutChanged("Verify metadata type changed")
    if new_spec is not None:
        for name in _SPEC_BRANCH_FIELDS:
            if getattr(old_spec, name, None) != getattr(new_spec, name, None):
                raise MixedGraphLayoutChanged(f"Verify host branch changed: {name}")
        _validate_buffers(old_spec, new_spec, label="Verify")
    _validate_buffers(target, source, label="Batch")


def _copy_batch(target, source):
    # Check every buffer/branch before changing any captured input storage.
    _validate_batch(target, source)
    for name, item in vars(source).items():
        old = getattr(target, name, None)
        if isinstance(item, torch.Tensor):
            old.copy_(item)
        elif name == "spec_info" and item is not None:
            for key, value in vars(item).items():
                if isinstance(value, torch.Tensor):
                    getattr(old, key).copy_(value)
                else:
                    setattr(old, key, value)
        else:
            setattr(target, name, _clone(item))


def _signature(metadata):
    return tuple(
        (
            name,
            ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
            if isinstance(value, torch.Tensor)
            else ("none",)
            if value is None
            else ("value", value),
        )
        for name, value in vars(metadata).items()
    )


def _refresh_metadata(pinned, fresh):
    if _signature(pinned) != _signature(fresh):
        raise MixedGraphLayoutChanged("GDN metadata topology changed")
    for name, value in vars(fresh).items():
        if isinstance(value, torch.Tensor):
            getattr(pinned, name).copy_(value)


class MixedTargetGraph:
    @torch.no_grad()
    def __init__(
        self,
        model_runner,
        prefill_batch,
        verify_batch,
        *,
        prefill_capacity,
        restore,
        prefill_capacity_limit=None,
        linear_tile=(16, 32, 4),
        moe_block_size=8,
        variable_batch=False,
        stable_numerics=True,
    ):
        self.mr = model_runner
        self.variable_batch = variable_batch
        self.stable_numerics = stable_numerics
        self.numerical_options = dict(
            linear_tile=linear_tile, moe_block_size=moe_block_size
        )
        if prefill_capacity_limit is not None and (
            type(prefill_capacity_limit) is not int
            or not 0 < prefill_capacity_limit <= 32768
        ):
            raise MixedGraphLayoutChanged("Invalid experimental P capacity limit")
        if prefill_capacity not in SUPPORTED_PREFILL_CAPACITIES and not (
            prefill_capacity_limit is not None
            and 0 < prefill_capacity <= prefill_capacity_limit
        ):
            raise MixedGraphLayoutChanged("Unsupported offline P graph capacity")
        self.layout = validate_mixed_layout(
            prefill_batch, verify_batch, prefill_capacity,
            variable_batch=self.variable_batch,
        )
        self.prefill_batch = copy.copy(prefill_batch)
        self.verify_batch = copy.copy(verify_batch)
        for target, source in [
            (self.prefill_batch, prefill_batch),
            (self.verify_batch, verify_batch),
        ]:
            for name, value in vars(source).items():
                setattr(target, name, _clone(value))
        self.prefill_backend = get_attention_backend(
            model_runner=model_runner, init_new_workspace=True
        )
        self.verify_backend = get_attention_backend(
            model_runner=model_runner, init_new_workspace=True
        )
        for backend, batch in [
            (self.prefill_backend, self.prefill_batch),
            (self.verify_backend, self.verify_batch),
        ]:
            full = backend.full_attn_backend
            full.init_cuda_graph_state(batch.batch_size, len(batch.input_ids))
            full.init_forward_metadata_out_graph(batch, in_capture=True)
            backend.linear_attn_backend.init_forward_metadata(batch)
        self.pinned_metadata = [
            self.prefill_backend.linear_attn_backend.forward_metadata,
            self.verify_backend.linear_attn_backend.forward_metadata,
        ]
        self._keep_chunk_metadata()
        self.graph = torch.cuda.CUDAGraph()
        self.replays = 0
        try:
            # Warm planner/JIT paths with the exact pinned buffers used by capture.
            restore()
            with self._numerical_context():
                self._forward()
            torch.cuda.synchronize()
            restore()
            with self._numerical_context(), torch.cuda.graph(self.graph):
                self.outputs = self._forward()
            torch.cuda.synchronize()
        finally:
            restore()

    def _numerical_context(self):
        return (
            stable_marlin(self.mr.model, **self.numerical_options)
            if self.stable_numerics else nullcontext()
        )

    def _keep_chunk_metadata(self):
        from sglang.kernels.ops.attention.fla.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
            prepare_lens,
        )

        query = self.pinned_metadata[0].query_start_loc
        # FLA's identity cache can evict another graph's index tensors. Retain
        # every capture-time chunk buffer so recorded pointers cannot dangle.
        self.chunk_buffers = [prepare_lens(query)]
        for size in [16, 32, 64]:
            self.chunk_buffers.extend(
                [prepare_chunk_indices(query, size), prepare_chunk_offsets(query, size)]
            )

    def _forward(self):
        return run_mixed_eager(
            self.mr.model,
            self.prefill_batch,
            self.verify_batch,
            self.prefill_backend,
            self.verify_backend,
            prefill_capacity=self.layout.prefill_capacity,
            _validated_layout=self.layout,
        )

    @torch.no_grad()
    def prepare(self, prefill_batch, verify_batch):
        layout = validate_mixed_layout(
            prefill_batch, verify_batch, self.layout.prefill_capacity,
            variable_batch=self.variable_batch,
        )
        if layout != self.layout:
            raise MixedGraphLayoutChanged("P valid length/capacity changed")
        _validate_batch(self.prefill_batch, prefill_batch)
        _validate_batch(self.verify_batch, verify_batch)
        _copy_batch(self.prefill_batch, prefill_batch)
        _copy_batch(self.verify_batch, verify_batch)
        for index, (backend, batch) in enumerate(
            [
                (self.prefill_backend, self.prefill_batch),
                (self.verify_backend, self.verify_batch),
            ]
        ):
            backend.full_attn_backend.init_forward_metadata_out_graph(batch)
            backend.linear_attn_backend.init_forward_metadata(batch)
            fresh = backend.linear_attn_backend.forward_metadata
            _refresh_metadata(self.pinned_metadata[index], fresh)
            backend.linear_attn_backend.forward_metadata = self.pinned_metadata[index]

    @torch.no_grad()
    def replay(self, prefill_batch, verify_batch):
        self.prepare(prefill_batch, verify_batch)
        self.graph.replay()
        self.replays += 1
        return self.outputs
