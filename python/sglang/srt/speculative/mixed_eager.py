"""Offline Qwen3.5 P + full-block DFlash verify prototype.

Not registered with the scheduler or graph runners. Callers own isolated state
snapshots, acceptance, and cache commit. Only token-local work is combined;
attention/recurrence retains each subbatch's native semantics and metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context


@dataclass(frozen=True)
class MixedTokenLayout:
    prefill_tokens: int
    prefill_capacity: int
    verify_requests: int = 3
    block_size: int = 8

    def __post_init__(self):
        if not 0 < self.prefill_tokens <= self.prefill_capacity:
            raise ValueError("P must have positive valid tokens within its capacity")
        if self.verify_requests != 3 or self.block_size != 8:
            raise ValueError("Offline prototype requires one P + D3 x block8")

    @property
    def verify_tokens(self):
        return self.verify_requests * self.block_size

    @property
    def total_tokens(self):
        return self.prefill_capacity + self.verify_tokens

    def prefill_view(self, value, axis=0):
        return value.narrow(axis, 0, self.prefill_tokens)

    def verify_view(self, value, axis=0):
        return value.narrow(axis, self.prefill_capacity, self.verify_tokens)


class MixedAttentionRouter:
    """Route native attention calls; padding never reaches either backend."""

    def __init__(
        self, layout, prefill_batch, verify_batch, prefill_backend, verify_backend
    ):
        if prefill_backend is verify_backend:
            raise ValueError("P and V must own independent attention metadata")
        if prefill_backend.req_to_token_pool is not verify_backend.req_to_token_pool:
            raise ValueError("Subbatches must use the same request pool")
        if prefill_backend.token_to_kv_pool is not verify_backend.token_to_kv_pool:
            raise ValueError("Subbatches must use the same target KV pool")
        self.layout = layout
        self.prefill_batch, self.verify_batch = prefill_batch, verify_batch
        self.prefill_backend, self.verify_backend = prefill_backend, verify_backend
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.verify_backend, name)

    def forward(
        self,
        q=None,
        k=None,
        v=None,
        layer=None,
        forward_batch=None,
        save_kv_cache=True,
        mixed_qkv=None,
        a=None,
        b=None,
        **kwargs,
    ):
        if kwargs:
            raise NotImplementedError(
                f"Unsupported mixed attention arguments: {kwargs.keys()}"
            )
        linear = mixed_qkv is not None
        if linear and not isinstance(mixed_qkv, torch.Tensor):
            raise NotImplementedError(
                "Fused decode projection/Conv is not a mixed path"
            )
        outputs = []
        for batch, backend, view in (
            (self.prefill_batch, self.prefill_backend, self.layout.prefill_view),
            (self.verify_batch, self.verify_backend, self.layout.verify_view),
        ):
            with forward_context(ForwardContext(attn_backend=backend)):
                if linear:
                    value = backend.forward(
                        layer=layer,
                        forward_batch=batch,
                        mixed_qkv=view(mixed_qkv),
                        a=view(a, 1 if a.ndim == 3 else 0),
                        b=view(b, 1 if b.ndim == 3 else 0),
                    )
                else:
                    value = backend.forward(
                        q=view(q),
                        k=view(k) if k is not None else None,
                        v=view(v) if v is not None else None,
                        layer=layer,
                        forward_batch=batch,
                        save_kv_cache=save_kv_cache,
                    )
            outputs.append(value)
        axis = 1 if linear and outputs[0].ndim == 4 else 0
        shape = list(outputs[0].shape)
        shape[axis] = self.layout.total_tokens
        result = outputs[0].new_zeros(shape)
        self.layout.prefill_view(result, axis).copy_(outputs[0])
        self.layout.verify_view(result, axis).copy_(outputs[1])
        self.calls.append((layer.layer_id, "gdn" if linear else "attention"))
        return result


def _text_head(model):
    """Find the text head whose body is a decoder stack, without copying weights."""
    candidates = [model]
    for name in ("language_model", "model"):
        candidate = getattr(model, name, None)
        if candidate is not None:
            candidates.append(candidate)
    for candidate in candidates:
        body = getattr(candidate, "model", None)
        if hasattr(body, "layers") and hasattr(candidate, "logits_processor"):
            return candidate
    raise NotImplementedError("Offline mixed prototype requires a Qwen3.5 text decoder")


def validate_mixed_layout(prefill_batch, verify_batch, prefill_capacity):
    """Validate request ownership outside CUDA capture/replay."""
    if prefill_batch.batch_size != 1 or verify_batch.batch_size != 3:
        raise ValueError("Expected P1 + D3")
    if not prefill_batch.forward_mode.is_extend_without_speculative():
        raise ValueError("P must be a native prefill/continuation batch")
    if not verify_batch.forward_mode.is_target_verify():
        raise ValueError("D must contain the complete native target verify block")
    spec = getattr(verify_batch, "spec_info", None)
    if getattr(spec, "draft_token_num", None) != 8:
        raise ValueError("Verify metadata must retain block8")
    if (
        getattr(spec, "topk", 1) != 1
        or getattr(spec, "ragged_verify_layout", None) is not None
    ):
        raise ValueError(
            "Offline mixed execution requires complete linear DFlash blocks"
        )
    for batch in (prefill_batch, verify_batch):
        for name in ("input_embeds", "replace_embeds", "replace_positions"):
            if getattr(batch, name, None) is not None:
                raise ValueError(f"Unsupported offline mixed input: {name}")
        for name in ("mm_inputs", "lora_ids"):
            values = getattr(batch, name, None)
            if values is not None and not (
                isinstance(values, (list, tuple)) and all(v is None for v in values)
            ):
                raise ValueError(f"Unsupported offline mixed input: {name}")
    if torch.isin(prefill_batch.req_pool_indices, verify_batch.req_pool_indices).any():
        raise ValueError("P and V request rows must be disjoint")
    layout = MixedTokenLayout(len(prefill_batch.input_ids), prefill_capacity)
    if len(verify_batch.input_ids) != layout.verify_tokens:
        raise ValueError("Cannot truncate or pad a DFlash block to fit P")
    return layout


@torch.inference_mode()
def run_mixed_eager(
    model,
    prefill_batch,
    verify_batch,
    prefill_backend,
    verify_backend,
    *,
    prefill_capacity,
    _validated_layout=None,
):
    """Return native P/V logits outputs with one shared transformer forward.

    P padding never reaches stateful calls. The graph adapter validates each
    input frame before capture/replay; its private layout skips GPU-to-host
    validation during capture. Original batch/backend pointers stay untouched.
    """
    layout = _validated_layout or validate_mixed_layout(
        prefill_batch, verify_batch, prefill_capacity
    )
    assert layout.prefill_tokens == len(prefill_batch.input_ids)
    assert layout.prefill_capacity == prefill_capacity
    router = MixedAttentionRouter(
        layout,
        prefill_batch,
        verify_batch,
        prefill_backend,
        verify_backend,
    )
    packed = copy.copy(verify_batch)
    padding = prefill_capacity - layout.prefill_tokens
    packed.input_ids = torch.cat(
        (
            prefill_batch.input_ids,
            prefill_batch.input_ids.new_zeros(padding),
            verify_batch.input_ids,
        )
    )
    packed.positions = torch.cat(
        (
            prefill_batch.positions,
            prefill_batch.positions.new_zeros(padding),
            verify_batch.positions,
        )
    )
    packed.batch_size = 4
    packed.req_pool_indices = torch.cat(
        (prefill_batch.req_pool_indices, verify_batch.req_pool_indices)
    )
    packed.seq_lens = torch.cat((prefill_batch.seq_lens, verify_batch.seq_lens))
    packed.seq_lens_cpu = torch.cat(
        (prefill_batch.seq_lens_cpu, verify_batch.seq_lens_cpu)
    )
    packed.seq_lens_sum = prefill_batch.seq_lens_sum + verify_batch.seq_lens_sum
    packed.out_cache_loc = torch.cat(
        (
            prefill_batch.out_cache_loc,
            prefill_batch.out_cache_loc.new_full((padding,), -1),
            verify_batch.out_cache_loc,
        )
    )
    packed.global_num_token_non_padded_cpu = layout.total_tokens
    for name in ("global_num_token_non_padded", "num_token_non_padded"):
        if getattr(packed, name, None) is not None:
            setattr(packed, name, getattr(packed, name).new_tensor(layout.total_tokens))
    packed._original_batch_size = None
    packed._original_num_tokens = None
    packed._original_forward_mode = None
    packed.forward_mode = ForwardMode.TARGET_VERIFY
    packed.input_embeds = None
    packed.mrope_positions = packed.positions.unsqueeze(0).expand(3, -1)
    head = _text_head(model)
    with forward_context(ForwardContext(attn_backend=router)):
        values = head.model(packed.input_ids, packed.positions, packed)
    hidden, auxiliary = values if isinstance(values, tuple) else (values, None)
    outputs = []
    for batch, view in (
        (prefill_batch, layout.prefill_view),
        (verify_batch, layout.verify_view),
    ):
        aux = None if auxiliary is None else [view(value) for value in auxiliary]
        outputs.append(
            head.logits_processor(
                batch.input_ids,
                view(hidden),
                head.lm_head,
                batch,
                aux,
            )
        )
    return outputs[0], outputs[1], router.calls
