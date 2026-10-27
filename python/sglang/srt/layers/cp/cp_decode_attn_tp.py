"""CP Decode Attention TP context.

When CP (Context Parallel) mode sets tp_size=1 (repeat weights), decode can
partition attention weights across CP ranks matching normal TP behavior.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import copy
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.cp.utils import is_cp_active
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# HF architectures whose attention linears are replicated with tp_size=1 under
# context parallelism, so slicing them to the local CP partition during decode is
# equivalent to a normal TP-layout GEMM. Any model not on this list must not use
# CP decode attention TP. Single arch-string source of truth for the whitelist.
CP_DECODE_ATTN_TP_SUPPORTED_ARCHS: Tuple[str, ...] = (
    # DeepSeek-V4
    "DeepseekV4ForCausalLM",
    "DeepseekV4ForCausalLMNextN",
    "DeepseekV4ForCausalLMDSpark",
    # GLM-5.x (inherits DeepseekV2 attention; DSA path)
    "GlmMoeDsaForCausalLM",
    "GlmMoeDsaForCausalLMNextN",
)

_global_cp_decode_attn_tp_ctx: CpDecodeAttnTpContext | None = None


def get_cp_decode_attn_tp_ctx() -> CpDecodeAttnTpContext:
    """Return the global CpDecodeAttnTpContext singleton."""
    global _global_cp_decode_attn_tp_ctx
    if _global_cp_decode_attn_tp_ctx is None:
        _global_cp_decode_attn_tp_ctx = CpDecodeAttnTpContext()
    return _global_cp_decode_attn_tp_ctx


class CpDecodeAttnTpContext:
    """Slices replicated attention weights across CP ranks during decode."""

    def __init__(self):
        enable_attn_tp = get_parallel().enable_cp_decode_attn_tp

        if enable_attn_tp and get_parallel().attn_cp_size > 1:
            self.decode_tp_rank = get_parallel().attn_cp_rank
            self.decode_tp_size = get_parallel().attn_cp_size
            logger.info("Enable CP decode attention TP")
        else:
            self.decode_tp_rank = None
            self.decode_tp_size = None

        self.use_decode_attn_tp = False
        self._slice_cache: Dict = {}

    @property
    def is_enabled(self) -> bool:
        return self.decode_tp_size is not None and self.decode_tp_size > 1

    def set_decode_attn_tp(self, forward_batch: ForwardBatch):
        self.use_decode_attn_tp = self.should_use_decode_attn_tp(forward_batch)

    def should_use_decode_attn_tp(self, forward_batch: ForwardBatch) -> bool:
        """Pure selection: PDMux must not publish a process-wide forward flag."""
        if not self.is_enabled:
            return False
        # Skip during prefill context parallel (needs all heads); apply on every
        # other forward, which includes decode.
        return not is_cp_active(forward_batch) and not dsa_use_prefill_cp(forward_batch)

    def _slice(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim in (0, 1)
        chunk = tensor.shape[dim] // self.decode_tp_size
        sliced = tensor.narrow(dim, self.decode_tp_rank * chunk, chunk)
        return sliced if dim == 0 else sliced.contiguous()

    # ==================== Unified activate/restore ====================

    def _activate(self, obj, attr_name: str, dim: int):
        """Replace obj.attr_name with its TP-sliced version. No-op if attr is None."""
        tensor = getattr(obj, attr_name, None)
        if tensor is None:
            return
        is_param = isinstance(tensor, torch.nn.Parameter)
        raw = tensor.data if is_param else tensor
        assert isinstance(raw, torch.Tensor) and raw.dim() > dim, (
            f"CP decode attn TP: {type(obj).__name__}.{attr_name} is not sliceable "
            f"(type={type(tensor).__name__}, dim={raw.dim()}, required_dim>{dim})"
        )
        assert raw.shape[dim] % self.decode_tp_size == 0, (
            f"CP decode attn TP: {type(obj).__name__}.{attr_name}.shape[{dim}]={raw.shape[dim]} "
            f"not divisible by decode_tp_size={self.decode_tp_size}"
        )

        cache_key = (id(obj), attr_name)
        cache = self._slice_cache.get(cache_key)
        if cache is None:
            cache = (raw, self._slice(raw, dim), is_param)
            self._slice_cache[cache_key] = cache

        if cache[2]:
            tensor.data = cache[1]
        else:
            setattr(obj, attr_name, cache[1])

    def _restore(self, obj, attr_name: str):
        cache = self._slice_cache.get((id(obj), attr_name))
        if cache is None:
            return
        orig, _, is_param = cache
        if is_param:
            getattr(obj, attr_name).data = orig
        else:
            setattr(obj, attr_name, orig)

    # ==================== Linear helpers ====================

    def _get_linear_attrs(self, linear_instance) -> List[Tuple]:
        """Return (obj, attr_name, dim) list for a linear layer."""
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

        if isinstance(linear_instance, RowParallelLinear):
            dim = 1
        elif isinstance(linear_instance, ColumnParallelLinear):
            dim = 0
        else:
            return []

        attrs = [(linear_instance, "weight", dim)]
        for scale_name in ("weight_scale_inv", "weight_scale"):
            if getattr(linear_instance, scale_name, None) is not None:
                attrs.append((linear_instance, scale_name, dim))
        return attrs

    @staticmethod
    def _copy_module(module):
        # nn.Module's shallow copy otherwise shares its registration dictionaries;
        # replacing a child or Parameter would still modify the prefill module.
        view = copy(module)
        view._parameters = module._parameters.copy()
        view._buffers = module._buffers.copy()
        view._modules = module._modules.copy()
        return view

    @torch.inference_mode(False)
    def make_decode_attention_view(self, attention, *, radix_attn_name: str):
        """A persistent DSV4 decode module, without rebinding prefill weights.

        The dim-0 shards alias immutable source storage. Row-parallel shards
        retain the contiguous copies required by the existing slice path.
        The caller caches this unregistered module so CUDA graphs retain stable
        weight addresses without adding duplicate entries to the checkpoint.
        """
        assert self.is_enabled
        assert attention.n_heads % self.decode_tp_size == 0
        assert attention.n_groups % self.decode_tp_size == 0
        view = self._copy_module(attention)
        from sglang.srt.layers.linear import RowParallelLinear

        for name in ("wq_b", "wo_a", "wo_b"):
            linear = self._copy_module(getattr(attention, name))
            for obj, attr_name, dim in self._get_linear_attrs(linear):
                tensor = getattr(obj, attr_name, None)
                if tensor is None:
                    continue
                assert (
                    tensor.dim() > dim and tensor.shape[dim] % self.decode_tp_size == 0
                ), (
                    f"CP decode attn TP: {name}.{attr_name} shape {tensor.shape} "
                    f"is not divisible by CP={self.decode_tp_size} along dim {dim}"
                )
                sliced = self._slice(tensor.detach(), dim)
                if isinstance(tensor, torch.nn.Parameter):
                    # Preserve quantized Parameter subclasses without rerunning
                    # their loader-specific __init__ methods.
                    sliced = torch.nn.Parameter.__new__(
                        type(tensor), sliced, requires_grad=tensor.requires_grad
                    )
                # FP8 kernels read metadata attached to scale tensors.
                sliced.__dict__.update(tensor.__dict__)
                setattr(obj, attr_name, sliced)
            size_attr = (
                "input_size_per_partition"
                if isinstance(linear, RowParallelLinear)
                else "output_size_per_partition"
            )
            setattr(
                linear, size_attr, getattr(linear, size_attr) // self.decode_tp_size
            )
            if isinstance(linear, RowParallelLinear):
                linear.use_decode_attn_tp = True
            setattr(view, name, linear)

        radix_attn = self._copy_module(getattr(attention, radix_attn_name))
        radix_attn.tp_q_head_num //= self.decode_tp_size
        setattr(view, radix_attn_name, radix_attn)
        view.n_local_heads = attention.n_heads // self.decode_tp_size
        view.n_local_groups = attention.n_groups // self.decode_tp_size
        view.attn_tp_rank = self.decode_tp_rank
        view.attn_tp_size = self.decode_tp_size
        # Never inherit a sink cached for prefill's full head count.
        view._attn_sink_local = None
        return view

    def refresh_decode_attention_view(self, attention, view):
        """Refresh refitted weights in place, preserving captured addresses."""
        with torch.inference_mode():
            for name in ("wq_b", "wo_a", "wo_b"):
                for _, attr_name, dim in self._get_linear_attrs(getattr(view, name)):
                    source = getattr(getattr(attention, name), attr_name, None)
                    if source is not None:
                        dest = getattr(getattr(view, name), attr_name)
                        dest.copy_(self._slice(source.detach(), dim))
                        dest.__dict__.update(source.__dict__)
            sink = view._attn_sink_local
            if sink is not None:
                sink.zero_()
                start = self.decode_tp_rank * view.n_local_heads
                sink[: view.n_local_heads].copy_(
                    attention.attn_sink[start : start + view.n_local_heads]
                )

    # ==================== Context manager ====================

    @contextmanager
    def maybe_use_decode_attn_tp(
        self,
        forward_batch: ForwardBatch,
        modules: list,
        tensor_attrs: List[Tuple] = None,
        radix_attn: Optional[RadixAttention] = None,
    ):
        """Activate decode attention TP for the duration of the block.

        Args:
            modules: Linear layers (ColumnParallel/RowParallel) to slice.
            tensor_attrs: (obj, attr_name, dim) tuples for absorbed weights.
            radix_attn: RadixAttention instance whose tp_q_head_num should be
                overridden to match the sliced head count during decode TP.
        """
        self.set_decode_attn_tp(forward_batch)
        if not self.use_decode_attn_tp:
            yield
            return

        all_attrs = []  # (obj, attr_name) pairs to restore
        size_overrides = []  # (linear, size_attr, orig_size)
        row_parallel_decode_flags = []  # (RowParallelLinear, orig_flag) to restore
        orig_tp_q_head_num = None
        try:
            for linear in modules:
                for obj, attr_name, dim in self._get_linear_attrs(linear):
                    self._activate(obj, attr_name, dim)
                    all_attrs.append((obj, attr_name))
                from sglang.srt.layers.linear import RowParallelLinear

                size_attr = (
                    "input_size_per_partition"
                    if isinstance(linear, RowParallelLinear)
                    else "output_size_per_partition"
                )
                orig_size = getattr(linear, size_attr)
                setattr(linear, size_attr, orig_size // self.decode_tp_size)
                size_overrides.append((linear, size_attr, orig_size))

                # Set the decode attn TP flag on RowParallelLinear instances
                if isinstance(linear, RowParallelLinear):
                    row_parallel_decode_flags.append(
                        (linear, linear.use_decode_attn_tp)
                    )
                    linear.use_decode_attn_tp = True

            if tensor_attrs:
                for obj, attr_name, dim in tensor_attrs:
                    self._activate(obj, attr_name, dim)
                    all_attrs.append((obj, attr_name))

            if radix_attn is not None:
                orig_tp_q_head_num = radix_attn.tp_q_head_num
                radix_attn.tp_q_head_num = orig_tp_q_head_num // self.decode_tp_size
            yield
        finally:
            if radix_attn is not None and orig_tp_q_head_num is not None:
                radix_attn.tp_q_head_num = orig_tp_q_head_num
            for linear, orig_flag in reversed(row_parallel_decode_flags):
                linear.use_decode_attn_tp = orig_flag
            for linear, size_attr, orig_size in reversed(size_overrides):
                setattr(linear, size_attr, orig_size)
            for obj, attr_name in reversed(all_attrs):
                self._restore(obj, attr_name)
