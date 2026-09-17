# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cross-stream ownership for the tensors a ScheduleBatch carries between the
PD-Multiplexing prefill and decode lanes.

Both lanes rebind batch state on their own stream and the other lane then reads
it: the prefill lane's admission (priority preemption -> `filter_batch`) and its
finalize (`merge_batch`) both allocate on the prefill stream, and the decode
lane consumes the result. `record_stream` tells the caching allocator that a
tensor allocated on one stream is still live on another, so its memory is not
handed out again while the consumer is still reading.

`record_stream` covers allocator reuse only. It is **not** a data-ready
dependency -- that is what the loop's cross-lane events (E1 formation_done,
E2 merge_done, E3 decode_done) provide. Both are required; neither substitutes
for the other.

Traversal is deliberately bounded. A ScheduleBatch also references the KV pool,
the tree cache and the Req list, whose buffers are engine-lifetime and must not
be registered per batch; recursing the object graph would sweep them in. Only
the containers listed in `iter_carried_tensors` are visited, one level each.

Fields intentionally not visited, and why:

- ``req_to_token_pool`` / ``token_to_kv_pool_allocator`` / ``tree_cache`` --
  engine-lifetime pools, not per-batch state.
- ``reqs`` and everything reachable from a ``Req`` -- request-lifetime; a
  request outlives the batch it is in and is never freed by the allocator on a
  lane boundary.
- ``out_cache_loc_dsv4`` -- an allocation bundle produced and consumed inside
  one forward on one lane; it is not rebound by ``filter_batch`` /
  ``merge_batch``.
- CPU-side mirrors (``seq_lens_cpu``, ``req_pool_indices_cpu``, ...) -- host
  memory has no stream ownership.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Iterator

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def _field_values(obj: Any) -> Iterator[Any]:
    """Attribute values of one object, for dataclass / msgspec / plain classes.

    Reflection rather than a hand-kept name list: a name list silently goes
    stale when a field is added, and the failure mode is an unregistered tensor
    rather than a test failure. The ratchet test asserts the *classification* of
    every declared field instead.
    """
    if dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            yield getattr(obj, field.name)
        return
    struct_fields = getattr(type(obj), "__struct_fields__", None)  # msgspec.Struct
    if struct_fields is not None:
        for name in struct_fields:
            yield getattr(obj, name)
        return
    yield from vars(obj).values()


def _is_carried_device_tensor(value: Any) -> bool:
    """Whether this value is memory the caching allocator can hand back.

    Named rather than inlined so the traversal's *reach* -- which objects are
    walked and which are deliberately not -- can be asserted without a CUDA
    device by substituting the predicate.
    """
    return torch.is_tensor(value) and value.is_cuda


def _own_device_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Device tensors held directly by `obj`. Does not descend."""
    if obj is None:
        return
    for value in _field_values(obj):
        if _is_carried_device_tensor(value):
            yield value


def _grammar_vocab_mask(sampling_info: Any) -> Iterator[torch.Tensor]:
    """The filled vocab mask of the batch's grammar, if there is one.

    Read by name rather than by the generic attribute walk: `GrammarMask` is a
    `NamedTuple` (constrained/base_grammar_backend.py), so it has no `__dict__`
    and `vars()` raises on it. Naming the field also keeps the walk out of the
    sibling `grammar` handle, which is backend state shared across batches, not
    per-batch memory the allocator may hand back.
    """
    grammar_mask = sampling_info.grammar_mask
    if grammar_mask is None:
        return
    vocab_mask = grammar_mask.vocab_mask
    if _is_carried_device_tensor(vocab_mask):
        yield vocab_mask


def _custom_logit_processor_masks(sampling_info: Any) -> Iterator[torch.Tensor]:
    """Per-request masks of the custom logit processors.

    `SamplingBatchInfo.custom_logit_processor` maps a processor key to
    ``(processor, mask)``; `merge_custom_logit_processor` cats the masks, so the
    merged mask is allocated on the merging lane's stream.
    """
    processors = sampling_info.custom_logit_processor
    if not processors:
        return
    for entry in processors.values():
        for value in entry:
            if _is_carried_device_tensor(value):
                yield value


def iter_carried_tensors(batch: ScheduleBatch) -> Iterator[torch.Tensor]:
    """Device tensors this batch carries across a lane boundary.

    Visits the batch itself plus four named sub-objects: the sampling info, its
    grammar mask, its custom-logit-processor masks, and each prepared penalizer;
    and the speculative state. Penalizer tensors are created in ``_prepare()``
    rather than declared on the class, so they are only reachable this way, and
    the grammar mask is a ``NamedTuple`` whose one carried tensor is read by
    name (see ``_grammar_vocab_mask``).
    """
    yield from _own_device_tensors(batch)

    sampling_info = batch.sampling_info
    if sampling_info is not None:
        yield from _own_device_tensors(sampling_info)
        yield from _grammar_vocab_mask(sampling_info)
        yield from _custom_logit_processor_masks(sampling_info)
        orchestrator = sampling_info.penalizer_orchestrator
        if orchestrator is not None:
            for penalizer in orchestrator.penalizers.values():
                yield from _own_device_tensors(penalizer)

    if batch.spec_info is not None:
        yield from _own_device_tensors(batch.spec_info)


def publish_carried_tensors(batch: ScheduleBatch, stream) -> None:
    """Register this batch's carried tensors as live on `stream`.

    Call at the point the consuming lane is about to read them, with that lane's
    *current* stream: a stream-group switch replaces the stream object, so a
    registration taken earlier does not describe the stream that ends up
    consuming the batch.
    """
    for tensor in iter_carried_tensors(batch):
        tensor.record_stream(stream)
