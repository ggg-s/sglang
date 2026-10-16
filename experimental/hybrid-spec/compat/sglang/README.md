# SGLang compatibility

Base commit: `a12814c1585b0c954234f776bba93dbdd04b1151`.

The working tree supplies draft extension API version 2 in four files:

1. `base_spec_worker.py`: optional worker-owned `draft_extension`, default `None`.
2. `eagle_info.py`: transient `EagleDraftInput.draft_override`, default `None`.
3. `eagle_worker_v2.py`: scope around draft execution and per-step selection after
   hot-token mapping, including repair of the fast candidate matrix.
4. `eagle_draft_cuda_graph_runner.py`: static allocation, capture binding and replay
   staging through the extension's graph buffers.

The extension interface is duck typed: `bind(batch, draft_input)` is a context
manager; `create_graph_buffers(max_bs)` returns an object with
`bind_capture_view(batch_size)` and `stage(view, raw_bs, padded_bs)`; a view exposes
`select_step(neural_tokens, step)`. Step is a capture-time loop index, not a replay
counter. A device-history view additionally exposes `prepare(seq_lens)`, called
inside `draft_forward` before token selection, so pending GPU feedback, SA updates
and lookup are included in the draft graph. Version 2 requires the plugin's
prefill and verify hooks for feedback; old CPU-history views need no `prepare`.
The prefill AROUND hook submits a GIL-free host build before target prefill,
stages the GPU bonus after success and cancels matching leases on exceptions.
At the first draft consumer it imports used pinned CPU prefixes on the forward
stream, preserving GPU bonus feedback. Copy sources are retained until a CUDA
event completes; background construction never touches CUDA or request objects.
Registration installs resources after both ordinary and Standalone draft
constructors, before memory-pool planning and graph capture.

The plugin checks the API markers, resolved configuration and model capabilities.
There is no promise of compatibility with later commits. Run the CPU and CUDA
tests and real-model comparisons whenever rebasing. The native sampling, verify,
draft_extend, tree building and KV allocation algorithms are unchanged.

Single-node TP uses the existing `get_tp_group().device_group`, with the first
global rank in `get_tp_group().ranks` as the broadcast source. Each rank owns an
SA replica. Before draft execution, the plugin broadcasts candidates and reduces
eligibility/candidate agreement with MIN. A disagreement falls back on all ranks.
This relies on the native TP scheduler providing the same ordered batch and K.
CPU-history collectives run outside CUDA Graph capture/replay. Device history
runs consensus inside `prepare(seq_lens)`, AFTER local SA lookup and BEFORE
token selection. Eager and captured paths use the TP group's existing PyNccl
communicator on the current stream, with explicit enable/restore and conversion
from global source rank to group-relative root. Device TP fails if PyNccl is
unavailable. Graph outputs are copied back into the existing tokens/mask storage,
so verify metrics see the final unanimous mask. TP=1 creates no synchronizer.
CPU Gloo bridge tests do not establish GPU/NCCL or model correctness.
DP/PP/EP/CP and multi-node remain rejected.
