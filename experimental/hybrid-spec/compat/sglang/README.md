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
counter. A device-history or TP CPU-history view additionally exposes
`prepare(seq_lens)`, called inside `draft_forward` before token selection.
Device history includes pending GPU feedback, SA updates and lookup in the
draft graph; CPU history stages local candidates and captures TP consensus.
Version 2 requires the plugin's
prefill and verify hooks for feedback; old CPU-history views need no `prepare`.
Device verify accumulates the six existing source counters in the feedback
kernel, including neural rows with no live SA slot. Prefill does not increment
these counters. Diagnostic readback is still explicit and outside decode.
Worker-created device views carry an immutable host mapping key. Graph staging
may reuse slot/generation inputs only when that key and raw/padded batch sizes
match; keyless mutable views always stage. Selection outputs are never cached.
The prefill AROUND hook submits a GIL-free host build before target prefill,
stages the GPU bonus after success and cancels matching leases on exceptions.
With chunked prefill, device history does this only for final-chunk rows. The
batch-local `chunked_req` marks the intermediate row; its transient sampled
token is discarded and it reserves no SA slot. Other final rows in that same
batch remain eligible. The final chunk snapshots the full request history,
including cached/earlier chunks, before launch. Re-prefill invalidates old
leases even on intermediate chunks. The legacy GPU SA CPU-history mode still
requires non-overlap scheduling and disabled chunked prefill; Arctic uses its
own GPU feedback bridge described below. Mixed prefill/decode remains rejected.
At the first draft consumer it imports used pinned CPU prefixes on the forward
stream, preserving GPU bonus feedback. Copy sources are retained until a CUDA
event completes; background construction never touches CUDA or request objects.
Registration installs resources after both ordinary and Standalone draft
constructors, before memory-pool planning and graph capture.

The plugin checks the API markers, resolved configuration and model capabilities.
There is no promise of compatibility with later commits. Run the CPU and CUDA
tests and real-model comparisons whenever rebasing. The native sampling, verify,
draft_extend, tree building and KV allocation algorithms are unchanged.

Single-node TP uses the existing `get_tp_group().device_group`. Each rank owns
an SA replica. One MIN all-reduce over `[tokens, ~tokens, eligibility]` computes
exact candidate agreement and unanimous eligibility, without a broadcast root.
Bitwise NOT reverses signed-int64 ordering, including the extrema. A disagreement
falls back on all ranks. This relies on the native TP scheduler providing the
same ordered batch and K; local eligibility never controls collective participation.
Both history modes run consensus inside `prepare(seq_lens)`, AFTER local SA
lookup and BEFORE token selection. Eager and captured paths use the TP group's
existing PyNccl communicator on the current stream, with explicit enable/restore.
Both modes fail on CUDA TP if PyNccl is unavailable. Graph outputs are copied
back into the existing tokens/mask storage; CPU-history staging aliases the
pending view's mask to this storage so verify metrics see the final consensus.
K=1 retains the native eager draft path; TP=1 creates no synchronizer.
CPU Gloo bridge tests do not establish GPU/NCCL or model correctness.
DP/PP/EP/CP and multi-node remain rejected.

Split graphs additionally require worker and runner split API version 5. EAGLE/MTP,
EAGLE3 and STANDALONE share the GPU dynamic row compaction and gate=0..K path.
The prefix callback receives a dense target-vocabulary draft chain even with a
hot-token map; mapped IDs are written after selection in both prefix and tail.
The compact tail supports an absent recurrent hidden state (STANDALONE), and
otherwise stages the actual prefix output shape (including EAGLE3). Full mode
retains its existing token-list fallback for mapped vocabularies. Triton draft
attention and native dsv4 MTP attention are supported; other model/backend and
parallelism restrictions still apply. See the [4060 validation scope](../../validation/draft-algorithms-split-20261017.md).

The optional official ArcticInference CPU suffix backend supports overlap and
chunked prefill using the same verify/prefill hooks. It asynchronously copies
packed GPU accept paths to pinned CPU memory and waits on their CUDA event at
the next query consumer. Final-chunk snapshots include cached/earlier prefixes;
intermediate bonus tokens are discarded. Private local trees follow GPU verify,
while a separate shared tree receives only CPU-confirmed response increments.
CPU lookup and H2D staging happen before replay; shared GatedOverride performs
final gate/TP consensus before GPU row compaction. No worker API version change
is needed beyond split v5. See [Arctic setup](../../validation/arctic-suffix.md).
