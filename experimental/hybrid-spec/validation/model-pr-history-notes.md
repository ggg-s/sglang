# Qwen3.6 MTP + SA adaptation evidence

Baseline: `1b0f58030790726b3146f46786a38b23a7f735e8`.

The local Qwen3.5 and Qwen3.6 PR-history dossiers were consulted before changing the model capability gate. Historical evidence guides inspection; it does not establish correctness on this checkout.

- [PR 19391](https://github.com/sgl-project/sglang/pull/19391): Qwen3.5 MTP support for SpecV2; draft-extend must use the V2 path. Current `qwen3_5_mtp.py` handles `is_draft_extend_v2()` and constructs a single full-attention MTP layer.
- [PR 20864](https://github.com/sgl-project/sglang/pull/20864): Qwen3.5 speculative mRoPE and device-side position handling. Current `eagle_worker_v2.py` advances both regular and mRoPE positions; graph staging copies all three mRoPE axes. SA selection must change tokens without changing positions.
- [PR 23486](https://github.com/sgl-project/sglang/pull/23486): Qwen3.6 retains a GDN backbone. The downloaded 35B-A3B FP8 checkpoint declares `Qwen3_5MoeForConditionalGeneration`, with linear/full-attention layers and an embedded MTP head.

Current source verification: `run_eagle_verify` calls `commit_mamba_states_after_verify` using native acceptance indices and lengths. GDN verification writes speculative intermediate states; the commit path selects only the accepted prefix (including the anchor input, not the unprocessed bonus token). The extension keeps this path intact. Qwen's MTP layer is full attention, so it does not introduce a separate recurrent draft-state rollback protocol.

Scope: only the Qwen3.5/3.6 target and `Qwen3_5ForCausalLMMTP` pair, both configured with `--language-only`. Other linear-attention architectures, sliding-window pools, and actual multimodal requests remain rejected. GPU token parity, rejection recovery, graph replay, and measured SA usage remain separate acceptance criteria from these source findings.

## Plugin performance review (2026-10-16)

Pre-change HEAD: `4030650794`; the supplied Qwen3-14B/0.6B report used
`1b0f580307`. Read the local `model-pr-history-knowledge/sglang/qwen3-core/README.en.md`
implementation coverage and PR #7745 card (Qwen EAGLE3 auxiliary hidden-state
integration), plus the existing Qwen3.5/3.6 notes above. Read the maintained
`AI-Infra-Auto-Driven-Skills/docs/upstream-source-contracts.md` through the skill's
resolved junction target: historical dispatch and trace attribution do not
certify the current runtime or a performance cause.

Decision: optimize the shared plugin consensus/graph boundary, retaining native
model hidden-state capture, token mapping, verification and acceptance paths.
No model-kernel replacement follows from the report's matching GEMM/attention
tables. Current `draft_forward` invokes `prepare` before the first selection;
the scheduler budgets KV after draft construction using currently free GPU
memory. These source observations, rather than historical PR titles, determine
the changes. GPU/NCCL timing and model parity remain deployment acceptance gates.

Direct sa_spec source comparison was rechecked on 2026-10-16 at clean local
HEAD `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969`: `src/api.cc` builds prompt
states on the host and copies used prefixes when requests enter generation;
`src/api.cu` consumes accepted device tokens and performs lookup in one
current-stream kernel; `src/memory.cu` uses asynchronous copies on the active
Torch stream. This supports the existing device-history path, selected explicitly
by `device-k3.json` or `device-sa-spec.json`. The standalone library has no TP
consensus protocol, which is not evidence that SGLang may safely remove consensus
while allowing rank-local slot/version fallback. No new GPU performance claim
is inferred from the reference implementation.

## Device overhead follow-up (2026-10-16)

Baseline: `2d279d4f34`. Reused the Qwen3-core PR #7745 evidence and current
native draft/verify boundaries above; no model-specific forward changes.
Re-read sa_spec `src/api.cu`, `src/sa_spec/util/flat_graph.hpp` and
`flat_hash_map.hpp`, plus the locally retained TRT-LLM #10951 head
`304ec62dd635cc7ad865c2ae35b4ae02910396d0` `reference-mtp.py` (acceptance,
`sa_spec.extend`, per-step `draft_sampler`, and next-input update).

The reference retains neural forward and selects SA tokens at each sampling
step. The first optimization targets here are our additional six per-verify
PyTorch statistics reductions and repeated stable graph-input copies, not
moving the source-selection point or changing model recurrence. Metrics are
fused into the existing private plugin feedback operator; no new public
`sglang.kernels` entry point or experimental-package relocation is introduced.
Reference hash-based transition lookup versus our linked adjacency scan is a
separate long-history performance risk; changing layout needs its own CPU/GPU
differential validation and length/vocabulary sweep. Source comparison alone
does not establish its contribution to the supplied EAGLE3 slowdown.

## Hash transition follow-up (2026-10-16)

Baseline: `44637fe669`. Read the same pinned sa_spec `flat_graph.hpp` and
`flat_hash_map.hpp`: inline single edge, then per-node open addressing, growth
by five with a full copy and no allocator reclamation. This implementation uses
an independently written exact `(state, token)` chained hash index with linear
bucket splitting. Up to four outgoing edges stay on a short-list path to avoid
hashing low-diversity histories; larger lists are indexed. The adjacency list
is retained for clone enumeration. Used prefixes remain relocatable between
compact host construction and larger GPU slots, with no device allocation or
full-table rebuild during graph replay. These storage choices are specific to
our bounded runtime and are not claims about sa_spec benchmark equivalence.
See VALIDATION.md for native measurements and the separate CUDA validation gap.

## Compact inline transitions (2026-10-16)

Baseline: `88f8d971eb`; original-list control: `44637fe669`. Re-read sa_spec
`flat_graph.hpp`: its inline edge avoids an allocated edge record for degree-one
states, and clones copy inline state directly. This implementation extends the
idea to a tagged second target while retaining the bounded hash overflow pool.
Primary labels come from the source's first-position successor; secondary labels
come from the destination's first-position token. Clone retargeting preserves
these labels. The pool bound follows from an injective mapping of non-primary
edges to suffix paths whose last non-primary edge is the mapped edge; the full
string follows primary edges only. Thus there are at most n-1 non-primary edges,
and stored overflow edges are a subset. Tests exercise the tight distinct-token
case, unary/binary zero-pool cases, and transitions to higher degree. This is a
layout-specific argument, not a capacity guess from observed corpora.

Native checksums now canonicalize semantic transitions instead of physical
adjacency addresses. Both original-list and previous-hash A/B measurements are
retained. No model forward, acceptance rule, TP consensus, or scheduling change.
