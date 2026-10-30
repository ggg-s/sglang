# D scheduler overlap — local validation, 2026-10-30

Scope: P keeps overlap disabled. D Remote batches serialize at the native
`Scheduler.is_disable_overlap_for_batch` boundary. The final queued Remote result
is consumed at the Local boundary; stable Local resumes the native predicate and
full CUDA graph bank. Mixed Remote/Local batches conservatively serialize while
any progressive target lease remains. No SMG or SGL core files changed.

## Completion and ownership

- Attach a batch proof before launching each forward involving a progressive
  target, retaining the native result through GPU completion. PREBUILT has no
  model forward and does not create a proof; an inner idle forward re-enters the
  ordinary hook.
- Reuse native `GenerationBatchResult.copy_done`: its copy stream follows the
  forward stream after forward/future publication/sampling. No additional CUDA
  event is recorded. A delayed sampling closure must clear before its event can
  be queried: CUDA can report an unrecorded event ready. CC `HostCopyDone.error`
  is checked as well as readiness. A partial forward without a result, a missing
  completion event or a failed copy fails closed.
- Native transfer/scatter completion and GPU proofs must all be ready. The
  existing TP retirement reduction also agrees that either every rank queued a
  native KV release or none did. Deferred cancellation/finish frees run only
  after this vote. State, receiver, metadata and native release retire once.
- The queue reaper runs before native retraction/polling gates, preventing held
  releases from starving under memory pressure. Native `release_req` drains the
  relevant TP lease before KV backup/free/reset, preserving native retraction.
- A prepared overshoot still needs Remote attention after the preceding result
  finishes its Req. Such state is retained until voted retirement; changing its
  request row while leased is rejected. Manual state-mutating controls (including import/resume/reset/release) are gated
  while D overlap is enabled.
- Stable Local takes an empty-target fast path: no plugin event allocation,
  event query, CPU reduction or synchronization. Native SGL checks and fences
  remain. Startup also compares overlap policy across TP ranks.

## Executed checks

The local lightweight CPU harness source-loads native pool/cache/queue methods
and isolates CUDA/model imports. Final checks: **84 tests passed in 11.722 s**, including actual two-process
TP2/Gloo; **27 source hooks / 67 core API/state contracts** passed. Ruff, isort
and diff whitespace checks passed.

- Native PD overlap loop: the first Remote forward waits for the preceding Local
  result, consecutive Remote forwards serialize, and the second Local forward
  launches before the previous Local result is processed again.
- Native disabled-radix allocator release: GPU-pending / IO-completed state keeps
  both metadata and KV allocated; completion recovers all 32 slots exactly once.
  The native process queue's retraction early-return cannot starve this release.
- Delayed sampling, unrecorded-event semantics, failed CC copy, partial forward,
  drain timeout, native retraction backup/reset and finished-Req overshoot cases.
- Actual TP2/Gloo: rank-skewed delayed sampling and release requests hold both
  shards until agreement; a retraction waits for the slower rank. Local bursts
  execute without plugin collectives or proof queries, and a new admission
  revokes the Local policy. Rank-inconsistent overlap configuration fails startup.
- Regression counterexamples using the pre-change `e99cc5ce81` methods:
  old `Target.ready_to_finish` releases metadata/KV while the GPU proof is still
  pending; old state binding prunes a finished overshoot's Remote state. Both
  updated cases fail under these restored methods and pass with the changes.

In a full SGL development environment:

```bash
PYTHONPATH=python:plugins/remote_attn python test/registered/unit/plugins/test_remote_attn.py
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only --sglang-root .
```

No CUDA, Qwen3 PD, NCCL or performance test was run for this change. A proposed
upload to gpuhome was rejected by automatic approval review for missing explicit
source-transfer authorization; the user selected local-only testing and push.
No source package was uploaded. The temporary, unexecuted CUDA probe is excluded
from this commit. Event behavior in these CPU tests is a device-boundary stand-in,
not measured CUDA evidence. GPU correctness and throughput remain to be validated
with the [native launch/verification commands](../SINGLE-PAIR-ZH.md).
