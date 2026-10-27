# PDMux allocator index lifetime and HiCache pump ordering

The reported SWE failures used commit `657c717729`, 4219 requests and a
16384-token prefill chunk. The first failure surfaced in GDN/FLA preparation
after roughly 96 requests; the second surfaced in the guarded native KV store
after 331. The original logs are unavailable. These observations do not prove
that the earlier NCCL broadcast deadlock has been fixed or reproduced.

## Reproduced defect

PDMux shares allocators between the prefill and decode CUDA streams. Their
free-list tensors can be created on one stream, consumed on another, then
replaced by concatenation or sorting on the original stream. The allocator
object remains alive, but its previous tensor storage can return to the CUDA
caching allocator before the other stream's queued read finishes. Batch input
keep-alive does not cover these allocator-internal tensors.

On a single RTX 4060, the production paged allocator reproduced corrupt output
slots when the original stream recycled the old list. The synthetic reproducer
deliberately fills reclaimed memory with page ID 1000000; at page size 16 this
produces slots around 16000000 in a pool with fewer than 32784 rows. Those
numbers describe the reproducer, not values recovered from the SWE failures.

The fix records the consuming CUDA stream for paged/token free lists, Mamba
slot lists and deferred free-group inputs before their reads. This protects
storage lifetime; the scheduler still supplies data-ready dependencies.

The in-flight layer-split HiCache pump now executes on the decode stream,
following previous decode writes and preceding this iteration's allocation.
Previously it could submit allocator mutations on the prefill stream without
a decode-to-prefill dependency. Batch-formation pumping retains its existing
prefill stream and event ordering. Pump counts and rank-uniform cadence are
unchanged.

## Small regression checks

```bash
python -m pytest -v \
  test/manual/pdmux/test_pdmux_allocator_lifetime_cuda.py \
  test/registered/unit/multiplex/test_pdmux_hicache_events.py \
  test/registered/unit/multiplex/test_pdmux_tp_overlap.py
```

The CUDA test uses real paged allocation kernels, token/Mamba allocators,
16384-token chunks, a green-context stream and a plain stream. Both stream
directions and immediate/sorted releases are covered. It compares raw indices
before launching the real guarded KV store, so a broken version fails without
poisoning the CUDA context. It also checks state-slot advanced indexing. It
does not execute a full GDN model or emulate two-GPU NCCL on one GPU.

All 16 CUDA cases failed with the allocator files from `657c717729` and passed
with the fix. The pump ordering test also rejects the previous placement. See
`pdmux_allocator_lifetime_validation_20261027.json` for final regression results.

## Deployment acceptance

Use two independent fresh server launches with the original frozen 4219 SWE
requests, request order, concurrency, model/draft paths, TP topology and HiCache
configuration. Keep chunk size 16384 and CPU overlap enabled. Retain the full
server log from startup through completion, including the first CUDA error if
one occurs. Both runs must complete all requests without CUDA asserts,
exceptions, watchdog termination or collective timeout before declaring this
load stable. Recheck output correctness and performance against the control.

Passing single-GPU regression does not establish that either production crash
or the original roughly-request-1030 NCCL deadlock is resolved. Until the full
deployment workload passes, the original deadlock remains unverified.
