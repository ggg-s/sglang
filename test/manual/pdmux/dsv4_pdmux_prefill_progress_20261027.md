# DSV4 PDMux prefill progress and PCP memory

The reported four-arm run uses mem-fraction-static 0.8 and client concurrency
24. Its capacities are 3.58M (base), 2.66M (DSpark), 2.58M (opt) and 2.01M (PCP).
Thus the incremental PDMux arm loses about 3% capacity; the larger drops are
DSpark and PCP. Low observed KV usage with a queue can reflect a prefill-service
bottleneck, request-slot/admission limits or sampling bias. It is not evidence
that the token allocator alone is blocking admission.

## Scheduling review and fixes

`update_split_prefill_batch` owns one active split **batch/chunk**. Its guard
protects that batch's live model state and attention metadata. After the last
layer completes on every rank, `_merge_completed_prefill_batch` stashes a middle
chunk and removes it from decode; `_merge_finished_prefill_batch` releases the
split slot. The batch can include multiple requests. The shared scheduler also
has one `chunked_req` continuation, so long-request chunk serialization is not
unique to PDMux.

Admission still uses the normal `PrefillAdder`: request slots, max-running and
prefill request limits, available/evictable KV, reservations for output growth,
chunk/input budgets and cache readiness. Free full-KV tokens do not bypass
those independent limits. The reported client concurrency does not specify the
server's max-running-requests or prefill limits.

Two independently reproducible defects were fixed:

1. Every new/completed chunk requested a stream-group selection, and both
   layer-split and standard loops drained P/D streams **before** checking
   whether the physical group changed. Repeated selections of the same group
   now preserve queued work and the backend binding. Actual changes still
   drain both old streams before rebinding. Existing formation/merge events
   retain the dependencies between lanes.
2. Default logging/metrics chose `attn_tp_rank == 0` without checking CP rank.
   CP=TP makes every CP rank an attention-TP leader, duplicating logical-batch
   reports. Only CP rank zero now reports by default, per DP worker. Explicit
   all-scheduler metrics remain available. A count of merged PCP log lines
   cannot be interpreted as independent batch work or a speedup.

The new two-process Gloo regression fails against the old loop's redundant
drains, and the reporter regression fails against the old CP selection. New
CUDA coverage queues prefill tensor work and decode graph replays across
unchanged selections, then verifies that an actual transition drains both old
lanes before the backend changes.

## Interpreting the provisional 32k split budget

With a configured budget of 32768 token-layers and a 16384-token global chunk,
the active-decode interval is **2 layers**. A 43-layer example needs 22 prefill
submissions for one chunk while decode remains active. With no decode work the
implementation finishes the remaining layers in one call. A configured
`split_forward_max_layers` may reduce the active-decode interval further.

PCP does not divide this global token count by CP size for the layer budget:
the plain-TP MoE still processes the gathered global tokens. Dividing by eight
would silently allow a much larger compute interval and can regress decode
latency. The new budget regression also checks the equivalent DP8 gathered
count and the no-decode fast path.

The unneeded chunk-boundary drains are removed; the budget remains explicit.
No full-model speedup is claimed without rerunning the deployment workload.
Multiple independently active prefill batches would require separate backend
metadata and activation ownership, not removal of the current guard.

## Why PCP does not increase this implementation's KV capacity

This DSV4 PCP path shards token computation, then
`MQALayer._forward_prepare` uses `cp_materialize_global_token_order` to gather
KV before cache writes. `DSV4PoolConfigurator` prices all PP-local target layers;
`_build_dsv4_kv_pool` passes the PP layer range and has no CP layer-ownership
split. Its SWA, C4/C128 and compressor state are therefore not reduced to 1/CP
per rank by enabling PCP. This is separate from GLM/DSA cache-layer splitting.

Ordinary TP already replicates the model's single KV head across query-head
partitions. PCP keeps that cache ownership while changing attention TP to one
for prefill. `wq_b`, `wo_a` and `wo_b` become full-width per-rank weights instead
of ordinary TP shards. `--enable-cp-decode-attn-tp` selects sharded decode views,
but prefill still needs the full source weights; row-parallel decode views also
retain a contiguous shard. Thus additional weights can reduce the free bytes
used for KV sizing without any compensating 1/CP KV reduction.

True KV-capacity scaling would need a separate DSV4 KV ownership scheme,
attention reads/reductions and compressed-state/indexer routing, including
DSpark and HiCache restore. This patch does not introduce that architecture.

For exact capacity attribution compare the startup `DSV4 memory calculation`
fields (`available_bytes`, `bytes_per_full_token`, C128/SWA/C4 fixed bytes) and
weight memory across the four arms. The same mem-fraction does not imply the
same residual bytes after target/draft weights and fixed state.

## Deployment validation

Retain resolved server settings, YAML, source commit and single-CP-rank logs.
Use `--log-level debug` to collect existing `PDMux split submit` records with
tokens, layers, start layer and decode batch size. Compare max-running-requests,
prefill-max-requests, chunk size, split budget, maximum split layers and SM caps.
Use time-weighted queue/KV occupancy and separate queue wait from GPU prefill
service. Rerun the same 400-request/c=24 workload after the fix; the 4060 checks
validate ordering and progress, not DSV4 end-to-end TTFT or throughput.

## Validation results

Tested source: `703d479a282324f9cfaa06339f42469982473081`, pulled over SSH on
the RTX 4060 Laptop workstation (PyTorch 2.13.0+cu130). The complete registered
`unit/multiplex` directory plus `test_dsv4_pcp_resolution.py` and
`observability/test_forward_pass_metrics.py` passed: **234 tests and 298
subtests**, no failures. `test_pdmux_tp_overlap_cuda.py` passed both the native
singleton-NCCL case and the new CUDA graph/group-transition case; the distinct
two-GPU NCCL case skipped. Both pre-fix regression checks failed with the
expected assertions against the parent source without changing workspace files.

Workstation evidence is under `/workspace/cp-decode-review-20261027/` in
`prefill-progress-unit.log` and `prefill-progress-cuda.log`. The source changes
do not implement additional active prefill batches, alter the configured split
budget or shard DSV4 KV storage.
