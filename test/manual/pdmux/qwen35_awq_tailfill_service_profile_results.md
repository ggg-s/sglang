# A100 live tail-fill service profile — 2026-10-26

The native tail-fill mode changes scheduling but does not fuse Target computation.
Every one of its 15 paired rounds finishes Verify before starting Prefill.
This directly explains why counting successful pairs has not established a
throughput gain: no Target weight read or GEMM is shared by the current path.
A token-budget remainder is not automatically a GPU execution slot.

## Setup

A100 PCIe 40 GB, TP1, Qwen3.5-35B-A3B AWQ Target and DFlash draft. Both actual
server-info files confirm chunked_prefill_size=4096, max_prefill_tokens=32768,
CPU overlap enabled, HiCache disabled, max_running_requests=4, KV budget65536.
One warmup and one measured recorded SWE replay per mode: six trajectories,
three assistant prefixes each, concurrency4, 128 generated tokens/request.
Both measured passes have 18 requests,2304 output tokens,26197 actual Prefill
tokens and115008 cached tokens. This is replay, not SWE task-success evaluation.

Only the measured pass is profiled. No admission capture, numerical gate,
state re-execution or additional correctness workload runs. Driver exits zero;
GPU compute-process list is empty after cleanup.

## Direct observations

| Observation | DFlash | Native tail fill |
| --- | ---: | ---: |
| Draft Graph hits/calls | 289/289 | 299/299 |
| Verify Graph hits/calls | 289/289 | 299/299 |
| CUDA Graph launches in trace | 578 | 598 |
| Prefill batches | 16 | 17 |
| Paired rounds | 0 | 15 |
| Joint Mixed Graph replay | 0 | 0 |
| Prefill kernel activity sum, ms | 1634.73 | 1635.88 |
| Draft kernel activity sum, ms | 585.41 | 603.08 |
| Verify kernel activity sum, ms | 3923.63 | 4043.90 |
| Other kernel activity sum, ms | 137.74 | 141.07 |
| Union of all kernel intervals, ms | 5737.01 | 5854.84 |
| First-to-last kernel window, ms | 8106.64 | 8983.19 |
| No-kernel time within that window, ms | 2369.63 | 3128.35 |

Kernel activity sums include concurrent streams; they are not additive wall-time
components. CPU/driver launch correlation, including CUDA Graph launch
correlation, assigns all kernels to stages; no kernel remains unmatched.

All 15 paired Verify stages precede their paired Prefill. The gap from the last
Verify kernel to the first Prefill kernel is0.294–0.622ms. There is no P/V
kernel overlap. Source inspection agrees: native mode calls RUN_BATCH(decode),
then _finish_pair calls RUN_BATCH(prefill), retaining two native forwards.
The existing independent Verify Graph remains useful, but does not itself
combine Verify tokens with Prefill GEMMs.

The paired15/299 Verify rounds represent5.02% of the rounds. Their kernel
activity sum is197.32ms and their per-round union of busy intervals totals
164.04ms; paired Draft activity adds29.38ms. Even hiding the entire paired
Verify busy time would cover only1.83% of this instrumented GPU window. This
is an opportunity-size estimate for this trace, not a rigorous speedup ceiling
for all fusion implementations or a prediction of unprofiled throughput.
Most Verify work remains outside Prefill pairing opportunities at concurrency4.

## Limits and next change

Profiling changes host overhead and arrival/batching timing. Instrumented wall
measurements are8.171s and9.059s; do not use them as throughput A/B evidence.
The extra759ms of no-kernel time cannot be assigned to the scheduler alone:
it includes launch gaps, waits and profiler overhead. More Verify calls and a
different Prefill batch count also confound direct phase totals. Native Verify
activity is nevertheless the largest GPU component in both traces, and neither
its launch count nor total activity falls in the paired mode.

The preceding unprofiled single-pass results remain DFlash340.22tok/s,
PDMux319.86tok/s, and native tail fill336.24tok/s; they show no demonstrated gain.

The next implementation work is to make the admitted tail path actually share
Target linear/MoE computation between P and full Verify blocks, while retaining
Draft Graph and the original Verify Graph as fallback. The earlier variable-shape
joint paths failed native state parity, so simply enabling them is not justified.
Localize the first state divergence on an already captured failing shape and
fix that path before another throughput comparison. Do not compensate by
shrinking global chunks or changing max_prefill_tokens. Once real joint replay
exists, these same traces suggest measuring pairing coverage before expecting
large gains at concurrency4; no new dataset is required to establish this defect.

## Evidence

[Result JSON](qwen35_awq_tailfill_service_profile_results.json) records counters,
paired timelines, kernel names, settings, source hashes and archive digests.
Local artifacts:
`/Users/guoqing/.codex/visualizations/2026/10/26/01a175cf-dbdf-7ce0-b26f-3707e8e44e1c/pdmux-service-profile-a100-20261026/`.
`service-profile-traces.tar.gz` contains both full Chrome traces;
`service-profile-evidence.tar.gz` contains logs, summaries and counters.
`analyze_service_trace.py` reproduces the correlation-based summaries.

Opt-in tracing is enabled by run_live_mixed_target.py --profile-service.
The client starts tracing after warmup/cache flush and waits for export after
the measured requests. Normal runs do not enable the profiler.
