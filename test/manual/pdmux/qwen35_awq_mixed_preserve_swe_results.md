# A100 SWE replay with preserved serving settings (2026-10-26)

The private Mixed experiment now retains the original chunk32768 and CPU
overlap. The corrected warm throughput is 305.44 output tok/s. The small
difference from the preceding controls does **not** establish a stable benefit:
there is one measured pass, controls were reused, and only two joint replays.
The first workload warmup takes 139.83 seconds including online admission.

| Path | Output tok/s | Versus DFlash | Wall seconds | TTFT p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Ordinary DFlash, preceding control | 295.10 | baseline | 7.808 | 139 / 1510 |
| PDMux, preceding control | 297.44 | +0.79% | 7.746 | 248 / 1752 |
| Corrected Mixed | 305.44 | +3.50% | 7.543 | 139 / 1541 |

Mixed is +2.69% against PDMux in these single observations. All actual
server-info settings match for chunked_prefill_size=32768,
max_prefill_tokens=32768, max_total_tokens=65536, max_running_requests=4,
disable_overlap_schedule=false, enable_hierarchical_cache=false and TP1.
Target is Qwen3.5-35B-A3B-AWQ with DFlash block8 on A100 PCIe 40 GB.
PDMux retains its P88 configuration and layer split policy.

The same six recorded SWE trajectories with three whole assistant prefixes
each are replayed at concurrency four, producing exactly 2304 tokens across
18 requests. Each request generates 128 tokens. Prompts are 6909–9773 tokens;
GPU prefix caching is enabled. Mixed and DFlash both report 95872 cached tokens;
PDMux reports 98688. This is frozen-history inference replay, without tools or
SWE task scoring. Baselines are retained in
[the preceding result](qwen35_awq_mixed_swe_quick_results.json).

## Scheduling changes

All driver groups inherit one common serving configuration. The private hook
uses the actual native Prefill chunk length, with an opt-in capacity limit up
to 32768. It never splits a request to fit a 128-token Graph. Unsupported
pairings, rejected shapes, insufficient workspace and unprofitable candidates
use the native path. HiCache/PDMux pairing remains unsupported.

CPU overlap keeps the native result queue depth and processing order. The
native Draft, acceptance/commit and future-map machinery remains active; only
an admitted Target forward is replaced. Captured output ownership and decode
snapshot lifetime are retained until the paired results are processed.

## Actual CUDA Graph use

| Measured stage | Graph hits / calls |
| --- | ---: |
| Draft | 292 / 292 |
| Independent native Target Verify | 290 / 290 |
| Joint Prefill + full Target Verify | 2 / 2 |
| New admission captures | 0 |

Joint execution covers 2/14 Prefill batches (14.3%) and 964/45333 Prefill tokens
(2.13%). The two joint replays replace two independent Verify calls. Draft
and native Verify hits come from their actual forward-result flags; joint
counts increment after successful replay. Warmup retains three admitted graphs.

## Admission decisions and remaining costs

| Actual Prefill tokens | Decision | Native / joint wall ms |
| ---: | --- | ---: |
| 430 | admitted | 116.29 / 99.90 |
| 1576 | native fallback, no net gain | 123.02 / 259.33 |
| 274 | admitted | 106.87 / 86.45 |
| 405 | native fallback, greedy decision mismatch | 122.69 / — |
| 534 | admitted | 125.90 / 115.82 |
| 6909 | native fallback, workspace budget | — |
| 2451 | native fallback, no net gain | 170.73 / 347.15 |
| 343 | native fallback, workspace budget | — |

New shapes compare joint outputs and state writes with an independent repaired
Separate Graph and check exact greedy decisions against native execution before
admission. Candidate wall time must beat native Prefill plus the original Verify
Graph by 2%. Successful observed shapes are 274, 430 and 534; the capacity limit
does not establish correctness or benefit for every larger shape.

The 139.83-second warmup includes snapshots, validation, calibration and twelve
admission Graph captures, including discarded controls/candidates. Its throughput
is only 16.48 tok/s. The 305.44 tok/s measurement excludes this initial cost.
Online calibration is therefore still too expensive for a cold request path;
persisted calibration or bounded startup preparation is needed before treating
this as a deployable optimization. Memory preflight is conservative because it
does not reclaim allocator reservations. The 405-token mismatch remains
unresolved and is blocked by the gate, so overall quality equivalence is not
established.

Four focused CPU checks cover configuration preservation, overlap result order,
output lifetime and exception restoration. The GPU run completed with driver
exit zero and no remaining compute processes. No additional datasets or broad
test suites were added. This remains a private TP1/greedy text experiment.

[Result JSON](qwen35_awq_mixed_preserve_swe_results.json) records the complete
counters, actual configuration, source hashes, decisions, raw archive location
and digest. The earlier chunk128 comparison is historical and confounded by
changed chunking and overlap settings; use this report for the corrected setup.

## Source-level bottleneck analysis

Follow-up: the [single-shape GPU profile](qwen35_awq_mixed_target_profile_results.md)
now measures the missing repaired Separate control and attributes the dominant
P1576 costs to the replacement FP16 GEMMs and Marlin MoE. The following records
the source-level reasoning before that diagnostic.

The large-shape slowdown is already present in GPU-event timing: P1576 costs
122.96 ms native versus 259.23 ms joint; P2451 costs 170.60 versus 347.06 ms.
Their corresponding wall/event gaps are only about 0.06–0.10 ms. This strongly
points to the GPU execution path rather than host scheduling as the dominant
cause of those particular regressions. Event timing alone does not attribute
individual kernels.

`MixedTargetGraph` captures under `stable_marlin(model)` with its default
contract. That contract replaces eligible unquantized FP16 linear methods with
a fixed BM16/BN32/K64 Triton GEMM, forces MoE M-block8, enables full-K reduction,
and disables Marlin atomic reduction. Native execution keeps its adaptive
kernel path. Thus this comparison combines batching with a different numerical
implementation. The fixed contract is the leading source-level suspect for
large-M inefficiency; FP16 projection versus MoE contributions are not measured
separately. The existing projection-tile candidate is not enabled in this run.

The router still invokes Prefill and Verify attention/GDN backends separately,
then zero-initializes and copies their outputs into a packed buffer at every
layer. Shared dense/MoE forwards can save work, but attention/state operations
are not a single fused P/V kernel. Packing copies scale with actual Prefill
length. There is no extra Prefill-capacity padding in the exact-length run.

Historical offline full-target controls also separate an important confounder:
for one P32 fixture the GPU medians are native Separate 101.24 ms, repaired
Separate Graph 34.21 ms, repaired Mixed Graph 24.97 ms. The 76.27 ms total gap
cannot all be credited to joint math: changing to captured repaired Separate
already accounts for 67.03 ms. At P512, the comparable values are
109.90 / 87.19 / 72.55 ms. These are different fixtures from SWE and are evidence
of the distinction, not an extrapolated serving speedup. Large-shape admission
captures a repaired Separate control but currently does not time it, leaving
numerical-repair cost versus batching benefit unresolved at P1576/P2451.

Coverage is constrained by exact P1+D3 admission, per-length Graph keys, four
retained graphs, rejected numerical/unprofitable shapes and workspace budget.
Measured counters identify five unpaired Prefill batches, three unsupported,
two previously rejected, one attempted fallback and two successful pairings.
Counters are not a mutually exclusive partition because a mixed-shape attempt
can fail before reaching the target hook. These filters explain why kernel
savings apply to only a small part of the workload.

The two admitted measured lengths (430 and 534) save about 26.46 ms in their
warmup calibration samples, roughly 0.35% of a 7.54-second workload. These are
not measured-phase per-call timings and cannot establish a causal speedup;
they show why the observed 264 ms difference from the reused DFlash control
cannot safely be assigned entirely to joint execution.

Cold admission invokes an offline-style `StateSnapshot`: it copies full Mamba
state and request mappings plus selected target/draft KV data to CPU, performs
repeated restores, and captures two complete Graphs per evaluated shape.
This explains concrete sources of the 139.83-second warmup cost; their individual
time shares were not profiled. It is separate from the steady large-shape
kernel regression.

The next minimal diagnostic should time the already-captured repaired Separate
control for one rejected large shape, then profile only that native/control/joint
triple to separate fixed FP16 GEMM, Marlin MoE, attention and packing time. This
requires no new dataset or broad correctness suite. Optimizing the dominant
kernel while retaining numerical gates takes priority over widening admission
or changing the global chunk size.
