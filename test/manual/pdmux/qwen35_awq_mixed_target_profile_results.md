# A100 single-shape Mixed target diagnosis (2026-10-26)

Follow-up: [kernel optimization results](qwen35_awq_mixed_optimization_results.md)
record the tested FP16/MoE candidates and their serving limitations.

The fixed numerical kernels dominate the P1576 regression. Joint execution
saves about 8 ms against a repaired Separate Graph, but both repaired paths
are much slower than native. Graph launch reduction does not offset this cost.

The original recorded SWE workload triggers one P1576 + D3x8 pairing on the
same A100 PCIe 40 GB / TP1 AWQ target and DFlash draft. Global chunk32768,
Prefill budget32768 and CPU overlap remain enabled; HiCache remains disabled.
The existing shape correctness gate passes before the diagnostic runs. Each
path starts from the same restored snapshot. Three timing samples rotate path
order, followed by one separate profiler replay per path. State restoration,
capture and validation are outside timing. Native Verify remains a CUDA Graph.

| Target path | Median GPU ms | Median wall ms |
| --- | ---: | ---: |
| Native Separate: eager Prefill + original Verify Graph | 123.38 | 123.43 |
| Repaired Separate Graph | 227.92 | 227.97 |
| Repaired Mixed Graph | 220.04 | 220.10 |

The combined repaired-kernel/capture change adds 104.54 ms versus native.
Joint batching recovers only 7.88 ms (3.46% of repaired Separate), leaving Mixed
96.66 ms slower than native. Repaired Separate also changes capture behavior,
so the first difference is not a pure numerical-kernel ablation. The native
first timing sample is 168.61 ms; the later samples are 122.90/123.38 ms. These
three-sample medians are diagnostic, not a confidence interval.

## Kernel evidence

| GPU kernel family, one profiled replay | Native ms | Repaired Separate ms | Mixed ms |
| --- | ---: | ---: | ---: |
| FP16 GEMMs, including LM head | 27.21 | 102.21 | 95.35 |
| Marlin MoE GEMMs | 46.38 | 80.22 | 79.22 |
| Attention/GDN/Conv and related kernels | 26.12 | 26.86 | 28.76 |
| Copy/fill/cat name matches | 2.28 | 2.17 | 2.42 |
| Other kernels | 13.86 | 12.85 | 11.75 |
| Sum of kernel durations | 115.85 | 224.31 | 217.50 |
| Kernel count | 2555 | 2384 | 1701 |

These are profiler activity sums, not a critical-path decomposition or the
unprofiled timing medians. Concurrent kernels may overlap. Name-based groups
are retained in the JSON and raw traces. The copy/fill/cat category includes
operations beyond Mixed packing; its small total does not support packing as
the main cause of this regression.

The fixed `_router` kernel name represents the replacement FP16 GEMM used for
eligible linear layers, not just the MoE router. It costs 100.74 ms across 460
calls in repaired Separate and 93.87 ms across 230 calls in Mixed. Remaining
native GEMMs, including LM head, add about 1.48 ms in either repaired path.
Thus Mixed roughly halves those launch counts but retains most of the work
in a large-M-inefficient implementation. Native FP16 GEMMs total 27.21 ms.

`MixedTargetGraph` captures under `stable_marlin` defaults: fixed
BM16/BN32/K64 FP16 GEMM, MoE M-block8/full-K, and disabled atomic reduction.
Native Marlin uses its adaptive path for Prefill. The trace confirms these
different kernel variants and their duration difference. Within Marlin,
the effects of M tile, full-K and reduction policy have not been isolated.

The prior source-level suspicion is therefore narrowed: prioritize the fixed
FP16 projection implementation, then Marlin MoE. Attention fusion, packing
cleanup and scheduler eligibility are secondary for this large-shape regression.
Simply widening admission would expose more requests to this slow path.

## Next implementation boundary

Evaluate the existing shape-based projection tile candidate first while
keeping the arithmetic contract and numerical gates. Then evaluate a larger
fixed MoE tile with full-K retained. A candidate must pass output/state checks
and native greedy parity for this shape before admission; the earlier
405-token mismatch remains unresolved. More relaxed gates would invalidate
the current correctness argument. Global chunk size should stay unchanged.

No new dataset or broad test suite was added. This profiling run is excluded
from throughput claims. Driver exit is zero and GPU compute processes are empty
after cleanup. The [result JSON](qwen35_awq_mixed_target_profile_results.json)
contains all nine timing samples, kernel groups, source hashes and the raw
trace archive location/digest. Diagnostic code is opt-in via
`MIXED_TARGET_PROFILE_DIR`, with `MIXED_TARGET_PROFILE_MIN_TOKENS=1000`; at most
one qualified shape is profiled per worker process.
