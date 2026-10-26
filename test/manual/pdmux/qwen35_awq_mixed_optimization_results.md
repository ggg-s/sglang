# A100 Mixed kernel optimization (2026-10-26)

The P1576 + D3x8 target stage now takes 119.47 ms instead of 220.04 ms in the
earlier diagnostic, a 45.7% reduction. This is a single-shape kernel result,
not a demonstrated end-to-end serving improvement. The final SWE pass executes
only one joint Graph and measures 282.09 output tok/s.

## Candidate progression

| Mixed implementation | P1576 GPU median ms | Fixed FP16 GEMM trace ms | Marlin MoE trace ms |
| --- | ---: | ---: | ---: |
| Legacy fixed FP16 tile, MoE M8 | 220.04 | 93.87 | 79.22 |
| Projection-dependent FP16 tile, MoE M8 | 176.49 | 49.28 | 79.63 |
| Projection-dependent FP16 tile, MoE M32 | 134.36 | 48.43 | 37.88 |
| Fixed FP16 BM64/BN128, 8 warps, MoE M32 | 119.47 | 34.12 | 38.35 |

Each row uses the same recorded SWE workload and actual P1576 geometry.
Rows are separate runs with three rotated timing samples per target path,
followed by one profiler replay. Kernel trace durations are activity sums;
they are not the unprofiled timing medians. Different runs can have different
request timing and state context, so the progression is diagnostic evidence
rather than a statistically controlled speedup estimate.

The final run's native Separate samples are 178.37, 127.37 and 123.45 ms,
including a first-call outlier. Its median is 127.37 ms; optimized Mixed samples
are 119.47, 119.34 and 119.78 ms. A separate admission sample compares native
124.08 ms against candidate 121.35 ms. The candidate clears the existing 2%
wall-time admission threshold, but the small native-relative advantage needs
confirmation. The much larger improvement over the legacy Mixed implementation
is consistent with the GEMM and MoE trace reductions.

## Implementation and correctness boundary

`MixedTargetGraph` accepts opt-in `linear_tile` and `moe_block_size` arguments;
legacy defaults are retained. The graph captures the selected fixed arithmetic
path. Full-K Marlin and the K64 FP32-accumulating FP16 GEMM remain enabled.
The final candidate uses FP16 BM64/BN128 with eight warps and MoE M-block32.

Private live admission compares candidate state writes, logits and hidden
states against the legacy repaired Separate Graph, and checks exact greedy
decisions against native execution. Only candidates passing those checks and
the performance threshold are retained. The P1576 final candidate passes.
This does not establish equivalence for every shape, context or generated
sequence; shape-specific numerical rejection remains necessary.

The successful bounded configuration is expressed as string-valued fields in
the private driver's base-config JSON:

```json
{
  "LIVE_MIXED_LINEAR_TILE": "wide",
  "LIVE_MIXED_MOE_BLOCK": "32",
  "LIVE_MIXED_MIN_TOKENS": "1576",
  "LIVE_MIXED_CAPACITY_LIMIT": "1576",
  "LIVE_MIXED_GRAPH_LIMIT": "1"
}
```

These limits filter eligibility; they do not split or resize native Prefill.
The other serving settings remain chunk32768, max_prefill32768, KV budget65536,
CPU overlap enabled, HiCache disabled, TP1 and concurrency four. Other Prefill
sizes use the original serving path. `LIVE_MIXED_LINEAR_TILE` accepts `legacy`,
`projection` or `wide`. The older `LIVE_MIXED_PROJECTION_TILE=1` remains supported
when the newer choice is absent. Profiling is separately opt-in through
`MIXED_TARGET_PROFILE_DIR`; omit it for a performance run.

## Serving observations and memory limit

The final bounded run completes 18 requests and 2304 output tokens in 8.168 s
(282.09 tok/s). It reports Draft Graph 297/297, independent Verify Graph 296/296,
and one joint Graph replay. There are no measured captures. Joint execution
covers 1576/45333 Prefill tokens. This is below preceding native controls and
does not demonstrate an end-to-end gain; the run follows diagnostic profiling
and uses restricted admission. Its 43.10-second warmup includes capture,
validation and profiling and is excluded from the measured throughput.

An earlier broad MoE32 run completed warmup but failed during the first measured
native Prefill batch of 28532 tokens with CUDA OOM. It retained several small
candidate Graphs before this larger native workspace allocation. The allocation
failure cannot be assigned solely to the MoE tile; retained graphs, backend
buffers and native batch size compete for memory. That run has no valid
measured throughput. Its warmup also rejects P430, P274 and P2451 on native
greedy mismatch. The single-shape/single-Graph retry completes without OOM;
this is diagnostic containment, not a general production memory-budget fix.

The remaining serving work is to budget retained Graph/backend memory against
the original maximum Prefill workspace, move expensive first-shape validation
off the request path, and increase useful pairing coverage only where both
numerical and performance gates pass. The global chunk size should stay intact.

No new dataset or broad test suite was added. Python syntax and whitespace
checks pass. The final driver exits zero and GPU compute processes are empty.
The [result JSON](qwen35_awq_mixed_optimization_results.json) retains timing
samples, Graph counters, numerical decisions, source hashes and all three raw
archives, including the failed broad run. The candidate remains opt-in.
