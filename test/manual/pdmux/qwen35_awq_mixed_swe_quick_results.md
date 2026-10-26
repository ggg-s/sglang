# A100 recorded SWE quick A/B (2026-10-26)

The current private Mixed implementation does **not** retain the synthetic
workload's throughput benefit on the recorded SWE workload.

| Path | Output tok/s | Versus ordinary DFlash | TTFT p50 | TTFT p95 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary DFlash | 295.10 | baseline | 139 ms | 1510 ms |
| PDMux | 297.44 | +0.79% | 248 ms | 1752 ms |
| Mixed | 45.20 | -84.68% | 6211 ms | 12908 ms |

Each group performs one complete workload warmup, a cache flush, and one
measured pass: six recorded SWE-rebench/OpenHands trajectories, three assistant
prefixes each, concurrency four, and exactly 128 generated tokens per request.
All 18 requests complete, producing 2304 output tokens per group. Input lengths
range from 6909 to 9773 tokens. Recorded histories are replayed as frozen
prompts; no tools are executed and no SWE task success score is computed.
The workload SHA256 is
`687c751bd2ad026771bc31ee737e19a02fc17605bf9d5c764a58eda1a535b123`.

## CUDA Graph observations, measured pass only

| Path | Draft Graph hits/calls | Native Verify Graph hits/calls | Joint Graph replays |
| --- | ---: | ---: | ---: |
| Ordinary DFlash | 290/290 | 290/290 | 0 |
| PDMux | 339/339 | 339/339 | 0 |
| Mixed | 341/341 | 245/245 | 96 |

The Mixed target Graph is used: 96 of 475 prefill batches (20.2%) execute
jointly. The other 379 prefill batches use the small-chunk native path. There
are no graph-bank fallback events. However, the full warmup happens to trigger
zero joint executions, so the measured pass includes one first Graph capture.
Draft/Verify hit counts come from their actual forward-result flags; the joint
counter increments after `MixedTargetGraph.replay`. These observations establish
that decode Graphs remain active, rather than attributing the slowdown to eager
Verify fallback.

## Configuration and interpretation

All groups use the same A100 PCIe 40 GB, TP1 FP16 AWQ-Marlin target, unquantized
DFlash draft, block8, 65536 KV tokens, 24 Mamba slots and HiCache disabled. GPU
radix prefix caching remains enabled. The baselines retain their original
chunk32768 and CPU overlap; PDMux uses P88 with a 32768 token-layer budget and
16-layer split maximum. The current Mixed hook requires chunk128 and disables
CPU overlap. This measures the complete configurations, not an isolated kernel
speed difference.

Native DFlash reuses 95872 cached tokens, PDMux 98688, and Mixed 81226. Mixed
therefore also executes more uncached prompt work. PDMux's prefill counters
count layer-split submissions, so its reported prefill-token counter must not
be treated as unique input-token volume.

The immediate engineering priorities suggested by these counters are large
chunk fallback when no eligible D3 verify is available, preserving CPU overlap,
and maintaining prefix reuse. The existing P1+D3x8 admission is too restrictive
to justify forcing every prefill through chunk128 on this workload. No extra
datasets, correctness suites, profiler runs or repeated trials were added.

This single measured pass has no confidence interval. PDMux and Mixed each
match ordinary DFlash's complete output token IDs on 11/18 requests; PDMux and
Mixed also match each other on 11/18. Every request produces its required 128
tokens. These comparisons do not establish task-quality equivalence or identify
the cause of output differences under different scheduling/numerical paths.

The [result JSON](qwen35_awq_mixed_swe_quick_results.json) includes configuration,
timings, acceptance statistics, Graph counters, source hashes and the local raw
archive path/digest. GPU compute processes were empty after all three servers
exited.
