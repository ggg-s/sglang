# A100 quick live Mixed scheduling A/B (2026-10-26)

The private Mixed scheduler completed real streaming HTTP requests with higher
output throughput on this synthetic workload. This is one measured trial per
group after one excluded warmup, not a production throughput claim.

| Metric | Ordinary DFlash | Mixed | Change |
| --- | ---: | ---: | ---: |
| Mixed-workload output tokens/s | 375.06 | 733.68 | +95.61% |
| Mixed-workload elapsed time | 4.437 s | 2.268 s | -48.88% |
| Long-prefill request mean TTFT | 1598.44 ms | 744.71 ms | -53.41% |
| Decode-only output tokens/s | 1199.02 | 1143.57 | -4.62% |

The fixture starts three 64-token code prompts generating 512 tokens each.
After all three emit their first tokens, it submits two sequential 2048-token
prefills, each generating 64 tokens. Peak concurrency is four. Each measured
mixed-workload round completes five requests and 1664 output tokens. Salts and
between-round cache flushes prevent prompt reuse. All eight measured requests
across mixed-workload and decode-only phases have identical output token IDs
between the two groups.

Both groups use A100 PCIe 40 GB, TP1, Qwen3.5-35B-A3B-AWQ in FP16 AWQ-Marlin,
the unquantized DFlash draft, block8, chunk128, 65536 KV tokens, 24 Mamba slots,
and HiCache disabled. Ordinary DFlash retains CPU overlap; the initial private
Mixed scheduling hook runs without CPU overlap. That difference also limits
interpretation of the single decode-only measurement.

The Mixed run records 64 joint Graph executions over warmup plus measurement,
one capture and zero graph-bank fallbacks. All 134 Draft calls hit their Graph;
all 70 remaining native Verify calls hit the original Verify Graph. The
ordinary DFlash group records 138/138 Draft and 138/138 Verify Graph hits.
The hook admits P1+D3x8 at P32/P128, routes acceptance/commit through the
existing DFlash worker, and preserves invocation-owned output objects. This
run exercises P128, fixed legacy repaired arithmetic, and fixed admission; it
does not exercise the adaptive six-candidate operator selector.

Before the live run, the current isolated sources passed all 18 stable Eager
and 18 Mixed Graph cases, including natural/rejection lanes and graph reuse.
The added Separate timing control initially shared Mixed metadata and crashed
on reuse. Giving it independent inputs/backends and retaining its capture-time
buffers resolved that failure in the complete rerun. The first failed run's
timings are excluded from these results.

The repeated-code fixture has high draft acceptance. Real SWE/agent traffic,
PDMux comparison, HiCache, asynchronous Mixed scheduling and repeated trials
remain unmeasured here. The -4.62% decode-only result is an observation from
one trial, not an established regression. GPU compute processes were empty
after both private servers exited.

The compact [result JSON](qwen35_awq_mixed_live_quick_results.json) records the
fixture digest, source hashes, runtime settings and raw timing/counter values.
Raw reports, output IDs, server logs and the passing full-model matrix are in
the local [evidence archive](/Users/guoqing/.codex/visualizations/2026/10/26/01a175cf-dbdf-7ce0-b26f-3707e8e44e1c/pdmux-a100-20261026/mixed-live-quick-evidence-20261026.tar.gz);
its SHA256 is retained in the result JSON.
