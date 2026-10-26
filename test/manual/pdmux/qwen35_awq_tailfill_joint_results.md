# A100 tail-fill joint execution follow-up — 2026-10-26

The experimental service path now executes a real shared Target Graph for an
admitted P2184 + D2x8 layout. Draft Graph and independent Verify Graph fallback
remain enabled. This is a bounded prototype, not general production admission.

Both final runs use A100 PCIe40GB, TP1, Qwen3.5-35B-A3B AWQ with DFlash,
chunked_prefill_size4096, max_prefill_tokens32768, CPU overlap enabled,
HiCache disabled and max_running_requests4. The same recorded SWE replay has
18 requests,128 output tokens each,26197 actual Prefill tokens and115008 cached
tokens. Warmup/capture/qualification precede the measured pass. No profiler is
enabled in this final comparison.

| Final measured pass | DFlash | Admitted joint path |
| --- | ---: | ---: |
| Output tok/s | 339.2315 | 346.0708 |
| Wall seconds | 6.79182 | 6.65760 |
| Draft Graph hits/calls | 293/293 | 293/293 |
| Independent Verify Graph hits/calls | 293/293 | 292/292 |
| Joint Graph replays | 0 | 1 |
| Measured captures | 0 | 0 |
| Prefill batches | 17 | 16 |

The single-pass throughput difference is +2.02%, not a demonstrated stable gain.
Only one joint call occurs. Its qualification timing falls from147.795ms to
141.457ms (4.29%), about6.34ms; this cannot alone explain the134ms wall difference.
Batch formation and ordinary run variability also differ. All requests finish;
driver exit is zero and no GPU compute process remains after cleanup.

## Numerical diagnosis and qualification

For the initial P3866 + D1x8 failure, early projection/attention outputs agree,
then discrepancies first appear in MoE/MLP and grow through later layers.
Restoring the same snapshot and repeating native execution itself produces
1122551 Conv elements outside rtol0.02/atol0.002; native recapture produces
1129389. Native Marlin enables atomic reduction. These observations implicate
native numerical nondeterminism rather than establish an initial state-routing
bug. They do not rule out other bugs in untested layouts.

The default native-state gate remains unchanged in meaning. A separate explicit
LIVE_TAILFILL_REFERENCE=stable_separate option compares full mutable state,
logits and hidden outputs against Separate using fixed arithmetic, and still
requires exact native greedy decisions and a native-relative speed advantage.
It does not assert full state equality with native arithmetic. Qualification is
snapshot/layout-specific; it does not establish arbitrary-context equivalence.
The final Separate reference uses FP16 BM64/BN128,8 warps,2 stages and MoE M32.
The candidate uses BM128/BN128,4 warps,3 stages with the same K64/FP32 accumulation
scheme and MoE M32. A prior pipeline attempt failed native greedy and was
rejected; that failure is retained in the remote evidence archive.

## Optimization and limits

The P3866 fixed FP16 kernel activity falls from81.461ms to53.872ms. Its full
joint Target median falls from259.678ms to234.05ms in separate diagnostic runs,
but remains slower than native at about230ms and is rejected. These profiles
are not serving throughput measurements. P2184 passes state/output/native
selection checks and the2% performance gate in both the subsequent service
run and the final unprofiled A/B. Its final admission uses three alternating
native/candidate timing samples and medians to reduce cold-start bias.

Final bounded launch settings in addition to normal model/config arguments:

```json
{
  "LIVE_TAILFILL_REFERENCE": "stable_separate",
  "LIVE_MIXED_LINEAR_TILE": "pipeline",
  "LIVE_MIXED_MOE_BLOCK": "32",
  "LIVE_MIXED_MIN_TOKENS": "2184",
  "LIVE_MIXED_CAPACITY_LIMIT": "2184",
  "LIVE_MIXED_GRAPH_LIMIT": "1"
}
```

These are experimental admission filters, not chunk-size settings. Native
chunk4096 is preserved. Other layouts retain the native path. Run the private
run_live_mixed_target.py driver with groups dflash tailfill, max-prefill-tokens
32768, cpu-overlap1 and hicache0; omit --profile-service and
MIXED_TARGET_PROFILE_DIR. No production scheduler registration is added.

Next useful work is profitable pairing coverage and offline qualification,
not assuming every underfilled chunk benefits or forcing all shapes through
Mixed. The observed one-call coverage sharply limits expected end-to-end gain.

Remote evidence remains at /root/rivermind-data/validation/ under:
- mixed-tailfill-diagnose-20261026 and mixed-tailfill-repeat-20261026
- mixed-tailfill-reference-20261026 and mixed-tailfill-shape-profile-20261026
- mixed-tailfill-pipeline-20261026 and mixed-tailfill-pipeline-final-20261026
- mixed-tailfill-joint-ab-20261026

/root/rivermind-data/tmp/tail-joint-evidence.tar.gz contains these results,
failed attempts, source snapshots/hashes, microbenchmarks and raw target traces.
The attempted local archive download did not complete; remote evidence exists.
