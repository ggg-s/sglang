# Qwen3.5 AWQ: three-group SWE replay on A100

Measured on 2026-10-24 with `QuantTrio/Qwen3.5-35B-A3B-AWQ`, FP16 activations,
an A100 PCIe 40 GB (108 SMs), and CPU overlap enabled in every group. The PDMux
configuration caps concurrent prefill at **88 SMs**; decode uses a high-priority
stream covering **all 108 SMs**. Idle prefill uses the full device.

The repeated-history throughput is highest with PDMux, but ordinary DFlash with
HiCache has better TTFT and cold/incremental throughput. This configuration is
not an overall win on every latency and throughput metric.

## Performance

Concurrency 4, two trials per phase. Values below are medians of per-trial
metrics. Quantiles are not pooled across trials.

| Group | Cold/incremental output tok/s | Repeated output tok/s | Cold TTFT p50 (ms) | Repeated TTFT p50 (ms) | Repeated TPOT mean (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR baseline, HiCache off | 257.3 | 300.1 | 243.9 | 206.4 | 7.79 |
| DFlash + HiCache | 270.5 | 353.7 | 194.6 | 130.4 | 7.88 |
| DFlash + HiCache + overlap split PDMux, P88 | 262.2 | 365.7 | 468.2 | 155.3 | 7.23 |

P88 improves repeated-history throughput by 3.4% over ordinary DFlash + HiCache
and 21.9% over AR. Its repeated TPOT is 8.3% lower than ordinary DFlash. Its
cold/incremental throughput is 3.1% lower, and its TTFT remains higher.

PDMux also reduces long pauses after streaming has begun. Cold/incremental
maximum push gaps were **95.7 / 81.6 ms** in its two trials, versus
**812.4 / 375.8 ms** for ordinary DFlash and **771.0 / 774.5 ms** for AR.
No outlier was removed. Repeated-history maximum push gaps were 59.5 / 60.3 ms
with P88 and 119.5 / 113.8 ms with ordinary DFlash. A speculative push may
contain multiple tokens; a push gap is not normalized inter-token latency.

All 216 measured C4 requests completed with 128 output tokens and zero
retractions. This is a runtime-completion observation, not an accuracy result.
The complete per-trial metrics and counters are in
[`qwen35_awq_swe_results.json`](qwen35_awq_swe_results.json).

## Acceptance and remaining bottlenecks

On repeated histories, ordinary DFlash accepted 24.36% of proposed drafts and
committed 2.705 tokens per verification. P88 accepted 23.80% and committed
2.666 tokens per verification; 40.29% of its verifications accepted no draft.
The target evaluates blocks of eight while committing about 2.67 tokens on
average, in addition to running the draft. This limits speculative speedup.
P88's throughput gain did not come from an increased acceptance rate.

The draft was trained against the unquantized target. Whether AWQ causes the
low acceptance requires a matched unquantized control, which this 40 GB lane
cannot hold fully on GPU. Do not attribute a measured acceptance loss to AWQ
without that control.

An earlier eager AWQ trace identified unquantized GEMMs, Marlin MoE and GDN as
large kernel-time contributors. Its timings are not formal P88 timings.
Additional profiling was cancelled at the user's request to deliver this
performance comparison quickly. The P88 TTFT regression still needs scheduling
analysis; increasing the prefill cap alone did not eliminate it.

## Numerical status

Two runtime fixes and one deterministic-control change are included:

- Preserve AWQ zero points in registers across their quantization group, rather
  than reloading a shared pipeline stage that prefetch can reuse.
- Save GDN prefix checkpoints from the FP32 accumulator, and match generic GDN
  sigmoid-beta rounding to prefill and packed AR decode. KDA keeps FP32 beta.
- In deterministic Marlin mode, use a fixed M tile and complete K reductions
  so batch size and PDMux SM counts do not change accumulation boundaries.

The selected GPU checks passed: **25 tests and 32 subtests**, including
FP16/BF16 prefix/resume state checks, eight-step verify versus packed decode,
AWQ GEMM oracles, shape/SM invariance and CUDA graph routing replay.

**Full generated-token equality has not passed.** In the separate deterministic
SWE control, DFlash P88 matches independent AR on 11/18 cold/incremental requests
and 9/18 repeated requests. DFlash cold versus repeated matches 16/18.
Its C1 versus C4 comparisons match 16/18 in both cold trials and 16/18 or 17/18
in repeated trials. The AR self-repeat control also has residual differences.
These results must not be presented as complete numerical correctness or a
SWE task-resolution score. Further diagnostic runs were deferred when the user
prioritized quick performance measurements.

## Reproduction

Use the three commands in
[`qwen35_dflash_overlap_runbook.md`](qwen35_dflash_overlap_runbook.md), with the
same budgets: max running requests 4, Mamba slots 24, static memory fraction
0.9, KV tokens 32768, context limit 16384, and both prefill budgets 32768.
Target and draft dense attention use FlashInfer, linear attention uses Triton,
decode uses full CUDA graphs, prefill graphs are disabled, and DFlash block
size is 8. HiCache-enabled groups use 4 GB, kernel I/O and page-first layout.

The frozen workload contains six SWE-rebench OpenHands trajectories with three
complete recorded assistant prefixes each (6909-9773 input tokens). Its SHA256
is `687c751bd2ad026771bc31ee737e19a02fc17605bf9d5c764a58eda1a535b123`.
Generate 128 tokens, greedy sampling, `ignore_eos=true`; each trajectory's
prefixes run sequentially, with up to four trajectories active. Flush once
before each cold/incremental trial, then repeat without flushing. Shared
prefixes can already hit during the cold/incremental phase.

Warmup is excluded. Tokenization and tool execution are outside this replay.
The AR baseline was reused because the later GDN beta fix changes the generic
recurrent path, while this AR lane uses packed decode; its other executed
runtime sources and budgets match the final candidate. Source hashes, model
revisions, generated IDs, launch arguments and logs are retained in the
validation artifacts. Runtime fix commits are `224d19ba7d` and `8c7de916ce`.
