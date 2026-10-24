# Qwen3.5 DFlash with CPU overlap, PDMux split prefill and HiCache

This lane supports text-only Qwen3.5 MoE with DFlash, TP=1 and PP=1.
CPU overlap is enabled by default; omit `--disable-overlap-schedule`.
Other speculative algorithms and PDMux `standard` prefill still require the
synchronous scheduler. Grammar and penalties consume the pending CPU result
before preparing their next decode.

For the A100 40 GB AWQ lane, use the FP16 AWQ-Marlin target and the
unquantized DFlash draft:

```bash
export TARGET_MODEL=/path/to/QuantTrio/Qwen3.5-35B-A3B-AWQ
export DRAFT_MODEL=/path/to/Qwen3.5-35B-A3B-DFlash
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
bash test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh
```

The launcher sets both `--chunked-prefill-size` and `--max-prefill-tokens`
to 32768. These are batch token budgets; the per-request context limit remains
16384 to reproduce the SWE replay. The companion YAML caps prefill at 88 SMs,
allows decode to use all 108 SMs, and computes the split interval from a
4096 token-layer budget with an eight-layer upper bound. Without active decode,
prefill executes all remaining layers in one call.

The AWQ launcher defaults to FlashInfer target and draft attention
and a draft block of 8; linear attention uses Triton. It follows the target's
FP16 activation dtype, including the convolution cache. The checkpoint's training
block is 16. Set `BLOCK_SIZE=16` for that control, or `BLOCK_SIZE=4` for the
smaller-block control. Keep the backends
and block identical when comparing CPU overlap on/off. Append
`--disable-overlap-schedule` to run the synchronous CPU control.
Target and draft decode use full CUDA graphs; prefill graphs are disabled.
The AWQ launcher exposes `DFLASH=0 HICACHE=0 PDMUX=0` for the AR baseline, `PDMUX=0` for
ordinary DFlash, `HICACHE=0` for the cache ablation, and `CPU_OVERLAP=0` for the
synchronous scheduler. Hold the request, Mamba-state and GPU-memory budgets
constant across comparisons. `MAX_RUNNING_REQUESTS`, `MAMBA_CACHE` and
`MEM_FRACTION` expose those budgets; a target fitting in memory does not imply
that 32 speculative requests and their intermediate recurrent states also fit.
The validated A100 40 GB defaults are four running requests, 24 Mamba slots and
a static memory fraction of 0.9. Use these same budgets for the three main groups:

```bash
# 1. AR baseline (device radix cache remains enabled).
DFLASH=0 HICACHE=0 PDMUX=0 bash test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh
# 2. DFlash + HiCache, ordinary CPU overlap scheduler.
DFLASH=1 HICACHE=1 PDMUX=0 bash test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh
# 3. DFlash + HiCache + overlap split PDMux.
DFLASH=1 HICACHE=1 PDMUX=1 bash test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh
```

The supplied reference command requires several changes on this A100. Its
`trtllm_mha` target backend and FlashInfer linear-attention kernels require newer
GPU architectures. FA4's SM80 implementation rejects paged KV, which the serving
backend uses for the draft. This lane uses FlashInfer dense attention for both
models and Triton linear attention. HiCache disables `tc_piecewise` prefill graphs in this runtime,
so the launcher explicitly selects disabled prefill graphs. TP=1 has no inter-GPU
all-reduce to fuse.

Replay the frozen six SWE-rebench OpenHands trajectories as three sequential
assistant prefixes per trajectory, preserving the complete recorded histories.
Use outer concurrency 1 and 4, 128 generated tokens per request, greedy
sampling, and `ignore_eos=true`. Flush before each cold/incremental trial, then
repeat the same histories without flushing. Check host cache hits separately
from device prefix hits. Tokenization and tool execution are outside this
inference-only replay; it does not measure SWE task resolution.

Compare CPU overlap on/off under the same 32K prefill limits, then compare with
ordinary DFlash plus HiCache. Record generated token IDs, completion reasons,
retractions, TTFT, output tokens/s, normalized inter-token latency and stream
push gaps. Speculative responses can contain several tokens per push, so
normalized inter-token latency differs from the raw push gap.

Runtime completion and an accepted-token check against the actual target verify
logits do not establish numerical equivalence to an independent AR run. Compare
the complete generated token IDs against AR, across concurrency levels, and
between cold/incremental and repeated phases. Retain mismatches and the AR
self-repeat control when evaluating changes. Report acceptance rate together
with committed tokens per verification and zero-acceptance frequency: reducing
the block size alone raises the normalized acceptance fraction.

For the numerical reproducibility control, set `DETERMINISTIC=1`. This selects
Triton attention by default to retain radix/HiCache, enables batch invariant
operators and disables Marlin MoE's
atomic reduction. It fixes the Marlin M tile to eight and assigns complete K
reductions to output tiles, preventing batch shape and PDMux SM partitions from
changing split-K accumulation boundaries. AWQ zero points must remain in
registers across the same group as the scales, because the shared pipeline stage
can already be reused by asynchronous prefetch. Test this mode separately:
complete K reductions have a performance cost. Stable repeated outputs do not prove
cross-shape or cached/uncached equivalence to AR.

GDN prefix tracking stores the chunk accumulator directly in an FP32 snapshot,
then copies it to the recurrent-state pool. Reading the activation-typed `h`
would round a state through FP16/BF16 before saving it in an FP32 pool.
The generic GDN recurrent kernel also rounds sigmoid beta to the activation
dtype, matching prefill and packed AR decode. KDA retains its FP32 beta contract.
Regression coverage compares complete prefix/resume states and every state in
an eight-step verify block against independent packed decode, in FP16 and BF16.

HiCache-enabled AWQ groups use `kernel` I/O with `page_first` layout. A long deterministic
replay encountered an intermittent native host crash in direct Mamba copies;
the identical instrumented replay completed, so the root cause is unresolved.
The legacy `direct` path remains available through `HICACHE_IO_BACKEND=direct`;
compare performance
only with an AR control using the same I/O backend and layout.

Earlier GPTQ runs on the A100 identified long-prefix Triton attention as
the main decode cost. The AWQ lane uses FlashInfer dense attention and needs its
own profile: quantized MoE, unquantized projections and GDN can dominate instead.
Preserve cold/incremental and repeated phases in reports instead of extrapolating
the repeated-history result to cache misses.
HiCache reduces repeated prefix work; it does not guarantee that PDMux or
DFlash improves TTFT on a cold cache or with a low prefix hit rate.

The original `Qwen/Qwen3.5-35B-A3B` has about 72 GB of BF16 weights and does not
fit entirely on this A100 40 GB. The QuantTrio AWQ checkpoint has approximately
25.46 GB of weight shards. Treat it as a separate target from earlier GPTQ or
BF16 measurements. The older GPTQ launcher remains only for reproducing those
earlier experiments.

See [`qwen35_awq_swe_results.md`](qwen35_awq_swe_results.md) for the measured
three-group P88 comparison, acceptance counters and remaining numerical gaps.
