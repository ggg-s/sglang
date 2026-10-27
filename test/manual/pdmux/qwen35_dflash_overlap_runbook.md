# Qwen3.5 DFlash with CPU overlap, PDMux split prefill and HiCache

## Experimental full-device temporal multiplexing

`SGLANG_PDMUX_TEMPORAL_LAYERS=2|4|8` enables a separately gated temporal
variant of `layer_split` PDMux. The default is zero, preserving the existing
spatial PDMux path. This experiment requires CUDA, DFlash overlap scheduling,
TP=PP=attention-DP=1 and `SGLANG_PDMUX_DECODE_READY=off`. The target must support
the existing DFlash auxiliary-state-preserving split-prefill interface.

Both logical lanes use one ordinary full-device CUDA stream and retain separate
attention metadata/backends. No green context is created; the YAML's SM
divisions and token-layer split budget do not select the temporal quantum.
One complete DFlash round alternates with a fixed number of complete target
layers while decode is active. Without decode work, prefill finishes all
remaining layers in one call. An event credit permits at most one unfinished
prefill quantum in the GPU queue, while allowing the next decode to be prepared
and submitted. Polling consumes the previous decode result at most once and
does not perform a device synchronization. Existing cache/allocator publication
and final merge ordering are retained.

The final target segment still performs ordinary DFlash draft-KV initialization
and result-copy/merge handling. That tail is **not split** (experiment C is not
part of this mode), so the layer quantum is not an upper bound on every output
stall.

Use the same target, draft, workload and capacity/cache settings for each group:

```bash
# Ordinary DFlash control, with the current cache and overlap configuration.
PDMUX=0 SGLANG_PDMUX_TEMPORAL_LAYERS=0 SGLANG_PDMUX_DECODE_READY=off \
  bash test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh
# Repeat separately with 2, 4 and 8; leave all other settings identical.
PDMUX=1 SGLANG_PDMUX_TEMPORAL_LAYERS=2 SGLANG_PDMUX_DECODE_READY=off \
  bash test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh
```

Measure the client-observed interval between consecutive token-bearing stream
responses, excluding TTFT and tokenless completion messages. Report its p95/p99
separately from normalized inter-token latency: DFlash can deliver multiple
tokens per response. Retain each trial's TTFT, TPOT, output throughput and cache
phase, and compare per-trial statistics across independent server launches.

### A100 temporal A/B, 2026-10-24

The comparison used the same frozen six SWE trajectories, three turns each,
outer concurrency four, 128 output tokens, greedy sampling and ignore-EOS.
Each group had two independent server launches and two trials per launch;
every trial flushed before cold/incremental replay and then repeated the same
histories without flushing. Launch order was baseline, 2, 4, 8, baseline, 8, 4,
2. The ordinary DFlash baseline had PDMux disabled; both prior decode-ready
experiments were off in every group.

Fixed settings: A100 PCIe 40 GB / 108 SMs, FP16 AWQ-Marlin target, unquantized
DFlash draft, FlashInfer target/draft attention, Triton linear attention, full
decode graphs, disabled prefill graphs, draft block eight, four running requests,
24 Mamba slots, memory fraction 0.9, 65536 total tokens, 32768 prefill limits,
16384 context, HiCache 16 GB write-through and ReplaySSM-spec cache length 32.
Resolved server settings differed only in PDMux mode/configuration and runtime
metadata. Actual cache-hit token counts differed with scheduling and are
retained per trial; cache configuration and input histories were identical.

Numbers below are medians of the four per-trial statistics for each phase.
All latency columns are milliseconds; TTFT is p50, TPOT is the per-request mean,
and throughput is total output tokens divided by phase wall time. The gap
quantiles are calculated within trials, not pooled across launches.

| Phase | Mode | Push gap p95 | Push gap p99 | TTFT p50 | TPOT | Output token/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cold/incremental | Ordinary DFlash | 29.59 | 123.85 | 166.56 | 7.70 | 286.39 |
| Cold/incremental | Temporal 2 layers | 40.07 | 68.72 | 474.89 | 7.09 | 261.85 |
| Cold/incremental | Temporal 4 layers | 42.20 | 94.68 | 300.49 | 7.70 | 279.19 |
| Cold/incremental | Temporal 8 layers | 51.19 | 80.95 | 244.38 | 7.85 | 288.94 |
| Repeated | Ordinary DFlash | 23.96 | 112.05 | 134.11 | 7.29 | 372.73 |
| Repeated | Temporal 2 layers | 25.59 | 42.82 | 385.89 | 5.92 | 351.55 |
| Repeated | Temporal 4 layers | 26.72 | 48.21 | 211.51 | 6.34 | 381.93 |
| Repeated | Temporal 8 layers | 27.78 | 57.42 | 169.23 | 6.52 | 389.02 |

Temporal multiplexing reduced p99 but increased p95 and median TTFT in both
phases. Eight layers was the less disruptive TTFT/throughput tradeoff: repeated
p99 improved 48.8%, TPOT improved 10.6% and throughput increased 4.4%, while
TTFT worsened 26.2% and p95 worsened 15.9%. Two layers gave the smallest p99 but
lost throughput in both phases. No candidate met the original combined
TTFT/TPOT/throughput gate. Keep the temporal switch at zero; C was not executed.
These are workload observations, not a guarantee on the maximum stall or a
statistical claim beyond the recorded trial ranges.

All 576 measured requests completed with 73728 output tokens and zero
retractions. Final PDMux regression coverage passed 139 tests. Audit samples
confirmed a shared physical stream, (108, 108) SM counts, actual 2/4/8 layer
quanta while decode was active and preserved decode graph replay. GPU memory
returned to 0 MiB after all owned servers exited. The six modified runtime
files exactly match the tested remote source hashes. Full generated outputs
are not bit-identical under this unchanged nondeterministic configuration;
baseline self-repeats also differ. This test does not establish numerical
equivalence to an independent AR run.

See [the machine-readable results](qwen35_awq_temporal_results.json) for all
trial values/ranges, configurations, source hashes, audit samples and generated
ID comparisons. Raw SSE timestamps, output IDs and logs are archived locally at
`.tmp/pdmux-temporal-20261024/raw-artifacts.tar.gz`.

The original spatial-PDMux instructions and prior experiments follow below.

This lane supports text-only Qwen3.5 MoE with DFlash, plain TP and PP=1.
CPU overlap is enabled by default; omit `--disable-overlap-schedule`.
Both launchers accept `TP_SIZE=2` (default: 1). At TP>1 the overlap lane
requires EP, MoE DP, attention CP and DCP sizes of one, and DP attention
disabled; their subgroup communicators are not duplicated by lane. The
temporal and decode-ready experiments above/below still require TP=1.
Other speculative algorithms and PDMux `standard` prefill still require the
synchronous scheduler. Grammar and penalties consume the pending CPU result
before preparing their next decode.

TP overlap routes full-width attention/MoE TP aliases and sampling sync to
the current lane's communicator. Both PDMux modes use the ordinary logits
all-gather instead of sharing one multimem buffer/signal pad across streams.
The CPU regression includes two real Gloo ranks, unequal completion timings,
multiple prefill chunks, HiCache pumps and overlap on/off:

```bash
PYTHONPATH=python python -m pytest \
  test/registered/unit/multiplex/test_pdmux_tp_overlap.py -q
python -m pytest test/manual/pdmux/test_pdmux_tp_overlap_cuda.py -q
```

The CUDA file runs a singleton NCCL/graph/stream check on one GPU, and an
additional two-rank check only when two distinct CUDA GPUs are available.
These are operator/control-flow checks without checkpoint weights. On the
deployment host, also replay the original long-prompt, multi-chunk workload
past the reported ~1030-request failure point, with HiCache and ReplaySSM
enabled, before drawing reliability or throughput conclusions.
The [2026-10-26 validation record](qwen35_tp_overlap_validation_20261026.json)
contains the 4060 results, source hashes and remaining deployment checks.

The subsequent collective-safety fixes keep HiCache DMA readiness polling
rank-local: ack retirement and its collectives run only in the scheduler's
unconditional event pump. This also covers the disaggregated decode restore
path, where ranks may poll different numbers of times. FlashInfer all-reduce
fusion is unavailable while PDMux communicators exist: its shared workspace
and signal pad are not isolated by lane. PDMux keeps using its ordinary,
lane-specific TP communicators; CPU and prefill/decode overlap remain enabled.
Workspace reuse outside PDMux also requires the same communicator and rank,
including the legacy rendezvous-group argument, in addition to matching size.
Run `test/registered/unit/multiplex/test_pdmux_collective_safety.py` for the
two-rank polling and workspace regression checks.
The [2026-10-27 validation record](qwen35_tp_collective_safety_validation_20261027.json)
records this follow-up's tests, source hashes and hardware limits.

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

## ReplaySSM memory control

For DFlash's GDN verify path, use speculative ReplaySSM:

```bash
REPLAYSSM_SPEC=1 REPLAYSSM_CACHE_LEN=32 HICACHE_SIZE=16 MAX_TOTAL_TOKENS=65536 \
  PDMUX_CONFIG=test/manual/pdmux/qwen35_dflash_overlap_replayssm_a100.yaml \
  bash test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh
```

This selects `--enable-linear-replayssm-spec`, keeps the `extra_buffer` radix
strategy and replaces per-draft full SSM snapshots with compact request-keyed
records. The separate `--enable-linear-replayssm` flag is the ordinary decode
control and requires `no_buffer`; it is not the DFlash memory control.
`MAX_TOTAL_TOKENS` changes the aggregate GPU KV pool limit. Both prefill budgets
remain 32768, and the per-request context limit remains 16384.

DFlash commits accepted GDN replay records using the actual request indices and
the physical state indices selected by the decode backend. Tracked prefix states
materialize at the accepted interval crossing, including under PDMux state
translation. Recurrent and replay verify use the same activation-dtype beta
rounding. The focused GPU regression covers circular wrap, a capacity fold,
deferred active state, translated track destinations and convolution rollback.
This does not establish complete generated-token equivalence to independent AR.

ReplaySSM remains opt-in. Compare its throughput and TPOT before selecting it:
freeing Mamba memory does not itself guarantee a latency improvement.
The ReplaySSM companion YAML keeps P88 and full-device decode, and uses a 32768
token-layer split budget with a 16-layer limit. This changes how many layers run
between decode submissions; the two 32768 prefill token limits are independent.
See the [ReplaySSM follow-up results](qwen35_awq_replayssm_swe_results.md) for
the measured 64K comparison and the remaining TTFT and numerical limitations.
The [scheduler TTFT investigation](qwen35_awq_scheduler_ttft_results.md) records
the CPU-length mirror wait, P88/P104 controls and rejected scheduling changes.
Its clean rerun retained the same runtime and configuration.

## Decode-ready experiments: A and B

The 2026-10-24 follow-up implements two opt-in experiments without changing
the selected launch arguments or YAML. `SGLANG_PDMUX_DECODE_READY=off` is the
default and retains the original scheduling order. `mirror` selects A:
publish-time length snapshots, private-stream D2H and a generation-owned
completion ticket. `dispatch` selects A+B: use the pending-ticket window for
single-layer prefill submissions, then yield at full-layer boundaries when
the next length ticket is ready. Both require CUDA, DFlash CPU overlap,
`layer_split`, TP=1, PP=1 and attention DP=1.

GPU state dependencies remain in place. A ticket only permits CPU planning;
it does not certify draft KV or recurrent state completion. Snapshot slots
wait for their previous D2H before reuse, request generations reject recycled
pool rows, and consumed CPU lengths do not alias the reusable pinned slots.
B prepares each decode once and shares the existing token-layer budget
between pre-decode and post-decode prefill submissions. Final draft KV
materialization remains part of the last target submission; C was not run.

All groups used the original P88/full-device D108 streams, block eight,
65536-token KV pool, 24 Mamba slots, memory fraction 0.9, HiCache size 16,
ReplaySSM ring 32, token-layer budget 32768 and 16-layer upper bound. Actual
resolved server arguments matched across groups, excluding runtime metadata
and ephemeral NCCL ports. The frozen SWE workload SHA256 remains
`687c751bd2ad026771bc31ee737e19a02fc17605bf9d5c764a58eda1a535b123`.
Concurrency was four, with 18 requests and 128 output tokens per phase/trial.
Every clean trial used the original warmup, cache flush and cold/incremental
then repeated phases. No profiler or diagnostic timing probe was enabled.

Values below are medians of per-trial metrics, not pooled request quantiles.
The patched-off controls bracket A's two independent server launches. B's
large TTFT regression was observed in both trials and was not selected for
another launch.

| Runtime | Trials per phase | Cold/incremental TTFT p50 (ms) | Repeated TTFT p50 (ms) | Repeated TPOT mean (ms) | Repeated output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original runtime control | 2 | 212.8 | 146.3 | 7.083 | 380.6 |
| Patched runtime, `off` | 4 | 212.3 | 143.7 | 7.216 | 376.8 |
| A, `mirror` | 4 | 230.4 | 137.4 | 6.775 | 397.2 |
| A+B, `dispatch` | 2 | 262.2 | 188.1 | 6.191 | 396.8 |

Against the patched-off controls, A improved repeated TTFT by 4.4%, TPOT by
6.1% and output throughput by 5.4%, but cold/incremental TTFT was 8.5% higher
and cold throughput was essentially flat (-0.2%). Cold TTFT varied substantially
between trials and launches; these measurements do not establish an improvement
in both phases. B improved decode metrics but increased repeated TTFT by 30.9%.
In the first repeated trial its prefill span p50 was 159.1 ms versus 124.8 ms
for the original control; queue p50 remained about 1 ms. This is evidence of a
prefill service tradeoff, not proof of a particular kernel or copy bottleneck.

Neither experiment passed the complete adoption gate. Keep the selected
configuration and default `off`; do not proceed to C on the basis of these
results. The implementations remain available for isolated experiments.

All 432 measured requests completed with 128 output tokens and zero retractions.
Validation passed 69 focused CPU checks and six CUDA ticket checks, with the
CUDA checks also passing CI's poisoned-relay assertions. Ruff's configured
checks and formatting passed. Generated token IDs are retained: the original
and patched-off controls themselves did not match on every request, so this
performance experiment does not establish AR token equivalence or resolve the
previous numerical gap.

Per-trial metrics, completion/acceptance counters, generated-ID comparisons,
source and raw-artifact hashes, discarded startup attempts and hardware
details are retained in
[`qwen35_awq_decode_ready_results.json`](qwen35_awq_decode_ready_results.json).

## Full-device temporal scheduling and throughput adaptation

`SGLANG_PDMUX_TEMPORAL_LAYERS=0` preserves the selected configuration.
Positive values use one ordinary physical CUDA stream with all 108 SMs for
both P and D. Separate attention backends remain, but green-context SM
partitioning is bypassed. The fixed-quantum experiment tested 2, 4 and 8 layers
against ordinary baseline+DFlash, with decode-ready experiments disabled.
Each DFlash round and each target layer remains atomic. One unfinished P
quantum is allowed in the GPU queue. C (splitting final draft-KV work) remains
unimplemented. Frozen results are in
[`qwen35_awq_temporal_results.json`](qwen35_awq_temporal_results.json).

The initial fixed-eight experiment improved repeated output throughput by
4.4% and push-gap p99 by 48.8%, but raised TTFT p50 by 26.2% and push-gap p95
by 15.9%. These are four per-trial medians across two startups, rather than
pooled-request quantiles. Subsequent matched controls show that the throughput
gain varies between runs; it is not an established configuration-wide gain.

The TTFT attribution used the earlier unprofiled raw requests. Repeated
prefill residence p50 rose from 111.2 to 142.4 ms while queue p50 remained
about 1 ms. Cold/incremental residence rose from 117.8 to 183.0 ms. Repeated
push gaps exceeding 20 ms increased from 8.0% to 18.2%, while gaps exceeding
80 ms fell from 3.6% to zero. Fixed interleaving replaces rare long pauses
with more frequent moderate pauses, which explains opposite p95/p99 changes.
Additional complete D rounds also lengthen P residence.

A separate native Torch Profiler capture covered a warmed mixed workload,
TP=PP=1 (the only rank), and 26,548 GPU kernels. It is diagnostic evidence,
not formal A/B timing. Four steady eight-layer small-prefill spans used
23–25 ms of host dispatch and 17–19 ms of GPU elapsed span, with only
4.2–5.1 ms of GPU kernel union inside each span. Copies and memset barely
changed the union; the four spans contained no host CUDA synchronization.
This supports exposed launch/submission gaps for short cache-hit tails.
Unprofiled instrumented middle segments also had host p50 16.6 ms versus
CUDA-event span p50 8.2 ms. Small-prefill finalization was about 1.5 ms host
and 0.95 ms GPU, so it does not explain most of this regression. Large cold
prompt finalization is still indivisible and can dominate an individual gap.

| Kernel evidence in the mixed trace | Attribution | Source confidence |
| --- | --- | --- |
| Marlin MoE kernels and FlashInfer paged attention during P | Existing quantized target prefill dispatch | Kernel names verified; Python stack mapping unavailable |
| `gdn_replayssm_spec_circular_kernel` and graph-launched GEMMs during D | Existing DFlash target/verify path | Actual configured graph path retained |
| `transfer_mamba_load_kernel` | Existing recurrent-state transfers | A timing share alone does not establish an overlapping opportunity |

| Scheduling opportunity | Dependency constraint | Implemented response |
| --- | --- | --- |
| Ready D output waits behind CPU P dispatch | Must preserve result order and full DFlash completion | Consume previous result before P and current result only after actual `copy_done` |
| Repeated short P submissions expose CPU overhead | Whole target layer and DFlash round boundaries remain atomic | Learn host/GPU service cost and choose larger quanta when useful |
| Small D batch can postpone admitting P into D | Decide before KV allocation and Mamba counter changes | Compare current and expanded-batch useful-token rates before preparing D |

| Fusion evidence | Assessment |
| --- | --- |
| Residual-add RMSNorm kernel | Fusion already present; unchanged |
| Analyzer grouped-topk/gate match on gated-delta kernels | Rejected as a router-attribution false positive |
| Missing Python mapping | No new kernel-fusion claim or implementation |

`SGLANG_PDMUX_TEMPORAL_ADAPTIVE=1` requires temporal layers eight and the
existing CUDA/DFlash overlap, layer-split, single-rank and supported-model
guards. It is an experimental throughput policy, default off. It measures
completed useful output through each request's actual stop boundary, including
zero-output rounds; speculative proposals and overshoot tokens are not reward.
EWMA estimates track D service and useful tokens by batch size, and P service
by prompt-token bucket. CPU and GPU elapsed time use their maximum, not their
sum. First-segment setup and final draft-KV work do not contaminate per-layer
estimates. Timing events are polled only when ready, with a bounded FIFO.

The controller predicts the time to finish current D output plus the output
that unfinished P will admit. It compares 4/8/16/32-layer quanta and an eligible
remainder. It favors fewer scheduling boundaries within a 2% prediction band;
this band is controller hysteresis, not statistical significance. Unknown P
shapes bootstrap at eight layers. Unknown output limits use a prediction
horizon of 128 tokens without changing generation limits. Chunked P requests
that cannot join D yet are excluded from admission benefit.

The action-level version also compares current D useful token/ms against the
predicted expanded-batch rate. If expansion is worthwhile, it may advance P
before another D round; otherwise it services D first. This decision precedes
`update_running_batch` and `prepare_for_decode`. Deferred D iterations only
filter finished requests and never allocate another decode KV step or advance
its Mamba counters. The approximate 64 ms service/deferral limit and 32-layer
exploration cap are soft guards, not latency guarantees. First setup, a slow
four-layer minimum and unchanged C can exceed them. Calibration is a model,
not an online proof of optimality; whole-workload unprofiled A/B remains the
adoption criterion.

The quantum-only prototype is retained as a negative control: in its fresh
matched experiment, repeated output throughput was 381.0 tok/s versus 391.1
for fixed eight and 390.9 for baseline+DFlash. Its shorter repeated TTFT did
not constitute a throughput improvement. Latency-budget prototype runs were
interrupted when the objective changed and are excluded from formal throughput
comparisons. All experiments retain the original concurrency four, greedy
128-token outputs, cache sequence, precision, graph settings, replay and KV
budgets. No profiler or boundary-timing probe runs in formal measurements.

The corrected action-level policy was tested in a fresh six-start matrix:
baseline, fixed eight, adaptive, adaptive, fixed eight, baseline. Each startup
ran two complete cold/incremental plus repeated trials. All values below are
medians of four per-trial measurements per group/phase. Push gaps are intervals
between token-bearing client SSE responses, excluding TTFT; multiple DFlash
tokens may share one response, so these are not normalized inter-token gaps.

| Phase | Runtime | Output tok/s | TTFT p50 ms | TPOT mean ms | Push p95 ms | Push p99 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cold/incremental | Baseline+DFlash | 282.82 | 156.74 | 7.86 | 30.09 | 124.81 |
| Cold/incremental | Fixed eight | 291.70 | 244.65 | 7.80 | 49.53 | 78.39 |
| Cold/incremental | Action-adaptive | 294.61 | 187.48 | 7.90 | 53.65 | 125.41 |
| Repeated | Baseline+DFlash | 384.55 | 131.38 | 7.04 | 22.54 | 109.95 |
| Repeated | Fixed eight | 380.56 | 174.65 | 6.80 | 27.60 | 66.31 |
| Repeated | Action-adaptive | 386.65 | 128.76 | 7.06 | 33.86 | 97.28 |

Against the matched controls, action-adaptive output throughput changed by:

- cold_then_incremental: +4.17% versus baseline+DFlash, +1.00% versus fixed eight; adaptive per-trial range 289.6–295.2 tok/s.
- repeated: +0.55% versus baseline+DFlash, +1.60% versus fixed eight; adaptive per-trial range 385.4–394.8 tok/s.

Combined-phase useful throughput (sum tokens / sum serving time within each trial, then median) was 326.83 baseline, 330.27 fixed eight and 334.51 adaptive tok/s. Adaptive changed by +2.35% versus baseline and +1.28% versus fixed eight. Startup, warmup and cache flushing are excluded from request-serving time.

The throughput improvement is modest and the trial ranges overlap. Against
fixed eight, adaptive reduced TTFT p50 by 23.4% cold and 26.3% repeated, but
raised TPOT by 1.3% and 3.7%, and push-gap p99 by 60.0% and 46.7%. Against
baseline, repeated p99 improved by 11.5% while p95 increased by 50.2%; cold
TTFT p50 remained 19.6% higher. This is a throughput/fill-batch tradeoff, not a
simultaneous latency improvement. Throughput remains the primary objective;
the small observed gain and overlapping trial ranges are insufficient evidence
for enabling the mode by default.

These measurements are workload-specific observations with two independent
startups per group, rather than evidence of a general optimum. The experiment
keeps the mode default off and the selected configuration unchanged. C remains
unimplemented. Completion covered 432 requests and 55,296 output tokens with
zero incomplete requests and zero retractions; resolved configuration checks
found no unexpected differences. Validation passed 139 PDMux regressions,
16 throughput-policy checks and two Mamba boundary checks. The original
nondeterministic runtime is retained: first-token and completion checks do not
prove full generated-ID or independent AR equivalence. An earlier action
startup was interrupted after detecting the admission-fullness reset edge
case; it is excluded, as are abandoned latency-objective measurements.

Per-trial metrics, source hashes, negative-control results, configuration
comparisons, corrected profiler interpretation and raw archive SHA256 are in
[`qwen35_awq_throughput_adaptive_results.json`](qwen35_awq_throughput_adaptive_results.json).
The raw archive retains SSE timestamps/output IDs, logs, source snapshots,
diagnostic spans, the native trace and collection scripts.

## Initial offline mixed prefill + complete DFlash verification (2026-10-25)

This section records the original failed candidate. The numerical repair and
subsequent Mixed Graph validation are recorded in the next section.

`python/sglang/srt/speculative/mixed_eager.py` implements an offline P1 + D3×8
target forward. Linear projections and MoE see one packed token axis. Each
layer routes real P rows and the complete 24 verification rows to independent
native Attention/GDN backends. P capacity padding never reaches those stateful
calls. Auxiliary target hidden states are split back into the native logits
processors and DFlash draft-KV append path. This module has no scheduler flag
or normal-runtime import. The original Draft/Verify Graph path remains the
serving path, with no SM partition and no C implementation.

The actual-model experiment retains QuantTrio Qwen3.5-35B-A3B-AWQ, the existing
z-lab DFlash checkpoint, A100 PCIe 40 GB, TP1, block8, concurrency four, FP16,
AWQ Marlin, FlashInfer, ReplaySSM length 32, cache budgets and graph settings.
The resolved server configuration is compared with the preceding baseline.
This is a correctness experiment; CPU snapshots, synchronization and diagnostic
hooks make its elapsed times unsuitable for throughput or latency comparisons.

To reproduce on the Linux GPU host, use the original string-valued launch
environment JSON and a fresh output directory:

```bash
python test/manual/pdmux/run_mixed_eager.py \
  --base-config /validation/existing-baseline-env.json \
  --model-path /models/QuantTrio/Qwen3.5-35B-A3B-AWQ \
  --draft-model-path /models/z-lab/Qwen3.5-35B-A3B-DFlash \
  --output-dir /validation/mixed-eager-original-new \
  --numerics original
```

The supplied baseline's existing compatibility `sitecustomize.py`, if needed,
can be retained with `--compat-script /existing/compat/sitecustomize.py`.
`--smoke` exercises one case in each lane. The driver owns a private server
process group, keeps native draft/verify graphs, records graph hits and source
hashes, and cleans up its server. Exit code **2** means correctness did not pass;
it is an expected failure signal for this initial candidate, not a passed run.
This private process must not receive external inference traffic.

The full matrix uses successive P chunks of 23, 32, 40, 64, 65, 128, 257, 430
and 512 tokens, with capacities 32, 32, 64, 64, 128, 128, 512, 512 and 512.
Each round starts its branches from the same target/draft KV, Conv/SSM,
ReplaySSM and request-mapping snapshot. Native draft graphs produce the
candidates. A natural lane keeps them unchanged; a rejection fixture changes
two candidate IDs and lets native target verification determine acceptance.
It does not force accepted lengths. Ring rotation preserves logical history
while exercising physical read/write wrap. P advances through real native
continuation chunks between comparisons. The mixed branch is restored after
each case; this is not an independently chained long mixed trajectory.
Floating states are compared against Separate Eager; the native Verify Graph
reference additionally checks acceptance/bonus, rather than full state equality.

The numerical gate uses fixed `rtol=0.02, atol=0.002`; integer mappings/cursors
and untouched regions require exact equality. KV snapshots cover all locations
allocated to the four requests plus sentinel zero, rather than the entire
unallocated pool. Boundary checks protect padding, other request/SSM slots,
non-Verify replay rows and rejected draft-KV tails. Separate kernel regressions
cover nonidentity physical-state mapping, commit/fold, tracking and Conv rollback.
The actual-model D prefix stays below the 256-token tracking boundary, so a
full-model crossing of that boundary remains additional coverage for a future
candidate.

Both real mixed lanes completed all nine cases with matching acceptance and
bonus decisions and passed the checked ownership/write boundaries. Both failed
the floating-state criterion, including Conv/SSM, ReplaySSM and KV differences.
Matching acceptance on this fixture is insufficient to establish state
equivalence or general output correctness.

Diagnosis retains two distinct controls. Repeating separate eager execution
under the unchanged nondeterministic AWQ configuration also produces floating
differences. A diagnostic that disables atomic reductions and selects
batch-invariant Marlin MoE stabilizes that self-repeat, but real mixed execution
still fails the original state tolerance. A feature probe finds small initial
projection differences that amplify through MoE. This supports sensitivity to
batch shape and reduction order; it does not prove the discrepancies harmless.
These diagnostic reduction choices are not adopted as a new baseline.

A further **diagnostic only** lane substitutes the separate run's token-local
projection/MoE outputs into the mixed forward to isolate Attention/GDN routing
and cache commit. Its results are recorded separately. Even a fully matching
frozen-intermediate lane cannot qualify the actual mixed candidate or its
graph gate. Tolerances have not been relaxed after observing failures.

The complete matrix retains its frozen harness snapshot. The final standalone
entry point additionally enforces graph-hit eligibility and allocates lazy
worker planning buffers under `no_grad`, so later native health/warmup forwards
can update them. Its first smoke attempt exposed inference-tensor reuse and is
preserved as a failed run; a separate smoke validates the lifecycle correction.

**Mixed Graph is withheld.** No throughput benefit is claimed. Before graph
capture, actual mixed AWQ computation must satisfy the state criterion and
broader output/continuation checks under the agreed configuration. Potential
P capacities are 32/64/128/512: prior short tails of 18–46 tokens motivate
smaller buckets, while P512 provides a larger option. These are unmeasured
candidates, not an optimal bucket set. Graph implementation would also need
static storage with in-place metadata refresh and explicit branch eligibility;
the prototype's validation, packing and Python routing are not graph-ready.

Per-case results, source hashes, diagnostics, configuration comparison,
reproduction smoke and raw archive SHA256 are recorded in
[`qwen35_awq_mixed_eager_results.json`](qwen35_awq_mixed_eager_results.json).

## Offline numerical repair and Mixed Graph (2026-10-25)

The repaired candidate preserves the same checkpoint, DFlash, A100, TP1,
block8, concurrency four, launch configuration, cache budgets and tolerances
(`rtol=0.02`, `atol=0.002`). It changes the candidate's operator arithmetic
explicitly. Formal serving still uses the original Draft/Verify Graphs; the
new modules have no scheduler registration. C and SM partitioning remain absent.

The checkpoint's exclusions matter: Attention/GDN projections, shared experts
and router use ordinary FP16 GEMMs. There are 39 AWQ MoE layers and one
unquantized MoE layer, with **zero ordinary AWQ Marlin Linear dispatches**.
The initial projection discrepancy must therefore be attributed to FP16 GEMM
batch-shape sensitivity, rather than Marlin Linear. Native Marlin expert
atomic split-K reduction introduces a second source of self-repeat noise.
Small upstream differences can change expert sets near routing ties; native
TopK is retained throughout the repair.

An actual-kernel and six-layer routing audit distinguishes the two effects:

| Audit | Separate repeat: state fields over tolerance | Mixed: state fields over tolerance | Repeat / mixed expert-set row differences |
| --- | ---: | ---: | ---: |
| Original arithmetic | 36 | 38 | 22 / 43 |
| Repaired arithmetic | 0 | 0 | 0 / 0 |

The routing numbers sum differing token rows over layers 0, 3, 7, 15, 31 and
39; they are not counts of distinct requests or tokens. Native expert calls
used atomic reduction with full-K disabled. Inside the repaired body, the audit
observed non-atomic, full-K, block-M8 expert calls. Native seeding and history
advancement outside the adapter still used the original modes. Audit-only
reports never qualify the correctness gate.

`mixed_marlin.py` provides an explicit offline context manager. It replaces
the target's excluded FP16 Linear/router GEMMs with fixed M16/N32/K64 tiles,
FP32 accumulation and FP16 outputs, retaining original FP16 inputs and weights.
It selects the existing full-K Marlin expert path and a consistent sum path for
the unquantized first MoE layer. The native router, checkpoint quantization,
packed expert weights, scales and zeros remain intact. All patched methods and
module-local helpers are restored on exit, including exceptions. The adapter
does not enable the global deterministic-inference setting. A fixed-Linear-only
ablation still fails, demonstrating that the expert reduction repair is also
needed. Frozen intermediates, forced routes and FP32-router experiments are
diagnostic controls and are excluded from acceptance.

The floating-state reference is **repaired Separate Eager**. Repaired Separate
and Mixed execute the same arithmetic, while the original native Verify Graph
is an additional acceptance/bonus control. This is not a claim of equality to
the original nondeterministic runtime's floating states, and the failed original
reports are retained unchanged.

After this repair, both nine-case Eager lanes pass: **18/18**, with identical
proposals to the previous experiment, identical acceptance/bonus decisions,
maximum absolute state difference **0**, and passing padding/ownership/write
checks. All 18 separate-repeat controls also have zero state difference. The
matrix retains the previous P chunks, padding, circular read/write wrap,
continuation and request/state mapping checks. Native Draft and Verify Graph
hits remain required.

Only after both full Eager gates passed, `mixed_graph.py` captured the actual
mixed target body. Capacities are P32/P128/P512 with D3×8; exact P valid length
is also an eligibility condition. P23 with capacity 32 and P32 with capacity 32
therefore use distinct graph entries. Private P/V attention backends retain
native paged-attention math. FlashInfer uses its graph wrappers, GDN metadata
is refreshed in place into pinned buffers, and capture-time FLA chunk metadata
is retained to keep graph pointers alive. Metadata topology, shape, dtype and
buffer removal checks reject unsupported changes before replay. The prototype
does not fall back to eager Verify.

The two Graph lanes also pass **18/18**, with maximum absolute state difference
**0**, identical acceptance/bonus decisions and passing boundaries. The graph
schedule is P32/32/32/128/128/512/512/23/23, with capacities
32/32/32/128/128/512/512/32/32. It deliberately exercises **10 reused replays**
and eight captures across the two lanes. The proposals are exactly the prior
natural/rejection proposals. Different P chunk schedules make this a
correctness/reuse test, not a performance comparison. Additional P23→P32 and
P65→P128 padded graph cases pass; the latter enables Mamba tracking and uses
physical P state slot 10 and D state slots 1/4/7, independently of request IDs.

The first full Graph attempt ran out of memory in the private validator before
capture. It retained dead backend cycles and staged a redundant 1.46 GiB SSM
restore allocation. Collecting those private objects and copying the CPU
snapshot directly into the existing GPU pool fixed the validator without
changing model/cache budgets. This failed attempt remains in the evidence.

Regression coverage passes 15 checks: six mixed-layout CPU tests, four fixed
GEMM/graph/adapter tests, one native GDN direct-commit test and four native GDN
spec-fold tests. The standalone driver can rerun the full sequence on a fresh
private server:

```bash
python test/manual/pdmux/run_mixed_eager.py \
  --base-config /validation/existing-baseline-env.json \
  --model-path /models/QuantTrio/Qwen3.5-35B-A3B-AWQ \
  --draft-model-path /models/z-lab/Qwen3.5-35B-A3B-DFlash \
  --output-dir /validation/mixed-stable-graph-new \
  --numerics stable \
  --proposals-from /validation/previous-mixed-eager \
  --mixed-graph
```

`--proposals-from` reads the previous `native-reductions-natural-report.json`
and `native-reductions-rejection-report.json`, validates anchors and replays
their full proposal tensors. The graph driver first requires both full stable
Eager lanes to pass. It rejects `--mixed-graph` with `--smoke` or original
arithmetic. Without `--mixed-graph`, `--smoke` runs one case per Eager lane and
a separate frozen-intermediate diagnostic; this diagnostic is never a gate.

No TTFT, TPOT, output-gap or throughput gain is claimed from these correctness
runs. A later performance experiment must compare the unchanged original
baseline, repaired separate arithmetic and Mixed Graph separately, so numerical
kernel costs are not confused with mixed-execution gains. The fixed-tile
projections may be slower than native GEMMs. Scheduler profitability/admission,
long contexts, finish/admission transitions and independently chained mixed
generation remain future work. D prefixes in the full-model matrix stay below
the 256-token tracking boundary; native component tests cover tracking/fold/
rollback and the extra P65 case covers actual-model P tracking. KV snapshots
cover allocated request pages and sentinel zero, not the entire unallocated pool.

Current repair results and retrieval status are recorded in
[`qwen35_awq_mixed_state_fix_results.json`](qwen35_awq_mixed_state_fix_results.json).
The full Eager/Graph matrices, extra P65 case, audits and 15 regressions were
confirmed from remote reports for the pre-shutdown snapshot. The server was
then powered off while the fresh standalone CLI smoke was running. Its final
exit status, GPU cleanup and raw-report/archive retrieval remain unconfirmed.
The JSON records confirmed observations and this retrieval limitation; it does
not claim a completed standalone smoke or a downloaded raw archive.

### Code adaptation while the A100 is off

The current revision adds stricter offline contracts around the validated
arithmetic. It does not modify the fixed GEMM or Marlin expert kernels. New
Full-model A100 execution of this revision is pending; the earlier 18/18 Eager
and 18/18 Graph observations describe their retained source snapshots. Local
WSL operator and regression results are recorded below.

`--prepare-only` runs on a CPU host, including Windows, without importing
torch, starting a server or loading model weights. It checks the existing
baseline configuration, validates complete D3×8 proposal tensors, freezes both
proposal inputs and records their digests, candidate source hashes and all four
jobs in `plan.json`. It exits successfully for preparation only and records
`gpu_executed=false` and `correctness_passed=null`.

```bash
python test/manual/pdmux/run_mixed_eager.py \
  --base-config /validation/existing-baseline-env.json \
  --model-path /models/QuantTrio/Qwen3.5-35B-A3B-AWQ \
  --draft-model-path /models/z-lab/Qwen3.5-35B-A3B-DFlash \
  --output-dir /validation/mixed-plan-new \
  --numerics stable \
  --proposals-from /validation/previous-mixed-eager \
  --mixed-graph --prepare-only
```

Run the full command without `--prepare-only` on the Linux A100 host, using
another fresh output directory. Plans contain host-specific paths and are not
a portable resume format. The driver validates inputs before starting its
private process and records both original and effective launch environments.
Explicit defaults match the existing launch script; memory/cache budgets are
retained. Conflicting dtype, block size, deterministic mode, graph mode or
PDMux settings cause an error instead of an implicit experiment change.

Before Mixed Graph capture, both full Eager reports must match the current
candidate/harness source hashes, exact P schedule, original tolerances and
natural/rejection lane identities. Each must explicitly include all nine
separate-repeat comparisons, native Draft/Verify Graph hits, acceptance/bonus,
state metrics and write-boundary checks, and successful cache cleanup. An
absent check, nonfinite metric, diagnostic-only report or stale source report
cannot qualify. Legacy passing reports remain historical evidence and cannot
authorize capture for this new revision. Proposals are checked again before
submission and inside the worker, in addition to the existing anchor checks.
When no previous proposal directory is supplied, Graph jobs replay the matching
Eager lane's newly recorded proposals; they do not generate an independent
Graph fixture. Supplying the original directory is still required to compare
with the previous experiment's exact proposals.

Graph input validation now checks P and V buffers before copying either batch.
It rejects removed/changed buffers, changed host control branches and changed
speculative metadata types or block semantics. GDN metadata also checks device
identity. Embedding overrides, real multimodal/LoRA inputs and non-linear or
ragged speculative layouts are outside this prototype and are rejected before
stateful execution. Unsupported layouts require a separate capture or the
native Verify Graph in a future integration; there is no eager Verify fallback.

Worker reports and triggers are published atomically. Worker errors, cache
cleanup failure and nonfinite results leave a terminal failed report. The
driver preserves a terminal summary on runtime failure and sends termination
to its private process group even if the launcher exits before its workers.
An Eager-only run records `mixed_graph_gate=null`; successful preparation or
Eager-only execution cannot be reported as a Mixed Graph pass.

The GPU-free regression suite is runnable directly:

```bash
python test/registered/unit/spec/test_mixed_validation.py
```

Its 14 tests pass locally, including a real `--prepare-only` subprocess and
private-hook failure/cleanup paths with mocked runtime collaborators. These
tests establish orchestration/contract behavior, not GPU numerical correctness.

The current source snapshot additionally passes **31/31 tests, no skips**, on
local WSL2 with RTX 4060 Laptop 8 GB, PyTorch 2.13.0+cu130, CUDA 13.0 and
Triton 3.7.1:

| File | Passed |
| --- | ---: |
| `test_mixed_eager.py` | 7 |
| `test_mixed_marlin.py` | 5 |
| `test_gdn_replayssm_direct_commit.py` | 1 |
| `test_gdn_replayssm_spec_fold.py` | 4 |
| `test_mixed_validation.py` | 14 |

The GPU tests execute actual fixed FP16 GEMMs, CUDA Graph replay and native
GDN commit/fold/tracking/rollback kernels, without numerical mocks or changed
tolerances. CPU layout, branch and report checks are included in the 31 total.
The isolated environment follows the repository's PyTorch 2.13 pin: reusing the
older local 2.14 runtime initially failed to load the prebuilt sglang-kernel ABI.
Initial import/ABI/dependency failures are retained with the final logs and
environment package list. Source hashes match the working-tree files, and the
post-run GPU compute-process list is empty.

These tests do not load the 35B checkpoint or exercise its complete
`MixedTargetGraph` body. After power-on, run the full fresh-server Eager→Graph
command with the same saved proposals on A100 before performance A/B or
scheduler integration. Formal serving still uses its original Draft/Verify
Graphs.

### Local WSL operator A/B: repair cost versus batching benefit (2026-10-25)

The operator experiment is complete on RTX 4060 Laptop 8 GB under WSL2,
PyTorch 2.13.0+cu130, CUDA 13.0 and Triton 3.7.1. Formal serving and the model,
TP1, DFlash block8 and concurrency4 baseline configuration are unchanged.
The synthetic shape contains one P request plus three D requests with eight
verify rows each; it does not execute a serving workload or DFlash proposals.

The benchmark uses dimensions and quantization exclusions from the
[checkpoint config at revision 2f4fac0](https://huggingface.co/QuantTrio/Qwen3.5-35B-A3B-AWQ/blob/2f4fac011fc06e736782b0b59394577790da7d79/config.json):
hidden size 2048, 256 routed experts, top8, intermediate size 512, AWQ4,
group128 and zero points. Synthetic weights are packed by the native AWQ test
helper. Projections, router and shared expert retain the checkpoint's FP16
exclusions. Trained checkpoint weights and measured checkpoint routing are
not used. `hot32` adds a synthetic bias toward 32 experts; `uniform` has no
such bias and does not prescribe an exact uniform expert assignment.

There are **55 cases**, covering five P/capacity pairs (23/32, 32/32, 65/128,
128/128, 512/512), eight operator groups and two routing profiles for the
expert-containing groups. Each case has six lanes, 12 trials and 64 replays
per trial, with balanced lane order. CUDA-event trial means are retained in
the raw data; reported times below are their medians. Capturing, JIT,
quantization and warmup are outside timing. The same weights and input values
are used across lanes within each case.

The primary native Separate, repaired Separate and Mixed lanes each replay
**one CUDA Graph**, matching graph submission count. Separate executes P and D
operators independently inside that graph. Native means original operator
kernels, not the whole serving baseline. Mixed includes packing and padding
of the inputs consumed by the tested operators. Auxiliary native/repaired
two-graph lanes and a prepacked Mixed lane measure submission and packing
effects separately. Static input refresh and CPU graph metadata are excluded.

All four repaired output controls pass in all 55 cases at the original
`rtol=0.02, atol=0.002`; their maximum absolute difference is **0.0**. The native
full-K Marlin CUDA Graph replay probe also passes separately. These checks
cover operator outputs, not full-model acceptance, KV, Conv or SSM state.

For the GDN-sized token-local combination, times in milliseconds are:

| Routing | P / capacity | Native Separate | Repaired Separate | Mixed with packing | Mixed versus native |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniform | 32 / 32 | 2.752 | 2.746 | 1.849 | -32.8% |
| uniform | 128 / 128 | 3.555 | 3.780 | 2.735 | -23.1% |
| uniform | 512 / 512 | 5.459 | 8.373 | 7.459 | +36.6% |
| hot32 | 32 / 32 | 1.196 | 1.192 | 0.647 | -45.9% |
| hot32 | 128 / 128 | 1.604 | 1.860 | 1.601 | -0.2% |
| hot32 | 512 / 512 | 3.622 | 5.360 | 5.021 | +38.6% |

Negative percentages mean faster. At uniform P128, the repaired Separate
path costs an extra **0.225 ms**, while mixing saves **1.046 ms** against that
repaired path, leaving **0.821 ms net savings** against native. At P512, repair
costs **2.914 ms** and mixing recovers only **0.914 ms (31.4%)**, leaving it
**2.000 ms slower**. Removing Mixed packing at P512 reduces 7.459 ms to
7.401 ms, still well above native 5.459 ms; packing is not the main loss there.
The two-graph controls closely track their one-graph counterparts (uniform
P128 native 3.555 ms, repaired 3.781 ms), so the main savings are not explained
by avoiding one host graph launch.

Full-attention-sized token-local combinations show the same boundary:
uniform P32/P128 improve 32.9%/25.1%, while P512 regresses 34.6%; hot32 P512
regresses 38.1%. Hot32 P128 is inconclusive in both combinations: Mixed wins
only 6/12 GDN-sized and 7/12 attention-sized paired trials, with median-time
differences around 0.2%. Padding matters: P65 into capacity128 improves
15.7%/17.7% for uniform routing, but regresses **13.7%/11.1%** for hot32.
P23 into capacity32 still improves in both profiles. Use the live-router
combinations for padding interpretation: the isolated routed-expert operator
freezes routes and uses zero IDs/scores for padded rows.

The larger-shape loss appears in both repaired computation paths. At P512,
isolated QKVZ FP16 GEMMs take 1.228 / 1.965 / 1.862 ms (native / repaired /
Mixed), and uniform routed MoE takes 3.493 / 5.311 / 4.603 ms. Dispatch auditing
confirms native atomic reduction without full-K and repaired non-atomic
full-K with block-M8. Native selects block-M32 at P512, compared with block-M8
at P128. Thus the measured repair cost includes the fixed FP16 tile and the
combined Marlin full-K/non-atomic/block-M8 choice; it cannot be attributed
solely to atomic reduction overhead. No arithmetic implementation was changed
for this performance experiment.

These combinations execute router, routed experts, shared expert and then
projection operators serially. They are not a faithful transformer-layer
timeline and do not overlap the shared expert on another stream. They omit
Attention/GDN recurrence, state commits, norms/residuals, the first unquantized
expert layer, logits and Draft/Verify execution. Neither a completed
`MixedTargetGraph` body nor end-to-end throughput is measured. Timing p95 in
the JSON describes trial means, not token output intervals. Laptop results
cannot establish A100 profitability.

The evidence supports testing P32 and selected P128 shapes next, with actual
routing and padding occupancy in the admission decision. Enabling every
capacity, especially P512, would not be supported by these results. A100
full-model correctness and throughput A/B remain pending power-on; scheduler
integration and changes to the numerical kernels remain outside this run.

Reproduce in the existing private WSL environment with a fresh output path:

```bash
cd /workspace/mixed-adapt-20261025/source
export PYTHONPATH=python:test/manual/pdmux
export CUDA_HOME=/workspace/conditional-graph-20261019/cuda13
export PATH=/workspace/mixed-adapt-20261025/venv/bin:$CUDA_HOME/bin:$PATH
export CC=gcc-12 CXX=g++-12 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
python test/manual/pdmux/bench_mixed_operators.py \
  --config /workspace/mixed-operator-ab-20261025/checkpoint-config.json \
  --output /workspace/mixed-operator-ab-20261025/repeat-new \
  --trials 12 --repeats 64
```

The benchmark is [`bench_mixed_operators.py`](bench_mixed_operators.py).
All 55 compact results and dispatch audits are in
[`qwen35_awq_mixed_operator_results.json`](qwen35_awq_mixed_operator_results.json).
Local raw trial JSON, logs, frozen dependencies, checkpoint-config revision,
source snapshots and checksums are retained under
[`.tmp/mixed-operator-ab-20261025`](../../../.tmp/mixed-operator-ab-20261025),
including `operator-ab.csv`, `raw-artifacts.tar.gz` and `raw-artifacts.sha256`.
The post-run GPU compute-process list is empty.

### Fixed-reduction tile tuning on local WSL (2026-10-25)

This follow-up measures whether P512's repair cost can be reduced without
relaxing `rtol=0.02, atol=0.002`. It uses the same WSL GPU, runtime, checkpoint
geometry and synthetic-weight operator harness described above. It retains
FP16 inputs/output, FP32 accumulation, K64 and two pipeline stages in the
fixed GEMMs. Marlin retains full-K and disables atomic reduction throughout.
No serving/model launch setting changes.

The opt-in `linear_tile="projection"` profile in `mixed_marlin.py` uses only
immutable weight output dimensions: N>=8192 selects M32/N128/eight warps;
N>=1024 selects M32/N64/four warps; smaller matrices keep M16/N32/four warps.
It never selects a tile using P, D or graph capacity. Keeping a small router
tile also avoids its observed performance loss with wider tiles. Defaults
still select the original M16/N32/four-warp implementation.

`fused_marlin_moe` now accepts an explicit `deterministic_block_size_m`, used
only when batch-invariant arithmetic is active. Its default remains eight;
native mode retains its existing block selection. The offline adapter can
pass a constant 16 or 32 for all P/D/Mixed shapes in a candidate. It restores
the original function and arithmetic-mode hooks on exit, including failure.
No C++ kernel or reduction algorithm was changed. The new Marlin source is
also included in the full-model validation source gate.

```python
with stable_marlin(model, linear_tile="projection", moe_block_size=16):
    # Offline candidate only; validate full-model state on A100 before use.
    ...
```

The FP16 sweep tries seven BM/BN/warp combinations. The MoE sweep compares
fixed M16/M32/M48/M64 at P128/P512 under uniform and hot32 routing. Initial
observations support M16 as a broader candidate, while M32 is a P512 candidate
that can regress P128. Full-K stripes already give each output tile complete
K ownership; M8 is not required by that scheduling rule. Larger M tiles can
reduce repeated expert-weight loading/dequantization, but increase padded
work for sparsely occupied experts. This is a code-based explanation of the
timing trend, not a kernel-level bandwidth measurement. M32 also moves the
C++ dispatch from its small-batch to its large-batch thread-config candidates,
so its cost difference is not an isolated M-axis change.

M16 candidate validation interleaves nine single-graph lanes: native
Separate, frozen old Separate/Mixed, FP16-only Separate/Mixed, Marlin-only
Separate/Mixed, and combined Separate/Mixed. Mixed includes packing/padding.
The M32 follow-up replaces the FP16-only pair with a combined M16 control,
so M16/M32 are compared on exactly the same weights, routes and inputs.
The frozen reference imports the exact `mixed_marlin.py` retained from the
previous operator experiment; its SHA256 is checked against that experiment.
Every candidate is compared to that reference and between Separate/Mixed;
router IDs require exact equality. Reused graphs also receive changed inputs
in the new CUDA regression tests. These tests do not cover full-model
acceptance, KV, Conv/SSM or ReplaySSM state, which remain A100 work.

The completed matrix contains 80 M16 cases (the earlier 55-case matrix plus
25 extra FP16-only random-input repeats labelled hot32), six P512 M32 cases
with same-input M16 controls, and 12 P257/P511-to-capacity512 boundary cases.
All **98 final cases** pass both old-reference and Separate/Mixed checks.
The eight exploratory sweep cases pass too. The maximum valid-output absolute
difference across all repaired operator variants is **0.0**, with exact route
IDs and unchanged tolerances. The three relevant regression files pass
**28/28 tests, no skips**: mixed Marlin seven, mixed Eager seven, GPU-free
validation 14. The added CUDA tests replay fresh inputs through larger-tile
FP16 graphs and full-K M16/M32 expert graphs. Post-run compute processes are
empty. Formatting, lint and source/archive checksum checks also pass.

For P512, the M32 experiment compares all variants on identical tensors
(seed43, 18 balanced trials x 64 replays per lane). Times below are median
CUDA-event trial means in milliseconds; both M16/M32 include the projection
profile and Mixed packing:

| Token-local bundle / routing | Native Separate | Old Separate | Old Mixed | M16 Separate / Mixed | M32 Separate / Mixed |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN-sized / uniform | 5.529 | 8.373 | 7.437 | 5.892 / 4.865 | 6.093 / 4.568 |
| Attention-sized / uniform | 5.195 | 7.867 | 6.959 | 5.525 / 4.561 | 5.724 / 4.266 |
| GDN-sized / hot32 | 3.641 | 5.400 | 5.091 | 3.940 / 3.548 | 3.938 / 3.456 |
| Attention-sized / hot32 | 3.368 | 4.871 | 4.570 | 3.614 / 3.227 | 3.603 / 3.192 |

At uniform GDN-sized P512, old repair adds 2.844 ms over native Separate.
The M16 candidate reduces that remaining cost to 0.363 ms (87.2% removed),
and M32 to 0.563 ms (80.2% removed). M32 then saves 1.525 ms by mixing,
covering its remaining repair cost and finishing **17.4% faster than native**.
Relative to old repaired arithmetic, its Separate/Mixed times improve
**27.2% / 38.6%**. The attention-sized counterpart is 17.9% faster than native
in Mixed. Under hot32, M32 Mixed gains about 5.1%/5.2% against native.

This comparison also shows the tradeoff: at uniform P512, M32 Mixed improves
another 6.1%/6.5% over same-input M16, but M32 Separate regresses 3.4%/3.6%
against M16. The broader M16 run independently improves P512 Mixed by
11.6%/12.0% against native under uniform routing and 4.7%/3.1% under hot32.
These two seed cohorts should not be compared by their absolute times;
the M32 experiment includes its own M16 controls for that reason.

The performance boundary remains relevant even after numerical repair is
cheapened. In the boundary run, P257 padded to capacity512 still regresses
about 34.1%/32.7% against native under hot32, although every output check
passes. At small P, the broader run's attention-sized hot32 P23/P32 Mixed
times regress 6.5%/4.9% against old Mixed. In the expert sweep, uniform P128
M32 Mixed is 2.370 ms versus old M8 1.891 ms and M16 1.810 ms. The candidate
therefore remains opt-in. Actual occupancy/routing and A100 end-to-end
measurements must determine admission and any capture-bucket choices.

To reproduce a P512 comparison with an explicit M16 control:

```bash
# Use the same private WSL environment/PYTHONPATH/CUDA_HOME as above.
python test/manual/pdmux/bench_mixed_reduction_tuning.py \
  --config /workspace/mixed-operator-ab-20261025/checkpoint-config.json \
  --reference-linear /workspace/mixed-reduction-tuning-20261025/source-before/python/sglang/srt/speculative/mixed_marlin.py \
  --stage validate --output /workspace/mixed-reduction-tuning-20261025/repeat-new \
  --operators routed_moe,gdn_token_local_bundle,attn_token_local_bundle \
  --cases 512:512 --linear-tile projection --moe-block 32 --control-block 16 \
  --seed 43 --trials 18 --repeats 64
```

[`bench_mixed_reduction_tuning.py`](bench_mixed_reduction_tuning.py) contains
the sweep, ablations and correctness checks. Compact results are in
[`qwen35_awq_mixed_reduction_tuning_results.json`](qwen35_awq_mixed_reduction_tuning_results.json).
Raw trials, source snapshots, frozen old reference, regression logs,
environment package versions and SHA256 archive are retained in
[`.tmp/mixed-reduction-tuning-20261025`](../../../.tmp/mixed-reduction-tuning-20261025).
Formal serving and the adapter's legacy defaults remain unchanged. No A100
full-model throughput, TTFT, TPOT or acceptance/state result is claimed here.

### Offline adaptive selection with CUDA Graph replay (2026-10-25)

The opt-in selector compares six pre-captured actions: legacy/M16/M32 repaired
arithmetic, each with Separate or packed Mixed execution. Each action is one
CUDA Graph. Legacy retains the original fixed M16/N32/K64 FP16 projections
and full-K/non-atomic Marlin M8. M16/M32 use the projection profile from the
preceding experiment. The selector does not modify native Draft/Verify serving
or the legacy adapter defaults, and does not call the model eagerly on replay.

The CPU cost table uses an exact capacity bucket and interpolates measured
costs in P and effective expert count. P/capacity represents useful-token and
padding occupancy. Routing diversity is estimated from valid P/D top-k IDs;
the histogram estimator removes same-token top-k pairs from its collision
denominator so sample size is comparable across P. The cache rounds effective
experts to bins of eight. This is a compact routing proxy, not a model of all
expert correlations. The full 17-value summary additionally records active
experts, peak share, P/D overlap and padded block counts at M8/M16/M32.
Physical padding costs come from the measured P/capacity curves, not from
counting padded tokens as useful routing observations.

`mixed_selection_graph.py` captures the histogram/reduction kernel into each
candidate graph. CPU selection consumes a previous completed route summary.
Two pinned host slots receive asynchronous D2H copies, and the CPU reads a
slot only after its CUDA event query succeeds. Replay performs no `.item()`,
CUDA synchronization, graph capture or eager model call. A changed context
immediately submits a fresh summary; pending data from another context or
data older than the age limit is discarded. Missing/expired summaries use a
shape-only prior. The default samples D2H every eight replays and accepts
feedback up to 16 logical steps old; the histogram itself still runs in every
candidate graph. The routing-transition validation samples every replay with
an age limit of two. A step/context must be a CPU integer.

CPU decisions keep the current graph when its predicted cost is within 2%
or 3 us of the predicted best. Unknown calibration buckets choose the captured
legacy Separate action. All six graphs must already be validated before a
bank is constructed. Graphs retain their input/output storage and fixed kernel
configuration; a new geometry needs its own offline bank. This implements
host selection of existing CUDA Graphs with asynchronous feedback. It does
not use device-conditional CUDA Graph nodes or replay the complete model's
`MixedTargetGraph` body.

The independent evaluation harness is `bench_mixed_selection.py`. It first
calibrates 42 operator cases: two token-local families, P23/P32/P65/P128/P257/
P430/P512, and uniform/hot32/hot128 synthetic routing. The cost table is frozen
before 48 evaluation cases using independent random inputs, P16/P31/P96/P127/
P300/P448/P511/P512, and uniform/hot32/hot64 routing. P512 repeats a calibrated
shape with fresh inputs; the other seven P values are new. Hot64 is absent
from calibration. Every candidate is checked against frozen old repaired
Separate, and integer route IDs require exact equality. Sixteen additional
steps test graph reuse while routing changes, using lagged GPU feedback rather
than profile labels. Evaluation timings never update the cost table.

Calibration uses 12 balanced trials x 16 replays across six actions; evaluation
uses 14 x 16 across those actions plus the adaptive replay path. Every graph
includes the same routing-summary kernel, and Mixed includes input packing.
JIT, capture, quantization and input refresh are outside timing. CPU cache-hit
and cold lookup costs are measured separately, as are replay submission and
bare/statistics/adaptive graph controls. The workload is an equal-weight
operator matrix, not a full-model throughput trace. Its per-case oracle is
the fastest of the six measured fixed candidates and is used only for reporting
selection error.

The completed run passes all 42 calibration cases, 48 independent evaluation
cases and 16 routing-transition steps at the original tolerances. Every
candidate and adaptive output comparison has maximum valid-output absolute
difference **0.0**, including exact route IDs. Five CPU policy tests and three
real CUDA Graph tests pass. The latter check changed routes, pinned-slot reuse,
stale/context-mismatched feedback, and replay with capture, synchronization and
`Tensor.item` explicitly prohibited. No serving process is started, and the
final GPU compute-process list is empty.

In the main 48-case matrix, the adaptive path totals 120.515 ms versus
125.792 ms for fixed M16 Mixed and 119.590 ms for the per-case six-candidate
oracle. All six fixed candidates in that matrix include the same histogram
kernel for selection-quality comparison. The adaptive-oracle total gap is
0.77%, median per-case gap 0.89%, and worst gap **8.42%** (GDN-sized P127,
unseen hot64 routing: M16 Mixed selected, M32 Mixed fastest). This demonstrates
the interpolation model's limits; it does not guarantee an optimal choice for
every routing distribution. Of 48 choices, six use legacy Mixed, 20 M16 Mixed,
four M16 Separate and 18 M32 Mixed. P300/capacity512 under concentrated routing
selects Separate, whereas P511/P512 select Mixed.

A second, stricter run repeats all 48 input fixtures with the fixed baselines'
histogram and feedback completely removed. The adaptive path retains both:

| Path | Sum of per-case median GPU replay time |
| --- | ---: |
| Legacy repaired Mixed, no statistics | 174.963 ms |
| Fixed M16 Mixed, no statistics | 124.064 ms |
| Adaptive, including statistics and feedback | 119.612 ms |

Thus the measured net total-time gain is **3.59% against fixed M16 Mixed**
and **31.64% against legacy Mixed** on this artificial operator mix. All 48
strict-baseline comparisons pass, and the frozen policy makes the same choices
as the main run. The two runs use separate balanced measurement schedules;
do not compare one run's absolute adaptive time with the other run's oracle.
These are steady-input replay measurements after feedback warmup. Transition
steps establish correctness and graph reuse, not dynamic-workload throughput.

CPU cache-hit lookup median is 0.59 us, cold lookup median 5.84 us (worst
per-case p95 23.21 us). The complete adaptive replay submission adds about
14.70 us of CPU time over a direct graph already containing statistics,
including polling, feedback processing and asynchronous-copy submission.
Independent small statistics graphs take 10.12-25.89 us including graph launch;
empty-graph controls are retained. In real operator controls, adaptive versus
the same selected graph with statistics removed adds a median **18.54 us**
of GPU-event time (maximum 28.90 us). These quantities describe different
boundaries and should not be added together as an end-to-end latency estimate.

Reproduce in the same private WSL environment described above:

```bash
python test/manual/pdmux/bench_mixed_selection.py \
  --config /workspace/mixed-operator-ab-20261025/checkpoint-config.json \
  --reference-linear /workspace/mixed-reduction-tuning-20261025/source-before/python/sglang/srt/speculative/mixed_marlin.py \
  --output /workspace/mixed-selection-20261025/repeat-new \
  --seed 20261025 --trials 12 --eval-trials 14 --repeats 16
```

Cost tables include the device/capability, model geometry, torch version,
configuration digest and numerical/selector source digests. Loading with a
different expected context fails; the RTX4060 table must be recalibrated on
A100. Runtime code is in
[`mixed_selection.py`](../../../python/sglang/srt/speculative/mixed_selection.py)
and [`mixed_selection_graph.py`](../../../python/sglang/srt/speculative/mixed_selection_graph.py).
The harness is [`bench_mixed_selection.py`](bench_mixed_selection.py), and
[`qwen35_awq_mixed_selection_results.json`](qwen35_awq_mixed_selection_results.json)
contains compact results and limitations. Raw trials, calibration, strict
fixed-baseline verification, source snapshots, tests, package versions and
archive SHA256 are in
[`.tmp/mixed-selection-20261025`](../../../.tmp/mixed-selection-20261025).
The interrupted first smoke is retained as incomplete evidence; the second
smoke and final full runs pass. Formal serving continues to use native
Draft/Verify Graphs. Checkpoint weights, whole-model acceptance/KV/Conv/SSM,
TTFT/TPOT and serving throughput remain untested in this local operator run.

### Repository packaging checks

Before pushing, CUDA tests were placed under `test/registered/kernel/speculative/`
(`test_eager_seq_lens_cpu.py`, `test_mixed_marlin.py`, and
`test_mixed_selection_graph.py`) and registered with the kernel CI suite.
The CPU selector test now registers CPU CI without importing the serving
runtime. Revalidation passes 70 GPU-free tests and 16 CUDA tests on local WSL.
Four files received import-order-only cleanup; their non-import AST bodies
were checked unchanged. Historical experiment hashes and test paths describe
their retained snapshots. These packaging edits do not introduce new numerical
kernel, configuration or serving-default changes.

The two largest generated result JSON files use compact JSON to meet the
repository's 1500 KiB limit, with parsed contents checked identical before and
after formatting. Raw logs and caches remain local. Pre-push validation paths,
counts and working-tree source hashes are recorded in
[`qwen35_awq_push_validation_results.json`](qwen35_awq_push_validation_results.json).

### Full-model target-stage benchmark

`run_mixed_eager.py --mixed-graph --benchmark-trials 12` first runs both fresh
stable Eager gates, then measures each validated Mixed Graph fixture. Retain
the model paths, baseline environment JSON and optional compatibility script
from the full validation command above, and use a fresh output directory.

The three paths are native Separate (native Prefill plus the original Target
Verify Graph), repaired Separate (one captured graph containing both forwards),
and repaired Mixed Graph. The additional Separate Graph must match Mixed
outputs and target pool writes before timing. Native Verify must report a
graph hit on every measured call. Each trial starts from the same restored
state, with rotated/reversed execution order and three discarded warmups.
Reports retain all GPU-event and wall-time samples. Preparation is included;
snapshot restoration, graph capture, and final snapshot comparisons are not.

These are complete target-model forward measurements, not serving throughput:
Draft, acceptance/commit, admission, request completion and cache scheduling
remain outside timing. The existing `diagnostic_wall_s` fields still cannot
be used as performance measurements. Use `cases[].target_benchmark` only when
the containing Graph report and its correctness gate both pass.

The driver retries temporary `/health_generate` failures while native warmup
finishes; a successful server-info response alone does not establish readiness.

### Private live Mixed experiment with preserved serving settings

Correction: the historical runs below explicitly used chunk32768. That is
not the intended unchanged native chunk configuration. The driver and launcher
now leave chunk size unspecified unless supplied by the existing base config,
environment or an explicit CLI override. Only max-prefill-tokens defaults to
32768. The current engine resolves an unspecified chunk to 4096 on A100 40GB;
confirm the actual value in server-info before comparing groups. Historical
throughput results do not establish gains under this corrected configuration.

`run_live_mixed_target.py` applies the same chunk, Prefill budget, CPU overlap
and HiCache settings to every group. It inherits the base configuration unless
explicitly overridden. For the corrected A100 SWE replay, append:

```bash
--swe-workload /path/to/swe-multiturn.json \
--max-prefill-tokens 32768 \
--cpu-overlap 1 --hicache 0 --groups mixed --trials 1
```

Retain the required model, base-config, compatibility, output and full-model
gate arguments. The hook preserves actual native chunk lengths and native CPU
overlap. Only validated, profitable P1+D3 block8 pairings use a joint Graph;
other batches follow the native path. Existing Draft and independent Verify
Graphs remain active. This is a private experiment, not a serving flag.

The [corrected SWE report](qwen35_awq_mixed_preserve_swe_results.md) supersedes
the old chunk128/overlap-off interpretation. It measures 305.44 tok/s versus
preceding controls of 295.10/297.44, with only two joint replays and no measured
capture. One pass with reused controls cannot establish a stable improvement.
First-shape admission makes warmup take 139.83 seconds; this excluded cost and
the rejected greedy-mismatch shape remain material limitations.

The [kernel optimization report](qwen35_awq_mixed_optimization_results.md)
records an opt-in P1576 candidate using `LIVE_MIXED_LINEAR_TILE=wide` and
`LIVE_MIXED_MOE_BLOCK=32`. Its validated bounded configuration also sets
`LIVE_MIXED_MIN_TOKENS=1576`, `LIVE_MIXED_CAPACITY_LIMIT=1576` and
`LIVE_MIXED_GRAPH_LIMIT=1` as strings in the base-config JSON. These eligibility
limits preserve the native chunk shape. Omit `MIXED_TARGET_PROFILE_DIR` for
unprofiled runs. Legacy tile defaults remain unchanged.

This candidate reduces the profiled target-stage median from 220.04 to
119.47 ms and executes a joint Graph in the real request loop. End-to-end
throughput improvement is not established. A broad MoE32 attempt exhausted
memory in a subsequent large native Prefill, so the bounded candidate must
not be treated as validated full-workload admission or a production memory fix.

## Native-chunk tail-fill follow-up

See [joint execution results](qwen35_awq_tailfill_joint_results.md) for the
chunk4096/max-prefill32768 A100 run, the opt-in fixed-arithmetic reference,
the pipeline projection candidate and the bounded P2184 admission settings.
The single-pass +2% serving result is not a demonstrated stable gain.
