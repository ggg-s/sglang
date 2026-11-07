# Generic PDMux segmented and standard BCG

The implementation reuses a model's existing prefill BCG body and records
decoder-module boundaries during capture. There is no Qwen architecture
allowlist in interval dispatch. Models need an ordered `layers` collection and
a working eager split fallback. ModuleList and ModuleDict are supported.
Missing, out-of-order or repeated layer execution declines partial replay
while preserving full BCG. This does not certify every model/backend pair.

The initial serving profile is **text, plain TP, PP=DP=CP=EP=1, no speculative
decoding/MTP**. Disable the prefill backend to retain existing eager profiles.

## Execution and ownership

- `run_split_prefill` selects an interval graph or the existing eager split.
  The scheduler still chooses the layer budget and submits decode between
  intervals; model code does not recursively call decode.
- Temporary forward hooks mark complete decoder modules and are removed in
  `finally`. Capture records a whole transformer body once per token bucket
  and stream group. Contiguous intervals reuse its layer map instead of
  capturing every possible `(start, end)` pair.
- The first interval includes input/PLE preparation; the last includes PLE
  commit and the transformer tail. Only the final interval invokes the eager
  logits tail. The graph retains cross-layer hidden/HC tensors at stable
  addresses; replay does not depend on Python assignments to model state.
- A continuation has one owner, one static batch, one captured shape/lane and
  an advancing cursor. I/O loading happens once. The scheduler pins the stream
  group until the final interval. A live session cannot change to eager,
  another shape or another graph midway through.
  Consequently an active prefill keeps its original SM allocation even if
  decode shrinks or finishes; repartitioning resumes after the final interval.
- Each interval fully exits graph/model/forward contexts before decode runs.
  The outer model's temporary forward replacement exists only for the final
  tail and is restored on exit. Replay-control kwargs stay out of model APIs.
- Prefill graph storage is separate from decode storage. Graph-pool borrowing
  is disabled for concurrent PDMux lanes. Prefill executable deduplication is
  disabled because its signature omits green-context lane identity. HiCache
  graph events are bound on the first interval, not rewritten between yields.
- Standard BCG uses per-group prefill streams and stream-indexed keys, without
  adding decoder-layer breaks. It replays the entire body and the eager tail.

This preserves layer-level scheduling opportunities; it does not preempt an
already running kernel. Capturing configured stream groups and buckets can
increase startup time and graph memory. Measure that cost on the full model.

## Worker integration regression

The merged `ForwardBatch` lacked `split_index` and `model_specific_states`
declarations. A fresh worker batch raised `AttributeError` before its first
eager split; earlier small-layer fixtures supplied those attributes themselves.
They now have per-batch defaults. A functional test constructs the production
dataclass fields and runs the real split dispatcher, checking independent batch
state and eager/graph dispatch. It fails against `8f4fb9f900` and passes after
the fix: [before](forward-batch-before.log), [after](dispatch-unit.log).

`split_prefill_can_run_graph` is refreshed on each interval and propagated to
`ModelRunnerOutput.can_run_graph`, so segmented replay reports graph use and
the eager fallback resets the flag.

## Real-layer CUDA results

Tested on RTX 4060 Laptop (SM89, 8,188 MiB), Torch 2.14.1+cu130, from
`050f5683330bef795bd7162b009b1f293eb7d193`. All **16 configurations passed
32 dynamic-input pairs each** (512 pairs), with zero tolerance for all checked
outputs/state/routing. Eight timing trials per configuration add 4,096 mixed
pairs, excluding warmup and diagnostics.

Prefill CPU submission time, ms (median of trial means):

| Dtype / tokens | Eager split | Split BCG | Standard eager | Standard BCG |
| --- | ---: | ---: | ---: | ---: |
| BF16 / 65 | 9.394 | 4.330 | 9.193 | 4.265 |
| BF16 / 129 | 9.048 | 4.256 | 10.703 | 4.178 |
| FP8 / 65 | 10.083 | 4.371 | 10.307 | 4.262 |
| FP8 / 129 | 10.254 | 4.456 | 10.161 | 4.407 |

Mixed-schedule CUDA-event elapsed time per pair, ms:

| Dtype / tokens | Eager split | Split BCG | Standard eager | Standard BCG |
| --- | ---: | ---: | ---: | ---: |
| BF16 / 65 | 9.710 | 4.596 | 9.492 | 4.538 |
| BF16 / 129 | 9.333 | 4.514 | 11.058 | 4.437 |
| FP8 / 65 | 10.359 | 4.772 | 10.592 | 4.602 |
| FP8 / 129 | 10.541 | 4.767 | 10.438 | 4.674 |

For FP8 / 65 tokens, the longest uninterrupted prefill call is
**2.340 ms** with split BCG versus
**4.262 ms** with standard BCG.
Both BCG modes substantially reduce submission cost here. Standard BCG's
aggregate time is slightly lower in these medians; the small difference is
not sufficient evidence of a universal winner. Split retains shorter host
submission quanta. Neither value is a measured serving p99 or preemption bound.

Raw data and full trial samples: [BF16/65](bf16-tokens65.json),
[BF16/129](bf16-tokens129.json), [FP8/65](fp8-tokens65.json),
[FP8/129](fp8-tokens129.json), with matching `.log` files.
[Environment/provenance](environment.json) records the collection revision
separately from each GPU report's tested commit. Subsequent import sorting
changes no non-import AST in the production runner/dispatcher.


The fixture runs production GDN → QSA decoder forwards with HC, four-expert
top-k-two MoE, shared dense experts and two PLE layers. Hidden width is 256,
HC count four, expert intermediate width 256, prefill 65/129 tokens and decode
one token. FP8 adds two FP8 expert blocks, 13 FP8 ordinary projections and
pinned FP8 PLE tables. Routers/norm/convolution keep their appropriate BF16 paths.
CUDA math, native alignment/activation and pinned-host UVA lookup are real;
small synthetic weights are shared by both streams.

The harness uses production BCG capture/replay, eager breaks, layer markers,
`run_split_prefill` and `PdmuxBCGExecutor`. PLE prepare/forward/commit and the QSA
live-context indexer bridge use original eager-break definitions. The original
ATen weak-ref implementation is compiled for this Torch ABI; the cached AOT
wheel is incompatible.

Construction, TP1 cache/communication metadata, registry I/O and outer logits
tail are explicit adapters. A complete server runner and checkpoint are not
loaded. QSA has a fixed compressed prefix and canonical ordering of the same
selected set in the test adapter, as explained in the
[earlier decoder report](../qwen38_decoder_cuda_20261107/README.md). Complete
compression/cache progression is outside this fixture.

Each mode alternates two token/hidden-state inputs in stable buffers and checks
15 output/state/routing leaves against serial eager controls. Decode uses its
own graph and high-priority stream. Verification reads graph-owned decode
selection/router tensors, not Python attributes rebound by eager controls.
Snapshot copies are excluded from timed loops. Request/cache resets are the
same across modes; this is not a progressive autoregressive serving workload.

## Timing definitions

CPU checks: **48 passed** across 18 BCG tests (with isolated service-import
dependencies), 13 dispatcher/integration tests, two capture-lifecycle source
method tests, three standard-admission source method tests, and 12 targeted TP
tests. The new TP case executes 32 two-interval trajectories with real two-rank
Gloo reductions, unequal P/D tensor sizes and separate groups. Its graph
segments are CPU callables, so it checks interval/collective order, not NCCL
CUDA graph capture. The old combined speculation/full-server test is not part
of this initial no-speculation profile or the pass count.

Receipts: [BCG](unit-eager-bcg.log), [dispatch](dispatch-unit.log),
[capture](capture-lifecycle-unit.log), [admission](standard-admission-unit.log),
[targeted TP/Gloo](tp-gloo.log). The TP command selects `TestPdmuxTPGroups`,
`TestPdmuxTPAdmission`, `TestPdmuxTPSampling` and
`TestPdmuxTPGloo.test_segmented_bcg_collectives_keep_their_lane`.

Both a missing ForwardBatch cursor and the stale TP finalizer fixture API were
found during integration. The new cursor regression retains its pre-fix
failure; the TP fixture now follows the upstream placement/group interfaces.

### Measurements

- `prefill_cpu_ms`: host time inside prefill forward/replay calls, summed over
  both split intervals. Excludes fixture resets and snapshots.
- `longest_prefill_call_ms`: longest uninterrupted prefill call per pair,
  averaged within each trial. This measures host scheduling granularity, not ITL.
- `cpu_submit_ms`: total mixed-pair host submission time, including fixture
  setup and context switching.
- `mixed_gpu_ms`: CUDA-event elapsed time over the mixed schedule, joining
  decode before the end event. Includes CPU-caused device idle gaps; it is not
  a sum of active kernel durations or a serving TPOT.

Mode order rotates over eight trials of 32 pairs. Reports retain every trial;
tables show medians of trial means. This small model is largely host/launch
bound and uses untuned configs on Windows/WSL. Ratios do not establish
full-model throughput or p99 ITL improvements.

## Reproduction

From the 4060 WSL Git checkout:

```bash
cd /mnt/c/Users/27839/AppData/Local/Temp/qwen38-decoder-source
export CUDA_HOME=/workspace/qwen38-decoder-cuda-20261107/cuda13/nvidia/cu13
export PATH="$CUDA_HOME/bin:/workspace/conditional-graph-20261019/venv/bin:$PATH"
export CXX=/usr/bin/g++-13
export PYTHONPATH=/workspace/qwen38-decoder-cuda-20261107/deps-pinned
export TILELANG_CACHE_DIR=/workspace/qwen38-decoder-cuda-20261107/tilelang-cache-cu13
export MAX_JOBS=1

python test/manual/pdmux/validate_segmented_bcg_cuda.py \
  --dtype bf16 --tokens 65 --rounds 32 --trials 8 --output /tmp/bf16-65.json
python test/manual/pdmux/validate_segmented_bcg_cuda.py \
  --dtype fp8 --tokens 65 --rounds 32 --trials 8 --output /tmp/fp8-65.json
```

Repeat with `--tokens 129`. Dependencies/toolchain match the
[previous environment](../qwen38_decoder_cuda_20261107/environment.json).
Code sync uses origin push, verified Git bundles/object packs and Windows
`git pull --ff-only`; source files are not copied individually. Source hashes
are raw Windows CRLF bytes, verified against the tested Git revision after
newline normalization.

On a complete compatible serving installation, use existing flags:

```bash
python -m sglang.launch_server --model-path "$MODEL_PATH" --tp-size 2 \
  --enable-pdmux --pdmux-prefill-mode layer_split \
  --cuda-graph-backend-prefill breakable
```

Select `--pdmux-prefill-mode standard` for full-body BCG, or
`--cuda-graph-backend-prefill disabled` for either eager baseline. Omit all
speculative-decoding flags. The model must support the corresponding PDMux/BCG
paths; tune buckets and SM groups for available memory.

## Remaining acceptance

The 4060 evidence is **TP1 with ordinary CUDA streams**, not real green-context
partitions. CPU TP/Gloo tests check collective ownership/order; they do not
establish multi-GPU NCCL graph correctness or speedup. Multi-GPU TP, actual
green-context placement, full-model loading and answer/logprob accuracy, long
mixed-request serving, HiCache restore, additional model/backend matrices and
server TTFT/TPOT/p99/throughput remain unverified. No MTP, other speculation,
DP, CP or EP support is advertised for this initial BCG profile.
