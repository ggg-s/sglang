# DSv4 + DSpark + PCP + CPU overlap + PDMux + HiCache + decode attention TP

The adaptation passes the available 4060 regressions. The measured projection
and Python routing paths show no material steady-state regression. Full-model
throughput and latency remain unmeasured: this host has one 8-GiB GPU and cannot
run DeepSeek-V4 or two-GPU NCCL.

Source: `ab9ba6ed7360383c8d5fc6ae1ab20712fb47155c`, pulled from GitHub over SSH
into `/workspace/cp-decode-review-20261027/repo`. Runtime: Python 3.12.3,
PyTorch 2.13.0+cu130, driver 592.82, RTX 4060 Laptop, SM89.
Raw samples, commands, source hashes and passing test outputs are recorded in
`dsv4_dspark_pcp_decode_tp_validation_20261027.json` beside this report.

## Correctness checks

- Compatibility unit suite: **82 passed**, including the 10 decode-view tests,
  actual CUDA graph replay interleaved with prefill, and two-process Gloo.
- Related CUDA suite: **19 passed, 1 skipped**. This covers singleton NCCL lane
  routing, DSpark finalization/KV projection on green-context streams, and
  allocator storage lifetime across streams. The skipped case requires two GPUs.
- The first extended run lacked the venv bin directory on PATH and could not
  launch `ninja` in 12 KV-store cases; that failing process also aborted during
  CUDA cleanup. With PATH corrected, both the allocator-only run (16 passed)
  and combined CUDA run above exited successfully. No allocator code changed.

The small model fixtures omit FlashMLA encoding and HiCache DMA. The full-model
acceptance tests in `test_dsv4_dspark_pcp_pdmux_hicache_tp2.py` still require the
deployment host and target/draft checkpoints.

## Measured adaptation overhead

The benchmark runs the production CP scope/view code with synthetic BF16
projections: 64 heads, head dimension 512, query rank 1024, 8 groups, output rank
1024 and hidden dimension 4096. Each comparison uses the same three GEMMs and
the same CP rank-0 partition. CP2/CP4 are shard sizes on one GPU, without NCCL.
There are 7 alternating-order samples of 60 iterations per mode and batch size.

The table shows wall time per CUDA graph replay, in microseconds. Positive
change means the persistent view took longer.

| CP size | Rows | Legacy slicing | Persistent view | Change |
|---|---:|---:|---:|---:|
| 2 | 1 | 410.07 | 409.87 | -0.05% |
| 2 | 8 | 433.41 | 431.94 | -0.34% |
| 2 | 32 | 458.31 | 458.12 | -0.04% |
| 2 | 128 | 574.10 | 573.70 | -0.07% |
| 4 | 1 | 208.60 | 208.77 | +0.08% |
| 4 | 8 | 228.43 | 228.58 | +0.07% |
| 4 | 32 | 236.22 | 235.21 | -0.42% |
| 4 | 128 | 315.04 | 314.27 | -0.25% |

Eager projection wall-time changes span -0.73% to +1.14%. The measurements do
not establish an end-to-end speedup. Python scope-only median cost falls from
29.8 us to 0.91–0.94 us per layer, because cached views avoid repeated Parameter
rebinding and metadata restoration. CUDA graph replay bypasses this scope.

Separate runtime getter measurements show approximately 0.23–0.24 us extra per
ForwardFlags read and 0.08–0.09 us extra per DP-length read, compared with direct
state access. Reusing CP metadata through `prepare_cp_forward` costs 2.4–2.6 us
per call in the helper fixture; the physical-length setter alone costs about
0.25 us. Split continuations must restore this padded length for correctness.
The preparation helper repeats small CPU conversions and slicing; these
measurements do not indicate a material cost worth changing the continuation
logic in this patch. Full runner timing includes additional batch checks.

Both implementations retain the same contiguous row-projection shard: 32 MiB
at CP2 and 16 MiB at CP4 for these dimensions. Scope warmup followed by 35,000
scope entries shows **zero growth in live CUDA allocation** for both modes.
Column projection shards alias the source weights. The persistent module adds
Python objects rather than another full GPU copy. Cold host timings in the raw
JSON are order-dependent first-use observations, not a startup-speed comparison.

## Feature costs that still matter

Relative to disabling decode attention TP, both the existing slicing path and
the new view path keep a contiguous `wo_b` shard. Its approximate per-layer,
per-rank weight cost is `wo_b.weight.numel() * element_size / CP`, plus scale
shards and the small attention sink. Account for every target layer and draft
stage when sizing memory; the 32/16 MiB values above are fixture dimensions.

Decode attention TP also trades projection work for a TP output reduction per
attention invocation. PCP introduces gathers, and DSpark captures auxiliary
states. PDMux scheduling introduces split boundaries and SM competition, and
HiCache transfers compete for memory/PCIe bandwidth. The adaptation adds lane
selection and state isolation around these existing operations; the benchmark
does not measure their network/transfer cost or whether enabling the entire
combination is faster for a particular workload.

Attention head padding can limit the gain from decode attention TP. On the
current non-SM120 FlashMLA target path, 64 total heads split across CP2 leave
32 local heads that are padded to 64 for the attention kernel. DSpark has its
own padded query path. Halving projection width therefore does not imply
halving attention-kernel work. The projection benchmark does not model this.

DeepSeek-V4's default breakable-prefill-graph resolution already disables
prefill CUDA graphs because of capture-pool memory pressure
(`arg_groups/cuda_graph_hook.py`). Requiring prefill graphs to remain disabled
does not introduce a new default-path performance loss. Decode graphs remain
available. Explicit prefill graph overrides are outside the admitted recipe.

Run the full-model graph and eager acceptance variants from the runbook, then
the original 4219-request workload, to compare throughput, TTFT, TPOT, memory,
acceptance rate and HiCache hit/reload behavior with each option enabled.

## Configuration-resolution follow-up

A subsequent real-pipeline regression exposed a configuration bug outside the
projection benchmark: `_data_parallelism_defaults` reset the DSV4 PCP model
hook's `enable_dp_attention=True` at DP1. The earlier unit fixtures supplied
the resolved flag directly and could not detect that reset. Preserve the model
decision in the DP1 pass and retain the PDMux gate's requirement that the final
flag is true. The new config-only regression starts with CLI defaults and
checks CP2/4/8, decode attention TP off/on, resolved readback and pickled worker
publication. A run that bypassed the gate while reporting DP attention false
must be rerun before comparing full-model performance with this recipe. The
isolated projection timings above do not validate configuration resolution.
