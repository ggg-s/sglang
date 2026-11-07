# Qwen3.8-Flash-Next-FP8 with PDMux

Checkpoint: `Qwen/Qwen3.8-Flash-Next-FP8`. Its config declares
`Qwen4ExpForConditionalGeneration` (`qwen4_exp`), with 48 hybrid GDN/QSA layers,
4 HyperConnection streams, PLE and block FP8 weights. Keep the model's existing
FP8 loader and quantization exclusions; no conversion to a Qwen3.5 architecture
is needed.

## Text serving

Example for a CUDA host with enough memory at TP4 (capacity must be checked on
the actual host; this command has not been validated with the checkpoint yet):

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.8-Flash-Next-FP8 \
  --tp-size 4 \
  --moe-runner-backend triton \
  --enable-pdmux \
  --pdmux-prefill-mode layer_split \
  --cuda-graph-backend-prefill disabled \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.80 \
  --reasoning-parser qwen3-thinking
```

Use PP=1, attention DP=1 and EP=1, without prefill CP or two-batch overlap, as
required by the existing PDMux overlap scheduler. The split path is text-only;
use `--pdmux-prefill-mode standard` for the ordinary full-forward path.

For block FP8 PDMux, automatic target/draft MoE backend selection resolves to
Triton. Explicit other MoE backends are rejected until their model-owned scratch
workspaces are isolated between lanes.

For native MTP, the split wrapper returns the widened HC hidden states used by
the existing NEXTN/EAGLE worker. Add the following after validating plain serving:

```bash
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

DFlash split-prefill admission is unchanged; this adaptation does not advertise
DFlash support for Qwen3.8.

## Execution and isolation

- Every segment uses the ordinary Qwen4 layer implementation and FP8 operators.
- HC residual state lives on the ForwardBatch. The prepared next-layer norm,
  widened hidden states, positions, PLE token layout and auxiliary captures are
  retained under `model_specific_states["qwen4_pdmux"]` between segments.
- PLE N-gram history is prepared once and committed after the final segment.
- PDMux disables PLE's model-wide prefetch slot, performing lookup on the active
  lane instead. Decode cannot consume a prefill's prefetched rows after a yield.
- PDMux also bypasses the Qwen4 single-group decode collective and QSA side-stream
  overlap. The ordinary lane-selected communicator and current stream are used.

## Validation

`test/registered/unit/models/test_qwen38_pdmux.py` compares full execution with
multiple segment sizes while interleaving decode through the same model. It
checks exact CPU tensor output equality, one PLE prepare/commit, state cleanup,
MTP hidden-state width and the PLE prefetch isolation gate.

GPU acceptance remains required: compare deterministic output against PDMux-off,
exercise concurrent long prompts and decode in both PDMux modes, prefix-cache
hits, and native MTP. Capture a trace to confirm lane overlap and run a long
mixed-request stress test for state/allocator races. CPU tests do not establish
FP8 kernel accuracy, serving accuracy or throughput on real hardware.

## 4060 CUDA component regression (2026-11-07)

[Results, logs and remaining acceptance requirements](qwen38_cuda_20261107/README.md)
are available. FP32/BF16 two-stream model-loop fixtures passed 24 configurations
(768 prefills and 1,792 decodes), with exact output equality. Existing real
singleton NCCL/CUDA Graph tests passed 2 cases; the distinct two-GPU case skipped.
These checks do not load the checkpoint or validate real GDN/QSA/FP8 operators,
green-context partitions, answer accuracy or serving performance.

## Real-operator CUDA regression and GDN fix (2026-11-07)

[Real-operator results and failure/fix evidence](qwen38_real_ops_cuda_20261107/README.md)
cover GDN chunk-prefill/recurrent decode, block FP8 GEMM, QSA sparse attention,
compressed-key gather/index expansion and native CUDA top-k 512. All six paths
passed 32 dual-stream iterations each with decode CUDA Graph replay. A real GDN
bug was fixed: the input guard copied non-contiguous mutable state/track pools,
so kernel writes failed to reach the caller's pool. The new registered regression
fails against the original source and passes with named state-buffer preservation.
Full decoder/model and serving-performance acceptance remain pending.

## Small real decoder and TileLang MQA regression (2026-11-07)

[Decoder/MQA evidence and reproduction](qwen38_decoder_cuda_20261107/README.md)
extend this to production decoder forwards with real GDN convolution, HC,
dense FFN, QSA scoring/selection/attention and nine FP8 projections. Four
TileLang MQA configurations and BF16/FP8 small decoder variants passed 32
dual-stream rounds each, including decode CUDA Graphs and layer-split prefill.
With a canonical order for the same selected token set, decoder outputs and
states match serial execution exactly. Native top-k's unspecified order yields
BF16 output variation in both dual-stream and ordinary single-stream controls;
the selected sets, state pools and KV contents remain exact. The report retains
that strict native-order failure and explains the controlled comparison.

The small configuration uses single-rank metadata adapters and a fixed
compressed prefix. Full checkpoint/MoE/PLE/indexer-update/MTP/TP acceptance,
answer accuracy and serving performance remain outside these results.

## Small real MoE/PLE and progressive graph regression (2026-11-07)

[MoE/PLE results, diagnostics and reproduction](qwen38_moe_ple_cuda_20261107/README.md)
add eight configurations, each passing 32 paired iterations with decode CUDA
Graph replay. Coverage includes real BF16/FP8 routed MoE, native alignment and
activation, pinned BF16/FP8 PLE table UVA lookup, working/track history and
short-conv state, and GDN → QSA decoder layers combining PLE, routed MoE and
shared experts. Dual-stream outputs/state/routing match serial controls exactly.
The combined FP8 decoder includes 13 FP8 projections and two FP8 MoE blocks.
A progressive PLE case changes inputs for 32 graph replays without resetting
decode history and matches every serial state transition exactly.

The report preserves the initial FP8 pointwise-reference failure and separates
near-midpoint quantization differences from GEMM error; stream comparisons stay
at zero tolerance. Construction/cache/TP1 metadata adapters and canonical QSA
token ordering remain explicit. This adds small-layer MoE/PLE coverage, while
full-checkpoint loading, complete serving/backend integration, indexer/cache
progression, file-backed PLE restore, MTP, TP/EP, model accuracy, long stress and
end-to-end performance acceptance remain pending.
