# Qwen3.8 small decoder and TileLang MQA CUDA acceptance

Tested on the RTX 4060 Laptop (SM89, 8,188 MiB), using the Git checkout at
`a5b254f2096e488fe828da3101cb2d4cb197293c`.

**The real BF16 and FP8 decoder computations preserve outputs and request state
across split prefill / decode streams when the selected tokens have a fixed
order. Native top-k returns an unordered set: its downstream BF16 rounding is
not bit invariant, including during ordinary single-stream execution.** This
report does not claim full-model accuracy or serving acceptance.

## Results

Each dual-stream case runs 32 paired iterations with independent request slots,
temporary allocation pressure and decode CUDA Graph replay. The decoder submits
its decode graph between prefill's two real layers, exercising the production
model's batch-owned continuation state.

| Case | Reference / serial comparison | Result |
| --- | --- | --- |
| TileLang BF16 MQA, absolute columns + paged decode | Max reference error 8.94e-8; dual-stream comparison exact | Passed |
| TileLang BF16 MQA, request-relative columns + paged decode | Max reference error 8.94e-8; dual-stream comparison exact | Passed |
| TileLang FP8 MQA, absolute columns + paged decode | Max reference error 7.39e-5; dual-stream comparison exact | Passed |
| TileLang FP8 MQA, request-relative columns + paged decode | Max reference error 7.22e-5; dual-stream comparison exact | Passed |
| BF16 GDN → QSA decoder, canonical token order | Terminal output, HC output, GDN/conv states, QSA K/V, GDN layer output and selected sets: all exact | Passed |
| Same decoder with **9 real FP8 projections**, canonical token order | All eight comparisons exact; unused slots unchanged | Passed |

The first four cases and the FP8 decoder share the final, logging-disabled run
in [combined.json](combined.json). The BF16 zero-tolerance run is in
[decoder-bf16-strict.json](decoder-bf16-strict.json). These six primary cases
cover 192 paired iterations, excluding warmup, diagnostic runs and earlier
operator checks. MQA reference checks use explicit FP32-logit tolerances;
stream-versus-serial and canonical decoder checks use zero tolerance.

Shapes: hidden width 256, HC count 4, prefill 65 tokens, decode 1 token; GDN
H=2/HV=6 with 128-dimensional heads; QSA QH=6/KVH=1 with head dimension 256,
context 4,096 and token budget 2,048. Standalone MQA also tests nonzero row starts,
request boundaries inside row tiles, incomplete tiles, six decode heads padded
to eight, and paged lengths 127/65/1 with page size 16.

## Native top-k order diagnosis

The original-order decoder **does not pass** the harness's strict output check:
its terminal mix differs by up to 0.0078125, while GDN/conv state, QSA K/V, GDN
layer output and the sorted selected-token sets are all exactly equal. The
failure is retained in [decoder-native-cu13.json](decoder-native-cu13.json).

The native CUDA top-k contract explicitly leaves atomic collection order
unspecified. Sparse attention's BF16 tile reductions can therefore change
rounding when the same selected set arrives in a different order. Sorting that
same set in the **test adapter**, without changing selection or any production
kernel, makes every decoder comparison exact. This was checked under the same
CUDA 13 toolchain as the original-order run.

A separate control uses only the default stream, with no splitting or graph.
Across 32 repeats, it also varies: maximum terminal difference 0.015625 and HC
difference 0.0078125. Its state/KV and selected sets remain exact. See
[serial-native-order.json](serial-native-order.json). Thus the observed variation
is also present without PDMux. The canonical experiment establishes stream
compatibility on this workload; it does not establish that native-order output
is bit identical, nor turn hidden-state differences into an answer-accuracy
verdict. Production top-k ordering was not changed.

## What is real, and what is adapted

The harness compiles selected production method bodies verbatim and executes
real CUDA math. It runs `Qwen4ExpModel.forward` with `split_interval`, the GDN and
QSA decoder `forward` methods, Qwen3.5 GDN projection orchestration, production
HC mix/combine and dense Qwen2 FFN orchestration. Attention uses real Triton
causal convolution, GDN chunk/recurrent, gated RMSNorm, QK norm/RoPE/gating,
TileLang MQA, native CUDA top-k 512, index expansion and sparse GQA kernels.
The FP8 variant adds production Triton per-token group quantization and
128×128 block-scaled FP8 GEMM; HC, convolution and the small B/A projection stay
BF16. Weights are small synthetic tensors, shared across both streams.

Construction, TP=PP=1 stage transport and cache metadata use explicit test
adapters. The QSA prefix is fixed and precompressed; the test calls production
`select_prefill_tokens` / `select_decode_tokens` but does not run the complete
indexer query/key preparation and compression-update orchestration. This is
production-forward / real-computation coverage, not validation of complete
production constructors, weight loading or server dispatch. No simulated
decoder layers or replacement attention outputs are used.

## Environment repairs

- PyTorch 2.14.1+cu130, Triton 3.8.0; TileLang **0.1.12**, matching the repo pin;
  compatible Apache TVM-FFI 0.1.11, plus isolated helper dependencies.
- System NVCC 12.0 first rejected GCC 13. GCC 12 allowed BF16 compilation, but
  CUTLASS's SM89 FP8 MMA guard requires CUDA >=12.4 and asserted at runtime.
  An isolated CUDA 13.0.88 compiler/NVVM/CRT, CUDA runtime 13.0.96 and CCCL
  13.0.85 fixed that requirement. A fresh TileLang cache prevents reuse of the
  earlier, successfully compiled but runtime-asserting FP8 cubin.
- The standalone loader formerly inherited its caller's future-annotations
  flag. That stringified TileLang's nested dynamic-shape annotations and caused
  `NameError: pages`. `compile(..., dont_inherit=True)` now preserves each source
  module's annotation semantics. Kernel math was not rewritten.
- Source sync uses origin push and a Windows Git sparse clone/pull. Windows
  downloads official wheels directly and checks SHA256; WSL runs CUDA through
  Windows SSH when its own SSH banner exchange times out. Existing system CUDA,
  the venv and proxy configuration were not changed.

[environment.json](environment.json) records the GPU, compiler and tested Git
commit. [cuda-toolchain-manifest.json](cuda-toolchain-manifest.json) records
official URLs and hashes. Source hashes in each result are raw Windows-checkout
bytes (CRLF); they were verified against the local LF sources after newline
normalization. The debug rerun additionally saved `mqa-api.log` (tracked as `mqa-api.log.gz`) using the
repository's kernel API logging; the final run disables that logging. Compiler
logs retain TileLang's symbolic race-check warnings for dynamic index
expressions; no checker was disabled to make the tests pass.

## Reproduction on this host

From WSL, using the Git checkout mounted from Windows:

```bash
cd /mnt/c/Users/27839/AppData/Local/Temp/qwen38-decoder-source
export CUDA_HOME=/workspace/qwen38-decoder-cuda-20261107/cuda13/nvidia/cu13
export PATH="$CUDA_HOME/bin:/workspace/conditional-graph-20261019/venv/bin:$PATH"
export CXX=/usr/bin/g++-13
export PYTHONPATH=/workspace/qwen38-decoder-cuda-20261107/deps-pinned
export TILELANG_CACHE_DIR=/workspace/qwen38-decoder-cuda-20261107/tilelang-cache-cu13

python test/manual/pdmux/validate_qwen38_decoder_cuda.py \
  --only all --canonical-topk --fp8-projections --rounds 32 --output /tmp/combined.json
python test/manual/pdmux/validate_qwen38_decoder_cuda.py \
  --only decoder --canonical-topk --rounds 32 --output /tmp/bf16.json
```

Omit `--canonical-topk` to reproduce the native-order strict-output diagnostic.
`--dtype bf16|fp8` and `--layout absolute|relative` isolate MQA cases; `--debug`
uses configured `SGLANG_KERNEL_API_*` logging. On another machine, install
compatible dependencies and a CUDA >=12.4 toolchain rather than copying these
host-specific paths.

## Remaining full-model acceptance

The full Qwen FP8 checkpoint, routed MoE, PLE, complete QSA indexer preparation
and cache updates, native MTP, multi-GPU TP, green-context SM partitions, long
progressive mixed-request stress, task accuracy and TTFT/TPOT/throughput are
outside this result. They require a complete compatible serving environment,
the model weights and a GPU/TP configuration with sufficient memory. The
4060's 8 GiB cannot provide full-checkpoint acceptance. FP8 projection stream
parity here does not measure checkpoint quantization quality versus BF16.
