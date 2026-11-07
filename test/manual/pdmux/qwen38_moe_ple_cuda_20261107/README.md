# Qwen3.8 real MoE and PLE dual-stream CUDA regression

Tested on the RTX 4060 Laptop (SM89, 8,188 MiB), from Git commit
`ffd4abe1d41e93941f12c5ea0e116fb1926b7d68`.

**All eight small real-computation configurations passed 32 paired iterations
each, including decode CUDA Graph replay. Dual-stream outputs, routing and
request state match serial controls exactly.** These 256 pairs exclude warmup
and diagnostic runs. This establishes compatibility for the exercised small
layers; it does not establish full-checkpoint serving accuracy or performance.

## Results

| Configuration | Checks | Result |
| --- | --- | --- |
| BF16 MoE, intermediate 256 | Expert output and router IDs/weights; independent Torch reference | Passed, stream/serial exact |
| FP8 MoE, intermediate 256 | Same, plus effective-operand GEMM and quantization checks | Passed, stream/serial exact |
| FP8 MoE, intermediate 192 | Partial 128-element weight-scale tile, same checks | Passed, stream/serial exact |
| Two BF16 PLE layers, pinned CPU tables | Hash, UVA lookup, working/track history and short-conv state | Passed, stream/serial exact |
| Two FP8 PLE layers, pinned FP8 tables and four FP8 projections | Same checks | Passed, stream/serial exact |
| Progressive FP8 PLE, 32 decode steps | Changing token IDs/hidden states, no history reset between graph replays, EOS boundaries | Passed, all step outputs/history/conv exact |
| BF16 GDN → QSA decoder with MoE, shared experts and PLE | Split prefill, high-priority decode graph, 16 output/state/routing comparisons | Passed, all differences zero |
| Same decoder with FP8 MoE, pinned FP8 PLE tables and 13 FP8 projections | Same checks; unused slots unchanged, shared PLE prefetch disabled | Passed, all differences zero |

Primary results and complete console logs are in [moe.json](moe.json),
[moe.log](moe.log), [ple.json](ple.json), [ple.log](ple.log),
[integrated.json](integrated.json) and [integrated.log](integrated.log).

## Real computations and explicit adapters

The harness compiles selected production definitions without changing their
method bodies. It uses small synthetic weights shared by both CUDA streams.
Allocation, TP=PP=EP=1 transport, cache indexing and construction boundaries use
explicit metadata adapters; this does not validate complete public-module
imports, production backend construction or checkpoint loading.

### MoE

Hidden width is 256, four routed experts, top-k two, intermediate width 256
(BF16/FP8) or 192 (FP8), with 65 prefill rows and one decode row. Execution uses
the real `Qwen2MoeSparseMoeBlock` forward/router/expert orchestration and
`fused_experts_impl`: token alignment, per-token group quantization, two Triton
expert GEMMs, native CUDA SiLU/multiply, route weighting and output reduction.
The router runs production `fused_topk_torch_native` on CUDA. The fused native
CUDA router is outside this coverage.

FP8 weights use 128×128 block scales. The supported `pad_expert_ids=True`
alignment option selects the in-tree JIT provider, avoiding this host's cached
`sgl_kernel` wheel with an incompatible Torch ABI. CUDA activation and alignment
compile the original repository headers through TVM-FFI. Default MoE/GEMM
configs are used; the retained warnings correctly say performance is untuned.

### PLE

The test runs real `Qwen4ExpPLELayer`, N-gram embedding, pinned-host embedding
and grouped-norm constructors/methods, plus production prepare/commit/track
orchestration and state-pool accessors. Only vocabulary sharding/weight-loader
metadata and pool allocation/routing are adapted. Lookup actually reads pinned
CPU table memory through the production UVA Triton gather; no embedding output
is substituted. The FP8 variant uses E4M3 table values with BF16 scales and four
real FP8 projection GEMMs across two layers.

Configuration: hidden/embedding width 256, HC count four (1,024-wide state),
N-gram size three, eight heads per N-gram (16 heads total), small prime table
vocabularies, convolution width four, dilation three and nine cached positions.
Standalone prefill has ragged request lengths 33/32 plus five physical padding
rows; decode has one token. The tests cover EOS handling, Python scalar hash
and CPU embedding/dequantization references, working and track slots, padding,
and untouched sentinel slots. The eight hash/embedding/history reference
comparisons per static PLE configuration have maximum error zero. Short-conv
outputs/state additionally match serial CUDA execution exactly.

The progressive case records prepare → two layers → commit inside the graph.
It resets the decode pool once after capture, then changes graph input buffers
for 32 consecutive steps, with EOS tokens at steps 5 and 17 (zero-based).
Every step's output, hash, working/track history and conv state matches the
independent serial trajectory. This checks that replay consumes the preceding
step's committed history rather than a stale capture-time value.

### Combined decoder

Production `Qwen4ExpModel.forward` runs layer intervals 0:1 and 1:2, with the
decode graph submitted between the GDN and QSA prefill layers. Both layers
contain real routed MoE plus a shared dense expert and real PLE. The FP8 case
has two FP8 MoE blocks and 13 FP8 ordinary projections; routers, HC, normalization
and convolution retain their appropriate BF16 computation. Decode uses stream
priority -1, prefill priority 0, and independent request/cache slots.

The 16 exact comparisons cover terminal mix, HC output, GDN state/conv, QSA K/V,
GDN layer output, selected QSA tokens, PLE working/track history and conv, and
both layers' router IDs/weights. PLE's shared prefetch state stays empty under
PDMux and unused slots remain unchanged.

As in the [earlier decoder report](../qwen38_decoder_cuda_20261107/README.md),
QSA uses a fixed compressed prefix and canonical ordering of the same selected
token set in the **test adapter**. Native unordered top-k's known BF16 reduction
variation was not changed in production. Complete QSA indexer preparation and
progressive QSA KV/cache updates are not exercised by this combined case.

## FP8 reference diagnosis and harness fixes

The first wholly independent Torch quantizer reference failed a pointwise
`atol=rtol=0.02` check: worst full-MoE absolute differences were 0.05078125
(I256) and 0.078125 (I192). That failed run is retained in
[moe-naive-reference-before.json](moe-naive-reference-before.json) and its
[log](moe-naive-reference-before.log); the additional isolation experiment is in
[moe-reference-diagnostics.json](moe-reference-diagnostics.json) and its
[log](moe-reference-diagnostics.log).

The diagnostic separates quantizer and GEMM behavior. A few FP8 values differ
between the independent Torch quantizer and production kernels near rounding
midpoints; this can propagate through the nonlinear two-GEMM path. The final
test therefore validates all of the following, rather than hiding the failed
comparison or weakening stream equality:

- The independent whole-path reference remains checked: normalized L2 error
  is below 2%, with observed worst value **1.795%**. This is an operator-level
  synthetic-weight check, not checkpoint quantization accuracy.
- Each quantized operand satisfies an E4M3 rounding-error bound, including
  subnormals. Differing quantized values are logged with midpoint distances.
- Before concurrency testing, temporary instrumentation copies the actual
  quantized operands and restores the original callables in `finally`.
  Independent FP32 Torch GEMMs using those operands give stage normalized L2
  errors below **0.091%** (limit 0.3%), with pointwise checks also enforced.
  Activation is independently checked against Torch SiLU/multiply.
- Serial versus dual-stream outputs and router results still require **zero
  difference**. Instrumentation does not replace arithmetic during these runs.

The standalone JIT bootstrap initially omitted production's
`--expt-relaxed-constexpr`; matching the production compiler flags fixed the
build. The shared AST loader now retains class constants needed by real PLE
constructors, and includes production `_add_ple_embedding` in the decoder
fixture. These are validation-harness changes. This round found no additional
production MoE/PLE stream-isolation defect requiring an arithmetic change.
The earlier production GDN mutable-state preservation fix remains in place.

## Environment, provenance and reproduction

[environment.json](environment.json) records the tested Git commit, GPU,
compiler flags and harness hashes. Runtime: PyTorch 2.14.1+cu130, Triton 3.8.0,
TileLang 0.1.12 (repository pin), TVM-FFI 0.1.11 and isolated CUDA 13.0.88.
The earlier [toolchain manifest](../qwen38_decoder_cuda_20261107/cuda-toolchain-manifest.json)
records the compiler/runtime package URLs and hashes.

Code was committed and pushed to origin. The Windows sparse clone received
Git bundles through SSH and performed `git pull --ff-only`; a Git object pack
populated missing sparse-checkout dependencies when GitHub HTTPS was unstable.
CUDA ran in WSL through Windows SSH. Source files were not copied individually.
Reported source hashes are raw Windows-checkout bytes (CRLF), verified against
the local LF sources after newline normalization.

From the Git checkout in WSL:

```bash
cd /mnt/c/Users/27839/AppData/Local/Temp/qwen38-decoder-source
export CUDA_HOME=/workspace/qwen38-decoder-cuda-20261107/cuda13/nvidia/cu13
export PATH="$CUDA_HOME/bin:/workspace/conditional-graph-20261019/venv/bin:$PATH"
export CXX=/usr/bin/g++-13
export PYTHONPATH=/workspace/qwen38-decoder-cuda-20261107/deps-pinned
export TILELANG_CACHE_DIR=/workspace/qwen38-decoder-cuda-20261107/tilelang-cache-cu13

python test/manual/pdmux/validate_qwen38_moe_ple_cuda.py \
  --only moe --rounds 32 --output /tmp/moe.json
python test/manual/pdmux/validate_qwen38_moe_ple_cuda.py \
  --only ple --rounds 32 --output /tmp/ple.json
python test/manual/pdmux/validate_qwen38_moe_ple_cuda.py \
  --only integrated --rounds 32 --output /tmp/integrated.json
```

These three commands reproduce the final runs in separate processes.
`--only all` is also available. The bootstrap explicitly targets SM89; adapt
runtime metadata/toolchain and install compatible dependencies on other hosts.

## Remaining acceptance

Full FP8 checkpoint loading, model answer/logprob accuracy, complete production
backend/weight-loading integration, fused CUDA routing, PLE file-backed offload
and HiCache restore, complete QSA indexer/cache progression, native MTP,
multi-GPU TP/EP communication, green-context SM partitions and long progressive
mixed-request serving stress remain unverified. The 4060's 8 GiB cannot hold
the full checkpoint. End-to-end TTFT/TPOT/throughput and actual overlap speedup
need a compatible full serving environment, sufficient GPU memory and traces;
this regression does not provide performance acceptance.
