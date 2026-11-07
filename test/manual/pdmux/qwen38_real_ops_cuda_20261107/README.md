# Real Qwen3.8 GDN / QSA / FP8 CUDA checks — 2026-11-07

**Result: six real operator paths passed dual-stream execution and decode CUDA
Graph replay on the RTX 4060. A real GDN state-pool writeback bug was reproduced,
fixed and covered by a registered regression.** No simulated decoder layers are
used in this suite.

## Environment and loading boundary

- Host: `ssh ubuntu` through Windows; RTX 4060 Laptop, SM89, 8,188 MiB.
- Python: `/workspace/conditional-graph-20261019/venv/bin/python`.
- PyTorch 2.14.1+cu130, Triton 3.8.0; driver 592.82.
- NVCC 12.0 and cached TVM-FFI build the repository's native top-k header for
  SM89, with PDL disabled as in the production architecture selection.
- Remote source/results: `/workspace/qwen38-realops-cuda-20261107/`.
- `einops==0.8.2` was installed **only** under that directory's `deps/`, using
  an offline wheel because the host's configured PyPI proxy was unreachable.

The harness omits package/runtime imports when loading standalone modules and
supplies explicit normal-numerics SM89 metadata. It does not replace kernel
math, rewrite Triton function bodies, or use mock layers. FLA chunk-prefill and
its helper kernels are imported from the snapshot normally through namespace
package bootstrap. CUDA top-k uses the unchanged repository header with an FFI
export wrapper; no cached incompatible `sgl_kernel` binary is used. This is
operator validation, not a test of the complete server import/dispatch stack.

## Tests

Each operator pair is warmed up serially and compared with an independent
PyTorch reference, then submitted on separate prefill/decode CUDA streams for
32 iterations. Decode runs in a captured CUDA Graph. Outputs and mutated states
must match their respective serial controls **exactly**, with no tolerance.
Temporary allocation pressure is included between submissions.

| Real operator path | Shapes / checks | Result |
| --- | --- | --- |
| Fused sigmoid-gated GDN recurrence | T=65 / T=1, H=2, HV=6, K=V=128; FP32 envelope-strided state pool, disjoint slots | Passed |
| FLA chunk GDN prefill + fused recurrent decode | Same real head dimensions and GQA ratio; shared state pool; unused slot unchanged | Passed after writeback fix |
| Triton block-scaled FP8 GEMM | M=65 / M=1, N=K=256, 128×128 weight scales, BF16 output, shared FP8 weights | Passed |
| Triton QSA sparse GQA prefill / decode | Q rows=65 / 1, Q heads=6, KV heads=1, dim=256, 128 keys, shared KV backing | Passed |
| Triton QSA compressed-key gather / index expansion | Compression ratio 4, packed key rows and relative block indices | Passed |
| Native CUDA QSA top-k 512 | 65 / 1 rows, nonzero row starts, full and <512 lengths; selected sets compared, since output order is unspecified | Passed |

Total: **192 paired dual-stream iterations** across the six paths. Each decode
path was replayed from a CUDA Graph. This count excludes warmup/reference work.
`realops.json` includes source hashes and measured reference errors. The maximum
reference differences were approximately 3.1e-5 for GDN output, 2.0e-4 for its
FP32 state, 0.001953125 for FP8 GEMM BF16 output, and 0.0078125 for QSA BF16
output. Gather/expand and top-k set references matched exactly. Reference
checks use explicit dtype tolerances; dual-stream-versus-serial checks use zero
tolerance. These are not task-accuracy or serving-throughput measurements.

## Bug and fix

`ChunkGatedDeltaRuleFunction.forward` used `input_guard`, which called
`.contiguous()` on every tensor, including mutable `initial_state` and
`track_state` buffers. For an envelope-strided pool, that created a copy. The
real kernel correctly read and wrote the copy, but the caller's pool retained
stale state. Its apparent output could still look correct for that prefill;
subsequent decode/prefix reuse would consume the stale state.

The fix adds opt-in named buffer preservation to `input_guard` and enables it
for those two GDN buffers. Other inputs retain existing contiguity conversion,
and device selection is preserved. The existing kernels already accept their
per-slot strides.

The registered case
`test_chunk_prefill_updates_strided_state_and_track_pools` compares contiguous
and envelope-strided pools using the real chunk kernel, checks both in-place
state and tracking writes, and checks untouched request slots. The same case
**fails on the original sources**: 98,304 / 294,912 state elements differ,
maximum absolute difference 0.5493546. It passes after the fix. Evidence:
`registered-before-fix.json`, plus `before-fix.json` / `before-fix.log` from the
original operator test. The fixed suite reports
`strided_state_track_regression: passed`.

## Reproduction

On this 4060, from the isolated snapshot's `source/` directory:

```bash
export PATH=/workspace/conditional-graph-20261019/venv/bin:$PATH
PYTHONPATH=../deps:/root/.cache/uv/archive-v0/HywQg704pUDeQISQXcUgC \
  python test/manual/pdmux/validate_qwen38_real_ops_cuda.py \
  --rounds 32 --output ../realops.json
```

The cached path is this host's TVM-FFI installation. Another host should install
a compatible TVM-FFI package and provide NVCC/ninja instead of using that path.
The harness intentionally fails unsupported/non-SM89 standalone bootstrap runs.

## Still outside this result

The full FP8 model, projections/causal convolution and complete real decoder
layers, TileLang QSA MQA scoring, native MTP end-to-end, real multi-GPU TP,
green-context SM partitions, model answer accuracy, long mixed-request stress,
and serving TTFT/TPOT/throughput remain unvalidated. The 8-GiB 4060 cannot provide
full-checkpoint acceptance. These successful kernel tests must not be reported
as full-model PDMux acceptance.
