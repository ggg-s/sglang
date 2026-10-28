# DeepSeek-V4 DSpark + CPU overlap + PDMux + HiCache, TP2

Use `codex/dsv4-pdmux-latest-main`. This configuration uses plain TP2, PP1,
EP1, MoE-DP1, attention-CP1 and DCP1. Leave DP attention disabled. PDMux's
duplicated communication groups cover full-width TP. The optional PCP path
below also uses full-width groups; other subgroup parallelism remains rejected.
CPU overlap is the asynchronous CPU/GPU scheduler for CUDA,
and is enabled by omitting `--disable-overlap-schedule`.

The DSpark prefill finalizer now broadcasts on the current prefill TP group;
draft proposal, verify planning and acceptance keep the decode group. Verify
metadata uses the per-stream decode backend. DeepSeek-V4 target/draft helper
streams are disabled under both PDMux modes so work stays on the managed lane.
The existing layer-split auxiliary-hidden capture and packed HiCache draft-KV
registration remain in use. CPU overlap supports `layer_split` and, as of
2026-10-28, `standard` with the DSV4 DSpark PCP recipe below. The temporal and
decode-ready experiments retain their separate restrictions.

## Full-model acceptance on the deployment host

Install this checkout in the deployment environment, then run:

```bash
export SGLANG_TEST_DSV4_FLASH_MODEL_PATH=/models/DeepSeek-V4
export SGLANG_TEST_DSV4_DSPARK_MODEL_PATH=/models/DeepSeek-V4-DSpark
# Omit the draft variable only if the target checkpoint bundles DSpark weights.
# An ordinary NextN/MTP head is not a DSpark draft.

python -m pytest -v \
  test/manual/pdmux/test_pdmux_tp_overlap_cuda.py::test_two_gpu_nccl_lanes_with_decode_graph \
  test/manual/pdmux/test_dsv4_dspark_pdmux_hicache_tp2.py
```

The full-model test requires two CUDA GPUs and the model-path variable; missing
resources cause explicit skips. It starts TP2 with CPU overlap, 1024-token
chunks, a 2048 token-layer split budget, full-device decode and capped prefill,
and HiCache write-through at ratio 2. It checks resolved settings, host reload
with identical greedy output IDs, and backup progress during concurrent
requests. Target prompts are 4096 tokens, forcing multiple prefill chunks.
The shared test recipe dequantizes FP4 experts for the plain-TP Triton runner;
ensure the host has enough memory for the selected checkpoint.

For a manual server launch, create a config matched to the deployment GPU:

```bash
python - <<'PY'
from pathlib import Path
import torch
sms = torch.cuda.get_device_properties(0).multi_processor_count
prefill_sms = max(8, (sms // 2 // 8) * 8)
Path('/tmp/dsv4-dspark-tp2.yaml').write_text(
    'sm_group_num: 3\n'
    'split_forward_token_budget: 2048\n'
    'overlap_decode_full_sm: true\n'
    'manual_divisions:\n'
    f'  - [{prefill_sms}, 0, 1]\n'
)
PY

SGLANG_DSV4_FP4_DEQUANT=1 \
SGLANG_DSV4_MHC_PREWARM=0 \
python -m sglang.launch_server \
  --model-path "$SGLANG_TEST_DSV4_FLASH_MODEL_PATH" \
  --speculative-draft-model-path "$SGLANG_TEST_DSV4_DSPARK_MODEL_PATH" \
  --trust-remote-code --tp 2 --ep-size 1 \
  --speculative-algorithm DSPARK \
  --enable-pdmux --pdmux-prefill-mode layer_split \
  --sm-group-num 3 --pdmux-config-path /tmp/dsv4-dspark-tp2.yaml \
  --chunked-prefill-size 1024 --max-running-requests 8 \
  --enable-hierarchical-cache --hicache-write-policy write_through \
  --hicache-ratio 2 --mem-fraction-static 0.7 \
  --max-total-tokens 32768 --enable-metrics
```

Keep `SGLANG_SPEC_TP_SYNC` at its default (`all`) during validation. Retain the
checkpoint's DSpark block size unless the checkpoint explicitly supports a
different size. After the smoke checks, rerun the original 4219-request long
prompt workload, retaining server logs and comparing outputs with the control.
The small tests do not establish long-run progress or performance gains.

## Optional prefill context parallelism (PCP)

The branch now admits an experimental DeepSeek-V4 DSpark PCP configuration:
interleave, CP=TP, DP1, EP1, MoE-DP1, DCP1, PP1, and MoE A2A backend `none`.
It requires a DeepSeek-V4 DSpark draft. Add these arguments to the launch above:

```bash
--enable-prefill-cp --cp-strategy interleave --dp-size 1 \
--moe-a2a-backend none --cuda-graph-backend-prefill disabled
```

DSV4 resolves `enable_dp_attention=true` for PCP even with DP1. This specific
case is admitted; actual DP attention with DP>1 is still rejected. CP and MoE-DP
aliases of the full TP group now follow the active PDMux lane. Per-forward DP
buffer sizes/counts and MoE collective flags have separate P/D storage. Split
continuations restore CP's padded physical length on every segment. DSpark
sampling/planning/acceptance sync spans the full CP group rather than the
singleton attention-TP group.

The DP1 default pass must preserve the PCP model hook's DP-attention decision.
`check_server_args` runs after resolution in the current launcher; weakening its
PCP gate to accept a false DP-attention value hides a later DP1 reset. Verify
`enable_dp_attention=true` in `/server_info` as well as in each worker's
published configuration. `test_dsv4_pcp_resolution.py` covers the real pipeline,
readback and pickled worker publication without loading model weights.
The subsequent full-chain audit, including real decode-context activation and
24 CLI combinations, is documented in `dsv4_pcp_configuration_audit_20261027.md`.
For chunk-boundary scheduling fixes, the provisional 32k split budget and why
this DSV4 PCP path does not shard KV storage, see
`dsv4_pdmux_prefill_progress_20261027.md`.

`--enable-cp-decode-attn-tp` can now be added to this configuration. The target
and DSpark draft each cache an unregistered decode attention module with their
own sliced projection Parameters, linear dimensions, radix head count and sink
cache. PDMux forwards select that module without changing the prefill module or
the context's global activation flag. Shards retain stable addresses for decode
graphs; model refits refresh their contents in place. KV pools and HiCache layer
identities remain shared, since KV layout does not depend on the query head TP.

`--enable-torch-compile`, A2A MoE, partial-width CP, and PDMux `standard` prefill
remain rejected for this path.
Prefill CUDA graphs must be disabled; decode CUDA graphs remain available.
Keep `SGLANG_RAGGED_VERIFY_MODE=static` (the default). `compact` is rejected
because the DSV4 ragged-verify backend does not support CP. Do not enable TBO
(`--enable-two-batch-overlap`): its micro-batch backend is not constructed by
the PDMux backend setup.

The eager draft logits path also keeps the full TP group under PCP. PCP's
DP-attention flag must not select a singleton attention-TP group for the
TP-sharded LM head. This matters when folded proposal is disabled, memory
headroom disables draft graphs, or a batch falls outside the graph buckets.

Run the full-model acceptance on the deployment host:

```bash
python -m pytest -v \
  test/manual/pdmux/test_pdmux_tp_overlap_cuda.py::test_two_gpu_nccl_lanes_with_decode_graph \
  test/manual/pdmux/test_dsv4_dspark_pcp_pdmux_hicache_tp2.py
```

The PCP file runs decode-graph and all-eager variants (folded proposal disabled),
each with CP decode attention TP off and on. All inherit the host-reload parity
and concurrent backup-progress checks. To run only the newly enabled variants:

```bash
python -m pytest -v \
  test/manual/pdmux/test_dsv4_dspark_pcp_pdmux_hicache_tp2.py::TestDSV4DSparkPCPDecodeTP2CPUOverlapHiCache \
  test/manual/pdmux/test_dsv4_dspark_pcp_pdmux_hicache_tp2.py::TestDSV4DSparkPCPDecodeTP2EagerCPUOverlapHiCache
```

Local regression uses an isolated PyTorch 2.8 CPU environment, including CP2/CP4
projection parity for target/draft fixtures, FP8 scales and parameter subclasses,
concurrent P/D scopes, stable addresses, refits and real two-process Gloo. The
numeric fixtures run the production view/context code with small PyTorch
linears; they do not run FlashMLA. A CUDA-only graph replay test is included in
`test/registered/unit/multiplex/test_pdmux_cp_decode_attn_tp.py` and skips locally.
The earlier validation JSON covers plain TP, not these PCP changes. Full-model
multi-GPU correctness, NCCL progress, HiCache DMA and performance still need the
acceptance run above and the original long-prompt workload.

## 4060 regression coverage

The PCP/decode-TP adaptation was pulled over SSH and validated on 2026-10-27:
82 compatibility unit tests passed, and the related CUDA suite had 19 passes
and one two-GPU skip. See
`dsv4_dspark_pcp_decode_tp_performance_20261027.md` for the performance audit and
`dsv4_dspark_pcp_decode_tp_validation_20261027.json` for raw CP2/CP4 benchmark
samples, exact commands, tested source commit and test outputs.

The 4060 is a single 8-GiB SM89 GPU and cannot load the full target model or
exercise two-GPU NCCL. GPU checks use synthetic tensors with the production
DSpark finalizer, MLA injector and multi-stage KV projection on alternating
PDMux green-context streams, interleaved with decode CUDA graph replays. The
store fixture omits FlashMLA encoding and HiCache DMA. Separate tests exercise
actual singleton NCCL operations and DSpark Triton-vs-Torch kernel parity.
Two-process Gloo tests check prefill/decode broadcast ownership and rank-uniform
chunk completion and HiCache pumping. See the accompanying validation JSON for
the exact results and limits.

## Standard chunk prefill + PCP + CPU overlap (2026-10-28)

The same DSV4/DSPARK/interleave/DP1/CP=TP/EP1/MoE-DP1/DCP1/PP1 recipe now
accepts `--pdmux-prefill-mode standard`. Retain `--enable-cp-decode-attn-tp`,
`--moe-a2a-backend none`, `--cuda-graph-backend-prefill disabled`,
`SGLANG_RAGGED_VERIFY_MODE=static`, and HiCache write-through. Leave
`--disable-overlap-schedule` unset. Torch compile, TBO and unified memory remain
unsupported with this recipe; temporal/decode-ready experiments are separate.

For the existing long-prompt launch, select:

```bash
--pdmux-prefill-mode standard --chunked-prefill-size 16384
```

Each admitted chunk is one ordinary target EXTEND through the model worker,
running every target layer before returning; it does not use the persistent
split-forward path or yield at layer intervals. DSpark still performs its
normal auxiliary-state/draft bootstrap. `split_forward_token_budget` and
`split_forward_max_layers` do not control standard mode. The existing SM
configuration still caps prefill and selects the decode lane.

The standard loop keeps one decode result pending: N is submitted before
N-1 is processed. Grammar/CPU penalizers and memory retraction drain the pending
result before preparation. A rank-uniform ready vote drains the current decode
before prefill merge and FutureMap seeding. Prefill inputs/results are pinned
until completion, independently of the decode ring. Formation/merge events and
carried-tensor stream registration protect allocator/KV state, while HiCache
event pumping continues on the decode lane during prefill.

Standard mode still permits only one prefill batch in flight; this change
removes layer-interval submissions, not the one-prefill admission limit. There
is no full-model throughput/TTFT claim without rerunning the deployment load.

Two-GPU full-model acceptance, with greedy host-reload parity and concurrent
HiCache progress inherited from the original tests:

```bash
python -m pytest -v \
  test/manual/pdmux/test_dsv4_dspark_pcp_pdmux_hicache_tp2.py::TestDSV4DSparkPCPDecodeTP2StandardCPUOverlapHiCache \
  test/manual/pdmux/test_dsv4_dspark_pcp_pdmux_hicache_tp2.py::TestDSV4DSparkPCPDecodeTP2StandardEagerCPUOverlapHiCache
```

Validated source `2764ef4f90` was pushed and pulled over SSH on the RTX 4060
Laptop workstation, PyTorch 2.13.0+cu130:

- Complete `unit/multiplex`, PCP CLI/runtime resolution, scheduler DP metadata
  and CUDA graph buffer registry: **274 tests, 440 subtests passed**. The real
  two-process Gloo test now runs three standard chunks with unequal rank-local
  readiness and verifies that both ranks finalize on the same iterations.
- Standard CUDA loop plus existing TP/NCCL/graph lane checks: **4 passed,
  1 skipped** (the distinct two-GPU NCCL case). The new two cases use real CUDA
  streams, whole-chunk fixture forwards, delayed completion and decode graph
  replays, with and without the grammar preparation barrier.
- GPU-hidden PCP configuration/runtime tests: **12 passed, 1 skipped,
  149 subtests passed**; the skipped case is CUDA output replay.
- Local baseline check: three new progress/FutureMap regressions fail against the pre-change
  `5088a34a92` loop, loaded in memory without modifying the checkout. The local
  standard-loop harness passes all 18 cases with the fix.
- The two new full-model acceptance classes collect successfully; their four
  cases skip on the 4060 because two GPUs/local checkpoint resources are absent.

Workstation logs are under `/workspace/cp-decode-review-20261027/`:
`standard-pcp-unit-20261028.log`, `standard-pcp-cuda-20261028.log`,
`standard-pcp-cpu-20261028.log`, and `standard-pcp-full-model-20261028.log`.
The 4060 checks use small model/HiCache collaborators; they do not run DSV4
FlashMLA, distributed model attention or HiCache DMA. The full-model tests need
two GPUs and local checkpoints and skip when those resources are absent.
