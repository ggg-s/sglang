# DeepSeek-V4 DSpark + CPU overlap + PDMux + HiCache, TP2

Use `codex/pdmux-tp-cpu-overlap`. This configuration uses plain TP2, PP1,
EP1, MoE-DP1, attention-CP1 and DCP1. Leave DP attention disabled. PDMux's
duplicated communication groups cover full-width TP; subgroup parallelism
remains rejected. CPU overlap is the asynchronous CPU/GPU scheduler for CUDA,
and is enabled by omitting `--disable-overlap-schedule`.

The DSpark prefill finalizer now broadcasts on the current prefill TP group;
draft proposal, verify planning and acceptance keep the decode group. Verify
metadata uses the per-stream decode backend. DeepSeek-V4 target/draft helper
streams are disabled under both PDMux modes so work stays on the managed lane.
The existing layer-split auxiliary-hidden capture and packed HiCache draft-KV
registration remain in use. This patch admits CPU overlap with `layer_split`;
the `standard`, temporal and decode-ready experiments retain their existing
restrictions.

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

## 4060 regression coverage

The 4060 is a single 8-GiB SM89 GPU and cannot load the full target model or
exercise two-GPU NCCL. GPU checks use synthetic tensors with the production
DSpark finalizer, MLA injector and multi-stage KV projection on alternating
PDMux green-context streams, interleaved with decode CUDA graph replays. The
store fixture omits FlashMLA encoding and HiCache DMA. Separate tests exercise
actual singleton NCCL operations and DSpark Triton-vs-Torch kernel parity.
Two-process Gloo tests check prefill/decode broadcast ownership and rank-uniform
chunk completion and HiCache pumping. See the accompanying validation JSON for
the exact results and limits.
