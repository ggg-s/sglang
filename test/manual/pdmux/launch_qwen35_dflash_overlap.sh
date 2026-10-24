#!/usr/bin/env bash
# Single A100 40 GB, FP16 GPTQ-Marlin target, unquantized DFlash draft.
set -euo pipefail
: "${TARGET_MODEL:?Set TARGET_MODEL to Qwen3.5-35B-A3B-GPTQ-Int4}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to Qwen3.5-35B-A3B-DFlash}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SGLANG_MAMBA_CONV_DTYPE=float16
exec python -m sglang.launch_server \
  --model-path "$TARGET_MODEL" --language-only \
  --quantization gptq_marlin --dtype float16 \
  --linear-attn-backend triton --attention-backend triton --tp-size 1 \
  --context-length 16384 --max-running-requests 4 \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --max-total-tokens 32768 --max-mamba-cache-size 20 --mem-fraction-static 0.92 \
  --speculative-algorithm DFLASH --speculative-num-steps 1 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens "${BLOCK_SIZE:-16}" \
  --speculative-draft-model-path "$DRAFT_MODEL" \
  --speculative-draft-model-quantization unquant \
  --speculative-moe-runner-backend triton --speculative-draft-attention-backend triton \
  --enable-pdmux --pdmux-prefill-mode layer_split --sm-group-num 3 \
  --pdmux-config-path "${PDMUX_CONFIG:-$script_dir/qwen35_dflash_overlap_a100.yaml}" \
  --enable-hierarchical-cache --hicache-size 4 --hicache-write-policy write_through \
  --hicache-io-backend direct --hicache-mem-layout page_first_direct \
  --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 4 --random-seed 42 --watchdog-timeout 300 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-39925}" "$@"
