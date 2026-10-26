#!/usr/bin/env bash
# QuantTrio AWQ target, unquantized DFlash draft, A100-compatible backends.
set -euo pipefail
: "${TARGET_MODEL:?Set TARGET_MODEL to QuantTrio/Qwen3.5-35B-A3B-AWQ}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to z-lab/Qwen3.5-35B-A3B-DFlash}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}
dtype=${DTYPE:-float16}
export SGLANG_MAMBA_CONV_DTYPE="$dtype"
if [[ ${DETERMINISTIC:-0} == 1 ]]; then
  attention_backend=${ATTENTION_BACKEND:-triton}
  draft_backend=${DRAFT_ATTENTION_BACKEND:-triton}
else
  attention_backend=${ATTENTION_BACKEND:-flashinfer}
  draft_backend=${DRAFT_ATTENTION_BACKEND:-flashinfer}
fi
hicache_io=${HICACHE_IO_BACKEND:-kernel}
case "$hicache_io" in
  kernel) hicache_layout=${HICACHE_LAYOUT:-page_first} ;;
  direct) hicache_layout=${HICACHE_LAYOUT:-page_first_direct} ;;
  *) echo "HICACHE_IO_BACKEND must be kernel or direct" >&2; exit 2 ;;
esac
args=(
  --model-path "$TARGET_MODEL" --trust-remote-code --language-only
  --quantization awq_marlin --dtype "$dtype" --tp-size 1
  --attention-backend "$attention_backend"
  --linear-attn-prefill-backend triton --linear-attn-decode-backend triton
  --mamba-radix-cache-strategy extra_buffer
  --context-length 16384 --max-prefill-tokens "${MAX_PREFILL_TOKENS:-32768}"
  --max-total-tokens "${MAX_TOTAL_TOKENS:-32768}" --max-mamba-cache-size "${MAMBA_CACHE:-24}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-4}"
  --cuda-graph-max-bs-decode "${MAX_RUNNING_REQUESTS:-4}"
  --cuda-graph-backend-decode "${DECODE_GRAPH:-full}" --cuda-graph-backend-prefill disabled
  --mem-fraction-static "${MEM_FRACTION:-0.9}"
  --random-seed 42 --watchdog-timeout 300 --enable-metrics
  --host "${HOST:-127.0.0.1}" --port "${PORT:-39926}"
)
if [[ -n ${CHUNKED_PREFILL_SIZE:-} ]]; then
  args+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
fi
if [[ ${DFLASH:-1} == 1 ]]; then
  args+=(--speculative-algorithm DFLASH --speculative-dflash-block-size "${BLOCK_SIZE:-8}"
    --speculative-draft-model-path "$DRAFT_MODEL"
    --speculative-draft-model-quantization unquant
    --speculative-moe-runner-backend triton
    --speculative-draft-attention-backend "$draft_backend")
fi
if [[ ${PDMUX:-${DFLASH:-1}} == 1 ]]; then
  args+=(--enable-pdmux --pdmux-prefill-mode layer_split --sm-group-num 3
    --pdmux-config-path "${PDMUX_CONFIG:-$script_dir/qwen35_dflash_overlap_a100.yaml}")
fi
if [[ ${HICACHE:-1} == 1 ]]; then
  args+=(--enable-hierarchical-cache --hicache-size "${HICACHE_SIZE:-4}" --hicache-write-policy write_through
    --hicache-io-backend "$hicache_io" --hicache-mem-layout "$hicache_layout")
fi
if [[ ${REPLAYSSM_SPEC:-0} == 1 ]]; then
  args+=(--enable-linear-replayssm-spec
    --linear-replayssm-cache-len "${REPLAYSSM_CACHE_LEN:-32}")
fi
if [[ ${DETERMINISTIC:-0} == 1 ]]; then
  args+=(--enable-deterministic-inference)
fi
if [[ ${CPU_OVERLAP:-1} == 0 ]]; then
  args+=(--disable-overlap-schedule)
fi
exec python -m sglang.launch_server "${args[@]}" "$@"
