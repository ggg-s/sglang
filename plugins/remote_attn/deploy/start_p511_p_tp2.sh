#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HANDOFF_CTL_DIR=${HANDOFF_CTL_DIR:-${HANDOFF_CTL_ROOT:-/workspace/handoff_ctl_p511}/p}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$HANDOFF_PYTHON" -m sglang.launch_server \
  --model-path "$HANDOFF_MODEL_PATH" --tp 2 --page-size 1 \
  --host 127.0.0.1 --port 31016 \
  --dtype bfloat16 --attention-backend flashinfer \
  --context-length 40960 --mem-fraction-static 0.85 \
  --max-running-requests 8 --chunked-prefill-size 2048 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 35409 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} "$@"
