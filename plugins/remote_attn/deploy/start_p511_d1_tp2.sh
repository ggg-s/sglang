#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export HANDOFF_CTL_DIR=${HANDOFF_CTL_DIR:-${HANDOFF_CTL_ROOT:-/workspace/handoff_ctl_p511}/dst}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$HANDOFF_PYTHON" -m sglang.launch_server \
  --model-path "$HANDOFF_MODEL_PATH" --tp 2 --page-size 1 \
  --host 127.0.0.1 --port 31018 \
  --dtype bfloat16 --attention-backend flashinfer \
  --context-length 40960 --mem-fraction-static 0.75 \
  --max-running-requests 16 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --disaggregation-mode decode \
  --disaggregation-decode-retraction-backup "$HANDOFF_RETRACTION_BACKUP" \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 35409 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} "$@"
