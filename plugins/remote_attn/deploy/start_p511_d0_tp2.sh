#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export HANDOFF_CTL_DIR=${HANDOFF_CTL_DIR:-${HANDOFF_CTL_ROOT:-/workspace/handoff_ctl_p511}/src}
export HANDOFF_REMOTE_HIGH_PRIORITY=${HANDOFF_REMOTE_HIGH_PRIORITY:-1}
export HANDOFF_FAST_FREEZE=${HANDOFF_FAST_FREEZE:-1}
export HANDOFF_FAST_BUFFERS=${HANDOFF_FAST_BUFFERS:-2}
export HANDOFF_FAST_COPY_STREAMS=${HANDOFF_FAST_COPY_STREAMS:-2}
export HANDOFF_FAST_ASYNC=${HANDOFF_FAST_ASYNC:-1}
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$HANDOFF_PYTHON" -m sglang.launch_server \
  --model-path "$HANDOFF_MODEL_PATH" --tp 2 --page-size 1 \
  --host 127.0.0.1 --port 31017 \
  --dtype bfloat16 --attention-backend flashinfer \
  --context-length 40960 --mem-fraction-static 0.75 \
  --max-running-requests 8 --chunked-prefill-size 2048 \
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule \
  --disaggregation-mode decode \
  --disaggregation-decode-retraction-backup "$HANDOFF_RETRACTION_BACKUP" \
  --disaggregation-transfer-backend mooncake_tcp \
  --disaggregation-bootstrap-port 35409 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} "$@"
