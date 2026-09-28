#!/usr/bin/env bash
# DeepSeek-V4-Flash DCP equivalence regression on one 8x H100 node.
#
# Baseline and candidate each use all eight GPUs, so run them in sequence.
#
# Typical workflow:
#   ACTION=serve-baseline bash scripts/playground/dcp_equivalence_run.sh
#   # In another shell while baseline is healthy:
#   ACTION=capture-baseline bash scripts/playground/dcp_equivalence_run.sh
#   # Stop baseline, then start candidate:
#   ACTION=serve-candidate bash scripts/playground/dcp_equivalence_run.sh
#   # In another shell while candidate is healthy:
#   ACTION=compare-candidate bash scripts/playground/dcp_equivalence_run.sh
#
# ACTION values:
#   serve-baseline      launch the dcp_size=1 server for this node
#   serve-candidate     launch the dcp_size=N server for this node
#   capture-baseline    save baseline responses to RESULTS_FILE
#   compare-candidate   compare candidate responses with RESULTS_FILE
#   live-compare        compare two already-running endpoints
set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/dcp_equiv_logs}"
mkdir -p "${LOG_DIR}"

ACTION="${ACTION:-serve-candidate}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-4}"
DCP_SIZE="${DCP_SIZE:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-127.0.0.1:30300}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
BASELINE_URL="${BASELINE_URL:-http://127.0.0.1:${PORT}}"
CANDIDATE_URL="${CANDIDATE_URL:-http://127.0.0.1:${PORT}}"
RESULTS_FILE="${RESULTS_FILE:-${LOG_DIR}/baseline_dcp1_results.json}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
CONCURRENCY="${CONCURRENCY:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

COMMON_ENV=(SGLANG_OPT_USE_ONLINE_COMPRESS=0)

COMMON_SERVER_ARGS=(
  --trust-remote-code
  --model-path "${MODEL_PATH}"
  --tp "${TP_SIZE}"
  --dp-size "${DP_SIZE}"
  --enable-dp-attention
  --cuda-graph-max-bs 128
  --max-running-requests 256
  --enable-metrics
  --host "${HOST}"
  --port "${PORT}"
  --mem-fraction-static 0.75
  --moe-runner-backend marlin
  --moe-a2a-backend none
  --dist-init-addr "${DIST_INIT_ADDR}"
  --nnodes "${NNODES}"
  --node-rank "${NODE_RANK}"
  --tool-call-parser deepseekv4
  --reasoning-parser deepseek-v4
  --chunked-prefill-size 4096
  --disable-cuda-graph
)

run_server() {
  local dcp_size="$1"
  local label="$2"
  local logfile="${LOG_DIR}/${label}_node${NODE_RANK}_dcp${dcp_size}.log"
  local dcp_env=()
  local dcp_args=()

  if [[ "${dcp_size}" -gt 1 ]]; then
    dcp_env=(SGLANG_DSV4_ENABLE_DCP=1)
    dcp_args=(--dcp-size "${dcp_size}")
  else
    dcp_args=(--dcp-size 1)
  fi

  echo "[serve] label=${label} node_rank=${NODE_RANK} dcp_size=${dcp_size}"
  echo "[serve] log=${logfile}"
  # shellcheck disable=SC2086
  env "${COMMON_ENV[@]}" "${dcp_env[@]}" \
    sglang serve \
    "${COMMON_SERVER_ARGS[@]}" \
    "${dcp_args[@]}" \
    ${EXTRA_ARGS} \
    2>&1 | tee "${logfile}"
}

run_check() {
  python "${WORK_DIR}/scripts/playground/dcp_equivalence_check.py" "$@"
}

case "${ACTION}" in
  serve-baseline)
    run_server 1 baseline
    ;;
  serve-candidate)
    run_server "${DCP_SIZE}" "candidate"
    ;;
  capture-baseline)
    run_check \
      --capture-url "${BASELINE_URL}" \
      --capture-output "${RESULTS_FILE}" \
      --model-path "${MODEL_PATH}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-tokens "${MAX_TOKENS}" \
      --concurrency "${CONCURRENCY}"
    ;;
  compare-candidate)
    run_check \
      --baseline-results "${RESULTS_FILE}" \
      --candidate-url "${CANDIDATE_URL}" \
      --model-path "${MODEL_PATH}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-tokens "${MAX_TOKENS}" \
      --concurrency "${CONCURRENCY}"
    ;;
  live-compare)
    run_check \
      --baseline-url "${BASELINE_URL}" \
      --candidate-url "${CANDIDATE_URL}" \
      --model-path "${MODEL_PATH}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-tokens "${MAX_TOKENS}" \
      --concurrency "${CONCURRENCY}"
    ;;
  *)
    echo "Unknown ACTION=${ACTION}" >&2
    exit 2
    ;;
esac
