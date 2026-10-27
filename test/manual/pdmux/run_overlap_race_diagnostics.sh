#!/usr/bin/env bash
# Wrap the original server launch; run the frozen client load separately.
set -euo pipefail

usage() {
  echo "Usage: $0 shadow|memcheck -- <original server command and arguments>" >&2
  exit 2
}

[[ $# -ge 3 ]] || usage
mode=$1
shift
[[ $1 == -- ]] || usage
shift
[[ $# -gt 0 ]] || usage
case "$mode" in
  shadow|memcheck) ;;
  *) usage ;;
esac

log_dir=${PDMUX_RACE_LOG_DIR:-/tmp/pdmux-race-$(date -u +%Y%m%dT%H%M%SZ)-$$}
mkdir -p "$log_dir"
export SGLANG_INVARIANT_CHECK=${SGLANG_INVARIANT_CHECK:-2}
if [[ $mode == shadow ]]; then
  export SGLANG_PDMUX_INDEX_SHADOW=1
else
  export SGLANG_PDMUX_INDEX_SHADOW=0
  sanitizer=${COMPUTE_SANITIZER:-compute-sanitizer}
  command -v "$sanitizer" >/dev/null 2>&1 || {
    echo "compute-sanitizer not found: set COMPUTE_SANITIZER to its path" >&2
    exit 127
  }
  "$sanitizer" --version > "$log_dir/sanitizer-version.txt" 2>&1 || true
fi

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  git rev-parse HEAD 2>/dev/null || true
  printf 'mode=%s\nSGLANG_PDMUX_INDEX_SHADOW=%s\nSGLANG_INVARIANT_CHECK=%s\n' \
    "$mode" "$SGLANG_PDMUX_INDEX_SHADOW" "$SGLANG_INVARIANT_CHECK"
  printf 'CUDA_LAUNCH_BLOCKING=%s\n' "${CUDA_LAUNCH_BLOCKING:-unset}"
  printf 'SGLANG_ENABLE_OVERLAP_PLAN_STREAM=%s\n' "${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-unset}"
  printf 'COMPUTE_SANITIZER_INJECTION_PATH=%s\n' "${COMPUTE_SANITIZER_INJECTION_PATH:-unset}"
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,uuid,memory.total --format=csv || true
  fi
} > "$log_dir/launch.txt"
echo "PDMux race diagnostics: $mode, logs: $log_dir"

if [[ $mode == memcheck ]]; then
  sanitizer_args=(--tool memcheck --target-processes all
    --error-exitcode 86 --log-file "$log_dir/memcheck.%p.log")
  if [[ -n ${COMPUTE_SANITIZER_INJECTION_PATH:-} ]]; then
    sanitizer_args+=(--injection-path "$COMPUTE_SANITIZER_INJECTION_PATH")
  fi
  set -- "$sanitizer" "${sanitizer_args[@]}" "$@"
fi
set +e
"$@" 2>&1 | tee "$log_dir/server.log"
server_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$server_status" > "$log_dir/exit-code.txt"
exit "$server_status"
