#!/bin/bash
set -euo pipefail

deploy_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
profile=${D2D_PROFILE:-optimized}
case "$profile" in
  baseline|optimized) ;;
  *) echo "D2D_PROFILE must be baseline or optimized" >&2; exit 2 ;;
esac

case "${1:-}" in
  controller)
    shift
    flags=(--event-driven-reconcile --worker-status-snapshots)
    if [[ "$profile" == optimized ]]; then
      flags+=(--chain-drain)
    fi
    exec "${HANDOFF_PYTHON:-python3}" "$deploy_dir/../control_plane/migration_controller.py" \
      "${flags[@]}" "$@"
    ;;
  decode)
    shift
    if [[ $# -eq 0 ]]; then
      echo "Pass the existing Decode launch command after decode" >&2
      exit 2
    fi
    export HANDOFF_D2D_STATUS_SNAPSHOT=1
    export HANDOFF_D2D_WARM_TRANSPORT=0
    if [[ "$profile" == optimized ]]; then
      export HANDOFF_D2D_WARM_TRANSPORT=1
      export HANDOFF_D2D_WARM_RDMA_SLOTS=${HANDOFF_D2D_WARM_RDMA_SLOTS:-2}
    fi
    exec "$@"
    ;;
  *)
    echo "Usage: $0 controller --config FILE [ARGS...] | decode COMMAND [ARGS...]" >&2
    exit 2
    ;;
esac
