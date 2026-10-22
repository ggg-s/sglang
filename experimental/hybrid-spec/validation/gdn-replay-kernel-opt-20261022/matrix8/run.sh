#!/usr/bin/env bash
set -uo pipefail
source /root/rivermind-data/activate-sglang.sh
out="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONPATH="$out/source/python:$out/source/experimental/hybrid-spec/src"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > "$out/gpu-telemetry.csv" &
monitor=$!
echo "$monitor" > "$out/telemetry.pid"
trap 'kill "$monitor" 2>/dev/null || true' EXIT
python "$out/run_matrix.py" > "$out/matrix.log" 2>&1
code=$?
echo "$code" > "$out/matrix.exit"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$out/post-gpu.txt"
exit "$code"
