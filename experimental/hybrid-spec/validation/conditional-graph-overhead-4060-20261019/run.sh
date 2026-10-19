#!/bin/sh
root=/workspace/conditional-graph-20261019
cd "$root/repo" || exit 1
PYTHONPATH="$root/repo/experimental/hybrid-spec/src" "$root/venv/bin/python" "$root/overhead/ablate_dispatch.py" --probe-dir "$root" --output-dir "$root/overhead" > "$root/overhead/run.log" 2>&1
printf '%s\n' "$?" > "$root/overhead/run.exit"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$root/overhead/post-test-gpu.txt"
