#!/bin/sh
root=/workspace/conditional-graph-20261019
out=/workspace/conditional-switch-20261019
cd "$out/repo" || exit 1
export PATH="$root/venv/bin:$root/cuda13/bin:$PATH"
export PYTHONPATH=experimental/hybrid-spec/src
export CUDA_HOME="$root/cuda13"
export CC=gcc-12 CXX=g++-12 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
python "$out/compare_dispatch.py" --helpers "$root/overhead" --probe-dir "$root" --output-dir "$out/measurements" > "$out/benchmark.log" 2>&1
printf '%s\n' "$?" > "$out/benchmark.exit"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$out/post-test-gpu.txt"
