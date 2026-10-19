#!/bin/sh
root=/workspace/conditional-graph-20261019
out=/workspace/conditional-switch-fused-20261019
cd "$out/repo" || exit 1
export PATH="$root/venv/bin:$root/cuda13/bin:$PATH"
export PYTHONPATH=experimental/hybrid-spec/src
export CUDA_HOME="$root/cuda13"
export CC=gcc-12 CXX=g++-12 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
for seed in 327 911; do
  python "$out/compare_rounds.py" --baseline-src /workspace/conditional-switch-20261019/repo/experimental/hybrid-spec/src --helpers "$root/overhead" --probe-dir "$root" --output-dir "$out/seed-$seed" --seed "$seed" > "$out/seed-$seed.log" 2>&1
  result=$?
  printf '%s\n' "$result" > "$out/seed-$seed.exit"
  if [ "$result" -ne 0 ]; then exit "$result"; fi
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$out/post-test-gpu.txt"
