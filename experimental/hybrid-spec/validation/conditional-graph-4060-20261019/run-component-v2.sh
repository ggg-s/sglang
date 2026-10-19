#!/bin/sh
root=/workspace/conditional-graph-20261019
cd "$root/repo" || exit 1
export PATH="$root/venv/bin:$root/cuda13/bin:$PATH"
export PYTHONPATH="$root:experimental/hybrid-spec/src"
export CUDA_HOME="$root/cuda13"
export CC=gcc-12 CXX=g++-12 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
python -m pytest experimental/hybrid-spec/tests/test_core.py experimental/hybrid-spec/tests/test_compact_tail.py experimental/hybrid-spec/tests/test_compact_verify.py experimental/hybrid-spec/tests/test_draft_recurrence.py -q --tb=short --junitxml="$root/component-regression-v2.xml" > "$root/component-regression-v2.log" 2>&1
printf '%s\n' "$?" > "$root/component-regression-v2.exit"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$root/post-test-gpu.txt"
