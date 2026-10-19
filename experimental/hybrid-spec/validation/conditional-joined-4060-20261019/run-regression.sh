#!/bin/sh
root=/workspace/conditional-graph-20261019
out=/workspace/conditional-joined-20261019
cd "$out/repo" || exit 1
export PATH="$root/venv/bin:$root/cuda13/bin:$PATH"
export PYTHONPATH="$root:experimental/hybrid-spec/src"
export CUDA_HOME="$root/cuda13"
export CC=gcc-12 CXX=g++-12 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9
python -m pytest -p lean_cuda_plugin -p capture_gc_plugin experimental/hybrid-spec/tests/test_cuda.py -k '(compact_tail_graph and not arctic) or gpu_dynamic_rows' -q --tb=short --junitxml="$out/cuda.xml" > "$out/cuda.log" 2>&1
printf '%s\n' "$?" > "$out/cuda.exit"
python -m pytest experimental/hybrid-spec/tests/test_core.py experimental/hybrid-spec/tests/test_compact_tail.py experimental/hybrid-spec/tests/test_compact_verify.py experimental/hybrid-spec/tests/test_draft_recurrence.py experimental/hybrid-spec/tests/test_switch_graph.py -q --tb=short --junitxml="$out/components.xml" > "$out/components.log" 2>&1
printf '%s\n' "$?" > "$out/components.exit"
