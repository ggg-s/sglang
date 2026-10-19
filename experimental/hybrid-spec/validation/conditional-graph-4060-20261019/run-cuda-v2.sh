#!/bin/sh
root=/workspace/conditional-graph-20261019
cd "$root/repo" || exit 1
PATH="$root/venv/bin:$PATH" PYTHONPATH="$root:experimental/hybrid-spec/src" python -m pytest -p lean_cuda_plugin -p capture_gc_plugin experimental/hybrid-spec/tests/test_cuda.py -k '(compact_tail_graph and not arctic) or gpu_dynamic_rows' -q --tb=short --junitxml="$root/cuda-regression-v2.xml" > "$root/cuda-regression-v2.log" 2>&1
printf '%s\n' "$?" > "$root/cuda-regression-v2.exit"
