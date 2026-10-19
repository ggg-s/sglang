#!/bin/sh
root=/workspace/conditional-graph-20261019
while ! test -f "$root/focused.exit"; do
 if test -f "$root/install.exit" && test "$(cat "$root/install.exit")" != 0; then exit 1; fi
 sleep 5
done
if test "$(cat "$root/focused.exit")" != 0; then exit 1; fi
cd "$root/repo" || exit 1
export PATH="$root/venv/bin:$PATH"
export PYTHONPATH="$root:experimental/hybrid-spec/src"
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests/test_cuda.py -k '(compact_tail_graph and not arctic) or gpu_dynamic_rows' -q --tb=short --junitxml="$root/cuda-regression.xml" > "$root/cuda-regression.log" 2>&1
printf '%s\n' "$?" > "$root/cuda-regression.exit"
python -m pytest experimental/hybrid-spec/tests/test_core.py experimental/hybrid-spec/tests/test_compact_tail.py experimental/hybrid-spec/tests/test_compact_verify.py experimental/hybrid-spec/tests/test_draft_recurrence.py -q --tb=short --junitxml="$root/component-regression.xml" > "$root/component-regression.log" 2>&1
printf '%s\n' "$?" > "$root/component-regression.exit"
python "$root/dispatch_probe.py" > "$root/dispatch-probe.json" 2> "$root/dispatch-probe.err"
printf '%s\n' "$?" > "$root/dispatch-probe.exit"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader > "$root/post-test-gpu.txt"
