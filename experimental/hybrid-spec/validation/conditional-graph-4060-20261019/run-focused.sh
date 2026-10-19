#!/bin/sh
root=/workspace/conditional-graph-20261019
while ! test -f "$root/install.exit"; do sleep 5; done
if test "$(cat "$root/install.exit")" != 0; then exit 1; fi
cd "$root/repo" || exit 1
export PATH="$root/venv/bin:$PATH"
export PYTHONPATH="$root:experimental/hybrid-spec/src"
python - <<'PY' > "$root/environment.txt" 2>&1
import torch,sys,triton
print(sys.version)
print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__)
print('device',torch.cuda.get_device_name())
print('conditional bindings',hasattr(torch.cuda.CUDAGraph,'begin_capture_to_if_node'))
x=torch.ones(2,device='cuda')
print('cuda computation', (x+x).cpu().tolist())
PY
python -m pytest -p lean_cuda_plugin experimental/hybrid-spec/tests/test_cuda.py -k 'conditional and standalone and not arctic' -q --tb=short --junitxml="$root/focused.xml" > "$root/focused.log" 2>&1
result=$?
printf '%s\n' "$result" > "$root/focused.exit"
