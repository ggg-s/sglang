#!/bin/sh
root=/workspace/conditional-graph-20261019
UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 /root/.local/bin/uv pip install --python "$root/venv/bin/python" ninja 'nvidia-cuda-nvcc==13.0.88' 'nvidia-cuda-crt==13.0.88' 'nvidia-nvvm==13.0.88' > "$root/compiler-install.log" 2>&1
printf '%s\n' "$?" > "$root/compiler-install.exit"
