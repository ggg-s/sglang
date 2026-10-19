#!/bin/sh
/root/.local/bin/uv pip install --python /workspace/conditional-graph-20261019/venv/bin/python 'torch==2.14.1' pytest msgspec numpy --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match > /workspace/conditional-graph-20261019/install.log 2>&1
result=$?
printf '%s\n' "$result" > /workspace/conditional-graph-20261019/install.exit
