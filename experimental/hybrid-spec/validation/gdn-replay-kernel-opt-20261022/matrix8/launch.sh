#!/usr/bin/env bash
set -euo pipefail
source /root/rivermind-data/activate-sglang.sh
out="$(cd -- "$(dirname -- "$0")" && pwd)"
src="$out/source"
cd "$src"
export PYTHONPATH="$src/python:$src/experimental/hybrid-spec/src"
export SGLANG_PLUGINS=hybrid_spec
export SGLANG_MAMBA_CONV_DTYPE=float16
unset HYBRID_SPEC_CONFIG SGLANG_RAGGED_VERIFY_MODE SGLANG_MAMBA_SSM_DTYPE
k=16
export HYBRID_SPEC_CONFIG="$out/sa16-n8.json"
replay=(--enable-linear-replayssm-spec --linear-replayssm-cache-len 64)
case "$1" in
 baseline-replay) ;;
 optimized-replay) src="$out/source-opt" ;;
 *) echo "Unknown comparison mode" >&2; exit 2 ;;
esac
cd "$src"
export PYTHONPATH="$src/python:$src/experimental/hybrid-spec/src"
unset AGENT_TRACE_SCOPES
python - "$src" "$out" "$1" <<'PYCHECK'
import sys,json,hashlib
from pathlib import Path
src,out,mode=sys.argv[1:]
src,out=Path(src),Path(out)
name="source-opt-manifest.json" if mode=="optimized-replay" else "source-manifest.json"
manifest=json.loads((out/name).read_text())
for f,h in manifest.items(): assert hashlib.sha256((src/f).read_bytes()).hexdigest()==h,f
print("Validated source files before launch:",len(manifest),flush=True)
PYCHECK
exec python -m sglang.launch_server \
 --model-path /root/rivermind-data/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
 --language-only --quantization gptq_marlin --dtype float16 \
 --speculative-algorithm EAGLE --speculative-num-steps "$k" \
 --speculative-eagle-topk 1 --speculative-num-draft-tokens "$((k+1))" \
 --speculative-draft-model-path /root/rivermind-data/validation/split-graph-20261016/mtp-unquant-view \
 --speculative-draft-model-quantization unquant --speculative-moe-runner-backend triton \
 --linear-attn-backend triton --linear-attn-verify-backend triton \
 --attention-backend triton --speculative-draft-attention-backend triton \
 --mamba-ssm-dtype float32 "${replay[@]}" \
 --tp-size 1 --context-length 16384 --max-running-requests 8 \
 --max-mamba-cache-size 40 --max-total-tokens 131072 --mem-fraction-static 0.94 \
 --random-seed 42 --watchdog-timeout 1800 --cuda-graph-max-bs-decode 8 \
 --cuda-graph-backend-decode full --cuda-graph-backend-prefill disabled \
 --host 127.0.0.1 --port 39924
