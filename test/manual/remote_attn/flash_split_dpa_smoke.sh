#!/usr/bin/env bash
# Same-node CUDA smoke test. Use a separate GPU set for each native PD role.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${MODEL:?Set MODEL to the DeepSeek-V4-Flash checkpoint with native DSpark draft}"
PYTHON=${PYTHON:-python3}
TP=${TP:-4}
P_GPUS=${P_GPUS:-0,1,2,3}
D_GPUS=${D_GPUS:-4,5,6,7}
P_PORT=${P_PORT:-31116}
D_PORT=${D_PORT:-31117}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-31997}
SMG=${SMG:-0}
SMG_PORT=${SMG_PORT:-31100}
P_PCP=${P_PCP:-0}
D_DPA=${D_DPA:-1}
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/flash-split-dpa.XXXXXX")
echo "Logs: $RUN_DIR"

export PYTHONPATH="$PWD/python:$PWD/plugins/remote_attn${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_PLUGINS=remote_attn
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=0 SGLANG_DSV4_UNIFIED_KV_FP8=0
export HANDOFF_PRE_LAYERWISE=remote HANDOFF_PD_DIR="$RUN_DIR/pd"
export HANDOFF_CTL_DIR="$RUN_DIR/control"
export HANDOFF_HYBRID_SPLIT_PREFILL=1
export HANDOFF_EXPERIMENTAL_PD_REMOTE=0 HANDOFF_PD_NETWORK=0 HANDOFF_REMOTE_GPU_TRANSFER=0
export HANDOFF_SPLIT_ADAPTIVE=0 HANDOFF_SPLIT_CAP_AUTO=0
export HANDOFF_SPLIT_DEFER_TICKS=0 HANDOFF_HYBRID_SPLIT_PREFILL_CHUNK_SIZE=1024
export HANDOFF_SPLIT_SUFFIX_TOKENS=4096
mkdir -p "$HANDOFF_PD_DIR" "$HANDOFF_CTL_DIR"

common=(
  --model-path "$MODEL" --trust-remote-code --language-model-only
  --tp "$TP" --moe-a2a-backend none
  --speculative-algorithm DSPARK --dsv4-attn-backend flashmla --page-size 256
  --disaggregation-transfer-backend mooncake
  --disaggregation-bootstrap-port "$BOOTSTRAP_PORT"
  --host 127.0.0.1 --context-length 32768
  --chunked-prefill-size 4096 --max-running-requests 32
  --cuda-graph-backend-prefill disabled --skip-server-warmup
)
if [[ -n ${DRAFT:-} ]]; then
  common+=(--speculative-draft-model-path "$DRAFT")
fi

p_groups=$TP
p_split_dpa=1
prefill_args=(--disaggregation-mode prefill --port "$P_PORT")
if [[ "$P_PCP" == 1 ]]; then
  p_groups=1
  p_split_dpa=0
  prefill_args+=(--enable-prefill-cp --cp-strategy interleave)
else
  prefill_args+=(--enable-dp-lm-head)
fi
prefill_args+=(--attn-dp-size "$p_groups")
d_groups=1
decode_args=(--disaggregation-mode decode --port "$D_PORT" --disable-radix-cache)
if [[ "$D_DPA" == 1 ]]; then
  d_groups=$TP
  decode_args+=(--enable-dp-lm-head)
fi
decode_args+=(--attn-dp-size "$d_groups")
if [[ "$SMG" == 1 ]]; then
  # SMG selects a P group independently of the random bootstrap room.
  export SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1
else
  prefill_args+=(--disable-radix-cache)
fi

"$PYTHON" -m sglang_remote_attn.check
HANDOFF_SPLIT_DPA="$p_split_dpa" SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION="$p_split_dpa" \
  CUDA_VISIBLE_DEVICES="$P_GPUS" "$PYTHON" -m sglang.launch_server "${common[@]}" \
  "${prefill_args[@]}" >"$RUN_DIR/p.log" 2>&1 &
p_pid=$!
HANDOFF_SPLIT_DPA="$D_DPA" SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION="$D_DPA" \
  SGLANG_LOG_DECODE_GRAPH_KEY=1 CUDA_VISIBLE_DEVICES="$D_GPUS" \
  "$PYTHON" -m sglang.launch_server "${common[@]}" \
  "${decode_args[@]}" \
  >"$RUN_DIR/d.log" 2>&1 &
d_pid=$!
cleanup() {
  kill "$p_pid" "$d_pid" 2>/dev/null || true
  wait "$p_pid" "$d_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$PYTHON" - "$p_groups" "$d_groups" "$P_PORT" "$D_PORT" "$BOOTSTRAP_PORT" "$RUN_DIR" \
  "$p_pid" "$d_pid" "$SMG" "$SMG_PORT" <<'PY'
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

p_groups, groups, prefill_port, decode_port, bootstrap_port = map(int, sys.argv[1:6])
log_dir = Path(sys.argv[6])
pids = list(map(int, sys.argv[7:9]))
smg, smg_port = map(int, sys.argv[9:11])
prefill, decode = (f"http://127.0.0.1:{port}" for port in (prefill_port, decode_port))
deadline = time.monotonic() + 1200
for url in (prefill, decode):
    while time.monotonic() < deadline:
        for pid in pids:
            os.kill(pid, 0)
        try:
            if requests.get(f"{url}/model_info", timeout=2).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        raise TimeoutError(f"Server did not start: {url}")

def smoke_smg():
    router_url = f"http://127.0.0.1:{smg_port}"
    with (log_dir / "smg.log").open("w") as log:
        router = subprocess.Popen(
            [
                sys.executable, "-m", "sglang_router.launch_router",
                "--host", "127.0.0.1", "--port", str(smg_port),
                "--pd-disaggregation", "--dp-aware",
                "--prefill", prefill, str(bootstrap_port), "--decode", decode,
                "--prefill-policy", "cache_aware", "--decode-policy", "power_of_two",
            ],
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 120
            counts = None
            while time.monotonic() < deadline:
                if router.poll() is not None:
                    raise RuntimeError(f"SMG exited. See {log.name}")
                try:
                    response = requests.get(f"{router_url}/workers", timeout=2)
                    if response.ok:
                        listing = response.json()
                        stats = listing["stats"]
                        counts = stats["prefill_count"], stats["decode_count"]
                        if counts == (p_groups, groups) and all(
                            worker["is_healthy"] for worker in listing["workers"]
                        ):
                            break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"Expected {p_groups} P and {groups} D groups, got counts={counts}. "
                    f"Check SMG attention-DP discovery in {log.name}"
                )
            print(f"SMG registered P={counts[0]} and D={counts[1]} groups", flush=True)

            # The HTTP cache_aware policy matches the request text. Leave ranks
            # and bootstrap fields unset so SMG selects both groups itself.
            prefix = " cache" * 8192
            cached = []
            for tail in (" cold", " warm"):
                response = requests.post(
                    f"{router_url}/generate",
                    json={
                        "text": prefix + tail * 6144,
                        "sampling_params": {
                            "temperature": 0, "max_new_tokens": 64, "ignore_eos": True
                        },
                    },
                    timeout=300,
                )
                response.raise_for_status()
                meta = response.json()["meta_info"]
                assert meta["completion_tokens"] == 64, meta
                assert meta.get("spec_verify_ct", 0) > 0, meta
                rid = meta["id"]
                decode_log = (log_dir / "d.log").read_text()
                assert f"[handoff-split] hybrid PD rid={rid} " in decode_log, meta
                cached.append(meta["cached_tokens"])
                print(f"{rid}: cached={cached[-1]}, verify={meta['spec_verify_ct']}", flush=True)
            assert cached[0] == 0 and cached[1] >= 256, cached
            print("PASS: SMG DP-aware routing, Flash split + DSpark, and warm P radix hit")
        finally:
            if router.poll() is None:
                router.terminate()
                try:
                    router.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    router.kill()
                    router.wait()


if smg:
    smoke_smg()
    sys.exit(0)

ready = threading.Event()
stop = threading.Event()


def send_pair(rid, room_index, rank, prompt, output, *, keep=False):
    prefill_rank = rank % p_groups
    payload = {
        "rid": rid,
        "input_ids": [1234] * prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": output, "ignore_eos": True},
        "routed_dp_rank": rank,
        "disagg_prefill_dp_rank": prefill_rank,
        "bootstrap_host": "127.0.0.1",
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": room_index * groups + rank,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        p = executor.submit(
            requests.post, f"{prefill}/generate",
            json=dict(payload, routed_dp_rank=prefill_rank), timeout=300,
        )
        with requests.post(
            f"{decode}/generate",
            json=dict(payload, stream=keep),
            stream=keep,
            timeout=300,
        ) as response:
            response.raise_for_status()
            if keep:
                for line in response.iter_lines(chunk_size=64):
                    if line.startswith(b"data: ") and line != b"data: [DONE]":
                        meta = json.loads(line[6:]).get("meta_info", {})
                        if meta.get("completion_tokens", 0) >= 2:
                            ready.set()
                    if stop.is_set():
                        break
            else:
                meta = response.json()["meta_info"]
                assert meta["completion_tokens"] == output, (rid, meta)
                assert meta.get("spec_verify_ct", 0) > 0, (rid, meta)
                print(f"{rid}: {meta['spec_verify_ct']} verify steps", flush=True)
        p.result().raise_for_status()


def has_step(rid, peer_mode, coordinated):
    return any(
        "extend" in modes.split(",")
        and peer_mode in modes.split(",")
        and mixed == str(coordinated)
        for modes, mixed in re.findall(
            rf"\[handoff-split-dpa\] rid={rid} modes=([\w,]+) coordinated=(\w+)",
            (log_dir / "d.log").read_text(),
        )
    )


# No peer request during the first suffix.
d_log = log_dir / "d.log"
before = len(d_log.read_text())
send_pair("flash-smoke-idle", 10, 0, 16384, 64)
isolated_log = d_log.read_text()[before:]
assert "[handoff-split] hybrid PD rid=flash-smoke-idle " in isolated_log, isolated_log
if groups > 1:
    assert has_step("flash-smoke-idle", "idle", False), "Expected suffix + idle"
for worker in ("draft", "target"):
    assert re.search(
        rf"Decode graph replay: worker={worker} .*mode=TARGET_VERIFY raw_bs=[1-9]",
        isolated_log,
    ), f"Expected active {worker} CUDA graph replay after the split"

if groups > 1:
    # Wait until group 1 is generating tokens before asking group 0 to split.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        decoding = executor.submit(send_pair, "flash-smoke-peer", 11, 1, 1024, 16384, keep=True)
        try:
            assert ready.wait(120), "Peer did not start DSpark decode"
            send_pair("flash-smoke-mixed", 12, 0, 16384, 64)
            assert has_step("flash-smoke-mixed", "verify", True), "Expected coordinated suffix + verify"
        finally:
            stop.set()
        decoding.result()

# A pinned 4096-token suffix on a 6144-token prompt would leave P 2048
# tokens, below the earliest cut 3072. D offers; P sends the full prompt.
send_pair("flash-smoke-fallback", 13, 0, 6144, 64)
logs = (log_dir / "d.log").read_text()
if groups > 1:
    assert "stage=decode_commit reason=native_pd rid=flash-smoke-fallback" in logs, (
        "Expected native PD commit after declining a split offer"
    )
assert "[handoff-split] hybrid PD rid=flash-smoke-fallback " not in logs
print("PASS: split with draft/verify graph replay and native PD fallback")
PY
