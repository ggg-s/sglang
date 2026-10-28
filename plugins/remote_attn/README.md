# Remote attention / progressive KV handoff plugin

中文部署入口：[内网部署手册](DEPLOYMENT-ZH.md)，包含交付清单、Mooncake 版本、离线准备、安装、启动和迁移验收。

This ports the supplied P0–P5.18 experiment onto local `main` at
`1335d372f0a58857422e40b89a097daa7d3bf86c`, using the existing
`sglang.srt.plugins` entry point. It is an experimental port, with CPU contract
tests and single-GPU CUDA/Mooncake probes. User-reported 5090 TP2 equivalence,
fault results and the resulting fixes are recorded in
[validation/5090-intranet.md](validation/5090-intranet.md); the merged fixes
still need acceptance on the target deployment.

## Files and provenance

- `sglang_remote_attn/handoff_control.py`: adapted runtime, including Copy, Layer,
  Remote, Fast Freeze/Resume, RemoteHub, progressive LOCAL, epochs, fences and leases.
- `sglang_remote_attn/hooks.py`: 21 function-boundary hooks. No scheduler event
  loop, graph runner, allocation function or Mooncake connector is copied/replaced.
- `control_plane/p2d_driver.py`: original supplied driver. `p4_gateway.py` now
  carries a targeted C1 reservation/epoch fix; its protocol and fixed state paths
  remain experimental. Original input hashes stay in the provenance manifest.
- `deploy/`: the three functional P/D0/D1 TP2 launch configurations and common
  transport defaults. Ports, GPU assignments and topology retain the supplied defaults.
- `migration/`: the supplied reference patches and porting instructions, plus
  `source-manifest.json`. These patches are provenance, **not patches to apply to main**.

The reported experimental SGLang base is `3a869635c` on `feat/dsv4-pdmux-opt`,
with uncommitted changes; experiment commit `17261ca0` is the supplied
`pd-handoff-p518-stable` reference. The actual input runtime MD5 is
`ad5290feffe0109dc8bbd18c94a4bafb`, which differs from the documented
`e545304a50a64f64e1f1cf9e679e634f`. The original file was not modified.
The manifest records the input SHA256 values so this discrepancy stays visible.
P5.19 now has an opt-in same-host pre-layerwise PD prototype; see the
[implementation and experiment guide](P5.19-PRE-LAYERWISE-ZH.md). GPU E2E validation is pending.
The [functional parity checklist](FUNCTIONAL-PARITY.md) maps the supplied
P0–P5.18 features to this port and distinguishes code coverage from deployment acceptance.

## Integration boundaries

| Original hook | Plugin target / behavior |
| --- | --- |
| All normal / overlap / decode input loops | Wrap `Scheduler.process_input_requests`; ordinary input first, broadcast controls next, housekeeping last, including empty polls |
| Command fan-out | Inject a sentinel before `_broadcast_reqs_across_ranks`, preserving local timeout aborts |
| TP command agreement | Compare command outcomes and logical scheduling state on the existing TP CPU group; stop the engine if ranks diverge |
| Hub startup / shutdown | Wrap scheduler construction and `run_event_loop` with cleanup in `finally` |
| Imported prefix | Skip tree matching for pending imports; supply their mapping through `_prefix_kv_indices`; upstream allocation writes the row; clear pending mapping only on success |
| Decode-style resume | Preserve `Req.kv` and its allocated row; set committed and allocated KV lengths; use the current `process_prebuilt(future_map)` API |
| KV release / retraction | Fence target receiver writes before upstream frees KV; handle imports aborted before row admission |
| Normal SSE output | Filter `_handoff_req` out of the regular output streamer; gateway/status retains output ownership |
| CUDA graphs | Gate decode replay on remote work, prefill replay on target states; stamp graph step metadata at `execute` |
| Mooncake | Create the send mutex before constructor threads start; serialize process-wide transport batches; probe changed peers before upstream registration publication |

Main's existing `prefill_decode_interval` implements P5.10. The old scheduler
instrumentation in the reference patch is not installed.

Core integration includes the typed `SGLANG_MOONCAKE_SEND_LOCK` declaration,
optional Mooncake buffer-subset/completion fields, and a Decode metadata commit
option that retains its receiver. These additions keep their ordinary behavior
by default; the P5.19 adapter is enabled only through the plugin and its explicit mode. No source patch in Mooncake C++ is needed.

## Supported port envelope

The initial port targets the supplied single-host Qwen3-8B setup: CUDA,
plain TP (including TP2, with the same TP width across roles), dense NHD MHA KV in BF16/FP16, page size 1,
radix cache disabled, and scheduler overlap disabled. The functional launchers
also disable CUDA graphs and select FlashInfer, as in the original experiment.
For TP > 1, disabling both prefill and decode CUDA graphs is mandatory and
checked at startup: rank-local transfer progress cannot choose graph versus
eager execution consistently without a separate replay agreement protocol.

The plugin rejects PP/DP/CP/DCP, speculative decoding, mixed chunks, unified KV,
HiCache, hybrid/MLA/quantized KV pools and live PD role switching. These layouts
need their own transfer/state-lifetime adapters. Rejecting them avoids silently
running the dense KV protocol against an incompatible pool.

Disabling scheduler overlap does **not** disable this feature's transfer/compute
overlap: Fast Freeze stages KV on copy streams while Remote attention serves
layers still on the source, then each completed layer switches to LOCAL.

The matching Mooncake reference is the CUDA 13 wheel
`mooncake-transfer-engine-cuda13==0.3.13.post1`. Its `TransferEngine.send_probe`
API is required by this port; a missing API fails startup instead of silently
omitting the restarted-peer repair. If three probes fail, the bootstrap thread
keeps running and the peer stays blacklisted for upstream failed-session recovery.

## Install and preflight on the GPU host

Run from the SGLang checkout using the serving Python environment (Python 3.12+):

```bash
python -m pip install -e ./python
python -m pip install --no-deps -e ./plugins/remote_attn
python -m sglang_remote_attn.check
```

Preflight checks entry-point installation, all 16 target interfaces, the env
declaration and Mooncake's `send_probe`. It does not start a server or validate
GPU transfer correctness. With an explicitly enabled plugin, registration errors
or missing hooks stop startup. Ordinary package installation without
`HANDOFF_CTL_DIR` registers no hooks.

For interface checks on a machine without CUDA/Mooncake:

```bash
PYTHONPATH=plugins/remote_attn python -m sglang_remote_attn.check --source-only
```

## Local validation

- 36 CPU contract tests passed, using real CPU PyTorch tensors and source-loaded
  upstream pool/KV methods. The local environment lacks the full SGLang model/eval
  dependencies, so its reduced test harness also source-loaded the repository's
  `CustomTestCase`; full installed-runtime import was not exercised locally.
- A real two-process Gloo test covers leader-only command polling, broadcast
  alongside ordinary/timeout-abort requests, equal rejections, rank failures
  and scheduling divergence. It does not exercise CUDA or NCCL.
- Transfer tests cover a three-buffer/one-slot ACK cycle, rank device selection,
  source staging reuse, receiver/source fences, unknown TE completion, RPC
  stream fences, incomplete-layer wakeups and teardown failures.
- All 16 hook targets and their positional/keyword call shapes match this checkout.
- Python compilation, Ruff F checks, isort and formatting, shell syntax, wheel build and
  wheel entry-point metadata checks passed.
- Launcher checks captured the real shell commands and environment without starting
  servers: all three retain GPU pairs, ports, TP2, role directories and required envs.
- The driver still matches its supplied SHA256; the gateway changes are covered
  by C1 reservation, stale-epoch and cutover/drain ordering regressions.
- RTX 4060 Laptop (8 GB): FP16/BF16 split-prefix attention matched an FP32
  reference; five real two-process CUDA/Mooncake transfer configurations and
  a cancellation/reuse check passed. A CUDA trace recorded H2D/GEMM overlap.
  See [validation/4060.md](validation/4060.md) for the scope and raw results.
- After the intranet fixes, all 10 CUDA probes passed, including two real
  RemoteHub/TargetClient socket cases with a missing local layer, three RPCs
  each and FP32 reference comparisons. Missing layers stayed Remote and copied
  layers stayed LOCAL. See [the new results](validation/4060-intranet-fixes-results.json).

In a complete SGLang test environment, run the registered test directly:

```bash
python test/registered/unit/plugins/test_remote_attn.py
```

See [REVIEW.md](REVIEW.md) for the review findings, fixes and remaining GPU
acceptance requirements. On uncertain transport completion, registered host
buffers are quarantined until completion is proven; `quarantined_transfers` in
status reports them. A completed local-drop transfer releases host storage
without requiring every layer to be LOCAL. For disconnects, sender completion
records under the shared control root allow idle-loop retries after CUDA fencing.
Native unregister failures retain storage and retry too. A CUDA reader/writer that cannot be fenced prevents KV reuse.
These failures can retain memory and require a process restart. TP command
agreement stops inconsistent scheduling; it does not provide atomic rollback
or automatic recovery after a partial-rank failure.
The full repository Ruff rule set still reports style/exception-policy issues
in the imported experimental runtime; this is not a clean full-lint result.

The standalone CUDA probe needs CUDA PyTorch, FlashInfer (including a matching
nvcc/ninja toolchain), and the Mooncake wheel. With these installed:

```bash
PYTHONPATH=plugins/remote_attn python test/manual/remote_attn/cuda_handoff_probe.py \
  --output /tmp/remote-attn-cuda --profile
```

It uses separate source/target processes on one GPU, small KV storage fixtures,
and real transport, copy and attention operations. It does not start an SGLang
model server or exercise TP2.

## Functional startup

Each launcher retains P/D0/D1 ports `31016/31017/31018`, TP2, bootstrap port
`35409`, GPU pairs `0,1` / `2,3` / `4,5`, and the original memory/context limits.
Set `HANDOFF_PYTHON`, `HANDOFF_MODEL_PATH`, `CUDA_VISIBLE_DEVICES` or
`HANDOFF_CTL_ROOT` to adapt paths/topology. Extra CLI arguments can be passed
directly to each script. Existing simple `HANDOFF_SERVER_EXTRA` word splitting
is retained; pass quoted arguments directly on the command line.

```bash
export HANDOFF_CTL_ROOT=/workspace/handoff_ctl_p511
export HANDOFF_PYTHON=/opt/conda/envs/sglang/bin/python
export HANDOFF_MODEL_PATH=/workspace/models/Qwen3-8B

bash plugins/remote_attn/deploy/start_p511_p_tp2.sh  > /tmp/remote-attn-p.log 2>&1 &
bash plugins/remote_attn/deploy/start_p511_d0_tp2.sh > /tmp/remote-attn-d0.log 2>&1 &
bash plugins/remote_attn/deploy/start_p511_d1_tp2.sh > /tmp/remote-attn-d1.log 2>&1 &
```

Launchers add `remote_attn` to the `SGLANG_PLUGINS` allowlist, create each role's
control directory and run preflight before launching SGLang. The role directories
are `p`, `src` and `dst`; the gateway uses their parent directory.

`deploy/common.sh` retains the functional transport defaults for all roles:

```bash
MC_FORCE_TCP=1
SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES=2048
MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER=8192
MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER=8192
SGLANG_ENABLE_FAILED_SESSION_PROBE=1
SGLANG_FAILED_SESSION_PROBE_INTERVAL_S=1.0
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0
SGLANG_MOONCAKE_SEND_LOCK=1
```

The native PD connector on this main actively splits index batches into
synchronous TE calls. The former deployment value `8` introduced thousands
of calls at 32K context. The default is now `2048`, matching the launchers'
`--chunked-prefill-size 2048`. If changing chunk size, review the batch bound
and transport queue limits together. An existing exported value still overrides
the default; clear a stale `8` or explicitly export `2048` before restarting.

Remote mode can keep serving a frozen prefix after a local import/drop error.
Only H2D-completed layers become LOCAL; missing layers keep using Remote RPC.
Abort, missing clients and unsafe GPU copy fences still reject serving. A local
import error also prevents the slow-RPC policy from switching into layer-wait.

D0 additionally retains `HANDOFF_FAST_FREEZE=1`, `HANDOFF_FAST_BUFFERS=2`,
`HANDOFF_FAST_COPY_STREAMS=2`, `HANDOFF_FAST_ASYNC=1`,
`HANDOFF_REMOTE_HIGH_PRIORITY=1`. Both decode roles retain
`HANDOFF_REMOTE_FALLBACK_MS=10000`. Existing env values may override defaults.

Start the gateway:

```bash
HANDOFF_CTL_DIR="$HANDOFF_CTL_ROOT" \
HANDOFF_PD_PREFILL_URL=http://127.0.0.1:31016 \
HANDOFF_PD_BOOTSTRAP_HOST=127.0.0.1 \
HANDOFF_PD_BOOTSTRAP_PORT=35409 \
HANDOFF_ADMISSION_POLICY=decoupled \
"$HANDOFF_PYTHON" plugins/remote_attn/control_plane/p4_gateway.py --port 31080
```

The control plane still stores its journal under
`/workspace/handoff_state`. Use separate state paths/containers when running
multiple experiments; changing those control-plane paths is outside this port.

## GPU acceptance still required

1. Check `/health` on all three servers and `ready.<rank>.json` for both ranks
   in each role directory. Check that rank and TP width are correct.
2. Run ordinary PD generation, then D0→D1 migrations using Copy, Layer and Remote.
   Compare deterministic token IDs/logprobs with an unmigrated reference.
3. Drain a full batch; check no duplicate output, no permanently full admission
   queue, and no leaked rows, reservations, registrations or source KV leases.
4. Exercise pre-commit rollback and both controller/decode restarts. Confirm a
   restarted D peer is probed before its first chunk.
5. Profile Remote to confirm KV transfer overlaps attention and layers become
   LOCAL progressively. TP graph-on support first needs a synchronized replay
   decision protocol; the current plugin rejects that configuration.

The source module's original local test/control surface and authentication
assumptions are retained. This port does not add a production router, automatic
scaling, gateway SSE reconnect or P5.19 functionality.

See [C1 and host-reclaim fixes](validation/C1-RECLAIM.md) for completion proof,
retry cleanup, compatibility and the remaining unprovable-completion boundary.
