# Qwen3.6 MTP + PDMux crash regression, 2026-10-29

Base: `codex/dsv4-pdmux-latest-main`, `9f16562144d9d586df408943058bc147c85a8cff`.
The branch was pulled with `--ff-only`; it was already current.

## Fix

`commit_mamba_states_after_verify()` used `model_runner.attn_backend`, which
retains prefill metadata under PDMux. TARGET_VERIFY runs on the decode lane.
When a live prefill had three requests and verify had five, the state scatter
failed with `indices length mismatch: dst_indices_raw.shape[0]=3 vs
step_indices_raw.shape[0]=5`. Equal batch sizes could instead write the wrong
state slots without a shape error.

The commit now resolves `decode_lane_attn_backend(model_runner)`, matching
verify planning and execution. Ordinary scheduling retains its original backend.
Regression fixtures cover unequal and equal P/D batch sizes in both prefill
modes, plus non-PDMux controls. Stale EAGLE admission expectations were updated
to match the pulled branch's existing support.

## RTX 6000D validation

- One NVIDIA RTX 6000D, 156 SMs, CUDA driver 580.119.02.
- Existing checkpoint: `/root/rivermind-data/models/Qwen/Qwen3.6-35B-A3B-FP8`,
  architecture `Qwen3_5MoeForConditionalGeneration`, one bundled MTP layer.
- Python 3.12, PyTorch `2.14.1+cu130`; existing sgl-kernel compatibility shim
  retained equally for controls and PDMux. Hashes are recorded in the JSON.
- TP1, EAGLE/MTP with three steps and four draft tokens, CPU overlap enabled,
  FlashInfer full attention, Triton linear attention, BF16 SSM states,
  extra-buffer prefix caching, 2048-token chunks, eight running requests.
- Unfixed full-model run reproduced the scatter exception during concurrent
  prefill/decode. The new unit regression also failed in all four PDMux cases.
- Fixed regression suite: **243 tests and 288 subtests passed**.

| Layout | Prefill | Decode execution | Completed requests |
| --- | --- | --- | --- |
| Fixed 72/80 SM | layer_split | CUDA graph | 32/32 |
| Fixed 72/80 SM | standard | CUDA graph | 32/32 |
| Full SM | layer_split | CUDA graph | 32/32 |
| Full SM | standard | CUDA graph | 32/32 |
| Full SM | layer_split | eager | 32/32 |
| Full SM | standard | eager | 32/32 |

Each run generated 6656 output tokens, with eight serial requests followed by
three eight-request concurrent rounds. Prompt lengths ranged from 64 to 6144
tokens; concurrent rounds reused cached prefixes. Logs confirm live P/D work.
Graph runs replayed draft graphs; genuine eager runs resolved the decode graph
backend to `disabled` and did not replay draft graphs. Two earlier attempts
using only `--disable-cuda-graph` were overridden by the explicit `full` backend
and are excluded from eager coverage.

This verifies the runtime crash regression, not bitwise output parity or model
quality. Some FP8 greedy sequences differ from the control. A logprob probe
found tied highest scores for newline tokens under PDMux, versus 0.125–0.25
separation in the control. This observation does not establish the cause of
every output difference. No throughput improvement is claimed.

Raw server logs, commands, token IDs and configuration snapshots remain at
`/root/pdmux-mtp-fix-20261029/`. The compact result is
`qwen36_mtp_pdmux_validation_20261029.json`. Additional answer checks were stopped
following the user's request to avoid nonessential validation. All task GPU
processes are stopped. No Qwen3.5 download was retained.

The reusable runner is `validate_qwen36_mtp_pdmux.py`. Eager validation must
explicitly select `--cuda-graph-backend-decode disabled`; the runner checks the
resolved backend before sending requests.
