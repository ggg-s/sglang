# Hybrid PD remote-window diagnostics

Hybrid PD now shares one parent refresh among its K/V resources and parent in
one `Retirement.observe()` pass. The snapshot (including any refresh failure)
is discarded at the end of that pass. Final TP votes and `finish()` still query
fresh GPU/transport readiness; receipt timing never authorizes a local read.

Each successfully retired hybrid PD request's existing `done.json` gains
`pd_window`. No additional per-layer log writes, CUDA synchronization, network
messages or cross-machine timestamp subtraction are introduced. Values are
rank-local; compare all TP ranks, not only rank 0. Existing remote calls are
counted per request/layer, even when multiple rows share one batched RPC.

## Fields

All times are milliseconds on D's monotonic clock relative to early admission.
For each full-attention layer, timestamps are the maximum across its K and V:

- `receipt_ms`: D first observed validated completed ranges covering the entire
  prefix. This includes P publication and D polling delay. It is **not** the
  instant the DMA finished, nor does it prove GPU visibility.
- `visible_ms`: D observed complete coverage after receive visibility/event
  checks. A complete receipt alone cannot set this field.
- `local_ms`: both K/V resources passed the ordered retirement protocol and
  were marked local on this rank. TP control remains the authority.
- `receipt_to_visible_ms`, `visible_to_local_ms`: differences of those times;
  unknown endpoints remain null, not zero.
- `remote_calls`, `remote_after_receipt`, `remote_after_visible`: remote routing
  decisions for this layer, including decisions after the respective complete
  layer observation. Successful completed requests should match their existing
  `remote_calls` total; these counters are not a measurement of RPC duration.

`remote_steps` counts observed decode steps with at least one remote layer;
`remote_layer_calls` counts all remote layer decisions. `observed_steps` covers
only steps while the PD target is tracked, not the entire generated response.
`recent_steps` retains at most 256 steps with `at_ms`, `layers_seen`, and
`remote_layers`. `dropped_steps` reports older omitted steps; aggregate counters
are never truncated. The final observed step may be partial if tracking ends.
No whole-request completion record is promised for aborted/quarantined requests.

## Comparison

Run the same model, dtype, KV prefix cache state, output length, and transport
settings at 32K/128K and C1/C4/C8. Retain all per-rank `done.json` records, token
hashes, E2E/TTFT and decode latency. Separate cold-cache runs.

1. Compare total `remote_layer_calls`, not `remote_steps * 16`. Layers already
   switch locally independently; a shorter step can increase window step count.
2. Large `remote_after_visible` or `visible_to_local_ms` points to notification /
   retirement delay rather than KV bandwidth.
3. Large `receipt_to_visible_ms` points to receive visibility/event scheduling.
4. Late receipts with small downstream delays motivate P queue instrumentation
   and a fair active-request KV priority experiment. Receipt latency alone cannot
   distinguish DMA, sender publication, or D polling.
5. Validate C1 separately: inter-request priority cannot remove contention from
   other requests when none exist. Do not infer an E2E saving from these counters.

The hybrid adapter still requires native NHD BF16/FP16 KV; FP8 is not a supported
one-flag ablation. This change does not alter bulk scheduling, transfer bytes,
attention math, or the request admission boundary.
