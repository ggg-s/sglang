# DSV4 DSpark PCP/PDMux/HiCache configuration audit

After the DP1 fix (`ac486fb5e2`), no additional configuration overwrite was
found in the audited chain. This checks configuration and selection behavior;
it does not claim full-model execution or multi-GPU NCCL validation.

## Expected resolved state

For TP8/CP8 with `--enable-cp-decode-attn-tp`, interleave PCP, CPU overlap,
PDMux layer split, HiCache and prefill graphs explicitly disabled:

| Setting | Expected value |
|---|---|
| `speculative_algorithm` | `DSPARK` |
| `enable_prefill_cp`, `cp_strategy` | `true`, `interleave` |
| `enable_dp_attention` | `true`, preserved through the DP1 default pass |
| `tp_size`, `attn_cp_size` | `8`, `8` |
| `dp_size`, `pp_size`, `ep_size`, `moe_dp_size`, `dcp_size` | all `1` |
| `moe_dense_tp_size`, `moe_a2a_backend` | `1`, `none` |
| `enable_pdmux`, `pdmux_prefill_mode` | `true`, `layer_split` |
| `disable_overlap_schedule` | `false` |
| `enable_hierarchical_cache` | `true` |
| `hicache_ratio`, `hicache_write_policy` | explicit `2.0`, `write_through` retained |
| `enable_cp_decode_attn_tp` | `true` |
| prefill / decode graph backend | `disabled` / `full` |
| `enable_dp_lm_head` | `false` at DP1; this does not disable PCP or decode TP |

Explicit `--disable-cuda-graph` changes decode graphs to `disabled`; it does not
disable CPU overlap, PCP, HiCache or decode attention TP.

## Reader path and potentially confusing indicators

`--enable-cp-decode-attn-tp` is parsed into the `parallel` namespace. Model
resolution validates the architecture allowlist but does not clear the flag.
`ServerArgs.resolved_dict()` (the server-info configuration source), the
pickled worker record and `get_parallel().enable_cp_decode_attn_tp` agree.

The real `CpDecodeAttnTpContext` reads that published flag and `attn_cp_size`.
With TP8/CP8 and the flag enabled it has `is_enabled=True`, `decode_tp_size=8`
and the worker's CP rank. Decode, target verify and draft extend select decode
TP; active full PCP prefill and split prefill retain all heads. With the flag
omitted, the context is disabled in every rank.

Two values must not be used as evidence that the flag is ineffective:

- The base `get_parallel().attn_tp_size` remains **1** for PCP's replicated
  attention. The selected decode module has its own `attn_tp_size=CP`.
- The context's legacy `use_decode_attn_tp` flag remains **false** in the PDMux
  view path. Selection is pure and per-forward; changing a shared activation
  flag would reintroduce interference between concurrent prefill and decode.

The target and DSpark stage both call attention through the selected module.
The DSpark MoE draft inherits the published target parallel configuration;
it skips the dense-draft TP override, and its runner does not republish a raw
record over the target's resolved configuration.

## Coverage

`test/registered/unit/server_args/test_dsv4_pcp_resolution.py` now checks:

- 24 actual CLI combinations: CP2/4/8, flag off/on, graph/eager decode, and
  bundled/separate draft config. The YAML configuration file is also validated.
- Every declaration of critical fields, including graph backends, rather than
  only the last value. A temporary overwrite fails even if a later pass repairs
  it. Explicit chunk size, request limit and memory fraction are retained.
- 48 pickled configuration publications across tokenizer and scheduler roles,
  with assertions against the actual `parallel`, `disagg`, `schedule`, `memory`,
  `spec` and `exec.graph` namespaces. Resolution re-entry preserves the result.
- 28 real decode-context constructions across all CP ranks with the flag off
  and on, with 140 forward-mode selection checks. Only the process's CP rank
  is supplied by the fixture; the flag and CP size come from the published
  configuration. No process groups or model weights are created.
- The earlier real-resolution/default-reset regressions and plain-TP controls.

Run on the Ubuntu workstation with GPUs hidden:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover \
  -s test/registered/unit/server_args -p test_dsv4_pcp_resolution.py -v
```

The existing `test_pdmux_cp_decode_attn_tp.py` separately checks the production
view creation and target/draft dispatch using small PyTorch attention fixtures,
including projection parity, FP8 metadata, stable addresses and CUDA replay.
The new configuration cases bridge CLI parsing and publication to the real
decode context; they do not substitute small fixtures for full-model acceptance.

## Overwrite review

The DP1 default pass was the problematic writer: it must preserve the model's
PCP DP-attention declaration. Other reviewed writers are conditional on
different configurations: pipeline parallelism, non-CUDA runtimes, hybrid
Mamba/no-buffer mode, diffusion models or the sparse embedding-head environment
option can disable overlap; diffusion mode can disable HiCache. None applies
to the tested DSV4/CUDA/PP1/static-ragged recipe. Prefill graph disabling is
intentional and decode graph disabling remains opt-in for this recipe.

Rerun the full-model acceptance and workload on the deployment host before
claiming throughput or latency. The former `enable_dp_attention=false` workload
results are not equivalent to the resolved configuration audited here.

## Runtime gather / CUDA graph follow-up, 2026-10-28

Preserving the PCP declaration exposed a separate runtime assumption at
`c9880b53bb`: `require_mlp_tp_gather()` asserted that DP attention implies
DP > 1. `GraphSharedOutput.create_for_model_runner()` reaches this predicate
through logits-row sizing and decode graph bucket alignment, before serving
requests. The configuration audit above did not cover that startup entry.

The fix keeps DP attention enabled and changes two predicates:

- MLP TP gather requires DP attention **and DP > 1**. At DP1 there are no
  independent attention-DP batches to gather. DSV4's prefill CP token gathers
  remain model-owned and are not disabled by this predicate.
- Attention TP gather tests the actual `attn_tp_size > 1`, rather than
  `dp_size < tp_size`. The latter also counts CP ranks and incorrectly asks
  for an upstream attention gathered buffer at TP8/CP8/DP1. A blanket DP1
  exemption would break ordinary TP sequence-scattering / A2A configurations.

For the supported recipe, the resulting state is:

| Runtime value | TP8/CP8/DP1 |
|---|---|
| base attention TP / CP width | 1 / 8 |
| MLP TP gather / attention TP gather | false / false |
| upstream gathered-buffer requirement | false |
| MLP-sync metadata preparation | true |
| decode graph token-row alignment | 8, retained for CP |

`--enable-cp-decode-attn-tp` still selects the separate decode module views;
it does not rewrite the base attention TP width used by these predicates.

The follow-up review covered attention-width derivation, attention-DP rank
arithmetic and model-parallel group construction, the DP1 scheduler-local
metadata path, DSpark's DP-only argument checks and CP-aware synchronization,
and graph bucket / buffer metadata consumers. DP1 still generates one local
token-count entry and skips cross-DP scheduler gathering. The separate DP>=2
requirements for unsupported MoE backends are not relaxed.

Regression coverage now includes the real resolved/pickled/published PCP
configuration reaching `GraphSharedOutput -> ModelRunner.max_decode_logits_rows
-> get_batch_sizes_to_capture -> get_cuda_graph_batch_size_alignment`, for
CP2/4/8, decode attention TP off/on, and target/draft token widths. It checks
request-pool clamping and CP-aligned buckets. On CUDA it also allocates the
shared logits output and captures/replays a small tensor operation. Additional
controls cover ordinary TP, multi-DP, mixed CP/attention-TP widths, the gather
opt-out, and DP1 scheduler metadata with CP2/CP8.

The actual graph-output startup entry was also run with the two old gather
functions loaded in memory from `c9880b53bb`, reproducing the reported assertion
without modifying the workstation checkout. The old attention predicate also
returns true for the same published configuration; the fixed predicate returns
false while retaining CP alignment 8.

These checks do not execute a TP8 DSV4 model, FlashMLA, MoE or HiCache transfers.
A deployment-host startup and request test is still required for full-model
acceptance; the 4060 only validates the planning path and small CUDA graphs.

Validation on the RTX 4060 Laptop workstation (PyTorch 2.13.0+cu130), tested
source `ec76ab0881`:

- PCP resolution/runtime tests, scheduler DP metadata, CUDA graph buffer
  registry and the complete `unit/multiplex` directory: **265 tests and 355
  subtests passed**, including the new CUDA output allocation/replay case.
- `test_pdmux_tp_overlap_cuda.py`: **2 passed, 1 skipped**; the skipped case
  needs two distinct GPUs.
- PCP resolution/runtime tests with `CUDA_VISIBLE_DEVICES=''`: **9 passed,
  1 skipped, 102 subtests passed**; only the CUDA replay case skipped.
- Formatting, focused static checks and `git diff --check` passed locally.

Workstation evidence is under `/workspace/cp-decode-review-20261027/`:
`pcp-graph-baseline-20261028.log`, `pcp-graph-regression-20261028.log`,
`pcp-graph-lanes-20261028.log` and `pcp-graph-cpu-20261028.log`.
