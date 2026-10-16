# Qwen3.6 MTP + SA adaptation evidence

Baseline: `1b0f58030790726b3146f46786a38b23a7f735e8`.

The local Qwen3.5 and Qwen3.6 PR-history dossiers were consulted before changing the model capability gate. Historical evidence guides inspection; it does not establish correctness on this checkout.

- [PR 19391](https://github.com/sgl-project/sglang/pull/19391): Qwen3.5 MTP support for SpecV2; draft-extend must use the V2 path. Current `qwen3_5_mtp.py` handles `is_draft_extend_v2()` and constructs a single full-attention MTP layer.
- [PR 20864](https://github.com/sgl-project/sglang/pull/20864): Qwen3.5 speculative mRoPE and device-side position handling. Current `eagle_worker_v2.py` advances both regular and mRoPE positions; graph staging copies all three mRoPE axes. SA selection must change tokens without changing positions.
- [PR 23486](https://github.com/sgl-project/sglang/pull/23486): Qwen3.6 retains a GDN backbone. The downloaded 35B-A3B FP8 checkpoint declares `Qwen3_5MoeForConditionalGeneration`, with linear/full-attention layers and an embedded MTP head.

Current source verification: `run_eagle_verify` calls `commit_mamba_states_after_verify` using native acceptance indices and lengths. GDN verification writes speculative intermediate states; the commit path selects only the accepted prefix (including the anchor input, not the unprocessed bonus token). The extension keeps this path intact. Qwen's MTP layer is full attention, so it does not introduce a separate recurrent draft-state rollback protocol.

Scope: only the Qwen3.5/3.6 target and `Qwen3_5ForCausalLMMTP` pair, both configured with `--language-only`. Other linear-attention architectures, sliding-window pools, and actual multimodal requests remain rejected. GPU token parity, rejection recovery, graph replay, and measured SA usage remain separate acceptance criteria from these source findings.

## Plugin performance review (2026-10-16)

Pre-change HEAD: `4030650794`; the supplied Qwen3-14B/0.6B report used
`1b0f580307`. Read the local `model-pr-history-knowledge/sglang/qwen3-core/README.en.md`
implementation coverage and PR #7745 card (Qwen EAGLE3 auxiliary hidden-state
integration), plus the existing Qwen3.5/3.6 notes above. Read the maintained
`AI-Infra-Auto-Driven-Skills/docs/upstream-source-contracts.md` through the skill's
resolved junction target: historical dispatch and trace attribution do not
certify the current runtime or a performance cause.

Decision: optimize the shared plugin consensus/graph boundary, retaining native
model hidden-state capture, token mapping, verification and acceptance paths.
No model-kernel replacement follows from the report's matching GEMM/attention
tables. Current `draft_forward` invokes `prepare` before the first selection;
the scheduler budgets KV after draft construction using currently free GPU
memory. These source observations, rather than historical PR titles, determine
the changes. GPU/NCCL timing and model parity remain deployment acceptance gates.

Direct sa_spec source comparison was rechecked on 2026-10-16 at clean local
HEAD `1cc7e8c86b1914a9c05eb60fb11ebe52daf47969`: `src/api.cc` builds prompt
states on the host and copies used prefixes when requests enter generation;
`src/api.cu` consumes accepted device tokens and performs lookup in one
current-stream kernel; `src/memory.cu` uses asynchronous copies on the active
Torch stream. This supports the existing device-history path, selected explicitly
by `device-k3.json` or `device-sa-spec.json`. The standalone library has no TP
consensus protocol, which is not evidence that SGLang may safely remove consensus
while allowing rank-local slot/version fallback. No new GPU performance claim
is inferred from the reference implementation.
