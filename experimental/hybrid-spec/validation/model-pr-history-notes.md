# Qwen3.6 MTP + SA adaptation evidence

Baseline: `1b0f58030790726b3146f46786a38b23a7f735e8`.

The local Qwen3.5 and Qwen3.6 PR-history dossiers were consulted before changing the model capability gate. Historical evidence guides inspection; it does not establish correctness on this checkout.

- [PR 19391](https://github.com/sgl-project/sglang/pull/19391): Qwen3.5 MTP support for SpecV2; draft-extend must use the V2 path. Current `qwen3_5_mtp.py` handles `is_draft_extend_v2()` and constructs a single full-attention MTP layer.
- [PR 20864](https://github.com/sgl-project/sglang/pull/20864): Qwen3.5 speculative mRoPE and device-side position handling. Current `eagle_worker_v2.py` advances both regular and mRoPE positions; graph staging copies all three mRoPE axes. SA selection must change tokens without changing positions.
- [PR 23486](https://github.com/sgl-project/sglang/pull/23486): Qwen3.6 retains a GDN backbone. The downloaded 35B-A3B FP8 checkpoint declares `Qwen3_5MoeForConditionalGeneration`, with linear/full-attention layers and an embedded MTP head.

Current source verification: `run_eagle_verify` calls `commit_mamba_states_after_verify` using native acceptance indices and lengths. GDN verification writes speculative intermediate states; the commit path selects only the accepted prefix (including the anchor input, not the unprocessed bonus token). The extension keeps this path intact. Qwen's MTP layer is full attention, so it does not introduce a separate recurrent draft-state rollback protocol.

Scope: only the Qwen3.5/3.6 target and `Qwen3_5ForCausalLMMTP` pair, both configured with `--language-only`. Other linear-attention architectures, sliding-window pools, and actual multimodal requests remain rejected. GPU token parity, rejection recovery, graph replay, and measured SA usage remain separate acceptance criteria from these source findings.
