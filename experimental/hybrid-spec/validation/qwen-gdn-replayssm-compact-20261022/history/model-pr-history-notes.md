# ReplaySSM community and source audit (2026-10-22)

- Read local `model-pr-history-knowledge/sglang/qwen35/README.en.md`; its earlier model history is discovery context, not proof that current compact replay works. No focused ReplaySSM entry was returned by the family query.
- [SGLang #28695](https://github.com/sgl-project/sglang/pull/28695): GDN circular verify, FP32 checkpoint, per-request cursors and explicit acceptance commit. Already present on this branch; no wholesale cherry-pick needed.
- [SGLang #35544](https://github.com/sgl-project/sglang/pull/35544): request-scoped compact history, compensated high/low update vectors and normalized keys, physical checkpoint ownership, lifecycle handling. These ownership and precision rules guide the integration.
- [Author reference](https://github.com/Johnny-Liou/ReplaySSM): checkpoint plus bounded input history, pointer rollback, separate output reconstruction and checkpoint materialization. Community results on other GPUs do not establish A100 performance here.

Current dispatch audited: `attention_hook.handle_linear_attn_backend`; `GDNAttnBackend.forward_extend/_replayssm_target_verify`; capped `query_start_loc` in `MambaAttnBackendBase`; `commit_mamba_states_after_verify`; `MambaPool`; the GDN circular kernel and materialization kernels.

Local branch already has varlen kernel inputs and compact conv handling. Changes retain FP32 checkpoint storage, allow the Triton GDN ragged combination, initialize graph slack, check the ring capacity, and test cached-history reconstruction. New CUDA tests compare live effective state and tracked state with recurrent snapshots through partial commits and row reordering.
