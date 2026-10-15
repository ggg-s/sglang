from __future__ import annotations


def validate_config(config, cfg):
    """cfg is SGLang's resolved_view, never the pristine ServerArgs seed."""
    errors = []
    required = {
        "device": "cuda",
        "dp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
        "attn_dp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "moe_dp_size": 1,
        "nnodes": 1,
        "enable_torch_compile": False,
        "speculative_eagle_topk": 1,
        "speculative_num_steps": config.num_drafts,
        "speculative_num_draft_tokens": config.num_drafts + 1,
        "disable_overlap_schedule": True,
        "enable_mixed_chunk": False,
        "enable_multi_layer_eagle": False,
        "speculative_adaptive": False,
        "enable_lora": False,
        "disaggregation_mode": "null",
        "decoupled_spec_role": "null",
    }
    for name, expected in required.items():
        if getattr(cfg, name, None) != expected:
            errors.append(f"{name}={expected!r} required")
    tp_size = getattr(cfg, "tp_size", None)
    if type(tp_size) is not int or tp_size < 1:
        errors.append("tp_size must be a positive integer")
    if cfg.chunked_prefill_size not in (None, -1):
        errors.append("chunked_prefill_size=-1 required")
    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3", "STANDALONE"):
        errors.append("only ordinary EAGLE/MTP, EAGLE3 and STANDALONE are wired")
    if config.backend != "gpu_sa":
        errors.append("reference backend is test-only")
    if errors:
        raise ValueError("hybrid-spec unsupported configuration: " + "; ".join(errors))


def validate_model(worker):
    from sglang.srt.configs.hybrid_arch import mambaish_config

    for runner in (worker.target_worker.model_runner, worker.draft_runner):
        model_config = runner.model_config
        if (
            model_config.is_multimodal
            or model_config.model_is_mrope
            or getattr(model_config, "is_hybrid_swa", False)
            or mambaish_config(model_config) is not None
        ):
            raise ValueError("hybrid-spec requires a text-only full-attention model")
