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
        "enable_mixed_chunk": False,
        "enable_multi_layer_eagle": False,
        "speculative_adaptive": False,
        "disaggregation_mode": "null",
        "decoupled_spec_role": "null",
    }
    for name, expected in required.items():
        if getattr(cfg, name, None) != expected:
            errors.append(f"{name}={expected!r} required")
    if config.draft_graph_mode != "full":
        for name, expected in {
            "speculative_use_rejection_sampling": False,
            "disable_cuda_graph": False,
            "enable_memory_saver": False,
        }.items():
            if getattr(cfg, name, None) != expected:
                errors.append(f"split draft graphs require {name}={expected!r}")
        limit = getattr(cfg, "max_running_requests", None)
        if type(limit) is not int or not 1 <= limit <= config.max_requests:
            errors.append(
                "split draft graphs require max_running_requests within max_requests"
            )
    if config.verify_mode == "compact":
        for name in (
            "enable_two_batch_overlap",
            "disable_cuda_graph_padding",
            "enable_memory_saver",
        ):
            if getattr(cfg, name, False):
                errors.append(f"compact verify requires {name}=False")
    if (
        config.history_mode == "cpu"
        and config.backend != "arctic"
        and not cfg.disable_overlap_schedule
    ):
        errors.append("CPU history requires disable_overlap_schedule=True")
    # CLI store_true has no False spelling: no flag/no adapters leaves None,
    # which SGLang treats as disabled. Never require literal enable_lora=False.
    if getattr(cfg, "enable_lora", None) not in (None, False) or getattr(
        cfg, "lora_paths", None
    ):
        errors.append("LoRA must be disabled")
    tp_size = getattr(cfg, "tp_size", None)
    if type(tp_size) is not int or tp_size < 1:
        errors.append("tp_size must be a positive integer")
    if (
        config.history_mode == "cpu"
        and config.backend != "arctic"
        and cfg.chunked_prefill_size not in (None, -1)
    ):
        errors.append("CPU history requires chunked_prefill_size=-1")
    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3", "STANDALONE"):
        errors.append("only ordinary EAGLE/MTP, EAGLE3 and STANDALONE are wired")
    if config.backend not in ("gpu_sa", "arctic"):
        errors.append("reference backend is test-only")
    if errors:
        raise ValueError("hybrid-spec unsupported configuration: " + "; ".join(errors))


def validate_model(worker, config=None):
    target = worker.target_worker.model_runner.model_config
    draft = worker.draft_runner.model_config
    target_arch = getattr(target.hf_config, "architectures", [])
    draft_arch = getattr(draft.hf_config, "architectures", [])
    if config is not None and config.verify_mode == "compact":
        from sglang.srt.configs.hybrid_arch import mambaish_config

        if target.model_is_mrope or mambaish_config(target) is not None:
            raise ValueError(
                "compact verify currently requires non-Mamba, non-mRoPE targets"
            )
        backend = worker.target_worker.model_runner.attn_backend
        if not getattr(backend, "supports_ragged_verify_graph", False):
            raise ValueError(
                "compact verify requires a ragged-capable target attention backend"
            )
    dsv4_mtp = target_arch == ["DeepseekV4ForCausalLM"] and draft_arch == [
        "DeepseekV4ForCausalLMNextN"
    ]
    if not dsv4_mtp and any(
        arch.startswith("DeepseekV4") for arch in target_arch + draft_arch
    ):
        raise ValueError(
            "hybrid-spec DeepSeek V4 requires a DeepseekV4ForCausalLM target "
            "and its DeepseekV4ForCausalLMNextN MTP draft"
        )
    qwen_mtp = (
        target_arch
        in (
            ["Qwen3_5ForConditionalGeneration"],
            ["Qwen3_5MoeForConditionalGeneration"],
            ["Qwen3_5ForCausalLM"],
            ["Qwen3_5MoeForCausalLM"],
        )
        and draft_arch == ["Qwen3_5ForCausalLMMTP"]
        and getattr(target.hf_config, "language_only", False)
        and getattr(draft.hf_config, "language_only", False)
    )

    for model_config in (target, draft):
        if getattr(model_config, "is_hybrid_swa", False) and not dsv4_mtp:
            raise ValueError(
                "hybrid-spec does not support hybrid sliding-window attention"
            )
        if dsv4_mtp:
            if model_config.model_is_mrope or (
                model_config.is_multimodal
                and not getattr(model_config.hf_config, "language_only", False)
            ):
                raise ValueError(
                    "hybrid-spec DeepSeek V4 MTP requires text-only serving"
                )
            # V4's native MTP owns SWA metadata and KV window planning. SA
            # replaces token IDs before native forwards/verify; it never crops
            # the SWA/compressed pools or rewrites compression commit state.
            continue
        if qwen_mtp:
            # Qwen3.6 uses the Qwen3.5 architecture. Its MTP layer is full
            # attention; the target's GDN verify writes intermediate states and
            # commit_mamba_states_after_verify commits only the accepted prefix.
            # Token replacement must stay before each draft forward and native
            # verify/commit. Keep native mRoPE positions for text-only requests;
            # DraftExtension.bind still rejects actual multimodal input.
            continue
        if model_config.is_multimodal or model_config.model_is_mrope:
            raise ValueError(
                "hybrid-spec requires full attention or Qwen3.5/3.6 MTP "
                "with --language-only"
            )
        from sglang.srt.configs.hybrid_arch import mambaish_config

        if mambaish_config(model_config) is not None:
            raise ValueError(
                "hybrid-spec supports linear attention only with Qwen3.5/3.6 MTP"
            )
