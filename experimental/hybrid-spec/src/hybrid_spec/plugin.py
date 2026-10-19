"""Entry point registration is CPU-only; resources belong to each draft worker."""

from __future__ import annotations

import os

from hybrid_spec.config import HybridConfig


def register():
    # An installed package remains inert until explicitly configured. Parse in
    # the invoked hooks so plugin-loader exception logging cannot silently
    # disable a malformed configuration.
    if not os.environ.get("HYBRID_SPEC_CONFIG"):
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.check_server_args",
        _validate,
        HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.create_worker",
        _validate_factory,
        HookType.BEFORE,
    )
    for path in (
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.__init__",
        "sglang.srt.speculative.standalone_worker_v2.StandaloneDraftWorker.__init__",
    ):
        HookRegistry.register(path, _attach, HookType.AFTER)
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft",
        _begin_draft,
        HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft",
        _configure_verify,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner."
        "DecodeCudaGraphRunner._build_ragged_verify_token_buckets",
        _verify_token_buckets,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.speculative.base_spec_worker.BaseSpecWorker.note_request_finished",
        _finished,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.speculative.base_spec_worker.BaseSpecWorker.clear_cache_pool",
        _reset,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2.verify",
        _verify,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2._forward_prefill_batch",
        _prefill,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        _validate_sampling,
        HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler_components.internal_state."
        "SchedulerInternalStateController.get_internal_state",
        _diagnostics,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler_components.internal_state."
        "SchedulerInternalStateController._update_handlers",
        _profile_update_handlers,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.managers.io_struct.GenerateReqInput.normalize_batch_and_arguments",
        _validate_inputs,
        HookType.BEFORE,
    )


def _config():
    return HybridConfig.from_file(os.environ["HYBRID_SPEC_CONFIG"])


def _validate(server_args):
    from sglang.srt.arg_groups.overrides import resolved_view

    from hybrid_spec.integrations.sglang.compatibility import validate_config

    config = _config()
    if config.dynamic_verify_profile:
        from sglang.srt.environ import envs

        if envs.SGLANG_SIMULATE_ACC_LEN.get() != 1:
            raise ValueError("hybrid cost profiling requires SGLANG_SIMULATE_ACC_LEN=1")
    validate_config(config, resolved_view(server_args))
    if config.verify_mode == "compact":
        os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "compact"
    elif os.environ.get("SGLANG_RAGGED_VERIFY_MODE", "static") == "compact":
        raise ValueError(
            "HYBRID_SPEC_CONFIG must select verify_mode=compact with ragged graphs"
        )


def _validate_factory(algorithm, server_args):
    _validate(server_args)


def _validate_sampling(params, vocab_size):
    # Reject unsupported API requests before they reach the GPU scheduler.
    if params.top_k != 1:
        raise ValueError("hybrid-spec supports greedy requests only (temperature=0)")
    if any((params.json_schema, params.regex, params.ebnf, params.structural_tag)):
        raise ValueError("hybrid-spec does not support grammar constraints")
    if params.beam_width is not None or params.watermark is not None:
        raise ValueError("hybrid-spec does not support beam search or watermarking")


def _validate_inputs(request):
    # Qwen's text-only MTP still advertises multimodal capabilities. Reject
    # unsupported requests in the tokenizer process, before GPU scheduling.
    if request.contains_mm_input() or request.input_embeds is not None:
        raise ValueError(
            "hybrid-spec requires text or token IDs without multimodal inputs"
        )


def _attach(result, worker, *args, **kwargs):
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )

    from hybrid_spec.integrations.sglang.compatibility import validate_model
    from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
    from hybrid_spec.integrations.sglang.tp_sync import TPSynchronizer

    _validate(worker.server_args)
    if (
        getattr(worker, "DRAFT_EXTENSION_API_VERSION", None) != 2
        or getattr(EAGLEDraftCudaGraphRunner, "DRAFT_EXTENSION_API_VERSION", None) != 2
    ):
        raise RuntimeError(
            "hybrid-spec requires the SGLang draft extension API v2 patch"
        )
    config = _config()
    if config.verify_mode == "compact":
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )

        if getattr(DecodeCudaGraphRunner, "COMPACT_VERIFY_API_VERSION", None) != 1:
            raise RuntimeError(
                "compact verify requires the SGLang compact verify API v1 patch"
            )
    validate_model(worker, config)
    if config.draft_graph_mode != "full" and (
        getattr(worker, "DRAFT_SPLIT_GRAPH_API_VERSION", None) != 5
        or getattr(EAGLEDraftCudaGraphRunner, "DRAFT_SPLIT_GRAPH_API_VERSION", None)
        != 5
    ):
        raise RuntimeError(
            "split draft graphs require the worker and Graph split API v5"
        )
    tp_group = get_tp_group()
    communicator = None
    if tp_group.world_size > 1:
        communicator = tp_group.pynccl_comm
        if communicator is None or not communicator.available:
            raise ValueError("hybrid-spec TP requires SGLang's PyNccl communicator")
    tp_sync = (
        TPSynchronizer(tp_group.device_group, communicator)
        if tp_group.world_size > 1
        else None
    )
    # Construction occurs before the scheduler budgets/allocates model KV.
    # cuda:<gpu_id> pins storage to this worker even with a different default device.
    extension_type = DraftExtension
    if config.backend == "arctic":
        from hybrid_spec.backends.arctic import ArcticBackend
        from hybrid_spec.integrations.sglang.arctic_extension import (
            ArcticDraftExtension,
        )

        backend = ArcticBackend(config, f"cuda:{worker.gpu_id}")
        extension_type = ArcticDraftExtension
    else:
        from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton

        backend = GPUSuffixAutomaton(config, f"cuda:{worker.gpu_id}")
    if config.history_mode == "device":
        from hybrid_spec.integrations.sglang.device_extension import (
            DeviceDraftExtension,
        )

        extension_type = DeviceDraftExtension
    worker.draft_extension = extension_type(
        config,
        backend,
        worker.target_worker.model_runner.model_config.vocab_size,
        tp_sync=tp_sync,
    )
    if config.dynamic_verify:
        import torch

        target = worker.target_worker.model_runner
        worker.draft_extension.cost_runtime = {
            "target_model": target.model_config.model_path,
            "draft_model": worker.draft_runner.model_config.model_path,
            "gpu": torch.cuda.get_device_name(worker.gpu_id),
            "tp_size": tp_group.world_size,
            "dtype": str(target.dtype),
            "kv_cache_dtype": str(target.kv_cache_dtype),
        }
    if config.dynamic_verify and config.backend == "arctic":
        from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton

        # Compile before capture; create the calibration tensors lazily on the
        # owning forward stream, exactly like the device-history feedback.
        GPUSuffixAutomaton.load_module()


def _finished(result, worker, *, rid, natural_stop):
    draft = worker.draft_worker
    if draft is not None and draft.draft_extension is not None:
        draft.draft_extension.close_request(rid, retain=True)


def _reset(result, worker, *args, **kwargs):
    draft = worker.draft_worker
    if draft is not None and draft.draft_extension is not None:
        draft.draft_extension.reset()
        draft.draft_extension.dynamic_verify = None


def _verify(result, worker, *args, **kwargs):
    extension = worker.draft_worker.draft_extension
    if extension is not None:
        from hybrid_spec.integrations.sglang.compact_verify import (
            finalize_verify_counters,
        )

        if getattr(extension, "pending_verify_tokens", None) is not None:
            finalize_verify_counters(
                extension,
                result,
                worker.target_worker.model_runner.decode_cuda_graph_runner,
            )
        dynamic = getattr(extension, "dynamic_verify", None)
        if dynamic is not None:
            batch = args[0] if args else kwargs["batch"]
            if not batch.forward_mode.is_idle():
                if result.can_run_cuda_graph:
                    graph = worker.target_worker.model_runner.decode_cuda_graph_runner
                    dynamic.graph_tokens = graph._ragged_graph_size
                else:
                    dynamic.discard_verify_timing()
                dynamic.feedback(result.accept_lens)
        if on_device_verify := getattr(extension, "on_device_verify", None):
            on_device_verify(result, args[0] if args else kwargs["batch"])
        else:
            extension.on_verify_result(result)


def _prefill(original, worker, batch, *args, **kwargs):
    extension = getattr(worker.draft_worker, "draft_extension", None)
    if getattr(extension, "dynamic_verify", None) is not None:
        extension.dynamic_verify.discard_timing()
    if extension is None or not hasattr(extension, "begin_prefill"):
        return original(worker, batch, *args, **kwargs)
    ticket = extension.begin_prefill(batch)
    try:
        result = original(worker, batch, *args, **kwargs)
        extension.on_prefill(batch, result, ticket)
        return result
    except BaseException:
        extension.cancel_prefill(ticket)
        raise


def _diagnostics(result, controller, *args, **kwargs):
    worker = controller.scheduler.draft_worker
    draft = getattr(worker, "draft_worker", None)
    extension = getattr(draft, "draft_extension", None)
    if extension is not None:
        # Explicit diagnostic readback only; never synchronize each decode step.
        result.internal_state["hybrid_spec"] = extension.metrics.snapshot()
        if hasattr(extension, "compact_verify_counters"):
            result.internal_state["hybrid_spec_compact_verify"] = dict(
                extension.compact_verify_counters
            )
        if hasattr(extension, "split_graph_counters"):
            result.internal_state["hybrid_spec_split_graph"] = dict(
                extension.split_graph_counters
            )
        if getattr(extension, "dynamic_verify", None) is not None:
            result.internal_state["hybrid_spec_dynamic_verify"] = (
                extension.dynamic_verify.snapshot()
            )


def _profile_update_handlers(result, controller):
    """Expose calibration commands only on an explicitly dedicated profile server."""
    worker = getattr(controller.scheduler, "draft_worker", None)
    draft = getattr(worker, "draft_worker", None)
    extension = getattr(draft, "draft_extension", None)
    if extension is None or not extension.config.dynamic_verify_profile:
        return result
    from sglang.srt.managers.scheduler_components.internal_state import (
        UpdateRejected,
        _UpdateHandler,
    )

    def validate(key, value):
        graph = worker.target_worker.model_runner.decode_cuda_graph_runner
        if (
            type(value) is not int
            or value < 0
            or graph is None
            or not graph.capture_num_tokens
        ):
            raise UpdateRejected(
                "hybrid profiling requires a nonnegative integer and CUDA graphs"
            )
        if (
            key.endswith("graph_tokens")
            and value
            and value not in graph.capture_num_tokens
        ):
            raise UpdateRejected("hybrid profile graph tier was not captured")
        if key.endswith("batch_size") and value > extension.config.max_requests:
            raise UpdateRejected("hybrid profile batch exceeds max_requests")

    def apply(key, value):
        from hybrid_spec.integrations.sglang.compact_verify import get_dynamic_verify

        device = getattr(extension, "device", None)
        if device is None:
            device = extension.backend.device
        dynamic = get_dynamic_verify(extension, device)
        dynamic.set_profile_cell(key.removeprefix("hybrid_spec_"), value)

    for name in ("profile_batch_size", "profile_graph_tokens"):
        result["hybrid_spec_" + name] = _UpdateHandler(validate, apply)
    return result


def _configure_verify(result, worker, batch, **kwargs):
    if worker.draft_extension is not None:
        from hybrid_spec.integrations.sglang.compact_verify import configure_verify

        configure_verify(
            result[0] if isinstance(result, tuple) else result, worker, batch
        )


def _verify_token_buckets(result, runner):
    config = _config()
    if config.verify_mode == "compact" and not runner.model_runner.is_draft_worker:
        # Worker construction precedes attention initialization. Validate the
        # concrete backend here, when the target builds its capture grid.
        if not getattr(runner.attn_backend, "supports_ragged_verify_graph", False):
            raise ValueError(
                "compact verify requires a ragged-capable target attention backend"
            )
        from sglang.srt.utils import get_cuda_graph_batch_size_alignment

        from hybrid_spec.integrations.sglang.compact_verify import token_buckets

        alignment = get_cuda_graph_batch_size_alignment(
            gathered_buffer_required=runner.require_gathered_buffer,
        )
        return token_buckets(config, result, runner.max_bs, alignment)


def _begin_draft(worker, batch, **kwargs):
    if worker.draft_extension is not None:
        worker.draft_extension.compact_num_neural = None
        extension = worker.draft_extension
        if extension.config.dynamic_verify and not batch.forward_mode.is_idle():
            from hybrid_spec.integrations.sglang.compact_verify import (
                get_dynamic_verify,
            )

            device = getattr(extension, "device", None)
            if device is None:
                device = extension.backend.device
            if (
                hasattr(extension, "cost_runtime")
                and "attention_backend" not in extension.cost_runtime
            ):
                target = worker.target_worker.model_runner
                extension.cost_runtime["attention_backend"] = (
                    type(target.attn_backend).__module__
                    + "."
                    + type(target.attn_backend).__name__
                )
                graph = target.decode_cuda_graph_runner
                extension.cost_runtime["capture_max_bs"] = graph.max_bs if graph else 0
                extension.cost_runtime["capture_token_grid"] = (
                    list(graph.capture_num_tokens or []) if graph else []
                )
            get_dynamic_verify(extension, device).begin(batch)
        elif getattr(extension, "dynamic_verify", None) is not None:
            extension.dynamic_verify.discard_timing()
