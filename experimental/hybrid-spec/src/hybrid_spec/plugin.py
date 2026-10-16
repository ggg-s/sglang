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
        "sglang.srt.managers.io_struct.GenerateReqInput.normalize_batch_and_arguments",
        _validate_inputs,
        HookType.BEFORE,
    )


def _config():
    return HybridConfig.from_file(os.environ["HYBRID_SPEC_CONFIG"])


def _validate(server_args):
    from sglang.srt.arg_groups.overrides import resolved_view

    from hybrid_spec.integrations.sglang.compatibility import validate_config

    validate_config(_config(), resolved_view(server_args))


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

    from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
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
    validate_model(worker)
    config = _config()
    tp_group = get_tp_group()
    communicator = None
    if config.history_mode == "device" and tp_group.world_size > 1:
        communicator = tp_group.pynccl_comm
        if communicator is None or not communicator.available:
            raise ValueError("device history TP requires SGLang's PyNccl communicator")
    tp_sync = (
        TPSynchronizer(tp_group.device_group, tp_group.ranks[0], communicator)
        if tp_group.world_size > 1
        else None
    )
    # Construction occurs before the scheduler budgets/allocates model KV.
    # cuda:<gpu_id> pins storage to this worker even with a different default device.
    backend = GPUSuffixAutomaton(config, f"cuda:{worker.gpu_id}")
    extension_type = DraftExtension
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


def _finished(result, worker, *, rid, natural_stop):
    draft = worker.draft_worker
    if draft is not None and draft.draft_extension is not None:
        draft.draft_extension.close_request(rid)


def _reset(result, worker, *args, **kwargs):
    draft = worker.draft_worker
    if draft is not None and draft.draft_extension is not None:
        draft.draft_extension.reset()


def _verify(result, worker, *args, **kwargs):
    extension = worker.draft_worker.draft_extension
    if extension is not None:
        if on_device_verify := getattr(extension, "on_device_verify", None):
            on_device_verify(result, args[0] if args else kwargs["batch"])
        else:
            extension.on_verify_result(result)


def _prefill(original, worker, batch, *args, **kwargs):
    extension = getattr(worker.draft_worker, "draft_extension", None)
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
