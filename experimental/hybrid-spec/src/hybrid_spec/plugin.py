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
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        _validate_sampling,
        HookType.BEFORE,
    )


def _config():
    return HybridConfig.from_file(os.environ["HYBRID_SPEC_CONFIG"])


def _validate(server_args):
    from hybrid_spec.integrations.sglang.compatibility import validate_config

    from sglang.srt.arg_groups.overrides import resolved_view

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


def _attach(result, worker, *args, **kwargs):
    from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
    from hybrid_spec.integrations.sglang.compatibility import validate_model
    from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
    from hybrid_spec.integrations.sglang.tp_sync import TPSynchronizer

    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )

    _validate(worker.server_args)
    if (
        getattr(worker, "DRAFT_EXTENSION_API_VERSION", None) != 1
        or getattr(EAGLEDraftCudaGraphRunner, "DRAFT_EXTENSION_API_VERSION", None) != 1
    ):
        raise RuntimeError(
            "hybrid-spec requires the SGLang draft extension API v1 patch"
        )
    validate_model(worker)
    config = _config()
    # Construction occurs before the scheduler budgets/allocates model KV.
    # cuda:<gpu_id> pins storage to this worker even with a different default device.
    backend = GPUSuffixAutomaton(config, f"cuda:{worker.gpu_id}")
    tp_group = get_tp_group()
    tp_sync = (
        TPSynchronizer(tp_group.device_group, tp_group.ranks[0])
        if tp_group.world_size > 1
        else None
    )
    worker.draft_extension = DraftExtension(
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
    if worker.draft_worker.draft_extension is not None:
        worker.draft_worker.draft_extension.on_verify_result(result)
