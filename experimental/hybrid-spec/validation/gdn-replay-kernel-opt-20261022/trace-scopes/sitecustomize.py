"""Install diagnostic scopes only at the first profiler request, after timing."""

import functools
import os

if os.environ.get("AGENT_TRACE_SCOPES") == "1":
    from sglang.srt.managers.scheduler_components.profiler_manager import (
        SchedulerProfilerManager,
    )

    _original_init = SchedulerProfilerManager._init_profile
    _installed = False

    def _wrap(owner, attribute, label):
        import torch

        original = getattr(owner, attribute)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with torch.profiler.record_function("diag/" + label):
                return original(*args, **kwargs)

        setattr(owner, attribute, measured)

    def _install():
        global _installed
        if _installed:
            return
        from sglang.srt.speculative import eagle_compact_verify
        from sglang.srt.speculative.eagle_worker_v2 import (
            EagleDraftWorker,
            EAGLEWorkerV2,
        )

        for owner, attribute, label in [
            (EagleDraftWorker, "draft", "draft"),
            (EagleDraftWorker, "_draft_extend_for_decode", "draft_extend"),
            (EAGLEWorkerV2, "verify", "verify"),
            (EAGLEWorkerV2, "_forward_prefill_batch", "prefill"),
            (eagle_compact_verify, "prepare_compact_verify", "compact_pack"),
            (eagle_compact_verify, "scatter_verify_output", "compact_scatter"),
        ]:
            _wrap(owner, attribute, label)
        if os.environ.get("HYBRID_SPEC_CONFIG"):
            from hybrid_spec.integrations.sglang import compact_graph, compact_verify

            _wrap(compact_graph, "replay_compact", "split_replay")
            _wrap(compact_verify, "configure_verify", "verify_plan")
        _installed = True

    def _init(self, *args, **kwargs):
        _install()
        return _original_init(self, *args, **kwargs)

    SchedulerProfilerManager._init_profile = _init
