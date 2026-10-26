"""Opt-in measured-pass service tracing, with no model/state re-execution."""

import json
import os
from contextlib import nullcontext


class ServiceProfile:
    def __init__(self, root):
        self.root = root
        self.enabled = os.environ.get("LIVE_SERVICE_PROFILE") == "1"
        self.profiler = None
        self.active = False
        self.finished = False

    def maybe_start(self):
        if not self.enabled or self.profiler is not None:
            return
        if not (self.root / "profile-start").exists():
            return
        import torch

        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False, with_stack=False,
        )
        self.profiler.start()
        self.active = True

    def scope(self, name):
        if not self.active:
            return nullcontext()
        import torch

        return torch.profiler.record_function(name)

    def maybe_finish(self):
        if not self.active or not (self.root / "profile-stop").exists():
            return
        import torch

        torch.cuda.synchronize()
        self.profiler.stop()
        self.active = False
        self.profiler.export_chrome_trace(str(self.root / "service-trace.json"))
        averages = self.profiler.key_averages()
        stages = []
        for event in averages:
            if event.key.startswith(("stage/", "round/")):
                stages.append(dict(
                    name=event.key, count=event.count,
                    cpu_ms=event.cpu_time_total / 1000,
                    device_ms=event.device_time_total / 1000,
                ))
        (self.root / "service-profile.json").write_text(json.dumps(dict(
            stages=stages,
            table=averages.table(sort_by="self_cuda_time_total", row_limit=50),
            scope="one measured SWE replay; profiler overhead included, not throughput evidence",
        ), indent=2))
        self.finished = True
        (self.root / "profile-complete").touch()
