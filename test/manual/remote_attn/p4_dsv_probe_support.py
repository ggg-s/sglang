"""P4 coordinator against real D2D sockets/bytes and explicit native queue ABI fixtures."""

import copy
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang_remote_attn import dsv_d2d as d2d
from sglang_remote_attn import dsv_gateway_control as control


class Driver:
    def __init__(self, env):
        self.env = env
        self.native_prefill = control.prefill
        self.calls, self.prefills, self.cancels = [], [], []
        self.after, self.before = {}, {}
        self.controllers = dict(src=env.source, dst=env.target)
        native_kv = copy.copy(env.req.kv)
        output = list(env.req.output_ids)
        for role, controller in self.controllers.items():
            scheduler = controller.scheduler

            def idle(scheduler=scheduler):
                from sglang_remote_attn.pre_layerwise import is_fully_idle

                def native_idle(*args, **kwargs):
                    return (
                        not scheduler.waiting_queue
                        and not any(
                            r.kv.holds_kv
                            for r in scheduler.handoff_tracked_reqs.values()
                        )
                        and not any(
                            not r.finished() for r in scheduler.running_batch.reqs
                        )
                    )

                return is_fully_idle(native_idle, scheduler)

            scheduler.is_fully_idle = idle

            def queue(req, role=role, scheduler=scheduler):
                req.kv = copy.copy(native_kv)
                req.kv.req_pool_idx = 0 if role == "src" else 1
                req.output_ids[:] = output
                req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
                req.cache_request_handle = (req.rid, 0)
                scheduler.running_batch.reqs = [req]
                if role == "src":
                    scheduler.handoff_tracked_reqs.pop(env.req.rid, None)
                    scheduler.handoff_tracked_reqs[req.rid] = req
                    env.req = req

            scheduler._add_request_to_queue = queue

            def abort(recv, scheduler=scheduler):
                self.cancels.append((scheduler, recv.rid))
                req = scheduler.handoff_tracked_reqs[recv.rid]
                req.finished = lambda: True
                from sglang.srt.mem_cache.common import release_kv_cache

                if req.kv.holds_kv:
                    release_kv_cache(req, scheduler.tree_cache, checkpoint=False)

            scheduler.abort_request = abort

        def prepare_abort(req, *args, **kwargs):
            req.finished = lambda: True
            req.finished_reason = "aborted"

        for controller in self.controllers.values():
            controller.scheduler.output_streamer = NS(stream_output=lambda *args: None)
        env.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.utils": NS(prepare_abort=prepare_abort),
                },
            )
        )
        env.stack.enter_context(
            patch.object(
                control,
                "prefill",
                side_effect=lambda req, payload: self.prefills.append(payload),
            )
        )
        env.stack.enter_context(
            patch.dict(
                sys.modules,
                {"sglang.srt.managers.io_struct": NS(AbortReq=lambda **kw: NS(**kw))},
            )
        )

    def send(self, role, payload, timeout=30):
        key = payload["cmd"]
        self.calls.append((role, dict(payload)))
        if self.before.get(key):
            raise self.before[key].pop(0)
        result = d2d.handle(self.controllers[role], payload)
        if key == "d2d_worker_status" and not getattr(self, "admission_plan", False):
            result["admission_plan_supported"] = False
        if self.after.get(key):
            raise self.after[key].pop(0)
        return dict(result, tp_rank_results=[dict(ok=True, **result)])

    def finish_local(self):
        self.env.gate.set()
        self.env.epoch = next(
            e for e, t in self.env.target.d2d_targets.items() if t.admitted
        )
        target = self.env.wait_copy()
        self.env.target.drain_targets(required=target)
        source = next(s for s in self.env.source.d2d_sources.values() if s.committed)
        deadline = time.monotonic() + 5
        while not source.poll():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        source.retire()
        self.env.source.sources.remove(source)

    def count(self, cmd):
        return sum(p["cmd"] == cmd for _, p in self.calls)


def load_gateway(root):
    import importlib.util

    path = root / "plugins/remote_attn/control_plane/p4_dsv_gateway.py"
    spec = importlib.util.spec_from_file_location("p4_dsv_gateway", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DsvGateway


def gateway(driver, root, folder, **kwargs):
    kwargs.setdefault("autostart", False)
    return load_gateway(root)(
        driver,
        NS(encode=lambda value: [1, 2, 3]),
        folder,
        prefill_url="http://127.0.0.1:30000/generate",
        **kwargs,
    )


def until(gateway_, rid, phase, limit=20):
    for _ in range(limit):
        if gateway_.entry(rid)["phase"] == phase:
            return gateway_.entry(rid)
        gateway_.tick()
    raise AssertionError(gateway_.entry(rid))
