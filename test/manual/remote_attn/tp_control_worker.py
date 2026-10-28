"""Subprocess fixture for the CPU-only TP control agreement regression test."""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
from sglang_remote_attn.tp import agree_control_result, inject_control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("rendezvous")
    parser.add_argument("output")
    args = parser.parse_args()
    dist.init_process_group(
        "gloo",
        init_method=Path(args.rendezvous).as_uri(),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    polls = []

    def poll():
        polls.append(args.rank)
        return {
            "__handoff_ctl__": True,
            "data": {"cmd": "import", "nonce": "same-on-all-ranks"},
        }

    try:
        requests = inject_control(
            ["request"] if args.rank == 0 else None, leader=args.rank == 0, poll=poll
        )
        # These are the upstream work-channel timeout aborts, appended after
        # plugin injection and before the same CPU-group broadcast.
        if requests is not None:
            requests = [*requests, "timeout-abort"]
        message = [requests]
        dist.broadcast_object_list(message, src=0)
        outcomes = []
        for mode in ("success", "failed-shard", "changed-schedule", "both-reject"):
            result = {
                "ok": not (
                    mode == "both-reject" or mode == "failed-shard" and args.rank == 1
                )
            }
            if not result["ok"]:
                result["error"] = "injected import rejection"
            state = (
                ("request",)
                if not (mode == "changed-schedule" and args.rank == 1)
                else ()
            )
            agreed, fatal = agree_control_result(
                result, state, group=dist.group.WORLD, world_size=2
            )
            outcomes.append({"mode": mode, "ok": agreed["ok"], "fatal": fatal})
        # Real two-rank retirement: the faster rank must not return allocator
        # capacity before its peer has drained, nor shrink the next collective.
        from sglang_remote_attn.pre_layerwise import Config, Controller, Source

        controller = Controller.__new__(Controller)
        controller.config = Config("remote", Path(args.output).with_suffix(".journal"))
        controller.scheduler = SimpleNamespace(attn_tp_cpu_group=dist.group.WORLD)
        controller.tp_size = 2
        released = []
        sources = []
        for name in ("a", "b"):
            req = SimpleNamespace(disagg_kv_sender=SimpleNamespace(poll=lambda: 4))
            source = Source(
                controller,
                req,
                controller.config.directory / name,
                name,
                [],
                slice(0, 1),
                (),
                1,
                responded=True,
                finalized=True,
                release=lambda name=name: released.append(name),
            )
            req._pre_layerwise_source = source
            sources.append(source)
        controller.sources = list(sources)
        retirements = []
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.schedule_batch": SimpleNamespace(
                    FINISH_LENGTH=None
                ),
                "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                    KVPoll=SimpleNamespace(Success=4, Failed=0)
                ),
            },
        ):
            for step in range(3):
                if step >= args.rank:
                    sources[0].remote_closed.set()
                if step >= 2 - args.rank:
                    sources[1].remote_closed.set()
                controller.poll_sources()
                retirements.append(list(released))
        Path(args.output).write_text(
            json.dumps(
                {
                    "requests": message[0],
                    "polls": polls,
                    "outcomes": outcomes,
                    "retirements": retirements,
                }
            )
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
