"""Subprocess fixture for the CPU-only TP control agreement regression test."""

import argparse
import json
from datetime import timedelta
from pathlib import Path

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
        Path(args.output).write_text(
            json.dumps({"requests": message[0], "polls": polls, "outcomes": outcomes})
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
