"""Subprocess fixture for the CPU-only TP control agreement regression test."""

import argparse
import json
import sys
import threading
from dataclasses import make_dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
from sglang_remote_attn.tp import agree_control_result, inject_control


def progressive_graph_leases(args, context):
    """Real admission/retirement votes with transport and model boundaries on CPU."""
    import torch
    from sglang_remote_attn import handoff_control as runtime
    from sglang_remote_attn import hooks
    from sglang_remote_attn.pre_layerwise import (
        Config,
        Controller,
        LayerCoverage,
        Target,
        write_json,
    )

    root = Path(args.output).with_suffix(".leases")
    config = make_dataclass("GraphConfig", [("decode", str), ("prefill", str)])(
        "full", "breakable"
    )
    context.get_exec = lambda: SimpleNamespace(
        graph=SimpleNamespace(cuda_graph_config=config)
    )
    model_runner = SimpleNamespace(model=SimpleNamespace(modules=lambda: []))
    decode = SimpleNamespace(model_runner=model_runner, capture_bs=[1, 2, 4], max_bs=4)
    prefill = SimpleNamespace(
        model_runner=model_runner, capture_num_tokens=[4, 8], max_bs=4
    )
    model_runner.decode_cuda_graph_runner, model_runner.prefill_cuda_graph_runner = (
        decode,
        prefill,
    )
    scheduler = SimpleNamespace(
        attn_tp_cpu_group=dist.group.WORLD,
        tp_worker=SimpleNamespace(model_runner=model_runner),
    )
    controller = Controller.__new__(Controller)
    controller.scheduler, controller.tp_size, controller.rank = scheduler, 2, args.rank
    controller.layers, controller.early_admissions, controller.targets = 1, 0, []
    controller.config = Config("remote", root)
    scheduler._pre_layerwise = model_runner._pre_layerwise = controller
    freed = []
    queue = SimpleNamespace(
        queue=[],
        metadata_buffers=SimpleNamespace(bootstrap_room=torch.full((3, 1), 7)),
        req_to_metadata_buffer_idx_allocator=SimpleNamespace(free=freed.append),
        _commit_transfer_to_req=lambda *a, **kw: None,
    )

    def offer(index):
        req = SimpleNamespace(
            rid=str(index),
            bootstrap_room=7,
            output_ids=[],
            kv=SimpleNamespace(req_pool_idx=index),
            finished=lambda: False,
        )
        receiver = SimpleNamespace(status=2, clear=lambda: None)
        receiver.poll = lambda: receiver.status
        dr = SimpleNamespace(req=req, kv_receiver=receiver, metadata_buffer_index=index)
        target = Target(
            controller, queue, dr, root / str(index), str(index), LayerCoverage(2, 1)
        )
        dr._pre_layerwise_offer = target
        queue.queue.append(dr)
        return target

    def ready(target):
        write_json(
            target.path / "ready.json",
            {
                "generation": target.generation,
                "prefix_len": 2,
                "service": {"handoff_id": target.generation},
            },
        )

    def complete(target):
        target.decode_req.kv_receiver.status = 4
        write_json(
            target.path / "layer-0.json",
            {"generation": target.generation, "layer": 0, "start": 0, "end": 2},
        )

    reductions = 0
    actual_reduce = dist.all_reduce

    def counted_reduce(*a, **kw):
        nonlocal reductions
        reductions += 1
        return actual_reduce(*a, **kw)

    snapshots = []
    with (
        patch.dict(
            sys.modules,
            {
                "sglang.srt.runtime_context": context,
                "sglang.srt.disaggregation.base.conn": SimpleNamespace(
                    KVPoll=SimpleNamespace(Success=4, Failed=0)
                ),
                "sglang.srt.layers.radix_attention": SimpleNamespace(
                    RadixAttention=type("RadixAttention", (), {})
                ),
            },
        ),
        patch.object(runtime, "Path", return_value=root),
        patch.object(runtime, "PROFILE", False),
        patch.object(runtime, "RPC_PROFILE", False),
        patch.object(runtime, "_TARGET_STATES", []),
        patch.object(dist, "all_reduce", counted_reduce),
    ):
        controller.validate_graphs()
        with patch.object(
            runtime, "_TARGET_STATES", [object()] if args.rank == 0 else []
        ):
            controller.validate_graphs()
            existing_state_fallback = not controller.graph_fast_path
        # The same capture mismatch must fail startup on *both* ranks.
        if args.rank == 1:
            decode.capture_bs.append(8)
        try:
            controller.validate_graphs()
            mismatch_rejected = False
        except RuntimeError:
            mismatch_rejected = True
        if args.rank == 1:
            decode.capture_bs.pop()
        controller.validate_graphs()

        def sample():
            before = reductions
            verdicts = []
            # Include a Local steady-state burst, not just one lucky step.
            for _ in range(16):
                verdicts.append(
                    [
                        hooks._decode_graph(lambda *a: True, decode, None),
                        hooks._prefill_graph(lambda *a: True, prefill, None),
                    ]
                )
            snapshots.append(
                {
                    "leases": len(controller.targets),
                    "verdicts": verdicts,
                    "forward_reductions": reductions - before,
                }
            )

        a = offer(0)
        if args.rank == 0:
            ready(a)
        controller.early_targets(queue, None)  # One rank ready: neither admits.
        sample()
        ready(a)
        controller.early_targets(queue, None)
        sample()
        b = offer(1)
        ready(b)
        controller.early_targets(queue, None)
        sample()
        complete(a)
        controller.early_targets(queue, None)
        sample()  # A retired; B still prevents replay.
        if args.rank == 0:
            complete(b)
        controller.early_targets(queue, None)
        sample()  # The fast shard must keep B's lease.
        if args.rank == 1:
            complete(b)
        controller.early_targets(queue, None)
        sample()
        c = offer(2)
        ready(c)
        controller.early_targets(queue, None)
        sample()  # A new admission revokes Local before the next forward.
        complete(c)
        controller.early_targets(queue, None)
        sample()
    return {
        "snapshots": snapshots,
        "mismatch_rejected": mismatch_rejected,
        "existing_state_fallback": existing_state_fallback,
        "boundary_reductions": reductions,
        "freed": freed,
    }


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
        # Exercise the actual plugin can_run_graph hooks over real Gloo. The
        # native predicate is rank-local too, so it must participate in the vote.
        from sglang_remote_attn import handoff_control as runtime
        from sglang_remote_attn import hooks

        context = SimpleNamespace(
            get_parallel=lambda: SimpleNamespace(
                tp_size=2,
                tp_rank=args.rank,
                tp_group=SimpleNamespace(cpu_group=dist.group.WORLD),
            )
        )
        event = threading.Event()
        state = SimpleNamespace(
            mode="remote", events=[event], error=None, aborted=False
        )
        decode, prefill = SimpleNamespace(), SimpleNamespace()
        graphs = []
        with patch.dict(sys.modules, {"sglang.srt.runtime_context": context}):
            for step in range(9):
                # Start Local, then admit only rank 1's target, then give both
                # ranks a target with rank 0 already ready. Later only rank 0
                # sees a newly admitted incomplete target.
                states = [state] if step in (1, 2, 3, 4, 5, 6, 7) else []
                if (step == 1 and args.rank == 0) or (step == 6 and args.rank == 1):
                    states = []
                if (step >= 3 and step != 6) or (step == 2 and args.rank == 0):
                    event.set()
                else:
                    event.clear()
                state.error = "failed import" if step == 7 and args.rank == 1 else None
                eligible = not (step == 4 and args.rank == 1)
                with patch.object(runtime, "_TARGET_STATES", states):
                    graphs.append(
                        [
                            hooks._decode_graph(lambda *a: eligible, decode, None),
                            hooks._prefill_graph(lambda *a: eligible, prefill, None),
                        ]
                    )
        leases = progressive_graph_leases(args, context)
        from sglang_remote_attn.segmented_graph import (
            RemoteGraphBackend,
            agree_capture_settings,
        )

        startup = []
        for scenario in ("same", "flag-mismatch", "bucket-mismatch"):
            report = (
                not (scenario == "flag-mismatch" and args.rank == 1),
                (1, 2)
                if scenario == "bucket-mismatch" and args.rank == 1
                else (1, 2, 4),
            )
            try:
                agree_capture_settings(report, group=dist.group.WORLD, world_size=2)
                startup.append(True)
            except RuntimeError:
                startup.append(False)
        controller = SimpleNamespace(graph_fast_path=True, targets=[])
        runner = SimpleNamespace(
            model_runner=SimpleNamespace(_pre_layerwise=controller)
        )
        backend = RemoteGraphBackend.__new__(RemoteGraphBackend)
        backend.runner, backend.remote_keys = (
            runner,
            {size: SimpleNamespace(size=size) for size in (1, 2, 4)},
        )
        backend.remote = SimpleNamespace(can_run=lambda *a: True)
        runner.backend = backend
        routing = []
        import torch

        with (
            patch.dict(sys.modules, {"sglang.srt.runtime_context": context}),
            patch.object(
                dist, "all_reduce", side_effect=AssertionError("per-forward graph vote")
            ),
        ):
            for remote_active in (False, True, False, True):
                controller.targets = [object()] if remote_active else []
                row = []
                for size in (1, 2, 3, 4, 8):
                    batch = SimpleNamespace(
                        batch_size=size,
                        input_ids=torch.zeros(size, dtype=torch.int64),
                        input_embeds=None,
                        replace_embeds=None,
                        forward_mode=SimpleNamespace(is_decode=lambda: True),
                    )
                    row.append(hooks._decode_graph(lambda *a: True, runner, batch))
                routing.append(row)
        Path(args.output).write_text(
            json.dumps(
                {
                    "requests": message[0],
                    "polls": polls,
                    "outcomes": outcomes,
                    "retirements": retirements,
                    "graphs": graphs,
                    "leases": leases,
                    "segmented_startup": startup,
                    "segmented_routing": routing,
                }
            )
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
