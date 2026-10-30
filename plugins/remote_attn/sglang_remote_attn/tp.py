"""CPU-group agreement for control mutations and graph/eager dispatch."""


def agree_graph(eligible, all_local, *, group, world_size):
    """Replay only if every shard is eligible and its prefix is GPU-readable.

    Always enter this collective, including ranks with no target states and
    ranks rejected by the native graph runner. A rank-local short circuit here
    can send peers into different model collective sequences.
    """
    if world_size == 1:
        return bool(eligible and all_local), bool(all_local)
    import torch
    import torch.distributed as dist

    votes = torch.tensor([eligible, all_local], dtype=torch.int32, device="cpu")
    dist.all_reduce(votes, op=dist.ReduceOp.MIN, group=group)
    eligible, all_local = votes.tolist()
    return bool(eligible and all_local), bool(all_local)


def inject_control(recv_reqs, *, leader, poll):
    if leader:
        control = poll()
        if control is not None:
            return [*(recv_reqs or []), control]
    return recv_reqs


def scheduling_state(scheduler):
    """Compare logical ownership, never rank-local physical slot numbers."""

    def rows(reqs):
        result = []
        for item in reqs or []:
            req = getattr(item, "req", item)
            kv = req.kv
            result.append(
                (
                    req.rid,
                    len(req.output_ids),
                    int(kv.kv_committed_len),
                    kv.req_pool_idx is not None,
                    bool(getattr(req, "_handoff_frozen", False)),
                )
            )
        return tuple(result)

    def batch(name):
        value = getattr(scheduler, name, None)
        return rows(value.reqs if value is not None else [])

    queues = []
    for name in ("disagg_decode_prealloc_queue", "disagg_decode_transfer_queue"):
        queue = getattr(scheduler, name, None)
        queues.append(rows(getattr(queue, "queue", [])))
    return (
        batch("running_batch"),
        batch("last_batch"),
        rows(scheduler.waiting_queue),
        rows(getattr(scheduler, "_handoff_held_reqs", [])),
        tuple(queues),
    )


def agree_control_result(result, state, *, group, world_size):
    """A failed/mutated shard must not let its peers enter a model collective.

    This is one CPU collective per command, not per decode step. A divergent
    command stops the engine on all ranks; it is not claimed to be an atomic
    cross-rank rollback of externally visible transport operations.
    """
    if world_size == 1:
        return result, False
    import torch.distributed as dist

    local = {
        "ok": bool(result.get("ok")) and not result.get("errors"),
        "state": state,
        "error": result.get("error") or result.get("errors"),
    }
    outcomes = [None] * world_size
    dist.all_gather_object(outcomes, local, group=group)
    divergent = any(
        row["ok"] != outcomes[0]["ok"] or row["state"] != outcomes[0]["state"]
        for row in outcomes
    )
    if divergent:
        return {
            **result,
            "ok": False,
            "error": "TP handoff command diverged; engine must stop before scheduling",
            "tp_errors": [row["error"] for row in outcomes],
            "fatal": True,
        }, True
    return result, False
