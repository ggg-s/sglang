"""Scheduler-owned pre-freeze FULL/SWA/row reservations for native DSV D2D.

Tickets never expire implicitly: a lost coordinator can reconcile status and
cancel, but cannot cause memory to be reused while an import may own it.
"""

from dataclasses import dataclass

from . import dsv_d2d as d2d


@dataclass
class Ticket:
    plan: dict
    req: object
    complete: bool = False
    consumed: bool = False
    cancelled: bool = False
    release_started: bool = False
    error: str | None = None
    ready: object = None


def records(controller):
    if not hasattr(controller, "d2d_plans"):
        controller.d2d_plans, controller.d2d_reservations = {}, {}
    return controller.d2d_plans, controller.d2d_reservations


def status(controller, epoch):
    _, tickets = records(controller)
    t = tickets.get(epoch)
    if t is None:
        return dict(cancelled=True) if epoch in tickets else None
    return dict(
        rid=t.plan["target_rid"],
        lower=t.plan["lower"],
        upper=t.plan["upper"],
        reserved=t.complete and not t.cancelled and not t.consumed,
        consumed=t.consumed,
        cancelled=t.cancelled,
        error=t.error,
    )


def select(controller, data, key):
    from .handoff_control import _select_transfer

    return _select_transfer(
        {"transfer": data.get(key), "transfers": data.get(key + "s")},
        controller.rank,
        controller.tp_size,
    )


def check_plan(controller, plan, epoch):
    if (
        plan["version"] != d2d.VERSION
        or plan["epoch"] != epoch
        or plan["tp_size"] != controller.tp_size
        or plan["rank"] != controller.rank
        or plan["schema"] != controller._schema(d2d.manager(controller))
    ):
        raise ValueError("D2D reservation epoch/rank/TP/schema mismatch")
    if not plan["destination"] or not plan["target_rid"]:
        raise ValueError("empty D2D reservation destination/rid")
    lower, upper = plan["lower"], plan["upper"]
    if type(lower) is not int or type(upper) is not int or not 0 < lower <= upper:
        raise ValueError("invalid D2D reservation interval")
    if lower != len(plan["origin_input_ids"]) + len(plan["output_ids"]) - 1:
        raise ValueError("invalid D2D reservation history")
    if upper > controller.scheduler.req_to_token_pool.req_to_token.shape[1]:
        raise ValueError("D2D reservation exceeds native request context")


def handle(controller, data):
    plans, tickets = records(controller)
    sources, targets = d2d.records(controller)
    epoch, cmd = data["migration_epoch"], data["cmd"]
    if cmd == "d2d_plan":
        if epoch in sources or epoch in targets or epoch in tickets:
            raise ValueError("D2D epoch already has an owner")
        if epoch in plans:
            plan, req = plans[epoch]
            if (data["rid"], data["target_rid"], data["destination"]) != (
                req.rid,
                plan["target_rid"],
                plan["destination"],
            ):
                raise ValueError("D2D plan retry identity mismatch")
            return dict(plan=plan, duplicate=True)
        req = d2d.find_req(controller.scheduler, data["rid"])
        if req.finished() or getattr(req, "_dsv_d2d_source", None) is not None:
            raise ValueError("D2D plan requires an active unowned request")
        params = d2d.sampling(req)
        lower = len(req.origin_input_ids) + len(req.output_ids) - 1
        growth = data.get("growth_tokens", 64)
        if not req.output_ids or type(growth) is not int or growth < 0:
            raise ValueError("D2D plan requires sampled output and nonnegative growth")
        upper = min(
            lower + growth,
            len(req.origin_input_ids) + params["max_new_tokens"] - 1,
            controller.scheduler.req_to_token_pool.req_to_token.shape[1],
        )
        plan = dict(
            version=d2d.VERSION,
            epoch=epoch,
            rank=controller.rank,
            tp_size=controller.tp_size,
            rid=req.rid,
            destination=str(data["destination"]),
            target_rid=str(data["target_rid"]),
            schema=controller._schema(d2d.manager(controller)),
            lower=lower,
            upper=upper,
            origin_input_ids=list(req.origin_input_ids),
            output_ids=list(req.output_ids),
            sampling=params,
        )
        check_plan(controller, plan, epoch)
        plans[epoch] = (plan, req)
        return dict(plan=plan)
    if cmd == "d2d_reserve_cancel":
        t = tickets.get(epoch)
        if t is None:
            # Keep a tombstone even if the reserve RPC has not arrived yet.
            # An out-of-order reserve must never resurrect a cancelled epoch.
            tickets[epoch] = None
            return dict(cancelled=True)
        if t.cancelled:
            return dict(cancelled=True, duplicate=True)
        if t.consumed:
            raise ValueError("D2D reservation consumed; use target abort/cancellation")
        if not t.complete or t.release_started or t.error:
            raise d2d.TransportFailure(
                "D2D reservation allocation/release unproven; retaining memory"
            )
        t.release_started = True
        try:
            t.ready.synchronize()
            from sglang.srt.mem_cache.common import release_kv_cache

            release_kv_cache(t.req, controller.scheduler.tree_cache, checkpoint=False)
            t.cancelled = True
        except BaseException as exc:
            t.error = repr(exc)
            raise
        return dict(cancelled=True)
    if cmd != "d2d_reserve":
        raise ValueError("unsupported D2D reservation command")
    plan = select(controller, data, "plan")
    if epoch in tickets:
        t = tickets[epoch]
        if t is None or t.cancelled or t.consumed or t.error:
            raise ValueError("D2D reservation epoch retired/failed/consumed")
        if t.plan != plan:
            raise ValueError("D2D reservation retry plan mismatch")
        return dict(reserved=t.complete, reservation=plan, duplicate=True)
    error = None
    try:
        check_plan(controller, plan, epoch)
        if epoch in sources or epoch in targets:
            raise ValueError("D2D reservation epoch already used")
        from sglang.srt.sampling.sampling_params import SamplingParams
        from .handoff_control import _new_req, _tracked

        if plan["target_rid"] in _tracked(controller.scheduler) or any(
            t is not None
            and not t.cancelled
            and t.plan["target_rid"] == plan["target_rid"]
            for t in tickets.values()
        ):
            raise ValueError("D2D reservation target rid already in use")
        params = SamplingParams(**plan["sampling"])
        params.normalize(controller.scheduler.tokenizer)
        output = plan["output_ids"] + [0] * (plan["upper"] - plan["lower"])
        req = _new_req(
            controller.scheduler,
            plan["target_rid"],
            plan["origin_input_ids"],
            output,
            params,
        )
        prealloc = controller.scheduler.disagg_decode_prealloc_queue
        start = None
        full, swa = prealloc._prealloc_required_tokens(req)
        if not getattr(controller, "hybrid_gdn", False):
            page = controller.pool.page_size
            start = (
                max(0, plan["lower"] - controller.pool.sliding_window) // page * page
            )
            # Charge the union of windows, not only the upper-bound tail.
            if controller.pool.request_window is None:
                swa = max(swa, (plan["upper"] - start + page - 1) // page * page)
        d2d.allocation_preflight(controller, req, required=(full, swa))
    except Exception as exc:
        error = repr(exc)
    if not controller.agree([int(error is None)])[0]:
        raise ValueError(
            f"TP D2D reserve preflight rejected: {error or 'peer rejected'}"
        )
    t = tickets[epoch] = Ticket(plan, req)
    try:
        prealloc._pre_alloc(req, swa_tail_start=start)
        t.ready = d2d.dp.event(d2d.pool_device(controller))
        t.ready.synchronize()
        t.complete = True
    except BaseException as exc:
        t.error = repr(exc)
    if not controller.agree([int(t.complete)])[0]:
        # Completed peers keep their tickets for explicit cancellation; failed
        # peers retain uncertain partial native allocations for diagnosis.
        raise d2d.TransportFailure(
            f"TP D2D reservation allocation failed: {t.error or 'peer failed'}"
        )
    return dict(reserved=True, reservation=plan)


def check_freeze(controller, data, req):
    plans, _ = records(controller)
    certificate = select(controller, data, "reservation")
    if certificate is None:
        if (
            getattr(controller, "hybrid_gdn", False)
            or not data.get("allow_unreserved", False)
            or data["migration_epoch"] in plans
        ):
            raise ValueError("D2D freeze requires target reservation")
        return  # Explicit diagnostic compatibility path, never the default.
    saved = plans.get(data["migration_epoch"])
    if saved is None or saved[0] != certificate or saved[1] is not req:
        raise ValueError("D2D freeze reservation does not match source plan/request")
    if (data["destination"], data["target_rid"]) != (
        certificate["destination"],
        certificate["target_rid"],
    ) or not certificate["lower"] <= req.kv.kv_committed_len <= certificate["upper"]:
        raise ValueError("D2D freeze outside reserved destination/length interval")


def consume(controller, data, transfer, req):
    _, tickets = records(controller)
    epoch = data["migration_epoch"]
    if epoch not in tickets:
        if (
            getattr(controller, "hybrid_gdn", False)
            or data.get("require_reservation", False)
            or transfer.get("reservation_required", False)
        ):
            raise ValueError("D2D import requires a live native reservation")
        return False
    t = tickets[epoch]
    if t is None or t.cancelled or t.consumed or t.error or not t.complete:
        raise ValueError("D2D import reservation unavailable")
    plan = t.plan
    if (
        req.rid != plan["target_rid"]
        or transfer["destination"] != plan["destination"]
        or transfer["origin_input_ids"] != plan["origin_input_ids"]
        or transfer["sampling"] != plan["sampling"]
        or transfer["output_ids"][: len(plan["output_ids"])] != plan["output_ids"]
        or not plan["lower"] <= transfer["kv_len"] <= plan["upper"]
    ):
        raise ValueError("D2D frozen manifest does not fit reservation")
    t.ready.synchronize()
    # Transfer ownership before any allocator mutation. Failure leaves the
    # tracked Target owning the row; reserve_cancel can no longer free it.
    req.kv, t.consumed = t.req.kv, True
    length = transfer["kv_len"]
    page = controller.pool.page_size
    end = (length + page - 1) // page * page
    if end < plan["upper"]:
        row = controller.scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx]
        controller.scheduler.token_to_kv_pool_allocator.free(row[end : plan["upper"]])
    req.kv.kv_allocated_len = req.kv.kv_committed_len = length
    req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
    req.prefix_len = 0
    req.extend_end = length
    # Keep the reservation's earlier SWA eviction cursor. Native ratcheting
    # releases those extra pages; setting it to the later cursor would leak.
    return True
