"""Native Qwen D2D reservation tickets; no implicit expiry or fallback allocation."""

from array import array
from types import SimpleNamespace

from . import d2d_ranges as ranges
from .transport_failure import TransportFailure


def runtime():
    from . import handoff_control

    return handoff_control


def agree(scheduler, good):
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    if p.tp_size == 1:
        return good
    import torch.distributed as dist

    votes = [None] * p.tp_size
    dist.all_gather_object(votes, bool(good), group=p.tp_group.cpu_group)
    return all(votes)


def schema(scheduler):
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    k, v = pool.get_key_buffer(0), pool.get_value_buffer(0)
    if (
        len(k.shape) != 3
        or k.shape[1:] != v.shape[1:]
        or k.dtype != v.dtype
        or str(k.dtype) not in ("torch.float16", "torch.bfloat16")
    ):
        raise ValueError("Qwen D2D requires matching FP16/BF16 native K/V rows")
    return dict(
        layers=pool.layer_num,
        key=list(k.shape[1:]),
        value=list(v.shape[1:]),
        dtype=str(k.dtype),
        page=scheduler.token_to_kv_pool_allocator.page_size,
    )


def plans(scheduler):
    if not hasattr(scheduler, "_qwen_d2d_plans"):
        scheduler._qwen_d2d_plans = {}
    return scheduler._qwen_d2d_plans


def select(scheduler, rows):
    r = runtime()
    return r._select_transfer(
        dict(transfers=rows), r._tp_rank(scheduler), r._tp_size(scheduler)
    )


def plan(scheduler, data):
    r = runtime()
    epoch = str(data["epoch"])
    result = []
    for item in data["items"]:
        req = r._tracked(scheduler)[item["rid"]]
        key = (epoch, req.rid)
        if not hasattr(scheduler, "_qwen_d2d_plan_owners"):
            scheduler._qwen_d2d_plan_owners = {}
        if key in plans(scheduler):
            if scheduler._qwen_d2d_plan_owners.get(key) is not req:
                raise ValueError("D2D source request identity changed")
            saved = plans(scheduler)[key]
            if saved["target_rid"] != item["target_rid"]:
                raise ValueError("D2D plan identity changed")
            result.append(saved)
            continue
        lower = int(req.kv.kv_committed_len)
        expected = len(req.origin_input_ids) + len(req.output_ids) - 1
        if (
            req.finished()
            or not req.output_ids
            or lower != expected
            or getattr(req, "_handoff_frozen", False)
        ):
            raise ValueError("D2D plan requires a live complete Decode boundary")
        growth = item.get("growth_tokens", 64)
        if type(growth) is not int or growth < 0:
            raise ValueError("invalid D2D growth budget")
        sampling = r._sampling_dict(req)
        upper = min(
            lower + growth,
            len(req.origin_input_ids) + sampling["max_new_tokens"] - 1,
            scheduler.req_to_token_pool.req_to_token.shape[1],
        )
        if upper < lower:
            raise ValueError("D2D plan exceeds request context")
        p = dict(
            protocol=ranges.PROTOCOL,
            epoch=epoch,
            rid=req.rid,
            target_rid=item["target_rid"],
            lower=lower,
            upper=upper,
            rank=r._tp_rank(scheduler),
            tp_size=r._tp_size(scheduler),
            schema=schema(scheduler),
            origin_input_ids=list(req.origin_input_ids),
            output_ids=list(req.output_ids),
            sampling=sampling,
        )
        plans(scheduler)[key] = p
        scheduler._qwen_d2d_plan_owners[key] = req
        result.append(p)
    return dict(plans=result)


def reserve(scheduler, data):
    r = runtime()
    selected = select(scheduler, data["plans"])
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.tp_size > 1:
        import torch.distributed as dist

        rows = [None] * parallel.tp_size
        identity = [{k: v for k, v in p.items() if k != "rank"} for p in selected]
        dist.all_gather_object(rows, identity, group=parallel.tp_group.cpu_group)
        if any(x != identity for x in rows):
            raise ValueError("TP D2D source plans disagree")
    output = {}
    for p in selected:
        rid, epoch = p["target_rid"], str(data["epoch"])
        old = r._reservations(scheduler).get(rid)
        parallel = get_parallel()
        if parallel.tp_size > 1:
            states = [None] * parallel.tp_size
            dist.all_gather_object(
                states, old is not None, group=parallel.tp_group.cpu_group
            )
            if any(state != states[0] for state in states):
                raise ValueError("TP D2D reservation ownership differs across ranks")
        if not agree(
            scheduler,
            old is None
            or old.get("plan") == p
            and not old.get("error")
            and old.get("indices") is not None,
        ):
            raise ValueError("TP D2D reservation already consumed/failed/changed")
        if old is not None:
            output[rid] = p
            continue
        error = None
        req = None
        try:
            if (
                p["protocol"] != ranges.PROTOCOL
                or p["epoch"] != epoch
                or p["rank"] != r._tp_rank(scheduler)
                or p["tp_size"] != r._tp_size(scheduler)
                or p["schema"] != schema(scheduler)
            ):
                raise ValueError("D2D reservation schema/epoch/TP mismatch")
            if (
                type(p["lower"]) is not int
                or type(p["upper"]) is not int
                or p["lower"] != len(p["origin_input_ids"]) + len(p["output_ids"]) - 1
                or not p["output_ids"]
                or not isinstance(p["target_rid"], str)
                or not p["target_rid"]
            ):
                raise ValueError("invalid D2D native reservation history/interval")
            if (
                not 0
                < p["lower"]
                <= p["upper"]
                <= scheduler.req_to_token_pool.req_to_token.shape[1]
            ):
                raise ValueError("D2D reservation outside context")
            if rid in r._tracked(scheduler):
                raise ValueError("D2D target rid already live")
            sp = r._sampling_from(
                p["sampling"], p["sampling"]["max_new_tokens"], scheduler.tokenizer
            )
            req = r._new_req(
                scheduler,
                rid,
                p["origin_input_ids"],
                p["output_ids"] + [0] * (p["upper"] - p["lower"]),
                sp,
            )
            stats = r._capacity_stats(scheduler)
            required = scheduler.disagg_decode_prealloc_queue._prealloc_required_tokens(
                req
            )[0]
            if stats["free_slots"] < 1 or required > stats["free_kv"]:
                raise ValueError(
                    "native D2D budget conflicts with ordinary admission leases"
                )
            if (epoch, rid) in getattr(scheduler, "_qwen_d2d_retired", set()):
                raise ValueError("D2D reservation epoch retired")
            from .dsv_d2d import allocation_preflight

            allocation_preflight(SimpleNamespace(scheduler=scheduler), req)
        except Exception as exc:
            error = repr(exc)
        if not agree(scheduler, error is None):
            raise ValueError(f"TP native D2D reserve preflight rejected: {error}")
        rec = dict(
            plan=p,
            native_req=req,
            epoch=epoch,
            tokens=p["upper"],
            indices=None,
            slots=0,
            slot_consumed=True,
            error=None,
            consumed=False,
        )
        r._reservations(scheduler)[rid] = rec
        try:
            req._d2d_native_alloc = True
            scheduler.disagg_decode_prealloc_queue._pre_alloc(req)
            device = scheduler.req_to_token_pool.req_to_token.device
            from .remote_data_plane import event

            rec["indices"] = scheduler.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, : p["upper"]
            ].clone()
            rec["ready"] = event(device)
            rec["ready"].synchronize()
        except BaseException as exc:
            rec["error"] = repr(exc)
        if not agree(scheduler, rec["error"] is None):
            raise TransportFailure(
                "native D2D partial allocation unproven; retaining row/KV"
            )
        output[rid] = p
    return dict(reserved=output, reservations=output, errors={})


def confirm(scheduler, data):
    selected = select(scheduler, data["certificates"])
    for rid, certificate in selected.items():
        key = (str(data["epoch"]), certificate["rid"])
        if plans(scheduler).get(key) != certificate or certificate["target_rid"] != rid:
            raise ValueError("D2D target reservation does not match source plan")
    if not hasattr(scheduler, "_qwen_d2d_confirmed"):
        scheduler._qwen_d2d_confirmed = {}
    scheduler._qwen_d2d_confirmed.update(
        {(str(data["epoch"]), p["rid"]): p for p in selected.values()}
    )
    return dict(confirmed=True)


def check_freeze(scheduler, req, epoch):
    p = getattr(scheduler, "_qwen_d2d_confirmed", {}).get((str(epoch), req.rid))
    if (
        getattr(scheduler, "_qwen_d2d_plan_owners", {}).get((str(epoch), req.rid))
        is not req
    ):
        raise ValueError("D2D freeze requires original source request")
    if p is None or not p["lower"] <= req.kv.kv_committed_len <= p["upper"]:
        raise ValueError(
            "D2D freeze requires a live matching target reservation interval"
        )
    if (
        req.finished()
        or int(req.kv.kv_committed_len)
        != len(req.origin_input_ids) + len(req.output_ids) - 1
    ):
        raise ValueError("D2D freeze requires a complete Decode boundary")
    if (
        list(req.origin_input_ids) != p["origin_input_ids"]
        or list(req.output_ids[: len(p["output_ids"])]) != p["output_ids"]
        or runtime()._sampling_dict(req) != p["sampling"]
    ):
        raise ValueError("D2D source history/sampling changed after planning")
    return p


def consume(scheduler, rid, meta, epoch):
    r = runtime()
    rec = r._reservations(scheduler).get(rid)
    if rec is None or "native_req" not in rec:
        raise ValueError("D2D range import requires native reservation")
    p, length = (
        rec["plan"],
        ranges.check_history(
            meta["origin_input_ids"], meta["output_ids"], meta["kv_committed_len"]
        ),
    )
    if (
        rec.get("error")
        or rec["consumed"]
        or rec["indices"] is None
        or str(epoch) != p["epoch"]
        or meta["rid"] != p["rid"]
        or meta["target_rid"] != rid
        or meta["origin_input_ids"] != p["origin_input_ids"]
        or meta["output_ids"][: len(p["output_ids"])] != p["output_ids"]
        or meta["sampling"] != p["sampling"]
        or not p["lower"] <= length <= p["upper"]
    ):
        raise ValueError("D2D frozen manifest does not fit native reservation")
    if meta.get("d2d_range_protocol"):
        shape = p["schema"]
        chunks = meta.get("chunks")
        if (
            meta["d2d_range_protocol"] != ranges.PROTOCOL
            or not isinstance(chunks, list)
            or len(chunks) != shape["layers"]
            or meta.get("kv_heads") != shape["key"][0]
            or meta.get("head_dim") != shape["key"][1]
            or meta.get("dtype") != shape["dtype"].removeprefix("torch.")
            or any(
                not isinstance(part, list)
                or not part
                or any(type(n) is not int or n < 1 for n in part)
                or sum(part) != length
                for part in chunks
            )
        ):
            raise ValueError("D2D frozen geometry does not match native reservation")
    rec["ready"].synchronize()
    req = rec["native_req"]
    # Publish ownership before releasing any whole trailing pages. Unknown
    # partial free is never retried and reserve_cancel cannot free this row.
    rec["consumed"] = True
    if not hasattr(scheduler, "_qwen_d2d_retired"):
        scheduler._qwen_d2d_retired = set()
    scheduler._qwen_d2d_retired.add((p["epoch"], rid))
    try:
        page = scheduler.token_to_kv_pool_allocator.page_size
        end = (length + page - 1) // page * page
        if end < p["upper"]:
            scheduler.token_to_kv_pool_allocator.free(rec["indices"][end:])
        indices = rec["indices"][:length]
        req.output_ids = (
            array("q", meta["output_ids"])
            if isinstance(req.origin_input_ids, array)
            else list(meta["output_ids"])
        )
        req.kv.kv_committed_len = req.kv.kv_allocated_len = length
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        req.prefix_len, req.extend_end = 0, length
        rec["indices"] = None
    except BaseException as exc:
        rec["error"] = repr(exc)
        raise
    return req, indices


def cancel(scheduler, rid, rec):
    if rec.get("error") or rec.get("release_started") or rec["consumed"]:
        raise TransportFailure("native D2D allocation/release unproven or consumed")
    rec["release_started"] = True
    if not hasattr(scheduler, "_qwen_d2d_retired"):
        scheduler._qwen_d2d_retired = set()
    scheduler._qwen_d2d_retired.add((rec["epoch"], rid))
    try:
        rec["ready"].synchronize()
        from sglang.srt.mem_cache.common import release_kv_cache

        release_kv_cache(rec["native_req"], scheduler.tree_cache, checkpoint=False)
    except BaseException as exc:
        rec["error"] = repr(exc)
        raise
    runtime()._reservations(scheduler).pop(rid)


def retire_source(scheduler, req):
    owners = getattr(scheduler, "_qwen_d2d_plan_owners", {})
    for key, owner in list(owners.items()):
        if owner is req:
            owners.pop(key)
            plans(scheduler).pop(key, None)
            getattr(scheduler, "_qwen_d2d_confirmed", {}).pop(key, None)
