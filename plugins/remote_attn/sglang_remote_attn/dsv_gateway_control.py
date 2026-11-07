"""Native PD submission and durable-in-worker D2D facts for the P4 coordinator.

All mutations execute on the TP scheduler thread. No legacy Qwen allocator,
forced KV free, token replay, or HTTP stream from the source is used here.
"""

import http.client
import json
import threading
import uuid
from urllib.parse import urlparse

from . import dsv_d2d as d2d
from . import dsv_d2d_reservation as reservations

COMMANDS = {"d2d_worker_status", "d2d_submit_pd", "d2d_cancel"}


def instance(controller):
    if not hasattr(controller, "d2d_instance_id"):
        controller.d2d_instance_id = uuid.uuid4().hex
    return controller.d2d_instance_id


def request_status(controller, req):
    return dict(
        rid=req.rid,
        output_ids=list(req.output_ids),
        finished=bool(req.finished()),
        finished_reason=str(getattr(req, "finished_reason", None)),
        kv_len=int(req.kv.kv_committed_len),
        kv_released=not req.kv.holds_kv,
        frozen=bool(getattr(req, "_handoff_frozen", False)),
        error=getattr(req, "_d2d_submission_error", None),
        migration_ready=bool(
            req.output_ids
            and not req.finished()
            and not getattr(req, "_handoff_frozen", False)
            and getattr(req, "_pre_layerwise_target", None) is None
            and getattr(req, "_dsv41_remote", None) is None
            and d2d.can_quiesce(controller.scheduler)
        ),
    )


def migration_status(controller, epoch):
    sources, targets = d2d.records(controller)
    source, target = sources.get(epoch), targets.get(epoch)
    return dict(
        reservation=reservations.status(controller, epoch),
        source_cancelled=epoch
        in getattr(controller, "d2d_source_cancelled_epochs", ()),
        target_cancelled=epoch in getattr(controller, "d2d_cancelled_epochs", ()),
        source=None
        if source is None
        else dict(
            rid=source.req.rid,
            committed=source.committed,
            copied=source.copied,
            released=source.released,
            resumed=source.released and not source.committed,
            error=source.error,
            kv_released=not source.req.kv.holds_kv,
            transfer=getattr(source, "meta", None),
        ),
        target=None
        if target is None
        else dict(
            rid=target.decode_req.req.rid,
            admitted=target.admitted,
            group_size=target.group_size,
            group_frames=target.group_frames,
            max_entries_per_frame=target.grouped_entries,
            prepared=target.allocation_complete
            and target.rpc is not None
            and not target.error,
            copied=target.copied,
            local=target.closed and not target.cancelled,
            aborted=target.cancelled,
            error=target.error,
            kv_released=not target.decode_req.req.kv.holds_kv,
        ),
    )


def candidates(controller):
    scheduler = controller.scheduler
    reqs = list(d2d.remote.runtime()._tracked(scheduler).values())
    for name in ("running_batch", "last_batch"):
        batch = getattr(scheduler, name, None)
        reqs.extend(getattr(batch, "reqs", ()))
    for batch, _ in getattr(scheduler, "result_queue", ()):
        reqs.extend(batch.reqs)
    request_lists = getattr(scheduler, "request_lists", None)
    if callable(request_lists):
        # Native ReqList iteration yields Req, including held/restoring/releasing
        # requests that are no longer all represented by the old three queues.
        for queue in request_lists().values():
            reqs.extend(queue)
    else:
        # Small standalone gateway fixtures may not own a native Scheduler.
        reqs.extend(getattr(scheduler, "waiting_queue", ()))
        for name in ("disagg_decode_prealloc_queue", "disagg_decode_transfer_queue"):
            queue = getattr(scheduler, name, None)
            for field in ("queue", "pending_reqs", "retracted_queue"):
                reqs.extend(getattr(r, "req", r) for r in getattr(queue, field, ()))
    return [r for r in reqs if hasattr(r, "rid")]


def handle(controller, data):
    scheduler, cmd = controller.scheduler, data["cmd"]
    epoch = data["migration_epoch"]
    if cmd == "d2d_worker_status":
        sources, targets = d2d.records(controller)
        _, tickets = reservations.records(controller)
        epochs = set(sources) | set(targets) | set(tickets)
        epochs.update(getattr(controller, "d2d_cancelled_epochs", ()))
        epochs.update(getattr(controller, "d2d_source_cancelled_epochs", ()))
        reqs = candidates(controller)
        reqs.extend(s.req for s in sources.values())
        reqs.extend(t.decode_req.req for t in targets.values())
        return dict(
            idle=bool(getattr(scheduler, "is_fully_idle", lambda: False)()),
            protocol=d2d.VERSION,
            instance_id=instance(controller),
            rank=controller.rank,
            tp_size=controller.tp_size,
            requests={r.rid: request_status(controller, r) for r in reqs},
            migrations={e: migration_status(controller, e) for e in sorted(epochs)},
        )
    tracked = d2d.remote.runtime()._tracked(scheduler)
    rid = str(data["rid"])
    if cmd == "d2d_cancel":
        req = tracked.get(rid)
        if req is None:
            raise KeyError("P4 cancel requires a known native request")
        if req.finished():
            return dict(cancelled=True, finished=True, kv_released=not req.kv.holds_kv)
        if getattr(req, "_handoff_frozen", False):
            raise ValueError(
                "frozen source must fence readers and resume before native cancel"
            )
        # Native AbortReq uses startswith. Do not let an exact gateway cancel
        # accidentally abort another live request with the same prefix.
        if any(r.rid != rid and r.rid.startswith(rid) for r in candidates(controller)):
            raise ValueError("native cancellation prefix overlaps another request")
        from sglang.srt.managers.io_struct import AbortReq

        scheduler.abort_request(AbortReq(rid=rid))
        return dict(
            cancelled=True,
            finished=bool(req.finished()),
            kv_released=not req.kv.holds_kv,
        )
    if cmd != "d2d_submit_pd":
        raise ValueError("unsupported P4 native command")
    if not hasattr(controller, "d2d_submissions"):
        controller.d2d_submissions = {}
        controller.d2d_submission_requests = {}
    payload = {
        k: data[k]
        for k in (
            "rid",
            "input_ids",
            "sampling",
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "prefill_url",
        )
    }
    previous = controller.d2d_submissions.get(epoch)
    if previous is not None:
        if previous != payload:
            raise ValueError("native PD submission epoch payload changed")
        if getattr(
            controller.d2d_submission_requests[epoch], "_d2d_submission_error", None
        ):
            raise ValueError("native PD submission failed; refusing enqueue retry")
        return dict(queued=True, rid=rid, duplicate=True)
    error = None
    try:
        if rid in tracked or any(r.rid == rid for r in candidates(controller)):
            raise ValueError("native PD request rid already in use")
        if not payload["input_ids"] or int(payload["bootstrap_room"]) <= 0:
            raise ValueError("native PD requires prompt tokens and a bootstrap room")
        from sglang.srt.sampling.sampling_params import SamplingParams
        from .handoff_control import _new_req

        url = urlparse(payload["prefill_url"])
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("native PD prefill requires an HTTP(S) URL")
        params = SamplingParams(**payload["sampling"])
        params.normalize(scheduler.tokenizer)
        req = _new_req(
            scheduler,
            rid,
            payload["input_ids"],
            [],
            params,
            bootstrap_host=payload["bootstrap_host"],
            bootstrap_port=int(payload["bootstrap_port"]),
            bootstrap_room=int(payload["bootstrap_room"]),
            disagg_mode=scheduler.disaggregation_mode,
        )
    except Exception as exc:
        error = repr(exc)
    if not controller.agree([int(error is None)])[0]:
        raise ValueError(
            f"TP native PD submission rejected: {error or 'peer rejected'}"
        )
    # Publish before enqueue. An interrupted enqueue cannot be retried as a
    # second generation of the same request; status/recovery must reconcile it.
    controller.d2d_submissions[epoch] = payload
    controller.d2d_submission_requests[epoch] = req
    tracked[rid] = req
    req._handoff_pd = True
    try:
        scheduler._add_request_to_queue(req)
        if controller.rank == 0:
            req._d2d_prefill_thread = threading.Thread(
                target=prefill,
                args=(req, payload),
                daemon=True,
                name="p4-native-prefill",
            )
            req._d2d_prefill_thread.start()
    except BaseException as exc:
        req._d2d_submission_error = repr(exc)
        raise
    return dict(queued=True, rid=rid, bootstrap_room=int(payload["bootstrap_room"]))


def prefill(req, payload):
    """D-worker-owned P HTTP request survives a gateway restart."""
    conn = None
    try:
        url = urlparse(payload["prefill_url"])
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("native PD prefill requires an HTTP(S) URL")
        body = dict(
            rid="prefill-" + payload["rid"],
            bootstrap_host=payload["bootstrap_host"],
            bootstrap_port=payload["bootstrap_port"],
            bootstrap_room=payload["bootstrap_room"],
            stream=False,
        )
        path = url.path.rstrip("/") or "/generate"
        if path == "/v1/completions":
            body.update(prompt=payload["input_ids"], max_tokens=1, temperature=0.0)
        else:
            body.update(
                input_ids=payload["input_ids"],
                sampling_params=dict(
                    max_new_tokens=1,
                    temperature=0.0,
                    ignore_eos=True,
                ),
            )
        cls = (
            http.client.HTTPSConnection
            if url.scheme == "https"
            else http.client.HTTPConnection
        )
        conn = cls(url.hostname, url.port, timeout=180)
        conn.request(
            "POST", path, json.dumps(body), {"Content-Type": "application/json"}
        )
        response = conn.getresponse()
        response.read()
        if response.status != 200:
            raise RuntimeError(f"native P prefill HTTP {response.status}")
    except Exception as exc:
        # Keep native receiver ownership; cancellation must use its drain path.
        req._d2d_submission_error = repr(exc)
    finally:
        if conn is not None:
            conn.close()
