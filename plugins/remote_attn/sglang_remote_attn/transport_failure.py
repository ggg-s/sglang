"""Request-scoped transport failure without changing the configured mode.

Quarantine is entered only on a TP-agreed scheduler boundary. It is not a
retirement proof: no RPC context, cache pin, ring allocation, metadata row or
deferred free callback belonging to the failed request is released here.
"""

import logging

logger = logging.getLogger(__name__)


class TransportFailure(RuntimeError):
    pass


def request_of(item):
    if hasattr(item, "chunk"):
        return item.chunk.req
    if hasattr(item, "decode_req"):
        return item.decode_req.req
    return item.req


def quarantine(owner, item, reason):
    # SourceOwner wraps the P overlap controller for ordered retirement.
    controller = getattr(owner, "controller", owner)
    req = request_of(item)
    records = getattr(controller, "transport_quarantine", None)
    if records is None:
        records = controller.transport_quarantine = {}
    record = getattr(req, "_remote_attn_transport_quarantine", None)
    if record is None:
        record = dict(req=req, items=[], releases=[])
    records[id(req)] = record
    req._remote_attn_transport_quarantine = record
    if not any(x is item for x in record["items"]):
        record["items"].append(item)
    # Other requests retain their own resources and configured transfer mode.
    for name in ("sources", "targets"):
        active = getattr(controller, name, [])
        for candidate in list(active):
            if request_of(candidate) is req:
                record["items"].append(candidate)
                active.remove(candidate)
    overlap = getattr(controller, "prefill_overlap", None)
    if overlap is not None:
        for lease in list(overlap.leases):
            if request_of(lease) is req:
                record["items"].append(lease)
                overlap.leases.remove(lease)
    target = getattr(req, "_pre_layerwise_target", None) or getattr(
        req, "_pre_layerwise_native_target", None
    )
    if target is not None:
        record["items"].append(target)
        # Do not pass this receiver to native timeout cleanup: that would free
        # its metadata row and ring lease while an old write may still arrive.
        target.queue.queue = [
            dr for dr in target.queue.queue if dr is not target.decode_req
        ]
    # Keep the original object as well as all callbacks/engine registrations.
    # It may still be referenced by an already-launched GPU batch or RPC.
    from sglang.srt.disaggregation.utils import prepare_abort

    if not req.finished():
        prepare_abort(
            req,
            f"KV transport failed; request storage quarantined: {reason}",
            status_code=503,
        )
        if not any(getattr(x, "responded", False) for x in record["items"]):
            controller.scheduler.output_streamer.stream_output(
                [req], req.return_logprob, None
            )
    # Stop future chunk scheduling through the native ordered abort boundary.
    # Quarantine may be entered after prepare_abort already marked the request.
    scheduler = getattr(controller, "scheduler", None)
    if (
        scheduler is not None
        and getattr(scheduler, "chunked_req", None) is req
        and getattr(scheduler, "_pending_chunked_abort_req", None) is None
    ):
        scheduler._pending_chunked_abort_req = req
    # Retirement and admission commands are serialized on the TP scheduler.
    # A quarantined certificate must not subsequently enter revoke/drain:
    # unknown IO cannot produce that drain proof. Preserve its peer/cache pins.
    certificate = getattr(req, "_dsv41_p_certificate", None)
    if certificate is not None:
        certificate.done = True
        if not any(x is certificate for x in record["items"]):
            record["items"].append(certificate)
    logger.error(
        "Remote-attn request %s quarantined; transfer mode unchanged: %s",
        req.rid,
        reason,
    )


def hold_release(req, release):
    record = getattr(req, "_remote_attn_transport_quarantine", None)
    if record is None:
        return False
    record["releases"].append(release)
    return True


def retain_release(req, item, release):
    """Hold a rank-local callback; active-list changes still await TP agreement."""
    if getattr(req, "_remote_attn_transport_quarantine", None) is None:
        req._remote_attn_transport_quarantine = dict(req=req, items=[item], releases=[])
    hold_release(req, release)
