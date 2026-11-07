"""Freeze an admitted native PD target before its immutable entries are Local.

P->D0 retains its receiver/ring/RPC leases. The ordinary ordered retirement
protocol drains them while D0 stops decoding this request; other requests run.
Only then may D0 publish a D2D manifest or expose bytes to D1.
"""

from .transport_failure import TransportFailure


def can_freeze(req):
    from .dsv41_progressive import ProgressiveTarget

    target = getattr(req, "_pre_layerwise_target", None)
    rpc = getattr(req, "_dsv41_remote", None)
    if not isinstance(target, ProgressiveTarget) or target.closed:
        return False
    if (
        not target.native_staging
        or target.release is not None
        or target.release_failed
        or target.cache_lease is not None
        or rpc is not target.rpc
        or not rpc.progressive
        or rpc.closed
        or rpc.controller is not target.owner
        or target.decode_req.req is not req
    ):
        return False
    protocol = target.owner.decode_overlap.retirement
    entry = protocol.registered.get(target.admission_id)
    if entry is None or entry[0] is not target or target not in target.owner.targets:
        return False
    # Early admission already installed indexer/SWA/carry/metadata. Check the
    # still-live attempt and transport, without waiting for global entries.
    target.refresh()
    return True


def prepare(controller, source):
    """An idempotent TP command: report pending, or publish a fully drained source."""
    error = None
    ready = False
    try:
        source.check_read()
        req = source.req
        if req.finished() or source.release is not None:
            raise TransportFailure("frozen source requested release before D2D export")
        if (
            req.kv.kv_committed_len != source.prefix_len
            or tuple(req.output_ids) != source.frozen_output
        ):
            raise TransportFailure("frozen source token history changed")
        previous = source.previous_target
        if previous is not None and not previous.closed:
            previous.refresh()
        ready = previous is None or previous.closed
    except Exception as exc:
        error = repr(exc)
    votes = controller.agree([int(error is None), int(ready and error is None)])
    if not votes[0]:
        source.error = f"TP previous PD lifetime failed: {error or 'peer failed'}"
        raise TransportFailure(source.error)
    if not votes[1]:
        return dict(
            frozen=True,
            pending_retirement=True,
            source_partial_remote=source.partial_remote,
            source_remote_entries=source.remote_entries_at_freeze,
        )
    try:
        # A previous native target closes its RPC before clearing receiver and
        # metadata. This also handles an already-Local compatibility RPC.
        rpc = source.previous_rpc
        if rpc is not None and not rpc.closed:
            rpc.close("d2d-frozen-source")
    except Exception as exc:
        error = repr(exc)
    if not controller.agree([int(error is None)])[0]:
        source.error = f"TP previous RPC retirement failed: {error or 'peer failed'}"
        raise TransportFailure(source.error)
    try:
        source.initialize()
    except Exception as exc:
        error = repr(exc)
    if not controller.agree([int(error is None)])[0]:
        source.error = f"TP D2D source preparation failed: {error or 'peer failed'}"
        raise TransportFailure(source.error)
    return dict(transfer=source.meta)
