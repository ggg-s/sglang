"""Native source metadata and external KV ownership across deferred retirement."""

from dataclasses import dataclass


@dataclass
class MetadataHold:
    req: object
    handle: object
    allocator: object
    index: int
    original: object
    closed: bool = False
    failed: bool = False

    def release(self):
        if self.closed:
            return
        if self.failed:
            raise RuntimeError("source metadata release failed; refusing retry")
        if (
            self.req.cache_request_handle != self.handle
            or self.req.metadata_buffer_index != self.index
        ):
            raise RuntimeError("source metadata ownership changed before retirement")
        # The native callback may mutate the allocator before raising.
        self.failed = True
        self.original(self.req, self.allocator)
        self.closed, self.failed = True, False


def release_metadata(original, req, allocator):
    """Also patches imported native aliases through HookRegistry propagation."""
    hold = getattr(req, "_remote_attn_metadata_hold", None)
    if hold is not None and hold.closed:
        if req.metadata_buffer_index < 0:
            return original(req, allocator)
        # A retracted Req object can acquire a new metadata slot/attempt.
        req._remote_attn_metadata_hold = hold = None
    if req.metadata_buffer_index < 0:
        return original(req, allocator)
    from .source_cache import lease_for

    cache = lease_for(req)
    source = getattr(req, "_pre_layerwise_source", None)
    overlap = getattr(req, "_dsv41_p_lease", None)
    unsettled = (
        getattr(req, "_remote_attn_transport_quarantine", None) is not None
        or (cache is not None and not cache.closed)
        or (source is not None and not source.released)
        or (overlap is not None and not overlap.closed)
    )
    if unsettled:
        if hold is None:
            req._remote_attn_metadata_hold = MetadataHold(
                req,
                req.cache_request_handle,
                allocator,
                req.metadata_buffer_index,
                original,
            )
        elif hold.allocator is not allocator or hold.index != req.metadata_buffer_index:
            raise RuntimeError("source metadata release changed allocator or slot")
        return
    if hold is not None:
        return hold.release()
    return original(req, allocator)


def retire_metadata(req):
    # Called only at the existing TP-agreed GPU/IO/Remote retirement boundary.
    # A failed room never supplies this proof, even if its GPU has completed.
    if getattr(req, "_remote_attn_transport_quarantine", None) is not None:
        return
    hold = getattr(req, "_remote_attn_metadata_hold", None)
    if hold is not None:
        hold.release()


def owned_requests(controller):
    """Snapshot native Req owners; the checker deduplicates KV records."""
    reqs = []
    guard = getattr(controller, "cache_guard", None)
    if guard is not None:
        reqs.extend(lease.req for lease in tuple(guard.leases.values()))
    reqs.extend(source.req for source in tuple(controller.sources))
    overlap = getattr(controller, "prefill_overlap", None)
    if overlap is not None:
        reqs.extend(lease.chunk.req for lease in tuple(overlap.leases))
    reqs.extend(target.decode_req.req for target in tuple(controller.targets))
    reqs.extend(
        record["req"]
        for record in tuple(getattr(controller, "transport_quarantine", {}).values())
    )
    return reqs


def install_accounting(controller):
    checker = controller.scheduler.invariant_checker
    previous = getattr(checker, "get_external_kv_owners", lambda: ())
    checker.get_external_kv_owners = lambda: [*previous(), *owned_requests(controller)]
