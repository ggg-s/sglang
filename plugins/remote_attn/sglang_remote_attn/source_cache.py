"""P-only radix read leases; ordinary requests retain the native cache path.

Pin the cached prefix independently. Keep the suffix request-owned by delaying
checkpoint/dedup/release until transfers and Remote readers drain across TP.
"""

from dataclasses import dataclass


def cache_signature(cache):
    if cache is None:
        return None
    return (
        type(cache).__module__,
        type(cache).__qualname__,
        cache.disable,
        tuple(c.name for c in getattr(cache, "tree_components", ())),
        getattr(cache, "cache_controller", None) is not None,
        getattr(cache, "linker", None) is not None,
        getattr(cache, "enable_session_radix_cache", False),
    )


@dataclass
class ReadLease:
    guard: object
    req: object
    handle: object
    lock: object
    cached_tokens: int
    checkpoints: int = 0
    pending_checkpoint: tuple | None = None
    published_checkpoint: tuple | None = None
    replayed_checkpoint: bool = False
    published_tokens: int = 0
    publication_declined: bool = False
    finish_call: object = None
    abort: bool = False
    retiring: bool = False
    failed: bool = False
    closed: bool = False

    def check(self):
        if self.failed or self.closed or self.req.cache_request_handle != self.handle:
            raise RuntimeError(
                "P radix read lease changed attempt or is no longer usable"
            )

    def defer_checkpoint(self, up_to, kwargs=None):
        self.check()
        if self.pending_checkpoint is None or up_to >= self.pending_checkpoint[0]:
            self.pending_checkpoint = (up_to, dict(kwargs or {}))

    def retire(self, release):
        self.check()
        self.retiring = True
        try:
            # Publish before the callback can release/reuse the request row.
            # All callers must already hold the GPU/IO/Remote retirement proof.
            # MAMBA's mutable state now describes the final frontier. Native
            # final release publishes/donates it once; replaying an earlier
            # deferred frontier would attach the wrong state to that prefix.
            if (
                self.pending_checkpoint is not None
                and not self.abort
                and not getattr(self.guard, "allow_mamba", False)
            ):
                up_to, kwargs = self.pending_checkpoint
                if not getattr(self.req, "skip_radix_cache_insert", False):
                    from .swa_checkpoints import replay

                    replay(self, up_to)
                    self.guard._checkpoint(self.req, up_to=up_to, **kwargs)
                    self.replayed_checkpoint = not self.publication_declined
            release()
            if self.finish_call is not None:
                self.finish_call()
            self.guard.cache.unlock(self.lock)
        except BaseException:
            # Checkpoint/release may already have changed allocator state.
            # Retain the pin and reject retries after unknown completion.
            self.failed = True
            raise
        finally:
            self.retiring = False
        self.lock = None
        self.closed = True
        del self.guard.leases[self.handle]
        del self.guard.by_req[id(self.req)]


class CacheGuard:
    def __init__(self, cache, *, allow_swa=False, scheduler=None, allow_mamba=False):
        self.allow_swa = allow_swa
        self.allow_mamba = allow_mamba
        self.scheduler = scheduler
        self.cache = cache
        self.leases = {}
        self.by_req = {}
        self.checkpoint = cache.checkpoint
        self.finish = cache.finish
        self.reset = cache.reset
        # Instance-local hooks also cover inherited BasePrefixCache.finish.
        cache.checkpoint = self._checkpoint
        cache.finish = self._finish
        cache.reset = self._reset

    def hold(self, req):
        handle = req.cache_request_handle
        if handle in self.leases or id(req) in self.by_req:
            raise RuntimeError("P radix request already has a read lease")
        if req.kv.cache_protected_len and req.lock is None:
            raise RuntimeError("P radix matched prefix has no native request lock")
        pin = self.cache.lock(req.last_node)
        if pin is None:
            raise RuntimeError("P radix cache did not return a read pin")
        lease = ReadLease(self, req, handle, pin, int(req.kv.cache_protected_len))
        self.leases[handle] = lease
        self.by_req[id(req)] = lease
        return lease

    def _checkpoint(self, req, *, up_to, **kwargs):
        lease = self.by_req.get(id(req)) or self.leases.get(req.cache_request_handle)
        if lease is not None:
            if lease.req is not req:
                raise RuntimeError(
                    "P radix checkpoint belongs to a different request object"
                )
            lease.check()
            if not lease.retiring:
                self._evict_swa(req, up_to)
                lease.checkpoints += 1
                lease.defer_checkpoint(up_to, kwargs)
                return
            if lease.abort or lease.publication_declined:
                return
            if lease.published_checkpoint == (up_to, kwargs):
                return  # Native final release may request the same checkpoint.
            if self.allow_swa:
                # The deferred intermediate branch may no longer have SWA.
                # At retirement publish the completed frontier and its live
                # tail, rather than truncating back to that obsolete branch.
                branch = getattr(req, "swa_branching_seqlen", None)
                if branch is not None and branch <= up_to:
                    req.swa_branching_seqlen = None
            result = self.checkpoint(req, up_to=up_to, **kwargs)
            lease.published_tokens = int(req.kv.cache_protected_len)
            lease.published_checkpoint = (up_to, dict(kwargs))
            lease.pending_checkpoint = None
            return result
        return self.checkpoint(req, up_to=up_to, **kwargs)

    def _evict_swa(self, req, up_to):
        scheduler = self.scheduler
        if (
            not self.allow_swa
            or scheduler is None
            or not self.cache.supports_swa()
            or req.finished()
        ):
            return
        # Only request-owned, out-of-window SWA is released. FULL/indexer
        # storage and the cached-prefix pin stay live for DMA/Remote readers.
        # Clamp to the prompt so even a repeated/late checkpoint cannot reclaim
        # the final SWA payload while its independent transfer is in flight.
        pre_len = min(up_to, len(req.origin_input_ids)) - 1
        # A scheduler checkpoint is NOT proof of GPU completion under overlap.
        # The core WAR fastpath may fence only metadata reads. Fence the entire
        # already-enqueued forward before clearing SWA mappings/reusing pages;
        # the next forward already waits on schedule_stream. No CPU sync, and
        # no dependency on a future forward (which would create a stream cycle).
        with scheduler.device_module.stream(scheduler.schedule_stream):
            scheduler.schedule_stream.wait_stream(scheduler.forward_stream)
            from .swa_checkpoints import retain

            retain(self, req, min(up_to, len(req.origin_input_ids)))
            from sglang.srt.mem_cache.allocation import retire_window

            retire_window(req, pre_len, self.cache)

    def _finish(self, handle, outcome):
        lease = self.leases.get(handle)
        if lease is None:
            return self.finish(handle, outcome)
        lease.check()
        if getattr(outcome, "name", None) == "ABORT":
            lease.abort = True
        # A late SUCCESS cannot overwrite an earlier ABORT.
        if lease.finish_call is None or getattr(outcome, "name", None) == "ABORT":
            lease.finish_call = lambda: self.finish(handle, outcome)

    def _reset(self):
        if self.leases:
            raise RuntimeError("P radix read leases are live; cannot reset cache")
        return self.reset()


def install(cache, *, allow_swa=False, scheduler=None, allow_mamba=False):
    if cache.disable:
        return None
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if type(cache) not in (RadixCache, UnifiedRadixCache):
        raise ValueError("P radix Remote reads require a built-in dense cache")
    if (allow_swa or allow_mamba) and type(cache) is not UnifiedRadixCache:
        raise ValueError("stateful P prefix cache requires native UnifiedRadixCache")
    supported = (
        ((ComponentType.FULL,), (ComponentType.FULL, ComponentType.SWA))
        if allow_swa
        else ((ComponentType.FULL,),)
    )
    if allow_mamba:
        supported = ((ComponentType.FULL, ComponentType.MAMBA),)
    if type(cache) is UnifiedRadixCache and (
        cache.tree_components not in supported
        or cache.cache_controller is not None
        or cache.linker is not None
        or cache.enable_session_radix_cache
    ):
        raise ValueError(
            "P radix reads support device-only FULL"
            + ("/MAMBA" if allow_mamba else "/SWA" if allow_swa else "")
            + " cache without session refs"
        )
    if hasattr(cache, "_remote_attn_source_cache"):
        return cache._remote_attn_source_cache
    guard = CacheGuard(
        cache, allow_swa=allow_swa, scheduler=scheduler, allow_mamba=allow_mamba
    )
    cache._remote_attn_source_cache = guard
    return guard


def lease_for(req):
    source = getattr(req, "_pre_layerwise_source", None)
    lease = getattr(source, "cache_lease", None)
    if lease is None:
        lease = getattr(getattr(req, "_dsv41_p_lease", None), "cache_lease", None)
    if lease is None:
        lease = getattr(getattr(req, "_dsv41_p_certificate", None), "cache_lease", None)
    return lease


def checkpoint_allowed(req, requested):
    lease = lease_for(req)
    return requested and not (lease is not None and lease.abort)
