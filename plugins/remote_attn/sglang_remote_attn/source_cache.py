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

    def retire(self, release):
        self.check()
        self.retiring = True
        try:
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
    def __init__(self, cache):
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
                lease.checkpoints += 1
                return
        return self.checkpoint(req, up_to=up_to, **kwargs)

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


def install(cache, *, allow_swa=False):
    if cache.disable:
        return None
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if type(cache) not in (RadixCache, UnifiedRadixCache):
        raise ValueError("P radix Remote reads require a built-in dense cache")
    if allow_swa and type(cache) is not UnifiedRadixCache:
        raise ValueError("DSV4.1 P prefix cache requires native UnifiedRadixCache")
    supported = (
        ((ComponentType.FULL,), (ComponentType.FULL, ComponentType.SWA))
        if allow_swa
        else ((ComponentType.FULL,),)
    )
    if type(cache) is UnifiedRadixCache and (
        cache.tree_components not in supported
        or cache.cache_controller is not None
        or cache.linker is not None
        or cache.enable_session_radix_cache
    ):
        raise ValueError(
            "P radix reads support device-only FULL"
            + ("/SWA" if allow_swa else "")
            + " cache without session refs"
        )
    if hasattr(cache, "_remote_attn_source_cache"):
        return cache._remote_attn_source_cache
    guard = CacheGuard(cache)
    cache._remote_attn_source_cache = guard
    return guard


def checkpoint_allowed(req, requested):
    source = getattr(req, "_pre_layerwise_source", None)
    lease = getattr(source, "cache_lease", None)
    if lease is None:
        lease = getattr(getattr(req, "_dsv41_p_lease", None), "cache_lease", None)
    return requested and not (lease is not None and lease.abort)
