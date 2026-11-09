"""D cache publication barrier over native locks, checkpoints and incremental PD.

Only early-admitted requests acquire a lease. Ordinary native PD and retired
Local requests keep their normal cache path. No readiness polling or collective
is added to the decode steady state.
"""

import hashlib
import json
from dataclasses import dataclass


def key(req):
    return hashlib.sha256(
        json.dumps(
            [
                list(req.origin_input_ids),
                req.extra_key,
                getattr(req, "cache_salt", None),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()


def lease_for(req):
    return getattr(req, "_remote_attn_destination_cache", None)


def checkpoint_allowed(req, requested):
    aborted = getattr(req, "_remote_attn_cache_abort_handle", None)
    return requested and (
        aborted is None or aborted != getattr(req, "cache_request_handle", None)
    )


def intent(target):
    lease = getattr(target, "cache_lease", None)
    return lease.intent() if isinstance(lease, Lease) else None


def suppress(req):
    lease = lease_for(req)
    if lease is not None:
        lease.check()
        lease.abort = True


def cancel_pending(req):
    match = getattr(req, "_remote_attn_pending_cache", None)
    if match is not None:
        match.guard.drop_match(req)


@dataclass
class Match:
    guard: object
    req: object
    handle: object
    result: object
    pin: object
    digest: str
    attrs: dict
    scope: tuple

    def reuse(self):
        req = self.req
        if (
            req.cache_request_handle != self.handle
            or req.kv.req_pool_idx is not None
            or self.scope
            != (
                req.extra_key,
                getattr(req, "cache_salt", None),
                len(req.origin_input_ids),
            )
        ):
            raise RuntimeError("pending D cache match changed attempt or scope")
        for name, value in self.attrs.items():
            setattr(req, name, value)
        if req.lock is None:
            req.lock = self.guard.cache.lock(req.last_node)
        req._remote_attn_decode_match = (
            self.handle,
            self.result.l1_prefix_len,
            self.digest,
            req.lock,
        )
        return self.result


@dataclass
class Lease:
    guard: object
    req: object
    handle: object
    row: int
    prefix: int
    native_lock: object
    pin: object
    digest: str
    pending: int | None = None
    finish_outcome: object = None
    abort: bool = False
    retiring: bool = False
    failed: bool = False
    closed: bool = False

    def check(self):
        if (
            self.closed
            or self.failed
            or self.req.cache_request_handle != self.handle
            or self.req.kv.req_pool_idx != self.row
            or self.req.kv.cache_protected_len != self.prefix
            or self.req.lock is not self.native_lock
        ):
            raise RuntimeError(
                "D cache lease changed attempt, row or prefix; retaining pages"
            )

    def intent(self):
        self.check()
        return [
            self.prefix,
            self.digest,
            self.pending,
            self.abort,
            getattr(self.finish_outcome, "name", None),
        ]

    def retire(self, release):
        self.check()
        if self.abort:
            self.req._remote_attn_cache_abort_handle = self.handle
        self.retiring = True
        try:
            if release is not None:
                # Native release recomputes the valid KV length and owns its
                # checkpoint/free. Never replay a second checkpoint after free.
                release()
            elif self.pending is not None and not self.abort:
                self.req.refresh_fill_ids()
                self.guard.checkpoint(
                    self.req, up_to=min(self.pending, self.req.owned_kv_len())
                )
            if self.finish_outcome is not None:
                self.guard.finish(self.handle, self.finish_outcome)
            self.guard.cache.unlock(self.pin)
        except BaseException:
            # Native checkpoint/dedup can already have changed ownership.
            # Keep the extra pin and disallow retry after partial failure.
            self.failed = True
            raise
        finally:
            self.retiring = False
        self.closed = True
        self.pin = None
        self.guard.leases.pop(self.handle)
        del self.req._remote_attn_destination_cache


class Guard:
    def __init__(self, cache, *, page_size=1, skip_lock_components=()):
        self.cache = cache
        self.page_size = page_size
        self.skip_lock_components = skip_lock_components
        self.leases = {}
        self.pending = {}
        self.poisoned = False
        self.checkpoint, self.finish, self.reset = (
            cache.checkpoint,
            cache.finish,
            cache.reset,
        )
        cache.checkpoint, cache.finish, cache.reset = (
            self._checkpoint,
            self._finish,
            self._reset,
        )

    def pin(self, node):
        if self.skip_lock_components:
            return self.cache.lock(node, skip_lock_components=self.skip_lock_components)
        return self.cache.lock(node)

    def check(self):
        if self.poisoned:
            raise RuntimeError(
                "D cache publication failed across TP; cache is quarantined"
            )

    def hold(self, req):
        self.check()
        if getattr(req, "session_id", None) is not None:
            raise ValueError("D progressive cache does not support sessions")
        stamp = getattr(req, "_remote_attn_decode_match", None)
        prefix = int(req.kv.cache_protected_len)
        if (
            stamp is None
            or stamp[:3] != (req.cache_request_handle, prefix, key(req))
            or stamp[3] is not req.lock
            or req.kv.req_pool_idx is None
        ):
            raise RuntimeError("D cache match was not certified before PD metadata")
        if req.cache_request_handle in self.leases or lease_for(req) is not None:
            raise RuntimeError("duplicate D cache lease")
        if prefix and req.lock is None:
            raise RuntimeError("D cached prefix has no native lock")
        self.drop_match(req)
        pin = self.pin(req.last_node)
        if pin is None:
            raise RuntimeError("D cache did not return an independent prefix pin")
        lease = Lease(
            self,
            req,
            req.cache_request_handle,
            req.kv.req_pool_idx,
            prefix,
            req.lock,
            pin,
            stamp[2],
            pending=len(req.origin_input_ids),
        )
        self.leases[lease.handle] = lease
        req._remote_attn_destination_cache = lease
        req._remote_attn_decode_cache_enabled = True
        return lease

    def _checkpoint(self, req, *, up_to, **kwargs):
        self.check()
        lease = lease_for(req)
        if lease is None:
            if not checkpoint_allowed(req, True):
                return
            return self.checkpoint(req, up_to=up_to, **kwargs)
        if lease.retiring:
            if not lease.abort:
                return self.checkpoint(req, up_to=up_to, **kwargs)
            return
        lease.check()
        if kwargs:
            raise ValueError(
                "D progressive cache supports native up_to checkpoints only"
            )
        if not lease.abort:
            lease.pending = max(int(up_to), lease.pending or 0)

    def _finish(self, handle, outcome):
        self.check()
        lease = self.leases.get(handle)
        if lease is None:
            match = self.pending.get(handle)
            if match is not None:
                self.drop_match(match.req)
            return self.finish(handle, outcome)
        if not lease.retiring:
            lease.check()
        if getattr(outcome, "name", None) == "ABORT":
            lease.abort = True
        if lease.finish_outcome is None or getattr(outcome, "name", None) == "ABORT":
            lease.finish_outcome = outcome

    def _reset(self):
        self.check()
        if self.leases or self.pending:
            raise RuntimeError("D cache leases are live; cannot reset")
        return self.reset()

    def drop_match(self, req):
        match = getattr(req, "_remote_attn_pending_cache", None)
        if match is None:
            return
        if match.req is not req or match.handle != req.cache_request_handle:
            raise RuntimeError("pending D cache match belongs to another attempt")
        self.cache.unlock(match.pin)
        self.pending.pop(match.handle)
        del req._remote_attn_pending_cache
        if req.kv.req_pool_idx is None and req.lock is not None:
            self.cache.unlock(req.lock)
            req.lock = None


def install(cache, *, allow_swa=False):
    if cache.disable:
        return None
    from sglang.srt.mem_cache.radix_cache import RadixCache
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    supported = ((ComponentType.FULL,),)
    if allow_swa:
        supported += ((ComponentType.FULL, ComponentType.SWA),)
    if (
        (allow_swa and type(cache) is not UnifiedRadixCache)
        or type(cache) not in (RadixCache, UnifiedRadixCache)
        or (
            type(cache) is UnifiedRadixCache
            and (
                cache.tree_components not in supported
                or cache.cache_controller is not None
                or cache.linker is not None
                or cache.enable_session_radix_cache
            )
        )
    ):
        raise ValueError(
            "D cache requires native device-only FULL"
            + ("/SWA UnifiedRadixCache" if allow_swa else " cache")
            + " without session refs"
        )
    if not hasattr(cache, "_remote_attn_destination_guard"):
        cache._remote_attn_destination_guard = Guard(
            cache,
            page_size=cache.page_size if allow_swa else 1,
            skip_lock_components=(ComponentType.SWA,) if allow_swa else (),
        )
    return cache._remote_attn_destination_guard


def match_prefix(original, queue, req):
    """One pre-metadata consensus in the native TP-symmetric admission order."""
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    guard = getattr(controller, "destination_cache", None)
    if guard is None:
        return original(queue, req)
    pending = getattr(req, "_remote_attn_pending_cache", None)
    if pending is not None:
        guard.check()
        return pending.reuse()
    previous_lock = getattr(req, "lock", None)
    result = error = pin = None
    try:
        guard.check()
        result = original(queue, req)
        prefix = int(result.l1_prefix_len)
        if (
            prefix % guard.page_size
            or prefix != result.decode_prefix_len
            or not 0 <= prefix <= len(req.origin_input_ids)
        ):
            raise RuntimeError("D cache supports device-only prefix matches")
        pin = guard.pin(req.last_node)
        if pin is None:
            raise RuntimeError("D pending cache match has no native pin")
        report = (
            str(req.rid),
            str(req.bootstrap_room),
            prefix,
            key(req),
            getattr(req.cache_request_handle, "attempt_id", None),
        )
    except Exception as exc:
        error = exc
        report = None
    reports = [report]
    if controller.tp_size > 1:
        import torch.distributed as dist

        reports = [None] * controller.tp_size
        dist.all_gather_object(reports, report, group=queue.scheduler.attn_tp_cpu_group)
    if report is None or any(value != report for value in reports):
        if pin is not None:
            guard.cache.unlock(pin)
        lock = getattr(req, "lock", None)
        if lock is not previous_lock:
            guard.cache.unlock(lock)
            req.lock = None
        raise RuntimeError(
            "D cache TP prefix match differs before PD metadata"
        ) from error
    req._remote_attn_decode_match = (
        req.cache_request_handle,
        report[2],
        report[3],
        req.lock,
    )
    req._remote_attn_decode_cache_enabled = True
    pending = Match(
        guard,
        req,
        req.cache_request_handle,
        result,
        pin,
        report[3],
        {
            name: getattr(req, name)
            for name in (
                "prefix_len",
                "last_node",
                "last_host_node",
                "best_match_node",
                "host_hit_length",
                "swa_host_hit_length",
                "mamba_host_hit_length",
                "num_matched_prefix_tokens",
            )
            if hasattr(req, name)
        },
        (req.extra_key, getattr(req, "cache_salt", None), len(req.origin_input_ids)),
    )
    guard.pending[pending.handle] = pending
    req._remote_attn_pending_cache = pending
    return result


def send_metadata(
    original,
    queue,
    decode_req,
    kv_indices,
    page_size,
    state_indices=None,
    metadata_buffer_preassigned=False,
    non_blocking=False,
    **metadata_kwargs,
):
    """Allocation succeeded: the native request lock now owns prefix lifetime."""
    from .qwen_split_prefill import advertise

    advertise(queue, decode_req, metadata_kwargs.get("decode_prefix_len"))
    match = getattr(decode_req.req, "_remote_attn_pending_cache", None)
    if match is None:
        return original(
            queue,
            decode_req,
            kv_indices,
            page_size,
            state_indices,
            metadata_buffer_preassigned=metadata_buffer_preassigned,
            non_blocking=non_blocking,
            **metadata_kwargs,
        )
    req = decode_req.req
    if (
        metadata_kwargs.get("decode_prefix_len", 0) != match.result.l1_prefix_len
        or req.kv.cache_protected_len != match.result.l1_prefix_len
        or req.lock is None
    ):
        raise RuntimeError("D cache prefix changed before native PD metadata")
    from .dsv41_receive import cache_metadata

    state_indices = cache_metadata(queue, decode_req, state_indices)
    try:
        return original(
            queue,
            decode_req,
            kv_indices,
            page_size,
            state_indices,
            metadata_buffer_preassigned=metadata_buffer_preassigned,
            non_blocking=non_blocking,
            **metadata_kwargs,
        )
    finally:
        match.guard.drop_match(req)
