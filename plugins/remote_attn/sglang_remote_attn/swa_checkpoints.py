"""Bounded, request-owned SWA windows for deferred source checkpoints.

No FULL row is published/rebound here until the source retirement proof holds.
Core eviction, accounting and release understand the retained SWA exceptions.
"""


def retain(guard, req, up_to):
    # Called on schedule_stream AFTER its dependency on all issued forwards.
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    cache = guard.cache
    swa = getattr(cache, "components", {}).get(ComponentType.SWA)
    if swa is None or not swa.gates_prefix_match:
        return
    page = cache.page_size
    window = (max(swa.sliding_window_size, page) + page - 1) // page * page
    end = (up_to - int(cache.tree_core.is_eagle)) // page * page
    kv = req.kv
    # Keep an extra page: replay caps N by W before page alignment. An N=P*k+4
    # prompt must still have a window at P*(k-1), not just at P*k.
    start = max(
        0,
        end - window - page,
        kv.cache_protected_len,
        kv.get_evicted_seqlen(ComponentType.SWA),
    )
    ranges = kv.swa_retained_ranges
    if start < end:
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(end, ranges[-1][1]))
        else:
            ranges.append((start, end))

    # A single global budget across source leases, not one budget per request.
    # Drop older windows under pressure; retained checkpoints are a cache,
    # never a reason to hold the whole prefix or exhaust the sliding pool.
    allocator = cache.token_to_kv_pool_allocator
    budget = allocator.swa_attn_allocator.size // (4 * page) * page
    leases = [
        lease
        for lease in guard.leases.values()
        if getattr(lease.req.kv, "swa_retained_ranges", ())
    ]
    total = sum(
        hi - lo for lease in leases for lo, hi in lease.req.kv.swa_retained_ranges
    )
    for lease in leases:
        # A budget is not a retirement proof. Unknown completion/partial
        # insertion must never become reclaimable through another request.
        if (
            lease.failed
            or lease.closed
            or getattr(lease.req, "_remote_attn_transport_quarantine", None) is not None
            or getattr(lease.req, "_handoff_transfer_unsafe", False)
        ):
            continue
        owner = lease.req.kv
        while total > budget and owner.swa_retained_ranges:
            lo, hi = owner.swa_retained_ranges[0]
            cut = min(hi, lo + total - budget)
            # Only the portion already passed by native eviction is an extra
            # allocation. The current forward/final DMA tail stays native-owned.
            free_lo = max(lo, owner.swa_dead_lo(page))
            free_hi = min(cut, owner.get_evicted_seqlen(ComponentType.SWA))
            if free_lo < free_hi:
                row = cache.req_to_token_pool.req_to_token[owner.req_pool_idx]
                allocator.free_swa_segment(row[free_lo:free_hi], start_pos=free_lo)
            total -= cut - lo
            if cut == hi:
                owner.swa_retained_ranges.pop(0)
            else:
                owner.swa_retained_ranges[0] = (cut, hi)


def replay(lease, up_to):
    """Publish retained windows in order, only after GPU/IO/Remote retirement.

    Each native insert sees one contiguous live tail. Restore the real high
    water mark even on a partial failure so quarantine accounting stays valid.
    """
    req, cache = lease.req, lease.guard.cache
    ranges = list(getattr(req.kv, "swa_retained_ranges", ()))
    if not ranges:
        return
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    kv = req.kv
    original = kv.get_evicted_seqlen(ComponentType.SWA)
    try:
        for lo, hi in ranges:
            end = hi + int(cache.tree_core.is_eagle)
            if end > up_to or hi <= kv.cache_protected_len:
                continue
            kv.set_evicted_seqlen(ComponentType.SWA, max(lo, kv.cache_protected_len))
            lease.guard._checkpoint(req, up_to=end)
            if kv.cache_protected_len < hi:
                # Native insertion may decline a rotation-conflicting path.
                # Leave the suffix request-owned for normal release, without
                # attempting a final insert that would hide its retained holes.
                lease.publication_declined = True
                break
            kv.swa_retained_ranges = [
                (max(a, kv.cache_protected_len), b)
                for a, b in kv.swa_retained_ranges
                if b > kv.cache_protected_len
            ]
    finally:
        kv.set_evicted_seqlen(
            ComponentType.SWA, max(original, kv.get_evicted_seqlen(ComponentType.SWA))
        )
