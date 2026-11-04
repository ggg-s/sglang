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
    # Other owners (including locally quarantined ones) reserve their recorded
    # budget identically on every rank. Reclaim ONLY this checkpoint's owner:
    # choosing a different victim from rank-local failure flags diverges both
    # allocator state and the subsequent radix tree. No per-chunk TP vote.
    reserved = sum(
        hi - lo
        for lease in guard.leases.values()
        if lease.req is not req
        for lo, hi in getattr(lease.req.kv, "swa_retained_ranges", ())
    )
    budget = max(0, budget - reserved)
    total = sum(hi - lo for lo, hi in ranges)
    while total > budget and ranges:
        lo, hi = ranges[0]
        cut = min(hi, lo + total - budget)
        # The portion beyond the cursor is still the native live window.
        # Removing its retention mark does NOT mean that its pages were freed.
        free_lo = max(lo, kv.swa_dead_lo(page))
        free_hi = min(cut, kv.get_evicted_seqlen(ComponentType.SWA))
        if free_lo < free_hi:
            row = cache.req_to_token_pool.req_to_token[kv.req_pool_idx]
            allocator.free_swa_segment(row[free_lo:free_hi], start_pos=free_lo)
        total -= cut - lo
        if cut == hi:
            ranges.pop(0)
        else:
            ranges[0] = (cut, hi)


def replay(lease, up_to):
    """Publish actual live intervals after GPU/IO/Remote retirement.

    The real eviction cursor stays unchanged. Native insertion validates that
    each explicit frontier accounts for all private SWA it takes over.
    """
    req, cache = lease.req, lease.guard.cache
    kv = req.kv
    if not getattr(kv, "swa_retained_ranges", ()):
        return
    bigram = int(cache.tree_core.is_eagle)
    ranges = kv.swa_live_ranges(cache.page_size, up_to - bigram)
    for lo, hi in ranges:
        lease.guard._checkpoint(req, up_to=hi + bigram, swa_checkpoint_start=lo)
        if kv.cache_protected_len < hi:
            # A rotation conflict declines the insert without taking ownership.
            # Let release return the private suffix; don't hide its live holes
            # behind a second, ordinary final checkpoint.
            lease.publication_declined = True
            break
        kv.swa_retained_ranges = [
            (max(a, kv.cache_protected_len), b)
            for a, b in kv.swa_retained_ranges
            if b > kv.cache_protected_len
        ]
