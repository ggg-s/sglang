"""Private split-dispatch kernel; buffers belong to each captured prefix."""

import triton
import triton.language as tl


@triton.jit
def _compact_neural_rows(
    mask,
    rows,
    count,
    live_bs,
    raw_bs,
    NO_SKIP: tl.constexpr,
    WRITE_LIVE_BS: tl.constexpr,
    READ_RAW_BS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One ordered scan, including batches larger than a block. Atomic append
    # would allow different row permutations across TP ranks.
    offset = 0
    if READ_RAW_BS:
        raw_bs = tl.load(raw_bs)
        limit = CAPACITY
    else:
        limit = raw_bs
    for start in range(tl.cdiv(limit, BLOCK)):
        row = start * BLOCK + tl.arange(0, BLOCK)
        hit = tl.load(mask + row, row < raw_bs, other=0)
        take = (row < raw_bs) & ((~hit) | NO_SKIP)
        rank = tl.cumsum(take.to(tl.int32)) - 1
        tl.store(rows + offset + rank, row.to(tl.int64), take)
        offset += tl.sum(take.to(tl.int32))
    tl.store(count, offset)
    if WRITE_LIVE_BS:
        tl.store(live_bs, raw_bs)


def compact_neural_rows(
    mask, rows, count, raw_bs, no_skip, *, live_bs=None, capacity=None
):
    """Write stable original-row indices and one count, without allocations."""
    _compact_neural_rows[(1,)](
        mask,
        rows,
        count,
        live_bs,
        raw_bs,
        no_skip,
        live_bs is not None,
        capacity is not None,
        capacity or 0,
        256,
    )
