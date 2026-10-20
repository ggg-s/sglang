"""Triton implementations of compact online C128 metadata and state banks."""

import triton
import triton.language as tl


@triton.jit
def _plan_kernel(
    seq,
    lens,
    reqs,
    starts,
    out,
    capacity: tl.constexpr,
    state_offset: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    prefix = tl.load(seq + row)
    length = tl.load(lens + row)
    req = tl.load(reqs + row)
    step = tl.arange(0, BLOCK)
    token = tl.load(starts + row) + step
    end = prefix + step + 1
    close = end % 128 == 0
    valid = (step < length) & (token < capacity) & (req >= 0)
    segment = close | (step + 1 == length)
    size = end - tl.maximum(prefix, (end - 1) // 128 * 128)
    base = ((tl.where(close, 0, capacity) + token) * 4).to(tl.int64)
    tl.store(out + base, end.to(tl.int32), valid & segment)
    tl.store(out + base + 1, (token | (size << 16)).to(tl.int32), valid & segment)
    tl.store(out + base + 2, (req + state_offset).to(tl.int32), valid & segment)
    tl.store(out + base + 3, req.to(tl.int32), valid & segment)


@triton.jit
def _write_kernel(
    kv,
    seq,
    reqs,
    lens,
    starts,
    ape,
    state,
    kv_stride: tl.constexpr,
    state_row: tl.constexpr,
    bank_stride: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(reqs + row)
    length = tl.load(lens + row)
    prefix = tl.load(seq + row)
    start = tl.load(starts + row)
    d = tl.arange(0, BLOCK)
    active = (req >= 0) & (length > 0)
    partial = active & (prefix % 128 != 0)
    maximum = tl.load(state + req * state_row + d, partial & (d < D), other=0).to(
        tl.float32
    )
    total = tl.load(state + req * state_row + D + d, partial & (d < D), other=0).to(
        tl.float32
    )
    value = tl.load(state + req * state_row + 2 * D + d, partial & (d < D), other=0).to(
        tl.float32
    )
    for step in range(length):
        pos = (prefix + step) % 128
        item = tl.load(
            kv + (start + step) * kv_stride + d, active & (d < D), other=0
        ).to(tl.float32)
        score = tl.load(
            kv + (start + step) * kv_stride + D + d, active & (d < D), other=0
        ).to(tl.float32)
        score += tl.load(ape + pos * D + d, active & (d < D), other=0)
        new_max = tl.maximum(maximum, score)
        old_sum = total * tl.exp(maximum - new_max)
        new_sum = tl.exp(score - new_max)
        value = tl.where(
            pos == 0, item, (value * old_sum + item * new_sum) / (old_sum + new_sum)
        )
        total = tl.where(pos == 0, 1.0, old_sum + new_sum)
        maximum = tl.where(pos == 0, score, new_max)
        slot = req + (step + 1) * bank_stride
        store = active & (d < D) & ((prefix + step + 1) % 128 != 0)
        tl.store(state + slot * state_row + d, maximum, store)
        tl.store(state + slot * state_row + D + d, total, store)
        tl.store(state + slot * state_row + 2 * D + d, value, store)


@triton.jit
def _commit_kernel(
    seq,
    reqs,
    pending,
    state,
    num_reqs: tl.constexpr,
    width: tl.constexpr,
    state_row: tl.constexpr,
    bank_stride: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(reqs + row)
    valid = (req >= 0) & (req < num_reqs)
    old = tl.load(pending + req, valid, other=-1)
    current = tl.load(seq + row)
    count = tl.minimum(tl.maximum(current - old, 0), width)
    valid &= (old >= 0) & (count > 0) & ((old + count) % 128 != 0)
    d = tl.arange(0, BLOCK)
    value = tl.load(
        state + (req + count * bank_stride) * state_row + d, valid & (d < D), other=0
    )
    tl.store(state + req * state_row + d, value, valid & (d < D))
