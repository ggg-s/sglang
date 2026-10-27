"""Real allocator cross-stream reuse regression, without model weights.

The owner replaces its free list while the other lane has queued reads of
the old list. Churn on the owner can recycle that storage before those reads.
Check raw indices before launching KV stores so the broken version fails a
host assertion instead of poisoning the CUDA context with a device assert.
This does not reproduce the full SWE workload or validate TP2 NCCL progress.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SIZE = 32768
CHUNK = 16384
PAGE = 16
POISON = 1_000_000


@pytest.fixture(scope="module")
def lanes():
    context = torch.cuda.GreenContext.create(num_sms=8)
    yield torch.cuda.Stream(), context.Stream()
    torch.cuda.synchronize()


def make_pool(kind, need_sort):
    if kind == "mamba":
        return MambaSlotAllocator(SIZE, "cuda")
    if kind == "token":
        return TokenToKVPoolAllocator(SIZE, torch.float16, "cuda", None, need_sort)
    return PagedTokenToKVPoolAllocator(
        SIZE, PAGE, torch.float16, "cuda", None, need_sort
    )


def prepare(kind, mode, need_sort):
    pool = make_pool(kind, need_sort)
    # Clone: token/Mamba alloc returns a view of the original free list. The
    # independent owned allocation must not accidentally keep that list alive.
    owned = pool.alloc(PAGE).clone()
    prefix_cpu = torch.tensor([0], dtype=torch.int64)
    seq_cpu = torch.tensor([PAGE + 1 if mode == "decode" else CHUNK])
    metadata = (
        prefix_cpu.cuda(),
        prefix_cpu,
        seq_cpu.cuda(),
        seq_cpu,
        torch.tensor([2 * PAGE - 1 if mode == "decode" else -1], device="cuda"),
    )
    return pool, owned, metadata


def consume(pool, kind, mode, metadata):
    prefix, prefix_cpu, seq, seq_cpu, last = metadata
    if kind != "paged":
        return pool.alloc(CHUNK).clone()
    if mode == "decode":
        return pool.alloc_decode(seq, seq_cpu, last)
    if mode == "extend":
        return pool.alloc_extend(prefix, prefix_cpu, seq, seq_cpu, last, CHUNK)
    return pool.alloc(CHUNK)


def exercise(kind, mode, need_sort, owner, consumer):
    entries = SIZE // PAGE if kind == "paged" else SIZE
    # Warm operators, Triton compilation and both streams' allocation caches
    # before introducing a delayed reader. First-use synchronization would
    # otherwise hide the race.
    with torch.cuda.stream(owner):
        pool, owned, metadata = prepare(kind, mode, need_sort)
        churn = [
            torch.empty(entries, dtype=torch.int64, device="cuda") for _ in range(64)
        ]
    consumer.wait_stream(owner)
    with torch.cuda.stream(consumer):
        warm = consume(pool, kind, mode, metadata)
    torch.cuda.synchronize()
    with torch.cuda.stream(owner):
        pool.free(owned)
        if need_sort:
            pool.merge_and_sort_free()
    torch.cuda.synchronize()
    del pool, owned, warm, churn, metadata

    for trial in range(3):
        with torch.cuda.stream(owner):
            pool, owned, metadata = prepare(kind, mode, need_sort)
        consumer.wait_stream(owner)
        with torch.cuda.stream(consumer):
            torch.cuda._sleep(300_000_000)
            output = consume(pool, kind, mode, metadata)
        with torch.cuda.stream(owner):
            pool.free(owned)
            if need_sort:
                pool.merge_and_sort_free()
            churn = [
                torch.full((entries,), POISON, dtype=torch.int64, device="cuda")
                for _ in range(64)
            ]
        owner.synchronize()
        consumer.synchronize()

        count = 1 if mode == "decode" else CHUNK
        first = 2 * PAGE if kind == "paged" else PAGE + 1
        expected = torch.arange(first, first + count, dtype=torch.int64)
        torch.testing.assert_close(
            output.cpu(), expected, msg=f"{kind=} {mode=} {need_sort=} {trial=}"
        )
        if kind == "paged":
            # Exercise the actual guarded native KV store with these indices.
            k = torch.ones(count, 64, dtype=torch.float16, device="cuda")
            v = k + 1
            k_cache = torch.zeros(SIZE + PAGE, 64, dtype=k.dtype, device="cuda")
            v_cache = torch.zeros_like(k_cache)
            store_cache(k, v, k_cache, v_cache, output)
            torch.testing.assert_close(k_cache[output], k)
            torch.testing.assert_close(v_cache[output], v)
        else:
            # State-index consumers use advanced indexing, as GDN preparation
            # does. Slot indices must remain valid after the list replacement.
            state = torch.arange(SIZE + 1, device="cuda")
            torch.testing.assert_close(state[output].cpu(), expected)
        torch.cuda.synchronize()
        del pool, owned, metadata, output, churn


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("need_sort", [False, True])
@pytest.mark.parametrize("mode", ["extend", "decode", "alloc"])
def test_paged_free_list_survives_lane_replacement(lanes, reverse, need_sort, mode):
    owner, consumer = lanes[::-1] if reverse else lanes
    exercise("paged", mode, need_sort, owner, consumer)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["token", "mamba"])
def test_slot_free_list_survives_lane_replacement(lanes, reverse, kind):
    owner, consumer = lanes[::-1] if reverse else lanes
    exercise(kind, "alloc", False, owner, consumer)
