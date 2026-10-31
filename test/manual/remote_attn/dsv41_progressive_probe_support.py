"""CUDA scatter/ordered-switch fixture for the real RemoteHub probe.

Native checkpoint/indexer kernels are outside this small 4060 test. Real native
paged bytes, pinned H2D, CUDA events, eager RPC and ordered plugin retirement run.
"""

import time
from types import SimpleNamespace

import torch
from sglang_remote_attn import dsv41_progressive as pg
from sglang_remote_attn.dsv41_receive import scatter_prefix
from sglang_remote_attn.retirement import Retirement


def start(f, source, target):
    schema = dict(entries=[["kv", 1, 1], ["kv", 2, 2]])
    f.controller._schema = lambda m: schema
    f.controller.scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        kv_manager=None
    )
    f.controller.rank, f.controller.tp_size = 0, 1
    f.controller.config = SimpleNamespace(drain_timeout=5)
    votes = []
    f.controller.agree = lambda flags: votes.append(list(flags)) or flags
    target.progressive = True
    source.schema = schema
    source.barrier_ready = True
    saved = {ratio: p.kv_buffer[0].clone() for ratio, p in f.d_pool.kv_pools.items()}
    for ratio in (1, 2):
        for physical in f.d_pages[:4]:
            f.d_pool.kv_pools[ratio].kv_buffer[0][physical].zero_()
    dr = SimpleNamespace(req=f.req, _staging_completed_layers=[])
    pending = {}

    def advance(dr):
        for entry, event in pending.items():
            if event.query() and not any(
                entry in ids for ids, _, _ in dr._staging_completed_layers
            ):
                dr._staging_completed_layers.append(((entry,), 0, 5))

    parent = SimpleNamespace(
        closed=False,
        decode_req=dr,
        path=source.path,
        generation=source.generation,
        rpc=target,
        queue=SimpleNamespace(
            staging_handler=SimpleNamespace(
                advance_scatter=advance, is_failed=lambda dr: False
            )
        ),
    )
    resources = [pg.Resource(parent, i, i + 1) for i in (0, 1)]
    f.controller.targets = resources.copy()
    protocol = Retirement(f.controller)
    for resource in resources:
        protocol.register(resource)
    for _ in range(32):
        protocol.observe()
    assert not votes
    return SimpleNamespace(
        saved=saved,
        pending=pending,
        protocol=protocol,
        resources=resources,
        votes=votes,
    )


def transfer(f, state, entry):
    ratio = entry + 1
    pool = f.d_pool.kv_pools[ratio]
    layout, cpp = pool.kv_layout, 8 // ratio
    source = f.p_pool.kv_pools[ratio].kv_buffer[0]
    # Real pinned CPU ring -> GPU shadow -> native-layout immutable scatter.
    staging = source[torch.tensor(f.p_pages, device="cuda")].cpu().pin_memory()
    shadow = torch.empty_like(staging, device="cuda")
    # Warm allocations on the very same stream. CUDA allocator blocks are
    # stream-owned; warming only the default stream can still cudaMalloc-sync.
    page_ids = torch.tensor(f.d_pages, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm = pool.kv_buffer[0].clone()
        scatter_prefix(warm, shadow, page_ids, 0, 33, 8, ratio, layout)
    stream.synchronize()
    before = pool.kv_buffer[0].clone()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1_000_000_000)
        shadow.copy_(staging, non_blocking=True)
        scatter_prefix(
            pool.kv_buffer[0],
            shadow,
            page_ids,
            0,
            33,
            8,
            ratio,
            layout,
        )
        event = torch.cuda.Event()
        event.record(stream)
    state.pending[entry] = event
    # Keep the DMA source and destination alive until actual GPU completion.
    state.keepalive = (staging, shadow, stream)
    state.protocol.observe()
    assert entry not in state.resources[entry].parent.rpc.local_entries
    assert not event.query()
    for _ in range(8):
        # GPU may finish during these polls. A notice still cannot switch
        # ownership before the ordered command/final vote below.
        state.protocol.observe()
        assert entry not in state.resources[entry].parent.rpc.local_entries
        assert len(state.votes) == entry
    event.synchronize()
    after = pool.kv_buffer[0]
    ds, base, ss = layout.row_addressing(cpp)
    last = f.d_pages[-1]
    # The partial page's D-owned rows/scales, plus padding, are untouched.
    mask = torch.zeros(after.shape[1], dtype=torch.bool, device="cuda")
    for row in range((33 % 8) // ratio):
        mask[row * ds : row * ds + layout.data_bytes] = True
        mask[base + row * ss : base + row * ss + layout.scale_bytes] = True
    assert torch.equal(after[last, ~mask], before[last, ~mask])
    for logical, physical in enumerate(f.d_pages[:4]):
        assert torch.equal(after[physical], state.saved[ratio][physical])
    state.protocol.observe()
    command = state.protocol.poll_command()
    assert command is not None
    state.protocol.arm(command)
    state.protocol.finalize()
    assert entry in state.resources[entry].parent.rpc.local_entries
    assert len(state.votes) == entry + 1
    return time.monotonic()


def finish(f, state, target):
    assert not state.protocol.owner.targets
    assert target.local_entries == {0, 1}
    assert [len(v) for v in state.votes] == [5, 5]
    before = len(state.votes)
    for _ in range(32):
        state.protocol.observe()
    assert len(state.votes) == before
    target.close("progressive-tp-local-probe")
    state.protocol.close()
    return dict(
        per_resource_final_votes=[5, 5],
        stable_remote_votes=0,
        stable_local_votes=0,
        pinned_h2d=True,
        odd_partial_page_tail_preserved=True,
        global_incomplete_during_decode=True,
    )
