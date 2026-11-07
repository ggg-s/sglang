"""Exercise live copies, immutable-page seals and native D2D ownership."""

import time
import uuid

import torch
from sglang_remote_attn import d2d_precopy as pc
from sglang_remote_attn import dsv_d2d as d2d


def command(controller, epoch, cmd, **payload):
    return d2d.handle(
        controller, dict(cmd="d2d_" + cmd, migration_epoch=epoch, **payload)
    )


def reserve(e, req=None, epoch="precopy", target_rid="target", growth=16):
    req = req or e.req
    common = dict(rid=req.rid, target_rid=target_rid, destination="D1")
    plan = command(e.source, epoch, "plan", growth_tokens=growth, **common)["plan"]
    command(e.target, epoch, "reserve", plan=plan)
    if e.device == "cuda":
        e.source.scheduler.forward_stream = torch.cuda.current_stream()
    return common, plan


def copy_frame(e, epoch, entry, start, count, retry=False):
    f = dict(entry=entry, start=start, count=count, chunk=uuid.uuid4().hex)
    received = command(e.target, epoch, "precopy_prepare", frame=f)["receive"]
    if retry:
        assert (
            command(e.target, epoch, "precopy_prepare", frame=f)["receive"] == received
        )
    proof = command(e.source, epoch, "precopy_copy", frame=f, receive=received)["proof"]
    if retry:
        assert (
            command(e.source, epoch, "precopy_copy", frame=f, receive=received)["proof"]
            == proof
        )
    receipt = command(e.target, epoch, "precopy_apply", frame=f, proof=proof)["receipt"]
    if retry:
        assert (
            command(e.target, epoch, "precopy_apply", frame=f, proof=proof)["receipt"]
            == receipt
        )
    return receipt


def copy_available(e, epoch, retry=False):
    available = command(e.source, epoch, "precopy_status")["available"]
    receipts = []
    for i, count in enumerate(available):
        if count:
            receipts.append(copy_frame(e, epoch, i, 0, count, retry=retry))
    return available, receipts


def finish(e, epoch, meta):
    e.epoch = epoch
    e.import_(meta)
    target = e.target.d2d_targets[epoch]
    assert not target.admitted and e.req.kv.holds_kv and not e.freed
    command(e.target, epoch, "commit")
    e.gate.set()
    target = e.wait_copy()
    # Exact closed-page bytes and mutable state, using native row mappings.
    src_pages = pc.page_ids(
        e.source, e.req, 0, (meta["kv_len"] + e.f.page - 1) // e.f.page
    )
    dst_pages = pc.page_ids(e.target, target.decode_req.req, 0, src_pages.numel())
    for a, b in zip(pc.tables(e.source), pc.tables(e.target)):
        # Partial physical rows contain native padding; compare only full pages.
        n = meta["kv_len"] // e.f.page
        assert torch.equal(a[src_pages[:n]], b[dst_pages[:n]])
    for (_, a, ai), (_, b, bi) in zip(
        d2d.state_rows(e.source, e.req, meta["kv_len"]),
        d2d.state_rows(e.target, target.decode_req.req, meta["kv_len"]),
    ):
        assert torch.equal(a[ai.long()], b[bi.long()])
    e.target.drain_targets(required=target)
    source = e.source.d2d_sources[epoch]
    deadline = time.monotonic() + 10
    while not source.poll():
        assert time.monotonic() < deadline
        time.sleep(0.001)
    source.retire()
    source.retire()
    assert e.freed.count(e.req.rid) == 1 and not e.req.kv.holds_kv
    return target
