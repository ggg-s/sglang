"""Bounded live-source copies into an unadmitted native reservation.

Only closed, locally complete pages are immutable. Every scheduler command
fences its native GPU reads before returning; network DMA addresses private
staging, never a live native row. A frozen manifest seals BOTH sides' receipts.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace as NS

import torch

from . import d2d_bulk as bulk
from . import dsv_d2d as d2d
from . import dsv_d2d_reservation as reservation

PROTOCOL = "closed-page-precopy-v1"
COMMANDS = {"d2d_precopy_" + s for s in ("status", "prepare", "copy", "apply", "close")}


@dataclass
class Source:
    req: object
    attempt: tuple
    plan_digest: str
    sender: object = None
    closed: bool = False
    failed: bool = False
    proofs: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    close_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class Destination:
    receiver: object
    ends: list
    plan_digest: str
    explicit_tcp: bool = False
    pending: dict | None = None
    receipts: dict = field(default_factory=dict)
    closed: bool = False
    failed: bool = False


def rdma_available(controller):
    from . import mooncake_backend as transport

    return transport.policy() != "tcp" and transport.gpu_capability(
        transport.handle(d2d.manager(controller).engine), d2d.pool_device(controller)
    )


def tcp_engine(controller):
    from . import mooncake_backend as transport

    native = d2d.manager(controller).engine
    engine = transport.handle(native)
    if engine.protocol == "tcp":
        return engine
    return transport.tcp_engine(getattr(engine.native, "hostname", "127.0.0.1"))


def plan_hash(plan):
    return hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def sources(controller):
    if not hasattr(controller, "d2d_precopies"):
        controller.d2d_precopies = {}
    return controller.d2d_precopies


def mark_unsafe(controller, slot):
    slot.poison()
    controller.d2d_precopy_unsafe = True


def stable_pages(controller, req):
    """Native allocation/committed_len alone is NOT a receive completion proof."""
    schema = controller._schema(d2d.manager(controller))
    ends = [0] * len(schema["entries"])
    if req.finished() or not req.kv.holds_kv or getattr(req, "_handoff_frozen", False):
        return ends
    parent = getattr(req, "_pre_layerwise_target", None)
    rpc = getattr(req, "_dsv41_remote", None)
    complete = set()
    if parent is not None and not parent.closed:
        from .dsv41_progressive import ProgressiveTarget, completed

        if not isinstance(parent, ProgressiveTarget):
            return ends
        complete = completed(parent.queue, parent.decode_req)
        if rpc is not None:
            complete |= rpc.local_entries
    elif rpc is not None and not rpc.all_local():
        return ends
    elif req.output_ids:
        complete = set(range(len(ends)))
    else:
        # Before early admission, the native receive wrapper owns the offer.
        from .compat import decode_queue_items
        from .dsv41_progressive import completed

        queue = getattr(controller.scheduler, "disagg_decode_transfer_queue", None)
        if queue is not None:
            for dr in decode_queue_items(queue):
                if (
                    dr.req is req
                    and getattr(dr, "_pre_layerwise_offer", None) is not None
                ):
                    complete = completed(queue, dr)
                    break
    count = int(req.kv.kv_committed_len) // controller.pool.page_size
    for i in complete:
        ends[i] = count
    return ends


def frame_layout(value, entries, *, rdma):
    """Return validated native ranges and their shard-local byte lengths."""
    if not isinstance(value, dict):
        raise ValueError("invalid precopy frame")
    if set(value) == {"entry", "start", "count", "chunk"}:
        parts = (value,)
    elif (
        rdma
        and set(value) == {"segments", "chunk"}
        and isinstance(value["segments"], list)
        and 2 <= len(value["segments"]) <= len(entries)
    ):
        parts = value["segments"]
        if any(
            not isinstance(part, dict) or set(part) != {"entry", "start", "count"}
            for part in parts
        ):
            raise ValueError("invalid packed precopy segment")
    else:
        raise ValueError("invalid precopy frame")
    if not isinstance(value["chunk"], str) or len(value["chunk"]) != 32:
        raise ValueError("invalid precopy chunk")

    layout = []
    previous = -1
    for part in parts:
        i, start, count = (part[k] for k in ("entry", "start", "count"))
        if (
            any(type(x) is not int for x in (i, start, count))
            or not previous < i < len(entries)
            or start < 0
            or count < 1
        ):
            raise ValueError("precopy frame outside bounded native layout")
        layout.append((i, start, count, count * entries[i][-1]))
        previous = i
    size = sum(nbytes for _, _, _, nbytes in layout)
    if size > (bulk.RDMA_CAPACITY if rdma else d2d.LIMIT):
        raise ValueError("precopy frame outside bounded native layout")
    return layout, size


def source_record(controller, epoch):
    plan, req, attempt = reservation.records(controller)[0][epoch]
    record = sources(controller).get(epoch)
    if record is None:
        record = sources(controller)[epoch] = Source(req, attempt, plan_hash(plan))
        if not hasattr(req, "_d2d_precopies"):
            req._d2d_precopies = []
        req._d2d_precopies.append((controller, record))
    if (
        record.req is not req
        or record.attempt != attempt
        or reservation.source_attempt(req) != attempt
    ):
        raise ValueError("precopy source attempt changed")
    return record


def ticket(controller, epoch):
    t = reservation.records(controller)[1].get(epoch)
    if t is None or not t.complete or t.consumed or t.cancelled or t.error:
        raise ValueError("precopy requires a live unadmitted reservation")
    return t


def tables(controller):
    from .dsv41_staging import buffers

    return [
        d2d.matrix(t, entry[-1])
        for t, entry in zip(
            buffers(controller.pool, d2d.manager(controller)),
            controller._schema(d2d.manager(controller))["entries"],
        )
    ]


def page_ids(controller, req, start, count):
    page = controller.pool.page_size
    return (
        controller.scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, start * page : (start + count) * page : page
        ]
        // page
    ).long()


def close_source(controller, epoch, pending_chunk=None):
    record = sources(controller).get(epoch)
    if record is None:
        # Tombstone before acknowledging: delayed COPY cannot resurrect it.
        closed = getattr(controller, "d2d_precopy_closed", None)
        if closed is None:
            closed = controller.d2d_precopy_closed = set()
        closed.add(epoch)
        return dict(
            protocol=PROTOCOL,
            epoch=epoch,
            closed=True,
            safe=True,
            pending_chunk=pending_chunk,
            sent=None,
        )
    record.closed = True
    safe = not record.failed
    if record.sender is not None and not record.sender.closed:
        if record.sender.unsafe:
            safe = False
            controller.d2d_precopy_unsafe = True
        else:
            try:
                record.sender.close()
            except BaseException:
                safe = False
                mark_unsafe(controller, record.sender)
    return dict(
        protocol=PROTOCOL,
        epoch=epoch,
        closed=True,
        safe=safe,
        close_id=record.close_id,
        pending_chunk=pending_chunk,
        sent=record.proofs.get(pending_chunk),
    )


def release_source(req, release):
    # Called on native release/retract. All live-source GPU reads are already
    # synchronous at command boundaries; uncertain TE only owns private backing.
    held = False
    for controller, record in getattr(req, "_d2d_precopies", ()):
        if getattr(record, "gpu_hold", None) is not None:
            from .transport_failure import retain_release

            retain_release(req, record, release)
            controller.d2d_precopy_unsafe = held = True
        for epoch, saved in sources(controller).items():
            if saved is record:
                close_source(controller, epoch)
    return held


def close_destination(controller, t, proof=None):
    state = getattr(t, "precopy", None)
    if state is None or state.closed:
        return
    state.closed = True  # No later APPLY may touch the native reservation.
    receiver = state.receiver
    if receiver is None:
        return
    # APPLY fenced all writes into the native row before acknowledging. Late
    # source writes can only reach this private slot. An unacknowledged chunk
    # has no positive producer-stop proof: quarantine it, but not native KV.
    safe = state.pending is None and not state.failed
    if state.pending is not None:
        safe = bool(
            proof
            and proof.get("protocol") == PROTOCOL
            and proof.get("epoch") == t.plan["epoch"]
            and proof.get("closed")
            and proof.get("safe")
            and proof.get("pending_chunk") == state.pending["frame"]["chunk"]
            and not state.failed
        )
    if safe:
        try:
            sent = proof.get("sent") if state.pending is not None else None
            if sent is not None:
                if sent["receive"] != state.pending:
                    raise ValueError(
                        "precopy producer-stop receipt differs from pending slot"
                    )
                # TE send success may precede peer application on TCP. A stop
                # ACK does not prove that the final bytes arrived; observe the
                # receive footer before deregistering even for discarded data.
                discarded = receiver.consume(
                    dict(
                        bytes=sent["bytes"],
                        bulk_protocol=receiver.protocol,
                        bulk_nonce=receiver.nonce,
                        bulk_sequence=receiver.sequence,
                    ),
                    sent["bytes"],
                    controller.config.drain_timeout,
                )
                receiver.drain()
                del discarded
            receiver.close()
        except BaseException:
            mark_unsafe(controller, receiver)
            raise
    else:
        mark_unsafe(controller, receiver)


def handle(controller, data):
    from . import mooncake_backend as transport

    epoch, cmd = data["migration_epoch"], data["cmd"]
    error, result = None, None
    finished = False
    # Every command has exactly two collective boundaries, including retries
    # and failures. No rank may enter a native mutation when a peer rejects.
    try:
        if cmd == "d2d_precopy_close":
            pass
        elif cmd in ("d2d_precopy_status", "d2d_precopy_copy"):
            record = source_record(controller, epoch)
            if record.failed:
                raise ValueError("precopy source failed")
            finished = record.req.finished()
            if not finished:
                if (
                    epoch in getattr(controller, "d2d_precopy_closed", ())
                    or record.closed
                ):
                    raise ValueError("precopy source closed")
                if cmd.endswith("status"):
                    ends = stable_pages(controller, record.req)
                else:
                    dest = reservation.select(controller, data, "receive")
                    region = bulk.descriptor(dest["region"])
                    rdma = region["protocol"] == transport.GPU
                    explicit_tcp = dest.get("tcp") is True
                    if explicit_tcp and (rdma or transport.policy() == "rdma"):
                        raise ValueError("precopy source did not offer TCP fallback")
                    f = data["frame"]
                    entries = controller._schema(d2d.manager(controller))["entries"]
                    layout, size = frame_layout(f, entries, rdma=rdma)
                    if size > region["size"] - bulk.FOOTER:
                        raise ValueError("precopy frame exceeds published slot")
                    if rdma and not rdma_available(controller):
                        raise ValueError("precopy source did not offer GPU RDMA")
                    if (
                        dest["frame"] != f
                        or dest["epoch"] != epoch
                        or dest["plan_hash"] != record.plan_digest
                    ):
                        raise ValueError("precopy destination certificate mismatch")
                    saved = record.proofs.get(f["chunk"])
                    if saved is not None:
                        if saved["receive"] != dest:
                            raise ValueError("precopy retry changed descriptor")
                    else:
                        available = stable_pages(controller, record.req)
                        if any(available[i] < lo + count for i, lo, count, _ in layout):
                            raise ValueError(
                                "precopy page is still mutable or remotely owned"
                            )
        else:
            t = ticket(controller, epoch)
            state = getattr(t, "precopy", None)
            if cmd.endswith("prepare"):
                rdma = data.get("rdma") is True
                explicit_tcp = data.get("tcp") is True
                if explicit_tcp and (rdma or transport.policy() == "rdma"):
                    raise ValueError("precopy destination did not offer TCP fallback")
                if rdma and not rdma_available(controller):
                    raise ValueError("precopy destination did not offer GPU RDMA")
                capacity = bulk.frame_capacity(
                    data.get("capacity", bulk.LEGACY_CAPACITY), rdma=rdma
                )
                if state is not None and (
                    state.receiver.capacity != capacity
                    or rdma != (state.receiver.backend == transport.GPU)
                    or explicit_tcp != state.explicit_tcp
                ):
                    raise ValueError("precopy slot capacity/backend changed")
            else:
                rdma = state is not None and state.receiver.backend == transport.GPU
            if state is not None and (state.closed or state.failed):
                raise ValueError("precopy destination closed/failed")
            f = data["frame"]
            entries = t.plan["schema"]["entries"]
            layout, size = frame_layout(f, entries, rdma=rdma)
            if any(
                (lo + count) * controller.pool.page_size > t.plan["upper"]
                for _, lo, count, _ in layout
            ):
                raise ValueError("precopy frame exceeds reservation")
            if cmd.endswith("prepare"):
                if size > capacity:
                    raise ValueError("precopy frame exceeds destination slot")
                if state is not None and f["chunk"] in state.receipts:
                    raise ValueError("precopy chunk already applied; reconcile receipt")
                if (
                    state is not None
                    and state.pending is not None
                    and state.pending["frame"] != f
                ):
                    raise ValueError("precopy slot already published")
                if any(
                    lo != (state.ends[i] if state is not None else 0)
                    for i, lo, _, _ in layout
                ):
                    raise ValueError("precopy prefix gap/duplicate")
            elif cmd.endswith("apply"):
                proof = reservation.select(controller, data, "proof")
                if state is None:
                    raise ValueError("precopy receive slot missing")
                old = state.receipts.get(f["chunk"])
                expected = old or dict(receive=state.pending, bytes=size)
                if proof != expected or proof["receive"]["frame"] != f:
                    raise ValueError("precopy send proof differs from published slot")
            else:
                raise ValueError("unsupported precopy command")
    except Exception as exc:
        error = repr(exc)
    healthy, running = controller.agree([int(error is None), int(not finished)])
    if not healthy:
        raise ValueError(f"TP precopy preflight rejected: {error or 'peer rejected'}")
    if not running:
        return dict(deferred=True, migration_deferred_reason="request_finished")
    scatter_started = False
    try:
        if cmd.endswith("close"):
            result = close_source(controller, epoch, data.get("pending_chunk"))
        elif cmd.endswith("status"):
            result = dict(protocol=PROTOCOL, available=ends)
        elif cmd.endswith("prepare"):
            if state is None:
                receiver = bulk.Receiver(
                    (
                        transport.handle(d2d.manager(controller).engine)
                        if rdma
                        else d2d.manager(controller).engine
                    ),
                    NS(),
                    d2d.pool_device(controller),
                    backend=transport.GPU if rdma else transport.TCP,
                    capacity=capacity,
                )
                state = t.precopy = Destination(
                    receiver,
                    [0] * len(t.plan["schema"]["entries"]),
                    plan_hash(t.plan),
                    explicit_tcp=explicit_tcp,
                )
            if state.pending is None:
                state.pending = dict(
                    epoch=epoch,
                    plan_hash=state.plan_digest,
                    frame=f,
                    region=state.receiver.prepare(),
                    sequence=state.receiver.sequence,
                )
                if explicit_tcp:
                    state.pending["tcp"] = True
            result = dict(receive=state.pending)
        elif cmd.endswith("copy"):
            replayed = saved is not None
            if saved is None:
                device = d2d.pool_device(controller)
                # Scheduler's current stream is ordered after model submission;
                # index_select gets independent backing. Fence before native TE.
                with d2d.on_device(device):
                    if torch.device(device).type == "cuda":
                        torch.cuda.current_stream(device).wait_stream(
                            controller.scheduler.forward_stream
                        )
                    native = tables(controller)
                    rows = [
                        (native[i], page_ids(controller, record.req, lo, count))
                        for i, lo, count, _ in layout
                    ]
                    engine = (
                        tcp_engine(controller)
                        if explicit_tcp
                        else d2d.manager(controller).engine
                    )
                    capacity = region["size"] - bulk.FOOTER
                    if record.sender is None:
                        catalog = transport.Catalog(engine, device) if rdma else None
                        record.sender = bulk.Sender(
                            engine,
                            NS(),
                            device,
                            catalog=catalog,
                            capacity=min(capacity, bulk.CAPACITY),
                            rdma_capacity=capacity if rdma else None,
                        )
                    if rdma:
                        ready = d2d.dp.event(device)
                        try:
                            size = record.sender.write_rows(
                                rows, ready, region, dest["sequence"]
                            )
                        except BaseException:
                            if record.sender.copy_owner is not None:
                                record.gpu_hold = record.sender.copy_owner
                            raise
                    else:
                        ((table, ids),) = rows
                        data_tensor = ready = None
                        try:
                            data_tensor = table.index_select(0, ids)
                            ready = d2d.dp.event(device)
                            ready.synchronize()
                        except BaseException:
                            d2d.dp.io_for(record.req).poison()
                            record.req._handoff_transfer_unsafe = True
                            record.gpu_hold = (table, ids, data_tensor, ready)
                            raise
                        size = record.sender.write(
                            data_tensor, ready, region, dest["sequence"]
                        )
                    saved = dict(receive=dest, bytes=size)
                    record.bindings[f["chunk"]] = [
                        (ids.cpu().tolist(), table.data_ptr()) for table, ids in rows
                    ]
                    record.proofs[f["chunk"]] = saved
            result = dict(
                proof=saved,
                replayed=replayed,
                copied_payload_bytes=0 if replayed else saved["bytes"],
            )
        else:
            if old is None:
                receiver = state.receiver
                device = d2d.pool_device(controller)
                with d2d.on_device(device):
                    header = dict(
                        bytes=proof["bytes"],
                        bulk_protocol=receiver.protocol,
                        bulk_nonce=receiver.nonce,
                        bulk_sequence=receiver.sequence,
                    )
                    from .dsv41_receive import scatter_prefix

                    native = tables(controller)
                    page_size = controller.pool.page_size
                    rows = [
                        (
                            native[i],
                            page_ids(controller, t.req, lo, count),
                            d2d.key_layout(controller, entries[i][2]),
                        )
                        for i, lo, count, _ in layout
                    ]

                    def scatter(copied):
                        nonlocal scatter_started
                        scatter_started = True
                        offset = 0
                        for (i, lo, count, nbytes), (table, ids, key_layout) in zip(
                            layout, rows
                        ):
                            scatter_prefix(
                                table,
                                copied[offset : offset + nbytes].reshape(count, -1),
                                ids,
                                lo,
                                (lo + count) * page_size,
                                page_size,
                                entries[i][1],
                                key_layout,
                            )
                            offset += nbytes

                    if rdma:
                        receiver.consume(
                            header,
                            proof["bytes"],
                            controller.config.drain_timeout,
                            apply=scatter,
                        )
                    else:
                        scatter(
                            receiver.consume(
                                header, proof["bytes"], controller.config.drain_timeout
                            )
                        )
                    # Native row reuse must wait for the scatter, independently
                    # of private Receiver retirement.
                    t.ready = d2d.dp.event(device)
                    t.ready.synchronize()
                for i, _, count, _ in layout:
                    state.ends[i] += count
                state.receipts[f["chunk"]] = proof
                state.pending = None
            result = dict(receipt=proof, prefix_pages=state.ends)
    except BaseException as exc:
        error = repr(exc)
        if cmd.endswith("copy"):
            record.failed = True
            if record.sender is not None:
                mark_unsafe(controller, record.sender)
        elif cmd.endswith("prepare"):
            controller.d2d_precopy_unsafe = True
        elif cmd.endswith("apply"):
            state.failed = True
            # A failed CUDA scatter can still touch native KV. Unlike a late
            # network write, it must retain that reservation as well.
            mark_unsafe(controller, state.receiver)
            if scatter_started:
                t.error = error
                d2d.dp.io_for(t.req).poison()
    native_safe = not (
        cmd.endswith("copy")
        and getattr(record, "gpu_hold", None) is not None
        or cmd.endswith("apply")
        and error is not None
        and scatter_started
    )
    success, native_safe = controller.agree([int(error is None), int(native_safe)])
    if not native_safe:
        if cmd.endswith("copy"):
            from .transport_failure import quarantine

            quarantine(
                controller,
                NS(req=record.req, responded=True),
                error or "peer GPU read failed",
            )
        else:
            t.error = error or "peer precopy scatter failed; native allocation retained"
            d2d.dp.io_for(t.req).poison()
    if not success:
        raise d2d.TransportFailure(
            f"TP precopy operation failed: {error or 'peer failed'}"
        )
    return result


def seal(controller, source, receipts):
    """Verify that only acknowledged immutable pages survive into the manifest."""
    record = sources(controller).get(source.epoch)
    entries = source.owner._schema(d2d.manager(controller))["entries"]
    ends = [0] * len(entries)
    if record is None:
        if receipts:
            raise ValueError("precopy source receipts missing")
        return dict(protocol=PROTOCOL, prefix_pages=ends, receipts=[])
    if record.failed or reservation.source_attempt(source.req) != record.attempt:
        raise ValueError("precopy source failed/attempt changed at seal")
    native = tables(controller)
    page_size = controller.pool.page_size
    for proof in receipts:
        f = proof["receive"]["frame"]
        chunk = f["chunk"]
        if record.proofs.get(chunk) != proof:
            raise ValueError("precopy receipt not a contiguous acknowledged send")
        for position, part in enumerate(f.get("segments", (f,))):
            i, lo, count = (part[k] for k in ("entry", "start", "count"))
            if lo != ends[i]:
                raise ValueError("precopy receipt not a contiguous acknowledged send")
            ids = page_ids(controller, source.req, lo, count)
            if (
                record.bindings[chunk][position]
                != (ids.cpu().tolist(), native[i].data_ptr())
                or (lo + count) * page_size > source.prefix_len
            ):
                raise ValueError("precopy source page ownership changed before freeze")
            ends[i] += count
    close_source(controller, source.epoch)
    return dict(protocol=PROTOCOL, prefix_pages=ends, receipts=receipts)


def consume(controller, t, transfer):
    seal = transfer.get("precopy")
    state = getattr(t, "precopy", None)
    if seal is None:
        if state is not None:
            raise ValueError("frozen manifest omitted precopy receipts")
        return
    expected = dict(
        protocol=PROTOCOL,
        prefix_pages=state.ends if state else [0] * len(t.plan["schema"]["entries"]),
        receipts=list(state.receipts.values()) if state else [],
    )
    if (
        seal != expected
        or state is not None
        and (state.pending is not None or state.failed)
    ):
        raise ValueError("precopy seal differs from destination applied receipts")
    close_destination(controller, t)
