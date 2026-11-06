"""Bounded Qwen layer-group x token-range Host/Mooncake transport."""

import hmac
import secrets
import socket
import threading
import time

import torch

from . import d2d_ranges as plan
from . import mooncake_backend as transport
from .d2d_bulk import CAPACITY, FOOTER, marker
from .d2d_groups import size
from .remote_data_plane import Visibility


def runtime():
    from . import handoff_control

    return handoff_control


def batches(meta, slots):
    for members, lo, hi in plan.rectangles(
        len(meta["chunks"]),
        meta["kv_committed_len"],
        size(meta["d2d_group_size"]),
        meta["d2d_token_range"],
    ):
        for offset in range(0, len(members), slots):
            yield members[offset : offset + slots], lo, hi


def source(scheduler, req, meta, remote, snapshot=None):
    r = runtime()
    token = secrets.token_hex(16)
    meta = r._with_transfer_ticket(token, dict(meta))
    group = min(size(meta["d2d_group_size"]), len(meta["chunks"]))
    row_bytes = meta["kv_heads"] * meta["head_dim"] * 4  # K + V, FP16/BF16
    width = min(plan.tokens(meta["d2d_token_range"]), CAPACITY // row_bytes)
    if not width:
        raise ValueError("Qwen KV row exceeds bounded D2D slot")
    meta["d2d_token_range"] = width
    wire_bytes = width * row_bytes + FOOTER
    device = torch.cuda.current_device()
    catalog = transport.Catalog(r._get_te(), torch.device("cuda", device))
    meta["d2d_capabilities"] = catalog.offer
    stages = []
    allocated_backend = None
    registration_safe = True

    def allocate(backend):
        nonlocal stages, allocated_backend, registration_safe
        engine = catalog.engines[backend]
        stages = [
            torch.empty(
                wire_bytes,
                dtype=torch.uint8,
                device=(device if backend == transport.GPU else "cpu"),
                pin_memory=backend == transport.TCP,
            )
            for _ in range(group)
        ]
        owner = r._registration_for(token, engine)
        try:
            if backend == transport.GPU:
                # Allocator reuse can carry work on the allocating stream.
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(device))
                ready.synchronize()
            owner.register(
                token,
                [x.data_ptr() for x in stages],
                [wire_bytes] * group,
                label="d2d-ranges-source",
            )
        except BaseException:
            registration_safe = False
            r._release_transfer_buffers(
                token,
                stages,
                "D2D source registration unknown",
                remote_complete=False,
                gpu_safe=False,
            )
            raise
        allocated_backend = backend

    # Preserve eager TCP registration and its setup failure ownership contract.
    if list(catalog.engines) == [transport.TCP]:
        allocate(transport.TCP)
    listener = None
    service = None
    started = False
    try:
        device = torch.cuda.current_device()
        producer = torch.cuda.Event()
        producer.record(torch.cuda.current_stream(device))
        if remote:
            service = r._start_remote_attention_service(scheduler, req, meta)
            meta["remote_service"] = service
        listener, port = r._open_transfer_listener()
        cancel = threading.Event()
        req._handoff_transfer_cancel = (listener, cancel)
        r._mark_source_transfer(scheduler, token, True)
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        indices = scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : meta["kv_committed_len"]
        ]

        def serve():
            conn = stream = None
            inputs, submitted = [], []
            native_safe, gpu_safe, arrival_safe = True, True, True
            sent, sequence = 0, 0
            try:
                torch.cuda.set_device(device)
                deadline = time.monotonic() + 180
                while not cancel.is_set():
                    try:
                        conn, _ = listener.accept()
                        break
                    except socket.timeout:
                        if time.monotonic() > deadline:
                            raise TimeoutError("D2D receiver did not connect")
                if conn is None:
                    return
                req._handoff_transfer_conn = conn
                conn.settimeout(300)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hello = r._recv_json(conn)
                if not hmac.compare_digest(str(hello.get("token", "")), token):
                    raise ValueError("bad D2D transfer token")
                r._send_json(conn, meta)
                peer = r._recv_json(conn)
                backend = catalog.select(peer["capabilities"])
                if peer.get("backend") != backend:
                    raise ValueError("D2D backend negotiation mismatch")
                if allocated_backend is None:
                    allocate(backend)
                elif allocated_backend != backend:
                    raise ValueError("D2D backend changed after registration")
                slots = peer["slots"]
                if (
                    type(slots) is not int
                    or not 0 < slots <= group
                    or len(peer["addrs"]) != slots
                    or len(peer["nonces"]) != slots
                ):
                    raise ValueError("invalid D2D range receiver slots")
                stream = r._fast_copy_streams(device, 1)[0]
                stream.wait_event(producer)
                te = catalog.engines[backend]
                for members, lo, hi in batches(meta, slots):
                    if cancel.is_set():
                        return
                    controls, sizes = [], []
                    for slot, layer in enumerate(members):
                        sequence += 1
                        nbytes = (hi - lo) * row_bytes
                        if snapshot is None:
                            with torch.cuda.stream(stream):
                                # Retain the GPU input until its D2H proof.
                                k = pool.get_key_buffer(layer)[indices[lo:hi]]
                                v = pool.get_value_buffer(layer)[indices[lo:hi]]
                                data = torch.cat((k.reshape(-1), v.reshape(-1))).view(
                                    torch.uint8
                                )
                                inputs.append(data)
                                stages[slot][:nbytes].copy_(data, non_blocking=True)
                        else:
                            offset, pieces = 0, []
                            for k, v in snapshot[layer]:
                                end = offset + k.shape[0]
                                if lo < end and hi > offset:
                                    pieces.append(
                                        (
                                            k[
                                                max(0, lo - offset) : min(
                                                    k.shape[0], hi - offset
                                                )
                                            ],
                                            v[
                                                max(0, lo - offset) : min(
                                                    v.shape[0], hi - offset
                                                )
                                            ],
                                        )
                                    )
                                offset = end
                            data = torch.cat(
                                (
                                    torch.cat([x[0] for x in pieces]).reshape(-1),
                                    torch.cat([x[1] for x in pieces]).reshape(-1),
                                )
                            ).view(torch.uint8)
                            with torch.cuda.stream(stream):
                                stages[slot][:nbytes].copy_(data, non_blocking=True)
                        controls.append(
                            dict(
                                layer=layer,
                                token_range=[lo, hi],
                                sequence=sequence,
                                slot=slot,
                                bytes=nbytes,
                            )
                        )
                        sizes.append(
                            nbytes + (FOOTER if backend == transport.TCP else 0)
                        )
                    if snapshot is None or backend == transport.GPU:
                        ready = torch.cuda.Event()
                        ready.record(stream)
                        ready.synchronize()
                        inputs.clear()
                    for control in controls if backend == transport.TCP else ():
                        slot, n = control["slot"], control["bytes"]
                        stages[slot][n : n + FOOTER].copy_(
                            torch.tensor(
                                list(
                                    marker(peer["nonces"][slot], control["sequence"], n)
                                ),
                                dtype=torch.uint8,
                            )
                        )
                    native_safe = False
                    arrival_safe = False
                    src = [stages[i].data_ptr() for i in range(len(members))]
                    # Both FAST_ASYNC settings use one submission. The native
                    # sync convenience API may hide failed, still-live retries.
                    te.batch_transfer_sync(
                        peer["session"], src, peer["addrs"][: len(members)], sizes
                    )
                    native_safe = True
                    for control in controls:
                        r._send_json(conn, control)
                    for control in controls:
                        ack = r._recv_json(conn)
                        if ack != dict(
                            free=control["slot"], sequence=control["sequence"]
                        ):
                            raise ValueError(
                                "D2D slot ACK does not match in-flight range"
                            )
                    arrival_safe = True
                    sent += sum(sizes)
                    submitted.append(dict(layers=list(members), token_range=[lo, hi]))
                r._send_json(conn, dict(done=True))
                r._TRANSFER_STATS[token] = dict(
                    backend=backend,
                    group_size=group,
                    token_range=width,
                    submitted_groups=submitted,
                    bytes=sent,
                    rc=0,
                )
            except Exception as exc:
                if not cancel.is_set():
                    r._TRANSFER_ERRORS[token] = repr(exc)
                    try:
                        r._send_json(conn, dict(ok=False, error=repr(exc)))
                    except Exception:
                        pass
            finally:
                if stream is not None:
                    try:
                        stream.synchronize()
                        inputs.clear()
                    except Exception:
                        gpu_safe = False
                        req._handoff_transfer_unsafe = True
                if not native_safe or not registration_safe:
                    req._handoff_transfer_unsafe = True
                r._publish_transfer_complete(
                    token,
                    native_safe and arrival_safe and gpu_safe and registration_safe,
                )
                r._release_transfer_buffers(
                    token,
                    (stages, inputs, producer),
                    "D2D range source completion unknown",
                    remote_complete=native_safe,
                    gpu_safe=gpu_safe and registration_safe,
                )
                r._mark_source_transfer(scheduler, token, False)
                req._handoff_transfer_conn = None
                if getattr(req, "_handoff_transfer_cancel", None) == (listener, cancel):
                    req._handoff_transfer_cancel = None
                if conn is not None:
                    conn.close()
                listener.close()

        thread = threading.Thread(target=serve, daemon=True, name="handoff-d2d-ranges")
        req._handoff_transfer_thread = thread
        thread.start()
        started = True
        return dict(
            host=__import__("os").environ.get("HANDOFF_TRANSFER_HOST", "127.0.0.1"),
            port=port,
            token=token,
            mode="remote" if remote else "layer",
            remote_service=service,
            d2d_range_protocol=plan.PROTOCOL,
        )
    finally:
        if not started:
            r._release_transfer_buffers(
                token, stages, "D2D source setup rollback", remote_complete=True
            )
            r._mark_source_transfer(scheduler, token, False)
            if listener is not None:
                listener.close()
            if service is not None:
                r.get_remote_hub(scheduler).registry.discard(
                    str(req.rid), expected_req=req
                )


def setup(conn, meta, owner):
    r = runtime()
    if meta.get("d2d_range_protocol") != plan.PROTOCOL:
        raise ValueError("unknown D2D range protocol")
    width = plan.tokens(meta["d2d_token_range"])
    group = min(size(meta["d2d_group_size"]), len(meta["chunks"]))
    slots = min(group, max(1, int(r.TARGET_STAGING_SLOTS) or group))
    capacity = width * meta["kv_heads"] * meta["head_dim"] * 4
    if not 0 < capacity <= CAPACITY or meta["dtype"] not in ("float16", "bfloat16"):
        raise ValueError("invalid bounded Qwen range manifest")
    device = torch.device("cuda", torch.cuda.current_device())
    catalog = transport.Catalog(r._get_te(), device)
    backend = catalog.select(meta.get("d2d_capabilities"))
    engine = catalog.engines[backend]
    visibility = (
        Visibility(device, protocol="rdma") if backend == transport.GPU else None
    )
    meta["_range_backend"] = backend
    meta["_range_visibility"] = visibility
    buffers = [
        torch.empty(
            capacity + FOOTER,
            dtype=torch.uint8,
            device=(device if backend == transport.GPU else "cpu"),
            pin_memory=backend == transport.TCP,
        )
        for _ in range(slots)
    ]
    nonces = [secrets.token_hex(16) for _ in buffers]
    addrs = [x.data_ptr() for x in buffers]
    registered = False
    try:
        if backend == transport.GPU:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
            ready.synchronize()
        r._registration_for(owner, engine).register(
            owner, addrs, [capacity + FOOTER] * slots, label="d2d-ranges-target"
        )
        registered = True
        meta["_range_nonces"] = nonces
        r._send_json(
            conn,
            dict(
                session=engine.session,
                capabilities=catalog.offer,
                backend=backend,
                addrs=addrs,
                slots=slots,
                nonces=nonces,
            ),
        )
    except BaseException:
        r._release_transfer_buffers(
            owner,
            buffers,
            "D2D range handshake outcome unknown",
            remote_complete=False,
            gpu_safe=registered,
        )
        raise
    return buffers, addrs, (capacity + FOOTER) * slots


def receive(scheduler, state, conn, meta, buffers, indices, owner, device):
    r = runtime()
    stream = None
    complete, gpu_safe = False, True
    try:
        torch.cuda.set_device(device)
        stream = r._fast_copy_streams(device, 1, role="target")[0]
        if getattr(state, "indices_ready", None) is not None:
            stream.wait_event(state.indices_ready)
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        coverage = plan.Coverage(len(state.events), meta["kv_committed_len"])
        sequence = 0
        for members, lo, hi in batches(meta, len(buffers)):
            for slot, layer in enumerate(members):
                sequence += 1
                msg = r._recv_json(conn)
                n = (hi - lo) * meta["kv_heads"] * meta["head_dim"] * 4
                expected = dict(
                    layer=layer,
                    token_range=[lo, hi],
                    sequence=sequence,
                    slot=slot,
                    bytes=n,
                )
                if msg != expected:
                    raise ValueError("D2D range control duplicate/gap/mismatch")
                if meta["_range_backend"] == transport.GPU:
                    # Control is sent only after ALL native descriptors finish.
                    # External writes must become visible before CUDA consumes.
                    meta["_range_visibility"].receive()
                else:
                    stamp = marker(meta["_range_nonces"][slot], sequence, n)
                    deadline = time.monotonic() + 30
                    while bytes(buffers[slot][n : n + FOOTER].tolist()) != stamp:
                        if state.abort_event.is_set() or time.monotonic() > deadline:
                            raise TimeoutError("D2D Host payload arrival unproven")
                        time.sleep(0.001)
                if r._layer_dropped(scheduler, layer):
                    state.error = f"layer {layer} dropped by test hook"
                else:
                    delay = r._layer_delay_s()
                    if delay:
                        time.sleep(delay)
                    with torch.cuda.stream(stream):
                        dtype = getattr(torch, meta["dtype"])
                        data = (
                            buffers[slot][:n]
                            .to(device, non_blocking=True, copy=True)
                            .view(dtype)
                            .reshape(2, hi - lo, meta["kv_heads"], meta["head_dim"])
                        )
                        pool.get_key_buffer(layer)[indices[lo:hi]] = data[0]
                        pool.get_value_buffer(layer)[indices[lo:hi]] = data[1]
                    stream.synchronize()
                    if coverage.add(layer, lo, hi):
                        state.layer_ready_at[layer] = time.monotonic()
                        state.events[layer].set()
                    state.layer_bytes[layer] = (state.layer_bytes[layer] or 0) + n
                r._send_json(conn, dict(free=slot, sequence=sequence))
        final = r._recv_json(conn)
        complete = final == dict(done=True)
        if not complete:
            raise ValueError("D2D final transfer proof missing")
        r._finish_layer_import(state)
    except Exception as exc:
        r._record_local_import_error(state, repr(exc))
    finally:
        if stream is not None:
            try:
                stream.synchronize()
            except Exception as exc:
                gpu_safe = False
                state.receiver_unsafe = True
                state.error = f"D2D target GPU completion unknown: {exc!r}"
        r._release_transfer_buffers(
            owner,
            buffers,
            state.error or "D2D ranges incomplete",
            remote_complete=complete,
            gpu_safe=gpu_safe,
        )
        conn.close()
