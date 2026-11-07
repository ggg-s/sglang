"""Native Mooncake GPU payloads; RemoteHub TCP carries control frames only.

One bounded lane per D/source endpoint. Unknown DMA completion quarantines the
registered allocation; neither a socket close nor an epoch permits reuse.
"""

import os
import secrets
import socket
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from . import mooncake_backend as transport

WIRE_VERSION = "remote-mooncake-v1"
FOOTER = 32

Q_BYTES = 2 * 1024 * 1024
O_BYTES = 4 * 1024 * 1024
LSE_BYTES = 16 * 1024
TOTAL = Q_BYTES + O_BYTES + LSE_BYTES
OFFSETS = (0, Q_BYTES, Q_BYTES + O_BYTES)
CAPACITIES = (Q_BYTES, O_BYTES, LSE_BYTES)
WIRE_OFFSETS = (0, Q_BYTES + FOOTER, Q_BYTES + O_BYTES + 2 * FOOTER)
WIRE_TOTAL = TOTAL + 3 * FOOTER
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def enabled():
    value = os.environ.get("HANDOFF_REMOTE_GPU_TRANSFER", "0")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_REMOTE_GPU_TRANSFER must be 0 or 1")
    return value == "1"


def manager_for(scheduler):
    name = (
        "disagg_prefill_bootstrap_queue"
        if scheduler.disaggregation_mode.value == "prefill"
        else "disagg_decode_prealloc_queue"
    )
    return getattr(scheduler, name).kv_manager


class Visibility:
    def __init__(self, device, protocol=None):
        self.device = torch.device(device)
        self.flush = None
        if self.device.type == "cpu":  # explicit CPU contract-test boundary
            return
        if protocol == "tcp" or (
            protocol is None
            and (
                os.environ.get("MC_FORCE_TCP") == "1"
                or os.environ.get("MOONCAKE_PROTOCOL") == "tcp"
            )
        ):
            return  # Native CUDA copy completion, not external RDMA writes.
        from cuda.bindings import runtime as cuda

        with torch.cuda.device(self.device):
            attr = cuda.cudaDeviceAttr
            rc, ordering = cuda.cudaDeviceGetAttribute(
                attr.cudaDevAttrGPUDirectRDMAWritesOrdering, self.device.index
            )
            if int(rc):
                raise RuntimeError("cannot query GPUDirect write ordering")
            scope = cuda.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner
            if int(ordering) >= int(scope):
                return
            rc, options = cuda.cudaDeviceGetAttribute(
                attr.cudaDevAttrGPUDirectRDMAFlushWritesOptions, self.device.index
            )
            if int(rc) or not int(options) & 1:
                raise RuntimeError(
                    "GPU payload transport lacks receive-side RDMA visibility support"
                )
            target = cuda.cudaFlushGPUDirectRDMAWritesTarget.cudaFlushGPUDirectRDMAWritesTargetCurrentDevice

            def flush():
                result = cuda.cudaDeviceFlushGPUDirectRDMAWrites(target, scope)
                if int(result[0]):
                    raise RuntimeError("GPUDirect receive visibility failed")

            self.flush = flush

    def receive(self):
        if self.flush is not None:
            with torch.cuda.device(self.device):
                self.flush()


def event(device):
    if torch.device(device).type == "cpu":
        return SimpleNamespace(query=lambda: True, synchronize=lambda: None)
    result = torch.cuda.Event()
    result.record(torch.cuda.current_stream(device))
    return result


class RequestIO:
    def __init__(self):
        self.lock = threading.Lock()
        self.pending = 0
        self.events = []
        self.unsafe = False

    def begin(self):
        with self.lock:
            if self.unsafe:
                raise RuntimeError("request GPU transport is quarantined")
            self.pending += 1

    def finish(self, proof):
        with self.lock:
            self.pending -= 1
            self.events.append(proof)

    def poison(self):
        with self.lock:
            self.unsafe = True

    def ready(self):
        with self.lock:
            if self.unsafe:
                raise RuntimeError("unproven GPU DMA; retaining request/cache leases")
            self.events = [e for e in self.events if not e.query()]
            return not self.pending and not self.events


def io_for(req):
    result = getattr(req, "_remote_attn_gpu_io", None)
    if result is None:
        result = req._remote_attn_gpu_io = RequestIO()
    return result


def ready(req):
    io = getattr(req, "_remote_attn_gpu_io", None)
    return io is None or io.ready()


class Slab:
    def __init__(self, plane, backend=transport.TCP):
        self.plane, self.backend = plane, backend
        self.engine = plane.catalog.engines[backend]
        self.nonce = secrets.token_hex(16)
        self.closed = self.quarantined = False
        self.stream = None
        self.copy_owner = None
        self.tensor = torch.empty(TOTAL, dtype=torch.uint8, device=plane.device)
        self.initialized = event(plane.device)
        self.wire = (
            self.tensor
            if backend == transport.GPU
            else torch.empty(
                WIRE_TOTAL,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=plane.device.type == "cuda",
            )
        )
        self.offsets = OFFSETS if backend == transport.GPU else WIRE_OFFSETS
        self.visibility = (
            Visibility(plane.device, protocol="rdma")
            if backend == transport.GPU
            else None
        )
        # Durable engine ownership BEFORE any registration can partially succeed.
        with plane.lock:
            retained = self.engine.__dict__.setdefault("_qo_slabs", [])
            retained.append(self)
        try:
            self.initialized.synchronize()
            if plane.device.type == "cuda":
                self.stream = torch.cuda.Stream(device=plane.device)
            with plane.send_guard():
                rc = self.engine.batch_register(
                    [self.wire.data_ptr()], [self.wire.numel()]
                )
            if rc != 0:
                raise RuntimeError("Mooncake Q/O registration failed")
        except BaseException:
            self.poison()
            raise

    def descriptor(self):
        return dict(
            base=self.wire.data_ptr(),
            size=self.wire.numel(),
            nonce=self.nonce,
            backend=self.backend,
        )

    def view(self, region, spec):
        shape, dtype, size = tensor_spec(spec, region)
        return self.tensor.narrow(0, OFFSETS[region], size).view(dtype).reshape(shape)

    def stamp(self, nonce, sequence, region, size):
        if type(sequence) is not int or not 0 < sequence < 2**64:
            raise ValueError("invalid Q/O transfer sequence")
        return (
            bytes.fromhex(nonce)
            + sequence.to_bytes(8, "little")
            + ((size << 2) | region).to_bytes(8, "little")
        )

    def copy_context(self):
        return (
            torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()
        )

    def drain(self):
        if self.stream is not None:
            self.stream.synchronize()

    def prepare(self, remote, entries, sequence):
        sizes = [tensor_spec(value, region)[2] for region, value in entries]
        if self.backend == transport.TCP:
            producer = event(self.plane.device)
            with self.copy_context():
                if self.stream is not None:
                    self.stream.wait_event(producer)
                for (region, _), size in zip(entries, sizes):
                    self.wire[self.offsets[region] : self.offsets[region] + size].copy_(
                        self.tensor[OFFSETS[region] : OFFSETS[region] + size],
                        non_blocking=True,
                    )
            self.drain()  # Independent D2H must finish before native reads Host.
            for (region, _), size in zip(entries, sizes):
                offset = self.offsets[region] + size
                self.wire[offset : offset + FOOTER].copy_(
                    torch.tensor(
                        list(self.stamp(remote["nonce"], sequence, region, size)),
                        dtype=torch.uint8,
                    )
                )
            sizes = [size + FOOTER for size in sizes]
        return sizes

    def receive(self, entries, sequence):
        if self.quarantined or self.closed:
            raise RuntimeError("Q/O slot is quarantined/closed")
        if self.backend == transport.GPU:
            self.visibility.receive()
            return
        # Each region's payload/footer is ONE ordered TCP descriptor. Native
        # sender completion alone does not prove receiver bytes have arrived.
        deadline = time.monotonic() + float(
            os.environ.get("HANDOFF_PD_DRAIN_TIMEOUT_S", "300")
        )
        for region, value in entries:
            size = tensor_spec(value, region)[2]
            offset = self.offsets[region] + size
            stamp = self.stamp(self.nonce, sequence, region, size)
            while bytes(self.wire[offset : offset + FOOTER].tolist()) != stamp:
                if self.quarantined or time.monotonic() >= deadline:
                    raise TimeoutError("Q/O Host payload arrival unproven")
                time.sleep(0.001)
        with self.copy_context():
            if self.stream is not None:
                self.stream.wait_event(self.initialized)
            for region, value in entries:
                size = tensor_spec(value, region)[2]
                self.tensor[OFFSETS[region] : OFFSETS[region] + size].copy_(
                    self.wire[self.offsets[region] : self.offsets[region] + size],
                    non_blocking=True,
                )
        self.drain()  # Prove H2D before another stream consumes or sender reuses.

    def close(self):
        if self.closed or self.quarantined:
            return
        try:
            self.drain()
            with self.plane.send_guard():
                rc = self.engine.batch_deregister([self.wire.data_ptr()])
            if rc != 0:
                raise RuntimeError("Mooncake Q/O deregistration failed")
        except BaseException:
            self.poison()
            raise
        self.closed = True
        with self.plane.lock:
            self.engine._qo_slabs.remove(self)

    def poison(self):
        if not self.quarantined:
            self.quarantined = True
            self.plane.quarantine.append(self)


def tensor_spec(spec, region):
    shape = spec["shape"]
    if (
        not isinstance(shape, list)
        or not 1 <= len(shape) <= 3
        or any(type(n) is not int or not 0 < n <= 65536 for n in shape)
    ):
        raise ValueError("invalid GPU tensor shape")
    dtype = DTYPES[spec["dtype"]]
    count = 1
    for n in shape:
        count *= n
    size = count * torch.empty((), dtype=dtype).element_size()
    if (
        size > CAPACITIES[region]
        or (region == 0 and dtype not in (torch.bfloat16, torch.float16))
        or (region == 2 and dtype != torch.float32)
    ):
        raise ValueError("GPU tensor exceeds slot/dtype contract")
    return shape, dtype, size


def spec(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype).split(".")[-1])


def descriptor(value, backend=None):
    if (
        not isinstance(value, dict)
        or value.get("backend") not in (transport.GPU, transport.TCP)
        or (backend is not None and value.get("backend") != backend)
        or type(value.get("base")) is not int
        or value["base"] <= 0
        or value.get("size")
        != (TOTAL if value.get("backend") == transport.GPU else WIRE_TOTAL)
        or not isinstance(value.get("nonce"), str)
        or len(value["nonce"]) != 32
        or any(c not in "0123456789abcdef" for c in value["nonce"])
    ):
        raise ValueError("invalid registered GPU region")
    return value


def source_for(ctx):
    return getattr(ctx, "dsv41_source", None) or getattr(
        ctx.req, "_pre_layerwise_source", None
    )


def allowed(ctx, session):
    source = source_for(ctx)
    return source is not None and any(
        peer.mooncake_session_id == session for peer, _ in source.peers
    )


class Plane:
    def __init__(self, manager, device):
        self.manager, self.engine, self.device = (
            manager,
            manager.engine,
            torch.device(device),
        )
        self.catalog = transport.Catalog(
            self.engine, self.device, policy_name="HANDOFF_REMOTE_BACKEND"
        )
        self.engine = self.catalog.primary
        self.session = self.engine.session  # Native PD identity, not fallback endpoint.
        for engine in self.catalog.engines.values():
            for name in ("batch_transfer_async_write", "get_batch_transfer_status"):
                if not callable(getattr(engine.raw, name, None)):
                    raise RuntimeError(
                        f"Mooncake Q/O requires single-submit API {name}"
                    )
        self.clients, self.lanes, self.quarantine = {}, set(), []
        self.lock = threading.RLock()
        self.closed = False
        self.transfers = self.payload_bytes = 0
        print(
            f"[remote-gpu] native Mooncake GPU payloads session={self.session}",
            flush=True,
        )

    def send_guard(self):
        lock = getattr(self.manager, "_remote_attn_send_lock", None)
        if lock is not None and hasattr(lock, "priority"):
            return lock.priority()
        return lock if lock is not None else nullcontext()

    def stats(self):
        with self.lock:
            return dict(
                transfers=self.transfers,
                payload_bytes=self.payload_bytes,
                clients=len(self.clients),
                lanes=len(self.lanes),
                quarantined=len(self.quarantine),
                tcp_tensor_payloads=0,
                send_lock_wait_ns=list(
                    getattr(
                        getattr(self.manager, "_remote_attn_send_lock", None),
                        "wait_ns",
                        (),
                    )
                ),
            )

    def write(self, slab, remote, session, entries, sequence):
        remote = descriptor(remote, slab.backend)
        if slab.quarantined or slab.closed:
            raise RuntimeError("GPU slot is not reusable")
        try:
            lengths = slab.prepare(remote, entries, sequence)
            local = [
                slab.wire.data_ptr() + slab.offsets[region] for region, _ in entries
            ]
            target = [remote["base"] + slab.offsets[region] for region, _ in entries]
            with self.send_guard():
                slab.engine.batch_transfer_sync(session, local, target, lengths)
        except BaseException:
            slab.poison()  # Includes partial submit, hidden receiver work and CUDA failure.
            raise
        with self.lock:
            self.transfers += 1
            self.payload_bytes += sum(lengths)

    def client(self, route):
        key = route["host"], int(route["port"])
        with self.lock:
            if self.closed:
                raise RuntimeError("GPU data plane closed")
            client = self.clients.get(key)
            if client is not None and (
                client.closed or (client.slab is not None and client.slab.quarantined)
            ):
                # A new request can use fresh backing memory. The failed
                # request's IO remains poisoned and may never be retried here.
                client = None
            if client is None:
                if (len(self.clients) >= 16 and key not in self.clients) or len(
                    self.quarantine
                ) >= 16:
                    raise RuntimeError("GPU peer lane limit reached")
                client = self.clients[key] = Client(self, route)
            return client

    def close(self):
        self.closed = True
        for client in list(self.clients.values()):
            client.close()
        for lane in list(self.lanes):
            lane.drop()


def plane_for(controller):
    result = getattr(controller, "gpu_data_plane", None)
    if result is None:
        manager = manager_for(controller.scheduler)
        result = controller.gpu_data_plane = Plane(
            manager, torch.device("cuda", torch.cuda.current_device())
        )
    return result


class Client:
    def __init__(self, plane, route):
        self.plane, self.route = plane, route
        self.slab = None  # Allocate/register only after a common backend is selected.
        self.lock = threading.Lock()
        self.sock, self.remote, self.session = None, None, None
        self.sequence = 0
        self.consumed = None
        self.closed = False

    def rpc(self, frame):
        from .handoff_control import _recv_json, _send_json

        _send_json(self.sock, frame)
        result = _recv_json(self.sock)
        if not result.get("ok"):
            raise RuntimeError(f"GPU Remote rejected: {result.get('error')}")
        return result

    def connect(self):
        try:
            self._connect()
        except BaseException:
            self.closed = True
            if self.slab is not None:
                self.slab.poison()
            if self.sock is not None:
                self.sock.close()
            raise

    def _connect(self):
        self.sock = socket.create_connection(
            (self.route["host"], int(self.route["port"])), timeout=10
        )
        self.sock.settimeout(float(os.environ.get("HANDOFF_PD_DRAIN_TIMEOUT_S", "300")))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.rpc(dict(cmd="hello", proto=2))
        result = self.rpc(
            dict(
                cmd="te_negotiate",
                version=WIRE_VERSION,
                route=self.route,
                native_session=self.plane.session,
                capabilities=self.plane.catalog.offer,
                sessions={
                    key: engine.session
                    for key, engine in self.plane.catalog.engines.items()
                },
            )
        )
        backend = self.plane.catalog.select(result["capabilities"])
        if result.get("version") != WIRE_VERSION or result.get("backend") != backend:
            raise ValueError("Q/O backend negotiation mismatch")
        self.session = result["session"]
        self.slab = Slab(self.plane, backend)
        result = self.rpc(
            dict(
                cmd="te_open",
                version=WIRE_VERSION,
                backend=backend,
                session=self.slab.engine.session,
                region=self.slab.descriptor(),
            )
        )
        self.remote = descriptor(result["region"], backend)
        if (
            result.get("session") != self.session
            or result.get("version") != WIRE_VERSION
        ):
            raise ValueError("Q/O engine identity changed during open")

    def call(self, queries, rows, reqs):
        if (
            not 1 <= len(rows) <= 4
            or len(rows) != len(queries)
            or len(reqs) != len(rows)
        ):
            raise ValueError("GPU payload requires 1..4 mapped rows")
        ios = [io_for(req) for req in reqs]
        with self.lock:
            if self.closed or (self.slab is not None and self.slab.quarantined):
                raise RuntimeError("GPU lane is quarantined/closed")
            try:
                for io in ios:
                    io.begin()
                if self.sock is None:
                    self.connect()
                if self.consumed is not None:
                    self.consumed.synchronize()
                    self.slab.copy_owner = None
                if any(q.device != self.plane.device for q in queries):
                    raise ValueError("Q belongs to another GPU")
                query = torch.stack(queries)
                query_spec = spec(query)
                self.slab.copy_owner = query
                self.slab.view(0, query_spec).copy_(query)
                event(self.plane.device).synchronize()
                self.slab.copy_owner = None
                self.plane.write(
                    self.slab,
                    self.remote,
                    self.session,
                    [(0, query_spec)],
                    self.sequence + 1,
                )
                self.sequence += 1
                result = self.rpc(
                    dict(cmd="te_attn", sequence=self.sequence, q=query_spec, rows=rows)
                )
                if result.get("sequence") != self.sequence:
                    raise ValueError("GPU result sequence changed")
                self.slab.receive([(1, result["o"]), (2, result["lse"])], self.sequence)
                # Return owned tensors; the arena never aliases a future graph
                # segment. Reuse waits for the final slot->owned D2D reads.
                self.slab.copy_owner = []
                out = self.slab.view(1, result["o"]).clone()
                self.slab.copy_owner.append(out)
                lse = self.slab.view(2, result["lse"]).clone()
                self.slab.copy_owner.append(lse)
                if out.shape != query.shape or lse.shape != query.shape[:-1]:
                    raise ValueError("GPU result shape changed")
                self.consumed = event(self.plane.device)
                self.rpc(dict(cmd="te_ack", sequence=self.sequence))
                for io in ios:
                    io.finish(self.consumed)
                return result, out, lse
            except BaseException:
                if self.slab is not None:
                    self.slab.poison()
                for io in ios:
                    io.poison()
                if self.sock is not None:
                    self.sock.close()
                self.closed = True
                raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.slab is None:
                if self.sock is not None:
                    self.sock.close()
                return
            if self.slab.quarantined:
                return
            try:
                if self.consumed is not None:
                    self.consumed.synchronize()
                if self.sock is not None:
                    self.rpc(dict(cmd="te_close", sequence=self.sequence))
                    self.sock.close()
                self.slab.copy_owner = None
                self.slab.close()
            except Exception:
                self.slab.poison()
                raise


class Lane:
    def __init__(
        self, plane, remote, session, backend=transport.TCP, native_session=None
    ):
        self.plane, self.remote, self.session = (
            plane,
            descriptor(remote, backend),
            session,
        )
        self.native_session = native_session or session
        self.slab = Slab(plane, backend)
        self.sequence = 0
        self.state = "idle"
        self.lock = threading.Lock()

    def begin(self, sequence):
        with self.lock:
            if (
                self.state != "idle"
                or type(sequence) is not int
                or sequence != self.sequence + 1
            ):
                raise RuntimeError("GPU lane reused before ACK or stale sequence")
            self.sequence, self.state = sequence, "running"

    def ack(self, sequence):
        with self.lock:
            if self.state != "returned" or sequence != self.sequence:
                raise RuntimeError("GPU lane ACK is stale/unproven")
            self.state = "idle"

    def drop(self):
        with self.lock:
            if self.state != "closed":
                self.state = "quarantined"
                self.slab.poison()


class Job:
    gpu_transport = True

    def __init__(self, hub, hc, lane, header, contexts):
        self.hub, self.hc, self.lane, self.header = hub, hc, lane, header
        self.contexts, self.stream = contexts, None
        self.done = False
        self.unsafe = False

    def execute(self, stream, cache):
        from . import dsv41_remote
        from .handoff_control import execute_remote_attention_batch

        plane = self.lane.plane
        self.lane.slab.receive([(0, self.header["q"])], self.lane.sequence)
        with (
            torch.cuda.stream(stream) if plane.device.type == "cuda" else nullcontext()
        ):
            query = self.lane.slab.view(0, self.header["q"])
            items = [
                SimpleNamespace(
                    hub=self.hub,
                    ctx=ctx,
                    epoch=ctx.owner_epoch,
                    header=row,
                    q_cpu=query[i],
                    gpu_transport=True,
                    rpc_id=self.lane.sequence,
                    recv_delay_ms=0,
                    recv_ms=0,
                )
                for i, (ctx, row) in enumerate(zip(self.contexts, self.header["rows"]))
            ]
            if any(row.header.get("qwen_hybrid") for row in items):
                from .qwen_hybrid import execute

                if not all(row.header.get("qwen_hybrid") for row in items):
                    raise ValueError("GPU RPC spans different model contracts")
                outs = [execute(row, stream, cache) for row in items]
            elif all(row.header.get("dsv41_sparse") for row in items):
                outs = [dsv41_remote.execute(row, stream, cache) for row in items]
            elif any(row.header.get("dsv41_sparse") for row in items):
                raise ValueError("GPU RPC spans different model contracts")
            else:
                _, outs = execute_remote_attention_batch(
                    SimpleNamespace(
                        hub=self.hub,
                        items=items,
                        rpc_id=self.lane.sequence,
                        gpu_transport=True,
                    ),
                    stream,
                    cache,
                )
            out, lse = (
                torch.stack([o for _, o, _ in outs]),
                torch.stack([l for _, _, l in outs]),
            )
            self.lane.slab.copy_owner = (outs, out, lse)
            o_spec, l_spec = spec(out), spec(lse)
            self.lane.slab.view(1, o_spec).copy_(out)
            self.lane.slab.view(2, l_spec).copy_(lse)
            event(plane.device).synchronize()
            self.lane.slab.copy_owner = None
            plane.write(
                self.lane.slab,
                self.lane.remote,
                self.lane.session,
                [(1, o_spec), (2, l_spec)],
                self.lane.sequence,
            )
        with self.lane.lock:
            if self.lane.state != "running":
                raise RuntimeError("GPU connection dropped during DMA")
            self.lane.state = "returned"
        self.hub._send(
            self.hc,
            dict(
                ok=True,
                cmd="te_attn",
                sequence=self.lane.sequence,
                o=o_spec,
                lse=l_spec,
                rows=[r for r, _, _ in outs],
            ),
        )

    def fail(self, exc):
        self.unsafe = True
        self.lane.drop()
        with self.hub._stats_lock:
            self.hub.rpc_failed += 1
        for ctx in self.contexts:
            ctx.req._handoff_transfer_unsafe = True
        try:
            self.hub._send(self.hc, dict(ok=False, cmd="te_attn", error=str(exc)))
        except OSError:
            pass

    def finish(self):
        from .handoff_control import _fence_remote_job

        if self.done:
            return
        self.done = True
        if self.unsafe or not _fence_remote_job(self, self.contexts):
            return
        for ctx in self.contexts:
            self.hub.registry.release_ref(ctx.handoff_id, ctx.owner_epoch)
        with self.hc.lock:
            self.hc.in_flight = max(0, self.hc.in_flight - 1)
        self.hub._job_finished(self.hc)
        with self.hub._stats_lock:
            self.hub.rpc_completed += 1


def handle(hub, hc, frame):
    import hmac

    controller = getattr(hub.scheduler, "_pre_layerwise", None)
    if controller is None or not enabled():
        raise RuntimeError("GPU payload transport is disabled")
    plane = plane_for(controller)
    command = frame["cmd"]
    if command == "te_negotiate":
        if (
            frame.get("version") != WIRE_VERSION
            or hc.gpu_negotiation is not None
            or hc.gpu_lane is not None
        ):
            raise ValueError("Q/O negotiation version/state mismatch")
        route = frame["route"]
        ctx = hub.registry.get(route["handoff_id"], route["epoch"])
        native_session = frame["native_session"]
        if (
            ctx is None
            or not hmac.compare_digest(str(route["token"]), ctx.token)
            or not allowed(ctx, native_session)
        ):
            raise RuntimeError("GPU lane is not bound to the native PD peer")
        backend = plane.catalog.select(frame["capabilities"])
        sessions = frame.get("sessions")
        if not isinstance(sessions, dict) or set(sessions) != set(
            frame["capabilities"]["backends"]
        ):
            raise ValueError("Q/O capability endpoints missing")
        session = sessions.get(backend)
        if not isinstance(session, str) or not 0 < len(session) <= 512:
            raise ValueError("Q/O capability endpoint invalid")
        hc.gpu_negotiation = (backend, native_session, session)
        hub._send(
            hc,
            dict(
                ok=True,
                cmd=command,
                version=WIRE_VERSION,
                capabilities=plane.catalog.offer,
                backend=backend,
                session=plane.catalog.engines[backend].session,
            ),
        )
        return
    if command == "te_open":
        if hc.gpu_negotiation is None or frame.get("version") != WIRE_VERSION:
            raise ValueError("Q/O open requires capability negotiation")
        backend, native_session, session = hc.gpu_negotiation
        if (
            frame.get("backend") != backend
            or frame.get("session") != session
            or not isinstance(frame.get("session"), str)
            or not 0 < len(frame["session"]) <= 512
        ):
            raise ValueError("Q/O negotiated backend/session changed")
        with plane.lock:
            if len(plane.lanes) >= 16 or hc.gpu_lane is not None:
                raise RuntimeError("GPU lane capacity/binding exceeded")
            lane = hc.gpu_lane = Lane(
                plane, frame["region"], frame["session"], backend, native_session
            )
            plane.lanes.add(lane)
        hub._send(
            hc,
            dict(
                ok=True,
                cmd=command,
                version=WIRE_VERSION,
                region=lane.slab.descriptor(),
                session=lane.slab.engine.session,
            ),
        )
        return
    lane = getattr(hc, "gpu_lane", None)
    if lane is None:
        raise RuntimeError("GPU command has no registered lane")
    if command == "te_ack":
        lane.ack(frame["sequence"])
        hub._send(hc, dict(ok=True, cmd=command))
    elif command == "te_close":
        with lane.lock:
            if lane.state != "idle" or frame["sequence"] != lane.sequence:
                raise RuntimeError("cannot deregister an active GPU lane")
            lane.slab.close()
            lane.state = "closed"
        hub._send(hc, dict(ok=True, cmd=command))
        with plane.lock:
            plane.lanes.discard(lane)
    elif command == "te_attn":
        rows = frame["rows"]
        shape, _, _ = tensor_spec(frame["q"], 0)
        if len(shape) != 3 or not 1 <= len(rows) <= 4 or shape[0] != len(rows):
            raise ValueError("GPU RPC row/shape mismatch")
        lane.begin(frame["sequence"])
        contexts = []
        try:
            for row in rows:
                ctx = hub.registry.acquire(row["handoff_id"], row["epoch"])
                if ctx is None:
                    raise RuntimeError("stale GPU source lease")
                contexts.append(ctx)
                if not hmac.compare_digest(str(row["token"]), ctx.token) or not allowed(
                    ctx, lane.native_session
                ):
                    raise RuntimeError("GPU RPC peer capability changed")
            job = Job(hub, hc, lane, frame, contexts)
            with hc.lock:
                hc.in_flight += 1
            with hub._active_lock:
                hub._in_flight += 1
                hub.scheduler._handoff_remote_active = True
            try:
                hub._pool.submit(job)
            except BaseException:
                with hc.lock:
                    hc.in_flight = max(0, hc.in_flight - 1)
                hub._job_finished(hc)
                raise
        except BaseException:
            lane.drop()
            for ctx in contexts:
                hub.registry.release_ref(ctx.handoff_id, ctx.owner_epoch)
            raise
    else:
        raise ValueError("unknown GPU control command")


def call(controller, route, queries, headers, reqs):
    return plane_for(controller).client(route).call(queries, headers, reqs)


def mha_call(group):
    first = group[0]["state"]
    target = getattr(first.req, "_pre_layerwise_target", None)
    if target is None:
        raise RuntimeError("GPU MHA transport requires native progressive PD ownership")
    headers = [
        {**p["row_header"], "token": p["state"].remote_service["token"]} for p in group
    ]
    return call(
        target.owner,
        first.remote_service,
        [p["q_row"] for p in group],
        headers,
        [p["state"].req for p in group],
    )
