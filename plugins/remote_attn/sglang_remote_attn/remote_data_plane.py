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
MAX_ROWS = 4
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
            scope = (
                cuda.cudaFlushGPUDirectRDMAWritesScope.cudaFlushGPUDirectRDMAWritesToOwner
            )
            if int(ordering) >= int(scope):
                return
            rc, options = cuda.cudaDeviceGetAttribute(
                attr.cudaDevAttrGPUDirectRDMAFlushWritesOptions, self.device.index
            )
            if int(rc) or not int(options) & 1:
                raise RuntimeError(
                    "GPU payload transport lacks receive-side RDMA visibility support"
                )
            target = (
                cuda.cudaFlushGPUDirectRDMAWritesTarget.cudaFlushGPUDirectRDMAWritesTargetCurrentDevice
            )

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
            if proof is not None:
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
        # Only reused under the lane's exclusive write ownership. RequestIO
        # completion proofs have a different lifetime and stay single-use.
        self.producer = event(plane.device) if plane.device.type == "cuda" else None
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
            with plane.send_guard(shared=True):
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

    def view(self, region, spec, *, offset=None):
        shape, dtype, size = tensor_spec(spec, region)
        offset = OFFSETS[region] if offset is None else offset
        return self.tensor.narrow(0, offset, size).view(dtype).reshape(shape)

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

    def record_producer(self):
        if self.producer is not None:
            self.producer.record(torch.cuda.current_stream(self.plane.device))
        return self.producer

    def wire_layout(self, entries, *, packed=False):
        sizes = [tensor_spec(value, region)[2] for region, value in entries]
        offsets = [self.offsets[region] for region, _ in entries]
        if packed:
            if [r for r, _ in entries] != [1, 2]:
                raise ValueError("packed return requires O/LSE regions")
            # Align GPU LSE for typed FP32 access. TCP keeps its exact byte
            # layout, including the existing arrival footer on the descriptor.
            offsets[1] = (
                packed_lse_offset(sizes[0])
                if self.backend == transport.GPU
                else offsets[0] + sizes[0]
            )
            spans = [(offsets[0], offsets[1] + sizes[1] - offsets[0], 3)]
        else:
            spans = [
                (offset, size, r)
                for offset, size, (r, _) in zip(offsets, sizes, entries)
            ]
        return sizes, offsets, spans

    def prepare(self, remote, entries, sequence, *, packed=False):
        sizes, offsets, spans = self.wire_layout(entries, packed=packed)
        producer = self.record_producer()
        if self.backend == transport.TCP:
            with self.copy_context():
                if self.stream is not None:
                    self.stream.wait_event(producer)
                for (region, _), size, offset in zip(entries, sizes, offsets):
                    self.wire[offset : offset + size].copy_(
                        self.tensor[OFFSETS[region] : OFFSETS[region] + size],
                        non_blocking=True,
                    )
            self.drain()  # Independent D2H must finish before native reads Host.
            for offset, size, tag in spans:
                self.wire[offset + size : offset + size + FOOTER].copy_(
                    torch.tensor(
                        list(self.stamp(remote["nonce"], sequence, tag, size)),
                        dtype=torch.uint8,
                    )
                )
        elif producer is not None:
            # RDMA reads the GPU slab directly. TCP already fenced its D2H.
            producer.synchronize()
        footer = FOOTER if self.backend == transport.TCP else 0
        return [(offset, size + footer) for offset, size, _ in spans]

    def receive(self, entries, sequence, *, owned=False, packed=False):
        if self.quarantined or self.closed:
            raise RuntimeError("Q/O slot is quarantined/closed")
        sizes, offsets, spans = self.wire_layout(entries, packed=packed)
        if self.backend == transport.GPU:
            self.visibility.receive()
            return
        # Each region's payload/footer is ONE ordered TCP descriptor. Native
        # sender completion alone does not prove receiver bytes have arrived.
        deadline = time.monotonic() + float(
            os.environ.get("HANDOFF_PD_DRAIN_TIMEOUT_S", "300")
        )
        for start, size, tag in spans:
            offset = start + size
            stamp = self.stamp(self.nonce, sequence, tag, size)
            while bytes(self.wire[offset : offset + FOOTER].tolist()) != stamp:
                if self.quarantined or time.monotonic() >= deadline:
                    raise TimeoutError("Q/O Host payload arrival unproven")
                time.sleep(0.001)
        outputs = []
        if owned:
            self.copy_owner = outputs
        with self.copy_context():
            if self.stream is not None:
                self.stream.wait_event(self.initialized)
            for (region, value), size, offset in zip(entries, sizes, offsets):
                shape, dtype, _ = tensor_spec(value, region)
                if owned:
                    dst = torch.empty(shape, device=self.plane.device, dtype=dtype)
                    outputs.append(dst)
                    dst = dst.view(torch.uint8).reshape(-1)
                else:
                    dst = self.tensor[OFFSETS[region] : OFFSETS[region] + size]
                dst.copy_(
                    self.wire[offset : offset + size],
                    non_blocking=True,
                )
        self.drain()  # Prove H2D before consumers run or the sender reuses Host.
        if owned:
            if self.stream is not None:
                consumer = torch.cuda.current_stream(self.plane.device)
                for output in outputs:
                    output.record_stream(consumer)
            self.copy_owner = None
            return outputs

    def close(self):
        if self.closed or self.quarantined:
            return
        try:
            self.drain()
            with self.plane.send_guard(shared=True):
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


def packed_lse_offset(o_size):
    return OFFSETS[1] + (o_size + 3) // 4 * 4


def pack(slab, region, tensors, *, offset=None):
    first = tensors[0]
    if any(t.shape != first.shape or t.dtype != first.dtype for t in tensors):
        raise ValueError("Q/O frame rows have different shapes or dtypes")
    value = spec(first)
    value["shape"].insert(0, len(tensors))
    dst = slab.view(region, value, offset=offset)
    if len(tensors) == 1:
        dst[0].copy_(first)
    else:
        torch.stack(tensors, out=dst)
    return value


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
        self._transfer_lock = threading.Lock()
        self.closed = False
        self.transfers = self.payload_bytes = 0
        self.transfer_regions = self.packed_returns = 0
        self.attention_frames = self.attention_rows = 0
        self.attention_ms = {}
        print(
            f"[remote-gpu] native Mooncake GPU payloads session={self.session}",
            flush=True,
        )

    def send_guard(self, shared=False):
        # Serialize Q/O traffic without queuing behind native KV pushes.
        # Registration lifetime changes still coordinate with peer publication.
        if not shared:
            return self._transfer_lock
        lock = getattr(self.manager, "_remote_attn_send_lock", None)
        if lock is not None and hasattr(lock, "priority"):
            return lock.priority()
        return lock if lock is not None else nullcontext()

    def stats(self):
        with self.lock:
            return dict(
                transfers=self.transfers,
                transfer_regions=self.transfer_regions,
                packed_returns=self.packed_returns,
                payload_bytes=self.payload_bytes,
                attention_frames=self.attention_frames,
                attention_rows=self.attention_rows,
                attention_ms=dict(self.attention_ms),
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

    def record_attention(self, rows, timings):
        with self.lock:
            self.attention_frames += 1
            self.attention_rows += rows
            for key, value in timings.items():
                self.attention_ms[key] = self.attention_ms.get(key, 0.0) + value

    def write(self, slab, remote, session, entries, sequence, *, packed=False):
        remote = descriptor(remote, slab.backend)
        if slab.quarantined or slab.closed:
            raise RuntimeError("GPU slot is not reusable")
        started = time.perf_counter()
        try:
            spans = slab.prepare(remote, entries, sequence, packed=packed)
            local = [slab.wire.data_ptr() + offset for offset, _ in spans]
            target = [remote["base"] + offset for offset, _ in spans]
            lengths = [size for _, size in spans]
            prepared = time.perf_counter()
            with self.send_guard():
                submitted = time.perf_counter()
                slab.engine.batch_transfer_sync(session, local, target, lengths)
            completed = time.perf_counter()
        except BaseException:
            slab.poison()  # Includes partial submit, hidden receiver work and CUDA failure.
            raise
        with self.lock:
            self.transfers += 1
            self.payload_bytes += sum(lengths)
            self.transfer_regions += len(lengths)
            self.packed_returns += int(packed)
        return dict(
            source_return_prepare=(prepared - started) * 1000,
            source_return_slot_wait=(submitted - prepared) * 1000,
            source_return_transfer=(completed - submitted) * 1000,
        )

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
        self.inline_ack = False
        self.packed_return = False
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
                inline_ack=True,
                packed_return=True,
                gpu_packed_return=True,
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
        self.inline_ack = result.get("inline_ack") is True
        self.packed_return = result.get("packed_return") is True
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
            not 1 <= len(rows) <= MAX_ROWS
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
                t0 = time.perf_counter()
                if self.consumed is not None:
                    self.consumed.synchronize()
                    self.slab.copy_owner = None
                if any(q.device != self.plane.device for q in queries):
                    raise ValueError("Q belongs to another GPU")
                self.slab.copy_owner = queries
                query_spec = pack(self.slab, 0, queries)
                t_pack = time.perf_counter()
                self.plane.write(
                    self.slab,
                    self.remote,
                    self.session,
                    [(0, query_spec)],
                    self.sequence + 1,
                )
                self.slab.copy_owner = None
                t_write = time.perf_counter()
                frame = dict(
                    cmd="te_attn", sequence=self.sequence + 1, q=query_spec, rows=rows
                )
                if self.inline_ack and self.sequence:
                    frame["ack_sequence"] = self.sequence
                self.sequence += 1
                result = self.rpc(frame)
                t_reply = time.perf_counter()
                if result.get("sequence") != self.sequence:
                    raise ValueError("GPU result sequence changed")
                entries = [(1, result["o"]), (2, result["lse"])]
                if self.slab.backend == transport.TCP:
                    # Copy registered Host data straight into owned outputs.
                    # The H2D fence in receive proves the slab has no readers.
                    out, lse = self.slab.receive(
                        entries, self.sequence, owned=True, packed=self.packed_return
                    )
                    self.consumed = None
                else:
                    self.slab.receive(entries, self.sequence)
                    # RDMA writes the registered GPU slab. Keep the backing
                    # and its consumption proof until the owned D2D reads finish.
                    self.slab.copy_owner = []
                    if self.packed_return:
                        sizes, offsets, spans = self.slab.wire_layout(
                            entries, packed=True
                        )
                        start, size, _tag = spans[0]
                        owned = self.slab.tensor.narrow(0, start, size).clone()
                        self.slab.copy_owner.append(owned)
                        out, lse = [
                            owned.narrow(0, offset - start, length)
                            .view(tensor_spec(value, region)[1])
                            .reshape(value["shape"])
                            for (region, value), offset, length in zip(
                                entries, offsets, sizes
                            )
                        ]
                    else:
                        out = self.slab.view(1, result["o"]).clone()
                        self.slab.copy_owner.append(out)
                        lse = self.slab.view(2, result["lse"]).clone()
                        self.slab.copy_owner.append(lse)
                    self.consumed = event(self.plane.device)
                shape = tuple(query_spec["shape"])
                if out.shape != shape or lse.shape != shape[:-1]:
                    raise ValueError("GPU result shape changed")
                t_receive = time.perf_counter()
                if not self.inline_ack:
                    self.rpc(dict(cmd="te_ack", sequence=self.sequence))
                for io in ios:
                    io.finish(self.consumed)
                self.plane.record_attention(
                    len(rows),
                    dict(
                        pack=(t_pack - t0) * 1000,
                        write=(t_write - t_pack) * 1000,
                        rpc=(t_reply - t_write) * 1000,
                        receive=(t_receive - t_reply) * 1000,
                        ack=(time.perf_counter() - t_receive) * 1000,
                        **result.get("source_timing_ms", {}),
                    ),
                )
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
                    frame = dict(cmd="te_close", sequence=self.sequence)
                    if self.inline_ack and self.sequence:
                        frame["ack_sequence"] = self.sequence
                    self.rpc(frame)
                    self.sock.close()
                self.slab.copy_owner = None
                self.slab.close()
            except Exception:
                self.slab.poison()
                raise


class Lane:
    def __init__(
        self,
        plane,
        remote,
        session,
        backend=transport.TCP,
        native_session=None,
        inline_ack=False,
        packed_return=False,
    ):
        self.plane, self.remote, self.session = (
            plane,
            descriptor(remote, backend),
            session,
        )
        self.native_session = native_session or session
        self.inline_ack = inline_ack
        self.packed_return = packed_return
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
            if (
                self.state != "returned"
                or type(sequence) is not int
                or sequence != self.sequence
            ):
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
        self.return_fenced = False

    def execute(self, stream, cache):
        from . import dsv41_remote
        from .handoff_control import execute_remote_attention_batch

        plane = self.lane.plane
        t_start = time.perf_counter()
        self.lane.slab.receive([(0, self.header["q"])], self.lane.sequence)
        t_receive = time.perf_counter()
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
                    fence_output_at_write=True,
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
            slab = self.lane.slab
            slab.copy_owner = outs
            o_spec = pack(slab, 1, [o for _, o, _ in outs])
            lse_offset = None
            if slab.backend == transport.GPU and self.lane.packed_return:
                o_size = tensor_spec(o_spec, 1)[2]
                o_end = OFFSETS[1] + o_size
                lse_offset = packed_lse_offset(o_size)
                if o_end != lse_offset:
                    slab.tensor[o_end:lse_offset].zero_()
            l_spec = pack(slab, 2, [l for _, _, l in outs], offset=lse_offset)
            t_compute = time.perf_counter()
            return_timing = plane.write(
                slab,
                self.lane.remote,
                self.lane.session,
                [(1, o_spec), (2, l_spec)],
                self.lane.sequence,
                packed=self.lane.packed_return,
            )
            # A successful write already fenced the attention stream before
            # D2H/RDMA. Registry release need not synchronize it a second time.
            self.return_fenced = True
            slab.copy_owner = None
        t_write = time.perf_counter()
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
                source_timing_ms=dict(
                    source_queue=(t_start - self.t_enqueue) * 1000,
                    source_receive=(t_receive - t_start) * 1000,
                    source_compute_submit=(t_compute - t_receive) * 1000,
                    source_return=(t_write - t_compute) * 1000,
                    **return_timing,
                ),
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
        if self.unsafe or (
            not self.return_fenced and not _fence_remote_job(self, self.contexts)
        ):
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
        inline_ack = frame.get("inline_ack") is True
        packed_return = frame.get("packed_return") is True and (
            backend == transport.TCP or frame.get("gpu_packed_return") is True
        )
        hc.gpu_negotiation = (
            backend,
            native_session,
            session,
            inline_ack,
            packed_return,
        )
        hub._send(
            hc,
            dict(
                ok=True,
                cmd=command,
                version=WIRE_VERSION,
                capabilities=plane.catalog.offer,
                inline_ack=inline_ack,
                packed_return=packed_return,
                backend=backend,
                session=plane.catalog.engines[backend].session,
            ),
        )
        return
    if command == "te_open":
        if hc.gpu_negotiation is None or frame.get("version") != WIRE_VERSION:
            raise ValueError("Q/O open requires capability negotiation")
        backend, native_session, session, inline_ack, packed_return = hc.gpu_negotiation
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
                plane,
                frame["region"],
                frame["session"],
                backend,
                native_session,
                inline_ack=inline_ack,
                packed_return=packed_return,
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
    if "ack_sequence" in frame:
        if not lane.inline_ack or command not in ("te_attn", "te_close"):
            raise ValueError("inline Q/O ACK was not negotiated")
        lane.ack(frame["ack_sequence"])
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
        if len(shape) != 3 or not 1 <= len(rows) <= MAX_ROWS or shape[0] != len(rows):
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
