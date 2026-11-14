"""Bounded D2D bulk slots on the native Mooncake engine.

RemoteHub authenticates and sequences requests; it carries no KV payload.
An uncertain write or response permanently quarantines its registered backing.
Slots are private to a migration, independent of Remote Attention Q/O lanes.
"""

import secrets
import threading
import time
from types import SimpleNamespace

import torch

from . import mooncake_backend as transport
from . import remote_data_plane as dp

LEGACY_CAPACITY = 4 * 1024 * 1024
CAPACITY = 16 * 1024 * 1024
RDMA_CAPACITY = 64 * 1024 * 1024
PROTOCOL = "mooncake-host-v1"
FOOTER = 32
WIRE_CAPACITY = CAPACITY + FOOTER
_OWNERS_LOCK = threading.Lock()


class ReadRejected(RuntimeError):
    """Authenticated source rejected this read before submitting any write."""


def marker(nonce, sequence, size):
    return (
        bytes.fromhex(nonce)
        + sequence.to_bytes(8, "little")
        + size.to_bytes(8, "little")
    )


def new_slot(capacity, device):
    from sglang.srt.disaggregation.pinned_staging_buffer import CudaPinnedSlot

    return CudaPinnedSlot(capacity, torch.device(device).index)


def owners(engine):
    engine = transport.handle(engine)
    # Engine ownership outlives sockets, scheduler records and failed constructors.
    if "_d2d_bulk_owners" not in engine.__dict__:
        engine._d2d_bulk_owners = []
        try:
            engine.native._d2d_bulk_owners = engine._d2d_bulk_owners
        except AttributeError:
            pass  # Some pybind engines do not allow dynamic attributes.
    return engine._d2d_bulk_owners


def descriptor(value):
    if (
        not isinstance(value, dict)
        or value.get("protocol") not in (PROTOCOL, transport.GPU)
        or type(value.get("base")) is not int
        or value["base"] <= 0
        or type(value.get("size")) is not int
        or (
            value["size"] not in (LEGACY_CAPACITY + FOOTER, WIRE_CAPACITY)
            and not (
                value["protocol"] == transport.GPU
                and value["size"] == RDMA_CAPACITY + FOOTER
            )
        )
        or not isinstance(value.get("session"), str)
        or not 0 < len(value["session"]) <= 512
        or not isinstance(value.get("nonce"), str)
        or len(value["nonce"]) != 32
        or any(c not in "0123456789abcdef" for c in value["nonce"])
    ):
        raise ValueError("invalid D2D Mooncake bulk region")
    return dict(value)


def frame_capacity(value, *, rdma=False):
    allowed = (
        (LEGACY_CAPACITY, CAPACITY, RDMA_CAPACITY)
        if rdma
        else (LEGACY_CAPACITY, CAPACITY)
    )
    if type(value) is not int or value not in allowed:
        raise ValueError("unsupported D2D bulk frame capacity")
    return value


class Registered:
    def __init__(self, engine, req):
        self.engine, self.req = engine, req
        self.ownership_engine = engine
        self.closed = self.unsafe = False
        self.tensor = None
        self.registered = False
        with _OWNERS_LOCK:
            retained = owners(engine)
            if len(retained) >= 32:
                raise RuntimeError("D2D bulk slot budget exhausted")
            retained.append(self)

    def register(self):
        # A failed registration can have an uncertain native outcome too.
        self.registered = True
        try:
            rc = self.engine.batch_register([self.tensor.data_ptr()], [self.tensor.numel()])
            if rc != 0:
                raise RuntimeError("D2D bulk registration failed")
        except BaseException:
            self.poison()
            raise

    def poison(self):
        self.unsafe = True
        self.req._d2d_bulk_unsafe = True
        dp.io_for(self.req).poison()

    def check(self):
        if self.closed or self.unsafe:
            raise RuntimeError("D2D bulk slot is closed/quarantined")

    def close(self):
        if self.closed:
            return
        self.check()
        try:
            self.drain()
            if self.registered:
                rc = self.engine.batch_deregister([self.tensor.data_ptr()])
                if rc != 0:
                    raise RuntimeError("D2D bulk deregistration failed")
                self.registered = False
            self.dispose()
        except BaseException:
            self.poison()
            raise
        self.closed = True
        with _OWNERS_LOCK:
            owners(self.ownership_engine).remove(self)
        self.tensor = None

    def dispose(self):
        pass


class GpuSendBuffer(Registered):
    """An RDMA staging allocation that never exposes its address to peers."""

    def __init__(self, engine, device, capacity):
        super().__init__(engine, SimpleNamespace())
        self.stream = None
        try:
            self.tensor = torch.empty(
                capacity + FOOTER, dtype=torch.uint8, device=device
            )
            dp.event(device).synchronize()
            self.stream = torch.cuda.Stream(device=device)
            self.register()
        except BaseException:
            if not self.registered:
                self.close()
            raise

    def drain(self):
        if self.stream is not None:
            self.stream.synchronize()


class GpuReceiveBuffer(Registered):
    """Registered backing shared across separately fenced receive leases."""

    def __init__(self, engine, device, capacity):
        super().__init__(engine, SimpleNamespace())
        try:
            self.tensor = torch.empty(
                capacity + FOOTER, dtype=torch.uint8, device=device
            )
            self.visibility = dp.Visibility(device, protocol="rdma")
            dp.event(device).synchronize()
            self.register()
        except BaseException:
            if not self.registered:
                self.close()
            raise

    def drain(self):
        # The receiver lease fences its own consumers before returning backing.
        pass


class _GpuBufferPool:
    def __init__(self):
        self.lock = threading.Lock()
        self.available = []
        self.closed = False

    def prewarm(self, engine, device, count):
        for _ in range(count):
            self.available.append(self.buffer_type(engine, device, RDMA_CAPACITY))

    def acquire(self):
        with self.lock:
            return self.available.pop() if self.available else None

    def release(self, buffer):
        with self.lock:
            if not self.closed:
                self.available.append(buffer)
                return
        buffer.close()

    def close(self):
        with self.lock:
            self.closed = True
            buffers, self.available = self.available, []
        for buffer in buffers:
            buffer.close()


class GpuSendPool(_GpuBufferPool):
    buffer_type = GpuSendBuffer


class GpuReceivePool(_GpuBufferPool):
    buffer_type = GpuReceiveBuffer


class Sender(Registered):
    def __init__(
        self,
        engine,
        req,
        device,
        catalog=None,
        *,
        capacity=CAPACITY,
        rdma_capacity=None,
        gpu_send_pool=None,
    ):
        self.capacity = frame_capacity(capacity)
        self.rdma_capacity = (
            self.capacity
            if rdma_capacity is None
            else frame_capacity(rdma_capacity, rdma=True)
        )
        super().__init__(engine, req)
        self.catalog = catalog
        self.gpu_send_pool = gpu_send_pool
        self.gpu_buffer = None
        self.prewarmed = False
        self.device = torch.device(device)
        self.backend = None
        self.stream = None
        self.lock = threading.Lock()
        self.lanes = {}
        self.session = None
        self.slot = None
        # The raw CUDA copy stream is invisible to the producer's allocator.
        # Engine ownership must retain BOTH endpoints if its fence fails.
        self.copy_owner = None
        if catalog is None or list(catalog.engines) == [transport.TCP]:
            self.configure(transport.TCP)

    def configure(self, backend):
        if self.backend is not None:
            if self.backend != backend:
                raise ValueError("D2D backend changed after registration")
            return
        if self.catalog is not None:
            if backend not in self.catalog.engines:
                raise ValueError("D2D source did not offer this backend")
            self.engine = self.catalog.engines[backend]
        elif backend != transport.TCP:
            raise ValueError("D2D GPU backend was not negotiated")
        self.backend = backend
        try:
            if backend == transport.GPU:
                if self.gpu_send_pool is not None:
                    self.gpu_buffer = self.gpu_send_pool.acquire()
                if self.gpu_buffer is not None:
                    self.tensor = self.gpu_buffer.tensor
                    self.stream = self.gpu_buffer.stream
                    self.prewarmed = True
                    return
                self.tensor = torch.empty(
                    self.rdma_capacity + FOOTER, dtype=torch.uint8, device=self.device
                )
                dp.event(self.device).synchronize()
                self.stream = torch.cuda.Stream(device=self.device)
            else:
                self.slot = new_slot(self.capacity + FOOTER, self.device)
                self.tensor = self.slot.host
            self.register()
        except BaseException:
            self.poison()
            raise

    def write(self, data, ready, region, sequence):
        region = descriptor(region)
        size = data.numel() * data.element_size()
        return self._write(data, None, size, ready, region, sequence)

    def write_rows(self, rows, ready, region, sequence):
        region = descriptor(region)
        if region["protocol"] != transport.GPU:
            raise ValueError("direct D2D gather requires a GPU region")
        size = sum(table.shape[1] * indices.numel() for table, indices in rows)
        return self._write(None, rows, size, ready, region, sequence)

    def _write(self, data, rows, size, ready, region, sequence):
        gpu = region["protocol"] == transport.GPU
        capacity = self.rdma_capacity if gpu else self.capacity
        if (
            not 0 < size <= min(capacity, region["size"] - FOOTER)
            or type(sequence) is not int
            or not 0 < sequence < 2**64
        ):
            raise ValueError("invalid D2D bulk size/sequence")
        with self.lock:
            self.check()
            self.configure(transport.GPU if gpu else transport.TCP)
            if self.session is not None and self.session != region["session"]:
                raise ValueError("D2D bulk destination session changed")
            lane = self.lanes.get(region["nonce"])
            if lane is None:
                # One setup reader and one background bulk reader per migration.
                if len(self.lanes) >= 2:
                    raise ValueError("D2D bulk receiver lane limit exceeded")
                lane = (region, 0)
            if region != lane[0] or sequence != lane[1] + 1:
                raise ValueError("D2D bulk region/sequence changed")
            self.session = region["session"]
            self.lanes[region["nonce"]] = (region, sequence)
            try:
                # CudaPinnedSlot uses a private nonblocking D2H stream, waits
                # only for the producing event and proves completion before TE.
                self.copy_owner = (data, rows, ready)
                if gpu:
                    with torch.cuda.stream(self.stream):
                        self.stream.wait_event(ready)
                        if rows is None:
                            self.tensor[:size].copy_(
                                data.reshape(-1).view(torch.uint8), non_blocking=True
                            )
                        else:
                            offset = 0
                            for table, indices in rows:
                                count, stride = indices.numel(), table.shape[1]
                                end = offset + count * stride
                                dst = self.tensor[offset:end].view(count, stride)
                                torch.index_select(table, 0, indices, out=dst)
                                offset = end
                        copied = dp.event(self.device)
                    copied.synchronize()
                    self.copy_owner = None
                    wire_size = size
                else:
                    self.slot.copy([(data.data_ptr(), 0, size)], ready)
                    # copy() returns only after its completion event synchronized.
                    # A later TE failure still retains Host backing, but no longer
                    # needs the GPU input. An exception above keeps copy_owner.
                    self.copy_owner = None
                    footer = torch.tensor(
                        list(marker(region["nonce"], sequence, size)),
                        dtype=torch.uint8,
                        device="cpu",
                    )
                    self.tensor[size : size + FOOTER].copy_(footer)
                    wire_size = size + FOOTER
                rc = transport.handle(self.engine).batch_transfer_sync(
                    region["session"],
                    [self.tensor.data_ptr()],
                    [region["base"]],
                    [wire_size],
                )
                if rc != 0:
                    raise RuntimeError("D2D Mooncake write completion is unknown")
            except BaseException:
                self.poison()
                self.req._handoff_transfer_unsafe = True
                raise
        return size

    def poison(self):
        if self.gpu_buffer is not None:
            self.gpu_buffer.unsafe = True
        super().poison()

    def drain(self):
        if self.stream is not None:
            self.stream.synchronize()
        if self.slot is not None:
            self.slot.stream.synchronize()

    def dispose(self):
        if self.gpu_buffer is not None:
            # Registered.close already fenced the copy stream. A failed or
            # ambiguous write poisons this Sender, so it never returns the slot.
            self.gpu_send_pool.release(self.gpu_buffer)
            self.gpu_buffer = None
            self.stream = None
        if self.slot is not None:
            self.slot.close()
            self.slot = None


class Receiver(Registered):
    def __init__(
        self,
        engine,
        req,
        device,
        backend=transport.TCP,
        *,
        capacity=CAPACITY,
        gpu_receive_pool=None,
    ):
        self.capacity = frame_capacity(capacity, rdma=backend == transport.GPU)
        super().__init__(engine, req)
        self.device = torch.device(device)
        self.backend = backend
        self.protocol = transport.GPU if backend == transport.GPU else PROTOCOL
        self.visibility = None
        self.lock = threading.Lock()
        self.sequence = 0
        self.consumed = None
        self.nonce = secrets.token_hex(16)
        self.gpu_receive_pool = gpu_receive_pool
        self.gpu_buffer = None
        self.pending = False
        try:
            if gpu_receive_pool is not None and backend == transport.GPU:
                self.gpu_buffer = gpu_receive_pool.acquire()
            if self.gpu_buffer is not None:
                self.tensor = self.gpu_buffer.tensor
                self.visibility = self.gpu_buffer.visibility
                return
            self.tensor = torch.empty(
                self.capacity + FOOTER,
                dtype=torch.uint8,
                device=self.device if backend == transport.GPU else "cpu",
                pin_memory=backend == transport.TCP and self.device.type == "cuda",
            )
            if backend == transport.GPU:
                self.visibility = dp.Visibility(self.device, protocol="rdma")
                dp.event(self.device).synchronize()
            self.register()
        except BaseException:
            self.poison()
            raise

    def prepare(self):
        self.check()
        if self.gpu_buffer is not None and self.pending:
            raise RuntimeError("D2D receive slot still has an outstanding write")
        self.drain()
        self.sequence += 1
        self.pending = True
        return dict(
            protocol=self.protocol,
            base=self.tensor.data_ptr(),
            size=self.capacity + FOOTER,
            session=self.engine.get_session_id(),
            nonce=self.nonce,
        )

    def consume(self, header, size, timeout, *, apply=None):
        self.check()
        if (
            type(size) is not int
            or (self.gpu_buffer is not None and not self.pending)
            or header.get("bulk_protocol") != self.protocol
            or header.get("bulk_nonce") != self.nonce
            or header.get("bulk_sequence") != self.sequence
            or header.get("bytes") != size
            or not 0 < size <= self.capacity
        ):
            raise ValueError("D2D bulk completion does not match receive slot")
        if self.backend == transport.GPU:
            # Authenticated RPC result is emitted after every native write has
            # completed. Flush external writes BEFORE launching CUDA consumers.
            self.visibility.receive()
        else:
            # Some TCP TE versions report send completion before the peer applies
            # the last bytes. The footer travels in the SAME contiguous descriptor,
            # after the payload; an RPC ACK alone must never publish stale host data.
            expected = marker(self.nonce, self.sequence, size)
            deadline = time.monotonic() + timeout
            while bytes(self.tensor[size : size + FOOTER].tolist()) != expected:
                if time.monotonic() >= deadline:
                    raise TimeoutError("D2D bulk host arrival is unproven")
                time.sleep(0.001)
        if apply is not None:
            # The callback enqueues all readers on this copy stream. The slot
            # stays registered and cannot be overwritten before that fence.
            if self.backend != transport.GPU:
                raise ValueError("direct D2D consumption requires a GPU slot")
            apply(self.tensor[:size])
            self.consumed = dp.event(self.device)
            self.pending = False
            return None
        # Return independent GPU backing; slot reuse waits for this copy event.
        data = self.tensor[:size].to(self.device, non_blocking=True, copy=True)
        self.consumed = dp.event(self.device)
        self.pending = False
        return data

    def poison(self):
        if self.gpu_buffer is not None:
            self.gpu_buffer.unsafe = True
        super().poison()

    def dispose(self):
        if self.gpu_buffer is not None:
            if self.pending:
                raise RuntimeError("D2D receive write completion is unknown")
            self.gpu_receive_pool.release(self.gpu_buffer)
            self.gpu_buffer = None

    def drain(self):
        if self.consumed is not None:
            self.consumed.synchronize()
