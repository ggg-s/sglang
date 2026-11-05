"""Bounded D2D bulk slots on the native Mooncake engine.

RemoteHub authenticates and sequences requests; it carries no KV payload.
An uncertain write or response permanently quarantines its registered backing.
Slots are private to a migration, independent of Remote Attention Q/O lanes.
"""

import secrets
import threading
import time

import torch

from . import remote_data_plane as dp

CAPACITY = 4 * 1024 * 1024
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
    # Engine ownership outlives sockets, scheduler records and failed constructors.
    if not hasattr(engine, "_d2d_bulk_owners"):
        engine._d2d_bulk_owners = []
    return engine._d2d_bulk_owners


def descriptor(value):
    if (
        not isinstance(value, dict)
        or value.get("protocol") != PROTOCOL
        or type(value.get("base")) is not int
        or value["base"] <= 0
        or value.get("size") != WIRE_CAPACITY
        or not isinstance(value.get("session"), str)
        or not 0 < len(value["session"]) <= 512
        or not isinstance(value.get("nonce"), str)
        or len(value["nonce"]) != 32
        or any(c not in "0123456789abcdef" for c in value["nonce"])
    ):
        raise ValueError("invalid D2D Mooncake host region")
    return dict(value)


class Registered:
    def __init__(self, engine, req):
        self.engine, self.req = engine, req
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
            rc = self.engine.batch_register([self.tensor.data_ptr()], [WIRE_CAPACITY])
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
            owners(self.engine).remove(self)
        self.tensor = None

    def dispose(self):
        pass


class Sender(Registered):
    def __init__(self, engine, req, device):
        super().__init__(engine, req)
        self.lock = threading.Lock()
        self.lanes = {}
        self.session = None
        self.slot = None
        # The raw CUDA copy stream is invisible to the producer's allocator.
        # Engine ownership must retain BOTH endpoints if its fence fails.
        self.copy_owner = None
        try:
            self.slot = new_slot(WIRE_CAPACITY, device)
            self.tensor = self.slot.host
            self.register()
        except BaseException:
            self.poison()
            raise

    def write(self, data, ready, region, sequence):
        region = descriptor(region)
        size = data.numel() * data.element_size()
        if (
            not 0 < size <= CAPACITY
            or type(sequence) is not int
            or not 0 < sequence < 2**64
        ):
            raise ValueError("invalid D2D bulk size/sequence")
        with self.lock:
            self.check()
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
                self.copy_owner = (data, ready)
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
                rc = self.engine.batch_transfer_sync(
                    region["session"],
                    [self.tensor.data_ptr()],
                    [region["base"]],
                    [size + FOOTER],
                )
                if rc != 0:
                    raise RuntimeError("D2D Mooncake write completion is unknown")
            except BaseException:
                self.poison()
                self.req._handoff_transfer_unsafe = True
                raise
        return size

    def drain(self):
        self.slot.stream.synchronize()

    def dispose(self):
        self.slot.close()
        self.slot = None


class Receiver(Registered):
    def __init__(self, engine, req, device):
        super().__init__(engine, req)
        self.device = torch.device(device)
        self.lock = threading.Lock()
        self.sequence = 0
        self.consumed = None
        self.nonce = secrets.token_hex(16)
        try:
            self.tensor = torch.empty(
                WIRE_CAPACITY,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.device.type == "cuda",
            )
            self.register()
        except BaseException:
            self.poison()
            raise

    def prepare(self):
        self.check()
        self.drain()
        self.sequence += 1
        return dict(
            protocol=PROTOCOL,
            base=self.tensor.data_ptr(),
            size=WIRE_CAPACITY,
            session=self.engine.get_session_id(),
            nonce=self.nonce,
        )

    def consume(self, header, size, timeout):
        self.check()
        if (
            type(size) is not int
            or header.get("bulk_protocol") != PROTOCOL
            or header.get("bulk_nonce") != self.nonce
            or header.get("bulk_sequence") != self.sequence
            or header.get("bytes") != size
            or not 0 < size <= CAPACITY
        ):
            raise ValueError("D2D bulk completion does not match receive slot")
        # Some TCP TE versions report send completion before the peer applies
        # the last bytes. The footer travels in the SAME contiguous descriptor,
        # after the payload; an RPC ACK alone must never publish stale host data.
        expected = marker(self.nonce, self.sequence, size)
        deadline = time.monotonic() + timeout
        while bytes(self.tensor[size : size + FOOTER].tolist()) != expected:
            if time.monotonic() >= deadline:
                raise TimeoutError("D2D bulk host arrival is unproven")
            time.sleep(0.001)
        # Receive backing is HOST memory, not a GPUDirect GPU allocation.
        data = self.tensor[:size].to(self.device, non_blocking=True, copy=True)
        self.consumed = dp.event(self.device)
        return data

    def drain(self):
        if self.consumed is not None:
            self.consumed.synchronize()
