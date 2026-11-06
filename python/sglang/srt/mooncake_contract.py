"""Classic Mooncake initialization and transfer completion contracts.

The Python protocol argument is not an installed-transport query. Read the
engine's own P2P metadata instead. This adapter supports the classic socket
metadata wire format (native uint64 length, uint8 type, JSON); unknown formats,
multi-transport descriptors and TENT fail closed for D2D capability negotiation.
"""

import json
import os
import socket
import struct
import threading
import time

INIT_LOCK = threading.RLock()
NON_CLASSIC = ("MC_USE_TENT", "MC_USE_TEV1", "MC_FORCE_SHM")


def local_protocol(engine, hostname):
    """Query only this initialized engine, before advertising any capability."""
    port = engine.get_rpc_port()
    deadline = time.monotonic() + 2
    with socket.create_connection((hostname.strip("[]"), port), timeout=2) as conn:
        # Metadata=1. The classic server returns its descriptor and does not
        # install the supplied peer metadata. No buffer or transfer is created.
        payload = b"\x01{}"
        conn.sendall(struct.pack("=Q", len(payload)) + payload)

        def read(size):
            data = bytearray()
            while len(data) < size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Mooncake metadata query timed out")
                conn.settimeout(remaining)
                part = conn.recv(size - len(data))
                if not part:
                    raise RuntimeError("Mooncake metadata response truncated")
                data.extend(part)
            return data

        length = struct.unpack("=Q", read(8))[0]
        if not 2 <= length <= 1024 * 1024:
            raise RuntimeError("unsupported Mooncake metadata length")
        response = read(length)
        if response[0] != 1:
            raise RuntimeError("unsupported Mooncake metadata response")
        desc = json.loads(response[1:])
        if (
            not isinstance(desc, dict)
            or not isinstance(desc.get("name"), str)
            or desc["name"].rsplit(":", 1)[-1] != str(port)
            or desc.get("protocol") not in ("tcp", "rdma")
            or "protocols" in desc
        ):
            raise RuntimeError("unverified Mooncake transport descriptor")
        return desc["protocol"]


def initialize_engine(engine, hostname, protocol, device_name, *, force_tcp=False):
    """Return (native status, verified transport); never guess from protocol.

    Classic bindings lack a per-instance TCP installation switch. Serialize all
    SGLang/D2D initializers, scope MC_FORCE_TCP to this call, then verify metadata.
    Keep the lock/override inside the deadline worker so a timeout cannot expose
    an unfinished initializer to an early environment restore.
    """
    with INIT_LOCK:
        non_classic = any(name in os.environ for name in NON_CLASSIC)
        if force_tcp and non_classic:
            raise RuntimeError("D2D TCP fallback requires classic Mooncake")
        previous = os.environ.get("MC_FORCE_TCP")
        if force_tcp:
            os.environ["MC_FORCE_TCP"] = "1"
        try:
            rc = engine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name)
            effective = "unsupported"
            if rc == 0 and not non_classic:
                try:
                    effective = local_protocol(engine, hostname)
                except (OSError, ValueError, RuntimeError):
                    # Native PD can continue, but D2D must not advertise GPU.
                    pass
            if force_tcp and (rc != 0 or effective != "tcp"):
                raise RuntimeError("D2D TCP fallback transport could not be verified")
            return rc, effective
        finally:
            if force_tcp:
                if previous is None:
                    os.environ.pop("MC_FORCE_TCP", None)
                else:
                    os.environ["MC_FORCE_TCP"] = previous


def write_once(engine, session, sources, targets, sizes):
    """Wait for exactly one native batch, without the sync API's hidden retry.

    Callers must mark buffers unsafe BEFORE entering and retain registrations and
    backing on any exception. A failed status need not mean sibling DMA drained.
    Never poll a failed ticket again: some bindings free it on terminal status.
    """
    submit = getattr(engine, "batch_transfer_async_write", None)
    wait = getattr(engine, "get_batch_transfer_status", None)
    if not callable(submit) or not callable(wait):
        raise RuntimeError("D2D requires Mooncake single-submit batch/status APIs")
    batch = submit(session, sources, targets, sizes)
    if type(batch) is not int or not 0 < batch < 2**64 - 1:
        raise RuntimeError("D2D native batch submission is unknown")
    if wait([batch]) != 0:
        raise RuntimeError("D2D native batch completion is unknown")
    return 0
