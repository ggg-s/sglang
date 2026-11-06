"""Shared native engines and pre-transfer D2D transport negotiation.

An engine's verified initialization metadata, never an environment guess, decides
its capability. TCP fallback is a separate process-wide engine when native PD
uses RDMA. No failed/uncertain transfer is retried on another backend.
"""

import os
import sys
import threading

import torch
from sglang.srt.mooncake_contract import (
    initialize_engine,
    write_once,
)

TCP = "tcp-host-v1"
GPU = "rdma-gpu-v1"
# v2 requires verified transport identity and single-submit completion. Reject
# old peers whose successful sync return may hide a still-live failed attempt.
VERSION = "d2d-mooncake-v2"
_LOCK = threading.RLock()
_HANDLES = {}
_TCP = {}
_PROBES = {}
_RETAINED = []


class Engine:
    def __init__(self, native, protocol, session):
        self.native, self.protocol, self.session = native, protocol, session
        self.raw = getattr(native, "engine", native)

    def __getattr__(self, name):
        return getattr(self.native, name)

    def batch_register(self, pointers, sizes):
        fn = getattr(self.native, "batch_register", None)
        return (
            fn(pointers, sizes)
            if fn
            else self.raw.batch_register_memory(pointers, sizes)
        )

    def batch_deregister(self, pointers):
        fn = getattr(self.native, "batch_deregister", None)
        return fn(pointers) if fn else self.raw.batch_unregister_memory(pointers)

    def batch_transfer_sync(self, session, sources, targets, sizes):
        return write_once(self.raw, session, sources, targets, sizes)

    def get_session_id(self):
        return self.session

    # RegistrationManager also accepts the native Python binding spelling.
    batch_register_memory = batch_register
    batch_unregister_memory = batch_deregister


def remember(native, protocol, session):
    if protocol not in ("tcp", "rdma"):
        # Unsupported native transports can still negotiate a separate TCP TE.
        protocol = "unsupported"
    with _LOCK:
        result = _HANDLES.get(id(native))
        if result is None:
            result = Engine(native, protocol, session)
            _HANDLES[id(native)] = result
            _HANDLES[id(result.raw)] = result
        elif (result.protocol, result.session) != (protocol, session):
            raise RuntimeError("Mooncake engine identity changed")
        return result


def handle(native):
    if isinstance(native, Engine):
        return native
    with _LOCK:
        known = _HANDLES.get(id(native))
        if known is not None:
            return known
    # Native SGLang wrapper records its effective initialization protocol.
    protocol = (
        getattr(native, "protocol", "unsupported")
        if getattr(native, "d2d_protocol_verified", False) is True
        else "unsupported"
    )
    session_fn = getattr(native, "get_session_id", None)
    session = session_fn() if session_fn else f"{host()}:{native.get_rpc_port()}"
    return remember(native, protocol, session)


def host(default="127.0.0.1"):
    return os.environ.get("HANDOFF_MOONCAKE_HOST", default)


def tcp_engine(default_host="127.0.0.1"):
    hostname = host(default_host)
    with _LOCK:
        if hostname in _TCP:
            return _TCP[hostname]
        from mooncake.engine import TransferEngine

        for dst, src in (
            ("MC_TCP_LANES_PER_PEER", "HANDOFF_MC_TCP_LANES"),
            ("MC_NUM_TCP_IO_THREADS", "HANDOFF_MC_TCP_IO_THREADS"),
            ("MC_TCP_SLICE_SIZE", "HANDOFF_MC_TCP_SLICE"),
        ):
            if os.environ.get(src):
                os.environ[dst] = os.environ[src]
        te = TransferEngine()
        # Keep a failed initialization alive; it can have native resources.
        _RETAINED.append(te)
        _, verified = initialize_engine(te, hostname, "tcp", "", force_tcp=True)
        result = remember(te, verified, f"{hostname}:{te.get_rpc_port()}")
        _TCP[hostname] = result
        return result


def shared_raw():
    """Use native PD's rank-local TE; diagnostics use the shared TCP fallback."""
    # Native PD initializes this module before accepting migrations. Standalone
    # probes must not import all of SGLang just to obtain their TCP engine.
    module = sys.modules.get(
        "sglang.srt.distributed.device_communicators.mooncake_transfer_engine"
    )
    native = module.get_mooncake_transfer_engine() if module is not None else None
    if native is not None:
        return handle(native).raw
    return tcp_engine().raw


def policy(name="HANDOFF_D2D_BACKEND"):
    value = os.environ.get(name, "auto")
    if value not in ("auto", "tcp", "rdma"):
        raise ValueError(f"{name} must be auto, tcp or rdma")
    return value


def gpu_capability(engine, device):
    """Probe registration + receive ordering once; retain any uncertain MR."""
    from .remote_data_plane import Visibility

    device = torch.device(device)
    if (
        engine.protocol != "rdma"
        or device.type != "cuda"
        or not getattr(engine.native, "d2d_rdma_eligible", True)
    ):
        return False
    key = (
        id(engine),
        device.index if device.index is not None else torch.cuda.current_device(),
    )
    with _LOCK:
        if key in _PROBES:
            return _PROBES[key]
        _PROBES[key] = False
        tensor = None
        try:
            Visibility(device, protocol="rdma")
            tensor = torch.empty(4096, dtype=torch.uint8, device=device)
            # Put backing under durable ownership BEFORE registration.
            retained = (engine, tensor)
            _RETAINED.append(retained)
            if engine.batch_register([tensor.data_ptr()], [tensor.numel()]) != 0:
                return False
            if engine.batch_deregister([tensor.data_ptr()]) != 0:
                return False
            _RETAINED[:] = [item for item in _RETAINED if item is not retained]
            _PROBES[key] = True
        except Exception:
            # Visibility/registration failure permits TCP only; never free an
            # allocation whose registration may have partially succeeded.
            return False
        return True


class Catalog:
    def __init__(self, native, device, *, policy_name="HANDOFF_D2D_BACKEND"):
        self.primary, self.device = handle(native), torch.device(device)
        self.engines = {}
        selected = policy(policy_name)
        if selected != "tcp" and gpu_capability(self.primary, self.device):
            self.engines[GPU] = self.primary
        if selected != "rdma":
            if self.primary.protocol == "tcp":
                self.engines[TCP] = self.primary
            else:
                self.engines[TCP] = tcp_engine(
                    getattr(self.primary.native, "hostname", "127.0.0.1")
                )
        if not self.engines:
            raise RuntimeError(
                "requested D2D backend unavailable (RDMA registration/visibility)"
            )
        self.offer = dict(version=VERSION, backends=list(self.engines))

    def select(self, peer):
        offered = validate(peer)
        for backend in (GPU, TCP):
            if backend in self.engines and backend in offered:
                return backend
        raise ValueError("no common D2D Mooncake backend")


def validate(offer):
    if (
        not isinstance(offer, dict)
        or offer.get("version") != VERSION
        or not isinstance(offer.get("backends"), list)
        or not offer["backends"]
        or any(x not in (GPU, TCP) for x in offer["backends"])
        or len(set(offer["backends"])) != len(offer["backends"])
    ):
        raise ValueError("invalid D2D Mooncake capabilities")
    return offer["backends"]


def tcp_offer():
    return dict(version=VERSION, backends=[TCP])
