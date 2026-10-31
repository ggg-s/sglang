"""Request-scoped journal replication over native PD-selected peer connections.

No KV payloads, scheduler calls or distributed collectives run on these threads.
Each instance keeps its own files; the wire grants access to one room/attempt.
"""

import dataclasses
import json
import os
import re
import secrets
import socket
import struct
import threading
import time
from pathlib import Path

MAGIC = b"REMOTE_ATTN_PD_V1:"
LIMIT = 2 * 1024 * 1024
P_FILES = re.compile(
    r"(?:layer-\d+|resource-\d+|prefill|ready|sparse-ready|remote-service|error|released)\.json\Z"
)
D_FILES = re.compile(r"(?:done|remote-done|admitted|local-\d+|layer-\d+|error)\.json\Z")


def enabled():
    value = os.environ.get("HANDOFF_PD_NETWORK", "0")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_PD_NETWORK must be 0 or 1")
    return value == "1"


def instance_id():
    value = os.environ.get("HANDOFF_INSTANCE_ID", "")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise ValueError(
            "network PD requires a unique HANDOFF_INSTANCE_ID per P/D instance"
        )
    return value


def startup_signature(config):
    # Reuse the existing unconditional TP startup exchange before a network-only
    # epoch broadcast. A rank-local flag must not change collective ordering.
    return tuple(
        os.environ.get(key, "")
        for key in (
            "HANDOFF_PD_NETWORK",
            "HANDOFF_INSTANCE_ID",
            "HANDOFF_NETWORK_BIND_HOST",
            "HANDOFF_NETWORK_ADVERTISE_HOST",
            "HANDOFF_CTL_DIR",
            "HANDOFF_REMOTE_GPU_TRANSFER",
        )
    ), str(config.directory) if enabled() else None


def address(scheduler):
    if not enabled():
        return "127.0.0.1", "127.0.0.1"
    instance_id()
    prefill = scheduler.disaggregation_mode.value == "prefill"
    queue = (
        scheduler.disagg_prefill_bootstrap_queue
        if prefill
        else scheduler.disagg_decode_prealloc_queue
    )
    bind = os.environ.get("HANDOFF_NETWORK_BIND_HOST", queue.kv_manager.local_ip)
    host = os.environ.get("HANDOFF_NETWORK_ADVERTISE_HOST", bind)
    if host in ("0.0.0.0", "::", ""):
        raise ValueError("PD network requires a reachable advertised host")
    return bind, host


def local_write(path, value):
    from .pre_layerwise import write_json

    write_json(Path(path), value)


def send(sock, value):
    data = json.dumps(value).encode()
    if len(data) > LIMIT:
        raise ValueError("PD journal frame too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive(sock):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            block = sock.recv(size - len(result))
            if not block:
                raise EOFError("PD journal disconnected")
            result.extend(block)
        return result

    size = struct.unpack("!I", exact(4))[0]
    if size > LIMIT:
        raise ValueError("PD journal frame too large")
    return json.loads(exact(size))


class JournalPath:
    """Path facade: ordinary reads/stats, explicitly replicated atomic writes."""

    def __init__(self, room, path=None):
        self.room = room
        self.local = Path(path) if path is not None else room.root

    def __truediv__(self, child):
        return JournalPath(self.room, self.local / child)

    @property
    def parent(self):
        parent = self.local.parent
        return (
            JournalPath(self.room, parent)
            if parent.is_relative_to(self.room.root)
            else parent
        )

    def __fspath__(self):
        return str(self.local)

    def __str__(self):
        return str(self.local)

    def __getattr__(self, name):
        return getattr(self.local, name)

    def publish(self, value):
        self.room.publish(str(self.local.relative_to(self.room.root)), value)


class Room:
    def __init__(self, root, role, descriptor):
        self.root, self.role, self.descriptor = Path(root), role, descriptor
        self.root.mkdir(parents=True, exist_ok=True)
        self.condition = threading.Condition()
        self.values = {}
        self.revision = 0
        self.remote_revision = 0
        self.stopped = threading.Event()
        self.sockets = set()
        self.owner = None
        self.terminal_at = None
        self.created_at = time.monotonic()
        self.error = None

    def validate(self, name, value, role):
        parts = name.split("/")
        if name == "offer.json":
            if role != "D" or not re.fullmatch(
                r"[a-f0-9]{32}", value.get("generation", "")
            ):
                raise ValueError("invalid PD offer writer/generation")
            old = self.values.get(name)
            if old and old["generation"] != value["generation"]:
                raise ValueError("PD room attempt changed")
            return
        offer = self.values.get("offer.json")
        if (
            len(parts) != 2
            or not offer
            or parts[0] != offer["generation"]
            or value.get("generation") != offer["generation"]
            or not (P_FILES if role == "P" else D_FILES).fullmatch(parts[1])
        ):
            raise ValueError("PD journal write outside the request capability")

    def publish(self, name, value):
        with self.condition:
            self.validate(name, value, self.role)
            local_write(self.root / name, value)
            self.values[name] = value
            self.revision += 1
            if name.endswith("/done.json"):
                self.terminal_at = time.monotonic()
            if name.endswith("/released.json"):
                self.terminal_at = time.monotonic()
            self.condition.notify_all()

    def apply(self, revision, values):
        with self.condition:
            if type(revision) is not int or revision < self.remote_revision:
                raise ValueError("PD journal revision regressed")
            if revision == self.remote_revision:
                return
            role = "D" if self.role == "P" else "P"
            # Offer is serialized first so snapshots can validate its children.
            for name, value in values.items():
                if not isinstance(name, str) or not isinstance(value, dict):
                    raise ValueError("invalid PD journal value")
                self.validate(name, value, role)
                local_write(self.root / name, value)
                if name == "offer.json":
                    self.values[name] = value
            self.remote_revision = revision

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
            sockets = list(self.sockets)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def connection(self, sock):
        with self.condition:
            if self.sockets:
                raise ValueError("PD journal connection already active")
            self.sockets.add(sock)
        ended = threading.Event()

        def writer():
            sent = -1
            try:
                while not self.stopped.is_set():
                    with self.condition:
                        self.condition.wait_for(
                            lambda: (
                                self.stopped.is_set()
                                or ended.is_set()
                                or self.revision != sent
                            )
                        )
                        if self.stopped.is_set() or ended.is_set():
                            return
                        # D owns offer; P keeps a copy only for validating facts.
                        values = {
                            k: v
                            for k, v in self.values.items()
                            if self.role == "D" or k != "offer.json"
                        }
                        revision = self.revision
                    send(sock, dict(revision=revision, values=values))
                    sent = revision
            except (OSError, ValueError) as exc:
                self.error = str(exc)
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        thread = threading.Thread(target=writer, daemon=True, name="pd-journal-send")
        thread.start()
        try:
            while not self.stopped.is_set():
                data = receive(sock)
                self.apply(data["revision"], data["values"])
        finally:
            ended.set()
            with self.condition:
                self.sockets.discard(sock)
                self.condition.notify_all()
            sock.close()
            # Wake the writer without revoking this request's native leases.
            self.error = "PD journal disconnected; awaiting same-attempt reconnect"


class Mesh:
    def __init__(self, root, instance, epoch, rank, bind, host, source_endpoint):
        self.root = Path(root)
        self.instance, self.epoch, self.rank = instance, epoch, rank
        self.source_endpoint = source_endpoint
        self.rooms, self.peers = {}, {}
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.listener = socket.create_server(
            (bind, 0), family=socket.AF_INET6 if ":" in bind else socket.AF_INET
        )
        self.listener.settimeout(0.2)
        self.host, self.port = host, self.listener.getsockname()[1]
        self.thread = threading.Thread(
            target=self.accept, daemon=True, name="pd-journal-listen"
        )
        self.thread.start()

    def new_room(self, number, session, expected):
        if self.stopped.is_set():
            return None
        self.prune()
        with self.lock:
            counter = self.root / str(number) / str(self.rank) / "attempt.json"
            previous = json.loads(counter.read_text()) if counter.exists() else 0
            attempt = previous + 1
            local_write(counter, attempt)
        descriptor = dict(
            host=self.host,
            port=self.port,
            token=secrets.token_hex(32),
            instance=self.instance,
            epoch=self.epoch,
            rank=self.rank,
            room=int(number),
            attempt=attempt,
            session=session,
            expected=list(expected),
        )
        # New native metadata = new capability; never reuse the previous attempt.
        room = Room(
            self.root / str(number) / str(descriptor["attempt"]) / str(self.rank),
            "D",
            descriptor,
        )
        with self.lock:
            if len(self.rooms) >= 4096:
                return None
            self.rooms[descriptor["token"]] = room
        return room

    def open_peer(self, peer):
        if self.stopped.is_set():
            return
        descriptor = getattr(peer, "_remote_attn_journal", None)
        if not descriptor or peer.is_dummy:
            return
        if (
            descriptor["room"] != peer.room
            or descriptor["session"] != peer.mooncake_session_id
            or descriptor["rank"] != self.rank
            or descriptor["expected"] != list(self.source_endpoint)
        ):
            return
        key = descriptor["token"]
        self.prune()
        with self.lock:
            if key not in self.peers:
                if len(self.peers) >= 4096:
                    return
                root = (
                    self.root
                    / "peers"
                    / descriptor["instance"]
                    / descriptor["epoch"]
                    / str(peer.room)
                    / str(descriptor["attempt"])
                    / str(self.rank)
                )
                room = self.peers[key] = Room(root, "P", descriptor)
                thread = threading.Thread(
                    target=self.connect,
                    args=(room,),
                    daemon=True,
                    name="pd-journal-connect",
                )
                thread.start()
            peer._remote_attn_room = self.peers[key]

    def connect(self, room):
        descriptor = room.descriptor
        owner = [self.instance, self.epoch, self.rank, *self.source_endpoint]
        while not self.stopped.is_set() and not room.stopped.is_set():
            try:
                with socket.create_connection(
                    (descriptor["host"], descriptor["port"]), timeout=2
                ) as sock:
                    send(sock, dict(token=descriptor["token"], owner=owner))
                    if receive(sock) != dict(ok=True):
                        raise ValueError("PD journal capability rejected")
                    sock.settimeout(None)
                    room.connection(sock)
            except (OSError, EOFError, ValueError, KeyError) as exc:
                room.error = str(exc)
            room.stopped.wait(0.1)

    def accept(self):
        while not self.stopped.is_set():
            try:
                sock, _ = self.listener.accept()
            except socket.timeout:
                self.prune()
                continue
            except OSError:
                return
            threading.Thread(
                target=self.serve, args=(sock,), daemon=True, name="pd-journal-receive"
            ).start()

    def serve(self, sock):
        try:
            sock.settimeout(2)
            hello = receive(sock)
            with self.lock:
                room = self.rooms.get(hello.get("token"))
            if room is None:
                raise ValueError("unknown PD journal capability")
            owner = hello["owner"]
            with room.condition:
                if (
                    owner[-2:] != room.descriptor["expected"]
                    or (room.owner is not None and room.owner != owner)
                    or room.sockets
                ):
                    raise ValueError("PD journal source/attempt already bound")
                room.owner = owner
            send(sock, dict(ok=True))
            sock.settimeout(None)
            room.connection(sock)
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            sock.close()

    def close(self):
        self.stopped.set()
        self.listener.close()
        for room in list(self.rooms.values()) + list(self.peers.values()):
            room.close()
        self.thread.join(timeout=2)
        lease = getattr(self, "instance_lease", None)
        if lease is not None:
            lease.close()

    def prune(self):
        # Only terminal native/TP facts permit background channel reclamation.
        # Dropped sockets alone never authorize freeing a cache or KV page.
        with self.lock:
            for table in (self.rooms, self.peers):
                for key, room in list(table.items()):
                    if (
                        room.terminal_at and time.monotonic() - room.terminal_at > 5
                    ) or (
                        "offer.json" not in room.values
                        and time.monotonic() - room.created_at > 300
                    ):
                        table.pop(key).close()


def install(controller):
    if not enabled():
        return
    import torch.distributed as dist

    scheduler = controller.scheduler
    lease = None
    control = os.environ.get("HANDOFF_CTL_DIR")
    if controller.rank == 0 and control:
        import fcntl

        root = Path(control) / instance_id()
        root.mkdir(parents=True, exist_ok=True)
        lease = (root / "instance.lock").open("a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lease.close()
            raise RuntimeError(
                "HANDOFF_INSTANCE_ID is already serving in this control root"
            )
    epoch = [secrets.token_hex(16) if controller.rank == 0 else None]
    if controller.tp_size > 1:
        group = scheduler.attn_tp_cpu_group
        dist.broadcast_object_list(
            epoch, src=dist.get_global_rank(group, 0), group=group
        )
    config = controller.config
    root = config.directory / "instances" / instance_id() / epoch[0]
    controller.config = dataclasses.replace(config, directory=root)
    prefill = scheduler.disaggregation_mode.value == "prefill"
    queue = (
        scheduler.disagg_prefill_bootstrap_queue
        if prefill
        else scheduler.disagg_decode_prealloc_queue
    )
    manager = queue.kv_manager
    bind, host = address(scheduler)
    manager._remote_attn_mesh = controller.mesh = Mesh(
        root,
        instance_id(),
        epoch[0],
        controller.rank,
        bind,
        host,
        (manager.local_ip, manager.rank_port),
    )
    controller.mesh.instance_lease = lease


def bootstrap(original, receiver, *, bootstrap_info, message, operation):
    mesh = getattr(receiver.kv_mgr, "_remote_attn_mesh", None)
    if (
        mesh is not None
        and operation == "send_metadata"
        and not bootstrap_info["is_dummy"]
    ):
        if len(message) != 10:
            raise ValueError("native PD metadata wire changed")
        room = mesh.new_room(
            receiver.bootstrap_room,
            receiver.kv_mgr.get_session_id(),
            (bootstrap_info["rank_ip"], bootstrap_info["rank_port"]),
        )
        receiver._remote_attn_room = room
        if room is not None:
            message = [*message, MAGIC + json.dumps(room.descriptor).encode()]
    return original(
        receiver, bootstrap_info=bootstrap_info, message=message, operation=operation
    )


def parse_transfer(original, *args, **kwargs):
    value = original(*args, **kwargs)
    message = kwargs.get("msg", args[-1] if args else None)
    if len(message) == 11 and message[-1].startswith(MAGIC):
        try:
            if len(message[-1]) > 4096:
                raise ValueError("oversized PD capability")
            descriptor = json.loads(message[-1][len(MAGIC) :])
            # Invalid extensions disable adoption, never kill native bootstrap.
            for key in ("instance", "epoch", "token"):
                if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", descriptor[key]):
                    raise ValueError("invalid PD journal identity")
            for key in ("port", "rank", "room", "attempt"):
                if type(descriptor[key]) is not int or descriptor[key] < 0:
                    raise ValueError("invalid PD capability number")
            if not 0 < descriptor["port"] < 65536 or descriptor["attempt"] == 0:
                raise ValueError("invalid PD capability port/attempt")
            for key in ("host", "session"):
                if (
                    not isinstance(descriptor[key], str)
                    or not descriptor[key]
                    or len(descriptor[key]) > 1024
                ):
                    raise ValueError("invalid PD capability address/session")
            expected = descriptor["expected"]
            if (
                not isinstance(expected, list)
                or len(expected) != 2
                or not isinstance(expected[0], str)
                or type(expected[1]) is not int
            ):
                raise ValueError("invalid PD source endpoint")
            value._remote_attn_journal = descriptor
        except (ValueError, KeyError, TypeError):
            pass
    return value


def resolve_peers(original, manager, transfer_infos):
    result = original(manager, transfer_infos)
    mesh = getattr(manager, "_remote_attn_mesh", None)
    if mesh is not None:
        for peer in transfer_infos.values():
            mesh.open_peer(peer)
    return result


def source_root(controller, req):
    manager = req.disagg_kv_sender.kv_mgr
    if getattr(manager, "_remote_attn_mesh", None) is not None:
        entries = list(manager.transfer_infos.get(req.bootstrap_room, {}).values())
        if len(entries) == 1:
            # Native registration may have been deferred behind an active P
            # certificate. Start the peer once its ordered replacement commits.
            room = getattr(entries[0], "_remote_attn_room", None)
            if room is None:
                manager._remote_attn_mesh.open_peer(entries[0])
                room = getattr(entries[0], "_remote_attn_room", None)
            if room is not None:
                return JournalPath(room)
        return (
            controller.config.directory
            / "unmatched"
            / str(req.bootstrap_room)
            / str(controller.rank)
        )
    return (
        controller.config.directory
        / str(int(req.bootstrap_room))
        / str(controller.rank)
    )
