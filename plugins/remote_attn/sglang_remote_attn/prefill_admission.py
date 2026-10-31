"""TP-committed P admission certificates; no collective on stable chunks.

First checks use the native shared batch order. Later changes are advisory
notices until an ordered command is carried by the native request broadcast.
Registration writers cannot replace a pinned room/session beneath queued IO.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

from .retirement import Mailbox

CONTROL = "__remote_attn_prefill_admission__"
logger = logging.getLogger(__name__)


def key_of(req):
    handle = getattr(req, "cache_request_handle", None)
    return (
        str(req.rid),
        str(req.bootstrap_room),
        int(getattr(handle, "attempt_id", 0)),
        int(getattr(req, "retraction_count", 0)),
    )


def fingerprint(value):
    """Rank-local registration stamp, including state/aux pointers and indices."""

    def encode(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "__dict__"):
            return vars(item)
        raise TypeError(type(item).__name__)

    return hashlib.sha256(
        json.dumps(value, default=encode, sort_keys=True).encode()
    ).hexdigest()


class RoomPeers(dict):
    def __init__(self, guard, room, values=()):
        self.guard, self.room = guard, room
        super().__init__(values)

    def __setitem__(self, key, value):
        self.guard.room_write(self.room, key, value)

    def pop(self, key, *default):
        with self.guard.lock:
            if key not in self:
                return dict.pop(self, key, *default)
            value = self[key]
            self.guard.change(
                self.room, lambda: dict.pop(self, key, None), "peer-removed"
            )
            return value

    def __delitem__(self, key):
        self.pop(key)

    def clear(self):
        for key in tuple(self):
            self.pop(key)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self.get(key, default)

    def popitem(self):
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = next(reversed(self))
        return key, self.pop(key)

    def __ior__(self, other):
        self.update(other)
        return self


class Rooms(dict):
    def __init__(self, guard, values):
        self.guard = guard
        super().__init__((k, RoomPeers(guard, k, v)) for k, v in values.items())

    def __setitem__(self, room, value):
        self.guard.change(
            room,
            lambda: dict.__setitem__(self, room, RoomPeers(self.guard, room, value)),
            "room-replaced",
        )

    def pop(self, room, *default):
        with self.guard.lock:
            if room not in self:
                return dict.pop(self, room, *default)
            old = dict.__getitem__(self, room)
            self.guard.change(room, lambda: dict.pop(self, room, None), "room-removed")
            return old

    __delitem__ = RoomPeers.__delitem__
    clear = RoomPeers.clear
    update = RoomPeers.update
    setdefault = RoomPeers.setdefault
    popitem = RoomPeers.popitem
    __ior__ = RoomPeers.__ior__


class RegistrationGuard:
    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.RLock()
        self.pins = {}  # common request key -> (room, sessions)
        self.dirty = set()
        self.pending = []
        manager.transfer_infos = Rooms(self, manager.transfer_infos)

    def change(self, room, action, reason):
        with self.lock:
            keys = [k for k, (r, _) in self.pins.items() if r == room]
            if keys:
                self.dirty.update(keys)
                self.pending.append((set(keys), action))
            else:
                action()

    def room_write(self, room, session, value):
        with self.lock:
            old = self.manager.transfer_infos.get(room, {}).get(session)
            if old is not None and fingerprint(old) == fingerprint(value):
                return

            def apply():
                rooms = self.manager.transfer_infos
                if room not in rooms:
                    dict.__setitem__(rooms, room, RoomPeers(self, room))
                dict.__setitem__(rooms[room], session, value)

            self.change(room, apply, "room-peer-changed")

    def publish(self, session, registration, action):
        with self.lock:
            old = self.manager.decode_kv_args_table.get(session)
            keys = [k for k, (_, sessions) in self.pins.items() if session in sessions]
            if (
                keys
                and old is not None
                and fingerprint(old) == fingerprint(registration)
            ):
                return None  # identical re-registration cannot unblacklist a live lease
            if keys:
                self.dirty.update(keys)
                self.pending.append((set(keys), action))
                return None
            return action()

    def pin(self, key, candidate):
        room = int(candidate[0].bootstrap_room)
        sessions = tuple(peer.mooncake_session_id for peer, _ in candidate[4])
        self.pins[key] = room, sessions

    def blocked(self, candidate):
        room = int(candidate[0].bootstrap_room)
        sessions = {peer.mooncake_session_id for peer, _ in candidate[4]}
        return any(
            k in self.dirty and (r == room or sessions.intersection(peers))
            for k, (r, peers) in self.pins.items()
        )

    def unpin(self, key):
        with self.lock:
            self.pins.pop(key, None)
            self.dirty.discard(key)
            blocked, ready = [], []
            for keys, action in self.pending:
                (blocked if keys.intersection(self.pins) else ready).append(
                    (keys, action)
                )
            self.pending = blocked
            for _, action in ready:
                action()  # lock order: guard -> native publish/send lock


@dataclass(eq=False)
class Certificate:
    key: tuple
    admission: object
    req: object
    batch: object
    version: int = 0
    active: bool = False
    done: bool = False
    candidate: object = None
    stamp: str | None = None
    reported: object = None
    revision: int = 0
    acked: int = 0
    notice: object = None
    assessment_stamp: object = None
    revoking: bool = False
    watch: object = None
    observed_ready: bool = False
    cache_lease: object = None

    @property
    def logical_key(self):
        return self.key[:-1]


class Admission:
    def __init__(self, overlap):
        self.overlap, self.owner = overlap, overlap.controller
        self.rank, self.size = self.owner.rank, self.owner.tp_size
        self.epoch = overlap.retirement.epoch
        self.mailbox = (
            Mailbox(self.epoch + "-admit", self.rank) if self.size > 1 else None
        )
        self.records = {}
        self.guards = {}
        self.reports = {}
        self.acks = {}
        self.pending = self.inflight = None
        self.last_seq = 0
        self.births = 0
        self.bad_arm = False
        self.checks = self.revalidations = self.revocations = 0
        logger.info(
            "[dsv41-p-admission] enabled: TP certificates, cached middle-to-final decisions"
        )

    def guard(self, manager):
        guard = self.guards.get(id(manager))
        if guard is None:
            guard = self.guards[id(manager)] = RegistrationGuard(manager)
            manager._dsv41_admission_guard = guard
        return guard

    def candidate(self, cert):
        req = cert.req
        if (
            key_of(req) != cert.logical_key
            or getattr(req, "extend_end", None) is None
            or getattr(req.kv, "req_pool_idx", None) is None
        ):
            return None
        # Earlier native sends had no plugin read pin. Do not acquire a new
        # cache lease across their unknown physical-page history. Wait for a
        # local IO edge, then revalidate through the shared command path.
        if getattr(self.owner, "cache_guard", None) is not None:
            ledger = self.overlap.ledgers.get(int(req.bootstrap_room))
            if ledger is not None:
                with ledger.lock:
                    if ledger.pending or ledger.error:
                        return None
        return self.owner.candidates(cert.batch, allow_middle=True)[0]

    def local_change(self, cert):
        if key_of(cert.req) != cert.logical_key:
            return True
        manager = cert.req.disagg_kv_sender.kv_mgr
        guard = self.guard(manager)
        if cert.key in guard.dirty:
            return True
        if cert.req.pending_bootstrap:
            return True
        for peer, _ in cert.candidate[4]:
            session = peer.mooncake_session_id
            if session in getattr(manager, "failed_sessions", ()) or session in getattr(
                manager, "state_layout_rejections", ()
            ):
                return True
        # No schema rebuild or offer JSON read on the stable path.
        offer_path = cert.candidate[2] / "offer.json"
        try:
            modified = offer_path.stat().st_mtime_ns
        except OSError:
            return True
        if modified != cert.stamp:
            from .pre_layerwise import read_json

            if read_json(offer_path) != cert.candidate[3]:
                return True
            cert.stamp = modified
        return cert.req.disagg_kv_sender is not cert.candidate[1]

    def assess(self, cert):
        manager = cert.req.disagg_kv_sender.kv_mgr
        guard = self.guard(manager)
        with guard.lock:
            candidate = self.candidate(cert)
            if candidate is not None and guard.blocked(candidate):
                candidate = None
            if candidate is not None:
                cert.assessment_stamp = (candidate[2] / "offer.json").stat().st_mtime_ns
                cache = getattr(self.owner, "cache_guard", None)
                if cache is not None and cert.cache_lease is None:
                    cert.cache_lease = cache.hold(cert.req)
                guard.pin(cert.key, candidate)  # provisional pin closes check/vote race
            return candidate

    def watch(self, cert):
        req, manager = cert.req, cert.req.disagg_kv_sender.kv_mgr
        path = (
            self.owner.config.directory
            / str(int(req.bootstrap_room))
            / str(self.rank)
            / "offer.json"
        )
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = None
        guard = self.guard(manager)
        with guard.lock:
            peers = manager.transfer_infos.get(int(req.bootstrap_room), {})
            peer_stamp = tuple(
                (session, id(manager.decode_kv_args_table.get(session)))
                for session in peers
            )
            dirty = cert.key in guard.dirty
        ledger = self.overlap.ledgers.get(int(req.bootstrap_room))
        with ledger.lock if ledger is not None else nullcontext():
            old_io = bool(ledger is not None and (ledger.pending or ledger.error))
        page = getattr(getattr(self.owner, "pool", None), "page_size", 1)
        start, end = getattr(req, "start_send_idx", 0), getattr(req, "extend_end", None)
        cursor_ok = (
            end is not None
            and start < end
            and start % page == 0
            and getattr(req.disagg_kv_sender, "curr_idx", start // page)
            == start // page
        )
        return (
            key_of(req),
            req.pending_bootstrap,
            modified,
            peer_stamp,
            getattr(req.kv, "req_pool_idx", None) is not None,
            getattr(req, "extend_end", None) is not None,
            dirty,
            cursor_ok,
            getattr(req, "disagg_decode_prefix_len", 0),
            cert.batch.is_prefill_only,
            cert.batch.return_hidden_states,
            cert.batch.input_embeds is not None,
            cert.batch.replace_embeds is not None,
            old_io,
        )

    def commit(self, cert, candidate, accepted):
        cert.version += 1
        cert.active = bool(accepted)
        cert.candidate = candidate if accepted else None
        if accepted:
            cert.stamp = cert.assessment_stamp
        else:
            if cert.cache_lease is not None:
                cert.cache_lease.retire(lambda: None)
                cert.cache_lease = None
            self.guard(cert.req.disagg_kv_sender.kv_mgr).unpin(cert.key)
        cert.reported = None
        cert.watch = self.watch(cert)
        cert.observed_ready = candidate is not None
        self.reports.pop(cert.key, None)

    def select(self, batch):
        # This boundary is TP-symmetric: native batch keys/order + committed
        # certificates only. Local eligibility NEVER decides whether to vote.
        first, certs = [], []
        for req in batch.reqs:
            cert = getattr(req, "_dsv41_p_certificate", None)
            if cert is None or cert.logical_key != key_of(req):
                if cert is not None:
                    if cert.active:
                        raise RuntimeError(
                            "active P certificate changed without ordered drain"
                        )
                    cert.done = True
                    self.records.pop(cert.key, None)
                snapshot = SimpleNamespace(
                    reqs=[req],
                    **{
                        name: getattr(batch, name)
                        for name in (
                            "forward_mode",
                            "is_prefill_only",
                            "return_hidden_states",
                            "input_embeds",
                            "replace_embeds",
                        )
                    },
                )
                self.births += 1
                cert = Certificate((*key_of(req), self.births), self, req, snapshot)
                req._dsv41_p_certificate = cert
                self.records[cert.key] = cert
                first.append(cert)
            else:
                for name in (
                    "forward_mode",
                    "is_prefill_only",
                    "return_hidden_states",
                    "input_embeds",
                    "replace_embeds",
                ):
                    setattr(cert.batch, name, getattr(batch, name))
            certs.append(cert)
        if first:
            if (
                self.pending is not None
                or self.inflight is not None
                or self.overlap.retirement.pending is not None
            ):
                raise RuntimeError(
                    "P admission command not consumed before first check"
                )
            candidates, errors = [], []
            for cert in first:
                try:
                    candidates.append(self.assess(cert))
                except Exception as exc:
                    candidates.append(None)
                    errors.append(repr(exc))
            votes = self.owner.agree(
                [int(not errors), *[int(c is not None) for c in candidates]]
            )
            if not votes[0]:
                raise RuntimeError(
                    "P first admission failed; retaining provisional pins"
                )
            votes = votes[1:]
            self.last_seq += 1
            self.checks += 1
            for cert, candidate, take in zip(first, candidates, votes):
                self.commit(cert, candidate, take)
        batch_ok = (
            not batch.forward_mode.is_split_prefill()
            and not batch.forward_mode.is_mixed()
            and not batch.is_prefill_only
            and not batch.return_hidden_states
            and batch.input_embeds is None
            and batch.replace_embeds is None
        )
        return [
            c.candidate if c.active and not c.done and batch_ok else None for c in certs
        ], [int(c.active and not c.done and batch_ok) for c in certs]

    def observe(self):
        for cert in list(self.records.values()):
            if cert.done:
                continue
            try:
                if cert.active:
                    lease = getattr(cert.req, "_dsv41_p_lease", None)
                    changed = (
                        False
                        if lease is not None and lease.finalized
                        else self.local_change(cert)
                    )
                    ready = not changed
                else:
                    watch = self.watch(cert)
                    if watch != cert.watch:
                        cert.observed_ready = self.candidate(cert) is not None
                        cert.watch = watch
                    changed, ready = False, cert.observed_ready
            except Exception:
                # A local read/validation error cannot choose a collective or
                # escape into a different model path. Revoke an active attempt;
                # an inactive attempt remains native and cannot be activated.
                changed, ready = cert.active, False
            state = (cert.version, ready, changed)
            if cert.reported != state:
                cert.revision += 1
                cert.notice = dict(
                    epoch=self.epoch,
                    key=list(cert.key),
                    version=cert.version,
                    revision=cert.revision,
                    rank=self.rank,
                    ready=ready,
                    revoke=changed,
                )
                cert.reported = state
            if self.rank == 0:
                self.accept(cert.notice)
            elif cert.acked < cert.revision:
                self.mailbox.send(cert.notice)  # resend until native broadcast ACK

    def accept(self, notice):
        if not isinstance(notice, dict) or notice.get("epoch") != self.epoch:
            return
        try:
            key = tuple(notice["key"])
            cert = self.records.get(key)
        except (TypeError, KeyError):
            return
        rank, revision = notice.get("rank"), notice.get("revision")
        if (
            cert is None
            or cert.done
            or type(rank) is not int
            or not 0 <= rank < self.size
            or notice.get("version") != cert.version
            or type(revision) is not int
            or type(notice.get("ready")) is not bool
            or type(notice.get("revoke")) is not bool
        ):
            return
        old = self.reports.setdefault(key, {}).get(rank)
        if old is None or revision > old["revision"]:
            self.reports[key][rank] = notice
        if rank != 0:
            self.acks[key, rank] = max(revision, self.acks.get((key, rank), 0))

    def proposal(self, *, allow_command):
        if self.rank != 0:
            return None
        if self.mailbox is not None:
            for notice in self.mailbox.receive():
                self.accept(notice)
        entries = []
        if allow_command and self.inflight is None:
            for key, cert in self.records.items():
                reports = self.reports.get(key, {})
                revoke = cert.active and any(n["revoke"] for n in reports.values())
                ready = (
                    not cert.active
                    and len(reports) == self.size
                    and all(n["ready"] for n in reports.values())
                )
                if not cert.done and (revoke or ready):
                    entries.append(
                        dict(key=list(key), version=cert.version, revoke=revoke)
                    )
        command = None
        if entries:
            command = dict(epoch=self.epoch, seq=self.last_seq + 1, entries=entries)
            self.inflight = command
        acks = [
            dict(key=list(key), rank=rank, revision=revision)
            for (key, rank), revision in self.acks.items()
        ]
        self.acks.clear()
        return (
            dict(epoch=self.epoch, command=command, acks=acks)
            if command or acks
            else None
        )

    def arm(self, envelope):
        if envelope.get("epoch") != self.epoch:
            raise RuntimeError("P admission epoch mismatch")
        for ack in envelope.get("acks", []):
            cert = self.records.get(tuple(ack["key"]))
            if cert is not None and ack["rank"] == self.rank:
                cert.acked = max(cert.acked, ack["revision"])
        command = envelope.get("command")
        if command is not None:
            if self.pending is not None:
                self.bad_arm = True
            else:
                self.pending = command

    def finalize(self):
        if self.pending is None:
            return
        command, self.pending = self.pending, None
        entries = command["entries"]
        valid = (
            command.get("epoch") == self.epoch
            and command.get("seq") == self.last_seq + 1
            and not self.bad_arm
        )
        self.bad_arm = False
        certs, candidates, flags = [], [], []
        for item in entries:
            cert = self.records.get(tuple(item["key"]))
            ok = cert is not None and not cert.done and cert.version == item["version"]
            try:
                candidate = self.assess(cert) if ok and not item["revoke"] else None
            except Exception:
                candidate, ok = None, False
            valid &= ok
            certs.append(cert)
            candidates.append(candidate)
            flags.append(int(ok and (item["revoke"] or candidate is not None)))
        votes = self.owner.agree([int(valid), *flags])
        self.last_seq = command["seq"]
        self.inflight = None
        if not votes[0]:
            raise RuntimeError("ordered P admission validation failed; retaining KV")
        for item, cert, candidate, take in zip(entries, certs, candidates, votes[1:]):
            if cert is None or cert.done:
                continue
            if item["revoke"]:
                if not take:
                    raise RuntimeError("P revoke identity mismatch; retaining KV")
                # A peer/layout replacement invalidates the native PD attempt,
                # too. Never continue a partially exported request against a new
                # destination. Drain source readers, abort using the OLD pinned
                # view, then make the deferred registration visible.
                cert.revoking = True
                lease = getattr(cert.req, "_dsv41_p_lease", None)
                if lease is not None:
                    lease.cancelling = True
                    self.overlap.retirement.drain(lease)
                from sglang.srt.disaggregation.utils import prepare_abort

                prepare_abort(cert.req, "PD admission qualification changed")
                cert.req.disagg_kv_sender.abort()
                scheduler = self.overlap.scheduler
                if getattr(scheduler, "chunked_req", None) is cert.req:
                    scheduler._pending_chunked_abort_req = cert.req
                cert.revoking = False
                cert.active, cert.done = False, True
                if cert.cache_lease is not None and not cert.cache_lease.closed:
                    cert.cache_lease.retire(lambda: None)
                cert.cache_lease = None
                self.guard(cert.req.disagg_kv_sender.kv_mgr).unpin(cert.key)
                cert.version += 1
                cert.candidate = None
                self.records.pop(cert.key, None)
                self.reports.pop(cert.key, None)
                self.revocations += 1
            else:
                self.commit(cert, candidate, take)
                self.revalidations += 1

    def retired(self, req):
        cert = getattr(req, "_dsv41_p_certificate", None)
        if cert is not None and cert.logical_key == key_of(req):
            if cert.cache_lease is not None and not cert.cache_lease.closed:
                cert.cache_lease.retire(lambda: None)
            cert.cache_lease = None
            cert.done, cert.active = True, False
            self.records.pop(cert.key, None)
            self.reports.pop(cert.key, None)
            if not cert.revoking:
                self.guard(req.disagg_kv_sender.kv_mgr).unpin(cert.key)

    def close(self):
        if self.mailbox is not None:
            self.mailbox.close()


def split_controls(requests):
    ordinary, commands = [], []
    for value in requests or []:
        if isinstance(value, dict) and CONTROL in value:
            commands.append(value[CONTROL])
        else:
            ordinary.append(value)
    return ordinary, commands
