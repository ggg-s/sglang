"""Ordered D retirement: local notifications, broadcast intent, one final vote.

Only the scheduler thread calls torch.distributed. Unix datagrams are advisory
notifications within the existing same-host deployment; they never retire KV.
"""

import errno
import json
import logging
import os
import secrets
import socket
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
CONTROL = "__remote_attn_retire_intent__"


class Mailbox:
    def __init__(self, epoch, rank):
        self.path = Path(f"/tmp/sgl-ra-retire-{os.getuid()}-{epoch}.sock")
        self.rank = rank
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        if rank == 0:
            self.sock.bind(str(self.path))
            os.chmod(self.path, 0o600)

    def send(self, notice):
        data = json.dumps(notice).encode()
        if len(data) > 65536:
            raise RuntimeError("retirement notification exceeds mailbox limit")
        try:
            self.sock.sendto(data, str(self.path))
            return True
        except OSError as exc:
            # Rank 0 may still be binding at startup, or its queue may be full.
            if exc.errno in (
                errno.EAGAIN,
                errno.EWOULDBLOCK,
                errno.ENOBUFS,
                errno.ENOENT,
                errno.ECONNREFUSED,
            ):
                return False
            raise

    def receive(self):
        for _ in range(1024):
            try:
                data = self.sock.recv(65536)
            except BlockingIOError:
                return
            try:
                value = json.loads(data)
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict):
                yield value

    def close(self):
        self.sock.close()
        if self.rank == 0:
            self.path.unlink(missing_ok=True)


class Retirement:
    def __init__(self, controller):
        self.owner = controller
        self.rank, self.size = controller.rank, controller.tp_size
        self.group = getattr(controller.scheduler, "attn_tp_cpu_group", None)
        self.epoch = secrets.token_hex(16) if self.rank == 0 else None
        if self.size > 1:
            import torch.distributed as dist

            values = [self.epoch]
            dist.broadcast_object_list(
                values, src=dist.get_global_rank(self.group, 0), group=self.group
            )
            self.epoch = values[0]
        self.mailbox = Mailbox(self.epoch, self.rank) if self.size > 1 else None
        self.registered = {}
        self.last_admission = 0
        self.sent = {}
        self.outbox = {}
        self.revisions = {}
        self.report_versions = {}
        self.reports = {}
        self.needs_gpu_notice = set()
        self.last_seq = self.issued_seq = 0
        self.inflight = None
        self.pending = None
        self.arm_error = False
        self.final_votes = self.commits = self.retries = 0
        self.closed = False

    @staticmethod
    def identity(target):
        req = target.decode_req.req
        return {
            "id": target.admission_id,
            "rid": str(req.rid),
            "room": str(getattr(req, "bootstrap_room", "")),
        }

    def register(self, target):
        identity = self.identity(target)
        index = identity["id"]
        if type(index) is not int or index <= self.last_admission:
            raise RuntimeError("duplicate or missing TP retirement admission id")
        self.last_admission = index
        self.registered[index] = (target, identity, target.generation)

    def _active(self):
        return {t.admission_id: t for t in self.owner.targets}

    def _accept(self, notice):
        if notice.get("epoch") != self.epoch:
            return
        rank, identity = notice.get("rank"), notice.get("identity")
        if (
            type(rank) is not int
            or not 0 <= rank < self.size
            or not isinstance(identity, dict)
        ):
            return
        index = identity.get("id")
        if type(index) is not int:
            return
        entry = self.registered.get(index)
        if entry is None or identity != entry[1] or index not in self._active():
            return
        # Transport generations differ across TP ranks; the command carries
        # each rank's own generation, keyed by the common admission ordinal.
        if (
            not isinstance(notice.get("generation"), str)
            or type(notice.get("release")) is not bool
        ):
            return
        version = notice.get("version")
        if type(version) is not int or version <= self.report_versions.get(
            (index, rank), 0
        ):
            return
        self.report_versions[index, rank] = version
        self.reports.setdefault(index, {})[rank] = notice

    def observe(self):
        # Empty Local does not query events, touch sockets, or enter a collective.
        for target in self.owner.targets:
            entry = self.registered.get(target.admission_id)
            if entry is None:
                raise RuntimeError(
                    "progressive target lacks ordered retirement registration"
                )
            error = None
            try:
                gpu_ready = target.gpu_ready()  # Advisory; never retires KV.
                ready = target.transport_ready()
                if target.admission_id in self.needs_gpu_notice and not gpu_ready:
                    continue
            except Exception as exc:
                error, ready = f"{type(exc).__name__}: {str(exc)[:256]}", True
            if not ready:
                continue
            state = (target.release is not None, error)
            index = target.admission_id
            if self.sent.get(index) == state:
                continue
            pending = self.outbox.get(index)
            if pending is None or pending[0] != state:
                self.revisions[index] = self.revisions.get(index, 0) + 1
                notice = {
                    "epoch": self.epoch,
                    "rank": self.rank,
                    "identity": entry[1],
                    "generation": entry[2],
                    "release": state[0],
                    "error": error,
                    "version": self.revisions[index],
                }
                self.outbox[index] = (state, notice)
            else:
                notice = pending[1]
            if self.rank == 0:
                self._accept(notice)
                delivered = True
            else:
                delivered = self.mailbox.send(notice)
            if delivered:
                self.sent[index] = state
                self.outbox.pop(index, None)
                self.needs_gpu_notice.discard(index)

    def _command(self, entries, kind):
        self.issued_seq += 1
        command = {
            "epoch": self.epoch,
            "seq": self.issued_seq,
            "kind": kind,
            "targets": entries,
        }
        self.inflight = command
        return command

    def poll_command(self):
        if self.rank != 0 or self.inflight is not None or not self.owner.targets:
            return None
        if self.mailbox is not None:
            for notice in self.mailbox.receive():
                self._accept(notice)
        entries, abort = [], False
        for index, target in self._active().items():
            reports = self.reports.get(index, {})
            failed = any(n.get("error") for n in reports.values())
            if not failed and (
                len(reports) != self.size
                or len({n["release"] for n in reports.values()}) != 1
            ):
                continue
            abort |= failed
            entries.append(
                {
                    **self.identity(target),
                    "generations": [
                        reports[r]["generation"] if r in reports else None
                        for r in range(self.size)
                    ],
                }
            )
        return (
            self._command(entries, "abort" if abort else "retire") if entries else None
        )

    def inject(self, requests):
        command = self.poll_command()
        return (
            [*(requests or []), {CONTROL: command}] if command is not None else requests
        )

    def arm(self, command):
        if self.pending is not None:
            self.arm_error = True
        else:
            self.pending = command

    def finalize(self, *, wait=False):
        if self.pending is None:
            return
        command, self.pending = self.pending, None
        entries = command.get("targets", []) if isinstance(command, dict) else []
        if not isinstance(entries, list):
            entries = []
        protocol_ok = (
            isinstance(command, dict)
            and command.get("epoch") == self.epoch
            and type(command.get("seq")) is int
            and command["seq"] == self.last_seq + 1
            and command.get("kind") in ("retire", "drain", "abort")
            and bool(entries)
            and not self.arm_error
        )
        self.arm_error = False
        active, selected, ready, releases = self._active(), [], [], []
        error_free = (
            command.get("kind") != "abort" if isinstance(command, dict) else False
        )
        seen = set()
        deadline = time.monotonic() + self.owner.config.drain_timeout
        for item in entries:
            index = item.get("id") if isinstance(item, dict) else None
            if type(index) is not int:
                index = None
            entry = self.registered.get(index)
            target = active.get(index)
            valid = entry is not None and target is entry[0] and index not in seen
            if valid:
                generations = item.get("generations")
                valid = (
                    {k: item.get(k) for k in ("id", "rid", "room")} == entry[1]
                    and self.identity(target) == entry[1]
                    and target.generation == entry[2]
                    and (
                        generations is None
                        or (
                            isinstance(generations, list)
                            and len(generations) == self.size
                            and (
                                generations[self.rank] == entry[2]
                                or command.get("kind") == "abort"
                            )
                        )
                    )
                )
            protocol_ok &= valid
            seen.add(index)
            selected.append(target)
            local_ready = False
            if valid:
                try:
                    local_ready = target.ready_to_finish()
                    while wait and not local_ready:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("D GPU/PD lifetime did not drain")
                        time.sleep(0.005)
                        local_ready = target.ready_to_finish()
                except Exception:
                    error_free = False
            ready.append(int(local_ready))
            releases.append(int(target is not None and target.release is not None))
        # Fixed vector shape/order comes only from the shared command, never
        # from a rank-local target list or local readiness branch.
        votes = self.owner.agree(
            [
                int(protocol_ok),
                int(error_free),
                *ready,
                *releases,
                *(1 - r for r in releases),
            ]
        )
        self.final_votes += 1
        if votes[0]:
            # Even a failed GPU drain consumed this shared command. Do not
            # let a later explicit recovery boundary reuse or skip its seq.
            self.last_seq = command["seq"]
            self.inflight = None
        if not votes[0] or not votes[1]:
            raise RuntimeError(
                "ordered D retirement validation failed; proof unproven, retaining KV"
            )
        n = len(entries)
        accepted = all(
            votes[2 + i] and (votes[2 + n + i] or votes[2 + 2 * n + i])
            for i in range(n)
        )
        self.inflight = None
        if accepted:
            for target in selected:
                target.finish(wait=False)
                self.owner.targets.remove(target)
                index = target.admission_id
                self.registered.pop(index, None)
                self.sent.pop(index, None)
                self.outbox.pop(index, None)
                self.revisions.pop(index, None)
                for rank in range(self.size):
                    self.report_versions.pop((index, rank), None)
                self.reports.pop(index, None)
                self.needs_gpu_notice.discard(index)
            self.commits += 1
            logger.info(
                "[remote-attn-retire] committed seq=%s targets=%s", self.last_seq, n
            )
        else:
            self.retries += 1
            # Republish current state once. A persistent release mismatch then
            # blocks proposals until another real state edge, not another loop.
            for i, item in enumerate(entries):
                self.sent.pop(item["id"], None)
                self.outbox.pop(item["id"], None)
                self.reports.pop(item["id"], None)
                if not votes[2 + i]:
                    self.needs_gpu_notice.add(item["id"])
            logger.info(
                "[remote-attn-retire] retry after state change seq=%s", self.last_seq
            )

    def drain(self, required=None):
        # Called only at TP-symmetric native retraction/shutdown boundaries.
        # Consume any earlier broadcast intent before reserving another seq.
        self.finalize(wait=True)
        if not self.owner.targets or (required is not None and required.closed):
            return
        command = None
        if self.rank == 0:
            targets = self.owner.targets if required is None else [required]
            command = self._command(
                [{**self.identity(t), "generations": None} for t in targets], "drain"
            )
        if self.size > 1:
            import torch.distributed as dist

            values = [command]
            dist.broadcast_object_list(
                values, src=dist.get_global_rank(self.group, 0), group=self.group
            )
            command = values[0]
        self.arm(command)
        self.finalize(wait=True)
        if (
            required is None
            and self.owner.targets
            or required is not None
            and not required.closed
        ):
            raise RuntimeError(
                "ordered D drain could not agree on release state; retaining KV"
            )

    def close(self):
        if not self.closed:
            self.closed = True
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


def wrap_receiver(receiver, retirement):
    # Extend the frozen receiver through its public dataclass constructor;
    # preserve every native field and the existing broadcast hook unchanged.
    if not is_dataclass(receiver):
        raise RuntimeError(
            "ordered retirement requires the native dataclass request receiver"
        )
    base = type(receiver)

    @dataclass(frozen=True, slots=True, kw_only=True)
    class OrderedReceiver(base):
        retire_protocol: object

        def _broadcast_reqs_across_ranks(self, recv_reqs, local_reqs=None):
            return base._broadcast_reqs_across_ranks(
                self, self.retire_protocol.inject(recv_reqs), local_reqs
            )

    return OrderedReceiver(
        **{f.name: getattr(receiver, f.name) for f in fields(receiver)},
        retire_protocol=retirement,
    )
