"""DSV mode of the experimental P4 gateway: native PD, D2D, SSE and recovery.

One durable coordinator owns migration intents. A COMMIT intent is irreversible
from its perspective, including when an RPC reply was lost. Client SSE uses
sequence ids; a reconnect supplies Last-Event-ID/after to acknowledge receipt.
"""

from __future__ import annotations

from collections import deque, namedtuple
import copy
import fcntl
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qs

PROTOCOL = "dsv-d2d-v2"
TERMINAL = {"DONE", "CANCELLED"}
PRECOMMIT = {"PLAN", "RESERVE", "WAIT_CUT", "FREEZE", "IMPORT"}
ACTIVE_MIGRATION_PHASES = PRECOMMIT | {"COMMIT", "ROLLBACK", "TARGET"}
STATUS_VISIBILITY_GRACE_S = 30
WAVE_SIZE = 8
WAVE_PHASES = {"RESERVE", "WAIT_CUT", "FREEZE", "IMPORT", "COMMIT"}
Command = namedtuple("Command", "role name epoch payload")


# Immutable large fields never participate in hot intent rewrites/deepcopies.
PAYLOAD_FIELDS = (
    "input_ids",
    "plans",
    "reservations",
    "transfers",
    "admission_plans",
    "precopy_receipts",
)


HANDOFF_MARKERS = (
    "cut_observed_at",
    "freeze_requested_at",
    "freeze_descriptor_ready_at",
    "import_ready_at",
    "target_admitted_at",
    "dst_first_token_observed_at",
)


def handoff_timing(entry):
    timing = {key: entry.get(key) for key in HANDOFF_MARKERS}
    for name, start, end in zip(
        (
            "cut_to_freeze_request",
            "freeze_request_to_descriptor",
            "descriptor_to_import",
            "import_to_target",
            "target_to_first_token",
        ),
        HANDOFF_MARKERS,
        HANDOFF_MARKERS[1:],
    ):
        a, b = timing[start], timing[end]
        timing[name + "_ms"] = (
            (b - a) * 1000 if a is not None and b is not None and b >= a else None
        )
    a, b = timing[HANDOFF_MARKERS[0]], timing[HANDOFF_MARKERS[-1]]
    timing["cut_to_first_token_ms"] = (
        (b - a) * 1000 if a is not None and b is not None and b >= a else None
    )
    admitted = timing["target_admitted_at"]
    for marker in (
        "dst_copy_observed_at", "dst_local_observed_at", "src_release_observed_at"
    ):
        observed = timing[marker] = entry.get(marker)
        timing["target_to_" + marker.removesuffix("_at") + "_ms"] = (
            (observed - admitted) * 1000
            if observed is not None and admitted is not None and observed >= admitted
            else None
        )
    return timing


def remaining_copy_bytes(transfers):
    """Unique planned payload after sealed precopy, including import state."""
    by_rank = []
    for t in transfers:
        try:
            schema = t["schema"]
            page = schema["page_size"]
            pages = (t["kv_len"] + page - 1) // page
            entries = schema["entries"]
            prefixes = (t.get("precopy") or {}).get("prefix_pages", [0] * len(entries))
            if len(prefixes) != len(entries) or any(
                not 0 <= n <= pages for n in prefixes
            ):
                return None
            by_rank.append(
                sum((pages - n) * e[-1] for e, n in zip(entries, prefixes))
                + sum(d["rows"] * d["stride"] for d in t["states"])
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    return dict(by_rank=by_rank, tp_sum=sum(by_rank))


class Payloads:
    def __init__(self, root):
        self.root, self.cache = root, {}
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, value):
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        key = hashlib.sha256(raw).hexdigest()
        path = self.root / (key + ".json")
        if key not in self.cache:
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != key:
                    raise ValueError("corrupt D2D journal payload")
            else:
                tmp = self.root / (key + "." + uuid.uuid4().hex + ".tmp")
                try:
                    with tmp.open("xb") as stream:
                        os.fchmod(stream.fileno(), 0o600)
                        stream.write(raw)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(tmp, path)
                    fd = os.open(self.root, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                finally:
                    tmp.unlink(missing_ok=True)
            # Independent immutable snapshot, never alias caller-owned arrays.
            self.cache[key] = json.loads(raw)
        return key

    def get(self, key):
        if re.fullmatch(r"[0-9a-f]{64}", key or "") is None:
            raise ValueError("invalid D2D journal payload reference")
        if key not in self.cache:
            raw = (self.root / (key + ".json")).read_bytes()
            if hashlib.sha256(raw).hexdigest() != key:
                raise ValueError("corrupt D2D journal payload")
            self.cache[key] = json.loads(raw)
        return self.cache[key]

    def pack(self, fields):
        fields = dict(fields)
        for name in PAYLOAD_FIELDS:
            if name in fields:
                value = fields.pop(name)
                fields[name + "_ref"] = None if value is None else self.put(value)
        return fields

    def view(self, entry, *, copy_payloads=False):
        value = dict(entry)
        for name in PAYLOAD_FIELDS:
            if name + "_ref" in value:
                key = value[name + "_ref"]
                payload = None if key is None else self.get(key)
                value[name] = copy.deepcopy(payload) if copy_payloads else payload
        return value


class DrainConflict(RuntimeError):
    pass


class StatusBusy(Exception):
    """Skip advisory polling while a mutation is queued; keep the last facts."""


def latency_histogram(samples):
    limits = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
    values = list(samples)
    return dict(
        count=len(values),
        sum_ms=sum(values),
        max_ms=max(values, default=0),
        buckets={
            **{str(n): sum(v <= n for v in values) for n in limits},
            "+Inf": len(values),
        },
    )


class DsvGateway:
    protocol = PROTOCOL
    journal_filename = "dsv_gateway_journal.json"

    def __init__(
        self,
        driver,
        tokenizer,
        state_dir,
        *,
        prefill_url,
        bootstrap_host="127.0.0.1",
        bootstrap_port=35409,
        poll_ms=10,
        out=None,
        autostart=True,
        async_handoff=False,
        disable_precopy=False,
        decode_workers=None,
        precopy_window_ms=0,
    ):
        self.driver, self.tokenizer = driver, tokenizer
        self.decode_workers = decode_workers or {}
        self.control_workers = tuple(self.decode_workers or ("src", "dst"))
        self.precopy_window_s = precopy_window_ms / 1000
        if precopy_window_ms < 0:
            raise ValueError("precopy window must be nonnegative")
        self.prefill_url, self.bootstrap_host = prefill_url, bootstrap_host
        self.bootstrap_port = int(bootstrap_port)
        self.poll_s = max(float(poll_ms) / 1000, 0.003)
        self.out = Path(out) if out else None
        self.path = Path(state_dir) / self.journal_filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payloads = Payloads(self.path.parent / "dsv_payloads")
        self.tick_cursor = 0
        self.tick_samples = deque(maxlen=256)
        self.precopy_frame_samples = deque(maxlen=256)
        self.precopy_stage_samples = {}
        self.disable_precopy = disable_precopy
        self.precopy_events = deque(maxlen=256)
        self.precopy_active = {}
        self.precopy_ready = {}
        self.lock = threading.RLock()
        self.work_lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.facts, self.fact_errors = {}, {}
        self.facts_at = {}
        self.connections = set()
        self.role_locks = {role: threading.Lock() for role in self.control_workers}
        self.command_waiters = dict.fromkeys(self.control_workers, 0)
        self.dst_slot_holder = None
        self.dst_slot_history = deque(maxlen=256)
        self.control_latencies = {}
        self.control_breakdown = {}
        self.control_wave_sizes = {}
        self.control_wave_members = {}
        self.background = autostart
        self.async_handoff = async_handoff
        self.status_threads = []
        self.fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.journal = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else dict(
                    version=2,
                    protocol=PROTOCOL,
                    entries={},
                    drain=None,
                )
            )
            if (
                self.journal.get("version") not in (1, 2)
                or self.journal.get("protocol") != PROTOCOL
            ):
                raise ValueError(
                    "unsupported DSV gateway journal; refusing to reset it"
                )
            if not isinstance(self.journal.get("entries"), dict):
                raise ValueError("invalid DSV gateway journal entries")
            for rid, entry in self.journal["entries"].items():
                if (
                    not isinstance(entry, dict)
                    or entry.get("rid") != rid
                    or (
                        entry.get("phase")
                        not in TERMINAL
                        | PRECOMMIT
                        | {
                            "SUBMIT",
                            "SOURCE",
                            "DIRECT",
                            "COMMIT",
                            "TARGET",
                            "LOCAL",
                            "ROLLBACK",
                        }
                    )
                    or not (entry.get("input_ids") or entry.get("input_ids_ref"))
                    or not {
                        "owner",
                        "initial_owner",
                        "instances",
                        "source_rid",
                        "target_rid",
                        "submission_epoch",
                        "sampling",
                        "bootstrap_room",
                        "bootstrap_host",
                        "bootstrap_port",
                        "prefill_url",
                        "max_new",
                        "cut",
                        "growth_tokens",
                        "cancel_requested",
                        "cut_attempted",
                        "migration_requested",
                        "attempts",
                    }.issubset(entry)
                ):
                    raise ValueError(
                        "incomplete DSV gateway journal; refusing recovery"
                    )
            self._validate_worker_bindings()
            migrated = self.journal["version"] == 1
            for rid, entry in list(self.journal["entries"].items()):
                self.journal["entries"][rid] = self.payloads.pack(entry)
                self.payloads.view(self.journal["entries"][rid])
            self.journal["version"] = 2
            if migrated:
                self._persist()
        except BaseException:
            os.close(self.fd)
            raise
        self.thread = None
        if autostart:
            for role in self.control_workers:
                poller = threading.Thread(
                    target=self._poll_role,
                    args=(role,),
                    daemon=True,
                    name="dsv-worker-status-" + role,
                )
                self.status_threads.append(poller)
                poller.start()
            self.thread = threading.Thread(
                target=self._loop, daemon=True, name="dsv-reconcile"
            )
            self.thread.start()

    def _validate_worker_bindings(self):
        configured = {name: str(path) for name, path in self.decode_workers.items()}
        if self.path.exists():
            if self.journal.get("decode_workers", {}) != configured:
                raise ValueError("migration journal Decode registry changed")
        elif configured:
            self.journal.update(decode_workers=configured, fenced_workers={})

    @staticmethod
    def _worker(entry, role):
        workers = entry.get("workers") if entry is not None else None
        return role if workers is None else workers[role]

    def _call_for(self, entry, role, cmd, epoch, **payload):
        if self.decode_workers:
            payload["expected_instances"] = entry["instances"][role]
        return self._call(self._worker(entry, role), cmd, epoch, **payload)

    def close(self):
        if self.fd < 0:
            return
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError(
                    "DSV coordinator still in RPC; keep journal lock held"
                )
        os.close(self.fd)
        self.fd = -1

    def _persist(self, journal=None):
        # A failed write must stop the transition before its side effect. The
        # caller updates in-memory state only after the durable file is replaced.
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(
                self.journal if journal is None else journal,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _update(self, rid, **changes):
        started = time.monotonic()
        try:
            result = self._update_impl(rid, **changes)
            if "phase" in changes and changes["phase"] != "WAIT_CUT":
                epoch = self.journal["entries"][rid].get("epoch")
                for key in list(self.precopy_ready):
                    if key[0] == epoch:
                        self.precopy_ready.pop(key, None)
            return result
        finally:
            trace = self.precopy_active.get(rid)
            if trace is not None:
                trace["journal_ms"] += (time.monotonic() - started) * 1000
                pending = changes.get("precopy_pending")
                if pending is not None and trace.get("frame") is None:
                    trace["frame"] = dict(pending["frame"])
                    trace["available_pages"] = pending.get("available_pages")
                    trace["available_pages_by_rank"] = pending.get("available_pages_by_rank")
                    trace["reservation_upper_pages"] = pending.get(
                        "reservation_upper_pages"
                    )

    def _update_impl(self, rid, **changes):
        with self.cv:
            changes = self.payloads.pack(changes)
            if all(
                self.journal["entries"][rid].get(key) == value
                for key, value in changes.items()
            ):
                return
            previous = copy.deepcopy(self.journal)
            self.journal["entries"][rid].update(changes, updated_at=time.time())
            try:
                self._persist()
            except BaseException:
                self.journal = previous
                raise
            self.cv.notify_all()
            if (
                self.out is not None
                and any(changes.get(key) is not None for key in (
                    "dst_first_token_observed_at", "src_release_observed_at"
                ))
            ):
                state = self._entry(rid)
                self._log_trace(
                    dict(
                        type="handoff_timing",
                        rid=rid,
                        epoch=state["epoch"],
                        handoff_timing=handoff_timing(state),
                        remaining_copy_bytes=state.get("remaining_copy_bytes"),
                    )
                )
            if self.out is not None and any(
                previous["entries"][rid].get(key)
                != self.journal["entries"][rid].get(key)
                for key in (
                    "phase",
                    "migration_deferred",
                    "migration_deferred_reason",
                    "freeze_pending",
                )
            ):
                try:
                    self.out.parent.mkdir(parents=True, exist_ok=True)
                    state = self._entry(rid)
                    with self.out.open("a") as metrics:
                        metrics.write(
                            json.dumps(
                                dict(
                                    handoff_timing=handoff_timing(state),
                                    remaining_copy_bytes=state.get(
                                        "remaining_copy_bytes"
                                    ),
                                    **{
                                        key: state.get(key)
                                        for key in (
                                            "rid",
                                            "epoch",
                                            "phase",
                                            "owner",
                                            "updated_at",
                                            "error",
                                            "migration_deferred",
                                            "migration_deferred_reason",
                                            "migration_deferred_count",
                                            "migration_deferred_wait_ms",
                                            "migration_outcome",
                                            "freeze_pending",
                                            "source_partial_remote",
                                            "migration_skip_reason",
                                            "plan_before_first_token",
                                            "freeze_ack_latency_ms",
                                            "prepare_latency_ms",
                                            "handoff_latency_ms",
                                            "precopy_bytes",
                                            "precopy_pages",
                                        )
                                    },
                                )
                            )
                            + "\n"
                        )
                except OSError:
                    pass  # The durable ownership journal remains authoritative.

    def entry(self, rid, *, copy_payloads=True):
        with self.lock:
            entry = self.payloads.view(
                copy.deepcopy(self.journal["entries"][rid]), copy_payloads=copy_payloads
            )
        entry["migration_deferred_wait_ms"] = self._deferred_wait_ms(entry)
        return entry

    @staticmethod
    def _deferred_wait_ms(entry):
        elapsed = entry.get("migration_deferred_ms", 0)
        since = entry.get("migration_deferred_since")
        if entry.get("migration_deferred") and since is not None:
            elapsed += max(0, time.time() - since) * 1000
        return elapsed

    def _entry(self, rid):
        return self.entry(rid, copy_payloads=False)

    def _defer_migration(self, entry, reason, **changes):
        active = entry.get("migration_deferred", False)
        changes.update(
            migration_deferred=True,
            migration_deferred_reason=reason,
            migration_deferred_since=entry["migration_deferred_since"]
            if active
            else time.time(),
            migration_deferred_count=entry.get("migration_deferred_count", 0)
            + int(not active or entry.get("migration_deferred_reason") != reason),
        )
        # Polling unchanged readiness must not rewrite the journal or inflate
        # the counter. Count actual waiting/reason transitions, not poll ticks.
        if any(entry.get(key) != value for key, value in changes.items()):
            self._update(entry["rid"], **changes)

    def _end_deferral(self, entry):
        return dict(
            migration_deferred=False,
            migration_deferred_reason=None,
            migration_deferred_since=None,
            migration_deferred_ms=self._deferred_wait_ms(entry),
        )

    @staticmethod
    def migration_fields(entry):
        return dict(
            migration_deferred=entry.get("migration_deferred", False),
            migration_deferred_reason=entry.get("migration_deferred_reason"),
            migration_deferred_count=entry.get("migration_deferred_count", 0),
            migration_deferred_wait_ms=entry.get("migration_deferred_wait_ms", 0),
            migration_outcome=entry.get("migration_outcome"),
            freeze_pending=entry.get("freeze_pending", False),
            source_partial_remote=entry.get("source_partial_remote", False),
            migration_skip_reason=entry.get("migration_skip_reason"),
            plan_before_first_token=entry.get("plan_before_first_token", False),
            freeze_ack_latency_ms=entry.get("freeze_ack_latency_ms"),
            prepare_latency_ms=entry.get("prepare_latency_ms"),
            handoff_latency_ms=entry.get("handoff_latency_ms"),
            precopy_bytes=entry.get("precopy_bytes", 0),
            precopy_pages=entry.get("precopy_pages"),
            handoff_timing=handoff_timing(entry),
            remaining_copy_bytes=entry.get("remaining_copy_bytes"),
        )

    @staticmethod
    def rows(response):
        rows = response.get("tp_rank_results") or [response]
        if not rows or any(r.get("ok") is False or r.get("errors") for r in rows):
            raise RuntimeError("DSV command failed on a TP rank")
        return rows

    def _call(self, role, cmd, epoch="worker", **payload):
        status = cmd == "d2d_worker_status"
        started = time.monotonic()
        acquired = False
        rows = ()
        slot_trace = holder = None
        commands = (
            [
                dict(cmd=c.get("cmd"), epoch=c.get("migration_epoch"))
                for c in payload.get("commands", [])
            ]
            if cmd == "d2d_control_wave"
            else [dict(cmd=cmd, epoch=epoch)]
        )
        with self.lock:
            if status and self.command_waiters[role]:
                raise StatusBusy()
            if not status:
                self.command_waiters[role] += 1
        try:
            if not status and "expected_instances" not in payload:
                payload["expected_instances"] = [
                    r["instance_id"] for r in self.facts[role]
                ]
            acquired = (
                self.role_locks[role].acquire(blocking=False)
                if status
                else self.role_locks[role].acquire(timeout=1)
            )
            if not acquired:
                if status:
                    raise StatusBusy()
                raise TimeoutError(f"{role} control busy; command not submitted")
            submitted = time.monotonic()
            submitted_at = time.time()
            if role == "dst":
                holder = dict(cmd=cmd, members=commands, started=submitted)
                with self.lock:
                    self.dst_slot_holder = holder
            command = dict(cmd=cmd, migration_epoch=epoch, **payload)
            timeout = 2 if status else 30
            if self.decode_workers:
                response = self.driver.send_to(
                    self.decode_workers[role], role, command, timeout=timeout
                )
            else:
                response = self.driver.send(role, command, timeout=timeout)
            return self.rows(response)
        finally:
            ended = time.monotonic()
            with self.lock:
                if role == "dst" and not status:
                    waited_until = submitted if acquired else ended
                    blockers = []
                    holders = list(self.dst_slot_history)
                    if (
                        self.dst_slot_holder is not None
                        and self.dst_slot_holder is not holder
                    ):
                        holders.append(dict(self.dst_slot_holder, ended=ended))
                    for previous in reversed(holders):
                        if previous["ended"] <= started:
                            break
                        overlap = min(previous["ended"], waited_until) - max(
                            previous["started"], started
                        )
                        if overlap > 0:
                            blockers.append(
                                dict(
                                    cmd=previous["cmd"],
                                    members=previous["members"],
                                    overlap_ms=overlap * 1000,
                                )
                            )
                    slot_trace = dict(
                        slot_wait_ms=(waited_until - started) * 1000,
                        holders=blockers,
                    )
                if holder is not None:
                    self.dst_slot_history.append(dict(holder, ended=ended))
                    self.dst_slot_holder = None
                if acquired:
                    self.role_locks[role].release()
                for trace in list(self.precopy_active.values()):
                    if any(c["epoch"] == trace["epoch"] for c in commands):
                        trace["slot_wait_ms"] += (
                            (submitted if acquired else ended) - started
                        ) * 1000
                        if acquired:
                            trace["rpc_ms"] += (ended - submitted) * 1000
                        trace["rpc_attempts"] += 1
                        if slot_trace is not None:
                            trace.setdefault("dst_slot_waits", []).append(
                                dict(cmd=cmd, members=commands, **slot_trace)
                            )
                        if cmd == "d2d_control_wave":
                            trace["wave_envelope"] = True
                if not status:
                    self.command_waiters[role] -= 1
                if acquired:
                    key = role + "/" + cmd
                    self._observe(
                        self.control_latencies, key, (time.monotonic() - started) * 1000
                    )
                    breakdown = self.control_breakdown.setdefault(key, {})
                    self._observe(breakdown, "gateway_slot_wait", (submitted - started) * 1000)
                    for field in ("wait_before_exec_ms", "exec_ms"):
                        times = [r.get("_ctl", {}).get(field) for r in rows]
                        if times and all(t is not None for t in times):
                            self._observe(breakdown, field, max(times))
            if slot_trace is not None or any(
                c["cmd"] == "d2d_freeze" for c in commands
            ):
                self._log_trace(
                    dict(
                        type="d2d_control_slot",
                        role=role,
                        cmd=cmd,
                        members=commands,
                        submitted_at=submitted_at if acquired else None,
                        rpc_ms=(ended - submitted) * 1000 if acquired else None,
                        **(
                            slot_trace
                            or dict(
                                slot_wait_ms=(
                                    (submitted if acquired else ended) - started
                                )
                                * 1000,
                                holders=[],
                            )
                        ),
                    )
                )

    def _log_trace(self, trace):
        if self.out is not None:
            try:
                with self.out.open("a") as stream:
                    stream.write(json.dumps(trace) + "\n")
            except OSError:
                pass

    @staticmethod
    def _observe(metrics, key, value):
        sample = latency_histogram([value])
        old = metrics.get(key)
        if old:
            sample["count"] += old["count"]
            sample["sum_ms"] += old["sum_ms"]
            sample["max_ms"] = max(sample["max_ms"], old["max_ms"])
            for bucket in sample["buckets"]:
                sample["buckets"][bucket] += old["buckets"][bucket]
        metrics[key] = sample

    def latency_metrics(self):
        with self.lock:
            entries = list(self.journal["entries"].values())
            skipped, outcomes = {}, {}
            for entry in entries:
                outcome = entry.get("migration_outcome")
                if outcome:
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
                reason = entry.get("migration_skip_reason")
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
            return dict(
                control_latency_ms=copy.deepcopy(self.control_latencies),
                control_breakdown_ms=copy.deepcopy(self.control_breakdown),
                control_wave_sizes=copy.deepcopy(self.control_wave_sizes),
                control_wave_members=dict(self.control_wave_members),
                precopy_stage_wait_ms=copy.deepcopy(self.precopy_stage_samples),
                precopy_enabled=not self.disable_precopy,
                precopy_frame_events=copy.deepcopy(list(self.precopy_events)),
                precopy_bytes_scope="applied_payload_sum_across_tp_ranks",
                tick_recent_latency_ms=latency_histogram(self.tick_samples),
                journal_bytes=self.path.stat().st_size if self.path.exists() else 0,
                journal_version=self.journal["version"],
                handoff_latency_ms=latency_histogram(
                    e["handoff_latency_ms"]
                    for e in entries
                    if e.get("handoff_latency_ms") is not None
                ),
                migration_skip_reasons=skipped,
                migration_outcomes=outcomes,
                precopy_frame_recent_latency_ms=latency_histogram(
                    self.precopy_frame_samples
                ),
            )

    @staticmethod
    def _check_worker(rows):
        if not rows or any(
            r.get("protocol") != PROTOCOL
            or r.get("rank") != rank
            or r.get("tp_size") != len(rows)
            or not r.get("instance_id")
            for rank, r in enumerate(rows)
        ):
            raise RuntimeError("DSV worker protocol/TP identity mismatch")

    def _refresh_role(self, role):
        with self.lock:
            drain = self.journal["drain"]
            if self.decode_workers:
                if role in self.journal["fenced_workers"] and (
                    role != drain["src"] or drain.get("source_released_at")
                ):
                    return
            elif role == "src" and (drain or {}).get("stopped"):
                return
        try:
            rows = self._call(role, "d2d_worker_status")
            self._check_worker(rows)
            with self.cv:
                drain = self.journal["drain"]
                if (
                    self.decode_workers
                    and drain
                    and drain["active"]
                    and drain.get("abort_requested_at") is None
                    and not drain.get("source_released_at")
                ):
                    expected = drain["instances"].get(role)
                    if expected is not None and [r["instance_id"] for r in rows] != expected:
                        raise RuntimeError(f"{role} restarted during migration")
                self.facts[role] = rows
                self.facts_at[role] = time.monotonic()
                self.fact_errors.pop(role, None)
                self.cv.notify_all()
        except (StatusBusy, BlockingIOError, TimeoutError):
            # Advisory status may miss its short deadline while a control wave
            # or KV copy occupies the worker. Keep the last validated facts and
            # any existing error until a successful poll; this is not death proof.
            return
        except Exception as exc:
            with self.cv:
                self.fact_errors[role] = repr(exc)
                self.cv.notify_all()

    def _poll_role(self, role):
        while not self.stop.is_set():
            self._refresh_role(role)
            self.stop.wait(self.poll_s)

    def _refresh(self):
        for role in self.control_workers:
            self._refresh_role(role)

    def _bound(self, entry, roles=None):
        for role in roles or entry["instances"]:
            worker = self._worker(entry, role)
            if worker in self.fact_errors or worker not in self.facts:
                raise ConnectionError(f"{worker} unavailable; ownership unchanged")
            actual = [r["instance_id"] for r in self.facts[worker]]
            if actual != entry["instances"][role]:
                raise RuntimeError(
                    f"{role} worker restarted; refusing stale epoch recovery"
                )

    def _request(self, entry, role):
        rid = entry["source_rid"] if role == "src" else entry["target_rid"]
        rows = [
            r.get("requests", {}).get(rid)
            for r in self.facts.get(self._worker(entry, role), ())
        ]
        if not rows or any(row is None for row in rows):
            return None
        outputs = [r["output_ids"] for r in rows]
        # Polls may land at adjacent scheduler boundaries on TP shards. Only
        # publish the common prefix; never mistake lag for different ownership.
        size = min(map(len, outputs))
        prefix = outputs[0][:size]
        finished = all(r["finished"] for r in rows)
        if any(o[:size] != prefix or (finished and len(o) != size) for o in outputs):
            raise RuntimeError("TP output token history diverged")
        return dict(
            output_ids=prefix,
            finished=finished,
            finish_reason=rows[0].get("finish_type") if finished else None,
            kv_released=all(r["kv_released"] for r in rows),
            plan_ready=all(
                r.get("plan_ready", r.get("migration_ready", False)) for r in rows
            ),
            plan_deferred_reason=next(
                (
                    r.get("plan_deferred_reason")
                    or r.get("migration_deferred_reason")
                    or "worker_not_ready"
                    for r in rows
                    if not r.get("plan_ready", r.get("migration_ready", False))
                ),
                None,
            ),
            migration_ready=all(r.get("migration_ready", False) for r in rows),
            migration_deferred_reason=next(
                (
                    r.get("migration_deferred_reason") or "worker_not_ready"
                    for r in rows
                    if not r.get("migration_ready", False)
                ),
                None,
            ),
            error=next((r.get("error") for r in rows if r.get("error")), None),
        )

    def _states(self, entry, role):
        return [
            r.get("migrations", {}).get(entry["epoch"], {}) for r in self.facts[self._worker(entry, role)]
        ]

    def _admission(self, role):
        drain = self.journal["drain"]
        role = role or ("dst" if drain and drain["active"] else "src")
        if role not in self.facts or role in self.fact_errors:
            raise RuntimeError(f"{role} worker not ready")
        return dict(
            owner=role,
            initial_owner=role,
            instances={role: [r["instance_id"] for r in self.facts[role]]},
            prefill_url=self.prefill_url,
            bootstrap_host=self.bootstrap_host,
            bootstrap_port=self.bootstrap_port,
        )

    def _external_admission(self, role, bootstrap):
        raise ValueError("this controller does not accept external PD requests")

    def accept(self, body, role=None, *, external_pd=None):
        prompt = body.get("prompt_ids", body.get("input_ids", body.get("prompt")))
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(x) is not int or x < 0 for x in prompt)
        ):
            raise ValueError("DSV P4 requires nonempty prompt token ids or prompt text")
        if not self.prefill_url and not self.decode_workers:
            raise ValueError("DSV P4 requires HANDOFF_PD_PREFILL_URL")
        rid = str(body.get("rid") or uuid.uuid4().hex)
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", rid) is None:
            raise ValueError("invalid P4 request rid")
        sampling = dict(body.get("sampling_params") or {})
        max_new = int(
            body.get(
                "max_new_tokens",
                body.get("max_tokens", sampling.get("max_new_tokens", 128)),
            )
        )
        if max_new < 1:
            raise ValueError("max_new_tokens must be positive")
        sampling.update(max_new_tokens=max_new)
        sampling.setdefault("temperature", 0.0)
        sampling.setdefault("ignore_eos", True)
        with self.cv:
            if rid in self.journal["entries"]:
                raise ValueError("rid already exists; reconnect using /v1/attach")
            if external_pd is None:
                admission = self._admission(role)
            else:
                admission = self._external_admission(role, external_pd)
            room = admission.pop("bootstrap_room", None)
            if room is None:
                room = (uuid.uuid4().int & ((1 << 63) - 1)) or 1
            uid = uuid.uuid4().hex
            entry = dict(
                rid=rid,
                source_rid="p4s-" + uid,
                target_rid="p4d-" + uid,
                submission_epoch="pd-" + uid,
                admission_epoch="move-" + uid,
                epoch=None,
                phase="SUBMIT",
                input_ids=prompt,
                sampling=sampling,
                max_new=max_new,
                cut=int(body.get("cut", max_new)),
                growth_tokens=int(body.get("growth_tokens", 128)),
                bootstrap_room=room,
                **admission,
                attempts=0,
                cancel_requested=False,
                cut_attempted=False,
                migration_requested=False,
                migration_deferred=False,
                migration_deferred_reason=None,
                migration_deferred_count=0,
                migration_deferred_since=None,
                migration_deferred_ms=0,
                migration_outcome=None,
                freeze_pending=False,
                source_partial_remote=False,
                created_at=time.time(),
                error=None,
                blocked=False,
            )
            if entry["growth_tokens"] < 0 or entry["cut"] < 0:
                raise ValueError("cut/growth_tokens must be nonnegative")
            self.journal["entries"][rid] = self.payloads.pack(entry)
            try:
                self._persist()
            except BaseException:
                self.journal["entries"].pop(rid)
                raise
            self.cv.notify_all()
        return rid

    def cancel(self, rid):
        self._update(rid, cancel_requested=True)
        return dict(
            ok=True,
            rid=rid,
            cancellation_pending=self._entry(rid)["phase"] not in TERMINAL,
        )

    def _migration_target(self, entry):
        if "dst" not in self.facts or "dst" in self.fact_errors:
            raise RuntimeError("target unavailable before reservation")
        return "dst"

    def _begin(self, entry):
        target = self._migration_target(entry)
        if target is None:
            return
        instances = dict(
            entry["instances"], dst=[r["instance_id"] for r in self.facts[target]]
        )
        if len(instances["src"]) != len(instances["dst"]):
            raise RuntimeError("source/target TP size mismatch")
        binding = (
            {"workers": {**entry["workers"], "dst": target}}
            if "workers" in entry
            else {}
        )
        self._update(
            entry["rid"],
            phase="PLAN",
            **binding,
            epoch=entry["admission_plans"][0]["epoch"]
            if entry.get("admission_plans")
            else "move-" + uuid.uuid4().hex,
            instances=instances,
            attempts=entry["attempts"] + 1,
            cut_attempted=True,
            error=None,
            blocked=False,
            freeze_pending=False,
            source_partial_remote=False,
            prepare_started_at=time.time(),
            freeze_requested_at=None,
            freeze_descriptor_ready_at=None,
            import_ready_at=None,
            target_admitted_at=None,
            dst_first_token_observed_at=None,
            dst_copy_observed_at=None,
            dst_local_observed_at=None,
            src_release_observed_at=None,
            remaining_copy_bytes=None,
            freeze_ack_latency_ms=None,
            plan_before_first_token=False,
            precopy_pending=None,
            precopy_capacity=None,
            precopy_backend=None,
            precopy_receipts=None,
            precopy_bytes=0,
            precopy_pages=None,
            migration_outcome=None,
            prepare_latency_ms=None,
            handoff_latency_ms=None,
            migration_skip_reason=None,
        )

    def _cut_due(self, entry, request):
        if not request["output_ids"]:
            return False
        if len(request["output_ids"]) >= entry["cut"]:
            return True
        if not entry["migration_requested"]:
            return False
        if not self.precopy_window_s or self.disable_precopy:
            return True
        if entry["phase"] in {"SUBMIT", "SOURCE", "PLAN", "RESERVE"}:
            return False
        if not all(
            r.get("precopy_protocol") == "closed-page-precopy-v1"
            for role in ("src", "dst")
            for r in self.facts.get(self._worker(entry, role), ())
        ):
            return True
        return entry.get("precopy_until", 0) <= time.time()

    def _async_handoff(self, entry=None):
        worker = "dst" if entry is None else self._worker(entry, "dst")
        workers = self.facts.get(worker, ())
        return self.async_handoff and bool(workers) and all(
            r.get("async_handoff_supported") for r in workers
        )

    def _control_waves(self):
        return not self.decode_workers and self._async_handoff() and all(
            rows and all(row.get("control_wave_supported") for row in rows)
            for rows in (self.facts.get("src"), self.facts.get("dst"))
        )

    @staticmethod
    def _command(role, name, epoch, **payload):
        return Command(role, name, epoch, payload)

    def _precopy_source(self, entry, cmd, **payload):
        rows = (yield self._command("src", cmd, entry["epoch"], **payload))
        if not any(r.get("deferred") for r in rows):
            return rows
        if not all(
            r.get("deferred") and r.get("migration_deferred_reason") == "request_finished"
            for r in rows
        ):
            raise RuntimeError("TP precopy source completion disagreed")
        self._update(
            entry["rid"],
            phase="ROLLBACK",
            migration_skip_reason="request_finished",
            migration_outcome="migration_cancelled_by_finish",
            error=None,
        )
        return None

    def _precopy(self, entry, *, cut_due=False):
        rid, epoch = entry["rid"], entry["epoch"]
        pending = entry.get("precopy_pending")
        if pending is None and self.disable_precopy:
            if cut_due:
                self._update(rid, phase="FREEZE")
            return
        started = time.monotonic()
        frame = pending["frame"] if pending else None
        key = (epoch, frame["chunk"]) if frame else None
        ready = self.precopy_ready.pop(key, None)
        trace = dict(
            type="precopy_stage",
            rid=rid,
            epoch=epoch,
            frame=dict(frame) if frame else None,
            available_pages=(pending or {}).get("available_pages"),
            available_pages_by_rank=(pending or {}).get("available_pages_by_rank"),
            reservation_upper_pages=(pending or {}).get("reservation_upper_pages"),
            stage=pending["stage"] if pending else "PREPARE",
            tick_wait_ms=(started - ready) * 1000 if ready is not None else None,
            timing_recovered=pending is not None and ready is None,
            rpc_ms=0.0,
            slot_wait_ms=0.0,
            journal_ms=0.0,
            dispatch_wait_ms=0.0,
            rpc_attempts=0,
            wave_envelope=False,
        )
        self.precopy_active[rid] = trace
        step = self._precopy_step(entry, cut_due=cut_due)
        yielded_ms = 0.0
        try:
            cmd = next(step)
            while True:
                sent = time.monotonic()
                measured = trace["rpc_ms"] + trace["slot_wait_ms"]
                try:
                    try:
                        result = yield cmd
                    finally:
                        elapsed = (time.monotonic() - sent) * 1000
                        yielded_ms += elapsed
                        trace["dispatch_wait_ms"] += max(
                            0,
                            elapsed
                            - (trace["rpc_ms"] + trace["slot_wait_ms"] - measured),
                        )
                except Exception as exc:
                    cmd = step.throw(exc)
                else:
                    cmd = step.send(result)
        except StopIteration:
            trace["ok"] = True
        except BaseException:
            trace["ok"] = False
            raise
        finally:
            step.close()
            self.precopy_active.pop(rid, None)
            ended = time.monotonic()
            trace["active_ms"] = (ended - started) * 1000
            trace["processing_ms"] = max(
                0, trace["active_ms"] - yielded_ms - trace["journal_ms"]
            )
            current = self._entry(rid)
            trace["phase"] = current["phase"]
            frame = trace["frame"]
            if frame is not None:
                plans = entry["plans"]
                index = frame["entry"]
                trace["page_bytes_by_rank"] = [
                    p["schema"]["entries"][index][-1] for p in plans
                ]
                trace["payload_bytes_by_rank"] = [
                    frame["count"] * n for n in trace["page_bytes_by_rank"]
                ]
                trace["payload_bytes_tp_sum"] = sum(trace["payload_bytes_by_rank"])
                trace["page_unit"] = "native_pool_page"
                trace["applied_pages_before"] = (
                    entry.get("precopy_pages")
                    or [0] * len(plans[0]["schema"]["entries"])
                )[index]
                trace["applied_bytes_tp_sum_after"] = current.get("precopy_bytes", 0)
                key = (epoch, frame["chunk"])
                if current["phase"] == "WAIT_CUT" and current.get("precopy_pending"):
                    self.precopy_ready[key] = ended
                with self.lock:
                    self.precopy_events.append(trace)
                if self.out is not None:
                    try:
                        with self.out.open("a") as stream:
                            stream.write(json.dumps(trace) + "\n")
                    except OSError:
                        pass

    def _precopy_step(self, entry, *, cut_due=False):
        """Advance one bounded, durable frame; Decode runs between commands."""
        rid, epoch = entry["rid"], entry["epoch"]
        pending = entry.get("precopy_pending")
        capacity = entry.get("precopy_capacity")
        backend = entry.get("precopy_backend")
        if pending is None:
            if cut_due:
                self._update(rid, phase="FREEZE")
                return
            # Named drains bound new frames even before the first output token.
            deadline = entry.get("precopy_until", 0) if self.decode_workers else None
            if deadline is not None and time.time() >= deadline:
                return
            if not all(
                r.get("precopy_protocol") == "closed-page-precopy-v1"
                for role in ("src", "dst")
                for r in self.facts.get(self._worker(entry, role), ())
            ):
                return
            # A reservation owns one receiver for all frames. Pin its negotiated
            # size with the first durable PREPARE, including across retries.
            if capacity is None and entry.get("precopy_receipts"):
                capacity = 4 * 1024 * 1024
            schema = entry["plans"][0]["schema"]
            upper_pages = entry["plans"][0]["upper"] // schema["page_size"]
            if capacity is None:
                max_bytes = upper_pages * max(e[-1] for e in schema["entries"])
                cap = 16 * 1024 * 1024 if max_bytes > 4 * 1024 * 1024 else 4 * 1024 * 1024
                facts = [
                    self.facts.get(self._worker(entry, role), ())
                    for role in ("src", "dst")
                ]
                if all(facts) and all(
                    row.get("precopy_rdma") is True for rows in facts for row in rows
                ):
                    backend = "rdma"
                elif all(facts) and all(
                    row.get("precopy_tcp") is True for rows in facts for row in rows
                ):
                    backend = "tcp"
                elif all(facts) and all(
                    "precopy_tcp" in row for rows in facts for row in rows
                ):
                    return  # No common transport. Leave native cut handling intact.
                if backend == "rdma":
                    capacity = 64 * 1024 * 1024 if max_bytes > cap else cap
                else:
                    capacity = min(
                        [cap] + [
                            row.get("precopy_frame_bytes", 4 * 1024 * 1024)
                            for rows in facts for row in rows
                        ]
                    )
            rows = yield from self._precopy_source(entry, "d2d_precopy_status")
            if rows is None:
                return
            if deadline is not None and time.time() >= deadline:
                return
            available = [
                min(r["available"][i] for r in rows)
                for i in range(len(rows[0]["available"]))
            ]
            ends = entry.get("precopy_pages") or [0] * len(available)
            # Same frame on every shard; each has its own authenticated slot.
            for i, end in enumerate(available):
                end = min(end, upper_pages)
                count = min(
                    end - ends[i], capacity // schema["entries"][i][-1]
                )
                if count > 0:
                    pending = dict(
                        stage="PREPARE",
                        available_pages=available[i],
                        available_pages_by_rank=[r["available"][i] for r in rows],
                        reservation_upper_pages=entry["plans"][0]["upper"]
                        // schema["page_size"],
                        started_at=time.time(),
                        stage_at=time.time(),
                        frame=dict(
                            entry=i, start=ends[i], count=count, chunk=uuid.uuid4().hex
                        ),
                    )
                    self._update(
                        rid, precopy_pending=pending, precopy_capacity=capacity,
                        precopy_backend=backend,
                    )
                    break
            if pending is None:
                return
        frame = pending["frame"]
        if "stage_at" in pending:
            with self.lock:
                self._observe(
                    self.precopy_stage_samples,
                    pending["stage"],
                    max(0, (time.time() - pending["stage_at"]) * 1000),
                )
        if pending["stage"] == "PREPARE":
            rows = yield self._command(
                "dst", "d2d_precopy_prepare", epoch, frame=frame,
                capacity=capacity or 4 * 1024 * 1024,
                **({backend: True} if backend else {}),
            )
            self._update(
                rid,
                precopy_pending=dict(
                    pending,
                    stage="COPY",
                    stage_at=time.time(),
                    receives=[r["receive"] for r in rows],
                ),
            )
        elif pending["stage"] == "COPY":
            rows = yield from self._precopy_source(
                entry, "d2d_precopy_copy", frame=frame, receives=pending["receives"]
            )
            if rows is None:
                return
            trace = self.precopy_active.get(rid)
            if trace is not None:
                trace["copied_payload_bytes_by_rank"] = [
                    r.get("copied_payload_bytes") for r in rows
                ]
                trace["copy_replayed_by_rank"] = [r.get("replayed") for r in rows]
            self._update(
                rid,
                precopy_pending=dict(
                    pending,
                    stage="APPLY",
                    stage_at=time.time(),
                    proofs=[r["proof"] for r in rows],
                ),
            )
        else:
            rows = yield self._command(
                "dst", "d2d_precopy_apply", epoch, frame=frame, proofs=pending["proofs"]
            )
            if any(r["prefix_pages"] != rows[0]["prefix_pages"] for r in rows):
                raise RuntimeError("TP precopy installed coverage disagreed")
            receipts = copy.deepcopy(
                entry.get("precopy_receipts") or [[] for _ in rows]
            )
            for receipt, row in zip(receipts, rows):
                receipt.append(row["receipt"])
            self._update(
                rid,
                phase="FREEZE" if cut_due else "WAIT_CUT",
                precopy_pending=None,
                precopy_pages=rows[0]["prefix_pages"],
                precopy_receipts=receipts,
                precopy_bytes=entry.get("precopy_bytes", 0)
                + sum(r["receipt"]["bytes"] for r in rows),
            )
            if pending.get("started_at") is not None:
                with self.lock:
                    self.precopy_frame_samples.append(
                        max(0, (time.time() - pending["started_at"]) * 1000)
                    )

    def _drain_abort_requested(self, rid):
        drain = self.journal["drain"]
        return (
            drain is not None
            and drain.get("abort_requested_at") is not None
            and rid in drain["members"]
        )

    def _advance(self, entry):
        rid, phase = entry["rid"], entry["phase"]
        if phase in PRECOMMIT and self._drain_abort_requested(rid):
            self._update(
                rid,
                phase="ROLLBACK",
                migration_requested=False,
                migration_outcome="drain_aborted",
            )
            return
        self._bound(
            entry,
            roles=(
                [entry["owner"]]
                if phase in {"SUBMIT", "SOURCE", "DIRECT", "LOCAL"}
                else None
            ),
        )
        request = self._request(entry, entry["owner"])
        if (
            phase in {"SOURCE", "PLAN", "RESERVE", "WAIT_CUT", "FREEZE"}
            and not entry.get("cut_observed_at")
            and request is not None
            and not entry["cancel_requested"]
            and (entry["cut"] < entry["max_new"] or entry["migration_requested"])
            and self._cut_due(entry, request)
        ):
            self._update(rid, cut_observed_at=time.time())
        if (
            phase in {"TARGET", "LOCAL"}
            and entry.get("target_admitted_at") is not None
            and entry.get("dst_first_token_observed_at") is None
            and request is not None
        ):
            frozen = entry["frozen_output"]
            ids = request["output_ids"]
            if len(ids) > len(frozen) and ids[: len(frozen)] == frozen:
                self._update(rid, dst_first_token_observed_at=time.time())
        if (
            request
            and request["error"]
            and not entry["cancel_requested"]
            and phase in {"SUBMIT", "SOURCE", "DIRECT"}
        ):
            raise RuntimeError(request["error"])
        if phase == "SUBMIT":
            role = entry["initial_owner"]
            backend_rid = entry["source_rid"] if role == "src" else entry["target_rid"]
            external = entry.get("external_prefill", False)
            if external and entry["cancel_requested"]:
                if entry.get("submit_started"):
                    rows = self._call_for(
                        entry, role, "d2d_cancel", entry["submission_epoch"],
                        rid=backend_rid, cancel_pending_submit=True,
                    )
                    if not all(row.get("kv_released") for row in rows):
                        return
                self._update(
                    rid, phase="CANCELLED", final_output_ids=[], final_finish_reason="abort"
                )
                return
            if request is None:
                prepare = None
                if (
                    role == "src"
                    and entry["cut"] < entry["max_new"]
                    and all(
                        r.get("admission_plan_supported") for r in self.facts[self._worker(entry, "src")]
                    )
                ):
                    prepare = dict(
                        migration_epoch=entry.get("admission_epoch")
                        or ("move-" + entry["submission_epoch"]),
                        target_rid=entry["target_rid"],
                        destination="D1",
                        growth_tokens=max(entry["growth_tokens"], entry["max_new"]),
                    )
                if external and not entry.get("submit_started"):
                    self._update(rid, submit_started=True)
                rows = self._call_for(
                    entry, role,
                    "d2d_submit_pd",
                    entry["submission_epoch"],
                    rid=backend_rid,
                    input_ids=entry["input_ids"],
                    sampling=entry["sampling"],
                    bootstrap_room=entry["bootstrap_room"],
                    bootstrap_host=entry["bootstrap_host"],
                    bootstrap_port=entry["bootstrap_port"],
                    prefill_url=entry["prefill_url"],
                    prepare_migration=prepare,
                    **({"external_prefill": True} if external else {}),
                )
                if prepare is not None and all(r.get("plan") for r in rows):
                    self._update(
                        rid,
                        admission_plans=[r["plan"] for r in rows],
                        admission_plan_at=time.time(),
                        plan_before_first_token=all(
                            not p["plan"]["output_ids"]
                            for p in rows
                        ),
                    )
            self._update(
                rid,
                phase="SOURCE" if role == "src" else "DIRECT",
                submission_confirmed_at=time.time(),
                request_seen=request is not None,
            )
            return
        if phase in {"SOURCE", "DIRECT"}:
            if request is None and entry.get("external_prefill") and entry["cancel_requested"]:
                rows = self._call_for(
                    entry, entry["owner"], "d2d_cancel", entry["submission_epoch"],
                    rid=entry["source_rid"], cancel_pending_submit=True,
                )
                if all(row.get("kv_released") for row in rows):
                    self._update(
                        rid, phase="CANCELLED", final_output_ids=[], final_finish_reason="abort"
                    )
                return
            if request is None:
                confirmed = entry.get("submission_confirmed_at")
                if (
                    not entry.get("request_seen", False)
                    and confirmed is not None
                    and time.time() - confirmed < STATUS_VISIBILITY_GRACE_S
                ):
                    # The submit reply can precede the next advisory snapshot.
                    # Wait without resubmitting or emitting a terminal error.
                    self._update(rid, awaiting_visibility=True, error=None)
                    return
                raise RuntimeError("native request missing; refusing resubmission")
            if not entry.get("request_seen", False) or entry.get("awaiting_visibility"):
                self._update(rid, request_seen=True, awaiting_visibility=False)
            if (
                phase == "SOURCE"
                and entry.get("admission_plans")
                and not entry["cut_attempted"]
                and not entry["cancel_requested"]
                and not self._drain_abort_requested(rid)
            ):
                self._begin(entry)
                return
            if entry["cancel_requested"] and not request["finished"]:
                role = entry["owner"]
                backend_rid = (
                    entry["source_rid"] if role == "src" else entry["target_rid"]
                )
                self._call_for(
                    entry, role, "d2d_cancel",
                    entry["submission_epoch"], rid=backend_rid
                )
            if request["finished"] and request["kv_released"]:
                self._update(
                    rid,
                    phase="CANCELLED" if entry["cancel_requested"] else "DONE",
                    final_output_ids=request["output_ids"],
                    final_finish_reason=request["finish_reason"],
                    migration_outcome=entry.get("migration_outcome")
                    or (
                        "completed_on_source_before_migration"
                        if phase == "SOURCE" and entry.get("migration_deferred")
                        else None
                    ),
                    **self._end_deferral(entry),
                )
            elif (
                phase == "SOURCE"
                and not entry["cancel_requested"]
                and not request["finished"]
            ):
                prepare = entry["migration_requested"] or (
                    not entry["cut_attempted"] and entry["cut"] < entry["max_new"]
                )
                if prepare and not self._drain_abort_requested(rid):
                    if not request["plan_ready"]:
                        self._defer_migration(
                            entry,
                            request["plan_deferred_reason"] or "sampled_output_pending",
                        )
                    elif time.time() >= entry.get("retry_at", 0):
                        self._begin(entry)
            return
        if entry["cancel_requested"] and phase in PRECOMMIT:
            self._update(rid, phase="ROLLBACK")
            return
        epoch = entry["epoch"]
        destination = entry.get("workers", {}).get("dst", "D1")
        common = dict(
            destination=destination,
            target_rid=entry["target_rid"],
            rid=entry["source_rid"],
        )
        if phase == "PLAN":
            rows = (
                [dict(plan=p) for p in entry["admission_plans"]]
                if entry.get("admission_plans")
                else self._call_for(
                    entry, "src",
                    "d2d_plan",
                    epoch,
                    growth_tokens=max(entry["growth_tokens"], entry["max_new"]),
                    **common,
                )
            )
            if any(r.get("deferred") for r in rows):
                if not all(r.get("deferred") for r in rows):
                    raise RuntimeError(
                        "TP plan deferral disagreed; refusing reservation"
                    )
                reason = next(
                    r["migration_deferred_reason"]
                    for r in rows
                    if r.get("migration_deferred_reason")
                )
                self._defer_migration(
                    entry,
                    reason,
                    migration_skip_reason=reason
                    if reason == "request_finished"
                    else None,
                    migration_outcome="window_missed_request_finished"
                    if reason == "request_finished"
                    else None,
                    phase="SOURCE",
                    epoch=None,
                    cut_attempted=False,
                    migration_requested=not self._drain_abort_requested(rid),
                    attempts=entry["attempts"] - 1,
                    retry_at=time.time() + max(self.poll_s, 0.1),
                )
                return
            plans = [r["plan"] for r in rows]
            if len({(p["lower"], p["upper"]) for p in plans}) != 1:
                raise RuntimeError("TP plan length mismatch")
            self._update(
                rid,
                plans=plans,
                phase="RESERVE",
                plan_before_first_token=all(not p["output_ids"] for p in plans),
                **self._end_deferral(entry),
            )
        elif phase == "RESERVE":
            rows = (yield self._command("dst", "d2d_reserve", epoch, plans=entry["plans"]))
            if not all(r.get("reserved") for r in rows):
                raise RuntimeError("target did not reserve on every rank")
            self._update(
                rid,
                reservations=[r["reservation"] for r in rows],
                phase="WAIT_CUT",
                precopy_until=(
                    time.time() + self.precopy_window_s if self.precopy_window_s else 0
                ),
                prepare_latency_ms=max(
                    0,
                    (time.time() - entry.get("prepare_started_at", time.time())) * 1000,
                ),
            )
        elif phase == "WAIT_CUT":
            if request is None:
                raise RuntimeError("prepared source request missing")
            cut_due = self._cut_due(entry, request)
            if request["finished"]:
                if entry.get("precopy_pending") is not None:
                    outcome = "migration_cancelled_by_finish"
                elif cut_due:
                    outcome = "window_missed_request_finished"
                else:
                    outcome = "completed_on_source_before_cut"
                self._update(
                    rid,
                    phase="ROLLBACK",
                    migration_skip_reason="request_finished",
                    migration_outcome=outcome,
                    error=None,
                )
            else:
                # At the cut, finish only an already durable frame. Its APPLY
                # receipt must be sealed by FREEZE, including after a restart.
                yield from self._precopy(entry, cut_due=cut_due)
        elif phase == "FREEZE":
            if entry.get("freeze_requested_at") is None:
                self._update(rid, freeze_requested_at=time.time())
            rows = (yield self._command(
                "src",
                "d2d_freeze",
                epoch,
                reservations=entry["reservations"],
                precopy_receipts=entry.get("precopy_receipts")
                or [[] for _ in entry["reservations"]],
                **common,
            ))
            if any(r.get("deferred") for r in rows):
                if not all(r.get("deferred") for r in rows):
                    raise RuntimeError("TP freeze deferral disagreed")
                reason = next(
                    (
                        r.get("migration_deferred_reason")
                        for r in rows
                        if r.get("migration_deferred_reason")
                    ),
                    "peer_not_ready",
                )
                if reason == "request_finished":
                    self._update(
                        rid,
                        phase="ROLLBACK",
                        migration_skip_reason=reason,
                        migration_outcome="window_missed_request_finished",
                    )
                elif not all(r.get("retryable", True) for r in rows):
                    self._update(
                        rid,
                        phase="ROLLBACK",
                        migration_skip_reason=reason,
                        migration_outcome="semantic_rejection",
                    )
                else:
                    self._defer_migration(entry, reason)
                return
            if entry.get("freeze_ack_latency_ms") is None:
                self._update(
                    rid,
                    freeze_ack_latency_ms=max(
                        0,
                        (time.time() - self._entry(rid)["freeze_requested_at"]) * 1000,
                    ),
                )
            if any(r.get("pending_retirement") for r in rows):
                if not all(
                    r.get("frozen") and r.get("pending_retirement") for r in rows
                ):
                    raise RuntimeError("TP frozen PD retirement disagreed")
                self._update(
                    rid,
                    freeze_pending=True,
                    source_partial_remote=any(
                        r.get("source_partial_remote", False) for r in rows
                    ),
                )
                return
            transfers = [r["transfer"] for r in rows]
            if any(
                t["output_ids"] != transfers[0]["output_ids"]
                or t["kv_len"] != transfers[0]["kv_len"]
                for t in transfers
            ):
                raise RuntimeError("TP freeze token history mismatch")
            self._update(
                rid,
                transfers=transfers,
                freeze_descriptor_ready_at=time.time(),
                remaining_copy_bytes=remaining_copy_bytes(transfers),
                frozen_output=transfers[0]["output_ids"],
                phase="IMPORT",
                freeze_pending=False,
                source_partial_remote=any(
                    t.get("source_partial_remote", False) for t in transfers
                ),
                **self._end_deferral(entry),
            )
        elif phase == "IMPORT":
            async_handoff = self._async_handoff(entry)
            targets = [r.get("target") for r in self._states(entry, "dst")]
            if async_handoff and any(t and t.get("import_pending") for t in targets):
                return
            payload = (
                {}
                if async_handoff and targets and all(targets)
                else {"transfers": entry["transfers"]}
            )
            rows = (yield self._command(
                "dst",
                "d2d_import",
                epoch,
                destination=destination,
                require_reservation=True,
                async_handoff=async_handoff,
                defer_bulk=self._control_waves(),
                **payload,
            ))
            if any(r.get("import_pending") for r in rows):
                if not all(r.get("import_pending") for r in rows):
                    raise RuntimeError("TP import readiness disagreed")
                return
            if not all(r.get("imported") for r in rows):
                raise RuntimeError("D2D target did not confirm import")
            # Serialize irreversible intent with the operator abort, which can
            # be accepted while the IMPORT RPC is outstanding.
            with self.cv:
                aborted = self._drain_abort_requested(rid)
                cancelled = self.journal["entries"][rid]["cancel_requested"]
                self._update(
                    rid,
                    phase="ROLLBACK" if aborted or cancelled else "COMMIT",
                    import_ready_at=time.time(),
                    **({"migration_outcome": "drain_aborted"} if aborted else {}),
                )
        elif phase == "COMMIT":
            # A timeout, partial TP commit or gateway crash can never send us
            # back to SOURCE. The per-rank backend commit is idempotent.
            if self.background and any(
                (r.get("target") or {}).get("commit_pending")
                for r in self._states(entry, "dst")
            ):
                return
            rows = (yield self._command("dst", "d2d_commit", epoch))
            if all(r.get("commit_pending") for r in rows):
                return
            if not all(r.get("admitted") and r.get("source_confirmed") for r in rows):
                raise RuntimeError("TP commit not yet confirmed; keeping commit intent")
            # The backend's collective commit certificate confirms ALL source
            # shards before ANY target shard admits. Status pollers may lag.
            if not self.background:
                self._refresh()
            self._update(
                rid,
                owner="dst",
                phase="TARGET",
                target_admitted_at=time.time(),
                error=None,
                blocked=False,
                migration_outcome="migrated",
                handoff_latency_ms=max(
                    0,
                    (time.time() - (entry.get("freeze_requested_at") or time.time()))
                    * 1000,
                ),
            )
        elif phase == "ROLLBACK":
            states = self._states(entry, "src")
            if any(
                s.get("source", {}).get("committed") for s in states if s.get("source")
            ):
                self._update(
                    rid,
                    phase="COMMIT",
                    error="observed committed source during rollback",
                )
                return
            closes = self._call_for(
                entry, "src",
                "d2d_precopy_close",
                epoch,
                pending_chunk=(entry.get("precopy_pending") or {})
                .get("frame", {})
                .get("chunk"),
            )
            rows = self._call_for(entry, "dst", "d2d_abort", epoch, precopy_close=closes)
            if not all(r.get("aborted") for r in rows):
                return
            rows = self._call_for(entry, "src", "d2d_resume", epoch)
            if all(r.get("resumed") for r in rows):
                drain = self.journal["drain"]
                retry = bool(
                    drain
                    and drain["active"]
                    and drain.get("abort_requested_at") is None
                    and rid in drain["members"]
                    and entry["attempts"] < 3
                )
                self._update(
                    rid,
                    phase="SOURCE",
                    migration_requested=retry,
                    admission_plans=None,
                    freeze_pending=False,
                    retry_at=time.time() + 1,
                    blocked=False,
                )
        elif phase == "LOCAL":
            if request is None:
                raise RuntimeError("target output history missing")
            if entry["cancel_requested"] and not request["finished"]:
                self._call_for(entry, "dst", "d2d_cancel", epoch, rid=entry["target_rid"])
            if request["finished"] and request["kv_released"]:
                self._update(
                    rid,
                    phase="CANCELLED" if entry["cancel_requested"] else "DONE",
                    final_output_ids=request["output_ids"],
                    final_finish_reason=request["finish_reason"],
                )
        elif phase == "TARGET":
            if request is None:
                raise RuntimeError("target output history missing")
            if entry["cancel_requested"] and not request["finished"]:
                self._call_for(entry, "dst", "d2d_cancel", epoch, rid=entry["target_rid"])
            sources = [s.get("source") for s in self._states(entry, "src")]
            targets = [s.get("target") for s in self._states(entry, "dst")]
            if any(
                (s and s.get("error")) or (t and t.get("error"))
                for s, t in zip(sources, targets)
            ):
                raise RuntimeError(
                    "D2D transport/retirement failed; retaining ownership"
                )
            local = bool(targets) and all(t and t["local"] for t in targets)
            released = bool(sources) and all(
                s and s["released"] and s["kv_released"] for s in sources
            )
            copied = bool(targets) and all(t and t.get("copied") for t in targets)
            observed_at = time.time()
            updates = {
                marker: observed_at
                for marker, seen in (
                    ("dst_copy_observed_at", copied),
                    ("dst_local_observed_at", local),
                    ("src_release_observed_at", released),
                )
                if seen and entry.get(marker) is None
            }
            if local and released:
                if request["finished"] and request["kv_released"]:
                    updates.update(
                        phase="CANCELLED" if entry["cancel_requested"] else "DONE",
                        final_output_ids=request["output_ids"],
                        final_finish_reason=request["finish_reason"],
                    )
                else:
                    updates["phase"] = "LOCAL"
            if updates:
                self._update(rid, **updates)

    def _drive(self, step, entry=None):
        try:
            cmd = next(step)
            while True:
                try:
                    rows = self._call_for(
                        entry, cmd.role, cmd.name, cmd.epoch, **cmd.payload
                    )
                except Exception as exc:
                    cmd = step.throw(exc)
                else:
                    cmd = step.send(rows)
        except StopIteration:
            pass

    def _advance_one(self, entry, started):
        self._drive(self._step(entry), entry)
        while self.background and time.monotonic() - started < 0.05:
            current = self._entry(entry["rid"])
            continue_frame = (
                entry["phase"] == current["phase"] == "WAIT_CUT"
                and current.get("precopy_pending") is not None
                and current["precopy_pending"] != entry.get("precopy_pending")
            )
            start_commit = (
                entry["phase"] == "IMPORT"
                and current["phase"] == "COMMIT"
                and self._async_handoff(current)
            )
            if not (continue_frame or start_commit):
                break
            entry = current
            self._drive(self._step(entry), entry)

    def _advance_error(self, rid, exc):
        current = self._entry(rid)
        # A timed-out mutation may have completed. Retry the same epoch.
        phase = current["phase"]
        rollback = phase in PRECOMMIT and not isinstance(
            exc, (TimeoutError, ConnectionError)
        )
        outcome = current.get("migration_outcome")
        if rollback and outcome is None:
            outcome = "precommit_failed"
        self._update(
            rid,
            phase="ROLLBACK" if rollback else phase,
            migration_outcome=outcome,
            error=repr(exc),
            blocked=not rollback,
        )

    def _step(self, entry):
        rid = entry["rid"]
        try:
            yield from self._advance(entry)
            current = self._entry(rid)
            if current["blocked"]:
                self._update(rid, blocked=False, error=None)
        except Exception as exc:
            self._advance_error(rid, exc)

    def _wave_priority(self, entry):
        phase = entry["phase"]
        if phase in {"COMMIT", "IMPORT", "FREEZE"}:
            return ("COMMIT", "IMPORT", "FREEZE").index(phase)
        if phase == "WAIT_CUT":
            # Advisory priority only. _advance validates the common history
            # and classifies divergence before issuing any mutation.
            rows = [
                row.get("requests", {}).get(entry["source_rid"])
                for row in self.facts.get("src", ())
            ]
            if rows and all(
                row and (row["finished"] or self._cut_due(entry, row))
                for row in rows
            ):
                return 2
            if entry.get("precopy_pending") is not None:
                return 3
        return 4

    def _wave_results(self, role, members):
        expected = members[0][2][role]
        if any(instances[role] != expected for _, _, instances in members):
            raise ConnectionError("D2D wave contains different worker instances")
        commands = [
            dict(cmd=cmd.name, migration_epoch=cmd.epoch, **cmd.payload)
            for _, cmd, _ in members
        ]
        with self.lock:
            sizes = self.control_wave_sizes.setdefault(role, {})
            size = str(len(commands))
            sizes[size] = sizes.get(size, 0) + 1
            for item in commands:
                key = role + "/" + item["cmd"]
                self.control_wave_members[key] = self.control_wave_members.get(key, 0) + 1
        return self._call(
            role, "d2d_control_wave", commands=commands, expected_instances=expected
        )

    def _run_wave(self, entries):
        steps = []
        copies = 0
        for entry in sorted(entries, key=self._wave_priority):
            # An already durable COPY can wait for the next round. Bound
            # synchronous CUDA/TE time so one slice cannot hold both slots.
            if (
                copies == 2
                and entry["phase"] == "WAIT_CUT"
                and (entry.get("precopy_pending") or {}).get("stage") == "COPY"
            ):
                continue
            step = self._step(entry)
            cmd = next(step, None)
            if cmd is not None:
                copies += int(cmd.name == "d2d_precopy_copy")
                steps.append((step, cmd, entry["instances"]))
        while steps:
            following = []
            # Keep destination admission ahead of speculative source copying.
            for role in ("dst", "src"):
                wave = [
                    (step, cmd, instances)
                    for step, cmd, instances in steps
                    if cmd.role == role
                ]
                if not wave:
                    continue
                try:
                    rows = self._wave_results(role, wave)
                    results = [
                        [row["results"][cmd.epoch] for row in rows]
                        for _, cmd, _ in wave
                    ]
                except Exception as exc:
                    # A lost envelope can follow successful member mutations.
                    # Reconcile each same durable phase/epoch, including precopy.
                    for step, _, _ in wave:
                        try:
                            step.throw(
                                ConnectionError(f"D2D control wave unacknowledged: {exc}")
                            )
                        except StopIteration:
                            pass
                    continue
                for (step, _, instances), replies in zip(wave, results):
                    try:
                        error = next((r["error"] for r in replies if "error" in r), None)
                        cmd = (
                            step.throw(RuntimeError(error)) if error else step.send(replies)
                        )
                        following.append((step, cmd, instances))
                    except StopIteration:
                        pass
            steps = following

    def tick(self):
        with self.work_lock:
            if not self.background:
                self._refresh()
            with self.lock:
                entries = [
                    copy.deepcopy(e)
                    for e in self.journal["entries"].values()
                    if e["phase"] not in TERMINAL
                ]
            offset = self.tick_cursor % len(entries) if entries else 0
            entries = entries[offset:] + entries[:offset]
            started, served = time.monotonic(), 0
            waves = self._control_waves()
            while served < len(entries):
                entry = self.payloads.view(entries[served])
                if waves:
                    slice_entries = [
                        self.payloads.view(raw)
                        for raw in entries[served : served + WAVE_SIZE]
                    ]
                    self._run_wave([e for e in slice_entries if e["phase"] in WAVE_PHASES])
                    for item in slice_entries:
                        if item["phase"] not in WAVE_PHASES:
                            self._drive(self._step(item), item)
                    served += len(slice_entries)
                else:
                    self._advance_one(entry, started)
                    served += 1
                if self.background and time.monotonic() - started >= 0.05:
                    break
            if entries:
                self.tick_cursor = (offset + served) % len(entries)
            with self.lock:
                self.tick_samples.append((time.monotonic() - started) * 1000)

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception:
                # Disk errors are fail-closed: no subsequent side effects from
                # the failed transition. A later tick may retry its durable intent.
                pass
            self.stop.wait(self.poll_s)

    def events(self, rid, after=-1):
        entry = self._entry(rid)
        with self.lock:
            if entry["phase"] in TERMINAL and "final_output_ids" in entry:
                req = dict(output_ids=entry["final_output_ids"], finished=True)
            else:
                self._bound(entry, roles=[entry["owner"]])
                req = self._request(entry, entry["owner"])
        ids = list(req["output_ids"]) if req else []
        frozen = entry.get("frozen_output", [])
        if entry["owner"] == "dst" and frozen:
            overlap = min(len(ids), len(frozen))
            if ids[:overlap] != frozen[:overlap]:
                raise RuntimeError("destination output does not continue the frozen prefix")
        if entry["owner"] == "src" and entry["phase"] not in {
            "SOURCE",
            "SUBMIT",
            "ROLLBACK",
        }:
            if (
                ids[: min(len(ids), len(frozen))]
                != frozen[: min(len(ids), len(frozen))]
            ):
                raise RuntimeError("frozen source output history changed")
            if len(frozen) > len(ids):
                ids = frozen
        if after < -1 or after >= entry["max_new"]:
            raise ValueError("SSE cursor outside request token budget")
        result = [
            dict(
                seq=i,
                token_id=int(ids[i]),
                owner=self._worker(entry, entry["owner"]),
                epoch=entry["epoch"] or entry["submission_epoch"],
                rid=rid,
            )
            for i in range(after + 1, len(ids))
        ]
        finished = bool(req and req["finished"]) or entry["phase"] in TERMINAL
        return result, finished

    def finish_reason(self, rid, output_count):
        with self.lock:
            entry = self._entry(rid)
            if entry["phase"] in TERMINAL:
                reason = entry.get("final_finish_reason")
            else:
                reason = (self._request(entry, entry["owner"]) or {}).get("finish_reason")
            if reason is not None:
                return reason
            return "length" if output_count == entry["max_new"] else "stop"

    def start_drain(self, mode="d2d"):
        if mode not in {"d2d", "natural"}:
            raise ValueError("drain mode must be d2d or natural")
        with self.cv:
            if self.journal["drain"] and self.journal["drain"]["active"]:
                return self.drain_status()
            members = [
                rid
                for rid, e in self.journal["entries"].items()
                if e["initial_owner"] == "src" and e["phase"] not in TERMINAL
            ]
            if mode == "natural" and any(
                self.journal["entries"][rid].get("epoch") for rid in members
            ):
                raise ValueError(
                    "natural comparison requires source requests without active migration"
                )
            previous = copy.deepcopy(self.journal)
            self.journal["drain"] = dict(
                id=uuid.uuid4().hex,
                active=True,
                members=members,
                started_at=time.time(),
                mode=mode,
            )
            for rid in members:
                self.journal["entries"][rid]["migration_requested"] = mode == "d2d"
                if mode == "natural":
                    self.journal["entries"][rid]["cut"] = self.journal["entries"][rid][
                        "max_new"
                    ]
            try:
                self._persist()
            except BaseException:
                self.journal = previous
                raise
            self.cv.notify_all()
            return self.drain_status()

    def _source_idle(self, source):
        return (
            source not in self.fact_errors
            and bool(self.facts.get(source))
            and all(row.get("idle", False) for row in self.facts[source])
        )

    def drain_status(self):
        with self.lock:
            drain = copy.deepcopy(self.journal["drain"])
            if not drain:
                return dict(state="READY", active=False)
            entries = [self.journal["entries"][rid] for rid in drain["members"]]
            source = drain.get("src", "src")
            abort_requested = drain.get("abort_requested_at") is not None
            aborting = abort_requested and any(
                e["phase"] in ACTIVE_MIGRATION_PHASES for e in entries
            )
            safe = not abort_requested and (
                all(e["phase"] in TERMINAL | {"LOCAL"} for e in entries)
                and self._source_idle(source)
            )
            if (
                not abort_requested
                and self.decode_workers
                and drain.get("source_released_at") is not None
            ):
                safe = True
            if (
                safe and drain.get("source_released_at") is None
                or abort_requested and not aborting and drain.get("aborted_at") is None
            ):
                previous = self.journal["drain"]
                drain["aborted_at" if abort_requested else "source_released_at"] = (
                    time.time()
                )
                if self.decode_workers:
                    drain["active"] = False
                self.journal["drain"] = drain
                try:
                    self._persist()
                except BaseException:
                    self.journal["drain"] = previous
                    raise
            blocked = any(
                e["blocked"] or e["phase"] == "SOURCE" and e["cut_attempted"]
                for e in entries
            )
            return dict(
                **drain,
                source_release_ms=(
                    max(0, (drain["source_released_at"] - drain["started_at"]) * 1000)
                    if drain.get("source_released_at") is not None
                    and drain.get("started_at") is not None
                    else None
                ),
                migration_deferred=sum(
                    bool(e.get("migration_deferred")) for e in entries
                ),
                migration_deferred_count=sum(
                    e.get("migration_deferred_count", 0) for e in entries
                ),
                freeze_pending=sum(bool(e.get("freeze_pending")) for e in entries),
                state="STOPPED"
                if drain.get("stopped")
                else "ABORTING"
                if aborting
                else "ABORTED"
                if abort_requested
                else "SAFE_TO_STOP"
                if safe
                else "BLOCKED"
                if blocked
                else "NATURAL_DRAINING"
                if time.time() < drain.get("natural_drain_until", 0)
                else "MIGRATING",
                members_state={
                    e["rid"]: dict(
                        phase=e["phase"],
                        owner=self._worker(e, e["owner"]),
                        error=e["error"],
                        migration_attempts=e["attempts"],
                        migration_deferred=e.get("migration_deferred", False),
                        migration_deferred_reason=e.get("migration_deferred_reason"),
                        migration_deferred_count=e.get("migration_deferred_count", 0),
                        migration_deferred_wait_ms=self._deferred_wait_ms(e),
                        migration_outcome=e.get("migration_outcome"),
                        freeze_pending=e.get("freeze_pending", False),
                        source_partial_remote=e.get("source_partial_remote", False),
                        precopy_bytes=e.get("precopy_bytes", 0),
                        remaining_copy_bytes=e.get("remaining_copy_bytes"),
                        handoff_timing=handoff_timing(e),
                    )
                    for e in entries
                },
            )

    @staticmethod
    def json_response(handler, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def stream(self, handler, rid, after=-1):
        # One active connection per logical request. A reconnect carries its
        # own acknowledged cursor; the journal is NOT a TCP delivery receipt.
        with self.lock:
            if rid in self.connections:
                return self.json_response(
                    handler, 409, dict(error="request already has an SSE reader")
                )
            if rid not in self.journal["entries"]:
                return self.json_response(handler, 404, dict(error="unknown rid"))
            self.connections.add(rid)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        last = time.monotonic()
        last_status = None
        try:
            while not self.stop.is_set():
                try:
                    events, finished = self.events(rid, after)
                except (ConnectionError, RuntimeError):
                    # Backend unreachability/identity mismatch never permits
                    # changing the output owner. Keep the client connection alive.
                    events, finished = [], False
                state = self._entry(rid)
                status_key = (
                    state["phase"],
                    state["blocked"],
                    state["error"],
                    state.get("migration_deferred"),
                    state.get("migration_deferred_reason"),
                    state.get("migration_deferred_count"),
                    state.get("migration_outcome"),
                    state.get("freeze_pending"),
                    state.get("migration_skip_reason"),
                )
                if status_key != last_status and (
                    state["blocked"]
                    or state.get("migration_deferred")
                    or state.get("freeze_pending")
                    or last_status is not None
                    or state.get("migration_outcome")
                    == "completed_on_source_before_migration"
                ):
                    status = json.dumps(
                        dict(
                            rid=rid,
                            phase=state["phase"],
                            blocked=state["blocked"],
                            error=state["error"],
                            **self.migration_fields(state),
                        )
                    )
                    handler.wfile.write(
                        ("event: status\ndata: " + status + "\n\n").encode()
                    )
                    handler.wfile.flush()
                    last_status = status_key
                for event in events:
                    data = json.dumps(event, ensure_ascii=False)
                    handler.wfile.write(
                        f"id: {event['seq']}\ndata: {data}\n\n".encode()
                    )
                    handler.wfile.flush()
                    after = event["seq"]
                if finished:
                    handler.wfile.write(b"data: [DONE]\n\n")
                    handler.wfile.flush()
                    break
                if time.monotonic() - last > 0.5:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    last = time.monotonic()
                self.stop.wait(min(self.poll_s, 0.05))
        except (BrokenPipeError, ConnectionResetError):
            pass  # Disconnect keeps the request alive for explicit /attach.
        except ValueError as exc:
            handler.wfile.write(
                ("data: " + json.dumps(dict(error=str(exc))) + "\n\n").encode()
            )
            handler.wfile.flush()
        finally:
            with self.lock:
                self.connections.discard(rid)

    def http_get(self, handler, url):
        query = parse_qs(url.query)
        if url.path == "/v1/attach":
            cursor = (
                query.get("after") or [handler.headers.get("Last-Event-ID", "-1")]
            )[0]
            try:
                cursor = int(cursor)
                if cursor < -1:
                    raise ValueError("negative SSE cursor")
            except ValueError:
                return self.json_response(
                    handler, 400, dict(error="invalid SSE cursor")
                )
            return self.stream(handler, (query.get("rid") or [""])[0], cursor)
        if url.path == "/v1/drain/status":
            return self.json_response(
                handler, 200, dict(protocol=PROTOCOL, **self.drain_status())
            )
        if url.path == "/v1/router/status":
            with self.lock:
                migrations = {
                    rid: dict(phase=entry["phase"], **self.migration_fields(entry))
                    for rid in self.journal["entries"]
                    for entry in (self._entry(rid),)
                }
            return self.json_response(
                handler,
                200,
                dict(
                    protocol=PROTOCOL,
                    drain=self.drain_status(),
                    worker_errors=dict(self.fact_errors),
                    migrations=migrations,
                    **self.latency_metrics(),
                ),
            )
        return self.json_response(handler, 404, dict(error="unknown DSV route"))

    def http_post(self, handler, path, body):
        try:
            if path in {"/v1/completions", "/generate"}:
                rid = self.accept(body)
                return self.stream(handler, rid)
            if path in {"/v1/cancel", "/v1/release"}:
                return self.json_response(handler, 200, self.cancel(body["rid"]))
            if path == "/v1/drain":
                if body.get("mode", "d2d") not in {"d2d", "natural"} or body.get(
                    "targets", ["dst"]
                ) != ["dst"]:
                    raise ValueError("DSV P4 supports D2D drain to dst")
                return self.json_response(
                    handler, 200, self.start_drain(body.get("mode", "d2d"))
                )
            if path == "/v1/drain/stop":
                with self.lock:
                    state = self.drain_status()["state"]
                    if state == "STOPPED":
                        return self.json_response(handler, 200, self.drain_status())
                    if state != "SAFE_TO_STOP":
                        return self.json_response(
                            handler, 409, dict(error="source not safe")
                        )
                    previous = copy.deepcopy(self.journal)
                    self.journal["drain"].update(stopped=True, stopped_at=time.time())
                    try:
                        self._persist()
                    except BaseException:
                        self.journal = previous
                        raise
                    # Keep source admission fenced; status polling no longer
                    # waits on a worker the operator may now stop externally.
                    return self.json_response(handler, 200, self.drain_status())
            if path == "/v1/background":
                return self.json_response(
                    handler, 200, dict(rid=self.accept(body, role="dst"))
                )
            return self.json_response(handler, 404, dict(error="unknown DSV route"))
        except (ValueError, KeyError) as exc:
            return self.json_response(handler, 400, dict(error=str(exc)))
        except RuntimeError as exc:
            return self.json_response(handler, 503, dict(error=str(exc)))
