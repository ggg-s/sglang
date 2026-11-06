"""DSV mode of the experimental P4 gateway: native PD, D2D, SSE and recovery.

One durable coordinator owns migration intents. A COMMIT intent is irreversible
from its perspective, including when an RPC reply was lost. Client SSE uses
sequence ids; a reconnect supplies Last-Event-ID/after to acknowledge receipt.
"""

from __future__ import annotations

import copy
import fcntl
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
PRECOMMIT = {"PLAN", "RESERVE", "FREEZE", "IMPORT"}


class DsvGateway:
    protocol = PROTOCOL

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
    ):
        self.driver, self.tokenizer = driver, tokenizer
        self.prefill_url, self.bootstrap_host = prefill_url, bootstrap_host
        self.bootstrap_port = int(bootstrap_port)
        self.poll_s = max(float(poll_ms) / 1000, 0.003)
        self.out = Path(out) if out else None
        self.path = Path(state_dir) / "dsv_gateway_journal.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.work_lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.stop = threading.Event()
        self.facts, self.fact_errors = {}, {}
        self.connections = set()
        self.role_locks = {role: threading.Lock() for role in ("src", "dst")}
        self.background = autostart
        self.status_threads = []
        self.fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.journal = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else dict(
                    version=1,
                    protocol=PROTOCOL,
                    entries={},
                    drain=None,
                )
            )
            if (
                self.journal.get("version") != 1
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
                    or not {
                        "owner",
                        "initial_owner",
                        "instances",
                        "source_rid",
                        "target_rid",
                        "submission_epoch",
                        "input_ids",
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
        except BaseException:
            os.close(self.fd)
            raise
        self.thread = None
        if autostart:
            for role in ("src", "dst"):
                poller = threading.Thread(
                    target=self._poll_role,
                    args=(role,),
                    daemon=True,
                    name="p4-dsv-status-" + role,
                )
                self.status_threads.append(poller)
                poller.start()
            self.thread = threading.Thread(
                target=self._loop, daemon=True, name="p4-dsv-reconcile"
            )
            self.thread.start()

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

    def _persist(self):
        # A failed write must stop the transition before its side effect. The
        # caller updates in-memory state only after the durable file is replaced.
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(self.journal, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _update(self, rid, **changes):
        with self.cv:
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
                and previous["entries"][rid]["phase"]
                != self.journal["entries"][rid]["phase"]
            ):
                try:
                    self.out.parent.mkdir(parents=True, exist_ok=True)
                    state = self.journal["entries"][rid]
                    with self.out.open("a") as metrics:
                        metrics.write(
                            json.dumps(
                                {
                                    key: state.get(key)
                                    for key in (
                                        "rid",
                                        "epoch",
                                        "phase",
                                        "owner",
                                        "updated_at",
                                        "error",
                                    )
                                }
                            )
                            + "\n"
                        )
                except OSError:
                    pass  # The durable ownership journal remains authoritative.

    def entry(self, rid):
        with self.lock:
            return copy.deepcopy(self.journal["entries"][rid])

    @staticmethod
    def rows(response):
        rows = response.get("tp_rank_results") or [response]
        if not rows or any(r.get("ok") is False or r.get("errors") for r in rows):
            raise RuntimeError("DSV command failed on a TP rank")
        return rows

    def _call(self, role, cmd, epoch="worker", **payload):
        if cmd != "d2d_worker_status":
            payload["expected_instances"] = [r["instance_id"] for r in self.facts[role]]
        if not self.role_locks[role].acquire(timeout=1):
            raise TimeoutError(f"{role} control busy; command not submitted")
        try:
            return self.rows(
                self.driver.send(
                    role,
                    dict(cmd=cmd, migration_epoch=epoch, **payload),
                    timeout=2 if cmd == "d2d_worker_status" else 30,
                )
            )
        finally:
            self.role_locks[role].release()

    def _refresh_role(self, role):
        with self.lock:
            if role == "src" and (self.journal["drain"] or {}).get("stopped"):
                return
        try:
            rows = self._call(role, "d2d_worker_status")
            if any(
                r.get("protocol") != PROTOCOL
                or r.get("rank") != rank
                or r.get("tp_size") != len(rows)
                or not r.get("instance_id")
                for rank, r in enumerate(rows)
            ):
                raise RuntimeError("DSV worker protocol/TP identity mismatch")
            with self.cv:
                self.facts[role] = rows
                self.fact_errors.pop(role, None)
                self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                self.fact_errors[role] = repr(exc)
                self.cv.notify_all()

    def _poll_role(self, role):
        while not self.stop.is_set():
            self._refresh_role(role)
            self.stop.wait(self.poll_s)

    def _refresh(self):
        for role in ("src", "dst"):
            self._refresh_role(role)

    def _bound(self, entry, roles=None):
        for role in roles or entry["instances"]:
            if role in self.fact_errors or role not in self.facts:
                raise RuntimeError(f"{role} unavailable; ownership unchanged")
            actual = [r["instance_id"] for r in self.facts[role]]
            if actual != entry["instances"][role]:
                raise RuntimeError(
                    f"{role} worker restarted; refusing stale epoch recovery"
                )

    def _request(self, entry, role):
        rid = entry["source_rid"] if role == "src" else entry["target_rid"]
        rows = [r.get("requests", {}).get(rid) for r in self.facts.get(role, [])]
        if not rows or any(row is None for row in rows):
            return None
        outputs = [r["output_ids"] for r in rows]
        # Polls may land at adjacent scheduler boundaries on TP shards. Only
        # publish the common prefix; never mistake lag for different ownership.
        size = min(map(len, outputs))
        prefix = outputs[0][:size]
        if any(o[:size] != prefix for o in outputs):
            raise RuntimeError("TP output token history diverged")
        return dict(
            output_ids=prefix,
            finished=all(r["finished"] for r in rows),
            kv_released=all(r["kv_released"] for r in rows),
            migration_ready=all(r.get("migration_ready", False) for r in rows),
            error=next((r.get("error") for r in rows if r.get("error")), None),
        )

    def _states(self, entry, role):
        return [
            r.get("migrations", {}).get(entry["epoch"], {}) for r in self.facts[role]
        ]

    def accept(self, body, role=None):
        prompt = body.get("prompt_ids", body.get("input_ids", body.get("prompt")))
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(type(x) is not int or x < 0 for x in prompt)
        ):
            raise ValueError("DSV P4 requires nonempty prompt token ids or prompt text")
        if not self.prefill_url:
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
            drain = self.journal["drain"]
            role = role or ("dst" if drain and drain["active"] else "src")
            if role not in self.facts or role in self.fact_errors:
                raise RuntimeError(f"{role} worker not ready")
            uid = uuid.uuid4().hex
            entry = dict(
                rid=rid,
                source_rid="p4s-" + uid,
                target_rid="p4d-" + uid,
                submission_epoch="pd-" + uid,
                epoch=None,
                phase="SUBMIT",
                owner=role,
                initial_owner=role,
                input_ids=prompt,
                sampling=sampling,
                max_new=max_new,
                cut=int(body.get("cut", max_new)),
                growth_tokens=int(body.get("growth_tokens", 128)),
                bootstrap_room=(uuid.uuid4().int & ((1 << 63) - 1)) or 1,
                bootstrap_host=self.bootstrap_host,
                bootstrap_port=self.bootstrap_port,
                prefill_url=self.prefill_url,
                instances={role: [r["instance_id"] for r in self.facts[role]]},
                attempts=0,
                cancel_requested=False,
                cut_attempted=False,
                migration_requested=False,
                created_at=time.time(),
                error=None,
                blocked=False,
            )
            if entry["growth_tokens"] < 0 or entry["cut"] < 0:
                raise ValueError("cut/growth_tokens must be nonnegative")
            self.journal["entries"][rid] = entry
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
            cancellation_pending=self.entry(rid)["phase"] not in TERMINAL,
        )

    def _begin(self, entry):
        if "dst" not in self.facts or "dst" in self.fact_errors:
            raise RuntimeError("target unavailable before reservation")
        instances = dict(
            entry["instances"], dst=[r["instance_id"] for r in self.facts["dst"]]
        )
        if len(instances["src"]) != len(instances["dst"]):
            raise RuntimeError("source/target TP size mismatch")
        self._update(
            entry["rid"],
            phase="PLAN",
            epoch="move-" + uuid.uuid4().hex,
            instances=instances,
            attempts=entry["attempts"] + 1,
            cut_attempted=True,
            error=None,
            blocked=False,
        )

    def _advance(self, entry):
        rid, phase = entry["rid"], entry["phase"]
        self._bound(
            entry,
            roles=[entry["owner"]]
            if phase in {"SUBMIT", "SOURCE", "DIRECT", "LOCAL"}
            else None,
        )
        request = self._request(entry, entry["owner"])
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
            if request is None:
                self._call(
                    role,
                    "d2d_submit_pd",
                    entry["submission_epoch"],
                    rid=backend_rid,
                    input_ids=entry["input_ids"],
                    sampling=entry["sampling"],
                    bootstrap_room=entry["bootstrap_room"],
                    bootstrap_host=entry["bootstrap_host"],
                    bootstrap_port=entry["bootstrap_port"],
                    prefill_url=entry["prefill_url"],
                )
            self._update(rid, phase="SOURCE" if role == "src" else "DIRECT")
            return
        if phase in {"SOURCE", "DIRECT"}:
            if request is None:
                raise RuntimeError("native request missing; refusing resubmission")
            if entry["cancel_requested"] and not request["finished"]:
                role = entry["owner"]
                backend_rid = (
                    entry["source_rid"] if role == "src" else entry["target_rid"]
                )
                self._call(
                    role, "d2d_cancel", entry["submission_epoch"], rid=backend_rid
                )
            if request["finished"] and request["kv_released"]:
                self._update(
                    rid,
                    phase="CANCELLED" if entry["cancel_requested"] else "DONE",
                    final_output_ids=request["output_ids"],
                )
            elif (
                phase == "SOURCE"
                and not entry["cancel_requested"]
                and not request["finished"]
            ):
                due = entry["migration_requested"] or (
                    not entry["cut_attempted"]
                    and len(request["output_ids"]) >= entry["cut"]
                )
                if (
                    due
                    and request["migration_ready"]
                    and request["output_ids"]
                    and time.time() >= entry.get("retry_at", 0)
                ):
                    self._begin(entry)
            return
        if entry["cancel_requested"] and phase in PRECOMMIT:
            self._update(rid, phase="ROLLBACK")
            return
        epoch = entry["epoch"]
        common = dict(
            destination="D1", target_rid=entry["target_rid"], rid=entry["source_rid"]
        )
        if phase == "PLAN":
            rows = self._call(
                "src", "d2d_plan", epoch, growth_tokens=entry["growth_tokens"], **common
            )
            plans = [r["plan"] for r in rows]
            if len({(p["lower"], p["upper"]) for p in plans}) != 1:
                raise RuntimeError("TP plan length mismatch")
            self._update(rid, plans=plans, phase="RESERVE")
        elif phase == "RESERVE":
            rows = self._call("dst", "d2d_reserve", epoch, plans=entry["plans"])
            if not all(r.get("reserved") for r in rows):
                raise RuntimeError("target did not reserve on every rank")
            self._update(
                rid, reservations=[r["reservation"] for r in rows], phase="FREEZE"
            )
        elif phase == "FREEZE":
            rows = self._call(
                "src", "d2d_freeze", epoch, reservations=entry["reservations"], **common
            )
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
                frozen_output=transfers[0]["output_ids"],
                phase="IMPORT",
            )
        elif phase == "IMPORT":
            self._call(
                "dst",
                "d2d_import",
                epoch,
                destination="D1",
                require_reservation=True,
                transfers=entry["transfers"],
            )
            # Persist irreversible intent BEFORE the first possible commit RPC.
            self._update(
                rid,
                phase="ROLLBACK" if self.entry(rid)["cancel_requested"] else "COMMIT",
            )
        elif phase == "COMMIT":
            # A timeout, partial TP commit or gateway crash can never send us
            # back to SOURCE. The per-rank backend commit is idempotent.
            rows = self._call("dst", "d2d_commit", epoch)
            if not all(r.get("admitted") and r.get("source_confirmed") for r in rows):
                raise RuntimeError("TP commit not yet confirmed; keeping commit intent")
            # The backend's collective commit certificate confirms ALL source
            # shards before ANY target shard admits. Status pollers may lag.
            if not self.background:
                self._refresh()
            self._update(rid, owner="dst", phase="TARGET", error=None, blocked=False)
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
            rows = self._call("dst", "d2d_abort", epoch)
            if not all(r.get("aborted") for r in rows):
                return
            rows = self._call("src", "d2d_resume", epoch)
            if all(r.get("resumed") for r in rows):
                drain = self.journal["drain"]
                retry = bool(
                    drain
                    and drain["active"]
                    and rid in drain["members"]
                    and entry["attempts"] < 3
                )
                self._update(
                    rid,
                    phase="SOURCE",
                    migration_requested=retry,
                    retry_at=time.time() + 1,
                    blocked=False,
                )
        elif phase == "LOCAL":
            if request is None:
                raise RuntimeError("target output history missing")
            if entry["cancel_requested"] and not request["finished"]:
                self._call("dst", "d2d_cancel", epoch, rid=entry["target_rid"])
            if request["finished"] and request["kv_released"]:
                self._update(
                    rid,
                    phase="CANCELLED" if entry["cancel_requested"] else "DONE",
                    final_output_ids=request["output_ids"],
                )
        elif phase == "TARGET":
            if request is None:
                raise RuntimeError("target output history missing")
            if entry["cancel_requested"] and not request["finished"]:
                self._call("dst", "d2d_cancel", epoch, rid=entry["target_rid"])
            sources = [s.get("source") for s in self._states(entry, "src")]
            targets = [s.get("target") for s in self._states(entry, "dst")]
            if any(
                (s and s.get("error")) or (t and t.get("error"))
                for s, t in zip(sources, targets)
            ):
                raise RuntimeError(
                    "D2D transport/retirement failed; retaining ownership"
                )
            local = all(t and t["local"] for t in targets)
            released = all(s and s["released"] and s["kv_released"] for s in sources)
            if local and released:
                if request["finished"] and request["kv_released"]:
                    self._update(
                        rid,
                        phase="CANCELLED" if entry["cancel_requested"] else "DONE",
                        final_output_ids=request["output_ids"],
                    )
                elif phase != "LOCAL":
                    self._update(rid, phase="LOCAL")

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
            for entry in entries:
                try:
                    self._advance(entry)
                    if self.entry(entry["rid"])["blocked"]:
                        self._update(entry["rid"], blocked=False, error=None)
                except Exception as exc:
                    current = self.entry(entry["rid"])
                    # RPC timeouts have unknown completion. Retrying the SAME
                    # epoch is safe; only explicit precommit errors roll back.
                    phase = current["phase"]
                    rollback = phase in PRECOMMIT and not isinstance(
                        exc, (TimeoutError, ConnectionError)
                    )
                    self._update(
                        entry["rid"],
                        phase="ROLLBACK" if rollback else phase,
                        error=repr(exc),
                        blocked=not rollback,
                    )

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
        entry = self.entry(rid)
        with self.lock:
            if entry["phase"] in TERMINAL and "final_output_ids" in entry:
                req = dict(output_ids=entry["final_output_ids"], finished=True)
            else:
                self._bound(entry, roles=[entry["owner"]])
                req = self._request(entry, entry["owner"])
        ids = list(req["output_ids"]) if req else []
        frozen = entry.get("frozen_output", [])
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
                owner=entry["owner"],
                epoch=entry["epoch"] or entry["submission_epoch"],
                rid=rid,
            )
            for i in range(after + 1, len(ids))
        ]
        finished = bool(req and req["finished"]) or entry["phase"] in TERMINAL
        return result, finished

    def start_drain(self):
        with self.cv:
            if self.journal["drain"] and self.journal["drain"]["active"]:
                return self.drain_status()
            members = [
                rid
                for rid, e in self.journal["entries"].items()
                if e["initial_owner"] == "src" and e["phase"] not in TERMINAL
            ]
            previous = copy.deepcopy(self.journal)
            self.journal["drain"] = dict(
                id=uuid.uuid4().hex, active=True, members=members
            )
            for rid in members:
                self.journal["entries"][rid]["migration_requested"] = True
            try:
                self._persist()
            except BaseException:
                self.journal = previous
                raise
            self.cv.notify_all()
            return self.drain_status()

    def drain_status(self):
        with self.lock:
            drain = copy.deepcopy(self.journal["drain"])
            if not drain:
                return dict(state="READY", active=False)
            entries = [self.journal["entries"][rid] for rid in drain["members"]]
            safe = all(e["phase"] in TERMINAL | {"LOCAL"} for e in entries)
            safe = (
                safe
                and "src" not in self.fact_errors
                and bool(self.facts.get("src"))
                and all(row.get("idle", False) for row in self.facts.get("src", []))
            )
            blocked = any(
                e["blocked"] or e["phase"] == "SOURCE" and e["cut_attempted"]
                for e in entries
            )
            return dict(
                **drain,
                state="STOPPED"
                if drain.get("stopped")
                else "SAFE_TO_STOP"
                if safe
                else "BLOCKED"
                if blocked
                else "MIGRATING",
                members_state={
                    e["rid"]: dict(phase=e["phase"], owner=e["owner"], error=e["error"])
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
        last_error = None
        try:
            while not self.stop.is_set():
                try:
                    events, finished = self.events(rid, after)
                except RuntimeError:
                    # Backend unreachability/identity mismatch never permits
                    # changing the output owner. Keep the client connection alive.
                    events, finished = [], False
                state = self.entry(rid)
                if state["blocked"] and state["error"] != last_error:
                    status = json.dumps(
                        dict(
                            rid=rid,
                            phase=state["phase"],
                            blocked=True,
                            error=state["error"],
                        )
                    )
                    handler.wfile.write(
                        ("event: status\ndata: " + status + "\n\n").encode()
                    )
                    handler.wfile.flush()
                    last_error = state["error"]
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
            return self.json_response(
                handler,
                200,
                dict(
                    protocol=PROTOCOL,
                    drain=self.drain_status(),
                    worker_errors=dict(self.fact_errors),
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
                if body.get("mode", "d2d") != "d2d" or body.get("targets", ["dst"]) != [
                    "dst"
                ]:
                    raise ValueError("DSV P4 supports D2D drain to dst")
                return self.json_response(handler, 200, self.start_drain())
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
