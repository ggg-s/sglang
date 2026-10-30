#!/usr/bin/env python3
"""P4.4 minimal streaming gateway.

The gateway owns the client SSE connection.  It submits the request to the
source decode instance, forwards committed tokens, and at cutover moves the
logical output owner to the target.  Tokens are identified by ``seq`` (index in
``output_ids``), so a target that starts from the source prefix can never
duplicate or reorder already-sent tokens.

This is deliberately a thin test gateway, not SGLang Model Gateway:
  * one request per HTTP connection;
  * status polling (3 ms) instead of a native token push API;
  * P4.4 v1 requires --disable-radix-cache.
"""
from __future__ import annotations

import argparse
import copy
import functools
import http.client
import json
import os
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/5090_handoff_m1")
import p2d_driver as d  # noqa: E402

d.CTL = Path(os.environ.get("HANDOFF_CTL_DIR", "/workspace/handoff_ctl_p4"))
STATE = Path("/workspace/handoff_state")
STATE.mkdir(parents=True, exist_ok=True)
MODEL = "/workspace/models/Qwen3-8B"
WAVE_DEFAULT = os.environ.get("HANDOFF_WAVE_DEFAULT", "1") == "1"
WAVE_JOIN_TIMEOUT_S = float(
    os.environ.get("HANDOFF_WAVE_JOIN_TIMEOUT_S", "30")
)
CANCEL_TTL_S = 120.0
KILL_AT = os.environ.get("HANDOFF_GW_KILL_AT", "").strip()
DRAIN_KILL_AT = os.environ.get("HANDOFF_GW_KILL_DRAIN_AT", "").strip()
DRAIN_NOOP = os.environ.get("HANDOFF_DRAIN_NOOP", "0") == "1"
DRAIN_IMPORT_FAIL = (
    os.environ.get("HANDOFF_DRAIN_IMPORT_FAIL", "0") == "1"
)
ADMISSION_POLICY = os.environ.get(
    "HANDOFF_ADMISSION_POLICY", "decoupled"
).strip().lower()
# Strict PD: when set, every NEW request submitted to a decode worker is
# preceded by a paired P-only prefill using the same bootstrap room.
PD_PREFILL_URL = os.environ.get("HANDOFF_PD_PREFILL_URL", "").strip()
PD_BOOTSTRAP_HOST = os.environ.get(
    "HANDOFF_PD_BOOTSTRAP_HOST", "127.0.0.1"
).strip()
PD_BOOTSTRAP_PORT = int(os.environ.get("HANDOFF_PD_BOOTSTRAP_PORT", "35409"))
_KILL_DONE = False
_DRAIN_KILL_DONE = False

print("[gw] P57_CODE_MARK 059f-run", flush=True)
TOK = AutoTokenizer.from_pretrained(MODEL)


def _pd_prefill_request(rid: str, prompt_ids: list[int], room: int) -> None:
    """Trigger P for one native PD decode request.

    The D worker has already registered its receiver with P's bootstrap
    server; P only needs to run the prompt prefill and transfer KV.  Use a
    distinct P-side rid so a retry cannot collide with the D request id.
    """
    if not PD_PREFILL_URL:
        return
    try:
        u = urlparse(PD_PREFILL_URL)
        host = u.hostname or "127.0.0.1"
        port = int(u.port or 80)
        path = u.path if u.path not in ("", "/") else "/v1/completions"
        body = {
            "rid": f"pd-p-{rid}",
            "prompt": prompt_ids,
            "max_new_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "ignore_eos": True,
            "stream": False,
            "bootstrap_host": PD_BOOTSTRAP_HOST,
            "bootstrap_port": PD_BOOTSTRAP_PORT,
            "bootstrap_room": room,
        }
        conn = http.client.HTTPConnection(host, port, timeout=180)
        conn.request(
            "POST", path, json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        if resp.status != 200:
            print(
                f"[gw] P prefill {rid} room={room} failed "
                f"{resp.status}: {raw[:300]!r}",
                flush=True,
            )
        else:
            print(
                f"[gw] P prefill {rid} room={room} ok", flush=True
            )
    except Exception as exc:
        print(
            f"[gw] P prefill {rid} room={room} error: {exc!r}",
            flush=True,
        )


class Role:
    """One control-plane role with a shared status poller and a command lock."""

    def __init__(self, role: str, poll_s: float = 0.003):
        self.role = role
        self.poll_s = poll_s
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.snapshot: dict[str, tuple[tuple[int, ...], bool, int]] = {}
        self.targets: dict[str, dict] = {}
        # Finished target states survive in the decode process after the
        # live target_states list drops them; keep them so the final summary
        # still carries the complete per-step remote-window series.
        self.completed_targets: dict[str, dict] = {}
        self.transfer_stats: dict[str, dict] = {}
        self.error = None
        self.error_since = None
        self.fail_streak = 0
        self._stop = False
        self.seq = 0
        self.thread = threading.Thread(target=self._poll, daemon=True, name=f"p4-{role}-poll")
        self.thread.start()

    def _poll(self):
        while not self._stop:
            try:
                # The file-based control protocol has a single command slot,
                # so polls and sends must stay serialized.  Keep the lock but
                # bound the call: a dead scheduler blocks senders for at most
                # the driver timeout instead of the old 300 s default.
                with self.lock:
                    st = d.status(self.role, timeout=4)
                    self.snapshot = {
                        r["rid"]: (
                            tuple(int(x) for x in r["output_ids"]),
                            bool(r["finished"]),
                            int(r["kv_committed_len"]),
                        )
                        for r in (st.get("reqs") or [])
                    }
                    self.targets = {
                        s.get("handoff_id"): s
                        for s in (st.get("target_states") or [])
                        if s.get("handoff_id")
                    }
                    self.completed_targets = {
                        s.get("handoff_id"): s
                        for s in (st.get("completed_target_states") or [])
                        if s.get("handoff_id")
                    }
                    self.transfer_stats = st.get("transfer_stats") or {}
                    self.error = None
                    self.error_since = None
                    self.fail_streak = 0
                    self.seq += 1
                    self.cv.notify_all()
            except Exception as exc:
                with self.lock:
                    # A single slow status probe is normal while the target
                    # scheduler is inside a large import or a P->D KV
                    # transfer.  Only a sustained outage may mark the target
                    # unreachable; otherwise a 1-2 s import stall produces a
                    # transient target_unreachable/BLOCKED drain.
                    self.fail_streak += 1
                    if self.fail_streak >= 2:
                        self.error = repr(exc)
                        if self.error_since is None:
                            self.error_since = time.monotonic()
                    self.cv.notify_all()
                time.sleep(0.2)
            time.sleep(self.poll_s)

    def send(self, payload: dict, timeout: float = 300.0):
        # A dead worker makes the status poller hold this lock for its whole
        # timeout.  Bound the wait so health/rollback decisions cannot stall
        # behind it.
        wait = min(max(1.0, float(timeout)), 30.0)
        if not self.lock.acquire(timeout=wait):
            raise TimeoutError(
                f"{self.role} control lock busy for {wait:.1f}s"
            )
        try:
            return d.send(self.role, payload, timeout=timeout)
        finally:
            self.lock.release()

    def wait_output(self, rid: str, after: int, timeout: float):
        """Wait until the request has more than ``after`` tokens or finishes."""
        deadline = time.monotonic() + timeout
        with self.cv:
            while True:
                item = self.snapshot.get(rid)
                if item is not None and (len(item[0]) > after or item[1]):
                    return item
                if self.error:
                    # A status control command can briefly exceed its 2 s
                    # poll budget while a 32K C4 batch is prefilling.  That is
                    # not evidence that the worker died; keep the client
                    # stream alive through a bounded outage instead of ending
                    # a migrated request with an SSE error.
                    grace = float(
                        os.environ.get("HANDOFF_ROLE_ERROR_GRACE_S", "30")
                    )
                    since = self.error_since or time.monotonic()
                    if time.monotonic() - since > grace:
                        raise RuntimeError(
                            f"{self.role} status: {self.error}"
                        )
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self.cv.wait(min(0.05, left))

    def get(self, rid: str):
        with self.lock:
            return self.snapshot.get(rid)


class RequestState:
    def __init__(self, rid: str, prompt_len: int, max_new: int, cut: int):
        self.rid = rid
        self.drid = f"dst-{rid}"
        self.prompt_len = prompt_len
        self.max_new = max_new
        self.cut = cut
        self.owner = "src"
        self.owner_epoch = "src"
        self.sent = 0
        self.last_send_t = None
        self.last_source_t = None
        self.first_target_t = None
        self.last_target_send_t = None
        self.max_gap_ms = 0.0
        self.max_backlog = 0
        self.target_itl = []
        self.cutover = {}
        self.reserve = None
        self.freeze_t = None
        self.freeze_send_t = None
        self.freeze_return_t = None
        self.release_t = None
        self.source_released_t = None
        self.transfer_token = None
        self.target_import_done_at = None
        self.target_last_remote_at = None
        self.target_created_at = None
        self.target_admitted_at = None
        self.target_first_step_at = None
        self.target_first_remote_at = None
        self.target_first_token_at = None
        self.target_step_starts = None
        # P5.14 remote-window attribution: per-step count of layers still
        # served remotely, plus the target-side RPC profile.
        self.target_step_remote_layers = None
        self.target_step_imported_layers = None
        self.target_mode = None
        self.target_rpc_n = None
        self.target_rpc_stage_ms = None
        self.target_admit_at = None
        self.target_import_timeline = None
        self.target_layer_ready_at = None
        self.target_layer_copy_ms = None
        self.target_layer_wait_ms = None
        self.target_staging_bytes = None
        self.source_transfer = None
        self.freeze_t = None
        self.error = None
        self.target_committed = False
        self.group = None
        self.switch_ready = False
        self.prefix_len = 0
        self.registered = False
        self.group_size = 1
        self.batch_failed = False
        self.wave = False
        self.wave_freeze = "each"
        self.drain_requested = False
        self.drain_failed = False
        self.transfer = None
        self.freeze_epoch = None
        self.freeze_result = None
        # P5.2-D per-phase control timeline: send/claim/exec/return timestamps.
        self.ctl = {}


def _serialize_migration(function):
    @functools.wraps(function)
    def call(self, *args, **kwargs):
        # Fence admission before waiting for this lock (start_drain). A legacy
        # cutover that wins the lock next must finish preflight before freeze.
        with self.migration_lock:
            if function.__name__ != "_prepare_drain":
                self._prepare_drain()
                drain, _ = self._drain_snapshot()
                if drain and drain.get("state") == "FAILED_UNSAFE":
                    raise RuntimeError("drain_failed_unsafe")
            return function(self, *args, **kwargs)
    return call


class Gateway:
    def __init__(self, out: Path, poll_ms: float = 3.0):
        self.src = Role("src", poll_ms / 1000.0)
        self.dst = Role("dst", poll_ms / 1000.0)
        self.out = out
        self.lock = threading.Lock()
        self.group_lock = threading.Lock()
        self.groups: dict[str, dict] = {}
        self._hb_stop = False
        self.journal_path = STATE / "controller_journal.json"
        self.journal_lock = threading.Lock()
        # Serializes "decide worker + journal + register active" against the
        # drain member snapshot, so a request cannot read READY and then be
        # submitted to a source that already committed DRAINING/SAFE.
        self.admission_lock = threading.Lock()
        self.migration_lock = threading.RLock()
        self.journal: dict[str, dict] = {}
        self.draining = False
        self.drain_started_t = None
        self.drain_targets: list[str] = []
        self.drain_kv_baseline = 0
        self.active: dict[str, RequestState] = {}
        try:
            if self.journal_path.exists():
                self.journal = json.loads(
                    self.journal_path.read_text(encoding="utf-8")
                )
        except Exception:
            self.journal = {}
        _drain = self.journal.get("__drain__")
        if isinstance(_drain, dict):
            self.draining = _drain.get("state") in self.DRAIN_LIVE + ("BLOCKED",)
            self.drain_targets = list(_drain.get("targets") or ["dst"])
            started = float(_drain.get("created_at", time.time()))
            self.drain_started_t = time.monotonic() - max(
                0.0, time.time() - started
            )
            if _drain.get("state") in (
                "REQUESTED", "DRAINING", "PREPARING", "MIGRATING",
                "RECONCILING",
            ):
                # Unified restart entry: rebuild from the journal and let the
                # reconcile loop replay the persisted wave intent.
                _drain["restart_at"] = time.time()
                _drain["restart_from"] = _drain.get("state")
                _drain["state"] = "RECONCILING"
                _drain["reconciled_at"] = None
                try:
                    tmp = self.journal_path.with_suffix(".tmp")
                    tmp.write_text(
                        json.dumps(self.journal, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    os.replace(tmp, self.journal_path)
                except OSError:
                    pass
        threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="gw-heartbeat"
        ).start()
        self._facts_at = 0.0
        self._src_facts: dict = {}
        self._dst_facts: dict = {}
        self._target_ok = True
        self._target_dbg_t = 0.0
        # Requests admitted by the Router while this controller was down are
        # recorded as sidecars; adopt them so a later drain sees them.
        self.proxy_dir = STATE / "proxy_requests"
        self._adopt_at = 0.0
        # D1 tokens admitted but not yet visible in target facts (the request
        # is in flight between admission and its first KV allocation).
        self._admit_pending = 0
        self._target_fail_since = None
        self._dst_fact_fails = 0
        threading.Thread(
            target=self._reconcile_loop, daemon=True, name="gw-reconcile"
        ).start()

    def _maybe_kill(self, phase: str):
        """B0 fault injection: SIGKILL this gateway at a named phase once."""
        global _KILL_DONE
        if not KILL_AT or _KILL_DONE or phase != KILL_AT:
            return
        _KILL_DONE = True
        if phase == "transfer":
            # Import has returned and the background TE is still running.
            def die():
                os.kill(os.getpid(), signal.SIGKILL)
            threading.Timer(0.3, die).start()
            print(f"[gw] kill scheduled at {phase}", flush=True)
            return
        print(f"[gw] killing self at {phase}", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)

    def _journal_update(self, st: RequestState, force: bool = False, **fields):
        """Persist enough state for a restarted controller to adopt the stream."""
        now = time.monotonic()
        if not force and now - getattr(st, "_journal_t", 0.0) < 0.05:
            return
        st._journal_t = now
        with self.journal_lock:
            rec = self.journal.setdefault(
                st.rid,
                {
                    "rid": st.rid,
                    "drid": st.drid,
                    "owner": "src",
                    "epoch": "src",
                    "phase": "accepted",
                    "sent": 0,
                    "prefix_len": 0,
                    "max_new": st.max_new,
                    "cut": st.cut,
                    "transfer": st.transfer or "remote",
                    "finished": False,
                },
            )
            rec.update(fields)
            try:
                tmp = self.journal_path.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(self.journal, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(tmp, self.journal_path)
            except OSError:
                pass

    def _journal_sent(self, st: RequestState):
        # Per-token writes only advance delivery progress.  owner/phase belong
        # to the reconcile loop and must never be clobbered by a stream thread.
        self._journal_update(st, sent=int(st.sent))

    def _maybe_drain_kill(self, phase: str):
        global _DRAIN_KILL_DONE
        if not DRAIN_KILL_AT or _DRAIN_KILL_DONE or phase != DRAIN_KILL_AT:
            return
        _DRAIN_KILL_DONE = True
        if phase == "transfer":
            def die():
                os.kill(os.getpid(), signal.SIGKILL)
            threading.Timer(0.3, die).start()
            print(f"[gw] drain kill scheduled at {phase}", flush=True)
            return
        print(f"[gw] drain killing self at {phase}", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)

    def _heartbeat_loop(self):
        """Durable liveness marker for the source/target frozen watchdogs."""
        dirs = [d.CTL / "src", d.CTL / "dst"]
        payload = {"pid": os.getpid()}
        while not self._hb_stop:
            payload["ts"] = time.time()
            body = json.dumps(payload)
            for root in dirs:
                try:
                    root.mkdir(parents=True, exist_ok=True)
                    tmp = root / "heartbeat.tmp"
                    tmp.write_text(body, encoding="utf-8")
                    os.replace(tmp, root / "heartbeat.json")
                except OSError:
                    pass
            time.sleep(0.1)

    def _event(self, st: RequestState, seq: int, token: int, owner: str):
        now = time.monotonic()
        if st.last_send_t is not None:
            st.max_gap_ms = max(st.max_gap_ms, (now - st.last_send_t) * 1000.0)
        st.last_send_t = now
        st.sent = max(st.sent, seq + 1)
        if owner == "src":
            st.last_source_t = now
        if owner == "dst":
            if st.first_target_t is None:
                st.first_target_t = now
            # Refresh every target event: the remote-window series grows one
            # entry per target decode step and the finished state can vanish
            # from the live list before the final summary is written.
            self._absorb_target(st, self._target_view(st.rid))
            if st.last_target_send_t is not None:
                st.target_itl.append((now - st.last_target_send_t) * 1000.0)
            st.last_target_send_t = now
        self._journal_sent(st)
        return {
            "seq": seq,
            "token_id": int(token),
            "owner": owner,
            "epoch": st.owner_epoch,
            "t": now,
        }

    def _target_view(self, rid: str):
        tt = self.dst.targets.get(rid)
        if tt is None:
            tt = self.dst.completed_targets.get(rid)
        return tt

    def _absorb_target(self, st: RequestState, tt):
        """Copy the complete target-side series, not just the first token.

        Remote Window attribution needs the final per-step layer mask and RPC
        counters.  Sampling only at first target token leaves one step in the
        series, so refresh on every target event and again when the request
        finishes (live state, then the retained completed state).
        """
        if not tt:
            return
        for src_key, dst_attr in (
            ("created_at", "target_created_at"),
            ("admitted_at", "target_admitted_at"),
            ("first_step_at", "target_first_step_at"),
            ("first_remote_at", "target_first_remote_at"),
            ("first_token_at", "target_first_token_at"),
            ("step_starts", "target_step_starts"),
            ("step_remote_layers", "target_step_remote_layers"),
            ("step_imported_layers", "target_step_imported_layers"),
            ("mode", "target_mode"),
            ("admit_at", "target_admit_at"),
            ("import_timeline", "target_import_timeline"),
            ("import_done_at", "target_import_done_at"),
            ("last_remote_at", "target_last_remote_at"),
            ("layer_ready_at", "target_layer_ready_at"),
            ("layer_copy_ms", "target_layer_copy_ms"),
            ("layer_wait_ms", "target_layer_wait_ms"),
            ("host_staging_bytes", "target_staging_bytes"),
            ("rpc_n", "target_rpc_n"),
            ("rpc_stage_ms", "target_rpc_stage_ms"),
        ):
            val = tt.get(src_key)
            if val is None:
                continue
            if isinstance(val, (list, dict)) and not val:
                continue
            if src_key in ("created_at", "admitted_at", "first_step_at",
                           "first_remote_at", "first_token_at",
                           "step_starts", "import_done_at"):
                # First-seen timestamps stay put; later series refresh.
                if getattr(st, dst_attr, None) is not None and src_key not in (
                    "step_starts",
                ):
                    continue
            setattr(st, dst_attr, val)

    @staticmethod
    def _remote_window(st: RequestState) -> dict:
        """Split target decode steps into Remote and LOCAL windows.

        ``step_remote_layers[i]`` is the number of layers not yet local at
        the start of step ``i``; the interval from ``step_starts[i]`` to
        ``step_starts[i+1]`` is therefore attributed to step ``i``'s mode.
        """
        starts = list(st.target_step_starts or [])
        counts = list(st.target_step_remote_layers or [])
        n = min(len(starts), len(counts))
        if n == 0:
            return {}
        first_local = next((i for i in range(n) if counts[i] == 0), None)
        remote_itl = [
            (starts[i + 1] - starts[i]) * 1000.0
            for i in range(n - 1)
            if counts[i] > 0
        ]
        local_from = first_local if first_local is not None else n
        local_itl = [
            (starts[i + 1] - starts[i]) * 1000.0
            for i in range(local_from, n - 1)
        ]

        def pct(vals, q):
            if not vals:
                return None
            vals = sorted(vals)
            k = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
            return round(vals[k], 3)

        remote_steps = first_local if first_local is not None else n
        local_steps = n - remote_steps
        return {
            "steps": n,
            "remote_steps": remote_steps,
            "local_steps": local_steps,
            "remote_window_ms": (
                round((starts[first_local] - starts[0]) * 1000.0, 3)
                if first_local is not None and first_local > 0
                else (round((starts[-1] - starts[0]) * 1000.0, 3)
                      if first_local is None and n > 1 else 0.0)
            ),
            "local_window_ms": (
                round((starts[-1] - starts[first_local]) * 1000.0, 3)
                if first_local is not None and n - 1 > first_local else 0.0
            ),
            "remote_itl_p50": pct(remote_itl, 0.5),
            "remote_itl_p90": pct(remote_itl, 0.9),
            "remote_itl_max": round(max(remote_itl), 3) if remote_itl else None,
            "local_itl_p50": pct(local_itl, 0.5),
            "local_itl_p90": pct(local_itl, 0.9),
            "local_itl_max": round(max(local_itl), 3) if local_itl else None,
            "remote_steps_total_ms": round(sum(remote_itl), 3),
            "local_steps_total_ms": round(sum(local_itl), 3),
            "max_remote_layers": max(counts[:n]) if counts else None,
        }

    def _metrics(self, st: RequestState) -> dict:
        tt = self._target_view(st.rid)
        if tt:
            self._absorb_target(st, tt)
            st.target_created_at = st.target_created_at or tt.get("created_at")
            st.target_admitted_at = st.target_admitted_at or tt.get("admitted_at")
            st.target_first_step_at = (
                st.target_first_step_at or tt.get("first_step_at")
            )
            st.target_first_remote_at = (
                st.target_first_remote_at or tt.get("first_remote_at")
            )
            st.target_first_token_at = (
                st.target_first_token_at or tt.get("first_token_at")
            )
            st.target_step_starts = (
                st.target_step_starts or tt.get("step_starts")
            )
            st.target_admit_at = st.target_admit_at or tt.get("admit_at")
            st.target_import_timeline = (
                st.target_import_timeline or tt.get("import_timeline")
            )
            st.target_import_done_at = (
                st.target_import_done_at or tt.get("import_done_at")
            )
        gap = None
        if st.last_source_t is not None and st.first_target_t is not None:
            gap = (st.first_target_t - st.last_source_t) * 1000.0
        if st.source_transfer is None and st.transfer_token:
            ts = self.src.transfer_stats.get(st.transfer_token)
            if ts is not None:
                st.source_transfer = {
                    k: ts.get(k)
                    for k in (
                        "start_at", "accept_at", "done_at", "send_ms",
                        "engine", "pipeline", "streams", "async",
                        "target_slots", "bytes", "layers",
                    )
                }
        def delta_ms(a, b):
            if a is None or b is None:
                return None
            return round((a - b) * 1000.0, 3)
        itl = st.target_itl
        itl_sorted = sorted(itl)
        def pct(q):
            if not itl_sorted:
                return None
            k = min(len(itl_sorted) - 1, max(0, int(round(q * (len(itl_sorted) - 1)))))
            return round(itl_sorted[k], 3)
        return {
            "rid": st.rid,
            "owner": st.owner,
            "sent": st.sent,
            "cut": st.cut,
            "source_last_to_target_first_ms": gap,
            "max_token_gap_ms": round(st.max_gap_ms, 3),
            "target_itl_p50": pct(0.5),
            "target_itl_p90": pct(0.9),
            "target_itl_max": round(max(itl), 3) if itl else None,
            "max_backlog": st.max_backlog,
            "source_freeze_to_release_ms": (
                None
                if st.freeze_t is None or st.source_released_t is None
                else round((st.source_released_t - st.freeze_t) * 1000.0, 3)
            ),
            "last_source_to_freeze_ms": (
                None
                if st.freeze_t is None or st.last_source_t is None
                else round((st.freeze_t - st.last_source_t) * 1000.0, 3)
            ),
            "freeze_to_first_target_ms": (
                None
                if st.freeze_t is None or st.first_target_t is None
                else round((st.first_target_t - st.freeze_t) * 1000.0, 3)
            ),
            # P5.2-C attribution (all timestamps are host CLOCK_MONOTONIC).
            "freeze_return_to_import_done_ms": delta_ms(
                st.target_import_done_at, st.freeze_return_t
            ),
            "freeze_return_to_last_remote_ms": delta_ms(
                st.target_last_remote_at, st.freeze_return_t
            ),
            "import_done_to_source_release_ms": delta_ms(
                st.source_released_t, st.target_import_done_at
            ),
            "last_remote_to_source_release_ms": delta_ms(
                st.source_released_t, st.target_last_remote_at
            ),
            "target_staging_bytes": st.target_staging_bytes,
            "target_layer_ready_at": st.target_layer_ready_at,
            "target_layer_copy_ms": st.target_layer_copy_ms,
            "target_layer_wait_ms": st.target_layer_wait_ms,
            "source_transfer": st.source_transfer,
            "timeline": {
                "last_source_at": st.last_source_t,
                "freeze_send_at": st.freeze_send_t,
                "freeze_return_at": st.freeze_t,
                "first_target_at": st.first_target_t,
                "target_created_at": st.target_created_at,
                "target_admit_at": st.target_admit_at,
                "target_first_step_at": st.target_first_step_at,
                "target_first_remote_at": st.target_first_remote_at,
                "target_first_token_at": st.target_first_token_at,
                "target_import_done_at": st.target_import_done_at,
                "source_released_at": st.source_released_t,
            },
            "freeze_profile": (
                (st.freeze_result or {}).get("freeze_profile")
                if isinstance(st.freeze_result, dict) else None
            ),
            "freeze_timeline": (
                ((st.freeze_result or {}).get("transfer") or {}).get(
                    "timeline"
                )
                if isinstance(st.freeze_result, dict) else None
            ),
            "target_import_timeline": st.target_import_timeline,
            "target_step_starts": st.target_step_starts,
            # P5.14: per-step remote layer mask and the derived split of
            # target decode into Remote Window vs LOCAL Window.
            "target_step_remote_layers": st.target_step_remote_layers,
            "target_step_imported_layers": st.target_step_imported_layers,
            "target_import_done_at": st.target_import_done_at,
            "target_last_remote_at": st.target_last_remote_at,
            "target_rpc_n": st.target_rpc_n,
            "target_rpc_stage_ms": st.target_rpc_stage_ms,
            "target_mode": st.target_mode,
            "remote_window": self._remote_window(st),
            "ctl": st.ctl,
            "wave": st.wave,
            "cutover": st.cutover,
            "error": st.error,
            "target_committed": st.target_committed,
        }

    def _send_timed(self, role, payload: dict, timeout: float, st: RequestState,
                    phase: str):
        """Send one control command and record its end-to-end timeline.

        ``send_to_claim`` is the scheduler wait (the command file is only
        polled once per iteration); ``exec_ms`` is the handler cost; the rest
        is socket/file return.  All timestamps are CLOCK_MONOTONIC on the same
        host, so the target/source numbers are directly comparable.
        """
        t0 = time.monotonic()
        result = role.send(payload, timeout=timeout)
        t1 = time.monotonic()
        c = result.get("_ctl") or {}
        claim = c.get("claim_mono")
        exec_start = c.get("exec_start_mono")
        exec_end = c.get("exec_end_mono")
        rec = {
            "phase": phase,
            "cmd": payload.get("cmd"),
            "send_t": t0,
            "recv_t": t1,
            "wall_ms": round((t1 - t0) * 1000.0, 3),
            "iter": c.get("iter"),
            "iter_period_ms": c.get("iter_period_ms"),
            "claim_mono": claim,
            "exec_start_mono": exec_start,
            "exec_end_mono": exec_end,
            "send_to_claim_ms": (
                None if claim is None else round((claim - t0) * 1000.0, 3)
            ),
            "claim_to_exec_ms": c.get("wait_before_exec_ms"),
            "exec_ms": c.get("exec_ms"),
            "recv_after_exec_ms": (
                None if exec_end is None
                else round((t1 - exec_end) * 1000.0, 3)
            ),
        }
        st.ctl.setdefault(phase, []).append(rec)
        return result

    def _copy_ctl(self, members, phase: str):
        """Share one wave command's timeline with every member."""
        if not members:
            return
        recs = members[0].ctl.get(phase) or []
        if not recs:
            return
        rec = recs[-1]
        for st in members[1:]:
            st.ctl.setdefault(phase, []).append(rec)

    def _cancel_epochs(self, members):
        """Tombstone a wave's (rid, epoch) before resuming its sources.

        The marker file is checked by the target after any command-side delay,
        so a control command that times out but still executes later cannot
        create a target request after the source resumed.
        """
        dst_dir = d.CTL / "dst"
        items = []
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        for st in members:
            if not st.freeze_epoch:
                continue
            marker = dst_dir / f"cancel.{st.drid}.{st.freeze_epoch}.json"
            try:
                tmp = marker.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps({"at": time.time()}), encoding="utf-8"
                )
                os.replace(tmp, marker)
            except OSError:
                pass
            items.append({"rid": st.drid, "epoch": st.freeze_epoch})
        if items:
            try:
                self.dst.send({"cmd": "cancel_import", "items": items},
                              timeout=30)
            except Exception:
                pass

    def _write_metrics(self, st: RequestState):
        rec = self._metrics(st)
        with self.lock:
            with self.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _send_event(self, handler, st: RequestState, payload):
        handler.wfile.write(
            ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()
        )
        handler.wfile.flush()

    def _keepalive(self, handler):
        handler.wfile.write(b": keepalive\n\n")
        handler.wfile.flush()

    def _drain_source(self, handler, st: RequestState, src_ids, sent: int):
        for seq in range(sent, len(src_ids)):
            self._send_event(handler, st, self._event(st, seq, src_ids[seq], "src"))
        return len(src_ids)

    def _cutover_epoch(self, members):
        with self.journal_lock:
            drain = self.journal.get("__drain__") or {}
            if drain.get("state") in self.DRAIN_LIVE:
                covered = [st.rid in (drain.get("members") or []) for st in members]
                if any(covered):
                    if not all(covered):
                        raise RuntimeError("cutover group crosses drain membership")
                    epoch = str(drain.get("reserved_epoch") or drain.get("epoch") or "")
                    if not epoch or epoch != str(drain.get("epoch") or ""):
                        raise RuntimeError("drain reservation epoch is inconsistent")
                    return epoch
        return uuid.uuid4().hex

    @_serialize_migration
    def _cutover(self, handler, st: RequestState, transfer: str, inject: str):
        print(f"[gw] enter _cutover rid={st.rid} wave={st.wave} "
              f"group={st.group}", flush=True)
        epoch = self._cutover_epoch([st])
        n = st.prompt_len + st.cut + 64
        if inject == "capacity":
            n += 10_000_000
        try:
            st_now = self.dst.send({"cmd": "status"}, timeout=4)
            tps = st_now.get("tp_rank_results") or [st_now]
            max_run = min(int(x.get("max_running") or 0) for x in tps)
            load = max(len(x.get("reqs") or []) for x in tps)
            if max_run and load + 1 > max_run:
                st.cutover["capacity_rejected"] = {
                    "slots": f"{load}+1>{max_run}"
                }
                st.error = f"capacity slots {load}+1>{max_run}"
                return False
        except Exception as exc:
            print(f"[gw] cutover slot gate skipped: {exc!r}", flush=True)
        t0 = time.monotonic()
        rv = self._send_timed(
            self.dst,
            {
                "cmd": "reserve",
                "items": [{"rid": st.drid, "tokens": n}],
                "epoch": epoch,
            },
            120,
            st,
            "reserve",
        )
        st.cutover["reserve_ms"] = (time.monotonic() - t0) * 1000.0
        if rv.get("errors"):
            st.cutover["capacity_rejected"] = rv.get("errors")
            return False
        self._maybe_kill("reserve")
        self._journal_update(st, force=True, phase="reserved", epoch=epoch)

        t_freeze = time.monotonic()
        st.freeze_send_t = t_freeze
        fr = self._send_timed(
            self.src,
            {
                "cmd": "freeze",
                "rid": st.rid,
                "transfer": transfer,
                "epoch": epoch,
            },
            600,
            st,
            "freeze",
        )
        st.cutover["freeze_ms"] = (time.monotonic() - t_freeze) * 1000.0
        st.freeze_t = time.monotonic()
        st.freeze_return_t = st.freeze_t
        st.freeze_epoch = epoch
        tp_fr = fr.get("tp_rank_results") or [fr]
        for idx, rr in enumerate(tp_fr):
            if rr.get("errors"):
                raise RuntimeError(
                    f"freeze rank {rr.get('tp_rank', idx)}: "
                    f"{rr['errors']}"
                )
        outs = {int(r.get("output_len") or 0) for r in tp_fr}
        kvs = {int(r.get("kv_committed_len") or 0) for r in tp_fr}
        if len(outs) != 1 or len(kvs) != 1:
            raise RuntimeError(
                f"freeze TP mismatch rid={st.rid} out={outs} kv={kvs}"
            )
        st.freeze_result = tp_fr[0]
        st.freeze_results_all = tp_fr
        st.transfer_token = (
            tp_fr[0].get("transfer") or {}
        ).get("token")
        self._maybe_kill("freeze")
        n_tp = len(tp_fr)
        transfers = [r.get("transfer") for r in tp_fr]
        if inject == "import_fail":
            transfers[0] = dict(transfers[0])
            transfers[0]["token"] = "0" * 32

        # Send every source token that was committed by the freeze, then hand
        # the stream to the target.  This closes the poll-vs-freeze race.
        # The freeze result is authoritative for the cut point; the poller
        # snapshot may lag by one token.  Using a stale length here makes the
        # target's max_new_tokens one too large and leaks an extra token.
        prefix_len = int(tp_fr[0].get("output_len", 0) or 0)
        deadline = time.monotonic() + 2.0
        src_ids = ()
        while time.monotonic() < deadline:
            item = self.src.get(st.rid)
            if item is not None and len(item[0]) >= prefix_len:
                src_ids = item[0]
                break
            self.src.wait_output(st.rid, prefix_len - 1, 0.05)
        if len(src_ids) < prefix_len:
            raise RuntimeError(
                f"source snapshot {len(src_ids)} < freeze prefix {prefix_len}"
            )
        self._drain_source(handler, st, src_ids[:prefix_len], st.sent)
        st.sent = prefix_len
        st.cutover["prefix_len"] = prefix_len
        st.prefix_len = prefix_len
        self._journal_update(
            st, force=True, phase="frozen", epoch=epoch,
            prefix_len=prefix_len, owner="src",
            freeze_transfer=fr.get("transfer"),
        )
        st.owner = "dst"
        st.owner_epoch = epoch

        t_imp = time.monotonic()
        item = {
            "rid": st.drid,
            "remaining": st.max_new - prefix_len,
            "epoch": epoch,
        }
        if n_tp > 1:
            item["transfers"] = transfers
        else:
            item["transfer"] = transfers[0]
        print(
            f"[gw] cutover rid={st.rid} n_tp={n_tp} "
            f"keys={sorted(item.keys())} transfers={len(transfers)}",
            flush=True,
        )
        try:
            if n_tp > 1:
                payload = {
                    "cmd": "import_wave", "items": [item],
                    "admit": False, "epoch": epoch,
                }
                timeout = 600
                if inject == "timeout_once":
                    payload["debug_delay_ms"] = 1500
                    timeout = 0.35
                imp = self._send_timed(
                    self.dst, payload, timeout, st, "import_wave"
                )
                if imp.get("errors"):
                    raise RuntimeError(f"import_wave: {imp['errors']}")
                ad = self._send_timed(
                    self.dst,
                    {"cmd": "admit_decode", "rids": [st.drid]},
                    60, st, "admit_decode",
                )
                ad_errs = self._all_rank_errors(ad)
                if ad_errs:
                    raise RuntimeError(f"admit_decode: {ad_errs}")
            elif st.wave:
                payload = {
                    "cmd": "import_wave",
                    "items": [item],
                    "admit": True,
                    "epoch": epoch,
                }
                timeout = 600
                if inject == "timeout_once":
                    payload["debug_delay_ms"] = 1500
                    timeout = 0.35
                imp = self._send_timed(
                    self.dst, payload, timeout, st, "import_wave"
                )
                if imp.get("errors"):
                    raise RuntimeError(f"import_wave: {imp['errors']}")
                if inject == "duplicate_wave":
                    dup = self._send_timed(
                        self.dst,
                        {
                            "cmd": "import_wave",
                            "items": [item],
                            "admit": False,
                            "epoch": epoch,
                        },
                        600,
                        st,
                        "import_wave_dup",
                    )
                    if dup.get("errors") or not all(
                        x.get("duplicate") for x in dup.get("imports", [])
                    ):
                        raise RuntimeError(f"duplicate import_wave: {dup}")
            else:
                imp = self._send_timed(
                    self.dst, {"cmd": "import", **item}, 600, st, "import"
                )
                self._send_timed(self.dst, {"cmd": "admit"}, 30, st, "admit")
            st.cutover["import_ms"] = (time.monotonic() - t_imp) * 1000.0
            self._journal_update(
                st, force=True, phase="imported", owner="dst", epoch=epoch,
                prefix_len=prefix_len,
            )
            self._maybe_kill("import")
            self._maybe_kill("transfer")
            threading.Thread(
                target=self._release_source_when_local,
                args=(st, transfer),
                daemon=True,
            ).start()
            return True
        except Exception as exc:
            st.cutover["import_error"] = repr(exc)
            # Tombstone the epoch before undoing: a timed-out command may still
            # execute on the target after this controller has resumed the
            # source, and must not create a second owner.
            self._cancel_epochs([st])
            # Fail-closed if the target had already committed; otherwise undo
            # the import and resume the source.
            ab = self.dst.send(
                {"cmd": "abort_import", "rid": st.drid}, timeout=120
            )
            st.cutover["abort"] = ab
            if ab.get("committed"):
                st.target_committed = True
                st.error = "target_committed"
                return False
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel", "rids": [st.drid]}, timeout=60
                )
            except Exception:
                pass
            if n_tp > 1:
                rv2 = self.src.send(
                    {"cmd": "resume_many_decode", "rids": [st.rid]},
                    timeout=120,
                )
                res = (rv2.get("results") or {}).get(st.rid) or {}
                st.cutover["resume"] = res
                if rv2.get("errors") or not res.get("resumed"):
                    st.error = (
                        f"resume failed: {rv2.get('errors') or res}"
                    )
                    return False
            else:
                rs = self.src.send(
                    {"cmd": "resume", "rid": st.rid}, timeout=60
                )
                st.cutover["resume"] = rs
                if not rs.get("resumed"):
                    st.error = f"resume failed: {rs}"
                    return False
            st.owner = "src"
            st.owner_epoch = "src"
            return False

    @staticmethod
    def _rank_results(res: dict) -> list:
        return res.get("tp_rank_results") or [res]

    @classmethod
    def _all_rank_errors(cls, res: dict) -> dict:
        """Merge per-rank ``errors`` maps; rank0 alone is not authoritative."""
        errs = {}
        for rr in cls._rank_results(res):
            for k, v in (rr.get("errors") or {}).items():
                errs[k] = v
        return errs

    @classmethod
    def _release_all_ranks(cls, res: dict):
        """Return (all_ranks_released, per-rank detail).

        A deferred/released=False result means that rank still holds the
        source KV (an inflight remote-attention RPC).  ``missing`` means the
        rank has no tracked request and therefore no source KV for this rid.
        """
        detail = []
        ok = True
        for idx, rr in enumerate(cls._rank_results(res)):
            rank = int(rr.get("tp_rank") or idx)
            released = bool(rr.get("released"))
            deferred = bool(rr.get("deferred"))
            missing = bool(rr.get("missing"))
            detail.append({
                "rank": rank,
                "released": released,
                "deferred": deferred,
                "missing": missing,
            })
            if not (released or missing):
                ok = False
        return ok, detail

    def _release_every_rank(self, role, rid: str, timeout: float = 10.0):
        rv = role.send(
            {"cmd": "release", "rid": rid, "unknown_ok": True},
            timeout=timeout,
        )
        ok, detail = self._release_all_ranks(rv)
        return ok, detail, rv

    def _release_source_when_local(self, st: RequestState, transfer: str):
        """Free the source only after every rank owns all KV it will read."""
        deadline = time.monotonic() + 120.0
        # The handler drains the source tokens committed at freeze before it
        # flips the owner.  Releasing the source earlier removes that req from
        # status and makes the handler wait out its drain deadline.
        while time.monotonic() < deadline:
            if st.owner == "dst" and st.sent >= st.prefix_len:
                break
            time.sleep(0.005)
        # Layer/remote ranks keep reading source KV until their LOCAL event is
        # set.  Require all ranks, not just rank 0.  If the target already
        # finished, no further RPC can be issued and release is safe even when
        # a layer never became local (the failure-injection path).
        while time.monotonic() < deadline:
            if transfer in ("mooncake", "socket"):
                break
            try:
                _src, _dst, states, completed = self._facts_bundle()
                t = states.get(st.rid) or completed.get(st.rid)
            except Exception:
                t = None
            if t is not None and (
                t.get("target_local_done") or t.get("finished_at")
            ):
                break
            time.sleep(0.02)
        t = None
        try:
            _src, _dst, states, completed = self._facts_bundle()
            t = states.get(st.rid) or completed.get(st.rid)
        except Exception:
            pass
        if t is not None:
            st.target_import_done_at = t.get("import_done_at")
            st.target_last_remote_at = t.get("last_remote_at")
            st.target_created_at = t.get("created_at")
            st.target_admitted_at = t.get("admitted_at")
            st.target_first_step_at = t.get("first_step_at")
            st.target_first_remote_at = t.get("first_remote_at")
            st.target_first_token_at = t.get("first_token_at")
            st.target_step_starts = t.get("step_starts")
            st.target_admit_at = t.get("admit_at")
            st.target_import_timeline = t.get("import_timeline")
            st.target_layer_ready_at = t.get("layer_ready_at")
            st.target_layer_copy_ms = t.get("layer_copy_ms")
            st.target_staging_bytes = t.get("host_staging_bytes")
        ts = self.src.transfer_stats.get(st.transfer_token)
        if ts is not None:
            st.source_transfer = {
                k: ts.get(k)
                for k in (
                    "start_at", "accept_at", "done_at", "send_ms",
                    "engine", "pipeline", "streams", "async",
                    "target_slots", "bytes", "layers",
                )
            }
        while time.monotonic() < deadline:
            if st.owner != "dst":
                return
            try:
                ok, detail, _rv = self._release_every_rank(
                    self.src, st.rid, timeout=60
                )
                st.cutover["source_release_ranks"] = detail
                if ok:
                    st.source_released_t = time.monotonic()
                    return
            except Exception as exc:
                st.cutover["source_release_error"] = repr(exc)
            time.sleep(0.05)


    @_serialize_migration
    def _wave_freeze(self, handler, st: RequestState, transfer: str, inject: str):
        print(f"[gw] enter _wave_freeze rid={st.rid} group={st.group}",
              flush=True)
        """Freeze one wave member at its own cut boundary and hold it.

        Reservations stay per-request because capacity must be checked before
        that request freezes.  The wave batches the *target* import/admit in a
        single command once all members are frozen.
        """
        epoch = self._cutover_epoch([st])
        st.freeze_epoch = epoch
        st.transfer = transfer
        with self.group_lock:
            g = self.groups.get(st.group) or {}
            if g.get("aborted"):
                st.batch_failed = True
                st.error = st.error or str(g.get("aborted"))
                return False
        n = st.prompt_len + st.cut + 64
        if inject == "capacity":
            n += 10_000_000
        try:
            # Slot-aware capacity gate (per-request wave path): count target
            # rows plus members already frozen but not yet imported, so an
            # over-capacity wave aborts before this member freezes.
            try:
                st_now = self.dst.send({"cmd": "status"}, timeout=4)
                tps = st_now.get("tp_rank_results") or [st_now]
                max_run = min(
                    int(x.get("max_running") or 0) for x in tps
                )
                load = max(len(x.get("reqs") or []) for x in tps)
                frozen_other = sum(
                    1 for x in (g.get("members") or [])
                    if x is not st and x.freeze_result is not None
                )
                if max_run and load + frozen_other + 1 > max_run:
                    raise RuntimeError(
                        f"target slots load={load} + frozen={frozen_other}"
                        f" + 1 > max_running={max_run}"
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"[gw] wave slot gate skipped: {exc!r}", flush=True)
            rv = self._send_timed(
                self.dst,
                {
                    "cmd": "reserve",
                    "items": [{"rid": st.drid, "tokens": n}],
                    "epoch": epoch,
                },
                120,
                st,
                "reserve",
            )
            if rv.get("errors"):
                st.cutover["capacity_rejected"] = rv["errors"]
                raise RuntimeError(f"reserve: {rv['errors']}")
            self._maybe_kill("reserve")
            t_freeze = time.monotonic()
            st.freeze_send_t = t_freeze
            fr = self._send_timed(
                self.src,
                {
                    "cmd": "freeze",
                    "rid": st.rid,
                    "transfer": transfer,
                    "epoch": epoch,
                },
                600,
                st,
                "freeze",
            )
            st.freeze_t = time.monotonic()
            st.freeze_return_t = st.freeze_t
            st.cutover["freeze_ms"] = (
                st.freeze_t - t_freeze
            ) * 1000.0
            tp_fr = fr.get("tp_rank_results") or [fr]
            for idx, rr in enumerate(tp_fr):
                if rr.get("errors"):
                    raise RuntimeError(
                        f"freeze rank {rr.get('tp_rank', idx)}: "
                        f"{rr['errors']}"
                    )
            outs = {int(r.get("output_len") or 0) for r in tp_fr}
            kvs = {int(r.get("kv_committed_len") or 0) for r in tp_fr}
            if len(outs) != 1 or len(kvs) != 1:
                raise RuntimeError(
                    f"freeze TP mismatch rid={st.rid} out={outs} kv={kvs}"
                )
            st.freeze_results_all = tp_fr
            st.freeze_result = tp_fr[0]
            st.transfer_token = (
                tp_fr[0].get("transfer") or {}
            ).get("token")
            self._maybe_kill("freeze")
            prefix_len = int(tp_fr[0].get("output_len", 0) or 0)
            st.prefix_len = prefix_len
            st.cutover["prefix_len"] = prefix_len
            deadline = time.monotonic() + 2.0
            src_ids = ()
            while time.monotonic() < deadline:
                item = self.src.get(st.rid)
                if item is not None and len(item[0]) >= prefix_len:
                    src_ids = item[0]
                    break
                self.src.wait_output(st.rid, prefix_len - 1, 0.05)
            if len(src_ids) < prefix_len:
                raise RuntimeError(
                    f"source snapshot {len(src_ids)} < freeze prefix "
                    f"{prefix_len}"
                )
            self._drain_source(handler, st, src_ids[:prefix_len], st.sent)
            st.sent = prefix_len
            return True
        except Exception as exc:
            st.batch_failed = True
            st.error = repr(exc)
            threading.Thread(
                target=self._wave_abort,
                args=(st.group, f"freeze: {exc!r}"),
                daemon=True,
            ).start()
            return False

    def _wave_abort(self, group: str, reason: str):
        """Resume every wave member whose source was already frozen."""
        with self.group_lock:
            g = self.groups.get(group)
            if g is None or g.get("aborted"):
                return
            g["aborted"] = reason
            members = list(g.get("all") or g.get("members") or [])
        n_tp = max(
            (len(getattr(x, "freeze_results_all", []) or [])
             for x in members), default=1
        )
        if n_tp > 1:
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel",
                     "rids": [x.drid for x in members]},
                    timeout=10,
                )
            except Exception:
                pass
            to_resume = []
            for st in members:
                st.batch_failed = True
                if st.error is None:
                    st.error = reason
                if st.owner != "src" or st.freeze_result is None:
                    continue
                if getattr(st, "_wave_resumed", False):
                    continue
                to_resume.append(st)
            if to_resume:
                try:
                    rv = self.src.send(
                        {"cmd": "resume_many_decode",
                         "rids": [x.rid for x in to_resume]},
                        timeout=120,
                    )
                    for st in to_resume:
                        st._wave_resumed = True
                        res = (rv.get("results") or {}).get(
                            st.rid
                        ) or {}
                        st.cutover["resume"] = res
                        if rv.get("errors") or not res.get("resumed"):
                            st.error = (
                                f"resume failed: "
                                f"{rv.get('errors') or res}"
                            )
                except Exception as exc:
                    for st in to_resume:
                        st.error = f"resume error: {exc!r}"
            return
        for st in members:
            st.batch_failed = True
            if st.error is None:
                st.error = reason
            if st.owner != "src" or not st.freeze_epoch:
                continue
            if getattr(st, "_wave_resumed", False):
                continue
            st._wave_resumed = True
            try:
                rs = self.src.send({"cmd": "resume", "rid": st.rid}, timeout=60)
                st.cutover["resume"] = rs
                if not rs.get("resumed"):
                    st.error = f"resume failed: {rs}"
            except Exception as exc:
                st.error = f"resume error: {exc!r}"

    def _wave_finish(self, group: str, transfer: str, inject: str):
        print(f"[gw] enter _wave_finish group={group}", flush=True)
        with self.group_lock:
            g = self.groups.get(group)
            if g is None or g.get("aborted"):
                return
            members = list(g["members"])
        items = []
        n_tp = 1
        for st in members:
            rows = getattr(st, "freeze_results_all", None) or [
                st.freeze_result or {}
            ]
            n_tp = max(n_tp, len(rows))
            trs = [dict((r or {}).get("transfer") or {}) for r in rows]
            if inject == "import_fail" and st is members[0]:
                trs[0]["token"] = "0" * 32
            item = {
                "rid": st.drid,
                "remaining": st.max_new - st.prefix_len,
                "epoch": st.freeze_epoch,
            }
            if n_tp > 1:
                item["transfers"] = trs
            else:
                item["transfer"] = trs[0]
            items.append(item)
        print(
            f"[gw] wave_finish n_tp={n_tp} items="
            f"{[sorted(x.keys()) for x in items]}", flush=True,
        )
        payload = {
            "cmd": "import_wave", "items": items,
            "admit": n_tp <= 1,
        }
        timeout = 1200
        if inject == "timeout_once":
            payload["debug_delay_ms"] = 1500
            timeout = 0.35
        try:
            rv = self._send_timed(
                self.dst, payload, timeout, members[0], "import_wave"
            )
            if n_tp > 1 and not rv.get("errors"):
                ad = self._send_timed(
                    self.dst,
                    {"cmd": "admit_decode",
                     "rids": [st.drid for st in members]},
                    60,
                    members[0],
                    "admit_decode",
                )
                ad_errs = self._all_rank_errors(ad)
                if ad_errs:
                    rv["errors"] = ad_errs
                else:
                    rv["admitted"] = ad.get("count")
            if inject == "duplicate_wave":
                dup = self._send_timed(
                    self.dst,
                    {"cmd": "import_wave", "items": items, "admit": False},
                    1200,
                    members[0],
                    "import_wave_dup",
                )
                if dup.get("errors") or not all(
                    x.get("duplicate") for x in dup.get("imports", [])
                ):
                    raise RuntimeError(f"duplicate import_wave: {dup}")
        except Exception as exc:
            rv = {"errors": {"wave": repr(exc)}}
        recs = members[0].ctl.get("import_wave") or []
        rec = recs[-1] if recs else {"wall_ms": None}
        for st in members[1:]:
            st.ctl.setdefault("import_wave", []).append(rec)
        for st in members:
            st.cutover["import_wave_ms"] = rec.get("wall_ms")
            st.cutover["admitted"] = rv.get("admitted")
        if rv.get("errors"):
            print(f"[gw] wave import failed {group}: {rv.get('errors')}",
                  flush=True)
            with self.group_lock:
                g = self.groups.get(group) or {}
                g["aborted"] = f"import_wave: {rv.get('errors')}"
            self._cancel_epochs(members)
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel",
                     "rids": [st.drid for st in members]},
                    timeout=10,
                )
            except Exception:
                pass
            if n_tp > 1:
                for st in members:
                    try:
                        self.dst.send(
                            {"cmd": "abort_import", "rid": st.drid},
                            timeout=10,
                        )
                    except Exception:
                        pass
                to_resume = []
                for st in members:
                    st.batch_failed = True
                    st.error = (
                        st.error or f"import_wave: {rv.get('errors')}"
                    )
                    if getattr(st, "_wave_resumed", False):
                        continue
                    if st.freeze_result is None:
                        continue
                    to_resume.append(st)
                if to_resume:
                    try:
                        rv2 = self.src.send(
                            {"cmd": "resume_many_decode",
                             "rids": [x.rid for x in to_resume]},
                            timeout=120,
                        )
                        for st in to_resume:
                            st._wave_resumed = True
                            res = (rv2.get("results") or {}).get(
                                st.rid
                            ) or {}
                            st.cutover["resume"] = res
                            if rv2.get("errors") or not res.get(
                                    "resumed"):
                                st.error = (
                                    f"resume failed: "
                                    f"{rv2.get('errors') or res}"
                                )
                    except Exception as exc:
                        for st in to_resume:
                            st.error = f"resume error: {exc!r}"
            else:
                for st in members:
                    st.batch_failed = True
                    st.error = (
                        st.error or f"import_wave: {rv.get('errors')}"
                    )
                    if getattr(st, "_wave_resumed", False):
                        continue
                    st._wave_resumed = True
                    try:
                        rs = self.src.send(
                            {"cmd": "resume", "rid": st.rid}, timeout=60
                        )
                        st.cutover["resume"] = rs
                    except Exception as exc:
                        st.error = f"resume error: {exc!r}"
            return
        self._maybe_kill("import")
        self._maybe_kill("transfer")
        for st in members:
            st.owner = "dst"
            st.owner_epoch = st.freeze_epoch
            st.switch_ready = True
            threading.Thread(
                target=self._release_source_when_local,
                args=(st, transfer),
                daemon=True,
            ).start()
        print(f"[gw] wave cutover done {group} n={len(members)}", flush=True)

    def _join_timeout(self, group: str):
        with self.group_lock:
            g = self.groups.get(group)
            if g is None or g.get("started") or g.get("aborted"):
                return
        print(f"[gw] wave join timeout {group}", flush=True)
        self._wave_abort(group, "wave join timeout")

    def _join_group(self, st: RequestState, transfer: str, inject: str):
        with self.group_lock:
            g = self.groups.setdefault(
                st.group,
                {"members": [], "expected": st.group_size, "started": False,
                 "all": [st], "aborted": None},
            )
            g.setdefault("all", [])
            g.setdefault("aborted", None)
            if not g.get("watcher"):
                g["watcher"] = True
                timer = threading.Timer(
                    WAVE_JOIN_TIMEOUT_S,
                    self._join_timeout,
                    args=(st.group,),
                )
                timer.daemon = True
                timer.start()
            if g.get("aborted"):
                return
            if not any(x is st for x in g["members"]):
                g["members"].append(st)
            print(
                f"[gw] join rid={st.rid} group={st.group} "
                f"{len(g['members'])}/{g['expected']}",
                flush=True,
            )
            if not g["started"] and len(g["members"]) >= g["expected"]:
                g["started"] = True
                print(f"[gw] {'wave' if st.wave else 'batch'} start {st.group}",
                      flush=True)
                target = (
                    self._wave_finish
                    if (st.wave and st.wave_freeze != "all")
                    else self._batch_cutover
                )
                threading.Thread(
                    target=target,
                    args=(st.group, transfer, inject),
                    daemon=True,
                ).start()

    @_serialize_migration
    def _batch_cutover(self, group: str, transfer: str, inject: str):
        print(f"[gw] enter _batch_cutover group={group}", flush=True)
        with self.group_lock:
            members = list(self.groups[group]["members"])
        wave_all = bool(members) and all(
            st.wave and st.wave_freeze == "all" for st in members
        )
        epoch = self._cutover_epoch(members)
        rids = [st.rid for st in members]
        n_tp = 1
        try:
            t_res = time.monotonic()
            rv = self._send_timed(
                self.dst,
                {
                    "cmd": "reserve",
                    "items": [
                        {
                            "rid": st.drid,
                            "tokens": st.prompt_len + st.cut
                            + (10_000_000 if inject == "capacity" else 64),
                        }
                        for st in members
                    ],
                    "epoch": epoch,
                },
                120,
                members[0],
                "reserve",
            )
            self._copy_ctl(members, "reserve")
            if rv.get("errors"):
                raise RuntimeError(f"reserve: {rv['errors']}")
            # Slot-aware capacity gate: a KV reservation does not bound target
            # request slots.  Over-capacity admits can make imported requests
            # disappear, so reject before freezing and keep the source live.
            try:
                st_now = self.dst.send({"cmd": "status"}, timeout=4)
                tps = st_now.get("tp_rank_results") or [st_now]
                max_run = min(
                    int(x.get("max_running") or 0) for x in tps
                )
                load = max(len(x.get("reqs") or []) for x in tps)
                print(
                    f"[gw] drain slot gate load={load} "
                    f"members={len(members)} max_run={max_run}",
                    flush=True,
                )
                if max_run and load + len(members) > max_run:
                    raise RuntimeError(
                        f"target slots load={load} + members={len(members)}"
                        f" > max_running={max_run}"
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"[gw] slot gate skipped: {exc!r}", flush=True)
            self._maybe_kill("reserve")
            t_fr = time.monotonic()
            for st in members:
                st.freeze_send_t = t_fr
                st.freeze_epoch = epoch
            fr = self._send_timed(
                self.src,
                {
                    "cmd": "freeze_all",
                    "rids": rids,
                    "transfer": transfer,
                    "epoch": epoch,
                    "cut": 0,
                    "target_rids": {st.rid: st.drid for st in members},
                },
                1200,
                members[0],
                "freeze_all",
            )
            self._copy_ctl(members, "freeze_all")
            if fr.get("errors") or fr.get("pending"):
                raise RuntimeError(f"freeze_all: {fr.get('errors')}")
            self._maybe_kill("freeze")
            now_fr = time.monotonic()
            for st in members:
                st.freeze_t = now_fr
                st.freeze_return_t = now_fr
                st.cutover["reserve_ms"] = (t_fr - t_res) * 1000.0
                st.cutover["freeze_all_ms"] = (now_fr - t_fr) * 1000.0
            t_imp = time.monotonic()
            tp_fr = fr.get("tp_rank_results") or [fr]
            n_tp = len(tp_fr)
            for idx, rr in enumerate(tp_fr):
                if rr.get("errors"):
                    raise RuntimeError(
                        f"freeze_all rank {rr.get('tp_rank', idx)}: "
                        f"{rr['errors']}"
                    )
            per_rank = [
                {r["rid"]: r for r in (rr.get("results") or [])}
                for rr in tp_fr
            ]
            if not wave_all and n_tp <= 1:
                self._send_timed(
                    self.dst, {"cmd": "hold_admission"}, 30, members[0],
                    "hold_admission",
                )
                self._copy_ctl(members, "hold_admission")
            items = []
            for st in members:
                rows = [pr.get(st.rid) for pr in per_rank]
                if not all(rows):
                    raise RuntimeError(
                        f"missing freeze result for {st.rid}: "
                        f"{[r is not None for r in rows]}"
                    )
                outs = {int(r["output_len"]) for r in rows}
                kvs = {int(r["kv_committed_len"]) for r in rows}
                eps = {str(r.get("migration_epoch")) for r in rows}
                if len(outs) != 1 or len(kvs) != 1 or eps != {epoch}:
                    raise RuntimeError(
                        f"freeze TP mismatch {st.rid}: out={outs} "
                        f"kv={kvs} epoch={eps} want={epoch}"
                    )
                st.prefix_len = outs.pop()
                st.freeze_result = rows[0]
                st.freeze_results_all = rows
                st.transfer_token = (
                    rows[0].get("transfer") or {}
                ).get("token")
                item = {
                    "rid": st.drid,
                    "remaining": st.max_new - st.prefix_len,
                    "epoch": epoch,
                }
                if n_tp > 1:
                    item["transfers"] = [r["transfer"] for r in rows]
                else:
                    item["transfer"] = rows[0]["transfer"]
                if inject == "import_fail" and st is members[0]:
                    if n_tp > 1:
                        item["transfers"] = list(item["transfers"])
                        item["transfers"][0] = dict(item["transfers"][0])
                        item["transfers"][0]["token"] = "0" * 32
                    else:
                        item["transfer"] = dict(item["transfer"])
                        item["transfer"]["token"] = "0" * 32
                items.append(item)
            if n_tp > 1:
                # TP target: import every rank in one wave, then admit the
                # whole batch as decode (PD-style).  Extend-style admission
                # would recompute the pending token with a different kernel.
                self._send_timed(
                    self.dst, {"cmd": "hold_admission"}, 30, members[0],
                    "hold_admission",
                )
                self._copy_ctl(members, "hold_admission")
                payload = {
                    "cmd": "import_wave", "items": items, "admit": False
                }
                timeout = 1200
                if inject == "timeout_once":
                    payload["debug_delay_ms"] = 1500
                    timeout = 0.35
                imp = self._send_timed(
                    self.dst, payload, timeout, members[0], "import_wave"
                )
                self._copy_ctl(members, "import_wave")
                if imp.get("errors"):
                    raise RuntimeError(f"import_wave: {imp['errors']}")
                if int(imp.get("count") or 0) != len(items):
                    raise RuntimeError(f"import_wave count: {imp}")
                ad = self._send_timed(
                    self.dst,
                    {"cmd": "admit_decode",
                     "rids": [st.drid for st in members]},
                    60,
                    members[0],
                    "admit_decode",
                )
                self._copy_ctl(members, "admit_decode")
                ad_errs = self._all_rank_errors(ad)
                if ad_errs:
                    raise RuntimeError(f"admit_decode: {ad_errs}")
                if int(ad.get("count") or 0) != len(members):
                    raise RuntimeError(f"admit_decode count: {ad}")
            elif wave_all:
                payload = {
                    "cmd": "import_wave", "items": items, "admit": True
                }
                timeout = 1200
                if inject == "timeout_once":
                    payload["debug_delay_ms"] = 1500
                    timeout = 0.35
                imp = self._send_timed(
                    self.dst, payload, timeout, members[0], "import_wave"
                )
                self._copy_ctl(members, "import_wave")
                if imp.get("errors"):
                    raise RuntimeError(f"import_wave: {imp['errors']}")
                if inject == "duplicate_wave":
                    dup = self._send_timed(
                        self.dst,
                        {"cmd": "import_wave", "items": items,
                         "admit": False},
                        1200,
                        members[0],
                        "import_wave_dup",
                    )
                    self._copy_ctl(members, "import_wave_dup")
                    if dup.get("errors") or not all(
                        x.get("duplicate") for x in dup.get("imports", [])
                    ):
                        raise RuntimeError(f"duplicate import_wave: {dup}")
            elif transfer == "mooncake":
                t_imp = time.monotonic()
                imp = self._send_timed(
                    self.dst, {"cmd": "import_many", "items": items}, 1200,
                    members[0], "import_many",
                )
                self._copy_ctl(members, "import_many")
                if imp.get("errors"):
                    raise RuntimeError(f"import_many: {imp['errors']}")
            else:
                for st, item in zip(members, items):
                    self._send_timed(
                        self.dst, {"cmd": "import", **item}, 1200, st, "import"
                    )
                self._send_timed(
                    self.dst, {"cmd": "admit"}, 30, members[0], "admit"
                )
                self._copy_ctl(members, "admit")
            self._maybe_kill("import")
            self._maybe_kill("transfer")
            t_adm = time.monotonic()
            for st in members:
                st.owner_epoch = epoch
                st.switch_ready = True
                st.cutover["import_ms"] = (t_adm - t_imp) * 1000.0
                threading.Thread(
                    target=self._release_source_when_local,
                    args=(st, transfer),
                    daemon=True,
                ).start()
        except Exception as exc:
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel",
                     "rids": [st.drid for st in members]},
                    timeout=10,
                )
            except Exception:
                pass
            if wave_all:
                self._cancel_epochs(members)
            if n_tp > 1:
                to_resume = []
                for st in members:
                    st.batch_failed = True
                    st.error = f"batch cutover: {exc!r}"
                    try:
                        ab = self.dst.send(
                            {"cmd": "abort_import", "rid": st.drid},
                            timeout=60,
                        )
                        if not ab.get("committed"):
                            if st.freeze_result is not None:
                                to_resume.append(st)
                        else:
                            st.error = "target_committed"
                    except Exception:
                        pass
                if to_resume:
                    try:
                        rv = self.src.send(
                            {"cmd": "resume_many_decode",
                             "rids": [x.rid for x in to_resume]},
                            timeout=120,
                        )
                        if rv.get("errors"):
                            raise RuntimeError(rv["errors"])
                        for st in to_resume:
                            res = (rv.get("results") or {}).get(
                                st.rid
                            ) or {}
                            st.cutover["resume"] = res
                            if not res.get("resumed"):
                                st.error = f"resume failed: {res}"
                    except Exception as rexc:
                        for st in to_resume:
                            st.error = f"resume failed: {rexc!r}"
            else:
                for st in members:
                    st.batch_failed = True
                    st.error = f"batch cutover: {exc!r}"
                    try:
                        ab = self.dst.send(
                            {"cmd": "abort_import", "rid": st.drid},
                            timeout=60,
                        )
                        if not ab.get("committed"):
                            if st.freeze_result is None:
                                continue
                            rs = self.src.send(
                                {"cmd": "resume", "rid": st.rid},
                                timeout=60,
                            )
                            st.cutover["resume"] = rs
                            if not rs.get("resumed"):
                                st.error = f"resume failed: {rs}"
                    except Exception:
                        pass
            print(f"[gw] batch failed {group}: {exc!r}", flush=True)
            try:
                self.dst.send({"cmd": "admit"}, timeout=10)
            except Exception:
                pass

    def _state_from_entry(self, entry: dict) -> RequestState:
        st = RequestState(
            entry["rid"],
            int(entry.get("prompt_len", 0)),
            int(entry.get("max_new", 128)),
            int(entry.get("cut", 0)),
        )
        st.drid = str(entry.get("drid") or f"dst-{st.rid}")
        st.transfer = entry.get("transfer") or "remote"
        st.prefix_len = int(entry.get("prefix_len") or 0)
        st.freeze_epoch = entry.get("epoch")
        st.owner = entry.get("owner", "src")
        st.owner_epoch = entry.get("epoch", "src")
        return st

    # ------------------------------------------------------------------
    # P5.4-B2: journal-driven drain state machine.
    #
    #   REQUESTED -> DRAINING -> PREPARING -> MIGRATING -> RECONCILING
    #             -> SAFE_TO_STOP -> STOPPED
    #             -> BLOCKED -> FAILED_UNSAFE (deadline, ownership still fenced)
    #
    # Each step first persists its intent, then runs idempotent Source/Target
    # commands, then records the result.  A controller killed at any point can
    # therefore replay from the journal plus authoritative Source/Target facts.
    # Only the reconcile thread writes owner/phase; stream handlers write only
    # their own delivery progress (``sent``/``finished``).
    # ------------------------------------------------------------------

    DRAIN_LIVE = ("REQUESTED", "DRAINING", "PREPARING", "MIGRATING",
                  "RECONCILING")
    DRAIN_DONE = ("SAFE_TO_STOP", "STOPPED")
    DRAIN_FENCED = DRAIN_LIVE + DRAIN_DONE + ("BLOCKED", "FAILED_UNSAFE")
    _MIG_FIELDS = (
        "owner", "phase", "prefix_len", "epoch", "freeze_transfer",
        "freeze_transfers", "tp_size",
        "source_released", "outcome", "committed_at", "observed_committed",
        "target_local_done", "target_finished", "target_released",
        "source_release_ranks", "source_release_deferred",
        "target_release_ranks", "resume_result",
        "cancel_confirmed", "migration_abandoned", "needs_cancel",
        "drain_id", "wave_id", "drain_member", "last_error",
    )

    def _drain_persist(self, d, entries=None):
        with self.journal_lock:
            cur = self.journal.get("__drain__")
            if (isinstance(cur, dict)
                    and cur.get("drain_id") != d.get("drain_id")):
                # A prior rollback's reconcile tick may still be in flight
                # when a new drain publishes its fence. Never overwrite it.
                return
            if isinstance(cur, dict):
                cur.update(d)
            else:
                self.journal["__drain__"] = dict(d)
            if entries:
                for rid, e in entries.items():
                    rec = self.journal.get(rid)
                    if not isinstance(rec, dict):
                        continue
                    # Write every field, including None: the journal is
                    # merged, so an omitted key would keep a stale value
                    # (e.g. needs_cancel=true forever after one cancel).
                    for k in self._MIG_FIELDS:
                        rec[k] = e.get(k)
            try:
                tmp = self.journal_path.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(self.journal, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(tmp, self.journal_path)
            except OSError:
                pass

    def src_worker_state(self) -> str:
        """D0 state: source health first, then the persistent drain state."""
        if self._facts_at and "reqs" not in self._src_facts:
            return "UNAVAILABLE"
        with self.journal_lock:
            raw = self.journal.get("__drain__")
            d = copy.deepcopy(raw) if isinstance(raw, dict) else None
        if d:
            state = d.get("state")
            if state in self.DRAIN_LIVE or state in ("BLOCKED", "FAILED_UNSAFE"):
                return "DRAINING"
            if state in self.DRAIN_DONE:
                return "SAFE_TO_STOP"
            # ROLLED_BACK / unknown -> READY.
        return "READY"

    def dst_worker_state(self) -> str:
        if self._facts_at and not self._target_ok:
            return "UNAVAILABLE"
        return "READY"

    def _drain_snapshot(self):
        with self.journal_lock:
            raw = self.journal.get("__drain__")
            d = copy.deepcopy(raw) if isinstance(raw, dict) else None
            entries = {
                k: copy.deepcopy(v)
                for k, v in self.journal.items()
                if k != "__drain__" and isinstance(v, dict)
            }
        return d, entries

    def _journal_flush(self):
        with self.journal_lock:
            try:
                tmp = self.journal_path.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(self.journal, ensure_ascii=False),
                    encoding="utf-8",
                )
                os.replace(tmp, self.journal_path)
            except OSError:
                pass

    def _adopt_proxy_requests(self):
        """Merge Router-direct admissions recorded while the controller was down."""
        try:
            files = list(self.proxy_dir.glob("*.json"))
        except OSError:
            return
        if not files:
            return
        changed = False
        for path in files:
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rid = str(rec.get("rid") or "")
            if not rid:
                continue
            with self.journal_lock:
                cur = self.journal.get(rid)
            if isinstance(cur, dict):
                if not cur.get("adopted"):
                    continue
                fin = bool(rec.get("finished"))
                out = int(rec.get("output_len") or 0)
                if cur.get("finished") != fin or int(cur.get("sent") or 0) != out:
                    with self.journal_lock:
                        entry = self.journal.get(rid)
                        if isinstance(entry, dict):
                            entry["finished"] = fin
                            entry["sent"] = out
                            if fin:
                                entry["phase"] = "done"
                    changed = True
                continue
            owner = str(rec.get("owner") or "src")
            fin = bool(rec.get("finished"))
            entry = {
                "rid": rid,
                "drid": str(rec.get("drid") or f"dst-{rid}"),
                "owner": owner,
                "phase": "done" if fin
                else ("direct" if owner == "dst" else "accepted"),
                "epoch": "direct" if owner == "dst" else "proxy",
                "direct": owner == "dst",
                "transfer": "direct",
                "max_new": int(rec.get("max_new") or 0),
                "prompt_len": int(rec.get("prompt_len") or 0),
                "sent": int(rec.get("output_len") or 0),
                "finished": fin,
                "adopted": True,
            }
            with self.journal_lock:
                self.journal[rid] = entry
            print(
                f"[gw] adopt proxy rid={rid} owner={owner} finished={fin}",
                flush=True,
            )
            changed = True
        if changed:
            self._journal_flush()

    def _admission_capacity_wait(self, need: int, forced: str,
                                rid: str = ""):
        """Admission path selector.

        ``hold`` preserves the P5.8 behaviour: every new D1 request waits for
        Drain→SAFE.  ``decoupled`` asks the target for an atomic capacity
        ticket and admits immediately if the remaining capacity fits.
        """
        if ADMISSION_POLICY == "decoupled" and rid:
            return self._ordinary_admission_wait(need, forced, rid)
        """Hold new D1 traffic until a live d2d drain reaches a terminal state.

        The wait must be lightweight: polling the source/target control
        channel from every waiting admission thread contends with the drain's
        own freeze/import RPCs and can push the wave past the frozen watchdog.
        Only the journal file is read here; the reconcile thread owns all
        worker-status polls.
        """
        deadline = time.monotonic() + float(
            os.environ.get("HANDOFF_ADMISSION_WAIT_S", "30")
        )
        wait_start = None
        last_log = 0.0
        while time.monotonic() < deadline:
            if forced == "src":
                return
            d, _entries = self._drain_snapshot()
            if not d or d.get("state") not in self.DRAIN_LIVE:
                if wait_start is not None:
                    print(
                        f"[gw] admission held={time.monotonic() - wait_start:.2f}s "
                        f"need={need} released",
                        flush=True,
                    )
                return
            if (d.get("mode") or "d2d") == "natural":
                return
            if self.src_worker_state() == "READY":
                return
            now = time.monotonic()
            if wait_start is None:
                wait_start = now
                print(
                    f"[gw] admission hold start need={need} "
                    f"state={d.get('state')} "
                    f"wave={(d.get('wave') or {}).get('phase')}",
                    flush=True,
                )
            elif now - last_log > 5.0:
                last_log = now
                print(
                    f"[gw] admission hold t={now - wait_start:.1f}s "
                    f"state={d.get('state')} "
                    f"wave={(d.get('wave') or {}).get('phase')}",
                    flush=True,
                )
            time.sleep(0.05)
        print(
            f"[gw] admission hold timeout need={need} "
            f"waited={time.monotonic() - (wait_start or time.monotonic()):.1f}s",
            flush=True,
        )

    def _ordinary_admission_wait(self, need: int, forced: str, rid: str):
        """Admit new traffic against Target's authoritative leftover capacity.

        ``rid`` is the *target* request id (``dst-<source rid>``), because the
        capacity lease is keyed by the id the target will actually submit.
        """
        if forced == "src" and not PD_PREFILL_URL:
            return None
        # Strict PD: acquire the D0 ordinary lease before triggering P.  If
        # D0 has no KV/slot capacity, fail the admission here instead of
        # starting a P prefill whose bootstrap receiver will never exist
        # (the source of the "Failed to send kv chunk"/connection-refused
        # cascade).
        role = self.src if forced == "src" else self.dst
        deadline = time.monotonic() + float(
            os.environ.get("HANDOFF_ADMISSION_WAIT_S", "30")
        )
        last_error = None
        last_log = 0.0
        while time.monotonic() < deadline:
            d, _entries = self._drain_snapshot()
            src_state = self.src_worker_state()
            drain_live = bool(
                d and d.get("state") in self.DRAIN_LIVE
            )
            if (
                forced != "dst"
                and not drain_live
                and src_state == "READY"
            ):
                return None
            if d and d.get("preflight_pending"):
                time.sleep(0.01)
                continue
            if d and d.get("state") == "FAILED_UNSAFE":
                return {"error": "drain_failed_unsafe", "rid": rid}
            epoch = str((d or {}).get("epoch") or "")
            try:
                rv = role.send(
                    {"cmd": "admit_ordinary", "rid": rid,
                     "tokens": int(need), "epoch": epoch},
                    timeout=4,
                )
            except Exception as exc:
                last_error = repr(exc)
                time.sleep(0.05)
                continue
            rows = self._rank_results(rv)
            if rows and all(bool(r.get("admitted")) for r in rows):
                return {
                    "ticket": rid,
                    "tokens": int(need),
                    "epoch": epoch,
                    "capacity": rv.get("capacity"),
                }
            if any(bool(r.get("admitted")) for r in rows):
                self._cancel_capacity_ticket(rid, role=role)
            bad = next(
                (r for r in rows if not r.get("admitted")), {}
            )
            last_error = bad.get("reason") or rv.get("reason") or "capacity"
            now = time.monotonic()
            if now - last_log > 5.0:
                last_log = now
                print(
                    f"[gw] ordinary admission waiting rid={rid} "
                    f"need={need} reason={last_error} "
                    f"state={(d or {}).get('state')}",
                    flush=True,
                )
            time.sleep(0.05)
        return {
            "error": "capacity_timeout",
            "reason": last_error,
            "rid": rid,
        }

    def _target_ticket_wait(self, need: int, target_rid: str,
                            timeout: float = 30.0):
        """Acquire a target capacity ticket; wait for slots/KV to free."""
        deadline = time.monotonic() + float(
            os.environ.get("HANDOFF_ADMISSION_WAIT_S", str(timeout))
        )
        last = None
        while time.monotonic() < deadline:
            d, _ = self._drain_snapshot()
            if d and d.get("preflight_pending"):
                time.sleep(0.01)
                continue
            if d and d.get("state") == "FAILED_UNSAFE":
                return {"error": "drain_failed_unsafe", "rid": target_rid}
            try:
                rv = self.dst.send(
                    {"cmd": "admit_ordinary", "rid": target_rid,
                     "tokens": int(need), "epoch": ""},
                    timeout=4,
                )
            except Exception as exc:
                last = repr(exc)
                time.sleep(0.05)
                continue
            rows = self._rank_results(rv)
            if rows and all(bool(r.get("admitted")) for r in rows):
                return {
                    "ticket": target_rid, "tokens": int(need),
                    "epoch": "", "capacity": rv.get("capacity"),
                }
            if any(bool(r.get("admitted")) for r in rows):
                self._cancel_capacity_ticket(target_rid)
            bad = next(
                (r for r in rows if not r.get("admitted")), {}
            )
            last = bad.get("reason") or rv.get("reason") or "capacity"
            time.sleep(0.05)
        return {
            "error": "capacity_timeout", "reason": last,
            "rid": target_rid,
        }

    def _cancel_capacity_ticket(self, rid: str, role=None) -> None:
        try:
            (role or self.dst).send(
                {"cmd": "capacity_cancel", "rid": rid}, timeout=4
            )
        except Exception:
            pass

    def _precommit_rollback_safe(self, d, entries) -> bool:
        if self._target_ok or self._target_fail_since is None:
            return False
        if time.monotonic() - self._target_fail_since < 1.5:
            return False
        wave = d.get("wave") or {}
        if (wave.get("phase") or "idle") in ("importing", "imported"):
            return False
        for rid in d.get("members") or []:
            e = entries.get(rid) or {}
            # Import/commit means the target may own KV.  A ``done`` member
            # simply finished on the source and must not block the rollback.
            if e.get("phase") in ("importing", "imported", "committed"):
                return False
            if e.get("owner") == "dst" and e.get("observed_committed"):
                return False
        return True

    def _rollback_target(self, d, entries):
        """Target disappeared before any import: source still owns all KV."""
        members = list(d.get("members") or [])
        d["state"] = "ROLLED_BACK"
        d["rolled_back_at"] = time.time()
        d["failed_reason"] = "target_unavailable"
        d["blocked_reason"] = None
        d["reserve_tokens"] = 0
        self.draining = False
        src_reqs = {
            r.get("rid"): r for r in (self._src_facts.get("reqs") or [])
        }
        for rid in members:
            e = entries.get(rid) or {}
            e["needs_cancel"] = False
            e["migration_abandoned"] = True
            if e.get("phase") != "done":
                e["owner"] = "src"
                e["phase"] = "accepted"
            if src_reqs.get(rid, {}).get("frozen"):
                try:
                    e["resume_result"] = self._resume_source_fast(
                        rid, timeout=120
                    )
                except Exception as exc:
                    e["resume_error"] = repr(exc)
        rids = [
            (entries.get(rid) or {}).get("drid")
            for rid in members
            if (entries.get(rid) or {}).get("drid")
        ]
        if rids:
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel", "rids": rids}, timeout=10
                )
                d["reserve_cleanup"] = []
            except Exception:
                d["reserve_cleanup"] = rids
        print(
            f"[gw] drain ROLLED_BACK reason=target_unavailable "
            f"members={len(members)}",
            flush=True,
        )

    def _refresh_facts(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self._facts_at < 0.5:
            return
        self._facts_at = now
        if "reqs" in self._dst_facts:
            self._last_target_facts = copy.deepcopy(self._dst_facts)
        # The role poller already probes status every few ms; during a worker
        # outage its error flag is authoritative and lets us fail over
        # immediately instead of queueing another status RPC behind it.
        if getattr(self.src, "error", None) is not None:
            self._src_facts = {"error": "src"}
        else:
            try:
                self._src_facts = self.src.send({"cmd": "status"}, timeout=4)
            except Exception:
                self._src_facts = {"error": "src"}
        if getattr(self.dst, "error", None) is not None:
            # The poller saw a sustained failure.  Before declaring the
            # target down, make one independent status attempt: the poller
            # can be starved by a long import while the target is still
            # serving, and a false target_unreachable blocks the drain.
            try:
                self._dst_facts = self.dst.send(
                    {"cmd": "status"}, timeout=4
                )
                self._target_ok = True
                self._target_fail_since = None
                self._dst_fact_fails = 0
            except Exception:
                self._dst_facts = {"error": "dst"}
                self._dst_fact_fails += 1
                self._target_ok = False
                if (
                    self._dst_fact_fails >= 2
                    and self._target_fail_since is None
                ):
                    self._target_fail_since = time.monotonic()
        else:
            try:
                self._dst_facts = self.dst.send({"cmd": "status"}, timeout=4)
                self._target_ok = True
                self._target_fail_since = None
                self._dst_fact_fails = 0
            except Exception:
                self._dst_facts = {"error": "dst"}
                self._dst_fact_fails += 1
                self._target_ok = False
                if (
                    self._dst_fact_fails >= 2
                    and self._target_fail_since is None
                ):
                    self._target_fail_since = time.monotonic()

    @staticmethod
    def _merge_req_rows(res: dict) -> dict:
        """Merge per-rank rows; length/finished require every TP rank."""
        tp = res.get("tp_rank_results") or [res]
        per = [{r["rid"]: r for r in (rr.get("reqs") or [])} for rr in tp]
        if not per:
            return {}
        merged = {}
        for rid, r0 in per[0].items():
            rows = [p.get(rid) for p in per]
            m = dict(r0)
            all_ranks = all(r is not None for r in rows)
            m["all_ranks"] = all_ranks
            if all_ranks:
                m["output_len"] = min(
                    int(r.get("output_len") or 0) for r in rows
                )
                m["finished"] = all(bool(r.get("finished")) for r in rows)
                m["kv_committed_len"] = min(
                    int(r.get("kv_committed_len") or 0) for r in rows
                )
            else:
                m["output_len"] = 0
                m["finished"] = False
            merged[rid] = m
        return merged

    @staticmethod
    def _merge_states(res: dict, key: str) -> dict:
        """Merge target/completed states; commit requires every TP rank."""
        tp = res.get("tp_rank_results") or [res]
        per = [
            {s.get("handoff_id"): s for s in (rr.get(key) or [])}
            for rr in tp
        ]
        if not per:
            return {}
        merged = {}
        for hid, s0 in per[0].items():
            rows = [p.get(hid) for p in per]
            m = dict(s0)
            all_ranks = all(r is not None for r in rows)
            m["all_ranks"] = all_ranks
            if all_ranks:
                m["new_tokens"] = min(
                    int(r.get("new_tokens") or 0) for r in rows
                )
                done = [r.get("import_done_at") for r in rows]
                m["import_done_at"] = (
                    max(done) if all(x is not None for x in done) else None
                )
                fin = [r.get("finished_at") for r in rows]
                m["finished_at"] = (
                    max(fin) if all(x is not None for x in fin) else None
                )
                if any("local_ready" in r for r in rows):
                    # LOCAL is an all-rank fact.  A rank that still has a
                    # remote layer must keep the source lease alive.
                    m["target_local_done"] = all(
                        bool(r.get("local_ready")) for r in rows
                    )
                    m["layers_ready_min"] = min(
                        int(r.get("layers_ready") or 0) for r in rows
                    )
                    m["layers_total"] = max(
                        int(r.get("layers_total") or 0) for r in rows
                    )
                elif any("target_local_done" in r for r in rows):
                    m["target_local_done"] = all(
                        bool(r.get("target_local_done")) for r in rows
                    )
            else:
                m["new_tokens"] = 0
                m["import_done_at"] = None
                m["finished_at"] = None
            merged[hid] = m
        return merged

    def _facts_bundle(self):
        src, dst = self._src_facts, self._dst_facts
        src_reqs = (
            self._merge_req_rows(src) if "reqs" in src else {}
        )
        dst_reqs = (
            self._merge_req_rows(dst) if "reqs" in dst else {}
        )
        states = (
            self._merge_states(dst, "target_states")
            if "target_states" in dst else {}
        )
        completed = (
            self._merge_states(dst, "completed_target_states")
            if "completed_target_states" in dst else {}
        )
        return src_reqs, dst_reqs, states, completed

    def _sync_member(self, e, src_reqs, dst_reqs, states, completed):
        """Derive owner/phase from authoritative facts (monotone forward)."""
        if e.get("phase") == "done":
            return
        rid = e.get("rid")
        drid = e.get("drid")
        sr = src_reqs.get(rid)
        dr = dst_reqs.get(drid)
        # Live target states and completed snapshots are keyed by the *source*
        # rid recorded in the freeze metadata; the target request itself is
        # tracked under the destination rid.
        stt = states.get(rid)
        if stt is not None and not stt.get("all_ranks", True):
            # A single rank's state (or a partial import after a rank fault)
            # is never sufficient to adopt target ownership.
            stt = None
        if stt is not None and not self._member_epoch_match(e, stt):
            stt = None
        comp = completed.get(rid)
        if comp is not None and not comp.get("all_ranks", True):
            comp = None
        if comp is not None and not self._member_epoch_match(e, comp):
            comp = None
        base = int(e.get("prefix_len") or 0)
        observed_commit = False
        imported = False
        if stt is not None:
            observed_commit = int(stt.get("new_tokens") or 0) > 0
            imported = bool(stt.get("import_done_at")) or observed_commit
            if stt.get("target_local_done"):
                e["target_local_done"] = True
            # A finished target can no longer issue remote-attention RPCs.
            # Propagate that fact even when one layer never became local
            # (LOCAL Drop hook), otherwise the source stays frozen forever
            # and the drain never leaves RECONCILING.
            if stt.get("finished_at"):
                e["target_finished"] = True
        elif dr is not None:
            # A live target row with new tokens is commit evidence.  A row
            # that was created by an import that timed out is not: it may be
            # a late partial import after the source was resumed.  Only
            # adopt it once it has actually generated past the frozen prefix.
            observed_commit = int(dr.get("output_len", 0)) > base
            imported = observed_commit
            if dr.get("finished"):
                e["target_finished"] = True
        elif comp is not None:
            # The target finished this handoff and already dropped its live
            # state; the completed snapshot is durable commit evidence.
            observed_commit = (
                int(comp.get("new_tokens") or 0) > 0
                or bool(comp.get("finished_at"))
            )
            if comp.get("target_local_done") or comp.get("import_done_at"):
                e["target_local_done"] = True
            if comp.get("finished_at"):
                e["target_finished"] = True
        if observed_commit or imported:
            # A cancel inference made while the target was unreachable is
            # obsolete as soon as authoritative commit/import facts are back.
            e["needs_cancel"] = False
        if observed_commit and not e.get("observed_committed"):
            print(
                f"[gw] commit observed rid={rid} phase={e.get('phase')} "
                f"kill_at={DRAIN_KILL_AT or '-'}",
                flush=True,
            )
            e["observed_committed"] = True
            e.setdefault("committed_at", time.time())
            self._maybe_drain_kill("commit")
        if e.get("observed_committed"):
            e["owner"] = "dst"
            e.setdefault("_owner_dst_at", time.monotonic())
            if e.get("phase") != "committed":
                e["phase"] = "committed"
            # ``target_local_done`` is derived above from the live state or
            # the completed snapshot; the committed fast-path must not hide it.
            return
        if imported:
            e["owner"] = "dst"
            e.setdefault("_owner_dst_at", time.monotonic())
            e["phase"] = "imported"
            return
        if sr is not None:
            e["src_output_len"] = int(sr.get("output_len", 0))
            if e.get("owner") == "dst":
                if not self._target_ok:
                    # Missing target facts during a partition are not evidence
                    # that the target dropped the request.
                    return
                # The phase switch to owner=dst happens before the next facts
                # refresh observes the target-side import state.  Do not
                # treat that one-refresh window as a lost target: require a
                # target facts snapshot taken after the owner switch and a
                # short grace period.
                switched = e.get("_owner_dst_at")
                if (
                    switched is None
                    or self._facts_at < switched
                    or time.monotonic() - switched < 1.0
                ):
                    return
                # Target objects disappeared without commit evidence.  Do not
                # fall back to the source silently; resolve via cancel fence.
                self._note_cancel(e, "target_gone_source_alive")
                return
            e["owner"] = "src"
            if sr.get("frozen"):
                # Source still being frozen cannot erase a persisted import
                # intent: an unreachable target may already have committed.
                if e.get("phase") not in ("importing", "imported", "committed"):
                    e["phase"] = "frozen"
            elif sr.get("finished"):
                e["phase"] = "done"
                e.setdefault("outcome", "drained")
            elif e.get("phase") in ("reserved",):
                pass
            elif e.get("phase") in ("frozen", "imported", "committed"):
                # Monotone: a stale facts refresh from before the freeze must
                # not move a frozen member back to "accepted" and make the
                # import wave see an empty member set.
                pass
            else:
                e["phase"] = "accepted"
            return
        # No live object on either side.
        if not self._target_ok:
            # Facts unavailable: keep the last known phase and let the
            # terminal rule go BLOCKED instead of inventing an outcome.
            return
        phase = e.get("phase")
        if e.get("owner") == "dst":
            self._note_cancel(e, "no_objects_owner_dst")
            return
        if phase in ("frozen", "reserved", "imported", "committed"):
            self._note_cancel(e, "no_objects_phase")
            return
        e["phase"] = "done"
        e.setdefault("outcome", "drained")

    @staticmethod
    def _member_epoch_match(e, obj) -> bool:
        want = str(e.get("epoch") or "")
        got = str(obj.get("migration_epoch") or "")
        if not want:
            # Before this member is frozen any target object with an epoch is
            # a leftover from an earlier run with recycled rids.
            return not got
        if not got:
            return True
        return want == got

    def _note_cancel(self, e, why: str):
        print(
            f"[gw] needs_cancel rid={e.get('rid')} why={why} "
            f"phase={e.get('phase')} owner={e.get('owner')} "
            f"epoch={e.get('epoch')} local={e.get('target_local_done')}",
            
        )
        e["needs_cancel"] = True

    def _resume_source_fast(self, rid: str, timeout: float = 120.0):
        """Resume a frozen source request on the same kernel path as Natural.

        TP2 must use the decode-style resume; the plain extend resume computes
        the pending token with a different kernel and breaks the numeric
        canary even when recompute_tokens=0.
        """
        try:
            stp = self.src.send({"cmd": "status_lite"}, timeout=2)
            n_tp = len(stp.get("tp_rank_results") or [stp])
        except Exception:
            n_tp = 1
        if n_tp > 1:
            rv = self.src.send(
                {"cmd": "resume_many_decode", "rids": [rid]},
                timeout=timeout,
            )
            res = (rv.get("results") or {}).get(rid) or {}
            if rv.get("errors") or not res.get("resumed"):
                raise RuntimeError(
                    f"resume_many_decode: {rv.get('errors') or res}"
                )
            return res
        return self.src.send(
            {"cmd": "resume", "rid": rid}, timeout=timeout
        )

    def _cancel_and_resume(self, e, d=None):
        """Fence late commits (epoch tombstone + abort), then resume.

        Never resumes the source on local absence alone: the target must
        confirm that no token was committed for this epoch.
        """
        rid = e.get("rid")
        drid = e.get("drid")
        epoch = str(e.get("epoch") or e.get("drain_epoch") or "")
        # A member that already finished on the source has nothing to fence
        # or resume.  This is common for a short P->D request that completes
        # at the drain boundary; resuming it returns "not_frozen" and used to
        # turn the whole drain into BLOCKED.
        if e.get("phase") == "done" and e.get("owner") in (None, "src"):
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel", "rids": [drid]}, timeout=5
                )
            except Exception:
                pass
            e["needs_cancel"] = False
            e["owner"] = "src"
            e["cancel_confirmed"] = True
            e.setdefault("outcome", "drained")
            return "src"
        # A request that never froze or imported (typically a new P->D short
        # that finished at the drain boundary) has no target-side request to
        # fence.  Only its capacity reservation exists.  Drop that
        # reservation and let the source finish it; do not turn this normal
        # race into a drain BLOCKED / resume path.  This must run before the
        # target-health gate: an unrelated target status timeout must not
        # block a drain on a request that was never migrated.
        if (
            e.get("phase") == "reserved"
            and e.get("freeze_result") is None
            and not e.get("freeze_transfers")
            and e.get("epoch") in (None, "src")
            and not e.get("observed_committed")
        ):
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel", "rids": [drid]}, timeout=5
                )
            except Exception as exc:
                e["last_error"] = f"reserve_cancel: {exc!r}"
                if d is not None:
                    cleanup = d.setdefault("reserve_cleanup", [])
                    if drid not in cleanup:
                        cleanup.append(drid)
            e["cancel_confirmed"] = True
            e["owner"] = "src"
            e["migration_abandoned"] = True
            e["needs_cancel"] = False
            if e.get("phase") != "done":
                e["phase"] = "accepted"
            return "src"
        if not self._target_ok:
            e["last_error"] = "target_unreachable"
            return "blocked"
        try:
            if epoch:
                self.dst.send(
                    {"cmd": "cancel_import",
                     "items": [{"rid": drid, "epoch": epoch}]},
                    timeout=5,
                )
            ab = self.dst.send(
                {"cmd": "abort_import", "rid": drid}, timeout=5
            )
        except Exception as exc:
            e["last_error"] = f"abort_import: {exc!r}"
            return "blocked"
        if ab.get("committed"):
            e["observed_committed"] = True
            e["owner"] = "dst"
            e.setdefault("_owner_dst_at", time.monotonic())
            e["phase"] = "committed"
            e["needs_cancel"] = False
            e.setdefault("committed_at", time.time())
            self._maybe_drain_kill("commit")
            return "dst"
        was_imported = (
            e.get("phase") in ("imported", "committed")
            or e.get("observed_committed")
            or e.get("target_local_done")
        )
        if ab.get("missing") and was_imported:
            # The target no longer holds the request and there is no durable
            # commit snapshot.  "Missing" is not proof that nothing committed,
            # so never resume the source from here.
            e["last_error"] = "target_missing_after_import"
            print(
                f"[gw] cancel ambiguous rid={rid} (target missing after "
                f"import, no durable commit evidence)",
                flush=True,
            )
            return "blocked"
        e["cancel_confirmed"] = True
        try:
            self.dst.send(
                {"cmd": "reserve_cancel", "rids": [drid]}, timeout=5
            )
        except Exception:
            pass
        try:
            res = self._resume_source_fast(rid, timeout=120)
            e["resume_result"] = res
        except Exception as exc:
            if "unknown rid" in repr(exc):
                e["phase"] = "done"
                e.setdefault("outcome", "drained")
                e["owner"] = "src"
                e["needs_cancel"] = False
                return "src"
            e["last_error"] = f"resume: {exc!r}"
            return "blocked"
        e["owner"] = "src"
        if e.get("phase") != "done":
            e["phase"] = "accepted"
        e["migration_abandoned"] = True
        e["needs_cancel"] = False
        return "src"

    def _drain_prepare(self, d, entries, src_reqs, members):
        """Reserve capacity then freeze; replay-safe for the same epoch."""
        wave = d.setdefault("wave", {})
        wave.setdefault("timeline", {})["prepare_at"] = time.monotonic()
        epoch = str(
            wave.get("epoch") or d.get("epoch") or uuid.uuid4().hex
        )
        wave["epoch"] = epoch
        target = []
        for rid in members:
            e = entries.get(rid)
            if e is None or e.get("phase") == "done":
                continue
            if e.get("migration_abandoned") or e.get("needs_cancel"):
                continue
            sr = src_reqs.get(rid)
            if sr is None:
                continue
            if int(sr.get("output_len", 0)) < 1 and not sr.get("frozen"):
                continue
            target.append(rid)
        if getattr(self.dst, "error", None) is not None:
            # The poller already knows the target is gone; never queue a
            # reserve RPC behind a dead worker (it would block the rollback).
            return False
        # Intent first: a replay after a kill reuses this epoch and the same
        # member set, so reserve/freeze are idempotent.
        wave["phase"] = "freezing"
        wave["members"] = list(target)
        wave["intent"] = {"op": "freeze", "epoch": epoch,
                          "members": list(target), "ts": time.time()}
        d["state"] = "PREPARING"
        self._drain_persist(d, entries)
        if not target:
            wave["phase"] = "failed"
            return False
        target_reserved = (self._dst_facts.get("reserved") or {})
        items = []
        for rid in target:
            e = entries[rid]
            if e.get("phase") in ("reserved", "frozen"):
                continue
            if e["drid"] in target_reserved:
                e["phase"] = "reserved"
                continue
            out = int(src_reqs[rid].get("output_len", 0))
            items.append({
                "rid": e["drid"],
                # Output can grow between reserve and freeze/import; keep a
                # 512-token margin or import_wave fails the reservation check.
                "tokens": int(e.get("prompt_len", 0)) + out + 512,
                "slots": 1,
            })
        if items:
            # Test hook: widen the drain-start -> reserve window so admission
            # races can be exercised deterministically.
            delay = float(os.environ.get("HANDOFF_RESERVE_DELAY_S", "0"))
            if delay > 0 and not wave.get("_delay_applied"):
                wave["_delay_applied"] = True
                print(f"[gw] reserve delay {delay}s", flush=True)
                time.sleep(delay)
            try:
                wave["timeline"]["reserve_send_at"] = time.monotonic()
                rv = self.dst.send(
                    {"cmd": "reserve", "items": items, "epoch": epoch},
                    timeout=5,
                )
                wave["timeline"]["reserve_done_at"] = time.monotonic()
            except Exception as exc:
                rv = {"errors": {it["rid"]: repr(exc) for it in items}}
            errs = self._all_rank_errors(rv)
            for rid in target:
                e = entries[rid]
                if e["drid"] in errs:
                    why = str(errs[e["drid"]])
                    e["needs_cancel"] = True
                    e["last_error"] = f"reserve: {why}"
                    d["last_error"] = e["last_error"]
                    if "insufficient KV" in why or "alloc failed" in why:
                        d["failed_reason"] = "capacity"
                else:
                    e["phase"] = "reserved"
        # If any reservation failed, do not freeze anyone: freezing a subset
        # would strand those requests while the rollback reopens D0.
        reserve_errors = bool(rv.get("errors")) if items else False
        if reserve_errors:
            wave["phase"] = "failed"
            wave["members"] = []
            self._drain_persist(d, entries)
            return False
        self._maybe_drain_kill("join")
        target = [r for r in target if not entries[r].get("needs_cancel")]
        if not target:
            wave["phase"] = "failed"
            return False
        def _ready(rid):
            sr = src_reqs.get(rid) or {}
            out = int(sr.get("output_len", 0))
            kv = int(sr.get("kv_committed_len", 0))
            prompt = int(entries[rid].get("prompt_len", 0))
            # A resumed request re-extends its output before it is legal to
            # freeze again; see handoff_control._freeze_request.
            return out > 0 and kv >= prompt + out - 1

        to_freeze = [
            r for r in target if not src_reqs.get(r, {}).get("frozen")
        ]
        not_ready = [r for r in to_freeze if not _ready(r)]
        to_freeze = [r for r in to_freeze if _ready(r)]
        if not_ready:
            since = wave.setdefault("not_ready_since", time.time())
            e0 = entries[not_ready[0]]
            d["last_error"] = (
                f"freeze pending: {len(not_ready)} request(s) still "
                f"re-extending (kv={src_reqs.get(not_ready[0], {}).get('kv_committed_len')})"
            )
            if time.time() - float(since) > 15.0:
                for rid in not_ready:
                    self._note_cancel(entries[rid], "freeze_not_ready")
            target = [r for r in target if not entries[r].get("needs_cancel")]
            if not target:
                wave["phase"] = "failed"
                return False
        if to_freeze:
            try:
                stp = self.src.send({"cmd": "status_lite"}, timeout=2)
                n_tp = len(stp.get("tp_rank_results") or [stp])
            except Exception:
                n_tp = 1
            arm = str(d.get("transfer") or "").strip().lower()
            aliases = {
                "copy": "mooncake",
                "copy_wait": "mooncake",
                "mooncake": "mooncake",
                "layer": "layer",
                "layer_wait": "layer",
                "remote": "remote",
                "remote_progressive": "remote",
            }
            if arm:
                mode = aliases.get(arm, arm)
            else:
                mode = "mooncake" if n_tp > 1 else "remote"
            if mode not in ("mooncake", "layer", "remote", "socket"):
                raise RuntimeError(f"unsupported transfer arm {arm!r}")
            wave["transfer_mode"] = mode
            try:
                wave["timeline"]["freeze_send_at"] = time.monotonic()
                fr = self.src.send(
                    {"cmd": "freeze_all", "rids": list(to_freeze),
                     "transfer": mode, "epoch": epoch, "cut": 0,
                     "target_rids": {
                         rid: entries[rid]["drid"] for rid in to_freeze
                     }},
                    timeout=120,
                )
                wave["timeline"]["freeze_done_at"] = time.monotonic()
            except Exception as exc:
                fr = {"errors": {r: repr(exc) for r in to_freeze}}
            tp_fr = fr.get("tp_rank_results") or []
            rank_maps = [
                {x.get("rid"): x for x in (rr.get("results") or [])}
                for rr in tp_fr
            ]
            results = {r.get("rid"): r for r in (fr.get("results") or [])}
            errs = dict(fr.get("errors") or {})
            for idx, rr in enumerate(tp_fr):
                if rr.get("errors"):
                    errs.update(rr["errors"])
            for rid in to_freeze:
                e = entries[rid]
                if len(rank_maps) > 1:
                    rows = [m.get(rid) for m in rank_maps]
                    if not all(rows):
                        errs[rid] = errs.get(rid) or "missing rank result"
                        res = None
                    else:
                        outs = {int(r.get("output_len") or 0) for r in rows}
                        kvs = {int(r.get("kv_committed_len") or 0)
                               for r in rows}
                        eps = {str(r.get("migration_epoch")) for r in rows}
                        if len(outs) != 1 or len(kvs) != 1 or eps != {epoch}:
                            errs[rid] = (
                                f"TP mismatch out={outs} kv={kvs} "
                                f"epoch={eps}"
                            )
                            res = None
                        else:
                            res = rows[0]
                            e["freeze_transfers"] = [
                                r.get("transfer") for r in rows
                            ]
                            e["tp_size"] = len(rows)
                else:
                    res = results.get(rid)
                if res is not None:
                    e["prefix_len"] = int(res.get("output_len", 0))
                    e["epoch"] = epoch
                    e["freeze_transfer"] = res.get("transfer")
                    e["phase"] = "frozen"
                    e["owner"] = "src"
                    for s in self.active.values():
                        if s.rid == rid:
                            s.freeze_result = res
                            s.transfer_token = (
                                (res.get("transfer") or {}).get("token")
                            )
                else:
                    why = errs.get(rid) or "no result"
                    e["last_error"] = f"freeze: {why}"
                    d["last_error"] = e["last_error"]
                    if "resume_pending" in why or "no committed KV" in why:
                        e["freeze_retry"] = True
                    else:
                        self._note_cancel(e, "freeze_error")
        frozen = [
            r for r in target if entries[r].get("phase") == "frozen"
        ]
        t_frz = wave["timeline"].get("freeze_done_at")
        t_frs = wave["timeline"].get("freeze_send_at")
        t_res_s = wave["timeline"].get("reserve_send_at")
        t_res_d = wave["timeline"].get("reserve_done_at")
        for s in self.active.values():
            if s.rid not in frozen:
                continue
            s.freeze_send_t = t_frs
            s.freeze_t = t_frz
            s.freeze_return_t = t_frz
            if t_frs is not None and t_frz is not None:
                s.cutover["freeze_all_ms"] = (t_frz - t_frs) * 1000.0
            if t_res_s is not None and t_res_d is not None:
                s.cutover["reserve_ms"] = (t_res_d - t_res_s) * 1000.0
        if frozen and not_ready:
            # A request that could not share this freeze boundary drains on
            # the source instead of holding the whole wave hostage.
            for rid in not_ready:
                e = entries[rid]
                if e.get("phase") == "frozen" or e.get("needs_cancel"):
                    continue
                e["migration_abandoned"] = True
                e["last_error"] = "freeze_not_ready_abandoned"
                try:
                    self.dst.send(
                        {"cmd": "reserve_cancel", "rids": [e["drid"]]},
                        timeout=5,
                    )
                except Exception:
                    pass
        if not frozen:
            if any(entries[r].get("freeze_retry") for r in target):
                # Stay in "freezing"; the next reconcile retries the freeze.
                return False
            wave["phase"] = "failed"
            return False
        wave["members"] = frozen
        wave["phase"] = "frozen"
        print(
            f"[gw] drain_prepare frozen={frozen} "
            f"phases={ {r: entries[r].get('phase') for r in target} } "
            f"mode={wave.get('transfer_mode')} n_tp={n_tp}",
            flush=True,
        )
        self._maybe_drain_kill("freeze")
        self._drain_persist(d, entries)
        return True

    def _drain_import(self, d, entries, states, dst_reqs):
        wave = d.setdefault("wave", {})
        members = [
            r for r in (wave.get("members") or [])
            if entries.get(r, {}).get("phase") == "frozen"
        ]
        if not members:
            wm = list(wave.get("members") or [])
            phases = {
                r: (entries.get(r) or {}).get("phase") for r in wm
            }
            print(
                f"[gw] drain_import no-frozen wave_members={wm} "
                f"phases={phases} entries={len(entries)}",
                flush=True,
            )
            if wm and not all(
                p in ("imported", "committed", "done",
                      "migration_abandoned")
                for p in phases.values()
            ):
                # A stale facts refresh briefly showed "accepted" after the
                # freeze.  Keep the wave in "frozen" so the next reconcile
                # imports it; never declare success with no import.
                return False
            wave["phase"] = "imported"
            for rid in d.get("members") or []:
                e = entries.get(rid) or {}
                if e.get("phase") != "imported":
                    continue
                for st in self.active.values():
                    if st.rid == rid:
                        st.prefix_len = int(e.get("prefix_len", 0))
                        st.cutover["prefix_len"] = st.prefix_len
                        st.owner_epoch = e.get("epoch")
                        st.switch_ready = True
            return True
        epoch = wave.get("epoch")
        by_drid = {entries[r]["drid"]: r for r in members}
        print(
            f"[gw] drain_import members={members} "
            f"states={sorted((states or {}).keys())} "
            f"dst_reqs={sorted((dst_reqs or {}).keys())}",
            flush=True,
        )
        already = []
        items = []
        for rid in members:
            e = entries[rid]
            drid = e["drid"]
            st = states.get(rid)
            if st is not None and not self._member_epoch_match(e, st):
                st = None
            if st is not None or drid in dst_reqs:
                e["owner"] = "dst"
                e.setdefault("_owner_dst_at", time.monotonic())
                e["phase"] = "imported"
                already.append(rid)
                continue
            item = {
                "rid": drid,
                "remaining": int(e.get("max_new", 128))
                - int(e.get("prefix_len", 0)),
                "epoch": e.get("epoch") or epoch,
            }
            tps = e.get("freeze_transfers")
            if tps:
                item["transfers"] = tps
            else:
                item["transfer"] = e.get("freeze_transfer")
            items.append(item)
        wave["phase"] = "importing"
        wave["intent"] = {
            "op": "import", "epoch": epoch,
            "members": [it["rid"] for it in items], "ts": time.time(),
        }
        d["state"] = "MIGRATING"
        self._drain_persist(d, entries)
        rv = {}
        if items:
            if DRAIN_IMPORT_FAIL and not d.get("import_failed_once"):
                d["import_failed_once"] = True
                rv = {"errors": {it["rid"]: "injected import failure"
                                 for it in items}}
            else:
                n_tp = max(
                    len(e.get("freeze_transfers") or [])
                    for e in (entries[r] for r in members)
                )
                try:
                    wave["timeline"]["import_send_at"] = time.monotonic()
                    rv = self.dst.send(
                        {"cmd": "import_wave", "items": items,
                         "admit": True, "admit_decode": n_tp > 1},
                        timeout=30,
                    )
                    wave["timeline"]["import_done_at"] = time.monotonic()
                    if n_tp > 1 and not rv.get("errors"):
                        if rv.get("admit_decode") is None:
                            # Compatibility: an older target still needs the
                            # separate TP2 admission command.
                            wave["timeline"]["admit_send_at"] = time.monotonic()
                            ad = self.dst.send(
                                {"cmd": "admit_decode",
                                 "rids": [it["rid"] for it in items]},
                                timeout=60,
                            )
                            wave["timeline"]["admit_done_at"] = time.monotonic()
                            if ad.get("errors"):
                                rv["errors"] = ad["errors"]
                            else:
                                rv["admitted"] = ad.get("count")
                        else:
                            wave["timeline"]["admit_send_at"] = wave[
                                "timeline"
                            ]["import_done_at"]
                            wave["timeline"]["admit_done_at"] = wave[
                                "timeline"
                            ]["import_done_at"]
                except Exception as exc:
                    rv = {"errors": {it["rid"]: repr(exc)
                                     for it in items}}
            errs = self._all_rank_errors(rv)
            if errs:
                # Fence late imports first: a target rank that was stopped
                # mid-import may finish after this controller timed out.  The
                # tombstone must exist before the abort/resume path runs.
                try:
                    self.dst.send(
                        {"cmd": "cancel_import",
                         "items": [
                             {"rid": it["rid"],
                              "epoch": it.get("epoch") or epoch}
                             for it in items
                         ]},
                        timeout=10,
                    )
                except Exception as exc:
                    d["last_error"] = f"cancel_import: {exc!r}"
                for it in items:
                    try:
                        self.dst.send(
                            {"cmd": "abort_import", "rid": it["rid"]},
                            timeout=10,
                        )
                    except Exception:
                        pass
            for it in items:
                rid = by_drid.get(it["rid"])
                e = entries[rid]
                if rid is None:
                    continue
                if errs:
                    # import_wave aborts every import of the wave on error.
                    e["needs_cancel"] = True
                    e["last_error"] = f"import_wave: {errs[it['rid']]}"
                    d["last_error"] = e["last_error"]
                else:
                    e["owner"] = "dst"
                    e.setdefault("_owner_dst_at", time.monotonic())
                    e["phase"] = "imported"
            if errs:
                wave["phase"] = "failed"
                self._drain_persist(d, entries)
                return False
        for rid in members:
            e = entries[rid]
            if e.get("phase") != "imported":
                continue
            for st in self.active.values():
                if st.rid == rid:
                    st.prefix_len = int(e.get("prefix_len", 0))
                    st.cutover["prefix_len"] = st.prefix_len
                    st.owner_epoch = e.get("epoch")
                    st.switch_ready = True
        if all(entries[r].get("phase") != "frozen" for r in members):
            wave["phase"] = "imported"
        else:
            wave["phase"] = "failed"
        self._maybe_drain_kill("import")
        self._maybe_drain_kill("transfer")
        self._drain_persist(d, entries)
        return wave["phase"] == "imported"

    def _drain_release(self, d, entries):
        """Release migrated source KV once the client consumed the prefix.

        ``release`` itself is refcounted: it defers the actual free while the
        target still reads remote KV and fires on the last free notification.
        """
        for rid in d.get("members") or []:
            e = entries.get(rid)
            if not e:
                continue
            if (
                e.get("target_finished")
                and not e.get("target_released")
            ):
                # The target finished this request.  Keep its output until the
                # adopted stream delivered it, otherwise a reconnecting client
                # would lose the tail; free it only when nothing is attached.
                sent = int(e.get("sent") or 0)
                total = int(e.get("max_new") or 0)
                if total and sent < total and rid in self.active:
                    pass
                else:
                    # Free its tracked entry and handoff state.
                    try:
                        ok, detail, _rv = self._release_every_rank(
                            self.dst, e["drid"], timeout=10
                        )
                        e["target_release_ranks"] = detail
                        if ok:
                            e["target_released"] = True
                    except Exception as exc:
                        e["last_error"] = f"target_release: {exc!r}"
            if e.get("source_released"):
                continue
            if e.get("phase") == "done" and e.get("owner") == "src":
                # NATURALLY_DRAINED: the request finished on the source.  Keep
                # it until the adopted stream delivered the tail, then free.
                out_len = int(e.get("src_output_len") or 0)
                # Fact refresh can miss a finished source request (it leaves
                # source facts before src_output_len is recorded).  ``max_new``
                # is then the only safe lower bound: an attached stream still
                # owes the client the full output, and releasing here would
                # strand it after the tokens already delivered.
                total = int(e.get("max_new") or 0)
                sent = int(e.get("sent") or 0)
                if sent < max(out_len, total) and rid in self.active:
                    continue
                try:
                    ok, detail, _rv = self._release_every_rank(
                        self.src, rid, timeout=10
                    )
                    e["source_release_ranks"] = detail
                    if ok:
                        e["source_released"] = True
                        e["source_released_t"] = time.time()
                    else:
                        e["source_release_deferred"] = [
                            d for d in detail if d.get("deferred")
                        ]
                except Exception as exc:
                    e["last_error"] = f"release: {exc!r}"
                continue
            if e.get("owner") != "dst":
                continue
            if e.get("phase") not in ("imported", "committed", "done"):
                continue
            sent = int(e.get("sent") or 0)
            max_new = int(e.get("max_new") or 0)
            if (
                not e.get("target_local_done")
                and e.get("phase") != "done"
                and not (
                    e.get("target_finished")
                    and max_new
                    and sent >= max_new
                )
            ):
                # Layer staging is still switching this request to local KV.
                # Releasing now would break the next remote-attention RPC.
                # If the target finished and the client consumed the full
                # output, no remote RPC can remain, so release is safe.
                continue
            prefix = int(e.get("prefix_len") or 0)
            if prefix > 0 and sent < prefix and rid in self.active:
                # Do not create an output gap while the client still needs
                # the frozen prefix from the source.
                continue
            try:
                ok, detail, _rv = self._release_every_rank(
                    self.src, rid, timeout=10
                )
                e["source_release_ranks"] = detail
                if ok:
                    e["source_released"] = True
                    e["source_released_t"] = time.time()
                else:
                    e["source_release_deferred"] = [
                        d for d in detail if d.get("deferred")
                    ]
            except Exception as exc:
                e["last_error"] = f"release: {exc!r}"

    def _all_determinate(self, entries, members):
        for rid in members:
            e = entries.get(rid) or {}
            if e.get("needs_cancel"):
                return False
            if e.get("owner") not in ("src", "dst"):
                return False
        return True

    def _advance_wave(self, d, entries, src_reqs, dst_reqs, states):
        members = list(d.get("members") or [])
        active = [
            r for r in members
            if (entries.get(r) or {}).get("phase") != "done"
        ]
        if not active:
            if d.get("state") in self.DRAIN_LIVE:
                d["state"] = "RECONCILING"
            return
        wave = d.setdefault("wave", {"phase": "idle", "members": []})
        phase = wave.get("phase") or "idle"
        if (d.get("mode") or "d2d") == "natural":
            # Natural drain: stop admissions, let members finish on D0.
            for rid in active:
                entries[rid]["migration_abandoned"] = True
            wave["phase"] = "failed"
            wave["members"] = []
            d["state"] = "RECONCILING"
            return
        if DRAIN_NOOP:
            # Control arm only: create the drain, never freeze or migrate.
            for rid in active:
                entries[rid]["migration_abandoned"] = True
            wave["phase"] = "failed"
            wave["members"] = []
            d["state"] = "RECONCILING"
            return
        if phase == "idle":
            ready = [
                r for r in active
                if r in src_reqs
                and not (entries[r].get("migration_abandoned"))
                and entries[r].get("owner") != "dst"
            ]
            if not ready:
                d["state"] = "RECONCILING"
                return
            waiting = any(
                int(src_reqs[r].get("output_len", 0)) < 1
                and not src_reqs[r].get("frozen")
                for r in ready
            )
            if waiting:
                d["state"] = "DRAINING"
                return
            epoch = str(d.get("reserved_epoch") or d.get("epoch") or uuid.uuid4().hex)
            wave.update({"phase": "freezing", "epoch": epoch,
                         "members": list(ready)})
            d["state"] = "PREPARING"
            self._drain_persist(d, entries)
            if self._drain_prepare(d, entries, src_reqs, ready):
                # Import in the same reconcile tick as the freeze.  Waiting
                # for the next tick used to add 250 ms between source freeze
                # and the target import command.
                d["state"] = "MIGRATING"
                self._drain_import(d, entries, states, dst_reqs)
        elif phase == "freezing":
            d["state"] = "PREPARING"
            if self._drain_prepare(
                d, entries, src_reqs, list(wave.get("members") or [])
            ):
                d["state"] = "MIGRATING"
                self._drain_import(d, entries, states, dst_reqs)
        elif phase in ("frozen", "importing"):
            # "importing" is the replay entry after a controller kill during
            # the import RPC; _drain_import skips members already on the target.
            if not self._target_ok:
                return
            d["state"] = "MIGRATING"
            self._drain_import(d, entries, states, dst_reqs)
        elif phase in ("imported", "failed"):
            if all(
                entries[r].get("phase") == "done"
                or entries[r].get("owner") == "dst"
                or entries[r].get("migration_abandoned")
                for r in active
            ):
                d["state"] = "RECONCILING"

    def _cleanup_drain_reservations(self, d):
        """Retry target reservation cancellation after a rollback.

        A target rank may be stopped while the rollback runs; the reservation
        then survives until the rank returns.  Keep a durable list on the
        drain and retry every reconcile tick until reserve_cancel succeeds.
        """
        rids = list(d.get("reserve_cleanup") or [])
        if not rids:
            return
        try:
            rv = self.dst.send(
                {"cmd": "reserve_cancel", "rids": rids}, timeout=5
            )
        except Exception:
            return
        d["reserve_cleanup"] = []
        d["reservations_cleaned_at"] = time.time()
        d["reserve_cleanup_result"] = {
            "cancelled": rv.get("cancelled"),
            "missing": rv.get("missing"),
        }

    def _rollback_capacity(self, d, entries):
        """Target cannot hold the wave: reopen D0 instead of taking it down."""
        members = list(d.get("members") or [])
        d["state"] = "ROLLED_BACK"
        d["rolled_back_at"] = time.time()
        d["blocked_reason"] = None
        d["reserve_tokens"] = 0
        d["wave"] = {"phase": "failed", "members": []}
        self.draining = False
        rids = []
        for rid in members:
            e = entries.get(rid) or {}
            e["needs_cancel"] = False
            e["migration_abandoned"] = True
            if e.get("phase") != "done":
                e["phase"] = "accepted"
                e["owner"] = "src"
            if e.get("drid"):
                rids.append(e["drid"])
        try:
            self.dst.send(
                {"cmd": "reserve_cancel", "rids": rids}, timeout=10
            )
            d["reserve_cleanup"] = []
        except Exception:
            d["reserve_cleanup"] = rids
        print(
            f"[gw] drain ROLLED_BACK reason=capacity members={len(members)}",
            flush=True,
        )

    def _drain_terminal(self, d, entries, src_reqs, blocked):
        if d.get("state") in self.DRAIN_DONE + ("ROLLED_BACK",):
            return
        members = list(d.get("members") or [])
        src_ok = "reqs" in self._src_facts
        with self.admission_lock:
            pending_local = any(
                st.owner == "src" for st in self.active.values()
            )
        # SAFE is an all-rank fact.  Every rank must have no request, no
        # reservation, no handoff lease, and no live TE registration.  Rank 0
        # alone is not enough: its KV can be gone while rank 1 still serves a
        # remote-attention RPC from the frozen source KV.
        src_rank_clean = []
        src_clean = src_ok and not pending_local
        if src_ok:
            for idx, rr in enumerate(self._rank_results(self._src_facts)):
                rank = int(rr.get("tp_rank") or idx)
                reg = rr.get("handoff_registry") or {}
                reg_mgr = rr.get("registration_manager") or {}
                row = {
                    "rank": rank,
                    "reqs": len(rr.get("reqs") or []),
                    "reserved": len(rr.get("reserved") or {}),
                    "handoffs": int(reg.get("handoffs") or 0),
                    "inflight": int(reg.get("inflight") or 0),
                    "te_live": int(reg_mgr.get("live") or 0),
                    "te_bytes_live": int(reg_mgr.get("bytes_live") or 0),
                    "kv_available": int(rr.get("kv_available") or 0),
                }
                row["clean"] = bool(
                    row["reqs"] == 0
                    and row["reserved"] == 0
                    and row["handoffs"] == 0
                    and row["inflight"] == 0
                    and row["te_live"] == 0
                    and row["te_bytes_live"] == 0
                )
                src_rank_clean.append(row)
                if not row["clean"]:
                    src_clean = False
        self._src_clean_ranks = src_rank_clean
        self._src_clean_all = src_clean
        if src_clean:
            for rid in members:
                e = entries.get(rid) or {}
                if not e.get("outcome"):
                    e["outcome"] = (
                        "migrated" if e.get("owner") == "dst" else "drained"
                    )
            d["state"] = "SAFE_TO_STOP"
            d["reserve_tokens"] = 0
            d.setdefault("safe_at", time.time())
            # ``_drain_persist`` merges dicts, so a key must be cleared
            # explicitly; popping it locally would leave the old value on disk.
            d["blocked_reason"] = None
            return
        if blocked:
            # Ownership cannot be determined and resources remain: hold.
            d["state"] = "BLOCKED"
            d["blocked_reason"] = blocked[0][1]
            return
        if d.get("state") == "BLOCKED":
            d["state"] = "RECONCILING"
        if members and all(
            (entries.get(r) or {}).get("phase") == "done" for r in members
        ):
            d["state"] = "RECONCILING"

    def _target_outage_expired(self):
        limit = max(1.0, float(os.environ.get("HANDOFF_DRAIN_TARGET_TIMEOUT_S", "30")))
        since = self._target_fail_since
        return bool(
            not self._target_ok and since is not None and time.monotonic() - since >= limit
        )

    def _fail_drain_target(self, d, entries):
        if (
            d.get("state") in self.DRAIN_DONE
            or d.get("mode") == "natural"
            or not self._target_outage_expired()
        ):
            return False
        unresolved = [entries.get(r) or {} for r in d.get("members") or []]
        if not any(
            e.get("phase") != "done"
            and not e.get("finished")
            and (
                e.get("phase") in ("importing", "imported", "committed")
                or (d.get("wave") or {}).get("phase") in ("importing", "imported")
                or e.get("owner") == "dst"
                or e.get("needs_cancel")
            )
            for e in unresolved
        ):
            return False
        # A pre-import rollback is handled first. After import, an absent
        # response is NOT proof the peer has aborted or stopped reading KV.
        d["state"] = "FAILED_UNSAFE"
        d["failed_reason"] = "target_unreachable_timeout"
        d["failed_at"] = time.time()
        d["blocked_reason"] = "ownership_unconfirmed"
        d["failure_evidence"] = {
            "captured_at": time.time(),
            "target_down_s": round(time.monotonic() - self._target_fail_since, 3),
            "wave": copy.deepcopy(d.get("wave")),
            "source": copy.deepcopy(self._src_facts),
            "target": copy.deepcopy(self._dst_facts),
            "target_last_healthy": copy.deepcopy(getattr(self, "_last_target_facts", {})),
            "members": {r: copy.deepcopy(entries.get(r)) for r in d.get("members") or []},
        }
        self.draining = False  # terminal; durable admission/ownership fence remains
        self._drain_persist(d, entries)
        print(
            f"[gw] drain FAILED_UNSAFE id={d.get('drain_id')} "
            "reason=target_unreachable_timeout; retained ownership and KV",
            flush=True,
        )
        return True

    def _stream_drain_error(self, st):
        with self.journal_lock:
            d = self.journal.get("__drain__") or {}
            member = st.rid in (d.get("members") or [])
            failed = d.get("state") == "FAILED_UNSAFE"
        if (member and failed) or (st.owner == "dst" and self._target_outage_expired()):
            return "target_unreachable_timeout"
        return None

    def _reconcile_loop(self):
        while True:
            # A live drain is latency-critical: every reconcile tick can sit
            # between freeze and import.  Keep the idle cadence at 250 ms but
            # poll fast while draining so the controller does not add a
            # quarter-second to every handoff.
            time.sleep(0.02 if self.draining else 0.25)
            try:
                self._reconcile_once()
            except Exception as exc:
                print(f"[gw] reconcile error: {exc!r}", flush=True)

    def _reconcile_once(self):
        d, entries = self._drain_snapshot()
        if d and d.get("state") in ("STOPPED", "FAILED_UNSAFE"):
            return
        now = time.monotonic()
        if now - self._adopt_at > 1.0:
            self._adopt_at = now
            self._adopt_proxy_requests()
            d, entries = self._drain_snapshot()
        if not d or d.get("state") in ("STOPPED", "FAILED_UNSAFE"):
            return
        if d.get("preflight_pending"):
            self._prepare_drain()
            return
        self._cleanup_drain_reservations(d)
        if d.get("state") == "ROLLED_BACK":
            # Completed members must still be released after a rollback;
            # their streams ended while the drain still looked live.  Refresh
            # facts and sync first so a request that finished during the
            # rollback is marked done before the release sweep.
            self._refresh_facts()
            src_reqs, dst_reqs, states, completed = self._facts_bundle()
            for rid in d.get("members") or []:
                e = entries.get(rid)
                if isinstance(e, dict) and e.get("drain_member"):
                    self._sync_member(
                        e, src_reqs, dst_reqs, states, completed
                    )
            self._drain_release(d, entries)
            self._drain_persist(d, entries)
            return
        self._refresh_facts()
        if not self._target_ok:
            dbg = time.monotonic()
            if dbg - self._target_dbg_t > 2.0:
                self._target_dbg_t = dbg
                age = (
                    None if self._target_fail_since is None
                    else round(dbg - self._target_fail_since, 2)
                )
                print(
                    f"[gw] target_down age={age} "
                    f"state={d.get('state')} "
                    f"wave={(d.get('wave') or {}).get('phase')}",
                    flush=True,
                )
        src_reqs, dst_reqs, states, completed = self._facts_bundle()
        members = list(d.get("members") or [])
        for rid in members:
            e = entries.get(rid)
            if e is None or not e.get("drain_member"):
                continue
            self._sync_member(e, src_reqs, dst_reqs, states, completed)
        if (
            d.get("failed_reason") == "capacity"
            and d.get("state") in self.DRAIN_LIVE
        ):
            self._rollback_capacity(d, entries)
            self._drain_persist(d, entries)
            return
        if (
            d.get("state") in self.DRAIN_LIVE
            and self._precommit_rollback_safe(d, entries)
        ):
            # Target is gone and nothing was imported: the source still owns
            # every KV, so reopen D0 instead of blocking all new traffic.
            self._rollback_target(d, entries)
            self._drain_persist(d, entries)
            return
        if self._fail_drain_target(d, entries):
            return
        blocked = []
        for rid in members:
            e = entries.get(rid)
            if not e or not e.get("needs_cancel"):
                continue
            res = self._cancel_and_resume(e, d)
            if res == "blocked":
                print(
                    f"[gw] cancel blocked rid={rid} "
                    f"target_ok={self._target_ok} "
                    f"phase={e.get('phase')} owner={e.get('owner')} "
                    f"freeze={e.get('freeze_result') is not None} "
                    f"last_error={e.get('last_error')}",
                    flush=True,
                )
                blocked.append((rid, "target_unreachable"))
            else:
                e.pop("needs_cancel", None)
                if res == "dst":
                    e.setdefault("outcome", "migrated")
        def _needs_target_liveness(entry):
            if not entry:
                return False
            phase = entry.get("phase")
            if phase == "done":
                return False
            # A reserved-only member has no target-side request yet.
            if (
                phase == "reserved"
                and entry.get("freeze_result") is None
                and not entry.get("freeze_transfers")
                and entry.get("epoch") in (None, "src")
            ):
                return False
            # Commit/local-ready facts are durable; a later status timeout
            # does not make the transfer unsafe.
            if (
                phase in ("imported", "committed")
                and (
                    entry.get("observed_committed")
                    or entry.get("target_local_done")
                )
            ):
                return False
            return phase in (
                "reserved", "frozen", "importing", "imported",
                "committed",
            ) or entry.get("owner") == "dst"

        if (
            not blocked
            and not self._target_ok
            and self._target_fail_since is not None
            and time.monotonic() - self._target_fail_since > 5.0
            and any(_needs_target_liveness(entries.get(r)) for r in members)
        ):
            # Never issue (or keep waiting on) target RPCs when the target is
            # unreachable; ownership is unknown, so hold the drain.
            blocked.append((members[0] if members else "", "target_unreachable"))
        if not blocked and d.get("state") not in self.DRAIN_DONE:
            self._advance_wave(d, entries, src_reqs, dst_reqs, states)
        self._drain_release(d, entries)
        self._drain_terminal(d, entries, src_reqs, blocked)
        if (
            d.get("restart_at")
            and not d.get("reconciled_at")
            and not blocked
            and self._all_determinate(entries, members)
            and (d.get("wave") or {}).get("phase")
            not in ("freezing", "importing")
        ):
            d["reconciled_at"] = time.time()
        self._drain_persist(d, entries)
        now = time.monotonic()
        if now - getattr(self, "_drain_dbg_t", 0.0) > 2.0:
            self._drain_dbg_t = now
            detail = " ".join(
                f"{rid[-6:]}:{entries[rid].get('phase')}"
                f"/local={entries[rid].get('target_local_done')}"
                f"/rel={entries[rid].get('source_released')}"
                for rid in members if rid in entries
            )
            print(
                f"[gw] drain={d.get('state')} wave="
                f"{(d.get('wave') or {}).get('phase')} "
                f"src={len(src_reqs)} dst={len(dst_reqs)} "
                f"states={len(states)} completed={len(completed)} "
                f"target_ok={self._target_ok} {detail}",
                flush=True,
            )

    def _pre_reserve_drain(self, drain, members):
        """Reserve migration KV + execution slots before any new admission.

        This is the authoritative capacity gate for the new policy.  It runs
        before freeze, so a failure is a clean ROLLED_BACK with no migration
        state and no target request created.  ``reserve`` is idempotent for
        the same drain epoch, which makes controller restart replay-safe.
        """
        epoch = str(drain.get("epoch") or "")
        items = []
        with self.journal_lock:
            for rid in members:
                e = self.journal.get(rid) or {}
                drid = str(e.get("drid") or f"dst-{rid}")
                tokens = (
                    int(e.get("prompt_len") or 0)
                    + int(e.get("max_new") or 0)
                    + 512
                )
                if tokens > 0:
                    items.append({
                        "rid": drid, "tokens": tokens, "slots": 1,
                    })
        if not items:
            return True
        try:
            rv = self.dst.send(
                {"cmd": "reserve", "items": items, "epoch": epoch},
                timeout=15,
            )
        except Exception as exc:
            drain["state"] = "ROLLED_BACK"
            drain["rolled_back_at"] = time.time()
            drain["failed_reason"] = f"reserve_unavailable: {exc!r}"
            drain["reserve_tokens"] = 0
            print(f"[gw] drain pre-reserve failed: {exc!r}", flush=True)
            return False
        errs = self._all_rank_errors(rv)
        if errs:
            rids = [it["rid"] for it in items]
            try:
                self.dst.send(
                    {"cmd": "reserve_cancel", "rids": rids}, timeout=10
                )
            except Exception:
                pass
            why = " ".join(
                f"{k}: {v}" for k, v in list(errs.items())[:4]
            )
            drain["state"] = "ROLLED_BACK"
            drain["rolled_back_at"] = time.time()
            drain["failed_reason"] = f"capacity_slots {why}"
            drain["reserve_tokens"] = 0
            print(
                f"[gw] drain pre-reserve ROLLED_BACK: {why}",
                flush=True,
            )
            return False
        drain["capacity_reserved"] = True
        drain["reserved_epoch"] = epoch
        drain["reserved_items"] = items
        print(
            f"[gw] drain pre-reserve ok members={len(items)} "
            f"capacity={rv.get('capacity')}",
            flush=True,
        )
        return True

    def start_drain(self, targets: list[str], mode: str = "d2d", transfer: str = ""):
        # Arm aliases are accepted at the API boundary; ``mode`` stays the
        # drain policy (d2d/natural) while ``transfer`` selects the data path.
        mode = str(mode or "d2d")
        arm = str(transfer or "").strip()
        if mode in ("copy", "copy_wait", "mooncake"):
            arm = arm or mode
            mode = "d2d"
        elif mode in ("layer", "layer_wait"):
            arm = arm or mode
            mode = "d2d"
        elif mode in ("remote", "remote_progressive"):
            arm = arm or mode
            mode = "d2d"
        if mode == "natural":
            arm = ""
        # Linearization point shared with stream admission: publish the fence
        # and snapshot BEFORE waiting for a prior cutover or any backend RPC.
        with self.admission_lock:
            with self.journal_lock:
                raw = self.journal.get("__drain__")
                existing = copy.deepcopy(raw) if isinstance(raw, dict) else None
            if existing is not None and existing.get("state") in self.DRAIN_FENCED:
                duplicate = True
            else:
                duplicate = False
                members = []
                with self.journal_lock:
                    for rid, e in self.journal.items():
                        if rid == "__drain__" or not isinstance(e, dict):
                            continue
                        if e.get("finished") or e.get("owner", "src") != "src":
                            continue
                        members.append(rid)
                for rid, st in self.active.items():
                    if st.owner == "src" and st.sent < st.max_new and rid not in members:
                        members.append(rid)
                reserve_tokens = 0
                if mode != "natural":
                    for rid in members:
                        e = self.journal.get(rid) or {}
                        reserve_tokens += (
                            int(e.get("prompt_len") or 0) + int(e.get("max_new") or 0) + 64
                        )
                drain_epoch = uuid.uuid4().hex
                drain = {
                    "drain_id": uuid.uuid4().hex,
                    "wave_id": uuid.uuid4().hex,
                    "epoch": drain_epoch,
                    "state": "REQUESTED",
                    "preflight_pending": True,
                    "admission_fenced_at": time.time(),
                    "members": members,
                    "targets": list(targets or ["dst"]),
                    "created_at": time.time(),
                    "mode": mode,
                    "transfer": arm,
                    "reserve_tokens": reserve_tokens,
                    "wave": {
                        "phase": "idle",
                        "members": [],
                        "epoch": drain_epoch,
                    },
                }
                self.draining = True
                self.drain_started_t = time.monotonic()
                self.drain_targets = drain["targets"]
                with self.journal_lock:
                    self.journal["__drain__"] = drain
                self._drain_persist(drain)
        if not duplicate:
            self._prepare_drain()
            self._maybe_drain_kill("created")
        return self.drain_status()

    def _finish_drain_preflight(self, drain):
        drain["preflight_pending"] = False
        drain["preflight_finished_at"] = time.time()
        drain["preflight_ms"] = round(
            (time.time() - drain["admission_fenced_at"]) * 1000, 1
        )
        with self.admission_lock:
            self.draining = drain.get("state") != "ROLLED_BACK"
            if not self.draining:
                drain.setdefault("rolled_back_at", time.time())
            self._drain_persist(drain)
        print(
            f"[gw] drain preflight state={drain['state']} "
            f"members={len(drain['members'])} ms={drain['preflight_ms']}",
            flush=True,
        )

    @_serialize_migration
    def _prepare_drain(self):
        # Idempotent after restart and serialized with every legacy cutover.
        drain, entries = self._drain_snapshot()
        if not drain or not drain.get("preflight_pending"):
            return
        mode = drain.get("mode") or "d2d"
        # A cutover already in progress at the fence may have finished while
        # we waited. It now owns its target reservation; do not reserve twice.
        members = [
            rid
            for rid in drain["members"]
            if not (entries.get(rid) or {}).get("finished")
            and (entries.get(rid) or {}).get("owner", "src") == "src"
        ]
        drain["members"] = members
        drain["preflight_started_at"] = time.time()
        self._drain_persist(drain)
        gate_started = time.monotonic()
        # Drain-wide slot gate: reject the whole drain before freezing
        # anything if the target cannot hold all members on top of its
        # current load.  This is the safe isolation for extreme capacity
        # contention (no target import request is ever created).
        if mode != "natural" and members:
            try:
                st_now = self.dst.send({"cmd": "status"}, timeout=4)
                tps = st_now.get("tp_rank_results") or [st_now]
                max_run = min(int(x.get("max_running") or 0) for x in tps)
                load = max(len(x.get("reqs") or []) for x in tps)
            except Exception as exc:
                max_run, load = 0, 0
                print(f"[gw] drain slot gate skipped: {exc!r}", flush=True)
            # Consecutive C4 runs can start while the previous run's
            # target requests are in their final cleanup.  Give those
            # rows a bounded window to disappear before rejecting the
            # whole drain; this is not a capacity failure.  Ten seconds
            # covers the previous wave's 5 s short-churn cleanup.
            if max_run and load + len(members) > max_run:
                wait_s = float(os.environ.get("HANDOFF_DRAIN_SLOT_WAIT_S", "10"))
                deadline = time.monotonic() + max(0.0, wait_s)
                while (
                    time.monotonic() < deadline
                    and max_run
                    and load + len(members) > max_run
                ):
                    time.sleep(0.1)
                    try:
                        st_now = self.dst.send({"cmd": "status"}, timeout=2)
                        tps = st_now.get("tp_rank_results") or [st_now]
                        max_run = min(int(x.get("max_running") or 0) for x in tps)
                        load = max(len(x.get("reqs") or []) for x in tps)
                    except Exception:
                        break
                print(
                    f"[gw] drain slot gate after wait "
                    f"load={load} members={len(members)} "
                    f"max_run={max_run}",
                    flush=True,
                )
            drain["slot_gate"] = {
                "load": load,
                "members": len(members),
                "max_running": max_run,
                "checked_at": time.time(),
                "elapsed_ms": round((time.monotonic() - gate_started) * 1000, 1),
            }
            if max_run and load + len(members) > max_run:
                drain["state"] = "ROLLED_BACK"
                drain["failed_reason"] = (
                    f"capacity_slots load={load} + "
                    f"members={len(members)} > max_running={max_run}"
                )
                drain["reserve_tokens"] = 0
                self._finish_drain_preflight(drain)
                print(
                    f"[gw] drain rejected before freeze: {drain['failed_reason']}",
                    flush=True,
                )
                return
        drain["slot_gate"] = {
            "load": load if mode != "natural" and members else 0,
            "members": len(members),
            "max_running": max_run if mode != "natural" and members else 0,
            "checked_at": time.time(),
            "elapsed_ms": round((time.monotonic() - gate_started) * 1000, 1),
        }
        if mode != "natural" and members:
            if not self._pre_reserve_drain(drain, members):
                self._finish_drain_preflight(drain)
                return
        with self.journal_lock:
            for rid in members:
                e = self.journal.get(rid)
                if isinstance(e, dict):
                    e["drain_id"] = drain["drain_id"]
                    e["wave_id"] = drain["wave_id"]
                    e["drain_epoch"] = drain["epoch"]
                    e["drain_member"] = True
        self.drain_kv_baseline = int(
            getattr(self, "_src_facts", {}).get("kv_available") or 0
        )
        self._finish_drain_preflight(drain)

    def stop_drain(self):
        d, entries = self._drain_snapshot()
        if not d:
            return {"state": "NONE"}
        if d.get("state") != "SAFE_TO_STOP":
            return {"error": "not_safe_to_stop", "state": d.get("state")}
        d["state"] = "STOPPED"
        d["stopped_at"] = time.time()
        d["reserve_tokens"] = 0
        self._drain_persist(d, entries)
        self.draining = False
        return self.drain_status()

    def drain_status(self):
        d, entries = self._drain_snapshot()
        if not d:
            return {"draining": False, "state": "NONE"}
        members = list(d.get("members") or [])
        rows = {rid: entries.get(rid, {}) for rid in members}
        pending = [
            rid for rid, e in rows.items()
            if e.get("owner", "src") == "src"
            and e.get("phase") != "done"
            and not e.get("migration_abandoned")
        ]
        migrated = [rid for rid, e in rows.items()
                    if e.get("owner") == "dst"]
        finished = [rid for rid, e in rows.items()
                    if e.get("phase") == "done"]
        failed = [st.rid for st in self.active.values() if st.drain_failed]
        now = time.time()
        created = float(d.get("created_at") or now)
        elapsed = (now - created) * 1000.0
        restart = d.get("restart_at")
        reconciled = d.get("reconciled_at")
        recovery_ms = None
        if restart:
            recovery_ms = ((reconciled or now) - float(restart)) * 1000.0
        src_kv = int(self._src_facts.get("kv_available", -1))
        return {
            "terminal": d.get("state") in self.DRAIN_DONE + ("ROLLED_BACK", "FAILED_UNSAFE"),
            "requires_intervention": d.get("state") == "FAILED_UNSAFE",
            "admission_fenced_at": d.get("admission_fenced_at"),
            "preflight_pending": bool(d.get("preflight_pending")),
            "preflight_ms": d.get("preflight_ms"),
            "preflight_started_at": d.get("preflight_started_at"),
            "preflight_finished_at": d.get("preflight_finished_at"),
            "failed_at": d.get("failed_at"),
            "slot_gate": d.get("slot_gate"),
            "failure_evidence": d.get("failure_evidence"),
            "draining": d.get("state") in self.DRAIN_LIVE,
            "state": d.get("state"),
            "drain_id": d.get("drain_id"),
            "wave_id": d.get("wave_id"),
            "wave_phase": (d.get("wave") or {}).get("phase"),
            "wave": d.get("wave"),
            "mode": d.get("mode") or "d2d",
            "transfer": d.get("transfer") or "",
            "reserve_tokens": int(d.get("reserve_tokens") or 0),
            "worker_state": self.src_worker_state(),
            "dst_worker_state": self.dst_worker_state(),
            "rolled_back_at": d.get("rolled_back_at"),
            "failed_reason": d.get("failed_reason"),
            "wave_members": list((d.get("wave") or {}).get("members") or []),
            "drain_ms": round(elapsed, 1),
            "recovery_ms": (
                None if recovery_ms is None else round(recovery_ms, 1)
            ),
            "created_at": d.get("created_at"),
            "restart_at": d.get("restart_at"),
            "reconciled_at": d.get("reconciled_at"),
            "safe_at": d.get("safe_at"),
            "stopped_at": d.get("stopped_at"),
            "members": members,
            "pending": pending,
            "migrated": migrated,
            "finished": finished,
            "outcomes": {
                rid: rows[rid].get("outcome") for rid in members
            },
            "failed": failed,
            "blocked_reason": d.get("blocked_reason"),
            "last_error": d.get("last_error"),
            "source_safe_to_stop": d.get("state") in self.DRAIN_DONE,
            "source_kv_available": src_kv,
            "source_ranks": getattr(self, "_src_clean_ranks", []),
            "source_clean_all_ranks": bool(
                getattr(self, "_src_clean_all", False)
            ),
            "target_capacity": [
                {
                    "rank": int(rr.get("tp_rank") or idx),
                    **(rr.get("capacity") or {}),
                }
                for idx, rr in enumerate(
                    self._rank_results(self._dst_facts)
                )
            ],
            "target_reserved": {
                rid: {
                    "tokens": int(v.get("tokens", 0)),
                    "slots": int(v.get("slots", 0)),
                    "slot_consumed": bool(v.get("slot_consumed")),
                    "kv_consumed": bool(v.get("kv_consumed")),
                }
                for rid, v in (
                    (self._dst_facts.get("tp_rank_results") or [
                        self._dst_facts
                    ])[0].get("reserved") or {}
                ).items()
            },
            "member_detail": {
                rid: {
                    "owner": rows[rid].get("owner"),
                    "phase": rows[rid].get("phase"),
                    "outcome": rows[rid].get("outcome"),
                    "prefix_len": rows[rid].get("prefix_len"),
                    "target_local_done": rows[rid].get("target_local_done"),
                    "source_released": rows[rid].get("source_released"),
                    "source_release_ranks": rows[rid].get(
                        "source_release_ranks"
                    ),
                    "source_release_deferred": rows[rid].get(
                        "source_release_deferred"
                    ),
                    "target_released": rows[rid].get("target_released"),
                    "target_release_ranks": rows[rid].get(
                        "target_release_ranks"
                    ),
                    "resume_result": rows[rid].get("resume_result"),
                    "last_error": rows[rid].get("last_error"),
                }
                for rid in members
            },
            "targets": d.get("targets"),
        }

    def _recover_owner(self, entry: dict) -> str:
        """Owner for a reconnecting stream.  Never performs migration RPCs."""
        rid = entry["rid"]
        drid = entry.get("drid") or f"dst-{rid}"
        with self.journal_lock:
            raw = self.journal.get("__drain__")
            drain = copy.deepcopy(raw) if isinstance(raw, dict) else None
        if drain is not None:
            state = drain.get("state")
            if entry.get("phase") == "done" or entry.get("finished"):
                return "done"
            if state == "BLOCKED":
                # Unknown ownership: stream neither side, but keep the client
                # connection alive until reconcile resolves.
                return "wait"
            return "dst" if entry.get("owner") == "dst" else "src"
        src_item = self.src.get(rid)
        dst_item = self.dst.get(drid)
        dst_state = self.dst.targets.get(rid)
        if dst_item is not None or dst_state is not None:
            try:
                ab = self.dst.send(
                    {"cmd": "abort_import", "rid": drid}, timeout=120
                )
            except Exception:
                return "ambiguous"
            if ab.get("committed"):
                return "dst"
            # Target confirmed it has no committed token; it is now aborted.
            try:
                self.src.send({"cmd": "resume", "rid": rid}, timeout=60)
            except Exception:
                pass
            return "src"
        if src_item is not None:
            if entry.get("phase") in ("reserved", "frozen"):
                try:
                    self.src.send({"cmd": "resume", "rid": rid}, timeout=60)
                except Exception:
                    pass
            return "src"
        if entry.get("phase") in ("imported", "commit") or entry.get(
            "owner"
        ) == "dst":
            return "ambiguous"
        return "done"

    def _attach_finish(self, handler, st, last, owner):
        self._journal_update(st, force=True, sent=last, finished=True)
        try:
            self._release_every_rank(self.src, st.rid, timeout=10)
        except Exception:
            pass
        if owner == "dst":
            try:
                self._release_every_rank(self.dst, st.drid, timeout=10)
            except Exception:
                pass
        self._send_event(handler, st, {
            "done": True, "metrics": self._metrics(st),
        })
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()

    def _attach_stream(self, handler, st: RequestState, entry: dict,
                       after: int):
        """Continue an adopted stream from the journal's authoritative owner.

        This path never resumes, aborts or releases migration state; the
        reconcile loop owns all of that.
        """
        last = int(after)
        started = time.monotonic()
        last_keepalive = started
        jt = 0.0
        st._j_owner = entry.get("owner", "src")
        st._j_phase = entry.get("phase", "accepted")
        st._j_done = bool(entry.get("finished") or st._j_phase == "done")
        st._j_state = None
        while True:
            drain_error = self._stream_drain_error(st)
            if drain_error:
                st.error = drain_error
                self._send_event(handler, st, {"error": drain_error})
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
                return
            now = time.monotonic()
            if now - jt > 0.25:
                jt = now
                with self.journal_lock:
                    jraw = self.journal.get(st.rid)
                    jentry = copy.deepcopy(jraw) if isinstance(jraw, dict) else {}
                    draw = self.journal.get("__drain__")
                    jdrain = copy.deepcopy(draw) if isinstance(draw, dict) else {}
                st._j_owner = jentry.get("owner") or st._j_owner
                st._j_phase = jentry.get("phase") or st._j_phase
                st._j_state = jdrain.get("state")
                st._j_done = bool(
                    jentry.get("finished") or st._j_phase == "done"
                )
            if st.switch_ready:
                st._j_owner = "dst"
                st.switch_ready = False
            if time.monotonic() - started > 900:
                self._send_event(handler, st, {
                    "error": "attach_timeout",
                    "metrics": self._metrics(st),
                })
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
                return
            owner = "dst" if st._j_owner == "dst" else "src"
            role = self.dst if owner == "dst" else self.src
            key = st.drid if owner == "dst" else st.rid
            wait_s = (
                0.02
                if (owner == "src" and st.freeze_t is not None)
                else 0.2
            )
            item = role.wait_output(key, last, wait_s)
            if item is None:
                if time.monotonic() - last_keepalive > 0.5:
                    self._keepalive(handler)
                    last_keepalive = time.monotonic()
                if st._j_done and role.get(key) is None:
                    self._attach_finish(handler, st, last, owner)
                    return
                continue
            ids, finished, _kv = item
            for seq in range(last + 1, len(ids)):
                self._send_event(handler, st, {
                    "seq": seq,
                    "token_id": int(ids[seq]),
                    "owner": owner,
                    "epoch": entry.get("epoch"),
                    "t": time.monotonic(),
                })
                last = seq
            if last > int(entry.get("sent", 0)):
                entry["sent"] = last
                self._journal_update(st, force=True, sent=last)
            if finished:
                self._attach_finish(handler, st, last, owner)
                return

    def attach(self, handler, rid: str, after: int):
        with self.journal_lock:
            rec = self.journal.get(str(rid))
            entry = dict(rec) if isinstance(rec, dict) else None
        if entry is None:
            self._json_response(handler, 404, {"error": "unknown rid"})
            return
        if entry.get("finished"):
            self._sse_headers(handler)
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
            return
        with self.journal_lock:
            draw = self.journal.get("__drain__")
            drain_live = (
                isinstance(draw, dict)
                and draw.get("state") not in self.DRAIN_DONE
            )
        if not drain_live:
            # Outside a drain the old ownership fence still applies: a target
            # that committed wins, otherwise abort+resume before streaming.
            owner = self._recover_owner(entry)
            if owner == "ambiguous":
                self._json_response(handler, 409, {
                    "error": "ownership_ambiguous", "rid": rid,
                })
                return
            if owner == "done":
                self._sse_headers(handler)
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
                return
            if owner == "dst":
                entry["owner"] = "dst"
        self._sse_headers(handler)
        st = self._state_from_entry(entry)
        st.sent = max(int(entry.get("sent") or 0), int(after) + 1)
        with self.journal_lock:
            rec = self.journal.get(str(rid))
            if isinstance(rec, dict):
                rec["sent"] = max(int(rec.get("sent") or 0), int(after) + 1)
        self.active[st.rid] = st
        try:
            self._attach_stream(handler, st, entry, int(after))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.active.pop(st.rid, None)

    def _sse_headers(self, handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.flush()

    def _json_response(self, handler, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def stream(self, handler, body: dict):
        # Router admission: a DRAINING/SAFE source admits no new requests;
        # they go to the other READY worker instead of a service-wide 503.
        # Tokenization happens before the critical section; the decision,
        # journal entry and active registration are one atomic step w.r.t.
        # start_drain, so a request is either in the drain member snapshot or
        # it observes DRAINING.
        prompt_ids = body.get("prompt_ids")
        if not prompt_ids:
            prompt_ids = TOK(body["prompt"], add_special_tokens=False)["input_ids"]
        max_new = int(body.get("max_new_tokens", 128))
        cut = int(body.get("cut", 32))
        transfer = str(body.get("transfer", "remote"))
        inject = str(body.get("inject", "none"))
        rid = str(body.get("rid") or f"gw-{uuid.uuid4().hex[:12]}")
        forced_hint = str(body.get("worker") or "")
        # Wait OUTSIDE admission_lock: the reconcile's terminal check also
        # takes this lock, and a long hold here would freeze the wave (the
        # source transfer listeners expire after 180 s, then import gets
        # connection refused).  The hold is a journal-only wait, so concurrent
        # waiters are cheap.
        need = len(prompt_ids) + max_new + 64
        # Strict-PD source requests are native decode requests: D0 physically
        # allocates the prompt KV during prealloc, so only the growth budget
        # should be charged against the D0 ordinary lease.
        admit_need = (
            max_new + 64
            if (PD_PREFILL_URL and forced_hint == "src")
            else need
        )
        target_rid = str(body.get("drid") or f"dst-{rid}")
        ticket_info = self._admission_capacity_wait(
            admit_need, forced_hint, target_rid
        )
        if isinstance(ticket_info, dict) and ticket_info.get("error"):
            self._json_response(handler, 503, {
                "error": ticket_info.get("error"),
                "reason": ticket_info.get("reason"),
                "rid": rid,
            })
            return
        capacity_ticket = (
            ticket_info.get("ticket")
            if isinstance(ticket_info, dict) else None
        )
        capacity_epoch = (
            ticket_info.get("epoch", "")
            if isinstance(ticket_info, dict) else ""
        )
        while True:
            with self.admission_lock:
                # Recheck under the SAME lock as the drain fence. A drain can
                # start after _admission_capacity_wait returned no ticket.
                use_dst = forced_hint == "dst" or (
                    forced_hint != "src" and self.src_worker_state() != "READY"
                )
                if use_dst and not capacity_ticket and self.dst_worker_state() == "READY":
                    need_ticket = True
                else:
                    need_ticket = False
                    forced = forced_hint
                    src_state = self.src_worker_state()
                    dst_state = self.dst_worker_state()
                    if rid in self.active:
                        # A retry must not create a second source/target submission
                        # under one rid; the proxy reattaches completed streams via
                        # GET /v1/attach instead of re-posting.
                        self._json_response(handler, 409, {
                            "error": "duplicate_rid", "rid": rid,
                        })
                        return
                    if forced == "src" and src_state != "READY":
                        if capacity_ticket:
                            self._cancel_capacity_ticket(capacity_ticket)
                        self._json_response(handler, 503, {
                            "error": "worker_not_ready", "worker": "src",
                            "state": src_state,
                        })
                        return
                    use_dst = forced == "dst" or (
                        forced != "src" and src_state != "READY"
                    )
                    if use_dst and dst_state != "READY":
                        if capacity_ticket:
                            self._cancel_capacity_ticket(capacity_ticket)
                        self._json_response(handler, 503, {
                            "error": "no_ready_worker",
                            "src": src_state, "dst": dst_state,
                        })
                        return
                    if not use_dst and src_state != "READY":
                        if capacity_ticket:
                            self._cancel_capacity_ticket(capacity_ticket)
                        self._json_response(handler, 503, {
                            "error": "no_ready_worker",
                            "src": src_state, "dst": dst_state,
                        })
                        return
                    if capacity_ticket and not use_dst:
                        self._cancel_capacity_ticket(capacity_ticket)
                        capacity_ticket = None
                    st = RequestState(rid, len(prompt_ids), max_new, cut)
                    st.group = body.get("group")
                    if st.group:
                        st.group = f"{st.group}:{body.get('run_id', '')}"
                    st.group_size = int(body.get("group_size", 1))
                    st.wave = bool(body.get("wave", WAVE_DEFAULT))
                    st.wave_freeze = str(body.get("wave_freeze", "each"))
                    st._capacity_ticket = capacity_ticket
                    st._capacity_epoch = capacity_epoch
                    with self.journal_lock:
                        admission_drain = dict(self.journal.get("__drain__") or {})
                    admission = {
                        "admitted_at": time.time(),
                        "admission_worker": "dst" if use_dst else "src",
                        "admission_drain_id": admission_drain.get("drain_id"),
                        "admission_drain_state": admission_drain.get("state"),
                    }
                    if use_dst:
                        # Direct-to-D1 admission: no handoff, journal owner is the
                        # target from the first token so attach/recovery need no
                        # migration state.
                        st.drid = str(body.get("drid") or f"dst-{rid}")
                        st.owner = "dst"
                        st.owner_epoch = "direct"
                        st.transfer = "direct"
                        self._journal_update(
                            st, force=True, phase="direct", owner="dst",
                            epoch="direct", direct=True, max_new=max_new, cut=cut,
                            transfer="direct", prompt_len=len(prompt_ids),
                            finished=False, **admission,
                        )
                    else:
                        self._journal_update(
                            st, force=True, phase="accepted", max_new=max_new, cut=cut,
                            transfer=transfer, prompt_len=len(prompt_ids),
                            finished=False, **admission,
                        )
                    self.active[st.rid] = st
                    if use_dst:
                        st._admit_need = need
                        self._admit_pending += need
                    if st.group:
                        # Register every member up front so a failed wave can abort
                        # members that have not reached their cut yet.
                        with self.group_lock:
                            g = self.groups.setdefault(
                                st.group,
                                {"members": [], "expected": st.group_size,
                                 "started": False, "all": [], "aborted": None},
                            )
                            g.setdefault("all", []).append(st)
                    break
            if need_ticket:
                ti = self._target_ticket_wait(need, target_rid)
                if ti.get("error"):
                    self._json_response(handler, 503, {
                        "error": ti.get("error"),
                        "reason": ti.get("reason"),
                        "rid": rid,
                    })
                    return
                capacity_ticket = ti.get("ticket")
                capacity_epoch = ti.get("epoch", "")

        handler.wfile.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: close\r\n\r\n"
        )
        handler.wfile.flush()

        try:
            submit_role = self.dst if use_dst else self.src
            submit_key = st.drid if use_dst else rid
            submit_payload = {
                "cmd": "submit",
                "rid": submit_key,
                "input_ids": prompt_ids,
                "max_new_tokens": max_new,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "ignore_eos": True,
            }
            if use_dst and capacity_ticket:
                submit_payload.update({
                    "capacity_ticket": capacity_ticket,
                    "capacity_epoch": capacity_epoch,
                    "capacity_tokens": need,
                })
            if PD_PREFILL_URL:
                # Strict PD: send P first so it is already waiting when D
                # registers the receiver and sends its metadata.  Sending P
                # after the D prealloc pass can lose the metadata race.
                room = uuid.uuid4().int & ((1 << 63) - 1)
                submit_payload.update({
                    "cmd": "submit_pd",
                    "prefill_url": PD_PREFILL_URL,
                    "bootstrap_host": PD_BOOTSTRAP_HOST,
                    "bootstrap_port": PD_BOOTSTRAP_PORT,
                    "bootstrap_room": room,
                })
                threading.Thread(
                    target=_pd_prefill_request,
                    args=(submit_key, prompt_ids, room),
                    daemon=True,
                ).start()
                _t_sub = time.monotonic()
                submitted = submit_role.send(submit_payload, timeout=120)
                print(
                    f"[gw] submit_pd rid={submit_key} role={submit_role.role} "
                    f"wait_ms={(time.monotonic() - _t_sub) * 1000:.1f} "
                    f"room={room} queues={submitted.get('queues')}",
                    flush=True,
                )
                got_room = int(submitted.get("bootstrap_room") or 0)
                if got_room and got_room != room:
                    raise RuntimeError(
                        f"submit_pd room mismatch: sent {room}, got {got_room}"
                    )
            else:
                submit_role.send(submit_payload, timeout=120)
            st.cutover["submit_t"] = time.monotonic()
            cut_done = False
            last_keepalive = time.monotonic()
            while st.sent < max_new:
                drain_error = self._stream_drain_error(st)
                if drain_error:
                    st.drain_failed = True
                    raise RuntimeError(drain_error)
                if (
                    getattr(st, "drain_requested", False)
                    and st.owner == "src"
                    and not cut_done
                    and st.sent >= 1
                ):
                    cut_done = True
                    st.cut = max(32, st.sent)
                    self._keepalive(handler)
                    if st.group and st.wave:
                        ok = self._wave_freeze(handler, st, transfer, inject)
                        if ok:
                            self._join_group(st, transfer, inject)
                    else:
                        ok = self._cutover(handler, st, transfer, inject)
                    st.drain_requested = False
                    if not ok:
                        st.drain_failed = True
                        st.error = st.error or "drain cutover failed"
                if st.switch_ready:
                    with self.journal_lock:
                        jraw = self.journal.get(st.rid)
                        jentry = copy.deepcopy(jraw) if isinstance(jraw, dict) else {}
                    if jentry.get("prefix_len") is not None:
                        st.prefix_len = int(jentry.get("prefix_len", 0))
                        st.cutover["prefix_len"] = st.prefix_len
                    deadline = time.monotonic() + 5.0
                    item = self.src.get(st.rid)
                    while (
                        item is None or len(item[0]) < st.prefix_len
                    ) and time.monotonic() < deadline:
                        self.src.wait_output(st.rid, st.prefix_len - 1, 0.05)
                        item = self.src.get(st.rid)
                    if item is not None:
                        self._drain_source(
                            handler, st, item[0][: st.prefix_len], st.sent
                        )
                    st.sent = st.prefix_len
                    st.owner = "dst"
                    st.switch_ready = False
                role = self.src if st.owner == "src" else self.dst
                # Once the source is frozen it will not emit another token;
                # a 200 ms wait here delays noticing switch_ready by up to
                # 200 ms.  Poll fast on the frozen source, keep 200 ms for a
                # live stream.
                wait_s = (
                    0.02
                    if (st.owner == "src" and st.freeze_t is not None)
                    else 0.2
                )
                item = role.wait_output(
                    st.rid if st.owner == "src" else st.drid,
                    st.sent if st.owner == "src" else max(st.sent, st.cutover.get("prefix_len", 0)),
                    wait_s,
                )
                if item is None:
                    if time.monotonic() - last_keepalive > 0.5:
                        self._keepalive(handler)
                        last_keepalive = time.monotonic()
                    continue
                ids, finished, _kv = item
                if st.owner == "src":
                    backlog = max(0, len(ids) - st.sent)
                    st.max_backlog = max(st.max_backlog, backlog)
                    new = len(ids) - st.sent
                    for seq in range(st.sent, len(ids)):
                        self._send_event(handler, st, self._event(st, seq, ids[seq], "src"))
                else:
                    prefix = int(
                        st.prefix_len
                        or st.cutover.get("prefix_len", 0)
                    )
                    start = max(st.sent, prefix)
                    if len(ids) > start:
                        st.max_backlog = max(st.max_backlog, len(ids) - start)
                        for seq in range(start, len(ids)):
                            first = st.first_target_t is None
                            self._send_event(handler, st, self._event(st, seq, ids[seq], "dst"))
                            if first:
                                # The client stream itself is commit evidence:
                                # the target produced a new token.  Keep it in
                                # the journal even if the target state is
                                # released/evicted under load.
                                with self.journal_lock:
                                    _rec = self.journal.get(st.rid)
                                    if isinstance(_rec, dict) and not _rec.get(
                                        "observed_committed"
                                    ):
                                        _rec["observed_committed"] = True
                                        _rec["committed_at"] = time.time()
                                self._maybe_kill("commit")
                if finished and (st.owner == "dst" or st.sent >= max_new):
                    break
                if not cut_done and st.owner == "src" and st.sent >= cut:
                    cut_done = True
                    print(
                        f"[gw] cut rid={rid} sent={st.sent} group={st.group}",
                        flush=True,
                    )
                    if st.group:
                        if st.wave and st.wave_freeze != "all":
                            if self._wave_freeze(handler, st, transfer, inject):
                                self._join_group(st, transfer, inject)
                        else:
                            self._join_group(st, transfer, inject)
                    else:
                        self._keepalive(handler)
                        ok = self._cutover(handler, st, transfer, inject)
                        if not ok and st.error == "target_committed":
                            self._send_event(handler, st, {"error": st.error, "metrics": self._metrics(st)})
                            break
                if finished and st.owner == "src":
                    # Natural finish before cut.
                    break
            self._journal_update(
                st, force=True, finished=True, sent=st.sent
            )
            self.active.pop(st.rid, None)
            self._send_event(handler, st, {"done": True, "metrics": self._metrics(st)})
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            st.error = "client_disconnect"
        except Exception as exc:
            st.error = repr(exc)
            try:
                self._send_event(handler, st, {"error": st.error, "metrics": self._metrics(st)})
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
            except Exception:
                pass
        finally:
            st.release_t = time.monotonic()
            with self.journal_lock:
                jrec = dict(self.journal.get(st.rid) or {})
                draw = self.journal.get("__drain__")
                jstate = draw.get("state") if isinstance(draw, dict) else None
            drain_live = jstate in (
                "REQUESTED", "DRAINING", "PREPARING", "MIGRATING",
                "RECONCILING", "BLOCKED", "FAILED_UNSAFE",
            )
            # A direct-to-D1 request never touches the source; release only
            # its target-side tracking, independent of any drain.
            if jrec.get("direct") and self._target_ok:
                try:
                    self.dst.send(
                        {"cmd": "release", "rid": st.drid}, timeout=60
                    )
                except Exception:
                    pass
            # During a live drain the reconcile loop owns every migration
            # decision: aborting an in-flight import or freeing a frozen
            # source KV here would race the wave.
            elif not jrec.get("direct") and not drain_live:
                # Release the source first: it is local and must not be held
                # behind a dead target's RPC timeout.  Only touch the target
                # when it is reachable (target failure leaves nothing to free).
                if st.owner == "src":
                    try:
                        self.src.send(
                            {"cmd": "release", "rid": st.rid}, timeout=10
                        )
                    except Exception:
                        pass
                if self._target_ok:
                    try:
                        self.dst.send(
                            {"cmd": "abort_import", "rid": st.drid},
                            timeout=10,
                        )
                    except Exception:
                        pass
                    try:
                        self.dst.send(
                            {"cmd": "release", "rid": st.drid}, timeout=10
                        )
                    except Exception:
                        pass
            self.active.pop(st.rid, None)
            need = int(getattr(st, "_admit_need", 0) or 0)
            if need:
                with self.admission_lock:
                    self._admit_pending = max(
                        0, self._admit_pending - need
                    )
            self._write_metrics(st)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gateway: Gateway = None

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/v1/attach":
            qs = parse_qs(u.query)
            rid = (qs.get("rid") or [""])[0]
            try:
                after = int((qs.get("after") or ["-1"])[0])
            except ValueError:
                after = -1
            self.gateway.attach(self, rid, after)
            return
        if u.path == "/v1/drain/status":
            self.gateway._json_response(self, 200, self.gateway.drain_status())
            return
        if u.path == "/v1/router/status":
            self.gateway._json_response(self, 200, {
                "src": self.gateway.src_worker_state(),
                "dst": self.gateway.dst_worker_state(),
                "drain": self.gateway.drain_status(),
            })
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/v1/drain":
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            out = self.gateway.start_drain(
                body.get("targets") or ["dst"],
                body.get("mode") or "d2d",
                body.get("transfer") or "",
            )
            self.gateway._json_response(self, 200, out)
            return
        if self.path == "/v1/drain/stop":
            out = self.gateway.stop_drain()
            code = 409 if out.get("error") else 200
            self.gateway._json_response(self, code, out)
            return
        if self.path == "/v1/background":
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            body.setdefault("rid", f"bg-{uuid.uuid4().hex[:10]}")
            try:
                res = self.gateway.dst.send(
                    {
                        "cmd": "submit",
                        "rid": body["rid"],
                        "input_ids": body["prompt_ids"],
                        "max_new_tokens": int(body.get("max_new_tokens", 256)),
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "top_k": 1,
                        "ignore_eos": True,
                    },
                    timeout=120,
                )
                out = {"ok": True, "rid": body["rid"], "result": res}
            except Exception as exc:
                out = {"ok": False, "error": repr(exc)}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path == "/v1/release":
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                res = self.gateway.dst.send(
                    {"cmd": "release", "rid": body["rid"]}, timeout=60
                )
                out = {"ok": True, "result": res}
            except Exception as exc:
                out = {"ok": False, "error": repr(exc)}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if self.path not in ("/v1/completions", "/generate"):
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        self.gateway.stream(self, body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=31080)
    p.add_argument("--poll-ms", type=float, default=3.0)
    p.add_argument("--out", default=str(STATE / "p4_gateway.jsonl"))
    args = p.parse_args()
    gw = Gateway(Path(args.out), args.poll_ms)
    Handler.gateway = gw
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        f"p4-gateway listening 127.0.0.1:{args.port} "
        f"kill_drain_at={DRAIN_KILL_AT or '-'} "
        f"kill_at={KILL_AT or '-'} pid={os.getpid()}",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()

          
        
       
