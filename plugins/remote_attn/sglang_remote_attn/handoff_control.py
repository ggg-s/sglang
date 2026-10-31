"""Minimal file-based handoff control for the 5090 Copy-and-Resume experiment.

Each scheduler (TP rank) polls ``$HANDOFF_CTL_DIR/cmd.<rank>.json`` at the top of
its event loop (a safe boundary in non-overlap mode), executes one command, and
writes ``result.<rank>.json``.  With a single rank the files are
``cmd.0.json``/``result.0.json``; the driver fans a command out to every rank.
At startup each rank publishes ``ready.<rank>.json`` (rank/tp_size/pid) so the
driver can discover the TP width.

Commands:
  submit  {rid, input_ids, max_new_tokens, temperature, top_p, top_k, ignore_eos}
  freeze  {rid, transfer:"socket"|"file", out?}
  import  {transfer:{host,port,token}, rid?, remaining?}  or  {state:"...pt"}
  status  {out?}
  release {rid}

KV transport:
  * socket: freeze exports KV to CPU and starts a one-shot authenticated TCP
    server on 127.0.0.1; import connects and streams tensors into its own slots.
  * file:   torch.save/load on a shared filesystem (legacy debug path).

This is a test control surface, not a production API.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import http.client
import json
import os
import queue
import secrets
import socket
import struct
import threading
import time
import traceback

try:
    import fcntl
except ImportError:  # pragma: no cover - scheduler runs on Linux
    fcntl = None
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import torch

from .compat import bind_decode_prefix, snapshot_layout

_CTL_DIR = os.environ.get("HANDOFF_CTL_DIR", "")
_SEQ = 0
HANDOFF_PREFIX_CLEARED = 0
# P5.2-D control-plane timeline: one id per scheduler iteration.  Rank 0
# increments it once per recv_requests()/poll cycle, which is the granularity
# at which a control command can be claimed.
_ITER = 0
_ITER_AT = time.monotonic()
_ITER_PERIOD_MS = 0.0
_ITER_PROFILE_FH = None
_ITER_PROFILE_N = 0


def _iter_profile(scheduler) -> None:
    """Optional per-scheduler-iteration trace for P5.9 C4 gap analysis."""
    global _ITER_PROFILE_FH, _ITER_PROFILE_N
    root = os.environ.get("HANDOFF_ITER_PROFILE_DIR", "").strip()
    if not root:
        return
    try:
        if _ITER_PROFILE_FH is None:
            Path(root).mkdir(parents=True, exist_ok=True)
            rank = _tp_rank(scheduler)
            _ITER_PROFILE_FH = open(Path(root) / f"iter.{rank}.jsonl", "a", buffering=1)
        rb = getattr(scheduler, "running_batch", None)
        lb = getattr(scheduler, "last_batch", None)
        wq = getattr(scheduler, "waiting_queue", [])

        def _mode(b):
            if b is None:
                return None
            fm = getattr(b, "forward_mode", None)
            return str(fm) if fm is not None else None

        rec = {
            "t": round(time.monotonic(), 6),
            "iter": int(_ITER),
            "dt_ms": round(float(_ITER_PERIOD_MS), 3),
            "running": len(getattr(rb, "reqs", []) or []),
            "running_mode": _mode(rb),
            "running_seqs": [
                int(getattr(r, "seqlen", 0))
                for r in (getattr(rb, "reqs", []) or [])[:8]
            ],
            "last": len(getattr(lb, "reqs", []) or []),
            "last_mode": _mode(lb),
            "waiting": len(wq),
            "waiting_lens": [len(getattr(r, "origin_input_ids", [])) for r in wq[:8]],
            "kv_available": _kv_available(scheduler),
            "pdi": int(getattr(scheduler, "prefill_decode_interval", 0) or 0),
            "pdi_remaining": int(
                getattr(scheduler, "_prefill_decode_interval_remaining", 0) or 0
            ),
        }
        _ITER_PROFILE_FH.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _ITER_PROFILE_N += 1
        if _ITER_PROFILE_N % 100 == 0:
            _ITER_PROFILE_FH.flush()
    except Exception:
        pass


# P5.4-A: epochs cancelled by the controller after a failed/timed-out wave.
# A late-arriving import with the same (rid, epoch) must not create a target
# request after the source has been resumed.
_CANCELLED_IMPORTS: dict[tuple[str, str], float] = {}
_CANCEL_TTL_S = 120.0

# P5.4-B0 source frozen watchdog.  The controller writes heartbeat.json into
# this role's ctl dir.  If it disappears, the source decides from the target's
# own imported/commit/done reports (delivered over the RemoteHub):
#   no import report  -> safe resume
#   imported          -> ambiguous, hold lease and never resume
#   committed         -> target owns output, never resume
#   done              -> target no longer reads, release the lease
_FROZEN_WD = None
_FROZEN_WD_LOCK = threading.Lock()
_FROZEN_WD_SEEN_HB = False
_FROZEN_WD_TIMEOUT_S = float(os.environ.get("HANDOFF_FROZEN_HEARTBEAT_S", "2.0"))
_FROZEN_WD_ACTIONS: dict[str, tuple[str, Any]] = {}
_FROZEN_WD_HANDLED: set[tuple[str, int]] = set()
_FROZEN_WD_LOGGED: set[tuple[str, int]] = set()
# Last target-fence outcome per rid, so an unreachable target is not logged
# on every scheduler iteration.
_FROZEN_WD_FENCE_STATE: dict[str, str] = {}
# A request that finishes while the controller is down still owns the only
# copy of its output.  Keep it long enough for a restarted controller (or an
# SSE attach) to stream the tail before the watchdog frees it.
_FINISHED_SEEN: dict[str, float] = {}
_FINISHED_RETAIN_S = float(os.environ.get("HANDOFF_FINISHED_RETAIN_S", "120"))
_RESV_HB_SEEN = False
# Finished target states are kept this long after controller loss so a
# restarted controller can stream the remaining output before releasing.
_DEAD_STATE_GRACE_S = float(
    os.environ.get("HANDOFF_DEAD_CONTROLLER_STATE_GRACE_S", "60")
)


def _expire_target_states(scheduler) -> None:
    """Drop target handoff states whose request finished after controller loss."""
    hb_path = _heartbeat_path()
    if hb_path is None or not hb_path.exists():
        return
    try:
        hb = json.loads(hb_path.read_text(encoding="utf-8"))
        ts = float(hb.get("ts", 0.0))
    except Exception:
        return
    if time.time() - ts < _FROZEN_WD_TIMEOUT_S:
        return
    for st in list(_TARGET_STATES):
        req = getattr(st, "req", None)
        if req is not None and getattr(req, "_pre_layerwise_target", None) is not None:
            # Native PD cancellation/reclamation fences its own receiver. The
            # experimental controller watchdog must not free its allocation.
            continue
        if req is not None and req.finished():
            if st.finished_at is None:
                st.finished_at = time.monotonic()
            if time.monotonic() - st.finished_at >= _DEAD_STATE_GRACE_S:
                # No controller is left to send release, and the retention
                # grace has expired.  Drop the handoff state *and* free the
                # finished request's slots; otherwise a controller kill at
                # commit leaves a finished req_pool row behind (P5.15).
                _drop_target_state_for(req)
                tracked = _tracked(scheduler)
                for _rid, _req in list(tracked.items()):
                    if _req is req:
                        tracked.pop(_rid, None)
                        break
                _free_source_request(scheduler, req)
                print(
                    f"[handoff-target] freed finished state id={st.handoff_id} "
                    f"after controller loss",
                    flush=True,
                )


def _expire_stale_reservations(scheduler) -> None:
    """Free pre-freeze reservations whose controller died before freeze.

    A heartbeat-stale target has no writer left to send reserve_cancel; the
    reservation is safe to drop because no request object exists for its rid.
    """
    global _RESV_HB_SEEN
    hb_path = _heartbeat_path()
    if hb_path is None or not hb_path.exists():
        return
    try:
        hb = json.loads(hb_path.read_text(encoding="utf-8"))
        ts = float(hb.get("ts", 0.0))
    except Exception:
        return
    _RESV_HB_SEEN = True
    if time.time() - ts < _FROZEN_WD_TIMEOUT_S:
        return
    tracked = _tracked(scheduler)
    reservations = _reservations(scheduler)
    now = time.monotonic()
    for rid in list(reservations):
        if rid in tracked:
            continue
        res = reservations.get(rid) or {}
        if now - float(res.get("created_at", now)) < _FROZEN_WD_TIMEOUT_S:
            continue
        if _reservation_free(scheduler, rid) is not None:
            print(
                f"[handoff-watchdog] expired stale reservation id={rid}",
                flush=True,
            )


def _heartbeat_path() -> Optional[Path]:
    if not _CTL_DIR:
        return None
    return Path(_CTL_DIR) / "heartbeat.json"


def _frozen_watchdog_loop(scheduler) -> None:
    global _FROZEN_WD_SEEN_HB
    if os.environ.get("HANDOFF_DISABLE_FROZEN_WATCHDOG", "0") == "1":
        return
    hb_path = _heartbeat_path()
    while True:
        time.sleep(0.2)
        try:
            if hb_path is None or not hb_path.exists():
                continue
            try:
                hb = json.loads(hb_path.read_text(encoding="utf-8"))
                ts = float(hb.get("ts", 0.0))
            except Exception:
                continue
            _FROZEN_WD_SEEN_HB = True
            if time.time() - ts < _FROZEN_WD_TIMEOUT_S:
                continue
            hub = getattr(scheduler, "_handoff_hub", None)
            if hub is None:
                continue
            tracked = dict(_tracked(scheduler))
            for rid, req in tracked.items():
                if not getattr(req, "_handoff_frozen", False):
                    # A request resumed by this watchdog (or completed after a
                    # dead controller) has no gateway left to release it.  It
                    # may still be the only copy of the output an adopted
                    # stream needs, so hold it for a bounded grace period.
                    if req.finished():
                        key = str(rid)
                        seen = _FINISHED_SEEN.setdefault(key, time.monotonic())
                        if time.monotonic() - seen >= _FINISHED_RETAIN_S:
                            _FINISHED_SEEN.pop(key, None)
                            _FROZEN_WD_ACTIONS[key] = ("free_finished", req)
                    else:
                        _FINISHED_SEEN.pop(str(rid), None)
                    continue
                ctx = hub.registry.get(rid)
                epoch = int(ctx.owner_epoch) if ctx is not None else -1
                key = (str(rid), epoch)
                if key in _FROZEN_WD_HANDLED:
                    continue
                if ctx is not None and ctx.done:
                    action = "release"
                elif ctx is not None and ctx.committed:
                    action = "hold_committed"
                elif ctx is not None and ctx.imported:
                    action = "hold_imported"
                elif ctx is None:
                    action = "hold_unknown"
                else:
                    action = "resume"
                if action == "resume":
                    # Keep re-evaluating: the first fence/enqueue may fail
                    # because the target is still importing.  The durable
                    # claim/tombstone prevents duplicate execution; the
                    # action is removed only after a successful resume.
                    if key not in _FROZEN_WD_LOGGED:
                        _FROZEN_WD_LOGGED.add(key)
                        print(
                            f"[handoff-watchdog] controller heartbeat stale "
                            f"id={rid} epoch={epoch} action={action}",
                            flush=True,
                        )
                    _FROZEN_WD_ACTIONS[str(rid)] = (action, req)
                elif action == "release":
                    _FROZEN_WD_HANDLED.add(key)
                    print(
                        f"[handoff-watchdog] controller heartbeat stale "
                        f"id={rid} epoch={epoch} action={action}",
                        flush=True,
                    )
                    _FROZEN_WD_ACTIONS[str(rid)] = (action, req)
                else:
                    # Hold actions are not terminal: re-check later so a
                    # late ``done`` can still release the lease.  Log once.
                    if key in _FROZEN_WD_LOGGED:
                        continue
                    _FROZEN_WD_LOGGED.add(key)
                    print(
                        f"[handoff-watchdog] holding id={rid} epoch={epoch} "
                        f"reason={action}",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[handoff-watchdog] loop error: {exc!r}", flush=True)


def _ensure_frozen_watchdog(scheduler) -> None:
    global _FROZEN_WD
    with _FROZEN_WD_LOCK:
        if _FROZEN_WD is not None:
            return
        _FROZEN_WD = threading.Thread(
            target=_frozen_watchdog_loop,
            args=(scheduler,),
            daemon=True,
            name="handoff-frozen-watchdog",
        )
        _FROZEN_WD.start()


def _drain_watchdog_actions(scheduler) -> None:
    if not _FROZEN_WD_ACTIONS:
        return
    hub = getattr(scheduler, "_handoff_hub", None)
    for rid, (action, req) in list(_FROZEN_WD_ACTIONS.items()):
        _FROZEN_WD_ACTIONS.pop(rid, None)
        try:
            if action == "resume":
                if not getattr(req, "_handoff_frozen", False):
                    continue
                if not _watchdog_resume_rank_allowed(scheduler):
                    continue
                ctx = hub.registry.get(rid) if hub is not None else None
                if ctx is not None and ctx.committed:
                    print(
                        f"[handoff-watchdog] resume refused id={rid}: target committed",
                        flush=True,
                    )
                    if ctx is not None:
                        _FROZEN_WD_HANDLED.add((str(rid), int(ctx.owner_epoch)))
                    continue
                epoch = int(ctx.owner_epoch) if ctx is not None else -1
                was_imported = bool(ctx is not None and ctx.imported)
                claim = f"{rid}:{epoch}"
                if _resume_blocked(claim):
                    continue
                if _resume_claim_pending(claim):
                    # A durable resume command is already queued/executing.
                    # Do not re-fence or publish a second command.
                    continue
                freeze_result = getattr(req, "_handoff_freeze_result", None)
                if not isinstance(freeze_result, dict):
                    freeze_result = {}
                migration_epoch = str(
                    freeze_result.get("migration_epoch")
                    or freeze_result.get("epoch")
                    or ""
                )
                target_rid = str(
                    getattr(req, "_handoff_target_rid", None) or f"dst-{rid}"
                )
                fence = _fence_target_import(
                    rid, target_rid, migration_epoch, was_imported
                )
                if fence != "safe":
                    prev = _FROZEN_WD_FENCE_STATE.get(str(rid))
                    if prev != fence:
                        _FROZEN_WD_FENCE_STATE[str(rid)] = fence
                        print(
                            f"[handoff-watchdog] resume fenced id={rid} "
                            f"target={target_rid} epoch={migration_epoch} "
                            f"state={fence}",
                            flush=True,
                        )
                    if fence in (
                        "committed",
                        "epoch_mismatch",
                        "ambiguous",
                        "unknown_epoch",
                    ):
                        # Target already owns this epoch (or holds a newer
                        # one).  Never resume the source into a dual-owner
                        # state; leave it frozen and stop retrying.
                        _mark_resume_blocked(claim, fence, rid)
                        _FROZEN_WD_HANDLED.add((str(rid), epoch))
                    continue
                if _FROZEN_WD_FENCE_STATE.get(str(rid)) != "safe":
                    _FROZEN_WD_FENCE_STATE[str(rid)] = "safe"
                    print(
                        f"[handoff-watchdog] fence ok id={rid} "
                        f"target={target_rid} epoch={migration_epoch}",
                        flush=True,
                    )
                nonce = ""
                try:
                    nonce = enqueue_broadcast_control_once(
                        {
                            "cmd": "resume_many_decode",
                            "rids": [rid],
                            "epoch": epoch,
                            "target_rid": target_rid,
                        },
                        claim,
                    )
                except Exception as exc:
                    print(
                        f"[handoff-watchdog] enqueue resume failed id={rid}: {exc!r}",
                        flush=True,
                    )
                    _FROZEN_WD_ACTIONS[str(rid)] = (action, req)
                    continue
                if nonce:
                    print(
                        f"[handoff-watchdog] queued resume id={rid} "
                        f"tp={_tp_rank(scheduler)} epoch={epoch} "
                        f"target={target_rid} nonce={nonce}",
                        flush=True,
                    )
                else:
                    print(
                        f"[handoff-watchdog] resume already queued id={rid} "
                        f"tp={_tp_rank(scheduler)} epoch={epoch}",
                        flush=True,
                    )
                continue
            elif action == "free_finished":
                _free_source_request(scheduler, req)
                _tracked(scheduler).pop(rid, None)
                print(f"[handoff-watchdog] freed finished id={rid}", flush=True)
            elif action == "release":
                if hub is None:
                    continue
                released = hub.registry.release(rid, expected_req=req)
                print(
                    f"[handoff-watchdog] release id={rid}: {released}",
                    flush=True,
                )
                if not released:
                    # Inflight remote read: retry after it drains.
                    _FROZEN_WD_ACTIONS[str(rid)] = (action, req)
        except Exception as exc:
            print(
                f"[handoff-watchdog] action {action} id={rid} failed: {exc!r}",
                flush=True,
            )


def _import_cancelled(rid: str, epoch) -> bool:
    if epoch is None:
        return False
    key = (str(rid), str(epoch))
    now = time.monotonic()
    if len(_CANCELLED_IMPORTS) > 1024:
        for k, ts in list(_CANCELLED_IMPORTS.items()):
            if now - ts > _CANCEL_TTL_S:
                _CANCELLED_IMPORTS.pop(k, None)
    ts = _CANCELLED_IMPORTS.get(key)
    if ts is not None and now - ts <= _CANCEL_TTL_S:
        return True
    # Controller-written marker: visible even while this command is sleeping
    # (a timeout injection or a late-arriving command after resume).
    if _CTL_DIR:
        safe = f"cancel.{str(rid).replace('/', '_')}.{str(epoch)}.json"
        marker = Path(_CTL_DIR) / safe
        try:
            if marker.exists():
                age = time.time() - marker.stat().st_mtime
                if age <= _CANCEL_TTL_S:
                    return True
                marker.unlink()
        except OSError:
            pass
    return False


_TRANSFER_ERRORS: dict[str, str] = {}
_TRANSFER_STATS: dict[str, dict] = {}
# On an unacknowledged transport failure, keep registered CPU buffers alive
# until completion and safe unregister are proven. Disconnect alone is not
# proof that a remote DMA/TCP write no longer targets them.
_TRANSFER_QUARANTINE: dict[str, tuple[Any, str]] = {}
_TRANSFER_TICKETS: dict[str, dict] = {}
_TRANSFER_RELEASES: dict[str, dict] = {}
_TRANSFER_RELEASE_LOCK = threading.RLock()


def _quarantine_transfer(owner, buffers, reason):
    with _TRANSFER_RELEASE_LOCK:
        _TRANSFER_QUARANTINE[owner] = (buffers, str(reason))
        _TRANSFER_ERRORS[owner] = str(reason)


def _with_transfer_ticket(token, meta):
    if not _CTL_DIR:
        return meta
    root = Path(_CTL_DIR) / "transfer-completion"
    root.mkdir(parents=True, exist_ok=True)
    ticket = {"path": str(root / f"{token}.json"), "token": token}
    _TRANSFER_TICKETS[token] = ticket
    return {**meta, "host_completion": ticket}


def _bind_transfer_ticket(owner, token, meta):
    ticket = meta.get("host_completion")
    if not _CTL_DIR or not isinstance(ticket, dict):
        return
    path = Path(str(ticket.get("path", ""))).resolve()
    # The supported topology shares one control root. Never delete arbitrary
    # paths supplied over the control socket.
    if (
        len(token) == 32
        and all(c in "0123456789abcdef" for c in token)
        and ticket.get("token") == token
        and path.name == f"{token}.json"
        and path.parent.name == "transfer-completion"
        and path.is_relative_to(Path(_CTL_DIR).resolve().parent)
    ):
        _TRANSFER_TICKETS[owner] = {"path": str(path), "token": token}


def _finish_transfer_ticket(ticket, *, publish):
    if ticket is None:
        return
    path = Path(ticket["path"])
    other = path.with_suffix(".ack") if publish else path
    destination = path if publish else path.with_suffix(".ack")
    tmp = destination.with_name(
        destination.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        tmp.write_text(json.dumps({"token": ticket["token"], "complete": True}))
        os.replace(tmp, destination)
        # Either publisher/consumer may win. The second writer removes both
        # records, avoiding a normal-transfer completion-file accumulation.
        if other.exists():
            other.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[handoff-transfer] completion record failed: {exc!r}", flush=True)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _publish_transfer_complete(owner, safe):
    if safe:
        _finish_transfer_ticket(_TRANSFER_TICKETS.get(owner), publish=True)


def _release_transfer_buffers(
    owner, buffers, reason, *, remote_complete=False, gpu_safe=True
):
    with _TRANSFER_RELEASE_LOCK:
        ticket = _TRANSFER_TICKETS.get(owner)
        if not remote_complete and ticket is not None:
            try:
                record = json.loads(Path(ticket["path"]).read_text())
                remote_complete = (
                    isinstance(record, dict)
                    and record.get("token") == ticket["token"]
                    and record.get("complete") is True
                )
            except (OSError, ValueError):
                pass
        if remote_complete and gpu_safe:
            try:
                released = _get_registration_manager().release_safe(owner)
            except Exception as exc:
                released = False
                reason = f"host unregister failed: {exc!r}"
            if released:
                _TRANSFER_QUARANTINE.pop(owner, None)
                _TRANSFER_RELEASES.pop(owner, None)
                _TRANSFER_TICKETS.pop(owner, None)
                # Source publication owns the source token. Only the target
                # acknowledges/deletes the shared completion proof.
                if ticket is not None and owner != ticket["token"]:
                    _finish_transfer_ticket(ticket, publish=False)
                return True
        _quarantine_transfer(owner, buffers, reason)
        _TRANSFER_RELEASES[owner] = {
            "remote_complete": remote_complete,
            "gpu_safe": gpu_safe,
        }
        return False


def reap_transfer_buffers():
    """Retry only proven-complete transfers; TTL alone never permits reuse."""
    with _TRANSFER_RELEASE_LOCK:
        for owner, flags in list(_TRANSFER_RELEASES.items()):
            entry = _TRANSFER_QUARANTINE.get(owner)
            if entry is None:
                _TRANSFER_RELEASES.pop(owner, None)
                continue
            buffers, reason = entry
            _release_transfer_buffers(owner, buffers, reason, **flags)


def _mark_source_transfer(scheduler, token, active):
    # Startup occurs on the scheduler thread before the first sender starts.
    guard = getattr(scheduler, "_handoff_transfer_guard", None)
    if guard is None:
        guard = scheduler._handoff_transfer_guard = threading.Lock()
        scheduler._handoff_transfer_tokens = set()
    with guard:
        tokens = scheduler._handoff_transfer_tokens
        if active:
            tokens.add(token)
        else:
            tokens.discard(token)
        scheduler._handoff_transfer_active = bool(tokens)


def stop_source_transfer(req):
    """Stop background GPU readers before a frozen source can free/reuse KV."""
    handle = getattr(req, "_handoff_transfer_cancel", None)
    if handle is not None:
        listener, cancel = handle
        cancel.set()
        listener.close()
    conn = getattr(req, "_handoff_transfer_conn", None)
    if conn is not None:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    thread = getattr(req, "_handoff_transfer_thread", None)
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("handoff source transfer did not stop before KV reuse")
    if getattr(req, "_handoff_transfer_unsafe", False):
        raise RuntimeError("handoff source CUDA readers could not be fenced")
    if getattr(req, "_handoff_transfer_cancel", None) is handle:
        req._handoff_transfer_cancel = None


_TE = None
_TE_LOCK = threading.Lock()
PROFILE = os.environ.get("HANDOFF_PROFILE", "0") == "1"
# P0 attribution: per-RPC stage timings on the target. Kept separate from
# PROFILE because PROFILE creates CUDA events per RPC (host-side ~130 ms tail).
RPC_PROFILE = os.environ.get("HANDOFF_RPC_PROFILE", "0") == "1"
# P1.1: aggregate all remote rows of one layer into a single RPC.  Set to 0
# to fall back to the old per-row RPC path for A/B measurements.
BATCH_ROWS = os.environ.get("HANDOFF_BATCH_ROWS", "1") == "1"
# P1.2: pinned host staging + async copy + batch merge.
ASYNC_COPY = os.environ.get("HANDOFF_ASYNC_COPY", "1") == "1"
# P1.3: one paged batch-decode kernel for all rows of a layer.  Measured on
# 32K x C4 it did not beat per-row single_decode (source is not the bottleneck),
# so it is opt-in; set HANDOFF_PAGED_BATCH=1 to enable it.
PAGED_BATCH = os.environ.get("HANDOFF_PAGED_BATCH", "0") == "1"
# P2.2: per-step attribution of attention host time, local rows, and the
# non-attention remainder (FFN/o_proj/scheduler).
P22_PROFILE = os.environ.get("HANDOFF_P22_PROFILE", "0") == "1"
# P5.0: per-request freeze breakdown (host timestamps only, low overhead).
FREEZE_PROFILE = os.environ.get("HANDOFF_FREEZE_PROFILE", "0") == "1"
# P5.1: freeze state immediately and stream KV from GPU through pinned layer
# staging buffers; do not materialize a full CPU snapshot in the freeze path.
FAST_FREEZE = os.environ.get("HANDOFF_FAST_FREEZE", "0") == "1"


def _env_int(name: str, default: int, lo: int = 1) -> int:
    try:
        return max(lo, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


# P5.2-A: staging pipeline ablation.  Async TE is required for any overlap.
FAST_BUFFERS = _env_int("HANDOFF_FAST_BUFFERS", 1)
FAST_COPY_STREAMS = _env_int("HANDOFF_FAST_COPY_STREAMS", 1)
FAST_ASYNC = os.environ.get("HANDOFF_FAST_ASYNC", "0") == "1"
# P5.2-B: target-side bounded layer staging slots (fast freeze only).
# 0 keeps the P5.1 full CPU mirror for a direct A/B baseline.
TARGET_STAGING_SLOTS = _env_int("HANDOFF_TARGET_STAGING_SLOTS", 2, lo=0)
_FAST_STREAMS: dict[tuple[str, str], list] = {}
_FAST_STREAM_LOCK = threading.Lock()
# P2.2a A/B knobs: sync Q D2H (before local tail) and one completion event
# covering both O and LSE H2D copies.
Q_SYNC = os.environ.get("HANDOFF_Q_SYNC", "0") == "1"
# V2 was the best A/B arm (async Q + one O/LSE completion event).
H2D_SINGLE = os.environ.get("HANDOFF_H2D_SINGLE", "1") == "1"
_GIL_GAPS = []
_GIL_PROBE_STARTED = False

# Hub listener knobs (one listener per source scheduler process).
_HUB_LISTEN_BACKLOG = 64
_HUB_ACCEPT_TIMEOUT_S = 1.0
_HUB_PROTO = 2
_HUB_FRAME_TIMEOUT_S = 120.0


class _HubConn:
    """State for one long-lived target connection (protocol v2)."""

    __slots__ = (
        "sock",
        "send_lock",
        "lock",
        "attached",
        "in_flight",
        "last_rx",
        "dead",
    )

    def __init__(self, sock):
        self.sock = sock
        self.send_lock = threading.Lock()
        self.lock = threading.Lock()
        self.attached: dict[str, int] = {}
        self.in_flight = 0
        self.last_rx = time.monotonic()
        self.dead = False


@dataclass
class HandoffContext:
    handoff_id: str
    req: Any
    req_pool_idx: int
    base_len: int
    kv_len: int
    token: str
    owner_epoch: int
    idx_start: int = -1
    idx_end: int = -1
    contiguous: bool = False
    inflight: int = 0
    closed: bool = False
    # P5.4-B0: target-reported ownership facts.  These are the authoritative
    # inputs for a source-side recovery decision when the controller dies.
    imported: bool = False
    committed: bool = False
    done: bool = False
    on_release: Optional[Callable[[HandoffContext], None]] = None


class HandoffRegistry:
    """Source-side lifetime registry for frozen handoff requests.

    The registry is the lease: an entry cannot be removed while any RPC holds
    an inflight reference. A release during inflight marks a tombstone; new
    readers are rejected and the last release_ref removes the entry.

    Re-registering an existing ``handoff_id`` replaces a live entry when it has
    no inflight references and frees the old generation; the old generation's
    ``on_release`` runs only when the new request object differs, so a shared
    request is not double-freed.

    ``on_release`` runs outside the lock after removal, exactly once, and
    exceptions propagate to the caller, except for the superseded-generation
    callback in :meth:`register`, whose failure is logged instead so the new
    registration stays reachable. ``release`` raises ``KeyError`` for an
    unknown id or when ``expected_req`` names a different generation, and
    returns ``False`` only when removal is deferred by inflight RPCs. Entries
    are keyed by string ``handoff_id``; callers must not release the same
    handoff id twice.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ctx: dict[str, HandoffContext] = {}
        self._epoch = 0

    def register(
        self,
        handoff_id: str,
        req: Any,
        req_pool_idx: int,
        base_len: int,
        kv_len: int,
        token: str,
        idx_start: int = -1,
        idx_end: int = -1,
        contiguous: bool = False,
        on_release: Optional[Callable[[HandoffContext], None]] = None,
    ) -> HandoffContext:
        hid = str(handoff_id)
        # Coerce/validate before mutating registry state so a failed call does
        # not consume an epoch.
        req_pool_idx = int(req_pool_idx)
        base_len = int(base_len)
        kv_len = int(kv_len)
        token = str(token)
        idx_start = int(idx_start)
        idx_end = int(idx_end)
        contiguous = bool(contiguous)
        superseded = None
        with self._lock:
            old = self._ctx.get(hid)
            if old is not None and old.inflight > 0:
                raise RuntimeError(
                    f"handoff {hid!r} still has {old.inflight} inflight RPCs"
                )
            self._epoch += 1
            ctx = HandoffContext(
                handoff_id=hid,
                req=req,
                req_pool_idx=req_pool_idx,
                base_len=base_len,
                kv_len=kv_len,
                token=token,
                owner_epoch=self._epoch,
                idx_start=idx_start,
                idx_end=idx_end,
                contiguous=contiguous,
                on_release=on_release,
            )
            self._ctx[hid] = ctx
            if old is not None and old.req is not req:
                superseded = old
        if superseded is not None and superseded.on_release is not None:
            try:
                superseded.on_release(superseded)
            except Exception as exc:
                print(
                    f"[handoff-hub] superseded release failed: {exc!r}",
                    flush=True,
                )
        return ctx

    def get(
        self, handoff_id: str, epoch: Optional[int] = None
    ) -> Optional[HandoffContext]:
        with self._lock:
            ctx = self._ctx.get(str(handoff_id))
            if ctx is None or ctx.closed:
                return None
            if epoch is not None and ctx.owner_epoch != int(epoch):
                return None
            return ctx

    def mark_state(self, handoff_id: str, epoch: int, kind: str) -> bool:
        """Apply a target-reported ownership fact (imported/committed/done)."""
        with self._lock:
            ctx = self._ctx.get(str(handoff_id))
            if ctx is None or ctx.owner_epoch != int(epoch):
                return False
            if kind == "imported":
                ctx.imported = True
            elif kind == "commit":
                ctx.committed = True
            elif kind == "done":
                ctx.done = True
            else:
                return False
            return True

    def acquire(self, handoff_id: str, epoch: int) -> Optional[HandoffContext]:
        with self._lock:
            ctx = self._ctx.get(str(handoff_id))
            if ctx is None or ctx.closed or ctx.owner_epoch != int(epoch):
                return None
            ctx.inflight += 1
            return ctx

    def release_ref(self, handoff_id: str, epoch: int) -> bool:
        hid = str(handoff_id)
        to_free = None
        with self._lock:
            ctx = self._ctx.get(hid)
            if ctx is None:
                return False
            owner_epoch = int(epoch)
            if ctx.owner_epoch != owner_epoch:
                return False
            if ctx.inflight <= 0:
                raise RuntimeError(f"handoff {hid!r} epoch {owner_epoch}: underflow")
            ctx.inflight -= 1
            if ctx.closed and ctx.inflight == 0:
                del self._ctx[hid]
                to_free = ctx
        if to_free is not None and to_free.on_release is not None:
            to_free.on_release(to_free)
        return True

    def release(self, handoff_id: str, expected_req: Any = None) -> bool:
        hid = str(handoff_id)
        to_free = None
        with self._lock:
            ctx = self._ctx.get(hid)
            if ctx is None:
                raise KeyError(f"unknown handoff {hid!r}")
            if expected_req is not None and ctx.req is not expected_req:
                raise KeyError(f"handoff {hid!r} belongs to a different request")
            if ctx.inflight > 0:
                ctx.closed = True
                return False
            del self._ctx[hid]
            to_free = ctx
        if to_free is not None and to_free.on_release is not None:
            to_free.on_release(to_free)
        return True

    def discard(self, handoff_id, expected_req=None) -> bool:
        """Drop a registration without firing on_release (setup failed).

        Unlike ``release`` this never runs the release callback: the request is
        still live and its pool slots must not be freed here.
        """
        hid = str(handoff_id)
        with self._lock:
            ctx = self._ctx.get(hid)
            if ctx is None:
                return False
            if expected_req is not None and ctx.req is not expected_req:
                return False
            if ctx.inflight > 0:
                return False
            del self._ctx[hid]
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "handoffs": len(self._ctx),
                "inflight": sum(c.inflight for c in self._ctx.values()),
                "closed": sum(1 for c in self._ctx.values() if c.closed),
            }


def _stop_receiver_io(state):
    conn = state.receiver_conn
    if conn is not None:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        conn.close()
    receiver = state.receiver_thread
    if receiver is not None and receiver is not threading.current_thread():
        receiver.join(timeout=5.0)
        if receiver.is_alive():
            raise RuntimeError("handoff receiver did not stop before KV release")
    if getattr(state, "receiver_unsafe", False):
        raise RuntimeError("handoff receiver CUDA writes could not be fenced")


def stop_target_receivers(req):
    """Fence background writes before upstream returns this request's KV slots."""
    matched = [st for st in _TARGET_STATES if st.req is req]
    for state in matched:
        if getattr(state, "import_done_at", None) is not None and state.error is None:
            _stop_receiver_io(state)
        else:
            state.abort("request KV released")
        receiver = state.receiver_thread
        if receiver is not None and receiver.is_alive():
            raise RuntimeError("handoff receiver did not stop before KV release")
        if getattr(state, "receiver_unsafe", False):
            raise RuntimeError("handoff receiver CUDA writes could not be fenced")
        state.notify_source("done")
    if matched:
        _drop_target_state_for(req)


def _free_source_request(scheduler, req):
    """Free a released source request's batches and KV exactly once."""
    # The registry guarantees at-most-once delivery, but a duplicate drain
    # (e.g. a release enqueued twice by a retried connection path) must not
    # free the same pool slots again. The flag is set before the frees on
    # purpose: a leak is preferable to a double free.
    if getattr(req, "_handoff_freed", False):
        return
    stop_target_receivers(req)
    stop_source_transfer(req)
    req._handoff_freed = True
    _remove_from_batches(scheduler, req)
    allocator = getattr(scheduler, "token_to_kv_pool_allocator", None)
    avail_before = _kv_available(scheduler)
    # A resumed request reuses its original prefix indices.  The scheduler's
    # normal free path only knows the tokens it wrote into req_to_token after
    # the resume, so release must free the frozen prefix explicitly.
    resume_prefix_len = int(getattr(req, "_handoff_resume_prefix_len", 0) or 0)
    pending = bool(getattr(req, "_handoff_resume_prefix_pending", False))
    prefix_freed = 0
    if pending:
        # Free the exact tensor captured at resume time.  The scheduler may
        # replace req.prefix_indices while extending, and freeing the current
        # value would release slots owned by another allocation.
        idx = getattr(req, "_handoff_resume_prefix_indices", None)
        if allocator is not None and idx is not None and int(idx.numel()) > 0:
            try:
                prefix_freed = int(idx.numel())
                allocator.free(idx)
            except Exception:
                prefix_freed = 0
        req._handoff_resume_prefix_pending = False
        req._handoff_resume_prefix_indices = None
    pool_idx = req.kv.req_pool_idx
    kv_len = int(req.kv.kv_committed_len)
    range_freed = 0
    imported_prefix = getattr(req, "handoff_prefix_indices", None)
    if pool_idx is None and imported_prefix is not None and not pending:
        # Imported/fast-resumed KV can own slots before it owns a request row.
        # A release at that boundary must free the pending mapping as well.
        if allocator is not None and not getattr(req, "_handoff_prefix_freed", False):
            allocator.free(imported_prefix)
            prefix_freed += int(imported_prefix.numel())
            req._handoff_prefix_freed = True
    if req.kv.req_pool_idx is not None:
        allocated_len = max(kv_len, req.kv.kv_allocated_len)
        if allocator is not None and allocated_len > resume_prefix_len:
            indices = scheduler.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, resume_prefix_len:allocated_len
            ]
            range_freed = max(0, allocated_len - resume_prefix_len)
            allocator.free(indices)
        scheduler.req_to_token_pool.free(req)
    req.handoff_prefix_indices = None
    req._handoff_prefix_indices = None
    req.kv.mark_kv_released()
    req.kv.kv_committed_len = 0
    avail_after = _kv_available(scheduler)
    print(
        f"[kv-free] rid={req.rid} pool_idx={pool_idx} kv_len={kv_len} "
        f"resume_prefix={resume_prefix_len} pending={pending} "
        f"full_reset={bool(getattr(req, '_handoff_full_reset', False))} "
        f"freed_prefix={prefix_freed} freed_range={range_freed} "
        f"avail={avail_before}->{avail_after} "
        f"delta={avail_after - avail_before}",
        flush=True,
    )


def _q_cpu_from_bytes(header: dict, raw: bytes) -> torch.Tensor:
    q_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[header["q_dtype"]]
    return (
        torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        .view(q_dtype)
        .reshape(tuple(int(x) for x in header["q_shape"]))
    )


def _recv_q(sock, header: dict) -> torch.Tensor:
    return _q_cpu_from_bytes(header, _recv_exact(sock, int(header["q_bytes"])))


def _fence_remote_job(job, contexts):
    """Keep source leases if a CUDA reader cannot be proven finished."""
    try:
        if job.stream is not None:
            job.stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
    except Exception as exc:
        for ctx in contexts:
            ctx.req._handoff_transfer_unsafe = True
            _TRANSFER_ERRORS[str(ctx.handoff_id)] = (
                f"remote reader fence failed: {exc!r}"
            )
        return False
    return True


class RemoteJob:
    """One remote attention RPC: GPU work is done by a pool worker."""

    __slots__ = (
        "hub",
        "conn",
        "hc",
        "send_lock",
        "ctx",
        "rpc_id",
        "epoch",
        "header",
        "q_cpu",
        "stream",
        "t_enqueue",
        "done",
        "recv_ms",
        "recv_delay_ms",
    )

    def __init__(self, hub, conn, hc, send_lock, ctx, rpc_id, epoch, header, q_cpu):
        self.hub = hub
        self.conn = conn
        self.hc = hc
        self.send_lock = send_lock
        self.ctx = ctx
        self.rpc_id = int(rpc_id)
        self.epoch = int(epoch)
        self.header = header
        self.q_cpu = q_cpu
        self.stream = None
        self.t_enqueue = time.perf_counter()
        self.done = False
        self.recv_ms = 0.0
        self.recv_delay_ms = 0.0

    def _mark_conn_dead(self) -> None:
        """Wake the connection reader after a failed send."""
        if self.hc is None:
            return
        self.hc.dead = True
        try:
            self.hc.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def reply(self, resp: dict, o=None, lse=None) -> bool:
        ok = True
        try:
            with self.send_lock:
                _send_json(
                    self.conn,
                    {"ok": True, "cmd": "attn", "rpc_id": self.rpc_id, **resp},
                )
                if o is not None:
                    _send_tensor(self.conn, o)
                if lse is not None:
                    _send_tensor(self.conn, lse)
        except OSError:
            ok = False
            self._mark_conn_dead()
        with self.hub._stats_lock:
            if ok:
                self.hub.rpc_completed += 1
            else:
                self.hub.rpc_failed += 1
        return ok

    def fail(self, exc) -> None:
        try:
            with self.send_lock:
                _send_json(
                    self.conn,
                    {
                        "ok": False,
                        "cmd": "attn",
                        "rpc_id": self.rpc_id,
                        "error": repr(exc),
                    },
                )
        except OSError:
            self._mark_conn_dead()
        with self.hub._stats_lock:
            self.hub.rpc_failed += 1

    def finish(self) -> None:
        if self.done:
            return
        self.done = True
        if not _fence_remote_job(self, [self.ctx]):
            return
        try:
            self.hub.registry.release_ref(self.ctx.handoff_id, self.epoch)
        except Exception:
            pass
        if self.hc is not None:
            with self.hc.lock:
                self.hc.in_flight = max(0, self.hc.in_flight - 1)
        self.hub._job_finished(self.hc)


class RemoteBatchJob:
    """One P1.1 RPC carrying several independent remote rows of one layer."""

    is_batch = True
    __slots__ = (
        "hub",
        "conn",
        "hc",
        "send_lock",
        "items",
        "rpc_id",
        "t_enqueue",
        "done",
        "stream",
    )

    def __init__(self, hub, conn, hc, send_lock, items, rpc_id):
        self.hub = hub
        self.conn = conn
        self.hc = hc
        self.send_lock = send_lock
        self.items = list(items)
        self.rpc_id = int(rpc_id)
        self.t_enqueue = time.perf_counter()
        self.done = False
        self.stream = None

    def _mark_conn_dead(self) -> None:
        if self.hc is None:
            return
        self.hc.dead = True
        try:
            self.hc.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def reply_batch(self, resp: dict, outs) -> bool:
        ok = True
        try:
            payload = dict(resp)
            payload["rows"] = [row_resp for row_resp, _o, _lse in outs]
            batch_o = batch_lse = None
            os_ = [o for _r, o, _l in outs if o is not None]
            ls_ = [l for _r, _o, l in outs if l is not None]
            if (
                len(os_) == len(outs)
                and len(ls_) == len(outs)
                and len({(tuple(o.shape), o.dtype) for o in os_}) == 1
                and len({(tuple(l.shape), l.dtype) for l in ls_}) == 1
            ):
                try:
                    batch_o = torch.stack(os_, 0)
                    batch_lse = torch.stack(ls_, 0)
                    payload["o_batch_shape"] = list(batch_o.shape)
                    payload["o_batch_dtype"] = (
                        "bfloat16"
                        if batch_o.dtype == torch.bfloat16
                        else "float16"
                        if batch_o.dtype == torch.float16
                        else "float32"
                    )
                    payload["lse_batch_shape"] = list(batch_lse.shape)
                except Exception:
                    batch_o = batch_lse = None
            with self.send_lock:
                _send_json(
                    self.conn,
                    {"ok": True, "cmd": "attn_batch", "rpc_id": self.rpc_id, **payload},
                )
                if batch_o is not None:
                    _send_tensor(self.conn, batch_o)
                    _send_tensor(self.conn, batch_lse)
                else:
                    for row_resp, o, lse in outs:
                        if o is not None:
                            _send_tensor(self.conn, o)
                        if lse is not None:
                            _send_tensor(self.conn, lse)
        except OSError:
            ok = False
            self._mark_conn_dead()
        with self.hub._stats_lock:
            if ok:
                self.hub.rpc_completed += 1
            else:
                self.hub.rpc_failed += 1
        return ok

    def fail(self, exc) -> None:
        try:
            with self.send_lock:
                _send_json(
                    self.conn,
                    {
                        "ok": False,
                        "cmd": "attn_batch",
                        "rpc_id": self.rpc_id,
                        "error": repr(exc),
                    },
                )
        except OSError:
            self._mark_conn_dead()
        with self.hub._stats_lock:
            self.hub.rpc_failed += 1

    def finish(self) -> None:
        if self.done:
            return
        self.done = True
        if not _fence_remote_job(self, [item.ctx for item in self.items]):
            return
        for item in self.items:
            try:
                self.hub.registry.release_ref(item.ctx.handoff_id, item.epoch)
            except Exception:
                pass
        if self.hc is not None:
            with self.hc.lock:
                self.hc.in_flight = max(0, self.hc.in_flight - len(self.items))
        self.hub._job_finished(self.hc, len(self.items))


def _trace_layer_diag(
    job,
    api_ms,
    q_h2d_ms,
    t_d2h0,
    t_d2h_o,
    t_d2h_lse,
    t_api_return,
    t_sync0,
    t_done,
    source_ms,
):
    try:
        if (
            int(job.header.get("step", -1)) == 0
            and Path("/workspace/handoff_state/layer_diag").exists()
        ):
            print(
                f"[handoff-layer] L{job.header.get('layer')} "
                f"recv={job.recv_delay_ms:.1f} h2d={q_h2d_ms:.1f} "
                f"api={api_ms:.1f} "
                f"d2h_o={(t_d2h_o - t_d2h0) * 1000.0:.1f} "
                f"d2h_lse={(t_d2h_lse - t_d2h_o) * 1000.0:.1f} "
                f"total={source_ms:.1f}",
                flush=True,
            )
    except Exception:
        pass


def _log_first_rpc(
    job, api_ms, q_h2d_ms, t_d2h0, t_d2h_o, t_d2h_lse, t_sync0, t_done, source_ms
):
    hub = job.hub
    with hub._stats_lock:
        if hub._first_rpc_logged:
            return
        hub._first_rpc_logged = True
    print(
        f"[handoff-first] step={job.header.get('step')} "
        f"layer={job.header.get('layer')} "
        f"recv_delay={job.recv_delay_ms:.1f} q_recv={job.recv_ms:.1f} "
        f"h2d={q_h2d_ms:.1f} api={api_ms:.1f} "
        f"sync_wait={(t_done - t_sync0) * 1000.0:.1f} "
        f"d2h_o={(t_d2h_o - t_d2h0) * 1000.0:.1f} "
        f"d2h_lse={(t_d2h_lse - t_d2h_o) * 1000.0:.1f} "
        f"total={source_ms:.1f}",
        flush=True,
    )


def _append_source_trace(
    job, resp, q_h2d_ms, t_api_return, t_d2h0, t_d2h_o, t_d2h_lse, t_done, source_ms
):
    try:
        path = Path("/workspace/handoff_state") / (
            f"source_trace_{job.ctx.handoff_id}_{job.ctx.owner_epoch}.jsonl"
        )
        rec = {
            "handoff_id": job.ctx.handoff_id,
            "step": job.header.get("step"),
            "seq": job.header.get("seq"),
            "layer": job.header.get("layer"),
            "rpc_id": job.rpc_id,
            "batch_size": job.header.get("batch_size"),
            "batch_row_index": job.header.get("batch_row_index"),
            "batch_rpc_id": job.header.get("batch_rpc_id"),
            "recv_delay_ms": job.recv_delay_ms,
            "q_recv_ms": job.recv_ms,
            "q_h2d_ms": q_h2d_ms,
            "api_ms": resp.get("api_ms"),
            "kernel_ms": resp.get("kernel_ms"),
            "queue_ms": resp.get("queue_ms"),
            "source_ms": source_ms,
            "d2h_o_ms": (t_d2h_o - t_d2h0) * 1000.0,
            "lse_d2h_ms": (t_d2h_lse - t_d2h_o) * 1000.0,
            "sync_ms": (t_done - t_d2h_lse) * 1000.0,
        }
        with path.open("a", buffering=1) as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def execute_remote_attention(job, stream, cache):
    """Q H2D -> frozen-KV attention -> O/LSE D2H. Returns (resp, o_cpu, lse_cpu).

    Runs on a pool worker thread. Performs no socket I/O: the caller sends the
    response, so a slow TCP peer can never hold a CUDA stream.
    """
    t_src = time.perf_counter()
    from contextlib import nullcontext

    import flashinfer

    hub = job.hub
    scheduler = hub.scheduler
    ctx = job.ctx
    header = job.header
    q_cpu = job.q_cpu
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    indices = scheduler.req_to_token_pool.req_to_token[
        ctx.req.kv.req_pool_idx, : ctx.kv_len
    ]
    layer = int(header["layer"])
    k_buf = pool.get_key_buffer(layer)
    v_buf = pool.get_value_buffer(layer)

    current_dev = torch.cuda.current_device()
    t_ctx0 = time.perf_counter()

    stream_ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    if PROFILE:
        pre_ev = torch.cuda.Event(enable_timing=True)
        q_ev = torch.cuda.Event(enable_timing=True)
        kv_ev = torch.cuda.Event(enable_timing=True)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)

    with stream_ctx:
        if PROFILE:
            pre_ev.record()
        t_q0 = time.perf_counter()
        q_gpu = cache.get("q_gpu")
        if (
            q_gpu is None
            or q_gpu.shape != q_cpu.shape
            or q_gpu.dtype != q_cpu.dtype
            or q_gpu.device.index != current_dev
        ):
            q_gpu = torch.empty(q_cpu.shape, dtype=q_cpu.dtype, device=current_dev)
            cache["q_gpu"] = q_gpu
        q_gpu.copy_(q_cpu, non_blocking=True)
        q = q_gpu
        if PROFILE:
            q_ev.record()
        q_h2d_ms = (time.perf_counter() - t_q0) * 1000.0
        if ctx.contiguous:
            k = k_buf[ctx.idx_start : ctx.idx_end]
            v = v_buf[ctx.idx_start : ctx.idx_end]
        else:
            k = k_buf[indices]
            v = v_buf[indices]
        if PROFILE:
            kv_ev.record()
            start_ev.record()
        t_api_start = time.perf_counter()
        o, lse = flashinfer.single_decode_with_kv_cache(
            q,
            k,
            v,
            kv_layout="NHD",
            pos_encoding_mode="NONE",
            sm_scale=float(header["sm_scale"]),
            return_lse=True,
        )
        t_api_return = time.perf_counter()
        if PROFILE:
            end_ev.record()
        if stream is not None:
            o.record_stream(stream)
            lse.record_stream(stream)
        t_d2h0 = time.perf_counter()
        o_cpu = o.detach().cpu()
        t_d2h_o = time.perf_counter()
        lse_cpu = lse.detach().cpu()
        t_d2h_lse = time.perf_counter()

    t_sync0 = time.perf_counter()
    if stream is not None:
        stream.synchronize()
    else:
        torch.cuda.current_stream().synchronize()
    t_done = time.perf_counter()

    if PROFILE:
        kernel_ms = start_ev.elapsed_time(end_ev)
        queue_ms = max(
            0.0,
            (t_done - t_src) * 1000.0
            - (t_api_return - t_api_start) * 1000.0
            - kernel_ms,
        )
    else:
        kernel_ms = None
        queue_ms = None
    api_ms = (t_api_return - t_api_start) * 1000.0
    source_ms = (t_done - t_src) * 1000.0
    pre_ms = 0.0  # payload receive happened on the reader thread
    setup_ms = (t_ctx0 - t_src) * 1000.0
    d2h_o_ms = (t_d2h_o - t_d2h0) * 1000.0
    d2h_lse_ms = (t_d2h_lse - t_d2h_o) * 1000.0
    sync_ms = (t_done - t_sync0) * 1000.0

    _trace_layer_diag(
        job,
        api_ms,
        q_h2d_ms,
        t_d2h0,
        t_d2h_o,
        t_d2h_lse,
        t_api_return,
        t_sync0,
        t_done,
        source_ms,
    )
    _log_first_rpc(
        job, api_ms, q_h2d_ms, t_d2h0, t_d2h_o, t_d2h_lse, t_sync0, t_done, source_ms
    )
    resp = {
        "o_shape": list(o.shape),
        "o_dtype": (
            "bfloat16"
            if o.dtype == torch.bfloat16
            else "float16"
            if o.dtype == torch.float16
            else "float32"
        ),
        "lse_shape": list(lse.shape),
        "source_ms": source_ms,
        "pre_ms": pre_ms,
        "kernel_ms": kernel_ms,
        "queue_ms": queue_ms,
        "api_ms": api_ms,
        "recv_delay_ms": job.recv_delay_ms,
        "q_recv_ms": job.recv_ms,
        "high_priority": stream is not None,
        "source_setup_ms": setup_ms,
        "q_h2d_ms": q_h2d_ms,
        "d2h_o_ms": d2h_o_ms,
        "d2h_lse_ms": d2h_lse_ms,
        "sync_ms": sync_ms,
    }
    if PROFILE:
        _append_source_trace(
            job,
            resp,
            q_h2d_ms,
            t_api_return,
            t_d2h0,
            t_d2h_o,
            t_d2h_lse,
            t_done,
            source_ms,
        )
    return resp, o_cpu, lse_cpu


def _paged_wrapper(cache, device, q_heads, kv_heads, head_dim, q_dtype):
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    key = (int(getattr(device, "index", 0) or 0), q_heads, kv_heads, head_dim, q_dtype)
    wrappers = cache.setdefault("paged_wrappers", {})
    wrapper = wrappers.get(key)
    if wrapper is None:
        ws = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", use_cuda_graph=False
        )
        wrappers[key] = wrapper
    return wrapper


def _execute_paged_batch(job, stream, cache):
    """P1.3: one paged decode kernel over independently-mapped requests."""
    import itertools

    items = job.items
    if not items:
        raise RuntimeError("empty batch")
    first = items[0]
    layer = int(first.header["layer"])
    if any(int(it.header["layer"]) != layer for it in items):
        raise RuntimeError("batch spans layers")
    sm_scale = float(first.header["sm_scale"])
    if any(float(it.header["sm_scale"]) != sm_scale for it in items):
        raise RuntimeError("batch spans sm_scale")
    q_cpu = [it.q_cpu for it in items]
    if len({tuple(x.shape) for x in q_cpu}) != 1 or len({x.dtype for x in q_cpu}) != 1:
        raise RuntimeError("batch spans q shapes")
    device = torch.cuda.current_device()
    scheduler = job.hub.scheduler
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    k_buf = pool.get_key_buffer(layer)
    v_buf = pool.get_value_buffer(layer)
    r2t = scheduler.req_to_token_pool.req_to_token
    t0 = time.perf_counter()
    indices_list = []
    lens = []
    for it in items:
        ctx = it.ctx
        n = int(ctx.kv_len)
        idx = r2t[ctx.req.kv.req_pool_idx, :n]
        indices_list.append(idx.to(dtype=torch.int32))
        lens.append(n)
    if any(n <= 0 for n in lens):
        raise RuntimeError("empty kv length")
    indices = torch.cat(indices_list).to(device=device)
    indptr = torch.tensor(
        [0] + list(itertools.accumulate(lens)),
        dtype=torch.int32,
        device=device,
    )
    last = torch.ones(len(items), dtype=torch.int32, device=device)
    with torch.cuda.stream(stream):
        q = torch.stack(q_cpu, 0).to(device=device, non_blocking=False)
        q_h2d_ms = (time.perf_counter() - t0) * 1000.0
        wrapper = _paged_wrapper(
            cache,
            device,
            int(q.shape[1]),
            int(k_buf.shape[1]),
            int(q.shape[2]),
            q.dtype,
        )
        t_plan = time.perf_counter()
        wrapper.plan(
            indptr,
            indices,
            last,
            int(q.shape[1]),
            int(k_buf.shape[1]),
            int(q.shape[2]),
            1,
            sm_scale=sm_scale,
            q_data_type=q.dtype,
            kv_data_type=k_buf.dtype,
        )
        t_run = time.perf_counter()
        o, lse = wrapper.run(
            q, (k_buf.unsqueeze(1), v_buf.unsqueeze(1)), return_lse=True
        )
        t_api = time.perf_counter()
        o_cpu = o.detach().cpu()
        t_d2h_o = time.perf_counter()
        lse_cpu = lse.detach().cpu()
        t_d2h_lse = time.perf_counter()
        stream.synchronize()
        t_done = time.perf_counter()
    total_ms = (t_done - t0) * 1000.0
    per_row_ms = total_ms / len(items)
    outs = []
    for i, it in enumerate(items):
        resp = {
            "o_shape": list(o.shape[1:]),
            "o_dtype": (
                "bfloat16"
                if o.dtype == torch.bfloat16
                else "float16"
                if o.dtype == torch.float16
                else "float32"
            ),
            "lse_shape": list(lse.shape[1:]),
            "source_ms": per_row_ms,
            "pre_ms": 0.0,
            "kernel_ms": total_ms,
            "queue_ms": None,
            "api_ms": (t_api - t_run) * 1000.0,
            "recv_delay_ms": it.recv_delay_ms,
            "q_recv_ms": it.recv_ms,
            "high_priority": stream is not None,
            "source_setup_ms": (t_run - t_plan) * 1000.0 / len(items),
            "q_h2d_ms": q_h2d_ms / len(items),
            "d2h_o_ms": (t_d2h_o - t_api) * 1000.0 / len(items),
            "d2h_lse_ms": (t_d2h_lse - t_d2h_o) * 1000.0 / len(items),
            "sync_ms": (t_done - t_d2h_lse) * 1000.0,
            "exec_ms": per_row_ms,
            "paged_batch": True,
        }
        resp["rpc_id"] = job.rpc_id
        outs.append((resp, o_cpu[i], lse_cpu[i]))
    return {"batch_size": len(items), "paged_batch": True}, outs


def execute_remote_attention_batch(job, stream, cache):
    """P1.3: one paged batch kernel; falls back to per-row single_decode."""
    if PAGED_BATCH:
        try:
            return _execute_paged_batch(job, stream, cache)
        except Exception as exc:
            try:
                print(
                    f"[handoff-hub] paged batch fallback: {exc!r}",
                    flush=True,
                )
            except Exception:
                pass
    outs = []
    for item in job.items:
        t_row = time.perf_counter()
        resp, o, lse = execute_remote_attention(item, stream, cache)
        resp["exec_ms"] = (time.perf_counter() - t_row) * 1000.0
        resp["rpc_id"] = job.rpc_id
        outs.append((resp, o, lse))
    return {"batch_size": len(outs)}, outs


def _execute_hub_job(job, stream, cache):
    if not getattr(job, "is_batch", False) and getattr(job, "header", {}).get(
        "dsv41_sparse"
    ):
        from .dsv41_remote import execute

        return execute(job, stream, cache)
    if getattr(job, "is_batch", False):
        return execute_remote_attention_batch(job, stream, cache)
    return execute_remote_attention(job, stream, cache)


class RemoteHub:
    """One listener per source scheduler; routes by handoff_id/token."""

    def __init__(self, scheduler, executor=None, stream_factory=None) -> None:
        self.scheduler = scheduler
        self.registry = HandoffRegistry()
        self.host = "127.0.0.1"
        self._closed = False
        # Active, validated connections only; the scheduler idles cooperatively
        # while any handoff RPC is in flight, so this is a count, not a flag.
        self._active = 0
        self._active_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._first_rpc_logged = False
        self._in_flight = 0
        self._conns: set[_HubConn] = set()
        self._conns_lock = threading.Lock()
        self.idle_timeout_s = float(
            os.environ.get("HANDOFF_HUB_IDLE_TIMEOUT_S", "1800")
        )
        self.read_poll_s = 0.5
        self.frame_timeout_s = _HUB_FRAME_TIMEOUT_S
        n = max(1, int(os.environ.get("HANDOFF_HUB_STREAMS", "2")))
        factory = stream_factory or _create_hub_streams
        self._pool = RemoteStreamPool(n, executor or _execute_hub_job, factory(n))
        self.rpc_completed = 0
        self.rpc_failed = 0
        self.idle_closes = 0
        self._conn_threads: set[threading.Thread] = set()
        self._conn_threads_lock = threading.Lock()
        # Frees requested by socket threads are drained on the scheduler thread.
        self._release_q: queue.Queue[HandoffContext] = queue.Queue()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((self.host, 0))
        self._listener.listen(_HUB_LISTEN_BACKLOG)
        self._listener.settimeout(_HUB_ACCEPT_TIMEOUT_S)
        self.port = int(self._listener.getsockname()[1])
        self._thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="handoff-hub"
        )
        self._thread.start()
        print(
            f"[handoff-hub] listening {self.host}:{self.port}",
            flush=True,
        )
        # Publish this rank's readiness so the driver can discover the TP width
        # and fan control commands out to every rank (one Hub per rank).
        if _CTL_DIR:
            try:
                root = Path(_CTL_DIR)
                root.mkdir(parents=True, exist_ok=True)
                path = root / f"ready.{_tp_rank(scheduler)}.json"
                tmp = path.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(
                        {
                            "rank": _tp_rank(scheduler),
                            "tp_size": _tp_size(scheduler),
                            "pid": os.getpid(),
                            "started_at": time.time(),
                        }
                    ),
                    encoding="utf-8",
                )
                os.replace(tmp, path)
            except Exception:
                pass

    def _job_finished(self, hc=None, n: int = 1) -> None:
        with self._active_lock:
            if self._in_flight > 0:
                self._in_flight = max(0, self._in_flight - int(n))
            self.scheduler._handoff_remote_active = self._in_flight > 0

    def register_handoff(self, req, meta: dict, *, on_release=None) -> dict:
        """Register a frozen request and return the target-side route info."""
        scheduler = self.scheduler
        kv_len = int(req.kv.kv_committed_len)
        kv_range = getattr(req, "_handoff_kv_range", None)
        if kv_range is not None:
            idx_start, idx_end, contiguous = kv_range
        else:
            r2t = scheduler.req_to_token_pool.req_to_token
            indices = r2t[req.kv.req_pool_idx, :kv_len]
            stream, _, _ = _create_remote_stream(
                os.environ.get("HANDOFF_REMOTE_HIGH_PRIORITY", "0") == "1"
            )
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    indices_cpu = indices.cpu()
                stream.synchronize()
            else:
                indices_cpu = indices.cpu()
            idx_start, idx_end, contiguous = _contiguous_range(indices_cpu)
        token = secrets.token_hex(16)
        ctx = self.registry.register(
            handoff_id=str(meta["rid"]),
            req=req,
            req_pool_idx=int(req.kv.req_pool_idx),
            base_len=kv_len,
            kv_len=kv_len,
            token=token,
            idx_start=idx_start,
            idx_end=idx_end,
            contiguous=contiguous,
            # Free on the scheduler thread; this callback runs in a socket
            # thread (or under the registry lock path) and must stay cheap.
            on_release=self._release_q.put if on_release is None else on_release,
        )
        print(
            f"[handoff-hub] register id={ctx.handoff_id} epoch={ctx.owner_epoch} "
            f"kv_len={ctx.kv_len} range={idx_start}:{idx_end} contig={contiguous}",
            flush=True,
        )
        return {
            "host": self.host,
            "port": self.port,
            "token": token,
            "handoff_id": ctx.handoff_id,
            "epoch": ctx.owner_epoch,
            "mode": "remote",
        }

    def drain_releases(self) -> None:
        """Free released source requests; must run on the scheduler thread."""
        while True:
            try:
                ctx = self._release_q.get_nowait()
            except queue.Empty:
                return
            try:
                _free_source_request(self.scheduler, ctx.req)
                tracked = _tracked(self.scheduler)
                if tracked.get(ctx.handoff_id) is ctx.req:
                    tracked.pop(ctx.handoff_id, None)
            except Exception as exc:
                print(f"[handoff-hub] release free failed: {exc!r}", flush=True)

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if self._closed:
                    break
                if exc.errno in (errno.EINTR, errno.ECONNABORTED):
                    continue
                print(f"[handoff-hub] accept failed: {exc!r}", flush=True)
                break
            t = threading.Thread(
                target=self._serve_conn,
                args=(conn,),
                daemon=True,
                name="handoff-hub-conn",
            )
            with self._conn_threads_lock:
                self._conn_threads.add(t)
            t.start()

    def _serve_conn(self, conn: socket.socket) -> None:
        hc = None
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(self.frame_timeout_s)
            try:
                hello = _recv_json(conn)
            except Exception:
                return
            if hello.get("cmd") != "hello" or int(hello.get("proto", 0)) != _HUB_PROTO:
                try:
                    _send_json(
                        conn,
                        {
                            "ok": False,
                            "cmd": "hello",
                            "rpc_id": hello.get("rpc_id"),
                            "error": "unsupported protocol",
                        },
                    )
                except OSError:
                    pass
                return
            conn.settimeout(None)
            hc = _HubConn(conn)
            with self._conns_lock:
                if self._closed:
                    return
                self._conns.add(hc)
                with self._active_lock:
                    self._active += 1
            with hc.send_lock:
                _send_json(
                    conn,
                    {
                        "ok": True,
                        "cmd": "hello",
                        "rpc_id": hello.get("rpc_id"),
                        "hub": self.port,
                    },
                )
            self._read_loop(hc)
        except Exception as exc:
            print(f"[handoff-hub] connection error: {exc!r}", flush=True)
        finally:
            if hc is not None:
                self._drop_conn(hc)
            else:
                try:
                    conn.close()
                except OSError:
                    pass
            with self._conn_threads_lock:
                self._conn_threads.discard(threading.current_thread())

    def _read_loop(self, hc: _HubConn) -> None:
        import select

        while not self._closed and not hc.dead:
            with hc.lock:
                inflight = hc.in_flight
            idle = time.monotonic() - hc.last_rx
            if inflight == 0 and idle > self.idle_timeout_s:
                with self._active_lock:
                    self.idle_closes += 1
                print(
                    f"[handoff-hub] idle close after {idle:.1f}s "
                    f"(rpc_completed={self.rpc_completed})",
                    flush=True,
                )
                return
            try:
                ready, _, _ = select.select([hc.sock], [], [], self.read_poll_s)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                hc.sock.settimeout(self.frame_timeout_s)
                header = _recv_json(hc.sock)
            except EOFError:
                return
            except Exception as exc:
                print(f"[handoff-hub] frame read failed: {exc!r}", flush=True)
                return
            finally:
                try:
                    hc.sock.settimeout(None)
                except OSError:
                    pass
            hc.last_rx = time.monotonic()
            cmd = header.get("cmd")
            if cmd == "attach":
                self._handle_attach(hc, header)
            elif cmd == "detach":
                hc.attached.pop(str(header.get("handoff_id", "")), None)
            elif cmd == "attn":
                self._handle_attn(hc, header)
            elif cmd == "attn_batch":
                self._handle_attn_batch(hc, header)
            elif cmd in ("imported", "commit", "done"):
                self._handle_state(hc, cmd, header)
            else:
                self._send(
                    hc,
                    {
                        "ok": False,
                        "cmd": "error",
                        "rpc_id": header.get("rpc_id"),
                        "error": f"unknown cmd {cmd!r}",
                    },
                )
                return

    def _handle_state(self, hc: _HubConn, kind: str, header: dict) -> None:
        """Target-reported ownership fact for frozen-lease recovery."""
        hid = str(header.get("handoff_id", ""))
        epoch = header.get("epoch")
        ctx = self.registry.get(hid, int(epoch) if epoch is not None else None)
        if ctx is not None and hmac.compare_digest(
            str(header.get("token", "")), ctx.token
        ):
            self.registry.mark_state(hid, int(ctx.owner_epoch), kind)
        # Best-effort ack; a target that lost its controller does not wait.
        try:
            self._send(hc, {"ok": ctx is not None, "cmd": kind, "handoff_id": hid})
        except Exception:
            pass

    def _handle_attach(self, hc: _HubConn, header: dict) -> None:
        hid = str(header.get("handoff_id", ""))
        rid = header.get("rpc_id")
        epoch = header.get("epoch")
        ctx = self.registry.get(hid, int(epoch) if epoch is not None else None)
        if ctx is None or not hmac.compare_digest(
            str(header.get("token", "")), ctx.token
        ):
            self._send(
                hc,
                {
                    "ok": False,
                    "cmd": "attach",
                    "rpc_id": rid,
                    "error": "unknown handoff or bad token",
                },
            )
            return
        hc.attached[hid] = int(ctx.owner_epoch)
        self._send(
            hc,
            {
                "ok": True,
                "cmd": "attach",
                "rpc_id": rid,
                "handoff_id": hid,
                "epoch": int(ctx.owner_epoch),
            },
        )

    def _handle_attn(self, hc: _HubConn, header: dict) -> None:
        hid = str(header.get("handoff_id", ""))
        try:
            rid = int(header.get("rpc_id"))
        except (TypeError, ValueError):
            hc.dead = True
            return
        epoch = header.get("epoch")
        try:
            send_ts = float(header.get("send_ts", time.monotonic()))
        except (TypeError, ValueError):
            send_ts = time.monotonic()
        try:
            epoch_i = int(epoch) if epoch is not None else None
        except (TypeError, ValueError):
            epoch_i = None
        q_bytes = int(header.get("q_bytes", 0))
        if q_bytes < 0 or q_bytes > 64 * 1024 * 1024:
            self._send(
                hc, {"ok": False, "cmd": "attn", "rpc_id": rid, "error": "bad q_bytes"}
            )
            hc.dead = True
            return
        attached_epoch = hc.attached.get(hid)
        ctx = None
        if attached_epoch is not None and epoch_i is not None:
            if epoch_i == attached_epoch:
                ctx = self.registry.acquire(hid, epoch_i)
        if ctx is None:
            try:
                hc.sock.settimeout(self.frame_timeout_s)
                _recv_exact(hc.sock, q_bytes)
            except Exception as exc:
                print(f"[handoff-hub] drain failed: {exc!r}", flush=True)
                hc.dead = True
                return
            finally:
                try:
                    hc.sock.settimeout(None)
                except OSError:
                    pass
            self._send(
                hc,
                {
                    "ok": False,
                    "cmd": "attn",
                    "rpc_id": rid,
                    "error": "stale or unattached handoff",
                },
            )
            return
        try:
            hc.sock.settimeout(self.frame_timeout_s)
            t_header = time.monotonic()
            q_cpu = _recv_q(hc.sock, header)
            q_recv_ms = (time.monotonic() - t_header) * 1000.0
        except Exception as exc:
            self.registry.release_ref(hid, int(epoch_i))
            print(f"[handoff-hub] q read failed: {exc!r}", flush=True)
            hc.dead = True
            return
        finally:
            try:
                hc.sock.settimeout(None)
            except OSError:
                pass
        job = None
        try:
            job = RemoteJob(
                hub=self,
                conn=hc.sock,
                hc=hc,
                send_lock=hc.send_lock,
                ctx=ctx,
                rpc_id=rid,
                epoch=int(epoch_i),
                header=header,
                q_cpu=q_cpu,
            )
            job.recv_delay_ms = (t_header - send_ts) * 1000.0
            job.recv_ms = q_recv_ms
            with hc.lock:
                hc.in_flight += 1
            with self._active_lock:
                self._in_flight += 1
                self.scheduler._handoff_remote_active = True
            self._pool.submit(job)
        except Exception:
            if job is None:
                self.registry.release_ref(hid, int(epoch_i))
            else:
                with hc.lock:
                    hc.in_flight = max(0, hc.in_flight - 1)
                with self._active_lock:
                    self._in_flight = max(0, self._in_flight - 1)
                    self.scheduler._handoff_remote_active = self._in_flight > 0
                self.registry.release_ref(hid, int(epoch_i))
            raise

    def _handle_attn_batch(self, hc: _HubConn, header: dict) -> None:
        """P1.1: one RPC carrying several rows with independent handoff ids."""
        try:
            rid = int(header.get("rpc_id"))
        except (TypeError, ValueError):
            hc.dead = True
            return
        rows = header.get("rows")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
            self._send(
                hc,
                {
                    "ok": False,
                    "cmd": "attn_batch",
                    "rpc_id": rid,
                    "error": "bad batch rows",
                },
            )
            return
        try:
            send_ts = float(header.get("send_ts", time.monotonic()))
        except (TypeError, ValueError):
            send_ts = time.monotonic()
        total = int(header.get("total_q_bytes", -1))
        if total < 0:
            total = sum(int(r.get("q_bytes", 0)) for r in rows)
        declared = sum(int(r.get("q_bytes", 0)) for r in rows)
        if total != declared or total > 64 * 1024 * 1024:
            self._send(
                hc,
                {
                    "ok": False,
                    "cmd": "attn_batch",
                    "rpc_id": rid,
                    "error": "bad total_q_bytes",
                },
            )
            hc.dead = True
            return
        t_header = time.monotonic()
        try:
            raw = _recv_exact(hc.sock, total)
            q_recv_ms = (time.monotonic() - t_header) * 1000.0
        except Exception as exc:
            print(f"[handoff-hub] batch q read failed: {exc!r}", flush=True)
            hc.dead = True
            return
        jobs = []
        acquired = []
        offset = 0
        try:
            for idx, row in enumerate(rows):
                hid = str(row.get("handoff_id", ""))
                try:
                    epoch_i = int(row.get("epoch"))
                except (TypeError, ValueError):
                    raise ValueError(f"row {idx}: bad epoch")
                if hc.attached.get(hid) != epoch_i:
                    raise ValueError(f"row {idx}: stale or unattached handoff")
                ctx = self.registry.acquire(hid, epoch_i)
                if ctx is None:
                    raise ValueError(f"row {idx}: handoff not registered")
                acquired.append((hid, epoch_i))
                qb = int(row.get("q_bytes", 0))
                if qb <= 0 or offset + qb > len(raw):
                    raise ValueError(f"row {idx}: bad q_bytes")
                q_cpu = _q_cpu_from_bytes(row, raw[offset : offset + qb])
                offset += qb
                row_header = dict(row)
                row_header["batch_size"] = len(rows)
                row_header["batch_rpc_id"] = rid
                row_header["batch_row_index"] = idx
                job = RemoteJob(
                    hub=self,
                    conn=None,
                    hc=None,
                    send_lock=None,
                    ctx=ctx,
                    rpc_id=rid,
                    epoch=epoch_i,
                    header=row_header,
                    q_cpu=q_cpu,
                )
                job.recv_delay_ms = (t_header - send_ts) * 1000.0
                job.recv_ms = q_recv_ms
                jobs.append(job)
            if offset != total:
                raise ValueError(f"q bytes consumed {offset} != {total}")
            batch = RemoteBatchJob(
                hub=self,
                conn=hc.sock,
                hc=hc,
                send_lock=hc.send_lock,
                items=jobs,
                rpc_id=rid,
            )
            with hc.lock:
                hc.in_flight += len(jobs)
            with self._active_lock:
                self._in_flight += len(jobs)
                self.scheduler._handoff_remote_active = True
            self._pool.submit(batch)
        except Exception as exc:
            for hid, epoch_i in acquired:
                try:
                    self.registry.release_ref(hid, epoch_i)
                except Exception:
                    pass
            self._send(
                hc, {"ok": False, "cmd": "attn_batch", "rpc_id": rid, "error": str(exc)}
            )
            return

    def _send(self, hc: _HubConn, obj: dict) -> bool:
        if hc.dead:
            return False
        try:
            with hc.send_lock:
                _send_json(hc.sock, obj)
            return True
        except OSError:
            hc.dead = True
            return False

    def _drop_conn(self, hc: _HubConn) -> None:
        hc.dead = True
        with self._conns_lock:
            self._conns.discard(hc)
            with self._active_lock:
                self._active = max(0, self._active - 1)
        try:
            hc.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            hc.sock.close()
        except OSError:
            pass

    def stats(self) -> dict:
        with self._conns_lock:
            conns = list(self._conns)
            active = len(conns)
        with self._active_lock:
            inflight = self._in_flight
        return {
            "active_conns": active,
            "attached_handoffs": sum(len(c.attached) for c in conns),
            "in_flight": inflight,
            "idle_closes": self.idle_closes,
            "rpc_completed": self.rpc_completed,
            "rpc_failed": self.rpc_failed,
            "pool": self._pool.stats(),
        }

    def close(self) -> None:
        """Stop accepting, close connections, and join hub threads."""
        if self._closed:
            return
        self._closed = True
        try:
            self._listener.close()
        except OSError:
            pass
        with self._conns_lock:
            conns = list(self._conns)
            self._conns.clear()
            with self._active_lock:
                self._active = 0
        for hc in conns:
            hc.dead = True
            try:
                hc.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                hc.sock.close()
            except OSError:
                pass
        self._pool.close()
        # The accept loop only notices the closed listener when its accept
        # timeout expires (closing a listening socket does not wake a blocked
        # accept on Linux), so allow one full timeout plus margin.
        self._thread.join(timeout=_HUB_ACCEPT_TIMEOUT_S + 1.0)
        with self._conn_threads_lock:
            threads = list(self._conn_threads)
        for t in threads:
            t.join(timeout=1.0)
        with self._active_lock:
            self.scheduler._handoff_remote_active = False


_HUB_CREATE_LOCK = threading.Lock()


def get_remote_hub(scheduler) -> RemoteHub:
    hub = getattr(scheduler, "_handoff_hub", None)
    if hub is not None:
        return hub
    with _HUB_CREATE_LOCK:
        hub = getattr(scheduler, "_handoff_hub", None)
        if hub is None:
            hub = RemoteHub(scheduler)
            scheduler._handoff_hub = hub
        return hub


def close_handoff_hub(scheduler) -> None:
    hub = getattr(scheduler, "_handoff_hub", None)
    if hub is not None:
        hub.close()
        scheduler._handoff_hub = None
        if _CTL_DIR:
            try:
                (Path(_CTL_DIR) / f"ready.{_tp_rank(scheduler)}.json").unlink()
            except OSError:
                pass


def shutdown_handoff(scheduler) -> None:
    """Fence plugin workers before scheduler/model resources are torn down."""
    errors = []
    actions = [lambda: close_handoff_hub(scheduler), _reset_target_states]
    actions.extend(
        lambda req=req: stop_source_transfer(req)
        for req in list(_tracked(scheduler).values())
    )
    for action in actions:
        try:
            action()
        except Exception as exc:
            errors.append(repr(exc))
    if errors:
        raise RuntimeError(f"handoff workers could not shut down safely: {errors}")


def _start_gil_probe():
    global _GIL_PROBE_STARTED
    if _GIL_PROBE_STARTED:
        return
    _GIL_PROBE_STARTED = True

    def loop():
        last = time.perf_counter()
        while True:
            time.sleep(0.001)
            now = time.perf_counter()
            gap = max(0.0, (now - last - 0.001) * 1000.0)
            _GIL_GAPS.append(gap)
            if len(_GIL_GAPS) > 20000:
                del _GIL_GAPS[:10000]
            last = now

    threading.Thread(target=loop, daemon=True, name="handoff-gil-probe").start()


_REMOTE_STREAM_CACHE: dict[bool, tuple] = {}


def _create_remote_stream(high_priority: bool):
    """Create a non-blocking CUDA stream (legacy default stream must not
    implicitly serialize it with the scheduler's prefill stream).

    The stream is cached per process and reused across handoff connections:
    cudaStreamCreateWithPriority is a host driver call that can block ~130 ms
    when the source is busy with prefill, and creating it per connection puts
    that cost on every handoff's first-token path.
    """
    if not high_priority:
        return None, None, None
    cached = _REMOTE_STREAM_CACHE.get(True)
    if cached is not None:
        return cached
    try:
        from cuda.bindings import runtime as cudart

        err, raw = cudart.cudaStreamCreateWithPriority(cudart.cudaStreamNonBlocking, -1)
        if int(err) != 0:
            raise RuntimeError(f"cudaStreamCreateWithPriority failed: {err}")
        try:
            _err, flags = cudart.cudaStreamGetFlags(raw)
        except Exception:
            flags = None
        result = (torch.cuda.ExternalStream(int(raw)), raw, flags)
        _REMOTE_STREAM_CACHE[True] = result
        return result
    except Exception:
        result = (torch.cuda.Stream(priority=-1), None, None)
        _REMOTE_STREAM_CACHE[True] = result
        return result


_HUB_STREAM_CACHE: dict[tuple, list] = {}
_HUB_STREAM_LOCK = threading.Lock()


def _create_hub_streams(n: int):
    """Create (or reuse) ``n`` highest-priority non-blocking CUDA streams.

    Streams are created once per process on the hub thread. Creating them per
    job or per connection puts a 100+ ms driver call on the first-token path
    when the source is busy with prefill (M1 root cause).
    """
    n = max(1, int(n))
    priority = -1 if os.environ.get("HANDOFF_REMOTE_HIGH_PRIORITY", "0") == "1" else 0
    try:
        _device = int(torch.cuda.current_device())
    except Exception:
        _device = -1
    key = (_device, n, priority)
    with _HUB_STREAM_LOCK:
        cached = _HUB_STREAM_CACHE.get(key)
        if cached is not None:
            return list(cached)
        try:
            from cuda.bindings import runtime as cudart

            slots = []
            for _ in range(n):
                err, raw = cudart.cudaStreamCreateWithPriority(
                    cudart.cudaStreamNonBlocking, priority
                )
                if int(err) != 0:
                    raise RuntimeError(f"cudaStreamCreateWithPriority failed: {err}")
                try:
                    _err, flags = cudart.cudaStreamGetFlags(raw)
                except Exception:
                    flags = None
                slots.append((torch.cuda.ExternalStream(int(raw)), raw, flags))
        except Exception as exc:
            print(
                f"[handoff-hub] high-priority streams unavailable: {exc!r}",
                flush=True,
            )
            slots = [
                (torch.cuda.Stream(priority=priority), None, None) for _ in range(n)
            ]
        _HUB_STREAM_CACHE[key] = list(slots)
        return slots


class RemoteStreamPool:
    """FIFO job queue with one pre-created stream per worker.

    Job contract:
    - ``run_job(job, stream, cache)`` may send results itself by calling
      ``job.reply(...)`` and returning ``None``, or return
      ``(resp, o, lse)`` and let the pool send them.
    - On exception the pool calls ``job.fail(exc)`` and then ``job.finish()``,
      exactly once each per job. ``fail()`` and ``finish()`` must not raise,
      and ``fail()`` must not call ``finish()``.
    - ``completed`` counts succeeded and failed jobs alike; ``failed`` is the
      failed subset.
    """

    def __init__(self, n, run_job, streams, name="handoff-exec"):
        self.n = max(1, int(n))
        self.run_job = run_job
        self._slots = list(streams)[: self.n]
        if len(self._slots) < self.n:
            self._slots += [(None, None, None)] * (self.n - len(self._slots))
        self._q: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._busy = 0
        self._queued = 0
        self._max_parallel = 0
        self.completed = 0
        self.failed = 0
        self.max_queue_ms = 0.0
        self.sum_exec_ms = 0.0
        try:
            self._device = torch.cuda.current_device()
        except Exception:
            self._device = None
        self._threads = [
            threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True,
                name=f"{name}-{i}",
            )
            for i in range(self.n)
        ]
        for t in self._threads:
            t.start()

    def submit(self, job) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("RemoteStreamPool is closed")
            job.t_enqueue = time.perf_counter()
            self._queued += 1
            self._q.put(job)

    def _worker(self, idx: int) -> None:
        if self._device is not None:
            try:
                torch.cuda.set_device(self._device)
            except Exception:
                pass
        cache: dict = {}
        while True:
            job = self._q.get()
            if job is None:
                return
            now = time.perf_counter()
            queue_ms = (now - getattr(job, "t_enqueue", now)) * 1000.0
            t0 = time.perf_counter()
            with self._lock:
                self._queued -= 1
                self._busy += 1
                self._max_parallel = max(self._max_parallel, self._busy)
                self.max_queue_ms = max(self.max_queue_ms, queue_ms)
            ok = True
            try:
                job.stream = self._slots[idx][0]
                result = self.run_job(job, job.stream, cache)
                if result is not None:
                    if getattr(job, "is_batch", False):
                        resp, outs = result
                        resp["queue_ms"] = queue_ms
                        resp["exec_ms"] = (time.perf_counter() - t0) * 1000.0
                        for row_resp, _o, _lse in outs:
                            row_resp["queue_ms"] = queue_ms
                        job.reply_batch(resp, outs)
                    else:
                        resp, o, lse = result
                        resp["queue_ms"] = queue_ms
                        resp["exec_ms"] = (time.perf_counter() - t0) * 1000.0
                        job.reply(resp, o, lse)
            except Exception as exc:
                ok = False
                try:
                    job.fail(exc)
                except Exception:
                    pass
            finally:
                try:
                    job.finish()
                except Exception:
                    pass
                with self._lock:
                    self._busy -= 1
                    self.completed += 1
                    if not ok:
                        self.failed += 1
                    self.sum_exec_ms += (time.perf_counter() - t0) * 1000.0

    def stats(self) -> dict:
        with self._lock:
            done = self.completed
            return {
                "n": self.n,
                "busy": self._busy,
                "queued": self._queued,
                "max_parallel": self._max_parallel,
                "completed": done,
                "failed": self.failed,
                "max_queue_ms": round(self.max_queue_ms, 3),
                "avg_exec_ms": round(self.sum_exec_ms / done, 3) if done else 0.0,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for _ in range(self.n):
                self._q.put(None)
        deadline = time.monotonic() + 1.0
        for t in self._threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(t.is_alive() for t in self._threads):
            raise RuntimeError("handoff RPC workers did not stop before model teardown")


def _gil_stats():
    vals = sorted(_GIL_GAPS[-5000:])
    if not vals:
        return None

    def q(p):
        return vals[min(len(vals) - 1, int(p * (len(vals) - 1)))]

    return {
        "n": len(vals),
        "p50_ms": q(0.50),
        "p99_ms": q(0.99),
        "max_ms": vals[-1],
    }


def _get_te():
    """Lazily create one Mooncake TransferEngine per scheduler process.

    The 5090 host has no RDMA HCA, so use Mooncake's TCP transport. The API is
    the same one used by the Mooncake PD connector; only the protocol differs.
    """
    global _TE
    if _TE is not None:
        return _TE
    with _TE_LOCK:
        if _TE is None:
            from mooncake.engine import TransferEngine

            # The scheduler child process does not always inherit MC_* from the
            # launcher, so mirror the handoff knobs here (before TE init).
            for dst, src in (
                ("MC_TCP_LANES_PER_PEER", "HANDOFF_MC_TCP_LANES"),
                ("MC_NUM_TCP_IO_THREADS", "HANDOFF_MC_TCP_IO_THREADS"),
                ("MC_TCP_SLICE_SIZE", "HANDOFF_MC_TCP_SLICE"),
            ):
                val = os.environ.get(src)
                if val:
                    os.environ[dst] = str(val)
            te = TransferEngine()
            rc = te.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "")
            if rc != 0:
                raise RuntimeError(f"Mooncake TransferEngine.initialize failed: {rc}")
            _TE = te
        return _TE


class RegistrationError(RuntimeError):
    pass


class _RegRange:
    __slots__ = ("addrs", "sizes", "spans", "key", "owner", "label", "refs", "rc")

    def __init__(self, addrs, sizes, spans, owner, label):
        self.addrs = addrs
        self.sizes = sizes
        self.spans = spans
        self.key = tuple(zip(addrs, sizes))
        self.owner = owner
        self.label = label
        self.refs = 1
        self.rc = None


class RegistrationManager:
    """Process-wide Mooncake TE register/unregister entry point.

    Register/unregister are serialized; identical registrations (same set of
    ``(addr, size)`` pairs, in any order) share a refcount, and any partial
    overlap is rejected before TE is called. This is what turns
    ``-7 ERR_ADDRESS_OVERLAPPED`` into a diagnosable Python error.

    Lease contract: one owner is one registration lifetime. ``unregister(owner)``
    releases every registration held by that owner, so the same owner
    registering twice releases both on a single ``unregister`` call.

    A failed TE unregistration drops the range from the live set but retains it
    in ``stale``: overlapping registrations are rejected until ``clear_stale()``
    is called, and the failure is reported through ``last_unregister_error``.
    """

    def __init__(self, te=None):
        self._te = te
        self._lock = threading.Lock()
        self._ranges: list[_RegRange] = []
        self._stale: list[_RegRange] = []
        self._by_owner: dict[str, list[_RegRange]] = {}
        self.registrations = 0
        self.unregistrations = 0
        self.overlap_rejections = 0
        self.stale_rejections = 0
        self.unregister_failures = 0
        self.bytes_live = 0
        self.max_bytes_live = 0
        self.last_unregister_error = None

    def _engine(self):
        return self._te if self._te is not None else _get_te()

    @staticmethod
    def _overlap(a, b):
        return a[0] < b[1] and b[0] < a[1]

    def register(self, owner, addrs, sizes, label=""):
        """Register CPU KV memory for ``owner`` through Mooncake TE.

        The lists actually stored and sent to TE are normalized: zero-size
        entries are dropped and the rest are sorted by address. Callers that
        care about the original order (for example to line the addresses up
        with a peer's list) must keep their own input lists.
        """
        owner = str(owner)
        addrs = [int(a) for a in addrs]
        sizes = [int(s) for s in sizes]
        if len(addrs) != len(sizes):
            raise ValueError("addrs/sizes length mismatch")
        if not addrs:
            raise ValueError("empty addrs")
        if any(s < 0 for s in sizes):
            raise ValueError("negative size")
        pairs = sorted((a, s) for a, s in zip(addrs, sizes) if s > 0)
        if not pairs:
            raise ValueError("all sizes are zero")
        addrs = [p[0] for p in pairs]
        sizes = [p[1] for p in pairs]
        key = tuple(pairs)
        spans = [(a, a + s) for a, s in pairs]
        for prev, cur in zip(spans, spans[1:]):
            if cur[0] < prev[1]:
                raise RegistrationError(
                    f"registration spans overlap within the same call: {spans}"
                )
        # Resolve (and lazily initialize) TE outside the manager lock: first
        # initialization can take ~0.4 s of topology discovery.
        engine = self._engine()
        with self._lock:
            for rr in self._ranges:
                if rr.key == key:
                    rr.refs += 1
                    self._by_owner.setdefault(owner, []).append(rr)
                    return rr
            for rr in self._stale:
                for span in spans:
                    for live in rr.spans:
                        if self._overlap(span, live):
                            self.stale_rejections += 1
                            raise RegistrationError(
                                f"TE range [{span[0]},{span[1]}) overlaps a range "
                                f"whose unregistration failed "
                                f"(owner={rr.owner} label={rr.label} rc={rr.rc}); "
                                f"call clear_stale() only after verifying TE state"
                            )
            for rr in self._ranges:
                for span in spans:
                    for live in rr.spans:
                        if self._overlap(span, live):
                            self.overlap_rejections += 1
                            raise RegistrationError(
                                f"TE range [{span[0]},{span[1]}) overlaps live "
                                f"registration [{live[0]},{live[1]}) "
                                f"owner={rr.owner} label={rr.label}"
                            )
            rc = int(engine.batch_register_memory(addrs, sizes))
            if rc != 0:
                raise RegistrationError(
                    f"batch_register_memory failed: rc={rc} owner={owner} "
                    f"label={label} first_addr={addrs[0]}"
                )
            rr = _RegRange(addrs, sizes, spans, owner, label)
            self._ranges.append(rr)
            self._by_owner.setdefault(owner, []).append(rr)
            self.registrations += 1
            nbytes = sum(sizes)
            self.bytes_live += nbytes
            self.max_bytes_live = max(self.max_bytes_live, self.bytes_live)
            return rr

    def unregister(self, owner):
        """Release every registration held by ``owner``.

        Returns True when the owner held at least one registration, even if a
        TE unregistration failed; such failures are reported through
        ``last_unregister_error`` and retained in ``stats()["stale"]``.
        """
        owner = str(owner)
        # Check ownership before resolving the engine: an unknown owner must not
        # pay (or trigger) TE initialization.
        with self._lock:
            if owner not in self._by_owner:
                return False
        engine = self._engine()
        with self._lock:
            held = self._by_owner.pop(owner, [])
            if not held:
                return False
            counts: dict[int, int] = {}
            for rr in held:
                counts[id(rr)] = counts.get(id(rr), 0) + 1
            to_unregister = []
            for rr in {id(r): r for r in held}.values():
                rr.refs -= counts[id(rr)]
                if rr.refs <= 0:
                    try:
                        self._ranges.remove(rr)
                    except ValueError:
                        pass
                    self.bytes_live -= sum(rr.sizes)
                    to_unregister.append(rr)
            for rr in to_unregister:
                try:
                    rc = int(engine.batch_unregister_memory(rr.addrs))
                except Exception as exc:
                    rc = -1
                    print(f"[registration] unregister raised: {exc!r}", flush=True)
                if rc != 0:
                    rr.rc = rc
                    rr.owner = owner
                    self._stale.append(rr)
                    self.unregister_failures += 1
                    self.last_unregister_error = (
                        f"rc={rc} owner={owner} label={rr.label}"
                    )
                    print(
                        f"[registration] unregister failed rc={rc} "
                        f"owner={owner} label={rr.label}",
                        flush=True,
                    )
                else:
                    self.unregistrations += 1
            return True

    def release_safe(self, owner):
        """Keep storage alive until native unregister succeeds, including retry."""
        owner = str(owner)
        with self._lock:
            live = owner in self._by_owner
        if live:
            self.unregister(owner)
        else:
            with self._lock:
                stale = [rr for rr in self._stale if rr.owner == owner]
                for rr in stale:
                    try:
                        rc = int(self._engine().batch_unregister_memory(rr.addrs))
                    except Exception:
                        rc = -1
                    if rc == 0:
                        self._stale.remove(rr)
                        self.unregistrations += 1
                if not self._stale:
                    self.last_unregister_error = None
        with self._lock:
            return owner not in self._by_owner and not any(
                rr.owner == owner for rr in self._stale
            )

    def clear_stale(self):
        """Forget ranges whose TE unregistration failed. Returns the count.

        Only call this after verifying the corresponding TE state by hand;
        otherwise a later registration may reuse an address TE still holds.
        """
        with self._lock:
            n = len(self._stale)
            self._stale.clear()
            self.last_unregister_error = None
            return n

    @staticmethod
    def _snapshot(rr, with_rc=False):
        entry = {
            "owner": rr.owner,
            "label": rr.label,
            "addrs": list(rr.addrs),
            "sizes": list(rr.sizes),
            "refs": rr.refs,
            "bytes": sum(rr.sizes),
        }
        if with_rc:
            entry["rc"] = rr.rc
        return entry

    def ranges(self):
        with self._lock:
            return {
                "live": [self._snapshot(rr) for rr in self._ranges],
                "stale": [self._snapshot(rr, True) for rr in self._stale],
            }

    def stats(self):
        with self._lock:
            return {
                "live": len(self._ranges),
                "owners": len(self._by_owner),
                "bytes_live": self.bytes_live,
                "max_bytes_live": self.max_bytes_live,
                "registrations": self.registrations,
                "unregistrations": self.unregistrations,
                "overlap_rejections": self.overlap_rejections,
                "stale_rejections": self.stale_rejections,
                "unregister_failures": self.unregister_failures,
                "stale": len(self._stale),
                "last_unregister_error": self.last_unregister_error,
            }


_REGISTRATION_MANAGER = None
_REGISTRATION_MANAGER_LOCK = threading.Lock()


def _get_registration_manager(te=None):
    global _REGISTRATION_MANAGER
    with _REGISTRATION_MANAGER_LOCK:
        if _REGISTRATION_MANAGER is None:
            _REGISTRATION_MANAGER = RegistrationManager(te=te)
        return _REGISTRATION_MANAGER


# ---------------------------------------------------------------------------
# small framing helpers
# ---------------------------------------------------------------------------
def _send_json(sock, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise EOFError("peer closed during transfer")
        buf.extend(part)
    return bytes(buf)


def _recv_json(sock):
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if size > 64 * 1024 * 1024:
        raise ValueError("control JSON too large")
    return json.loads(_recv_exact(sock, size))


def _send_tensor(sock, tensor: torch.Tensor):
    if tensor.is_cuda:
        tensor = tensor.detach().cpu()
    raw = tensor.contiguous().view(torch.uint8).reshape(-1).numpy()
    sock.sendall(memoryview(raw))


def _recv_into(sock, view):
    offset = 0
    while offset < len(view):
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise EOFError("peer closed during tensor receive")
        offset += n


def _recv_tensor(sock, shape, dtype, pinned=False):
    try:
        tensor = torch.empty(
            tuple(int(x) for x in shape), dtype=dtype, pin_memory=pinned
        )
    except Exception:
        tensor = torch.empty(tuple(int(x) for x in shape), dtype=dtype)
    view = tensor.view(torch.uint8).reshape(-1).numpy()
    _recv_into(sock, memoryview(view))
    return tensor


# ---------------------------------------------------------------------------
# socket server (source side) / client (target side)
# ---------------------------------------------------------------------------
def _start_transfer_server(scheduler, meta: dict, kv_cpu) -> dict:
    token = secrets.token_hex(16)
    scheduler._handoff_transfer_active = True
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(180)
    port = listener.getsockname()[1]

    def serve():
        conn = None
        t0 = time.perf_counter()
        nbytes = 0
        try:
            conn, _ = listener.accept()
            conn.settimeout(600)
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)
            hello = _recv_json(conn)
            if not hmac.compare_digest(str(hello.get("token", "")), token):
                raise ValueError("bad transfer token")
            _send_json(conn, meta)
            for layer in kv_cpu:
                for k, v in layer:
                    nbytes += k.numel() * k.element_size()
                    nbytes += v.numel() * v.element_size()
                    _send_tensor(conn, k)
                    _send_tensor(conn, v)
            _TRANSFER_STATS[token] = {
                "bytes": nbytes,
                "send_ms": (time.perf_counter() - t0) * 1000.0,
            }
            try:
                conn.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        except Exception as exc:
            _TRANSFER_ERRORS[token] = repr(exc)
        finally:
            scheduler._handoff_transfer_active = False
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            try:
                listener.close()
            except OSError:
                pass

    threading.Thread(target=serve, daemon=True, name="handoff-kv-send").start()
    return {"host": "127.0.0.1", "port": port, "token": token}


def _recv_transfer(transfer: dict):
    host = str(transfer["host"])
    port = int(transfer["port"])
    token = str(transfer["token"])
    with socket.create_connection((host, port), timeout=600) as conn:
        conn.settimeout(600)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        _send_json(conn, {"token": token})
        meta = _recv_json(conn)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[meta["dtype"]]
        hkv = int(meta["kv_heads"])
        head_dim = int(meta["head_dim"])
        chunks = meta["chunks"]
        kv_len = int(meta["kv_committed_len"])
        if not chunks or sum(int(n) for n in chunks[0]) != kv_len:
            raise ValueError(
                f"chunk manifest mismatch: kv_len={kv_len} chunks0={chunks[:1]}"
            )
        kv_cpu = []
        for layer_chunks in chunks:
            layer = []
            for n in layer_chunks:
                k = _recv_tensor(conn, (int(n), hkv, head_dim), dtype)
                v = _recv_tensor(conn, (int(n), hkv, head_dim), dtype)
                layer.append([k, v])
            kv_cpu.append(layer)
    return meta, kv_cpu


def _flatten_cpu(kv_cpu):
    tensors = []
    for layer in kv_cpu:
        for k, v in layer:
            tensors.append(k)
            tensors.append(v)
    return tensors


def _open_transfer_listener(timeout_s=1.0):
    """Create the loopback control listener shared by the source servers."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(timeout_s)
    return listener, int(listener.getsockname()[1])


def _start_mooncake_server(scheduler, req, meta: dict, kv_cpu) -> dict:
    """Source side: register CPU KV with Mooncake TE and serve addresses.

    A tiny TCP control channel exchanges the target's TE session and registered
    buffer addresses; the KV bytes themselves go through Mooncake TE
    (``batch_transfer_sync_write``, TCP transport on this host).
    """
    token = secrets.token_hex(16)
    meta = _with_transfer_ticket(token, meta)
    te = _get_te()
    tensors = _flatten_cpu(kv_cpu)
    addrs = [int(t.data_ptr()) for t in tensors]
    sizes = [int(t.numel() * t.element_size()) for t in tensors]
    mgr = _get_registration_manager()
    mgr.register(token, addrs, sizes, label=f"mooncake:{meta.get('rid')}")

    listener = None
    cancel = None
    try:
        listener, port = _open_transfer_listener()
        cancel = threading.Event()
        req._handoff_transfer_cancel = (listener, cancel)
        _mark_source_transfer(scheduler, token, True)

        def serve():
            hold = tensors  # keep CPU KV alive until TE unregisters
            hold_kv = kv_cpu
            conn = None
            t0 = time.perf_counter()
            sent = 0
            safe_cleanup = True
            try:
                deadline = time.monotonic() + 180.0
                while not cancel.is_set() and time.monotonic() < deadline:
                    try:
                        conn, _ = listener.accept()
                        break
                    except socket.timeout:
                        continue
                    except OSError:
                        if cancel.is_set():
                            break
                        raise
                if conn is None:
                    if cancel.is_set():
                        return
                    raise TimeoutError("transfer listener timed out")
                req._handoff_transfer_conn = conn
                if cancel.is_set():
                    return
                conn.settimeout(600)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hello = _recv_json(conn)
                if not hmac.compare_digest(str(hello.get("token", "")), token):
                    raise ValueError("bad transfer token")
                _send_json(conn, meta)
                reply = _recv_json(conn)
                session = str(reply["session"])
                peer = [int(x) for x in reply["addrs"]]
                if len(peer) != len(addrs):
                    raise ValueError(
                        f"address count mismatch: src={len(addrs)} dst={len(peer)}"
                    )
                for i in range(0, len(addrs), 64):
                    safe_cleanup = False
                    rc = te.batch_transfer_sync_write(
                        session,
                        addrs[i : i + 64],
                        peer[i : i + 64],
                        sizes[i : i + 64],
                    )
                    if rc != 0:
                        raise RuntimeError(f"Mooncake batch_transfer failed: rc={rc}")
                    safe_cleanup = True
                    sent += sum(sizes[i : i + 64])
                _TRANSFER_STATS[token] = {
                    "bytes": sent,
                    "send_ms": (time.perf_counter() - t0) * 1000.0,
                    "engine": "mooncake_tcp",
                    "rc": 0,
                }
                _send_json(conn, {"ok": True})
            except Exception as exc:
                if not cancel.is_set():
                    _TRANSFER_ERRORS[token] = repr(exc)
                    try:
                        _send_json(conn, {"ok": False, "error": repr(exc)})
                    except Exception:
                        pass
            finally:
                if getattr(req, "_handoff_transfer_conn", None) is conn:
                    req._handoff_transfer_conn = None
                _mark_source_transfer(scheduler, token, False)
                _publish_transfer_complete(token, safe_cleanup)
                _release_transfer_buffers(
                    token,
                    kv_cpu,
                    "source transfer completion unknown",
                    remote_complete=safe_cleanup,
                    gpu_safe=not getattr(req, "_handoff_transfer_unsafe", False),
                )
                if getattr(req, "_handoff_transfer_cancel", None) is not None:
                    handle = req._handoff_transfer_cancel
                    if handle[1] is cancel:
                        req._handoff_transfer_cancel = None
                del hold
                del hold_kv
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass
                try:
                    listener.close()
                except OSError:
                    pass

        thread = threading.Thread(target=serve, daemon=True, name="handoff-kv-mooncake")
        thread.start()
        req._handoff_transfer_thread = thread
    except Exception:
        # Rollback everything opened after the TE registration: if the thread
        # never started, its finally block cannot do this for us.
        _release_transfer_buffers(
            token, kv_cpu, "source setup rollback", remote_complete=True
        )
        _mark_source_transfer(scheduler, token, False)
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if cancel is not None:
            handle = getattr(req, "_handoff_transfer_cancel", None)
            if handle is not None and handle[1] is cancel:
                req._handoff_transfer_cancel = None
        raise
    return {"host": "127.0.0.1", "port": port, "token": token, "mode": "mooncake"}


def _start_remote_attention_service(scheduler, req, meta: dict) -> dict:
    """Register the frozen request with the scheduler-level RemoteHub."""
    hub = get_remote_hub(scheduler)
    # The hub's connection handler owns _handoff_remote_active: a registered
    # but never-connected handoff must not pin the scheduler's idle path.
    return hub.register_handoff(req, meta)


def _fast_copy_streams(device, n, *, role="source"):
    """Cached copy stream(s) per device (P5.2 ablation knob)."""
    n = max(1, int(n))
    with _FAST_STREAM_LOCK:
        key = (str(device), role)
        streams = _FAST_STREAMS.get(key)
        if streams is None:
            streams = []
            _FAST_STREAMS[key] = streams
        while len(streams) < n:
            streams.append(torch.cuda.Stream(device=device))
        return list(streams[:n])


def _fast_copy_stream(device):
    """One cached copy stream per device (P5.1 fast freeze)."""
    return _fast_copy_streams(device, 1)[0]


def _start_layer_gpu_server(scheduler, req, meta, remote):
    """P5.1/P5.2 source: pinned layer staging -> TE, reading frozen GPU KV.

    Freeze returns after this thread is started; the target can import and run
    remote attention while layers are still being copied/transferred.

    ``HANDOFF_FAST_BUFFERS`` + ``HANDOFF_FAST_COPY_STREAMS`` +
    ``HANDOFF_FAST_ASYNC`` select the staging pipeline.  With async TE, layer
    L's GPU->pinned copy overlaps layer L-1's Mooncake write.  The target may
    advertise a bounded ring of layer slots (P5.2-B); the source reuses a
    target slot only after the target acknowledges its H2D copy.
    """
    source_device = torch.cuda.current_device()
    source_ready = torch.cuda.Event()
    source_ready.record(torch.cuda.current_stream(source_device))
    token = secrets.token_hex(16)
    meta = _with_transfer_ticket(token, meta)
    timeline = {"enter_at": time.monotonic()}
    te = _get_te()
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    kv_len = int(meta["kv_committed_len"])
    layer_num = int(getattr(pool, "layer_num", len(meta["chunks"])))
    k0 = pool.get_key_buffer(0)
    shape = (kv_len, int(k0.shape[1]), int(k0.shape[2]))
    dtype = k0.dtype
    nbuf = max(1, min(int(FAST_BUFFERS), layer_num))
    nstreams = max(1, min(int(FAST_COPY_STREAMS), nbuf))
    if not FAST_ASYNC:
        # Without async TE a second staging buffer cannot overlap anything.
        nbuf = 1
        nstreams = 1
    t_stage0 = time.perf_counter()
    stage = [
        (
            torch.empty(shape, dtype=dtype, pin_memory=True),
            torch.empty(shape, dtype=dtype, pin_memory=True),
        )
        for _ in range(nbuf)
    ]
    t_stage_alloc = time.perf_counter()
    timeline["stage_alloc_at"] = time.monotonic()
    slot_addrs = [[int(k.data_ptr()), int(v.data_ptr())] for k, v in stage]
    flat_addrs = [a for pair in slot_addrs for a in pair]
    k_bytes = int(stage[0][0].numel() * stage[0][0].element_size())
    v_bytes = int(stage[0][1].numel() * stage[0][1].element_size())
    slot_sizes = [k_bytes, v_bytes]
    flat_sizes = slot_sizes * nbuf
    mgr = _get_registration_manager()
    mgr.register(token, flat_addrs, flat_sizes, label=f"fastlayer:{meta.get('rid')}")
    t_stage_reg = time.perf_counter()
    timeline["stage_register_at"] = time.monotonic()
    listener = None
    cancel = None
    remote_info = None
    try:
        remote_info = None
        if remote:
            remote_info = _start_remote_attention_service(scheduler, req, meta)
            timeline["remote_register_at"] = time.monotonic()
            meta = dict(meta)
            meta["remote_service"] = remote_info
        listener, port = _open_transfer_listener()
        timeline["listener_at"] = time.monotonic()
        cancel = threading.Event()
        req._handoff_transfer_cancel = (listener, cancel)
        _mark_source_transfer(scheduler, token, True)
        indices = scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :kv_len]

        def serve():
            conn = None
            t0 = time.monotonic()
            hold = stage  # keep pinned buffers alive until TE done
            sent = 0
            prof = {}
            ack_stop = threading.Event()
            ack_thread = None
            free_cv = threading.Condition()
            freed = []
            pending = {}
            streams = []
            safe_cleanup = True
            try:
                torch.cuda.set_device(source_device)
                deadline = time.monotonic() + 180.0
                while not cancel.is_set() and time.monotonic() < deadline:
                    try:
                        conn, _ = listener.accept()
                        break
                    except socket.timeout:
                        continue
                    except OSError:
                        if cancel.is_set():
                            break
                        raise
                if conn is None:
                    if cancel.is_set():
                        return
                    raise TimeoutError("transfer listener timed out")
                req._handoff_transfer_conn = conn
                if cancel.is_set():
                    return
                conn.settimeout(600)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hello = _recv_json(conn)
                if not hmac.compare_digest(str(hello.get("token", "")), token):
                    raise ValueError("bad transfer token")
                t_accept = time.monotonic()
                _send_json(conn, meta)
                reply = _recv_json(conn)
                session = str(reply["session"])
                peer = [int(x) for x in reply["addrs"]]
                relay_slots = int(reply.get("slots") or 0)
                layer_peer = []
                if relay_slots:
                    if len(peer) != 2 * relay_slots:
                        raise ValueError(
                            f"slot address count mismatch: {len(peer)} != "
                            f"{2 * relay_slots}"
                        )
                    layer_peer = [
                        [peer[2 * s], peer[2 * s + 1]] for s in range(relay_slots)
                    ]
                    freed = [0] * relay_slots

                    def ack_loop():
                        while not ack_stop.is_set():
                            try:
                                msg = _recv_json(conn)
                            except Exception:
                                return
                            if not isinstance(msg, dict):
                                return
                            slot = msg.get("free")
                            if slot is None:
                                continue
                            with free_cv:
                                freed[int(slot)] += 1
                                free_cv.notify_all()

                    ack_thread = threading.Thread(
                        target=ack_loop, daemon=True, name="handoff-fast-ack"
                    )
                    ack_thread.start()
                else:
                    # Full-mirror targets: 2 tensors per chunk per layer.
                    off = 0
                    for layer_chunks in meta["chunks"]:
                        n = 2 * len(layer_chunks)
                        layer_peer.append(peer[off : off + n])
                        off += n
                    if off != len(peer):
                        raise ValueError(
                            f"peer address count mismatch: {off} != {len(peer)}"
                        )
                streams = _fast_copy_streams(source_device, nstreams)
                for stream in streams:
                    stream.wait_event(source_ready)
                evs = [torch.cuda.Event() for _ in range(nbuf)]
                pending = {}
                ctrl_next = 0
                t_start = time.monotonic()

                def prof_rec(layer):
                    rec = prof.get(layer)
                    if rec is None:
                        rec = {"layer": layer}
                        prof[layer] = rec
                    return rec

                def wait_target_slot(layer, tslot):
                    if not relay_slots:
                        return
                    # Slot tslot has already carried layer uses 0..need-1; the
                    # source may write use ``need`` only after those H2D copies
                    # were acknowledged.
                    need = layer // relay_slots
                    if need <= 0:
                        return
                    deadline_s = time.monotonic() + 300.0
                    with free_cv:
                        while freed[tslot] < need and not cancel.is_set():
                            left = deadline_s - time.monotonic()
                            if left <= 0:
                                raise TimeoutError(
                                    f"target staging slot {tslot} not freed"
                                )
                            free_cv.wait(min(0.1, left))
                    if cancel.is_set():
                        raise RuntimeError("transfer cancelled")

                def issue_copy(layer):
                    s = layer % nbuf
                    rec = prof_rec(layer)
                    rec["copy_start_at"] = time.monotonic()
                    rec["copy_start"] = time.perf_counter()
                    k_buf = pool.get_key_buffer(layer)
                    v_buf = pool.get_value_buffer(layer)
                    with torch.cuda.stream(streams[s % nstreams]):
                        k_gpu = k_buf[indices]
                        v_gpu = v_buf[indices]
                        stage[s][0].copy_(k_gpu, non_blocking=True)
                        stage[s][1].copy_(v_gpu, non_blocking=True)
                        evs[s].record()

                def submit_te(layer):
                    nonlocal sent, ctrl_next, safe_cleanup
                    s = layer % nbuf
                    tslot = layer % relay_slots if relay_slots else None
                    rec = prof_rec(layer)
                    evs[s].synchronize()
                    rec["copy_done_at"] = time.monotonic()
                    rec["copy_done"] = time.perf_counter()
                    if FAST_ASYNC and relay_slots:
                        # The target can only free slot ``tslot`` after it has
                        # received the control message for an earlier layer.
                        # Controls lag by up to nbuf in the async pipeline, so
                        # flush them before blocking on the slot ack. Otherwise
                        # slots < nbuf deadlock (source waits for an ack that
                        # requires a control message it has not sent yet).
                        need_layer = layer - relay_slots
                        while ctrl_next <= need_layer and ctrl_next < layer_num:
                            complete_ctrl(ctrl_next)
                            ctrl_next += 1
                    wait_target_slot(layer, tslot)
                    rec["te_submit"] = time.perf_counter()
                    peer_l = layer_peer[tslot] if relay_slots else layer_peer[layer]
                    if FAST_ASYNC:
                        try:
                            bid = te.batch_transfer_async_write(
                                session, slot_addrs[s], peer_l, slot_sizes
                            )
                            if not bid or int(bid) <= 0:
                                raise RuntimeError(
                                    f"Mooncake async fast layer {layer} submit "
                                    f"failed: {bid}"
                                )
                        except Exception:
                            safe_cleanup = False
                            raise
                        pending[layer] = int(bid)
                    else:
                        safe_cleanup = False
                        rc = te.batch_transfer_sync_write(
                            session, slot_addrs[s], peer_l, slot_sizes
                        )
                        if rc != 0:
                            raise RuntimeError(
                                f"Mooncake fast layer {layer} failed: rc={rc}"
                            )
                        safe_cleanup = True
                        rec["te_done_at"] = time.monotonic()
                        rec["te_done"] = time.perf_counter()
                        sent += sum(slot_sizes)
                        msg = {"layer": layer}
                        if relay_slots:
                            msg["slot"] = tslot
                        _send_json(conn, msg)
                        rec["ctrl_at"] = time.monotonic()

                def complete_ctrl(layer):
                    nonlocal sent, safe_cleanup
                    rec = prof_rec(layer)
                    bid = pending.pop(layer)
                    try:
                        rc = te.get_batch_transfer_status([bid])
                    except Exception:
                        safe_cleanup = False
                        raise
                    if rc != 0:
                        safe_cleanup = False
                        raise RuntimeError(
                            f"Mooncake async fast layer {layer} failed: rc={rc}"
                        )
                    rec["te_done_at"] = time.monotonic()
                    rec["te_done"] = time.perf_counter()
                    sent += sum(slot_sizes)
                    msg = {"layer": layer}
                    if relay_slots:
                        msg["slot"] = layer % relay_slots
                    _send_json(conn, msg)
                    rec["ctrl_at"] = time.monotonic()

                if FAST_ASYNC:
                    if nbuf == 1:
                        # Single buffer: the next copy needs the previous TE to
                        # finish, so submit immediately after each copy.
                        for layer in range(layer_num):
                            if cancel.is_set():
                                return
                            while ctrl_next < layer:
                                complete_ctrl(ctrl_next)
                                ctrl_next += 1
                            issue_copy(layer)
                            submit_te(layer)
                    else:
                        # Double buffer: copy L is issued while TE L-1 runs.
                        for layer in range(layer_num):
                            if cancel.is_set():
                                return
                            while ctrl_next <= layer - nbuf:
                                complete_ctrl(ctrl_next)
                                ctrl_next += 1
                            issue_copy(layer)
                            if layer >= 1:
                                submit_te(layer - 1)
                        submit_te(layer_num - 1)
                else:
                    for layer in range(layer_num):
                        if cancel.is_set():
                            return
                        issue_copy(layer)
                        submit_te(layer)
                        ctrl_next = layer + 1
                while ctrl_next < layer_num:
                    complete_ctrl(ctrl_next)
                    ctrl_next += 1
                _send_json(conn, {"done": True})
                layers_out = []
                for layer in range(layer_num):
                    rec = prof.get(layer) or {}
                    copy_ms = None
                    if (
                        rec.get("copy_start") is not None
                        and rec.get("copy_done") is not None
                    ):
                        copy_ms = (rec["copy_done"] - rec["copy_start"]) * 1000.0
                    te_ms = None
                    if (
                        rec.get("te_submit") is not None
                        and rec.get("te_done") is not None
                    ):
                        te_ms = (rec["te_done"] - rec["te_submit"]) * 1000.0
                    layers_out.append(
                        {
                            "layer": layer,
                            "copy_ms": copy_ms,
                            "te_ms": te_ms,
                            "copy_start_at": rec.get("copy_start_at"),
                            "te_done_at": rec.get("te_done_at"),
                            "ctrl_at": rec.get("ctrl_at"),
                            "bytes": sum(slot_sizes),
                        }
                    )
                _TRANSFER_STATS[token] = {
                    "bytes": sent,
                    "send_ms": (time.perf_counter() - t0) * 1000.0,
                    "engine": "fast_gpu_layers",
                    "pipeline": nbuf,
                    "streams": nstreams,
                    "async": bool(FAST_ASYNC),
                    "target_slots": relay_slots,
                    "start_at": t_start,
                    "accept_at": t_accept,
                    "done_at": time.monotonic(),
                    "stage_alloc_ms": (t_stage_alloc - t_stage0) * 1000.0,
                    "stage_register_ms": (t_stage_reg - t_stage_alloc) * 1000.0,
                    "thread_start_at": t0,
                    "layers": layers_out,
                    "rc": 0,
                }
            except Exception as exc:
                if not cancel.is_set():
                    _TRANSFER_ERRORS[token] = repr(exc)
                    try:
                        _send_json(conn, {"ok": False, "error": repr(exc)})
                    except Exception:
                        pass
            finally:
                ack_stop.set()
                if getattr(req, "_handoff_transfer_conn", None) is conn:
                    req._handoff_transfer_conn = None
                _mark_source_transfer(scheduler, token, False)
                try:
                    if pending:
                        safe_cleanup = (
                            te.get_batch_transfer_status(list(pending.values())) == 0
                            and safe_cleanup
                        )
                        pending.clear()
                except Exception:
                    safe_cleanup = False
                try:
                    for stream in streams:
                        stream.synchronize()
                except Exception:
                    safe_cleanup = False
                    req._handoff_transfer_unsafe = True
                _publish_transfer_complete(token, safe_cleanup)
                _release_transfer_buffers(
                    token,
                    stage,
                    "source transfer completion unknown",
                    remote_complete=safe_cleanup,
                    gpu_safe=not getattr(req, "_handoff_transfer_unsafe", False),
                )
                if getattr(req, "_handoff_transfer_cancel", None) is not None:
                    handle = req._handoff_transfer_cancel
                    if handle[1] is cancel:
                        req._handoff_transfer_cancel = None
                del hold
                if conn is not None:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    try:
                        conn.close()
                    except OSError:
                        pass
                if ack_thread is not None:
                    ack_thread.join(timeout=0.5)
                try:
                    listener.close()
                except OSError:
                    pass

        thread = threading.Thread(target=serve, daemon=True, name="handoff-fast-layers")
        thread.start()
        timeline["thread_start_at"] = time.monotonic()
        req._handoff_transfer_thread = thread
    except Exception:
        _release_transfer_buffers(
            token, stage, "source setup rollback", remote_complete=True
        )
        if remote_info is not None:
            try:
                get_remote_hub(scheduler).registry.discard(
                    str(req.rid), expected_req=req
                )
            except Exception:
                pass
        _mark_source_transfer(scheduler, token, False)
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if cancel is not None:
            handle = getattr(req, "_handoff_transfer_cancel", None)
            if handle is not None and handle[1] is cancel:
                req._handoff_transfer_cancel = None
        raise
    return {
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "mode": "remote" if remote else "layer",
        "remote_service": remote_info,
        "fast_freeze": True,
        "fast_buffers": nbuf,
        "fast_streams": nstreams,
        "fast_async": bool(FAST_ASYNC),
        "timeline": dict(timeline, return_at=time.monotonic()),
    }


def _start_layer_mooncake_server(
    scheduler, req, meta: dict, kv_cpu, remote: bool
) -> dict:
    """Source side for layer_wait / remote_progressive.

    Registers all CPU KV, lets the target register its own CPU buffers, then
    transfers one layer at a time and sends a control message after each layer.
    The target's receiver copies that layer to GPU and sets its layer event.
    """
    token = secrets.token_hex(16)
    meta = _with_transfer_ticket(token, meta)
    timeline = {"enter_at": time.monotonic()}
    te = _get_te()
    tensors = _flatten_cpu(kv_cpu)
    addrs = [int(t.data_ptr()) for t in tensors]
    sizes = [int(t.numel() * t.element_size()) for t in tensors]
    mgr = _get_registration_manager()
    mgr.register(token, addrs, sizes, label=f"layer:{meta.get('rid')}")
    timeline["stage_register_at"] = time.monotonic()

    listener = None
    cancel = None
    try:
        remote_info = None
        if remote:
            remote_info = _start_remote_attention_service(scheduler, req, meta)
            timeline["remote_register_at"] = time.monotonic()
            meta = dict(meta)
            meta["remote_service"] = remote_info

        listener, port = _open_transfer_listener()
        timeline["listener_at"] = time.monotonic()
        cancel = threading.Event()
        req._handoff_transfer_cancel = (listener, cancel)
        _mark_source_transfer(scheduler, token, True)

        # Offsets into the flat tensor list, one pair (K,V) per chunk per layer.
        layer_slices = []
        off = 0
        for layer_chunks in meta["chunks"]:
            n = 2 * len(layer_chunks)
            layer_slices.append(slice(off, off + n))
            off += n
        try:
            pipeline = max(1, int(os.environ.get("HANDOFF_TRANSFER_PIPELINE", "1")))
        except (TypeError, ValueError):
            pipeline = 1

        def serve():
            hold = tensors  # keep CPU KV alive until TE unregisters
            hold_kv = kv_cpu
            conn = None
            t0 = time.perf_counter()
            sent = 0
            safe_cleanup = True
            outstanding = {}
            try:
                deadline = time.monotonic() + 180.0
                while not cancel.is_set() and time.monotonic() < deadline:
                    try:
                        conn, _ = listener.accept()
                        break
                    except socket.timeout:
                        continue
                    except OSError:
                        if cancel.is_set():
                            break
                        raise
                if conn is None:
                    if cancel.is_set():
                        return
                    raise TimeoutError("transfer listener timed out")
                req._handoff_transfer_conn = conn
                if cancel.is_set():
                    return
                conn.settimeout(600)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hello = _recv_json(conn)
                if not hmac.compare_digest(str(hello.get("token", "")), token):
                    raise ValueError("bad transfer token")
                _send_json(conn, meta)
                reply = _recv_json(conn)
                session = str(reply["session"])
                peer = [int(x) for x in reply["addrs"]]
                if len(peer) != len(addrs):
                    raise ValueError(
                        f"address count mismatch: src={len(addrs)} dst={len(peer)}"
                    )
                if pipeline <= 1:
                    for layer, sl in enumerate(layer_slices):
                        if cancel.is_set():
                            return
                        safe_cleanup = False
                        rc = te.batch_transfer_sync_write(
                            session, addrs[sl], peer[sl], sizes[sl]
                        )
                        if rc != 0:
                            raise RuntimeError(
                                f"Mooncake layer {layer} transfer failed: rc={rc}"
                            )
                        safe_cleanup = True
                        sent += sum(sizes[sl])
                        _send_json(conn, {"layer": layer})
                else:

                    def submit(layer):
                        nonlocal safe_cleanup
                        sl = layer_slices[layer]
                        try:
                            bid = te.batch_transfer_async_write(
                                session, addrs[sl], peer[sl], sizes[sl]
                            )
                            if not bid or int(bid) <= 0:
                                raise RuntimeError(
                                    f"Mooncake async layer {layer} submit failed: {bid}"
                                )
                        except Exception:
                            safe_cleanup = False
                            raise
                        outstanding[layer] = int(bid)

                    for layer in range(min(pipeline, len(layer_slices))):
                        submit(layer)
                    for layer in range(len(layer_slices)):
                        if cancel.is_set():
                            return
                        bid = outstanding.pop(layer)
                        try:
                            rc = te.get_batch_transfer_status([bid])
                            if rc != 0:
                                raise RuntimeError(
                                    f"Mooncake async layer {layer} failed: rc={rc}"
                                )
                        except Exception:
                            safe_cleanup = False
                            raise
                        sent += sum(sizes[layer_slices[layer]])
                        _send_json(conn, {"layer": layer})
                        nxt = layer + pipeline
                        if nxt < len(layer_slices):
                            submit(nxt)
                _send_json(conn, {"done": True})
                _TRANSFER_STATS[token] = {
                    "bytes": sent,
                    "send_ms": (time.perf_counter() - t0) * 1000.0,
                    "engine": "mooncake_tcp_layers",
                    "pipeline": pipeline,
                    "rc": 0,
                }
            except Exception as exc:
                if not cancel.is_set():
                    _TRANSFER_ERRORS[token] = repr(exc)
                    try:
                        _send_json(conn, {"ok": False, "error": repr(exc)})
                    except Exception:
                        pass
            finally:
                if getattr(req, "_handoff_transfer_conn", None) is conn:
                    req._handoff_transfer_conn = None
                _mark_source_transfer(scheduler, token, False)
                try:
                    if outstanding:
                        safe_cleanup = (
                            te.get_batch_transfer_status(list(outstanding.values()))
                            == 0
                            and safe_cleanup
                        )
                        outstanding.clear()
                except Exception:
                    safe_cleanup = False
                _publish_transfer_complete(token, safe_cleanup)
                _release_transfer_buffers(
                    token,
                    kv_cpu,
                    "source transfer completion unknown",
                    remote_complete=safe_cleanup,
                    gpu_safe=not getattr(req, "_handoff_transfer_unsafe", False),
                )
                if getattr(req, "_handoff_transfer_cancel", None) is not None:
                    handle = req._handoff_transfer_cancel
                    if handle[1] is cancel:
                        req._handoff_transfer_cancel = None
                del hold
                del hold_kv
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass
                try:
                    listener.close()
                except OSError:
                    pass

        thread = threading.Thread(target=serve, daemon=True, name="handoff-kv-layers")
        thread.start()
        timeline["thread_start_at"] = time.monotonic()
        req._handoff_transfer_thread = thread
    except Exception:
        # Rollback everything opened after the TE registration (remote hub
        # registration, listener, cancel handle): if the thread never started,
        # its finally block cannot do this for us.
        _release_transfer_buffers(
            token, kv_cpu, "source setup rollback", remote_complete=True
        )
        if remote_info is not None:
            # The hub registration has no transfer behind it now; drop it
            # without firing on_release (the request is still live).
            try:
                get_remote_hub(scheduler).registry.discard(
                    str(req.rid), expected_req=req
                )
            except Exception:
                pass
        _mark_source_transfer(scheduler, token, False)
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if cancel is not None:
            handle = getattr(req, "_handoff_transfer_cancel", None)
            if handle is not None and handle[1] is cancel:
                req._handoff_transfer_cancel = None
        raise
    return {
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "mode": "remote" if remote else "layer",
        "remote_service": remote_info,
        "timeline": dict(timeline, return_at=time.monotonic()),
    }


def _recv_transfer_layer_setup(transfer: dict):
    """Target side handshake for layer-wise transfer.

    Non-fast transfers allocate and register the full CPU KV mirror.  Fast
    transfers (P5.2-B) allocate a bounded ring of layer-sized staging slots
    (``HANDOFF_TARGET_STAGING_SLOTS``) and advertise the slot count; the source
    then reuses a slot only after the target acknowledges its H2D copy.
    """
    te = _get_te()
    host = str(transfer["host"])
    port = int(transfer["port"])
    token = str(transfer["token"])
    owner = f"target-layer:{token}"
    mgr = _get_registration_manager()
    registered = False
    exposed = False
    t_setup0 = time.monotonic()
    conn = socket.create_connection((host, port), timeout=600)
    t_connect = time.monotonic()
    conn.settimeout(600)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        _send_json(conn, {"token": token})
        meta = _recv_json(conn)
        _bind_transfer_ticket(owner, token, meta)
        t_meta = time.monotonic()
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[meta["dtype"]]
        hkv = int(meta["kv_heads"])
        head_dim = int(meta["head_dim"])
        chunks = meta["chunks"]
        kv_len = int(meta["kv_committed_len"])
        if not chunks or sum(int(n) for n in chunks[0]) != kv_len:
            raise ValueError(
                f"chunk manifest mismatch: kv_len={kv_len} chunks0={chunks[:1]}"
            )
        layer_num = len(chunks)
        slot_bufs = None
        fast = bool(meta.get("fast_freeze")) and int(TARGET_STAGING_SLOTS) > 0
        if fast:
            if any(len(layer_chunks) != 1 for layer_chunks in chunks):
                raise ValueError("fast freeze expects exactly one chunk per layer")
            slots = max(1, int(TARGET_STAGING_SLOTS))
            tensors = []
            slot_bufs = []
            for _ in range(slots):
                k = torch.empty((kv_len, hkv, head_dim), dtype=dtype, pin_memory=True)
                v = torch.empty((kv_len, hkv, head_dim), dtype=dtype, pin_memory=True)
                slot_bufs.append([(k, v)])
                tensors.extend([k, v])
            t_alloc = time.monotonic()
            kv_cpu = None
            addrs = [int(t.data_ptr()) for t in tensors]
            sizes = [int(t.numel() * t.element_size()) for t in tensors]
            mgr.register(owner, addrs, sizes, label="target-fast-slots")
            t_reg = time.monotonic()
            registered = True
            exposed = True
            _send_json(
                conn,
                {
                    "session": f"127.0.0.1:{te.get_rpc_port()}",
                    "addrs": addrs,
                    "slots": slots,
                },
            )
            t_reply = time.monotonic()
        else:
            kv_cpu = []
            tensors = []
            for layer_chunks in chunks:
                layer = []
                for n in layer_chunks:
                    k = torch.empty(
                        (int(n), hkv, head_dim), dtype=dtype, pin_memory=True
                    )
                    v = torch.empty(
                        (int(n), hkv, head_dim), dtype=dtype, pin_memory=True
                    )
                    layer.append([k, v])
                    tensors.extend([k, v])
                kv_cpu.append(layer)
            t_alloc = time.monotonic()
            addrs = [int(t.data_ptr()) for t in tensors]
            sizes = [int(t.numel() * t.element_size()) for t in tensors]
            mgr.register(owner, addrs, sizes, label="target-layer")
            t_reg = time.monotonic()
            registered = True
            exposed = True
            _send_json(
                conn,
                {"session": f"127.0.0.1:{te.get_rpc_port()}", "addrs": addrs},
            )
            t_reply = time.monotonic()
    except Exception:
        if registered:
            try:
                _release_transfer_buffers(
                    owner, tensors, "layer setup failed", remote_complete=not exposed
                )
            except Exception:
                pass
        conn.close()
        raise
    setup_tl = {
        "setup_enter_at": t_setup0,
        "connect_at": t_connect,
        "meta_at": t_meta,
        "alloc_at": t_alloc,
        "register_at": t_reg,
        "reply_at": t_reply,
        "fast": bool(fast),
        "slots": int(TARGET_STAGING_SLOTS) if fast else 0,
        "kv_len": int(kv_len),
    }
    meta["_setup_timeline"] = setup_tl
    print(
        f"[handoff-target] setup rid={meta.get('rid')} fast={fast} "
        f"connect_ms={(t_connect - t_setup0) * 1000:.1f} "
        f"meta_ms={(t_meta - t_connect) * 1000:.1f} "
        f"alloc_ms={(t_alloc - t_meta) * 1000:.1f} "
        f"register_ms={(t_reg - t_alloc) * 1000:.1f} "
        f"reply_ms={(t_reply - t_reg) * 1000:.1f} "
        f"total_ms={(t_reply - t_setup0) * 1000:.1f}",
        flush=True,
    )
    return (
        conn,
        meta,
        kv_cpu,
        te,
        addrs,
        owner,
        slot_bufs,
        layer_num,
        int(sum(sizes)),
    )


def _recv_transfer_mooncake(transfer: dict):
    """Target side: allocate/register CPU KV, hand addresses to the source,
    wait for Mooncake TE to fill them, then return the nested CPU tensors."""
    te = _get_te()
    host = str(transfer["host"])
    port = int(transfer["port"])
    token = str(transfer["token"])
    mgr = _get_registration_manager()
    owner = f"target-mooncake:{token}"
    with socket.create_connection((host, port), timeout=600) as conn:
        conn.settimeout(600)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send_json(conn, {"token": token})
        meta = _recv_json(conn)
        _bind_transfer_ticket(owner, token, meta)
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[meta["dtype"]]
        hkv = int(meta["kv_heads"])
        head_dim = int(meta["head_dim"])
        chunks = meta["chunks"]
        kv_len = int(meta["kv_committed_len"])
        if not chunks or sum(int(n) for n in chunks[0]) != kv_len:
            raise ValueError(
                f"chunk manifest mismatch: kv_len={kv_len} chunks0={chunks[:1]}"
            )
        kv_cpu = []
        tensors = []
        for layer_chunks in chunks:
            layer = []
            for n in layer_chunks:
                k = torch.empty((int(n), hkv, head_dim), dtype=dtype)
                v = torch.empty((int(n), hkv, head_dim), dtype=dtype)
                layer.append([k, v])
                tensors.extend([k, v])
            kv_cpu.append(layer)
        addrs = [int(t.data_ptr()) for t in tensors]
        sizes = [int(t.numel() * t.element_size()) for t in tensors]
        try:
            mgr.register(owner, addrs, sizes, label="target-mooncake")
        except Exception:
            conn.close()
            raise
        complete = False
        try:
            _send_json(
                conn,
                {
                    "session": f"127.0.0.1:{te.get_rpc_port()}",
                    "addrs": addrs,
                },
            )
            ack = _recv_json(conn)
            if not ack.get("ok"):
                raise RuntimeError(f"Mooncake transfer failed: {ack.get('error')}")
            complete = True
        finally:
            _release_transfer_buffers(
                owner,
                kv_cpu,
                "copy transfer disconnected before completion ACK",
                remote_complete=complete,
            )
    return meta, kv_cpu


# ---------------------------------------------------------------------------
# control hub
# ---------------------------------------------------------------------------
def _tp_rank(scheduler) -> int:
    from sglang.srt.runtime_context import get_parallel

    return int(get_parallel().tp_rank)


def _tp_size(scheduler) -> int:
    from sglang.srt.runtime_context import get_parallel

    return int(get_parallel().tp_size)


def _ctl_paths(scheduler=None):
    root = Path(_CTL_DIR)
    rank = _tp_rank(scheduler) if scheduler is not None else 0
    return root, root / f"cmd.{rank}.json", root / f"result.{rank}.json"


def _tracked(scheduler):
    tracked = getattr(scheduler, "handoff_tracked_reqs", None)
    if tracked is None:
        tracked = {}
        scheduler.handoff_tracked_reqs = tracked
    return tracked


def _reservations(scheduler):
    """Target-side pre-freeze KV reservations keyed by destination rid."""
    res = getattr(scheduler, "_handoff_reserved", None)
    if res is None:
        res = {}
        scheduler._handoff_reserved = res
    return res


def _reservation_free(scheduler, rid) -> Optional[dict]:
    res = _reservations(scheduler).pop(str(rid), None)
    if res is not None:
        try:
            scheduler.token_to_kv_pool_allocator.free(res["indices"])
        except Exception:
            pass
    return res


def _take_reserved(scheduler, rid, kv_len: int):
    """Consume a pre-freeze reservation's KV, keeping its execution slot.

    The slot reservation is released only after admit_decode has actually
    allocated a req_pool row (see ``_capacity_reconcile``).  This prevents
    ordinary admissions from taking the row between import and admit.
    """
    res = _reservations(scheduler).get(str(rid))
    if res is None:
        return None
    allocator = scheduler.token_to_kv_pool_allocator
    idx = res.get("indices")
    if idx is None:
        raise RuntimeError(f"reservation {rid!r} KV already consumed")
    n = int(idx.numel())
    if n < kv_len:
        _reservation_free(scheduler, rid)
        raise RuntimeError(f"reserved {n} KV slots < import needs {kv_len}")
    if n > kv_len:
        allocator.free(idx[kv_len:])
        idx = idx[:kv_len]
    res["indices"] = None
    res["kv_consumed_at"] = time.monotonic()
    return idx


def _ordinary_leases(scheduler):
    """Target-side capacity tickets for ordinary (non-migration) requests."""
    leases = getattr(scheduler, "_handoff_ordinary_leases", None)
    if leases is None:
        leases = {}
        scheduler._handoff_ordinary_leases = leases
    return leases


def _migration_slots(scheduler) -> int:
    return sum(
        max(0, int(r.get("slots", 0)))
        for r in _reservations(scheduler).values()
        if not r.get("slot_consumed")
    )


def _ordinary_slot_count(scheduler) -> int:
    return sum(
        1 for l in _ordinary_leases(scheduler).values() if not l.get("slot_released")
    )


def _ordinary_kv_count(scheduler) -> int:
    return sum(
        max(0, int(l.get("tokens", 0)))
        for l in _ordinary_leases(scheduler).values()
        if not l.get("kv_released")
    )


def _capacity_stats(scheduler) -> dict:
    """Authoritative target capacity: KV and execution slots.

    Migration KV is physically allocated, so ``available_size`` already
    excludes it.  Ordinary KV leases are soft capacity (the request has not
    necessarily prefilled yet), so they are subtracted separately.  Execution
    slots are counted from the real ReqToTokenPool, then reduced by migration
    reservations and ordinary slot leases that have not yet become live rows.
    """
    from sglang.srt.runtime_context import get_schedule

    pool = scheduler.req_to_token_pool
    allocator = scheduler.token_to_kv_pool_allocator
    max_run = int(get_schedule().max_running_requests or 0)
    free_rows = int(pool.available_size())
    mig_slots = _migration_slots(scheduler)
    ord_slots = _ordinary_slot_count(scheduler)
    kv_avail = int(allocator.available_size())
    ord_kv = _ordinary_kv_count(scheduler)
    return {
        "max_running": max_run,
        "free_rows": free_rows,
        "live_rows": max(0, max_run - free_rows),
        "migration_reserved_slots": mig_slots,
        "ordinary_slot_leases": ord_slots,
        "free_slots": max(0, free_rows - mig_slots - ord_slots),
        "kv_available": kv_avail,
        "ordinary_kv_leases": ord_kv,
        "free_kv": max(0, kv_avail - ord_kv),
        "migration_reservations": len(_reservations(scheduler)),
        "ordinary_leases": len(_ordinary_leases(scheduler)),
    }


def _ordinary_lease_acquire(scheduler, rid, tokens: int, epoch: str = ""):
    """Atomically reserve one slot and a soft KV budget for an ordinary req.

    Runs on the scheduler thread, so the check and insert are atomic with
    respect to every other control command on this TP rank.
    """
    rid = str(rid)
    tokens = int(tokens)
    leases = _ordinary_leases(scheduler)
    cur = leases.get(rid)
    if cur is not None:
        if int(cur.get("tokens", 0)) >= tokens:
            return {
                "admitted": True,
                "reused": True,
                "ticket": rid,
                "tokens": int(cur.get("tokens", 0)),
                "capacity": _capacity_stats(scheduler),
            }
        return {
            "admitted": False,
            "reason": "rid already leased with a smaller budget",
            "ticket": rid,
            "capacity": _capacity_stats(scheduler),
        }
    if rid in _tracked(scheduler):
        return {
            "admitted": False,
            "reason": "rid already live",
            "ticket": rid,
            "capacity": _capacity_stats(scheduler),
        }
    stats = _capacity_stats(scheduler)
    if stats["free_slots"] < 1:
        return {
            "admitted": False,
            "reason": "insufficient slots",
            "ticket": rid,
            "capacity": stats,
        }
    if tokens > stats["free_kv"]:
        return {
            "admitted": False,
            "reason": "insufficient KV",
            "ticket": rid,
            "capacity": stats,
        }
    leases[rid] = {
        "rid": rid,
        "tokens": tokens,
        "epoch": str(epoch or ""),
        "created_at": time.monotonic(),
        "slot_released": False,
        "kv_released": False,
        "req": None,
    }
    return {
        "admitted": True,
        "reused": False,
        "ticket": rid,
        "tokens": tokens,
        "capacity": _capacity_stats(scheduler),
    }


def _ordinary_lease_free(scheduler, rid, force: bool = False) -> bool:
    rid = str(rid)
    leases = _ordinary_leases(scheduler)
    lease = leases.get(rid)
    if lease is None:
        return False
    req = lease.get("req")
    if (
        not force
        and req is not None
        and req is not getattr(scheduler, "_handoff_placeholder", None)
        and not req.finished()
    ):
        return False
    leases.pop(rid, None)
    return True


def _capacity_reconcile(scheduler) -> None:
    """Convert leases/reservations into live rows, and expire abandoned ones."""
    now = time.monotonic()
    tracked = _tracked(scheduler)
    # Migration reservations: once admit_decode allocated a real row, the
    # slot reservation has done its job.
    for rid, res in list(_reservations(scheduler).items()):
        req = tracked.get(rid)
        if req is not None and req.kv.req_pool_idx is not None:
            res["slot_consumed"] = True
        if res.get("slot_consumed") and res.get("indices") is None:
            _reservations(scheduler).pop(rid, None)
    # Ordinary leases: release the slot count when the scheduler allocated the
    # real row; free the lease when the request finishes or is abandoned.
    lease_ttl = float(os.environ.get("HANDOFF_LEASE_TTL_S", "120"))
    for rid, lease in list(_ordinary_leases(scheduler).items()):
        req = lease.get("req")
        if req is None:
            req = tracked.get(rid)
            if req is not None:
                lease["req"] = req
        if req is not None and req.kv.req_pool_idx is not None:
            lease["slot_released"] = True
        if req is not None and req.finished():
            _ordinary_lease_free(scheduler, rid, force=True)
            continue
        if req is None and now - float(lease.get("created_at", now)) > lease_ttl:
            _ordinary_lease_free(scheduler, rid, force=True)


def _expire_capacity(scheduler) -> None:
    """Reservation/lease expiry independent of controller heartbeat state."""
    now = time.monotonic()
    tracked = _tracked(scheduler)
    mig_ttl = float(os.environ.get("HANDOFF_MIG_RESV_TTL_S", "1800"))
    for rid, res in list(_reservations(scheduler).items()):
        if rid in tracked:
            continue
        if now - float(res.get("created_at", now)) > mig_ttl:
            if _reservation_free(scheduler, rid) is not None:
                print(
                    f"[handoff-capacity] expired reservation {rid}",
                    flush=True,
                )
    lease_ttl = float(os.environ.get("HANDOFF_LEASE_TTL_S", "120"))
    for rid, lease in list(_ordinary_leases(scheduler).items()):
        if lease.get("req") is None and rid not in tracked:
            if now - float(lease.get("created_at", now)) > lease_ttl:
                _ordinary_lease_free(scheduler, rid, force=True)
                print(
                    f"[handoff-capacity] expired ordinary lease {rid}",
                    flush=True,
                )


def _kv_available(scheduler):
    try:
        return int(scheduler.token_to_kv_pool_allocator.available_size())
    except Exception:
        return None


_HANDOFF_CTL_KEY = "__handoff_ctl__"
# Carries the path of the file this command was claimed from through the TP
# broadcast, so the rank that claimed it can retire it after execution.
_HANDOFF_CLAIM_KEY = "__handoff_claimed__"
_CTRL_RANK = 0
_CTL_QUEUE_PREFIX = "ctlq."
_RESUME_CLAIM_TTL_S = float(os.environ.get("HANDOFF_RESUME_CLAIM_TTL_S", "30"))
_WD_FENCE_TIMEOUT_S = float(os.environ.get("HANDOFF_WD_FENCE_TIMEOUT_S", "30"))
_RESUME_BLOCK_TTL_S = float(os.environ.get("HANDOFF_RESUME_BLOCK_TTL_S", "600"))


def _resume_block_path(claim: str) -> Path:
    digest = hashlib.sha1(claim.encode("utf-8")).hexdigest()[:16]
    return Path(_CTL_DIR) / f"resume_block.{digest}.json"


def _resume_blocked(claim: str) -> bool:
    """True when any source TP rank proved the target owns this epoch."""
    if not _CTL_DIR:
        return False
    path = _resume_block_path(claim)
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return False
    if age > _RESUME_BLOCK_TTL_S:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def _mark_resume_blocked(claim: str, reason: str, rid: str) -> None:
    """Share a terminal ownership verdict across both source TP ranks."""
    if not _CTL_DIR:
        return
    path = _resume_block_path(claim)
    try:
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(
                {"rid": str(rid), "reason": str(reason), "at": time.time()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        pass


def _watchdog_resume_rank_allowed(scheduler) -> bool:
    """Test hook: pin resume initiation to one TP rank.

    Production leaves this unset, allowing either rank to publish the
    command (rank0 may be blocked in the TP broadcast).  The stress harness
    sets it to 0 or 1 to prove both initiation paths.
    """
    forced = os.environ.get("HANDOFF_WD_RESUME_RANK", "").strip()
    if not forced:
        return True
    try:
        return _tp_rank(scheduler) == int(forced)
    except ValueError:
        return True


def _resume_claim_path(claim: str) -> Path:
    digest = hashlib.sha1(claim.encode("utf-8")).hexdigest()[:16]
    return Path(_CTL_DIR) / f"resume_claim.{digest}"


def _resume_claim_pending(claim: str) -> bool:
    """True while a durable resume command for ``claim`` is in flight.

    The claim is retired only by ``execute_handoff_control`` after the
    broadcast command actually ran.  A stale claim (rank0 crashed before
    polling, or a command file was lost) expires so the watchdog can retry.
    """
    if not _CTL_DIR:
        return False
    path = _resume_claim_path(claim)
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    if time.time() - st.st_mtime > _RESUME_CLAIM_TTL_S:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def _prune_stale_control_files(root: Path) -> None:
    """Bound crash leftovers in the resume queue."""
    now = time.time()
    patterns = ("resume_claim.*", f"{_CTL_QUEUE_PREFIX}*.claimed.*")
    for pattern in patterns:
        for path in root.glob(pattern):
            try:
                if now - path.stat().st_mtime <= _RESUME_CLAIM_TTL_S:
                    continue
                path.unlink()
            except OSError:
                pass


def enqueue_broadcast_control_once(payload: dict, claim: str) -> str:
    """Publish one TP-broadcast control command, at most once per claim.

    The frozen watchdog may run on either rank while the other is already
    waiting in the TP broadcast.  Both ranks try to claim; the winner writes
    a *unique* ``ctlq.*.json`` file, which rank0's request receiver injects
    before the next broadcast.  A control command from the external driver
    uses ``cmd.0.json`` and therefore cannot overwrite a pending resume.

    The claim is not removed when the file is read: it stays as a crash-safe
    marker until ``execute_handoff_control`` has actually run the command.
    """
    if not _CTL_DIR:
        raise RuntimeError("HANDOFF_CTL_DIR is not set")
    root = Path(_CTL_DIR)
    root.mkdir(parents=True, exist_ok=True)
    _prune_stale_control_files(root)
    digest = hashlib.sha1(claim.encode("utf-8")).hexdigest()[:16]
    claim_path = root / f"resume_claim.{digest}"
    fd = None
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return ""
    finally:
        if fd is not None:
            os.close(fd)
    nonce = secrets.token_hex(16)
    data = {
        **payload,
        "nonce": nonce,
        "__resume_claim__": str(claim_path),
    }
    queue_path = root / f"{_CTL_QUEUE_PREFIX}{digest}.{nonce}.json"
    tmp = root / (f"{_CTL_QUEUE_PREFIX}{digest}.{nonce}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, queue_path)
    except Exception:
        try:
            os.unlink(claim_path)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return nonce


def _claim_control_file(path: Path, root: Path):
    """Atomically rename and parse one command file."""
    claimed_path = root / f"{path.name}.claimed.{os.getpid()}"
    try:
        os.replace(path, claimed_path)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        data = json.loads(claimed_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            os.replace(claimed_path, root / f"{path.name}.parse_error")
        except OSError:
            pass
        return None
    if not isinstance(data, dict):
        try:
            os.replace(claimed_path, root / f"{path.name}.parse_error")
        except OSError:
            pass
        return None
    data = dict(data)
    data[_HANDOFF_CLAIM_KEY] = str(claimed_path)
    return data


def poll_broadcast_control():
    """Rank 0: claim one pending command for the next TP broadcast.

    Returns the command payload, or None.  The request receiver appends it to
    ``recv_reqs`` *before* the TP broadcast, so every rank executes the same
    command in the same scheduler iteration.  This is the only safe control
    point: a file barrier that makes one rank wait while another is inside a
    forward collective desynchronizes the TP ranks and deadlocks.

    Watchdog resumes live in unique ``ctlq.*.json`` files and are served
    first.  The external driver's single ``cmd.0.json`` slot is served next,
    so a status/import command can never overwrite a pending resume.  Every
    file is claimed with ``os.replace`` *before* it is read, so a writer that
    publishes the next command while this one is being parsed cannot have its
    file renamed away by this poll.
    """
    global _SEQ, _ITER
    if not _CTL_DIR:
        return None
    _ITER += 1
    iter_id = _ITER
    root = Path(_CTL_DIR)
    _prune_stale_control_files(root)
    data = None
    queue = []
    try:
        queue = sorted(
            root.glob(f"{_CTL_QUEUE_PREFIX}*.json"),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        queue = []
    for path in queue:
        data = _claim_control_file(path, root)
        if data is not None:
            break
    if data is None:
        cmd_path = root / f"cmd.{_CTRL_RANK}.json"
        if not cmd_path.exists():
            return None
        data = _claim_control_file(cmd_path, root)
        if data is None:
            return None
    data["_ctl_iter"] = iter_id
    data["_ctl_claim_mono"] = time.monotonic()
    return {_HANDOFF_CTL_KEY: True, "data": data}


def _peer_ctl_dir() -> Optional[Path]:
    """Directory of the peer decode role (source D0 -> target D1)."""
    explicit = os.environ.get("HANDOFF_WD_TARGET_CTL_DIR", "").strip()
    if explicit:
        return Path(explicit)
    if not _CTL_DIR:
        return None
    root = Path(_CTL_DIR)
    role = os.environ.get("HANDOFF_WD_TARGET_ROLE", "dst").strip() or "dst"
    return root.parent / role


def _peer_control(payload: dict, timeout: float) -> dict:
    """Send one command to every target TP rank and wait for all results.

    This is the same file protocol as ``p2d_driver`` (nonce-tagged
    ``cmd.0.json`` -> ``result.<rank>.json``), but it is used only on the
    controller-death recovery path.  The target's own ``.control.lock`` is
    held for the whole transaction so a live driver and the watchdog can
    never interleave a command.
    """
    if fcntl is None:
        return {"error": "fcntl unavailable"}
    peer = _peer_ctl_dir()
    if peer is None or not peer.is_dir():
        return {"error": "peer ctl dir missing"}
    try:
        info0 = json.loads((peer / "ready.0.json").read_text(encoding="utf-8"))
        n = max(1, int(info0.get("tp_size", 1)))
    except Exception as exc:
        return {"error": f"peer not ready: {exc!r}"}
    ranks = []
    for r in range(n):
        try:
            info = json.loads((peer / f"ready.{r}.json").read_text(encoding="utf-8"))
            if int(info.get("rank", -1)) != r:
                raise RuntimeError(f"rank mismatch in ready.{r}")
            os.kill(int(info["pid"]), 0)
            ranks.append(r)
        except Exception:
            continue
    if ranks != list(range(n)):
        return {"error": f"peer ranks incomplete: {ranks}"}
    nonce = secrets.token_hex(16)
    data = {**payload, "nonce": nonce}
    lock_fh = None
    try:
        lock_fh = open(peer / ".control.lock", "a+")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    return {"error": "peer control lock timeout"}
                time.sleep(0.02)
        for r in range(n):
            try:
                (peer / f"result.{r}.json").unlink()
            except FileNotFoundError:
                pass
        tmp = peer / f"cmd.0.tmp.wd.{os.getpid()}.{nonce}"
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, peer / "cmd.0.json")
        results = {}
        pending = set(range(n))
        while pending and time.monotonic() < deadline:
            for r in list(pending):
                try:
                    res = json.loads(
                        (peer / f"result.{r}.json").read_text(encoding="utf-8")
                    )
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    continue
                if res.get("nonce") != nonce:
                    continue
                results[r] = res
                pending.discard(r)
            if pending:
                time.sleep(0.02)
        if pending:
            return {
                "error": f"peer result timeout: ranks {sorted(pending)}",
                "results": results,
                "nonce": nonce,
            }
        return {"results": results, "nonce": nonce}
    except Exception as exc:
        return {"error": repr(exc)}
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                lock_fh.close()
            except OSError:
                pass


def _fence_target_import(
    rid: str, target_rid: str, epoch: str, was_imported: bool = False
) -> str:
    """Fence a possibly in-flight target import before resuming the source.

    The controller can die after sending Import but before the target's
    ``imported`` report reaches the source.  Resuming on that gap would make
    both sides owners.  The target gets a cancel tombstone for this migration
    epoch and an epoch-checked abort:

      committed       -> target owns output; source must stay frozen
      epoch_mismatch  -> target owns a newer epoch; source must stay frozen
      aborted         -> target had not committed; source may resume
      missing         -> safe only if the source never observed an import
      unreachable     -> cannot prove safety; source stays frozen
    """
    if not epoch:
        # Without the migration epoch we cannot scope a cancel tombstone or
        # prove that the target request is the same generation.
        return "unknown_epoch"
    if not target_rid:
        target_rid = f"dst-{rid}"
    peer = _peer_ctl_dir()
    if peer is None:
        return "unreachable"
    # The marker is the fence that a sleeping/late target import checks after
    # its own delay.  It must land before the target admits the import; the
    # control command below cannot execute while that command is sleeping.
    safe = f"cancel.{str(target_rid).replace('/', '_')}.{str(epoch)}.json"
    marker = peer / safe
    try:
        peer.mkdir(parents=True, exist_ok=True)
        tmp = peer / f"{safe}.tmp.{os.getpid()}"
        tmp.write_text(
            json.dumps({"at": time.time(), "source": str(rid)}),
            encoding="utf-8",
        )
        os.replace(tmp, marker)
    except OSError:
        return "unreachable"
    rv = _peer_control(
        {"cmd": "fence_import", "rid": target_rid, "epoch": str(epoch)},
        _WD_FENCE_TIMEOUT_S,
    )
    if rv.get("error"):
        return "unreachable"
    results = rv.get("results") or {}
    rows = list(results.values())
    if not rows:
        return "unreachable"
    if any(r.get("committed") for r in rows):
        return "committed"
    if any(r.get("epoch_mismatch") for r in rows):
        return "epoch_mismatch"
    if any(not (r.get("ok") and (r.get("aborted") or r.get("missing"))) for r in rows):
        return "error"
    if was_imported and any(r.get("missing") for r in rows):
        # The source already saw an import for this epoch, but the target no
        # longer has the request.  "Missing" is not proof that nothing was
        # committed; mirror the gateway's cancel fence and stay frozen.
        return "ambiguous"
    return "safe"


def split_handoff_controls(recv_reqs):
    """Strip broadcast control sentinels from ``recv_reqs``.

    Returns ``(reqs, controls)``.  Both lists are identical on every TP rank
    because the sentinels travel through the normal request broadcast.
    """
    if not recv_reqs:
        return recv_reqs or [], []
    reqs = []
    controls = []
    for item in recv_reqs:
        if isinstance(item, dict) and item.get(_HANDOFF_CTL_KEY):
            data = item.get("data")
            if isinstance(data, dict):
                controls.append(data)
            continue
        reqs.append(item)
    return reqs, controls


def execute_handoff_control(scheduler, data):
    """Run one control command on this rank and publish result.<rank>.json."""
    global _SEQ
    root, _, result_path = _ctl_paths(scheduler)
    claimed = None
    resume_claim = None
    if isinstance(data, dict) and _HANDOFF_CLAIM_KEY in data:
        # Copy: every TP rank executes the same broadcast payload object.
        claimed = data.get(_HANDOFF_CLAIM_KEY)
        data = {k: v for k, v in data.items() if k != _HANDOFF_CLAIM_KEY}
    if isinstance(data, dict) and "__resume_claim__" in data:
        resume_claim = data.get("__resume_claim__")
        data = {k: v for k, v in data.items() if k != "__resume_claim__"}
    t_exec = time.monotonic()
    try:
        result = _handle(scheduler, data)
        result = {"ok": True, **result, "nonce": data.get("nonce")}
    except Exception as exc:
        result = {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "nonce": data.get("nonce"),
        }
    t_done = time.monotonic()
    claim = data.get("_ctl_claim_mono")
    result["_ctl"] = {
        "cmd": data.get("cmd"),
        "rank": _tp_rank(scheduler),
        "iter": data.get("_ctl_iter"),
        "iter_period_ms": _ITER_PERIOD_MS,
        "claim_mono": claim,
        "exec_start_mono": t_exec,
        "exec_end_mono": t_done,
        "wait_before_exec_ms": (
            None if claim is None else round((t_exec - claim) * 1000.0, 3)
        ),
        "exec_ms": round((t_done - t_exec) * 1000.0, 3),
    }
    from sglang.srt.runtime_context import get_parallel

    from .tp import agree_control_result, scheduling_state

    parallel = get_parallel()
    result, fatal = agree_control_result(
        result,
        scheduling_state(scheduler),
        group=parallel.tp_group.cpu_group if parallel.tp_size > 1 else None,
        world_size=parallel.tp_size,
    )
    _write_result(result_path, result)
    if resume_claim:
        # Retire the durable claim only after the command actually ran.
        # If this rank crashed between poll and execute, the claim TTL lets
        # the watchdog publish a fresh command.
        try:
            os.unlink(resume_claim)
        except OSError:
            pass
    if claimed:
        # Only the claiming rank (0) owns the file; other ranks see ENOENT.
        _SEQ += 1
        suffix = "done" if result.get("ok") else "bad"
        try:
            os.replace(
                claimed,
                root / f"cmd.{_tp_rank(scheduler)}.{os.getpid()}.{_SEQ}.{suffix}",
            )
        except OSError:
            pass
        if _SEQ % 256 == 0:
            _prune_done_files(root, _CTRL_RANK)
    if fatal:
        raise RuntimeError(result["error"])


def maybe_handle_handoff_control(scheduler):
    """Per-iteration handoff housekeeping (control runs via recv broadcast)."""
    global _ITER_AT, _ITER_PERIOD_MS
    if not _CTL_DIR:
        return
    now = time.monotonic()
    _ITER_PERIOD_MS = (now - _ITER_AT) * 1000.0
    _ITER_AT = now
    scheduler._handoff_iter_id = _ITER
    scheduler._handoff_iter_period_ms = _ITER_PERIOD_MS
    reap_transfer_buffers()
    _ensure_frozen_watchdog(scheduler)
    _drain_watchdog_actions(scheduler)
    _expire_stale_reservations(scheduler)
    _expire_capacity(scheduler)
    _capacity_reconcile(scheduler)
    _expire_target_states(scheduler)
    # Source-side frees requested by hub socket threads are executed here, on
    # the scheduler thread, before any new command touches batch state.
    hub = getattr(scheduler, "_handoff_hub", None)
    if hub is not None:
        hub.drain_releases()
    # Stamp the completion time of target-side requests once (monotonic, same
    # clock as freeze/step timestamps) so drivers can read y128 from status.
    for st in list(_TARGET_STATES):
        try:
            if not st._state_notified.get("imported"):
                st.notify_source("imported")
            if st.committed_new_tokens() > 0 and not st._state_notified.get("commit"):
                st.notify_source("commit")
            if st.finished_at is None and st.req is not None and st.req.finished():
                st.finished_at = time.monotonic()
            if (
                st.req is not None
                and st.req.finished()
                and not st._state_notified.get("done")
            ):
                st.notify_source("done")
        except Exception:
            pass
    # While the source's KV sender thread runs, yield the GIL so the Python
    # scheduler's idle admission loop does not throttle the socket transfer.
    if getattr(scheduler, "_handoff_transfer_active", False) or getattr(
        scheduler, "_handoff_remote_active", False
    ):
        time.sleep(0.002)
    _maybe_auto_freeze(scheduler)
    _iter_profile(scheduler)


def _prune_done_files(root: Path, rank: int, keep: int = 200) -> None:
    """Bound the per-rank done/bad marker files in the control directory."""
    try:
        files = list(root.glob(f"cmd.{rank}.*.done")) + list(
            root.glob(f"cmd.{rank}.*.bad")
        )
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files[keep:]:
            try:
                path.unlink()
            except OSError:
                pass
    except Exception:
        pass


def _write_result(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _sampling_from(state_sampling: dict, max_new_tokens: int, tokenizer=None):
    from sglang.srt.sampling.sampling_params import SamplingParams

    sp = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=float(state_sampling.get("temperature", 0.0)),
        top_p=float(state_sampling.get("top_p", 1.0)),
        top_k=int(state_sampling.get("top_k", -1)),
        ignore_eos=bool(state_sampling.get("ignore_eos", True)),
    )
    # The normal HTTP path normalizes SamplingParams in the tokenizer manager.
    # Injected requests bypass that path, so do it here.
    sp.normalize(tokenizer)
    return sp


def _new_req(
    scheduler,
    rid,
    origin_ids,
    output_ids,
    sampling_params,
    return_logprob=False,
    top_logprobs_num=0,
    bootstrap_host=None,
    bootstrap_port=None,
    bootstrap_room=None,
    disagg_mode=None,
):
    from sglang.srt.managers.schedule_batch import Req

    req = Req(
        rid,
        "",
        array("q", [int(x) for x in origin_ids]),
        sampling_params,
        return_logprob=bool(return_logprob),
        top_logprobs_num=int(top_logprobs_num or 0),
        vocab_size=scheduler.model_config.vocab_size,
        eos_token_ids=scheduler.model_config.hf_eos_token_id,
        bootstrap_host=bootstrap_host,
        bootstrap_port=bootstrap_port,
        bootstrap_room=bootstrap_room,
        disagg_mode=disagg_mode,
    )
    req.tokenizer = scheduler.tokenizer
    req._handoff_req = True
    req.output_ids = array("q", [int(x) for x in output_ids])
    req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
    if return_logprob:
        # Only the tokens generated from here on are of interest.  Setting the
        # start to the end of the prefix keeps the prefill step from computing
        # input logprobs for the whole prompt (14K+ tokens under SWE traffic).
        req.logprob_start_len = len(origin_ids) + len(output_ids)
    return req


def _pd_prefill_request(prefill_url: str, body: dict) -> None:
    """Trigger the P-only server for one native PD bootstrap room.

    The decode worker has already registered its receiver with the prefill
    bootstrap server; this request only makes P run the prompt prefill and
    transfer KV to that receiver.  It runs in a daemon thread so the D-side
    scheduler control loop never blocks on P.
    """
    try:
        u = urlparse(prefill_url)
        host = u.hostname or "127.0.0.1"
        port = int(u.port or 80)
        path = u.path if u.path not in ("", "/") else "/v1/completions"
        payload = {
            "rid": body["rid"],
            "prompt": body["input_ids"],
            "max_new_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "ignore_eos": True,
            "stream": False,
            "bootstrap_host": body["bootstrap_host"],
            "bootstrap_port": body["bootstrap_port"],
            "bootstrap_room": body["bootstrap_room"],
        }
        conn = http.client.HTTPConnection(host, port, timeout=180)
        conn.request(
            "POST",
            path,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        if resp.status != 200:
            print(
                f"[handoff-pd] prefill {body['rid']} failed "
                f"{resp.status}: {raw[:300]!r}",
                flush=True,
            )
    except Exception as exc:
        print(
            f"[handoff-pd] prefill {body.get('rid')} error: {exc!r}",
            flush=True,
        )


def _logprob_kwargs(data):
    if not data.get("return_logprob"):
        return {}
    return {
        "return_logprob": True,
        "top_logprobs_num": int(data.get("top_logprobs_num", 0) or 0),
    }


def _remove_from_batches(scheduler, req):
    # A prefill-to-decode freeze happens before get_next_batch_to_run merges
    # last_batch, so the request may still live in last_batch rather than
    # running_batch. Drop it from both.
    for name in ("running_batch", "last_batch"):
        rb = getattr(scheduler, name, None)
        if rb is not None:
            if any(r is req for r in rb.reqs):
                keep = [i for i, r in enumerate(rb.reqs) if r is not req]
                rb.filter_batch(keep_indices=keep)
            # A frozen request must not leave batch_is_full latched; otherwise
            # later sources in the same C=4 prefill can never be admitted.
            rb.batch_is_full = False
    scheduler.waiting_queue = [r for r in scheduler.waiting_queue if r is not req]


def _sampling_dict(req):
    sp = req.sampling_params
    return {
        "max_new_tokens": int(sp.max_new_tokens),
        "temperature": float(sp.temperature),
        "top_p": float(sp.top_p),
        "top_k": int(sp.top_k),
        "ignore_eos": bool(sp.ignore_eos),
    }


def _contiguous_range(indices_cpu) -> tuple[int, int, bool]:
    """Return (start, end, is_exactly_contiguous) for a token-index vector."""
    n = int(indices_cpu.numel())
    if n == 0:
        return -1, -1, False
    start = int(indices_cpu[0])
    end = int(indices_cpu[-1]) + 1
    if end - start != n:
        return start, end, False
    expected = torch.arange(start, end, dtype=indices_cpu.dtype)
    return start, end, bool(torch.equal(indices_cpu, expected))


def _freeze_request(
    scheduler, req, mode="mooncake", out=None, epoch=None, target_rid=None
):
    from sglang.srt.runtime_context import get_memory

    t_entry = time.perf_counter()
    t_entry_mono = time.monotonic()
    # P4.4/P4.5 v1 assumes the request exclusively owns its prefix slots.
    # With Radix Cache a shared prefix has its own refcount/lock lifecycle;
    # the explicit release used by resume would bypass it.
    if (
        not get_memory().disable_radix_cache
        and os.environ.get("HANDOFF_ALLOW_RADIX", "0") != "1"
    ):
        raise RuntimeError(
            "D2D migration v1 requires --disable-radix-cache; "
            "set HANDOFF_ALLOW_RADIX=1 only after validating prefix refcounts"
        )
    if getattr(scheduler, "enable_overlap", False):
        raise RuntimeError("freeze requires non-overlap scheduling")
    kv_len = int(req.kv.kv_committed_len)
    if kv_len <= 0:
        raise RuntimeError("request has no committed KV yet")
    # Decode invariant at a step boundary: every generated token except the
    # last sampled one is in KV.  A request resumed after a failed handoff
    # re-extends its output tokens; freezing it before kv_committed_len has
    # caught up would copy freed/stale slots and corrupt the target KV.
    _out_len = len(req.output_ids)
    _prompt_len = len(req.origin_input_ids)
    _expected = _prompt_len + (_out_len - 1 if _out_len > 0 else 0)
    if kv_len < _expected:
        raise RuntimeError(
            f"resume_pending: kv_committed_len={kv_len} < expected={_expected}"
        )
    if req.kv.req_pool_idx is None:
        raise RuntimeError("request has no req_pool_idx")
    token_indices = scheduler.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, :kv_len
    ].clone()
    t_state = time.perf_counter()
    t_state_mono = time.monotonic()
    # Compute the contiguous KV page range on the host *before* t_freeze, so
    # the remote-attention service does not need a blocking D2H on its first
    # (critical-path) call under source prefill pressure.
    try:
        _range_stream = torch.cuda.Stream(priority=-1)
        _range_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_range_stream):
            _idx_cpu = token_indices.cpu()
        req._handoff_kv_range = _contiguous_range(_idx_cpu)
    except Exception:
        req._handoff_kv_range = None
    t_range = time.perf_counter()
    t_range_mono = time.monotonic()
    fast = FAST_FREEZE and mode in ("layer", "remote") and out is None
    t_export = time.perf_counter()
    if fast:
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        k0 = pool.get_key_buffer(0)
        layer_num = int(getattr(pool, "layer_num", 0))
        export_ms = 0.0
        kv_cpu = None
    else:
        pool = None
        k0 = None
        layer_num = 0
        kv_cpu = scheduler.token_to_kv_pool_allocator.get_cpu_copy(token_indices)
        export_ms = (time.perf_counter() - t_export) * 1000.0
    t_export_done = time.perf_counter()
    t_export_done_mono = time.monotonic()

    sampling = _sampling_dict(req)
    origin_ids = [int(x) for x in req.origin_input_ids]
    output_ids = [int(x) for x in req.output_ids]
    migration_epoch = str(epoch or secrets.token_hex(8))
    target_rid = str(target_rid or f"dst-{req.rid}")
    chunks, kv_heads, head_dim, kv_dtype = snapshot_layout(
        kv_len, fast=fast, layer_num=layer_num, key=k0, kv_cpu=kv_cpu
    )
    meta = {
        "rid": req.rid,
        "migration_epoch": migration_epoch,
        "target_rid": target_rid,
        "trace_tag": f"{int(time.time() * 1000)}",
        "origin_input_ids": origin_ids,
        "output_ids": output_ids,
        "kv_committed_len": kv_len,
        "seqlen": int(req.seqlen),
        "sampling": sampling,
        "chunks": chunks,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": "bfloat16" if kv_dtype == torch.bfloat16 else "float16",
        "fast_freeze": bool(fast),
    }

    result = {
        "rid": req.rid,
        "migration_epoch": migration_epoch,
        "target_rid": target_rid,
        "kv_committed_len": kv_len,
        "output_len": len(output_ids),
        "seqlen": int(req.seqlen),
        "transport": mode,
        "export_ms": export_ms,
        "fast_freeze": bool(fast),
        "t_freeze_monotonic": time.monotonic(),
        "trace_tag": meta["trace_tag"],
    }
    t_meta = time.perf_counter()
    t_meta_mono = time.monotonic()
    if mode == "mooncake":
        result["transfer"] = _start_mooncake_server(scheduler, req, meta, kv_cpu)
    elif mode in ("layer", "remote"):
        if fast:
            result["transfer"] = _start_layer_gpu_server(
                scheduler, req, meta, remote=(mode == "remote")
            )
        else:
            result["transfer"] = _start_layer_mooncake_server(
                scheduler, req, meta, kv_cpu, remote=(mode == "remote")
            )
    elif mode == "socket":
        result["transfer"] = _start_transfer_server(scheduler, meta, kv_cpu)
    else:
        if out is None:
            raise ValueError("file transport requires out")
        state = {
            "rid": req.rid,
            "origin_input_ids": origin_ids,
            "output_ids": output_ids,
            "kv_committed_len": kv_len,
            "seqlen": int(req.seqlen),
            "token_indices": token_indices.cpu(),
            "kv_cpu": kv_cpu,
            "sampling": sampling,
        }
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, out)
        result["state"] = out

    t_transfer = time.perf_counter()
    t_transfer_mono = time.monotonic()
    _remove_from_batches(scheduler, req)
    req._handoff_frozen = True
    req._handoff_freeze_result = result
    req._handoff_target_rid = target_rid
    req._handoff_epoch = migration_epoch
    if FREEZE_PROFILE:
        if fast:
            itemsize = 2 if kv_dtype in (torch.bfloat16, torch.float16) else 4
            nbytes = layer_num * 2 * kv_len * kv_heads * head_dim * itemsize
        else:
            nbytes = sum(
                int(t.numel()) * int(t.element_size())
                for layer in kv_cpu
                for pair in layer
                for t in pair
            )
        total_ms = (time.perf_counter() - t_entry) * 1000.0
        result["freeze_profile"] = {
            "state_ms": (t_state - t_entry) * 1000.0,
            "range_ms": (t_range - t_state) * 1000.0,
            "export_ms": (t_export_done - t_export) * 1000.0,
            "meta_ms": (t_meta - t_export_done) * 1000.0,
            "transfer_ms": (t_transfer - t_meta) * 1000.0,
            "remove_ms": (time.perf_counter() - t_transfer) * 1000.0,
            "total_ms": total_ms,
            "bytes": nbytes,
            "gbs": (None if not export_ms else nbytes / (export_ms / 1000.0) / 1e9),
            "kv_len": kv_len,
        }
    # Always-on, cheap stage timeline for drain attribution.  All values are
    # CLOCK_MONOTONIC on the source rank and are directly comparable with the
    # gateway/target timestamps (same host for the RTX 5090 test bed).
    result["timeline"] = {
        "enter_at": t_entry_mono,
        "state_at": t_state_mono,
        "range_at": t_range_mono,
        "export_at": t_export_done_mono,
        "meta_at": t_meta_mono,
        "transfer_at": t_transfer_mono,
        "remove_at": time.monotonic(),
        "return_at": time.monotonic(),
        "fast": bool(fast),
        "mode": mode,
    }
    _tl = result["timeline"]
    _tr = (result.get("transfer") or {}).get("timeline") or {}

    def _ms(a, b):
        if a is None or b is None:
            return None
        return round((b - a) * 1000.0, 1)

    print(
        f"[handoff-src] freeze rid={req.rid} mode={mode} fast={fast} "
        f"kv_len={kv_len} "
        f"state_ms={_ms(_tl['enter_at'], _tl['state_at'])} "
        f"range_ms={_ms(_tl['state_at'], _tl['range_at'])} "
        f"export_ms={_ms(_tl['range_at'], _tl['export_at'])} "
        f"meta_ms={_ms(_tl['export_at'], _tl['meta_at'])} "
        f"transfer_ms={_ms(_tl['meta_at'], _tl['transfer_at'])} "
        f"total_ms={_ms(_tl['enter_at'], _tl['return_at'])} "
        f"stage_alloc_ms={_ms(_tr.get('enter_at'), _tr.get('stage_alloc_at'))} "
        f"stage_reg_ms={_ms(_tr.get('stage_alloc_at'), _tr.get('stage_register_at'))} "
        f"remote_reg_ms={_ms(_tr.get('stage_register_at'), _tr.get('remote_register_at'))} "
        f"listener_ms={_ms(_tr.get('remote_register_at'), _tr.get('listener_at'))} "
        f"thread_ms={_ms(_tr.get('listener_at'), _tr.get('thread_start_at'))}",
        flush=True,
    )
    return result


def _maybe_auto_freeze(scheduler):
    """Freeze a request at the prefill boundary (P->D start)."""
    for req in list(_tracked(scheduler).values()):
        if not getattr(req, "_handoff_pause_after_prefill", False):
            continue
        if getattr(req, "_handoff_frozen", False) or req.finished():
            continue
        if int(req.kv.kv_committed_len) != len(req.origin_input_ids):
            continue
        if len(req.output_ids) < 1:
            continue
        req._handoff_pause_after_prefill = False
        try:
            _freeze_request(
                scheduler,
                req,
                mode=getattr(req, "_handoff_freeze_mode", "mooncake"),
                out=getattr(req, "_handoff_freeze_out", None),
            )
        except Exception as exc:
            req._handoff_freeze_error = repr(exc)
            # Swallowing this permits another TP rank to remove the request
            # while this rank keeps decoding it. Let the supervisor stop TP.
            raise RuntimeError("automatic handoff freeze failed") from exc


# ---------------------------------------------------------------------------
# target-side multiplexed RPC client
# ---------------------------------------------------------------------------
class _RemoteRpcError(RuntimeError):
    pass


class _RemoteDisconnected(ConnectionError):
    pass


class _Pending:
    __slots__ = ("rpc_id", "event", "result", "exc")

    def __init__(self, rpc_id):
        self.rpc_id = int(rpc_id)
        self.event = threading.Event()
        self.result = None
        self.exc = None

    def set_result(self, value):
        self.result = value
        self.event.set()

    def set_exception(self, exc):
        self.exc = exc
        self.event.set()


class TargetClient:
    """One long-lived connection per (target, source) scheduler pair.

    RPCs carry a client-assigned ``rpc_id``; a background reader demultiplexes
    responses. A disconnect fails every pending RPC, and the next call
    reconnects and replays every live ``attach``.

    Stats semantics: ``rpc_failed`` counts failed attempts, so a permanently
    failed retried RPC contributes 2; ``drops`` counts reader-detected drops
    only; ``last_rpc_ms`` covers the winning attempt only. A timeout kills the
    shared connection: other pending RPCs are woken as disconnected and retry
    once with a fresh full deadline (attention reads are idempotent).
    """

    def __init__(
        self, host, port, connect_timeout_s=10.0, rpc_deadline_s=600.0, name=None
    ):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.rpc_deadline_s = float(rpc_deadline_s)
        self._conn = None
        self._reader = None
        self._conn_lock = threading.RLock()
        # Guards the ``_conn`` pointer itself. The reader must be able to
        # clear ``_conn`` while a connector holds ``_conn_lock`` and waits for
        # a reader-delivered hello reply, so it cannot take ``_conn_lock``.
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._attached: dict[str, tuple[str, int]] = {}
        self._next_rpc = 1
        self._closed = False
        self._stats = {
            "connects": 0,
            "drops": 0,
            "attaches": 0,
            "rpc_sent": 0,
            "rpc_failed": 0,
            "rpc_retried": 0,
            "last_rpc_ms": 0.0,
        }
        self._name = name or f"target-client-{self.port}"

    # -- public API -----------------------------------------------------
    def attach(self, handoff_id, token, epoch):
        if epoch is None:
            raise ValueError("TargetClient.attach requires owner_epoch")
        hid = str(handoff_id)
        epoch = int(epoch)
        with self._conn_lock:
            prev = self._attached.get(hid)
            if prev == (str(token), epoch):
                return
            try:
                # Connect (and replay older live attaches) first, so this new
                # attach is not replayed once more by the same connect.
                conn = self._ensure_conn_locked()
                self._attached[hid] = (str(token), epoch)
                self._send_attach_locked(conn, hid, str(token), epoch)
            except Exception:
                if self._attached.get(hid) == (str(token), epoch):
                    if prev is None:
                        self._attached.pop(hid, None)
                    else:
                        self._attached[hid] = prev
                raise

    def detach(self, handoff_id):
        hid = str(handoff_id)
        with self._conn_lock:
            self._attached.pop(hid, None)
            with self._state_lock:
                conn = self._conn
            if conn is None:
                return
            try:
                with self._send_lock:
                    _send_json(conn, {"cmd": "detach", "handoff_id": hid})
            except OSError:
                self._kill_conn_locked(conn)

    def call(self, header, q_bytes, deadline_s=None):
        return self._call(header, q_bytes, deadline_s, "attn")

    def call_batch(self, header, q_bytes, deadline_s=None):
        return self._call(header, q_bytes, deadline_s, "attn_batch")

    def _call(self, header, q_bytes, deadline_s, cmd):
        deadline = float(deadline_s if deadline_s is not None else self.rpc_deadline_s)
        for attempt in (0, 1):
            conn = None
            try:
                with self._conn_lock:
                    conn = self._ensure_conn_locked()
                    hid = str(header.get("handoff_id", ""))
                    if hid and hid not in self._attached:
                        raise _RemoteRpcError(f"handoff {hid!r} is not attached")
                    fut = self._new_pending_locked(conn)
                    frame = dict(header)
                    frame["cmd"] = cmd
                    frame["rpc_id"] = fut.rpc_id
                    try:
                        with self._send_lock:
                            _send_json(conn, frame)
                            conn.sendall(q_bytes)
                    except OSError as exc:
                        self._kill_conn_locked(conn)
                        raise _RemoteDisconnected(repr(exc)) from None
                    except BaseException:
                        with self._pending_lock:
                            self._pending.pop(fut.rpc_id, None)
                        raise
                t0 = time.perf_counter()
                result = self._wait(fut, deadline)
                with self._pending_lock:
                    self._stats["rpc_sent"] += 1
                    self._stats["last_rpc_ms"] = (time.perf_counter() - t0) * 1000.0
                return result
            except _RemoteDisconnected as exc:
                with self._pending_lock:
                    self._stats["rpc_failed"] += 1
                if conn is not None:
                    # The reader may not have reaped the dead connection yet;
                    # clear it before retrying so the retry reconnects instead
                    # of sending onto a half-closed socket.
                    with self._conn_lock:
                        self._kill_conn_locked(conn)
                if attempt == 0:
                    with self._pending_lock:
                        self._stats["rpc_retried"] += 1
                    continue
                raise ConnectionError(str(exc)) from None
            except Exception:
                with self._pending_lock:
                    self._stats["rpc_failed"] += 1
                raise

    def stats(self) -> dict:
        with self._conn_lock:
            with self._state_lock:
                connected = self._conn is not None
            attached = len(self._attached)
            with self._pending_lock:
                out = dict(self._stats)
                out["in_flight"] = len(self._pending)
        out["connected"] = connected
        out["attached"] = attached
        return out

    def close(self):
        self._closed = True
        with self._conn_lock:
            with self._state_lock:
                conn = self._conn
                self._conn = None
            if conn is not None:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    conn.close()
                except OSError:
                    pass
        self._fail_pending(_RemoteDisconnected("target client closed"))
        if self._reader is not None:
            self._reader.join(timeout=1.0)

    # -- internals ------------------------------------------------------
    def _ensure_conn_locked(self):
        if self._closed:
            raise RuntimeError("target client closed")
        with self._state_lock:
            conn = self._conn
        if conn is not None:
            return conn
        conn = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        )
        conn.settimeout(None)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        with self._state_lock:
            self._conn = conn
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(conn,),
            daemon=True,
            name=f"{self._name}-reader",
        )
        self._reader.start()
        try:
            header = self._rpc_locked(
                conn, {"cmd": "hello", "proto": 2}, None, self.connect_timeout_s
            )
            if header.get("ok") is False:
                raise _RemoteRpcError(str(header.get("error")))
            for hid, (token, epoch) in list(self._attached.items()):
                try:
                    self._send_attach_locked(conn, hid, token, epoch)
                except _RemoteRpcError as exc:
                    self._attached.pop(hid, None)
                    print(
                        f"[target-client] dropping stale attach {hid}: {exc}",
                        flush=True,
                    )
        except Exception:
            self._kill_conn_locked(conn)
            raise
        with self._pending_lock:
            self._stats["connects"] += 1
        return conn

    def _rpc_locked(self, conn, header, payload, timeout_s):
        fut = self._new_pending_locked(conn)
        frame = dict(header)
        frame["rpc_id"] = fut.rpc_id
        try:
            with self._send_lock:
                _send_json(conn, frame)
                if payload is not None:
                    conn.sendall(payload)
        except OSError as exc:
            self._kill_conn_locked(conn)
            raise _RemoteDisconnected(repr(exc)) from None
        except BaseException:
            with self._pending_lock:
                self._pending.pop(fut.rpc_id, None)
            raise
        response = self._wait(fut, timeout_s)
        return response[0]

    def _send_attach_locked(self, conn, hid, token, epoch):
        with self._pending_lock:
            self._stats["attaches"] += 1
        header = self._rpc_locked(
            conn,
            {"cmd": "attach", "handoff_id": hid, "token": token, "epoch": epoch},
            None,
            self.connect_timeout_s,
        )
        if header.get("ok") is False:
            raise _RemoteRpcError(str(header.get("error")))

    def _new_pending_locked(self, conn):
        # Do not nest _pending_lock under _state_lock here: the reader holds
        # _state_lock then _pending_lock.
        with self._pending_lock:
            rid = self._next_rpc
            self._next_rpc += 1
            fut = _Pending(rid)
            self._pending[rid] = fut
        # The reader may have reaped ``conn`` before this RPC was registered
        # (it clears _conn before failing pending). Do not wait on a future
        # that can no longer be failed by anybody.
        with self._state_lock:
            alive = self._conn is conn
        if not alive:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise _RemoteDisconnected("hub connection closed")
        return fut

    def _wait(self, fut, deadline_s):
        if not fut.event.wait(max(0.001, float(deadline_s))):
            with self._pending_lock:
                self._pending.pop(fut.rpc_id, None)
            with self._conn_lock:
                with self._state_lock:
                    conn = self._conn
                if conn is not None:
                    self._kill_conn_locked(conn)
            raise TimeoutError(f"remote RPC {fut.rpc_id} timed out")
        if fut.exc is not None:
            raise fut.exc
        return fut.result

    def _read_loop(self, conn):
        try:
            while True:
                header = _recv_json(conn)
                rid = header.get("rpc_id")
                with self._pending_lock:
                    fut = self._pending.get(rid)
                ok = header.get("ok", True) is not False
                o = lse = None
                batch_rows = None
                batch_tensors = None
                if ok and header.get("o_batch_shape"):
                    dtype = {
                        "bfloat16": torch.bfloat16,
                        "float16": torch.float16,
                        "float32": torch.float32,
                    }[header["o_batch_dtype"]]
                    batch_tensors = (
                        _recv_tensor(conn, header["o_batch_shape"], dtype, pinned=True),
                        _recv_tensor(
                            conn,
                            header["lse_batch_shape"],
                            torch.float32,
                            pinned=True,
                        ),
                    )
                elif ok and header.get("cmd") == "attn_batch":
                    batch_rows = []
                    for row in header.get("rows") or []:
                        ro = rl = None
                        if "o_shape" in row:
                            ro = _recv_tensor(
                                conn,
                                row["o_shape"],
                                {
                                    "bfloat16": torch.bfloat16,
                                    "float16": torch.float16,
                                    "float32": torch.float32,
                                }[row["o_dtype"]],
                                pinned=True,
                            )
                            rl = _recv_tensor(
                                conn,
                                row["lse_shape"],
                                torch.float32,
                                pinned=True,
                            )
                        batch_rows.append((ro, rl))
                elif ok and "o_shape" in header:
                    o = _recv_tensor(
                        conn,
                        header["o_shape"],
                        {
                            "bfloat16": torch.bfloat16,
                            "float16": torch.float16,
                            "float32": torch.float32,
                        }[header["o_dtype"]],
                    )
                    lse = _recv_tensor(conn, header["lse_shape"], torch.float32)
                with self._pending_lock:
                    if fut is not None and self._pending.get(rid) is fut:
                        self._pending.pop(rid, None)
                if fut is None:
                    continue  # late or duplicate result: payload drained
                if not ok:
                    fut.set_exception(_RemoteRpcError(str(header.get("error"))))
                elif batch_rows is not None:
                    fut.set_result((header, batch_rows))
                elif batch_tensors is not None:
                    fut.set_result((header, batch_tensors))
                else:
                    fut.set_result((header, o, lse))
        except Exception as exc:
            pending = []
            with self._state_lock:
                if self._conn is conn:
                    self._conn = None
                    with self._pending_lock:
                        self._stats["drops"] += 1
                        pending = list(self._pending.values())
                        self._pending.clear()
            for fut in pending:
                fut.set_exception(_RemoteDisconnected(f"hub connection lost: {exc!r}"))
            try:
                conn.close()
            except OSError:
                pass

    def _fail_pending(self, exc):
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for fut in pending:
            fut.set_exception(exc)

    def _kill_conn_locked(self, conn):
        pending = []
        with self._state_lock:
            if self._conn is not conn:
                return
            self._conn = None
            with self._pending_lock:
                pending = list(self._pending.values())
                self._pending.clear()
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass
        for fut in pending:
            fut.set_exception(_RemoteDisconnected("hub connection closed"))


# ---------------------------------------------------------------------------
# target-side per-layer gating / remote attention
# ---------------------------------------------------------------------------
# Both are mutated only on the scheduler thread (non-overlap loops); hub,
# reader and receiver threads never touch them.
_TARGET_STATES = []
_TARGET_METRICS: dict[str, dict] = {}


def _snapshot_state(st) -> None:
    try:
        _TARGET_METRICS[str(st.handoff_id)] = st.metrics()
    except Exception:
        return
    while len(_TARGET_METRICS) > 16:
        _TARGET_METRICS.pop(next(iter(_TARGET_METRICS)))


_FB_CACHE = {
    "fb": None,
    "states": [],
    "rp": [],
    "seq": [],
    "by_rpi": None,
    "counted_fb": None,
    "mixed_counted": False,
}
_TARGET_CLIENTS: dict[tuple[str, int], TargetClient] = {}
_TARGET_CLIENTS_LOCK = threading.Lock()


def _target_client_for(service: dict) -> TargetClient:
    key = (str(service["host"]), int(service["port"]))
    with _TARGET_CLIENTS_LOCK:
        client = _TARGET_CLIENTS.get(key)
        if client is None:
            client = TargetClient(key[0], key[1])
            _TARGET_CLIENTS[key] = client
        return client


def _close_target_clients() -> None:
    with _TARGET_CLIENTS_LOCK:
        clients = list(_TARGET_CLIENTS.values())
        _TARGET_CLIENTS.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


def _reset_target_states():
    for st in list(_TARGET_STATES):
        stop_target_receivers(st.req)
    _TARGET_STATES.clear()
    _TARGET_METRICS.clear()
    _close_target_clients()
    with _M4_LOCK:
        _M4_STATS.update(
            {
                "steps": 0,
                "batch4_steps": 0,
                "all_steps": 0,
                "all_batch4_steps": 0,
                "mixed_steps": 0,
                "local_rows": 0,
                "remote_rows": 0,
                "max_remote_wait_ms": 0.0,
            }
        )
    _FB_CACHE.update(
        {
            "fb": None,
            "states": [],
            "rp": [],
            "seq": [],
            "by_rpi": None,
            "counted_fb": None,
            "mixed_counted": False,
            "all_counted_fb": None,
        }
    )


def _bind_and_select_states(req_pool_indices) -> list:
    """Bind states to their Req by object identity, prune finished ones."""
    rp = {int(x) for x in req_pool_indices}
    selected = []
    for st in list(_TARGET_STATES):
        req = getattr(st, "req", None)
        target = getattr(req, "_pre_layerwise_target", None)
        leased = (
            target is not None
            and getattr(target.owner, "decode_overlap", None) is not None
            and not target.closed
        )
        # Overlap may have already prepared an overshoot decode before the
        # previous result marks this Req finished. It still needs Remote KV.
        # Only the TP retirement path may prune its state or close its client.
        if req is not None and req.finished() and not leased:
            _snapshot_state(st)
            try:
                _TARGET_STATES.remove(st)
            except ValueError:
                pass
            try:
                st.close()
            except Exception:
                pass
            continue
        if req is not None:
            cur = req.kv.req_pool_idx
            was = bool(getattr(st, "_admitted", False))
            if leased and (cur is None or (was and int(cur) != st.req_pool_idx)):
                raise RuntimeError("leased D request row changed before TP retirement")
            if cur is None:
                if was:
                    # Previously admitted, now freed/retracted: the imported
                    # prefix is gone; invalidate rather than rebind later.
                    _snapshot_state(st)
                    try:
                        _TARGET_STATES.remove(st)
                    except ValueError:
                        pass
                    try:
                        st.close()
                    except Exception:
                        pass
                # else: imported but not admitted yet; keep unbound.
                continue
            cur = int(cur)
            if was and st.req_pool_idx is not None and cur != st.req_pool_idx:
                _snapshot_state(st)
                try:
                    _TARGET_STATES.remove(st)
                except ValueError:
                    pass
                try:
                    st.close()
                except Exception:
                    pass
                continue
            st.req_pool_idx = cur
            st._admitted = True
        if st.req_pool_idx in rp:
            selected.append(st)
    return selected


def _drop_target_state_for(req) -> None:
    for st in list(_TARGET_STATES):
        if getattr(st, "req", None) is req:
            _snapshot_state(st)
            try:
                _TARGET_STATES.remove(st)
            except ValueError:
                pass
            try:
                st.close()
            except Exception:
                pass
    _FB_CACHE.update(
        {
            "fb": None,
            "states": [],
            "rp": [],
            "seq": [],
            "by_rpi": None,
            "counted_fb": None,
            "mixed_counted": False,
        }
    )


def _row_meta(forward_batch):
    # Count every ForwardBatch once, whether or not it carries handoff states
    # (the copy_wait arm has no target states, but its steps still matter).
    # Called per attention layer, so the identity cache dedupes to one count.
    if _FB_CACHE.get("all_counted_fb") is not forward_batch:
        rpi = getattr(forward_batch, "req_pool_indices", None)
        try:
            n_rows = int(rpi.numel()) if rpi is not None else 0
        except Exception:
            n_rows = 0
        with _M4_LOCK:
            if _FB_CACHE.get("all_counted_fb") is not forward_batch:
                _FB_CACHE["all_counted_fb"] = forward_batch
                _M4_STATS["all_steps"] += 1
                if n_rows == 4:
                    _M4_STATS["all_batch4_steps"] += 1
    if not _TARGET_STATES:
        if _FB_CACHE["fb"] is not forward_batch:
            _FB_CACHE.update(
                {
                    "fb": forward_batch,
                    "states": [],
                    "rp": [],
                    "seq": [],
                    "by_rpi": {},
                    "counted_fb": None,
                    "mixed_counted": False,
                }
            )
        return _FB_CACHE
    if _FB_CACHE["fb"] is forward_batch and _FB_CACHE.get("by_rpi") is not None:
        return _FB_CACHE
    rpi = forward_batch.req_pool_indices
    seq = forward_batch.seq_lens
    rp = [int(x) for x in rpi.tolist()] if rpi is not None else []
    seqs = [int(x) for x in seq.tolist()] if seq is not None else []
    states = _bind_and_select_states(rp) if rp else []
    by_rpi = {}
    for st in states:
        st.begin_step(forward_batch)
        by_rpi[int(st.req_pool_idx)] = st
    _FB_CACHE.update(
        {
            "fb": forward_batch,
            "states": states,
            "rp": rp,
            "seq": seqs,
            "by_rpi": by_rpi,
            "counted_fb": None,
            "mixed_counted": False,
        }
    )
    return _FB_CACHE


def _check_remote_header(header: dict, handoff_id: Optional[str] = None) -> None:
    """Raise if the source rejected this Q/RPC (e.g. a stale epoch)."""
    if header.get("ok") is False:
        raise RuntimeError(
            f"remote attention rejected for {handoff_id!r}: {header.get('error')!r}"
        )


_M4_LOCK = threading.Lock()
_M4_STATS = {
    "steps": 0,
    "batch4_steps": 0,
    "all_steps": 0,
    "all_batch4_steps": 0,
    "mixed_steps": 0,
    "local_rows": 0,
    "remote_rows": 0,
    "max_remote_wait_ms": 0.0,
}


def _get_kv_pool():
    from sglang.srt.model_executor.forward_context import get_attn_backend

    return get_attn_backend().token_to_kv_pool


def has_target_states() -> bool:
    return bool(_TARGET_STATES)


def has_remote_work() -> bool:
    """True while replay would bypass an import wait, RPC, or failure check."""
    if os.environ.get("HANDOFF_GRAPH_DISABLE_ANY_STATE", "0") == "1":
        return bool(_TARGET_STATES)
    for st in _TARGET_STATES:
        if st.error is not None or st.aborted:
            return True
        # Layerwise mode also needs Python to wait for the missing KV. Graph
        # replay bypasses that wait just as it bypasses Remote attention RPCs.
        if any(not e.is_set() for e in st.events):
            return True
    return False


def graph_step_begin(forward_batch) -> None:
    """Record handoff step metrics before a CUDA-graph replay.

    Replay bypasses the attention wrapper, so the normal begin_step path is
    not called.  This keeps step_starts / masks / state binding consistent for
    post-window (all-local) graph steps.
    """
    try:
        _row_meta(forward_batch)
    except Exception:
        pass


def _single_decode(q, k, v, sm_scale):
    import flashinfer

    return flashinfer.single_decode_with_kv_cache(
        q,
        k,
        v,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        sm_scale=float(sm_scale),
        return_lse=True,
    )


def _merge_states(local_o, local_lse, o, lse):
    import flashinfer

    return flashinfer.merge_state(
        local_o.unsqueeze(0),
        local_lse.unsqueeze(0),
        o.unsqueeze(0),
        lse.unsqueeze(0),
    )


def _merge_states_batch(local_o, local_lse, o, lse):
    """merge_state over B independent rows (already batched)."""
    import flashinfer

    return flashinfer.merge_state(local_o, local_lse, o, lse)


_PINNED_CACHE: dict = {}
_PINNED_LOCK = threading.Lock()
_COPY_STREAMS: dict = {}
_COPY_STREAMS_LOCK = threading.Lock()
_Q_EVENTS: dict = {}


def _q_events(device):
    idx = int(getattr(device, "index", None) or 0)
    evs = _Q_EVENTS.get(idx)
    if evs is None:
        evs = {
            "ready": torch.cuda.Event(),
            "done": torch.cuda.Event(),
        }
        _Q_EVENTS[idx] = evs
    return evs


def _copy_stream(device):
    idx = int(getattr(device, "index", None) or 0)
    with _COPY_STREAMS_LOCK:
        stream = _COPY_STREAMS.get(idx)
        if stream is None:
            stream = torch.cuda.Stream(device=idx)
            _COPY_STREAMS[idx] = stream
        return stream


def _pinned_empty(shape, dtype):
    key = (tuple(int(x) for x in shape), dtype)
    with _PINNED_LOCK:
        buf = _PINNED_CACHE.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            _PINNED_CACHE[key] = buf
        return buf


def _h2d_pair(o_cpu, lse_cpu, device):
    """One copy stream + one completion event covers both O and LSE H2D."""
    try:
        if not ASYNC_COPY or not torch.cuda.is_available():
            raise RuntimeError("fallback")
        cs = _copy_stream(device)
        cur = torch.cuda.current_stream(device)
        ev = torch.cuda.Event()
        with torch.cuda.stream(cs):
            o_gpu = torch.empty(o_cpu.shape, dtype=o_cpu.dtype, device=device)
            lse_gpu = torch.empty(lse_cpu.shape, dtype=lse_cpu.dtype, device=device)
            o_gpu.copy_(o_cpu, non_blocking=True)
            lse_gpu.copy_(lse_cpu, non_blocking=True)
            ev.record()
        cur.wait_event(ev)
        # Allocated on the copy stream, consumed on the compute stream.
        o_gpu.record_stream(cur)
        lse_gpu.record_stream(cur)
        return o_gpu, lse_gpu
    except Exception:
        return o_cpu.to(device), lse_cpu.to(device)


def _batch_q_bytes(q_rows, ready_event=None):
    """Return (concatenated Q bytes, pinned D2H ms) for one layer's rows.

    The copy runs on a dedicated stream.  ``ready_event`` must be recorded on
    the compute stream *before* the local-tail kernels are launched; the copy
    stream then waits only for Q (not for the tails), so both overlap.
    Falls back to per-row blocking copies if pinned memory or stream copies are
    unavailable.
    """
    if not q_rows:
        return b"", 0.0
    try:
        shapes = {tuple(r.shape) for r in q_rows}
        dtypes = {r.dtype for r in q_rows}
        if (
            not ASYNC_COPY
            or not torch.cuda.is_available()
            or len(shapes) != 1
            or len(dtypes) != 1
            or any(not r.is_cuda for r in q_rows)
        ):
            raise RuntimeError("fallback")
        n = len(q_rows)
        shape = (n,) + tuple(q_rows[0].shape)
        buf = _pinned_empty(shape, q_rows[0].dtype)
        dev = q_rows[0].device
        cs = _copy_stream(dev)
        cur = torch.cuda.current_stream(dev)
        evs = _q_events(dev)
        if ready_event is not None:
            cs.wait_event(ready_event)
        else:
            evs["ready"].record(cur)
            cs.wait_event(evs["ready"])
        t0 = time.perf_counter()
        with torch.cuda.stream(cs):
            q = q_rows[0] if n == 1 else torch.stack(q_rows, 0)
            buf.copy_(q, non_blocking=True)
            evs["done"].record(cs)
        # CPU may read the pinned buffer only after the DMA really completed.
        evs["done"].synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        raw = buf.view(torch.uint8).reshape(-1).numpy().tobytes()
        return raw, ms
    except Exception:
        parts = [
            r.contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
            for r in q_rows
        ]
        return b"".join(parts), 0.0


def _h2d_async(cpu_tensor, device):
    """Pinned CPU -> GPU copy on the copy stream; returns a GPU tensor."""
    try:
        if not ASYNC_COPY or not torch.cuda.is_available():
            raise RuntimeError("fallback")
        gpu = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype, device=device)
        cs = _copy_stream(device)
        cur = torch.cuda.current_stream(device)
        cs.wait_stream(cur)
        with torch.cuda.stream(cs):
            gpu.copy_(cpu_tensor, non_blocking=True)
        cur.wait_stream(cs)
        return gpu
    except Exception:
        return cpu_tensor.to(device)


def _get_attn_backend_req_to_token():
    from sglang.srt.model_executor.forward_context import get_attn_backend

    return get_attn_backend().req_to_token_pool.req_to_token


def _batch_remote_attention(module, q, k, v, forward_batch, meta, save_kv_cache):
    """Per-row decode attention for a batch containing at least one remote row.

    KV is written once for the whole batch; LOCAL rows use a single-decode
    over their full KV, REMOTE rows merge the local tail with the source-side
    prompt attention. Output is composed in the original row order.
    """
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc

    t_func0 = time.perf_counter()
    rp = meta["rp"]
    seq = meta["seq"]
    by_rpi = meta["by_rpi"]
    n = len(rp)
    q3 = q.reshape(n, module.tp_q_head_num, module.qk_head_dim)
    k3 = (
        k.reshape(n, module.tp_k_head_num, module.qk_head_dim)
        if k is not None
        else None
    )
    v3 = (
        v.reshape(n, module.tp_v_head_num, module.v_head_dim) if v is not None else None
    )
    pool = _get_kv_pool()
    if save_kv_cache and k3 is not None:
        t_kv = time.perf_counter()
        pool.set_kv_buffer(module, KVWriteLoc.for_batch(forward_batch), k3, v3)
        if P22_PROFILE:
            kv_ms = (time.perf_counter() - t_kv) * 1000.0
            for st in meta.get("states") or []:
                st._prof("kv_write_ms", kv_ms)
    k_buf = pool.get_key_buffer(module.layer_id)
    v_buf = pool.get_value_buffer(module.layer_id)
    out_dtype = v3.dtype if v3 is not None else q3.dtype
    if out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        out_dtype = torch.bfloat16
    out = torch.empty(
        (n, module.tp_q_head_num * module.v_head_dim),
        dtype=out_dtype,
        device=q3.device,
    )
    remote_rows = 0
    local_rows = 0
    batch_remote_wait = 0.0
    if not BATCH_ROWS:
        for i, rpi in enumerate(rp):
            st = by_rpi.get(int(rpi))
            if st is not None and st.is_remote_layer(module.layer_id):
                remote_rows += 1
                seq_len = int(seq[i])
                row = st.remote_row(
                    module,
                    q3[i],
                    k_buf,
                    v_buf,
                    seq_len,
                    seq_len - int(st.base_len),
                    row_index=i,
                )
                out[i].copy_(row.reshape(-1))
                batch_remote_wait = max(
                    batch_remote_wait,
                    float(getattr(st, "last_rpc_wall_ms", 0.0) or 0.0),
                )
            else:
                local_rows += 1
                idx = _get_attn_backend_req_to_token()[int(rpi), : int(seq[i])]
                t_local = time.perf_counter()
                o, _lse = _single_decode(q3[i], k_buf[idx], v_buf[idx], module.scaling)
                out[i].copy_(o.reshape(-1))
                if st is not None:
                    st.local_calls += 1
                    st._step_local += 1
                    st._prof("local_rows", 1)
                    st._prof(
                        "local_attn_ms",
                        (time.perf_counter() - t_local) * 1000.0,
                    )
    else:
        has_remote = any(
            by_rpi.get(int(rpi)) is not None
            and by_rpi[int(rpi)].is_remote_layer(module.layer_id)
            for rpi in rp
        )
        q_ready_ev = None
        if has_remote and not Q_SYNC and torch.cuda.is_available():
            try:
                q_ready_ev = _q_events(torch.cuda.current_device())["ready"]
                q_ready_ev.record()
            except Exception:
                q_ready_ev = None
        prepared = []
        for i, rpi in enumerate(rp):
            st = by_rpi.get(int(rpi))
            if st is not None and st.is_remote_layer(module.layer_id):
                remote_rows += 1
                seq_len = int(seq[i])
                prepared.append(
                    st.prepare_remote_row(
                        module,
                        q3[i],
                        k_buf,
                        v_buf,
                        seq_len,
                        seq_len - int(st.base_len),
                        row_index=i,
                    )
                )
            else:
                local_rows += 1
                idx = _get_attn_backend_req_to_token()[int(rpi), : int(seq[i])]
                t_local = time.perf_counter()
                o, _lse = _single_decode(q3[i], k_buf[idx], v_buf[idx], module.scaling)
                out[i].copy_(o.reshape(-1))
                if st is not None:
                    st.local_calls += 1
                    st._step_local += 1
                    st._prof("local_rows", 1)
                    st._prof(
                        "local_attn_ms",
                        (time.perf_counter() - t_local) * 1000.0,
                    )
        # One RPC per (source endpoint, layer): all remote rows of this layer
        # travel together but keep their own handoff_id/epoch/seq metadata.
        groups = {}
        for prep in prepared:
            st = prep["state"]
            if st.client is None:
                st.error = "remote_progressive has no target client"
                raise RuntimeError(st.error)
            groups.setdefault((st.client.host, st.client.port), []).append(prep)
        for group in groups.values():
            client = group[0]["state"].client
            for prep in group:
                st = prep["state"]
                try:
                    client.attach(
                        st.handoff_id,
                        str(st.remote_service["token"]),
                        st.owner_epoch,
                    )
                except Exception as exc:
                    st.error = repr(exc)
                    raise
            if Q_SYNC:
                q_blob = b"".join(p["q_bytes"] for p in group)
                q_d2h_ms = sum(float(p.get("q_sync_ms") or 0.0) for p in group)
            else:
                q_blob, q_d2h_ms = _batch_q_bytes(
                    [p["q_row"] for p in group], ready_event=q_ready_ev
                )
            for prep in group:
                prep["q_d2h_ms"] = q_d2h_ms / max(1, len(group))
            batch_header = {
                "cmd": "attn_batch",
                "batch_size": len(group),
                "total_q_bytes": len(q_blob),
                "send_ts": time.monotonic(),
                "rows": [p["row_header"] for p in group],
            }
            t_rpc = time.perf_counter()
            try:
                header, payload = client.call_batch(batch_header, q_blob)
            except Exception as exc:
                for prep in group:
                    prep["state"].error = repr(exc)
                raise
            t_call = time.perf_counter()
            rows_resp = header.get("rows") or []
            if len(rows_resp) != len(group):
                raise RuntimeError(
                    "attn_batch result count mismatch: "
                    f"{len(rows_resp)} rows for {len(group)} requests"
                )
            # P1.2 fast path: one pinned O/LSE batch + one batch merge.
            if (
                isinstance(payload, tuple)
                and len(payload) == 2
                and torch.is_tensor(payload[0])
            ):
                o_batch, lse_batch = payload
                device = group[0]["local_o"].device
                t_post0 = time.perf_counter()
                if H2D_SINGLE:
                    o_gpu, lse_gpu = _h2d_pair(o_batch, lse_batch, device)
                else:
                    o_gpu = _h2d_async(o_batch, device)
                    lse_gpu = _h2d_async(lse_batch, device)
                merged_batch = None
                try:
                    local_o_batch = torch.stack([p["local_o"] for p in group])
                    local_lse_batch = torch.stack([p["local_lse"] for p in group])
                    merged_batch, _ = _merge_states_batch(
                        local_o_batch, local_lse_batch, o_gpu, lse_gpu
                    )
                except Exception as exc:
                    group[0]["state"].error = repr(exc)
                    merged_batch = None
                post_ms = (time.perf_counter() - t_post0) * 1000.0
                if merged_batch is not None:
                    for j, (prep, row_resp) in enumerate(zip(group, rows_resp)):
                        st = prep["state"]
                        merged_row = merged_batch[j : j + 1]
                        st.finish_remote_row(
                            prep,
                            row_resp,
                            None,
                            None,
                            t_rpc,
                            t_call,
                            batch_size=len(group),
                            batch_rpc_id=header.get("rpc_id"),
                            batch_queue_ms=header.get("queue_ms"),
                            batch_exec_ms=header.get("exec_ms"),
                            merged=merged_row,
                            post_ms=post_ms / max(1, len(group)),
                            q_d2h_ms=prep["q_d2h_ms"],
                        )
                        out[prep["row_index"]].copy_(merged_row.squeeze(0).reshape(-1))
                        batch_remote_wait = max(
                            batch_remote_wait,
                            float(getattr(st, "last_rpc_wall_ms", 0.0) or 0.0),
                        )
                    continue
                # Batch merge failed: fall back to per-row H2D + merge.
                payload = [(o_batch[j], lse_batch[j]) for j in range(len(group))]
            if len(payload) != len(group):
                raise RuntimeError(
                    "attn_batch tensor count mismatch: "
                    f"{len(payload)} tensors for {len(group)} requests"
                )
            for prep, row_resp, (o_cpu, lse_cpu) in zip(group, rows_resp, payload):
                st = prep["state"]
                merged = st.finish_remote_row(
                    prep,
                    row_resp,
                    o_cpu,
                    lse_cpu,
                    t_rpc,
                    t_call,
                    batch_size=len(group),
                    batch_rpc_id=header.get("rpc_id"),
                    batch_queue_ms=header.get("queue_ms"),
                    batch_exec_ms=header.get("exec_ms"),
                    q_d2h_ms=prep["q_d2h_ms"],
                )
                out[prep["row_index"]].copy_(merged.reshape(-1))
                batch_remote_wait = max(
                    batch_remote_wait,
                    float(getattr(st, "last_rpc_wall_ms", 0.0) or 0.0),
                )
    first_layer = meta.get("counted_fb") is not forward_batch
    if first_layer:
        meta["counted_fb"] = forward_batch
        with _M4_LOCK:
            _M4_STATS["steps"] += 1
            if n == 4:
                _M4_STATS["batch4_steps"] += 1
    if remote_rows and local_rows and not meta.get("mixed_counted"):
        meta["mixed_counted"] = True
        with _M4_LOCK:
            _M4_STATS["mixed_steps"] += 1
    with _M4_LOCK:
        # local_rows/remote_rows count row-layers (one per module invocation).
        _M4_STATS["local_rows"] += local_rows
        _M4_STATS["remote_rows"] += remote_rows
        _M4_STATS["max_remote_wait_ms"] = max(
            _M4_STATS["max_remote_wait_ms"], batch_remote_wait
        )
    if P22_PROFILE:
        func_ms = (time.perf_counter() - t_func0) * 1000.0
        for st in meta.get("states") or []:
            st._prof("attn_ms", func_ms)
            st._prof("layers", 1)
    return out


def _can_serve_remote(state) -> bool:
    """A failed local import can retain its frozen source prefix via RPC."""
    return (
        getattr(state, "mode", None) == "remote"
        and getattr(state, "client", None) is not None
        and not getattr(state, "aborted", False)
        and not getattr(state, "receiver_unsafe", False)
        and not (
            getattr(state, "abort_event", None) is not None
            and state.abort_event.is_set()
        )
    )


def _check_target_state(state):
    error = getattr(state, "error", None)
    if (
        getattr(state, "aborted", False)
        or getattr(state, "receiver_unsafe", False)
        or (
            getattr(state, "abort_event", None) is not None
            and state.abort_event.is_set()
        )
    ):
        raise RuntimeError(f"handoff import is not usable: {error or 'aborted/unsafe'}")
    if error is None:
        return
    if _can_serve_remote(state):
        if not getattr(state, "_downgrade_logged", False):
            state._downgrade_logged = True
            print(
                f"[handoff-target] remote downgrade id={state.handoff_id} reason={error}",
                flush=True,
            )
        return
    raise RuntimeError(f"handoff import is not usable: {error}")


def _record_local_import_error(state, error):
    state.error = str(error)
    # Remote events are data-ready bits, not error wakeups. A dropped or
    # incomplete layer must remain remote in every subsequent step mask.
    if not _can_serve_remote(state):
        for event in state.events:
            event.set()


def _handoff_attention_forward(
    module, q, k, v, forward_batch, save_kv_cache=True, **kw
):
    """Dispatch one RadixAttention call for a batch with handoff states."""
    meta = _row_meta(forward_batch)
    states = meta["states"]
    if not states:
        return module._handoff_orig_forward(
            q, k, v, forward_batch, save_kv_cache=save_kv_cache, **kw
        )
    for st in states:
        _check_target_state(st)
        if st.mode == "layer":
            ev = st.events[module.layer_id]
            if not ev.is_set():
                st.wait_calls += 1
                if not ev.wait(300):
                    raise TimeoutError(
                        f"handoff layer {module.layer_id} did not arrive"
                    )
            _check_target_state(st)
    remote = [st for st in states if st.is_remote_layer(module.layer_id)]
    if not remote:
        for st in states:
            st.local_calls += 1
            st._step_local += 1
        t_local_layer = time.perf_counter()
        out = module._handoff_orig_forward(
            q, k, v, forward_batch, save_kv_cache=save_kv_cache, **kw
        )
        if P22_PROFILE:
            dur = (time.perf_counter() - t_local_layer) * 1000.0
            for st in states:
                st._prof("full_local_attn_ms", dur)
                st._prof("full_local_layers", 1)
        return out
    if forward_batch.forward_mode.is_mixed():
        raise RuntimeError("MIXED prefill+decode batches are not supported in M4")
    n_rows = len(meta["rp"])
    if q.shape[0] != n_rows:
        raise RuntimeError(
            "mixed LOCAL/REMOTE attention needs one token per row "
            f"(got {q.shape[0]} tokens for {n_rows} rows)"
        )
    if getattr(forward_batch, "mha_return_lse", False):
        raise RuntimeError("mha_return_lse is not supported in M4")
    return _batch_remote_attention(module, q, k, v, forward_batch, meta, save_kv_cache)


class _TargetHandoffState:
    def __init__(
        self,
        req_pool_idx,
        base_len,
        events,
        mode,
        remote_service=None,
        handoff_id=None,
        trace_tag=None,
        req=None,
        tp_rank=None,
        base_output_len=0,
        migration_epoch=None,
        host_staging_bytes=0,
    ):
        self.req_pool_idx = None if req_pool_idx is None else int(req_pool_idx)
        self.base_len = int(base_len)
        self.base_output_len = int(base_output_len or 0)
        self.migration_epoch = migration_epoch
        self.host_staging_bytes = int(host_staging_bytes or 0)
        self.receiver_conn = None
        self.receiver_thread = None
        self.abort_event = threading.Event()
        self.handoff_id = handoff_id or "unknown"
        self.trace_tag = trace_tag or f"{int(time.time() * 1000)}"
        self.req = req
        self._pending_fallback = False
        self._admitted = False
        self.trace_path = Path("/workspace/handoff_state") / (
            f"target_trace_{self.trace_tag}.jsonl"
        )
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_fh = (
            self.trace_path.open("a", buffering=1 if PROFILE else (1 << 20))
            if (PROFILE or RPC_PROFILE)
            else None
        )
        self.trace_lock = threading.Lock()
        self.events = events
        self.mode = mode
        self.remote_service = remote_service
        _epoch = (remote_service or {}).get("epoch")
        self.owner_epoch = int(_epoch) if _epoch is not None else None
        self.error = None
        self.aborted = False
        self.client = (
            _target_client_for(remote_service)
            if remote_service and remote_service.get("host") is not None
            else None
        )
        self._step_fb = None
        self._step_mask = None
        self.remote_calls = 0
        self.local_calls = 0
        self.wait_calls = 0
        self.remote_fallback = 0
        self.last_remote_ms = None
        self.last_pre_ms = None
        self.last_kernel_ms = None
        self.last_queue_ms = None
        self.last_rpc_wall_ms = None
        self.last_recv_delay_ms = None
        self.remote_stats = []
        self.remote_latencies = []
        self.remote_ms_threshold = float(
            os.environ.get("HANDOFF_REMOTE_FALLBACK_MS", "10")
        )
        self.step_log = []
        self._step_fb = None
        self._step_remote = 0
        self._step_local = 0
        self._step_seq = None
        self._step_t = None
        self._step_index = 0
        self._cur_step = None
        self.created_at = time.monotonic()
        self.first_token_at: Optional[float] = None
        self._token_watch_stop = threading.Event()
        if req is not None:
            threading.Thread(
                target=self._watch_first_token,
                daemon=True,
                name=f"handoff-first-token-{self.trace_tag}",
            ).start()
        self.admitted_at = None
        self.finished_at = None
        self._state_notified = {"imported": False, "commit": False, "done": False}
        self._state_notify_lock = threading.Lock()
        self.step_starts: list[float] = []
        # P0: remote-window attribution. Counts are appended per completed step
        # so ITL can be split into remote vs post-local intervals.
        self.tp_rank = tp_rank
        self.step_remote_counts: list[int] = []
        self.rpc_n = 0
        self.rpc_stage: dict[str, float] = {}
        self.rpc_layer: dict[int, dict] = {}
        self.rpc_step: dict[int, dict] = {}
        self.rpc_top: list[dict] = []
        self._rpc_step_records: list[dict] = []
        self.layer_ready_at: list[Optional[float]] = [None] * len(events)
        self.layer_msg_at: list[Optional[float]] = [None] * len(events)
        self.layer_wait_ms: list[Optional[float]] = [None] * len(events)
        self.layer_copy_ms: list[Optional[float]] = [None] * len(events)
        self.layer_sync_ms: list[Optional[float]] = [None] * len(events)
        self.layer_bytes: list[Optional[int]] = [None] * len(events)
        self.import_done_at: Optional[float] = None
        self.last_remote_at: Optional[float] = None
        self.first_remote_at: Optional[float] = None
        self.import_timeline: Optional[dict] = None
        self.step_remote_layers: list[int] = []
        self.step_imported_layers: list[int] = []
        self.step_profile: dict[int, dict] = {}

    def committed_new_tokens(self) -> int:
        """New tokens committed on the target since import (excludes prefix)."""
        if self.req is None:
            return 0
        try:
            return max(0, len(self.req.output_ids) - self.base_output_len)
        except Exception:
            return 0

    def notify_source(self, kind: str) -> None:
        """Tell the source registry that the target imported/committed/done.

        These facts are what allow a source-side frozen watchdog to recover
        without the controller: no ``imported`` means resume is safe, a
        ``commit`` forbids resume, and ``done`` means the lease can be freed.
        """
        with self._state_notify_lock:
            if self._state_notified.get(kind):
                return
            self._state_notified[kind] = True
        svc = self.remote_service or {}
        host = svc.get("host")
        port = svc.get("port")
        token = svc.get("token")
        if host is None or port is None:
            return

        def work():
            ok = False
            try:
                sock = socket.create_connection((host, int(port)), timeout=3.0)
                sock.settimeout(3.0)
                _send_json(
                    sock,
                    {
                        "cmd": "hello",
                        "proto": _HUB_PROTO,
                        "rpc_id": f"state-{kind}-{self.handoff_id}",
                    },
                )
                hello = _recv_json(sock)
                if not hello.get("ok"):
                    raise RuntimeError(f"hub hello rejected: {hello}")
                _send_json(
                    sock,
                    {
                        "cmd": kind,
                        "handoff_id": self.handoff_id,
                        "epoch": self.owner_epoch,
                        "token": token,
                    },
                )
                sock.close()
                ok = True
            except Exception as exc:
                print(
                    f"[handoff-target] notify {kind} failed "
                    f"id={self.handoff_id}: {exc!r}",
                    flush=True,
                )
            finally:
                if not ok:
                    with self._state_notify_lock:
                        self._state_notified[kind] = False

        threading.Thread(
            target=work, daemon=True, name=f"handoff-notify-{kind}"
        ).start()

    def abort(self, reason="aborted"):
        """Stop the receiver and wake any layer waiters without touching KV.

        The caller must free ``req.prefix_indices`` only after this returns;
        the receiver thread is joined here so it cannot write into slots that
        are about to be released.
        """
        self.error = self.error or reason
        self.aborted = True
        if self.req is not None:
            try:
                self.req._handoff_aborted = True
            except Exception:
                pass
        self.abort_event.set()
        for ev in self.events:
            try:
                ev.set()
            except Exception:
                pass
        _stop_receiver_io(self)

    def _prof(self, key, value=0.0):
        if not P22_PROFILE:
            return
        step = int(self._cur_step if self._cur_step is not None else -1)
        rec = self.step_profile.get(step)
        if rec is None:
            rec = {
                "layers": 0,
                "local_rows": 0,
                "remote_rows": 0,
                "attn_ms": 0.0,
                "local_attn_ms": 0.0,
                "remote_prep_ms": 0.0,
                "q_ms": 0.0,
                "rpc_ms": 0.0,
                "post_ms": 0.0,
                "kv_write_ms": 0.0,
                "full_local_attn_ms": 0.0,
                "full_local_layers": 0,
            }
            self.step_profile[step] = rec
        rec[key] = rec.get(key, 0) + value

    def _record_rpc(self, rec):
        if not RPC_PROFILE:
            return
        self.rpc_n += 1
        for key, val in rec.items():
            if key in (
                "step",
                "layer",
                "row",
                "tp_rank",
                "rpc_id",
                "batch_size",
                "batch_row_index",
                "batch_rpc_id",
            ):
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                self.rpc_stage[key] = self.rpc_stage.get(key, 0.0) + float(val)
        for store, key in (
            (self.rpc_layer, int(rec.get("layer", -1))),
            (self.rpc_step, int(rec.get("step", -1))),
        ):
            agg = store.get(key)
            if agg is None:
                agg = {
                    "n": 0,
                    "wall_sum": 0.0,
                    "wall_max": 0.0,
                    "queue_sum": 0.0,
                    "exec_sum": 0.0,
                    "source_sum": 0.0,
                    "post_sum": 0.0,
                    "local_sum": 0.0,
                    "q_d2h_sum": 0.0,
                    "recv_delay_sum": 0.0,
                }
                store[key] = agg
            agg["n"] += 1
            wall = float(rec.get("wall_ms") or 0.0)
            agg["wall_sum"] += wall
            agg["wall_max"] = max(agg["wall_max"], wall)
            agg["queue_sum"] += float(rec.get("queue_ms") or 0.0)
            agg["exec_sum"] += float(rec.get("exec_ms") or 0.0)
            agg["source_sum"] += float(rec.get("source_ms") or 0.0)
            agg["post_sum"] += float(rec.get("post_ms") or 0.0)
            agg["local_sum"] += float(rec.get("local_tail_ms") or 0.0)
            agg["q_d2h_sum"] += float(rec.get("q_d2h_ms") or 0.0)
            agg["recv_delay_sum"] += float(rec.get("recv_delay_ms") or 0.0)
        self.rpc_top.append(dict(rec))
        if len(self.rpc_top) > 64:
            self.rpc_top.sort(key=lambda r: r.get("wall_ms") or 0.0, reverse=True)
            del self.rpc_top[32:]
        self._rpc_step_records.append(rec)

    def _flush_rpc_records(self):
        if not RPC_PROFILE or self.trace_fh is None or not self._rpc_step_records:
            return
        with self.trace_lock:
            for rec in self._rpc_step_records:
                self.trace_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.trace_fh.flush()
        self._rpc_step_records.clear()

    def metrics(self) -> dict:
        self._flush_rpc_records()
        out = {
            "handoff_id": self.handoff_id,
            "trace_tag": self.trace_tag,
            "mode": self.mode,
            "base_len": self.base_len,
            "base_output_len": self.base_output_len,
            "new_tokens": self.committed_new_tokens(),
            "migration_epoch": self.migration_epoch,
            "aborted": self.aborted,
            "created_at": self.created_at,
            "admitted_at": self.admitted_at,
            "finished_at": self.finished_at,
            "step_starts": list(self.step_starts),
            "step_index": self._cur_step,
            "remote_calls": self.remote_calls,
            "local_calls": self.local_calls,
            "wait_calls": self.wait_calls,
            "remote_fallback": self.remote_fallback,
            "last_remote_ms": self.last_remote_ms,
            "error": self.error,
            "tp_rank": self.tp_rank,
            "step_remote_counts": (
                list(self.step_remote_counts)
                + ([int(self._step_remote)] if self._step_fb is not None else [])
            ),
            "step_remote_layers": list(self.step_remote_layers),
            "step_imported_layers": list(self.step_imported_layers),
            "layer_ready_at": list(self.layer_ready_at),
            "layer_msg_at": list(self.layer_msg_at),
            "layer_wait_ms": list(self.layer_wait_ms),
            "layer_copy_ms": list(self.layer_copy_ms),
            "layer_sync_ms": list(self.layer_sync_ms),
            "layer_bytes": list(self.layer_bytes),
            "import_done_at": self.import_done_at,
            "last_remote_at": self.last_remote_at,
            "first_remote_at": self.first_remote_at,
            "first_step_at": (self.step_starts[0] if self.step_starts else None),
            "first_token_at": self.first_token_at,
            "import_start_at": (
                getattr(self.req, "_handoff_import_start", None)
                if self.req is not None
                else None
            ),
            "admit_at": (
                getattr(self.req, "_handoff_admit_at", None)
                if self.req is not None
                else None
            ),
            "import_timeline": self.import_timeline,
            "local_ready": bool(
                self.import_done_at is not None
                and self.error is None
                and all(ev.is_set() for ev in self.events)
            ),
            "layers_total": len(self.events),
            "layers_ready": sum(1 for ev in self.events if ev.is_set()),
            "host_staging_bytes": self.host_staging_bytes,
            "output_ids": (list(self.req.output_ids) if self.req is not None else []),
        }
        if P22_PROFILE:
            out["step_profile"] = {
                str(k): {kk: round(vv, 3) for kk, vv in v.items()}
                for k, v in sorted(self.step_profile.items())
            }
        if RPC_PROFILE:
            out["rpc_n"] = self.rpc_n
            out["rpc_stage_ms"] = {
                k: round(v, 3) for k, v in sorted(self.rpc_stage.items())
            }
            out["rpc_layer"] = [
                {"layer": k, **{kk: round(vv, 3) for kk, vv in v.items()}}
                for k, v in sorted(self.rpc_layer.items())
            ]
            out["rpc_step"] = [
                {"step": k, **{kk: round(vv, 3) for kk, vv in v.items()}}
                for k, v in sorted(self.rpc_step.items())
            ]
            out["rpc_top"] = [
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in rec.items()
                }
                for rec in sorted(
                    self.rpc_top,
                    key=lambda r: r.get("wall_ms") or 0.0,
                    reverse=True,
                )[:32]
            ]
        return out

    def _trace(self, rec):
        if not PROFILE or self.trace_fh is None:
            return
        with self.trace_lock:
            self.trace_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.trace_fh.flush()

    def begin_step(self, forward_batch):
        _check_target_state(self)
        if forward_batch is self._step_fb:
            return
        now = time.monotonic()
        if self.admitted_at is None:
            self.admitted_at = now
            _imp_start = (
                getattr(self.req, "_handoff_import_start", None)
                if self.req is not None
                else None
            )
            print(
                f"[handoff-target] first_step id={self.handoff_id} "
                f"mode={self.mode} "
                + (
                    f"import_to_step_ms={(now - _imp_start) * 1000:.1f}"
                    if _imp_start is not None
                    else "import_to_step_ms=None"
                ),
                flush=True,
            )
        self.step_starts.append(now)
        if self.events:
            self.step_remote_layers.append(
                sum(1 for e in self.events if not e.is_set())
            )
            self.step_imported_layers.append(sum(1 for e in self.events if e.is_set()))
        else:
            self.step_remote_layers.append(0)
            self.step_imported_layers.append(0)
        if self._step_fb is not None:
            self.step_remote_counts.append(int(self._step_remote))
            self._flush_rpc_records()
        if self.mode == "remote" and self._pending_fallback:
            self._pending_fallback = False
            # Once the local import fails, waiting for it cannot recover.
            # Keep using the source even if an earlier slow RPC asked to wait.
            if self.error is None:
                self.mode = "layer"
                self.remote_fallback += 1
        if PROFILE and self._step_fb is not None:
            self.step_log.append(
                {
                    "seq_len": self._step_seq,
                    "remote": self._step_remote,
                    "local": self._step_local,
                    "t": self._step_t,
                }
            )
        self._cur_step = self._step_index
        self._step_index += 1
        self._step_fb = forward_batch
        self._step_mask = [e.is_set() for e in self.events]
        self._step_remote = 0
        self._step_local = 0
        try:
            self._step_seq = int(forward_batch.seq_lens[0])
        except Exception:
            self._step_seq = None
        self._step_t = time.perf_counter()

    def is_remote_layer(self, layer_id):
        return self.mode == "remote" and bool(
            self._step_mask is not None and not self._step_mask[layer_id]
        )

    def _watch_first_token(self):
        """Record when the target actually appends its first new token.

        Diagnostic only: separates target compute from control/status delivery
        latency when a first-token gap looks large.
        """
        base = int(self.base_output_len)
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline and not self._token_watch_stop.is_set():
            try:
                if len(self.req.output_ids) > base:
                    self.first_token_at = time.monotonic()
                    return
            except Exception:
                return
            time.sleep(0.001)

    def close(self):
        self._token_watch_stop.set()
        if self.client is not None:
            try:
                self.client.detach(self.handoff_id)
            except Exception:
                pass
            # The client is shared per endpoint; never close it here.
        try:
            self._flush_rpc_records()
            self.trace_fh.close()
        except Exception:
            pass
        self.trace_fh = None

    def prepare_remote_row(
        self, module, q_row, k_buf, v_buf, seq_len, seq_delta, row_index=None
    ):
        """Local-tail attention + Q download for one remote row.

        Returns a dict consumed by :meth:`finish_remote_row`.  The remote RPC is
        sent by the caller, so several rows of the same layer can share one
        request (P1.1).
        """
        t_row0 = time.perf_counter()
        t_row0_mono = time.monotonic()
        req_to_token = _get_attn_backend_req_to_token()
        seq_len = int(seq_len)
        tail_idx = req_to_token[self.req_pool_idx, self.base_len : seq_len]
        k_tail = k_buf[tail_idx]
        v_tail = v_buf[tail_idx]
        q_bytes = None
        q_sync_ms = 0.0
        if Q_SYNC:
            # Sync Q D2H before launching local tail: small Q (tens of KB) and
            # it lets local tail overlap the remote RPC instead of the copy.
            t_q = time.perf_counter()
            q_bytes = (
                q_row.contiguous().view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
            )
            q_sync_ms = (time.perf_counter() - t_q) * 1000.0
        local_o, local_lse = _single_decode(
            q_row.contiguous(), k_tail, v_tail, module.scaling
        )
        t_tail = time.perf_counter()
        # Keep Q on GPU; the batch orchestrator stages all rows into pinned
        # memory on a copy stream, overlapping local-tail completion.
        q_row_c = q_row
        t_qbuild = time.perf_counter()
        q_nbytes = int(q_row_c.numel()) * int(q_row_c.element_size())
        req = {
            "rid": f"p2d-{self.req_pool_idx}",
            "handoff_id": self.handoff_id,
            "step": int(self._cur_step if self._cur_step is not None else -1),
            "seq": int(seq_delta),
            "layer": int(module.layer_id),
            "q_shape": list(q_row.shape),
            "q_dtype": "bfloat16" if q_row.dtype == torch.bfloat16 else "float16",
            "sm_scale": float(module.scaling),
            "q_bytes": q_nbytes,
            "send_ts": time.monotonic(),
        }
        if self.owner_epoch is not None:
            req["epoch"] = int(self.owner_epoch)
        if P22_PROFILE:
            self._prof("remote_prep_ms", (t_tail - t_row0) * 1000.0)
        return {
            "state": self,
            "row_index": row_index,
            "layer": int(module.layer_id),
            "local_o": local_o,
            "local_lse": local_lse,
            "q_row": q_row_c,
            "q_nbytes": q_nbytes,
            "q_bytes": q_bytes,
            "q_sync_ms": q_sync_ms,
            "row_header": req,
            "t_row0": t_row0,
            "t_row0_mono": t_row0_mono,
            "t_tail": t_tail,
            "t_qbuild": t_qbuild,
            "q_d2h_ms": 0.0,
            "t_rpc": time.perf_counter(),
        }

    def finish_remote_row(
        self,
        prep,
        resp,
        o_cpu,
        lse_cpu,
        t_rpc,
        t_call,
        batch_size=None,
        batch_rpc_id=None,
        batch_queue_ms=None,
        batch_exec_ms=None,
        merged=None,
        post_ms=None,
        q_d2h_ms=None,
    ):
        """Merge one row's source result and account for it."""
        self.last_rpc_wall_ms = (t_call - t_rpc) * 1000.0
        self.remote_calls += 1
        self._step_remote += 1
        queue_ms = resp.get("queue_ms", batch_queue_ms)
        exec_ms = resp.get("exec_ms", batch_exec_ms)
        self.last_remote_ms = float(resp.get("source_ms", self.last_rpc_wall_ms))
        self.last_pre_ms = resp.get("pre_ms")
        self.last_kernel_ms = resp.get("kernel_ms")
        self.last_queue_ms = queue_ms
        self.last_recv_delay_ms = resp.get("recv_delay_ms")
        if PROFILE:
            self.remote_stats.append(
                {
                    "source_ms": self.last_remote_ms,
                    "pre_ms": self.last_pre_ms,
                    "kernel_ms": self.last_kernel_ms,
                    "queue_ms": queue_ms,
                    "rpc_wall_ms": self.last_rpc_wall_ms,
                    "recv_delay_ms": self.last_recv_delay_ms,
                    "batch_size": batch_size,
                }
            )
        self.remote_latencies.append(self.last_remote_ms)
        if (
            len(self.remote_latencies) >= 6
            and max(self.remote_latencies[-3:]) > self.remote_ms_threshold
        ):
            # Defer the mode switch to the next step boundary so all layers
            # of the current step see the same row mask.
            self._pending_fallback = True
        if merged is None:
            o = o_cpu.to(prep["local_o"].device)
            lse = lse_cpu.to(prep["local_o"].device)
            merged, _ = _merge_states(prep["local_o"], prep["local_lse"], o, lse)
            t_post = time.perf_counter()
            post_ms = (t_post - t_call) * 1000.0
        else:
            # Batch path: O/LSE H2D and merge were already issued once for all
            # rows; this is the accounting timestamp only.
            t_post = t_call + (post_ms or 0.0) / 1000.0
        if q_d2h_ms is None:
            q_d2h_ms = prep.get("q_d2h_ms", 0.0)
        self._last_attn_exit = t_post
        self.last_remote_at = time.monotonic()
        if self.first_remote_at is None:
            self.first_remote_at = self.last_remote_at
        if P22_PROFILE:
            self._prof("remote_rows", 1)
            self._prof("q_ms", q_d2h_ms or 0.0)
            self._prof(
                "rpc_ms",
                self.last_rpc_wall_ms / max(1, int(batch_size or 1)),
            )
            self._prof("post_ms", post_ms or 0.0)
        if RPC_PROFILE:
            self._record_rpc(
                {
                    "handoff_id": self.handoff_id,
                    "trace_tag": self.trace_tag,
                    "step": int(self._cur_step if self._cur_step is not None else -1),
                    "layer": prep["layer"],
                    "row": prep["row_index"],
                    "tp_rank": self.tp_rank,
                    "rpc_id": batch_rpc_id or resp.get("rpc_id"),
                    "batch_size": batch_size,
                    "batch_rpc_id": batch_rpc_id,
                    "wall_ms": (t_call - t_rpc) * 1000.0,
                    "t_mono": time.monotonic(),
                    "row0_mono": prep.get("t_row0_mono"),
                    "step_start_at": (
                        self.step_starts[-1] if self.step_starts else None
                    ),
                    "local_tail_ms": (prep["t_tail"] - prep["t_row0"]) * 1000.0,
                    "q_d2h_ms": q_d2h_ms,
                    "post_ms": post_ms,
                    "recv_delay_ms": resp.get("recv_delay_ms"),
                    "q_recv_ms": resp.get("q_recv_ms"),
                    "queue_ms": queue_ms,
                    "exec_ms": exec_ms,
                    "source_ms": resp.get("source_ms"),
                    "api_ms": resp.get("api_ms"),
                    "q_h2d_ms": resp.get("q_h2d_ms"),
                    "d2h_o_ms": resp.get("d2h_o_ms"),
                    "d2h_lse_ms": resp.get("d2h_lse_ms"),
                    "sync_ms": resp.get("sync_ms"),
                    "source_setup_ms": resp.get("source_setup_ms"),
                }
            )
        return merged.squeeze(0)

    def remote_row(
        self, module, q_row, k_buf, v_buf, seq_len, seq_delta, row_index=None
    ):
        """Local tail attention + source-side prompt attention for one row."""
        prep = self.prepare_remote_row(
            module,
            q_row,
            k_buf,
            v_buf,
            seq_len,
            seq_delta,
            row_index=row_index,
        )
        if Q_SYNC:
            q_bytes = prep["q_bytes"]
            q_d2h_ms = prep["q_sync_ms"]
        else:
            q_bytes, q_d2h_ms = _batch_q_bytes([prep["q_row"]])
        prep["q_d2h_ms"] = q_d2h_ms
        t_rpc = prep["t_rpc"]
        try:
            if self.client is None:
                raise RuntimeError("remote_progressive has no target client")
            self.client.attach(
                self.handoff_id, str(self.remote_service["token"]), self.owner_epoch
            )
        except Exception as exc:
            self.error = repr(exc)
            raise
        try:
            header, o_cpu, lse_cpu = self.client.call(prep["row_header"], q_bytes)
        except Exception as exc:
            self.error = repr(exc)
            raise
        t_call = time.perf_counter()
        return self.finish_remote_row(
            prep,
            header,
            o_cpu,
            lse_cpu,
            t_rpc,
            t_call,
            q_d2h_ms=q_d2h_ms,
        )


def _layer_delay_s() -> float:
    try:
        return (
            max(0.0, float(os.environ.get("HANDOFF_LAYER_DELAY_MS", "0") or 0)) / 1000.0
        )
    except (TypeError, ValueError):
        return 0.0


def _layer_dropped(scheduler, layer: int) -> bool:
    """Failure-injection hook: pretend one layer never became local.

    ``HANDOFF_LAYER_DROP`` accepts ``layer`` (all ranks) or ``rank:layer``.
    The receiver still drains the wire and acknowledges slot reuse, but leaves
    that layer's READY event unset so the controller must not treat the rank
    as locally complete.
    """
    spec = os.environ.get("HANDOFF_LAYER_DROP", "").strip()
    if not spec:
        return False
    try:
        if ":" in spec:
            rank_s, layer_s = spec.split(":", 1)
            want_rank = int(rank_s)
        else:
            want_rank = None
            layer_s = spec
        want_layer = int(layer_s)
    except (TypeError, ValueError):
        return False
    if want_layer != int(layer):
        return False
    return want_rank is None or want_rank == _tp_rank(scheduler)


def _finish_layer_import(state) -> None:
    """Mark local-ready only when every layer is materialized without error."""
    if state.error is None and all(ev.is_set() for ev in state.events):
        state.import_done_at = time.monotonic()
    else:
        if state.error is None:
            missing = [i for i, ev in enumerate(state.events) if not ev.is_set()]
            state.error = f"layer import incomplete: {missing[:8]}"
        # Keep import_done_at unset: the source KV is still required by at
        # least one layer, so the controller must not release it.
        if not _can_serve_remote(state):
            raise RuntimeError(state.error)


def _layer_receiver(
    scheduler, state, conn, meta, kv_cpu, new_indices, reg_owner, device
):
    copy_stream = None
    transfer_done = False
    try:
        # Each TP rank owns its own KV shard: the receiver must use the device
        # of the scheduler that started it, not local device 0.
        torch.cuda.set_device(device)
        copy_stream = _fast_copy_streams(device, 1, role="target")[0]
        ready = getattr(state, "indices_ready", None)
        if ready is not None:
            copy_stream.wait_event(ready)
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        delay = _layer_delay_s()
        for layer in range(len(kv_cpu)):
            t_msg = time.monotonic()
            msg = _recv_json(conn)
            if msg.get("ok") is False:
                raise RuntimeError(msg.get("error", "source transfer failed"))
            if msg.get("done"):
                transfer_done = True
                break
            l = int(msg["layer"])
            if l < 0 or l >= len(state.events) or state.events[l].is_set():
                raise ValueError(f"invalid or duplicate layer {l}")
            if _layer_dropped(scheduler, l):
                state.error = f"layer {l} dropped by test hook"
                continue
            if delay > 0:
                time.sleep(delay)
            t_copy0 = time.perf_counter()
            offset = 0
            k_buf = pool.get_key_buffer(l)
            v_buf = pool.get_value_buffer(l)
            nbytes = 0
            with torch.cuda.stream(copy_stream):
                for k_cpu, v_cpu in kv_cpu[l]:
                    n = int(k_cpu.shape[0])
                    idx = new_indices[offset : offset + n]
                    k_buf[idx] = k_cpu.to(k_buf.device, non_blocking=True)
                    v_buf[idx] = v_cpu.to(v_buf.device, non_blocking=True)
                    nbytes += int(k_cpu.numel()) * int(k_cpu.element_size()) + int(
                        v_cpu.numel()
                    ) * int(v_cpu.element_size())
                    offset += n
            t_sync0 = time.perf_counter()
            copy_stream.synchronize()
            t_sync = time.perf_counter()
            state.layer_wait_ms[l] = (t_copy0 - t_msg) * 1000.0
            state.layer_copy_ms[l] = (t_sync0 - t_copy0) * 1000.0
            state.layer_sync_ms[l] = (t_sync - t_sync0) * 1000.0
            state.layer_bytes[l] = nbytes
            state.layer_ready_at[l] = time.monotonic()
            state.events[l].set()
        _finish_layer_import(state)
        # Consume the source's final {"done": true} so its stats/cleanup can run.
        if not transfer_done:
            try:
                final = _recv_json(conn)
                transfer_done = (
                    final.get("done") is True and final.get("ok") is not False
                )
            except Exception:
                pass
    except Exception as exc:
        _record_local_import_error(state, repr(exc))
    finally:
        if copy_stream is not None:
            try:
                copy_stream.synchronize()
            except Exception as exc:
                state.receiver_unsafe = True
                state.error = f"target copy fence failed: {exc!r}"
        _release_transfer_buffers(
            reg_owner,
            kv_cpu,
            state.error or "target transfer incomplete",
            remote_complete=transfer_done
            or (
                state.import_done_at is not None
                and state.error is None
                and not state.abort_event.is_set()
            ),
            gpu_safe=not getattr(state, "receiver_unsafe", False),
        )
        try:
            conn.close()
        except OSError:
            pass


def _slot_layer_receiver(
    scheduler, state, conn, meta, slots, new_indices, reg_owner, device
):
    """P5.2-B target receiver: copy each arriving layer from a bounded slot.

    Layer ``L`` becomes READY only after its H2D copy finished (synchronous
    ``to`` + ``cuda.synchronize``) and the layer event is set.  The source may
    overwrite this slot only after the matching ``{"free": slot}`` ack, so a
    slot is never reused while a copy or an attention read could still use it.
    """
    copy_stream = None
    transfer_done = False
    try:
        torch.cuda.set_device(device)
        copy_stream = _fast_copy_streams(device, 1, role="target")[0]
        ready = getattr(state, "indices_ready", None)
        if ready is not None:
            copy_stream.wait_event(ready)
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        delay = _layer_delay_s()
        nslots = len(slots)
        # Layer H2D must not synchronize the whole device: the target is
        # decoding while layers stream in, and a device-wide sync after every
        # layer stalls the decode stream for the entire import window (the
        # 32K first-token outlier).  Copy on a dedicated cached stream and
        # wait only for that stream.
        for _ in range(len(meta["chunks"])):
            t_msg = time.monotonic()
            msg = _recv_json(conn)
            if msg.get("ok") is False:
                raise RuntimeError(msg.get("error", "source transfer failed"))
            if msg.get("done"):
                transfer_done = True
                break
            l = int(msg["layer"])
            if l < 0 or l >= len(state.events) or state.events[l].is_set():
                raise ValueError(f"invalid or duplicate layer {l}")
            slot = int(msg.get("slot", l % nslots))
            if slot < 0 or slot >= nslots:
                raise ValueError(f"layer {l} slot {slot} out of range")
            if _layer_dropped(scheduler, l):
                state.error = f"layer {l} dropped by test hook"
                # Return the slot so the source is not stalled; the layer
                # event stays unset so LOCAL is not falsely reported.
                try:
                    _send_json(conn, {"free": slot})
                except OSError:
                    pass
                continue
            if delay > 0:
                time.sleep(delay)
            t_copy0 = time.perf_counter()
            k_buf = pool.get_key_buffer(l)
            v_buf = pool.get_value_buffer(l)
            nbytes = 0
            offset = 0
            with torch.cuda.stream(copy_stream):
                for k_cpu, v_cpu in slots[slot]:
                    n = int(k_cpu.shape[0])
                    idx = new_indices[offset : offset + n]
                    k_buf[idx] = k_cpu.to(k_buf.device, non_blocking=True)
                    v_buf[idx] = v_cpu.to(v_buf.device, non_blocking=True)
                    nbytes += int(k_cpu.numel()) * int(k_cpu.element_size()) + int(
                        v_cpu.numel()
                    ) * int(v_cpu.element_size())
                    offset += n
            t_sync0 = time.perf_counter()
            copy_stream.synchronize()
            t_sync = time.perf_counter()
            state.layer_msg_at[l] = t_msg
            state.layer_wait_ms[l] = (t_copy0 - t_msg) * 1000.0
            state.layer_copy_ms[l] = (t_sync0 - t_copy0) * 1000.0
            state.layer_sync_ms[l] = (t_sync - t_sync0) * 1000.0
            state.layer_bytes[l] = nbytes
            state.layer_ready_at[l] = time.monotonic()
            state.events[l].set()
            # The source may already have sent ``done`` and closed its side
            # after the last layer; the ack is advisory and must not turn a
            # fully-materialized request into an import failure.
            try:
                _send_json(conn, {"free": slot})
            except OSError:
                pass
        _finish_layer_import(state)
        # Consume the source's final {"done": true} so its stats/cleanup can run.
        if not transfer_done:
            try:
                final = _recv_json(conn)
                transfer_done = (
                    final.get("done") is True and final.get("ok") is not False
                )
            except Exception:
                pass
    except Exception as exc:
        _record_local_import_error(state, repr(exc))
    finally:
        if copy_stream is not None:
            try:
                copy_stream.synchronize()
            except Exception as exc:
                state.receiver_unsafe = True
                state.error = f"target copy fence failed: {exc!r}"
        _release_transfer_buffers(
            reg_owner,
            slots,
            state.error or "target transfer incomplete",
            remote_complete=transfer_done
            or (
                state.import_done_at is not None
                and state.error is None
                and not state.abort_event.is_set()
            ),
            gpu_safe=not getattr(state, "receiver_unsafe", False),
        )
        try:
            conn.close()
        except OSError:
            pass


def _install_attention_wrappers(scheduler, state):
    import types

    from sglang.srt.layers.radix_attention import RadixAttention

    controller = getattr(scheduler, "_pre_layerwise", None)
    if controller is not None:
        target = getattr(getattr(state, "req", None), "_pre_layerwise_target", None)
        if (
            target is None
            or target.owner is not controller
            or target.state is not state
        ):
            # A legacy/manual import can progress independently of the
            # progressive TP lifetime votes. Permanently fall back to the
            # conservative graph vote for this engine (all command ranks
            # install their state before the next model forward).
            controller.graph_fast_path = False
    if state not in _TARGET_STATES:
        _TARGET_STATES.append(state)
    if PROFILE:
        _GIL_GAPS.clear()
        _start_gil_probe()
    _FB_CACHE.update(
        {
            "fb": None,
            "states": [],
            "rp": [],
            "seq": [],
            "by_rpi": None,
            "counted_fb": None,
            "mixed_counted": False,
        }
    )
    model = scheduler.tp_worker.model_runner.model
    for module in model.modules():
        if not isinstance(module, RadixAttention) or getattr(
            module, "_handoff_wrapped", False
        ):
            continue
        module._handoff_orig_forward = module.forward

        def wrapper(self, q, k, v, forward_batch, save_kv_cache=True, **kw):
            return _handoff_attention_forward(
                self, q, k, v, forward_batch, save_kv_cache, **kw
            )

        module.forward = types.MethodType(wrapper, module)
        module._handoff_wrapped = True


def _rollback_full_import(scheduler, tracked, rid, req, indices):
    if req is not None:
        stop_target_receivers(req)
        _remove_from_batches(scheduler, req)
        scheduler._handoff_held_reqs = [
            r for r in getattr(scheduler, "_handoff_held_reqs", []) if r is not req
        ]
        if tracked.get(rid) is req:
            tracked.pop(rid)
        req.handoff_prefix_indices = None
        req._handoff_prefix_indices = None
        req.kv.mark_kv_released()
        req.kv.kv_committed_len = 0
    scheduler.token_to_kv_pool_allocator.free(indices)


def _full_import_one(
    scheduler,
    tracked,
    transfer,
    rid,
    remaining=None,
    epoch=None,
    lp_kwargs=None,
):
    """Full-KV import of one request (copy_wait path)."""
    t_recv = time.perf_counter()
    if rid in tracked:
        prev = tracked[rid]
        prev_epoch = getattr(prev, "_handoff_epoch", None)
        if epoch is not None and prev_epoch == str(epoch):
            return {
                "rid": rid,
                "duplicate": True,
                "imported_kv_tokens": int(prev.kv.kv_committed_len),
                "output_len": len(prev.output_ids),
                "remaining": remaining,
            }
        raise RuntimeError(
            f"duplicate rid {rid!r}: already imported with epoch {prev_epoch!r}"
        )
    if transfer.get("mode") == "mooncake":
        meta, kv_cpu = _recv_transfer_mooncake(transfer)
    else:
        meta, kv_cpu = _recv_transfer(transfer)
    recv_ms = (time.perf_counter() - t_recv) * 1000.0
    origin = meta["origin_input_ids"]
    output = meta["output_ids"]
    kv_len = int(meta["kv_committed_len"])
    state_sampling = meta["sampling"]
    epoch = str(epoch or meta.get("migration_epoch") or "")

    allocator = scheduler.token_to_kv_pool_allocator
    t_alloc = time.perf_counter()
    new_indices = _take_reserved(scheduler, rid, kv_len)
    if new_indices is None:
        new_indices = allocator.alloc(kv_len)
    if new_indices is None:
        raise RuntimeError(f"target KV pool cannot allocate {kv_len} tokens")
    req = None
    try:
        alloc_ms = (time.perf_counter() - t_alloc) * 1000.0
        t_load = time.perf_counter()
        allocator.load_cpu_copy(kv_cpu, new_indices)
        load_ms = (time.perf_counter() - t_load) * 1000.0
        if remaining is None:
            remaining = max(1, int(state_sampling["max_new_tokens"]) - len(output))
        sp = _sampling_from(
            state_sampling, len(output) + int(remaining), scheduler.tokenizer
        )
        req = _new_req(scheduler, rid, origin, output, sp, **(lp_kwargs or {}))
        req.kv.kv_committed_len = kv_len
        req._handoff_prefix_indices = new_indices.to(dtype=torch.int64)
        req.handoff_prefix_indices = req._handoff_prefix_indices
        req._handoff_epoch = epoch
        req._handoff_base_output_len = len(output)
        if getattr(scheduler, "_handoff_hold_admission", False):
            scheduler._handoff_held_reqs = getattr(scheduler, "_handoff_held_reqs", [])
            scheduler._handoff_held_reqs.append(req)
        else:
            scheduler.waiting_queue.append(req)
        tracked[rid] = req
        state = _TargetHandoffState(
            None,
            kv_len,
            [],
            "local",
            None,
            handoff_id=meta.get("rid") or rid,
            trace_tag=meta.get("trace_tag"),
            req=req,
            tp_rank=_tp_rank(scheduler),
            base_output_len=len(output),
            migration_epoch=epoch,
        )
        state.import_done_at = time.monotonic()
        _install_attention_wrappers(scheduler, state)
        return {
            "rid": rid,
            "imported_kv_tokens": kv_len,
            "output_len": len(output),
            "remaining": int(remaining),
            "first_new_token_pos": kv_len,
            "recv_ms": recv_ms,
            "alloc_ms": alloc_ms,
            "load_ms": load_ms,
            "migration_epoch": epoch,
        }

    except Exception:
        _rollback_full_import(scheduler, tracked, rid, req, new_indices)
        raise


def _flush_decode_resume(scheduler, reqs):
    """Start resumed requests as a normal decode step.

    The fast-resume path used to re-extend the pending output token, which
    computes its KV with the extend/prefill kernel.  Natural decode computes
    the same token's KV with the decode kernel, so the two paths drift even
    under deterministic inference.  Build a prebuilt decode batch exactly like
    the PD decode side does after a KV transfer: prefix KV is already bound,
    the last output token is the pending decode input, and its KV is written by
    the first decode forward.
    """
    if not reqs:
        return
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    batch = ScheduleBatch.init_new(
        reqs,
        scheduler.req_to_token_pool,
        scheduler.token_to_kv_pool_allocator,
        scheduler.tree_cache,
        scheduler.model_config,
        scheduler.enable_overlap,
        scheduler.spec_algorithm,
    )
    batch.prepare_for_prebuilt()
    batch.process_prebuilt(scheduler.future_map)
    rb = getattr(scheduler, "running_batch", None)
    if rb is None or rb.is_empty():
        scheduler.running_batch = batch
    else:
        rb.merge_batch(batch)


def _select_transfer(data, rank, tp_size=1):
    """Pick this TP rank's transfer metadata from a fan-out import payload."""
    transfers = data.get("transfers")
    if transfers is not None:
        if len(transfers) != tp_size:
            raise ValueError(
                f"transfers has {len(transfers)} entries for tp_size={tp_size}"
            )
        if transfers[rank] is None:
            raise ValueError(f"transfers[{rank}] is None for tp_size={tp_size}")
        return transfers[rank]
    transfer = data.get("transfer")
    if transfer is not None and tp_size > 1:
        raise ValueError("bare transfer is not valid for tp_size>1")
    return transfer


def _handle(scheduler, data):
    from sglang.srt.runtime_context import get_schedule

    cmd = data.get("cmd")
    tracked = _tracked(scheduler)
    if cmd in ("import", "import_wave"):
        # Failure-injection only: leave the target import command in flight
        # long enough for the controller to die and the source watchdog to
        # exercise the epoch fence.
        try:
            delay_ms = float(os.environ.get("HANDOFF_TEST_IMPORT_DELAY_MS", "0") or 0)
        except ValueError:
            delay_ms = 0.0
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    if cmd == "reset":
        hub = getattr(scheduler, "_handoff_hub", None)
        for rid, req in list(tracked.items()):
            handle = getattr(req, "_handoff_transfer_cancel", None)
            if handle is not None:
                req._handoff_transfer_cancel = None
                _, ev = handle
                ev.set()
                try:
                    handle[0].close()
                except OSError:
                    pass
            if hub is not None:
                try:
                    # True: freed here and queued on_release, so the drain
                    # frees it exactly once.  False: deferred while an RPC is
                    # in flight; the drain still frees it exactly once.  Both
                    # leave the entry tracked until the drain pops it.
                    hub.registry.release(rid, expected_req=req)
                    continue
                except KeyError:
                    # Not registered here (target-side req or a mooncake
                    # source that never created a hub route): free inline.
                    pass
            _free_source_request(scheduler, req)
            tracked.pop(rid, None)
        _reset_target_states()
        for rid in list(_reservations(scheduler)):
            _reservation_free(scheduler, rid)
        _ordinary_leases(scheduler).clear()
        scheduler._handoff_held_reqs = []
        scheduler._handoff_hold_admission = False
        return {"reset": True}

    if cmd == "hold_admission":
        scheduler._handoff_hold_admission = True
        scheduler._handoff_held_reqs = []
        return {"hold": True}

    if cmd == "admit":
        held = getattr(scheduler, "_handoff_held_reqs", [])
        scheduler._handoff_held_reqs = []
        scheduler._handoff_hold_admission = False
        for req in held:
            scheduler.waiting_queue.append(req)
        return {"admitted": len(held)}

    if cmd == "capacity":
        return {
            "capacity": _capacity_stats(scheduler),
            "reservations": {
                rid: {
                    "tokens": int(r.get("tokens", 0)),
                    "slots": int(r.get("slots", 0)),
                    "slot_consumed": bool(r.get("slot_consumed")),
                    "epoch": r.get("epoch"),
                    "kv_consumed": r.get("indices") is None,
                }
                for rid, r in _reservations(scheduler).items()
            },
            "ordinary_leases": {
                rid: {
                    "tokens": int(l.get("tokens", 0)),
                    "slot_released": bool(l.get("slot_released")),
                    "epoch": l.get("epoch"),
                }
                for rid, l in _ordinary_leases(scheduler).items()
            },
            "tp_rank": _tp_rank(scheduler),
            "tp_size": _tp_size(scheduler),
        }

    if cmd == "admit_ordinary":
        rid = str(data.get("rid") or "")
        if not rid:
            raise ValueError("admit_ordinary requires rid")
        tokens = int(data.get("tokens", 0) or 0)
        if tokens <= 0:
            raise ValueError("admit_ordinary requires positive tokens")
        return _ordinary_lease_acquire(
            scheduler, rid, tokens, epoch=str(data.get("epoch") or "")
        )

    if cmd == "capacity_cancel":
        rid = str(data.get("rid") or "")
        leases = _ordinary_leases(scheduler)
        lease = leases.get(rid)
        if lease is None:
            return {"cancelled": False, "missing": True, "rid": rid}
        req = lease.get("req")
        if req is not None and not req.finished() and not data.get("force"):
            return {
                "cancelled": False,
                "in_use": True,
                "rid": rid,
                "capacity": _capacity_stats(scheduler),
            }
        _ordinary_lease_free(scheduler, rid, force=True)
        return {
            "cancelled": True,
            "rid": rid,
            "capacity": _capacity_stats(scheduler),
        }

    if cmd == "submit":
        sp = _sampling_from(
            data, int(data.get("max_new_tokens", 128)), scheduler.tokenizer
        )
        rid = str(data["rid"])
        tickets = _ordinary_leases(scheduler)
        ticket = str(data.get("capacity_ticket") or "")
        if ticket and ticket != rid:
            raise ValueError(f"capacity_ticket {ticket!r} does not match rid {rid!r}")
        if ticket:
            cap = _ordinary_lease_acquire(
                scheduler,
                rid,
                int(data.get("capacity_tokens", 0) or 0)
                or (len(data["input_ids"]) + int(data.get("max_new_tokens", 128)) + 64),
                epoch=str(data.get("capacity_epoch") or ""),
            )
        else:
            need = len(data["input_ids"]) + int(data.get("max_new_tokens", 128)) + 64
            cap = _ordinary_lease_acquire(
                scheduler,
                rid,
                need,
                epoch=str(data.get("capacity_epoch") or ""),
            )
        if not cap.get("admitted"):
            raise RuntimeError(
                f"admission capacity: {cap.get('reason')} ({cap.get('capacity')})"
            )
        req = _new_req(
            scheduler,
            rid,
            data["input_ids"],
            data.get("output_ids", []),
            sp,
            **_logprob_kwargs(data),
        )
        lease = tickets.get(rid)
        if lease is not None:
            lease["req"] = req
            req._handoff_capacity_lease = rid
        req.kv.kv_committed_len = int(data.get("kv_committed_len", 0))
        if len(req.output_ids) > 0 and req.kv.kv_committed_len == 0:
            raise ValueError("submit with output_ids must set kv_committed_len")
        if data.get("pause_after_prefill"):
            req._handoff_pause_after_prefill = True
            req._handoff_freeze_mode = data.get("transfer", "mooncake")
            req._handoff_freeze_out = data.get("out")
        scheduler.waiting_queue.append(req)
        tracked[rid] = req
        return {"rid": rid, "queued": True, "input_len": len(req.origin_input_ids)}

    if cmd == "submit_pd":
        """Create a native PD decode request and trigger its P prefill.

        The D scheduler owns the decode receiver; P is triggered by a daemon
        HTTP request using the same bootstrap room.  The request is tracked so
        the existing status/journal/SSE path can stream it exactly like a
        normal handoff request.
        """
        sp = _sampling_from(
            data, int(data.get("max_new_tokens", 128)), scheduler.tokenizer
        )
        rid = str(data["rid"])
        room = int(data.get("bootstrap_room") or secrets.randbits(63))
        host = str(
            data.get("bootstrap_host")
            or os.environ.get("HANDOFF_PD_BOOTSTRAP_HOST", "127.0.0.1")
        )
        port = int(
            data.get("bootstrap_port")
            or os.environ.get("HANDOFF_PD_BOOTSTRAP_PORT", "35409")
        )
        prefill_url = str(
            data.get("prefill_url") or os.environ.get("HANDOFF_PD_PREFILL_URL", "")
        )
        # Native decode prealloc physically allocates the prompt KV as soon
        # as the request enters the prealloc queue.  Lease only the growth
        # budget; charging prompt + growth double-counts the prompt and
        # falsely rejects a 4th 32K request at mem-fraction 0.75.
        need = int(data.get("max_new_tokens", 128)) + 64
        _t_sub0 = time.monotonic()
        cap = _ordinary_lease_acquire(
            scheduler,
            rid,
            need,
            epoch=str(data.get("capacity_epoch") or ""),
        )
        _t_cap = time.monotonic()
        if not cap.get("admitted"):
            raise RuntimeError(
                f"admission capacity: {cap.get('reason')} ({cap.get('capacity')})"
            )
        req = _new_req(
            scheduler,
            rid,
            data["input_ids"],
            [],
            sp,
            **_logprob_kwargs(data),
            bootstrap_host=host,
            bootstrap_port=port,
            bootstrap_room=room,
            disagg_mode=scheduler.disaggregation_mode,
        )
        lease = _ordinary_leases(scheduler).get(rid)
        if lease is not None:
            lease["req"] = req
            req._handoff_capacity_lease = rid
        req._handoff_pd = True
        # Decode-only scheduler: this goes through the native prealloc /
        # transfer queues and waits for P's KV before becoming runnable.
        scheduler._add_request_to_queue(req)
        _t_queue = time.monotonic()
        tracked[rid] = req
        print(
            f"[handoff-pd] submit rid={rid} room={room} "
            f"cap_ms={(_t_cap - _t_sub0) * 1000:.1f} "
            f"queue_ms={(_t_queue - _t_cap) * 1000:.1f} "
            f"max_new={int(data.get('max_new_tokens', 128))} "
            f"input_len={len(data['input_ids'])}",
            flush=True,
        )
        if not prefill_url:
            print(
                f"[handoff-pd] no prefill_url for {rid}; request will wait "
                "for an external P trigger",
                flush=True,
            )
        qstats = {}
        try:
            q = scheduler.disagg_decode_prealloc_queue
            qstats = {
                "prealloc": len(getattr(q, "queue", [])),
                "pending": len(getattr(q, "pending_reqs", [])),
                "waiting": len(scheduler.waiting_queue),
            }
        except Exception:
            pass
        return {
            "rid": rid,
            "queued": True,
            "bootstrap_room": room,
            "pd": True,
            "queues": qstats,
        }

    if cmd == "pd_queues":
        q = getattr(scheduler, "disagg_decode_prealloc_queue", None)
        t = getattr(scheduler, "disagg_decode_transfer_queue", None)
        return {
            "mode": str(getattr(scheduler, "disaggregation_mode", None)),
            "prealloc_queue": len(getattr(q, "queue", []) or []),
            "pending": len(getattr(q, "pending_reqs", []) or []),
            "retracted": len(getattr(q, "retracted_queue", []) or []),
            "transfer": len(getattr(t, "queue", []) or []),
            "waiting": len(getattr(scheduler, "waiting_queue", []) or []),
            "running": len(getattr(scheduler.running_batch, "reqs", []) or []),
        }

    if cmd == "submit_many":
        # Queue a batch of fresh requests in one scheduler iteration, so the
        # prefill scheduler sees all of them before it forms the first batch.
        rids = []
        for item in data.get("items", []):
            sp = _sampling_from(
                item, int(item.get("max_new_tokens", 128)), scheduler.tokenizer
            )
            rid = str(item["rid"])
            need = len(item["input_ids"]) + int(item.get("max_new_tokens", 128)) + 64
            cap = _ordinary_lease_acquire(
                scheduler,
                rid,
                need,
                epoch=str(item.get("capacity_epoch") or ""),
            )
            if not cap.get("admitted"):
                raise RuntimeError(f"admission capacity for {rid}: {cap.get('reason')}")
            req = _new_req(
                scheduler,
                rid,
                item["input_ids"],
                item.get("output_ids", []),
                sp,
                **_logprob_kwargs(item),
            )
            lease = _ordinary_leases(scheduler).get(rid)
            if lease is not None:
                lease["req"] = req
                req._handoff_capacity_lease = rid
            req.kv.kv_committed_len = int(item.get("kv_committed_len", 0))
            if len(req.output_ids) > 0 and req.kv.kv_committed_len == 0:
                raise ValueError("submit with output_ids must set kv_committed_len")
            if item.get("pause_after_prefill"):
                req._handoff_pause_after_prefill = True
                req._handoff_freeze_mode = item.get("transfer", "mooncake")
                req._handoff_freeze_out = item.get("out")
            scheduler.waiting_queue.append(req)
            tracked[rid] = req
            rids.append(rid)
        return {"rids": rids, "count": len(rids)}

    if cmd == "freeze":
        rid = str(data["rid"])
        req = tracked.get(rid)
        if req is None:
            raise KeyError(f"unknown rid {rid!r}")
        if req.finished():
            raise RuntimeError(f"request {rid!r} already finished; cannot freeze")
        return _freeze_request(
            scheduler,
            req,
            mode=data.get("transfer", "file"),
            out=data.get("out"),
            epoch=data.get("epoch"),
            target_rid=data.get("target_rid"),
        )

    if cmd == "freeze_all":
        # Drain primitive: freeze every listed in-flight request in the same
        # scheduler iteration, so all of them share one decode step boundary.
        # With ``cut`` set, the scheduler itself waits until every request has
        # generated at least that many tokens.  The check and the freeze happen
        # in one iteration, so a fast request cannot finish between the
        # controller's status poll and the freeze.
        mode = data.get("transfer", "mooncake")
        epoch = data.get("epoch")
        rids = [str(x) for x in data.get("rids", [])]
        target_rids = data.get("target_rids") or {}
        t_all = time.perf_counter()
        cut = int(data.get("cut", 0) or 0)
        lens = {}
        if cut > 0:
            for rid in rids:
                req = tracked.get(rid)
                if req is None:
                    return {"pending": True, "errors": {rid: "unknown rid"}}
                if req.finished():
                    return {
                        "pending": False,
                        "errors": {rid: "finished before cut"},
                        "cut": cut,
                        "lens": {rid: len(req.output_ids)},
                    }
                lens[rid] = len(req.output_ids)
            short = {rid: n for rid, n in lens.items() if n < cut}
            if short:
                return {"pending": True, "cut": cut, "lens": lens, "short": short}
        results = []
        errors = {}
        for rid in rids:
            req = tracked.get(rid)
            if req is None:
                errors[rid] = "unknown rid"
                continue
            if req.finished():
                errors[rid] = "already finished"
                continue
            try:
                results.append(
                    _freeze_request(
                        scheduler,
                        req,
                        mode=mode,
                        out=data.get("out"),
                        epoch=epoch,
                        target_rid=target_rids.get(rid),
                    )
                )
            except Exception as exc:
                errors[rid] = repr(exc)
        return {
            "pending": False,
            "results": results,
            "errors": errors,
            "count": len(results),
            "freeze_all_ms": (time.perf_counter() - t_all) * 1000.0,
        }

    if cmd == "import":
        t_recv = time.perf_counter()
        t_cmd_mono = time.monotonic()
        rank = _tp_rank(scheduler)
        # Idempotency check happens before consuming the transfer handshake.
        rid0 = str(data.get("rid", "handoff-dst"))
        epoch0 = data.get("epoch")
        if _import_cancelled(rid0, epoch0):
            raise RuntimeError(f"import epoch cancelled for rid {rid0!r}")
        prev0 = tracked.get(rid0)
        if prev0 is not None:
            prev_epoch = getattr(prev0, "_handoff_epoch", None)
            if epoch0 is not None and prev_epoch == str(epoch0):
                return {
                    "rid": rid0,
                    "duplicate": True,
                    "imported_kv_tokens": int(prev0.kv.kv_committed_len),
                    "output_len": len(prev0.output_ids),
                    "migration_epoch": prev_epoch,
                }
            raise RuntimeError(
                f"duplicate rid {rid0!r}: already imported with epoch {prev_epoch!r}"
            )
        # Multi-rank payloads carry one transfer per rank (each target rank
        # connects to its own source rank's transfer server / Hub).
        transfer = _select_transfer(data, rank, _tp_size(scheduler))
        if transfer and transfer.get("mode") in ("layer", "remote"):
            setup = _recv_transfer_layer_setup(transfer)
            t_handshake_mono = time.monotonic()
            if len(setup) == 6:
                # Backward-compatible shape used by older tests/fakes.
                conn, meta, kv_cpu, te, addrs, reg_owner = setup
                slot_bufs = None
                layer_num = len(kv_cpu)
                staging_bytes = 0
            else:
                (
                    conn,
                    meta,
                    kv_cpu,
                    te,
                    addrs,
                    reg_owner,
                    slot_bufs,
                    layer_num,
                    staging_bytes,
                ) = setup
            receiver_started = False
            state = None
            new_indices = None
            req = None
            try:
                origin = meta["origin_input_ids"]
                output = meta["output_ids"]
                kv_len = int(meta["kv_committed_len"])
                state_sampling = meta["sampling"]
                epoch = str(epoch0 or meta.get("migration_epoch") or "")
                recv_ms = (time.perf_counter() - t_recv) * 1000.0

                allocator = scheduler.token_to_kv_pool_allocator
                t_alloc = time.perf_counter()
                new_indices = _take_reserved(scheduler, rid0, kv_len)
                if new_indices is None:
                    new_indices = allocator.alloc(kv_len)
                if new_indices is None:
                    raise RuntimeError(
                        f"target KV pool cannot allocate {kv_len} tokens"
                    )
                alloc_ms = (time.perf_counter() - t_alloc) * 1000.0
                t_alloc_mono = time.monotonic()
                remaining = data.get("remaining")
                if remaining is None:
                    remaining = max(
                        1, int(state_sampling["max_new_tokens"]) - len(output)
                    )
                sp = _sampling_from(
                    state_sampling, len(output) + int(remaining), scheduler.tokenizer
                )
                rid = rid0
                req = _new_req(
                    scheduler, rid, origin, output, sp, **_logprob_kwargs(data)
                )
                req.kv.kv_committed_len = kv_len
                req._handoff_prefix_indices = new_indices.to(dtype=torch.int64)
                req.handoff_prefix_indices = req._handoff_prefix_indices
                req._handoff_epoch = epoch
                req._handoff_base_output_len = len(output)
                events = [threading.Event() for _ in range(layer_num)]
                state = _TargetHandoffState(
                    None,
                    kv_len,
                    events,
                    str(transfer["mode"]),
                    transfer.get("remote_service"),
                    handoff_id=meta.get("rid"),
                    trace_tag=meta.get("trace_tag"),
                    req=req,
                    tp_rank=_tp_rank(scheduler),
                    base_output_len=len(output),
                    migration_epoch=epoch,
                    host_staging_bytes=staging_bytes,
                )
                _install_attention_wrappers(scheduler, state)
                # Capture this scheduler's device while still on its own
                # thread; the receiver thread must not assume device 0.
                receiver_device = torch.cuda.current_device()
                state.indices_ready = torch.cuda.Event()
                state.indices_ready.record(torch.cuda.current_stream(receiver_device))
                receiver = threading.Thread(
                    target=(_slot_layer_receiver if slot_bufs else _layer_receiver),
                    args=(
                        scheduler,
                        state,
                        conn,
                        meta,
                        slot_bufs if slot_bufs else kv_cpu,
                        new_indices,
                        reg_owner,
                        receiver_device,
                    ),
                    daemon=True,
                    name="handoff-layer-recv",
                )
                state.receiver_conn = conn
                state.receiver_thread = receiver
                receiver.start()
                receiver_started = True
                if getattr(scheduler, "_handoff_hold_admission", False):
                    scheduler._handoff_held_reqs = getattr(
                        scheduler, "_handoff_held_reqs", []
                    )
                    scheduler._handoff_held_reqs.append(req)
                else:
                    scheduler.waiting_queue.append(req)
                tracked[rid] = req
                state.import_timeline = {
                    "cmd_recv_at": t_cmd_mono,
                    "handshake_done_at": t_handshake_mono,
                    "alloc_done_at": t_alloc_mono,
                    "receiver_start_at": time.monotonic(),
                    "queued_at": time.monotonic(),
                    "created_at": state.created_at,
                    "setup": meta.get("_setup_timeline"),
                }
                req._handoff_import_start = t_cmd_mono
                print(
                    f"[handoff-target] import rid={rid} "
                    f"handshake_ms={(t_handshake_mono - t_cmd_mono) * 1000:.1f} "
                    f"alloc_ms={(t_alloc_mono - t_handshake_mono) * 1000:.1f} "
                    f"setup_total_ms={
                        (
                            (meta.get('_setup_timeline') or {}).get('reply_at', 0)
                            - (meta.get('_setup_timeline') or {}).get(
                                'setup_enter_at', 0
                            )
                        )
                        * 1000:.1f}",
                    flush=True,
                )
                return {
                    "rid": rid,
                    "mode": transfer["mode"],
                    "imported_kv_tokens": kv_len,
                    "layers": len(events),
                    "output_len": len(output),
                    "remaining": int(remaining),
                    "first_new_token_pos": kv_len,
                    "recv_ms": recv_ms,
                    "alloc_ms": alloc_ms,
                    "load_ms": 0.0,
                    "migration_epoch": epoch,
                }
            except Exception:
                # A failed import must not leave a state bound to nothing: an
                # unbound entry in _TARGET_STATES can be matched by a later
                # batch and make it wait on events that will never be set.
                if state is not None:
                    stop_target_receivers(req)
                if req is not None:
                    try:
                        held = getattr(scheduler, "_handoff_held_reqs", [])
                        scheduler._handoff_held_reqs = [r for r in held if r is not req]
                        _remove_from_batches(scheduler, req)
                    except Exception:
                        pass
                if new_indices is not None:
                    try:
                        scheduler.token_to_kv_pool_allocator.free(new_indices)
                    except Exception:
                        pass
                if req is not None:
                    try:
                        tracked.pop(rid0, None)
                    except Exception:
                        pass
                if not receiver_started:
                    # The receiver thread never owned the connection or the
                    # registration, so this frame must release both.
                    try:
                        _release_transfer_buffers(
                            reg_owner,
                            slot_bufs or kv_cpu,
                            "target setup failed after advertising addresses",
                        )
                    except Exception:
                        pass
                    try:
                        conn.close()
                    except OSError:
                        pass
                raise

        if transfer:
            if transfer.get("mode") == "mooncake":
                meta, kv_cpu = _recv_transfer_mooncake(transfer)
            else:
                meta, kv_cpu = _recv_transfer(transfer)
            origin = meta["origin_input_ids"]
            output = meta["output_ids"]
            kv_len = int(meta["kv_committed_len"])
            state_sampling = meta["sampling"]
        else:
            state = torch.load(data["state"], map_location="cpu", weights_only=False)
            origin = state["origin_input_ids"]
            output = state["output_ids"]
            kv_len = int(state["kv_committed_len"])
            state_sampling = state["sampling"]
            kv_cpu = state["kv_cpu"]
        recv_ms = (time.perf_counter() - t_recv) * 1000.0
        if _import_cancelled(rid0, epoch0):
            raise RuntimeError(f"import epoch cancelled after receive for rid {rid0!r}")

        allocator = scheduler.token_to_kv_pool_allocator
        t_alloc = time.perf_counter()
        new_indices = _take_reserved(scheduler, rid0, kv_len)
        if new_indices is None:
            new_indices = allocator.alloc(kv_len)
        if new_indices is None:
            raise RuntimeError(f"target KV pool cannot allocate {kv_len} tokens")
        req = None
        try:
            alloc_ms = (time.perf_counter() - t_alloc) * 1000.0
            t_load = time.perf_counter()
            allocator.load_cpu_copy(kv_cpu, new_indices)
            load_ms = (time.perf_counter() - t_load) * 1000.0
            remaining = data.get("remaining")
            if remaining is None:
                remaining = max(1, int(state_sampling["max_new_tokens"]) - len(output))
            sp = _sampling_from(
                state_sampling, len(output) + int(remaining), scheduler.tokenizer
            )
            rid = rid0
            epoch = str(
                epoch0
                or (
                    meta.get("migration_epoch")
                    if transfer
                    else state.get("migration_epoch")
                )
                or ""
            )
            req = _new_req(scheduler, rid, origin, output, sp, **_logprob_kwargs(data))
            req.kv.kv_committed_len = kv_len
            req._handoff_prefix_indices = new_indices.to(dtype=torch.int64)
            req.handoff_prefix_indices = req._handoff_prefix_indices
            req._handoff_epoch = epoch
            req._handoff_base_output_len = len(output)
            if getattr(scheduler, "_handoff_hold_admission", False):
                scheduler._handoff_held_reqs = getattr(
                    scheduler, "_handoff_held_reqs", []
                )
                scheduler._handoff_held_reqs.append(req)
            else:
                scheduler.waiting_queue.append(req)
            tracked[rid] = req
            tstate = _TargetHandoffState(
                None,
                kv_len,
                [],
                "local",
                None,
                handoff_id=((meta.get("rid") if transfer else None) or rid),
                trace_tag=(
                    meta.get("trace_tag") if transfer else state.get("trace_tag")
                ),
                req=req,
                tp_rank=_tp_rank(scheduler),
                base_output_len=len(output),
                migration_epoch=epoch,
            )
            # Full-KV import: every layer is local as soon as load_cpu_copy
            # returns.  This is the target-side LOCAL flag the controller waits
            # for before releasing the source.
            tstate.import_done_at = time.monotonic()
            _install_attention_wrappers(scheduler, tstate)
            return {
                "rid": rid,
                "imported_kv_tokens": kv_len,
                "output_len": len(output),
                "remaining": int(remaining),
                "first_new_token_pos": kv_len,
                "recv_ms": recv_ms,
                "alloc_ms": alloc_ms,
                "load_ms": load_ms,
                "migration_epoch": epoch,
            }

        except Exception:
            _rollback_full_import(scheduler, tracked, rid0, req, new_indices)
            raise

    if cmd == "hold":
        scheduler.running_batch.batch_is_full = True
        return {"hold": True}

    if cmd == "release_hold":
        scheduler.running_batch.batch_is_full = False
        return {"hold": False}

    if cmd == "reap_bg":
        released = []
        for rid, req in list(tracked.items()):
            if not rid.startswith("bg-") or not req.finished():
                continue
            _remove_from_batches(scheduler, req)
            if req.kv.req_pool_idx is not None:
                kv_len = int(req.kv.kv_committed_len)
                if kv_len > 0:
                    indices = scheduler.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, :kv_len
                    ]
                    scheduler.token_to_kv_pool_allocator.free(indices)
                scheduler.req_to_token_pool.free(req)
            tracked.pop(rid, None)
            released.append(rid)
        return {"released": released}

    if cmd == "import_many":
        results = []
        errors = {}
        for item in data.get("items", []):
            rid = str(item["rid"])
            try:
                results.append(
                    _full_import_one(
                        scheduler,
                        tracked,
                        item["transfer"],
                        rid,
                        item.get("remaining"),
                        epoch=item.get("epoch"),
                        lp_kwargs=_logprob_kwargs(item),
                    )
                )
            except Exception as exc:
                # Per-item isolation: one failed request must not abort the
                # other imports or the target batch.
                errors[rid] = repr(exc)
                _reservation_free(scheduler, rid)
        return {
            "imports": results,
            "errors": errors,
            "count": len(results),
        }

    if cmd == "cancel_import":
        now = time.monotonic()
        cancelled = []
        for item in data.get("items", []):
            key = (str(item["rid"]), str(item.get("epoch", "")))
            _CANCELLED_IMPORTS[key] = now
            cancelled.append({"rid": key[0], "epoch": key[1]})
        return {"cancelled": cancelled, "count": len(cancelled)}

    if cmd == "fence_import":
        # Atomic source-recovery fence: tombstone this migration epoch, then
        # abort the uncommitted target request.  The epoch check in
        # abort_import prevents a stale source watchdog from aborting a newer
        # target generation that reused the same rid.
        rid = str(data["rid"])
        epoch = data.get("epoch")
        if epoch is None:
            raise RuntimeError("fence_import requires epoch")
        _CANCELLED_IMPORTS[(rid, str(epoch))] = time.monotonic()
        res = _handle(scheduler, {"cmd": "abort_import", "rid": rid, "epoch": epoch})
        print(
            f"[handoff-target] fence rid={rid} epoch={epoch} "
            f"result={json.dumps(res, ensure_ascii=False)[:400]}",
            flush=True,
        )
        return {"fenced": True, "cancel_epoch": str(epoch), **res}

    if cmd == "import_wave":
        # P5.2-D: one scheduler iteration starts all imports of a drain wave
        # and admits them together.  Each item is the same payload the
        # per-request "import" command accepts, so remote/layer/mooncake keep
        # their existing semantics.  A single failed item fails the wave: every
        # already-started import is aborted before any request is admitted.
        items = data.get("items", [])
        epoch = data.get("epoch")
        admit = bool(data.get("admit", True))
        admit_decode = bool(data.get("admit_decode", False))
        try:
            delay_ms = float(data.get("debug_delay_ms") or 0.0)
        except (TypeError, ValueError):
            delay_ms = 0.0
        if delay_ms > 0:
            # Failure-injection only: simulates a command that reaches the
            # target after the controller has timed out and resumed sources.
            time.sleep(delay_ms / 1000.0)
        prev_hold = getattr(scheduler, "_handoff_hold_admission", False)
        prev_held = list(getattr(scheduler, "_handoff_held_reqs", []))
        scheduler._handoff_hold_admission = True
        scheduler._handoff_held_reqs = []
        results = []
        errors = {}
        imported = []
        for item in items:
            rid = str(item.get("rid"))
            try:
                r = _handle(
                    scheduler,
                    {
                        "cmd": "import",
                        **item,
                        "epoch": item.get("epoch", epoch),
                    },
                )
                if _import_cancelled(rid, item.get("epoch", epoch)):
                    raise RuntimeError(
                        f"import cancelled by controller tombstone for {rid!r}"
                    )
                results.append(r)
                imported.append(rid)
            except Exception as exc:
                errors[rid] = repr(exc)
                _reservation_free(scheduler, rid)
        if errors:
            aborted = {}
            for rid in imported:
                try:
                    aborted[rid] = _handle(
                        scheduler, {"cmd": "abort_import", "rid": rid}
                    )
                except Exception as exc:
                    aborted[rid] = {"error": repr(exc)}
            scheduler._handoff_held_reqs = prev_held
            scheduler._handoff_hold_admission = prev_hold
            return {
                "imports": results,
                "errors": errors,
                "aborted": aborted,
                "admitted": 0,
                "count": len(results),
            }
        held = getattr(scheduler, "_handoff_held_reqs", [])
        if admit_decode:
            # TP2: admission needs the per-rank req_to_token row and the
            # prebuilt decode batch.  Run it in this same scheduler
            # iteration so new P->D admissions cannot slip between import and
            # admit, and remove a second control round-trip from the
            # first-token path.
            scheduler._handoff_held_reqs = prev_held + held
            scheduler._handoff_hold_admission = prev_hold
            ad = _handle(
                scheduler,
                {"cmd": "admit_decode", "rids": [r.rid for r in held]},
            )
            ad_errs = ad.get("errors") or {}
            if ad_errs:
                aborted = {}
                for rid in imported:
                    try:
                        aborted[rid] = _handle(
                            scheduler, {"cmd": "abort_import", "rid": rid}
                        )
                    except Exception as exc:
                        aborted[rid] = {"error": repr(exc)}
                scheduler._handoff_held_reqs = prev_held
                scheduler._handoff_hold_admission = prev_hold
                return {
                    "imports": results,
                    "errors": ad_errs,
                    "aborted": aborted,
                    "admitted": 0,
                    "count": len(results),
                }
            return {
                "imports": results,
                "errors": {},
                "admitted": int(ad.get("count") or 0),
                "admit_decode": ad,
                "count": len(results),
            }
        if admit:
            scheduler._handoff_hold_admission = False
            scheduler._handoff_held_reqs = []
            for req in held:
                scheduler.waiting_queue.append(req)
        else:
            scheduler._handoff_held_reqs = prev_held + held
            scheduler._handoff_hold_admission = prev_hold
        return {
            "imports": results,
            "errors": {},
            "admitted": len(held) if admit else 0,
            "count": len(results),
        }

    if cmd == "reserve":
        # Pre-freeze capacity reservation.  The target reserves KV *and* one
        # execution slot per migration request before the source freezes, so
        # ordinary admissions can use only the remaining authoritative
        # capacity and can never take the slot an import is about to need.
        items = data.get("items", [])
        epoch = str(data.get("epoch") or "")
        reserved = _reservations(scheduler)
        allocator = scheduler.token_to_kv_pool_allocator
        out = {}
        errors = {}
        for item in items:
            rid = str(item.get("rid"))
            try:
                tokens = int(item.get("tokens", 0))
                slots = int(item.get("slots", 1) or 0)
            except (TypeError, ValueError):
                errors[rid] = "invalid tokens"
                continue
            if tokens <= 0:
                errors[rid] = f"invalid token reservation {tokens}"
                continue
            if slots < 0:
                errors[rid] = f"invalid slot reservation {slots}"
                continue
            cur = reserved.get(rid)
            if cur is not None:
                if (
                    cur.get("epoch") == epoch
                    and int(cur.get("tokens", 0)) >= tokens
                    and int(cur.get("slots", 0)) >= slots
                ):
                    out[rid] = {
                        "tokens": int(cur.get("tokens", tokens)),
                        "slots": int(cur.get("slots", slots)),
                        "reused": True,
                    }
                else:
                    errors[rid] = "rid already reserved with another epoch/size"
                continue
            stats = _capacity_stats(scheduler)
            if slots > stats["free_slots"]:
                errors[rid] = (
                    f"insufficient slots: need {slots}, free {stats['free_slots']}"
                )
                continue
            avail = int(allocator.available_size())
            if tokens > stats["free_kv"]:
                errors[rid] = (
                    f"insufficient KV: need {tokens}, "
                    f"free {stats['free_kv']} (avail {avail}, "
                    f"ordinary leases {stats['ordinary_kv_leases']})"
                )
                continue
            idx = allocator.alloc(tokens)
            if idx is None:
                errors[rid] = f"alloc failed: need {tokens}, available {avail}"
                continue
            reserved[rid] = {
                "indices": idx,
                "tokens": tokens,
                "slots": slots,
                "slot_consumed": False,
                "epoch": epoch,
                "created_at": time.monotonic(),
            }
            out[rid] = {
                "tokens": tokens,
                "slots": slots,
                "available_before": avail,
            }
        return {
            "reserved": out,
            "errors": errors,
            "available": int(allocator.available_size()),
            "capacity": _capacity_stats(scheduler),
            "migration_epoch": epoch,
        }

    if cmd == "reserve_cancel":
        rids = [str(x) for x in data.get("rids", [])]
        cancelled = []
        missing = []
        for rid in rids:
            if _reservation_free(scheduler, rid) is not None:
                cancelled.append(rid)
            else:
                missing.append(rid)
        return {
            "cancelled": cancelled,
            "missing": missing,
            "available": int(scheduler.token_to_kv_pool_allocator.available_size()),
        }

    if cmd == "abort_import":
        # Undo a target-side import that has not committed any new token.
        # Once the target has appended even one output token the old source
        # state can no longer be resumed without replay; report it instead of
        # silently duplicating output.
        rid = str(data["rid"])
        epoch = data.get("epoch")
        req = tracked.get(rid)
        if req is not None and epoch is not None:
            req_epoch = getattr(req, "_handoff_epoch", None)
            if req_epoch is None or str(req_epoch) != str(epoch):
                return {
                    "aborted": False,
                    "committed": False,
                    "epoch_mismatch": True,
                    "target_epoch": req_epoch,
                }
        state = next(
            (st for st in _TARGET_STATES if req is not None and st.req is req),
            None,
        )
        new_tokens = (
            state.committed_new_tokens()
            if state is not None
            else (
                0
                if req is None
                else max(
                    0,
                    len(req.output_ids)
                    - int(getattr(req, "_handoff_base_output_len", 0)),
                )
            )
        )
        if new_tokens > 0:
            return {
                "aborted": False,
                "committed": True,
                "new_tokens": int(new_tokens),
                "reason": "target_committed",
            }
        _reservation_free(scheduler, rid)
        if req is None:
            return {
                "aborted": True,
                "committed": False,
                "new_tokens": 0,
                "missing": True,
            }
        held = getattr(scheduler, "_handoff_held_reqs", [])
        scheduler._handoff_held_reqs = [r for r in held if r is not req]
        _remove_from_batches(scheduler, req)
        before = _kv_available(scheduler)
        _free_source_request(scheduler, req)
        freed = max(0, _kv_available(scheduler) - before)
        tracked.pop(rid, None)
        _reservation_free(scheduler, rid)
        return {
            "aborted": True,
            "committed": False,
            "new_tokens": 0,
            "freed_kv_tokens": freed,
        }

    if cmd == "resume":
        # Source-side rollback: only valid while the request is frozen and the
        # target has confirmed that it committed no new token.
        rid = str(data["rid"])
        req = tracked.get(rid)
        if req is None:
            raise KeyError(f"unknown rid {rid!r}")
        if not getattr(req, "_handoff_frozen", False):
            return {"resumed": False, "reason": "not_frozen"}
        epoch_raw = data.get("epoch")
        if epoch_raw is not None:
            try:
                _claim_epoch = int(epoch_raw)
            except (TypeError, ValueError):
                return {
                    "resumed": False,
                    "reason": "bad_epoch",
                    "epoch": epoch_raw,
                }
            if _resume_blocked(f"{rid}:{_claim_epoch}"):
                # Another source rank proved the target owns this epoch
                # between fence and execution.  Refuse on every TP rank.
                return {
                    "resumed": False,
                    "reason": "resume_blocked",
                    "epoch": _claim_epoch,
                }
        hub = getattr(scheduler, "_handoff_hub", None)
        epoch = data.get("epoch")
        if epoch is not None:
            try:
                owner_epoch = int(epoch)
            except (TypeError, ValueError):
                return {
                    "resumed": False,
                    "reason": "bad_epoch",
                    "epoch": epoch,
                }
            ctx = hub.registry.get(rid) if hub is not None else None
            if ctx is not None and int(ctx.owner_epoch) != owner_epoch:
                return {
                    "resumed": False,
                    "reason": "epoch_mismatch",
                    "epoch": owner_epoch,
                    "owner_epoch": int(ctx.owner_epoch),
                }
        stop_source_transfer(req)
        if hub is not None:
            ctx = hub.registry.get(rid)
            if ctx is not None:
                if not hub.registry.discard(rid, expected_req=req):
                    return {
                        "resumed": False,
                        "pending": True,
                        "reason": "inflight_rpc",
                    }
        req._handoff_frozen = False
        req._handoff_freeze_result = None
        req._handoff_pause_after_prefill = False
        # A decode request's prefix_indices may only cover the prompt.  The
        # scheduler will re-extend the already-generated tokens after resume,
        # so their old KV slots become orphans; free them before re-queueing.
        # This avoids both the leak and a double free of the prompt prefix.
        kv_len = int(req.kv.kv_committed_len)
        prefix_len = 0
        idx = getattr(req, "_handoff_prefix_indices", None)
        if idx is not None:
            try:
                prefix_len = int(idx.numel())
            except Exception:
                prefix_len = 0
        try:
            avail = int(scheduler.token_to_kv_pool_allocator.available_size())
        except Exception:
            avail = kv_len
        avail_at_resume = avail
        fast = os.environ.get("HANDOFF_FAST_RESUME", "1") == "1"
        if fast:
            # Pre-commit fast resume: keep the frozen KV, the req_pool slot and
            # the materialized token mapping.  The scheduler consumes
            # handoff_prefix_indices, extends only the pending sampled token
            # (fill_ids[kv_len:seqlen]), and resumes decode without re-prefill.
            out_len = len(req.output_ids)
            prompt_len = len(req.origin_input_ids)
            expected = prompt_len + (out_len - 1 if out_len > 0 else 0)
            if req.kv.req_pool_idx is None or out_len < 1 or kv_len != expected:
                raise RuntimeError(
                    "fast_resume invariant failed: "
                    f"pool_idx={req.kv.req_pool_idx} kv_len={kv_len} "
                    f"expected={expected} out_len={out_len}"
                )
            mapping = (
                scheduler.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :kv_len]
                .to(dtype=torch.int64)
                .clone()
            )
            if int(mapping.numel()) != kv_len:
                raise RuntimeError(
                    f"fast_resume mapping short: {int(mapping.numel())} < {kv_len}"
                )
            try:
                _kv = scheduler.token_to_kv_pool_allocator.get_kvcache()
                _total = int(getattr(_kv, "size", 0))
                _mn = int(mapping.min())
                _mx = int(mapping.max())
                _uniq = int(torch.unique(mapping).numel())
                print(
                    f"[kv-fast-resume] map numel={int(mapping.numel())} "
                    f"min={_mn} max={_mx} uniq={_uniq} total={_total} "
                    f"dtype={mapping.dtype} seqlen={req.seqlen}",
                    flush=True,
                )
                if _mn < 0 or (_total and _mx > _total) or _uniq != kv_len:
                    raise RuntimeError(
                        f"fast_resume mapping invalid: min={_mn} max={_mx} "
                        f"uniq={_uniq} total={_total} kv_len={kv_len}"
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"[kv-fast-resume] map check skipped: {exc!r}", flush=True)
            old_pool = req.kv.req_pool_idx
            # A full-batch resume (all max_running requests) would otherwise
            # hold every req_pool slot before admission, making
            # get_num_allocatable_reqs()==0 and stalling the waiting queue.
            # Release the slot now; alloc_for_extend reassigns one and the
            # scheduler bind writes this mapping into the new slot.  KV slots
            # are untouched.
            try:
                if req.kv.req_pool_idx is not None:
                    scheduler.req_to_token_pool.free(req)
                    # The pending mapping is now owned by the plugin, not a
                    # registered Req.kv row. Admission restores ownership.
                    req.kv.mark_kv_released()
            except Exception as exc:
                print(f"[kv-fast-resume] pool free skipped: {exc!r}", flush=True)
            req._handoff_prefix_indices = mapping
            req.handoff_prefix_indices = mapping
            req._handoff_resume_prefix_len = 0
            req._handoff_resume_prefix_indices = None
            req._handoff_resume_prefix_pending = False
            req._handoff_full_reset = False
            req._handoff_fast_resume = True
            print(
                f"[kv-fast-resume] rid={rid} pool_idx={old_pool} "
                f"kv_len={kv_len} seqlen={prompt_len + out_len} "
                f"pending=1 avail={_kv_available(scheduler)}",
                flush=True,
            )
            if data.get("decode"):
                # Decode-style resume: bind the prefix into a fresh req_pool
                # row and let the pending output token be processed by the
                # normal decode kernel (same as a PD decode worker).
                bind_decode_prefix(req, scheduler.req_to_token_pool, mapping, kv_len)
                req._handoff_decode_resume = True
                if data.get("defer_batch"):
                    pending = getattr(scheduler, "_handoff_decode_resume_reqs", [])
                    if not any(r is req for r in pending):
                        pending.append(req)
                    scheduler._handoff_decode_resume_reqs = pending
                else:
                    _flush_decode_resume(scheduler, [req])
                return {
                    "resumed": True,
                    "rid": rid,
                    "fast": True,
                    "decode_resume": True,
                    "recompute_tokens": 0,
                    "prefix_len": kv_len,
                    "output_len": out_len,
                    "kv_committed_len": int(req.kv.kv_committed_len),
                }
            if not any(r is req for r in scheduler.waiting_queue):
                scheduler.waiting_queue.append(req)
            return {
                "resumed": True,
                "rid": rid,
                "fast": True,
                "recompute_tokens": 0,
                "prefix_len": kv_len,
                "output_len": out_len,
                "kv_committed_len": int(req.kv.kv_committed_len),
            }
        # With the radix cache disabled the scheduler does not reuse
        # ``prefix_indices`` when a request is re-queued: it re-prefills the
        # prompt.  Keeping the old prefix across resume therefore duplicates
        # the prompt KV (a hard 16 K slot leak) and can mix old/new mappings.
        # Free the whole frozen KV and let the scheduler rebuild it.
        keep_prefix = os.environ.get("HANDOFF_RESUME_KEEP_PREFIX", "0") == "1"
        full_reset = req.kv.req_pool_idx is not None and (
            avail < kv_len or not keep_prefix
        )
        if full_reset:
            # If the pool cannot also hold a fresh extend, keeping the frozen
            # prefix leaves the request stuck in waiting_queue (C4 at 16K: 4
            # prefixes fill the pool).  Free the whole KV once and let the
            # scheduler re-prefill prompt+output; failure recovery is allowed
            # to be slower, and the token stream is still authoritative.
            try:
                full = scheduler.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :kv_len
                ].clone()
                scheduler.token_to_kv_pool_allocator.free(full)
            except Exception:
                pass
            req._handoff_prefix_indices = None
            req.handoff_prefix_indices = None
            req.kv.kv_committed_len = 0
            scheduler.req_to_token_pool.free(req)
            req.kv.mark_kv_released()
            req.prefix_len = 0
            req.extend_end = None
            req._handoff_resume_prefix_len = 0
            req._handoff_resume_prefix_indices = None
            req._handoff_resume_prefix_pending = False
            req._handoff_full_reset = True
            print(
                f"[kv-resume] rid={rid} branch=full pool_idx={req.kv.req_pool_idx} "
                f"kv_len={kv_len} prefix_len={prefix_len} "
                f"avail={avail_at_resume}->{_kv_available(scheduler)}",
                flush=True,
            )
        elif req.kv.req_pool_idx is not None and 0 <= prefix_len < kv_len:
            # Keep the prompt prefix; free only the generated-token orphan
            # range so the scheduler can re-extend it.
            try:
                orphan = scheduler.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, prefix_len:kv_len
                ].clone()
                scheduler.token_to_kv_pool_allocator.free(orphan)
            except Exception:
                pass
            req._handoff_resume_prefix_len = prefix_len
            req._handoff_resume_prefix_indices = idx
            req._handoff_resume_prefix_pending = idx is not None
            print(
                f"[kv-resume] rid={rid} branch=partial pool_idx={req.kv.req_pool_idx} "
                f"kv_len={kv_len} prefix_len={prefix_len} "
                f"freed_orphan={max(0, kv_len - prefix_len)} "
                f"avail={avail_at_resume}->{_kv_available(scheduler)}",
                flush=True,
            )
        else:
            req._handoff_resume_prefix_len = prefix_len
            req._handoff_resume_prefix_indices = idx
            req._handoff_resume_prefix_pending = idx is not None
            print(
                f"[kv-resume] rid={rid} branch=keep pool_idx={req.kv.req_pool_idx} "
                f"kv_len={kv_len} prefix_len={prefix_len} avail={avail_at_resume}",
                flush=True,
            )
        if not any(r is req for r in scheduler.waiting_queue):
            scheduler.waiting_queue.append(req)
        return {
            "resumed": True,
            "rid": rid,
            "output_len": len(req.output_ids),
            "kv_committed_len": int(req.kv.kv_committed_len),
        }

    if cmd == "resume_many_decode":
        rids = [str(x) for x in (data.get("rids") or [])]
        epoch = data.get("epoch")
        results = {}
        errors = {}
        for rid in rids:
            try:
                results[rid] = _handle(
                    scheduler,
                    {
                        "cmd": "resume",
                        "rid": rid,
                        "decode": True,
                        "defer_batch": True,
                        "epoch": epoch,
                    },
                )
            except Exception as exc:
                errors[rid] = repr(exc)
        reqs = getattr(scheduler, "_handoff_decode_resume_reqs", [])
        scheduler._handoff_decode_resume_reqs = []
        if reqs:
            try:
                _flush_decode_resume(scheduler, reqs)
            except Exception as exc:
                errors["__flush__"] = repr(exc)
        print(
            f"[handoff-resume] tp={_tp_rank(scheduler)} "
            f"results={json.dumps(results, ensure_ascii=False)[:1200]} "
            f"errors={errors}",
            flush=True,
        )
        return {"results": results, "errors": errors, "count": len(reqs)}

    if cmd == "admit_decode":
        rids = [str(x) for x in (data.get("rids") or [])]
        results = {}
        errors = {}
        reqs = []
        held = list(getattr(scheduler, "_handoff_held_reqs", []))
        for rid in rids:
            try:
                req = tracked.get(rid)
                if req is None:
                    raise KeyError(f"unknown rid {rid!r}")
                kv_len = int(req.kv.kv_committed_len or 0)
                mapping = getattr(req, "handoff_prefix_indices", None)
                if mapping is None:
                    mapping = getattr(req, "_handoff_prefix_indices", None)
                if mapping is None or int(mapping.numel()) < kv_len:
                    raise RuntimeError(
                        f"rid {rid!r} has no imported mapping for kv={kv_len}"
                    )
                mapping = mapping[:kv_len].to(dtype=torch.int64)
                bind_decode_prefix(req, scheduler.req_to_token_pool, mapping, kv_len)
                req._handoff_decode_resume = True
                req._handoff_admit_at = time.monotonic()
                reqs.append(req)
                _imp_start = getattr(req, "_handoff_import_start", None)
                if _imp_start is not None:
                    _imp_ms = (req._handoff_admit_at - _imp_start) * 1000.0
                    print(
                        f"[handoff-target] admit rid={rid} "
                        f"import_to_admit_ms={_imp_ms:.1f}",
                        flush=True,
                    )
                results[rid] = {
                    "admitted": True,
                    "decode_resume": True,
                    "kv_committed_len": kv_len,
                    "output_len": len(req.output_ids),
                }
            except Exception as exc:
                errors[rid] = repr(exc)
        scheduler._handoff_held_reqs = [r for r in held if r not in reqs]
        scheduler._handoff_hold_admission = False
        if reqs:
            try:
                _flush_decode_resume(scheduler, reqs)
            except Exception as exc:
                errors["__flush__"] = repr(exc)
        return {"results": results, "errors": errors, "count": len(reqs)}

    if cmd == "debug_queue":
        rows = []
        for req in list(scheduler.waiting_queue):
            er = getattr(req, "extend_range", None)
            pi = getattr(req, "_handoff_prefix_indices", None)
            hp = getattr(req, "handoff_prefix_indices", None)
            rows.append(
                {
                    "rid": req.rid,
                    "kv_committed_len": int(req.kv.kv_committed_len),
                    "seqlen": int(req.seqlen),
                    "output_len": len(req.output_ids),
                    "req_pool_idx": req.kv.req_pool_idx,
                    "prefix_len": int(pi.numel()) if pi is not None else None,
                    "handoff_prefix_pending": hp is not None,
                    "extend_range": (
                        None
                        if er is None
                        else {
                            "start": int(er.start),
                            "end": int(er.end),
                            "length": int(er.length),
                        }
                    ),
                    "finished": bool(req.finished()),
                    "inflight_middle_chunks": int(
                        getattr(req, "inflight_middle_chunks", 0) or 0
                    ),
                    "held": bool(getattr(req, "handoff_held", False)),
                }
            )
        return {"waiting": rows, "count": len(rows)}

    if cmd == "profiler":
        # Diagnostic: drive an external nsys capture range from the driver.
        action = str(data.get("action") or "")
        if action == "start":
            torch.cuda.cudart().cudaProfilerStart()
        elif action == "stop":
            torch.cuda.cudart().cudaProfilerStop()
        else:
            raise ValueError(f"unknown profiler action {action!r}")
        return {"action": action, "ok": True}

    if cmd == "kv_digest":
        # Diagnostic: byte-exact digest of this rank's KV shard for a tracked
        # request.  Source and target rank r must produce identical hashes
        # after the same freeze/import, independently of token equality.
        rid = str(data.get("rid"))
        req = tracked.get(rid)
        if req is None:
            raise KeyError(f"unknown rid {rid!r}")
        kv_len = int(data.get("kv_len") or req.kv.kv_committed_len or 0)
        if kv_len <= 0:
            kv_len = len(req.origin_input_ids) + len(req.output_ids)
        r2t = scheduler.req_to_token_pool.req_to_token
        if req.kv.req_pool_idx is not None:
            idx = r2t[req.kv.req_pool_idx, :kv_len].to(dtype=torch.int64)
        else:
            # Imported but not yet scheduled: KV lives at the imported
            # prefix mapping and req_to_token is bound later by alloc_for_extend.
            hp = getattr(req, "handoff_prefix_indices", None)
            if hp is None:
                hp = getattr(req, "_handoff_prefix_indices", None)
            if hp is None:
                raise RuntimeError(f"rid {rid!r} has no KV mapping yet")
            idx = hp[:kv_len].to(dtype=torch.int64)
        max_tok = int(data.get("max_tokens") or 0)
        if max_tok > 0 and kv_len > max_tok:
            pos = (
                torch.linspace(0, kv_len - 1, max_tok, device=idx.device).round().long()
            )
            idx = idx[pos]
        pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        if int(getattr(scheduler, "page_size", 1) or 1) != 1:
            raise RuntimeError("kv_digest assumes page_size=1")
        layer_num = int(getattr(pool, "layer_num", 0))
        layers = {}
        total = hashlib.sha1()
        for layer in range(layer_num):
            k = pool.get_key_buffer(layer)[idx].detach().cpu().contiguous()
            v = pool.get_value_buffer(layer)[idx].detach().cpu().contiguous()
            # Hash raw storage bytes: bfloat16 is not a numpy dtype, and a
            # dtype-preserving byte view also covers fp8 KV caches.
            kb = k.flatten().view(torch.uint8).numpy().tobytes()
            vb = v.flatten().view(torch.uint8).numpy().tobytes()
            kh = hashlib.sha1(kb).hexdigest()
            vh = hashlib.sha1(vb).hexdigest()
            layers[str(layer)] = {
                "k": kh,
                "v": vh,
                "shape": list(k.shape),
                "dtype": str(k.dtype),
            }
            total.update(kh.encode())
            total.update(vh.encode())
        return {
            "rid": rid,
            "kv_len": kv_len,
            "sampled": int(idx.numel()),
            "layers": layers,
            "digest": total.hexdigest(),
            "tp_rank": _tp_rank(scheduler),
            "tp_size": _tp_size(scheduler),
            "max_running": int(get_schedule().max_running_requests or 0),
        }

    if cmd == "status_lite":
        rows = []
        for rid, req in tracked.items():
            rows.append(
                {
                    "rid": rid,
                    "output_len": len(req.output_ids),
                    "finished": bool(req.finished()),
                    "kv_committed_len": int(req.kv.kv_committed_len),
                    "frozen": bool(getattr(req, "_handoff_frozen", False)),
                }
            )
        return {
            "reqs": rows,
            "kv_available": _kv_available(scheduler),
            "capacity": _capacity_stats(scheduler),
            "tp_rank": _tp_rank(scheduler),
            "tp_size": _tp_size(scheduler),
            "max_running": int(get_schedule().max_running_requests or 0),
        }

    if cmd == "status":
        rows = []
        for rid, req in tracked.items():
            rows.append(
                {
                    "rid": rid,
                    "output_len": len(req.output_ids),
                    "output_ids": [int(x) for x in req.output_ids],
                    "kv_committed_len": int(req.kv.kv_committed_len),
                    "seqlen": int(req.seqlen),
                    "finished": bool(req.finished()),
                    "finished_reason": str(req.finished_reason),
                    "output_token_logprobs": (
                        [float(x) for x in req.logprob.output_token_logprobs_val]
                        if req.return_logprob and req.logprob.output_token_logprobs_val
                        else None
                    ),
                    "output_top_logprobs": (
                        [
                            {
                                "lp": [float(x) for x in vals],
                                "ids": [int(x) for x in ids],
                            }
                            for vals, ids in zip(
                                req.logprob.output_top_logprobs_val or [],
                                req.logprob.output_top_logprobs_idx or [],
                            )
                        ]
                        if req.return_logprob and req.logprob.output_top_logprobs_val
                        else None
                    ),
                    "req_pool_idx": req.kv.req_pool_idx,
                    "frozen": bool(getattr(req, "_handoff_frozen", False)),
                    "freeze_result": getattr(req, "_handoff_freeze_result", None),
                    "freeze_error": getattr(req, "_handoff_freeze_error", None),
                    "handoff_prefix_pending": getattr(
                        req, "handoff_prefix_indices", None
                    )
                    is not None,
                }
            )
        _prefix_cleared = HANDOFF_PREFIX_CLEARED
        result = {
            "reqs": rows,
            "running": [r.rid for r in scheduler.running_batch.reqs],
            "waiting": [r.rid for r in scheduler.waiting_queue],
            "reserved": {
                rid: {
                    "tokens": int(res.get("tokens", 0)),
                    "slots": int(res.get("slots", 0)),
                    "slot_consumed": bool(res.get("slot_consumed")),
                    "kv_consumed": res.get("indices") is None,
                    "epoch": res.get("epoch"),
                }
                for rid, res in _reservations(scheduler).items()
            },
            "capacity": _capacity_stats(scheduler),
            "ordinary_leases": {
                rid: {
                    "tokens": int(l.get("tokens", 0)),
                    "slot_released": bool(l.get("slot_released")),
                    "epoch": l.get("epoch"),
                }
                for rid, l in _ordinary_leases(scheduler).items()
            },
            "kv_available": _kv_available(scheduler),
            "handoff_prefix_cleared": _prefix_cleared,
            "tp_rank": _tp_rank(scheduler),
            "tp_size": _tp_size(scheduler),
            "max_running": int(get_schedule().max_running_requests or 0),
            "handoff_iter": {
                "id": _ITER,
                "period_ms": round(_ITER_PERIOD_MS, 3),
            },
            "transfer_errors": dict(_TRANSFER_ERRORS),
            "transfer_stats": dict(_TRANSFER_STATS),
            "gil_stats": _gil_stats() if PROFILE else None,
            "target_states": [
                {
                    "handoff_id": st.handoff_id,
                    "mode": st.mode,
                    "req_pool_idx": st.req_pool_idx,
                    "base_len": st.base_len,
                    "remote_calls": st.remote_calls,
                    "local_calls": st.local_calls,
                    "wait_calls": st.wait_calls,
                    "remote_fallback": st.remote_fallback,
                    "last_remote_ms": st.last_remote_ms,
                    "remote_ms_threshold": st.remote_ms_threshold,
                    "step_index": st._cur_step,
                    "created_at": st.created_at,
                    "admitted_at": st.admitted_at,
                    "finished_at": st.finished_at,
                    "step_starts": list(st.step_starts),
                    "step_remote": st._step_remote,
                    "step_local": st._step_local,
                    "step_mask": st._step_mask,
                    "local_ready": bool(
                        st.import_done_at is not None
                        and st.error is None
                        and all(ev.is_set() for ev in st.events)
                    ),
                    "layers_total": len(st.events),
                    "layers_ready": sum(1 for ev in st.events if ev.is_set()),
                    "error": st.error,
                    **st.metrics(),
                }
                for st in _TARGET_STATES
            ],
            "target_state": (
                None
                if not _TARGET_STATES
                else {
                    "mode": _TARGET_STATES[0].mode,
                    "req_pool_idx": _TARGET_STATES[0].req_pool_idx,
                    "base_len": _TARGET_STATES[0].base_len,
                    "remote_calls": _TARGET_STATES[0].remote_calls,
                    "local_calls": _TARGET_STATES[0].local_calls,
                    "wait_calls": _TARGET_STATES[0].wait_calls,
                    "remote_fallback": _TARGET_STATES[0].remote_fallback,
                    "last_remote_ms": _TARGET_STATES[0].last_remote_ms,
                    "remote_ms_threshold": _TARGET_STATES[0].remote_ms_threshold,
                    "last_pre_ms": _TARGET_STATES[0].last_pre_ms,
                    "last_kernel_ms": _TARGET_STATES[0].last_kernel_ms,
                    "last_queue_ms": _TARGET_STATES[0].last_queue_ms,
                    "remote_stats": (_TARGET_STATES[0].remote_stats if PROFILE else []),
                    "step_log": _TARGET_STATES[0].step_log if PROFILE else [],
                    "gil_stats": _gil_stats() if PROFILE else None,
                    "local_ready": bool(
                        _TARGET_STATES[0].import_done_at is not None
                        and _TARGET_STATES[0].error is None
                        and all(ev.is_set() for ev in _TARGET_STATES[0].events)
                    ),
                    "layers_total": len(_TARGET_STATES[0].events),
                    "layers_ready": sum(
                        1 for ev in _TARGET_STATES[0].events if ev.is_set()
                    ),
                    "error": _TARGET_STATES[0].error,
                    **_TARGET_STATES[0].metrics(),
                }
            ),
        }
        hub = getattr(scheduler, "_handoff_hub", None)
        result["handoff_registry"] = hub.registry.stats() if hub is not None else None
        result["hub_stats"] = hub.stats() if hub is not None else None
        with _TRANSFER_RELEASE_LOCK:
            result["quarantined_transfers"] = {
                key: reason for key, (_, reason) in _TRANSFER_QUARANTINE.items()
            }
        result["registration_manager"] = _get_registration_manager().stats()
        with _TARGET_CLIENTS_LOCK:
            clients = list(_TARGET_CLIENTS.items())
        result["target_clients"] = [
            {"host": host, "port": port, **client.stats()}
            for (host, port), client in clients
        ]
        result["batch_attention"] = dict(_M4_STATS)
        result["completed_target_states"] = list(_TARGET_METRICS.values())
        out = data.get("out")
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )
        return result

    if cmd == "kv_ledger":
        allocator = scheduler.token_to_kv_pool_allocator
        rows = []
        for rid, req in tracked.items():
            idx = getattr(req, "_handoff_prefix_indices", None)
            try:
                prefix_n = int(idx.numel()) if idx is not None else 0
            except Exception:
                prefix_n = -1
            rows.append(
                {
                    "rid": rid,
                    "req_pool_idx": req.kv.req_pool_idx,
                    "kv_committed_len": int(req.kv.kv_committed_len),
                    "output_len": len(req.output_ids),
                    "prefix_indices": prefix_n,
                    "resume_prefix_len": int(
                        getattr(req, "_handoff_resume_prefix_len", 0) or 0
                    ),
                    "resume_prefix_pending": bool(
                        getattr(req, "_handoff_resume_prefix_pending", False)
                    ),
                    "full_reset": bool(getattr(req, "_handoff_full_reset", False)),
                    "frozen": bool(getattr(req, "_handoff_frozen", False)),
                    "finished": bool(req.finished()),
                    "freed": bool(getattr(req, "_handoff_freed", False)),
                }
            )
        return {
            "available": _kv_available(scheduler),
            "tracked": rows,
            "reserved": {
                rid: {
                    "tokens": int(res["indices"].numel()),
                    "epoch": res.get("epoch"),
                }
                for rid, res in _reservations(scheduler).items()
            },
            "handoff_registry": (
                getattr(scheduler, "_handoff_hub", None).registry.stats()
                if getattr(scheduler, "_handoff_hub", None) is not None
                else None
            ),
        }

    if cmd == "release":
        rid = str(data["rid"])
        _ordinary_lease_free(scheduler, rid, force=True)
        req = tracked.get(rid)
        if req is None:
            if data.get("unknown_ok"):
                return {"rid": rid, "released": True, "missing": True}
            raise KeyError(f"unknown rid {rid!r}")
        hub = getattr(scheduler, "_handoff_hub", None)
        if hub is not None:
            try:
                released = hub.registry.release(rid, expected_req=req)
            except KeyError:
                # Absent, already evicted, or a stale generation. Clear any
                # stale entry best-effort; this request still needs its slots.
                try:
                    hub.registry.release(rid)
                except KeyError:
                    pass
                tracked.pop(rid, None)
                _free_source_request(scheduler, req)
                _drop_target_state_for(req)
                return {"rid": rid, "released": True}
            if not released:
                # An RPC is still reading the frozen KV. The registry will call
                # on_release when the last inflight reference drops.
                print(f"[handoff-hub] release deferred id={rid}", flush=True)
                _drop_target_state_for(req)
                return {"rid": rid, "released": False, "deferred": True}
            tracked.pop(rid, None)
            _drop_target_state_for(req)
            return {"rid": rid, "released": True}
        tracked.pop(rid, None)
        _free_source_request(scheduler, req)
        _drop_target_state_for(req)
        return {"rid": rid, "released": True}

    raise ValueError(f"unknown handoff command {cmd!r}")
