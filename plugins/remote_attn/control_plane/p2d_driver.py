#!/usr/bin/env python3
"""P->D arms driver: source freezes at the prefill boundary, target resumes.

Modes:
  copy_wait          full KV transfer before the target starts decode
  layer_wait         target starts decode; per-layer KV gate
  remote_progressive target starts decode; remote attention for non-local layers
"""
from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from pathlib import Path

try:
    import fcntl
except ImportError:  # local Windows mirrors never open the control channel
    fcntl = None

from transformers import AutoTokenizer

CTL = Path("/workspace/handoff_ctl")
STATE = Path("/workspace/handoff_state")
STATE.mkdir(parents=True, exist_ok=True)
MODEL = "/workspace/models/Qwen3-8B"
_READY_WAIT_S = 60.0
_BARE_LEGACY_WAIT_S = 2.0
INPUT_LEN = int(os.environ.get("HANDOFF_INPUT_LEN", "4096"))
TOTAL_NEW = int(os.environ.get("HANDOFF_TOTAL_NEW", "128"))
MODE = os.environ.get("HANDOFF_MODE", "copy_wait")
RESULT_NAME = os.environ.get("HANDOFF_RESULT_NAME", f"p2d_{MODE}.json")
SERVER_MODE = {"copy_wait": "mooncake", "layer_wait": "layer", "remote_progressive": "remote"}[MODE]
TRANSFER_OVERRIDE = os.environ.get("HANDOFF_TRANSFER")
if TRANSFER_OVERRIDE and MODE == "copy_wait":
    SERVER_MODE = TRANSFER_OVERRIDE
TRANSFER_MODE = SERVER_MODE


def _ready(role: str) -> list:
    """Live rank readiness files written atomically by RemoteHub on each rank.

    Reads exact names (no directory scan): ``ready.0.json`` defines tp_size and
    the remaining eager per-rank files are read directly.
    """
    base = CTL / role
    first = base / "ready.0.json"
    try:
        info0 = json.loads(first.read_text(encoding="utf-8"))
    except Exception:
        return []
    try:
        n = max(1, int(info0.get("tp_size", 1)))
    except Exception:
        n = 1
    out = []
    for r in range(n):
        path = base / f"ready.{r}.json"
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            pid = int(info.get("pid", -1))
        except Exception:
            continue
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        try:
            rank = int(info.get("rank", -1))
        except Exception:
            continue
        if rank != r:
            continue
        out.append(info)
    return out


def _send_legacy(d: Path, role: str, payload: dict, timeout: float) -> dict:
    """Pre-multi-rank control protocol: shared cmd.json/result.json."""
    res_path = d / "result.json"
    try:
        res_path.unlink()
    except FileNotFoundError:
        pass
    tmp = d / "cmd.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, d / "cmd.json")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if res_path.exists():
            try:
                res = json.loads(res_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                time.sleep(0.02)
                continue
            if not res.get("ok"):
                raise RuntimeError(f"{role} {payload['cmd']} failed: {json.dumps(res)[:4000]}")
            return res
        time.sleep(0.02)
    raise TimeoutError(f"{role} {payload['cmd']} timed out after {timeout}s")


def send(role: str, payload: dict, timeout: float = 300.0) -> dict:
    # The file control protocol has a single command slot per role.  The
    # Router (b1_proxy) and the Migration Controller (p4_gateway) can both be
    # alive after a failover, so serialize cross-process access with flock.
    with _role_lock(role):
        return _send_unlocked(role, payload, timeout)


@contextlib.contextmanager
def _role_lock(role: str):
    if fcntl is None:
        yield
        return
    d = CTL / role
    d.mkdir(parents=True, exist_ok=True)
    fh = open(d / ".control.lock", "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _send_unlocked(role: str, payload: dict, timeout: float = 300.0) -> dict:
    d = CTL / role
    d.mkdir(parents=True, exist_ok=True)
    infos = _ready(role)
    if not infos:
        # A new server may still be creating its Hub/ready file, but an old
        # single-rank server never publishes one: cap the wait, then go legacy.
        deadline = time.time() + _BARE_LEGACY_WAIT_S
        while not infos and time.time() < deadline:
            time.sleep(0.1)
            infos = _ready(role)
    if not infos:
        # Old single-rank servers publish no ready file: legacy protocol.
        return _send_legacy(d, role, payload, timeout)

    expected = max(1, max(int(i.get("tp_size", 1)) for i in infos))
    ranks = [int(i.get("rank", 0)) for i in infos]
    if ranks != list(range(expected)):
        # A rank may still be bringing its Hub up; give it a bounded window.
        deadline = time.time() + _READY_WAIT_S
        while time.time() < deadline:
            time.sleep(0.1)
            infos = _ready(role)
            if not infos:
                continue
            ranks = [int(i.get("rank", 0)) for i in infos]
            expected = max(1, max(int(i.get("tp_size", 1)) for i in infos))
            if ranks == list(range(expected)):
                break
    if ranks != list(range(expected)):
        raise RuntimeError(
            f"{role}: expected TP ranks 0..{expected - 1}, ready={ranks} "
            f"(ready files incomplete after {_READY_WAIT_S:.0f}s)"
        )

    # Tag this fan-out so a late result from an earlier command is never
    # mistaken for this one (legacy single-file path stays nonce-free).
    nonce = uuid.uuid4().hex
    payload = {**payload, "nonce": nonce}

    # Broadcast protocol: rank 0's request receiver claims ``cmd.0.json`` and
    # injects the payload into the TP broadcast, so every rank executes it.
    # Writing ``cmd.1.json`` (or any other rank's file) would never be read.
    cmd_ranks = [ranks[0]]
    for r in ranks:
        try:
            (d / f"result.{r}.json").unlink()
        except FileNotFoundError:
            pass
    for r in cmd_ranks:
        tmp = d / f"cmd.{r}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, d / f"cmd.{r}.json")

    t0 = time.time()
    results = {}
    pending = set(ranks)
    while pending and time.time() - t0 < timeout:
        for r in list(pending):
            p = d / f"result.{r}.json"
            if not p.exists():
                continue
            try:
                res = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if res.get("nonce") != nonce:
                # Stale response from an earlier command: discard, keep waiting.
                try:
                    p.unlink()
                except OSError:
                    pass
                continue
            results[r] = res
            pending.discard(r)
        if pending:
            time.sleep(0.02)
    if pending:
        raise TimeoutError(
            f"{role} {payload['cmd']} timed out after {timeout}s; "
            f"missing ranks {sorted(pending)}"
        )
    bad = [(r, results[r]) for r in ranks if not results[r].get("ok")]
    if bad:
        raise RuntimeError(
            f"{role} {payload['cmd']} failed on rank(s) {[r for r, _ in bad]}: "
            f"{json.dumps(bad)[:4000]}"
        )
    out = dict(results[ranks[0]])
    out["tp_rank_results"] = [results[r] for r in ranks]
    return out


def status(role: str, timeout: float = 300.0) -> dict:
    return send(role, {"cmd": "status"}, timeout=timeout)


def status_all(role: str) -> list:
    """Per-rank status dicts, sorted by rank (fan-out status)."""
    st = send(role, {"cmd": "status"})
    return st.get("tp_rank_results") or [st]


def wait_all(role: str, rid: str, pred, timeout: float = 300.0):
    """Wait until every rank's row for rid satisfies pred; return (status, rows)."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        st = status(role)
        rank_results = st.get("tp_rank_results") or [st]
        rows = [find(rr, rid) for rr in rank_results]
        if all(row is not None and pred(row) for row in rows):
            return st, rows
        last = rows
        time.sleep(0.02)
    raise TimeoutError(
        f"{role} {rid} all-rank wait timed out; last={json.dumps(last)[:2000]}"
    )


def find(res: dict, rid: str):
    for row in res.get("reqs", []):
        if row["rid"] == rid:
            return row
    return None


def freeze_all(role: str, rid: str, timeout: float = 300.0) -> list:
    """Wait until every TP rank reports the request frozen; return per-rank freeze_results."""
    st, rows = wait_all(role, rid, lambda r: r["frozen"] or r.get("freeze_error"), timeout)
    results = []
    for rr in st.get("tp_rank_results") or []:
        row = find(rr, rid)
        if row is None or not row.get("frozen"):
            raise RuntimeError(
                f"rank {rr.get('tp_rank')} not frozen for {rid}: {row}"
            )
        if row.get("freeze_error"):
            raise RuntimeError(
                f"rank {rr.get('tp_rank')} freeze error for {rid}: {row['freeze_error']}"
            )
        results.append(row["freeze_result"])
    if not results:
        row = find(st, rid)
        results = [row["freeze_result"]]
    return results


def wait_for(role: str, rid: str, pred, timeout: float = 300.0) -> dict:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        res = status(role)
        row = find(res, rid)
        if row is not None:
            last = row
            if pred(row):
                return row
        time.sleep(0.02)
    raise TimeoutError(f"{role} {rid} wait timed out; last={json.dumps(last)[:2000] if last else None}")


def release(role: str, rid: str):
    try:
        send(role, {"cmd": "release", "rid": rid}, timeout=30)
    except Exception as exc:
        print(f"release {role}/{rid} warning: {exc}", flush=True)


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "A distributed system must preserve request state across workers. "
        "Caching, scheduling, and migration interact in non-obvious ways. "
    )
    ids = tok(base, add_special_tokens=False)["input_ids"]
    while len(ids) < INPUT_LEN:
        ids = ids + ids
    ids = ids[:INPUT_LEN]
    print(f"mode={MODE} input_len={INPUT_LEN} total_new={TOTAL_NEW}", flush=True)
    send("dst", {"cmd": "reset"})
    send("src", {"cmd": "reset"})

    # Full local reference for token-level correctness (cached per input length;
    # greedy outputs are deterministic for the same model/input).
    ref_path = STATE / f"p2d_reference_{INPUT_LEN}_{TOTAL_NEW}.json"
    if ref_path.exists():
        ref_out = json.loads(ref_path.read_text(encoding="utf-8"))
    else:
        ref_rid = "p2d-ref"
        send("dst", {
            "cmd": "submit", "rid": ref_rid, "input_ids": ids,
            "max_new_tokens": TOTAL_NEW, "temperature": 0.0,
            "top_p": 1.0, "top_k": 1, "ignore_eos": True,
        })
        ref = wait_for("dst", ref_rid,
                       lambda r: r["finished"] or r["output_len"] >= TOTAL_NEW, 300)
        ref_out = ref["output_ids"]
        release("dst", ref_rid)
        ref_path.write_text(json.dumps(ref_out), encoding="utf-8")

    src_rid, dst_rid = "p2d-src", "p2d-dst"
    t_submit = time.monotonic()
    send("src", {
        "cmd": "submit", "rid": src_rid, "input_ids": ids,
        "max_new_tokens": TOTAL_NEW, "temperature": 0.0,
        "top_p": 1.0, "top_k": 1, "ignore_eos": True,
        "pause_after_prefill": True, "transfer": TRANSFER_MODE,
        "out": str(STATE / f"p2d_state_{RESULT_NAME}_{INPUT_LEN}.pt"),
    })
    # Source auto-freezes at kv_committed_len == prompt_len and output_len == 1.
    src_row = wait_for("src", src_rid,
                       lambda r: r["frozen"] or r.get("freeze_error"), 300)
    if src_row.get("freeze_error"):
        raise RuntimeError(f"source freeze failed: {src_row['freeze_error']}")
    freeze = src_row["freeze_result"]
    t_freeze = float(freeze["t_freeze_monotonic"])
    print(f"prefill/freeze: kv_len={freeze['kv_committed_len']} "
          f"output_len={freeze['output_len']} export={freeze['export_ms']:.1f}ms "
          f"since_submit={t_freeze - t_submit:.3f}s", flush=True)

    t_import = time.monotonic()
    import_payload = {
        "cmd": "import", "rid": dst_rid,
        "remaining": TOTAL_NEW - int(freeze["output_len"]),
    }
    if freeze.get("transfer"):
        import_payload["transfer"] = freeze["transfer"]
    else:
        import_payload["state"] = freeze["state"]
    imp = send("dst", import_payload)
    t_import_done = time.monotonic()
    print(f"import: {json.dumps({k: imp[k] for k in ('imported_kv_tokens','output_len','remaining')})} "
          f"import_wall={t_import_done - t_import:.3f}s", flush=True)

    marks = {}
    final_row = None
    milestone_targets = [t for t in (2, 16, 64, 128, 256, 512) if t < TOTAL_NEW]
    if TOTAL_NEW not in milestone_targets:
        milestone_targets.append(TOTAL_NEW)
    for target in milestone_targets:
        if target <= int(freeze["output_len"]):
            continue
        row = wait_for("dst", dst_rid,
                       lambda r, t=target: r["output_len"] >= t or r["finished"], 300)
        final_row = row
        now = time.monotonic()
        marks[f"y{target}_after_freeze_ms"] = (now - t_freeze) * 1000.0
        marks[f"y{target}_len"] = row["output_len"]
        print(f"y{target}: len={row['output_len']} "
              f"after_freeze={(now - t_freeze)*1000:.1f}ms", flush=True)
        if row["finished"]:
            break

    out_ids = final_row["output_ids"] if final_row else []
    dst_status = status("dst")
    src_status = status("src")
    target_state = dst_status.get("target_state")
    first_div = next(
        (i for i, (a, b) in enumerate(zip(out_ids, ref_out)) if a != b), None
    )
    if first_div is None and len(out_ids) != len(ref_out):
        first_div = min(len(out_ids), len(ref_out))

    result = {
        "mode": MODE,
        "input_len": INPUT_LEN,
        "total_new": TOTAL_NEW,
        "kv_len": freeze["kv_committed_len"],
        "source_output_len": freeze["output_len"],
        "transport": freeze["transport"],
        "trace_tag": freeze.get("trace_tag"),
        "freeze_export_ms": freeze["export_ms"],
        "import": {k: imp.get(k) for k in ("imported_kv_tokens", "output_len", "recv_ms", "load_ms", "remaining")},
        "import_wall_s": t_import_done - t_import,
        "marks": marks,
        "output_len": len(out_ids),
        "reference_len": len(ref_out),
        "full_match": out_ids == ref_out,
        "first_divergence": first_div,
        "target_state": target_state,
        "source_gil": src_status.get("gil_stats"),
    }
    (STATE / RESULT_NAME).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    release("dst", dst_rid)
    release("src", src_rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
