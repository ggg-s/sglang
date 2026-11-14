"""Serve a DSV model pool and coordinate operator-directed Decode migration."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from p4_dsv_gateway import (
    ACTIVE_MIGRATION_PHASES,
    CUTOVER_PHASES,
    TERMINAL,
    DrainConflict,
    DsvGateway,
)

DECODE_STATUS_MAX_AGE_S = 10
PREPARING_PHASES = {"PLAN", "RESERVE", "WAIT_CUT"}
MAX_INFLIGHT_MIGRATIONS = 8


@dataclass(frozen=True)
class Prefill:
    url: str
    bootstrap_host: str
    bootstrap_port: int

    def __post_init__(self):
        url = urlsplit(self.url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.path != "/generate"
            or url.query
            or url.fragment
            or not self.bootstrap_host
            or type(self.bootstrap_port) is not int
            or not 0 < self.bootstrap_port < 65536
        ):
            raise ValueError("invalid Prefill URL or bootstrap address")

    def health(self):
        url = urlsplit(self.url)
        health_url = urlunsplit((url.scheme, url.netloc, "/health", "", ""))
        try:
            with urlopen(health_url, timeout=1) as response:
                return response.status == 200
        except OSError:
            return False


class MigrationController(DsvGateway):
    journal_filename = "dsv_migration_journal.json"

    def __init__(
        self,
        driver,
        tokenizer,
        state_dir,
        *,
        prefills,
        decode_workers,
        standby=(),
        natural_drain_ms=0,
        max_concurrent_migrations=1,
        max_preparing_migrations=None,
        async_handoff=True,
        autostart=True,
        **kwargs,
    ):
        if natural_drain_ms < 0:
            raise ValueError("natural drain window must be nonnegative")
        if max_concurrent_migrations not in (1, 2, 4):
            raise ValueError("max_concurrent_migrations must be 1, 2, or 4")
        if max_preparing_migrations is None:
            max_preparing_migrations = max_concurrent_migrations
        if max_preparing_migrations not in range(1, 5):
            raise ValueError("max_preparing_migrations must be between 1 and 4")
        self.max_concurrent_migrations = max_concurrent_migrations
        self.max_preparing_migrations = max_preparing_migrations
        self.max_inflight_migrations = (
            1
            if max_concurrent_migrations == max_preparing_migrations == 1
            else min(
                MAX_INFLIGHT_MIGRATIONS,
                max_concurrent_migrations + max_preparing_migrations,
            )
        )
        self.natural_drain_s = natural_drain_ms / 1000
        self.prefills = dict(prefills)
        self.standby = frozenset(standby)
        if (
            not self.prefills
            or not decode_workers
            or not self.standby <= decode_workers.keys()
            or self.standby == decode_workers.keys()
        ):
            raise ValueError("configure at least one Prefill and one serving Decode")
        if len(set(decode_workers.values())) != len(decode_workers):
            raise ValueError("Decode instances must have distinct control directories")
        self.prefill_facts = {}
        self.admission_cursor = 0
        self.routing_thread = None
        super().__init__(
            driver,
            tokenizer,
            state_dir,
            prefill_url=None,
            decode_workers=dict(decode_workers),
            async_handoff=async_handoff,
            autostart=autostart,
            **kwargs,
        )
        if autostart:
            self.routing_thread = threading.Thread(
                target=self._poll_routing, daemon=True
            )
            self.routing_thread.start()

    def close(self):
        self.stop.set()
        if self.routing_thread is not None:
            self.routing_thread.join(timeout=5)
            if self.routing_thread.is_alive():
                raise RuntimeError(
                    "routing poll is still active; keep the journal lock held"
                )
        super().close()

    def _validate_worker_bindings(self):
        super()._validate_worker_bindings()
        if self.journal.get("native_pd", False) != self.native_pd:
            if self.path.exists():
                raise ValueError("migration journal ingress mode changed")
            self.journal["native_pd"] = self.native_pd

    native_pd = False

    def _poll_routing(self):
        self._poll_prefills()

    def _poll_prefills(self):
        while not self.stop.is_set():
            for name, prefill in self.prefills.items():
                if self.stop.is_set():
                    return
                if prefill.health():
                    self.prefill_facts[name] = time.monotonic()
                else:
                    self.prefill_facts.pop(name, None)
            self.stop.wait(1)

    def _available_prefills(self):
        earliest = time.monotonic() - max(3, 2 * len(self.prefills))
        return [
            name
            for name in self.prefills
            if self.prefill_facts.get(name, float("-inf")) >= earliest
        ]

    def _decode_ready(self, worker):
        return (
            worker not in self.fact_errors
            and bool(self.facts.get(worker))
            and time.monotonic() - self.facts_at.get(worker, float("-inf"))
            < DECODE_STATUS_MAX_AGE_S
        )

    def _available_decodes(self):
        fenced = self.journal["fenced_workers"]
        return [
            worker
            for worker in self.decode_workers
            if worker not in fenced
            and worker not in self.standby
            and self._decode_ready(worker)
        ]

    def ready(self):
        return bool(self._available_prefills() and self._available_decodes())

    def _admission(self, role):
        prefills, decodes = self._available_prefills(), self._available_decodes()
        if not prefills or not decodes:
            raise RuntimeError("model pool has no ready Prefill/Decode pair")
        # Rotate the Prefill offset after each Decode round so equally sized
        # pools still use every combination.
        round_index, decode_index = divmod(self.admission_cursor, len(decodes))
        prefill = self.prefills[prefills[(round_index + decode_index) % len(prefills)]]
        decode = decodes[decode_index]
        self.admission_cursor += 1
        return dict(
            owner="src",
            initial_owner="src",
            workers={"src": decode},
            instances={"src": [row["instance_id"] for row in self.facts[decode]]},
            prefill_url=prefill.url,
            bootstrap_host=prefill.bootstrap_host,
            bootstrap_port=prefill.bootstrap_port,
        )

    def _migration_target(self, entry):
        drain = self.journal["drain"]
        if (
            not drain
            or not drain["active"]
            or drain.get("abort_requested_at") is not None
            or entry["rid"] not in drain["members"]
        ):
            return None
        if time.time() < drain.get("natural_drain_until", 0):
            return None
        phases = [self.journal["entries"][rid]["phase"] for rid in drain["members"]]
        if (
            sum(phase in PREPARING_PHASES for phase in phases)
            >= self.max_preparing_migrations
            or sum(phase in ACTIVE_MIGRATION_PHASES for phase in phases)
            >= self.max_inflight_migrations
        ):
            return None
        target = drain["dst"]
        if not self._decode_ready(target):
            raise RuntimeError("destination unavailable before reservation")
        instances = [row["instance_id"] for row in self.facts[target]]
        if instances != drain["instances"][target]:
            raise RuntimeError("destination identity changed before reservation")
        return target

    def _begin(self, entry):
        # Serialize acquiring the migration slot with aborting the drain.
        with self.cv:
            super()._begin(entry)

    def _cut_due(self, entry, request):
        # Named drains decide the cut only after a request has the migration slot.
        return entry["phase"] not in {"SUBMIT", "SOURCE"} and super()._cut_due(
            entry, request
        )

    def _active_cutovers(self, drain):
        entries = self.journal["entries"]
        # Rollback can still hold a frozen source until all ranks resume.
        return sum(
            entries[rid]["phase"] in CUTOVER_PHASES
            or (
                entries[rid]["phase"] == "ROLLBACK"
                and entries[rid].get("freeze_requested_at") is not None
            )
            for rid in drain["members"]
        )

    def _can_freeze(self, entry):
        drain = self.journal["drain"]
        return bool(
            drain
            and drain["active"]
            and drain.get("abort_requested_at") is None
            and entry["rid"] in drain["members"]
            and self._active_cutovers(drain) < self.max_concurrent_migrations
        )

    def drain_status(self):
        with self.cv:
            status = super().drain_status()
            drain = self.journal["drain"]
            return {
                **status,
                "active_cutovers": self._active_cutovers(drain) if drain else 0,
                "controller_features": {
                    "event_driven_reconcile": self.event_driven_reconcile,
                    "worker_status_snapshots": self.worker_status_snapshots,
                    "chain_drain": self.chain_drain,
                },
                "migration_limits": {
                    "preparing": self.max_preparing_migrations,
                    "cutover": self.max_concurrent_migrations,
                    "inflight": self.max_inflight_migrations,
                },
                "worker_status_reads": {
                    worker: reads.copy()
                    for worker, reads in self.worker_status_reads.items()
                },
                "worker_snapshot_sample_ms": {
                    worker: sample.copy()
                    for worker, sample in self.worker_snapshot_sample_ms.items()
                },
                "control_wave_sizes": {
                    worker: sizes.copy()
                    for worker, sizes in self.control_wave_sizes.items()
                },
            }

    def _source_idle(self, source):
        return self._decode_ready(source) and super()._source_idle(source)

    def _existing_drain(self, src, dst):
        drain = self.journal["drain"]
        if drain:
            status = super().drain_status()
            if status["state"] == "ABORTED":
                return None
            if status["state"] == "ABORTING":
                raise DrainConflict("the previous migration is still cleaning up")
            if (drain["src"], drain["dst"]) == (src, dst):
                return status
            if status["state"] != "SAFE_TO_STOP":
                raise DrainConflict("a migration is already active in this model pool")
        return None

    def _validate_drain_pair(self, src, dst):
        if (
            not all(
                isinstance(worker, str) and worker in self.decode_workers
                for worker in (src, dst)
            )
            or src == dst
        ):
            raise ValueError("src and dst must be distinct registered Decode names")

    def _drain_instances(self, src, dst):
        if (
            src in self.journal["fenced_workers"]
            or dst in self.journal["fenced_workers"]
        ):
            raise DrainConflict("src or dst is already fenced")
        if not self._decode_ready(src) or not self._decode_ready(dst):
            raise RuntimeError("src or dst is unavailable")
        instances = {
            worker: [r["instance_id"] for r in self.facts[worker]]
            for worker in (src, dst)
        }
        if len(instances[src]) != len(instances[dst]) or set(instances[src]) & set(
            instances[dst]
        ):
            raise DrainConflict("src/dst TP sizes or instance identities conflict")
        return instances

    def start_named_drain(self, src, dst):
        self._validate_drain_pair(src, dst)
        return self._activate_drain(src, dst)

    def _publish_journal(self, **changes):
        journal = {**self.journal, **changes}
        self._persist(journal)
        self.journal = journal
        if "drain" in changes and changes["drain"]:
            for rid in changes["drain"].get("members", ()):
                self._queue_reconcile(rid, "drain")
        self.cv.notify_all()

    def abort_named_drain(self, drain_id):
        with self.cv:
            drain = self.journal["drain"]
            if not drain or drain["id"] != drain_id:
                raise DrainConflict("drain id does not match the current drain")
            if drain.get("abort_requested_at") is not None:
                return self.drain_status()
            source = drain["src"]
            fenced = self.journal["fenced_workers"].copy()
            if not drain["active"] or fenced.get(source) != drain_id:
                raise DrainConflict("drain is already complete")
            del fenced[source]
            entries = self.journal["entries"].copy()
            for rid in drain["members"]:
                entries[rid] = {**entries[rid], "migration_requested": False}
            changes = dict(
                drain={
                    **drain,
                    "abort_requested_at": time.time(),
                    "abort_reason": "operator_abort",
                },
                entries=entries,
                fenced_workers=fenced,
            )
            if "late_routes" in self.journal:
                routes = self.journal["late_routes"].copy()
                routes.pop(source, None)
                changes["late_routes"] = routes
            # Requests already bound to an engine keep their owner and epoch.
            self._publish_journal(**changes)
            return self.drain_status()

    def _activate_drain(self, src, dst, *, cordon=None):
        with self.cv:
            existing = self._existing_drain(src, dst)
            if existing:
                return existing
        # Rebase a LOCAL request only at a boundary between reconciliation ticks.
        with self.work_lock, self.cv:
            existing = self._existing_drain(src, dst)
            if existing:
                return existing
            instances = self._drain_instances(src, dst)
            if cordon is not None and (
                (self.journal.get("native_cordon") or {}).get("id") != cordon["id"]
                or instances != cordon["instances"]
                or time.time() >= cordon["cordon_deadline_at"]
            ):
                raise DrainConflict("SMG cordon expired or Decode identities changed")
            members = {
                rid: entry
                for rid, entry in self.journal["entries"].items()
                if entry["phase"] not in TERMINAL
                and self._worker(entry, entry["owner"]) == src
            }
            if any(
                entry["instances"][entry["owner"]] != instances[src]
                for entry in members.values()
            ):
                raise DrainConflict(
                    "a live request is still bound to a previous source instance"
                )

            activated_at = time.time()
            drain = dict(
                id=cordon["id"] if cordon else uuid.uuid4().hex,
                active=True,
                mode="d2d",
                src=src,
                dst=dst,
                members=list(members),
                instances=instances,
                started_at=cordon["started_at"] if cordon else activated_at,
                activated_at=activated_at,
            )
            if self.natural_drain_s:
                drain["natural_drain_until"] = activated_at + self.natural_drain_s
            previous = self.journal
            entries = dict(previous["entries"])
            for rid, entry in members.items():
                changes = {
                    "migration_requested": True,
                    "attempts": 0,
                    "cut_attempted": False,
                    "retry_at": 0,
                    "cut_observed_at": None,
                }
                if entry["phase"] == "LOCAL":
                    # D1 has the full generated prefix and its native request
                    # row. Preserve both the public rid and the SSE cursor.
                    changes.update(
                        phase="SOURCE",
                        owner="src",
                        initial_owner="src",
                        workers={"src": src},
                        instances={"src": instances[src]},
                        source_rid=entry["target_rid"],
                        target_rid="p4d-" + uuid.uuid4().hex,
                        epoch=None,
                        admission_plans_ref=None,
                        migration_outcome=None,
                        frozen_output=[],
                    )
                entries[rid] = {**entry, **changes}
            changes = {
                "drain": drain,
                "entries": entries,
                "fenced_workers": {**previous["fenced_workers"], src: drain["id"]},
            }
            if cordon is not None:
                changes["native_cordon"] = None
                changes["late_routes"] = {
                    **previous.get("late_routes", {}),
                    src: dst,
                }
            self._publish_journal(**changes)
            return self.drain_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31080)
    parser.add_argument("--admin-port", type=int, default=31081)
    parser.add_argument("--poll-ms", type=float, default=10)
    parser.add_argument(
        "--event-driven-reconcile",
        action="store_true",
        help="Advance affected requests on durable transitions and worker snapshots",
    )
    parser.add_argument(
        "--worker-status-snapshots",
        action="store_true",
        help="Read advisory TP status snapshots with status RPC fallback",
    )
    parser.add_argument(
        "--natural-drain-ms",
        type=int,
        default=0,
        help="Let source requests finish for this long after cordon before planning D2D",
    )
    parser.add_argument(
        "--chain-drain",
        action="store_true",
        help="Advance adjacent D2D drain stages in the same reconcile pass",
    )
    parser.add_argument("--precopy-window-ms", type=int, default=0)
    parser.add_argument(
        "--packed-precopy",
        action="store_true",
        help="Pack closed KV ranges into one RDMA frame when all ranks support it",
    )
    parser.add_argument(
        "--max-concurrent-migrations",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Max simultaneous Freeze-to-admission cutovers",
    )
    parser.add_argument(
        "--max-preparing-migrations",
        type=int,
        choices=range(1, 5),
        help="Max requests in PLAN, RESERVE or Pre-copy (defaults to cutover limit)",
    )
    parser.add_argument("--out", type=Path, help="Optional per-request handoff JSONL")
    args = parser.parse_args()
    if min(args.natural_drain_ms, args.precopy_window_ms) < 0:
        parser.error("drain and precopy windows must be nonnegative")
    try:
        config = json.loads(args.config.read_text())
        prefills = {
            name: Prefill(**values) for name, values in config["prefill"].items()
        }
        decode_workers = {
            name: Path(directory).resolve()
            for name, directory in config["decode"].items()
        }
        model_path = config["model_path"]
        state_dir = Path(config["state_dir"])
        model = config["served_model_name"]
        if not isinstance(model, str) or not model:
            raise ValueError("served_model_name must be nonempty")
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))

    from transformers import AutoTokenizer

    import p2d_driver
    from migration_http import MigrationAdminHandler, MigrationWorkerHandler

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not getattr(tokenizer, "backend_tokenizer", None):
        parser.error("a fast tokenizer is required")

    controller_type = MigrationController
    native_options = config.get("native_pd")
    if native_options is not None:
        from migration_native_pd import NativePdController, serve_native_pd

        controller_type = NativePdController
    controller = controller_type(
        p2d_driver,
        tokenizer,
        state_dir,
        prefills=prefills,
        decode_workers=decode_workers,
        standby=config.get("standby", ()),
        poll_ms=args.poll_ms,
        event_driven_reconcile=args.event_driven_reconcile,
        worker_status_snapshots=args.worker_status_snapshots,
        chain_drain=args.chain_drain,
        natural_drain_ms=args.natural_drain_ms,
        precopy_window_ms=args.precopy_window_ms,
        packed_precopy=args.packed_precopy,
        max_concurrent_migrations=args.max_concurrent_migrations,
        max_preparing_migrations=args.max_preparing_migrations,
        out=args.out,
        **(native_options or {}),
    )
    try:
        MigrationWorkerHandler.controller = MigrationAdminHandler.controller = (
            controller
        )
        MigrationWorkerHandler.model = model
        with ThreadingHTTPServer(
            ("127.0.0.1", args.admin_port), MigrationAdminHandler
        ) as admin:
            threading.Thread(target=admin.serve_forever, daemon=True).start()
            try:
                if native_options is not None:
                    serve_native_pd(controller, model, args.host)
                else:
                    with ThreadingHTTPServer(
                        (args.host, args.port), MigrationWorkerHandler
                    ) as worker:
                        worker.serve_forever()
            finally:
                admin.shutdown()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
