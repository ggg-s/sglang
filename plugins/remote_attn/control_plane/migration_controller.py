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

from p4_dsv_gateway import PRECOMMIT, TERMINAL, DrainConflict, DsvGateway

ACTIVE_MIGRATION_PHASES = PRECOMMIT | {"COMMIT", "ROLLBACK", "TARGET"}
DECODE_STATUS_MAX_AGE_S = 10


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
        autostart=True,
        **kwargs,
    ):
        if natural_drain_ms < 0:
            raise ValueError("natural drain window must be nonnegative")
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
        if not drain or entry["rid"] not in drain["members"]:
            return None
        if time.time() < drain.get("natural_drain_until", 0):
            return None
        if any(
            record["phase"] in ACTIVE_MIGRATION_PHASES
            for record in self.journal["entries"].values()
        ):
            return None
        target = drain["dst"]
        if not self._decode_ready(target):
            raise RuntimeError("destination unavailable before reservation")
        instances = [row["instance_id"] for row in self.facts[target]]
        if instances != drain["instances"][target]:
            raise RuntimeError("destination identity changed before reservation")
        return target

    def _cut_due(self, entry, request):
        # Named drains decide the cut only after a request has the migration slot.
        return entry["phase"] not in {"SUBMIT", "SOURCE"} and super()._cut_due(
            entry, request
        )

    def _source_idle(self, source):
        return self._decode_ready(source) and super()._source_idle(source)

    def _existing_drain(self, src, dst):
        drain = self.journal["drain"]
        if drain:
            status = super().drain_status()
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
            ):
                raise DrainConflict("SMG cordon or Decode identities changed")
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

            drain = dict(
                id=cordon["id"] if cordon else uuid.uuid4().hex,
                active=True,
                mode="d2d",
                src=src,
                dst=dst,
                members=list(members),
                instances=instances,
                started_at=cordon["started_at"] if cordon else time.time(),
            )
            if self.natural_drain_s:
                drain["natural_drain_until"] = time.time() + self.natural_drain_s
            previous = self.journal
            entries = dict(previous["entries"])
            for rid, entry in members.items():
                changes = {"migration_requested": True}
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
                        attempts=0,
                        cut_attempted=False,
                        retry_at=0,
                        cut_observed_at=None,
                        admission_plans_ref=None,
                        migration_outcome=None,
                        frozen_output=[],
                    )
                entries[rid] = {**entry, **changes}
            self.journal = {
                **previous,
                "drain": drain,
                "entries": entries,
                "fenced_workers": {**previous["fenced_workers"], src: drain["id"]},
            }
            if cordon is not None:
                self.journal["native_cordon"] = None
                self.journal["late_routes"] = {
                    **previous.get("late_routes", {}),
                    src: dst,
                }
            try:
                self._persist()
            except BaseException:
                self.journal = previous
                raise
            self.cv.notify_all()
            return self.drain_status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31080)
    parser.add_argument("--admin-port", type=int, default=31081)
    parser.add_argument("--poll-ms", type=float, default=10)
    parser.add_argument(
        "--natural-drain-ms",
        type=int,
        default=0,
        help="Let source requests finish for this long after cordon before planning D2D",
    )
    parser.add_argument("--precopy-window-ms", type=int, default=0)
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
        natural_drain_ms=args.natural_drain_ms,
        precopy_window_ms=args.precopy_window_ms,
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
