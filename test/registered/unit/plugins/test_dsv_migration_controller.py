"""Pool ownership and SSE through the real controller with simulated worker RPCs."""

import asyncio
import copy
import http.client
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback
from tokenizers.models import BPE

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CONTROL = Path(__file__).resolve().parents[4] / "plugins/remote_attn/control_plane"


class WorkerTransport:
    def __init__(self):
        self.requests = {worker: {} for worker in ("D0", "D1", "D2")}
        self.retired = {}
        self.moves = {}
        self.migrations = {worker: {} for worker in self.requests}
        self.commands = []
        self.submissions = []
        self.auto_release = True
        self.precopy = False
        self.waves = []
        self.wave_supported = False
        self.import_ready = True

    def send_to(self, directory, worker, payload, timeout):
        assert directory.name == worker
        cmd, epoch = payload["cmd"], payload["migration_epoch"]
        requests = self.requests[worker]
        self.commands.append((worker, cmd, epoch))
        if cmd == "d2d_worker_status":
            if "request_rid" in payload:
                assert payload["expected_instances"] == [worker]
                rid = payload["request_rid"]
                return dict(
                    request=self.retired.get((worker, epoch, rid))
                    or (requests.get(rid) if payload.get("include_live") else None)
                )
            return dict(
                protocol="dsv-d2d-v2",
                rank=0,
                tp_size=1,
                instance_id=worker,
                idle=all(row["kv_released"] for row in requests.values()),
                requests=copy.deepcopy(requests),
                migrations=copy.deepcopy(self.migrations[worker]),
                terminal_request_lookup_supported=True,
                precopy_protocol="closed-page-precopy-v1" if self.precopy else None,
                async_handoff_supported=self.wave_supported,
                control_wave_supported=self.wave_supported,
            )
        if cmd == "d2d_control_wave":
            assert self.wave_supported and payload["expected_instances"] == [worker]
            commands = payload["commands"]
            self.waves.append((worker, copy.deepcopy(commands)))
            return {
                "results": {
                    command["migration_epoch"]: self.send_to(
                        directory, worker, command, timeout
                    )
                    for command in commands
                }
            }
        if cmd == "d2d_submit_pd":
            requests[payload["rid"]] = self.request([1, 2])
            self.submissions.append((worker, payload["prefill_url"]))
            return {}
        if cmd == "d2d_plan":
            return {
                "plan": {
                    "lower": 4,
                    "upper": 12,
                    "output_ids": requests[payload["rid"]]["output_ids"],
                    "schema": {"page_size": 4, "entries": [["full", 128]]},
                }
            }
        if cmd == "d2d_reserve":
            return {"reserved": True, "reservation": {}}
        if cmd == "d2d_precopy_status":
            return {"available": [1]}
        if cmd == "d2d_precopy_prepare":
            return {"receive": {}}
        if cmd == "d2d_precopy_copy":
            return {"proof": {}}
        if cmd == "d2d_precopy_apply":
            return {"prefix_pages": [1], "receipt": {"bytes": 128}}
        if cmd == "d2d_freeze":
            transfer = dict(
                output_ids=list(requests[payload["rid"]]["output_ids"]),
                kv_len=4,
                source_rid=payload["rid"],
                target_rid=payload["target_rid"],
                source_worker=worker,
            )
            return {"transfer": transfer}
        if cmd == "d2d_import":
            if payload.get("async_handoff") and not self.import_ready:
                return {"import_pending": True}
            self.moves[epoch] = payload["transfers"][0]
            return {"imported": True}
        if cmd == "d2d_commit":
            move = self.moves[epoch]
            suffix = [3, 4] if worker == "D1" else [5]
            requests[move["target_rid"]] = self.request(
                move["output_ids"] + suffix, finished=worker == "D2"
            )
            move["target_worker"] = worker
            if self.auto_release:
                self.release(epoch)
            return {"admitted": True, "source_confirmed": True}
        raise AssertionError(cmd)

    def release(self, epoch):
        move = self.moves[epoch]
        src, dst = move["source_worker"], move["target_worker"]
        self.requests[src][move["source_rid"]]["kv_released"] = True
        self.migrations[src][epoch] = {
            "source": {"released": True, "kv_released": True}
        }
        self.migrations[dst][epoch] = {"target": {"local": True}}

    @staticmethod
    def request(tokens, finished=False):
        return dict(
            output_ids=tokens,
            finished=finished,
            kv_released=finished,
            finish_type="stop" if finished else None,
            plan_ready=True,
            migration_ready=True,
        )


class TextTokenizer:
    def __init__(self):
        self.backend_tokenizer = Tokenizer(
            BPE(
                vocab={"a": 0, "A": 1, "<0xE4>": 2, "<0xB8>": 3, "<0xAD>": 4, "!": 5},
                merges=[],
            )
        )
        self.backend_tokenizer.decoder = ByteFallback()

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size()


class TestDsvMigrationController(CustomTestCase):
    def setUp(self):
        sys.path.insert(0, str(CONTROL))
        self.addCleanup(sys.path.remove, str(CONTROL))
        from migration_controller import MigrationController, Prefill

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = WorkerTransport()
        self.controller_type = MigrationController
        self.config = dict(
            prefills={
                name: Prefill(f"http://{name}:31016/generate", name, 35409)
                for name in ("p0", "p1")
            },
            decode_workers={
                worker: self.root / worker for worker in self.driver.requests
            },
            autostart=False,
        )
        self.controller = self.open_controller()

    def open_controller(self, **changes):
        controller = self.controller_type(
            self.driver,
            TextTokenizer(),
            self.root / "journal",
            **(self.config | changes),
        )
        self.addCleanup(controller.close)
        controller.prefill_facts.update(
            dict.fromkeys(controller.prefills, time.monotonic())
        )
        controller.tick()
        return controller

    def server(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_port

    def http(self, port, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
        self.addCleanup(conn.close)
        if body is None:
            conn.request("GET", path, headers={"Connection": "close"})
        else:
            conn.request(
                "POST", path, json.dumps(body), {"Content-Type": "application/json"}
            )
        return conn.getresponse()

    def ticks(self, count=12):
        for _ in range(count):
            self.controller.tick()

    @staticmethod
    def frame(response):
        while line := response.readline():
            if line.startswith(b"data: "):
                return line.removeprefix(b"data: ").strip()
        raise AssertionError("stream ended before next frame")

    def test_same_sse_survives_two_drains_and_admission_skips_each_source(self):
        from migration_http import MigrationAdminHandler, MigrationWorkerHandler

        class Worker(MigrationWorkerHandler):
            controller = self.controller
            model = "test-model"

        class Admin(MigrationAdminHandler):
            controller = self.controller

        worker, admin = self.server(Worker), self.server(Admin)
        response = self.http(
            worker,
            "/v1/completions",
            {"model": "test-model", "prompt": [0], "stream": True, "max_tokens": 12},
        )
        self.assertEqual(response.status, 200)
        self.ticks(2)
        first = json.loads(self.frame(response))
        self.assertEqual(first["choices"][0]["text"], "A")

        first_drain = {"src": "D0", "dst": "D1"}
        self.assertEqual(self.http(worker, "/v1/drain", first_drain).status, 404)
        accepted = self.http(admin, "/v1/drain", first_drain)
        self.assertEqual(accepted.status, 202)
        drain_id = json.load(accepted)["id"]
        self.assertEqual(self.http(worker, "/ready").status, 200)
        self.assertEqual(
            self.http(admin, "/v1/drain", {"src": "D1", "dst": "D2"}).status, 409
        )
        self.assertEqual(
            json.load(self.http(admin, "/v1/drain", first_drain))["id"], drain_id
        )

        other = self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        self.assertEqual(self.controller.entry(other)["workers"]["src"], "D2")
        self.ticks()
        second = json.loads(self.frame(response))
        self.assertEqual(second["choices"][0]["text"], "中")
        self.assertEqual(second["id"], first["id"])
        status = self.controller.drain_status()
        self.assertEqual(status["state"], "SAFE_TO_STOP")
        self.assertFalse(status["active"])
        member = status["members"][0]
        self.assertEqual(status["members_state"][member]["owner"], "D1")
        self.assertEqual(
            self.driver.submissions,
            [("D0", "http://p0:31016/generate"), ("D2", "http://p1:31016/generate")],
        )

        accepted = self.http(admin, "/v1/drain", {"src": "D1", "dst": "D2"})
        self.assertEqual(accepted.status, 202)
        self.assertEqual(json.load(accepted)["members"], [member])
        self.ticks()
        last = json.loads(self.frame(response))
        self.assertEqual(last["choices"][0]["text"], "!")
        self.assertEqual(last["id"], first["id"])
        self.assertEqual(
            json.loads(self.frame(response))["choices"][0]["finish_reason"], "stop"
        )
        self.assertEqual(self.frame(response), b"[DONE]")
        self.assertEqual(self.controller.drain_status()["state"], "SAFE_TO_STOP")
        self.assertEqual(self.controller.entry(member)["workers"]["dst"], "D2")
        self.assertEqual(self.controller.entry(other)["workers"]["src"], "D2")

    def test_three_prefills_and_three_decodes_use_all_pairs(self):
        from migration_controller import Prefill

        self.controller.close()
        prefills = {
            **self.config["prefills"],
            "p2": Prefill("http://p2:31016/generate", "p2", 35409),
        }
        self.controller = self.open_controller(prefills=prefills)
        for _ in range(9):
            self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        self.ticks(2)
        expected = {
            (decode, prefill.url)
            for decode in self.config["decode_workers"]
            for prefill in prefills.values()
        }
        self.assertCountEqual(self.driver.submissions, expected)

    def test_next_request_cannot_start_until_previous_source_retires(self):
        self.driver.auto_release = False
        self.driver.wave_supported = True
        ids = [
            self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
            for _ in range(4)
        ]
        self.ticks(2)
        self.controller.start_named_drain("D0", "D1")
        self.ticks()
        active = [
            self.controller.entry(rid)
            for rid in (ids[0], ids[3])
            if self.controller.entry(rid)["epoch"]
        ]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["phase"], "TARGET")
        self.driver.release(active[0]["epoch"])
        self.ticks()
        self.assertEqual(
            sorted(self.controller.entry(rid)["phase"] for rid in (ids[0], ids[3])),
            ["LOCAL", "TARGET"],
        )

    def test_chained_drain_keeps_prefill_member_without_holding_a_slot(self):
        self.controller.close()
        self.driver.wave_supported = True
        self.controller = self.open_controller(standby=("D1", "D2"), chain_drain=True)
        controller = self.controller
        prefill, decoding = [
            controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
            for _ in range(2)
        ]
        self.ticks(2)
        request = controller.entry(prefill)["source_rid"]
        self.driver.requests["D0"][request] = self.driver.request([])
        controller.start_named_drain("D0", "D1")
        with patch("p4_dsv_gateway.time.monotonic", return_value=time.monotonic()):
            controller.tick()
        self.assertEqual(controller.entry(prefill)["phase"], "SOURCE")
        self.assertIsNone(controller.entry(prefill)["epoch"])
        self.assertEqual(controller.entry(decoding)["phase"], "TARGET")
        self.assertCountEqual(controller.drain_status()["members"], [prefill, decoding])
        self.assertNotEqual(controller.drain_status()["state"], "SAFE_TO_STOP")

        self.driver.requests["D0"][request] = self.driver.request([1], finished=True)
        self.ticks(2)
        self.assertEqual(controller.drain_status()["state"], "SAFE_TO_STOP")
        self.assertEqual(controller.entry(prefill)["final_output_ids"], [1])

    def test_named_workers_batch_four_async_cutovers(self):
        self.controller.close()
        self.driver.wave_supported = True
        self.controller = self.open_controller(
            standby=("D0", "D2"), max_concurrent_migrations=4
        )
        rids = [
            self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
            for _ in range(4)
        ]
        self.ticks(2)
        self.controller.start_named_drain("D1", "D2")
        self.ticks()
        status = self.controller.drain_status()
        self.assertEqual(status["state"], "SAFE_TO_STOP")
        self.assertCountEqual(status["members"], rids)
        self.assertEqual(
            status["migration_limits"], {"preparing": 4, "cutover": 4, "inflight": 8}
        )
        self.assertTrue(
            all(
                member["owner"] == "D2" and member["migration_outcome"] == "migrated"
                for member in status["members_state"].values()
            )
        )
        for worker, command in (
            ("D2", "d2d_reserve"),
            ("D1", "d2d_freeze"),
            ("D2", "d2d_import"),
            ("D2", "d2d_commit"),
        ):
            wave = next(
                commands
                for name, commands in self.driver.waves
                if name == worker and {item["cmd"] for item in commands} == {command}
            )
            self.assertEqual(len(wave), 4)
            if command == "d2d_import":
                self.assertTrue(
                    all(item["async_handoff"] and item["defer_bulk"] for item in wave)
                )

    def test_natural_finish_before_reserve_never_contacts_destination(self):
        """Completion in PLAN or before first RESERVE needs only source cleanup."""
        for phase in ("PLAN", "RESERVE"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as root:
                controller = self.controller_type(
                    self.driver,
                    TextTokenizer(),
                    Path(root),
                    **(self.config | dict(standby=("D1", "D2"))),
                )
                self.addCleanup(controller.close)
                self.driver.requests["D0"].clear()
                controller.tick()
                controller.prefill_facts.update(
                    dict.fromkeys(controller.prefills, time.monotonic())
                )
                rid = controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
                controller.tick()
                controller.tick()
                controller.start_named_drain("D0", "D1")
                controller.tick()
                if phase == "RESERVE":
                    controller.tick()
                entry = controller.entry(rid)
                self.assertEqual(entry["phase"], phase)
                finished = self.driver.request([1, 2, 3], finished=True)
                if phase == "PLAN":
                    finished["kv_released"] = False
                self.driver.requests["D0"][entry["source_rid"]] = finished
                send_to = self.driver.send_to

                def resume(directory, worker, payload, timeout):
                    if payload["cmd"] == "d2d_resume":
                        self.assertEqual(worker, "D0")
                        self.driver.commands.append(
                            (worker, payload["cmd"], payload["migration_epoch"])
                        )
                        return {"resumed": True}
                    return send_to(directory, worker, payload, timeout)

                with patch.object(self.driver, "send_to", side_effect=resume):
                    controller.tick()
                    if phase == "PLAN":
                        self.assertEqual(controller.entry(rid)["phase"], "SOURCE")
                        self.assertNotEqual(controller.drain_status()["state"], "SAFE_TO_STOP")
                        finished["kv_released"] = True
                    controller.tick()
                self.assertEqual(controller.entry(rid)["phase"], "DONE")
                self.assertEqual(controller.entry(rid)["final_output_ids"], [1, 2, 3])
                self.assertEqual(
                    [
                        cmd for worker, cmd, epoch in self.driver.commands
                        if worker == "D1" and epoch == entry["epoch"]
                    ],
                    [],
                )
                self.assertIn(("D0", "d2d_resume", entry["epoch"]), self.driver.commands)
                self.assertEqual(controller.drain_status()["state"], "SAFE_TO_STOP")
                controller.close()

    def test_natural_finish_after_lost_reserve_reply_waits_for_cleanup(self):
        """A recovered RESERVE may still own a destination row after source EOS."""
        self.controller.close()
        self.controller = self.open_controller(standby=("D1", "D2"))
        controller = self.controller
        rid = controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        self.ticks(2)
        controller.start_named_drain("D0", "D1")
        self.ticks(2)
        entry = controller.entry(rid)
        self.assertEqual(entry["phase"], "RESERVE")
        epoch = entry["epoch"]
        reserved = False
        allow_abort = False
        send_to = self.driver.send_to

        def drop_reserve_reply(directory, worker, payload, timeout):
            nonlocal reserved
            cmd = payload["cmd"]
            if payload["migration_epoch"] == epoch:
                if cmd == "d2d_reserve":
                    self.assertFalse(reserved, "finished request retried RESERVE")
                    reserved = True
                    raise TimeoutError("lost RESERVE reply")
                if cmd == "d2d_precopy_close":
                    return {"closed": True}
                if cmd == "d2d_abort":
                    if allow_abort:
                        reserved = False
                    return {"aborted": allow_abort}
                if cmd == "d2d_resume":
                    self.assertFalse(reserved)
                    return {"resumed": True}
            return send_to(directory, worker, payload, timeout)

        with patch.object(self.driver, "send_to", side_effect=drop_reserve_reply):
            self.ticks(1)
            self.assertTrue(reserved)
            self.assertEqual(controller.entry(rid)["phase"], "RESERVE")
            self.driver.requests["D0"][entry["source_rid"]] = self.driver.request(
                [1, 2, 3], finished=True
            )
            controller.close()
            self.controller = self.open_controller(standby=("D1", "D2"))
            self.ticks(2)
            self.assertTrue(reserved)
            self.assertEqual(self.controller.entry(rid)["phase"], "ROLLBACK")
            self.assertNotEqual(self.controller.drain_status()["state"], "SAFE_TO_STOP")
            allow_abort = True
            self.ticks(2)
        self.assertFalse(reserved)
        self.assertEqual(self.controller.entry(rid)["final_output_ids"], [1, 2, 3])
        self.assertEqual(self.controller.drain_status()["state"], "SAFE_TO_STOP")

    def test_retired_source_rolls_back_reservation_before_publishing_terminal(self):
        """EOS between polls leaves a destination reservation to clean up."""
        self.controller.close()
        self.controller = self.open_controller(standby=("D1", "D2"))
        controller = self.controller
        rid = controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        self.ticks(2)
        controller.start_named_drain("D0", "D1")
        self.ticks(3)
        entry = controller.entry(rid)
        self.assertEqual(entry["phase"], "WAIT_CUT")
        move_epoch = entry["epoch"]
        source_rid = entry["source_rid"]
        # An advisory status omission must not discard a live reservation.
        request = self.driver.requests["D0"][source_rid]
        request["output_ids"] = []
        send_to = self.driver.send_to

        def omitted_from_status(directory, worker, payload, timeout):
            reply = send_to(directory, worker, payload, timeout)
            if worker == "D0" and "requests" in reply:
                reply["requests"].pop(source_rid, None)
            return reply

        with patch.object(self.driver, "send_to", side_effect=omitted_from_status):
            self.ticks(1)
        self.assertEqual(controller.entry(rid)["phase"], "WAIT_CUT")
        self.driver.requests["D0"].pop(source_rid)
        self.driver.retired["D0", entry["submission_epoch"], source_rid] = (
            self.driver.request([1, 2, 3, 4], finished=True)
        )
        send_to = self.driver.send_to
        abort_acknowledged = False
        reserved = True
        cleanup = []

        def send(directory, worker, payload, timeout):
            nonlocal reserved
            if payload["migration_epoch"] == move_epoch:
                cmd = payload["cmd"]
                if cmd == "d2d_precopy_close":
                    self.assertEqual(worker, "D0")
                    return {"closed": True}
                if cmd == "d2d_abort":
                    self.assertEqual(worker, "D1")
                    self.assertEqual(payload["precopy_close"], [{"closed": True}])
                    if abort_acknowledged:
                        reserved = False
                        cleanup.append(cmd)
                    return {"aborted": abort_acknowledged}
                if cmd == "d2d_resume":
                    self.assertEqual(worker, "D0")
                    self.assertFalse(reserved)
                    cleanup.append(cmd)
                    return {"resumed": True}
            return send_to(directory, worker, payload, timeout)

        with patch.object(self.driver, "send_to", side_effect=send):
            self.ticks(3)
            self.assertTrue(reserved)
            self.assertEqual(controller.entry(rid)["phase"], "ROLLBACK")
            self.assertNotEqual(controller.drain_status()["state"], "SAFE_TO_STOP")
            abort_acknowledged = True
            self.ticks(3)

        self.assertEqual(cleanup, ["d2d_abort", "d2d_resume"])
        self.assertFalse(reserved)
        self.assertEqual(controller.drain_status()["state"], "SAFE_TO_STOP")
        self.assertEqual(controller.entry(rid)["final_output_ids"], [1, 2, 3, 4])
        events, finished = controller.events(rid, after=1)
        self.assertTrue(finished)
        self.assertEqual([event["token_id"] for event in events], [3, 4])
        self.assertEqual(controller.finish_reason(rid, 4), "stop")

    def test_wait_cut_without_first_token_rolls_back_and_frees_slot(self):
        self.controller.close()
        self.controller = self.open_controller(standby=("D1", "D2"))
        controller = self.controller
        rid, other = [
            controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
            for _ in range(2)
        ]
        self.ticks(2)
        controller.start_named_drain("D0", "D1")
        self.ticks(3)
        # Simulate a persisted WAIT_CUT from the older prefill admission path.
        entry, other = next(
            (controller.entry(candidate), alternate)
            for candidate, alternate in ((rid, other), (other, rid))
            if controller.entry(candidate)["phase"] == "WAIT_CUT"
        )
        self.driver.requests["D0"][entry["source_rid"]] = self.driver.request([])
        controller._update(entry["rid"], wait_cut_started_at=time.time() - 6)
        send_to = self.driver.send_to
        cleanup = []

        def send(directory, worker, payload, timeout):
            cmd = payload["cmd"]
            if payload["migration_epoch"] == entry["epoch"] and cmd in {
                "d2d_precopy_close",
                "d2d_abort",
                "d2d_resume",
            }:
                cleanup.append(cmd)
                return {
                    "d2d_precopy_close": {"closed": True},
                    "d2d_abort": {"aborted": True},
                    "d2d_resume": {"resumed": True},
                }[cmd]
            return send_to(directory, worker, payload, timeout)

        with patch.object(self.driver, "send_to", side_effect=send):
            self.ticks(12)
        self.assertEqual(cleanup, ["d2d_precopy_close", "d2d_abort", "d2d_resume"])
        self.assertEqual(controller.entry(entry["rid"])["phase"], "SOURCE")
        self.assertEqual(controller.entry(other)["migration_outcome"], "migrated")
        self.assertNotEqual(controller.drain_status()["state"], "SAFE_TO_STOP")

    def test_finish_at_freeze_holds_slot_until_rollback_cleans_destination(self):
        """EOS after the status poll must finish on source before the next move."""
        self.controller.close()
        self.controller = self.open_controller(standby=("D1", "D2"))
        rids = [
            self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
            for _ in range(2)
        ]
        self.ticks(2)
        self.controller.start_named_drain("D0", "D1")
        self.ticks(4)
        first = next(
            self.controller.entry(rid)
            for rid in rids
            if self.controller.entry(rid)["phase"] == "FREEZE"
        )
        second = next(rid for rid in rids if rid != first["rid"])
        send_to = self.driver.send_to
        abort_acknowledged = False

        def finish_at_freeze(directory, worker, payload, timeout):
            if payload["migration_epoch"] == first["epoch"]:
                cmd = payload["cmd"]
                if cmd == "d2d_freeze":
                    self.driver.requests[worker][payload["rid"]] = self.driver.request(
                        [1, 2, 3], finished=True
                    )
                    return {
                        "deferred": True,
                        "migration_deferred_reason": "request_finished",
                        "retryable": False,
                    }
                if cmd == "d2d_precopy_close":
                    return {"closed": True}
                if cmd == "d2d_abort":
                    return {"aborted": abort_acknowledged}
                if cmd == "d2d_resume":
                    return {"resumed": True}
            return send_to(directory, worker, payload, timeout)

        with patch.object(self.driver, "send_to", side_effect=finish_at_freeze):
            self.ticks()
            status = self.controller.drain_status()
            self.assertEqual(status["state"], "MIGRATING")
            self.assertEqual(status["members_state"][first["rid"]]["phase"], "ROLLBACK")
            self.assertEqual(status["members_state"][second]["phase"], "SOURCE")
            self.assertEqual(status["members_state"][second]["migration_attempts"], 0)
            abort_acknowledged = True
            self.ticks()

        status = self.controller.drain_status()
        self.assertEqual(status["state"], "SAFE_TO_STOP")
        members = status["members_state"]
        completed = members[first["rid"]]
        self.assertEqual((completed["owner"], completed["phase"]), ("D0", "DONE"))
        self.assertEqual(
            completed["migration_outcome"], "window_missed_request_finished"
        )
        self.assertEqual(
            self.controller.entry(first["rid"])["final_output_ids"], [1, 2, 3]
        )
        self.assertNotIn(first["epoch"], self.driver.moves)
        self.assertEqual(
            (members[second]["owner"], members[second]["phase"]), ("D1", "LOCAL")
        )
        self.assertEqual(members[second]["migration_outcome"], "migrated")

    def test_mixed_drain_waits_once_and_only_moves_a_surviving_request(self):
        self.controller.close()
        self.driver.precopy = True
        self.driver.auto_release = False
        self.controller = self.open_controller(
            standby=("D1", "D2"), natural_drain_ms=300, precopy_window_ms=400
        )
        ids = [
            self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
            for _ in range(3)
        ]
        self.ticks(2)
        now = time.time()
        with patch.object(time, "time", return_value=now) as clock:
            status = self.controller.start_named_drain("D0", "D1")
            self.assertEqual(status["state"], "NATURAL_DRAINING")
            self.assertCountEqual(status["members"], ids)
            self.assertNotIn("D0", self.controller._available_decodes())
            short = self.controller.entry(ids[0])["source_rid"]
            self.driver.requests["D0"][short] = self.driver.request(
                [1, 2], finished=True
            )
            self.ticks()
            self.assertEqual(self.controller.entry(ids[0])["phase"], "DONE")
            self.assertTrue(
                all(cmd != "d2d_plan" for _, cmd, _ in self.driver.commands)
            )

            clock.return_value = status["natural_drain_until"]
            self.ticks()
            active = [
                self.controller.entry(rid)
                for rid in ids
                if self.controller.entry(rid)["epoch"]
            ]
            self.assertEqual(len(active), 1)
            move = active[0]
            self.assertEqual(move["phase"], "WAIT_CUT")
            self.assertEqual(move["precopy_bytes"], 128)
            self.assertEqual(move["precopy_until"], clock.return_value + 0.4)
            commands = [cmd for _, cmd, _ in self.driver.commands]
            self.assertLess(
                commands.index("d2d_reserve"), commands.index("d2d_precopy_prepare")
            )
            self.assertNotIn("d2d_freeze", commands)

            clock.return_value = move["precopy_until"]
            self.ticks()
            self.assertEqual(self.controller.entry(move["rid"])["phase"], "TARGET")
            survivor = next(rid for rid in ids[1:] if rid != move["rid"])
            source_rid = self.controller.entry(survivor)["source_rid"]
            self.driver.requests["D0"][source_rid] = self.driver.request(
                [1, 2], finished=True
            )
            self.ticks()
            self.assertEqual(self.controller.entry(survivor)["phase"], "DONE")
            self.assertEqual(self.controller.drain_status()["state"], "MIGRATING")
            self.driver.release(move["epoch"])
            self.ticks()
            self.assertEqual(self.controller.drain_status()["state"], "SAFE_TO_STOP")
            self.assertEqual(
                sum(cmd == "d2d_plan" for _, cmd, _ in self.driver.commands), 1
            )

    def test_natural_window_survives_restart_and_can_finish_early(self):
        self.controller.close()
        self.controller = self.open_controller(natural_drain_ms=300)
        rid = self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        self.ticks(2)
        with patch.object(time, "time", return_value=time.time()):
            drain = self.controller.start_named_drain("D0", "D1")
            self.controller.close()
            self.controller = self.open_controller()
            status = self.controller.start_named_drain("D0", "D1")
            self.assertEqual(status["id"], drain["id"])
            self.assertEqual(
                status["natural_drain_until"], drain["natural_drain_until"]
            )
            self.assertEqual(status["state"], "NATURAL_DRAINING")
            source_rid = self.controller.entry(rid)["source_rid"]
            self.driver.requests["D0"][source_rid] = self.driver.request(
                [1, 2], finished=True
            )
            self.ticks()
            self.assertEqual(self.controller.drain_status()["state"], "SAFE_TO_STOP")
            self.assertIsNone(self.controller.entry(rid)["epoch"])

    def test_expired_precopy_window_does_not_start_a_frame_before_first_token(self):
        self.controller.close()
        self.driver.precopy = True
        self.controller = self.open_controller(precopy_window_ms=400)
        rid = self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        self.ticks(2)
        source = self.driver.requests["D0"][self.controller.entry(rid)["source_rid"]]
        source["output_ids"] = []
        with patch.object(time, "time", return_value=time.time()) as clock:
            self.controller.start_named_drain("D0", "D1")
            self.ticks(3)
            entry = self.controller.entry(rid)
            self.assertEqual(entry["phase"], "WAIT_CUT")
            clock.return_value = entry["precopy_until"]
            self.driver.commands.clear()
            self.ticks(2)
            self.assertEqual(self.controller.entry(rid)["phase"], "WAIT_CUT")
            self.assertFalse(
                any(cmd.startswith("d2d_precopy") for _, cmd, _ in self.driver.commands)
            )
            source["output_ids"] = [1, 2]
            self.ticks()
            self.assertEqual(self.controller.drain_status()["state"], "SAFE_TO_STOP")

    def test_restart_keeps_the_fence_and_physical_target(self):
        from p4_dsv_gateway import DrainConflict

        rid = self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        self.ticks(2)
        drain = self.controller.start_named_drain("D0", "D1")
        self.controller.close()
        with self.assertRaisesRegex(ValueError, "registry changed"):
            self.open_controller(
                decode_workers={
                    **self.config["decode_workers"],
                    "D1": self.root / "new",
                }
            )
        self.controller = self.open_controller()
        self.assertEqual(
            self.controller.start_named_drain("D0", "D1")["id"], drain["id"]
        )
        with self.assertRaises(DrainConflict):
            self.controller.start_named_drain("D0", "D2")
        self.ticks()
        self.assertEqual(self.controller.entry(rid)["workers"]["dst"], "D1")

    def test_failed_drain_persist_does_not_fence_or_change_request_ownership(self):
        rid = self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        with patch.object(
            self.controller, "_persist", side_effect=OSError("fsync failed")
        ):
            with self.assertRaisesRegex(OSError, "fsync failed"):
                self.controller.start_named_drain("D0", "D1")
        self.assertEqual(self.controller.drain_status()["state"], "READY")
        self.assertIn("D0", self.controller._available_decodes())
        entry = self.controller.entry(rid)
        self.assertEqual(entry["workers"], {"src": "D0"})
        self.assertFalse(entry["migration_requested"])

    def test_abort_admin_rejects_stale_id_and_is_idempotent_for_current_drain(self):
        from migration_http import MigrationAdminHandler

        class Admin(MigrationAdminHandler):
            controller = self.controller

        self.controller.accept({"prompt_ids": [0], "max_new_tokens": 12})
        port = self.server(Admin)
        pair = {"src": "D0", "dst": "D1"}
        first = json.load(self.http(port, "/v1/drain", pair))
        path = "/v1/drain/abort"
        self.assertEqual(self.http(port, path, {"drain_id": 0}).status, 400)
        aborted = self.http(port, path, {"drain_id": first["id"]})
        self.assertEqual(aborted.status, 202)
        self.assertEqual(json.load(aborted)["state"], "ABORTED")
        self.assertEqual(
            json.load(self.http(port, path, {"drain_id": first["id"]}))["state"],
            "ABORTED",
        )
        second = json.load(self.http(port, "/v1/drain", pair))
        self.assertNotEqual(second["id"], first["id"])
        self.assertEqual(self.http(port, path, {"drain_id": first["id"]}).status, 409)
        self.assertNotIn("D0", self.controller._available_decodes())


class NativeWorkerTransport(WorkerTransport):
    def __init__(self):
        super().__init__()
        self.bootstrap_submissions = []
        self.first_output = [1, 2]

    def send_to(self, directory, worker, payload, timeout):
        cmd = payload["cmd"]
        if cmd == "d2d_submit_pd":
            self.bootstrap_submissions.append((worker, copy.deepcopy(payload)))
            assert payload["external_prefill"] is True
            assert payload["prefill_url"] is None
            self.requests[worker][payload["rid"]] = self.request(self.first_output)
            return {}
        if cmd == "d2d_cancel":
            self.commands.append((worker, cmd, payload["migration_epoch"]))
            row = self.requests[worker].get(payload["rid"])
            if row:
                row.update(finished=True, kv_released=True, finish_type="abort")
            return dict(cancelled=True, kv_released=True)
        return super().send_to(directory, worker, payload, timeout)


class TestNativePdDecode(CustomTestCase):
    def setUp(self):
        sys.path.insert(0, str(CONTROL))
        self.addCleanup(sys.path.remove, str(CONTROL))
        from migration_controller import Prefill
        from migration_native_pd import NativeDecodeApp, NativePdController

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = NativeWorkerTransport()
        self.controller_type = NativePdController
        self.config = dict(
            prefills={"p1": Prefill("http://p1:31016/generate", "p1", 35409)},
            decode_workers={
                worker: self.root / worker for worker in self.driver.requests
            },
            decode_urls={
                worker: f"http://127.0.0.1:{31117 + index}"
                for index, worker in enumerate(self.driver.requests)
            },
            smg_admin_urls=["http://smg1:32000", "http://smg2:32000"],
            autostart=False,
        )
        self.controller = self.open_controller()
        self.app = NativeDecodeApp(self.controller, "test-model")

    def open_controller(self, **options):
        controller = self.controller_type(
            self.driver, TextTokenizer(), self.root, **(self.config | options)
        )
        self.addCleanup(controller.close)
        controller.tick()
        return controller

    @staticmethod
    def bootstrap(room):
        return dict(bootstrap_host="p1", bootstrap_port=35409, bootstrap_room=room)

    async def start_completion(self, room, port=31117, *, generate=False):
        requests, responses = asyncio.Queue(), asyncio.Queue()
        body = dict(
            model="test-model",
            prompt=[0],
            stream=True,
            max_tokens=16,
            n=1,
            echo=False,
            stop=None,
            **self.bootstrap(room),
        )
        if generate:
            body = dict(
                model="test-model",
                input_ids=[0],
                stream=True,
                sampling_params=dict(max_new_tokens=16, temperature=0, ignore_eos=True),
                log_metrics=True,
                return_text_in_logprobs=False,
                **self.bootstrap(room),
            )
        await requests.put(dict(type="http.request", body=json.dumps(body).encode()))
        task = asyncio.create_task(
            self.app(
                {
                    "server": ("127.0.0.1", port),
                    "path": "/generate" if generate else "/v1/completions",
                    "method": "POST",
                },
                requests.get,
                responses.put,
            )
        )
        return task, requests, responses

    async def wait_for_submission(self, count=1):
        async with asyncio.timeout(3):
            while len(self.driver.bootstrap_submissions) < count:
                await asyncio.to_thread(self.controller.tick)
                await asyncio.sleep(0.002)

    async def next_response(self, responses):
        return await asyncio.wait_for(responses.get(), 3)

    def test_uvicorn_port_identifies_each_logical_decode(self):
        import uvicorn

        with ExitStack() as stack:
            sockets = [
                stack.enter_context(socket.create_server(("127.0.0.1", 0)))
                for _ in range(3)
            ]
            self.controller.listeners = {
                sock.getsockname()[1]: f"D{index}" for index, sock in enumerate(sockets)
            }
            self.controller.start_named_drain("D1", "D2")
            server = uvicorn.Server(
                uvicorn.Config(
                    self.app,
                    lifespan="off",
                    access_log=False,
                    log_level="error",
                )
            )
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": sockets}, daemon=True
            )
            thread.start()
            try:
                deadline = time.monotonic() + 4
                while (
                    not server.started
                    and thread.is_alive()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                self.assertTrue(server.started)
                for index, sock in enumerate(sockets):
                    conn = http.client.HTTPConnection(
                        "127.0.0.1", sock.getsockname()[1], timeout=3
                    )
                    try:
                        conn.request("GET", "/health")
                        response = conn.getresponse()
                        self.assertEqual(response.status, 503 if index == 1 else 200)
                        self.assertEqual(json.load(response)["decode"], f"D{index}")
                    finally:
                        conn.close()
            finally:
                server.should_exit = True
                thread.join(timeout=4)
            self.assertFalse(thread.is_alive())

    def test_native_worker_enqueues_external_pd_without_starting_prefill(self):
        # Load the actual worker handler, replacing GPU/scheduler boundaries.
        package = ModuleType("native_pd_worker_fixture")
        package.__path__ = []
        tracked, submitted, prefill_calls = {}, [], []
        d2d = SimpleNamespace(
            remote=SimpleNamespace(
                runtime=lambda: SimpleNamespace(_tracked=lambda _: tracked)
            )
        )
        sampling = SimpleNamespace(
            SamplingParams=lambda **_: SimpleNamespace(normalize=lambda _: None)
        )
        handoff = SimpleNamespace(
            _new_req=lambda scheduler, rid, ids, outputs, params, **bootstrap: (
                SimpleNamespace(rid=rid, input_ids=ids, **bootstrap)
            )
        )
        name = f"{package.__name__}.dsv_gateway_control"
        path = CONTROL.parent / "sglang_remote_attn/dsv_gateway_control.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        modules = {
            package.__name__: package,
            f"{package.__name__}.dsv_d2d": d2d,
            f"{package.__name__}.dsv_d2d_reservation": SimpleNamespace(),
            f"{package.__name__}.handoff_control": handoff,
            "sglang.srt.sampling.sampling_params": sampling,
        }
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
            module.threading = SimpleNamespace(
                Thread=lambda target, args, **kwargs: SimpleNamespace(
                    start=lambda: prefill_calls.append(args[1]["prefill_url"])
                )
            )
            worker = SimpleNamespace(
                scheduler=SimpleNamespace(
                    disaggregation_mode="decode",
                    tokenizer=None,
                    _add_request_to_queue=submitted.append,
                ),
                rank=0,
                agree=lambda votes: votes,
            )
            payload = dict(
                cmd="d2d_submit_pd",
                rid="source",
                migration_epoch="submit",
                input_ids=[0],
                sampling={},
                prefill_url=None,
                external_prefill=True,
                **self.bootstrap(41),
            )
            first = module.handle(worker, payload)
            self.assertEqual(first["bootstrap_room"], 41)
            self.assertEqual(submitted[0].bootstrap_host, "p1")
            self.assertEqual(prefill_calls, [])

            late = {**payload, "rid": "late", "migration_epoch": "late-epoch"}
            stopped = module.handle(
                worker,
                {
                    **late,
                    "cmd": "d2d_cancel",
                    "cancel_pending_submit": True,
                },
            )
            self.assertTrue(stopped["kv_released"])
            with self.assertRaisesRegex(ValueError, "submission was cancelled"):
                module.handle(worker, late)
            self.assertEqual(len(submitted), 1)

            module.handle(
                worker,
                {
                    **payload,
                    "rid": "standalone",
                    "migration_epoch": "old-flow",
                    "external_prefill": False,
                    "prefill_url": "http://p1:31016/generate",
                },
            )
            self.assertEqual(prefill_calls, ["http://p1:31016/generate"])
            self.assertEqual(len(submitted), 2)

    def test_preserves_smg_bootstrap_and_one_sse_through_handoff(self):
        self.assert_sse_handoff()

    def test_generate_keeps_native_token_prefix_and_room_through_handoff(self):
        self.assert_sse_handoff(generate=True)

    def assert_sse_handoff(self, *, generate=False):
        async def run():
            task, requests, responses = await self.start_completion(
                731, generate=generate
            )
            try:
                await self.wait_for_submission()
                await asyncio.to_thread(self.controller.tick)
                worker, payload = self.driver.bootstrap_submissions[0]
                self.assertEqual(worker, "D0")
                self.assertEqual(
                    {key: payload[key] for key in self.bootstrap(731)},
                    self.bootstrap(731),
                )
                self.assertIsNone(payload["prefill_url"])
                self.assertEqual(payload["sampling"]["ignore_eos"], generate)
                self.assertEqual((await self.next_response(responses))["status"], 200)
                first = json.loads((await self.next_response(responses))["body"][6:])
                if generate:
                    self.assertEqual(first["text"], "A")
                    self.assertEqual(first["output_ids"], [1, 2])
                    self.assertIsNone(first["meta_info"]["finish_reason"])
                else:
                    self.assertEqual(first["choices"][0]["text"], "A")

                pending = self.controller.start_named_drain("D0", "D1")
                self.assertEqual(pending["state"], "CORDONING")
                self.assertFalse(self.controller.decode_ready("D0"))
                with patch.object(self.controller, "_smg_cordoned", return_value=True):
                    self.controller._reconcile_cordon()
                send_to = self.driver.send_to

                def lagging_commit(directory, worker, payload, timeout):
                    reply = send_to(directory, worker, payload, timeout)
                    if payload["cmd"] == "d2d_commit":
                        move = self.driver.moves[payload["migration_epoch"]]
                        self.driver.requests[worker][move["target_rid"]] = (
                            self.driver.request([1], finished=True)
                        )
                    return reply

                with patch.object(self.driver, "send_to", side_effect=lagging_commit):
                    for _ in range(12):
                        await asyncio.to_thread(self.controller.tick)
                rid = next(iter(self.controller.drain_status()["members"]))
                events, finished, reason = self.controller.stream_events(rid, after=1)
                self.assertEqual((events, finished, reason), ([], False, None))
                for rid_on_dst in self.driver.requests["D1"]:
                    self.driver.requests["D1"][rid_on_dst] = self.driver.request(
                        [1, 2, 3, 4]
                    )
                await asyncio.to_thread(self.controller.tick)
                result = json.loads((await self.next_response(responses))["body"][6:])
                if generate:
                    self.assertEqual(result["text"], "A中")
                    self.assertEqual(result["output_ids"], [1, 2, 3, 4])
                    self.assertEqual(
                        result["meta_info"]["id"], first["meta_info"]["id"]
                    )
                else:
                    self.assertEqual(result["choices"][0]["text"], "中")
                    self.assertEqual(result["id"], first["id"])
                status = self.controller.drain_status()
                self.assertEqual(status["state"], "SAFE_TO_STOP")
                self.assertEqual(status["id"], pending["id"])
                self.assertEqual(len(self.driver.bootstrap_submissions), 1)

                for row in self.driver.requests["D1"].values():
                    row.update(finished=True, kv_released=True, finish_type="stop")
                await asyncio.to_thread(self.controller.tick)
                final = json.loads((await self.next_response(responses))["body"][6:])
                if generate:
                    self.assertEqual(final["output_ids"], result["output_ids"])
                    self.assertEqual(
                        final["meta_info"]["finish_reason"], {"type": "stop"}
                    )
                else:
                    self.assertEqual(final["choices"][0]["finish_reason"], "stop")
                self.assertEqual(
                    (await self.next_response(responses))["body"], b"data: [DONE]\n\n"
                )
                await asyncio.wait_for(task, 3)
            finally:
                if not task.done():
                    await requests.put({"type": "http.disconnect"})
                    await task

        asyncio.run(run())

    def test_disconnect_before_headers_cancels_external_decode(self):
        async def run():
            self.driver.first_output = []
            task, requests, responses = await self.start_completion(811)
            await self.wait_for_submission()
            self.assertTrue(responses.empty())
            await requests.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 3)
            for _ in range(3):
                await asyncio.to_thread(self.controller.tick)
            self.assertTrue(responses.empty())
            self.assertEqual(len(self.driver.bootstrap_submissions), 1)
            self.assertTrue(
                any(
                    worker == "D0" and command == "d2d_cancel"
                    for worker, command, _ in self.driver.commands
                )
            )
            self.assertTrue(
                all(row["kv_released"] for row in self.driver.requests["D0"].values())
            )

        asyncio.run(run())

    def test_cordon_waits_for_every_smg_and_preserves_already_paired_requests(self):
        from io import BytesIO

        self.controller.natural_drain_s = 0.3
        pending = self.controller.start_named_drain("D0", "D1")
        body = {"prompt_ids": [0], "max_new_tokens": 16}
        early = self.controller.accept_native("D0", body, self.bootstrap(10))

        def answer(healthy):
            return BytesIO(
                json.dumps(
                    {
                        "workers": [
                            {
                                "url": self.controller.decode_urls["D0"],
                                "worker_type": "decode",
                                "is_healthy": healthy,
                            }
                        ]
                    }
                ).encode()
            )

        with patch(
            "migration_native_pd.urlopen", side_effect=[answer(False), answer(True)]
        ):
            self.controller._reconcile_cordon()
        self.assertEqual(self.controller.drain_status()["state"], "CORDONING")
        self.assertEqual(self.controller.entry(early)["workers"], {"src": "D0"})
        confirmed_at = pending["started_at"] + 3
        with (
            patch(
                "migration_native_pd.urlopen",
                side_effect=[answer(False), answer(False)],
            ),
            patch.object(time, "time", return_value=confirmed_at),
        ):
            self.controller._reconcile_cordon()
            state = self.controller.drain_status()
        self.assertEqual(state["state"], "NATURAL_DRAINING")
        self.assertEqual(state["natural_drain_until"], confirmed_at + 0.3)
        self.assertEqual(state["id"], pending["id"])
        self.assertEqual(state["members"], [early])
        late = self.controller.accept_native("D0", body, self.bootstrap(11))
        self.assertEqual(self.controller.entry(late)["workers"], {"src": "D1"})
        self.assertEqual(self.controller.entry(late)["bootstrap_room"], 11)
        self.assertNotIn(late, self.controller.drain_status()["members"])

    def test_native_pd_prepare_overlaps_cutover_and_waits_for_retirement(self):
        self.controller.close()
        self.driver.wave_supported = True
        self.driver.precopy = True
        self.driver.import_ready = False
        self.driver.auto_release = False
        self.controller = self.open_controller(
            max_concurrent_migrations=2,
            max_preparing_migrations=4,
            chain_drain=True,
            precopy_window_ms=400,
        )
        controller = self.controller
        rids = [
            controller.accept_native(
                "D0", {"prompt_ids": [0], "max_new_tokens": 12}, self.bootstrap(i)
            )
            for i in range(7)
        ]

        def ticks(count=12):
            for _ in range(count):
                controller.tick()
                self.assertLessEqual(controller.drain_status().get("active_cutovers", 0), 2)

        ticks(2)

        def entries_in(phase):
            return [
                entry
                for rid in rids
                if (entry := self.controller.entry(rid))["phase"] == phase
            ]

        now = time.monotonic()
        with (
            patch.object(time, "time", return_value=time.time()) as clock,
            patch.object(time, "monotonic", return_value=now) as monotonic,
        ):
            pending = controller.start_named_drain("D0", "D1")
            self.assertEqual(pending["state"], "CORDONING")
            self.assertTrue(controller.cordon_wakeup.is_set())
            with patch.object(controller, "_smg_cordoned", return_value=True):
                controller._reconcile_cordon()
            ticks()
            preparing = entries_in("WAIT_CUT")
            self.assertEqual(len(preparing), 4)
            self.assertEqual(len(entries_in("SOURCE")), 3)
            self.assertTrue(all(entry["precopy_bytes"] == 128 for entry in preparing))

            clock.return_value = preparing[0]["precopy_until"]
            ticks()
            self.assertEqual(len(entries_in("IMPORT")), 2)
            status = controller.drain_status()
            self.assertEqual(status["active_cutovers"], 2)
            self.assertTrue(status["controller_features"]["chain_drain"])
            # A concurrent all-request poll can arrive between two wave
            # envelopes. It must yield even while the role lock is free.
            run_waves = controller._run_drain_waves

            def poll_between_envelopes(members):
                before = self.driver.commands.count(("D1", "d2d_worker_status", "worker"))
                controller._refresh_role("D1")
                after = self.driver.commands.count(("D1", "d2d_worker_status", "worker"))
                self.assertEqual(before, after)
                return run_waves(members)

            with (
                patch.object(controller, "background", True),
                patch.object(controller, "_run_drain_waves", poll_between_envelopes),
            ):
                controller._advance_drain_waves(entries_in("IMPORT"))
            self.assertFalse(controller.drain_wave_workers)
            queued = entries_in("WAIT_CUT")
            self.assertEqual(len(queued), 4)
            self.assertTrue(all(entry["precopy_bytes"] == 128 for entry in queued))

            clock.return_value = queued[0]["precopy_until"]
            self.driver.waves.clear()
            ticks()
            self.assertCountEqual(
                [entry["rid"] for entry in entries_in("WAIT_CUT")],
                [entry["rid"] for entry in queued],
            )
            self.assertFalse(
                any(
                    item["cmd"] == "d2d_import"
                    for _, wave in self.driver.waves
                    for item in wave
                )
            )

            self.driver.import_ready = True
            # No new readiness facts from this worker: periodic recovery retries.
            for _ in range(8):
                monotonic.return_value += 0.30
                clock.return_value += 0.30
                ticks()
            targets = entries_in("TARGET")
            self.assertEqual(len(targets), 6)
            self.assertEqual(len(entries_in("SOURCE")), 1)
            self.assertEqual(self.controller.drain_status()["state"], "MIGRATING")
            self.driver.release(targets[0]["epoch"])
            ticks()
            self.assertEqual(len(entries_in("WAIT_CUT")), 1)

    def test_submit_bound_before_cordon_stays_in_drain_until_it_finishes(self):
        """A lost submit reply leaves a SUBMIT entry already owned by the source."""
        controller = self.controller
        body = {"prompt_ids": [0], "max_new_tokens": 16}
        pending = controller.accept_native("D0", body, self.bootstrap(12))
        send_to = self.driver.send_to

        def lose_submit_reply(directory, worker, payload, timeout):
            result = send_to(directory, worker, payload, timeout)
            if payload["cmd"] == "d2d_submit_pd":
                raise TimeoutError("lost submit acknowledgement")
            return result

        with patch.object(self.driver, "send_to", side_effect=lose_submit_reply):
            controller.tick()
        self.assertEqual(controller.entry(pending)["phase"], "SUBMIT")
        self.assertTrue(controller.entry(pending)["submit_started"])

        # EOS and native removal happen before the next status poll.
        entry = controller.entry(pending)
        source_rid = entry["source_rid"]
        self.driver.requests["D0"].pop(source_rid)
        self.driver.retired["D0", entry["submission_epoch"], source_rid] = (
            self.driver.request([1, 2, 3], finished=True)
        )
        controller.start_named_drain("D0", "D1")
        with patch.object(controller, "_smg_cordoned", return_value=True):
            controller._reconcile_cordon()
        status = controller.drain_status()
        self.assertEqual(status["members"], [pending])
        self.assertNotEqual(status["state"], "SAFE_TO_STOP")

        controller.tick()
        controller.tick()
        self.assertEqual(len(self.driver.bootstrap_submissions), 1)
        self.assertEqual(controller.entry(pending)["phase"], "DONE")
        self.assertEqual(controller.entry(pending)["final_output_ids"], [1, 2, 3])
        self.assertEqual(controller.drain_status()["state"], "SAFE_TO_STOP")

    def test_cordon_timeout_unfences_source_with_unreachable_replica(self):
        from io import BytesIO

        pending = self.controller.start_named_drain("D0", "D1")
        confirm = BytesIO(
            json.dumps(
                {
                    "workers": [
                        {
                            "url": self.controller.decode_urls["D0"],
                            "worker_type": "decode",
                            "is_healthy": False,
                        }
                    ]
                }
            ).encode()
        )
        with patch(
            "migration_native_pd.urlopen",
            side_effect=[confirm, TimeoutError("replica unavailable")],
        ):
            self.controller._reconcile_cordon()
        before = self.controller.drain_status()
        self.assertEqual(before["state"], "CORDONING")
        self.assertIn("smg2:32000/workers", before["cordon_error"])
        self.assertFalse(self.controller.decode_ready("D0"))
        with patch.object(time, "time", return_value=pending["cordon_deadline_at"]):
            self.controller._reconcile_cordon()
        self.controller.close()
        self.controller = self.open_controller()
        after = self.controller.drain_status()
        self.assertEqual(after["id"], pending["id"])
        self.assertEqual(after["state"], "ABORTED")
        self.assertEqual(after["abort_reason"], "cordon_timeout")
        self.assertEqual(after["cordon_error"], before["cordon_error"])
        self.assertTrue(self.controller.decode_ready("D0"))
        rid = self.controller.accept_native(
            "D0", {"prompt_ids": [0], "max_new_tokens": 16}, self.bootstrap(81)
        )
        self.assertEqual(self.controller.entry(rid)["workers"], {"src": "D0"})
        self.assertNotEqual(
            self.controller.start_named_drain("D0", "D1")["id"], pending["id"]
        )

    def test_cordon_watchdog_expires_while_ack_is_in_flight(self):
        controller = self.controller
        query_started, release_query = threading.Event(), threading.Event()
        now = time.time()

        def blocked_ack(_):
            query_started.set()
            return release_query.wait(3)

        with (
            patch.object(controller, "_smg_cordoned", side_effect=blocked_ack),
            patch.object(time, "time", return_value=now) as clock,
        ):
            pending = controller.start_named_drain("D0", "D1")
            routing = threading.Thread(target=controller._reconcile_cordon)
            watchdog = threading.Thread(target=controller._watch_cordon_timeout)
            routing.start()
            watchdog.start()
            try:
                self.assertTrue(query_started.wait(2))
                clock.return_value = pending["cordon_deadline_at"]
                with controller.cv:
                    self.assertTrue(
                        controller.cv.wait_for(
                            lambda: controller.drain_status()["state"] == "ABORTED",
                            timeout=2,
                        )
                    )
                self.assertTrue(controller.decode_ready("D0"))
                next_drain = controller.start_named_drain("D0", "D1")
                self.assertNotEqual(next_drain["id"], pending["id"])
            finally:
                release_query.set()
                routing.join(timeout=3)
                controller.stop.set()
                watchdog.join(timeout=3)
        self.assertFalse(routing.is_alive())
        self.assertFalse(watchdog.is_alive())
        status = controller.drain_status()
        self.assertEqual(status["id"], next_drain["id"])
        self.assertEqual(status["state"], "CORDONING")
        self.assertIsNone(status["cordon_error"])
        self.assertEqual(
            controller.abort_named_drain(next_drain["id"])["abort_reason"],
            "operator_abort",
        )
        self.assertTrue(controller.decode_ready("D0"))

    def test_abort_during_import_restores_ingress_and_rolls_back_without_commit(self):
        controller = self.controller
        body = {"prompt_ids": [0], "max_new_tokens": 16}
        members = [
            controller.accept_native("D0", body, self.bootstrap(room))
            for room in (90, 89)
        ]
        controller.tick()
        pending = controller.start_named_drain("D0", "D1")
        with patch.object(controller, "_smg_cordoned", return_value=True):
            controller._reconcile_cordon()
        routed = controller.accept_native("D0", body, self.bootstrap(91))
        self.assertEqual(controller.entry(routed)["workers"], {"src": "D1"})
        send_to = self.driver.send_to
        result = {}
        cleanup_ready = False

        def abort_during_import(directory, worker, payload, timeout):
            cmd = payload["cmd"]
            if cmd == "d2d_import" and not result:
                reply = send_to(directory, worker, payload, timeout)
                result.update(controller.abort_named_drain(pending["id"]))
                return reply
            if cmd == "d2d_precopy_close":
                return {"closed": True}
            if cmd == "d2d_abort":
                return {"aborted": cleanup_ready}
            if cmd == "d2d_resume":
                return {"resumed": True}
            return send_to(directory, worker, payload, timeout)

        with patch.object(self.driver, "send_to", side_effect=abort_during_import):
            for _ in range(10):
                controller.tick()
            self.assertEqual(result["state"], "ABORTING")
            rid = next(
                rid
                for rid in members
                if result["members_state"][rid]["phase"] == "IMPORT"
            )
            waiting = next(other for other in members if other != rid)
            self.assertEqual(controller.drain_status()["state"], "ABORTING")
            self.assertEqual(controller.entry(rid)["phase"], "ROLLBACK")
            self.assertTrue(controller.decode_ready("D0"))
            fresh = controller.accept_native("D0", body, self.bootstrap(92))
            self.assertEqual(controller.entry(fresh)["workers"], {"src": "D0"})
            cleanup_ready = True
            for _ in range(5):
                controller.tick()
        status = controller.drain_status()
        self.assertEqual(status["state"], "ABORTED")
        self.assertEqual(
            status["members_state"][rid]["migration_outcome"], "drain_aborted"
        )
        self.assertEqual(controller.entry(rid)["phase"], "SOURCE")
        self.assertFalse(controller.entry(rid)["cancel_requested"])
        self.assertFalse(controller.entry(rid)["migration_requested"])
        self.assertEqual(status["members_state"][waiting]["migration_attempts"], 0)
        self.assertNotIn("d2d_commit", [cmd for _, cmd, _ in self.driver.commands])
        retry = controller.start_named_drain("D0", "D1")
        self.assertNotEqual(retry["id"], pending["id"])
        with patch.object(controller, "_smg_cordoned", return_value=True):
            controller._reconcile_cordon()
        self.assertEqual(controller.entry(rid)["attempts"], 0)

    def test_abort_keeps_durable_commit_and_waits_for_source_retirement(self):
        controller = self.controller
        self.driver.auto_release = False
        rid = controller.accept_native(
            "D0", {"prompt_ids": [0], "max_new_tokens": 16}, self.bootstrap(93)
        )
        controller.tick()
        pending = controller.start_named_drain("D0", "D1")
        with patch.object(controller, "_smg_cordoned", return_value=True):
            controller._reconcile_cordon()
        for _ in range(6):
            controller.tick()
        entry = controller.entry(rid)
        self.assertEqual(entry["phase"], "COMMIT")
        self.assertEqual(
            controller.abort_named_drain(pending["id"])["state"], "ABORTING"
        )
        controller.close()
        self.controller = controller = self.open_controller()
        self.assertEqual(controller.entry(rid)["phase"], "TARGET")
        self.assertEqual(controller.drain_status()["state"], "ABORTING")
        self.assertTrue(controller.decode_ready("D0"))
        self.driver.release(entry["epoch"])
        controller.tick()
        status = controller.drain_status()
        self.assertEqual(status["state"], "ABORTED")
        self.assertEqual(status["members_state"][rid]["owner"], "D1")
        self.assertEqual(status["members_state"][rid]["migration_outcome"], "migrated")


if __name__ == "__main__":
    unittest.main()
