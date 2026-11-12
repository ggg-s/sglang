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
        self.moves = {}
        self.migrations = {worker: {} for worker in self.requests}
        self.commands = []
        self.submissions = []
        self.auto_release = True
        self.precopy = False

    def send_to(self, directory, worker, payload, timeout):
        assert directory.name == worker
        cmd, epoch = payload["cmd"], payload["migration_epoch"]
        requests = self.requests[worker]
        self.commands.append((worker, cmd, epoch))
        if cmd == "d2d_worker_status":
            return dict(
                protocol="dsv-d2d-v2",
                rank=0,
                tp_size=1,
                instance_id=worker,
                idle=all(row["kv_released"] for row in requests.values()),
                requests=copy.deepcopy(requests),
                migrations=copy.deepcopy(self.migrations[worker]),
                precopy_protocol="closed-page-precopy-v1" if self.precopy else None,
            )
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

        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.driver = NativeWorkerTransport()
        self.controller = NativePdController(
            self.driver,
            TextTokenizer(),
            root,
            prefills={"p1": Prefill("http://p1:31016/generate", "p1", 35409)},
            decode_workers={worker: root / worker for worker in self.driver.requests},
            decode_urls={
                worker: f"http://127.0.0.1:{31117 + index}"
                for index, worker in enumerate(self.driver.requests)
            },
            smg_admin_urls=["http://smg1:32000", "http://smg2:32000"],
            autostart=False,
        )
        self.addCleanup(self.controller.close)
        self.controller.tick()
        self.app = NativeDecodeApp(self.controller, "test-model")

    @staticmethod
    def bootstrap(room):
        return dict(bootstrap_host="p1", bootstrap_port=35409, bootstrap_room=room)

    async def start_completion(self, room, port=31117):
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
        await requests.put(dict(type="http.request", body=json.dumps(body).encode()))
        task = asyncio.create_task(
            self.app(
                {
                    "server": ("127.0.0.1", port),
                    "path": "/v1/completions",
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
        async def run():
            task, requests, responses = await self.start_completion(731)
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
                self.assertEqual((await self.next_response(responses))["status"], 200)
                first = json.loads((await self.next_response(responses))["body"][6:])
                self.assertEqual(first["choices"][0]["text"], "A")

                pending = self.controller.start_named_drain("D0", "D1")
                self.assertEqual(pending["state"], "CORDONING")
                self.assertFalse(self.controller.decode_ready("D0"))
                with patch.object(self.controller, "_smg_cordoned", return_value=True):
                    self.controller._reconcile_cordon()
                for _ in range(12):
                    await asyncio.to_thread(self.controller.tick)
                result = json.loads((await self.next_response(responses))["body"][6:])
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


if __name__ == "__main__":
    unittest.main()
