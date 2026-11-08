"""Gateway polling and SSE recovery with a transport-only worker fixture."""

import copy
import http.client
import importlib.util
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


class StatusDriver:
    def __init__(self, protocol):
        self.error = None
        self.submissions = []
        self.rows = {
            role: dict(
                protocol=protocol,
                rank=0,
                tp_size=1,
                instance_id=role + "-instance",
                requests={},
            )
            for role in ("src", "dst")
        }

    def send(self, role, payload, timeout):
        if payload["cmd"] == "d2d_submit_pd":
            self.submissions.append((role, payload))
            return dict(ok=True)
        assert payload["cmd"] == "d2d_worker_status"
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.rows[role])


class TestDsvGatewayStatus(CustomTestCase):
    def setUp(self):
        super().setUp()
        path = ROOT / "plugins/remote_attn/control_plane/p4_dsv_gateway.py"
        spec = importlib.util.spec_from_file_location("dsv_gateway_status_test", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.driver = StatusDriver(self.module.PROTOCOL)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.gw = self.module.DsvGateway(
            self.driver,
            SimpleNamespace(encode=lambda _: [1]),
            folder.name,
            prefill_url="http://unused/generate",
            autostart=False,
            async_handoff=True,
        )
        self.addCleanup(self.gw.close)
        self.gw._refresh()

    def request(self, rid, role):
        self.gw.accept(dict(rid=rid, prompt_ids=[1], max_new_tokens=4), role=role)
        self.gw._update(rid, phase="SOURCE" if role == "src" else "TARGET")
        entry = self.gw.entry(rid)
        native_rid = entry["source_rid" if role == "src" else "target_rid"]
        req = dict(output_ids=[31, 32], finished=False, kv_released=False)
        self.driver.rows[role]["requests"][native_rid] = req
        return req

    def test_submit_waits_for_first_visible_status_without_resubmitting(self):
        """A confirmed submit may precede many snapshots, even for one-token output."""
        for role in ("src", "dst"):
            with self.subTest(role=role):
                rid = self.gw.accept(
                    dict(
                        rid="delayed-" + role, prompt_ids=[1], max_new_tokens=1, cut=1
                    ),
                    role=role,
                )
                self.gw.tick()
                for _ in range(4):
                    self.gw.tick()
                    entry = self.gw.entry(rid)
                    self.assertIsNone(entry["error"])
                    self.assertFalse(entry["blocked"])
                    self.assertEqual(self.gw.events(rid), ([], False))
                self.assertTrue(entry["awaiting_visibility"])
                self.assertEqual(sum(r == role for r, _ in self.driver.submissions), 1)
                native = entry["source_rid"] if role == "src" else entry["target_rid"]
                self.driver.rows[role]["requests"][native] = dict(
                    output_ids=[31], finished=True, kv_released=True
                )
                self.gw.tick()
                events, finished = self.gw.events(rid)
                self.assertTrue(finished)
                self.assertEqual([e["token_id"] for e in events], [31])
                self.assertEqual(self.gw.entry(rid)["phase"], "DONE")
                self.assertIsNone(self.gw.entry(rid)["error"])

    def test_visibility_deadline_survives_restart_and_seen_request_loss_is_error(self):
        """Waiting is bounded and cannot hide disappearance of a known request."""
        rid = self.gw.accept(dict(rid="deadline", prompt_ids=[1], max_new_tokens=4))
        self.gw.tick()
        confirmed = self.gw.entry(rid)["submission_confirmed_at"]
        folder = self.gw.path.parent
        self.gw.close()
        self.gw = self.module.DsvGateway(
            self.driver,
            None,
            folder,
            prefill_url="http://unused/generate",
            autostart=False,
        )
        self.addCleanup(self.gw.close)
        self.gw.tick()
        self.assertIsNone(self.gw.entry(rid)["error"])
        with patch.object(self.module.time, "time", return_value=confirmed + 31):
            self.gw.tick()
        self.assertIn("native request missing", self.gw.entry(rid)["error"])
        entry = self.gw.entry(rid)
        requests = self.driver.rows["src"]["requests"]
        requests[entry["source_rid"]] = dict(
            output_ids=[31], finished=False, kv_released=False
        )
        self.gw.tick()
        self.assertIsNone(self.gw.entry(rid)["error"])
        requests.clear()
        self.gw.tick()
        self.assertIn("native request missing", self.gw.entry(rid)["error"])
        self.assertEqual(len(self.driver.submissions), 1)

    def test_status_timeout_preserves_facts_and_recovers_without_changing_owner(self):
        """A slow advisory poll must not poison every request on its worker."""
        for role in ("src", "dst"):
            with self.subTest(role=role):
                req = self.request(role, role)
                self.gw._refresh_role(role)
                before = self.gw.entry(role)
                facts = copy.deepcopy(self.gw.facts[role])
                self.driver.error = TimeoutError("status exceeded 2s during wave")
                self.gw._refresh_role(role)
                self.assertNotIn(role, self.gw.fact_errors)
                self.assertEqual(self.gw.facts[role], facts)
                events, finished = self.gw.events(role)
                self.assertEqual([e["token_id"] for e in events], [31, 32])
                self.assertFalse(finished)
                # A timeout must not clear an earlier confirmed failure either.
                self.gw.fact_errors[role] = "confirmed transport failure"
                self.gw._refresh_role(role)
                self.assertEqual(
                    self.gw.fact_errors[role], "confirmed transport failure"
                )
                self.driver.error = None
                req.update(output_ids=[31, 32, 33, 34], finished=True)
                self.gw._refresh_role(role)
                events, finished = self.gw.events(role, after=1)
                self.assertEqual([e["seq"] for e in events], [2, 3])
                self.assertEqual([e["token_id"] for e in events], [33, 34])
                self.assertTrue(finished)
                self.assertNotIn(role, self.gw.fact_errors)
                self.assertEqual(self.gw.entry(role), before)

    def test_unavailable_worker_keeps_eight_sse_connections_until_recovery(self):
        """One failed worker poll must not turn eight live SSE streams into EOF."""
        rids = [f"request-{i}" for i in range(8)]
        requests = [self.request(rid, "dst") for rid in rids]
        self.gw._refresh_role("dst")
        before = copy.deepcopy(self.gw.journal)
        gw = self.gw

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                gw.stream(self, self.path.lstrip("/"))

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        self.addCleanup(server.server_close)
        self.addCleanup(serving.join, 2)
        self.addCleanup(server.shutdown)
        seen = [[] for _ in rids]
        initial = [threading.Event() for _ in rids]
        heartbeat = [threading.Event() for _ in rids]
        done = [threading.Event() for _ in rids]
        errors = []
        outage = threading.Event()

        def client(i):
            conn = http.client.HTTPConnection(
                "127.0.0.1", server.server_port, timeout=5
            )
            try:
                conn.request("GET", "/" + rids[i])
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                while line := response.readline():
                    if line.startswith(b": keepalive") and outage.is_set():
                        heartbeat[i].set()
                    if not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done[i].set()
                        return
                    event = json.loads(data)
                    if "seq" in event:
                        seen[i].append(event)
                        if len(seen[i]) == 2:
                            initial[i].set()
                raise AssertionError("SSE EOF before [DONE]")
            except BaseException as exc:
                errors.append(repr(exc))
            finally:
                conn.close()

        clients = [threading.Thread(target=client, args=(i,)) for i in range(8)]
        for thread in clients:
            thread.start()
        try:
            self.assertTrue(all(e.wait(5) for e in initial), errors)
            self.driver.error = ConnectionError("worker temporarily unreachable")
            self.gw._refresh_role("dst")
            self.assertIn("dst", self.gw.fact_errors)
            outage.set()
            self.assertTrue(all(e.wait(3) for e in heartbeat), errors)
            self.assertFalse(errors, errors)
            self.assertFalse(any(e.is_set() for e in done))
            self.assertEqual(self.gw.journal, before)
            self.driver.error = None
            for req in requests:
                req.update(output_ids=[31, 32, 33, 34], finished=True)
            self.gw._refresh_role("dst")
            self.assertTrue(all(e.wait(5) for e in done), errors)
            self.assertFalse(errors, errors)
            for events in seen:
                self.assertEqual([e["seq"] for e in events], [0, 1, 2, 3])
                self.assertEqual([e["token_id"] for e in events], [31, 32, 33, 34])
                self.assertEqual({e["owner"] for e in events}, {"dst"})
            self.assertEqual(self.gw.journal, before)
        finally:
            self.gw.stop.set()
            for thread in clients:
                thread.join(6)
        self.assertFalse(any(thread.is_alive() for thread in clients))

    def test_timeout_does_not_authorize_missing_or_restarted_worker(self):
        """Advisory timeout tolerance must preserve the worker identity fence."""
        self.request("identity", "dst")
        self.gw.facts.pop("dst")
        self.driver.error = TimeoutError("no status yet")
        self.gw._refresh_role("dst")
        with self.assertRaisesRegex(ConnectionError, "unavailable"):
            self.gw.events("identity")
        self.driver.error = None
        self.driver.rows["dst"]["instance_id"] = "replacement-worker"
        self.gw._refresh_role("dst")
        with self.assertRaisesRegex(RuntimeError, "worker restarted"):
            self.gw.events("identity")
        self.driver.error = TimeoutError("replacement status delayed")
        self.gw._refresh_role("dst")
        with self.assertRaisesRegex(RuntimeError, "worker restarted"):
            self.gw.events("identity")
        self.assertEqual(self.gw.entry("identity")["owner"], "dst")


class TestDsvDrainSSE(CustomTestCase):
    def setUp(self):
        path = ROOT / "test/manual/remote_attn/bench_p4_dsv_drain.py"
        spec = importlib.util.spec_from_file_location("bench_dsv_sse_test", path)
        self.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bench)

    def consume(self, wire=None, error=None):
        row = dict(tokens=[], done=False)
        with patch.object(
            self.bench.urllib.request,
            "urlopen",
            side_effect=error,
            return_value=io.BytesIO(wire or b""),
        ):
            self.bench.stream("http://unused", {}, row, threading.Lock(), 1)
        return row

    def test_status_error_and_null_error_do_not_abort_tokens(self):
        """Advisory status errors must not terminate the benchmark reader."""
        row = self.consume(
            b'event: status\ndata: {"error":"waiting", "blocked":true}\n\n'
            b': keepalive\n\nevent: status\ndata: {"error":null}\n\n'
            b'data: {"token_id":31,"seq":0,"error":null}\n\n'
            b"data: [DONE]\n\n"
        )
        self.assertTrue(row["done"])
        self.assertNotIn("error", row)
        self.assertEqual([e["token_id"] for e in row["tokens"]], [31])

    def test_fatal_error_eof_and_transport_failure_remain_failures(self):
        for wire in (
            b'event: error\ndata: {"error":"fatal"}\n\ndata: [DONE]\n\n',
            b'data: {"error":"fatal"}\n\ndata: [DONE]\n\n',
            b'data: {"token_id":31}\n\n',
            b"data: [DONE]",
            b'event: status\ndata: {"error":null}\n\n',
        ):
            with self.subTest(wire=wire):
                row = self.consume(wire)
                self.assertFalse(row["done"])
                self.assertTrue(row["error"])
        for exc in (TimeoutError("timed out"), ConnectionError("reset")):
            row = self.consume(error=exc)
            self.assertFalse(row["done"])
            self.assertTrue(row["error"])

    def test_multiline_sse_resets_event_type_between_frames(self):
        wire = io.BytesIO(
            b'event: status\r\ndata: {"error":\r\ndata: null}\r\n\r\ndata: [DONE]\r\n\r\n'
        )
        self.assertEqual(
            list(self.bench.sse_events(wire)),
            [("status", '{"error":\nnull}'), ("message", "[DONE]")],
        )

    def test_warmup_uses_multiple_tokens_without_migration(self):
        bodies = []

        def capture(url, body, row, lock, timeout):
            bodies.append(body)
            row.update(done=False, error="stop after warmup")

        with patch.object(self.bench, "rpc", return_value={}), patch.object(
            self.bench, "stream", side_effect=capture
        ):
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                self.bench.run("http://unused", [1], 1, 128, 32, "natural", 1)
        self.assertGreater(bodies[0]["max_new_tokens"], 1)
        self.assertEqual(bodies[0]["cut"], bodies[0]["max_new_tokens"])


if __name__ == "__main__":
    unittest.main()
