"""Gateway polling and SSE recovery with a transport-only worker fixture."""

import copy
import http.client
import importlib.util
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


class StatusDriver:
    def __init__(self, protocol):
        self.error = None
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


if __name__ == "__main__":
    unittest.main()
