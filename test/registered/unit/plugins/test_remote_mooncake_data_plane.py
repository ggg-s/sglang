"""Q/O/LSE control/ownership contracts with explicit CPU NIC/CUDA boundaries."""

import ctypes
import os
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "plugins/remote_attn"))

from sglang_remote_attn import dsv41_remote
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import mooncake_backend as mb
from sglang_remote_attn import remote_data_plane as dp


class NIC:
    """Retains raw pointer work on failure; the runtime must own its backing."""

    d2d_protocol_verified = True

    def __init__(self, name, peers, protocol="tcp", gpu=True):
        self.name = self.hostname = name
        self.protocol, self.gpu = protocol, gpu
        self.peers = peers
        peers[name] = self
        self.regions, self.pending = {}, {}
        self.calls = []
        self.sequence = self.sync_calls = 0
        self.failure = None
        self.started = self.gate = None
        self.fail_register = self.fail_unregister = False

    def get_session_id(self):
        return self.name

    def batch_register(self, pointers, sizes):
        self.regions.update(zip(pointers, sizes))
        return -1 if self.fail_register else 0

    def batch_deregister(self, pointers):
        if self.fail_unregister:
            return -1
        for pointer in pointers:
            del self.regions[pointer]
        return 0

    def copy(self, args):
        session, sources, targets, sizes = args
        peer = self.peers[session]
        for source, target, size in zip(sources, targets, sizes):
            assert any(
                a <= source and source + size <= a + n for a, n in self.regions.items()
            )
            assert any(
                a <= target and target + size <= a + n for a, n in peer.regions.items()
            )
            ctypes.memmove(target, source, size)

    def batch_transfer_sync(self, *args):
        self.sync_calls += 1
        # Deliberately model the old unsafe convenience API: failed attempt is
        # pending, an internal retry succeeds, and the public return hides it.
        self.pending[-1] = args
        self.copy(args)
        return 0

    batch_transfer_sync_write = batch_transfer_sync

    def batch_transfer_async_write(self, *args):
        self.sequence += 1
        self.calls.append(args)
        self.pending[self.sequence] = args
        if self.failure is None:
            self.copy(args)
        return self.sequence

    def get_batch_transfer_status(self, tickets):
        if self.started is not None:
            self.started.set()
            if not self.gate.wait(3):
                raise TimeoutError("status gate")
        if self.failure == "delivery":
            return 0  # TCP sender completion precedes delayed receiver delivery.
        if self.failure == "exception":
            raise RuntimeError("injected status loss")
        if self.failure:
            return -1
        for ticket in tickets:
            del self.pending[ticket]
        return 0

    def flush_late(self):
        assert self.pending
        for args in self.pending.values():
            self.copy(args)
        self.pending.clear()


class TestRemoteMooncakeDataPlane(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "HANDOFF_REMOTE_GPU_TRANSFER": "1",
                    "HANDOFF_REMOTE_BACKEND": "auto",
                    "HANDOFF_D2D_BACKEND": "auto",
                },
            )
        )
        for key in ("_HANDLES", "_TCP", "_PROBES"):
            self.enterContext(patch.object(mb, key, {}))
        self.enterContext(patch.object(mb, "_RETAINED", []))

    def fixture(self, protocol="tcp", server_gpu=True, client_gpu=True):
        peers = {}
        p, d = (
            NIC("P", peers, protocol, server_gpu),
            NIC("D", peers, protocol, client_gpu),
        )
        pt, dt = NIC("P-tcp", peers), NIC("D-tcp", peers)
        self.enterContext(
            patch.object(
                mb,
                "gpu_capability",
                side_effect=lambda engine, device: (
                    engine.protocol == "rdma" and engine.native.gpu
                ),
            )
        )
        self.enterContext(
            patch.object(
                mb,
                "tcp_engine",
                side_effect=lambda host="127.0.0.1": mb.handle(peers[host + "-tcp"]),
            )
        )
        pp, dd = dp.Plane(NS(engine=p), "cpu"), dp.Plane(NS(engine=d), "cpu")
        hub = rt.RemoteHub(
            NS(_pre_layerwise=NS(gpu_data_plane=pp)),
            stream_factory=lambda n: [(None, None, None)] * n,
        )
        self.addCleanup(hub.close)
        self.addCleanup(pp.close)
        self.addCleanup(dd.close)
        freed = []
        source = NS(
            _pre_layerwise_source=NS(peers=((NS(mooncake_session_id="D"), None),))
        )
        ctx = hub.registry.register(
            "qo", source, 0, 8, 8, "secret", on_release=lambda ctx: freed.append(True)
        )
        route = dict(
            host=hub.host,
            port=hub.port,
            handoff_id="qo",
            epoch=ctx.owner_epoch,
            token="secret",
        )

        def compute(job, stream, cache):
            query = job.q_cpu
            return (
                {},
                query.float() * 3,
                torch.zeros(query.shape[0], dtype=torch.float32),
            )

        self.enterContext(patch.object(dsv41_remote, "execute", compute))
        self.enterContext(
            patch.object(
                rt,
                "_send_tensor",
                side_effect=AssertionError("control socket tensor payload"),
            )
        )
        return NS(
            p=p,
            d=d,
            pt=pt,
            dt=dt,
            pp=pp,
            dd=dd,
            hub=hub,
            source=source,
            ctx=ctx,
            route=route,
            freed=freed,
        )

    def call(self, fixture, req=None, client=None):
        req = req or NS()
        client = client or fixture.dd.client(fixture.route)
        query = torch.arange(16).reshape(2, 8).to(torch.bfloat16)
        result, out, lse = client.call(
            [query], [{**fixture.route, "dsv41_sparse": True}], [req]
        )
        torch.testing.assert_close(out, query.float()[None] * 3)
        torch.testing.assert_close(lse, torch.zeros(1, 2))
        return client, req, result

    def wait(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("condition did not become true")
            time.sleep(0.005)

    def test_tcp_return_layout_negotiation_and_inline_ack_use_single_submit(self):
        for mode in ("packed", "legacy_client", "legacy_server"):
            for inline_ack in (False, True):
                self.check_tcp_return(mode, inline_ack)

    def check_tcp_return(self, mode, inline_ack):
        with self.subTest(mode=mode, inline_ack=inline_ack):
            fixture = self.fixture()
            client = fixture.dd.client(fixture.route)
            rpc = client.rpc
            frames = []

            def capture(frame):
                frame = dict(frame)
                if not inline_ack and frame["cmd"] == "te_negotiate":
                    frame.pop("inline_ack", None)  # Legacy peer.
                if mode != "packed" and frame["cmd"] == "te_negotiate":
                    frame.pop("packed_return", None)
                frames.append(frame)
                result = rpc(frame)
                if mode == "legacy_server" and frame["cmd"] == "te_negotiate":
                    result.pop("packed_return", None)
                return result

            with patch.object(client, "rpc", side_effect=capture):
                _, req, result = self.call(fixture, client=client)
                self.call(fixture, client=client)
                client.close()
            self.assertEqual([len(call[1]) for call in fixture.d.calls], [1, 1])
            count = 1 if mode == "packed" else 2
            self.assertEqual([len(call[1]) for call in fixture.p.calls], [count, count])
            self.assertEqual(client.packed_return, mode == "packed")
            self.assertEqual(
                fixture.pp.stats()["packed_returns"], 2 if mode == "packed" else 0
            )
            self.assertEqual(fixture.pp.stats()["transfer_regions"], 2 * count)
            timing = result["source_timing_ms"]
            parts = [
                timing["source_return_" + part]
                for part in ("prepare", "slot_wait", "transfer")
            ]
            self.assertTrue(all(part >= 0 for part in parts))
            self.assertLessEqual(sum(parts), timing["source_return"])
            self.assertEqual(fixture.p.sync_calls + fixture.d.sync_calls, 0)
            self.assertTrue(dp.ready(req))
            self.assertFalse(fixture.p.regions or fixture.d.regions)
            self.assertEqual(client.inline_ack, inline_ack)
            self.assertEqual(
                [f["ack_sequence"] for f in frames if "ack_sequence" in f],
                [1, 2] if inline_ack else [],
            )
            self.assertEqual(
                sum(f["cmd"] == "te_ack" for f in frames), 0 if inline_ack else 2
            )

    def test_mixed_gpu_capabilities_fall_back_before_any_payload(self):
        fixture = self.fixture("rdma", server_gpu=False, client_gpu=True)
        client, req, _ = self.call(fixture)
        self.assertFalse(fixture.p.calls or fixture.d.calls)
        self.assertEqual(len(fixture.pt.calls), 1)
        self.assertEqual(len(fixture.dt.calls), 1)
        self.assertTrue(dp.ready(req))
        self.assertTrue(client.packed_return)
        self.assertEqual(len(fixture.pt.calls[0][1]), 1)
        client.close()
        self.assertFalse(fixture.pt.regions or fixture.dt.regions)

    def test_both_gpu_capable_use_rdma_without_tcp_payload_retry(self):
        fixture = self.fixture("rdma")
        client, req, _ = self.call(fixture)
        self.assertEqual(len(fixture.p.calls), 1)
        self.assertEqual(len(fixture.d.calls), 1)
        self.assertFalse(fixture.pt.calls or fixture.dt.calls)
        self.assertTrue(dp.ready(req))
        self.assertFalse(client.packed_return)
        self.assertEqual(len(fixture.p.calls[0][1]), 2)
        client.close()

    def test_late_q_write_keeps_both_backings_and_request_quarantined(self):
        fixture = self.fixture("rdma")
        fixture.d.failure = "late"
        client, req = fixture.dd.client(fixture.route), NS()
        with self.assertRaises(RuntimeError):
            self.call(fixture, req, client)
        self.wait(lambda: bool(fixture.pp.quarantine))
        source_regions, target_regions = (
            dict(fixture.d.regions),
            dict(fixture.p.regions),
        )
        self.assertTrue(source_regions and target_regions)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)
        client.close()
        fixture.d.flush_late()
        self.assertEqual(fixture.d.regions, source_regions)
        self.assertEqual(fixture.p.regions, target_regions)
        self.assertTrue(client.slab.quarantined)
        self.assertEqual(len(fixture.d.calls), 1)
        self.assertFalse(fixture.p.calls or fixture.pt.calls or fixture.dt.calls)

    def test_late_o_lse_write_keeps_reader_lease_after_cancel(self):
        fixture = self.fixture("rdma")
        fixture.p.failure = "late"
        req, client = NS(), fixture.dd.client(fixture.route)
        with self.assertRaises(RuntimeError):
            self.call(fixture, req, client)
        self.assertTrue(fixture.p.pending)
        self.assertEqual(len(next(iter(fixture.p.pending.values()))[1]), 2)
        self.assertFalse(
            fixture.hub.registry.release("qo", expected_req=fixture.source)
        )
        self.assertFalse(fixture.freed)
        before = dict(fixture.p.regions), dict(fixture.d.regions)
        client.close()
        fixture.p.flush_late()
        self.assertEqual((fixture.p.regions, fixture.d.regions), before)
        self.assertFalse(fixture.freed)
        self.assertTrue(fixture.source._handoff_transfer_unsafe)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)
        self.assertEqual(fixture.p.sync_calls + fixture.d.sync_calls, 0)
        self.assertEqual(len(fixture.p.calls), 1)

    def test_status_exception_keeps_old_slot_and_new_request_uses_fresh_memory(self):
        fixture = self.fixture("rdma")
        fixture.d.failure = "exception"
        req, old = NS(), fixture.dd.client(fixture.route)
        with self.assertRaisesRegex(RuntimeError, "status loss"):
            self.call(fixture, req, old)
        before = old.slab.wire.data_ptr()
        fixture.d.failure = None
        new, _, _ = self.call(fixture)
        self.assertNotEqual(new.slab.wire.data_ptr(), before)
        fixture.d.flush_late()
        self.assertTrue(old.slab.quarantined)
        self.assertIn(before, fixture.d.regions)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)
        self.assertEqual(len(fixture.d.calls), 2)
        new.close()

    def test_old_control_and_capability_versions_are_rejected_before_registration(self):
        for bad in ("control", "capability", "identity"):
            with self.subTest(bad=bad):
                fixture = self.fixture()
                client = fixture.dd.client(fixture.route)
                original = client.rpc

                def rpc(frame):
                    if frame.get("cmd") == "te_negotiate":
                        frame = dict(frame)
                        if bad == "control":
                            frame["version"] = "remote-mooncake-v0"
                        elif bad == "capability":
                            frame["capabilities"] = dict(
                                version="d2d-mooncake-v1", backends=[mb.TCP]
                            )
                        else:
                            frame["native_session"] = "not-the-native-peer"
                    return original(frame)

                client.rpc = rpc
                with self.assertRaises((ValueError, RuntimeError)):
                    self.call(fixture, client=client)
                self.assertFalse(fixture.p.calls or fixture.d.calls)
                self.assertFalse(fixture.p.regions or fixture.d.regions)

    def test_disconnect_during_pending_return_keeps_old_dma_and_source_lease(self):
        fixture = self.fixture("rdma")
        fixture.p.failure = "late"
        fixture.p.started, fixture.p.gate = threading.Event(), threading.Event()
        client, req = fixture.dd.client(fixture.route), NS()
        errors = []

        def run():
            try:
                self.call(fixture, req, client)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(fixture.p.started.wait(3))
            self.assertFalse(dp.ready(req))
            self.assertFalse(
                fixture.hub.registry.release("qo", expected_req=fixture.source)
            )
            self.assertFalse(fixture.freed)
            client.sock.shutdown(socket.SHUT_RDWR)
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertTrue(errors)
            before = dict(fixture.p.regions), dict(fixture.d.regions)
            self.assertTrue(before[0] and before[1])
        finally:
            fixture.p.gate.set()
            thread.join(3)
        self.wait(
            lambda: fixture.source.__dict__.get("_handoff_transfer_unsafe", False)
        )
        fixture.p.flush_late()
        self.assertEqual((fixture.p.regions, fixture.d.regions), before)
        self.assertFalse(fixture.freed)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)
        self.assertEqual(len(fixture.p.calls), 1)

    def test_remote_strict_policy_is_independent_and_rejects_tcp_only_peer(self):
        with patch.dict(
            os.environ, {"HANDOFF_REMOTE_BACKEND": "rdma", "HANDOFF_D2D_BACKEND": "tcp"}
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                self.fixture("tcp")

    def test_tcp_receiver_waits_for_payload_footer_after_native_send_completion(self):
        fixture = self.fixture()
        fixture.d.failure = "delivery"
        req, client = NS(), fixture.dd.client(fixture.route)
        results, errors = [], []

        def run():
            try:
                results.append(self.call(fixture, req, client))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.wait(lambda: bool(fixture.d.pending))
            self.assertFalse(dp.ready(req))
            self.assertFalse(fixture.p.calls)
            fixture.d.flush_late()
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)
            self.assertEqual(len(results), 1)
            self.assertTrue(dp.ready(req))
            self.assertEqual(len(fixture.p.calls), 1)
        finally:
            if thread.is_alive():
                if client.sock is not None:
                    client.sock.shutdown(socket.SHUT_RDWR)
                thread.join(3)

    def test_partial_registration_failure_retains_backing_before_owner_assignment(self):
        fixture = self.fixture()
        fixture.d.fail_register = True
        client, req = fixture.dd.client(fixture.route), NS()
        with self.assertRaisesRegex(RuntimeError, "registration"):
            self.call(fixture, req, client)
        self.assertTrue(fixture.d.regions)
        self.assertTrue(fixture.dd.quarantine)
        self.assertFalse(fixture.d.calls or fixture.p.calls)
        before = dict(fixture.d.regions)
        client.close()
        self.assertEqual(fixture.d.regions, before)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)

    def test_unregister_failure_keeps_remote_and_local_wire_backings(self):
        fixture = self.fixture()
        client, _, _ = self.call(fixture)
        fixture.p.fail_unregister = True
        before = dict(fixture.p.regions), dict(fixture.d.regions)
        with self.assertRaisesRegex(RuntimeError, "deregistration"):
            client.close()
        self.assertEqual((fixture.p.regions, fixture.d.regions), before)
        self.assertTrue(client.slab.quarantined)
        self.assertTrue(fixture.pp.quarantine)

    def test_tcp_markers_bind_each_region_size_nonce_and_sequence(self):
        fixture = self.fixture()
        client = fixture.dd.client(fixture.route)
        rpc = client.rpc

        def legacy(frame):
            frame = dict(frame)
            frame.pop("packed_return", None)
            return rpc(frame)

        with patch.object(client, "rpc", side_effect=legacy):
            _, _, result = self.call(fixture, client=client)
        lane = next(iter(fixture.pp.lanes))
        query_spec = dict(shape=[1, 2, 8], dtype="bfloat16")
        for slab, entries in (
            (lane.slab, [(0, query_spec)]),
            (client.slab, [(1, result["o"]), (2, result["lse"])]),
        ):
            desc = slab.descriptor()
            self.assertEqual(desc["base"], slab.wire.data_ptr())
            self.assertEqual(desc["size"], slab.wire.numel())
            for region, spec in entries:
                payload_bytes = dp.tensor_spec(spec, region)[2]
                offset = slab.offsets[region] + payload_bytes
                expected = (
                    bytes.fromhex(desc["nonce"])
                    + (1).to_bytes(8, "little")
                    + ((payload_bytes << 2) | region).to_bytes(8, "little")
                )
                self.assertLessEqual(offset + len(expected), desc["size"])
                self.assertEqual(
                    bytes(slab.wire[offset : offset + len(expected)].tolist()), expected
                )
        self.assertEqual(
            fixture.d.calls[0][3], [dp.tensor_spec(query_spec, 0)[2] + dp.FOOTER]
        )
        self.assertEqual(
            fixture.p.calls[0][3],
            [
                dp.tensor_spec(result[key], region)[2] + dp.FOOTER
                for region, key in ((1, "o"), (2, "lse"))
            ],
        )

    def test_packed_return_preserves_lse_bits_and_owned_outputs_across_shapes(self):
        """Compact LSE offsets follow actual O length, including batched frames."""
        fixture = self.fixture()
        client = fixture.dd.client(fixture.route)
        saved = []

        def compute(job, stream, cache):
            lse = torch.tensor([float("-inf"), -0.0, 1.25, float("nan")])
            return {}, job.q_cpu.clone(), lse

        with patch.object(dsv41_remote, "execute", compute):
            for count, width in ((1, 8), (4, 128), (2, 16), (1, 8)):
                q = [
                    torch.full((4, width), i + width, dtype=torch.bfloat16)
                    for i in range(count)
                ]
                _, out, lse = client.call(
                    q,
                    [{**fixture.route, "dsv41_sparse": True}] * count,
                    [NS() for _ in q],
                )
                expected_lse = torch.tensor(
                    [float("-inf"), -0.0, 1.25, float("nan")]
                ).repeat(count, 1)
                saved.append((out, torch.stack(q), lse, expected_lse))
                nbytes = (
                    out.numel() * out.element_size() + lse.numel() * lse.element_size()
                )
                self.assertEqual(fixture.p.calls[-1][3], [nbytes + dp.FOOTER])
                offset = dp.WIRE_OFFSETS[1] + nbytes
                stamp = client.slab.stamp(client.slab.nonce, client.sequence, 3, nbytes)
                self.assertEqual(
                    bytes(client.slab.wire[offset : offset + dp.FOOTER].tolist()), stamp
                )
        client.close()
        for out, expected, lse, expected_lse in saved:
            torch.testing.assert_close(out, expected)
            self.assertTrue(
                torch.equal(lse.view(torch.int32), expected_lse.view(torch.int32))
            )

    def test_packed_return_waits_for_combined_footer_after_sender_completion(self):
        """O bytes alone cannot publish a return whose LSE/footer are still in flight."""
        fixture = self.fixture()
        # Initialize the reused slot with a valid footer from the previous sequence.
        client, _, _ = self.call(fixture)
        fixture.p.failure = "delivery"
        req, results, errors = NS(), [], []
        waiting, deliver = threading.Event(), threading.Event()
        sleep = time.sleep

        def wait_for_delivery(delay):
            if threading.current_thread() is thread:
                # Gate only after receive rejected the previous frame's footer.
                waiting.set()
                if not deliver.wait(3):
                    raise TimeoutError("test return delivery gate")
            else:
                sleep(delay)

        def run():
            try:
                results.append(self.call(fixture, req, client))
            except Exception as exc:
                errors.append(exc)

        with patch.object(dp.time, "sleep", side_effect=wait_for_delivery):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(waiting.wait(3))
                self.assertFalse(results or errors)
                self.assertFalse(dp.ready(req))
                _, sources, targets, sizes = next(iter(fixture.p.pending.values()))
                self.assertEqual(len(sources), 1)
                # Deliver the payload, but leave the previous sequence's footer.
                ctypes.memmove(targets[0], sources[0], sizes[0] - dp.FOOTER)
                self.assertFalse(dp.ready(req))
                fixture.p.flush_late()
                deliver.set()
                thread.join(3)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors)
                self.assertEqual(len(results), 1)
                self.assertTrue(dp.ready(req))
            finally:
                if fixture.p.pending:
                    fixture.p.flush_late()
                deliver.set()
                thread.join(3)
        client.close()

    def test_failed_packed_return_retains_dma_backing_and_source_lease(self):
        """A failed single-region return must quarantine both ends just like RDMA."""
        fixture = self.fixture()
        fixture.p.failure = "late"
        client, req = fixture.dd.client(fixture.route), NS()
        with self.assertRaises(RuntimeError):
            self.call(fixture, req, client)
        self.assertEqual(len(fixture.p.calls[0][1]), 1)
        self.assertFalse(
            fixture.hub.registry.release("qo", expected_req=fixture.source)
        )
        before = dict(fixture.p.regions), dict(fixture.d.regions)
        self.assertTrue(before[0] and before[1])
        client.close()
        fixture.p.flush_late()
        self.assertEqual((fixture.p.regions, fixture.d.regions), before)
        self.assertFalse(fixture.freed)
        self.assertTrue(client.slab.quarantined)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)

    def test_failed_negotiation_without_slab_does_not_reuse_bad_socket(self):
        fixture = self.fixture()
        old = fixture.dd.client(fixture.route)
        original = old.rpc

        def bad_rpc(frame):
            if frame.get("cmd") == "te_negotiate":
                frame = {**frame, "version": "unknown"}
            return original(frame)

        old.rpc = bad_rpc
        req = NS()
        with self.assertRaises(RuntimeError):
            self.call(fixture, req, old)
        self.assertIsNone(old.slab)
        self.assertFalse(fixture.d.regions)
        new = fixture.dd.client(fixture.route)
        self.assertIsNot(new, old)
        self.call(fixture, client=new)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)
        new.close()

    def test_open_cannot_change_negotiated_transport_endpoint(self):
        fixture = self.fixture()
        client = fixture.dd.client(fixture.route)
        original = client.rpc

        def bad_rpc(frame):
            if frame.get("cmd") == "te_open":
                frame = {**frame, "session": "another-transport-session"}
            return original(frame)

        client.rpc = bad_rpc
        with self.assertRaises((RuntimeError, ValueError)):
            self.call(fixture, client=client)
        self.assertFalse(fixture.p.calls or fixture.d.calls)
        self.assertTrue(client.slab.quarantined)
        self.assertTrue(fixture.d.regions)

    def test_lost_open_reply_keeps_published_addresses_registered(self):
        fixture = self.fixture()
        client, req = fixture.dd.client(fixture.route), NS()
        original = client.rpc

        def lost_reply(frame):
            result = original(frame)
            if frame.get("cmd") == "te_open":
                raise OSError("injected open reply loss after registration")
            return result

        client.rpc = lost_reply
        with self.assertRaisesRegex(OSError, "reply loss"):
            self.call(fixture, req, client)
        self.wait(lambda: bool(fixture.pp.quarantine))
        before = dict(fixture.p.regions), dict(fixture.d.regions)
        self.assertTrue(before[0] and before[1])
        self.assertFalse(fixture.p.calls or fixture.d.calls)
        client.close()
        self.assertEqual((fixture.p.regions, fixture.d.regions), before)
        self.assertTrue(client.slab.quarantined)
        for slab in [*fixture.pp.quarantine, *fixture.dd.quarantine]:
            self.assertIn(slab, slab.engine._qo_slabs)
            self.assertIn(slab.wire.data_ptr(), slab.engine.native.regions)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)

    def test_q_producer_event_failure_retains_copy_input(self):
        fixture = self.fixture("rdma")
        client, req = fixture.dd.client(fixture.route), NS()
        client.connect()

        def fail():
            raise RuntimeError("injected producer event failure")

        with patch.object(
            client.slab, "record_producer", return_value=NS(synchronize=fail)
        ):
            with self.assertRaisesRegex(RuntimeError, "producer event"):
                self.call(fixture, req, client)
        (query,) = client.slab.copy_owner
        torch.testing.assert_close(
            query, torch.arange(16).reshape(2, 8).to(torch.bfloat16)
        )
        self.assertTrue(client.slab.quarantined)
        self.assertIn(client.slab, client.slab.engine._qo_slabs)
        self.assertFalse(fixture.d.calls)
        client.close()
        self.assertIsNotNone(client.slab.copy_owner)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)

    def test_o_lse_producer_event_failure_retains_compute_outputs_and_lease(self):
        fixture = self.fixture("rdma")
        client, req = fixture.dd.client(fixture.route), NS()
        client.connect()
        armed = threading.Event()
        real_producer = dp.Slab.record_producer

        def compute(job, stream, cache):
            armed.set()
            query = job.q_cpu
            return (
                {},
                query.float() * 3,
                torch.zeros(query.shape[0], dtype=torch.float32),
            )

        def gate_producer(slab):
            if not armed.is_set() or slab.plane is not fixture.pp:
                return real_producer(slab)

            def synchronize():
                raise RuntimeError("injected output event failure")

            return NS(synchronize=synchronize)

        with (
            patch.object(dsv41_remote, "execute", side_effect=compute),
            patch.object(dp.Slab, "record_producer", gate_producer),
        ):
            with self.assertRaisesRegex(RuntimeError, "output event"):
                self.call(fixture, req, client)
        self.wait(lambda: bool(fixture.pp.quarantine))
        slab = fixture.pp.quarantine[0]
        [(_, o, lse)] = slab.copy_owner
        self.assertEqual((o.shape, lse.shape), ((2, 8), (2,)))
        self.assertIn(slab, slab.engine._qo_slabs)
        self.assertFalse(fixture.p.calls)
        self.assertFalse(
            fixture.hub.registry.release("qo", expected_req=fixture.source)
        )
        self.assertFalse(fixture.freed)
        with self.assertRaisesRegex(RuntimeError, "unproven"):
            dp.ready(req)


if __name__ == "__main__":
    unittest.main()
