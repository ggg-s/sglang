"""Transport boundary contracts; CUDA/RDMA hardware tests live under manual/."""

import ast
import ctypes
import json
import os
import socketserver
import struct
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "plugins/remote_attn"))

from sglang.srt import mooncake_contract as contract
from sglang_remote_attn import d2d_bulk as bulk
from sglang_remote_attn import mooncake_backend as mb
from sglang_remote_attn import remote_data_plane as dp


class Native:
    d2d_protocol_verified = True

    def __init__(self, protocol="tcp", name="peer"):
        self.protocol, self.name = protocol, name
        self.regions, self.writes = {}, []
        self.fail_register = self.fail_unregister = self.fail_write = False

    def get_session_id(self):
        return self.name

    def batch_register(self, pointers, sizes):
        self.regions.update(zip(pointers, sizes))
        return -1 if self.fail_register else 0

    def batch_deregister(self, pointers):
        if self.fail_unregister:
            return -1
        for ptr in pointers:
            self.regions.pop(ptr)
        return 0

    def batch_transfer_sync(self, session, sources, targets, sizes):
        self.writes.append((session, sources, targets, sizes))
        for src, dst, size in zip(sources, targets, sizes):
            ctypes.memmove(dst, src, size)
        return -1 if self.fail_write else 0


class TestMooncakeBackend(unittest.TestCase):
    def setUp(self):
        for key in ("_HANDLES", "_TCP", "_PROBES"):
            self.enterContext(patch.object(mb, key, {}))
        self.enterContext(patch.object(mb, "_RETAINED", []))
        self.enterContext(patch.dict(os.environ, {"HANDOFF_D2D_BACKEND": "auto"}))

    def test_native_rank_engine_and_raw_identity_are_shared(self):
        raw = NS()
        native = Native("rdma")
        native.engine = raw
        module = NS(get_mooncake_transfer_engine=lambda: native)
        key = "sglang.srt.distributed.device_communicators.mooncake_transfer_engine"
        with (
            patch.dict(sys.modules, {key: module}),
            patch.object(mb, "tcp_engine") as tcp,
        ):
            self.assertIs(mb.shared_raw(), raw)
            self.assertIs(mb.handle(raw), mb.handle(native))
            self.assertEqual(mb.handle(raw).protocol, "rdma")
            tcp.assert_not_called()

    def test_effective_protocol_not_mutable_environment(self):
        native = Native("tcp")
        with (
            patch.dict(os.environ, {"MOONCAKE_PROTOCOL": "rdma"}),
            patch.object(mb, "tcp_engine") as factory,
        ):
            cat = mb.Catalog(native, "cpu")
            self.assertEqual(cat.offer["backends"], [mb.TCP])
            factory.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "identity"):
            mb.remember(native, "rdma", "peer")

    def test_gpu_intersection_and_pretransfer_tcp_fallback(self):
        native, tcp = Native("rdma"), mb.handle(Native("tcp", "fallback"))
        with (
            patch.object(mb, "gpu_capability", return_value=True),
            patch.object(mb, "tcp_engine", return_value=tcp),
        ):
            cat = mb.Catalog(native, "cuda:0")
            self.assertEqual(cat.select(cat.offer), mb.GPU)
            self.assertEqual(cat.select(mb.tcp_offer()), mb.TCP)
            self.assertIs(cat.engines[mb.GPU], mb.handle(native))
            self.assertIs(cat.engines[mb.TCP], tcp)

    def test_strict_rdma_rejects_unavailable_and_incompatible_peer(self):
        with patch.dict(os.environ, {"HANDOFF_D2D_BACKEND": "rdma"}):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                mb.Catalog(Native(), "cpu")
            with patch.object(mb, "gpu_capability", return_value=True):
                cat = mb.Catalog(Native("rdma"), "cuda:0")
                with self.assertRaisesRegex(ValueError, "common"):
                    cat.select(mb.tcp_offer())

    def test_capability_schema_rejects_unknown_duplicate_and_empty(self):
        for offer in (
            None,
            {},
            {"version": "d2d-mooncake-v1", "backends": [mb.GPU, mb.TCP]},
            {"version": mb.VERSION, "backends": []},
            {"version": mb.VERSION, "backends": [mb.TCP, mb.TCP]},
            {"version": mb.VERSION, "backends": ["rdma"]},
        ):
            with self.subTest(offer=offer), self.assertRaises(ValueError):
                mb.validate(offer)

    def probe(self, native, visibility=None):
        original = torch.empty

        def empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original(*args, **kwargs)

        with (
            patch.object(torch, "empty", side_effect=empty),
            patch.object(dp, "Visibility", visibility or Mock()),
        ):
            return mb.gpu_capability(mb.handle(native), "cuda:0")

    def test_registration_probe_success_is_cached_and_releases_only_after_unregister(
        self,
    ):
        native = Native("rdma")
        self.assertTrue(self.probe(native))
        self.assertFalse(native.regions)
        self.assertFalse(mb._RETAINED)
        native.fail_register = True
        self.assertTrue(self.probe(native))

    def test_uncertain_registration_probe_retains_backing_and_disables_gpu(self):
        native = Native("rdma")
        native.fail_register = True
        self.assertFalse(self.probe(native))
        self.assertTrue(native.regions)
        self.assertEqual(len(mb._RETAINED), 1)
        self.assertFalse(self.probe(native))
        self.assertEqual(len(mb._RETAINED), 1)

    def test_unregister_probe_failure_retains_gpu_memory(self):
        native = Native("rdma")
        native.fail_unregister = True
        self.assertFalse(self.probe(native))
        self.assertEqual(len(mb._RETAINED), 1)
        self.assertTrue(native.regions)

    def test_unsupported_visibility_does_not_register(self):
        native = Native("rdma")
        self.assertFalse(
            self.probe(native, Mock(side_effect=RuntimeError("no ordering")))
        )
        self.assertFalse(native.regions)
        self.assertFalse(mb._RETAINED)

    def test_unverified_transport_policy_cannot_offer_gpu(self):
        native = Native("rdma")
        native.d2d_rdma_eligible = False
        self.assertFalse(self.probe(native))
        self.assertFalse(native.regions)

    def test_unknown_engine_is_not_mislabelled_tcp(self):
        native = NS(get_session_id=lambda: "unknown")
        self.assertEqual(mb.handle(native).protocol, "unsupported")

    def test_forced_other_transport_cannot_be_advertised_as_tcp_fallback(self):
        for name in contract.NON_CLASSIC:
            for value in ("1", "0", "", "true"):
                with (
                    self.subTest(name=name, value=value),
                    patch.dict(os.environ, {name: value}, clear=True),
                ):
                    raw = NS(initialize=Mock())
                    with self.assertRaisesRegex(RuntimeError, "classic"):
                        contract.initialize_engine(
                            raw, "host", "tcp", "", force_tcp=True
                        )
                    raw.initialize.assert_not_called()

    def native_initializer(self):
        path = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py"
        )
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "MooncakeTransferEngine"
        )
        fn = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "initialize"
        )
        env = NS(
            ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=NS(get=lambda: False),
            MOONCAKE_PROTOCOL=NS(get=lambda: "rdma"),
            SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT=NS(get=lambda: 1),
        )
        scope = dict(
            os=os,
            Optional=__import__("typing").Optional,
            envs=env,
            run_with_deadline=lambda fn, **kw: fn(),
        )
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
        return scope["initialize"]

    def test_native_initialization_records_actual_not_requested_protocol(self):
        initialize = self.native_initializer()
        for value in ("1", "0", "", "true"):
            raw = NS(initialize=Mock(return_value=0))
            native = NS(engine=raw)
            with (
                patch.dict(os.environ, {"MC_FORCE_TCP": value}, clear=True),
                patch.object(contract, "local_protocol", return_value="tcp"),
            ):
                initialize(native, "host", "mlx5_0")
            self.assertEqual(native.protocol, "tcp")
            self.assertTrue(native.d2d_protocol_verified)
            self.assertFalse(native.d2d_rdma_eligible)
        # An unforced RDMA request can also auto-discover TCP (no HCA).
        for actual in ("tcp", "rdma"):
            native = NS(engine=NS(initialize=Mock(return_value=0)))
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(contract, "local_protocol", return_value=actual),
            ):
                initialize(native, "host", "")
            self.assertEqual(native.protocol, actual)
            self.assertEqual(native.d2d_rdma_eligible, actual == "rdma")

    def test_unknown_metadata_and_all_tent_aliases_fail_closed(self):
        initialize = self.native_initializer()
        for knobs in (
            {},
            *({name: value} for name in contract.NON_CLASSIC for value in ("0", "")),
        ):
            native = NS(engine=NS(initialize=Mock(return_value=0)))
            with (
                patch.dict(os.environ, knobs, clear=True),
                patch.object(
                    contract, "local_protocol", side_effect=RuntimeError("unknown wire")
                ) as query,
            ):
                initialize(native, "host", "")
            self.assertEqual(native.protocol, "unsupported")
            self.assertFalse(native.d2d_rdma_eligible)
            self.assertFalse(native.d2d_protocol_verified)
            if knobs:
                query.assert_not_called()

    def test_requested_protocol_without_verification_cannot_offer_gpu(self):
        raw = NS(protocol="rdma", get_session_id=lambda: "unverified")
        self.assertEqual(mb.handle(raw).protocol, "unsupported")
        self.assertFalse(self.probe(raw))

    def test_tcp_factory_forces_tcp_despite_hca_and_restores_environment(self):
        class Raw:
            def initialize(self, *args):
                self.actual = "tcp" if "MC_FORCE_TCP" in os.environ else "rdma"
                return 0

            def get_rpc_port(self):
                return 12345

        for previous in (None, "0", ""):
            with (
                self.subTest(previous=previous),
                patch.object(mb, "_TCP", {}),
                patch.dict(
                    os.environ,
                    {} if previous is None else {"MC_FORCE_TCP": previous},
                    clear=True,
                ),
                patch.dict(sys.modules, {"mooncake.engine": NS(TransferEngine=Raw)}),
                patch.object(
                    contract, "local_protocol", side_effect=lambda raw, host: raw.actual
                ),
            ):
                result = mb.tcp_engine()
                self.assertEqual(result.raw.actual, "tcp")
                self.assertEqual(result.protocol, "tcp")
                self.assertIs(result, mb.tcp_engine())
                self.assertEqual(os.environ.get("MC_FORCE_TCP"), previous)

    def test_tcp_factory_rejects_wrong_actual_transport_without_caching(self):
        raw = NS(initialize=Mock(return_value=0), get_rpc_port=lambda: 12345)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict(
                sys.modules, {"mooncake.engine": NS(TransferEngine=lambda: raw)}
            ),
            patch.object(contract, "local_protocol", return_value="rdma"),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not be verified"):
                mb.tcp_engine()
            self.assertNotIn("MC_FORCE_TCP", os.environ)
            self.assertFalse(mb._TCP)
            self.assertIn(raw, mb._RETAINED)

    def test_tcp_initialization_exception_restores_override(self):
        raw = NS(initialize=Mock(side_effect=RuntimeError("init failed")))
        with patch.dict(os.environ, {"MC_FORCE_TCP": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "init failed"):
                contract.initialize_engine(raw, "host", "tcp", "", force_tcp=True)
            self.assertEqual(os.environ["MC_FORCE_TCP"], "0")

    def test_concurrent_native_init_cannot_observe_tcp_override(self):
        entered, release, second_entered = (threading.Event() for _ in range(3))
        results, errors = [], []

        class Raw:
            def __init__(self, first):
                self.first = first

            def initialize(self, *args):
                if self.first:
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError("test gate")
                else:
                    second_entered.set()
                self.actual = "tcp" if "MC_FORCE_TCP" in os.environ else "rdma"
                return 0

        def run(first):
            try:
                results.append(
                    contract.initialize_engine(
                        Raw(first), "host", "rdma", "", force_tcp=first
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                contract, "local_protocol", side_effect=lambda raw, host: raw.actual
            ),
        ):
            first, second = (
                threading.Thread(target=run, args=(True,)),
                threading.Thread(target=run, args=(False,)),
            )
            first.start()
            self.assertTrue(entered.wait(2))
            second.start()
            try:
                self.assertFalse(second_entered.wait(0.05))
            finally:
                release.set()
                first.join(3)
                second.join(3)
            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertNotIn("MC_FORCE_TCP", os.environ)
        self.assertFalse(errors)
        self.assertCountEqual(results, [(0, "tcp"), (0, "rdma")])

    def query_metadata(self, response, *, kind=1, announced=None):
        class Handler(socketserver.BaseRequestHandler):
            def handle(handler):
                header = handler.request.recv(8)
                self.assertEqual(struct.unpack("=Q", header)[0], 3)
                self.assertEqual(handler.request.recv(3), b"\x01{}")
                value = dict(response) if isinstance(response, dict) else response
                if isinstance(value, dict):
                    value.setdefault("name", f"127.0.0.1:{server.server_address[1]}")
                payload = bytes([kind]) + json.dumps(value).encode()
                # Fragment headers/payload: no recv call may assume a full frame.
                frame = (
                    struct.pack("=Q", len(payload) if announced is None else announced)
                    + payload
                )
                for byte in frame:
                    handler.request.sendall(bytes([byte]))

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.handle_request)
            thread.start()
            try:
                return contract.local_protocol(
                    NS(get_rpc_port=lambda: server.server_address[1]), "127.0.0.1"
                )
            finally:
                thread.join(3)

    def test_local_metadata_query_uses_actual_single_transport(self):
        for protocol in ("tcp", "rdma"):
            self.assertEqual(self.query_metadata(dict(protocol=protocol)), protocol)

    def test_local_metadata_query_rejects_unknown_multi_and_wrong_identity(self):
        for response in (
            {},
            {"protocol": "efa"},
            {"protocol": ["rdma", "tcp"]},
            {"protocol": "rdma", "protocols": ["rdma", "tcp"]},
            {"protocol": "rdma", "name": "other:1"},
            [],
        ):
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                self.query_metadata(response)

    def test_single_submit_never_calls_hidden_retry_sync_api(self):
        late = []
        raw = NS(
            batch_transfer_sync_write=Mock(side_effect=AssertionError("hidden retry")),
            batch_transfer_async_write=Mock(
                side_effect=lambda *args: late.append(args) or 7
            ),
            get_batch_transfer_status=Mock(return_value=-1),
        )
        native = Native("rdma")
        native.engine = raw
        engine = mb.handle(native)
        with self.assertRaisesRegex(RuntimeError, "completion is unknown"):
            engine.batch_transfer_sync("peer", [11, 12], [21, 22], [8, 8])
        raw.batch_transfer_async_write.assert_called_once()
        raw.get_batch_transfer_status.assert_called_once_with([7])
        raw.batch_transfer_sync_write.assert_not_called()
        self.assertEqual(len(late), 1)  # Failed batch may still access these pointers.

    def test_single_submit_success_waits_same_ticket_and_checks_sentinels(self):
        raw = NS(
            batch_transfer_async_write=Mock(return_value=19),
            get_batch_transfer_status=Mock(return_value=0),
        )
        self.assertEqual(contract.write_once(raw, "peer", [1], [2], [8]), 0)
        raw.get_batch_transfer_status.assert_called_once_with([19])
        for invalid in (0, -1, 2**64 - 1, None, True):
            raw.batch_transfer_async_write.return_value = invalid
            raw.get_batch_transfer_status.reset_mock()
            with self.assertRaisesRegex(RuntimeError, "submission"):
                contract.write_once(raw, "peer", [1], [2], [8])
            raw.get_batch_transfer_status.assert_not_called()

    def test_missing_async_api_fails_before_native_sync_submission(self):
        raw = NS(batch_transfer_sync_write=Mock(return_value=0))
        with self.assertRaisesRegex(RuntimeError, "single-submit"):
            contract.write_once(raw, "peer", [1], [2], [8])
        raw.batch_transfer_sync_write.assert_not_called()

    def test_gpu_receiver_visibility_precedes_owned_copy_and_slot_reuse(self):
        # CPU memory is an explicit fake NIC/CUDA boundary for ordering checks.
        native, req = Native("rdma"), NS()
        visibility = NS(receive=Mock())
        with patch.object(dp, "Visibility", return_value=visibility):
            receiver = bulk.Receiver(native, req, "cpu", backend=mb.GPU)
        region = receiver.prepare()
        receiver.tensor[:8] = torch.arange(8, dtype=torch.uint8)
        header = dict(
            bulk_protocol=mb.GPU, bulk_nonce=region["nonce"], bulk_sequence=1, bytes=8
        )
        result = receiver.consume(header, 8, 0)
        visibility.receive.assert_called_once()
        receiver.tensor.zero_()
        self.assertEqual(result.tolist(), list(range(8)))
        event = Mock()
        receiver.consumed = event
        receiver.prepare()
        event.synchronize.assert_called_once()
        with self.assertRaisesRegex(ValueError, "completion"):
            receiver.consume(header, 8, 0)
        receiver.close()
        self.assertFalse(native.regions)

    def test_gpu_visibility_failure_cannot_publish_payload(self):
        native = Native("rdma")
        with patch.object(
            dp,
            "Visibility",
            return_value=NS(receive=Mock(side_effect=RuntimeError("flush failed"))),
        ):
            receiver = bulk.Receiver(native, NS(), "cpu", backend=mb.GPU)
        region = receiver.prepare()
        header = dict(
            bulk_protocol=mb.GPU, bulk_nonce=region["nonce"], bulk_sequence=1, bytes=8
        )
        with self.assertRaisesRegex(RuntimeError, "flush"):
            receiver.consume(header, 8, 0)
        self.assertIsNone(receiver.consumed)
        receiver.poison()  # Reader's exception path always poisons this slot.
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            receiver.close()
        self.assertTrue(native.regions)


if __name__ == "__main__":
    unittest.main()
