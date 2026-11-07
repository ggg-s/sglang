"""DSV adapters keep native scatter and auxiliary metadata lifetime contracts.

These CPU regressions run the production hook bodies and native lifecycle/address
logic. CUDA enqueue/event and Mooncake IO are boundaries; tensor scatter and the
byte-level metadata destination checks use real CPU storage.
"""

import ast
import ctypes
import functools
import logging
import struct
import sys
import threading
import time
import unittest
from collections import deque
from contextlib import contextmanager, nullcontext
from http import HTTPStatus
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
PLUGIN = ROOT / "plugins/remote_attn/sglang_remote_attn"
NATIVE = ROOT / "python/sglang/srt/disaggregation"


def definitions(path, names, namespace, *, owner=None):
    """Load source bodies without initializing the accelerator runtime."""
    tree = ast.parse(path.read_text())
    body = tree.body
    if owner is not None:
        body = next(
            node.body
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
    nodes = [
        node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing production definitions in {path}: {names}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    # Only the repository's reviewed source is executable test input here.
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def module(name, **attributes):
    value = ModuleType(name)
    vars(value).update(attributes)
    return value


class Stream:
    def __init__(self):
        self.fail_drain = False
        self.drains = 0
        self.waited = []

    def wait_event(self, event):
        self.waited.append(event)

    def synchronize(self):
        if self.fail_drain:
            raise RuntimeError("scatter stream drain failed")
        self.drains += 1


class Event:
    def __init__(self, *, fail_record=False):
        self.fail_record = fail_record
        self.done = False
        self.stream = None

    def record(self, stream):
        self.stream = stream
        if self.fail_record:
            raise RuntimeError("completion event record failed")

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


class TestRemoteMainABI(CustomTestCase):
    def scatter_fixture(self, fail_record):
        event, explicit, background = Event(fail_record=fail_record), Stream(), Stream()
        contexts, freed = [], []

        @contextmanager
        def stream_context(stream):
            contexts.append(stream)
            yield

        package = "sglang_remote_attn"
        native_buffer = "sglang.srt.disaggregation.common.staging_buffer"
        modules = {
            "sglang.srt.utils": module(
                "sglang.srt.utils",
                create_device_stream=lambda device: background,
                device_stream_context=stream_context,
            ),
            native_buffer: module(
                native_buffer, StagingAllocator=SimpleNamespace(CHUNK_SCATTERED=-3)
            ),
            package + ".destination_cache": module(
                package + ".destination_cache", lease_for=lambda req: None
            ),
            package + ".prefix_stream": module(
                package + ".prefix_stream", header=lambda offer: None
            ),
        }
        rx = definitions(
            PLUGIN / "dsv41_receive.py",
            {"scatter", "selected", "prefix_pages", "scatter_prefix", "teardown"},
            {"__package__": package},
        )
        ns = definitions(
            NATIVE / "common/staging_handler.py",
            {"DecodeStagingHandler"},
            {
                "torch": torch,
                "threading": threading,
                "deque": deque,
                "time": time,
                "create_device_event": lambda device: event,
                "logger": logging.getLogger(__name__),
            },
        )
        handler = object.__new__(ns["DecodeStagingHandler"])
        host = torch.arange(1, 9, dtype=torch.uint8)
        destination = torch.full((4, 4), 99, dtype=torch.uint8)
        allocations = {5: object()}

        def free(alloc_id):
            allocations.pop(alloc_id)
            freed.append(alloc_id)

        allocator = SimpleNamespace(
            total_size=8,
            _scatter_stream=background,
            allocations=allocations,
            buffer=SimpleNamespace(buffer=host, gpu_buffer=torch.zeros_like(host)),
            free=free,
            get_watermark=lambda: (0, 8),
            get_base_ptr=lambda: host.data_ptr(),
        )
        receiver = SimpleNamespace(
            chunk_staging_infos=[(5, 0, 0, 8, 2)],
            chunk_staging_layers={0: ((0,), 0, 2)},
            prefill_info=SimpleNamespace(attn_tp_size=1, pp_size=1),
        )
        req = SimpleNamespace(
            bootstrap_room=7,
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=0),
            origin_input_ids=[10, 11, 12, 13],
        )
        dr = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            _req_to_token_ready=object(),
            _chunk_events=[],
            _staging_layer_events={},
            _staging_completed_layers=[],
            _staging_scatter_streams=[],
            _staging_all_success=False,
        )
        manager = SimpleNamespace(
            _dsv41_layerwise_schema={"entries": [("index", 1, 0)]},
            kv_args=SimpleNamespace(page_size=2, kv_item_lens=[4]),
            _staging_ctx=SimpleNamespace(
                room_receivers={7: receiver}, room_bootstrap={7: []}
            ),
        )
        handler.kv_manager = manager
        handler.staging_allocator = allocator
        handler.kv_buffer_info = {"k_buffers": [destination]}
        handler.scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=torch.tensor([[2, 3, 6, 7]]))
        )
        handler.device, handler.gpu_id, handler.decode_tp = torch.device("cpu"), 0, 1
        handler.device_module = SimpleNamespace(set_device=lambda device: None)
        handler._room_to_decode_req, handler._room_to_receiver = {7: dr}, {7: receiver}
        handler._scatter_lock = threading.RLock()
        handler._staging_lifecycle = threading.Condition(handler._scatter_lock)
        handler._room_tokens = {7: object()}
        handler._inflight_submits, handler._claimed_chunks = {}, set()
        handler._closing_rooms, handler._writer_counts = set(), {}
        handler._pending_scatters = deque()
        handler._defer_scatter, handler._wm_subscribers = True, {}
        handler._scatter_region = functools.partial(
            rx["scatter"], type(handler)._scatter_region, handler
        )
        return SimpleNamespace(**locals())

    def release_queue(self, fixture):
        trace = []
        receiver = fixture.receiver
        receiver.clear = lambda: trace.append("receiver")
        receiver.kv_mgr = SimpleNamespace(
            clear_deferred_abort_state=lambda room: trace.append("abort-state")
        )
        queue = SimpleNamespace(
            enable_staging=True,
            staging_handler=fixture.handler,
            scheduler=SimpleNamespace(
                request_retirer=SimpleNamespace(
                    release=lambda req, **kw: trace.append("kv")
                )
            ),
            metadata_buffers=SimpleNamespace(
                clear_slot=lambda idx: trace.append("metadata-clear")
            ),
            req_to_metadata_buffer_idx_allocator=SimpleNamespace(
                free=lambda idx: trace.append("metadata-free")
            ),
            _is_host_destination=lambda req: False,
        )
        return queue, trace

    def test_failed_drain_retains_room_for_repeated_release_and_rejects_reuse(self):
        """A failed drain must not let the next release skip the GPU fence."""
        f = self.scatter_fixture(False)
        release = definitions(
            NATIVE / "decode.py", {"_do_release"}, {}, owner="DecodeTransferQueue"
        )["_do_release"]
        queue, trace = self.release_queue(f)
        with patch.dict(sys.modules, f.modules):
            f.handler.submit_chunk_scatter(7, 0, 0, 2, stream=f.explicit)
        events = f.dr._chunk_events
        ready = f.dr._req_to_token_ready
        f.explicit.fail_drain = True
        for attempt in range(2):
            with self.subTest(release_attempt=attempt):
                with self.assertRaisesRegex(RuntimeError, "drain failed"):
                    release(queue, f.dr, 3)
                self.assertTrue(f.handler.is_staging_room(7))
                self.assertIs(f.handler._room_to_receiver[7], f.receiver)
                self.assertNotIn(7, f.handler._room_tokens)
                self.assertEqual(trace, [])
                self.assertEqual(f.freed, [])
                self.assertIs(f.dr._chunk_events, events)
                self.assertEqual(events, [(f.event, 5)])
                # The same object must not lose events before the duplicate
                # registration check; a fresh object must not steal its room.
                replacement = SimpleNamespace(req=f.req, kv_receiver=f.receiver)
                for candidate in (f.dr, replacement):
                    with self.assertRaisesRegex(
                        RuntimeError, "still owns receive storage"
                    ):
                        f.handler.register_decode_req(7, candidate)
                self.assertIs(f.dr._chunk_events, events)
                self.assertIs(f.dr._req_to_token_ready, ready)
                self.assertFalse(hasattr(replacement, "_chunk_events"))
                with patch.dict(sys.modules, f.modules):
                    self.assertFalse(
                        f.handler.submit_chunk_scatter(7, 0, 0, 2, stream=f.explicit)
                    )
        f.explicit.fail_drain = False
        release(queue, f.dr, 3)
        self.assertEqual(f.explicit.drains, 1)
        self.assertEqual(f.freed, [5])
        self.assertFalse(f.handler.is_staging_room(7))
        self.assertEqual(f.manager._staging_ctx.room_receivers, {})
        self.assertEqual(f.manager._staging_ctx.room_bootstrap, {})
        self.assertEqual(
            trace, ["kv", "metadata-clear", "metadata-free", "abort-state", "receiver"]
        )
        self.assertIsNone(f.dr.kv_receiver)

    def test_live_staging_registration_is_idempotent_without_losing_scatter_proofs(
        self,
    ):
        f = self.scatter_fixture(False)
        with patch.dict(sys.modules, f.modules):
            f.handler.submit_chunk_scatter(7, 0, 0, 2, stream=f.explicit)
        events, ready = f.dr._chunk_events, f.dr._req_to_token_ready
        token = f.handler._room_tokens[7]
        f.dr._staging_success_ts = 123.0
        f.handler.register_decode_req(7, f.dr)
        self.assertIs(f.dr._chunk_events, events)
        self.assertEqual(events, [(f.event, 5)])
        self.assertIs(f.dr._req_to_token_ready, ready)
        self.assertIs(f.handler._room_tokens[7], token)
        self.assertEqual(f.dr._staging_success_ts, 123.0)
        with self.assertRaisesRegex(RuntimeError, "still owns receive storage"):
            f.handler.register_decode_req(
                7, SimpleNamespace(req=f.req, kv_receiver=f.receiver)
            )

    def test_first_dsv_registration_publishes_aux_strides_without_customized_info(self):
        native = definitions(
            NATIVE / "mooncake/conn.py",
            {"build_receiver_registration_message"},
            {"struct": struct, "pack_int_lists": lambda values, fmt: b""},
            owner="MooncakeKVManager",
        )["build_receiver_registration_message"]
        hook = definitions(
            PLUGIN / "dsv41_receive.py", {"registration_message"}, {"struct": struct}
        )["registration_message"]
        manager = SimpleNamespace(
            kv_args=SimpleNamespace(
                kv_data_ptrs=[10],
                aux_data_ptrs=[20, 30],
                aux_item_lens=[8, 6],
                state_data_ptrs=[],
                state_item_lens=[],
                state_layer_ids=[],
                engine_rank=0,
                kv_item_lens=[4],
                kv_layer_ids=[0],
            ),
            get_session_id=lambda: "decoder",
            local_ip="127.0.0.1",
            rank_port=1,
            attn_tp_size=1,
            dcp_size=1,
            dcp_rank=0,
            enable_staging=False,
            customized_info_enabled=False,
        )
        self.assertEqual(hook(native, manager)[6], b"")
        manager._dsv41_layerwise_schema = {"entries": [("kv",)]}
        message = hook(native, manager)
        self.assertEqual(struct.unpack("2Q", message[6]), (8, 6))
        self.assertEqual(message[:6], native(manager)[:6])
        self.assertEqual(message[7:], native(manager)[7:])
        manager.kv_args.aux_item_lens = [8]
        with self.assertRaisesRegex(ValueError, "registration strides"):
            hook(native, manager)

    def test_failed_transfer_without_hold_drains_before_native_kv_retirement(self):
        """The no-ACK-hold failure path must fence scatter before freeing KV."""
        f = self.scatter_fixture(False)
        queue, trace = self.release_queue(f)
        poll = SimpleNamespace(Failed=1, Success=2)
        consume = definitions(
            NATIVE / "decode.py",
            {"pop_transferred_with_outcomes"},
            {
                "torch": torch,
                "KVPoll": poll,
                "HiCacheRestoreResult": SimpleNamespace(FAILED=-1),
                "HTTPStatus": HTTPStatus,
                "logger": logging.getLogger(__name__),
                "prepare_abort": lambda *a, **kw: None,
                "all_reduce_flags": lambda flags, **kw: flags,
            },
            owner="DecodeTransferQueue",
        )["pop_transferred_with_outcomes"]
        pending = [f.dr]
        queue.queue = SimpleNamespace(
            items=lambda: iter(pending),
            leave=lambda reqs: pending.clear(),
        )
        queue.scheduler.enable_decode_hicache = False
        queue.scheduler.output_streamer = SimpleNamespace(stream_output=lambda *a: None)
        queue.enable_deferred_kv_release, queue.tp_rank, queue.gloo_group = (
            False,
            0,
            None,
        )
        queue._poll_with_staging = lambda: [poll.Failed]
        queue._abort_failed_transfer = lambda req: None
        queue._clean_hicache_prefetch_resources = lambda req: None
        queue._record_transfer_failure = lambda reason: None
        queue.publish_pending_metadata = lambda polls: None
        f.req.rid, f.req.return_logprob = "failed", False
        f.dr.hicache_restore_status, f.dr.metadata_buffer_index = 0, 3
        f.receiver.failure_exception = lambda: None
        with patch.dict(sys.modules, f.modules):
            f.handler.submit_chunk_scatter(7, 0, 0, 2, stream=f.explicit)
        f.explicit.fail_drain = True
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            consume(queue)
        self.assertEqual(trace, [])
        self.assertEqual(f.freed, [])
        self.assertIs(f.dr.kv_receiver, f.receiver)
        self.assertEqual(pending, [f.dr])
        self.assertTrue(f.handler.is_staging_room(7))
        f.explicit.fail_drain = False
        self.assertEqual(consume(queue), ([], {"failed": True}))
        self.assertEqual(f.freed, [5])
        self.assertFalse(f.handler.is_staging_room(7))
        self.assertEqual(trace, ["kv", "receiver", "metadata-clear", "metadata-free"])
        self.assertEqual(pending, [])

    def test_dsv_explicit_stream_proves_entries_and_retains_failed_publication(self):
        """An event on forward stream cannot certify a private-stream KV write."""
        for fail_record, fail_drain in ((False, False), (True, False), (True, True)):
            with self.subTest(fail_record=fail_record, fail_drain=fail_drain):
                f = self.scatter_fixture(fail_record)
                with patch.dict(sys.modules, f.modules):
                    if fail_record:
                        with self.assertRaisesRegex(RuntimeError, "event record"):
                            f.handler.submit_chunk_scatter(
                                7, 0, 0, 2, stream=f.explicit
                            )
                    else:
                        self.assertTrue(
                            f.handler.submit_chunk_scatter(
                                7, 0, 0, 2, stream=f.explicit
                            )
                        )
                self.assertEqual(f.contexts, [f.explicit])
                self.assertIs(f.event.stream, f.explicit)
                self.assertEqual(f.explicit.waited, [f.dr._req_to_token_ready])
                expected = torch.full((4, 4), 99, dtype=torch.uint8)
                expected[torch.tensor([1, 3])] = f.host.reshape(2, 4)
                torch.testing.assert_close(f.destination, expected)
                self.assertEqual(f.dr._staging_scatter_streams, [f.explicit])
                f.handler.advance_scatter(f.dr)
                self.assertEqual(f.dr._staging_completed_layers, [])
                self.assertEqual(f.freed, [])
                if fail_record:
                    # Both the allocation and explicit stream survive a failed
                    # completion publication; teardown cannot deregister them.
                    f.manager._staging_handler = f.handler
                    owners = [(f.manager, f.allocator, [f.destination])]
                    f.manager.engine = SimpleNamespace(_dsv41_receive_owners=owners)
                    with self.assertRaisesRegex(RuntimeError, "ring still active"):
                        f.rx["teardown"](lambda manager: None, f.manager)
                    self.assertEqual(len(owners), 1)
                    f.explicit.fail_drain = fail_drain
                    if fail_drain:
                        with self.assertRaisesRegex(RuntimeError, "drain failed"):
                            f.handler.release_room(7, f.dr, f.receiver)
                        self.assertIn(5, f.allocations)
                        self.assertEqual(f.freed, [])
                        self.assertEqual(f.dr._staging_scatter_streams, [f.explicit])
                    f.explicit.fail_drain = False
                    f.handler.release_room(7, f.dr, f.receiver)
                    self.assertEqual(f.explicit.drains, 1)
                    self.assertEqual(f.dr._staging_scatter_streams, [])
                else:
                    f.event.done = True
                    f.handler.advance_scatter(f.dr)
                    self.assertEqual(f.dr._staging_completed_layers, [((0,), 0, 2)])
                self.assertEqual(f.freed, [5])
                self.assertEqual(f.allocations, {})

    def test_progressive_aux_uses_destination_rows_and_rejects_overflow(self):
        """Different P/D strides must select the D metadata row, not its neighbor."""
        package = "sglang_remote_attn"
        progressive = definitions(
            PLUGIN / "dsv41_progressive.py",
            {"send_metadata", "send_aux"},
            {"__package__": package},
        )
        native = definitions(
            NATIVE / "mooncake/conn.py",
            {"send_aux"},
            {
                "envs": SimpleNamespace(
                    SGLANG_MOONCAKE_SEND_AUX_TCP=SimpleNamespace(get=lambda: False)
                )
            },
            owner="MooncakeKVManager",
        )["send_aux"]
        src = [
            torch.arange(1, 13, dtype=torch.uint8),
            torch.arange(21, 27, dtype=torch.uint8),
        ]
        dst = [
            torch.full((24,), 99, dtype=torch.uint8),
            torch.full((18,), 99, dtype=torch.uint8),
        ]
        writes = []

        def transfer(session, blocks):
            for source, target, length in blocks:
                ctypes.memmove(target, source, length)
            writes.append((session, blocks))
            return 0

        manager = SimpleNamespace(
            kv_args=SimpleNamespace(
                aux_data_ptrs=[t.data_ptr() for t in src], aux_item_lens=[4, 2]
            ),
            enable_custom_mem_pool=False,
            _transfer_data=transfer,
        )
        manager.send_aux = functools.partial(progressive["send_aux"], native, manager)
        peer = SimpleNamespace(mooncake_session_id="decoder", dst_aux_index=2)
        pointers, strides = [t.data_ptr() for t in dst], [8, 6]
        priority = module(
            package + ".transfer_priority", control=lambda manager: nullcontext()
        )
        with patch.dict(sys.modules, {priority.__name__: priority}):
            self.assertEqual(
                progressive["send_metadata"](manager, peer, 1, pointers, strides), 0
            )
            expected = [torch.full_like(t, 99) for t in dst]
            expected[0][16:20] = src[0][4:8]
            expected[1][12:14] = src[1][2:4]
            for actual, wanted in zip(dst, expected):
                torch.testing.assert_close(actual, wanted)
            self.assertEqual(len(writes), 1)
            for invalid in (None, [8], [3, 6], [8, 0]):
                with self.subTest(strides=invalid):
                    with self.assertRaisesRegex(ValueError, "destination strides"):
                        progressive["send_metadata"](
                            manager, peer, 1, pointers, invalid
                        )
                    self.assertEqual(len(writes), 1)
            # The canonical marker must not repeat the already-committed DMA.
            manager._dsv41_progressive_sources = {
                7: SimpleNamespace(
                    barrier_ready=True,
                    final_payload=(None, None, {"aux_index": 1}),
                    peers=[(peer, object())],
                )
            }
            self.assertEqual(manager.send_aux(peer, 1, pointers, strides), 0)
            self.assertEqual(len(writes), 1)


if __name__ == "__main__":
    unittest.main()
