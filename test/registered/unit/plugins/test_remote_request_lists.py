"""Remote migration keeps native ReqList membership and queue ownership."""

import ast
import importlib.util
import logging
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
PLUGIN = ROOT / "plugins/remote_attn/sglang_remote_attn"


def load_function(path, name, namespace):
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def request_lists():
    path = ROOT / "python/sglang/srt/managers/scheduler_components/request_list.py"
    spec = importlib.util.spec_from_file_location("native_request_list_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Request:
    def __init__(self, rid):
        self.rid = rid


class TestRemoteRequestLists(CustomTestCase):
    def setUp(self):
        self.native = request_lists()
        self.namespace = {}
        path = PLUGIN / "handoff_control.py"
        for name in ("_join_waiting", "_leave_waiting", "_remove_from_batches"):
            load_function(path, name, self.namespace)
        self.compat = ModuleType("sglang_remote_attn.compat")
        for name in ("decode_queue_items", "remove_decode_items"):
            load_function(PLUGIN / "compat.py", name, vars(self.compat))

    def test_admit_and_remove_use_native_methods_without_replacing_queue(self):
        calls = []

        class NativeQueue(self.native.IndexedReqList):
            def join(self, reqs):
                calls.append("join")
                super().join(reqs)

            def leave(self, reqs):
                calls.append("leave")
                super().leave(reqs)

        queue = NativeQueue(self.native.ListKind.WAITING)
        scheduler = SimpleNamespace(waiting_queue=queue)
        first, second = Request("first"), Request("second")
        self.namespace["_join_waiting"](scheduler, [first, second])
        self.namespace["_leave_waiting"](scheduler, first)
        self.assertIs(scheduler.waiting_queue, queue)
        self.assertEqual(list(queue), [second])
        self.assertEqual(calls, ["join", "leave"])

    def test_freeze_removes_running_last_and_waiting_membership(self):
        first, second = Request("first"), Request("second")
        queue = self.native.IndexedReqList(self.native.ListKind.WAITING)
        queue.join([first, second])

        class Batch:
            def __init__(self):
                self.reqs = [first, second]
                self.batch_is_full = True

            def filter_batch(self, *, keep_indices):
                self.reqs = [self.reqs[i] for i in keep_indices]

        scheduler = SimpleNamespace(
            waiting_queue=queue, running_batch=Batch(), last_batch=Batch()
        )
        self.namespace["_remove_from_batches"](scheduler, first)
        self.assertIs(scheduler.waiting_queue, queue)
        self.assertEqual(list(queue), [second])
        for batch in (scheduler.running_batch, scheduler.last_batch):
            self.assertEqual(batch.reqs, [second])
            self.assertFalse(batch.batch_is_full)

    def test_standalone_list_fixture_retains_container_identity(self):
        queue = []
        scheduler = SimpleNamespace(waiting_queue=queue)
        first, second = Request("first"), Request("second")
        self.namespace["_join_waiting"](scheduler, [first, second])
        self.namespace["_leave_waiting"](scheduler, first)
        self.assertIs(scheduler.waiting_queue, queue)
        self.assertEqual(queue, [second])

    def test_early_admission_keeps_decode_wrappers_and_native_queue_owner(self):
        # Native iteration exposes bare Req objects; early admission must retain
        # the wrappers carrying their receiver and metadata ownership instead.
        # Equal public IDs also must not collapse distinct request attempts.
        first, second = Request("retry"), Request("retry")
        wrappers = [
            SimpleNamespace(req=req, kv_receiver=object(), metadata_buffer_index=i)
            for i, req in enumerate((first, second))
        ]
        pending = self.native.IndexedReqList(
            self.native.ListKind.PD_RECEIVING, req_of=lambda item: item.req
        )
        pending.join(wrappers)
        queue = SimpleNamespace(queue=pending)
        self.assertEqual(list(pending), [first, second])
        offered = self.compat.decode_queue_items(queue)
        self.assertEqual(offered, tuple(wrappers))
        self.compat.remove_decode_items(queue, {id(offered[0])})
        self.assertIs(queue.queue, pending)
        self.assertEqual(list(pending), [second])
        self.assertEqual(list(pending.items()), [wrappers[1]])
        # Already admitted D2D targets may not belong to the native PD queue.
        self.compat.remove_decode_items(queue, {id(wrappers[0])})
        self.assertEqual(list(pending.items()), [wrappers[1]])

    def test_transport_quarantine_removes_wrapper_but_retains_native_owner_and_io(self):
        failed, healthy = Request("failed"), Request("healthy")
        failed.finished = lambda: True
        receiver = object()
        wrappers = [
            SimpleNamespace(req=req, kv_receiver=receiver) for req in (failed, healthy)
        ]
        pending = self.native.IndexedReqList(
            self.native.ListKind.PD_RECEIVING, req_of=lambda item: item.req
        )
        pending.join(wrappers)
        queue = SimpleNamespace(queue=pending)
        target = SimpleNamespace(decode_req=wrappers[0], queue=queue)
        failed._pre_layerwise_target = target
        owner = SimpleNamespace(targets=[target], scheduler=SimpleNamespace())
        namespace = {
            "__package__": "sglang_remote_attn",
            "logger": logging.getLogger(__name__),
        }
        for name in ("request_of", "quarantine", "hold_release"):
            load_function(PLUGIN / "transport_failure.py", name, namespace)
        native_utils = ModuleType("sglang.srt.disaggregation.utils")
        native_utils.prepare_abort = lambda *args, **kwargs: self.fail(
            "already finished request must not emit another abort"
        )
        with patch.dict(
            sys.modules,
            {
                self.compat.__name__: self.compat,
                native_utils.__name__: native_utils,
            },
        ):
            namespace["quarantine"](owner, target, "late write is unproven")
        self.assertIs(queue.queue, pending)
        self.assertEqual(list(pending.items()), [wrappers[1]])
        self.assertEqual(owner.targets, [])
        record = owner.transport_quarantine[id(failed)]
        self.assertIn(target, record["items"])
        self.assertIs(target.decode_req.kv_receiver, receiver)
        releases = []
        release = lambda: releases.append("released")
        self.assertTrue(namespace["hold_release"](failed, release))
        self.assertEqual(record["releases"], [release])
        self.assertEqual(releases, [])

    def test_gateway_uses_native_registry_for_wrapped_held_and_releasing_requests(self):
        frozen, receiving, held, releasing = [
            Request(name) for name in ("frozen", "receiving", "held", "releasing")
        ]
        lists = {}
        for kind, req in (
            (self.native.ListKind.PD_RECEIVING, receiving),
            (self.native.ListKind.PD_HELD, held),
            (self.native.ListKind.PD_RELEASING, releasing),
        ):
            queue = self.native.IndexedReqList(kind, req_of=lambda item: item.req)
            queue.join([SimpleNamespace(req=req)])
            lists[kind] = queue
        scheduler = SimpleNamespace(request_lists=lambda: lists)
        runtime = SimpleNamespace(_tracked=lambda scheduler: {frozen.rid: frozen})
        namespace = {
            "d2d": SimpleNamespace(remote=SimpleNamespace(runtime=lambda: runtime))
        }
        candidates = load_function(
            PLUGIN / "dsv_gateway_control.py", "candidates", namespace
        )
        result = candidates(SimpleNamespace(scheduler=scheduler))
        self.assertEqual(result, [frozen, receiving, held, releasing])


if __name__ == "__main__":
    unittest.main()
