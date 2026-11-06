"""PDMux communicator and forward-flag isolation through runtime context reads."""

from __future__ import annotations

import ast
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import Mock, patch

from test_pdmux_sxf_port import SRT, register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def load_definitions(path, names, namespace):
    tree = ast.parse((SRT / path).read_text())
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in definitions} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(SRT / path), "exec"), namespace)


class TestPdmuxParallelRuntime(unittest.TestCase):
    def setUp(self):
        self.decode_group = object()
        self.prefill_group = object()
        self.subgroup = object()
        self.parallel_state = types.ModuleType("sglang.srt.distributed.parallel_state")
        self.parallel_state.__dict__.update(
            _TP=self.decode_group,
            _PDMUX_PREFILL_TP_GROUP=self.prefill_group,
            _ENABLE_PDMUX_P_TP=ContextVar("test_pdmux_lane", default=False),
        )
        load_definitions(
            "distributed/parallel_state.py",
            {"set_pdmux_status", "is_pdmux_prefill_enabled", "resolve_pdmux_group"},
            self.parallel_state.__dict__,
        )
        self.enterContext(
            patch.dict(
                "sys.modules", {self.parallel_state.__name__: self.parallel_state}
            )
        )
        namespace = {"contextmanager": contextmanager}
        load_definitions("runtime_context.py", {"ParallelContext"}, namespace)
        self.parallel = namespace["ParallelContext"]()
        self.aliases = (
            "tp_group",
            "attn_tp_group",
            "attn_cp_group",
            "moe_tp_group",
            "moe_dp_group",
            "shared_experts_tp_group",
        )
        self.parallel._stamp.update(dict.fromkeys(self.aliases, self.decode_group))

    def test_stamped_aliases_follow_lane_and_real_subgroups_keep_membership(self):
        for enabled, group in ((False, self.decode_group), (True, self.prefill_group)):
            self.parallel_state.set_pdmux_status(enabled)
            for name in self.aliases:
                with self.subTest(enabled=enabled, group=name):
                    self.assertIs(getattr(self.parallel, name), group)
        self.parallel._stamp["attn_tp_group"] = self.subgroup
        self.assertIs(self.parallel.attn_tp_group, self.subgroup)

    def test_non_pdmux_group_read_does_not_touch_untraceable_lane_state(self):
        self.parallel_state._PDMUX_PREFILL_TP_GROUP = None
        self.parallel_state.is_pdmux_prefill_enabled = Mock(
            side_effect=AssertionError("ordinary model code must not read ContextVars")
        )
        self.assertIs(self.parallel.tp_group, self.decode_group)

    def test_explicit_draft_override_is_not_replaced_by_target_prefill_group(self):
        self.parallel_state.set_pdmux_status(True)
        self.parallel._overrides["tp_group"] = self.subgroup
        self.parallel._overrides["attn_cp_group"] = None
        self.assertIs(self.parallel.tp_group, self.subgroup)
        self.assertIsNone(self.parallel.attn_cp_group)
        self.assertIs(self.parallel.moe_tp_group, self.prefill_group)

    def test_parallel_reads_are_isolated_between_submitter_threads(self):
        barrier = threading.Barrier(2)

        def read_lane(enabled):
            self.parallel_state.set_pdmux_status(enabled)
            barrier.wait(timeout=5)
            return self.parallel.tp_group, self.parallel.attn_cp_group

        with ThreadPoolExecutor(max_workers=2) as pool:
            prefill = pool.submit(read_lane, True)
            decode = pool.submit(read_lane, False)
            self.assertEqual(prefill.result(), (self.prefill_group,) * 2)
            self.assertEqual(decode.result(), (self.decode_group,) * 2)
        self.assertIs(self.parallel.tp_group, self.decode_group)

    def test_forward_flags_follow_the_same_lane(self):
        context = types.SimpleNamespace(forward=object(), prefill_forward=object())
        namespace = {"_CONTEXT": context}
        load_definitions("runtime_context.py", {"get_forward"}, namespace)
        for enabled, flags in (
            (False, context.forward),
            (True, context.prefill_forward),
        ):
            self.parallel_state.set_pdmux_status(enabled)
            self.assertIs(namespace["get_forward"](), flags)
        context.prefill_forward = None
        self.assertIs(namespace["get_forward"](), context.forward)


if __name__ == "__main__":
    unittest.main()
