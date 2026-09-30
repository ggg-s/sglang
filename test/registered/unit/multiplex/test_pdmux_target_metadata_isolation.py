"""Run production lane routing over stub backends without a GPU runtime."""

import ast
import contextlib
import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from test_pdmux_sxf_port import SRT, bind, load_methods, register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "pdmux_isolation_forward_context", SRT / "model_executor/forward_context.py"
)
_context = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _context
_spec.loader.exec_module(_context)


def mode(name):
    return SimpleNamespace(is_target_verify=lambda: name == "verify", name=name)


class Backend:
    def __init__(self):
        self.forward_metadata = None
        self.plans = 0

    def init_forward_metadata(self, batch):
        self.plans += 1
        # DSV4 drops metadata even for a padded, nonempty logical IDLE.
        self.forward_metadata = None if batch.idle else batch.metadata


class TestTargetMetadataIsolation(unittest.TestCase):
    def setUp(self):
        namespace = dict(vars(_context))
        namespace.update(
            contextlib=contextlib,
            device_timer_ctx=lambda *args: contextlib.nullcontext(),
            is_cp_v2_active=lambda batch: False,
            _is_hip=False,
        )
        self.methods = load_methods(
            "model_executor/runner/eager_runner.py",
            "EagerRunner",
            [
                "_caller_published_attn_backend",
                "_resolve_decode_pdmux",
                "_resolve_extend_pdmux",
                "_execute_extend",
                "_execute_idle",
            ],
            **namespace,
        )
        self.split = load_methods(
            "model_executor/model_runner.py",
            "ModelRunner",
            ["forward_split_prefill"],
            device_timer_ctx=lambda *args: contextlib.nullcontext(),
        )["forward_split_prefill"]
        self.prefill = Backend()
        self.decode = Backend()
        self.observed = []

        def forward(inputs, positions, batch, *args):
            backend = _context.get_attn_backend()
            self.observed.append(backend)
            self.assertIs(
                backend.forward_metadata, None if batch.idle else batch.metadata
            )
            return backend.forward_metadata

        self.mr = SimpleNamespace(
            attn_backend=self.prefill,
            decode_attn_backend=self.decode,
            ps=SimpleNamespace(attn_dcp_size=1),
            device_timer=None,
            prefill_cuda_graph_runner=None,
            model_config=SimpleNamespace(num_hidden_layers=3),
            model=SimpleNamespace(forward=forward, forward_split_prefill=forward),
            _extend_forward_kwargs=lambda *args: {},
            _pp_kwargs=lambda *args: {},
        )
        self.runner = bind(
            SimpleNamespace(
                model_runner=self.mr,
                enable_pdmux=True,
                pdmux_standard=False,
                load_batch=lambda batch, *args: batch,
            ),
            self.methods,
        )

    @staticmethod
    def batch(name, *, bs=1, ready=False, peer_prefill=False):
        return SimpleNamespace(
            forward_mode=mode(name),
            _original_forward_mode=None,
            batch_size=bs,
            input_ids=None,
            positions=None,
            idle=name == "idle",
            metadata=object(),
            split_index=0,
            is_extend_in_batch=peer_prefill,
            needs_forward_metadata_init=lambda: not ready,
        )

    def test_split_metadata_survives_verify_and_idle_between_segments(self):
        for ready in (False, True):
            for bs in (0, 1):
                for group in range(3):
                    with self.subTest(ready=ready, idle_bs=bs, group=group):
                        self.mr.decode_attn_backend = Backend()
                        prefill = self.batch("prefill")
                        plans_before = self.prefill.plans
                        with _context.forward_context(
                            _context.ForwardContext(attn_backend=self.prefill)
                        ):
                            self.split(self.mr, prefill)
                            verify = self.batch("verify", ready=ready)
                            if ready:
                                self.mr.decode_attn_backend.init_forward_metadata(
                                    verify
                                )
                            self.runner._execute_extend(verify)
                            self.assertIs(
                                self.observed[-1], self.mr.decode_attn_backend
                            )
                            self.split(self.mr, prefill)
                            self.runner._execute_idle(self.batch("idle", bs=bs))
                            self.assertIs(
                                self.observed[-1], self.mr.decode_attn_backend
                            )
                            self.split(self.mr, prefill)
                            self.assertIs(_context.get_attn_backend(), self.prefill)
                        self.assertEqual(prefill.split_index, 3)
                        self.assertEqual(self.prefill.plans, plans_before + 1)

    def test_idle_does_not_clear_pending_split_prefill(self):
        for bs in (0, 1):
            prefill = self.batch("prefill")
            with _context.forward_context(
                _context.ForwardContext(attn_backend=self.prefill)
            ):
                self.split(self.mr, prefill)
                self.runner._execute_idle(self.batch("idle", bs=bs))
                self.split(self.mr, prefill)
            self.assertIs(self.prefill.forward_metadata, prefill.metadata)

    def test_idle_preserves_explicit_draft_step_backend(self):
        for bs in (0, 1):
            draft = Backend()
            self.prefill.forward_metadata = marker = object()
            self.decode.forward_metadata = marker
            with _context.forward_context(_context.ForwardContext(attn_backend=draft)):
                self.runner._execute_idle(self.batch("idle", bs=bs))
                self.assertIs(_context.get_attn_backend(), draft)
            self.assertIs(self.observed[-1], draft)
            self.assertIs(self.prefill.forward_metadata, marker)
            self.assertIs(self.decode.forward_metadata, marker)

    def test_peer_prefill_idle_keeps_prefill_backend(self):
        self.decode.forward_metadata = marker = object()
        with _context.forward_context(
            _context.ForwardContext(attn_backend=self.prefill)
        ):
            self.runner._execute_idle(self.batch("idle", peer_prefill=True))
        self.assertIs(self.observed[-1], self.prefill)
        self.assertIs(self.decode.forward_metadata, marker)

    def test_dp_remapped_verify_retains_decode_routing(self):
        batch = self.batch("verify")
        batch._original_forward_mode = batch.forward_mode
        batch.forward_mode = mode("extend")
        with _context.forward_context(
            _context.ForwardContext(attn_backend=self.prefill)
        ):
            self.runner._execute_extend(batch)
        self.assertIs(self.observed[-1], self.decode)

    def test_non_pdmux_keeps_default_backend(self):
        self.runner.enable_pdmux = False
        with _context.forward_context(
            _context.ForwardContext(attn_backend=self.prefill)
        ):
            self.runner._execute_extend(self.batch("verify"))
            self.runner._execute_idle(self.batch("idle"))
        self.assertEqual(self.observed, [self.prefill, self.prefill])

    def test_preplanner_selects_decode_backend_in_both_pdmux_modes(self):
        tree = ast.parse(
            (SRT / "multiplex/pdmux_context.py").read_text(encoding="utf-8")
        )
        helper = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "decode_lane_attn_backend"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[helper], type_ignores=[]), "pdmux_context.py", "exec"
            ),
            namespace,
        )
        for enabled in (False, True):
            for prefill_mode in ("standard", "layer_split"):
                config = SimpleNamespace(
                    enable_pdmux=enabled, pdmux_prefill_mode=prefill_mode
                )
                module = SimpleNamespace(get_disagg=lambda: config)
                with patch.dict(sys.modules, {"sglang.srt.runtime_context": module}):
                    backend = namespace["decode_lane_attn_backend"](self.mr)
                self.assertIs(backend, self.decode if enabled else self.prefill)


if __name__ == "__main__":
    unittest.main()
