"""CPU regression contracts for FlashMLA metadata at full/remote bank boundaries.

The fake scheduler deliberately leaves capture-only allocations poisoned: CUDA
capture records its producer without executing it. The bank orchestration and
metadata reset methods are loaded from production source without CUDA imports.
"""

import ast
import logging
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
PLUGIN = ROOT / "plugins/remote_attn/sglang_remote_attn"
SCHEDULES = tuple(f"c{ratio}_flashmla_metadata" for ratio in (0, 1, 2, 4, 128))
POISON = -(1 << 30)


def _source_class(path, name, namespace, *, methods=None):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    if methods is not None:
        node.body = [
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods
        ]
        node.bases = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102
    return namespace[name]


class _Core:
    def __init__(self, *, has_c4=True, has_c128=True, low_ratios=(1, 2)):
        self.has_c4, self.has_c128, self.low_ratios = has_c4, has_c128, low_ratios
        active = {0, *low_ratios}
        if has_c4:
            active.add(4)
        if has_c128:
            active.add(128)
        for ratio in (0, 1, 2, 4, 128):
            setattr(
                self,
                f"c{ratio}_flashmla_metadata",
                _schedule() if ratio in active else None,
            )


class _Metadata:
    def __init__(self, core=None):
        self.core_attn_metadata = core or _Core()


def _schedule():
    return NS(tile_scheduler_metadata=None, num_splits=None)


def _load_runtime():
    namespace = {
        "DSV4Metadata": _Metadata,
        "DSV4AttnMetadata": _Core,
        "_create_flashmla_metadata": _schedule,
    }
    attention = _source_class(
        ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
        "DeepseekV4AttnBackend",
        namespace,
        methods={"on_after_cuda_graph_warmup"},
    )
    namespace = {
        "contextmanager": contextmanager,
        "logger": logging.getLogger(__name__),
        "REMOTE_BATCHES": (1, 2, 4, 8, 16),
    }
    remote = _source_class(
        PLUGIN / "segmented_graph.py", "RemoteGraphBackend", namespace
    )
    dsv41 = _source_class(PLUGIN / "dsv41_graph.py", "DSV41GraphBackend", namespace)
    return attention, remote, dsv41


class _Bank:
    """Preserve the native warmup/hook/warmup/hook/capture execution order."""

    def __init__(self, fixture, name):
        self.fixture, self.name = fixture, name
        self.keys = []
        self.fail_capture = False

    def capture_session(self, stream):
        return nullcontext()

    def capture_one(self, key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        f = self.fixture
        try:
            for i in range(2):
                f.stage = (self.name, f"warmup{i}", key.size)
                forward_fn()
                if post_warmup_hook is not None:
                    post_warmup_hook()
            f.stage = (self.name, "capture", key.size)
            forward_fn()
            if self.fail_capture:
                raise RuntimeError("injected capture failure")
            self.keys.append(key)
        finally:
            f.stage = None


class _Fixture:
    def __init__(self, *, raw=False, variants=False):
        attention_type, remote_type, dsv41_type = _load_runtime()
        self.attention = attention_type.__new__(attention_type)
        self.raw = raw
        self.stage = None
        self.calls, self.records, self.restored, self.hooks = [], [], [], []
        self.full, self.remote = _Bank(self, "full"), _Bank(self, "remote")
        runner = NS(
            model_runner=NS(model_config=NS(num_hidden_layers=1)),
            device_module=NS(Event=lambda: object()),
            attention_graph_variants=NS(capture_labels=("a", "b")),
        )
        backend_type = dsv41_type if variants else remote_type
        self.backend = backend_type(
            runner,
            self.full,
            self.remote,
            lambda fn: fn,
            lambda: self.stage is not None and self.stage[1] == "capture",
            lambda: self.restored.append(True),
        )
        self.begin_bucket()

    def begin_bucket(self):
        self.attention.forward_metadata = _Metadata()
        self.attention._current_capture_raw = object() if self.raw else None

    def hook(self):
        self.hooks.append(self.stage)
        self.attention.on_after_cuda_graph_warmup()

    def forward(self):
        bank, phase, size = self.stage
        metadata = self.attention.forward_metadata
        if metadata is self.attention._current_capture_raw:
            # Native out-of-graph Raw -> Full rebuilding runs eagerly again.
            metadata = self.attention.forward_metadata = _Metadata()
        core = metadata.core_attn_metadata
        self.calls.append(self.stage)
        for name in SCHEDULES:
            schedule = getattr(core, name)
            if schedule is None:
                continue
            if schedule.tile_scheduler_metadata is None:
                schedule.tile_scheduler_metadata = torch.full(
                    (3, 8), POISON, dtype=torch.int32
                )
                schedule.num_splits = torch.full((size + 1,), POISON, dtype=torch.int32)
                if phase != "capture":
                    schedule.tile_scheduler_metadata.fill_(size)
                    schedule.num_splits.fill_(1)
            if phase == "capture":
                tensors = (schedule.tile_scheduler_metadata, schedule.num_splits)
                self.records.append(
                    (bank, size, tensors, tuple(t.clone() for t in tensors))
                )
            elif (
                schedule.num_splits.shape != (size + 1,)
                or torch.any(schedule.tile_scheduler_metadata < 0)
                or torch.any(schedule.num_splits < 0)
            ):
                raise RuntimeError("eager consumed capture-only scheduler metadata")
        if phase == "capture" and self.backend.capture_remote:
            self.backend.capture_layers.append(0)
            if hasattr(self.backend, "dynamic_trace"):
                self.backend.dynamic_trace.extend(
                    [("metadata", -1), ("state", 0), ("attention", 0)]
                )

    @staticmethod
    def key(size, variant=None):
        return NS(
            size=size, stream_idx=None, variant_label=None, attention_variant=variant
        )

    def capture(self, size, variant=None):
        self.begin_bucket()
        with self.backend.capture_session(object()):
            self.backend.capture_one(
                self.key(size, variant), self.forward, post_warmup_hook=self.hook
            )


class TestFlashMLACaptureLifecycle(unittest.TestCase):
    def test_capture_only_schedule_reproduces_first_eager_warmup_failure(self):
        f = _Fixture()
        key = f.key(2)
        f.full.capture_one(key, f.forward, post_warmup_hook=f.hook)
        with self.assertRaisesRegex(RuntimeError, "capture-only scheduler"):
            f.remote.capture_one(key, f.forward, post_warmup_hook=f.hook)
        self.assertEqual(f.calls[-1], ("remote", "warmup0", 2))

    def test_remote_buckets_reset_before_first_warmup_and_keep_old_graph_storage(self):
        for sizes in ((1,), (2, 1), (4, 1), (8, 4, 2, 1)):
            for raw in (False, True):
                with self.subTest(sizes=sizes, raw=raw):
                    f = _Fixture(raw=raw)
                    addresses = []
                    for size in sizes:
                        f.capture(size)
                        for tensors, pointers in addresses:
                            self.assertEqual(
                                tuple(t.data_ptr() for t in tensors), pointers
                            )
                        addresses = [
                            (r[2], tuple(t.data_ptr() for t in r[2])) for r in f.records
                        ]
                        for _, _, tensors, snapshots in f.records:
                            for tensor, snapshot in zip(tensors, snapshots):
                                self.assertTrue(torch.equal(tensor, snapshot))
                    self.assertEqual(sorted(f.backend.remote_keys), sorted(sizes))
                    self.assertEqual(len(f.hooks), 5 * len(sizes))
                    self.assertEqual(len(f.restored), len(sizes))
                    # All capture producers remain unexecuted, never in-place zeroed.
                    self.assertTrue(
                        all(torch.all(r[2][0] == POISON) for r in f.records)
                    )

    def test_reset_replaces_all_active_wrappers_and_both_schedule_tensors(self):
        for has_c4, has_c128, low_ratios in (
            (True, True, (1, 2)),
            (False, False, ()),
            (True, False, (1,)),
        ):
            with self.subTest(has_c4=has_c4, has_c128=has_c128, low_ratios=low_ratios):
                f = _Fixture()
                core = _Core(has_c4=has_c4, has_c128=has_c128, low_ratios=low_ratios)
                metadata = f.attention.forward_metadata = _Metadata(core)
                old = {name: getattr(core, name) for name in SCHEDULES}
                for schedule in old.values():
                    if schedule is not None:
                        schedule.tile_scheduler_metadata = torch.full((3, 8), POISON)
                        schedule.num_splits = torch.full((3,), POISON)
                f.hook()
                self.assertIs(f.attention.forward_metadata, metadata)
                for name, previous in old.items():
                    current = getattr(core, name)
                    if previous is None:
                        self.assertIsNone(current)
                    else:
                        self.assertIsNot(current, previous)
                        self.assertIsNone(current.tile_scheduler_metadata)
                        self.assertIsNone(current.num_splits)
                        self.assertTrue(
                            torch.all(previous.tile_scheduler_metadata == POISON)
                        )
                        self.assertTrue(torch.all(previous.num_splits == POISON))

    def test_hook_restores_raw_after_detaching_capture_only_wrappers(self):
        f = _Fixture(raw=True)
        metadata = f.attention.forward_metadata
        old = metadata.core_attn_metadata.c0_flashmla_metadata
        f.hook()
        self.assertIs(f.attention.forward_metadata, f.attention._current_capture_raw)
        self.assertIsNot(metadata.core_attn_metadata.c0_flashmla_metadata, old)
        f.hook()  # Already Raw is a valid, idempotent boundary.
        self.assertIs(f.attention.forward_metadata, f.attention._current_capture_raw)

    def test_nonremote_bucket_has_only_the_native_post_warmup_hooks(self):
        f = _Fixture()
        f.capture(32)
        self.assertEqual(len(f.hooks), 2)
        self.assertEqual(f.remote.keys, [])
        self.assertEqual(f.restored, [])

    def test_none_hook_keeps_other_attention_backends_compatible(self):
        f = _Fixture()
        calls = []

        def forward():
            calls.append(f.stage)
            if f.stage[1] == "capture" and f.backend.capture_remote:
                f.backend.capture_layers.append(0)

        with f.backend.capture_session(object()):
            f.backend.capture_one(f.key(1), forward, post_warmup_hook=None)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(f.remote.keys), 1)

    def test_dsv41_native_variants_share_one_remote_bank_per_bucket(self):
        f = _Fixture(raw=True, variants=True)
        for size in (2, 1):
            for variant in ("a", "b"):
                f.capture(size, variant)
        self.assertEqual(len(f.full.keys), 4)
        self.assertEqual(len(f.remote.keys), 2)
        self.assertEqual(len(f.backend.native_keys), 4)
        self.assertEqual(len(f.hooks), 14)

    def test_capture_exception_restores_bank_state_and_allows_fresh_bucket_retry(self):
        f = _Fixture(raw=True)
        f.remote.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
            f.capture(2)
        self.assertFalse(f.backend.capture_remote)
        self.assertFalse(f.backend.capturing)
        self.assertIsNone(f.backend.stream)
        self.assertEqual(f.backend.remote_keys, {})
        self.assertEqual(len(f.restored), 1)
        f.remote.fail_capture = False
        f.capture(2)
        f.capture(1)
        self.assertEqual(sorted(f.backend.remote_keys), [1, 2])


if __name__ == "__main__":
    unittest.main()
