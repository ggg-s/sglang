"""CUDA graph lifecycle probe; does not run SM90 FlashMLA or a model.

Run on any CUDA GPU:
    python test/manual/remote_attn/flashmla_capture_lifecycle_probe.py

Uses the production RemoteGraphBackend orchestration and DSV4 metadata hook
with tiny CUDA scheduling producers/consumers. A negative control skips the
full-to-remote boundary hook and must reject the first poisoned eager read.
"""

import ast
import json
import logging
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[3]
POISON = -(1 << 30)
FIELDS = tuple(f"c{ratio}_flashmla_metadata" for ratio in (0, 1, 2, 4, 128))


def source_class(path, name, namespace, *, methods=None):
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    if methods is not None:
        node.body = [
            item
            for item in node.body
            if isinstance(item, ast.FunctionDef) and item.name in methods
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


class Core:
    has_c4 = has_c128 = True
    low_ratios = (1, 2)

    def __init__(self, fixture):
        for name in FIELDS:
            setattr(self, name, fixture.schedule())


class Metadata:
    def __init__(self, fixture):
        self.core_attn_metadata = Core(fixture)


class Bank:
    def __init__(self, fixture, name):
        self.fixture, self.name = fixture, name
        self.records = []

    def capture_session(self, stream):
        return nullcontext()

    def capture_one(self, key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        f = self.fixture
        for _ in range(2):
            torch.cuda.synchronize()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=f.stream):
            forward_fn()
        schedules = [
            getattr(f.attention.forward_metadata.core_attn_metadata, name)
            for name in FIELDS
        ]
        torch.cuda.synchronize()
        # The graph recorded fill_ producers but has never replayed them.
        assert all(
            torch.all(s.tile_scheduler_metadata == POISON).item() for s in schedules
        )
        assert all(torch.all(s.num_splits == POISON).item() for s in schedules)
        self.records.append((key.size, graph, schedules))


class Fixture:
    def __init__(self):
        self.size = 2
        self.stream = torch.cuda.Stream()
        attention_type = source_class(
            ROOT / "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
            "DeepseekV4AttnBackend",
            {
                "DSV4Metadata": Metadata,
                "DSV4AttnMetadata": Core,
                "_create_flashmla_metadata": self.schedule,
            },
            methods={"on_after_cuda_graph_warmup"},
        )
        self.attention = attention_type.__new__(attention_type)
        self.attention._current_capture_raw = None
        remote_type = source_class(
            ROOT / "plugins/remote_attn/sglang_remote_attn/segmented_graph.py",
            "RemoteGraphBackend",
            {
                "contextmanager": contextmanager,
                "logger": logging.getLogger(__name__),
                "REMOTE_BATCHES": (1, 2, 4, 8, 16),
            },
        )
        self.full, self.remote = Bank(self, "full"), Bank(self, "remote")
        runner = NS(
            model_runner=NS(model_config=NS(num_hidden_layers=1)),
            device_module=torch.cuda,
        )
        self.backend = remote_type(
            runner,
            self.full,
            self.remote,
            lambda fn: fn,
            torch.cuda.is_current_stream_capturing,
            lambda: None,
        )
        self.begin_bucket(2)

    def schedule(self):
        # Prepoison outside capture to make unexecuted producers deterministic.
        # Metadata remains absent until the attention forward first uses it.
        return NS(
            tile_scheduler_metadata=None,
            num_splits=None,
            backing_tile=torch.full((3, 8), POISON, dtype=torch.int32, device="cuda"),
            backing_splits=torch.full(
                (self.size + 1,), POISON, dtype=torch.int32, device="cuda"
            ),
            output=torch.empty((), dtype=torch.int32, device="cuda"),
        )

    def begin_bucket(self, size):
        self.size = size
        self.attention.forward_metadata = Metadata(self)

    def forward(self):
        capturing = torch.cuda.is_current_stream_capturing()
        for name in FIELDS:
            schedule = getattr(self.attention.forward_metadata.core_attn_metadata, name)
            if schedule.tile_scheduler_metadata is None:
                schedule.tile_scheduler_metadata = schedule.backing_tile
                schedule.num_splits = schedule.backing_splits
                schedule.tile_scheduler_metadata.fill_(self.size)
                schedule.num_splits.fill_(1)
            elif not capturing:
                if (
                    torch.any(schedule.tile_scheduler_metadata < 0).item()
                    or torch.any(schedule.num_splits < 0).item()
                ):
                    raise RuntimeError("eager consumed capture-only scheduler metadata")
            schedule.output.copy_(
                schedule.tile_scheduler_metadata[0, 0] + schedule.num_splits[-1]
            )
        if capturing and self.backend.capture_remote:
            self.backend.capture_layers.append(0)

    @staticmethod
    def key(size):
        return NS(
            size=size, stream_idx=None, variant_label=None, attention_variant=None
        )

    def capture(self, size):
        self.begin_bucket(size)
        with self.backend.capture_session(self.stream):
            self.backend.capture_one(
                self.key(size),
                self.forward,
                post_warmup_hook=self.attention.on_after_cuda_graph_warmup,
            )

    def assert_old_graphs(self):
        records = self.full.records + self.remote.records
        pointers = [
            tuple(
                (s.tile_scheduler_metadata.data_ptr(), s.num_splits.data_ptr())
                for s in schedules
            )
            for _, _, schedules in records
        ]
        # CPU hook has detached wrappers, not modified old graph storage.
        for _, _, schedules in records:
            assert all(
                torch.all(s.tile_scheduler_metadata == POISON).item() for s in schedules
            )
            assert all(torch.all(s.num_splits == POISON).item() for s in schedules)
        for (size, graph, schedules), before in zip(records, pointers):
            graph.replay()
            torch.cuda.synchronize()
            assert (
                tuple(
                    (s.tile_scheduler_metadata.data_ptr(), s.num_splits.data_ptr())
                    for s in schedules
                )
                == before
            )
            assert all(s.output.item() == size + 1 for s in schedules)
            assert all(
                torch.all(s.tile_scheduler_metadata == size).item() for s in schedules
            )
            assert all(torch.all(s.num_splits == 1).item() for s in schedules)
        return len(records)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this probe does not validate SM90 FlashMLA")
    negative = Fixture()
    key = negative.key(2)
    negative.full.capture_one(
        key,
        negative.forward,
        post_warmup_hook=negative.attention.on_after_cuda_graph_warmup,
    )
    try:
        # Exactly the old full -> remote transition, with no boundary reset.
        negative.remote.capture_one(
            key,
            negative.forward,
            post_warmup_hook=negative.attention.on_after_cuda_graph_warmup,
        )
    except RuntimeError as error:
        if "capture-only scheduler" not in str(error):
            raise
    else:
        raise AssertionError("negative control did not reject stale scheduler metadata")
    outcomes = []
    for sizes in ((1,), (2, 1), (4, 1), (8, 4, 2, 1)):
        fixture = Fixture()
        for size in sizes:
            fixture.capture(size)
        graphs = fixture.assert_old_graphs()
        outcomes.append({"buckets": sizes, "graphs_replayed": graphs})
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "negative_control": "rejected_first_remote_eager_read",
                "lifecycle_cases": outcomes,
                "sm90_flashmla_executed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
