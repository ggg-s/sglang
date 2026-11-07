"""CPU tests of scoped eager replay without requiring CUDA graph capture."""

import gc
import sys
import weakref
from contextlib import contextmanager
from contextvars import Context, ContextVar
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.model_executor.runner_utils.forward_batch import (
    get_forward_batch,
    set_forward_batch,
)
from sglang.srt.model_executor.runner_utils.pdmux_bcg import (
    PdmuxBCGExecutor,
    capture_layer_boundaries,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


@contextmanager
def recording_capture():
    events = []
    graph = bcg.BreakableCUDAGraph()
    capture = SimpleNamespace(
        cuda_graph=graph,
        _end_current_segment=lambda: events.append("end"),
        _begin_new_segment=lambda: events.append("begin"),
    )
    token = bcg._current_capture_var.set(capture)
    try:
        yield graph, events
    finally:
        bcg._current_capture_var.reset(token)


def test_method_replay_reads_current_batch_and_preserves_output_address(monkeypatch):
    class Layer:
        @bcg.eager_on_graph
        def _eager_scale(self, x, out):
            out.copy_(x * get_forward_batch().scale)

    layer = Layer()
    x = torch.tensor([2.0])
    output = torch.zeros(1)
    with set_forward_batch(SimpleNamespace(scale=3)):
        with recording_capture() as (graph, events):
            assert layer._eager_scale(x, output) is None
    assert events == ["end", "begin"]
    assert output.item() == 0
    pointer = output.data_ptr()
    # Exercise the public, argument-free replay API between two segments.
    graph._segments = [
        SimpleNamespace(replay=lambda: events.append("first")),
        SimpleNamespace(replay=lambda: events.append("last")),
    ]
    monkeypatch.setattr(
        bcg, "get_device_module", lambda: SimpleNamespace(current_stream=lambda: None)
    )
    x.fill_(4)
    with set_forward_batch(SimpleNamespace(scale=5)):
        graph.replay()
    assert events == ["end", "begin", "first", "last"]
    assert output.item() == 20 and output.data_ptr() == pointer
    with pytest.raises(RuntimeError, match="No forward batch"):
        graph.replay()


def test_method_on_capture_and_real_replay():
    calls = []

    class Layer:
        def _on_capture(self, x, out):
            calls.append("stub")

        @bcg.eager_on_graph(on_capture=_on_capture)
        def _eager_scale(self, x, out):
            calls.append("real")
            out.copy_(x * get_forward_batch().scale)

    layer = Layer()
    output = torch.zeros(2)
    with recording_capture() as (graph, _):
        layer._eager_scale(torch.ones(2), output)
    assert calls == ["stub"]
    with set_forward_batch(SimpleNamespace(scale=7)):
        graph._break_fns[0]()
    assert calls == ["stub", "real"]
    torch.testing.assert_close(output, torch.full((2,), 7.0))


def test_plain_callable_and_eager_passthrough():
    @bcg.eager_on_graph
    def increment(x, out):
        out.copy_(x + 1)

    output = torch.tensor(0)
    increment(torch.tensor(2), output)
    assert output.item() == 3
    x = torch.tensor(3)
    with recording_capture() as (graph, _):
        increment(x, output)
    x.fill_(9)
    graph._break_fns[0]()
    assert output.item() == 10


def test_a_break_returning_a_value_fails():
    @bcg.eager_on_graph
    def increment(x):
        return x + 1

    with pytest.raises(AssertionError, match="must return None"):
        increment(torch.tensor(2))
    with recording_capture() as (graph, _):
        increment(torch.tensor(2))
    with pytest.raises(AssertionError, match="must return None"):
        graph._break_fns[0]()


def test_replay_does_not_retain_capture_or_serving_batches():
    class Batch:
        scale = 3

    @bcg.eager_on_graph
    def scale(x, out):
        out.copy_(x * get_forward_batch().scale)

    batch = Batch()
    capture_ref = weakref.ref(batch)
    with set_forward_batch(batch):
        with recording_capture() as (graph, _):
            scale(torch.ones(2), torch.zeros(2))
    del batch
    gc.collect()
    assert capture_ref() is None

    batch = Batch()
    live_ref = weakref.ref(batch)
    with set_forward_batch(batch):
        graph._break_fns[0]()
    del batch
    gc.collect()
    assert live_ref() is None


@pytest.mark.parametrize(
    "decorator", [bcg.eager_on_graph(True), bcg.eager_on_graph(enable=True)]
)
def test_legacy_decorator_replays(decorator):
    @decorator
    def increment(x, out):
        out.copy_(x + 1)

    output = torch.tensor(0)
    with recording_capture() as (graph, _):
        increment(torch.tensor(2), output)
    graph._break_fns[0]()
    assert output.item() == 3


def test_legacy_disabled_decorator_is_identity():
    fn = lambda x: x + 1
    assert bcg.eager_on_graph(False)(fn) is fn
    assert bcg.eager_on_graph(enable=False)(fn) is fn


def test_batch_scope_restores_outer_batch_after_exception():
    outer = SimpleNamespace(scale=2)
    with set_forward_batch(outer):
        with pytest.raises(ValueError, match="eager failure"):
            with set_forward_batch(SimpleNamespace(scale=9)):
                assert get_forward_batch().scale == 9
                raise ValueError("eager failure")
        assert get_forward_batch() is outer
    with pytest.raises(RuntimeError, match="No forward batch"):
        get_forward_batch()


def test_batch_scope_is_isolated_between_execution_contexts():
    with set_forward_batch(SimpleNamespace(scale=2)):
        with pytest.raises(RuntimeError, match="No forward batch"):
            Context().run(get_forward_batch)
        assert get_forward_batch().scale == 2


def test_run_eager_binds_instance_and_keeps_one_outer_break():
    calls = []

    class Layer:
        def __init__(self, scale):
            self.scale = scale

        def stub(self, x, *, offset=0):
            raise AssertionError("run_eager must bypass on_capture")

        @bcg.eager_on_graph(on_capture=stub)
        def inner(self, x, *, offset=0):
            calls.append(self.scale)
            return x * self.scale + offset

        @bcg.eager_on_graph
        def outer(self, x, out):
            out.copy_(self.inner.run_eager(x, offset=1))

    first, second = Layer(2), Layer(5)
    output = torch.zeros(1)
    with recording_capture() as (graph, events):
        first.outer(torch.ones(1), output)
    assert events == ["end", "begin"]
    assert len(graph._break_fns) == 1
    assert calls == [] and output.item() == 0
    assert second.inner.run_eager(torch.ones(1), offset=1).item() == 6
    graph._break_fns[0]()
    assert calls == [5, 2]
    assert output.item() == 3


def test_run_eager_plain_function_does_not_record_a_break():
    @bcg.eager_on_graph
    def increment(x):
        return x + 1

    with recording_capture() as (graph, events):
        assert increment.run_eager(torch.tensor(2)).item() == 3
    assert events == [] and graph._break_fns == []


def test_run_eager_method_torch_compile():
    class Layer(torch.nn.Module):
        @bcg.eager_on_graph
        def inner(self, x):
            return x + 1

        def forward(self, x):
            return self.inner.run_eager(x)

    layer = Layer()
    x = torch.ones(2)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), layer(x))


def test_split_intervals_partition_eager_breaks_and_run_tail_once(monkeypatch):
    """A yield must neither repeat preparation nor omit the final eager tail."""
    bindings = []
    graph = bcg.BreakableCUDAGraph(
        hicache_graph_events=SimpleNamespace(
            bind=lambda *args: bindings.append("bound")
        )
    )
    state = torch.zeros(1)
    events = []
    actions = [
        lambda: state.fill_(2),
        lambda: state.add_(3),
        lambda: state.mul_(4),
        lambda: state.add_(1),
        lambda: state.mul_(2),
    ]
    graph._segments = [SimpleNamespace(replay=fn) for fn in actions]
    graph._break_fns = [lambda i=i: events.append(i) for i in range(4)]
    graph._layer_ends = {1: 2, 2: 4}
    monkeypatch.setattr(
        bcg, "get_device_module", lambda: SimpleNamespace(current_stream=lambda: None)
    )
    graph.replay_layers(0, 1)
    assert state.item() == 5 and events == [0, 1]
    graph.replay_layers(1, 2)
    assert state.item() == 42 and events == [0, 1, 2, 3]
    assert bindings == ["bound"]
    state.zero_()
    graph.replay()
    assert state.item() == 42
    assert bindings == ["bound", "bound"]
    with pytest.raises(ValueError):
        graph.replay_layers(1, 3)
    graph._layer_ends = {2: 4}
    with pytest.raises(ValueError, match="every decoder layer"):
        graph.replay_layers(0, 2)


@pytest.mark.parametrize("mapping", [False, True])
def test_generic_layer_hooks_are_removed_between_capture_variants(mapping):
    """A second shape/lane must not retain the first capture's layer hooks."""
    layers = [torch.nn.Identity(), torch.nn.Identity()]
    model = SimpleNamespace(
        layers=(
            torch.nn.ModuleDict({str(i): l for i, l in enumerate(layers)})
            if mapping
            else torch.nn.ModuleList(layers)
        )
    )
    for _ in range(2):
        with capture_layer_boundaries(model, 2), recording_capture() as (graph, _):
            for layer in layers:
                layer(torch.ones(1))
            assert list(graph._layer_ends) == [1, 2]
        assert all(not layer._forward_hooks for layer in layers)


def test_repeated_decoder_invocation_declines_partial_graph_replay():
    """Recurrent layer execution can keep full BCG without a false split map."""
    with recording_capture() as (graph, _):
        bcg.mark_layer_end(1)
        bcg.mark_layer_end(1)
        bcg.mark_layer_end(2)
    with pytest.raises(ValueError):
        graph.layer_segment_range(0, 2)


def test_split_session_restores_live_context_and_keeps_one_owner():
    """Decode between intervals must not replace metadata or reset graph input."""
    active = ContextVar("split_test_batch", default=None)
    calls = []
    batch = SimpleNamespace(split_index=0, input_ids=[1, 2])

    @contextmanager
    def context(b, **kwargs):
        token = active.set(b)
        try:
            yield
        finally:
            active.reset(token)

    def load(b):
        calls.append("load")
        return b, None

    def replay(key, b, **kwargs):
        assert active.get() is batch
        calls.append(kwargs["pdmux_layer_interval"])
        return "logits"

    def tail(b, static, size, raw, key, **kwargs):
        calls.append("tail")
        return replay(key, b, **kwargs)

    runner = SimpleNamespace(
        model_runner=SimpleNamespace(model_config=SimpleNamespace(num_hidden_layers=2)),
        model_context=lambda: context(None),
        backend=SimpleNamespace(replay_session=lambda: context(None), replay=replay),
        load_batch=load,
        raw_num_tokens=2,
        _shape_key=lambda size, b: (size, 1),
        buffer_registry=SimpleNamespace(has_slot=lambda name: False),
        _prefill_forward_context=context,
        device_module=None,
        _execute_body_capture=tail,
        _finalize_execute_output=lambda out: out,
    )
    executor = PdmuxBCGExecutor(runner)
    assert executor.execute(batch, 1) is None
    assert active.get() is None and executor.owner is batch
    with pytest.raises(RuntimeError, match="owns"):
        executor.can_run(SimpleNamespace(), 1)
    # Replaying an interval twice would repeat a state update; reject it.
    with pytest.raises(RuntimeError, match="not finalized"):
        executor.execute(batch, 1)
    batch.split_index = 1
    with context("decode"):
        assert executor.execute(batch, 1) == "logits"
        assert active.get() == "decode"
    assert executor.owner is None and executor.session is None
    assert calls == ["load", (0, 1), "tail", (1, 2)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
