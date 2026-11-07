"""Single-owner continuation of a layer-marked prefill BCG.

The graph retains cross-layer tensor storage, rather than replaying Python
assignments to model_specific_states. A session cannot change shape/stream group
or fall back to eager halfway through: both would lose the captured activations.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@contextmanager
def capture_layer_boundaries(layer_model, num_layers):
    """Use ordinary decoder modules as boundaries; no model-family allowlist.

    Hooks are installed only around startup capture, never serving. The marker
    is a no-op during warmup. Models without an ordered layer list retain eager
    split; a graph without all boundaries cannot admit a partial replay.
    """
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        mark_layer_end,
    )

    layers = getattr(layer_model, "layers", ())
    layers = tuple(layers.values() if hasattr(layers, "values") else layers)
    handles = []
    if (
        len(layers) == num_layers
        and len({id(layer) for layer in layers}) == num_layers
        and all(
            callable(getattr(layer, "register_forward_hook", None)) for layer in layers
        )
    ):
        for index, layer in enumerate(layers):

            def after_layer(module, args, output, end=index + 1):
                mark_layer_end(end)

            handles.append(layer.register_forward_hook(after_layer))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@dataclass
class SplitBCGSession:
    static_batch: Any
    shape_key: Any
    raw_num_tokens: int
    next_layer: int = 0


class PdmuxBCGExecutor:
    # The runner owns live static buffers, context managers and eager logits
    # dispatch; this leaf needs its existing interface, not a copy of config.
    def __init__(self, runner):
        self.runner = runner
        self.owner = None
        self.session = None

    def can_run(self, batch, forward_count):
        if forward_count < 1:
            raise ValueError("Split BCG requires a positive layer count")
        if self.owner is batch:
            return True
        if self.owner is not None:
            raise RuntimeError("A split BCG already owns the prefill buffers")
        r = self.runner
        if (
            batch.split_index != 0
            or batch.spec_info is not None
            or batch.contains_mm_inputs()
            or not r.can_run_graph(batch)
        ):
            return False
        size = r._pad_to_bucket(len(batch.input_ids), r.capture_num_tokens)
        key = r._shape_key(size, batch)
        return r.backend.has_layer_range(
            key, 0, r.model_runner.model_config.num_hidden_layers
        )

    def execute(self, batch, forward_count):
        # Contexts are entered for EACH interval and fully restored before the
        # scheduler submits decode. No model.forward monkey-patch spans a yield.
        from sglang.srt.model_executor.runner_utils.shared_read_event import (
            maybe_publish_prefill_shared_read_done,
        )

        r = self.runner
        start = batch.split_index
        last = r.model_runner.model_config.num_hidden_layers
        end = min(start + forward_count, last)
        with r.model_context(), r.backend.replay_session():
            if start == 0:
                if self.owner is not None:
                    raise RuntimeError("Split BCG session was not finalized")
                static, shared_ends = r.load_batch(batch)
                size = len(static.input_ids)
                key = r._shape_key(size, batch)
                self.session = SplitBCGSession(static, key, r.raw_num_tokens)
                self.owner = batch
                maybe_publish_prefill_shared_read_done(
                    r.model_runner, shared_ends, r.device_module
                )
                if r.buffer_registry.has_slot("input_embeds"):
                    r._fill_input_embeds_slot((), {"input_ids": static.input_ids}, size)
            s = self.session
            if self.owner is not batch or s.next_layer != start:
                raise RuntimeError("Non-contiguous split BCG continuation")
            size = len(s.static_batch.input_ids)
            if r._shape_key(size, batch) != s.shape_key:
                raise RuntimeError("Split BCG cannot migrate between captured variants")
            with r._prefill_forward_context(
                s.static_batch, num_tokens=size, raw_num_tokens=s.raw_num_tokens
            ):
                if end == last:
                    output = r._execute_body_capture(
                        batch,
                        s.static_batch,
                        size,
                        s.raw_num_tokens,
                        s.shape_key,
                        pdmux_layer_interval=(start, end),
                    )
                    output = r._finalize_execute_output(output)
                else:
                    r.backend.replay(
                        s.shape_key,
                        s.static_batch,
                        pdmux_layer_interval=(start, end),
                    )
                    output = None
            s.next_layer = end
            if end == last:
                self.owner = self.session = None
            return output
