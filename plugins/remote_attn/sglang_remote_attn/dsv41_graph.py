"""DSV decode graph breaks over native metadata, prepare and attention methods.

Preparation includes fused SWA writes and compressor/indexer side streams. Keep
that native region eager as a unit so no captured kernel reads stale locations
and the native stream joins precede the next graph segment. Projection after
attention, norms, MLP/MoE and the remaining stable forward stay captured.
"""

import inspect
import types
from contextlib import contextmanager

from .segmented_graph import RemoteGraphBackend

PREPARE = (
    "_forward_prepare",
    "_forward_prepare_multi_stream",
    "_forward_prepare_low_ratio_multi_stream",
)


def is_model(runner):
    config = getattr(
        getattr(runner.model_runner, "model_config", None), "hf_config", None
    )
    return getattr(config, "model_type", None) == "deepseek_v41"


class DSV41GraphBackend(RemoteGraphBackend):
    def __init__(self, *args):
        super().__init__(*args)
        # Reuse the native eager decorator (same bridge/pool lifetime contract).
        self.dynamic_break = self._eager_on_graph(self._dynamic)
        self.dynamic_trace = []
        self.dynamic_boundaries = {}
        self.live_targets = {}
        self.bridge_specs = {}
        self.native_keys = []

    def validate_capture_key(self, key):
        variants = getattr(self.runner, "attention_graph_variants", None)
        labels = getattr(variants, "capture_labels", (None,))
        if (
            key.stream_idx is not None
            or key.variant_label is not None
            or key.attention_variant not in labels
        ):
            raise RuntimeError(
                "DSV Remote supports only native decode candidate variants"
            )

    def capture_one(self, key, forward_fn, capture_inputs=None, post_warmup_hook=None):
        self.validate_capture_key(key)
        if key.size in self.remote_keys:
            self.full.capture_one(key, forward_fn, capture_inputs, post_warmup_hook)
        else:
            super().capture_one(key, forward_fn, capture_inputs, post_warmup_hook)
        self.native_keys.append(key)

    def can_run(self, batch, shape_key):
        from .segmented_graph import leased

        if leased(self.runner):
            return self.supports_remote(batch) and shape_key in self.native_keys
        return self.full.can_run(batch, shape_key)

    def replay(self, shape_key, batch, **kwargs):
        if not self.use_remote:
            return super().replay(shape_key, batch, **kwargs)
        if shape_key not in self.native_keys or shape_key.size != batch.batch_size:
            raise RuntimeError("DSV replay has an uncaptured native candidate variant")
        # All candidate-dependent work is eager. One Remote bank per batch can
        # serve every native label, while retaining the native per-token policy.
        with attention_variant(shape_key.attention_variant):
            return super().replay(self.remote_keys[batch.batch_size], batch, **kwargs)

    def begin_remote_capture(self):
        self.dynamic_trace = []

    def validate_remote_capture(self, size):
        layers = range(self.runner.model_runner.model_config.num_hidden_layers)
        expected = [("metadata", -1)]
        expected += [
            item
            for layer in layers
            for item in (("prepare", layer), ("attention", layer))
        ]
        if self.dynamic_trace != expected:
            raise RuntimeError(
                f"DSV Remote dynamic boundaries differ: {self.dynamic_trace}, expected {expected}"
            )
        self.dynamic_boundaries[size] = tuple(self.dynamic_trace)

    def validate_live(self, batch):
        controller = self.runner.model_runner._pre_layerwise
        if not getattr(controller, "dsv41_remote", False):
            raise RuntimeError("DSV graph lacks progressive ownership")
        rows = getattr(batch, "req_pool_indices_cpu", None)
        if rows is None:
            rows = batch.req_pool_indices.cpu().tolist()
        self.live_targets = {}
        for row in rows:
            target = controller.remote_targets.get(int(row))
            if target is not None:
                if not target.progressive or target.closed:
                    raise RuntimeError("DSV graph requires live TP progressive targets")
                target.check()
                self.live_targets[int(row)] = (target, frozenset(target.local_entries))

    @contextmanager
    def execution(self, batch):
        try:
            with super().execution(batch):
                yield
        finally:
            self.live_targets.clear()

    def _dynamic(self, kind, owner, **arguments):
        prepare = kind in PREPARE
        trace_kind = "prepare" if prepare else kind
        capture = self.capture_remote
        batch = self.capture_batch if capture else self.active_batch
        if batch is None or (not capture and not self.use_remote):
            raise RuntimeError("DSV dynamic break has no live decode batch")
        if not capture:
            if self.is_capturing():
                raise RuntimeError("DSV replay unexpectedly capturing")
            controller = self.runner.model_runner._pre_layerwise
            for row, (target, local) in self.live_targets.items():
                target.check()
                if controller.remote_targets.get(
                    row
                ) is not target or local != frozenset(target.local_entries):
                    raise RuntimeError("DSV ownership changed inside a graph token")
        layer = (
            -1
            if kind == "metadata"
            else (
                arguments["layer"].layer_id if kind == "attention" else owner.layer_id
            )
        )
        if capture and self.is_capturing():
            self.dynamic_trace.append((trace_kind, layer))
            if kind == "attention":
                self.capture_layers.append(layer)
        arguments["forward_batch"] = batch
        if kind == "attention":
            # Native V4 shares K/V; separate weak-ref views would lose `k is v`.
            arguments["v"] = arguments["k"]
        try:
            result = owner._dsv41_graph_originals[kind](**arguments)
            key = (batch.batch_size, kind, layer)
            spec = output_signature(result)
            if capture:
                self.bridge_specs[key] = spec
            elif self.bridge_specs.get(key, object()) != spec:
                raise RuntimeError(
                    "DSV eager result differs from captured graph bridge"
                )
            return result
        except BaseException:
            # A failed native preparation can leave side streams without their
            # join. A main-stream event alone cannot prove its storage drained.
            if kind != "attention":
                self.unsafe = True
            raise
        finally:
            if not capture and kind == "attention":
                self.attention_calls += 1

    def manifest(self):
        return (
            tuple(sorted(self.dynamic_boundaries.items())),
            tuple((k.size, k.attention_variant) for k in self.native_keys),
        )

    def cleanup(self):
        super().cleanup()
        self.dynamic_boundaries.clear()
        self.live_targets.clear()
        self.bridge_specs.clear()
        self.native_keys.clear()


def bind(model_runner, owner, name, kind):
    originals = getattr(owner, "_dsv41_graph_originals", None)
    if originals is None:
        originals = owner._dsv41_graph_originals = {}
    if name in originals:
        return
    original = getattr(owner, name)
    signature = inspect.signature(original)
    originals[name] = original
    # Each prepare variant needs its own callable key, while trace uses prepare.
    key = name if kind == "prepare" else kind
    originals[key] = original

    def wrapped(self, *args, **kwargs):
        backend = model_runner._remote_segmented_backend
        if not backend.capture_remote:
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        batch = values.pop("forward_batch")
        if kind == "attention":
            if values.pop("v") is not values["k"]:
                raise ValueError("DSV graph requires shared K/V")
            # Native forward has varargs; compress_ratio is a keyword on the
            # model path, but accept its canonical positional spelling too.
            extra = values.pop("kwargs", {})
            positional = values.pop("args", ())
            if positional:
                if len(positional) != 1 or "compress_ratio" in extra:
                    raise ValueError("DSV graph attention arguments changed")
                extra["compress_ratio"] = positional[0]
            values.update(extra)
        previous, backend.capture_batch = backend.capture_batch, batch
        try:
            return backend.dynamic_break(
                kind if kind != "prepare" else key, self, **values
            )
        finally:
            backend.capture_batch = previous

    setattr(owner, name, types.MethodType(wrapped, owner))


def install(model_runner, runner):
    backend = getattr(runner, "attn_backend", model_runner.attn_backend)
    if (
        not getattr(backend, "is_dsv41", False)
        or backend.trtllm_attn
        or backend.mega_attn
    ):
        raise ValueError("DSV Remote graphs require the native flashmla backend")
    bind(model_runner, backend, "init_forward_metadata_in_graph", "metadata")
    bind(model_runner, backend, "forward", "attention")
    layers = model_runner.model.model.layers
    if len(layers) != model_runner.model_config.num_hidden_layers:
        raise ValueError("DSV Remote graph requires a complete PP=1 model")
    for layer in layers:
        for name in PREPARE:
            bind(model_runner, layer.self_attn, name, "prepare")


def capture_signature(model_runner, runner):
    """Pointer-free native ABI report, exchanged before any extra capture."""
    rows = []
    try:
        backend = runner.attn_backend
        valid = backend.is_dsv41 and not backend.trtllm_attn and not backend.mega_attn
        owners = [
            (backend, "init_forward_metadata_in_graph", ("forward_batch",)),
            (
                backend,
                "forward",
                ("q", "k", "v", "layer", "forward_batch", "args", "kwargs"),
            ),
        ]
        layers = model_runner.model.model.layers
        valid &= len(layers) == model_runner.model_config.num_hidden_layers
        for layer in layers:
            owners.extend(
                (
                    layer.self_attn,
                    name,
                    ("x", "positions", "forward_batch", "attn_backend"),
                )
                for name in PREPARE
            )
        for owner, name, required in owners:
            method = getattr(owner, "_dsv41_graph_originals", {}).get(
                name, getattr(owner, name)
            )
            signature = inspect.signature(method)
            valid &= set(required) <= signature.parameters.keys()
            rows.append((name, tuple(signature.parameters)))
        variants = getattr(runner, "attention_graph_variants", None)
        rows.append(
            (
                "candidate_variants",
                tuple(getattr(variants, "capture_labels", (None,))),
                tuple(getattr(variants, "graph_limits", ())),
            )
        )
        return bool(valid), tuple(rows)
    except (AttributeError, TypeError, ValueError):
        return False, tuple(rows)


def output_signature(value):
    import torch

    if value is None:
        return None
    if torch.is_tensor(value):
        return tuple(value.shape), str(value.dtype), value.device.type
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(output_signature(v) for v in value)
    raise TypeError(f"unsupported DSV graph bridge: {type(value)}")


@contextmanager
def attention_variant(label):
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        _set_capture_attention_variant,
        get_capture_attention_variant,
    )

    previous = get_capture_attention_variant()
    _set_capture_attention_variant(label)
    try:
        yield
    finally:
        _set_capture_attention_variant(previous)
