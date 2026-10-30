"""D-only Remote decode segments, with native full graphs for Local decode.

No model/scheduler loop is copied. Native backends capture the same forward;
attention is an eager break only while capturing the Remote bank. The break
retains static Q/K/V addresses and reads request state from the live replay.
"""

import importlib.util
import logging
import os
import types
from contextlib import contextmanager

logger = logging.getLogger(__name__)
REMOTE_BATCHES = (1, 2, 4)


def _runtime():
    from . import handoff_control

    return handoff_control


def leased(runner):
    controller = getattr(runner.model_runner, "_pre_layerwise", None)
    return bool(
        controller is not None and controller.graph_fast_path and controller.targets
    )


def agree_capture_settings(report, *, group, world_size):
    if world_size == 1:
        return
    import torch.distributed as dist

    reports = [None] * world_size
    dist.all_gather_object(reports, report, group=group)
    if any(item != reports[0] for item in reports):
        raise RuntimeError("Remote graph settings differ across TP ranks")


class RemoteGraphBackend:
    """Compose native full and breakable banks; own their replay lifetime."""

    def __init__(
        self, runner, full, remote, eager_on_graph, is_capturing, restore_pool
    ):
        self.runner = runner
        self.full = full
        self.remote = remote
        self.is_capturing = is_capturing
        self.restore_pool = restore_pool
        self.attention_break = eager_on_graph(self._attention)
        self.capture_remote = False
        self.capture_batch = None
        self.capturing = False
        self.active_batch = None
        self.use_remote = False
        self.capture_layers = []
        self.remote_keys = {}
        self.boundaries = {}
        self.stream = None
        self.done = runner.device_module.Event()
        self.recorded = False
        self.closed = False
        self.unsafe = False
        self.replays = {"remote": 0, "full": 0}
        self.attention_calls = 0

    @contextmanager
    def capture_session(self, stream):
        self.stream = stream
        self.capturing = True
        try:
            with self.full.capture_session(stream):
                yield
        finally:
            self.stream = None
            self.capturing = False

    def capture_one(
        self, shape_key, forward_fn, capture_inputs=None, post_warmup_hook=None
    ):
        self.full.capture_one(shape_key, forward_fn, capture_inputs, post_warmup_hook)
        if shape_key.size not in REMOTE_BATCHES:
            return
        if any(
            getattr(shape_key, key) is not None
            for key in ("stream_idx", "variant_label", "attention_variant")
        ):
            raise RuntimeError("Remote graph supports only plain decode graph keys")
        self.capture_layers = []
        self.capture_remote = True
        try:
            # The Remote bank owns a private pool. Opening its native session
            # per bucket also gives each bucket a separately owned output bridge.
            with self.remote.capture_session(self.stream):
                self.remote.capture_one(
                    shape_key, forward_fn, capture_inputs, post_warmup_hook
                )
        finally:
            self.capture_remote = False
            self.restore_pool()
        expected = list(range(self.runner.model_runner.model_config.num_hidden_layers))
        if self.capture_layers != expected:
            raise RuntimeError(
                f"Remote graph attention boundaries differ: {self.capture_layers}, expected {expected}"
            )
        self.remote_keys[shape_key.size] = shape_key
        self.boundaries[shape_key.size] = tuple(self.capture_layers)
        logger.info(
            "[remote-attn-graph] captured remote batch=%s attention_breaks=%s",
            shape_key.size,
            len(expected),
        )

    def supports_remote(self, batch):
        key = self.remote_keys.get(batch.batch_size)
        return bool(
            key is not None
            and batch.forward_mode.is_decode()
            and batch.input_ids.numel() == batch.batch_size
            and batch.input_embeds is None
            and batch.replace_embeds is None
            and self.remote.can_run(batch, key)
        )

    def can_run(self, batch, shape_key):
        if leased(self.runner):
            return (
                self.supports_remote(batch)
                and self.remote_keys[batch.batch_size] == shape_key
            )
        return self.full.can_run(batch, shape_key)

    @contextmanager
    def execution(self, batch):
        if self.closed or self.unsafe or self.active_batch is not None:
            raise RuntimeError("Remote graph backend is closed or already executing")
        use_remote = leased(self.runner)
        if use_remote and not self.supports_remote(batch):
            raise RuntimeError(
                "Remote graph must fall back before executing an unsupported batch"
            )
        if use_remote:
            # Freeze the per-token layer masks before the first segment. Never
            # retain a Req or a capture-time dummy ForwardBatch in the callback.
            meta = _runtime()._row_meta(batch)
            for state in meta["states"]:
                _runtime()._check_target_state(state)
        self.use_remote, self.active_batch = use_remote, batch
        try:
            yield
        finally:
            # Even a failed RPC can leave earlier segments queued. Retain both
            # banks and bridges until this stream has drained at cleanup.
            try:
                if self.use_remote:
                    self.done.record(self.runner.device_module.current_stream())
                    self.recorded = True
            except BaseException:
                self.unsafe = True
                raise
            finally:
                self.active_batch = None
                self.use_remote = False

    @contextmanager
    def replay_session(self):
        backend = self.remote if self.use_remote else self.full
        with backend.replay_session():
            yield

    def replay(self, shape_key, batch, **kwargs):
        if self.active_batch is not batch:
            raise RuntimeError("Remote graph replay has no matching live batch")
        name = "remote" if self.use_remote else "full"
        backend = self.remote if self.use_remote else self.full
        if self.use_remote and self.remote_keys.get(batch.batch_size) != shape_key:
            raise RuntimeError("Remote graph cannot replay a padded or mismatched key")
        output = backend.replay(shape_key, batch, **kwargs)
        self.replays[name] += 1
        from sglang.srt.environ import envs

        if envs.SGLANG_LOG_DECODE_GRAPH_KEY.get():
            logger.info(
                "[remote-attn-graph] replay bank=%s batch=%s", name, batch.batch_size
            )
        return output

    def _attention(self, module, q, k, v, save_kv_cache=True, **kwargs):
        if self.capture_remote:
            if self.capture_batch is None:
                raise RuntimeError("Remote capture has no dummy batch")
            if self.is_capturing():
                self.capture_layers.append(module.layer_id)
            # Warmup/capture uses native Local attention, never an RPC.
            return module._handoff_orig_forward(
                q, k, v, self.capture_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        batch = self.active_batch
        if not self.use_remote or batch is None or self.is_capturing():
            raise RuntimeError("Remote attention break is outside its live replay")
        if q.shape[0] != batch.batch_size:
            raise RuntimeError("Remote attention break cannot consume padding rows")
        result = _runtime()._handoff_attention_forward(
            module, q, k, v, batch, save_kv_cache=save_kv_cache, **kwargs
        )
        if (
            result.shape != (batch.batch_size, module.tp_q_head_num * module.v_head_dim)
            or result.dtype != q.dtype
        ):
            raise RuntimeError("Remote attention output differs from its graph bridge")
        self.attention_calls += 1
        return result

    def manifest(self):
        return tuple(sorted(self.boundaries.items()))

    def cleanup(self):
        if self.closed:
            return
        if self.unsafe:
            raise RuntimeError(
                "Remote graph completion is unproven; retaining graph banks"
            )
        if self.recorded:
            self.done.synchronize()
        self.remote.cleanup()
        self.full.cleanup()
        self.remote_keys.clear()
        self.boundaries.clear()
        self.closed = True


def install_attention(model_runner):
    from sglang.srt.layers.radix_attention import RadixAttention

    for module in model_runner.model.modules():
        if not isinstance(module, RadixAttention):
            continue
        if not getattr(module, "_handoff_wrapped", False):
            module._handoff_orig_forward = module.forward

        def forward(self, q, k, v, batch, save_kv_cache=True, **kwargs):
            backend = model_runner._remote_segmented_backend
            if backend.capture_remote:
                # Do not pass the dummy batch to eager_on_graph: it retains
                # non-tensor arguments in the replay closure.
                previous, backend.capture_batch = backend.capture_batch, batch
                try:
                    return backend.attention_break(
                        self, q, k, v, save_kv_cache=save_kv_cache, **kwargs
                    )
                finally:
                    backend.capture_batch = previous
            if backend.capturing or not _runtime().has_target_states():
                return self._handoff_orig_forward(
                    q, k, v, batch, save_kv_cache=save_kv_cache, **kwargs
                )
            return _runtime()._handoff_attention_forward(
                self, q, k, v, batch, save_kv_cache=save_kv_cache, **kwargs
            )

        module.forward = types.MethodType(forward, module)
        module._handoff_wrapped = True


def resolve_backend(original, runner):
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel

    parallel = get_parallel()
    option = os.environ.get("HANDOFF_REMOTE_SEGMENTED_GRAPH", "1")
    settings = (
        get_disagg().disaggregation_mode,
        os.environ.get("HANDOFF_PRE_LAYERWISE", "off"),
        option,
    )
    config = get_exec()
    try:
        cuda_python = importlib.util.find_spec("cuda.bindings.runtime") is not None
    except ModuleNotFoundError:
        cuda_python = False
    mr = runner.model_runner
    controller = getattr(mr, "_pre_layerwise", None)
    live = bool(
        _runtime().has_target_states()
        or controller is not None
        and (controller.targets or controller.sources)
    )
    report = (
        settings,
        tuple(runner.capture_bs),
        type(runner.model_runner.model).__name__,
        runner.enable_torch_compile,
        config.graph.cuda_graph_config.decode.backend,
        config.graph.debug_cuda_graph,
        config.features.enable_memory_saver,
        mr.model_config.num_hidden_layers,
        mr.lora_manager is not None,
        live,
        cuda_python if settings == ("decode", "remote", "1") else True,
    )
    # This vote precedes all extra capture barriers. Even a disabled rank joins,
    # so asymmetric plugin settings cannot hang a peer in segmented capture.
    agree_capture_settings(
        report,
        group=parallel.tp_group.cpu_group if parallel.tp_size > 1 else None,
        world_size=parallel.tp_size,
    )
    if option not in ("0", "1"):
        raise ValueError("HANDOFF_REMOTE_SEGMENTED_GRAPH must be 0 or 1")
    full = original(runner)
    if settings != ("decode", "remote", "1"):
        return full
    if not cuda_python:
        raise RuntimeError(
            "Remote segments require cuda-python (cuda.bindings.runtime)"
        )
    from sglang.srt.distributed.device_communicators.pynccl_allocator import (
        set_graph_pool_id,
    )
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend,
    )
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        is_capturing,
    )

    mr = runner.model_runner
    if (
        type(full) is not FullCudaGraphBackend
        or runner.enable_torch_compile
        or get_exec().graph.debug_cuda_graph
        or get_exec().features.enable_memory_saver
    ):
        raise ValueError(
            "Remote segments require native full decode graphs without torch.compile, debug-eager or memory saver"
        )
    if type(mr.model).__name__ != "Qwen3ForCausalLM" or mr.lora_manager is not None:
        raise ValueError("Remote segments require Qwen3ForCausalLM without LoRA")
    if not set(REMOTE_BATCHES).issubset(runner.capture_bs):
        raise ValueError(
            "Native decode capture buckets must include 1, 2, 4 for Remote segments"
        )
    if live:
        raise RuntimeError("Cannot recapture Remote graphs with live handoff leases")
    remote = BreakableCudaGraphBackend(runner)
    remote._pool = runner.device_module.graph_pool_handle()
    backend = RemoteGraphBackend(
        runner,
        full,
        remote,
        eager_on_graph,
        is_capturing,
        lambda: set_graph_pool_id(full._pool),
    )
    mr._remote_segmented_backend = backend
    install_attention(mr)
    return backend
