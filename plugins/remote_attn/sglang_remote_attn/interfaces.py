"""Progressive core contracts, checked without constructing a CUDA runtime."""

import ast
import dataclasses
import inspect
import pkgutil
import textwrap
from pathlib import Path

DISAGG = "sglang.srt.disaggregation."
MANAGER = DISAGG + "mooncake.conn.MooncakeKVManager."
STAGING = DISAGG + "common.staging_handler."

# Explicit optional parameters are required: **kwargs can silently discard them.
# Values pin the native, feature-disabled defaults as well as keyword support.
CORE_CALLS = {
    DISAGG + "decode.DecodeTransferQueue._commit_transfer_to_req": {
        "keep_receiver": False
    },
    MANAGER + "add_transfer_request": {
        "wait_event": None,
        "kv_buffer_indices": None,
        "on_complete": None,
        "transfer_fn": None,
        "staging_chunk_idx": None,
        "staging_layer_ids": None,
    },
    MANAGER + "send_kvcache": {"kv_buffer_indices": None, "transfer_fn": None},
    MANAGER + "_send_kvcache_generic": {"transfer_fn": None, "force_flat": False},
    DISAGG + "prefill.SchedulerDisaggregationPrefillMixin.send_kv_chunk": {
        "last_chunk": False,
        "end_idx": None,
    },
    DISAGG + "common.conn.CommonKVSender._prepare_send_indices": {
        "state_indices": None
    },
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool.get_contiguous_buf_infos": {},
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool.get_state_buf_infos": {},
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool.get_request_state_buf_infos": {},
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool.request_state_transfer_indices": {},
    MANAGER + "_transfer_data": {"stage": False},
    MANAGER + "request_layer_staging": {},
    MANAGER + "send_layer_group_to_staging": {},
    MANAGER + "ensure_host_stager": {},
    MANAGER + "set_kv_buffer_tensors": {},
    STAGING + "init_staging_allocator": {"pin_memory": False},
    STAGING + "PrefillStagingStrategy.check_ready": {"chunk_idx": None},
    STAGING + "DecodeStagingHandler._scatter_region": {
        "layer_ids": None,
        "staging_bytes": None,
    },
    STAGING + "DecodeStagingHandler.advance_scatter": {},
    STAGING + "DecodeStagingHandler.is_done": {},
    STAGING + "DecodeStagingHandler.is_failed": {},
    STAGING + "DecodeStagingHandler.register_decode_req": {},
    STAGING + "DecodeStagingHandler.unregister_decode_req": {},
    STAGING + "DecodeStagingHandler.submit_last_scatter_async": {},
    DISAGG + "common.staging_buffer.StagingBuffer.__init__": {"pin_memory": False},
    DISAGG + "common.staging_buffer.StagingAllocator.__init__": {"pin_memory": False},
    DISAGG + "mooncake.pinned_staging_core.CorePinnedStager.reserve": {},
    DISAGG + "mooncake.pinned_staging_core.CorePinnedStager.set_buffers": {},
    DISAGG + "mooncake.pinned_staging_core.CorePinnedStager.send": {},
    DISAGG + "mooncake.pinned_staging_core.CorePinnedStager.close": {},
    DISAGG + "pinned_staging_buffer.PinnedKVStager.reserve": {},
    DISAGG + "pinned_staging_buffer.PinnedKVStager.send": {
        "ready_event": inspect.Parameter.empty,
        "already_ready": False,
    },
    DISAGG + "pinned_staging_buffer.PinnedKVStager.close": {},
    DISAGG + "pinned_staging_buffer.CudaPinnedSlot.warmup": {},
    DISAGG + "pinned_staging_buffer.CudaPinnedSlot.copy": {},
    DISAGG + "pinned_staging_buffer.CudaPinnedSlot.close": {},
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph.eager_on_graph": {
        "capture_stub": None,
        "enable": None,
    },
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph.is_capturing": {},
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.capture_one": {
        "capture_inputs": None,
        "post_warmup_hook": None,
    },
    "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend.BreakableCudaGraphBackend.capture_one": {
        "capture_inputs": None,
        "post_warmup_hook": None,
    },
    "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend.BreakableCudaGraphBackend.capture_session": {
        "stream": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.lock": {
        "node": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.unlock": {
        "lock": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.base_prefix_cache.BasePrefixCache.finish": {
        "handle": inspect.Parameter.empty,
        "outcome": inspect.Parameter.empty,
    },
    "sglang.srt.mem_cache.radix_cache.RadixCache.checkpoint": {
        "up_to": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.radix_cache.RadixCache.reset": {},
    DISAGG
    + "prefill.SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk": {
        "req": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.checkpoint": {
        "up_to": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.lock": {
        "node_id": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.unlock": {
        "lock": inspect.Parameter.empty
    },
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.reset": {},
    "sglang.srt.managers.scheduler.Scheduler.is_disable_overlap_for_batch": {
        "batch": inspect.Parameter.empty,
        "last_batch": inspect.Parameter.empty,
    },
    "sglang.srt.managers.utils.GenerationBatchResult.copy_to_cpu": {
        "return_logprob": inspect.Parameter.empty,
        "return_hidden_states": True,
    },
    "sglang.srt.managers.async_d2h_copy_worker.HostCopyDone.query": {},
    "sglang.srt.managers.scheduler.Scheduler.process_batch_result": {
        "batch": inspect.Parameter.empty,
        "result": inspect.Parameter.empty,
    },
    "sglang.srt.managers.scheduler.Scheduler.launch_batch_sample_if_needed": {
        "batch_result": inspect.Parameter.empty,
        "cur_batch": inspect.Parameter.empty,
    },
}
CORE_PROPERTIES = (DISAGG + "mooncake.pinned_staging_core.CorePinnedStager.poisoned",)
CORE_FIELDS = {
    "sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver": (
        "get_last_batch",
        "recv_from_tokenizer",
        "recv_from_rpc",
        "max_recv_per_poll",
    ),
    "sglang.srt.managers.utils.GenerationBatchResult": (
        "copy_done",
        "delay_sample_func",
    ),
    "sglang.srt.mem_cache.base_prefix_cache.CacheRequestHandle": ("rid", "attempt_id"),
    "sglang.srt.mem_cache.base_prefix_cache.TreeLock": ("node", "receipt"),
    DISAGG + "common.utils.TransferKVChunk": (
        "kv_buffer_indices",
        "on_complete",
        "transfer_fn",
        "staging_chunk_idx",
        "staging_layer_ids",
    ),
}
# Instance fields cannot be inspected by constructing the real GPU owners.
# Inspect their initialization declarations in the actual loaded Python sources.
CORE_STATE = {
    "sglang.srt.managers.async_d2h_copy_worker.HostCopyDone.__init__": (
        "self",
        ("error",),
    ),
    "sglang.srt.managers.scheduler.Scheduler.run_batch": (
        "batch_result",
        ("copy_done",),
    ),
    "sglang.srt.managers.schedule_batch.Req.__init__": (
        "self",
        ("cache_request_handle", "last_node", "lock"),
    ),
    "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.__init__": (
        "self",
        ("tree_components", "cache_controller", "linker", "enable_session_radix_cache"),
    ),
    "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.__init__": (
        "self",
        ("_pool",),
    ),
    "sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend.BreakableCudaGraphBackend.__init__": (
        "self",
        ("_pool",),
    ),
    MANAGER + "__init__": ("self", ("_core_stager", "_host_staging_all")),
    DISAGG + "common.conn.CommonKVReceiver.send_metadata": (
        "self",
        ("chunk_staging_layers",),
    ),
    STAGING + "DecodeStagingHandler.__init__": ("self", ("_scatter_lock",)),
    STAGING + "DecodeStagingHandler.register_decode_req": (
        "decode_req",
        ("_staging_layer_events", "_staging_completed_layers"),
    ),
    DISAGG + "common.staging_buffer.StagingBuffer.__init__": (
        "self",
        ("pin_memory", "gpu_buffer"),
    ),
    DISAGG + "pinned_staging_buffer.PinnedKVStager.__init__": (
        "self",
        ("poisoned", "closed", "registered"),
    ),
}
CORE_ENVS = (
    "SGLANG_MOONCAKE_SEND_LOCK",
    "SGLANG_MOONCAKE_HOST_STAGING_ALL",
)


def incompatible(target, detail):
    raise RuntimeError(
        f"Incompatible SGLang core: {target}: {detail}. "
        "Install SGLang core and remote_attn from the same progressive revision."
    )


class SourceAPI:
    """Resolve declarations from a checkout without importing model dependencies."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def resolve(self, target):
        parts = target.split(".")
        for index in range(len(parts), 0, -1):
            path = self.root / "python" / Path(*parts[:index]).with_suffix(".py")
            if path.is_file():
                if path not in self.cache:
                    self.cache[path] = ast.parse(path.read_text())
                node = self.cache[path]
                break
        else:
            incompatible(target, "module is missing")
        for name in parts[index:]:
            node = next(
                (item for item in node.body if getattr(item, "name", None) == name),
                None,
            )
            if node is None:
                incompatible(target, "declaration is missing")
        return node


def source_signature(node):
    """Only literal defaults are relevant; never evaluate checkout expressions."""
    parameters = []
    args = node.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + args.defaults
    groups = (
        (positional, defaults),
        (args.kwonlyargs, args.kw_defaults),
    )
    for group, values in groups:
        for arg, value in zip(group, values):
            default = inspect.Parameter.empty
            if value is not None:
                try:
                    default = ast.literal_eval(value)
                except (ValueError, TypeError):
                    default = object()
            kind = inspect.Parameter.KEYWORD_ONLY
            if arg in args.posonlyargs:
                kind = inspect.Parameter.POSITIONAL_ONLY
            parameters.append((arg.arg, kind, default))
    return {name: (kind, default) for name, kind, default in parameters}


def check_parameters(target, actual, expected):
    for name, default in expected.items():
        if name not in actual:
            incompatible(target, f"missing explicit parameter {name}")
        kind, value = actual[name]
        if kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            incompatible(
                target, f"{name} must be an explicit keyword-compatible parameter"
            )
        if value is not default:
            incompatible(target, f"unexpected default for {name}; expected {default!r}")


def declared_fields(node):
    fields = set()
    for item in node.body:
        targets = item.targets if isinstance(item, ast.Assign) else []
        if isinstance(item, ast.AnnAssign):
            targets = [item.target]
        fields.update(target.id for target in targets if isinstance(target, ast.Name))
    return fields


def check_state(target, node, owner, fields):
    assigned = set()
    for item in ast.walk(node):
        targets = item.targets if isinstance(item, ast.Assign) else []
        if isinstance(item, ast.AnnAssign):
            targets = [item.target]
        for attribute in targets:
            if (
                isinstance(attribute, ast.Attribute)
                and isinstance(attribute.value, ast.Name)
                and attribute.value.id == owner
            ):
                assigned.add(attribute.attr)
    for name in fields:
        if name not in assigned:
            incompatible(target, f"missing state initialization {owner}.{name}")


def check_source_interfaces(root):
    api = SourceAPI(root)
    checked = []
    for target, expected in CORE_CALLS.items():
        node = api.resolve(target)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            incompatible(target, "callable declaration is missing")
        check_parameters(target, source_signature(node), expected)
        checked.append(target)
    for target, fields in CORE_FIELDS.items():
        declared = declared_fields(api.resolve(target))
        for name in fields:
            if name not in declared:
                incompatible(target, f"missing dataclass field {name}")
        checked.append(target)
    for target in CORE_PROPERTIES:
        node = api.resolve(target)
        if not any(
            isinstance(item, ast.Name) and item.id == "property"
            for item in getattr(node, "decorator_list", [])
        ):
            incompatible(target, "property declaration is missing")
        checked.append(target)
    for target, (owner, fields) in CORE_STATE.items():
        check_state(target, api.resolve(target), owner, fields)
        checked.append(target + ":state")
    envs = declared_fields(api.resolve("sglang.srt.environ.Envs"))
    for name in CORE_ENVS:
        if name not in envs:
            incompatible("sglang.srt.environ.Envs", f"missing env descriptor {name}")
        checked.append(name)
    return checked


def check_runtime_interfaces(resolve=None):
    """Check installed objects, not a potentially unrelated source checkout."""
    resolve = resolve or pkgutil.resolve_name

    def loaded(target):
        try:
            return resolve(target)
        except (ImportError, AttributeError, ValueError) as exc:
            incompatible(target, f"cannot resolve installed interface ({exc})")

    checked = []
    for target, expected in CORE_CALLS.items():
        function = loaded(target)
        if not callable(function):
            incompatible(target, "installed interface is not callable")
        actual = {
            name: (parameter.kind, parameter.default)
            for name, parameter in inspect.signature(function).parameters.items()
        }
        check_parameters(target, actual, expected)
        checked.append(target)
    for target, fields in CORE_FIELDS.items():
        cls = loaded(target)
        if not dataclasses.is_dataclass(cls):
            incompatible(target, "installed interface is not a dataclass")
        declared = {field.name for field in dataclasses.fields(cls)}
        for name in fields:
            if name not in declared:
                incompatible(target, f"missing dataclass field {name}")
        checked.append(target)
    for target in CORE_PROPERTIES:
        if not isinstance(loaded(target), property):
            incompatible(target, "installed interface is not a property")
        checked.append(target)
    for target, (owner, fields) in CORE_STATE.items():
        try:
            source = inspect.getsource(inspect.unwrap(loaded(target)))
        except (OSError, TypeError) as exc:
            incompatible(
                target, f"cannot inspect installed state initialization ({exc})"
            )
        check_state(target, ast.parse(textwrap.dedent(source)), owner, fields)
        checked.append(target + ":state")
    envs = loaded("sglang.srt.environ.envs")
    for name in CORE_ENVS:
        if not callable(getattr(getattr(envs, name, None), "get", None)):
            incompatible("sglang.srt.environ.envs", f"missing env descriptor {name}")
        checked.append(name)
    return checked
