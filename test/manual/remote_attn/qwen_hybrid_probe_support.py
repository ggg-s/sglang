"""Native row/KV/state allocation and real byte layouts for hybrid D2D probes.

Source loading isolates optional model imports; allocation/registration/state
methods are the production bodies. Only model execution and external I/O are
replaced in CPU cases. CUDA cases use the native Mooncake TCP engine.
"""

import ast
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from dsv_d2d_probe_support import Environment, Sampling
from qwen_reservation_probe_support import Fixture, load
from sglang_remote_attn import dsv_d2d as d2d
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import pre_layerwise as pl
from sglang_remote_attn import qwen_hybrid as hybrid


class HybridEnvironment(Environment):
    def __init__(self, root, device="cpu"):
        # Import before allocator fixtures install scoped runtime boundaries.
        # Never synthesize layer_types: native Qwen configs discard that HF key.
        from sglang.srt.configs.qwen3_5 import Qwen3_5Config
        from sglang.srt import runtime_context

        self.native_runtime = runtime_context

        config = Qwen3_5Config(
            architectures=["Qwen3_5ForConditionalGeneration"],
            language_model_only=True,
            text_config={
                "num_hidden_layers": 8,
                "full_attention_interval": 4,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 6,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 3,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "layer_types": ["linear_attention"] * 3
                + ["full_attention"]
                + ["linear_attention"] * 3
                + ["full_attention"],
            },
        )
        super().__init__(root, device=device, gpu=device == "cuda")
        self.stack.enter_context(
            patch.dict(
                __import__("os").environ,
                {"HANDOFF_D2D_GROUP_SIZE": "8", "HANDOFF_D2D_TOKEN_RANGE": "3"},
            )
        )
        self.native = []
        for controller in self.nodes:
            f = Fixture(root, device=device, page=1)
            self.native.append(f)
            self.stack.callback(f.close)
            old = controller.scheduler
            s = f.scheduler
            s.disaggregation_mode = old.disaggregation_mode
            s.enable_overlap = False
            s.result_queue, s.last_batch, s.running_batch = [], None, old.running_batch
            s._pre_layerwise = controller
            controller.scheduler = s
            controller.hybrid_gdn = True
            controller.layers = 8
            s.tp_worker = NS(
                model_runner=NS(
                    model_config=NS(
                        hf_config=config,
                        num_hidden_layers=config.text_config.num_hidden_layers,
                    )
                )
            )
            if device == "cuda":
                s.tp_worker.model_runner.forward_stream = torch.cuda.current_stream()
            # Native registration and logical->physical layer lookup.
            full_cls = load(
                root,
                "mem_cache/memory_pool.py",
                "MHATokenToKVPool",
                {
                    "get_key_buffer",
                    "get_value_buffer",
                    "_get_key_buffer",
                    "_get_value_buffer",
                    "_pd_registerable_tensors",
                    "get_contiguous_buf_infos",
                    "_build_kv_buffer_descs",
                    "_kv_tokens_per_row",
                },
                {
                    "np": __import__("numpy"),
                    "KvBufferDesc": load(
                        root, "mem_cache/memory_pool.py", "KvBufferDesc"
                    ),
                },
            )
            full = full_cls()
            full.start_layer, full.layer_transfer_counter = 0, None
            full.dtype, full.is_quantized_kv_cache = torch.bfloat16, False
            full.kv_cache_layout, full.use_hnd = "nhd", False
            full.size, full.page_size, full.layer_num, full.store_dtype = (
                64,
                1,
                2,
                torch.bfloat16,
            )
            full.k_buffer = [
                torch.zeros((65, 1, 256), device=device, dtype=torch.bfloat16)
                for _ in range(2)
            ]
            full.v_buffer = [torch.zeros_like(t) for t in full.k_buffer]
            full._kv_buffer_descs = full._build_kv_buffer_descs()
            mamba_cls = load(
                root,
                "mem_cache/memory_pool.py",
                "MambaPool",
                {"_iter_transfer_state_entries", "get_contiguous_buf_infos"},
            )
            m = mamba_cls()
            tree = ast.parse(
                (root / "python/sglang/srt/mem_cache/memory_pool.py").read_text()
            )
            native_m = next(
                n for n in tree.body if getattr(n, "name", None) == "MambaPool"
            )
            constant = next(
                n.value
                for n in native_m.body
                if isinstance(n, ast.Assign)
                and any(
                    getattr(t, "id", None) == "_NON_TRANSFER_STATE_FIELDS"
                    for t in n.targets
                )
            )
            m._NON_TRANSFER_STATE_FIELDS = eval(
                compile(ast.Expression(constant), "native-state-fields", "eval")
            )
            m.mamba_layer_ids = [0, 1, 2, 4, 5, 6]
            m._slot_siblings, m.conv_slice_axis = [], 0
            m.enable_linear_replayssm = m.enable_linear_replayssm_spec = False
            m.replayssm_spec_write_pos = m.replayssm_write_pos = None
            m.mamba_cache = NS(
                conv=[torch.zeros((6, 5, 640, 3), device=device, dtype=torch.bfloat16)],
                temporal=torch.zeros(
                    (6, 5, 3, 128, 128), device=device, dtype=torch.float32
                ),
            )
            hybrid_cls = load(
                root,
                "mem_cache/memory_pool.py",
                "HybridLinearKVPool",
                {
                    "get_contiguous_buf_infos",
                    "get_state_buf_infos",
                    "get_key_buffer",
                    "get_value_buffer",
                    "_transfer_full_attention_id",
                    "_wait_for_layer",
                },
            )
            pool = hybrid_cls()
            pool.full_kv_pool, pool.mamba_pool = full, m
            pool.full_attention_layer_id_mapping = {3: 0, 7: 1}
            pool.full_layer_nums, pool.page_size, pool.dtype = 2, 1, torch.bfloat16
            pool.layer_transfer_counter = None
            controller.pool = pool
            f.allocator.get_kvcache = lambda pool=pool: pool
            state_cls = load(
                root,
                "mem_cache/kv/state_storage.py",
                "StateStorage",
                {
                    "__init__",
                    "pending_request_slots",
                    "acquire",
                    "release",
                    "_give_back",
                    "alloc_ping_pong_buffer",
                },
                {
                    "async_h2d": lambda values, **kw: torch.tensor(values, **kw),
                    "StateSlotsExhausted": RuntimeError,
                },
            )
            slots_cls = load(root, "mem_cache/allocator/mamba.py", "MambaSlotAllocator")
            state = state_cls(
                sub_pool="mamba",
                pool=m,
                slots=slots_cls(4, device),
                layer_map={l: i for i, l in enumerate(m.mamba_layer_ids)},
                short_conv_pool=None,
                ngram_pool=None,
            )
            f.allocator.state = f.pool.state = state
            f.pool.device = device
            f.pool.holds_state_slots, f.pool.state_sub_pool = True, "mamba"
            f.pool.enable_mamba_extra_buffer = False
            f.pool.mamba_initial_tracking_slots = 2
            f.pool.req_index_to_mamba_index_mapping = torch.zeros(
                4, dtype=torch.int32, device=device
            )
            ids = __import__("dataclasses").dataclass(
                load(root, "mem_cache/kv/alloc/id_space.py", "IdSpace")
            )
            state_ids = load(
                root,
                "mem_cache/kv/alloc/id_space.py",
                "state_id_space",
                extra={"IdSpace": ids},
            )
            self.stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.mem_cache.kv.alloc.id_space": NS(
                            state_id_space=state_ids
                        )
                    },
                )
            )
            base = load(
                root,
                "mem_cache/allocator/base.py",
                "BaseTokenToKVPoolAllocator",
                {"state_ids", "state_id_spaces"},
            )
            f.allocator.state_ids = base.state_ids.__get__(f.allocator)
            f.allocator.state_id_spaces = base.state_id_spaces.__get__(f.allocator)
            capacity = load(root, "mem_cache/allocation.py", "ensure_mamba_capacity")
            self.stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.mem_cache.allocation": NS(
                            ensure_mamba_capacity=capacity
                        )
                    },
                )
            )
            # Fixture's native acquire_rows must see the native exhaustion type.
            alloc_ns = (
                f.queue._pre_alloc.__globals__["kv_storage_of"]
                .__globals__["StaticKVStorage"]
                .acquire.__globals__["acquire_rows"]
                .__globals__
            )
            alloc_ns["StateSlotsExhausted"] = RuntimeError
            f.scheduler.tree_cache.on_release = lambda req, checkpointed, f=f: (
                f.allocator.state.release(f.pool, req)
            )
            native_new = f.new

            def factory(scheduler, rid, origin, output, params, native_new=native_new):
                req = native_new(scheduler, rid, origin, output, params)
                req.cache_request_handle = (rid, 0)
                req.grammar, req.return_logprob, req.multimodal_inputs = (
                    None,
                    False,
                    None,
                )
                req._handoff_req, req._handoff_frozen = True, False
                req.full_untruncated_fill_ids = list(origin) + list(output)
                return req

            f.make = factory
            s.disagg_decode_prealloc_queue.kv_manager = (
                old.disagg_decode_prealloc_queue.kv_manager
            )
            s.disagg_decode_prealloc_queue.kv_manager.enable_staging = False
            args = s.disagg_decode_prealloc_queue.kv_manager.kv_args
            ptrs, sizes, strides = pool.get_contiguous_buf_infos()
            sp, sz, ss = pool.get_state_buf_infos()
            args.__dict__.update(
                kv_data_ptrs=ptrs,
                kv_data_lens=sizes,
                kv_item_lens=strides,
                state_types=[NS(name="MAMBA")],
                state_data_ptrs=[sp],
                state_data_lens=[sz],
                state_item_lens=[ss],
            )
            controller._schema = lambda mgr, controller=controller: hybrid.manifest(
                controller, mgr
            )
            controller.layer_entries = {
                l: tuple(
                    i
                    for i, e in enumerate(
                        controller._schema(d2d.manager(controller))["entries"]
                    )
                    if e[2] == l
                )
                for l in (3, 7)
            }
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {"sglang.srt.sampling.sampling_params": NS(SamplingParams=Sampling)},
            )
        )
        self.hub.scheduler = self.source.scheduler
        self.source.scheduler._handoff_hub = self.hub
        self.req = self.native[0].make(
            self.source.scheduler, "source", list(range(8)), [31, 32, 33], Sampling()
        )
        self.native[0].queue._pre_alloc(self.req)
        self.req.kv.mamba_needs_clear = False
        self.source.scheduler.running_batch.reqs = [self.req]
        self.source.scheduler.handoff_tracked_reqs["source"] = self.req
        self.prefix = self.req.kv.kv_committed_len
        for i, (desc, table, ids) in enumerate(
            hybrid.state_rows(self.source, self.req, self.prefix)
        ):
            table[ids] = (torch.arange(table.shape[1], device=device) % 233 + i).to(
                torch.uint8
            )
        for i, tensor in enumerate(hybrid.kv_buffers(self.source.pool)):
            tensor.copy_(
                (
                    torch.arange(tensor.numel(), device=device).reshape(tensor.shape)
                    % 31
                    + i
                ).to(tensor.dtype)
            )
        # Target slot 1 is occupied: migration must use its actual native slot 2.
        self.occupied = self.native[1].allocator.state.slots.alloc(1)
        self.stack.enter_context(patch.object(rt, "_new_req", self.native[1].make))
        self.stack.enter_context(
            patch(
                "sglang_remote_attn.dsv41_staging.buffers",
                side_effect=lambda pool, manager: hybrid.kv_buffers(pool),
            )
        )
        native_release = sys.modules["sglang.srt.mem_cache.common"].release_kv_cache

        def release(req, tree, *, checkpoint):
            apply = lambda: native_release(req, tree, checkpoint=checkpoint)
            if not pl.defer_release(req, apply, checkpoint=checkpoint):
                apply()

        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {"sglang.srt.mem_cache.common": NS(release_kv_cache=release)},
            )
        )
        self.stack.enter_context(
            patch.dict(
                __import__("os").environ,
                {"HANDOFF_REMOTE_GPU_TRANSFER": "1" if device == "cuda" else "0"},
            )
        )

    def startup(self, role, *, prefix_cache=None):
        """Run the real controller startup with native config and pool methods.

        Pool construction and the heavyweight attention model are boundaries;
        config construction, layer properties, validation and manifest are real.
        """
        from contextlib import ExitStack

        owner = self.source if role == "prefill" else self.target
        scheduler = NS(**vars(owner.scheduler))
        scheduler.disaggregation_mode = NS(value=role)
        scheduler.kv_checksum_computer = None
        if prefix_cache is not None:
            scheduler.tree_cache = prefix_cache
        manager = d2d.manager(owner)
        scheduler.disagg_prefill_bootstrap_queue = NS(kv_manager=manager)
        runner = NS(**vars(owner.scheduler.tp_worker.model_runner))
        runner.lora_manager = None
        scheduler.tp_worker = NS(model_runner=runner)

        class AttentionBoundary(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer_id = layer
                self.sliding_window_size = -1
                self.logit_cap = 0.0
                self.pos_encoding_mode = "NONE"

            def forward(self, *args, **kwargs):
                return self.layer_id

        runner.model = torch.nn.ModuleList([AttentionBoundary(i) for i in (3, 7)])
        pool_types = NS(
            HybridLinearKVPool=type(owner.pool),
            MHATokenToKVPool=type(owner.pool.full_kv_pool),
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.mem_cache.memory_pool": pool_types,
                        "sglang.srt.layers.radix_attention": NS(
                            RadixAttention=AttentionBoundary
                        ),
                        "sglang.srt.runtime_context": self.native_runtime,
                    },
                )
            )
            stack.enter_context(
                self.native_runtime.get_context().override_server_args(
                    disable_radix_cache=prefix_cache is None or prefix_cache.disable,
                    language_model_only=True,
                    disable_overlap_schedule=True,
                    disaggregation_mode=role,
                )
            )
            stack.enter_context(
                self.native_runtime.get_parallel().override(
                    tp_rank=0,
                    tp_size=1,
                )
            )
            stack.enter_context(
                patch.dict(
                    __import__("os").environ, {"HANDOFF_REMOTE_GPU_TRANSFER": "1"}
                )
            )
            controller = hybrid.Controller(scheduler, owner.config)
            controller.validate()
        return controller

    def close(self):
        super().close()
