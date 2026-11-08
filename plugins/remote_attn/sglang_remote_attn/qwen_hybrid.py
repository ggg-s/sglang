"""Qwen3.5-ABI (including Qwen3.8) text-only hybrid D2D adapter.

GDN always executes locally. Its working Conv/SSM slot is installed before
admission; immutable full-attention K/V entries retire independently thereafter.
The shared D2D protocol owns allocation, transport and quarantine, not this adapter.
"""

import functools

import torch

from . import dsv41_remote, pre_layerwise


def matches(runner):
    return runner.model_config.hf_config.model_type in ("qwen3_5", "qwen3_5_text")


def text_config(runner):
    hf = runner.model_config.hf_config
    return getattr(hf, "text_config", hf)


def native_layer_mapping(config, pool):
    # Use the same logical ids as native model construction and pool allocation.
    # Qwen3_5TextConfig consumes the HF layer_types key without retaining it.
    full = list(config.full_attention_layer_ids)
    linear = list(config.linear_layer_ids)
    if (
        sorted(full + linear) != list(range(config.num_hidden_layers))
        or set(full) != set(pool.full_attention_layer_id_mapping)
        or linear != list(pool.mamba_pool.mamba_layer_ids)
    ):
        raise ValueError("hybrid logical layer ownership differs from native pools")
    return {"full_attention_layer_ids": full, "linear_layer_ids": linear}


def kv_buffers(pool):
    full = pool.full_kv_pool
    tensors = [*full.k_buffer, *full.v_buffer]
    ptrs, sizes, strides = pool.get_contiguous_buf_infos()
    lookup = {t.data_ptr(): t for t in tensors}
    if len(lookup) != len(tensors) or set(ptrs) != set(lookup):
        raise ValueError("hybrid K/V pointer ownership mismatch")
    ordered = [lookup[p] for p in ptrs]
    for t, size, stride in zip(ordered, sizes, strides):
        if (
            not t.is_contiguous()
            or t.nbytes != size
            or stride != t[0].nbytes * pool.page_size
        ):
            raise ValueError("hybrid native K/V registration mismatch")
    return ordered


def state_entries(pool):
    entries = list(pool.mamba_pool._iter_transfer_state_entries())
    if not entries or {field for field, *_ in entries} != {"conv", "temporal"}:
        raise ValueError("hybrid D2D requires native GDN Conv and temporal state only")
    expected = {
        (field, layer)
        for layer in pool.mamba_pool.mamba_layer_ids
        for field in ("conv", "temporal")
    }
    if (
        len(entries) != len(expected)
        or {(field, layer) for field, t, axis, layer in entries} != expected
    ):
        raise ValueError("hybrid GDN state layer ownership incomplete")
    ptrs, sizes, strides = pool.get_state_buf_infos()
    if len(entries) != len(ptrs) or len(set(ptrs)) != len(ptrs):
        raise ValueError("hybrid GDN registration count/alias mismatch")
    for (field, t, axis, layer), ptr, size, stride in zip(
        entries, ptrs, sizes, strides
    ):
        if (
            not t.is_contiguous()
            or t.data_ptr() != ptr
            or t.nbytes != size
            or t[0].nbytes != stride
            or axis != 0
        ):
            raise ValueError("hybrid native GDN state registration mismatch")
    return entries


def manifest(controller, manager):
    pool, args = controller.pool, manager.kv_args
    tensors = kv_buffers(pool)
    ptrs, sizes, strides = pool.get_contiguous_buf_infos()
    states = state_entries(pool)
    sp, sz, ss = pool.get_state_buf_infos()
    if (
        list(args.kv_data_ptrs) != list(ptrs)
        or list(args.kv_data_lens) != list(sizes)
        or list(args.kv_item_lens) != list(strides)
        or [s.name for s in args.state_types] != ["MAMBA"]
        or list(map(list, args.state_data_ptrs)) != [list(sp)]
        or list(map(list, args.state_data_lens)) != [list(sz)]
        or list(map(list, args.state_item_lens)) != [list(ss)]
        or getattr(args, "num_draft_entries", 0)
    ):
        raise ValueError("hybrid native PD registration changed")
    owners = {}
    for layer, physical in pool.full_attention_layer_id_mapping.items():
        for name, buffers in (
            ("k", pool.full_kv_pool.k_buffer),
            ("v", pool.full_kv_pool.v_buffer),
        ):
            t = buffers[physical]
            owners[t.data_ptr()] = ["kv", 1, layer, name, str(t.dtype)]
    config = text_config(controller.scheduler.tp_worker.model_runner)
    return {
        "family": "qwen-gdn-d2d-v1",
        "page_size": pool.page_size,
        "layers": config.num_hidden_layers,
        "tp_size": controller.tp_size,
        "layer_mapping": native_layer_mapping(config, pool),
        "rope": getattr(config, "rope_parameters", None)
        or getattr(config, "rope_scaling", None),
        "rms_norm_eps": getattr(config, "rms_norm_eps", None),
        "geometry": [
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
        ],
        "entries": [owners[t.data_ptr()] + [int(n)] for t, n in zip(tensors, strides)],
        "states": [
            [field, layer, list(t.shape[1:]), str(t.dtype), int(n)]
            for (field, t, axis, layer), n in zip(states, ss)
        ],
    }


def validate_transfer(transfer):
    expected = [
        {"kind": field, "index": i, "layer": layer, "rows": 1, "stride": stride}
        for i, (field, layer, shape, dtype, stride) in enumerate(
            transfer["schema"]["states"]
        )
    ]
    if transfer.get("states") != expected:
        raise ValueError("hybrid GDN state manifest incomplete or reordered")


def freeze_ready(controller):
    device = controller.pool.full_kv_pool.k_buffer[0].device
    if device.type == "cuda":
        runner = controller.scheduler.tp_worker.model_runner
        torch.cuda.current_stream(device).wait_stream(runner.forward_stream)


def state_rows(controller, req, length):
    from .dsv_d2d import matrix

    scheduler, pool = controller.scheduler, controller.pool
    table, allocator = scheduler.req_to_token_pool, scheduler.token_to_kv_pool_allocator
    if not req.kv.holds_mamba:
        raise ValueError("hybrid D2D request has no native GDN working slot")
    # Use the same virtual->physical translation as native PD. A request row,
    # or the request's virtual Mamba id, is not a physical state index.
    slot = (
        allocator.state_ids(table.state_sub_pool)
        .ids(table.req_index_to_mamba_index_mapping[req.kv.req_pool_idx])
        .reshape(-1)
        .long()
    )
    result = []
    strides = pool.get_state_buf_infos()[2]
    for i, ((field, t, axis, layer), stride) in enumerate(
        zip(state_entries(pool), strides)
    ):
        if slot.numel() != 1 or bool(((slot <= 0) | (slot >= t.shape[0])).any()):
            raise ValueError("hybrid GDN physical slot outside native registration")
        result.append(
            (
                {
                    "kind": field,
                    "index": i,
                    "layer": layer,
                    "rows": 1,
                    "stride": int(stride),
                },
                matrix(t, int(stride)),
                slot,
            )
        )
    return result


def clear_state(controller, req, length):
    # Only touch this request's already allocated working slot. Native acquire
    # initializes tracking/replay metadata; it remains owned by native release.
    for desc, tensor, indices in state_rows(controller, req, length):
        tensor.index_fill_(0, indices, 0)


def state_capacity(controller, req):
    from sglang.srt.mem_cache.allocation import ensure_mamba_capacity

    s = controller.scheduler
    return ensure_mamba_capacity(
        s.req_to_token_pool, s.req_to_token_pool.state, [req], s.tree_cache
    )


class Target(dsv41_remote.Target):
    def call(self, query, layer, scale):
        out, lse = call_batch([self], [query], layer, scale)
        return out[0], lse[0]

    def header(self, layer, scale):
        self.check()
        if self.closed:
            raise RuntimeError("hybrid Remote target retired")
        return {
            "handoff_id": self.route["handoff_id"],
            "epoch": self.route["epoch"],
            "token": self.route["token"],
            "qwen_hybrid": True,
            "layer": int(layer),
            "sm_scale": float(scale),
        }


def call_batch(targets, queries, layer, scale):
    from . import remote_data_plane as dp

    if not dp.enabled():
        raise RuntimeError("hybrid Remote requires HANDOFF_REMOTE_GPU_TRANSFER=1")
    first = targets[0]
    _, out, lse = dp.call(
        first.controller,
        first.route,
        queries,
        [t.header(layer, scale) for t in targets],
        [t.req for t in targets],
    )
    for target in targets:
        target.calls += 1
        target.touched.add(layer)
    return out, lse


def attention(controller, original, module, q, k, v, fb, save_kv_cache=True, **kw):
    """Exact prefix/suffix GQA merge; GDN layers never enter this hook."""
    targets = controller.remote_targets
    if not targets:
        return original(q, k, v, fb, save_kv_cache=save_kv_cache, **kw)
    rows = fb.req_pool_indices.cpu().tolist()
    active = [targets.get(int(row)) for row in rows]
    entries = set(controller.layer_entries[module.layer_id])
    remote_rows = [
        t is not None and not entries <= t.local_entries for t in active
    ]
    for target, remote in zip(active, remote_rows):
        trace = getattr(target, "pd_trace", None)
        if trace is not None:
            trace.layer(module.layer_id, remote)
    if not any(remote_rows):
        return original(q, k, v, fb, save_kv_cache=save_kv_cache, **kw)
    if (
        not fb.forward_mode.is_decode()
        or q.shape[0] != len(rows)
        or k is None
        or v is None
        or kw
        or getattr(fb, "mha_return_lse", False)
    ):
        raise ValueError("hybrid Remote supports ordinary one-token text Decode only")
    import flashinfer

    from .remote_data_plane import MAX_ROWS

    pool = controller.pool
    queries = q.reshape(len(rows), module.tp_q_head_num, module.qk_head_dim)
    if save_kv_cache:
        pool.set_kv_buffer(
            module,
            fb.out_cache_loc,
            k.reshape(-1, module.tp_k_head_num, module.qk_head_dim),
            v.reshape(-1, module.tp_v_head_num, module.v_head_dim),
        )
    seq_lens_cpu = getattr(fb, "seq_lens_cpu", None)
    seqs = (fb.seq_lens.cpu() if seq_lens_cpu is None else seq_lens_cpu).tolist()
    output, local_lses, groups = [], [], {}
    for i, (row, length, target, remote) in enumerate(
        zip(rows, seqs, active, remote_rows)
    ):
        if target is not None:
            target.check()
        start = target.prefix_len if remote else 0
        indices = controller.scheduler.req_to_token_pool.req_to_token[
            row, start:length
        ].long()
        if not indices.numel():
            raise ValueError("hybrid Decode suffix must contain the pending token")
        local_o, local_lse = flashinfer.single_decode_with_kv_cache(
            queries[i],
            pool.get_key_buffer(module.layer_id)[indices],
            pool.get_value_buffer(module.layer_id)[indices],
            kv_layout="NHD",
            pos_encoding_mode="NONE",
            sm_scale=float(module.scaling),
            return_lse=True,
        )
        output.append(local_o)
        local_lses.append(local_lse)
        if remote:
            endpoint = target.route["host"], int(target.route["port"])
            groups.setdefault(endpoint, []).append(i)
    for group in groups.values():
        for start in range(0, len(group), MAX_ROWS):
            indices = group[start : start + MAX_ROWS]
            remote_o, remote_lse = call_batch(
                [active[i] for i in indices],
                [queries[i] for i in indices],
                module.layer_id,
                module.scaling,
            )
            i = indices[0]
            local_o, local_lse = (
                (output[i].unsqueeze(0), local_lses[i].unsqueeze(0))
                if len(indices) == 1
                else (
                    torch.stack([output[i] for i in indices]),
                    torch.stack([local_lses[i] for i in indices]),
                )
            )
            merged, _ = flashinfer.merge_state(
                local_o,
                local_lse,
                remote_o,
                remote_lse,
            )
            for i, item in zip(indices, merged):
                output[i] = item
    return torch.stack(output).reshape(len(rows), -1)


def execute(job, stream, cache):
    from .handoff_control import execute_remote_attention

    source = getattr(job.ctx, "dsv41_source", None)
    if source is None or not getattr(source.owner, "hybrid_gdn", False):
        raise ValueError("hybrid RPC has no native immutable-prefix lease")
    layer = job.header.get("layer")
    config = text_config(source.owner.scheduler.tp_worker.model_runner)
    if (
        source.service["epoch"] != job.epoch
        or not job.header.get("qwen_hybrid")
        or job.q_cpu.shape
        != (config.num_attention_heads // source.owner.tp_size, config.head_dim)
        or job.q_cpu.dtype != source.owner.pool.dtype
        or type(layer) is not int
        or layer not in source.owner.layer_entries
        or job.q_cpu.ndim != 2
    ):
        raise ValueError("hybrid RPC layer/query contract mismatch")
    # The hub holds a registry reference through compute and Q/O delivery.
    # Fence the actual freezing stream before a worker stream touches KV.
    source.check_read()
    if source.final_event is None:
        raise RuntimeError("hybrid frozen state has no CUDA readiness proof")
    stream.wait_event(source.final_event)
    return execute_remote_attention(job, stream, cache)


def validate_prefill_cache(cache):
    """Hybrid P requires native FULL+MAMBA checkpoint/CoW for PD read leases."""
    if cache.disable:
        return
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    if (
        type(cache) is not UnifiedRadixCache
        or cache.tree_components != (ComponentType.FULL, ComponentType.MAMBA)
        or cache.cache_controller is not None
        or cache.linker is not None
        or cache.enable_session_radix_cache
    ):
        raise ValueError(
            "hybrid Prefill prefix cache requires native device-only FULL+MAMBA "
            "UnifiedRadixCache without HiCache, external storage or sessions"
        )


class Controller(pre_layerwise.Controller):
    # Reuse the native-state D2D command/lifetime path, never DSV sparse math.
    hybrid_gdn = True
    dsv41_layerwise = True
    dsv41_remote = True

    def validate(self):
        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MHATokenToKVPool,
        )
        from sglang.srt.runtime_context import (
            get_disagg,
            get_memory,
            get_schedule,
            get_spec,
        )

        from . import remote_data_plane as dp

        s = self.scheduler
        runner = s.tp_worker.model_runner
        self.pool = s.token_to_kv_pool_allocator.get_kvcache()
        pool = self.pool
        if (
            self.config.mode != "remote"
            or not dp.enabled()
            or s.disaggregation_mode.value not in ("prefill", "decode")
            or not get_disagg().language_model_only
            or get_spec().speculative_algorithm is not None
            or getattr(s, "enable_hisparse", False)
            or runner.lora_manager is not None
            or s.kv_checksum_computer is not None
            or self.config.recv_host_staging
            or not get_schedule().disable_overlap_schedule
        ):
            raise ValueError(
                "hybrid Remote requires text-only native PD, remote GPU plane, overlap/MTP/LoRA disabled"
            )
        if s.disaggregation_mode.value == "decode":
            if not get_memory().disable_radix_cache or not s.tree_cache.disable:
                raise ValueError(
                    "hybrid Decode Remote requires --disable-radix-cache; "
                    "Prefill prefix cache may stay enabled"
                )
        else:
            validate_prefill_cache(s.tree_cache)
        if (
            type(pool) is not HybridLinearKVPool
            or type(pool.full_kv_pool) is not MHATokenToKVPool
            or pool.full_kv_pool.is_quantized_kv_cache
            or pool.full_kv_pool.store_dtype != pool.dtype
            or pool.page_size != 1
            or pool.full_kv_pool.kv_cache_layout != "nhd"
            or pool.dtype not in (torch.bfloat16, torch.float16)
            or pool.mamba_pool.enable_linear_replayssm
            or pool.mamba_pool.enable_linear_replayssm_spec
        ):
            raise ValueError(
                "hybrid Remote requires native NHD BF16/FP16 page-1 KV and ordinary GDN state"
            )
        config = text_config(runner)
        expected = native_layer_mapping(config, pool)["full_attention_layer_ids"]
        if config.num_hidden_layers != self.layers:
            raise ValueError("hybrid model layer count differs from native config")
        if any(
            t.dtype != torch.float32
            for field, t, *_ in state_entries(pool)
            if field == "temporal"
        ):
            raise ValueError("hybrid Remote first supports FP32 GDN SSM state")
        self.executing = None
        self.remote_targets = {}
        self.pd_visibility = dp.Visibility(pool.full_kv_pool.k_buffer[0].device)
        manager = (
            s.disagg_prefill_bootstrap_queue.kv_manager
            if s.disaggregation_mode.value == "prefill"
            else s.disagg_decode_prealloc_queue.kv_manager
        )
        if manager.enable_staging:
            raise ValueError(
                "hybrid Remote uses native PD without native staging buffers"
            )
        self.schema_signature = self._schema(manager)
        self.layer_entries = {
            layer: tuple(
                i
                for i, e in enumerate(self.schema_signature["entries"])
                if e[2] == layer
            )
            for layer in expected
        }
        from sglang.srt.layers.radix_attention import RadixAttention

        self.attention_modules = []
        for module in runner.model.modules():
            if isinstance(module, RadixAttention):
                if (
                    module.layer_id not in expected
                    or module.sliding_window_size != -1
                    or module.logit_cap
                    or module.pos_encoding_mode != "NONE"
                ):
                    raise ValueError("hybrid full-attention operator is incompatible")
                original = module.forward
                module.forward = functools.partial(attention, self, original, module)
                self.attention_modules.append(module)
        if sorted(m.layer_id for m in self.attention_modules) != expected:
            raise ValueError("hybrid full-attention module map is incomplete")
        self.config.directory.mkdir(parents=True, exist_ok=True)

    def _schema(self, manager):
        return manifest(self, manager)

    def prepare(self, batch):
        if (
            self.scheduler.disaggregation_mode.value == "prefill"
            and batch.forward_mode.is_extend()
        ):
            from .qwen_hybrid_pd import prepare

            prepare(self, batch.reqs)
            return [
                source
                for req in batch.reqs
                if (source := getattr(req, "_pre_layerwise_source", None)) is not None
                and req.extend_end == len(req.origin_input_ids)
            ]
        return None

    def offer(self, queue, dr):
        from .qwen_hybrid_pd import offer

        return offer(self, queue, dr)

    def early_targets(self, queue, rids_to_check):
        from .qwen_hybrid_pd import early_targets

        return early_targets(self, queue, rids_to_check)

    def forward(self, original, runner, fb, *args, **kwargs):
        if (
            any(x is not None for x in (getattr(fb, "mm_inputs", None) or ()))
            or getattr(fb, "input_embeds", None) is not None
            or getattr(fb, "replace_embeds", None) is not None
        ):
            raise ValueError("hybrid Remote first supports pure-text inputs only")
        result = original(runner, fb, *args, **kwargs)
        if self.scheduler.disaggregation_mode.value == "prefill" and self.plan:
            from .remote_data_plane import event

            freeze_ready(self)
            for source in self.plan:
                if source.req.extend_end == source.prefix_len and not source.finalized:
                    source.final_event = event(
                        self.pool.full_kv_pool.k_buffer[0].device
                    )
                    source.prefill_complete = True
                    source.prefix_stream.bound = True
        return result
