"""DSV4.1 text-only layerwise PD, with native final state/metadata completion.

Only immutable, compressed global KV/index-K page entries are pipelined. SWA
and compression carry state retain the native last-chunk path and D admission
barrier. The optional first post-PD eager sparse read closure is delegated to
dsv41_remote; early sparse admission and model/scheduler loops are not copied.
"""

import functools
import logging
import time
from dataclasses import dataclass, field

from . import dsv41_receive, dsv41_remote, dsv41_staging
from .pre_layerwise import Controller, Source, read_json, write_json

logger = logging.getLogger(__name__)


def is_dsv41(runner):
    return runner.model_config.hf_config.model_type == "deepseek_v41"


def manifest(pool, kv_args, config):
    """Describe native wire entries by their actual pointer owners, not layer count.

    FP4 index payload and scales may be distinct entries. Page item lengths
    already account for compression and index-page tiling in the native pool.
    Addresses and total pool capacity intentionally stay out of the peer schema.
    """
    if getattr(pool, "_unified_kv", False) or pool.uniform_fp8:
        raise ValueError(
            "DSV4.1 layerwise requires the native non-unified FlashMLA pools"
        )
    if any(
        r not in (0, 1, 2) for r in config.compress_ratios[: config.num_hidden_layers]
    ):
        raise ValueError("DSV4.1 layerwise supports compression ratios 0/1/2 only")
    owners = {}
    for ratio, buffers in pool.kv_pools.items():
        if buffers is None:
            continue
        sources = list(pool.sources_by_ratio.get(ratio, ()))
        if ratio not in (1, 2) or len(buffers.kv_buffer) != len(sources):
            raise ValueError("unsupported DSV4.1 compressed KV ownership")
        layout = buffers.kv_layout.value
        entries = [("kv", buffers.kv_buffer)]
        indexer = pool.index_pools.get(ratio)
        if indexer is not None:
            if getattr(indexer, "index_k_page8", False):
                raise ValueError(
                    "DSV4.1 layerwise does not support HIP index page layout"
                )
            entries.append(("index", indexer.contiguous_page_row_buffers()))
        for kind, tensors in entries:
            if not tensors or len(tensors) % len(sources):
                raise ValueError("incomplete DSV4.1 native buffer entries")
            for i, tensor in enumerate(tensors):
                producer = sources[i % len(sources)]
                if producer not in config.kv_source_layer_ids:
                    raise ValueError("DSV4.1 pool and model producer maps differ")
                pointer = tensor.data_ptr()
                if pointer in owners:
                    raise ValueError(
                        "aliased DSV4.1 native entries need explicit ownership"
                    )
                owners[pointer] = [kind, ratio, producer, layout, str(tensor.dtype)]
    pointers, _lengths, item_lens = pool.get_contiguous_buf_infos()
    if (
        list(pointers) != list(kv_args.kv_data_ptrs)
        or list(item_lens) != list(kv_args.kv_item_lens)
        or len(set(pointers)) != len(pointers)
        or set(pointers) != set(owners)
    ):
        raise ValueError("DSV4.1 native PD buffers differ from the resource map")
    if getattr(kv_args, "num_draft_entries", 0):
        raise ValueError("DSV4.1 layerwise requires DSpark/draft KV disabled")
    swa_pool = (
        pool.request_window.state
        if pool.request_window is not None
        else pool.swa_kv_pool
    )
    if swa_pool is None:
        raise ValueError("DSV4.1 layerwise requires native SWA state storage")
    return {
        "version": 1,
        "page_size": pool.page_size,
        "layers": config.num_hidden_layers,
        "ratios": list(config.compress_ratios[: config.num_hidden_layers]),
        "kv_sources": list(config.kv_source_layer_ids),
        "index_sources": list(config.index_source_layer_ids),
        "candidate_source": config.candidate_source_layer_id,
        "entries": [owners[p] + [int(n)] for p, n in zip(pointers, item_lens)],
        "state_types": [t.name for t in kv_args.state_types],
        "state_item_lens": [list(x) for x in kv_args.state_item_lens],
        "swa_layout": getattr(getattr(swa_pool, "kv_layout", None), "value", None),
        "swa_dtypes": [str(t.dtype) for t in swa_pool.kv_buffer],
        "compression_states": [
            [
                kind,
                layer,
                state.ratio,
                state.ring_size,
                state.request_scoped,
                str(state.kv_score_buffer.kv_score.dtype),
            ]
            for kind, states in enumerate(
                (pool.compress_state_pools, pool.indexer_compress_state_pools)
            )
            for layer, state in enumerate(states)
            if state is not None
        ],
    }


def resource_groups(schema, group_size):
    groups = {}
    for entry, (_, _, producer, *_) in enumerate(schema["entries"]):
        group = groups.setdefault(producer // group_size, [[], []])
        group[0].append(producer)
        group[1].append(entry)
    return tuple(
        (max(layers), tuple(indices)) for _, (layers, indices) in sorted(groups.items())
    )


def send_selected(
    original,
    manager,
    mooncake_session_id,
    prefill_kv_indices,
    dst_kv_ptrs,
    dst_kv_indices,
    executor,
    dst_layer_ids=None,
    dst_device_kv_indices=None,
    dst_kv_item_len=None,
    dst_attn_tp_size=None,
    kv_buffer_indices=None,
    transfer_fn=None,
):
    schema = getattr(manager, "_dsv41_layerwise_schema", None)
    if schema is None or kv_buffer_indices is None:
        return original(
            manager,
            mooncake_session_id,
            prefill_kv_indices,
            dst_kv_ptrs,
            dst_kv_indices,
            executor,
            dst_layer_ids=dst_layer_ids,
            dst_device_kv_indices=dst_device_kv_indices,
            dst_kv_item_len=dst_kv_item_len,
            dst_attn_tp_size=dst_attn_tp_size,
            kv_buffer_indices=kv_buffer_indices,
            transfer_fn=transfer_fn,
        )
    # The native selected-entry entry point is MHA-only. Keep the DSV4 adapter
    # here and reuse its generic byte/page sender after exact peer agreement.
    manager._validate_envelope_kv_layout(dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size)
    indices = tuple(kv_buffer_indices)
    peer = manager.decode_kv_args_table.get(mooncake_session_id)
    if (
        peer is None
        or getattr(peer, "requires_dcp_relayout", False)
        or dst_attn_tp_size != manager.attn_tp_size
        or manager.pp_size != 1
        or manager.enable_staging
        or dst_device_kv_indices is not None
        or (
            transfer_fn is not None
            and not dsv41_staging.is_transfer(manager, transfer_fn)
        )
        or dst_layer_ids
        or len(dst_kv_ptrs) != len(schema["entries"])
        or list(dst_kv_ptrs) != list(peer.dst_kv_ptrs)
        or len(manager.kv_args.kv_data_ptrs) != len(schema["entries"])
        or list(manager.kv_args.kv_item_lens) != [e[-1] for e in schema["entries"]]
        or not indices
        or len(set(indices)) != len(indices)
        or any(i < 0 or i >= len(dst_kv_ptrs) for i in indices)
    ):
        raise ValueError("DSV4.1 layerwise peer/layout changed before transfer")
    return manager._send_kvcache_generic(
        mooncake_session_id=mooncake_session_id,
        src_data_ptrs=[manager.kv_args.kv_data_ptrs[i] for i in indices],
        dst_data_ptrs=[dst_kv_ptrs[i] for i in indices],
        item_lens=[manager.kv_args.kv_item_lens[i] for i in indices],
        prefill_data_indices=prefill_kv_indices,
        dst_data_indices=dst_kv_indices,
        executor=executor,
        force_flat=True,
        transfer_fn=transfer_fn,
    )


@dataclass
class DSV41Source(Source):
    groups: tuple = ()
    submitted: set = field(default_factory=set)
    completed: set = field(default_factory=set)
    defer_native_release: bool = True
    indices_event: object = None
    indices_keepalive: object = None

    def complete_group(self, producer, indices):
        self.completed.update(indices)
        write_json(
            self.path / f"resource-{producer}.json",
            {
                "generation": self.generation,
                "producer": producer,
                "buffer_indices": list(indices),
                "at": time.monotonic(),
                "completion": "native_te",
                "decode_ready": False,
            },
        )

    def enqueue(self, producer, event, *, flush=False):
        sender = self.req.disagg_kv_sender
        for ordinal, (boundary, indices) in enumerate(self.groups):
            if boundary in self.submitted or (boundary > producer and not flush):
                continue
            sender.kv_mgr.add_transfer_request(
                sender.bootstrap_room,
                self.indices,
                self.index_slice,
                False,
                wait_event=event,
                indices_event=self.indices_event,
                transfer_fn=dsv41_staging.transfer(sender.kv_mgr)
                if self.stager is not None
                else None,
                kv_buffer_indices=indices,
                staging_chunk_idx=ordinal if self.native_staging else None,
                staging_layer_ids=indices if self.native_staging else None,
                on_complete=functools.partial(self.complete_group, boundary, indices),
                trace_ctx=sender.trace_ctx.copy_for_thread(),
            )
            self.submitted.add(boundary)

    def finalize_native(self, original, scheduler, req, *, last_chunk, end_idx):
        if self.finalized or len(self.submitted) != len(self.groups):
            raise RuntimeError("incomplete or duplicate DSV4.1 layerwise finalization")
        if self.owner.dsv41_remote:
            dsv41_remote.register(self)
        try:
            # req.start_send_idx and sender.curr_idx already cover these global
            # pages. Native finalization sends an empty KV tail AND every native
            # state component, aux metadata and the canonical Success signal.
            result = original(scheduler, req, last_chunk=last_chunk, end_idx=end_idx)
        except BaseException as exc:
            self.error = repr(exc)
            raise
        self.finalized = True
        if not self.owner.dsv41_remote:
            self.remote_closed.set()
        write_json(
            self.path / "prefill.json",
            {
                "generation": self.generation,
                "model_type": "deepseek_v41",
                "prefix_len": self.prefix_len,
                "final_chunk_start_page": self.index_slice.start,
                "final_chunk_end_page": self.index_slice.stop,
                "prefill_chunk_tokens": self.prefill_chunk_tokens,
                "layer_group_size": self.group_size,
                "native_final_states": True,
                "host_staging": self.stager is not None,
                "recv_host_staging": self.recv_host_staging,
                "native_staging": self.native_staging,
                "p_radix_read_lease": self.cache_lease is not None,
                "cached_prefix_tokens": self.cache_lease.cached_tokens
                if self.cache_lease is not None
                else 0,
                "resource_groups": len(self.groups),
                "at": time.monotonic(),
            },
        )
        return result

    def poll(self):
        if not self.owner.dsv41_remote:
            return super().poll()
        from .pre_layerwise import _poll_success

        ready = dsv41_remote.source_ready(self)
        return (
            _poll_success(self.req.disagg_kv_sender)
            and self.release is not None
            and ready
        )


class DSV41Controller(Controller):
    dsv41_layerwise = True

    def validate(self):
        from sglang.srt.runtime_context import (
            get_disagg,
            get_exec,
            get_memory,
            get_spec,
        )

        s = self.scheduler
        self.dsv41_remote = self.config.mode == "remote"
        if not hasattr(self, "remote_targets"):
            self.remote_targets = {}
        if self.config.mode not in ("layerwise", "remote"):
            raise ValueError("DSV4.1 requires layerwise or post-PD eager remote mode")
        if not get_disagg().language_model_only:
            raise ValueError("DSV4.1 layerwise requires --language-model-only")
        if get_spec().speculative_algorithm is not None:
            raise ValueError("DSV4.1 layerwise requires DSpark/speculation disabled")
        if (
            not get_memory().disable_radix_cache
            and s.disaggregation_mode.value != "prefill"
        ):
            raise ValueError("DSV4.1 layerwise requires --disable-radix-cache on D")
        if (
            (self.config.host_staging and s.disaggregation_mode.value != "prefill")
            or (
                self.config.recv_host_staging
                and s.disaggregation_mode.value != "decode"
            )
            or (s.enable_staging and not getattr(self, "receive_installed", False))
            or s.enable_pdmux
            or get_memory().enable_unified_memory
            or get_memory().enable_hierarchical_cache
        ):
            raise ValueError(
                "DSV4.1 staging requires P send / D receive roles; no HiCache/PD multiplexing"
            )
        if get_disagg().disaggregation_decode_allocation_policy == "prefill_complete":
            raise ValueError(
                "DSV4.1 layerwise requires destination allocation before prefill"
            )
        runner = s.tp_worker.model_runner
        if runner.lora_manager is not None:
            raise ValueError("DSV4.1 layerwise does not support LoRA")
        if s.kv_checksum_computer is not None:
            raise ValueError(
                "DSV4.1 layerwise does not support whole-prefix KV checksums"
            )
        self.pool = s.token_to_kv_pool_allocator.get_kvcache()
        self.hf_config = runner.model_config.hf_config
        self.executing = None
        self._layer_handles = []
        role = s.disaggregation_mode.value
        if role not in ("prefill", "decode"):
            raise ValueError("DSV4.1 layerwise requires a native PD role")
        if role == "prefill" and get_exec().graph.enable_torch_compile:
            raise ValueError(
                "DSV4.1 layerwise P completion hooks require torch.compile disabled"
            )
        manager = (
            s.disagg_prefill_bootstrap_queue.kv_manager
            if role == "prefill"
            else s.disagg_decode_prealloc_queue.kv_manager
        )
        if (
            manager.enable_staging and not getattr(self, "receive_installed", False)
        ) or (role != "prefill" and getattr(manager, "_host_staging_all", False)):
            raise ValueError(
                "DSV4.1 staging requires P send / D receive roles; native MHA staging must be disabled"
            )
        self._schema(manager)  # reject unsupported storage before serving
        if self.dsv41_remote:
            dsv41_remote.validate(self, runner, role)

        if role == "prefill" and (
            self.config.host_staging or getattr(manager, "_host_staging_all", False)
        ):
            self.stager = dsv41_staging.bind(self.pool, manager)
        if (
            role == "decode"
            and self.config.recv_host_staging
            and not getattr(self, "receive_installed", False)
        ):
            dsv41_receive.install(self, manager, role)
            self.receive_installed = True
        self.config.directory.mkdir(parents=True, exist_ok=True)
        # Attach no model hooks on D. Its native completion, graphs and overlap
        # remain authoritative; it never receives an early-admission offer.
        if s.disaggregation_mode.value == "prefill":
            layers = runner.model.model.layers
            if len(layers) != self.layers:
                raise ValueError("DSV4.1 layerwise requires a complete PP=1 model")
            for index, layer in enumerate(layers):
                self._layer_handles.append(
                    layer.register_forward_hook(
                        functools.partial(self._after_layer, index)
                    )
                )

    def _schema(self, manager):
        return manifest(self.pool, manager.kv_args, self.hf_config)

    def offer(self, queue, dr):
        target = super().offer(queue, dr)
        if target is not None:
            offer = read_json(target.path.parent / "offer.json")
            offer["dsv41"] = self._schema(dr.kv_receiver.kv_mgr)
            offer["mode"] = self.config.mode
            offer["remote_boundary"] = (
                "first-post-PD-decode" if self.dsv41_remote else None
            )
            if self.dsv41_remote:
                dr.req._dsv41_remote_offer = target.path, target.generation
            offer["staging_wire"] = (
                dsv41_receive.WIRE if self.config.recv_host_staging else None
            )
            # No early D admission/graph lease: native queue owns cancellation,
            # deferred abort ACK, H2D drain and metadata teardown exactly once.
            dr.req._pre_layerwise_native_target = None
            write_json(target.path.parent / "offer.json", offer)
        return target

    def early_targets(self, queue, rids_to_check):
        # Physical global entries are not dense model layers. Native staging
        # polling supplies the H2D+metadata+TP barrier; never run Target.refresh.
        return [], [self.offer(queue, dr) for dr in queue.queue]

    def candidates(self, batch, *, allow_middle=False):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender
        from sglang.srt.disaggregation.utils import DisaggregationMode

        s = self.scheduler
        if (
            s.disaggregation_mode != DisaggregationMode.PREFILL
            or not batch.forward_mode.is_extend()
            or batch.forward_mode.is_split_prefill()
            or batch.forward_mode.is_mixed()
            or batch.is_prefill_only
            or batch.return_hidden_states
            or batch.input_embeds is not None
            or batch.replace_embeds is not None
        ):
            return [None] * len(batch.reqs)
        candidates = []
        for req in batch.reqs:
            candidate = None
            sender = getattr(req, "disagg_kv_sender", None)
            lease = getattr(req, "_dsv41_p_lease", None)
            if (
                type(sender) is MooncakeKVSender
                and not req.pending_bootstrap
                and not getattr(req, "session_id", None)
                and (allow_middle or req.inflight_middle_chunks <= 0)
                and (allow_middle or req.extend_end == len(req.origin_input_ids))
                and req.multimodal_inputs is None
                and req.input_embeds is None
                and getattr(req, "_pre_layerwise_source", None) is None
                and not getattr(
                    getattr(req, "_dsv41_p_lease", None), "cancelling", False
                )
                and (
                    not allow_middle
                    or self.config.select_group_size(
                        self.layers, len(req.origin_input_ids), req.extend_len
                    )
                )
            ):
                mgr = sender.kv_mgr
                if self.stager is not None and (
                    self.stager is not getattr(mgr, "_core_stager", None)
                    or self.stager.poisoned
                ):
                    raise RuntimeError("DSV4.1 host staging unavailable; retaining KV")
                root = (
                    self.config.directory
                    / str(int(req.bootstrap_room))
                    / str(self.rank)
                )
                offer = read_json(root / "offer.json")
                peers = tuple(
                    (peer, mgr.decode_kv_args_table.get(peer.mooncake_session_id))
                    for peer in mgr.transfer_infos.get(req.bootstrap_room, {}).values()
                )
                schema = self._schema(mgr)
                page = schema["page_size"]
                if (
                    offer
                    and (lease is None or offer.get("generation") == lease.generation)
                    and offer.get("dsv41") == schema
                    and offer.get("mode") == self.config.mode
                    and offer.get("tp_size") == self.tp_size
                    and offer.get("prefix_len") == len(req.origin_input_ids)
                    and len(peers) == 1
                    and not peers[0][0].is_dummy
                    and peers[0][1] is not None
                    and offer.get("session") == peers[0][0].mooncake_session_id
                    and peers[0][1].dst_attn_tp_size == mgr.attn_tp_size
                    and not peers[0][1].requires_dcp_relayout
                    and peers[0][0].dst_device_kv_indices is None
                    and len(peers[0][1].dst_kv_ptrs) == len(schema["entries"])
                    and not mgr.enable_staging
                    and (
                        not offer.get("native_staging")
                        and not offer.get("recv_host_staging")
                        or offer.get("native_staging")
                        and offer.get("recv_host_staging")
                        and offer.get("staging_wire") == dsv41_receive.WIRE
                        and getattr(peers[0][1], "staging_base_ptr", 0) > 0
                        and getattr(peers[0][1], "staging_total_size", 0) > 0
                    )
                    and mgr.kv_shard_size == 1
                    and not mgr.enable_all_cp_ranks_for_transfer
                    and not mgr.is_dummy_cp_rank
                    and req.disagg_decode_prefix_len == 0
                    and 0 <= req.start_send_idx < req.extend_end
                    and req.start_send_idx % page == 0
                    and sender.curr_idx == req.start_send_idx // page
                    and sender.num_kv_indices
                    == (len(req.origin_input_ids) + page - 1) // page
                ):
                    candidate = req, sender, root, offer, peers, schema
            candidates.append(candidate)
        return candidates

    def prepare(self, batch, *, plan=None):
        s = self.scheduler
        if plan is None:
            candidates = self.candidates(batch)
            accepted = self.agree([int(c is not None) for c in candidates])
        else:
            candidates = plan.candidates
            accepted = [
                int(take and chunk.final)
                for take, chunk in zip(plan.accepted, plan.chunks)
            ]
        tokens = sum(req.extend_len for req, take in zip(batch.reqs, accepted) if take)
        group_size = self.config.select_group_size(
            self.layers,
            max(
                tokens,
                max(
                    (
                        len(req.origin_input_ids)
                        for req, take in zip(batch.reqs, accepted)
                        if take
                    ),
                    default=0,
                ),
            )
            if plan is not None
            else tokens,
            sum(req.extend_len for req in batch.reqs) // len(batch.reqs),
        )
        if not group_size:
            return None
        jobs = []
        for candidate, take in zip(candidates, accepted):
            if not take:
                continue
            req, sender, root, offer, peers, schema = candidate
            page = schema["page_size"]
            indices_event = keepalive = None
            if plan is not None:
                from .prefill_overlap import page_indices

                indices, indices_event, keepalive = page_indices(
                    s.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, req.start_send_idx : req.extend_end
                    ],
                    page,
                )
                plan.indices_refs.append(keepalive)
            else:
                indices = (
                    (
                        s.req_to_token_pool.req_to_token[
                            req.kv.req_pool_idx,
                            req.start_send_idx : req.extend_end : page,
                        ]
                        // page
                    )
                    .to("cpu")
                    .numpy()
                    .copy()
                )
            indices, index_slice, last, skip = sender._prepare_send_indices(indices)
            if skip or not last or len(indices) != index_slice.stop - index_slice.start:
                raise RuntimeError("unexpected DSV4.1 page reservation")
            groups = resource_groups(schema, group_size)
            if offer.get("native_staging"):
                dsv41_receive.install(self, sender.kv_mgr, "prefill")
                sender.kv_mgr.request_layer_staging(
                    sender.bootstrap_room,
                    index_slice,
                    tuple((indices, None) for _, indices in groups),
                )
            source = DSV41Source(
                self,
                req,
                root / offer["generation"],
                offer["generation"],
                indices,
                index_slice,
                peers,
                len(req.origin_input_ids),
                group_size=group_size,
                prefill_chunk_tokens=req.extend_len,
                groups=groups,
                stager=self.stager,
                native_staging=offer.get("native_staging", False),
                recv_host_staging=offer.get("recv_host_staging", False),
                cache_lease=(
                    getattr(getattr(req, "_dsv41_p_lease", None), "cache_lease", None)
                    if plan is not None
                    else self.cache_guard.hold(req)
                    if getattr(self, "cache_guard", None) is not None
                    else None
                ),
                indices_event=indices_event,
                indices_keepalive=keepalive,
            )
            sender.kv_mgr._dsv41_layerwise_schema = schema
            sender._record_transfer_indices(indices, None)
            sender._transfer_start_time = (
                sender._transfer_start_time or time.perf_counter()
            )
            req._pre_layerwise_source = source
            req.start_send_idx = req.extend_end
            self.sources.append(source)
            jobs.append(source)
        if jobs:
            self.batches += 1
            logger.info(
                "[dsv41-layerwise] batch=%s requests=%s group_size=%s final_chunk_tokens=%s",
                self.batches,
                len(jobs),
                group_size,
                tokens,
            )
        return jobs or None

    def _after_layer(self, layer_id, module, args, result):
        overlap = getattr(self, "prefill_overlap", None)
        plan = overlap.current if overlap is not None else None
        jobs = plan.jobs if plan is not None else self.executing
        if jobs is None or not any(
            boundary <= layer_id and boundary not in source.submitted
            for source in jobs
            for boundary, _ in source.groups
        ):
            return
        import torch

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        for source in jobs:
            source.enqueue(layer_id, event)

    def forward(self, original, runner, fb, *args, **kwargs):
        if self.dsv41_remote and self.scheduler.disaggregation_mode.value == "decode":
            return dsv41_remote.forward(self, original, runner, fb, *args, **kwargs)
        overlap = getattr(self, "prefill_overlap", None)
        if overlap is not None:
            return overlap.forward(original, runner, fb, *args, **kwargs)
        if self.plan is None:
            return original(runner, fb, *args, **kwargs)
        import torch

        jobs, self.plan = self.plan, None
        model = runner.model.model
        tail = model.decoder_replay_tail
        self.executing = jobs
        # Native bounded-replay math stays enabled. Only its optional nested
        # tail graph is bypassed so Python completion hooks run on this pass.
        model.decoder_replay_tail = None
        try:
            result = original(runner, fb, *args, **kwargs)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            for source in jobs:
                # Native replay may elide a producer's module call while still
                # writing its global KV. Export any such entries at forward end.
                source.enqueue(self.layers - 1, event, flush=True)
            return result
        except BaseException as exc:
            for source in jobs:
                source.error = repr(exc)
                write_json(
                    source.path / "error.json",
                    {
                        "generation": source.generation,
                        "error": source.error,
                    },
                )
            raise
        finally:
            model.decoder_replay_tail = tail
            self.executing = None
