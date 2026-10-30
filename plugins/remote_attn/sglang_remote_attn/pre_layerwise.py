"""Opt-in, same-host PD layer pipelining and Remote admission.

The layer-group execution / selected-buffer transfer design follows SGLang
#34905 (a57f4613803f). This is a deliberately narrower Qwen3/MHA adapter, not a
copy of that PR's hybrid/speculative support. Native Mooncake owns ALL payload
writes. A generation-scoped journal under the shared control directory carries
capabilities and completion proofs, never KV payloads.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    try:
        tmp.write_text(json.dumps(value))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class Config:
    mode: str
    directory: Path
    group_size: int = 0
    drain_timeout: float = 300.0
    host_staging: bool = False
    recv_host_staging: bool = False

    @classmethod
    def from_env(cls):
        mode = os.environ.get("HANDOFF_PRE_LAYERWISE", "off")
        if mode == "off":
            return None
        if mode not in ("layerwise", "remote"):
            raise ValueError("HANDOFF_PRE_LAYERWISE must be off, layerwise or remote")
        root = os.environ.get("HANDOFF_PD_DIR")
        if not root:
            raise ValueError(
                "HANDOFF_PD_DIR must name the same local directory on P and D"
            )
        group = int(os.environ.get("HANDOFF_PRE_LAYERWISE_GROUP_SIZE", "0"))
        timeout = float(os.environ.get("HANDOFF_PD_DRAIN_TIMEOUT_S", "300"))
        if group < 0 or timeout <= 0:
            raise ValueError(
                "layer group size must be nonnegative (0 = auto); "
                "drain timeout must be positive"
            )
        staging = os.environ.get(
            "HANDOFF_PRE_LAYERWISE_HOST_STAGING",
            "1" if os.environ.get("MC_FORCE_TCP") == "1" else "0",
        )
        if staging not in ("0", "1"):
            raise ValueError("HANDOFF_PRE_LAYERWISE_HOST_STAGING must be 0 or 1")
        receive = os.environ.get("HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING", "0")
        if receive not in ("0", "1"):
            raise ValueError("HANDOFF_PRE_LAYERWISE_RECV_HOST_STAGING must be 0 or 1")
        return cls(
            mode, Path(root).resolve(), group, timeout, staging == "1", receive == "1"
        )

    def select_group_size(self, layers, transfer_tokens, average_tokens):
        """Select layer groups from the scheduler's actual batch, never re-chunk it.

        Follow #34905's default 3072-token threshold and 10-to-4 stage
        heuristic. Remote keeps one whole-layer batch below the threshold:
        skipping the adapter there would also disable early decode admission.
        A positive override is retained for controlled layer-group experiments.
        """
        if self.group_size:
            return min(layers, self.group_size)
        if layers <= 4 or transfer_tokens < 3072:
            return layers if self.mode == "remote" else 0
        progress = min(1.0, max(0.0, (average_tokens - 3072) / (3 * 3072 - 3072)))
        target_groups = max(4, round(10 - progress * (10 - 4)))
        return max(1, layers // target_groups)


@dataclass
class LayerCoverage:
    """A completion covers a token interval, never just a submitted layer ID."""

    prefix_len: int
    layers: int
    intervals: dict = field(default_factory=dict)

    def complete(self, layer, start, end):
        if not 0 <= layer < self.layers or not 0 <= start < end <= self.prefix_len:
            raise ValueError("invalid layer completion range")
        ranges = sorted(self.intervals.get(layer, []) + [(start, end)])
        merged = []
        for lo, hi in ranges:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        self.intervals[layer] = merged

    def ready(self, layer):
        return self.intervals.get(layer) == [(0, self.prefix_len)]


def buffer_groups(layer_ids, layers, group_size):
    """Return whole MHA K/V pairs, respecting the actual pointer entry order."""
    if len(layer_ids) != 2 * layers or any(
        layer_ids.count(i) != 2 for i in range(layers)
    ):
        raise ValueError(
            "pre-layerwise requires exactly two MHA buffers per model layer"
        )
    return [
        (
            tuple(range(lo, min(lo + group_size, layers))),
            tuple(
                i
                for i, layer in enumerate(layer_ids)
                if lo <= layer < min(lo + group_size, layers)
            ),
        )
        for lo in range(0, layers, group_size)
    ]


def _runtime():
    from . import handoff_control

    return handoff_control


def _poll_success(receiver):
    from sglang.srt.disaggregation.base.conn import KVPoll

    poll = receiver.poll()
    if poll == KVPoll.Failed:
        raise RuntimeError(
            "pre-layerwise transport failed; KV reuse is not proven safe"
        )
    return poll == KVPoll.Success


@dataclass
class Source:
    owner: object
    req: object
    path: Path
    generation: str
    indices: object
    index_slice: slice
    peers: tuple
    prefix_len: int
    service: dict | None = None
    thread: object = None
    release: object = None
    remote_closed: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    responded: bool = False
    finalized: bool = False
    released: bool = False
    created_at: float = field(default_factory=time.monotonic)
    group_size: int = 0
    prefill_chunk_tokens: int | None = None
    stager: object = None
    native_staging: bool = False
    recv_host_staging: bool = False

    def complete(self, layers):
        if self.native_staging:
            # Network completion is not GPU completion. The native receiver's
            # scatter event, observed by Target.refresh, publishes readiness.
            return
        # add_transfer_request shards a room onto one FIFO. Earlier ordinary
        # chunks have settled before this final-chunk subset runs. Thus this
        # proof covers [0, prefix_len), not merely the final chunk. Native
        # staging, relayout and multi-destination fanout remain disallowed.
        for layer in layers:
            write_json(
                self.path / f"layer-{layer}.json",
                {
                    "generation": self.generation,
                    "layer": layer,
                    "start": 0,
                    "end": self.prefix_len,
                    "at": time.monotonic(),
                },
            )

    def finalize(self):
        import numpy as np
        import torch

        if self.finalized:
            raise RuntimeError("duplicate layer-pipelined finalization")
        self.finalized = True
        write_json(
            self.path / "prefill.json",
            {
                "generation": self.generation,
                "rid": self.req.rid,
                "prefix_len": self.prefix_len,
                "final_chunk_start": self.index_slice.start,
                "final_chunk_end": self.index_slice.stop,
                "prefill_chunk_tokens": self.prefill_chunk_tokens,
                "layer_group_size": self.group_size,
                "host_staging": self.stager is not None,
                "recv_host_staging": self.recv_host_staging,
                "native_staging": self.native_staging,
                "at": time.monotonic(),
            },
        )
        s = self.owner.scheduler
        s.disagg_metadata_buffers.set_buf(self.req)
        if self.owner.config.mode == "layerwise":
            self.req.disagg_kv_sender.send(np.empty(0, dtype=np.int32))
            self.remote_closed.set()
            return
        if int(self.req.kv.kv_committed_len) != self.prefix_len:
            raise RuntimeError("Remote prefix is not fully committed")
        self.service = (
            _runtime()
            .get_remote_hub(s)
            .register_handoff(
                self.req,
                {"rid": f"pd-{self.generation}"},
                on_release=lambda ctx: self.remote_closed.set(),
            )
        )
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        device = torch.cuda.current_device()

        def publish():
            try:
                torch.cuda.set_device(device)
                event.synchronize()
                sender = self.req.disagg_kv_sender
                # An independent metadata write can wait for the current TE
                # call's mutex, but does not sit behind the whole KV queue.
                for peer, registration in self.peers:
                    rc = sender.kv_mgr.send_aux(
                        peer, sender.aux_index, registration.dst_aux_ptrs
                    )
                    if rc != 0:
                        raise RuntimeError(f"early metadata transfer failed: {rc}")
                write_json(
                    self.path / "ready.json",
                    {
                        "generation": self.generation,
                        "prefix_len": self.prefix_len,
                        "service": self.service,
                        "at": time.monotonic(),
                    },
                )
                # Canonical completion remains at the tail. It is never used
                # to signal early admission, and cannot race the early writer.
                sender.send(np.empty(0, dtype=np.int32))
            except Exception as exc:
                self.error = repr(exc)
                write_json(
                    self.path / "error.json",
                    {"generation": self.generation, "error": self.error},
                )

        self.thread = threading.Thread(
            target=publish, name="pd-remote-metadata", daemon=True
        )
        self.thread.start()

    def poll(self):
        if self.released:
            return True
        native_complete = _poll_success(self.req.disagg_kv_sender)
        if self.error:
            raise RuntimeError(f"pre-layerwise source failed: {self.error}")
        if time.monotonic() - self.created_at > self.owner.config.drain_timeout:
            raise RuntimeError("pre-layerwise source lease timed out; retaining KV")
        done = read_json(self.path / "done.json")
        if done and done.get("generation") == self.generation and self.service:
            if not self.remote_closed.is_set():
                hub = _runtime().get_remote_hub(self.owner.scheduler)
                # release() tombstones the context before waiting for inflight
                # GPU RPC readers. Its callback is the only release proof.
                ctx = hub.registry.get(
                    self.service["handoff_id"], self.service["epoch"]
                )
                if ctx is not None:
                    hub.registry.release(
                        self.service["handoff_id"], expected_req=self.req
                    )
        return (
            native_complete and self.release is not None and self.remote_closed.is_set()
        )

    def retire(self):
        # Allocator mutations must occur only after every TP rank is ready.
        if self.released:
            return
        if self.thread is not None:
            self.thread.join(timeout=self.owner.config.drain_timeout)
            if self.thread.is_alive():
                raise RuntimeError("metadata writer did not drain")
        release, self.release = self.release, None
        if release is not None:
            release()
        self.req._pre_layerwise_source = None
        self.released = True
        write_json(
            self.path / "released.json",
            {"generation": self.generation, "at": time.monotonic()},
        )


@dataclass
class Target:
    owner: object
    queue: object
    decode_req: object
    path: Path
    generation: str
    coverage: LayerCoverage
    state: object = None
    closed: bool = False
    native_staging: bool = False

    def refresh(self):
        if self.native_staging:
            handler = self.queue.staging_handler
            handler.advance_scatter(self.decode_req)
            if handler.is_failed(self.decode_req):
                raise RuntimeError("native scatter failed; retaining destination KV")
            for layers, start, pages in self.decode_req._staging_completed_layers:
                if start + pages != self.coverage.prefix_len:
                    raise RuntimeError(
                        "native layer scatter does not cover final chunk"
                    )
                # Earlier same-TP ordinary chunks finished on the native sender
                # FIFO before any final-group write. Their prefix is already local.
                for layer in layers:
                    if not self.coverage.ready(layer):
                        self.coverage.complete(layer, 0, self.coverage.prefix_len)
                        write_json(
                            self.path / f"layer-{layer}.json",
                            {
                                "generation": self.generation,
                                "layer": layer,
                                "start": 0,
                                "end": self.coverage.prefix_len,
                                "at": time.monotonic(),
                                "completion": "scatter",
                            },
                        )
        failure = read_json(self.path / "error.json")
        if failure and failure.get("generation") == self.generation:
            raise RuntimeError(f"pre-layerwise peer failed: {failure['error']}")
        if self.state is None:
            return
        for layer in range(self.coverage.layers):
            if self.coverage.ready(layer):
                if not self.state.events[layer].is_set():
                    self.state.layer_ready_at[layer] = time.monotonic()
                    self.state.events[layer].set()
                continue
            proof = read_json(self.path / f"layer-{layer}.json")
            if proof and proof.get("generation") == self.generation:
                if proof.get("layer") != layer:
                    raise RuntimeError("layer completion identity mismatch")
                self.coverage.complete(layer, proof["start"], proof["end"])
                if self.coverage.ready(layer):
                    self.state.layer_ready_at[layer] = time.monotonic()
                    self.state.events[layer].set()
        if all(self.coverage.ready(i) for i in range(self.coverage.layers)):
            if self.state.import_done_at is None:
                self.state.import_done_at = time.monotonic()

    def finish(self, *, wait):
        if self.closed:
            return
        receiver = self.decode_req.kv_receiver
        deadline = time.monotonic() + self.owner.config.drain_timeout
        while not self.ready_to_finish():
            self.refresh()
            if not wait:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("PD writes did not drain; retaining destination KV")
            time.sleep(0.005)
        self.refresh()
        if self.state is not None and not all(
            self.coverage.ready(i) for i in range(self.coverage.layers)
        ):
            raise RuntimeError("canonical completion without all layer proofs")
        if self.native_staging:
            self.queue.staging_handler.unregister_decode_req(
                self.decode_req.req.bootstrap_room
            )
        # Called only between forwards or from release_kv_cache. No next step
        # can snapshot a Remote mask after the state is removed.
        _runtime()._drop_target_state_for(self.decode_req.req)
        receiver.clear()
        self.decode_req.kv_receiver = None
        idx = self.decode_req.metadata_buffer_index
        self.queue.metadata_buffers.bootstrap_room[idx] = 0
        self.queue.req_to_metadata_buffer_idx_allocator.free(idx)
        write_json(
            self.path / "done.json",
            {
                "generation": self.generation,
                "at": time.monotonic(),
                "early_admission": self.state is not None,
                "remote_calls": self.state.remote_calls
                if self.state is not None
                else 0,
                "first_step_at": self.state.admitted_at
                if self.state is not None
                else None,
                "import_done_at": self.state.import_done_at
                if self.state is not None
                else None,
            },
        )
        self.closed = True
        self.decode_req.req._pre_layerwise_target = None
        self.decode_req.req._pre_layerwise_native_target = None

    def ready_to_finish(self):
        self.refresh()
        return _poll_success(self.decode_req.kv_receiver) and (
            not self.native_staging
            or self.queue.staging_handler.is_done(self.decode_req)
        )


class Controller:
    def __init__(self, scheduler, config):
        self.scheduler = scheduler
        self.config = config
        self.rank = _runtime()._tp_rank(scheduler)
        self.tp_size = _runtime()._tp_size(scheduler)
        self.layers = scheduler.tp_worker.model_runner.model_config.num_hidden_layers
        self.sources = []
        self.targets = []
        self.plan = None
        self.batches = 0
        self.early_admissions = 0
        self.stager = None
        # Enabled only after graph configuration/capture manifests agree at
        # startup. Membership of targets is then changed only by TP votes.
        self.graph_fast_path = False

    def validate_graphs(self):
        from sglang.srt.runtime_context import get_exec

        self.graph_fast_path = False
        runner = self.scheduler.tp_worker.model_runner

        def manifest(graph_runner):
            if graph_runner is None:
                return None
            return (
                type(graph_runner).__module__,
                type(graph_runner).__qualname__,
                tuple(getattr(graph_runner, "capture_bs", []) or []),
                tuple(getattr(graph_runner, "capture_num_tokens", []) or []),
                getattr(graph_runner, "max_bs", None),
                getattr(
                    getattr(graph_runner, "backend", None), "manifest", lambda: None
                )(),
            )

        signature = (
            asdict(get_exec().graph.cuda_graph_config),
            manifest(runner.decode_cuda_graph_runner),
            manifest(runner.prefill_cuda_graph_runner),
        )
        clean = not _runtime().has_target_states()
        if self.tp_size > 1:
            import torch.distributed as dist

            reports = [None] * self.tp_size
            dist.all_gather_object(
                reports, (signature, clean), group=self.scheduler.attn_tp_cpu_group
            )
            if any(item[0] != reports[0][0] for item in reports):
                raise RuntimeError("remote_attn TP graph config/captures differ")
            clean = all(item[1] for item in reports)
        # An existing legacy import has no TP-agreed progressive lifetime.
        self.graph_fast_path = clean

    def validate(self):
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_disagg, get_memory

        if (
            self.config.recv_host_staging
            and not envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        ):
            raise ValueError(
                "receive host staging requires SGLANG_DISAGG_STAGING_BUFFER=1"
            )
        s = self.scheduler
        runner = s.tp_worker.model_runner
        if type(runner.model).__name__ != "Qwen3ForCausalLM":
            raise ValueError("pre-layerwise first supports Qwen3ForCausalLM only")
        if runner.lora_manager is not None:
            raise ValueError("pre-layerwise does not yet support LoRA")
        if not callable(getattr(runner.model, "forward_split_prefill", None)):
            raise ValueError("model has no split-prefill entry point")
        if s.kv_checksum_computer is not None:
            raise ValueError("pre-layerwise does not support whole-prefix KV checksums")
        if get_memory().enable_unified_memory or s.enable_pdmux:
            raise ValueError(
                "pre-layerwise does not support unified KV or PD multiplexing"
            )
        if get_disagg().disaggregation_decode_allocation_policy == "prefill_complete":
            raise ValueError(
                "pre-layerwise requires destination allocation before prefill"
            )
        self.config.directory.mkdir(parents=True, exist_ok=True)

    def agree(self, flags):
        import torch
        import torch.distributed as dist

        if not flags or self.tp_size == 1:
            return flags
        values = torch.tensor(flags, dtype=torch.int32)
        dist.all_reduce(
            values, op=dist.ReduceOp.MIN, group=self.scheduler.attn_tp_cpu_group
        )
        return values.tolist()

    def offer(self, queue, dr):
        previous = getattr(dr, "_pre_layerwise_offer", None)
        if previous is not None:
            return previous
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVReceiver

        if type(dr.kv_receiver) is not MooncakeKVReceiver or dr.is_rebootstrap:
            return None
        mgr = dr.kv_receiver.kv_mgr
        if queue._is_host_destination(dr):
            return None
        generation = secrets.token_hex(16)
        root = self.config.directory / str(int(dr.req.bootstrap_room)) / str(self.rank)
        path = root / generation
        target = Target(
            self,
            queue,
            dr,
            path,
            generation,
            LayerCoverage(len(dr.req.origin_input_ids), self.layers),
        )
        if mgr.enable_staging:
            if not queue.staging_handler.is_staging_room(dr.req.bootstrap_room):
                queue.staging_handler.register_decode_req(dr.req.bootstrap_room, dr)
            dr.kv_receiver.require_staging = True
            target.native_staging = True
            dr.req._pre_layerwise_native_target = target
            # An ordinary small request can already have concluded before its
            # first offer. Registration must not lose that terminal notification.
            if _poll_success(dr.kv_receiver):
                queue.staging_handler.submit_last_scatter_async(dr.req.bootstrap_room)
        write_json(
            root / "offer.json",
            {
                "generation": generation,
                "session": mgr.get_session_id(),
                "prefix_len": target.coverage.prefix_len,
                "tp_size": self.tp_size,
                "layers": self.layers,
                "mode": self.config.mode,
                "recv_host_staging": self.config.recv_host_staging,
                "native_staging": mgr.enable_staging,
            },
        )
        dr._pre_layerwise_offer = target
        return target

    def prepare(self, batch):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender
        from sglang.srt.disaggregation.utils import DisaggregationMode

        s = self.scheduler
        if (
            s.disaggregation_mode != DisaggregationMode.PREFILL
            or not batch.forward_mode.is_extend()
        ):
            return None
        if batch.forward_mode.is_split_prefill() or batch.forward_mode.is_mixed():
            return None
        # Splitting executes the entire batch, including non-pipelined rows.
        # The split entry point does not forward embedding overrides/captures.
        if (
            batch.is_prefill_only
            or batch.return_hidden_states
            or batch.input_embeds is not None
            or batch.replace_embeds is not None
        ):
            return None
        candidates = []
        for req in batch.reqs:
            candidate = None
            sender = getattr(req, "disagg_kv_sender", None)
            if (
                not req.pending_bootstrap
                and req.inflight_middle_chunks <= 0
                and req.extend_end == len(req.origin_input_ids)
                and req.multimodal_inputs is None
                and req.input_embeds is None
                and type(sender) is MooncakeKVSender
                and getattr(req, "_pre_layerwise_source", None) is None
            ):
                mgr = sender.kv_mgr
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
                if (
                    offer
                    and len(peers) == 1
                    and not peers[0][0].is_dummy
                    and peers[0][1] is not None
                    and bool(offer.get("native_staging", False)) == mgr.enable_staging
                    and (not mgr.enable_staging or peers[0][1].staging_base_ptr != 0)
                    and offer.get("session") == peers[0][0].mooncake_session_id
                    and offer.get("prefix_len") == len(req.origin_input_ids)
                    and offer.get("tp_size") == self.tp_size
                    and offer.get("layers") == self.layers
                    and offer.get("mode") == self.config.mode
                    and peers[0][1].dst_attn_tp_size == mgr.attn_tp_size
                    and not peers[0][1].requires_dcp_relayout
                    and peers[0][0].dst_device_kv_indices is None
                    and list(mgr.kv_args.kv_layer_ids)
                    == list(peers[0][1].dst_kv_layer_ids)
                    and len(mgr.kv_args.kv_data_ptrs) == len(peers[0][1].dst_kv_ptrs)
                    and not mgr.kv_args.state_data_ptrs
                    and 0 <= req.start_send_idx < len(req.origin_input_ids)
                    and sender.curr_idx == req.start_send_idx
                ):
                    candidate = (req, offer, root, peers)
            candidates.append(candidate)
        accepted = self.agree([int(c is not None) for c in candidates])
        if not any(accepted):
            return None
        # Resolve the policy only after TP agrees on eligible requests. Middle
        # chunks still contribute compute, but not the pipelined-token threshold.
        transfer_tokens = sum(
            req.extend_len for req, take in zip(batch.reqs, accepted) if take
        )
        group_size = self.config.select_group_size(
            self.layers,
            transfer_tokens,
            sum(req.extend_len for req in batch.reqs) // len(batch.reqs),
        )
        if group_size == 0:
            # Fall back before reserving indices or creating a source lease.
            return None
        jobs = []
        for candidate, take in zip(candidates, accepted):
            if not take:
                continue
            req, offer, root, peers = candidate
            sender = req.disagg_kv_sender
            # Dense MHA uses the documented k0..kN,v0..vN layout and
            # deliberately publishes no kv_layer_ids in this SGLang revision.
            layer_ids = list(sender.kv_mgr.kv_args.kv_layer_ids)
            if (
                not layer_ids
                and len(sender.kv_mgr.kv_args.kv_data_ptrs) == 2 * self.layers
            ):
                layer_ids = list(range(self.layers)) * 2
            groups = buffer_groups(layer_ids, self.layers, group_size)
            stager = None
            if self.config.host_staging or getattr(
                sender.kv_mgr, "_host_staging_all", False
            ):
                stager = sender.kv_mgr.ensure_host_stager()
                self.stager = stager
                stager.reserve(s.tp_worker.model_runner.gpu_id)
            indices = (
                s.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, req.start_send_idx : req.extend_end
                ]
                .to("cpu")
                .numpy()
                .copy()
            )
            indices, index_slice, last, skip = sender._prepare_send_indices(indices)
            if skip or not last or len(indices) != index_slice.stop - index_slice.start:
                raise RuntimeError("unexpected layer-pipeline reservation")
            source = Source(
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
                stager=stager,
                native_staging=sender.kv_mgr.enable_staging,
                recv_host_staging=offer.get("recv_host_staging", False),
            )
            sender._record_transfer_indices(indices, None)
            sender._transfer_start_time = (
                sender._transfer_start_time or time.perf_counter()
            )
            req._pre_layerwise_source = source
            req.start_send_idx = req.extend_end
            self.sources.append(source)
            if source.native_staging:
                sender.kv_mgr.request_layer_staging(
                    sender.bootstrap_room, index_slice, groups
                )
            jobs.append((source, groups))
        if jobs:
            self.batches += 1
            logger.info(
                "pre-layerwise batch=%s requests=%s group_size=%s "
                "pipelined_tokens=%s policy=%s host_staging=%s receive_staged_requests=%s",
                self.batches,
                len(jobs),
                group_size,
                transfer_tokens,
                "fixed" if self.config.group_size else "auto",
                self.config.host_staging,
                sum(source.native_staging for source, _ in jobs),
            )
        return jobs or None

    def forward(self, original, runner, fb, *args, **kwargs):
        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if self.plan is None:
            return original(runner, fb, *args, **kwargs)
        jobs, self.plan = self.plan, None
        old_mode = fb.forward_mode
        fb.forward_mode = ForwardMode.SPLIT_PREFILL
        fb.split_index = 0
        try:
            for group_index in range(len(jobs[0][1])):
                layers = jobs[0][1][group_index][0]
                result = original(
                    runner, fb, *args, **{**kwargs, "split_forward_count": len(layers)}
                )
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                for source, groups in jobs:
                    sender = source.req.disagg_kv_sender
                    sender.kv_mgr.add_transfer_request(
                        sender.bootstrap_room,
                        source.indices,
                        source.index_slice,
                        False,
                        wait_event=event,
                        kv_buffer_indices=groups[group_index][1],
                        on_complete=functools.partial(source.complete, layers),
                        transfer_fn=(
                            functools.partial(sender.kv_mgr._transfer_data, stage=True)
                            if source.stager is not None
                            else None
                        ),
                        staging_chunk_idx=group_index
                        if source.native_staging
                        else None,
                        staging_layer_ids=layers if source.native_staging else None,
                        trace_ctx=sender.trace_ctx.copy_for_thread(),
                    )
            return result
        finally:
            fb.forward_mode = old_mode

    def early_targets(self, queue, rids_to_check):
        # Retired receivers are not in queue.queue but still own metadata and
        # prefix writes. Reap them before considering new admissions.
        targets = list(self.targets)
        finished = [
            int(target.closed or target.ready_to_finish()) for target in targets
        ]
        for target, done in zip(targets, self.agree(finished)):
            if done:
                target.finish(wait=False)
                self.targets.remove(target)
            else:
                target.refresh()
        offered = [self.offer(queue, dr) for dr in queue.queue]
        candidates = []
        for target in offered:
            if target is not None:
                target.refresh()
            ready = None
            if target is not None and self.config.mode == "remote":
                data = read_json(target.path / "ready.json")
                dr = target.decode_req
                allowed = rids_to_check is None or dr.req.rid in rids_to_check
                if (
                    allowed
                    and not dr.req.finished()
                    and data
                    and data.get("generation") == target.generation
                    and data.get("prefix_len") == target.coverage.prefix_len
                    and int(
                        queue.metadata_buffers.bootstrap_room[dr.metadata_buffer_index][
                            0
                        ].item()
                    )
                    == int(dr.req.bootstrap_room)
                    and not target.ready_to_finish()
                ):
                    ready = data
            candidates.append(ready)
        accepted = self.agree([int(c is not None) for c in candidates])
        admitted = []
        removed = set()
        for target, ready, take in zip(offered, candidates, accepted):
            if not take:
                continue
            dr = target.decode_req
            queue._commit_transfer_to_req(dr, keep_receiver=True)
            if dr.kv_receiver is None or dr.req.finished():
                raise RuntimeError("early PD metadata commit failed")
            state = _runtime()._TargetHandoffState(
                dr.req.kv.req_pool_idx,
                target.coverage.prefix_len,
                [threading.Event() for _ in range(self.layers)],
                "remote",
                ready["service"],
                handoff_id=ready["service"]["handoff_id"],
                req=dr.req,
                tp_rank=self.rank,
                base_output_len=len(dr.req.output_ids),
                migration_epoch=target.generation,
            )
            # Completion is refreshed between steps, not by a receive thread.
            # A latency-triggered switch to blocking layer waits would prevent
            # that refresh and deadlock. Keep Remote until proofs are observed.
            state.remote_ms_threshold = float("inf")
            target.state = state
            target.refresh()
            dr.req._pre_layerwise_target = target
            _runtime()._install_attention_wrappers(self.scheduler, state)
            self.targets.append(target)
            removed.add(id(dr))
            admitted.append(dr.req)
            self.early_admissions += 1
            write_json(
                target.path / "admitted.json",
                {
                    "generation": target.generation,
                    "at": time.monotonic(),
                    "ready_layers": sum(
                        target.coverage.ready(i) for i in range(self.layers)
                    ),
                },
            )
        queue.queue = [dr for dr in queue.queue if id(dr) not in removed]
        return admitted, offered

    def poll_sources(self):
        # The P HTTP leg must finish before SMG forwards D's stream. Emit a
        # response-only snapshot; the live Req stays in the native transfer
        # queue, with its original KVPoll state and KV ownership intact.
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        sources = list(self.sources)
        ready = [
            int(
                not src.responded
                and self.config.mode == "remote"
                and read_json(src.path / "ready.json") is not None
            )
            for src in sources
        ]
        retired = []
        for src, respond in zip(sources, self.agree(ready)):
            if respond:
                snapshot = copy.copy(src.req)
                snapshot.finished_reason = FINISH_LENGTH(length=0)
                snapshot._pre_layerwise_response_snapshot = True
                self.scheduler.output_streamer.stream_output(
                    [snapshot], snapshot.return_logprob, None
                )
                src.responded = True
            complete = src.poll()
            if (
                self.config.mode == "layerwise"
                and src.finalized
                and src.req.kv.is_kv_released
            ):
                src.req._pre_layerwise_source = None
                complete = True
            retired.append(int(complete))
        # Do not change the collective's vector size independently by rank.
        for src, retire in zip(sources, self.agree(retired)):
            if retire:
                src.retire()
                self.sources.remove(src)


def initialize(scheduler):
    config = Config.from_env()
    if config is None:
        return
    controller = Controller(scheduler, config)
    controller.validate()
    controller.validate_graphs()
    scheduler._pre_layerwise = controller
    scheduler.tp_worker.model_runner._pre_layerwise = controller


def run_batch(original, scheduler, batch, *args, **kwargs):
    controller = getattr(scheduler, "_pre_layerwise", None)
    if controller is None:
        return original(scheduler, batch, *args, **kwargs)
    controller.plan = controller.prepare(batch)
    try:
        return original(scheduler, batch, *args, **kwargs)
    finally:
        controller.plan = None


def forward(original, runner, fb, *args, **kwargs):
    controller = getattr(runner, "_pre_layerwise", None)
    if controller is None:
        return original(runner, fb, *args, **kwargs)
    return controller.forward(original, runner, fb, *args, **kwargs)


def send_chunk(original, scheduler, req, last_chunk=False, end_idx=None):
    source = getattr(req, "_pre_layerwise_source", None)
    if source is None or not last_chunk:
        return original(scheduler, req, last_chunk=last_chunk, end_idx=end_idx)
    source.finalize()


def pop_transferred(original, queue, rids_to_check=None):
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    if controller is None:
        return original(queue, rids_to_check)
    admitted, offered = controller.early_targets(queue, rids_to_check)
    ordinary = original(queue, rids_to_check)
    for target in offered:
        if target is not None and any(req is target.decode_req.req for req in ordinary):
            target.decode_req.req._pre_layerwise_native_target = None
            write_json(
                target.path / "done.json",
                {
                    "generation": target.generation,
                    "at": time.monotonic(),
                    "early_admission": False,
                },
            )
    return ordinary + admitted


def defer_release(req, release):
    source = getattr(req, "_pre_layerwise_source", None)
    if source is not None and (
        source.error is not None
        or (source.stager is not None and source.stager.poisoned)
    ):
        raise RuntimeError("pre-layerwise staging is unsettled; retaining source KV")
    if source is not None and source.owner.config.mode == "remote":
        if not source.finalized:
            raise RuntimeError(
                "cannot recycle a failed Prefill with layer writes in flight"
            )
        if source.release is None:
            source.release = release
        return True
    target = getattr(req, "_pre_layerwise_target", None)
    if target is not None:
        target.finish(wait=True)
    else:
        native_target = getattr(req, "_pre_layerwise_native_target", None)
        if native_target is not None:
            # Before early admission, native queue cleanup still owns receiver
            # and metadata teardown. Fence only; do not clear/free them twice.
            deadline = time.monotonic() + native_target.owner.config.drain_timeout
            while not native_target.ready_to_finish():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "native staging did not drain; retaining destination KV"
                    )
                time.sleep(0.005)
            req._pre_layerwise_native_target = None
    return False


def poll(scheduler):
    controller = getattr(scheduler, "_pre_layerwise", None)
    if controller is not None:
        controller.poll_sources()


def is_fully_idle(original, scheduler, for_health_check=False, ignore_waiting=False):
    controller = getattr(scheduler, "_pre_layerwise", None)
    if (
        not for_health_check
        and controller is not None
        and (controller.sources or controller.targets)
    ):
        # Native queues no longer contain early-admitted/leased requests.
        # Prevent idle flush/reset and false leak checks while they own pages.
        return False
    return original(
        scheduler, for_health_check=for_health_check, ignore_waiting=ignore_waiting
    )


def shutdown(scheduler):
    controller = getattr(scheduler, "_pre_layerwise", None)
    if controller is None:
        return
    for target in controller.targets:
        target.finish(wait=True)
    for source in controller.sources:
        if source.thread is not None:
            source.thread.join(timeout=controller.config.drain_timeout)
            if source.thread.is_alive():
                raise RuntimeError("pre-layerwise metadata writer remains active")
        deadline = time.monotonic() + controller.config.drain_timeout
        while not _poll_success(source.req.disagg_kv_sender):
            if time.monotonic() >= deadline:
                raise RuntimeError("native PD writes remain active during shutdown")
            time.sleep(0.005)
    # Native manager owns shared staging teardown after its workers drain.
    if controller.stager is not None and controller.stager.poisoned:
        raise RuntimeError("pre-layerwise staging remains quarantined during shutdown")


def init_staging_allocator(
    original, register_fn, kv_args, *, device_type, pin_memory=False
):
    config = Config.from_env()
    if config is not None:
        pin_memory = config.recv_host_staging
    return original(
        register_fn, kv_args, device_type=device_type, pin_memory=pin_memory
    )


def init_staging_buffers(original, *args, **kwargs):
    if Config.from_env() is not None:
        # This plugin supports only same-TP selected-buffer sends and has its
        # own bounded D2H adapter. Do not allocate full-chunk GPU gather buffers
        # per transfer worker for the unused heterogeneous-TP gather path.
        return []
    return original(*args, **kwargs)
