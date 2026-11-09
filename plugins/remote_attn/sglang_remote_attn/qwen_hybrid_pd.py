"""Native hybrid PD: GDN/metadata barrier, immutable K/V ranges, Remote reads.

GDN state is transferred once using native physical working-slot indices. Only
full attention K/V is read on P; cached FULL+MAMBA state follows native CoW and
final publication after the read lease retires. No DSV math or receive ring.
"""

import functools
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from . import dsv41_remote, prefix_stream, qwen_hybrid
from .compat import decode_queue_items, remove_decode_items
from .pre_layerwise import LayerCoverage, Source as NativeSource, Target as NativeTarget
from .pre_layerwise import _poll_success, read_json, write_json
from .transport_failure import TransportFailure

logger = logging.getLogger(__name__)


class Source(NativeSource):
    def poll(self):
        try:
            return super().poll()
        except RuntimeError as exc:
            raise TransportFailure(str(exc)) from exc

    def check_read(self):
        if (
            self.error
            or self.released
            or not getattr(self, "barrier_ready", False)
            or self.req.cache_request_handle != self.handle
            or self.req.kv.req_pool_idx != self.row
            or getattr(self.req, "_pre_layerwise_source", None) is not self
            or getattr(self.req, "_remote_attn_transport_quarantine", None) is not None
            or getattr(self.req, "_handoff_transfer_unsafe", False)
        ):
            raise TransportFailure("hybrid PD immutable read lease is unproven")
        if self.cache_lease is not None:
            self.cache_lease.check()
        _poll_success(self.req.disagg_kv_sender)

    def finalize_native(self, original, scheduler, req, **kwargs):
        if not getattr(self, "prefill_complete", False):
            raise RuntimeError("hybrid final send before native forward completion")
        self.prefix_stream.bound = True
        return original(scheduler, req, **kwargs)

    def capture_final(self, send, indices, span, kwargs):
        if (
            self.finalized
            or kwargs.get("aux_index") is None
            or not kwargs.get("state_indices")
        ):
            raise RuntimeError("hybrid final envelope missing native GDN/metadata")
        self.finalized = True
        plan = self.prefix_stream
        priority = plan.priority()
        priority.__enter__()
        try:
            # Retain native final state indices and submit K/V as ordinary
            # bounded entry/range tasks. Canonical success cannot precede them.
            kw = dict(kwargs, state_indices=[], wait_event=self.final_event)
            plan.add(send, indices, span, kw)
            with plan.lock:
                if plan.sealed:
                    raise RuntimeError("duplicate hybrid canonical envelope")
                plan.sealed = True
                plan.final = functools.partial(
                    send,
                    self.req.disagg_kv_sender.kv_mgr,
                    self.req.bootstrap_room,
                    indices[:0],
                    slice(span.stop, span.stop),
                    True,
                    **dict(kw, num_kv_tokens=0),
                )
                plan.publish()
            self.thread = threading.Thread(
                target=self.publish_barrier,
                args=(dict(kwargs), priority),
                name="hybrid-pd-state",
                daemon=True,
            )
            self.thread.start()
        except BaseException:
            priority.__exit__(None, None, None)
            raise

    def publish_barrier(self, kwargs, priority):
        import torch
        from .dsv_d2d import pool_device
        from .transfer_priority import control

        sender = self.req.disagg_kv_sender
        manager = sender.kv_mgr
        try:
            device = pool_device(self.owner)
            if device.type == "cuda":
                torch.cuda.set_device(device)
            self.final_event.synchronize()
            event = kwargs.get("indices_event")
            if event is not None:
                event.synchronize()
            # Native state planners identify the working slot, not a cached
            # checkpoint or a request-row id. Verify it before native TE.
            physical = qwen_hybrid.state_rows(self.owner, self.req, self.prefix_len)[0][
                2
            ]
            if list(map(int, kwargs["state_indices"][0])) != physical.cpu().tolist():
                raise RuntimeError("hybrid final state no longer owns the working slot")
            with control(manager), ThreadPoolExecutor(max_workers=2) as executor:
                for peer, registration in self.peers:
                    if (
                        manager.maybe_send_extra(
                            peer, kwargs["state_indices"], executor, registration
                        )
                        != 0
                    ):
                        raise RuntimeError("hybrid native GDN transfer failed")
                    if (
                        manager.send_aux(
                            peer,
                            kwargs["aux_index"],
                            registration.dst_aux_ptrs,
                            registration.dst_aux_item_lens,
                        )
                        != 0
                    ):
                        raise RuntimeError("hybrid native metadata transfer failed")
            _poll_success(sender)  # Check failure; success is NOT required here.
            self.barrier_ready = True
            dsv41_remote.register(self)
            cached_tokens = getattr(self.req, "cached_tokens_device", None)
            write_json(
                self.path / "ready.json",
                dict(
                    generation=self.generation,
                    prefix_len=self.prefix_len,
                    service=self.service,
                    range_stream=True,
                    hybrid_state_ready=True,
                    p_cached_tokens=cached_tokens,
                    schema=self.owner.schema_signature,
                ),
            )
            logger.info(
                "[handoff-src] hybrid PD rid=%s prefix=%s p_cached_tokens=%s",
                self.req.rid,
                self.prefix_len,
                cached_tokens,
            )
        except BaseException as exc:
            self.error = repr(exc)
            manager._poison_unsettled_send(sender.bootstrap_room)
            manager.conclude_failure(
                bootstrap_room=sender.bootstrap_room, failure_reason=self.error
            )
            write_json(
                self.path / "error.json",
                dict(generation=self.generation, error=self.error),
            )
        finally:
            # Start bulk only after the barrier is published, or retain failed
            # tasks/backing. Native sender failure cannot become canonical OK.
            if self.error:
                self.prefix_stream.pump.quarantine()
            priority.__exit__(None, None, None)


def prepare(controller, reqs):
    """Called in the existing TP batch order, before native cached-prefix sends."""
    from .pd_mesh import source_root
    from .prefix_stream import Plan
    import numpy as np

    min_ratio = controller.remote_min_cache_hit_ratio
    candidates = []
    for req in reqs:
        candidate = None
        source = getattr(req, "_pre_layerwise_source", None)
        if (
            source is None
            and not getattr(req, "_hybrid_split_at", 0)
            and not req.pending_bootstrap
            and req.kv.holds_kv
            and req.start_send_idx == 0
            and (
                min_ratio == 0
                or req.cached_tokens_device > len(req.origin_input_ids) * min_ratio
            )
        ):
            # Native prepare_for_extend computes cached_tokens_device on the
            # first chunk. Later chunks' own KV must never qualify as a hit.
            # Earlier native chunks have no range receipts in this plan. Never
            # create a late stream that cannot prove coverage of its prefix.
            sender = getattr(req, "disagg_kv_sender", None)
            if sender is not None:
                manager = sender.kv_mgr
                root = source_root(controller, req)
                offer = read_json(root / "offer.json") or {}
                peers = tuple(
                    (p, manager.decode_kv_args_table.get(p.mooncake_session_id))
                    for p in manager.transfer_infos.get(req.bootstrap_room, {}).values()
                )
                if (
                    offer.get("hybrid_schema") == controller.schema_signature
                    and offer.get("mode") == "remote"
                    and offer.get("range_stream") == 1
                    and offer.get("gpu_transfer") is True
                    and offer.get("tp_size") == controller.tp_size
                    and offer.get("prefix_len") == len(req.origin_input_ids)
                    and offer.get("decode_prefix_len", 0) == 0
                    and len(peers) == 1
                    and peers[0][1] is not None
                    and offer.get("session") == peers[0][0].mooncake_session_id
                    and not peers[0][0].is_dummy
                    and not peers[0][1].requires_dcp_relayout
                    and not manager.enable_staging
                    and not req.multimodal_inputs
                    and req.input_embeds is None
                    and getattr(req, "session_id", None) is None
                ):
                    candidate = root, offer, peers
        candidates.append(candidate)
    accepted = controller.agree([int(c is not None) for c in candidates])
    if not any(accepted):
        return
    error = None
    for req, candidate, take in zip(reqs, candidates, accepted):
        if not take:
            continue
        try:
            root, offer, peers = candidate
            guard = controller.cache_guard
            lease = None if guard is None else guard.hold(req)
            plan = Plan(controller, req, offer, root / offer["generation"], lease)
            source = Source(
                controller,
                req,
                plan.path,
                plan.generation,
                np.empty(0, dtype=np.int32),
                slice(0, 0),
                peers,
                len(req.origin_input_ids),
                cache_lease=lease,
            )
            source.prefix_stream = plan
            source.handle, source.row = req.cache_request_handle, req.kv.req_pool_idx
            source.barrier_ready = source.prefill_complete = False
            plan.source = source
            req._remote_attn_prefix_stream = plan
            req._pre_layerwise_source = source
            manager = req.disagg_kv_sender.kv_mgr
            if not hasattr(manager, "_prefix_streams"):
                manager._prefix_streams = {}
            manager._prefix_streams[int(req.bootstrap_room)] = plan
            controller.sources.append(source)
            if isinstance(controller.plan, list) and req.extend_end == len(
                req.origin_input_ids
            ):
                controller.plan.append(source)
        except Exception as exc:
            error = repr(exc)
    if not controller.agree([int(error is None)])[0]:
        raise TransportFailure(
            f"TP hybrid PD source allocation failed: {error or 'peer failed'}"
        )


def _observe_refresh(parent, observations):
    key = id(parent)
    if key not in observations:
        try:
            parent.refresh()
        except Exception as exc:
            observations[key] = exc
        else:
            observations[key] = None
    error = observations[key]
    if error is not None:
        raise error


class Resource:
    def __init__(self, parent, entry, admission_id):
        self.parent, self.entry, self.admission_id = parent, entry, admission_id
        self.closed, self.release = False, None

    @property
    def decode_req(self):
        return self.parent.decode_req

    @property
    def generation(self):
        return self.parent.generation

    def gpu_ready(self):
        return self.parent.gpu_ready()

    def transport_ready(self):
        self.parent.refresh()
        return self.parent.closed or self.parent.coverage.ready(self.entry)

    def observe_transport_ready(self, observations):
        _observe_refresh(self.parent, observations)
        return self.parent.closed or self.parent.coverage.ready(self.entry)

    def ready_to_finish(self):
        return self.gpu_ready() and self.transport_ready()

    def finish(self, *, wait=False):
        if not self.ready_to_finish():
            raise RuntimeError("hybrid entry has no GPU/transport proof")
        if not self.parent.closed:
            self.parent.rpc.local_entries.add(self.entry)
            trace = getattr(self.parent, "pd_trace", None)
            if trace is not None:
                trace.local(self.entry)
        self.closed = True


class Target(NativeTarget):
    def refresh(self):
        super().refresh()
        self.owner.pd_visibility.receive()
        trace = getattr(self, "pd_trace", None)
        if trace is not None:
            trace.visible(self.coverage)

    def observe_transport_ready(self, observations):
        _observe_refresh(self, observations)
        # Hybrid PD explicitly disables native staging; canonical native
        # success is still required for retirement of the whole request.
        return _poll_success(self.decode_req.kv_receiver)

    def finish(self, *, wait=False):
        if self.closed:
            return
        if not self.ready_to_finish():
            raise RuntimeError("hybrid PD retirement unproven")
        self.rpc.close("hybrid-pd-tp-local")
        super().finish(wait=wait)
        value = read_json(self.path / "done.json")
        value.update(
            hybrid_pd=True,
            rid=self.decode_req.req.rid,
            remote_calls=self.rpc.calls,
            pd_window=self.pd_trace.snapshot(),
        )
        write_json(self.path / "done.json", value)


def offer(controller, queue, dr):
    from .pre_layerwise import Controller

    target = Controller.offer(controller, queue, dr)
    if target is not None and getattr(target, "_hybrid_coverage", None) is None:
        schema = controller.schema_signature
        value = read_json(target.path.parent / "offer.json")
        value.update(hybrid_schema=schema, remote_boundary="hybrid-gdn-state")
        write_json(target.path.parent / "offer.json", value)
        target.coverage = target._hybrid_coverage = LayerCoverage(
            len(dr.req.origin_input_ids), len(schema["entries"])
        )
    return target


def early_targets(controller, queue, rids_to_check):
    offered = [controller.offer(queue, dr) for dr in decode_queue_items(queue)]
    ready, healthy = [], []
    for target in offered:
        take = False
        try:
            data = None if target is None else read_json(target.path / "ready.json")
            if target is not None:
                _poll_success(target.decode_req.kv_receiver)
                dr = target.decode_req
                take = bool(
                    (rids_to_check is None or dr.req.rid in rids_to_check)
                    and not dr.req.finished()
                    and data
                    and data.get("generation") == target.generation
                    and data.get("prefix_len") == target.coverage.prefix_len
                    and data.get("hybrid_state_ready") is True
                    and data.get("schema") == controller.schema_signature
                    and prefix_stream.header(target) is not None
                    and int(
                        queue.metadata_buffers.bootstrap_room[dr.metadata_buffer_index][
                            0
                        ].item()
                    )
                    == int(dr.req.bootstrap_room)
                )
            healthy.append(1)
        except TransportFailure:
            healthy.append(0)
        ready.append(int(take))
    votes = controller.agree([*healthy, *ready])
    from .transport_failure import quarantine

    for target, ok in zip(offered, votes[: len(offered)]):
        if target is not None and not ok:
            quarantine(controller, target, "hybrid native PD receive failed")
    admitted, removed = [], set()
    for target, take in zip(offered, votes[len(offered) :]):
        if not take:
            continue
        dr = target.decode_req
        data = read_json(target.path / "ready.json")
        controller.pd_visibility.receive()
        queue._commit_transfer_to_req(dr, keep_receiver=True)
        if dr.kv_receiver is None or dr.req.finished():
            raise RuntimeError("hybrid early native metadata commit failed")
        # Native GDN payload has completed; a fresh working slot must survive
        # the first Decode prepare instead of being zeroed as a new context.
        dr.req.kv.mamba_needs_clear = False
        rpc = qwen_hybrid.Target(
            controller,
            dr.req,
            target.path,
            target.generation,
            data["service"],
            len(dr.req.origin_input_ids),
        )
        rpc.progressive = True
        parent = Target(
            controller, queue, dr, target.path, target.generation, target.coverage
        )
        from .pd_window import WindowTrace

        parent.pd_trace = rpc.pd_trace = WindowTrace(controller.layer_entries)
        parent.rpc = rpc
        dr.req._pre_layerwise_target, dr.req._dsv41_remote = parent, rpc
        controller.remote_targets[rpc.row] = rpc
        protocol = controller.decode_overlap.retirement
        for i in range(len(controller.schema_signature["entries"])):
            resource = Resource(parent, i, protocol.last_admission + 1)
            parent.resources.append(resource)
            protocol.register(resource)
            controller.targets.append(resource)
        parent.admission_id = protocol.last_admission + 1
        protocol.register(parent)
        controller.targets.append(parent)
        controller.early_admissions += 1
        removed.add(id(dr))
        admitted.append(dr.req)
        logger.info(
            "[handoff-target] hybrid PD rid=%s prefix=%s", dr.req.rid, rpc.prefix_len
        )
    remove_decode_items(queue, removed)
    return admitted, [
        x for x in offered if x is not None and id(x.decode_req) not in removed
    ]
