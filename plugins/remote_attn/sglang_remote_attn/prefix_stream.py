"""Request-owned prefix/middle/final PD ranges over native Mooncake tasks.

Only a device cache hit opts in, before its first native send. A bounded number of
payloads per manager is in flight; index entries and metadata precede queued
global work. Worker callbacks never enter a TP collective or free cache pages.
"""

import functools
import os
import threading
from bisect import bisect_left
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

from .pre_layerwise import LayerCoverage, Source, buffer_groups, read_json, write_json

VERSION = 1
MAX_BYTES = 4 * 1024 * 1024


def enabled():
    value = os.environ.get("HANDOFF_PREFIX_REMOTE_PROGRESSIVE", "1")
    if value not in ("0", "1"):
        raise ValueError("HANDOFF_PREFIX_REMOTE_PROGRESSIVE must be 0 or 1")
    return value == "1"


class Pump:
    def __init__(self, limit=None):
        if limit is None:
            limit = int(os.environ.get("HANDOFF_PREFIX_INFLIGHT", "2"))
        if limit not in (1, 2, 3, 4):
            raise ValueError("HANDOFF_PREFIX_INFLIGHT must be 1..4")
        self.limit = limit
        self.lock = threading.RLock()
        self.urgent, self.bulk = deque(), deque()
        self.active = {}
        self.controls = 0
        self.failed = False

    def submit(self, task, urgent):
        with self.lock:
            (self.urgent if urgent else self.bulk).append(task)
        self.drive()

    def drive(self):
        while True:
            with self.lock:
                if self.failed or len(self.active) >= self.limit:
                    return
                queue = self.urgent if self.urgent else self.bulk
                if not queue or (queue is self.bulk and self.controls):
                    return
                task = queue.popleft()
                self.active[id(task)] = task
            try:
                task()
            except Exception as exc:
                # Submission can already own DMA. Freeze this pump; preserve
                # every active ticket until its own completion is known.
                with self.lock:
                    self.failed = True
                task.plan.fail(exc)
                raise

    def done(self, task):
        with self.lock:
            if self.active.get(id(task)) is not task:
                raise RuntimeError("prefix task completion identity changed")
            del self.active[id(task)]
        self.drive()

    def quarantine(self):
        with self.lock:
            self.failed = True


@dataclass
class Task:
    plan: object
    send: object
    piece: dict
    indices: object
    kwargs: dict
    finish: object

    def __call__(self):
        p, piece = self.plan, self.piece
        p.check()
        kw = dict(self.kwargs)
        kw.update(kv_buffer_indices=tuple(piece["buffers"]), on_complete=self.complete)
        if p.staged:
            p.manager.request_layer_staging(
                p.room,
                slice(piece["lo"], piece["hi"]),
                ((tuple(piece["resources"]), None),),
                chunk_id_base=piece["id"],
                stream_generation=p.generation,
            )
            kw.update(
                staging_chunk_idx=piece["id"],
                staging_layer_ids=tuple(piece["resources"]),
            )
        self.send(
            p.manager,
            p.room,
            self.indices,
            slice(piece["lo"], piece["hi"]),
            False,
            **kw,
        )

    def complete(self):
        p, piece = self.plan, self.piece
        try:
            p.check()
            with p.lock:
                for resource in piece["resources"]:
                    p.sent.complete(
                        resource,
                        p.base + piece["lo"] * p.page,
                        min(p.length, p.base + piece["hi"] * p.page),
                    )
                p.completed.add(piece["id"])
                if not p.staged:
                    p.publish()
                self.finish()
                p.flush_callbacks()
                p.flush_final()
        except Exception as exc:
            p.pump.quarantine()
            p.fail(exc)
            raise
        finally:
            # Failure leaves the active task quarantined, including storage refs.
            if p.error is None:
                p.pump.done(self)


class Plan:
    def __init__(self, controller, req, offer, path, cache_lease):
        self.controller, self.req, self.manager = (
            controller,
            req,
            req.disagg_kv_sender.kv_mgr,
        )
        self.room, self.path, self.generation = (
            int(req.bootstrap_room),
            path,
            offer["generation"],
        )
        self.handle, self.row, self.sender = (
            req.cache_request_handle,
            req.kv.req_pool_idx,
            req.disagg_kv_sender,
        )
        self.cache_lease = cache_lease
        self.peers = tuple(
            (p, self.manager.decode_kv_args_table[p.mooncake_session_id])
            for p in self.manager.transfer_infos[self.room].values()
        )
        self.destinations = tuple(tuple(reg.dst_kv_ptrs) for _, reg in self.peers)
        self.page = self.manager.kv_args.page_size
        self.base, self.length = (
            int(getattr(req, "disagg_decode_prefix_len", 0)),
            len(req.origin_input_ids),
        )
        self.dsv = getattr(controller, "dsv41_layerwise", False)
        self.staged = bool(offer.get("native_staging"))
        self.byte_budget = MAX_BYTES
        if self.staged:
            self.byte_budget = min(
                MAX_BYTES, min(reg.staging_total_size for _, reg in self.peers)
            )
        if self.dsv:
            from .dsv41_receive import install

            install(controller, self.manager, "prefill")
            schema = controller._schema(self.manager)
            self.groups = [
                ((i,), (i,), e[0] == "index") for i, e in enumerate(schema["entries"])
            ]
            resources = len(schema["entries"])
        else:
            size = controller.config.select_group_size(
                controller.layers, req.prefix_len, req.prefix_len
            )
            ids = (
                list(self.manager.kv_args.kv_layer_ids)
                or list(range(controller.layers)) * 2
            )
            self.groups = [
                (layers, buffers, False)
                for layers, buffers in buffer_groups(ids, controller.layers, size)
            ]
            resources = controller.layers
        self.sent = LayerCoverage(self.length, resources)
        if self.base:
            for r in range(resources):
                self.sent.complete(r, 0, self.base)
        self.lock = threading.RLock()
        self.pieces, self.completed, self.callbacks = [], set(), []
        self.booked = {}
        self.final = self.source = self.error = None
        self.sealed = self.bound = False
        self.aborted = False
        if not hasattr(self.manager, "_prefix_stream_pump"):
            self.manager._prefix_stream_pump = Pump()
        self.pump = self.manager._prefix_stream_pump
        self.publish()

    def publish(self):
        write_json(
            self.path / "range-stream.json",
            dict(
                version=VERSION,
                generation=self.generation,
                length=self.length,
                base=self.base,
                page=self.page,
                resources=self.sent.layers,
                staged=self.staged,
                completed=sorted(self.completed) if not self.staged else [],
                count=len(self.pieces),
                sealed=self.sealed,
            ),
        )

    def check(self):
        if (
            self.error
            or self.req.cache_request_handle != self.handle
            or self.req.kv.req_pool_idx != self.row
            or self.req.disagg_kv_sender is not self.sender
        ):
            raise RuntimeError(
                f"prefix stream ownership unproven; retaining KV: {self.error}"
            )
        self.cache_lease.check()
        if any(
            self.manager.decode_kv_args_table.get(p.mooncake_session_id) is not reg
            or tuple(reg.dst_kv_ptrs) != self.destinations[i]
            for i, (p, reg) in enumerate(self.peers)
        ):
            raise RuntimeError("prefix destination registration changed; retaining KV")

    def fail(self, exc):
        if self.error is not None:
            return
        self.error = repr(exc)
        write_json(
            self.path / "error.json", dict(generation=self.generation, error=self.error)
        )

    @contextmanager
    def priority(self):
        with self.pump.lock:
            self.pump.controls += 1
        try:
            yield
        finally:
            with self.pump.lock:
                self.pump.controls -= 1
            self.pump.drive()

    def add(self, send, indices, span, kwargs):
        self.check()
        if self.aborted:
            raise RuntimeError("send after prefix stream cancellation")
        if (
            len(indices) != span.stop - span.start
            or span.start < 0
            or self.base + span.stop * self.page
            > ((self.length + self.page - 1) // self.page) * self.page
        ):
            raise ValueError(
                "prefix stream range differs from native sender reservation"
            )
        selected = kwargs.get("kv_buffer_indices")
        groups = (
            self.groups
            if selected is None
            else [
                (
                    tuple(kwargs["staging_layer_ids"])
                    if kwargs.get("staging_layer_ids")
                    else tuple(i for i in range(self.sent.layers) if i in selected),
                    tuple(selected),
                    self.dsv
                    and all(
                        self.controller.schema_signature["entries"][i][0] == "index"
                        for i in selected
                    ),
                )
            ]
        )
        # Qwen selected buffers are K/V ids, resources are supplied layer ids.
        if selected is not None and not self.dsv:
            groups = [
                (
                    tuple(
                        i
                        for i in range(self.sent.layers)
                        if i in selected or i + self.sent.layers in selected
                    ),
                    tuple(selected),
                    False,
                )
            ]
        callback = kwargs.get("on_complete")
        remaining = [0]
        tasks = []
        with self.lock:
            for resources, buffers, urgent in groups:
                stride = sum(self.manager.kv_args.kv_item_lens[i] for i in buffers)
                if self.staged and stride > self.byte_budget:
                    raise ValueError(
                        "one prefix page group exceeds the native receive ring"
                    )
                bound = max(1, self.byte_budget // stride)
                cap = getattr(self.manager, "max_transfer_batch_indices", 0)
                if cap > 0:
                    bound = min(bound, cap)
                for lo in range(span.start, span.stop, bound):
                    hi = min(span.stop, lo + bound)
                    for resource in resources:
                        booked = self.booked.setdefault(resource, [])
                        offset = bisect_left(booked, (lo, hi))
                        if (offset and booked[offset - 1][1] > lo) or (
                            offset < len(booked) and booked[offset][0] < hi
                        ):
                            raise RuntimeError(
                                "duplicate/overlapping prefix resource range"
                            )
                        booked.insert(offset, (lo, hi))
                    piece = dict(
                        id=len(self.pieces),
                        generation=self.generation,
                        lo=lo,
                        hi=hi,
                        resources=list(resources),
                        buffers=list(buffers),
                    )
                    self.pieces.append(piece)
                    if not self.staged:
                        write_json(self.path / f"range-{piece['id']}.json", piece)
                    kw = {
                        k: v
                        for k, v in kwargs.items()
                        if k
                        not in (
                            "on_complete",
                            "kv_buffer_indices",
                            "staging_chunk_idx",
                            "staging_layer_ids",
                        )
                    }
                    kw["num_kv_tokens"] = min(
                        self.length, self.base + hi * self.page
                    ) - (self.base + lo * self.page)
                    remaining[0] += 1

                    def finish(remaining=remaining):
                        remaining[0] -= 1

                    tasks.append(
                        (
                            Task(
                                self,
                                send,
                                piece,
                                indices[lo - span.start : hi - span.start],
                                kw,
                                finish,
                            ),
                            urgent,
                        )
                    )
            if callback is not None:
                # Final entry callbacks publish a barrier only after ALL ranges
                # of that entry; ordinary middle callbacks just drain their IO.
                resources = tuple(r for rs, _, _ in groups for r in rs)
                self.callbacks.append(
                    (remaining, callback, resources if selected is not None else ())
                )
            self.publish()
        for task, urgent in tasks:
            self.pump.submit(task, urgent)

    def flush_callbacks(self):
        pending = []
        for remaining, callback, resources in self.callbacks:
            if remaining[0] == 0 and all(self.sent.ready(r) for r in resources):
                callback()
            else:
                pending.append((remaining, callback, resources))
        self.callbacks = pending

    def flush_final(self):
        if (
            self.final is not None
            and len(self.completed) == len(self.pieces)
            and not self.callbacks
            and all(self.sent.ready(r) for r in range(self.sent.layers))
        ):
            final, self.final = self.final, None
            final()

    def drained(self):
        self.check()
        return len(self.completed) == len(self.pieces)


def stream_for(req):
    return getattr(req, "_remote_attn_prefix_stream", None)


def bind(req, source):
    plan = stream_for(req)
    if plan is None:
        return source
    plan.check()
    if plan.bound or source.generation != plan.generation:
        raise RuntimeError("prefix final plan changed identity or was bound twice")
    previous = plan.source
    created = previous.created_at
    previous.__dict__.update(source.__dict__)
    previous.created_at = created
    previous.prefix_stream = plan
    plan.bound = True
    return previous


def cleanup(req):
    plan = stream_for(req)
    if plan is not None:
        if not plan.drained():
            raise RuntimeError("prefix IO remains live at cleanup")
        plan.manager._prefix_streams.pop(plan.room, None)
        del req._remote_attn_prefix_stream


def before_mutation(scheduler, req):
    """Native P abort/requeue boundary, before it aborts the sender or frees KV."""
    plan = stream_for(req)
    if plan is None or plan.source.service is not None:
        return
    import time

    deadline = time.monotonic() + plan.controller.config.drain_timeout
    ready = False
    try:
        while not (ready := plan.drained()):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
    except Exception:
        ready = False
    # Enter once, in the existing TP-symmetric native mutation order. No local
    # readiness branch changes the number or shape of collectives.
    if not plan.controller.agree([int(ready)])[0]:
        raise RuntimeError("prefix abort IO unproven across TP; retaining pages")
    source = plan.source
    plan.aborted = True
    source.cache_lease.abort = True
    source.release = lambda: None  # Native mutation below still owns the real free.
    source.retire()
    plan.controller.sources.remove(source)


def cached_prefix(original, scheduler, req, allow_send=True):
    controller = getattr(scheduler, "_pre_layerwise", None)
    if (
        not allow_send
        or controller is None
        or not enabled()
        or controller.config.mode != "remote"
        or getattr(controller, "cache_guard", None) is None
        or req.pending_bootstrap
    ):
        return original(scheduler, req, allow_send=allow_send)
    if not getattr(req, "_remote_attn_prefix_stream_seen", False):
        from .pd_mesh import source_root

        req._remote_attn_prefix_stream_seen = True
        candidate = pin = None
        owned = False
        try:
            from sglang.srt.disaggregation.mooncake.conn import MooncakeKVSender

            from .destination_cache import key as request_digest
            from .remote_data_plane import enabled as gpu_payload

            sender = req.disagg_kv_sender
            manager = sender.kv_mgr
            root = source_root(controller, req)
            offer = read_json(root / "offer.json") or {}
            end = req.prefix_len - req.host_hit_length
            page = manager.kv_args.page_size
            overlap = getattr(controller, "prefill_overlap", None)
            active = getattr(overlap, "current", None)
            batch = (
                active.batch
                if active is not None
                else getattr(controller, "prefix_batch", None)
            )
            plain = batch is None or (
                not batch.is_prefill_only
                and not batch.return_hidden_states
                and batch.input_embeds is None
                and batch.replace_embeds is None
                and not batch.forward_mode.is_mixed()
                and not batch.forward_mode.is_split_prefill()
            )
            peers = tuple(
                (p, manager.decode_kv_args_table.get(p.mooncake_session_id))
                for p in manager.transfer_infos.get(req.bootstrap_room, {}).values()
            )
            if (
                type(sender) is MooncakeKVSender
                and plain
                and getattr(req, "multimodal_inputs", None) is None
                and getattr(req, "input_embeds", None) is None
                and bool(offer.get("gpu_transfer", False)) == gpu_payload()
                and (
                    not getattr(controller, "dsv41_layerwise", False)
                    or offer.get("native_staging")
                )
                and (
                    getattr(controller, "prefill_overlap", None) is None
                    or getattr(req, "_dsv41_p_lease", None) is not None
                )
                and offer.get("range_stream") == VERSION
                and offer.get("mode") == "remote"
                and req.start_send_idx == getattr(req, "disagg_decode_prefix_len", 0)
                and end > req.start_send_idx
                and end % page == 0
                and len(peers) == 1
                and peers[0][1] is not None
                and (
                    not offer.get("native_staging")
                    or getattr(peers[0][1], "staging_total_size", 0) > 0
                )
                and not peers[0][0].is_dummy
                and not peers[0][1].requires_dcp_relayout
                and not getattr(req, "session_id", None)
                and offer.get("session") == peers[0][0].mooncake_session_id
                and offer.get("prefix_len") == len(req.origin_input_ids)
                and offer.get("decode_prefix_len", 0) == req.start_send_idx
                and getattr(manager, "kv_shard_size", 1) == 1
            ):
                cert = getattr(req, "_dsv41_p_certificate", None)
                pin = getattr(cert, "cache_lease", None)
                if pin is None:
                    pin = controller.cache_guard.hold(req)
                    owned = True
                candidate = (
                    str(req.rid),
                    str(req.bootstrap_room),
                    end,
                    req.start_send_idx,
                    page,
                    getattr(req.cache_request_handle, "attempt_id", None),
                    request_digest(req),
                )
        except Exception:
            candidate = None
        reports = [candidate]
        if controller.tp_size > 1:
            import torch.distributed as dist

            reports = [None] * controller.tp_size
            dist.all_gather_object(
                reports, candidate, group=scheduler.attn_tp_cpu_group
            )
        if candidate is not None and all(r == candidate for r in reports):
            plan = Plan(controller, req, offer, root / offer["generation"], pin)
            req._remote_attn_prefix_stream = plan
            if not hasattr(manager, "_prefix_streams"):
                manager._prefix_streams = {}
            manager._prefix_streams[int(req.bootstrap_room)] = plan
            import numpy as np

            cls = Source
            if plan.dsv:
                from .dsv41_layerwise import DSV41Source

                cls = DSV41Source
            source = cls(
                controller,
                req,
                plan.path,
                plan.generation,
                np.empty(0, dtype=np.int32),
                slice(0, 0),
                peers,
                plan.length,
                cache_lease=pin,
                decode_prefix_len=plan.base,
                native_staging=plan.staged,
            )
            source.prefix_stream = plan
            plan.source = source
            req._pre_layerwise_source = source
            controller.sources.append(source)
            lease = getattr(req, "_dsv41_p_lease", None)
            if lease is not None:
                lease.source = source
        elif owned and pin is not None:
            pin.retire(lambda: None)
    return original(scheduler, req, allow_send=allow_send)


def track(original, manager, room, indices, span, last, *args, **kw):
    from .dsv41_progressive import canonical_track

    send = functools.partial(canonical_track, original)
    plan = getattr(manager, "_prefix_streams", {}).get(int(room))
    if plan is None:
        return send(manager, room, indices, span, last, *args, **kw)
    if args:
        raise ValueError("prefix ranges require native keyword metadata")
    if last and not len(indices):
        if plan.dsv and not kw.get("_dsv41_canonical", False):
            return send(manager, room, indices, span, last, **kw)
        with plan.lock:
            if plan.sealed:
                raise RuntimeError("duplicate prefix canonical marker")
            plan.sealed = True
            plan.final = functools.partial(
                send, manager, room, indices, span, last, **kw
            )
            plan.publish()
            plan.flush_final()
        return
    if last:
        raise RuntimeError("prefix final forward was not bound before native send")
    try:
        return plan.add(send, indices, span, kw)
    except Exception as exc:
        plan.fail(exc)
        raise


def header(target):
    value = read_json(target.path / "range-stream.json")
    if value is None:
        value = getattr(target.decode_req, "_prefix_range_header", None)
    if value is not None and (
        value.get("version") != VERSION
        or value.get("generation") != target.generation
        or value.get("length") != len(target.decode_req.req.origin_input_ids)
        or value.get("base")
        != getattr(target.decode_req.req.kv, "cache_protected_len", 0)
    ):
        raise RuntimeError("prefix range stream identity changed")
    if value is not None:
        identity = tuple(
            value[k]
            for k in (
                "version",
                "generation",
                "length",
                "base",
                "page",
                "resources",
                "staged",
            )
        )
        if getattr(target, "_range_identity", identity) != identity:
            raise RuntimeError("prefix range geometry changed")
        target._range_identity = identity
    return value


def descriptor(target, index, resources, start, pages):
    registry = getattr(target.decode_req, "_prefix_range_registry", {})
    if index in registry:
        if registry[index] != (tuple(resources), start, pages):
            raise RuntimeError("registered prefix range identity changed")
        return True
    h = header(target)
    if h is None:
        return False
    piece = read_json(target.path / f"range-{index}.json")
    if (
        piece is None
        or piece.get("generation") != target.generation
        or piece.get("id") != index
        or piece.get("resources") != list(resources)
        or piece.get("lo") != start
        or piece.get("hi") != start + pages
        or not 0 <= index < h["count"]
    ):
        raise RuntimeError("unregistered prefix range arrival")
    return True


def register_range(manager, msg):
    """The native control socket registers identity ahead of payload.

    Generation/room bind it without waiting for a cross-host journal mirror.
    """
    if len(msg) != 9:
        return False
    import struct

    dr = manager._staging_handler._room_to_decode_req.get(int(msg[1]))
    if dr is None:
        return False
    target = getattr(dr, "_pre_layerwise_offer", None)
    if target is None or msg[8].decode("ascii") != target.generation:
        raise RuntimeError("stale prefix range attempt")
    resources = tuple(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6]))
    index, start, pages = int(msg[2]), int(msg[7]), int(msg[3])
    page = manager.kv_args.page_size
    base = int(dr.req.kv.cache_protected_len)
    length = len(dr.req.origin_input_ids)
    count = (
        len(manager._dsv41_layerwise_schema["entries"])
        if getattr(manager, "_dsv41_layerwise_schema", None)
        else len(manager.kv_args.kv_item_lens) // 2
    )
    if (
        index < 0
        or not resources
        or tuple(sorted(set(resources))) != resources
        or min(resources) < 0
        or max(resources) >= count
        or start < 0
        or pages <= 0
        or base + (start + pages - 1) * page >= length
    ):
        raise ValueError("invalid prefix range envelope")
    registry = getattr(dr, "_prefix_range_registry", None)
    if registry is None:
        registry = dr._prefix_range_registry = {}
    item = (resources, start, pages)
    if index in registry and registry[index] != item:
        raise RuntimeError("prefix chunk id was reassigned")
    for other, (used, lo, n) in registry.items():
        if (
            other != index
            and set(used) & set(resources)
            and max(lo, start) < min(lo + n, start + pages)
        ):
            raise RuntimeError("overlapping prefix range arrival")
    registry[index] = item
    if not hasattr(dr, "_prefix_range_signatures"):
        dr._prefix_range_signatures = set(registry.values())
    else:
        dr._prefix_range_signatures.add(item)
    dr._prefix_range_header = dict(
        version=VERSION,
        generation=target.generation,
        length=length,
        base=base,
        page=page,
        resources=count,
        staged=True,
        count=max(registry) + 1,
    )
    return True


def allocate_mha_range(manager, msg):
    """Reuse native MHA allocation below its legacy final-tail-only guard."""
    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
    from sglang.srt.disaggregation.common.staging_handler import handle_staging_req

    handler = manager._staging_handler
    room, index = int(msg[1]), int(msg[2])
    with handler._scatter_lock:
        dr = handler._room_to_decode_req.get(room)
        if dr is None:
            return
        receiver = dr.kv_receiver
        if manager.pp_size != 1 or receiver.prefill_info.pp_size != 1:
            raise ValueError("prefix layer staging requires PP1")
        infos = receiver.chunk_staging_infos
        if index in receiver.chunk_staging_layers and (
            index >= len(infos)
            or infos[index][0] < 0
            and infos[index][1] != StagingAllocator.ALLOC_OVERSIZED
        ):
            return  # Never reallocate a consumed id after its GPU event freed it.
        handle_staging_req(
            msg,
            manager._staging_ctx.allocator,
            manager.kv_args,
            manager.attn_tp_size,
            receiver.prefill_info.attn_tp_size,
            getattr(manager, "kv_buffer_tensors", None),
            manager._staging_ctx.room_receivers,
            manager._staging_ctx.room_bootstrap,
        )
        handler.register_wm_subscriber(receiver, msg[4].decode("ascii"))


def update(target):
    h = header(target)
    if h is None:
        return None
    if not hasattr(target, "_range_coverage"):
        target._range_coverage = LayerCoverage(h["length"], h["resources"])
        if h["base"]:
            for r in range(h["resources"]):
                target._range_coverage.complete(r, 0, h["base"])
        target._range_events = {}
    coverage = target._range_coverage
    ranges = []
    if h["staged"]:
        handler = target.queue.staging_handler
        handler.advance_scatter(target.decode_req)
        if handler.is_failed(target.decode_req):
            raise RuntimeError("prefix scatter failed; retaining KV")
        done = target.decode_req._staging_completed_layers
        ranges = list(done[getattr(target, "_range_scatter_seen", 0) :])
        if not ranges:
            return coverage
        registry = getattr(target.decode_req, "_prefix_range_registry", {})
        valid = getattr(target.decode_req, "_prefix_range_signatures", None)
        if valid is None and registry:
            valid = set(registry.values())
        if valid is None:
            pieces = [
                read_json(target.path / f"range-{i}.json") for i in range(h["count"])
            ]
            valid = {
                (tuple(p["resources"]), p["lo"], p["hi"] - p["lo"])
                for p in pieces
                if p is not None and p["generation"] == target.generation
            }
        for resources, lo, pages in ranges:
            if (tuple(resources), lo, pages) not in valid:
                raise RuntimeError("GPU completion has no registered prefix range")
        target._range_scatter_seen = len(done)
    else:
        if not hasattr(target, "_range_consumed"):
            target._range_consumed = set()
        for index in h.get("completed", []):
            if index in target._range_consumed:
                continue
            piece = read_json(target.path / f"range-{index}.json")
            if piece is None:
                continue
            descriptor(
                target,
                index,
                piece["resources"],
                piece["lo"],
                piece["hi"] - piece["lo"],
            )
            if index not in target._range_events:
                from .remote_data_plane import Visibility
                from .remote_data_plane import event as completion_event

                if not hasattr(target.owner, "_prefix_visibility"):
                    device = (
                        target.queue.scheduler.req_to_token_pool.req_to_token.device
                    )
                    target.owner._prefix_visibility = Visibility(device)
                target.owner._prefix_visibility.receive()
                event = completion_event(
                    target.queue.scheduler.req_to_token_pool.req_to_token.device
                )
                target._range_events[index] = event
            if target._range_events[index].query():
                target._range_consumed.add(index)
                ranges.append(
                    (piece["resources"], piece["lo"], piece["hi"] - piece["lo"])
                )
    for resources, lo, pages in ranges:
        for r in resources:
            coverage.complete(
                r,
                h["base"] + lo * h["page"],
                min(h["length"], h["base"] + (lo + pages) * h["page"]),
            )
    return coverage


@dataclass(eq=False)
class Layer:
    parent: object
    entry: int
    admission_id: int
    closed: bool = False
    release: object = None

    @property
    def decode_req(self):
        return self.parent.decode_req

    @property
    def generation(self):
        return self.parent.generation

    def gpu_ready(self):
        return self.parent.gpu_ready()

    def transport_ready(self):
        from .pre_layerwise import _poll_success

        if not self.parent.closed:
            _poll_success(self.decode_req.kv_receiver)
        return self.parent.closed or update(self.parent).ready(self.entry)

    def ready_to_finish(self):
        return self.gpu_ready() and self.transport_ready()

    def finish(self, *, wait=False):
        if not self.ready_to_finish():
            raise RuntimeError("prefix layer Local lacks GPU/range proof")
        import time

        self.parent.state.layer_ready_at[self.entry] = time.monotonic()
        self.parent.state.events[self.entry].set()
        self.closed = True
