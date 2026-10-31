"""Native-PD sideband barrier and event-driven per-entry sparse ownership.

No scheduler/model loop is copied. Indexer plus native SWA/carry/metadata precede
admission; global bytes use a native receive ring and immutable-prefix scatter.
"""

import functools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import dsv41_remote, prefill_overlap
from .pre_layerwise import Target as NativeTarget
from .pre_layerwise import _poll_success, read_json, write_json


def groups(schema, group_size):
    # Index/scales are a barrier, global KV entries can switch independently.
    # Distinct physical ids (not consumer layer numbers) identify retirement.
    index = tuple(i for i, e in enumerate(schema["entries"]) if e[0] == "index")
    kv = tuple(i for i, e in enumerate(schema["entries"]) if e[0] == "kv")
    return tuple((i, (i,)) for i in (*index, *kv))


def track_transfer(original, manager, room, indices, index_slice, last, *args, **kw):
    source = getattr(manager, "_dsv41_progressive_sources", {}).get(int(room))
    tracked = functools.partial(prefill_overlap.track_transfer, original)
    if source is None or not last:
        return tracked(manager, room, indices, index_slice, last, *args, **kw)
    if args or source.final_payload is not None or len(indices):
        raise RuntimeError("invalid/duplicate native DSV4.1 final envelope")
    source.final_payload = (indices, index_slice, dict(kw))
    source.enqueue_index()


def send_states(source):
    """Runs inside a counted native worker callback, including abort ACK fence."""
    sender = source.req.disagg_kv_sender
    manager = sender.kv_mgr
    _, _, payload = source.final_payload
    with ThreadPoolExecutor(max_workers=2) as executor:
        for peer, registration in source.peers:
            if (
                manager.maybe_send_extra(
                    peer, payload.get("state_indices") or [], executor, registration
                )
                != 0
                or manager.send_aux(
                    peer, payload["aux_index"], registration.dst_aux_ptrs
                )
                != 0
            ):
                raise RuntimeError("DSV4.1 early native state/aux transfer failed")
    # State/aux TE completion is synchronous. Indexer GPU scatter is separately
    # required on D; this publication alone never enables a read or a free.
    from sglang.srt.disaggregation.base.conn import KVPoll

    if sender.poll() == KVPoll.Failed:
        raise RuntimeError("aborted during native state barrier")
    source.barrier_ready = True
    write_json(
        source.path / "sparse-ready.json",
        dict(
            generation=source.generation,
            prefix_len=source.prefix_len,
            barrier="indexer+swa+carry+metadata",
            at=time.monotonic(),
        ),
    )
    source.enqueue_global()
    indices, index_slice, kw = source.final_payload
    kw["state_indices"] = []  # D owns mutable SWA/carry now; never overwrite it.
    manager.add_transfer_request(
        sender.bootstrap_room, indices, index_slice, True, **kw, _dsv41_canonical=True
    )


def canonical_track(original, manager, room, indices, index_slice, last, *args, **kw):
    # Private plugin-only sentinel is consumed here, never passed into core.
    if kw.pop("_dsv41_canonical", False):
        return prefill_overlap.track_transfer(
            original, manager, room, indices, index_slice, last, *args, **kw
        )
    return track_transfer(
        original, manager, room, indices, index_slice, last, *args, **kw
    )


class SourceMixin:
    def enqueue_index(self):
        index_groups = [
            (n, e) for n, e in self.groups if self.schema["entries"][e[0]][0] == "index"
        ]
        if not index_groups:
            self.sideband_started = True
            # A counted empty native task covers the sideband even if the model
            # has no index entries. No global page is read/written by this task.
            sender = self.req.disagg_kv_sender
            sender.kv_mgr.add_transfer_request(
                sender.bootstrap_room,
                self.indices[:0],
                slice(self.index_slice.start, self.index_slice.start),
                False,
                wait_event=self.final_event,
                on_complete=self._sideband,
            )
        for ordinal, (boundary, entries) in enumerate(self.groups):
            if (boundary, entries) in index_groups:
                self._enqueue(ordinal, boundary, entries, self.final_event)

    def _sideband(self):
        try:
            send_states(self)
        except Exception as exc:
            self.error = repr(exc)
            manager = self.req.disagg_kv_sender.kv_mgr
            # Match native failed-TE semantics: a timed-out DMA may still write.
            # Poison abort ACKs and retain pages; do not kill a shared worker.
            manager._poison_unsettled_send(self.req.disagg_kv_sender.bootstrap_room)
            manager.conclude_failure(
                bootstrap_room=self.req.disagg_kv_sender.bootstrap_room,
                failure_reason=self.error,
            )
            write_json(
                self.path / "error.json",
                dict(generation=self.generation, error=self.error),
            )

    def enqueue_global(self):
        for ordinal, (boundary, entries) in enumerate(self.groups):
            if self.schema["entries"][entries[0]][0] == "kv":
                self._enqueue(ordinal, boundary, entries, self.final_event)

    def completed_entry(self, boundary, entries):
        # Native staging can defer/reorder groups. The *set*, never ordinal,
        # determines when all index/scales TE writes have completed.
        with self.completion_lock:
            self.complete_group(boundary, entries)
            index = {i for i, e in enumerate(self.schema["entries"]) if e[0] == "index"}
            if index <= self.completed and not self.sideband_started:
                self.sideband_started = True
                self._sideband()


def completed(queue, dr):
    handler = queue.staging_handler
    handler.advance_scatter(dr)
    if handler.is_failed(dr):
        raise RuntimeError("DSV4.1 receive scatter failed; retaining KV")
    return {i for entries, _, _ in dr._staging_completed_layers for i in entries}


def barrier_ready(offer):
    dr, queue = offer.decode_req, offer.queue
    data = read_json(offer.path / "sparse-ready.json")
    service = read_json(offer.path / "remote-service.json")
    if not data or data.get("generation") != offer.generation or not service:
        return None
    if (
        service.get("generation") != offer.generation
        or data.get("prefix_len") != len(dr.req.origin_input_ids)
        or service.get("prefix_len") != len(dr.req.origin_input_ids)
    ):
        raise RuntimeError("DSV4.1 sparse barrier identity changed")
    schema = offer.owner._schema(dr.kv_receiver.kv_mgr)
    index = {i for i, entry in enumerate(schema["entries"]) if entry[0] == "index"}
    if not index <= completed(queue, dr):
        return None
    if int(
        queue.metadata_buffers.bootstrap_room[dr.metadata_buffer_index][0].item()
    ) != int(dr.req.bootstrap_room):
        return None
    from sglang.srt.disaggregation.base.conn import KVPoll

    if dr.kv_receiver.poll() == KVPoll.Failed or dr.req.finished():
        return None
    return service


@dataclass(eq=False)
class Resource:
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
        return self.transport_ready()

    def transport_ready(self):
        return self.parent.closed or self.entry in completed(
            self.parent.queue, self.decode_req
        )

    def ready_to_finish(self):
        return self.transport_ready()

    def finish(self, *, wait=False):
        if not self.ready_to_finish():
            raise RuntimeError("unproven resource switch")
        self.parent.rpc.local_entries.add(self.entry)
        self.closed = True
        write_json(
            self.parent.path / f"local-{self.entry}.json",
            dict(
                generation=self.generation,
                entry=self.entry,
                tp_agreed=True,
                completion="native-scatter-gpu",
                at=time.monotonic(),
            ),
        )


@dataclass(eq=False)
class ProgressiveTarget(NativeTarget):
    rpc: object = None
    resources: list = field(default_factory=list)

    def refresh(self):
        if self.closed:
            return
        self.rpc.check()
        completed(self.queue, self.decode_req)
        failure = read_json(self.path / "error.json")
        if failure and failure.get("generation") == self.generation:
            raise RuntimeError(f"DSV4.1 peer failed; retaining KV: {failure}")

    def transport_ready(self):
        self.refresh()
        return _poll_success(
            self.decode_req.kv_receiver
        ) and self.queue.staging_handler.is_done(self.decode_req)

    def ready_to_finish(self):
        # Full canonical completion may supersede outstanding resource commands
        # at a TP-symmetric retraction boundary. All scatters still need proving.
        return self.gpu_ready() and self.transport_ready()

    def finish(self, *, wait=False):
        if not self.ready_to_finish():
            raise RuntimeError("unproven final DSV4.1 GPU/PD retirement")
        self.rpc.close("progressive-tp-local")
        super().finish(wait=wait)
        data = read_json(self.path / "done.json")
        data.update(
            early_admission=True, progressive_sparse=True, remote_calls=self.rpc.calls
        )
        write_json(self.path / "done.json", data)


def early_targets(controller, queue, rids_to_check):
    offered = [controller.offer(queue, dr) for dr in queue.queue]
    candidates = []
    for offer in offered:
        ready = None
        if offer is not None and (
            rids_to_check is None or offer.decode_req.req.rid in rids_to_check
        ):
            ready = barrier_ready(offer)
        candidates.append(ready)
    accepted = controller.agree([int(c is not None) for c in candidates])
    admitted, removed = [], set()
    retirement = controller.decode_overlap.retirement
    for offer, data, take in zip(offered, candidates, accepted):
        if not take:
            continue
        dr = offer.decode_req
        queue._commit_transfer_to_req(dr, keep_receiver=True)
        if dr.kv_receiver is None or dr.req.finished():
            raise RuntimeError("DSV4.1 early native metadata commit failed")
        rpc = dsv41_remote.Target(
            controller,
            dr.req,
            offer.path,
            offer.generation,
            data["route"],
            int(data["prefix_len"]),
        )
        rpc.progressive = True
        target = ProgressiveTarget(
            controller,
            queue,
            dr,
            offer.path,
            offer.generation,
            offer.coverage,
            native_staging=True,
            rpc=rpc,
        )
        dr.req._pre_layerwise_target = target
        dr.req._dsv41_remote = rpc
        controller.remote_targets[rpc.row] = rpc
        schema = controller._schema(dr.kv_receiver.kv_mgr)
        # One ordered registration sequence on every TP rank. Parent comes last
        # so batch GPU proofs attach to the lifetime, not to a byte resource.
        for i, entry in enumerate(schema["entries"]):
            if entry[0] != "kv":
                continue
            resource = Resource(target, i, retirement.last_admission + 1)
            target.resources.append(resource)
            retirement.register(resource)
            controller.targets.append(resource)
        target.admission_id = retirement.last_admission + 1
        retirement.register(target)
        controller.targets.append(target)
        removed.add(id(dr))
        admitted.append(dr.req)
        controller.early_admissions += 1
        write_json(
            target.path / "admitted.json",
            dict(
                generation=target.generation,
                boundary="indexer+swa+carry+metadata",
                global_entries_ready=len(
                    completed(queue, dr).intersection(
                        i
                        for i, entry in enumerate(schema["entries"])
                        if entry[0] == "kv"
                    )
                ),
                at=time.monotonic(),
            ),
        )
    queue.queue = [dr for dr in queue.queue if id(dr) not in removed]
    return admitted, offered


def send_aux(original, manager, req, prefill_aux_index, dst_aux_ptrs):
    peer, aux_index = req, prefill_aux_index
    # The canonical final marker must not race a second metadata DMA against
    # D's early native commit. The successful sideband already sent these bytes.
    for source in tuple(getattr(manager, "_dsv41_progressive_sources", {}).values()):
        if (
            source.barrier_ready
            and source.final_payload is not None
            and source.final_payload[2]["aux_index"] == aux_index
            and any(peer is p for p, _ in source.peers)
        ):
            return 0
    return original(manager, peer, aux_index, dst_aux_ptrs)


def poll_responses(controller):
    # A response-only snapshot never changes KV ownership or execution policy.
    # All shard barriers are immutable per-attempt files; no TP collective is
    # needed to return P's HTTP leg. Native streamer owns the output-rank gate.
    import copy

    from sglang.srt.managers.schedule_batch import FINISH_LENGTH

    for source in controller.sources:
        if source.responded or not getattr(source, "barrier_ready", False):
            continue
        ready = True
        room = controller.config.directory / str(int(source.req.bootstrap_room))
        for rank in range(controller.tp_size):
            root = room / str(rank)
            offer = read_json(root / "offer.json") or {}
            generation = offer.get("generation")
            if rank == controller.rank and generation != source.generation:
                raise RuntimeError("P response attempt changed; retaining source")
            data = (
                read_json(root / generation / "sparse-ready.json")
                if generation
                else None
            )
            if (
                not data
                or data.get("generation") != generation
                or data.get("prefix_len") != source.prefix_len
                or offer.get("mode") != "remote"
            ):
                ready = False
                break
        if ready:
            snapshot = copy.copy(source.req)
            snapshot.finished_reason = FINISH_LENGTH(length=0)
            snapshot._pre_layerwise_response_snapshot = True
            controller.scheduler.output_streamer.stream_output(
                [snapshot], snapshot.return_logprob, None
            )
            source.responded = True
