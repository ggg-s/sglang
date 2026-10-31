"""DSV4.1 P overlap: launch-owned plans and ordered source retirement.

The native loop, result queue, future map and sampling remain untouched. A
request lease spans all of its chunk plans; only the scheduler thread retires
it. Worker callbacks update an IO ledger and never enter TP collectives.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

from .decode_overlap import BatchProof
from .pre_layerwise import _poll_success, write_json
from .retirement import Retirement, wrap_receiver

logger = logging.getLogger(__name__)


def handle_of(req):
    return getattr(req, "cache_request_handle", None)


@dataclass(frozen=True)
class Chunk:
    req: object
    sender: object
    handle: object
    row: int
    start: int
    end: int
    final: bool

    def check(self):
        if (
            self.req.disagg_kv_sender is not self.sender
            or handle_of(self.req) != self.handle
            or self.req.kv.req_pool_idx != self.row
        ):
            raise RuntimeError(
                "P overlap request attempt/row changed before retirement"
            )


@dataclass(eq=False)
class LaunchPlan:
    sequence: int
    batch: object
    chunks: tuple
    candidates: list
    accepted: list
    proof: object = field(default_factory=BatchProof)
    jobs: list = field(default_factory=list)
    middle_sent: set = field(default_factory=set)
    processed: bool = False
    forward_done: object = None
    indices_refs: list = field(default_factory=list)
    error: str | None = None


@dataclass(eq=False)
class TransferLedger:
    pending: set = field(default_factory=set)
    lock: object = field(default_factory=threading.Lock)
    error: str | None = None


@dataclass(eq=False)
class SourceLease:
    overlap: object
    chunk: Chunk
    admission_id: int
    generation: str
    plans: list = field(default_factory=list)
    source: object = None
    release: object = None
    finalized: bool = False
    cancelling: bool = False
    closed: bool = False
    error: str | None = None
    ledger: TransferLedger = field(default_factory=TransferLedger)
    cache_lease: object = None

    @property
    def decode_req(self):
        # The ordered protocol consumes logical request identities, not D KV.
        return SimpleNamespace(req=self.chunk.req)

    def check(self):
        self.chunk.check()
        if self.cache_lease is not None:
            self.cache_lease.check()
        if self.error or self.ledger.error:
            raise RuntimeError(
                f"P overlap source failed; retaining KV: {self.error or self.ledger.error}"
            )
        for plan in self.plans:
            if plan.error:
                raise RuntimeError(
                    f"P overlap launch failed; retaining KV: {plan.error}"
                )

    def gpu_ready(self):
        self.check()
        for plan in self.plans:
            # Aborted outputs are discarded by the native path. Delayed
            # sampling reads logits, not source KV; forward_done fences KV.
            # Keep the plan/result alive for native sampling/result handling.
            if not self.cancelling and not plan.proof.ready():
                return False
            if plan.forward_done is None or not plan.forward_done.query():
                return False
        return True

    def transport_ready(self):
        self.check()
        with self.ledger.lock:
            outstanding = bool(self.ledger.pending)
        if self.cancelling:
            # Cancellation is fenced BEFORE native abort/clear. Every issued
            # envelope must have its successful completion callback, not merely
            # KVPoll.Failed or a zero dequeue counter.
            return not outstanding and self.gpu_ready()
        return (
            self.finalized
            and not outstanding
            and (self.cache_lease is None or self.release is not None)
            and _poll_success(self.chunk.sender)
            and self.gpu_ready()
        )

    def ready_to_finish(self):
        return self.gpu_ready() and self.transport_ready()

    def finish(self, *, wait=False):
        if not self.ready_to_finish():
            raise RuntimeError("P source GPU/IO proof is incomplete; retaining KV")
        req = self.chunk.req
        try:
            if self.source is not None:
                self.source.release = (
                    self.release or (lambda: None)
                    if self.cache_lease is not None
                    else self.release
                )
                self.source.retire()
                if self.source in self.overlap.controller.sources:
                    self.overlap.controller.sources.remove(self.source)
                self.source.indices_keepalive = None
            elif self.cache_lease is not None:
                self.cache_lease.retire(self.release or (lambda: None))
            elif self.release is not None:
                self.release()
        except BaseException as exc:
            # An allocator callback may have mutated ownership before failing.
            # Never retry it during another drain or shutdown.
            self.error = repr(exc)
            raise
        if getattr(req, "_dsv41_p_lease", None) is self:
            req._dsv41_p_lease = None
        self.overlap.by_room.pop(int(req.bootstrap_room), None)
        if self.overlap.ledgers.get(int(req.bootstrap_room)) is self.ledger:
            self.overlap.ledgers.pop(int(req.bootstrap_room), None)
        for plan in self.plans:
            plan.indices_refs.clear()
        self.plans.clear()
        self.closed = True
        self.overlap.admission.retired(req)


class SourceOwner:
    def __init__(self, overlap):
        controller = overlap.controller
        self.scheduler, self.config = controller.scheduler, controller.config
        self.rank, self.tp_size = controller.rank, controller.tp_size
        self.targets = overlap.leases
        self.agree = controller.agree


class PrefillOverlap:
    def __init__(self, controller):
        self.controller = controller
        self.scheduler = controller.scheduler
        self.active = contextvars.ContextVar("dsv41_prefill_launch", default=None)
        self.processing = contextvars.ContextVar("dsv41_prefill_result", default=None)
        self.leases = []
        self.by_room = {}
        self.ledgers = {}
        self.plans = {}
        self.sequence = self.admissions = 0
        self.retirement = Retirement(SourceOwner(self))
        from .prefill_admission import Admission

        self.admission = Admission(self)
        logger.info(
            "[dsv41-p-overlap] enabled: native two-forward loop, launch-owned plans"
        )

    @property
    def current(self):
        return self.active.get()

    def run_batch(self, original, scheduler, batch, *args, **kwargs):
        if not batch.forward_mode.is_extend() or batch.forward_mode.is_prebuilt():
            return original(scheduler, batch, *args, **kwargs)
        self.finalize_control()
        candidates, accepted = self.admission.select(batch)
        if not any(accepted) and not any(
            getattr(req, "_dsv41_p_lease", None) is not None for req in batch.reqs
        ):
            return original(scheduler, batch, *args, **kwargs)
        chunks = tuple(
            Chunk(
                req,
                getattr(req, "disagg_kv_sender", None),
                handle_of(req),
                req.kv.req_pool_idx,
                req.start_send_idx,
                req.extend_end,
                req.extend_end == len(req.origin_input_ids),
            )
            for req in batch.reqs
        )
        self.sequence += 1
        plan = LaunchPlan(self.sequence, batch, chunks, candidates, accepted)
        self.plans[plan.sequence] = plan
        for chunk, candidate, take in zip(chunks, candidates, accepted):
            lease = getattr(chunk.req, "_dsv41_p_lease", None)
            if not take and lease is None:
                continue
            if lease is None:
                self.admissions += 1
                lease = SourceLease(
                    self,
                    chunk,
                    self.admissions,
                    candidate[3]["generation"],
                    ledger=self.ledgers.setdefault(
                        int(chunk.req.bootstrap_room), TransferLedger()
                    ),
                )
                cache = getattr(self.controller, "cache_guard", None)
                if cache is not None:
                    lease.cache_lease = chunk.req._dsv41_p_certificate.cache_lease
                chunk.req._dsv41_p_lease = lease
                self.leases.append(lease)
                self.by_room[int(chunk.req.bootstrap_room)] = lease
                self.retirement.register(lease)
            lease.check()
            lease.plans.append(plan)
        token = self.active.set(plan)
        try:
            result = original(scheduler, batch, *args, **kwargs)
            plan.batch = batch.copy()
            plan.proof.result = result
            result._dsv41_p_plan = plan
            return result
        except BaseException as exc:
            plan.error = repr(exc)
            raise
        finally:
            self.active.reset(token)

    def forward(self, original, runner, fb, *args, **kwargs):
        plan = self.current
        if plan is None:
            return original(runner, fb, *args, **kwargs)
        import torch

        # Runs on native forward_stream AFTER its wait on schedule_stream.
        # No device-to-host .to('cpu') or CPU event wait is performed here.
        plan.jobs = self.controller.prepare(plan.batch, plan=plan) or []
        for source in plan.jobs:
            lease = source.req._dsv41_p_lease
            lease.source = source
        try:
            result = original(runner, fb, *args, **kwargs)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            plan.forward_done = event
            for source in plan.jobs:
                source.enqueue(self.controller.layers - 1, event, flush=True)
            for chunk, take in zip(plan.chunks, plan.accepted):
                if take and not chunk.final:
                    chunk.check()
                    # Full middle pages are immutable before the next forward;
                    # native send rounds partial pages down and owns the cursor.
                    chunk.sender._early_send_wait_event = event
                    self.scheduler.send_kv_chunk(
                        chunk.req, last_chunk=False, end_idx=chunk.end
                    )
                    plan.middle_sent.add(id(chunk.req))
            return result
        except BaseException as exc:
            plan.error = repr(exc)
            for source in plan.jobs:
                source.error = repr(exc)
                write_json(
                    source.path / "error.json",
                    {
                        "generation": source.generation,
                        "error": source.error,
                    },
                )
            raise

    def send_chunk(self, original, scheduler, req, *, last_chunk, end_idx):
        plan = self.processing.get()
        chunk = (
            next((c for c in plan.chunks if c.req is req), None)
            if plan is not None
            else None
        )
        if chunk is not None:
            chunk.check()
            if not last_chunk and id(req) in plan.middle_sent:
                return  # this exact middle interval is already queued
            if not last_chunk and req.start_send_idx >= chunk.end:
                return  # a later final plan already owns these pages
            end_idx = chunk.end
        lease = getattr(req, "_dsv41_p_lease", None)
        if lease is not None and lease.cancelling:
            return
        source = getattr(req, "_pre_layerwise_source", None)
        if last_chunk and source is not None:
            result = source.finalize_native(
                original, scheduler, req, last_chunk=True, end_idx=end_idx
            )
        else:
            result = original(scheduler, req, last_chunk=last_chunk, end_idx=end_idx)
        lease = getattr(req, "_dsv41_p_lease", None)
        if last_chunk and lease is not None:
            lease.finalized = True
        return result

    def defer_release(self, req, release):
        lease = getattr(req, "_dsv41_p_lease", None)
        if lease is None:
            return False
        lease.check()
        if lease.release is not None:
            raise RuntimeError("duplicate P native release callback")
        lease.release = release
        return True

    def poll(self):
        # No all_reduce, no mailbox polling on a stable empty engine.
        self.retirement.observe()
        self.admission.observe()
        self.reap_plans()

    def inject(self, requests):
        from .prefill_admission import CONTROL

        retirement = self.retirement
        envelope = self.admission.proposal(
            allow_command=(retirement.inflight is None and retirement.pending is None)
        )
        if (
            (envelope is None or envelope.get("command") is None)
            and self.admission.inflight is None
            and self.admission.pending is None
        ):
            requests = retirement.inject(requests)
        if envelope is not None:
            requests = [*(requests or []), {CONTROL: envelope}]
        return requests

    def finalize_control(self, *, wait=False):
        self.admission.finalize()
        self.retirement.finalize(wait=wait)

    def reap_plans(self):
        for seq, plan in list(self.plans.items()):
            if plan.error or not plan.processed or plan.proof.result is None:
                continue
            if plan.proof.ready():
                # Keep lightweight completion proof; release logits/hidden refs.
                result = plan.proof.result
                if getattr(result, "_dsv41_p_plan", None) is plan:
                    result._dsv41_p_plan = None
                plan.proof = CompletedProof()
                plan.batch = None
                if not any(plan in lease.plans for lease in self.leases):
                    plan.indices_refs.clear()
                    self.plans.pop(seq, None)

    def drain(self, req=None):
        self.finalize_control(wait=True)
        selected = [t for t in self.leases if req is None or t.chunk.req is req]
        for lease in selected:
            lease.cancelling = True
        if req is None:
            self.retirement.drain()
        elif selected:
            self.retirement.drain(selected[0])
        if req is not None:
            self.admission.retired(req)
        self.reap_plans()

    def close(self):
        try:
            self.drain()
        finally:
            self.retirement.close()
            self.admission.close()


class CompletedProof:
    result = True

    @staticmethod
    def ready():
        return True


def install(controller):
    if not controller.scheduler.enable_overlap:
        return None
    overlap = PrefillOverlap(controller)
    controller.scheduler.request_receiver = wrap_receiver(
        controller.scheduler.request_receiver, overlap
    )
    manager = controller.scheduler.disagg_prefill_bootstrap_queue.kv_manager
    overlap.admission.guard(manager)
    manager._dsv41_p_overlap = overlap
    controller.scheduler.tp_worker.model_runner.model.model._dsv41_p_overlap = overlap
    return overlap


def track_transfer(
    original,
    manager,
    bootstrap_room,
    kv_indices,
    index_slice,
    is_last_chunk,
    *args,
    **kwargs,
):
    overlap = getattr(manager, "_dsv41_p_overlap", None)
    if overlap is None:
        return original(
            manager,
            bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk,
            *args,
            **kwargs,
        )
    lease = overlap.by_room.get(int(bootstrap_room))
    if lease is not None:
        lease.check()
    ledger = (
        lease.ledger
        if lease is not None
        else overlap.ledgers.setdefault(int(bootstrap_room), TransferLedger())
    )
    marker = object()
    with ledger.lock:
        ledger.pending.add(marker)
    callback = kwargs.get("on_complete")

    def complete():
        try:
            if callback is not None:
                callback()
        except BaseException as exc:
            ledger.error = repr(exc)
            raise
        finally:
            with ledger.lock:
                ledger.pending.discard(marker)
                if (
                    not ledger.pending
                    and int(bootstrap_room) not in overlap.by_room
                    and overlap.ledgers.get(int(bootstrap_room)) is ledger
                ):
                    overlap.ledgers.pop(int(bootstrap_room), None)

    kwargs["on_complete"] = complete
    try:
        return original(
            manager,
            bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk,
            *args,
            **kwargs,
        )
    except BaseException as exc:
        ledger.error = repr(exc)
        raise


def page_indices(raw, page):
    import torch

    device_indices = raw[::page] // page
    host = torch.empty(
        device_indices.shape, dtype=device_indices.dtype, device="cpu", pin_memory=True
    )
    host.copy_(device_indices, non_blocking=True)
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    return host.numpy(), event, (device_indices, host)


def after_result(original, scheduler, batch, result):
    overlap = getattr(
        getattr(scheduler, "_pre_layerwise", None), "prefill_overlap", None
    )
    if overlap is None:
        return original(scheduler, batch, result)
    plan = getattr(result, "_dsv41_p_plan", None)
    if plan is None:
        return original(scheduler, batch, result)
    token = overlap.processing.set(plan)
    try:
        value = original(scheduler, batch, result)
        plan.processed = True
        return value
    except BaseException as exc:
        plan.error = repr(exc)
        raise
    finally:
        overlap.processing.reset(token)


def after_sample(original, scheduler, *args, **kwargs):
    overlap = getattr(
        getattr(scheduler, "_pre_layerwise", None), "prefill_overlap", None
    )
    if overlap is None:
        from .decode_overlap import after_result as decode_after_result

        return decode_after_result(original, scheduler, *args, **kwargs)
    value = original(scheduler, *args, **kwargs)
    overlap.poll()
    overlap.finalize_control()
    return value


def before_mutation(original, scheduler, req, *args, **kwargs):
    lease = getattr(req, "_dsv41_p_lease", None)
    if lease is not None:
        lease.overlap.drain(req)
    else:
        cert = getattr(req, "_dsv41_p_certificate", None)
        if cert is not None:
            cert.admission.overlap.drain(req)
    return original(scheduler, req, *args, **kwargs)


def tail_plan(original, tail, *args, **kwargs):
    overlap = getattr(tail._model, "_dsv41_p_overlap", None)
    plan = overlap.current if overlap is not None else None
    if plan is not None and plan.jobs:
        return False, None
    return original(tail, *args, **kwargs)
