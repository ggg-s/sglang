"""DSV Decode-to-Decode ownership over a frozen native request.

Bulk bytes use native Mooncake with private pinned slots; TCP carries control.
Attention reuses DSV sparse Remote.
All allocator mutations and ownership decisions stay on TP scheduler commands.
"""

import json
import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from . import d2d_bulk as bulk
from . import d2d_groups as grouping
from . import d2d_ranges as ranges
from . import dsv41_remote as remote
from . import mooncake_backend as transport
from . import remote_data_plane as dp
from .transport_failure import TransportFailure

VERSION = "dsv-d2d-v2"
LIMIT = bulk.CAPACITY
COMMANDS = {
    "d2d_worker_status",
    "d2d_submit_pd",
    "d2d_cancel",
    "d2d_plan",
    "d2d_reserve",
    "d2d_reserve_cancel",
    "d2d_freeze",
    "d2d_import",
    "d2d_commit",
    "d2d_release",
    "d2d_resume",
    "d2d_status",
    "d2d_abort",
}


def new_stream(device):
    if torch.device(device).type == "cpu":
        return NS(
            synchronize=lambda: None, wait_event=lambda event: event.synchronize()
        )
    return torch.cuda.Stream(device=device)


def on_stream(device, stream):
    return (
        torch.cuda.stream(stream)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


def on_device(device):
    return (
        torch.cuda.device(device)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


def current_stream(device):
    return (
        torch.cuda.current_stream(device)
        if torch.device(device).type == "cuda"
        else new_stream(device)
    )


def can_quiesce(scheduler):
    queue = getattr(scheduler, "result_queue", ())
    if not queue:
        return True
    if len(queue) != 1:
        return False
    batch, _ = queue[0]
    running, last = scheduler.running_batch, scheduler.last_batch
    return bool(
        batch.forward_mode.is_decode()
        and running is not None
        and last is not None
        and set(map(id, running.reqs)) == set(map(id, last.reqs))
    )


def matrix(tensor, stride):
    raw = tensor.view(torch.uint8).reshape(-1)
    if stride <= 0:
        raise ValueError("nonpositive native state stride")
    # Native state pools can end in a partial sentinel ring. Never address it.
    return raw[: raw.numel() // stride * stride].reshape(-1, stride)


def state_views(pool):
    """Pointer lookup only; indices/strides come from native state registration."""
    tensors = []
    swa = (
        pool.request_window.state
        if pool.request_window is not None
        else pool.swa_kv_pool
    )
    tensors.extend(swa.kv_buffer)
    for pools in (pool.compress_state_pools, pool.indexer_compress_state_pools):
        tensors.extend(p.kv_score_buffer.kv_score for p in pools if p is not None)
    return {t.data_ptr(): t for t in tensors}


def state_rows(controller, req, length):
    if getattr(controller, "hybrid_gdn", False):
        from .qwen_hybrid import state_rows as hybrid_rows

        return hybrid_rows(controller, req, length)
    pool, scheduler = controller.pool, controller.scheduler
    views = state_views(pool)
    page = pool.page_size
    start = max(0, length - pool.sliding_window) // page * page
    if pool.request_window is not None:
        swa = pool.request_window.page_indices(req.kv.req_pool_idx, start, length)
    else:
        full = scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, start:length
        ]
        allocator = scheduler.token_to_kv_pool_allocator
        slots = allocator.translate_swa_indices_for_transfer(full)
        # Same positional page selection as native prefill/decode state planner.
        swa = slots[::page] // page
    swa = torch.as_tensor(
        swa, dtype=torch.int64, device=pool.get_swa_key_buffer_radix(0).device
    )
    groups = []
    for kind, info, indices in (
        ("swa", pool.get_state_buf_infos(), swa),
        ("request", pool.get_request_state_buf_infos(), None),
    ):
        pointers, _sizes, strides = info
        if not pointers:
            continue
        if indices is None:
            indices = torch.as_tensor(
                pool.request_state_transfer_indices(req.kv.req_pool_idx, length),
                device=swa.device,
                dtype=torch.int64,
            )
        for i, (pointer, stride) in enumerate(zip(pointers, strides)):
            table = matrix(views[pointer], int(stride))
            if indices.numel() and bool(
                ((indices < 0) | (indices >= table.shape[0])).any()
            ):
                raise ValueError("native D2D state index outside registration")
            groups.append(
                (
                    dict(kind=kind, index=i, rows=indices.numel(), stride=int(stride)),
                    table,
                    indices,
                )
            )
    return groups


def pool_device(controller):
    pool = controller.pool
    if getattr(controller, "hybrid_gdn", False):
        return pool.full_kv_pool.k_buffer[0].device
    return pool.get_swa_key_buffer_radix(0).device


def key_layout(controller, layer):
    if getattr(controller, "hybrid_gdn", False):
        return None  # page_size=1: every copied row is an immutable full page.
    return controller.pool.get_extra_key_layout(layer)


def manager(controller):
    return controller.scheduler.disagg_decode_prealloc_queue.kv_manager


def validate(controller):
    s = controller.scheduler
    if not controller.dsv41_remote or s.disaggregation_mode.value != "decode":
        raise ValueError("DSV D2D requires a remote Decode worker")
    if not s.tree_cache.disable:
        raise ValueError("DSV D2D v1 requires Decode radix cache disabled")
    if getattr(s, "enable_hisparse", False):
        raise ValueError("DSV D2D does not support HiSparse")
    if any(
        t.name
        not in (
            ("MAMBA",)
            if getattr(controller, "hybrid_gdn", False)
            else ("SWA", "DSV4_REQUEST_STATE")
        )
        for t in manager(controller).kv_args.state_types
    ):
        raise ValueError("D2D native state component is unsupported")


def source_blocker(req):
    """Explain why the current owner cannot yet become a D2D source."""
    if req.finished():
        return "request_finished"
    if getattr(req, "_handoff_frozen", False):
        return "request_frozen"
    if getattr(req, "_dsv_d2d_source", None) is not None:
        return "d2d_owner_active"
    if getattr(req, "_remote_attn_transport_quarantine", None) is not None:
        return "transport_quarantined"
    rpc = getattr(req, "_dsv41_remote", None)
    if rpc is not None and not rpc.all_local():
        return "remote_entries_pending"
    target = getattr(req, "_pre_layerwise_target", None)
    if target is None or target.closed:
        return None
    # Local switches do not retire the receive ring or canonical PD envelope.
    # Admit only a provably drainable lifetime, with no deferred KV release.
    if target.release is not None:
        return "previous_kv_release_pending"
    if target.release_failed:
        return "previous_transfer_release_failed"
    if not target.ready_to_finish():
        return "previous_transfer_pending"
    return None


def source_local(req):
    return source_blocker(req) is None


def planning_blocker(req):
    reason = source_blocker(req)
    if reason in ("remote_entries_pending", "previous_transfer_pending"):
        from .dsv_d2d_partial import can_freeze

        if can_freeze(req):
            reason = None
    if reason is not None:
        return reason
    if not req.output_ids:
        return "sampled_output_pending"
    return None


def migration_blocker(scheduler, req):
    reason = planning_blocker(req)
    if reason is not None:
        return reason
    return None if can_quiesce(scheduler) else "decode_boundary_pending"


def quiesce(scheduler, req):
    """Consume one native overlap result before changing batch membership.

    Never drain an extend/prebuilt result by copying the scheduler's merge
    policy: the caller retries after that native transition has completed.
    """
    queue = getattr(scheduler, "result_queue", ())
    if queue:
        if len(queue) != 1:
            raise RuntimeError("retry D2D freeze at a single Decode result boundary")
        batch, result = queue[0]
        running, last = scheduler.running_batch, scheduler.last_batch
        if (
            not batch.forward_mode.is_decode()
            or last is None
            or running is None
            or set(map(id, running.reqs)) != set(map(id, last.reqs))
        ):
            raise RuntimeError("retry D2D freeze after native extend/prebuilt merge")
        # Native processing waits copy_done and commits sampling/output once.
        queue.popleft()
        scheduler.process_batch_result(batch, result)
        scheduler.last_batch = None
    if req.finished():
        raise ValueError("cannot migrate a finished request")
    if migration_blocker(scheduler, req) is not None:
        raise ValueError("D2D source must have completed its previous transfer")
    expected = len(req.origin_input_ids) + len(req.output_ids) - 1
    if not req.output_ids or req.kv.kv_committed_len != expected:
        raise ValueError("D2D freeze requires one pending sampled token")


def sampling(req):
    # Recreate native SamplingParams, including penalties/seed, from its public
    # constructor inputs. Grammar/custom processors carry additional live state.
    import inspect

    from sglang.srt.sampling.sampling_params import SamplingParams

    if (
        getattr(req, "multimodal_inputs", None) is not None
        or req.grammar is not None
        or getattr(req, "custom_logit_processor", None)
        or req.return_logprob
    ):
        raise ValueError("D2D v1 does not migrate grammar/custom logits/logprob state")
    values = {
        name: getattr(req.sampling_params, name)
        for name in inspect.signature(SamplingParams).parameters
        if hasattr(req.sampling_params, name)
    }
    return json.loads(
        json.dumps(
            values, default=lambda v: sorted(v) if isinstance(v, set) else _not_json(v)
        )
    )


def _not_json(value):
    raise ValueError(f"D2D sampling value is not serializable: {type(value).__name__}")


@dataclass
class Source:
    owner: object
    req: object
    epoch: str
    destination: str
    target_rid: str
    parameters: dict
    path: Path
    generation: str = ""
    is_d2d: bool = True
    finalized: bool = True
    responded: bool = True
    released: bool = False
    defer_native_release: bool = True
    release: object = None
    error: object = None
    cache_lease: object = None
    committed: bool = False
    aborted: bool = False
    copied: bool = False
    retirement_failed: bool = False
    service: object = None
    states: list = field(default_factory=list)
    globals: list = field(default_factory=list)
    prefix_len: int = 0
    handle: object = None
    row: object = None
    peers: tuple = ()
    gpu_session: object = None
    bulk_sender: object = None
    group_size: int = 1
    token_range: int = 1024
    created: float = field(default_factory=time.monotonic)
    remote_closed: object = field(default_factory=threading.Event)
    lock: object = field(default_factory=threading.RLock)
    previous_target: object = None
    previous_rpc: object = None
    partial_remote: bool = False
    remote_entries_at_freeze: tuple = ()
    frozen_output: tuple = ()

    def initialize(self):
        from .dsv41_staging import buffers

        s, req = self.owner.scheduler, self.req
        if self.previous_target is not None and not self.previous_target.closed:
            raise TransportFailure(
                "cannot export D2D bytes while native PD still owns writes"
            )
        self.generation = self.epoch
        self.handle, self.row = req.cache_request_handle, req.kv.req_pool_idx
        self.prefix_len = int(req.kv.kv_committed_len)
        self.decode_prefix_len = 0
        self.schema = self.owner._schema(manager(self.owner))
        self.barrier_ready = True
        if getattr(self.owner, "hybrid_gdn", False):
            from .qwen_hybrid import freeze_ready

            freeze_ready(self.owner)
        self.states = state_rows(self.owner, req, self.prefix_len)
        self.pages = (
            s.req_to_token_pool.req_to_token[
                self.row, : self.prefix_len : self.owner.pool.page_size
            ]
            // self.owner.pool.page_size
        ).clone()
        self.globals = [
            matrix(t, entry[-1])
            for t, entry in zip(
                buffers(self.owner.pool, manager(self.owner)), self.schema["entries"]
            )
        ]
        self.final_event = dp.event(self.pages.device)
        catalog = transport.Catalog(manager(self.owner).engine, self.pages.device)
        self.bulk_sender = bulk.Sender(
            manager(self.owner).engine, req, self.pages.device, catalog=catalog
        )
        remote.register(self)
        self.meta = dict(
            version=VERSION,
            bulk_transport=bulk.PROTOCOL,
            bulk_capabilities=catalog.offer,
            d2d_group_size=self.group_size,
            d2d_group_protocol=grouping.PACKED,
            d2d_range_protocol=ranges.PROTOCOL,
            d2d_token_range=self.token_range,
            reservation_required=self.epoch in getattr(self.owner, "d2d_plans", {}),
            epoch=self.epoch,
            destination=self.destination,
            target_rid=self.target_rid,
            origin_input_ids=list(req.origin_input_ids),
            output_ids=list(req.output_ids),
            kv_len=self.prefix_len,
            sampling=self.parameters,
            detokenize={"output_prefix_len": len(req.output_ids)},
            schema=self.schema,
            tp_size=self.owner.tp_size,
            states=[d for d, t, i in self.states],
            route=self.service,
            source_partial_remote=self.partial_remote,
            source_remote_entries=list(self.remote_entries_at_freeze),
        )
        self.meta["route"] = self.service

    def check_read(self, *, allow_aborted=False):
        if (
            getattr(self.req, "_remote_attn_transport_quarantine", None) is not None
            or self.error
            or getattr(self.req, "_d2d_bulk_unsafe", False)
            or self.released
            or (self.aborted and not allow_aborted)
            or getattr(self.req, "_handoff_transfer_unsafe", False)
            or self.req.cache_request_handle != self.handle
            or self.req.kv.req_pool_idx != self.row
            or getattr(self.req, "_dsv_d2d_source", None) is not self
        ):
            raise TransportFailure("D2D frozen source ownership is unproven")
        if not getattr(self.req, "_handoff_frozen", False):
            raise TransportFailure("D2D source resumed while read lease is live")

    def poll(self):
        if time.monotonic() - self.created > self.owner.config.drain_timeout:
            raise TransportFailure("D2D source lease timed out; retaining KV")
        self.check_read(allow_aborted=True)
        if getattr(self, "meta", None) is None:
            # Frozen native PD still owns writes, or a scheduler command has
            # not yet published the drained D2D source. No read service exists.
            return False
        if self.aborted:
            return self.remote_closed.is_set() and self.release is not None
        if getattr(self.req, "_handoff_transfer_unsafe", False):
            raise TransportFailure("D2D source GPU reader fence failed")
        if not (self.committed and self.copied and self.remote_closed.is_set()):
            return False
        self.bulk_sender.close()
        if self.release is None:
            from sglang.srt.mem_cache.common import release_kv_cache

            release_kv_cache(
                self.req, self.owner.scheduler.tree_cache, checkpoint=False
            )
        return self.release is not None

    def retire(self):
        if self.released:
            return
        if self.retirement_failed or not self.poll():
            raise RuntimeError("D2D source retirement is unproven or failed")
        self.retirement_failed = True
        self.bulk_sender.close()
        self.release()
        self.release = None
        self.released = True
        self.req._pre_layerwise_source = None
        self.req._dsv_d2d_source = None
        self.globals.clear()
        self.states.clear()
        self.pages = None
        self.final_event = None
        tracked = remote.runtime()._tracked(self.owner.scheduler)
        if tracked.get(self.req.rid) is self.req:
            tracked.pop(self.req.rid)
        self.req._handoff_freed = True


def reply(data, **header):
    raw = data.contiguous().view(torch.uint8).reshape(-1).cpu()
    pad = (-raw.numel()) % 4
    if pad:
        raw = torch.cat((raw, torch.zeros(pad, dtype=torch.uint8)))
    out = raw.view(torch.float32)
    lse = torch.empty(0, dtype=torch.float32)
    return (
        dict(
            o_shape=list(out.shape),
            o_dtype="float32",
            lse_shape=[0],
            bytes=data.numel() * data.element_size(),
            **header,
        ),
        out,
        lse,
    )


def execute(job, stream, cache):
    source = getattr(job.ctx, "dsv41_source", None)
    if not getattr(source, "is_d2d", False):
        raise ValueError("not a D2D snapshot lease")
    h = job.header
    if (
        h.get("destination") != source.destination
        or h.get("migration_epoch") != source.epoch
    ):
        raise ValueError("D2D destination/epoch changed")
    action = h.get("dsv_d2d")
    with source.lock:
        try:
            source.check_read()
        except TransportFailure:
            if action in {"read", "read_group"}:
                # Positive proof for prepared cancellation: this invocation
                # never reached a native write. A socket error is NOT this proof.
                return reply(torch.empty(0, dtype=torch.uint8), bulk_rejected=True)
            raise
        if action == "abort":
            if source.committed:
                raise ValueError("cannot abort committed D2D owner")
            source.aborted = True
            remote.runtime().get_remote_hub(source.owner.scheduler).registry.release(
                job.ctx.handoff_id, expected_req=source.req
            )
            return reply(torch.empty(0, dtype=torch.uint8))
        if action == "manifest":
            session = h.get("gpu_session")
            if session:
                if source.gpu_session is not None and source.gpu_session != session:
                    raise ValueError("D2D GPU destination changed")
                source.gpu_session = session
                source.peers = ((NS(mooncake_session_id=session), None),)
            return reply(torch.empty(0, dtype=torch.uint8), manifest=source.meta)
        if action == "commit":
            source.committed = True
            return reply(torch.empty(0, dtype=torch.uint8))
        if action == "copied":
            source.copied = True
            return reply(torch.empty(0, dtype=torch.uint8))
        if action == "done":
            if not source.committed or not source.copied:
                raise ValueError("D2D done before commit/copy proof")
            remote.runtime().get_remote_hub(source.owner.scheduler).registry.release(
                job.ctx.handoff_id, expected_req=source.req
            )
            return reply(torch.empty(0, dtype=torch.uint8))
        if action not in {"read", "read_group"}:
            raise ValueError("unknown D2D snapshot action")
    selected = None
    if action == "read_group":
        selected = grouping.validate_segments(source, h.get("segments"), LIMIT)
        rectangle = h.get("token_range")
        if (
            not isinstance(rectangle, list)
            or len(rectangle) != 2
            or not all(type(x) is int for x in rectangle)
            or type(h.get("layer_group")) is not int
        ):
            raise ValueError("invalid D2D rectangle")
        lo, hi = rectangle
        page = source.owner.pool.page_size
        step = max(page, source.token_range // page * page)
        if lo < 0 or lo % step or hi != min(lo + step, source.prefix_len) or lo >= hi:
            raise ValueError("D2D token range outside frozen plan")
        for segment in h["segments"]:
            if (
                not lo // page
                <= segment["start"]
                < segment["start"] + segment["count"]
                <= (hi + page - 1) // page
            ):
                raise ValueError("D2D segment outside token range")
            entry = source.schema["entries"][segment["entry"]]
            if entry[2] // source.group_size != h["layer_group"]:
                raise ValueError("D2D segment outside layer group")
    else:
        kind, entry = h.get("kind"), int(h["entry"])
        if kind == "global" and 0 <= entry < len(source.globals):
            table, indices = source.globals[entry], source.pages
        elif kind == "state" and 0 <= entry < len(source.states):
            _desc, table, indices = source.states[entry]
        else:
            raise ValueError("invalid D2D tensor group")
        lo, count = int(h["start"]), int(h["count"])
        if (
            lo < 0
            or count <= 0
            or lo + count > indices.numel()
            or count * table.shape[1] > LIMIT
        ):
            raise ValueError("D2D read range exceeds frozen ownership/budget")
    region = bulk.descriptor(h.get("bulk_region"))
    sequence = h.get("bulk_sequence")
    device = source.globals[0].device if selected is not None else table.device
    with on_device(device), on_stream(device, stream):
        if stream is not None:
            stream.wait_event(source.final_event)
        if selected is None:
            data = table.index_select(
                0, indices[lo : lo + count].to(torch.int64)
            ).contiguous()
        else:
            data = torch.cat(
                [
                    t.index_select(0, source.pages[start : start + n].to(torch.int64))
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(-1)
                    for t, start, n in selected
                ]
            )
        ready = dp.event(device)
        size = source.bulk_sender.write(data, ready, region, sequence)
    # Empty tensor envelope preserves the authenticated RemoteHub request lease;
    # Receiver checks the host arrival footer before consuming any KV bytes.
    return reply(
        torch.empty(0, dtype=torch.uint8),
        bulk_protocol=region["protocol"],
        bulk_nonce=region["nonce"],
        bulk_sequence=sequence,
        bulk_bytes=size,
    )


class Reader:
    def __init__(
        self, route, destination, epoch, timeout=600, *, owner, req, offer=None
    ):
        self.route, self.destination, self.epoch = route, destination, epoch
        self.client = remote.runtime().TargetClient(route["host"], route["port"])
        self.client.attach(route["handoff_id"], route["token"], route["epoch"])
        self.query = bytes(1024)
        self.timeout = timeout
        self.closed = False
        self.owner, self.req = owner, req
        self.offer = offer if offer is not None else transport.tcp_offer()
        self.bulk = None

    def call(self, action, **fields):
        if self.closed:
            raise RuntimeError("D2D reader already closed")
        h = dict(
            self.route,
            dsv_d2d=action,
            destination=self.destination,
            migration_epoch=self.epoch,
            q_shape=[1, 512],
            q_dtype="bfloat16",
            q_bytes=1024,
            **fields,
        )
        if action not in {"read", "read_group"}:
            response, out, _lse = self.client.call(
                h, self.query, deadline_s=self.timeout
            )
            if not response.get("ok"):
                raise RuntimeError(response.get("error", "D2D snapshot failed"))
            return response, out.view(torch.uint8)[: response["bytes"]]
        if self.bulk is None:
            device = pool_device(self.owner)
            catalog = transport.Catalog(manager(self.owner).engine, device)
            backend = catalog.select(self.offer)
            self.bulk = bulk.Receiver(
                catalog.engines[backend], self.req, device, backend=backend
            )
        with self.bulk.lock:
            try:
                h["bulk_region"] = self.bulk.prepare()
                h["bulk_sequence"] = self.bulk.sequence
                response, out, lse = self.client.call(
                    h, self.query, deadline_s=self.timeout, retry_on_disconnect=False
                )
                if not response.get("ok"):
                    raise RuntimeError(response.get("error", "D2D snapshot failed"))
                if response.get("bulk_rejected") is True:
                    raise bulk.ReadRejected("D2D read rejected before native write")
                if out.numel() or lse.numel() or response.get("bytes") != 0:
                    raise ValueError("D2D bulk unexpectedly returned TCP tensor bytes")
                size = response.get("bulk_bytes")
                # Existing callers validate row counts against their own schema.
                header = dict(response, bytes=size)
                data = self.bulk.consume(header, size, self.timeout)
                return header, data
            except bulk.ReadRejected:
                raise
            except BaseException:
                # Even an RPC timeout after a successful send is uncertain to
                # this receiver: never reuse/deregister its slot or free target KV.
                self.bulk.poison()
                raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.client.detach(self.route["handoff_id"])
        finally:
            self.client.close()
            if self.bulk is not None and not self.bulk.unsafe:
                self.bulk.close()


def copy_rows(reader, kind, entry, dst, indices, rows):
    if indices.numel() != rows:
        raise ValueError("D2D native source/destination state counts differ")
    stride = dst.shape[1]
    chunk = LIMIT // stride
    if not chunk:
        raise ValueError("native D2D row exceeds transfer budget")
    for lo in range(0, rows, chunk):
        n = min(chunk, rows - lo)
        _h, data = reader.call("read", kind=kind, entry=entry, start=lo, count=n)
        if data.numel() != n * stride:
            raise ValueError("truncated D2D snapshot")
        dst.index_copy_(
            0,
            indices[lo : lo + n].long(),
            data.reshape(n, stride).to(dst.device, non_blocking=True),
        )


@dataclass(eq=False)
class Resource:
    parent: object
    entry: int
    admission_id: int = 0
    closed: bool = False
    release: object = None

    @property
    def decode_req(self):
        return self.parent.decode_req

    @property
    def generation(self):
        return self.parent.generation

    def gpu_ready(self):
        return self.parent.closed or self.parent.gpu_ready()

    def transport_ready(self):
        if self.parent.closed:
            return True
        self.parent.refresh()
        if self.parent.cancelled:
            return self.parent.transport_ready()
        event = self.parent.events.get(self.entry)
        return event is not None and event.query()

    def ready_to_finish(self):
        return self.gpu_ready() and self.transport_ready()

    def finish(self, *, wait=False):
        if self.closed:
            return
        if not self.ready_to_finish():
            raise RuntimeError("D2D resource completion unproven")
        if not self.parent.closed and not self.parent.cancelled:
            self.parent.rpc.local_entries.add(self.entry)
        self.closed = True


from .pre_layerwise import Target as NativeTarget


@dataclass(eq=False)
class Target(NativeTarget):
    reader: object = None
    bulk_reader: object = None
    rpc: object = None
    events: dict = field(default_factory=dict)
    worker: object = None
    error: object = None
    copied: bool = False
    admitted: bool = False
    cancelled: bool = False
    abort_ready: object = None
    copy_stream: object = None
    setup_stream: object = None
    stop: object = field(default_factory=threading.Event)
    final_event: object = None
    allocation_complete: bool = False
    created: float = field(default_factory=time.monotonic)
    resources: list = field(default_factory=list)
    group_size: int = 1
    token_range: int = 1024
    group_frames: int = 0
    grouped_entries: int = 0

    def refresh(self):
        if (
            getattr(self.decode_req.req, "_remote_attn_transport_quarantine", None)
            is not None
        ):
            raise TransportFailure("D2D target is quarantined; retaining KV")
        if time.monotonic() - self.created > self.owner.config.drain_timeout:
            raise TransportFailure("D2D target lease timed out; retaining KV")
        if self.cancelled:
            return
        if self.rpc is not None:
            self.rpc.check()
        if self.error:
            raise TransportFailure(
                f"D2D copy failed; retaining target KV: {self.error}"
            )

    def transport_ready(self):
        self.refresh()
        if self.cancelled:
            return self.abort_ready is not None and self.abort_ready.query()
        return self.copied and self.final_event is not None and self.final_event.query()

    def gpu_ready(self):
        if self.cancelled and not self.admitted:
            return self.abort_ready is not None and self.abort_ready.query()
        return super().gpu_ready()

    def ready_to_finish(self):
        return (
            (self.admitted or self.cancelled)
            and self.gpu_ready()
            and self.transport_ready()
        )

    def finish(self, *, wait=False):
        if self.closed:
            return
        if self.release_failed or not self.ready_to_finish():
            raise RuntimeError("D2D target retirement unproven/failed")
        self.release_failed = True
        if self.worker is not None:
            self.worker.join()
        if not self.cancelled:
            self.reader.call("done")
        if self.rpc is not None:
            if self.cancelled:
                # Prepared imports have never entered model execution. The
                # Copy-stream fences prove local writes only; uncertain native
                # bulk writes are rejected by abort_import before retirement.
                if self.rpc.calls or self.forward_proofs:
                    raise RuntimeError("prepared D2D abort has model consumers")
                self.rpc.closed = True
                self.owner.remote_targets.pop(self.rpc.row, None)
                self.decode_req.req._dsv41_remote = None
                getattr(self.rpc, "_sparse_page_maps", {}).clear()
            else:
                self.rpc.close("d2d-tp-local")
        if self.reader is not None:
            self.reader.close()
        if self.release is not None:
            self.release()
            self.release = None
        self.decode_req.req._pre_layerwise_target = None
        self.closed = True
        self.release_failed = False
        self.events.clear()
        self.final_event = self.setup_stream = self.copy_stream = None
        self.decode_req.ready = None
        self.worker = None

    def transfer(self):
        from .dsv41_receive import scatter_prefix
        from .dsv41_staging import buffers

        req, pool = self.decode_req.req, self.owner.pool
        try:
            self.bulk_reader = Reader(
                self.reader.route,
                self.reader.destination,
                self.reader.epoch,
                self.reader.timeout,
                owner=self.owner,
                req=req,
                offer=self.reader.offer,
            )
            copy_device = pool_device(self.owner)
            self.copy_stream = new_stream(copy_device)
            with on_device(copy_device), on_stream(copy_device, self.copy_stream):
                self.copy_stream.wait_event(self.decode_req.ready)
                pages = (
                    self.owner.scheduler.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, : self.coverage.prefix_len : pool.page_size
                    ]
                    // pool.page_size
                ).long()
                tensors = buffers(pool, manager(self.owner))
                entries = self.owner._schema(manager(self.owner))["entries"]
                if self.token_range:
                    coverage = ranges.Coverage(len(entries), pages.numel())
                    for rectangle in grouping.range_frames(
                        entries,
                        self.coverage.prefix_len,
                        pool.page_size,
                        self.group_size,
                        self.token_range,
                        LIMIT,
                    ):
                        frame = rectangle["segments"]
                        if self.stop.is_set():
                            raise RuntimeError("D2D prepared import cancelled")
                        _h, data = self.bulk_reader.call("read_group", **rectangle)
                        expected = sum(
                            segment["count"] * entries[segment["entry"]][-1]
                            for segment in frame
                        )
                        if data.numel() != expected:
                            raise ValueError("truncated D2D packed frame")
                        self.group_frames += 1
                        self.grouped_entries = max(
                            self.grouped_entries, len({p["entry"] for p in frame})
                        )
                        offset = 0
                        for segment in frame:
                            i, lo, count = (
                                segment[k] for k in ("entry", "start", "count")
                            )
                            entry = entries[i]
                            dst = matrix(tensors[i], entry[-1])
                            size = count * dst.shape[1]
                            src = data[offset : offset + size].reshape(count, -1)
                            scatter_prefix(
                                dst,
                                src,
                                pages[lo : lo + count],
                                lo,
                                self.coverage.prefix_len,
                                pool.page_size,
                                entry[1],
                                key_layout(self.owner, entry[2]),
                            )
                            offset += size
                            if coverage.add(i, lo, lo + count):
                                self.events[i] = dp.event(copy_device)
                else:
                    for i, (tensor, entry) in enumerate(
                        zip(
                            buffers(pool, manager(self.owner)),
                            self.owner._schema(manager(self.owner))["entries"],
                        )
                    ):
                        if entry[0] != "kv":
                            continue
                        dst = matrix(tensor, entry[-1])
                        chunk = LIMIT // dst.shape[1]
                        if not chunk:
                            raise ValueError("D2D global row exceeds budget")
                        for lo in range(0, pages.numel(), chunk):
                            if self.stop.is_set():
                                raise RuntimeError("D2D prepared import cancelled")
                            n = min(chunk, pages.numel() - lo)
                            _h, data = self.bulk_reader.call(
                                "read", kind="global", entry=i, start=lo, count=n
                            )
                            if data.numel() != n * dst.shape[1]:
                                raise ValueError("truncated D2D global rows")
                            src = data.reshape(n, -1).to(copy_device, non_blocking=True)
                            scatter_prefix(
                                dst,
                                src,
                                pages[lo : lo + n],
                                lo,
                                self.coverage.prefix_len,
                                pool.page_size,
                                entry[1],
                                key_layout(self.owner, entry[2]),
                            )
                        self.events[i] = dp.event(copy_device)
                self.final_event = dp.event(copy_device)
                self.final_event.synchronize()
                if not self.stop.is_set():
                    self.reader.call("copied")
                    self.copied = True
        except BaseException as exc:
            self.error = repr(exc)
            dp.io_for(req).poison()
        finally:
            if self.bulk_reader is not None:
                self.bulk_reader.close()


def find_req(scheduler, rid):
    candidates = list(remote.runtime()._tracked(scheduler).values())
    for name in ("running_batch", "last_batch"):
        batch = getattr(scheduler, name, None)
        if batch is not None:
            candidates.extend(batch.reqs)
    for req in candidates:
        if req.rid == rid:
            return req
    raise KeyError(f"active request {rid!r} not found")


def records(controller):
    if not hasattr(controller, "d2d_sources"):
        controller.d2d_sources, controller.d2d_targets = {}, {}
    return controller.d2d_sources, controller.d2d_targets


def handle(controller, data):
    validate(controller)
    sources, targets = records(controller)
    cmd = data["cmd"]
    epoch = str(data.get("migration_epoch", ""))
    if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", epoch) is None:
        raise ValueError("D2D requires a safe explicit migration_epoch")
    from . import dsv_d2d_reservation as reservation
    from . import dsv_gateway_control as gateway

    expected = data.get("expected_instances")
    if expected is not None:
        matches = (
            isinstance(expected, list)
            and len(expected) == controller.tp_size
            and expected[controller.rank] == gateway.instance(controller)
        )
        if not controller.agree([int(matches)])[0]:
            raise ValueError("D2D worker instance changed; refusing stale command")
    if cmd in gateway.COMMANDS:
        return gateway.handle(controller, data)
    if cmd in {"d2d_plan", "d2d_reserve", "d2d_reserve_cancel"}:
        return reservation.handle(controller, data)
    if cmd == "d2d_status":
        return dict(
            migration_epoch=epoch,
            instance_id=gateway.instance(controller),
            **gateway.migration_status(controller, epoch),
        )
    if cmd == "d2d_freeze":
        if epoch in getattr(controller, "d2d_source_cancelled_epochs", ()):
            raise ValueError("D2D source epoch was cancelled")
        if epoch in sources:
            source = sources[epoch]
            if source.released or source.aborted:
                raise ValueError("D2D source epoch is retired")
            if source.error:
                raise RuntimeError(
                    "D2D source preparation failed; inspect status/resume"
                )
            if getattr(source, "meta", None) is not None:
                return dict(transfer=source.meta)
            from .dsv_d2d_partial import prepare

            return prepare(controller, source)
        if epoch in targets:
            raise ValueError("migration epoch was already used by a D2D import")
        error = None
        try:
            if epoch in getattr(controller, "d2d_reservations", {}):
                raise ValueError("migration epoch belongs to a target reservation")
            req = find_req(controller.scheduler, data["rid"])
            destination, target_rid = str(data["destination"]), str(data["target_rid"])
            if not destination or not target_rid:
                raise ValueError("D2D destination and target rid cannot be empty")
            params = sampling(req)
            group_size = grouping.size()
            token_range = ranges.tokens()
            blocker = migration_blocker(controller.scheduler, req)
        except Exception as exc:
            error = repr(exc)
        votes = controller.agree(
            [
                int(error is None),
                int(error is None and blocker is None),
                int(error is not None or blocker != "request_finished"),
                int(
                    error is not None
                    or blocker
                    in (
                        None,
                        "decode_boundary_pending",
                        "remote_entries_pending",
                        "previous_transfer_pending",
                        "sampled_output_pending",
                    )
                ),
            ]
        )
        if not votes[0]:
            raise ValueError(
                f"TP D2D freeze preflight rejected: {error or 'peer rejected'}"
            )
        if not votes[1]:
            return dict(
                deferred=True,
                migration_deferred_reason="request_finished"
                if not votes[2]
                else blocker or "peer_not_ready",
                retryable=bool(votes[3]),
            )
        error = None
        try:
            quiesce(controller.scheduler, req)
        except Exception as exc:
            error = repr(exc)
        votes = controller.agree(
            [
                int(error is None),
                int(not req.finished()),
                int(error is None or req.finished()),
            ]
        )
        if not votes[0]:
            if not votes[1] and votes[2]:
                return dict(
                    deferred=True,
                    migration_deferred_reason="request_finished",
                    retryable=False,
                )
            raise ValueError(f"TP D2D quiesce rejected: {error or 'peer rejected'}")
        # Quiescing can commit one overlap result; validate the final length
        # collectively before removing any rank from its runnable batches.
        error = None
        try:
            reservation.check_freeze(controller, data, req)
            source = Source(
                controller,
                req,
                epoch,
                destination,
                target_rid,
                params,
                controller.config.directory
                / "d2d"
                / epoch
                / str(controller.rank)
                / "source",
                group_size=group_size,
                token_range=token_range,
                previous_target=getattr(req, "_pre_layerwise_target", None),
                previous_rpc=getattr(req, "_dsv41_remote", None),
                prefix_len=int(req.kv.kv_committed_len),
                handle=req.cache_request_handle,
                row=req.kv.req_pool_idx,
                frozen_output=tuple(req.output_ids),
            )
            rpc = source.previous_rpc
            if rpc is not None and getattr(rpc, "progressive", False):
                schema = controller._schema(manager(controller))
                source.remote_entries_at_freeze = tuple(
                    i
                    for i, entry in enumerate(schema["entries"])
                    if entry[0] == "kv" and i not in rpc.local_entries
                )
                source.partial_remote = bool(source.remote_entries_at_freeze)
        except Exception as exc:
            error = repr(exc)
        if not controller.agree([int(error is None)])[0]:
            raise ValueError(f"TP D2D reservation rejected: {error or 'peer rejected'}")
        req._handoff_frozen = True
        req._handoff_req = True
        req._handoff_epoch = epoch
        req._dsv_d2d_source = req._pre_layerwise_source = source
        sources[epoch] = source
        controller.sources.append(source)
        remote.runtime()._tracked(controller.scheduler)[req.rid] = req
        error = None
        try:
            remote.runtime()._remove_from_batches(controller.scheduler, req)
        except Exception as exc:
            error = repr(exc)
        if not controller.agree([int(error is None)])[0]:
            source.error = f"TP D2D batch removal failed: {error or 'peer failed'}"
            raise TransportFailure(source.error)
        from .dsv_d2d_partial import prepare

        return prepare(controller, source)
    if cmd == "d2d_import":
        return import_request(controller, data)
    if cmd == "d2d_abort":
        target = targets.get(epoch)
        if not controller.agree([int(target is None or not target.admitted)])[0]:
            raise ValueError(
                "committed D2D target must use native request cancellation"
            )
        if not hasattr(controller, "d2d_cancelled_epochs"):
            controller.d2d_cancelled_epochs = set()
        controller.d2d_cancelled_epochs.add(epoch)
        result = (
            dict(aborted=True) if target is None else abort_import(controller, target)
        )
        ticket = getattr(controller, "d2d_reservations", {}).get(epoch)
        if result.get("aborted") and (ticket is None or not ticket.consumed):
            reservation.handle(
                controller, dict(cmd="d2d_reserve_cancel", migration_epoch=epoch)
            )
        return result
    if cmd == "d2d_commit":
        target = targets.get(epoch)
        error = None
        try:
            if target is None:
                raise ValueError("D2D target not prepared")
            if epoch in getattr(controller, "d2d_cancelled_epochs", ()) or target.error:
                raise ValueError("D2D commit epoch aborted/cancelled or target failed")
            if not target.admitted:
                if (
                    target.cancelled
                    or target.closed
                    or target.decode_req.req.finished()
                ):
                    raise ValueError("cannot commit aborted/finished D2D import")
                target.refresh()
        except Exception as exc:
            error = repr(exc)
        if not controller.agree([int(error is None)])[0]:
            raise ValueError(f"TP D2D commit rejected: {error or 'peer rejected'}")
        confirmed = target.admitted or getattr(target, "source_confirmed", False)
        if not confirmed:
            try:
                target.reader.call("commit")
                target.source_confirmed = confirmed = True
            except Exception as exc:
                error = repr(exc)
        # Never publish a rank-local Decode batch after only a subset of
        # source shards acknowledged ownership. Partial confirmation can only
        # roll forward; all target shards remain unadmitted until this vote.
        if not controller.agree([int(confirmed)])[0]:
            raise TransportFailure(
                f"TP D2D commit confirmation pending: {error or 'peer pending'}"
            )
        error = None
        req = target.decode_req.req
        if not target.admitted:
            target.admitted = True
            from .handoff_control import _flush_decode_resume

            try:
                _flush_decode_resume(controller.scheduler, [req])
            except BaseException as exc:
                error = repr(exc)
        if not controller.agree([int(error is None)])[0]:
            target.error = f"native D2D admission failed: {error or 'peer failed'}"
            dp.io_for(req).poison()
            # The control was broadcast and this failure was TP-agreed. Use
            # the existing native abort/quarantine boundary to stop every
            # shard and retain rows, readers, registrations and free callbacks.
            from .transport_failure import quarantine, request_of

            quarantine(controller, target, target.error)
            protocol = controller.decode_overlap.retirement
            for index, entry in list(protocol.registered.items()):
                if request_of(entry[0]) is req:
                    protocol.registered.pop(index)
            raise TransportFailure(target.error)
        return dict(admitted=True, source_confirmed=True, rid=req.rid)
    if cmd == "d2d_release":
        source = sources[epoch]
        return dict(
            released=source.released, committed=source.committed, copied=source.copied
        )
    if cmd == "d2d_resume":
        source = sources.get(epoch)
        present, absent = controller.agree(
            [int(source is not None), int(source is None)]
        )
        if not present and not absent:
            raise TransportFailure(
                "asymmetric TP source preparation; retaining frozen rows"
            )
        if not hasattr(controller, "d2d_source_cancelled_epochs"):
            controller.d2d_source_cancelled_epochs = set()
        controller.d2d_source_cancelled_epochs.add(epoch)
        if absent:
            return dict(resumed=True, never_frozen=True)
        return resume_source(controller, source)
    raise ValueError("unsupported D2D command")


def allocation_preflight(controller, req, required=None):
    prealloc = controller.scheduler.disagg_decode_prealloc_queue
    full, swa = (
        required if required is not None else prealloc._prealloc_required_tokens(req)
    )
    full_budget = prealloc._allocatable_token_budgets(retractable_tokens=0)
    swa_budget = (
        prealloc._swa_tail_allocatable_token_budget(retractable_tokens=0)
        if prealloc._uses_swa_reservation()
        else None
    )
    fits = bool(
        controller.scheduler.req_to_token_pool.available_size()
    ) and prealloc._prealloc_reservation_fits(
        full,
        swa,
        full_allocatable_tokens=full_budget,
        swa_allocatable_tokens=swa_budget,
    )
    if fits and getattr(controller, "hybrid_gdn", False):
        from .qwen_hybrid import state_capacity

        fits = state_capacity(controller, req)
    if not fits:
        raise ValueError("D2D target native KV/state/request-row budget unavailable")


def import_request(controller, data):
    from sglang.srt.sampling.sampling_params import SamplingParams

    from .dsv41_staging import buffers
    from .handoff_control import _new_req, _select_transfer, _tracked
    from .pre_layerwise import LayerCoverage

    sources, targets = records(controller)
    epoch = str(data["migration_epoch"])
    if epoch in getattr(controller, "d2d_cancelled_epochs", ()):
        raise ValueError("D2D import epoch cancelled")
    if epoch in sources or epoch in getattr(controller, "d2d_plans", {}):
        raise ValueError("migration epoch was already used by a D2D source")
    if epoch in targets:
        target = targets[epoch]
        if target.error or target.cancelled:
            raise ValueError("D2D import has failed/aborted; cannot retry allocation")
        return dict(rid=target.decode_req.req.rid, imported=True, duplicate=True)
    transfer = _select_transfer(data, controller.rank, controller.tp_size)
    if (
        transfer["version"] != VERSION
        or transfer.get("bulk_transport") != bulk.PROTOCOL
        or transfer["epoch"] != epoch
        or transfer["tp_size"] != controller.tp_size
    ):
        raise ValueError("D2D transfer epoch/TP mismatch")
    transport.validate(transfer.get("bulk_capabilities"))
    transport.Catalog(manager(controller).engine, pool_device(controller)).select(
        transfer["bulk_capabilities"]
    )
    ranges.check_history(
        transfer["origin_input_ids"], transfer["output_ids"], transfer["kv_len"]
    )
    if transfer["schema"] != controller._schema(manager(controller)):
        raise ValueError("D2D model/KV/state wire schema mismatch")
    if getattr(controller, "hybrid_gdn", False):
        from .qwen_hybrid import validate_transfer

        validate_transfer(transfer)
    if transfer.get("d2d_range_protocol") != ranges.PROTOCOL:
        raise ValueError("D2D requires layer/token range protocol")
    token_range = ranges.tokens(transfer["d2d_token_range"])
    group_size = grouping.size(transfer.get("d2d_group_size", 1))
    if group_size > 1 and transfer.get("d2d_group_protocol") != grouping.PACKED:
        raise ValueError("unsupported D2D packed group protocol")
    destination = str(data["destination"])
    if destination != transfer["destination"]:
        raise ValueError("D2D destination mismatch")
    rid = transfer["target_rid"]
    if rid in _tracked(controller.scheduler):
        raise ValueError("D2D target rid already live")
    params = SamplingParams(**transfer["sampling"])
    params.normalize(controller.scheduler.tokenizer)
    req = _new_req(
        controller.scheduler,
        rid,
        transfer["origin_input_ids"],
        transfer["output_ids"],
        params,
    )
    from .compat import restore_detokenize_prefix

    restore_detokenize_prefix(
        req, transfer["detokenize"].get("output_prefix_len", len(req.output_ids))
    )
    target = Target(
        controller,
        None,
        NS(req=req, kv_receiver=None),
        controller.config.directory / "d2d" / epoch / str(controller.rank) / "target",
        epoch,
        LayerCoverage(transfer["kv_len"], controller.layers),
    )
    target.group_size = group_size
    target.token_range = token_range
    target.queue = getattr(controller.scheduler, "disagg_decode_transfer_queue", None)
    if target.queue is None:
        from sglang.srt.managers.scheduler_components.request_list import (
            IndexedReqList,
            ListKind,
        )

        target.queue = NS(
            queue=IndexedReqList(ListKind.PD_RECEIVING, req_of=lambda item: item.req)
        )
    # D2D owns its frozen manifest and pending token already. It shares the
    # native queue interface for lifetime tracking but never enters the normal
    # PD metadata/token-commit queue and must not sample that token a second time.
    targets[epoch] = target
    _tracked(controller.scheduler)[rid] = req
    req._pre_layerwise_target = target
    reader = None
    try:
        target.setup_stream = current_stream(pool_device(controller))
        prealloc = controller.scheduler.disagg_decode_prealloc_queue
        from . import dsv_d2d_reservation as reservation

        if not reservation.consume(controller, data, transfer, req):
            allocation_preflight(controller, req)
            prealloc._pre_alloc(req)
        target.allocation_complete = True
        if req.kv.kv_committed_len != transfer["kv_len"]:
            raise RuntimeError("native D2D committed length mismatch")
        reader = target.reader = Reader(
            transfer["route"],
            destination,
            epoch,
            controller.config.drain_timeout,
            owner=controller,
            req=req,
            offer=transfer["bulk_capabilities"],
        )
        session = (
            dp.plane_for(controller).engine.get_session_id() if dp.enabled() else None
        )
        meta, _ = reader.call("manifest", gpu_session=session)
        if meta["manifest"] != transfer:
            raise ValueError("D2D frozen manifest changed")
        pool = controller.pool
        if getattr(controller, "hybrid_gdn", False):
            from .qwen_hybrid import clear_state

            clear_state(controller, req, transfer["kv_len"])
        else:
            pool.clear_request_scoped_state(req.kv.req_pool_idx)
        states = state_rows(controller, req, transfer["kv_len"])
        if [d for d, t, i in states] != transfer["states"]:
            raise ValueError("native D2D state layout/count changed")
        for i, (desc, table, indices) in enumerate(states):
            copy_rows(reader, "state", i, table, indices, desc["rows"])
        pages = (
            controller.scheduler.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, : transfer["kv_len"] : pool.page_size
            ]
            // pool.page_size
        ).long()
        for i, (tensor, entry) in enumerate(
            zip(buffers(pool, manager(controller)), transfer["schema"]["entries"])
        ):
            if entry[0] == "index":
                copy_rows(
                    reader, "global", i, matrix(tensor, entry[-1]), pages, pages.numel()
                )
        target.decode_req.ready = dp.event(pages.device)
        target.decode_req.ready.synchronize()
        if getattr(controller, "hybrid_gdn", False):
            req.kv.mamba_needs_clear = False
        rpc_class = remote.Target
        if getattr(controller, "hybrid_gdn", False):
            from .qwen_hybrid import Target as rpc_class

        rpc = target.rpc = rpc_class(
            controller,
            req,
            target.path,
            epoch,
            transfer["route"],
            transfer["kv_len"],
            progressive=True,
        )
        req._dsv41_remote = rpc
        controller.remote_targets[rpc.row] = rpc
        req.prefix_len = req.extend_end = transfer["kv_len"]
        req._handoff_epoch = epoch
        req._handoff_base_output_len = len(req.output_ids)
        retirement = controller.decode_overlap.retirement
        for i, entry in enumerate(transfer["schema"]["entries"]):
            if entry[0] == "kv":
                resource = Resource(target, i, retirement.last_admission + 1)
                retirement.register(resource)
                target.resources.append(resource)
                controller.targets.append(resource)
        target.admission_id = retirement.last_admission + 1
        retirement.register(target)
        controller.targets.append(target)
        target.worker = threading.Thread(
            target=target.transfer, daemon=True, name="dsv-d2d-copy"
        )
        target.worker.start()
        return dict(
            rid=rid,
            imported=True,
            kv_len=transfer["kv_len"],
            output_len=len(req.output_ids),
        )
    except BaseException as exc:
        target.error = repr(exc)
        # The target row may already contain GPU writes. Preserve it for an
        # explicit proof/abort path rather than rolling back blindly.
        dp.io_for(req).poison()
        raise


def resume_source(controller, source):
    # Revoke the service before looking for a positive final reader fence.
    # A concurrent destination commit is serialized by the same source lock.
    already_resumed = source.released and not source.committed
    can_resume = already_resumed or (
        not source.committed
        and not source.req.finished()
        and not source.released
        and not getattr(source.req, "_d2d_bulk_unsafe", False)
        and source.release is None
        and getattr(source.req, "_remote_attn_transport_quarantine", None) is None
        and not (
            source.previous_target is not None
            and not source.previous_target.closed
            and source.error
        )
    )
    if not controller.agree([int(can_resume)])[0]:
        raise ValueError("destination owns generation or source already released")
    with source.lock:
        if source.committed:
            raise ValueError("destination committed during source resume")
        source.aborted = True
        service = getattr(source, "service", None)
        if service is None:
            source.remote_closed.set()
        elif not source.remote_closed.is_set():
            hub = remote.runtime().get_remote_hub(controller.scheduler)
            hub.registry.release(service["handoff_id"], expected_req=source.req)
    safe = source.remote_closed.is_set() and not getattr(
        source.req, "_handoff_transfer_unsafe", False
    )
    if not controller.agree([int(safe)])[0]:
        return dict(resumed=False, pending_reader=True)
    if already_resumed:
        return dict(resumed=True, rid=source.req.rid, duplicate=True)
    if source.bulk_sender is not None:
        source.bulk_sender.close()
    req = source.req
    source.error = None
    source.released = True  # Retire the read lease, retaining the native KV row.
    req._pre_layerwise_source = req._dsv_d2d_source = None
    req._handoff_frozen = False
    req.prefix_len = req.extend_end = source.prefix_len
    source.globals.clear()
    source.states.clear()
    source.pages = None
    source.final_event = None
    if source in controller.sources:
        controller.sources.remove(source)
    remote.runtime()._flush_decode_resume(controller.scheduler, [req])
    return dict(resumed=True, rid=req.rid)


def abort_import(controller, target):
    if target.closed:
        return dict(aborted=target.cancelled, closed=True)
    req = target.decode_req.req
    if getattr(req, "_remote_attn_transport_quarantine", None) is not None or getattr(
        req, "_d2d_bulk_unsafe", False
    ):
        raise TransportFailure("D2D quarantine cannot be bypassed by prepared abort")
    if not target.allocation_complete and req.kv.req_pool_idx is not None:
        raise TransportFailure(
            "partial native D2D allocation is unproven; retaining row"
        )
    if target.release_failed:
        raise RuntimeError("D2D target release failed; refusing abort retry")
    if target.admitted:
        raise ValueError("committed D2D target must use native request cancellation")
    target.stop.set()
    if target.worker is not None:
        target.worker.join(timeout=1)
        if target.worker.is_alive():
            return dict(aborted=False, pending_reader=True)
    for reader in (target.reader, target.bulk_reader):
        if reader is not None and reader.bulk is not None and reader.bulk.unsafe:
            raise TransportFailure("D2D bulk completion unknown; retaining target KV")
    # Join prevents new submissions. Prove H2D/scatter completion as well as
    # native host writes before deregistering any slot or releasing target KV.
    for stream in (target.setup_stream, target.copy_stream):
        if stream is not None:
            stream.synchronize()
    for reader in (target.reader, target.bulk_reader):
        if reader is not None and reader.bulk is not None:
            reader.bulk.close()
    if target.reader is not None:
        try:
            target.reader.call("abort")
        except Exception:
            pass  # Source resume independently revokes/fences its lease.
    target.abort_ready = dp.event(pool_device(controller))
    target.abort_ready.synchronize()
    target.cancelled = True
    from sglang.srt.mem_cache.common import release_kv_cache

    req = target.decode_req.req
    if req.kv.holds_kv:
        release_kv_cache(req, controller.scheduler.tree_cache, checkpoint=False)
    if not target.admission_id:
        protocol = controller.decode_overlap.retirement
        target.admission_id = protocol.last_admission + 1
        protocol.register(target)
        controller.targets.append(target)
    return dict(aborted=True, pending_retirement=True)


def drain_sources(controller):
    """Shutdown consumes the same TP source proof; a D2D source has no PD sender."""

    def pending():
        return any(getattr(s, "is_d2d", False) for s in controller.sources)

    deadline = time.monotonic() + controller.config.drain_timeout
    while pending():
        controller.poll_sources()
        if pending():
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "D2D sources remain active during shutdown; retaining KV"
                )
            time.sleep(0.005)
