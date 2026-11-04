"""DSV Decode-to-Decode ownership over a frozen native request.

The bulk reader uses bounded TCP frames; attention reuses DSV sparse Remote.
All allocator mutations and ownership decisions stay on TP scheduler commands.
"""

import json
import re
import threading
import time
from array import array
from dataclasses import dataclass, field
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from . import dsv41_remote as remote, remote_data_plane as dp
from .transport_failure import TransportFailure

VERSION = "dsv-d2d-v1"
LIMIT = 4 * 1024 * 1024
COMMANDS = {
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
        t.name not in ("SWA", "DSV4_REQUEST_STATE")
        for t in manager(controller).kv_args.state_types
    ):
        raise ValueError("D2D native state component is unsupported")


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
    if (
        getattr(req, "_pre_layerwise_target", None) is not None
        or getattr(req, "_dsv41_remote", None) is not None
    ):
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
    created: float = field(default_factory=time.monotonic)
    remote_closed: object = field(default_factory=threading.Event)
    lock: object = field(default_factory=threading.RLock)

    def initialize(self):
        from .dsv41_staging import buffers

        s, req = self.owner.scheduler, self.req
        self.generation = self.epoch
        self.handle, self.row = req.cache_request_handle, req.kv.req_pool_idx
        self.prefix_len = int(req.kv.kv_committed_len)
        self.decode_prefix_len = 0
        self.schema = self.owner._schema(manager(self.owner))
        self.barrier_ready = True
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
        remote.register(self)
        self.meta = dict(
            version=VERSION,
            epoch=self.epoch,
            destination=self.destination,
            target_rid=self.target_rid,
            origin_input_ids=list(req.origin_input_ids),
            output_ids=list(req.output_ids),
            kv_len=self.prefix_len,
            sampling=self.parameters,
            detokenize={
                name: (list(value) if isinstance(value, array) else value)
                for name in (
                    "decoded_text",
                    "surr_offset",
                    "read_offset",
                    "cur_decode_ids_len",
                    "surr_and_decode_ids",
                )
                for value in (getattr(req, name, None),)
            },
            schema=self.schema,
            tp_size=self.owner.tp_size,
            states=[d for d, t, i in self.states],
            route=self.service,
        )
        self.meta["route"] = self.service

    def check_read(self, *, allow_aborted=False):
        if (
            getattr(self.req, "_remote_attn_transport_quarantine", None) is not None
            or self.error
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
        if self.aborted:
            return self.remote_closed.is_set() and self.release is not None
        if getattr(self.req, "_handoff_transfer_unsafe", False):
            raise TransportFailure("D2D source GPU reader fence failed")
        if not (self.committed and self.copied and self.remote_closed.is_set()):
            return False
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
        source.check_read()
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
        if action != "read":
            raise ValueError("unknown D2D snapshot action")
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
    device = table.device
    with on_device(device), on_stream(device, stream):
        if stream is not None:
            stream.wait_event(source.final_event)
        data = (
            table.index_select(0, indices[lo : lo + count].to(torch.int64))
            .contiguous()
            .cpu()
        )
    return reply(data)


class Reader:
    def __init__(self, route, destination, epoch, timeout=600):
        self.route, self.destination, self.epoch = route, destination, epoch
        self.client = remote.runtime().TargetClient(route["host"], route["port"])
        self.client.attach(route["handoff_id"], route["token"], route["epoch"])
        self.query = bytes(1024)
        self.timeout = timeout
        self.closed = False

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
        response, out, _lse = self.client.call(h, self.query, deadline_s=self.timeout)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "D2D snapshot failed"))
        return response, out.view(torch.uint8)[: response["bytes"]]

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.client.detach(self.route["handoff_id"])
        self.client.close()


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
                # copy-stream fence proves their TCP-only writes, even when a
                # copy error poisoned the ordinary request IO ledger.
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
        from .dsv41_staging import buffers
        from .dsv41_receive import scatter_prefix

        req, pool = self.decode_req.req, self.owner.pool
        try:
            self.bulk_reader = Reader(
                self.reader.route,
                self.reader.destination,
                self.reader.epoch,
                self.reader.timeout,
            )
            device = pool.get_swa_key_buffer_radix(0).device
            self.copy_stream = new_stream(device)
            with on_device(device), on_stream(device, self.copy_stream):
                self.copy_stream.wait_event(self.decode_req.ready)
                pages = (
                    self.owner.scheduler.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx, : self.coverage.prefix_len : pool.page_size
                    ]
                    // pool.page_size
                ).long()
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
                        src = data.reshape(n, -1).to(device, non_blocking=True)
                        scatter_prefix(
                            dst,
                            src,
                            pages[lo : lo + n],
                            lo,
                            self.coverage.prefix_len,
                            pool.page_size,
                            entry[1],
                            pool.get_extra_key_layout(entry[2]),
                        )
                    self.events[i] = dp.event(device)
                self.final_event = dp.event(device)
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
    if cmd == "d2d_status":
        src = sources.get(epoch)
        target = targets.get(epoch)
        return dict(
            migration_epoch=epoch,
            source=None
            if src is None
            else dict(
                committed=src.committed,
                copied=src.copied,
                released=src.released,
                error=src.error,
                resumed=src.released and not src.committed,
                kv_released=not src.req.kv.holds_kv,
            ),
            target=None
            if target is None
            else dict(
                admitted=target.admitted,
                copied=target.copied,
                local=target.closed and not target.cancelled,
                aborted=target.cancelled,
                error=target.error,
            ),
        )
    if cmd == "d2d_freeze":
        if epoch in sources:
            source = sources[epoch]
            if source.released or source.aborted:
                raise ValueError("D2D source epoch is retired")
            if getattr(source, "meta", None) is None or source.error:
                raise RuntimeError(
                    "D2D source preparation failed; inspect status/resume"
                )
            return dict(transfer=source.meta)
        if epoch in targets:
            raise ValueError("migration epoch was already used by a D2D import")
        error = None
        try:
            req = find_req(controller.scheduler, data["rid"])
            destination, target_rid = str(data["destination"]), str(data["target_rid"])
            if not destination or not target_rid:
                raise ValueError("D2D destination and target rid cannot be empty")
            params = sampling(req)
            if getattr(req, "_dsv_d2d_source", None) is not None or not can_quiesce(
                controller.scheduler
            ):
                raise ValueError(
                    "request has a D2D owner or needs native Decode boundary"
                )
        except Exception as exc:
            error = repr(exc)
        if not controller.agree([int(error is None)])[0]:
            raise ValueError(
                f"TP D2D freeze preflight rejected: {error or 'peer rejected'}"
            )
        quiesce(controller.scheduler, req)
        remote.runtime()._remove_from_batches(controller.scheduler, req)
        req._handoff_frozen = True
        req._handoff_req = True
        req._handoff_epoch = epoch
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
        )
        req._dsv_d2d_source = req._pre_layerwise_source = source
        sources[epoch] = source
        controller.sources.append(source)
        try:
            source.initialize()
        except BaseException as exc:
            source.error = repr(exc)
            raise
        remote.runtime()._tracked(controller.scheduler)[req.rid] = req
        return dict(transfer=source.meta)
    if cmd == "d2d_import":
        return import_request(controller, data)
    if cmd == "d2d_abort":
        return abort_import(controller, targets[epoch])
    if cmd == "d2d_commit":
        target = targets[epoch]
        if not target.admitted:
            if target.cancelled or target.closed or target.decode_req.req.finished():
                raise ValueError("cannot commit aborted/finished D2D import")
            target.refresh()
            target.reader.call("commit")
            target.admitted = True
            req = target.decode_req.req
            from .handoff_control import _flush_decode_resume

            _flush_decode_resume(controller.scheduler, [req])
        return dict(admitted=True, rid=target.decode_req.req.rid)
    if cmd == "d2d_release":
        source = sources[epoch]
        return dict(
            released=source.released, committed=source.committed, copied=source.copied
        )
    if cmd == "d2d_resume":
        return resume_source(controller, sources[epoch])
    raise ValueError("unsupported D2D command")


def allocation_preflight(controller, req):
    prealloc = controller.scheduler.disagg_decode_prealloc_queue
    full, swa = prealloc._prealloc_required_tokens(req)
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
    if not fits:
        raise ValueError("D2D target native FULL/SWA/request-row budget unavailable")


def import_request(controller, data):
    from .dsv41_staging import buffers
    from .handoff_control import _select_transfer, _new_req, _tracked
    from .pre_layerwise import LayerCoverage
    from sglang.srt.sampling.sampling_params import SamplingParams

    sources, targets = records(controller)
    epoch = str(data["migration_epoch"])
    if epoch in sources:
        raise ValueError("migration epoch was already used by a D2D source")
    if epoch in targets:
        target = targets[epoch]
        if target.error or target.cancelled:
            raise ValueError("D2D import has failed/aborted; cannot retry allocation")
        return dict(rid=target.decode_req.req.rid, imported=True, duplicate=True)
    transfer = _select_transfer(data, controller.rank, controller.tp_size)
    if (
        transfer["version"] != VERSION
        or transfer["epoch"] != epoch
        or transfer["tp_size"] != controller.tp_size
    ):
        raise ValueError("D2D transfer epoch/TP mismatch")
    if transfer["schema"] != controller._schema(manager(controller)):
        raise ValueError("D2D model/KV/state wire schema mismatch")
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
    for name, value in transfer["detokenize"].items():
        if name == "surr_and_decode_ids" and value is not None:
            value = array("q", value)
        if value is not None:
            setattr(req, name, value)
    target = Target(
        controller,
        None,
        NS(req=req, kv_receiver=None),
        controller.config.directory / "d2d" / epoch / str(controller.rank) / "target",
        epoch,
        LayerCoverage(transfer["kv_len"], controller.layers),
    )
    target.queue = NS(queue=[])
    targets[epoch] = target
    _tracked(controller.scheduler)[rid] = req
    req._pre_layerwise_target = target
    reader = None
    try:
        target.setup_stream = current_stream(
            controller.pool.get_swa_key_buffer_radix(0).device
        )
        prealloc = controller.scheduler.disagg_decode_prealloc_queue
        allocation_preflight(controller, req)
        prealloc._pre_alloc(req)
        target.allocation_complete = True
        if req.kv.kv_committed_len != transfer["kv_len"]:
            raise RuntimeError("native D2D committed length mismatch")
        reader = target.reader = Reader(
            transfer["route"], destination, epoch, controller.config.drain_timeout
        )
        session = (
            dp.plane_for(controller).engine.get_session_id() if dp.enabled() else None
        )
        meta, _ = reader.call("manifest", gpu_session=session)
        if meta["manifest"] != transfer:
            raise ValueError("D2D frozen manifest changed")
        pool = controller.pool
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
        rpc = target.rpc = remote.Target(
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
    can_resume = (
        not source.committed
        and not source.req.finished()
        and not source.released
        and source.release is None
        and getattr(source.req, "_remote_attn_transport_quarantine", None) is None
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
    if getattr(req, "_remote_attn_transport_quarantine", None) is not None:
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
    # All target operations are CPU-framed H2D on these owned streams. No
    # thread can submit another write after join; a CUDA error is not proof.
    for stream in (target.setup_stream, target.copy_stream):
        if stream is not None:
            stream.synchronize()
    if target.reader is not None:
        try:
            target.reader.call("abort")
        except Exception:
            pass  # Source resume independently revokes/fences its lease.
    device = controller.pool.get_swa_key_buffer_radix(0).device
    target.abort_ready = dp.event(device)
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
