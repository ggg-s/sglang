"""Shared-worker partial PD -> D2D fixtures, with real retirement and byte copies.

Model sampling and native allocation are explicit ABI boundaries. CUDA runs use
native pinned receive rings, H2D scatter/events and Mooncake TCP D2D transfers.
"""

import importlib
import logging
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from dsv41_remote_probe_support import definitions, encoded
from dsv_d2d_probe_support import Batch, Environment, Sampling, request
from sglang_remote_attn import dsv41_receive as rx
from sglang_remote_attn import dsv41_remote as remote
from sglang_remote_attn import dsv41_staging as staging
from sglang_remote_attn import dsv_d2d as d2d
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn.dsv41_progressive import ProgressiveTarget, Resource
from sglang_remote_attn.pre_layerwise import LayerCoverage


class ImmediateEvent:
    def record(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


class SharedEnvironment(Environment):
    def __init__(self, root, concurrency=4, device="cpu"):
        if device == "cuda":
            # Load native modules before Environment's module-boundary patches;
            # their transitive imports must survive repeated CUDA fixtures.
            for name in (
                "base.conn",
                "common.staging_buffer",
                "common.staging_handler",
            ):
                importlib.import_module("sglang.srt.disaggregation." + name)
        super().__init__(root, device)
        self.concurrency = concurrency
        self.requests, self.parents, self.saved = [], [], []
        self.metadata_freed, self.receivers_cleared = [], []
        self.target_rows = set()
        page, width = self.f.page, 128
        capacity = concurrency * (width // page) + 1
        self.buffers = []
        for n, c in enumerate(self.nodes):
            mgr = d2d.manager(c)
            old = staging.buffers(c.pool, mgr)
            resized = []
            schema = c._schema(mgr)
            for tensor, entry in zip(old, schema["entries"]):
                if entry[0] == "kv":
                    expanded = encoded(
                        c.pool.get_extra_key_layout(entry[2]),
                        capacity,
                        page // entry[1],
                        device,
                    )
                else:
                    expanded = (
                        torch.arange(capacity * tensor.shape[1], device=device).reshape(
                            capacity, -1
                        )
                        % 251
                    ).to(tensor.dtype)
                if n:
                    expanded.zero_()
                resized.append(expanded)
            # Preserve the CPU fixture's registered list identity.
            old[:] = resized
            self.buffers.append(old)
            schema = c._schema(mgr)
            for entry, tensor in zip(schema["entries"], resized):
                kind, ratio, layer = entry[:3]
                if kind == "kv":
                    kv = c.pool.kv_pools[ratio]
                    kv.kv_buffer[:] = [tensor]
                    kv.get_key_buffer = lambda i, tensor=tensor: tensor
                else:
                    c.pool.index_pools[ratio].contiguous_page_row_buffers = (
                        lambda tensor=tensor: [tensor]
                    )
            for i, tensor in enumerate(c.pool.swa_kv_pool.kv_buffer):
                c.pool.swa_kv_pool.kv_buffer[i] = encoded(
                    self.f.main_layout, capacity, page, device
                )
                if n:
                    c.pool.swa_kv_pool.kv_buffer[i].zero_()
            if not n:
                for state in c.pool.compress_state_pools:
                    if state is not None:
                        score = state.kv_score_buffer.kv_score
                        score.copy_(
                            torch.arange(score.numel(), device=device).reshape(
                                score.shape
                            )
                        )
            mgr.kv_args.kv_data_ptrs = [t.data_ptr() for t in resized]
            mgr.kv_args.kv_data_lens = [t.nbytes for t in resized]
            mgr.kv_args.page_size = page
            c.scheduler.req_to_token_pool.req_to_token = torch.arange(
                page, page + concurrency * width, device=device, dtype=torch.int32
            ).reshape(concurrency, width)
            c.scheduler.req_to_token_pool.available_size = lambda: (
                concurrency - len(self.target_rows)
            )
            c.scheduler.token_to_kv_pool_allocator.translate_swa_indices_for_transfer = (
                lambda full: full
            )
            c.scheduler.handoff_tracked_reqs.clear()
            c.scheduler.running_batch = Batch([])
        for i in range(concurrency):
            req = request(f"source-{i}", list(range(31)), [31, 32, 33], Sampling())
            req.bootstrap_room = 1000 + i
            req.kv = NS(
                req_pool_idx=i,
                kv_committed_len=33,
                kv_allocated_len=33,
                cache_protected_len=0,
                holds_kv=True,
                is_kv_released=False,
            )
            self.requests.append(req)
            self.source.scheduler.handoff_tracked_reqs[req.rid] = req
        self.source.scheduler.running_batch.reqs[:] = self.requests
        self.req = self.requests[0]

        def alloc(req, **kwargs):
            row = next(i for i in range(concurrency) if i not in self.target_rows)
            self.target_rows.add(row)
            req.kv.req_pool_idx = row
            req.kv.kv_committed_len = req.kv.kv_allocated_len = (
                len(req.origin_input_ids) + len(req.output_ids) - 1
            )
            req.kv.holds_kv, req.kv.is_kv_released = True, False

        self.target.scheduler.disagg_decode_prealloc_queue._pre_alloc = alloc

        def resume(scheduler, reqs):
            self.resumed.append((scheduler, list(reqs)))
            for req in reqs:
                assert req not in scheduler.running_batch.reqs
                scheduler.running_batch.reqs.append(req)

        self.stack.enter_context(patch.object(rt, "_flush_decode_resume", resume))
        self._incoming(root, device)

    def _incoming(self, root, device):
        if device == "cuda":
            from sglang.srt.disaggregation.base.conn import KVPoll
            from sglang.srt import utils
        else:
            KVPoll = definitions(
                root, "python/sglang/srt/disaggregation/base/conn.py", owner="KVPoll"
            )
            self.stack.enter_context(
                patch.dict(
                    sys.modules,
                    {"sglang.srt.disaggregation.base.conn": NS(KVPoll=KVPoll)},
                )
            )
            utils = sys.modules.get("sglang.srt.utils")
            if utils is None:
                utils = NS()
                self.stack.enter_context(
                    patch.dict(sys.modules, {"sglang.srt.utils": utils})
                )
        self.KVPoll = KVPoll
        if device == "cuda":
            from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
            from sglang.srt.disaggregation.common.staging_handler import (
                DecodeStagingHandler,
            )

            self.stream = torch.cuda.current_stream()
        else:
            StagingAllocator = definitions(
                root,
                "python/sglang/srt/disaggregation/common/staging_buffer.py",
                owner="StagingAllocator",
            )

            def cpu_buffer(size, *args, **kwargs):
                host = torch.empty(size, dtype=torch.uint8)
                return NS(
                    buffer=host,
                    gpu_buffer=host,
                    data_ptr=host.data_ptr(),
                    device=host.device,
                )

            StagingAllocator.__init__.__globals__.update(
                StagingBuffer=cpu_buffer,
                threading=threading,
                logger=logging.getLogger(__name__),
                _device_ptr_type_str=lambda *a: "",
            )
            DecodeStagingHandler = definitions(
                root,
                "python/sglang/srt/disaggregation/common/staging_handler.py",
                owner="DecodeStagingHandler",
            )
            self.stream = NS(synchronize=lambda: None, wait_event=lambda e: None)
            DecodeStagingHandler.__init__.__globals__.update(
                threading=threading,
                time=time,
                deque=deque,
                logger=logging.getLogger(__name__),
                create_device_event=lambda *a: ImmediateEvent(),
            )
            self.stack.enter_context(
                patch.dict(
                    sys.modules,
                    {
                        "sglang.srt.disaggregation.common.staging_buffer": NS(
                            StagingAllocator=StagingAllocator
                        )
                    },
                )
            )
            self.stack.enter_context(
                patch.object(
                    utils, "create_device_stream", lambda *a: self.stream, create=True
                )
            )
            self.stack.enter_context(
                patch.object(
                    utils,
                    "device_stream_context",
                    lambda *a: nullcontext(),
                    create=True,
                )
            )
        mgr, c = d2d.manager(self.source), self.source
        mgr.attn_tp_size, mgr.pp_size = 1, 1
        mgr._dsv41_layerwise_schema = c._schema(mgr)
        mgr._dsv41_progressive_pool = c.pool
        c.scheduler.schedule_stream = self.stream
        allocator = StagingAllocator(
            2 * 1024 * 1024, device, 0, pin_memory=device == "cuda"
        )
        mgr._staging_ctx = NS(allocator=allocator, room_receivers={}, room_bootstrap={})
        info = dict(k_buffers=self.buffers[0], v_buffers=[], page_size=self.f.page)
        # CPU stands in only for device stream/event construction.
        ctor_info = (
            info
            if device == "cuda"
            else dict(k_buffers=[NS(device=torch.device("cpu:0"))])
        )
        h = DecodeStagingHandler(mgr, allocator, ctor_info, 1, 1, 0, c.scheduler)
        h.kv_buffer_info = info
        h._defer_scatter = True
        h._scatter_region = lambda *args, **kw: rx.scatter(None, h, *args, **kw)
        self.handler, self.allocator = h, allocator
        q = NS(
            staging_handler=h,
            scheduler=c.scheduler,
            metadata_buffers=NS(
                bootstrap_room=torch.ones((self.concurrency, 1), dtype=torch.int64)
            ),
            req_to_metadata_buffer_idx_allocator=NS(free=self.metadata_freed.append),
        )
        self.queue = q
        for i, req in enumerate(self.requests):
            receiver = NS(
                prefill_info=NS(attn_tp_size=1, pp_size=1),
                kv_mgr=mgr,
                bootstrap_infos=[],
                chunk_staging_infos=[],
                chunk_staging_layers={},
                status=KVPoll.Transferring,
            )
            receiver.poll = lambda r=receiver: r.status
            receiver.clear = lambda i=i: self.receivers_cleared.append(i)
            dr = NS(req=req, kv_receiver=receiver, metadata_buffer_index=i)
            path = c.config.directory / "native-pd" / str(i)
            rpc = remote.Target(c, req, path, f"pd-{i}", {}, len(req.origin_input_ids))
            rpc.progressive = True
            rpc.local_entries.add(0)
            target = ProgressiveTarget(
                c,
                q,
                dr,
                path,
                f"pd-{i}",
                LayerCoverage(31, 4),
                native_staging=True,
                rpc=rpc,
            )
            req._pre_layerwise_target, req._dsv41_remote = target, rpc
            c.remote_targets[i] = rpc
            protocol = c.decode_overlap.retirement
            for entry in (1,):
                r = Resource(target, entry, protocol.last_admission + 1)
                target.resources.append(r)
                c.targets.append(r)
                protocol.register(r)
            target.admission_id = protocol.last_admission + 1
            c.targets.append(target)
            protocol.register(target)
            self.parents.append(target)
            h.register_decode_req(req.bootstrap_room, dr)
            mgr._staging_ctx.room_receivers[req.bootstrap_room] = receiver
            # One global entry is still P-owned. Poison only its immutable
            # compressed prefix, retaining D's partial-page tail and padding.
            tensor = self.buffers[0][1]
            ids = c.scheduler.req_to_token_pool.req_to_token[i, :32:8].long() // 8
            payload = tensor[ids].clone()
            rx.scatter_prefix(
                tensor,
                torch.zeros_like(payload),
                ids,
                0,
                31,
                8,
                2,
                c.pool.get_extra_key_layout(2),
            )
            assert payload.any() and not torch.equal(tensor[ids], payload)
            self.saved.append((ids, payload))
            size = payload.nbytes
            aid, off, rnd = allocator.assign(size)
            receiver.chunk_staging_infos.append((aid, off, rnd, off + size, 4))
            receiver.chunk_staging_layers[0] = ((1,), 0, 4)
            allocator.buffer.buffer[off : off + size].copy_(payload.flatten().cpu())

    def plan_freeze(self, index):
        req = self.requests[index]
        common = dict(
            migration_epoch=f"m-{index}", destination="D1", target_rid=f"target-{index}"
        )
        plan = d2d.handle(
            self.source, dict(cmd="d2d_plan", rid=req.rid, growth_tokens=0, **common)
        )["plan"]
        ticket = d2d.handle(self.target, dict(cmd="d2d_reserve", plan=plan, **common))
        data = dict(
            cmd="d2d_freeze", rid=req.rid, reservation=ticket["reservation"], **common
        )
        result = d2d.handle(self.source, data)
        return data, result

    def arrive(self, index, success=True):
        t = self.parents[index]
        room = t.decode_req.req.bootstrap_room
        assert self.handler.handle_chunk_arrived(room, 0, 0, 4, "writer")
        if success:
            t.decode_req.kv_receiver.status = self.KVPoll.Success
            self.handler.submit_last_scatter_async(room)

    def pump(self):
        self.handler.flush_pending_scatters(self.stream)
        p = self.source.decode_overlap.retirement
        p.observe()
        command = p.poll_command()
        if command:
            p.arm(command)
            p.finalize()

    def complete_pd(self):
        deadline = time.monotonic() + 10
        while not all(t.closed for t in self.parents):
            assert time.monotonic() < deadline, "PD retirement stalled"
            self.pump()
            time.sleep(0.001)


def exercise_shared(env):
    """Freeze c live partial-Remote requests on ONE worker; migrate every row."""
    e = env
    commands, sources = [], []
    for i, req in enumerate(e.requests):
        assert not req._dsv41_remote.all_local()
        data, reply = e.plan_freeze(i)
        commands.append(data)
        source = e.source.d2d_sources[data["migration_epoch"]]
        sources.append(source)
        assert reply["pending_retirement"] and reply["source_partial_remote"]
        assert source.service is None and source.bulk_sender is None
        assert not source.previous_target.closed and not source.previous_rpc.closed
        assert req._handoff_frozen and req.kv.holds_kv
        assert req not in e.source.scheduler.running_batch.reqs
        assert len(e.source.scheduler.running_batch.reqs) == e.concurrency - i - 1
        assert d2d.handle(e.source, data) == reply
        assert not source.poll() and not e.freed
    assert not e.source.scheduler.running_batch.reqs
    # Canonical Success alone must not retire an allocated, unarrived chunk.
    for t in e.parents:
        t.decode_req.kv_receiver.status = e.KVPoll.Success
        e.handler.submit_last_scatter_async(t.decode_req.req.bootstrap_room)
    e.pump()
    assert not any(t.closed for t in e.parents)
    assert all(d2d.handle(e.source, data)["pending_retirement"] for data in commands)
    assert e.allocator.allocations and not e.metadata_freed and not e.freed
    for i in range(e.concurrency):
        e.arrive(i)
    e.complete_pd()
    assert not e.allocator.allocations
    assert sorted(e.metadata_freed) == list(range(e.concurrency))
    assert sorted(e.receivers_cleared) == list(range(e.concurrency))
    assert not e.source.remote_targets and not e.freed
    for i, (ids, payload) in enumerate(e.saved):
        # Restoring immutable P rows cannot overwrite D's last partial page.
        assert torch.equal(e.buffers[0][1][ids], payload)
    for data, source in zip(commands, sources):
        result = d2d.handle(e.source, data)
        assert result["transfer"]["source_partial_remote"]
        assert result["transfer"]["output_ids"] == list(source.frozen_output)
        d2d.handle(
            e.target,
            dict(
                cmd="d2d_import",
                migration_epoch=source.epoch,
                destination="D1",
                transfer=result["transfer"],
            ),
        )
    assert e.blocked.wait(5)
    targets = [e.target.d2d_targets[s.epoch] for s in sources]
    assert len({t.decode_req.req.kv.req_pool_idx for t in targets}) == e.concurrency
    for source, target in zip(sources, targets):
        assert not target.copied and not source.poll()
        for (_, a, ai), (_, b, bi) in zip(
            d2d.state_rows(e.source, source.req, source.prefix_len),
            d2d.state_rows(e.target, target.decode_req.req, source.prefix_len),
        ):
            assert torch.equal(a[ai.long()], b[bi.long()])
        d2d.handle(e.target, dict(cmd="d2d_commit", migration_epoch=source.epoch))
        assert source.committed and not target.copied
    assert len(e.target.scheduler.running_batch.reqs) == e.concurrency
    assert not e.freed
    e.gate.set()
    for target in targets:
        target.worker.join(15)
        assert not target.worker.is_alive() and not target.error, target.error
        assert target.copied
        e.target.drain_targets(required=target)
    deadline = time.monotonic() + 10
    while not all(s.poll() for s in sources):
        assert time.monotonic() < deadline, "source reader lease stalled"
        time.sleep(0.001)
    assert not e.freed
    for source, target in zip(sources, targets):
        assert target.closed and source.remote_closed.is_set()
        req = target.decode_req.req
        assert req.output_ids == list(source.frozen_output)
        assert req.kv.kv_committed_len == source.prefix_len and req.kv.holds_kv
        assert req._dsv41_remote is None
        for tensor, entry, dst in zip(
            e.buffers[0], source.schema["entries"], e.buffers[1]
        ):
            # All complete native pages are copied exactly, including indexer.
            pages = source.prefix_len // e.f.page
            assert torch.equal(
                tensor[source.pages[:pages].long()],
                dst[
                    (
                        e.target.scheduler.req_to_token_pool.req_to_token[
                            req.kv.req_pool_idx, : pages * e.f.page : e.f.page
                        ]
                        // e.f.page
                    ).long()
                ],
            )
        source.retire()
        source.retire()
        assert e.freed.count(source.req.rid) == 1
        assert not source.req.kv.holds_kv and req.kv.holds_kv
    return dict(
        concurrency=e.concurrency,
        shared_workers=1,
        partial_remote_frozen=len(sources),
        migrations=len(targets),
        pending_chunk_blocks_export=True,
        native_pd_retired_before_export=True,
        distinct_destination_rows=True,
        state_bytes_exact=True,
        source_frees=len(e.freed),
        destination_admissions=len(e.resumed),
        token_history_exact=True,
        mooncake_transfers=0 if e.transport is None else len(e.transport.calls),
    )
