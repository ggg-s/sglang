"""Hybrid PD ownership probe: native pools/state ids, real range/RPC lifecycle.

Native sender queue and metadata commit are ABI boundaries. CUDA uses real
Mooncake TCP copies and FlashInfer Remote attention, not model weights.
"""

from dataclasses import replace
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from qwen_hybrid_probe_support import HybridEnvironment
from sglang_remote_attn import dsv_d2d as d2d, prefix_stream, qwen_hybrid as hybrid
from sglang_remote_attn import qwen_hybrid_pd as pd
from sglang_remote_attn.pre_layerwise import read_json, send_chunk


class Environment(HybridEnvironment):
    def __init__(self, root, device="cpu"):
        super().__init__(root, device)
        self.jobs, self.state_copies, self.canonical = [], 0, False
        self.allow_kv = threading.Event()
        self.allow_state = threading.Event()
        self.allow_state.set()
        self.workers = []
        p, d = self.source, self.target
        self.pm, self.dm = d2d.manager(p), d2d.manager(d)
        p.scheduler.disaggregation_mode = NS(value="prefill")
        p.scheduler.disagg_prefill_bootstrap_queue = NS(kv_manager=self.pm)
        self.stack.enter_context(
            patch.dict(__import__("os").environ, {"HANDOFF_REMOTE_GPU_TRANSFER": "1"})
        )
        self.pm.kv_args.page_size = self.dm.kv_args.page_size = 1
        self.pm.get_session_id = self.pm.engine.get_session_id
        self.dm.get_session_id = self.dm.engine.get_session_id
        for c in (p, d):
            c.schema_signature = c._schema(d2d.manager(c))
            c.cache_guard = None
            c.plan = None
            c.pd_visibility = d2d.dp.Visibility(torch.device(device), protocol="tcp")
            c.config = replace(
                c.config, directory=__import__("pathlib").Path(self.tmp) / "hybrid-pd"
            )
        self.prefix = 8
        p.remote_min_cache_hit_ratio = hybrid.Controller.remote_min_cache_hit_ratio
        self.req.origin_input_ids = list(range(8))
        self.req._hybrid_prefill_cached_tokens = 6  # First Prefill admission.
        self.req.output_ids = __import__("array").array("q", [31])
        self.req.extend_end = 8
        self.req.kv.kv_committed_len = 8
        self.req.pending_bootstrap, self.req.multimodal_inputs = False, None
        self.req.input_embeds = self.req.session_id = None
        self.req.start_send_idx, self.req.disagg_decode_prefix_len = 0, 0
        self.req.bootstrap_room = 1977
        self.dr_req = self.native[1].make(
            d.scheduler, "pd-target", list(range(8)), [], self.req.sampling_params
        )
        self.native[1].queue._pre_alloc(self.dr_req)
        self.dr_req.bootstrap_room = self.req.bootstrap_room
        self.dr_req.kv.mamba_needs_clear = True
        from dsv41_remote_probe_support import definitions

        KVPoll = definitions(
            root, "python/sglang/srt/disaggregation/base/conn.py", owner="KVPoll"
        )
        self.stack.enter_context(
            patch.dict(
                __import__("sys").modules,
                {"sglang.srt.disaggregation.base.conn": NS(KVPoll=KVPoll)},
            )
        )

        self.poll = KVPoll
        self.sender_status = KVPoll.Transferring
        self.receiver = NS(
            kv_mgr=self.dm, poll=lambda: self.sender_status, clear=lambda: None
        )
        self.dr = NS(
            req=self.dr_req,
            kv_receiver=self.receiver,
            is_rebootstrap=False,
            metadata_buffer_index=0,
        )
        self.queue = NS(
            scheduler=d.scheduler,
            queue=[self.dr],
            _is_host_destination=lambda dr: False,
            metadata_buffers=NS(bootstrap_room=torch.zeros((1, 1), dtype=torch.int64)),
            req_to_metadata_buffer_idx_allocator=NS(free=lambda i: None),
        )

        def commit(dr, *, keep_receiver):
            assert keep_receiver
            dr.req.output_ids.append(31)

        self.queue._commit_transfer_to_req = commit
        d.scheduler.disagg_decode_transfer_queue = self.queue
        peer = NS(mooncake_session_id=self.dm.engine.get_session_id(), is_dummy=False)
        reg = NS(
            dst_kv_ptrs=self.dm.kv_args.kv_data_ptrs,
            dst_aux_ptrs=[1],
            dst_aux_item_lens=[8],
            requires_dcp_relayout=False,
        )
        self.pm.transfer_infos = {
            self.req.bootstrap_room: {peer.mooncake_session_id: peer}
        }
        self.pm.decode_kv_args_table = {peer.mooncake_session_id: reg}
        self.sender = self.req.disagg_kv_sender = NS(
            kv_mgr=self.pm,
            bootstrap_room=self.req.bootstrap_room,
            poll=lambda: self.sender_status,
        )
        self.pm._poison_unsettled_send = lambda room: setattr(
            self, "sender_status", KVPoll.Failed
        )
        self.pm.conclude_failure = lambda **kw: setattr(
            self, "sender_status", KVPoll.Failed
        )
        self.pm.maybe_send_extra = self.send_state
        self.pm.send_aux = self.send_aux
        # Native receiver type/host-destination discovery is the ABI boundary.
        self.stack.enter_context(
            patch.dict(
                __import__("sys").modules,
                {
                    "sglang.srt.disaggregation.mooncake.conn": NS(
                        MooncakeKVReceiver=type(self.receiver)
                    )
                },
            )
        )
        d.offer = lambda queue, dr: pd.offer(d, queue, dr)
        pd.offer(d, self.queue, self.dr)
        self.pointers = []
        if device == "cuda":
            for c in (p, d):
                pointers = c.pool.get_contiguous_buf_infos()
                states = c.pool.get_state_buf_infos()
                engine = d2d.manager(c).engine
                ptrs, sizes = [*pointers[0], *states[0]], [*pointers[1], *states[1]]
                assert engine.batch_register(ptrs, sizes) == 0
                self.pointers.append((engine, ptrs))

    def copy(self, src, dst, si, di):
        if self.device == "cpu":
            dst[di] = src[si]
        else:
            stride = src[0].nbytes
            assert (
                d2d.transport.handle(self.pm.engine).batch_transfer_sync(
                    self.dm.engine.get_session_id(),
                    [src.data_ptr() + int(i) * stride for i in si],
                    [dst.data_ptr() + int(i) * stride for i in di],
                    [stride] * len(si),
                )
                == 0
            )

    def send_state(self, peer, indices, executor, registration):
        assert self.allow_state.wait(10)
        for (_, a, ai), (_, b, bi) in zip(
            hybrid.state_rows(self.source, self.req, 8),
            hybrid.state_rows(self.target, self.dr_req, 8),
        ):
            self.copy(a, b, ai.cpu().tolist(), bi.cpu().tolist())
        self.state_copies += 1
        return 0

    def send_aux(self, *args):
        self.queue.metadata_buffers.bootstrap_room[0, 0] = self.req.bootstrap_room
        return 0

    def native_send(self, manager, room, indices, span, last, **kwargs):
        if last:
            assert not len(indices) and kwargs.get("state_indices") == []
            self.canonical = True
            self.sender_status = self.poll.Success
            return

        def work():
            try:
                assert self.allow_kv.wait(120)
                event = kwargs.get("wait_event")
                if event is not None:
                    event.synchronize()
                token_ids = (
                    self.target.scheduler.req_to_token_pool.req_to_token[
                        self.dr_req.kv.req_pool_idx, span.start : span.stop
                    ]
                    .cpu()
                    .tolist()
                )
                for i in kwargs["kv_buffer_indices"]:
                    self.copy(
                        hybrid.kv_buffers(self.source.pool)[i],
                        hybrid.kv_buffers(self.target.pool)[i],
                        list(indices),
                        token_ids,
                    )
                kwargs["on_complete"]()
            except BaseException as exc:
                self.jobs.append(repr(exc))

        worker = threading.Thread(target=work)
        self.workers.append(worker)
        worker.start()

    def start(self, forward=None):
        pd.prepare(self.source, [self.req])
        self.source_actor = self.req._pre_layerwise_source
        self.source.plan = [self.source_actor]
        hybrid.Controller.forward(
            self.source,
            forward or (lambda runner, fb: None),
            self.source.scheduler.tp_worker.model_runner,
            NS(input_embeds=None, replace_embeds=None),
        )
        self.source.plan = None
        indices = (
            self.source.scheduler.req_to_token_pool.req_to_token[
                self.req.kv.req_pool_idx, :8
            ]
            .cpu()
            .numpy()
        )
        state = hybrid.state_rows(self.source, self.req, 8)[0][2].cpu().numpy()

        def native_final(scheduler, req, *, last_chunk, end_idx):
            assert last_chunk
            prefix_stream.track(
                self.native_send,
                self.pm,
                req.bootstrap_room,
                indices,
                slice(0, 8),
                True,
                aux_index=0,
                state_indices=[state],
            )

        send_chunk(native_final, self.source.scheduler, self.req, last_chunk=True)

    def wait_barrier(self):
        self.source_actor.thread.join(10)
        assert (
            not self.source_actor.thread.is_alive() and not self.source_actor.error
        ), self.source_actor.error
        assert read_json(self.source_actor.path / "ready.json")["hybrid_state_ready"]

    def admit(self):
        admitted, _ = pd.early_targets(self.target, self.queue, None)
        assert admitted == [self.dr_req] and not self.queue.queue
        return self.dr_req._pre_layerwise_target

    def local(self, parent):
        self.allow_kv.set()
        # Completion callbacks can start additional bounded native tasks.
        deadline = time.monotonic() + 10
        while not self.canonical:
            assert time.monotonic() < deadline, self.jobs
            time.sleep(0.001)
        assert not self.jobs
        self.target.drain_targets(required=parent)
        assert parent.closed
        from sglang.srt.mem_cache.common import release_kv_cache

        self.req.finished = lambda: True
        release_kv_cache(self.req, self.source.scheduler.tree_cache, checkpoint=False)
        deadline = time.monotonic() + 5
        while not self.source_actor.poll():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        self.source_actor.retire()
        self.source.sources.remove(self.source_actor)
        assert not self.req.kv.holds_kv and not self.req.kv.holds_mamba
        return parent.rpc.calls

    def close(self):
        self.allow_kv.set()
        self.allow_state.set()
        for worker in self.workers:
            worker.join(10)
        if hasattr(self, "source_actor") and self.source_actor.thread is not None:
            self.source_actor.thread.join(10)
        for engine, ptrs in getattr(self, "pointers", ()):
            engine.batch_deregister(ptrs)
        super().close()
