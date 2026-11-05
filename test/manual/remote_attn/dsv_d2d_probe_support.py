"""DSV D2D native byte/state fixtures and scheduler/allocator ABI boundaries."""

import ast
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from dsv41_remote_probe_support import definitions, encoded, fixture
from flash0731_probe_support import flash_fixture
from sglang_remote_attn import dsv_d2d as d2d
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import pre_layerwise as pl
from sglang_remote_attn.decode_overlap import DecodeOverlap


class Sampling:
    def __init__(
        self, max_new_tokens=10000, temperature=0.0, top_k=1, stop_token_ids=None
    ):
        self.max_new_tokens, self.temperature, self.top_k = (
            max_new_tokens,
            temperature,
            top_k,
        )
        self.stop_token_ids = stop_token_ids

    def normalize(self, tokenizer):
        self.stop_token_ids = set(self.stop_token_ids or ()) or None


def request(rid, origin, output, params):
    req = NS(
        rid=rid,
        origin_input_ids=list(origin),
        output_ids=list(output),
        sampling_params=params,
        grammar=None,
        return_logprob=False,
        cache_request_handle=(rid, 0),
        lock=None,
        _handoff_frozen=False,
        full_untruncated_fill_ids=list(origin) + list(output),
    )
    req.kv = NS(
        req_pool_idx=None,
        kv_committed_len=0,
        kv_allocated_len=0,
        cache_protected_len=0,
        holds_kv=False,
        is_kv_released=True,
    )
    req.finished = lambda: False
    return req


class Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)

    def filter_batch(self, keep_indices):
        self.reqs = [self.reqs[i] for i in keep_indices]


class Environment:
    def __init__(
        self, root, device="cpu", ratio=2, gpu=False, kv_layout="v4", online_c128=True
    ):
        self.stack = ExitStack()
        self.root = root
        self.device = device
        self.tmp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.f = (
            fixture(root, device, kv_layout=kv_layout)
            if ratio == 2
            else flash_fixture(root, device, ratio)
        )
        f = self.f
        self.ratio = ratio
        self.prefix = f.prefix
        page = f.page
        methods = definitions(
            root,
            "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
            owner="DeepSeekV4TokenToKVPool",
            members={
                "get_state_buf_infos",
                "get_request_state_buf_infos",
                "request_state_transfer_indices",
            },
        )
        carry = definitions(
            root,
            "python/sglang/srt/mem_cache/deepseek_v4_compress_state.py",
            owner="CompressStatePool",
            members={"transfer_indices"},
        )
        path = root / "python/sglang/srt/mem_cache/deepseek_v4_compress_state.py"
        node = next(
            n
            for n in ast.parse(path.read_text()).body
            if getattr(n, "name", None) == "request_scoped_state_transfer_indices"
        )
        ns = {}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(
                        body=[
                            ast.ImportFrom(
                                module="__future__",
                                names=[ast.alias(name="annotations")],
                                level=0,
                            ),
                            node,
                        ],
                        type_ignores=[],
                    )
                ),
                str(path),
                "exec",
            ),
            ns,
        )
        transfer = ns["request_scoped_state_transfer_indices"]
        import numpy as np

        transfer.__globals__["np"] = np
        carry.transfer_indices.__globals__.update(
            np=np, request_scoped_state_transfer_indices=transfer
        )
        methods.get_request_state_buf_infos.__globals__["ONLINE_C128"] = (
            ratio == 128 and online_c128
        )
        self.scores = []
        for p, pool in enumerate((f.p_pool, f.d_pool)):
            pool.sliding_window = page
            pool.swa_page_size = page
            swas = [encoded(f.main_layout, 8, page, device) for _ in range(4)]
            if p:
                swas = [torch.zeros_like(x) for x in swas]
            pool.swa_kv_pool = NS(
                page_size=page, kv_buffer=swas, get_key_buffer=swas.__getitem__
            )
            pool.uniform_fp8 = pool._unified_kv = False
            pool.compress_state_pools = [None] * 4
            pool.indexer_compress_state_pools = [None] * 4
            state = carry()
            state.ratio = ratio
            state.request_scoped = ratio != 4
            state.online = ratio == 128 and online_c128
            state.ring_size = (
                (1 if online_c128 else 128)
                if ratio == 128
                else (2 if ratio == 2 else 8)
            )
            score = torch.full(
                (512, 16), 71 if not p else 0, dtype=torch.float32, device=device
            )
            state.kv_score_buffer = NS(kv_score=score)
            pool.compress_state_pools[2 if ratio == 2 else 3] = state
            if ratio == 4:
                index = carry()
                index.ratio = 4
                index.request_scoped = False
                index.online = False
                index.ring_size = 8
                index.kv_score_buffer = NS(kv_score=score.clone())
                pool.indexer_compress_state_pools[3] = index
            self.scores.append(score)
            for name in (
                "get_state_buf_infos",
                "get_request_state_buf_infos",
                "request_state_transfer_indices",
            ):
                setattr(pool, name, getattr(methods, name).__get__(pool))

            def clear(idx, pool=pool):
                for st in pool.compress_state_pools:
                    if st is not None and st.request_scoped:
                        d2d.matrix(
                            st.kv_score_buffer.kv_score,
                            16 * 4 * (st.ring_size if st.ratio == 2 else 1),
                        )[idx].zero_()

            pool.clear_request_scoped_state = clear
            # Keep an actual indexer byte entry to prove it is copied BEFORE commit.
            idx = torch.full(
                (pool.get_extra_key_buffer(3).shape[0], 8),
                93 if not p else 0,
                dtype=torch.uint8,
                device=device,
            )
            pool.index_pools = {
                ratio: NS(contiguous_page_row_buffers=lambda idx=idx: [idx])
            }
        self.nodes = []
        cpu_buffers = {}
        self.freed = []
        self.resumed = []
        for node_index, pool, row in ((0, f.p_pool, f.p_row), (1, f.d_pool, f.d_row)):
            controller = pl.Controller.__new__(pl.Controller)
            controller.pool, controller.layers, controller.rank, controller.tp_size = (
                pool,
                4,
                0,
                1,
            )
            controller.hf_config = f.p_controller.hf_config
            controller.dsv41_remote = controller.dsv41_layerwise = (
                controller.graph_fast_path
            ) = True
            controller.config = pl.Config("remote", Path(self.tmp) / str(node_index))
            controller.sources, controller.targets, controller.remote_targets = (
                [],
                [],
                {},
            )
            controller.agree = lambda flags: flags
            controller.early_admissions = 0
            scheduler = NS(
                disaggregation_mode=NS(value="decode"),
                tree_cache=NS(disable=True),
                enable_overlap=False,
                tokenizer=None,
                waiting_queue=[],
                last_batch=None,
                result_queue=[],
                running_batch=Batch([]),
                handoff_tracked_reqs={},
                enable_hisparse=False,
            )
            scheduler._pre_layerwise = controller
            controller.scheduler = scheduler
            scheduler.req_to_token_pool = NS(
                req_to_token=row.repeat(2, 1), available_size=lambda: 1
            )
            scheduler.token_to_kv_pool_allocator = NS(
                translate_swa_indices_for_transfer=lambda full, n=node_index: (
                    torch.arange(full.numel(), device=full.device)
                    + (2 if n == 0 else 1) * page
                )
            )

            # Registration mirrors the pool's native global pointer order.
            tensors = []
            entries = []
            for r, kv in pool.kv_pools.items():
                sources = pool.sources_by_ratio[r]
                for layer, tensor in zip(sources, kv.kv_buffer):
                    tensors.append(tensor)
                    entries.append(
                        [
                            "kv",
                            r,
                            layer,
                            pool.get_extra_key_layout(layer).value,
                            str(tensor.dtype),
                            tensor[0].nbytes,
                        ]
                    )
                if r in pool.index_pools:
                    tensor = pool.index_pools[r].contiguous_page_row_buffers()[0]
                    tensors.append(tensor)
                    entries.append(
                        [
                            "index",
                            r,
                            sources[0],
                            "fixture",
                            str(tensor.dtype),
                            tensor[0].nbytes,
                        ]
                    )
            cpu_buffers[id(pool)] = tensors
            args = NS(
                kv_data_ptrs=[t.data_ptr() for t in tensors],
                kv_data_lens=[t.nbytes for t in tensors],
                kv_item_lens=[t[0].nbytes for t in tensors],
                state_types=[NS(name="SWA"), NS(name="DSV4_REQUEST_STATE")],
            )
            mgr = NS(kv_args=args)
            controller._d2d_kv_args = args

            def alloc(req, n=node_index):
                req.kv.req_pool_idx = 1
                req.kv.kv_committed_len = req.kv.kv_allocated_len = (
                    len(req.origin_input_ids) + len(req.output_ids) - 1
                )
                req.kv.holds_kv = True
                req.kv.is_kv_released = False

            scheduler.disagg_decode_prealloc_queue = NS(
                kv_manager=mgr,
                _pre_alloc=alloc,
                _prealloc_required_tokens=lambda req: (self.prefix, page),
                _allocatable_token_budgets=lambda **kw: 100000,
                _uses_swa_reservation=lambda: True,
                _swa_tail_allocatable_token_budget=lambda **kw: 100000,
                _prealloc_reservation_fits=lambda *args, **kw: True,
            )
            schema = dict(
                entries=entries, page_size=page, ratios=list(pool.compression_ratios)
            )
            controller._schema = lambda m, schema=schema: schema
            controller.decode_overlap = DecodeOverlap(controller)
            self.stack.callback(controller.decode_overlap.retirement.close)
            self.nodes.append(controller)
        if device == "cpu":
            self.stack.enter_context(
                patch(
                    "sglang_remote_attn.dsv41_staging.buffers",
                    side_effect=lambda pool, manager: cpu_buffers[id(pool)],
                )
            )
        self.source, self.target = self.nodes
        # Route actual native sparse metadata to the destination request row.
        f.controller = self.target
        f.backend._dsv41_remote = self.target
        f.fb.req_pool_indices.fill_(1)
        f.fb.req_pool_indices_cpu = [1]
        f.core.swa_page_indices = torch.tensor(
            [[page + 5, page + 6, -1]], device=device, dtype=torch.int32
        )
        hub = rt.RemoteHub(
            self.source.scheduler,
            stream_factory=lambda n: [
                (torch.cuda.Stream(), None, None)
                if device == "cuda"
                else (None, None, None)
                for _ in range(n)
            ],
        )
        self.hub = hub
        self.source.scheduler._handoff_hub = hub
        self.stack.callback(hub.close)
        self.stack.enter_context(
            patch.dict(
                __import__("sys").modules,
                {"sglang.srt.sampling.sampling_params": NS(SamplingParams=Sampling)},
            )
        )
        self.stack.enter_context(
            patch.object(
                rt,
                "_new_req",
                side_effect=lambda s, r, o, out, sp, **kw: request(r, o, out, sp),
            )
        )
        self.stack.enter_context(
            patch.object(
                rt,
                "_flush_decode_resume",
                side_effect=lambda s, reqs: self.resumed.extend(
                    (s, list(reqs)) for _ in [0]
                ),
            )
        )

        def free(req, tree, *, checkpoint):
            def apply():
                self.freed.append(req.rid)
                req.kv.holds_kv = False
                req.kv.is_kv_released = True
                req.kv.req_pool_idx = None

            if not pl.defer_release(req, apply, checkpoint=False):
                apply()

        self.stack.enter_context(
            patch.dict(
                __import__("sys").modules,
                {"sglang.srt.mem_cache.common": NS(release_kv_cache=free)},
            )
        )
        self.req = request(
            "source",
            list(range(self.prefix - 2)),
            [31, 32, 33],
            Sampling(stop_token_ids={999}),
        )
        self.req.kv = NS(
            req_pool_idx=0,
            kv_committed_len=self.prefix,
            kv_allocated_len=self.prefix,
            cache_protected_len=0,
            holds_kv=True,
            is_kv_released=False,
        )
        self.source.scheduler.running_batch = Batch([self.req])
        self.source.scheduler.handoff_tracked_reqs[self.req.rid] = self.req
        self.gate, self.blocked = threading.Event(), threading.Event()
        actual = d2d.Reader.call

        def read(reader, action, **kw):
            if (
                action == "read"
                and kw.get("kind") == "global"
                and threading.current_thread().name == "dsv-d2d-copy"
            ):
                self.blocked.set()
                if not self.gate.wait(30):
                    raise TimeoutError("D2D test transfer gate")
            return actual(reader, action, **kw)

        self.stack.enter_context(patch.object(d2d.Reader, "call", read))
        self.stack.callback(self.gate.set)
        self.transport = None
        if device == "cuda":
            from remote_gpu_probe_support import Transport

            self.transport = Transport(root, allow_tcp_snapshot=True)
            for controller in self.nodes:
                prealloc = controller.scheduler.disagg_decode_prealloc_queue
                self.transport.attach(controller, "decode")
                prealloc.kv_manager = controller.gpu_data_plane.manager
                controller.scheduler.disagg_decode_prealloc_queue = prealloc
                prealloc.kv_manager.kv_args = controller._d2d_kv_args
            if not gpu:
                self.stack.enter_context(
                    patch.dict(
                        __import__("os").environ, {"HANDOFF_REMOTE_GPU_TRANSFER": "0"}
                    )
                )
        else:
            # Real registered byte copies at the native transport boundary.
            # CUDA probes use the actual Mooncake engine and pinned CUDA slot.
            import ctypes

            from sglang_remote_attn import d2d_bulk

            engines = {}

            class Slot:
                def __init__(self, capacity, device):
                    self.host = torch.empty(capacity, dtype=torch.uint8)
                    self.stream = NS(synchronize=lambda: None)

                def copy(self, blocks, ready):
                    if ready is not None:
                        ready.synchronize()
                    offset = 0
                    for src, _, size in blocks:
                        ctypes.memmove(self.host.data_ptr() + offset, src, size)
                        offset += size

                def close(self):
                    pass

            self.stack.enter_context(patch.object(d2d_bulk, "new_slot", Slot))
            for n, controller in enumerate(self.nodes):
                engine = NS(regions={}, calls=[], session=f"cpu-d2d-{n}")
                engines[engine.session] = engine

                def register(ptrs, sizes, engine=engine):
                    engine.regions.update(zip(ptrs, sizes))
                    return 0

                def deregister(ptrs, engine=engine):
                    for ptr in ptrs:
                        del engine.regions[ptr]
                    return 0

                def transfer(session, sources, targets, sizes, engine=engine):
                    peer = engines[session]
                    for src, dst, size in zip(sources, targets, sizes):
                        assert engine.regions[src] >= size
                        assert peer.regions[dst] >= size
                        ctypes.memmove(dst, src, size)
                    engine.calls.append((session, list(sizes)))
                    return 0

                engine.batch_register = register
                engine.batch_deregister = deregister
                engine.batch_transfer_sync = transfer
                engine.get_session_id = lambda engine=engine: engine.session
                controller.scheduler.disagg_decode_prealloc_queue.kv_manager.engine = (
                    engine
                )
            self.stack.enter_context(
                patch.dict(
                    __import__("os").environ, {"HANDOFF_REMOTE_GPU_TRANSFER": "0"}
                )
            )

    def freeze(self, epoch="migration"):
        self.epoch = epoch
        result = d2d.handle(
            self.source,
            dict(
                cmd="d2d_freeze",
                rid="source",
                migration_epoch=epoch,
                destination="D1",
                target_rid="target",
            ),
        )
        return result["transfer"]

    def import_(self, transfer):
        return d2d.handle(
            self.target,
            dict(
                cmd="d2d_import",
                migration_epoch=self.epoch,
                destination="D1",
                transfer=transfer,
            ),
        )

    def command(self, controller, cmd):
        return d2d.handle(controller, dict(cmd=cmd, migration_epoch=self.epoch))

    def wait_copy(self):
        target = self.target.d2d_targets[self.epoch]
        target.worker.join(20)
        assert not target.worker.is_alive() and target.error is None, target.error
        return target

    def close(self):
        self.gate.set()
        for controller in self.nodes:
            for t in getattr(controller, "d2d_targets", {}).values():
                if t.worker is not None:
                    t.worker.join(5)
                if t.reader is not None:
                    t.reader.close()
            if hasattr(controller, "dsv41_rpc_batches"):
                controller.dsv41_rpc_batches.close()
        # No readers or producers may outlive test teardown. Successful slots
        # are deregistered; uncertain ones deliberately remain engine-owned.
        if self.transport is not None:
            self.transport.close()
        self.hub.close()
        for controller in self.nodes:
            engine = d2d.manager(controller).engine
            for slot in list(getattr(engine, "_d2d_bulk_owners", ())):
                if not slot.unsafe:
                    slot.close()
        self.stack.close()
