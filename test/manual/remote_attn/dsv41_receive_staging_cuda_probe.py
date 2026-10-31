"""Real DSV4.1 byte plans, CUDA H2D/scatter and Mooncake TCP on one GPU.

Small pools/control sockets are fixtures; native ring, watermark, event lifetime
and data transport are real. No model/server/NCCL or latency claim.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch
from dsv41_host_staging_cuda_probe import extract
from sglang_remote_attn import dsv41_receive as rx
from sglang_remote_attn import dsv41_staging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.sglang_root.resolve()
    native = root / "python/sglang/srt"
    os.environ["MC_FORCE_TCP"] = "1"
    torch.cuda.set_device(0)
    logger = logging.getLogger(__name__)
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.disaggregation",
        "sglang.srt.disaggregation.common",
        "sglang.srt.disaggregation.mooncake",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(root / "python" / name.replace(".", "/"))]
        sys.modules[name] = package
    from mooncake.engine import TransferEngine

    from sglang.srt.disaggregation.mooncake.pinned_staging_core import CorePinnedStager

    class Engine:
        def __init__(self):
            self.native = TransferEngine()
            assert self.native.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "") == 0
            self.session = f"127.0.0.1:{self.native.get_rpc_port()}"
            self.calls = []
            self.registered = set()

        def batch_register(self, pointers, sizes):
            rc = self.native.batch_register_memory(pointers, sizes)
            if rc == 0:
                self.registered.update(pointers)
            return rc

        def batch_deregister(self, pointers):
            rc = self.native.batch_unregister_memory(pointers)
            if rc == 0:
                self.registered.difference_update(pointers)
            return rc

        def register(self, pointer, size):
            assert self.batch_register([pointer], [size]) == 0

        def deregister(self, pointer):
            assert self.batch_deregister([pointer]) == 0

        def batch_transfer_sync(self, session, sources, targets, sizes):
            self.calls.append(list(zip(sources, targets, sizes)))
            return self.native.batch_transfer_sync_write(
                session, sources, targets, sizes
            )

    sender, receiver = Engine(), Engine()
    pool_path = native / "mem_cache/deepseek_v4_memory_pool.py"
    pool_cls = extract(
        pool_path, "DeepSeekV4TokenToKVPool", {"get_contiguous_buf_infos"}
    )
    index_cls = extract(
        pool_path, "DeepSeekV4IndexerPool", {"contiguous_page_row_buffers"}
    )

    def pool():
        p = pool_cls()
        p._unified_kv = p._unified_kv_fp8 = p.uniform_fp8 = False
        p.page_size = 8
        p.compression_ratios = [0, 0] + [2] * 18 + [1] * 20
        p._stage_start, p._stage_end = 0, 40
        p.sources_by_ratio = {2: [2, 8, 14], 1: [20]}
        p.kv_pools, p.index_pools = {}, {}
        tensors = []
        for ratio, producers in p.sources_by_ratio.items():
            kv = [
                torch.zeros((8, 131072), dtype=torch.uint8, device="cuda")
                for _ in producers
            ]
            p.kv_pools[ratio] = types.SimpleNamespace(kv_buffer=kv)
            index = index_cls()
            index.page_size, index.index_k_with_scale_buffer = 2, None
            rows = 8 * p.page_size // ratio // 2
            index.index_k_payload_buffer = [
                torch.zeros((rows, 16), dtype=torch.uint8, device="cuda")
                for _ in producers
            ]
            index.index_k_scale_buffer = [
                torch.zeros((rows, 4), dtype=torch.uint8, device="cuda")
                for _ in producers
            ]
            p.index_pools[ratio] = index
            tensors += kv + index.contiguous_page_row_buffers()
        return p, tensors

    source_pool, sources = pool()
    target_pool, targets = pool()
    src_ptrs, sizes, strides = source_pool.get_contiguous_buf_infos()
    dst_ptrs, dst_sizes, _ = target_pool.get_contiguous_buf_infos()
    assert sender.batch_register(src_ptrs, sizes) == 0
    assert receiver.batch_register(dst_ptrs, dst_sizes) == 0
    path = native / "disaggregation/common/staging_buffer.py"
    buffer_cls = extract(path, "StagingBuffer")
    allocator_cls = extract(path, "StagingAllocator")
    buffer_cls.__init__.__globals__.update(
        _STAGING_SUPPORTED_DEVICE_TYPES=("cuda",),
        _device_ptr_type_str=lambda *a: "",
        logger=logger,
    )
    allocator_cls.__init__.__globals__.update(
        StagingBuffer=buffer_cls,
        threading=threading,
        logger=logger,
        _device_ptr_type_str=lambda *a: "",
    )
    sys.modules["sglang.srt.disaggregation.common.staging_buffer"] = (
        types.SimpleNamespace(StagingAllocator=allocator_cls)
    )
    utilities = types.SimpleNamespace(
        create_device_event=lambda device: torch.cuda.Event(),
        create_device_stream=lambda device: torch.cuda.Stream(device=device),
        device_stream_context=torch.cuda.stream,
    )
    sys.modules["sglang.srt.utils"] = utilities
    sys.modules["sglang.srt.environ"] = types.SimpleNamespace(
        envs=types.SimpleNamespace(
            SGLANG_DISAGGREGATION_WAITING_TIMEOUT=types.SimpleNamespace(get=lambda: 30),
            SGLANG_DISAGG_STAGING_POOL_SIZE_MB=types.SimpleNamespace(get=lambda: 1),
        )
    )
    handler_path = native / "disaggregation/common/staging_handler.py"
    handler_cls = extract(handler_path, "DecodeStagingHandler")
    handler_cls.__init__.__globals__.update(
        threading=threading, logger=logger, time=time, **vars(utilities)
    )
    # Context decorators need dataclasses in the extraction namespace.
    # Native dataclass fields/defaults are preserved by this source loader.
    import ast

    def context(name):
        node = next(
            n
            for n in ast.parse(handler_path.read_text()).body
            if getattr(n, "name", None) == name
        )
        namespace = dict(dataclasses=dataclasses, threading=threading)
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
                str(handler_path),
                "exec",
            ),
            namespace,
        )
        return namespace[name]

    init_allocator = extract(handler_path, "init_staging_allocator")
    init_allocator.__globals__.update(
        _bare_device_type=lambda device: device.split(":")[0],
        _get_custom_mem_pool=lambda device: (None, None),
    )
    strategy_cls = extract(handler_path, "PrefillStagingStrategy")
    strategy_cls.__init__.__globals__.update(
        get_schedule=lambda: types.SimpleNamespace(chunked_prefill_size=16)
    )
    sys.modules[
        "sglang.srt.disaggregation.common.staging_buffer"
    ].staging_grid_tokens = extract(path, "staging_grid_tokens")
    sys.modules["sglang.srt.disaggregation.common.staging_handler"] = (
        types.SimpleNamespace(
            DecodeStagingContext=context("DecodeStagingContext"),
            PrefillStagingContext=context("PrefillStagingContext"),
            DecodeStagingHandler=handler_cls,
            init_staging_allocator=init_allocator,
            PrefillStagingStrategy=strategy_cls,
        )
    )
    conn = native / "disaggregation/mooncake/conn.py"
    cls = extract(
        conn,
        "MooncakeKVManager",
        {
            "set_kv_buffer_tensors",
            "_register_staging_memory",
            "ensure_host_stager",
            "_transfer_data",
            "_send_kvcache_generic",
            "request_layer_staging",
        },
    )
    cls.get_mla_kv_ptrs_with_pp = extract(
        native / "disaggregation/common/conn.py",
        "CommonKVManager",
        {"get_mla_kv_ptrs_with_pp"},
    ).get_mla_kv_ptrs_with_pp
    flat = extract(conn, "_uses_flat_entry_layout")
    state_types = types.SimpleNamespace(DSA="DSA", SWA="SWA")
    flat.__globals__["StateType"] = state_types
    cls._send_kvcache_generic.__globals__.update(
        group_concurrent_contiguous=extract(
            native / "disaggregation/common/utils.py", "group_concurrent_contiguous"
        ),
        _uses_flat_entry_layout=flat,
        StateType=state_types,
        is_elided_entry=extract(native / "disaggregation/utils.py", "is_elided_entry"),
    )
    cls.request_layer_staging.__globals__.update(
        struct=__import__("struct"),
        NetworkAddress=lambda *a: types.SimpleNamespace(
            to_tcp=lambda: "control-fixture", is_ipv6=False
        ),
    )
    entries = []
    for ratio, producers in source_pool.sources_by_ratio.items():
        for kind in ("kv", "index", "index"):
            for producer in producers:
                entries.append(
                    [kind, ratio, producer, "fp4", "torch.uint8", strides[len(entries)]]
                )
    schema = {"entries": entries}

    def manager(engine, ptrs, lens):
        m = cls()
        m.engine, m._core_stager, m.kv_buffer_tensors = engine, None, None
        m._host_staging_all = m.enable_staging = m.enable_custom_mem_pool = False
        m._staging_ctx = None
        m.kv_args = types.SimpleNamespace(
            kv_data_ptrs=ptrs,
            kv_data_lens=lens,
            kv_item_lens=strides,
            page_size=8,
            gpu_id=0,
            engine_rank=0,
        )
        m.attn_tp_size, m.pp_size, m.pp_rank = 1, 1, 0
        m.is_mla_backend, m.is_hybrid_mla_backend = True, False
        m.max_transfer_batch_indices = 2
        m._validate_envelope_kv_layout = lambda *a: None
        m._worker_threads = []
        return m

    p, d = manager(sender, src_ptrs, sizes), manager(receiver, dst_ptrs, dst_sizes)
    row = torch.cat(
        [torch.arange(i * 8, i * 8 + 8, device="cuda") for i in (5, 6, 1, 2, 3)]
    ).reshape(1, -1)
    scheduler = types.SimpleNamespace(
        schedule_stream=torch.cuda.current_stream(),
        req_to_token_pool=types.SimpleNamespace(req_to_token=row),
        disagg_decode_prealloc_queue=types.SimpleNamespace(
            transfer_queue=types.SimpleNamespace()
        ),
    )
    controller = types.SimpleNamespace(
        pool=target_pool,
        _schema=lambda manager: schema,
        config=types.SimpleNamespace(recv_host_staging=True),
        tp_size=1,
        rank=0,
        scheduler=scheduler,
    )
    rx.install(controller, d, "decode")
    handler, ring = d._staging_handler, d._staging_ctx.allocator
    handler._scatter_region = lambda *a, **kw: rx.scatter(None, handler, *a, **kw)
    pcontroller = types.SimpleNamespace(
        pool=source_pool, _schema=lambda manager: schema
    )
    rx.install(pcontroller, p, "prefill")
    watermark = extract(handler_path, "is_watermark_ready")
    p._is_watermark_ready = lambda session, rnd, end: watermark(
        p._staging_ctx, session, rnd, end
    )
    rsp = extract(handler_path, "handle_staging_rsp")
    rsp.__globals__.update(logger=logger)
    wm = extract(handler_path, "handle_watermark_msg")
    # Warm CUDA lazy initialization before checking a deliberately pending event.
    ring.buffer.gpu_buffer[:64].copy_(ring.buffer.buffer[:64], non_blocking=True)
    for tensor, stride in zip(targets, strides):
        tensor.view(torch.uint8).reshape(-1, stride).index_copy_(
            0,
            torch.zeros(1, device="cuda", dtype=torch.int64),
            torch.zeros((1, stride), device="cuda", dtype=torch.uint8),
        )
    torch.cuda.synchronize()
    results = []
    from sglang_remote_attn.dsv41_layerwise import resource_groups

    groups = resource_groups(schema, 1)

    for staged in (False, True):
        if staged:
            p._core_stager = CorePinnedStager(sender, capacity=65536)
            dsv41_staging.bind(source_pool, p)
        for tensor in targets:
            tensor.zero_()
        for i, tensor in enumerate(sources):
            tensor.fill_(i + 11)
        torch.cuda.synchronize()
        room = 101 + int(staged)
        peer = types.SimpleNamespace(
            is_dummy=False,
            endpoint="127.0.0.1",
            dst_port=1,
            mooncake_session_id=receiver.session,
            room=room,
        )
        peer.staging = types.SimpleNamespace(
            offsets=[], rounds=[], ends=[], set_chunk=lambda *a: None
        )
        # Use the real StagingTransferInfo class parsed by native metadata.
        stg_cls = context("StagingTransferInfo")
        peer.staging = stg_cls()
        p.transfer_infos = {room: {receiver.session: peer}}
        registration = types.SimpleNamespace(
            dst_kv_ptrs=dst_ptrs,
            dst_kv_item_len=strides[0],
            dst_attn_tp_size=1,
            requires_dcp_relayout=False,
            staging_base_ptr=ring.get_base_ptr(),
            staging_total_size=ring.total_size,
        )
        p.decode_kv_args_table = {receiver.session: registration}
        req = types.SimpleNamespace(
            bootstrap_room=room,
            origin_input_ids=list(range(33)),
            kv=types.SimpleNamespace(cache_protected_len=0, req_pool_idx=0),
        )
        recv = types.SimpleNamespace(
            chunk_staging_infos=[],
            chunk_staging_layers={},
            prefill_info=types.SimpleNamespace(attn_tp_size=1, pp_size=1),
            bootstrap_infos=[dict(pp_rank=0)],
        )

        def control(parts):
            if parts[0] == b"STAGING_RSP":
                rsp(parts, p.transfer_infos)
            else:
                wm(p._staging_ctx, parts)

        recv._connect_to_bootstrap_server = lambda bi: (
            types.SimpleNamespace(send_multipart=control),
            threading.Lock(),
        )
        dr = types.SimpleNamespace(req=req, kv_receiver=recv)
        handler.register_decode_req(room, dr)
        d._staging_ctx.room_receivers[room] = recv
        d._staging_ctx.room_bootstrap[room] = recv.bootstrap_infos
        p._send_multipart_locked = lambda address, msg, **kwargs: rx.allocate(
            None, d, msg
        )
        p.request_layer_staging(
            room, slice(2, 5), tuple((indices, None) for _, indices in groups)
        )
        assert len(ring.allocations) == 4
        strategy = rx.strategy(None, p, None)
        pending_seen = False
        wraps = []
        for ordinal, (_, indices) in enumerate(groups):
            ready, _, offset, rnd, _ = strategy.check_ready(
                peer, 2, 3, chunk_idx=ordinal
            )
            assert ready
            wraps.append(rnd)
            chunk = types.SimpleNamespace(
                staging_layer_ids=indices,
                kv_buffer_indices=indices,
                prefill_kv_indices=np.array([3, 4, 5], dtype=np.int32),
                transfer_fn=dsv41_staging.transfer(p) if staged else None,
            )
            assert rx.send_group(None, p, peer, chunk, registration, offset, None) == 0
            if ring._scatter_stream is None:
                ring._scatter_stream = torch.cuda.Stream()
            with torch.cuda.stream(ring._scatter_stream):
                torch.cuda._sleep(500_000_000)
            handler.handle_chunk_arrived(room, ordinal, 2, 3, receiver.session)
            handler.submit_last_scatter_async(room)
            handler.advance_scatter(dr)
            pending_seen |= (
                not handler.is_done(dr)
                and bool(dr._chunk_events)
                and not dr._chunk_events[-1][0].query()
            )
            assert not handler.is_done(dr) and ring.allocations
            ring._scatter_stream.synchronize()
            handler.advance_scatter(dr)
        assert pending_seen and handler.is_done(dr) and not ring.allocations
        torch.cuda.synchronize()
        for src, dst, stride in zip(sources, targets, strides):
            a, b = src.view(torch.uint8).reshape(-1), dst.view(torch.uint8).reshape(-1)
            for sp, dp in zip((3, 4, 5), (1, 2, 3)):
                assert torch.equal(
                    a[sp * stride : (sp + 1) * stride],
                    b[dp * stride : (dp + 1) * stride],
                )
            assert not b[5 * stride : 7 * stride].any()
        handler.unregister_decode_req(room)
        results.append(
            dict(
                p_host_staging=staged,
                bytes_exact=True,
                pending_h2d_observed=pending_seen,
                ring_wrapped=max(wraps) > min(wraps),
                watermark=ring.get_watermark(),
                native_ring_empty=True,
            )
        )
    # Cancel with a real H2D/scatter still queued. Native room release must
    # wait for that GPU work before freeing both issued and unsent allocations.
    room = 303
    peer.room = room
    peer.staging = context("StagingTransferInfo")()
    p.transfer_infos = {room: {receiver.session: peer}}
    req = types.SimpleNamespace(
        bootstrap_room=room,
        origin_input_ids=list(range(33)),
        kv=types.SimpleNamespace(cache_protected_len=0, req_pool_idx=0),
    )
    recv = types.SimpleNamespace(
        chunk_staging_infos=[],
        chunk_staging_layers={},
        prefill_info=types.SimpleNamespace(attn_tp_size=1, pp_size=1),
        bootstrap_infos=[dict(pp_rank=0)],
        _connect_to_bootstrap_server=lambda bi: (
            types.SimpleNamespace(send_multipart=control),
            threading.Lock(),
        ),
    )
    dr = types.SimpleNamespace(req=req, kv_receiver=recv)
    handler.register_decode_req(room, dr)
    d._staging_ctx.room_receivers[room] = recv
    d._staging_ctx.room_bootstrap[room] = recv.bootstrap_infos
    p.request_layer_staging(
        room, slice(2, 5), tuple((indices, None) for _, indices in groups)
    )
    ready, _, offset, _, _ = strategy.check_ready(peer, 2, 3, chunk_idx=0)
    assert ready
    indices = groups[0][1]
    chunk = types.SimpleNamespace(
        staging_layer_ids=indices,
        kv_buffer_indices=indices,
        prefill_kv_indices=np.array([3, 4, 5], dtype=np.int32),
        transfer_fn=dsv41_staging.transfer(p),
    )
    assert rx.send_group(None, p, peer, chunk, registration, offset, None) == 0
    with torch.cuda.stream(ring._scatter_stream):
        torch.cuda._sleep(500_000_000)
    handler.handle_chunk_arrived(room, 0, 2, 3, receiver.session)
    pending_event = dr._chunk_events[-1][0]
    assert not pending_event.query()
    handler.unregister_decode_req(room)
    assert pending_event.query() and not ring.allocations
    assert not handler.is_staging_room(room)
    for tensor in targets:
        tensor.fill_(201)
    ring._scatter_stream.synchronize()
    torch.cuda.synchronize()
    assert all(bool((tensor == 201).all()) for tensor in targets)
    ring_ptr = ring.get_base_ptr()
    # A clean native teardown boundary permits ring deregistration/storage release.
    rx.teardown(lambda manager: None, d)
    assert ring_ptr not in receiver.registered and not receiver._dsv41_receive_owners
    files = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py")) + [
        Path(__file__).resolve(),
        Path(__file__).with_name("dsv41_host_staging_cuda_probe.py").resolve(),
    ]
    files += [
        native / p
        for p in (
            "disaggregation/pinned_staging_buffer.py",
            "disaggregation/mooncake/pinned_staging_core.py",
            "disaggregation/mooncake/conn.py",
            "disaggregation/common/conn.py",
            "disaggregation/common/utils.py",
            "disaggregation/utils.py",
            "disaggregation/common/staging_handler.py",
            "disaggregation/common/staging_buffer.py",
            "mem_cache/deepseek_v4_memory_pool.py",
        )
    ]
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        entries=12,
        native_strides=strides,
        receive_ring_bytes=ring.total_size,
        p_send_slot_bytes=65536,
        cases=results,
        ring_deregistered_after_drain=True,
        cancel_waited_for_gpu=True,
        cancel_freed_issued_and_unsent=True,
        no_post_release_write=True,
        scope=__doc__,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
