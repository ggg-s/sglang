"""Real Mooncake TCP / two-GPU check of native ring H2D + scatter.

Use --native-ring for the GPU-ring control, add --receive-staging for pinned
host reception, or omit both for direct GPU writes. --busy-device destination
isolates receive-side default-stream dependencies. This is not PD/model E2E.
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import threading
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--destination-device", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--item-bytes", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--native-ring", action="store_true")
    parser.add_argument("--receive-staging", action="store_true")
    parser.add_argument(
        "--busy-device", choices=("source", "destination"), default="source"
    )
    parser.add_argument("--busy-cycles", type=int, default=1_000_000_000)
    args = parser.parse_args()
    if args.source_device == args.destination_device:
        parser.error("use distinct source and destination GPUs")
    if (
        min(
            args.tokens, args.item_bytes, args.layers, args.iterations, args.busy_cycles
        )
        <= 0
    ):
        parser.error("sizes, iterations and busy cycles must be positive")
    if os.environ.get("MC_FORCE_TCP", "1") != "1":
        parser.error("this probe requires MC_FORCE_TCP=1")
    os.environ["MC_FORCE_TCP"] = "1"
    args.native_ring |= args.receive_staging

    import torch
    from mooncake.engine import TransferEngine
    from sglang_remote_attn.pinned_staging import PinnedKVStager

    from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
    from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )

    def engine():
        native = TransferEngine()
        rc = native.initialize(args.host, "P2PHANDSHAKE", "rdma", "")
        if rc != 0:
            raise RuntimeError(f"Mooncake initialization failed: {rc}")
        wrapper = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        wrapper.engine = native
        return wrapper, f"{args.host}:{native.get_rpc_port()}"

    sender, _ = engine()
    receiver, session = engine()
    shape = (2 * args.layers, args.tokens, 1, args.item_bytes)
    src = torch.empty(shape, dtype=torch.uint8, device=args.source_device)
    dst = torch.empty(shape, dtype=torch.uint8, device=args.destination_device)
    for i, buffer in enumerate(src):
        buffer.fill_(i + 1)
    torch.cuda.synchronize(args.source_device)
    sender.register(src.data_ptr(), src.numel())
    receiver.register(dst.data_ptr(), dst.numel())
    manager = SimpleNamespace(engine=sender, _remote_attn_send_lock=threading.Lock())

    def transfer(target, plan):
        with manager._remote_attn_send_lock:
            sources, targets, sizes = zip(*plan)
            return sender.batch_transfer_sync(
                target, list(sources), list(targets), list(sizes)
            )

    manager._transfer_data = transfer
    stager = PinnedKVStager(manager, args.source_device)
    stager.reserve(src.numel())
    ready, busy_done = torch.cuda.Event(), torch.cuda.Event()
    source_stream = torch.cuda.default_stream(args.source_device)
    busy_device = (
        args.source_device if args.busy_device == "source" else args.destination_device
    )
    busy_stream = torch.cuda.default_stream(busy_device)
    ring = handler = None
    if args.native_ring:
        ring = StagingAllocator(
            src.numel(),
            "cuda",
            args.destination_device,
            pin_memory=args.receive_staging,
        )
        receiver.register(ring.base_ptr, ring.total_size)
        rx_manager = SimpleNamespace(
            pp_size=1,
            kv_args=SimpleNamespace(engine_rank=0),
            _staging_ctx=SimpleNamespace(room_receivers={}, room_bootstrap={}),
        )
        rows = torch.arange(args.tokens, device=args.destination_device).reshape(1, -1)
        scheduler = SimpleNamespace(
            schedule_stream=torch.cuda.default_stream(args.destination_device),
            req_to_token_pool=SimpleNamespace(req_to_token=rows),
        )
        handler = DecodeStagingHandler(
            rx_manager,
            ring,
            dict(
                k_buffers=list(dst[: args.layers]),
                v_buffers=list(dst[args.layers :]),
                page_size=1,
            ),
            1,
            1,
            0,
            scheduler,
        )
        torch.cuda.synchronize(args.destination_device)

    results, room = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for staged, busy in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            durations, overlapped = [], []
            # One warmup per case, outside recorded samples.
            for iteration in range(args.iterations + 1):
                room += 1
                dst.zero_()
                torch.cuda.synchronize(args.destination_device)
                blocks = [
                    (s.data_ptr(), d.data_ptr(), s.numel()) for s, d in zip(src, dst)
                ]
                if ring is not None:
                    alloc_id, offset, rnd = ring.assign(src.numel())
                    recv = SimpleNamespace(
                        chunk_staging_infos=[
                            (alloc_id, offset, rnd, offset + src.numel(), args.tokens)
                        ],
                        chunk_staging_layers={
                            0: (tuple(range(args.layers)), 0, args.tokens)
                        },
                        prefill_info=SimpleNamespace(attn_tp_size=1, pp_size=1),
                    )
                    dr = SimpleNamespace(
                        req=SimpleNamespace(
                            bootstrap_room=room,
                            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=0),
                        ),
                        kv_receiver=recv,
                    )
                    handler.register_decode_req(room, dr)
                    blocks = [
                        (
                            s.data_ptr(),
                            ring.base_ptr + offset + i * s.numel(),
                            s.numel(),
                        )
                        for i, s in enumerate(src)
                    ]
                with torch.cuda.device(args.source_device):
                    ready.record(source_stream)
                with torch.cuda.device(busy_device), torch.cuda.stream(busy_stream):
                    if busy:
                        torch.cuda._sleep(args.busy_cycles)
                    busy_done.record(busy_stream)
                start = time.perf_counter()
                operation = stager.send if staged else transfer
                kwargs = {"ready_event": ready} if staged else {}
                rc = executor.submit(operation, session, blocks, **kwargs).result()
                if rc != 0:
                    raise RuntimeError(
                        f"transfer failed: {rc}; buffers retained until exit"
                    )
                if handler is not None:
                    executor.submit(
                        handler.handle_chunk_arrived, room, 0, 0, args.tokens, "probe"
                    ).result()
                    handler.submit_last_scatter_async(room)
                    deadline = time.monotonic() + 300
                    while not handler.is_done(dr):
                        handler.advance_scatter(dr)
                        if handler.is_failed(dr) or time.monotonic() >= deadline:
                            raise RuntimeError(
                                "scatter did not complete; buffers retained until exit"
                            )
                        time.sleep(0.001)
                if iteration:
                    durations.append((time.perf_counter() - start) * 1000)
                    overlapped.append(busy and not busy_done.query())
                if handler is not None:
                    handler.unregister_decode_req(room)
                busy_done.synchronize()
                for i, buffer in enumerate(dst):
                    if not torch.all(buffer == i + 1).item():
                        raise AssertionError(f"KV bytes differ in buffer {i}")
            results.append(
                dict(
                    host_staging=staged,
                    native_ring=args.native_ring,
                    recv_host_staging=args.receive_staging,
                    busy_device=args.busy_device,
                    busy_default_stream=busy,
                    bytes=src.numel(),
                    p50_ms=statistics.median(durations),
                    samples_ms=durations,
                    completed_before_compute=overlapped,
                    data_verified=True,
                )
            )
    if not stager.close():
        raise RuntimeError("source staging did not drain")
    if ring is not None:
        receiver.deregister(ring.base_ptr)
    sender.deregister(src.data_ptr())
    receiver.deregister(dst.data_ptr())
    print(json.dumps(results, indent=2))
    if not any(results[-1]["completed_before_compute"]):
        raise SystemExit(
            "Transfer/scatter never beat the compute tail; inspect a CUDA trace or increase --busy-cycles."
        )


if __name__ == "__main__":
    main()
