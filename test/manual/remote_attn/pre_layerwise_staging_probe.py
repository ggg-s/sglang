"""Real Mooncake TCP / two-GPU check of the pre-layerwise staging adapters.

Run in the deployed Python environment with the plugin installed. This tests
transport and stream independence, not model accuracy or PD end-to-end latency.
The busy condition queues a CUDA sleep after KV-ready on the legacy stream;
it isolates stream ordering rather than simulating memory-bandwidth contention.
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path
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
        parser.error("payload sizes, iterations and busy cycles must be positive")
    if os.environ.get("MC_FORCE_TCP", "1") != "1":
        parser.error("this probe requires MC_FORCE_TCP=1")
    os.environ["MC_FORCE_TCP"] = "1"

    import torch
    from mooncake.engine import TransferEngine
    from sglang_remote_attn.pinned_staging import PinnedKVStager
    from sglang_remote_attn.receive_staging import HostDestination, ReceiveStaging

    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        MooncakeTransferEngine,
    )

    def engine():
        native = TransferEngine()
        rc = native.initialize(args.host, "P2PHANDSHAKE", "rdma", "")
        if rc != 0:
            raise RuntimeError(f"Mooncake TCP initialization failed: {rc}")
        # Use the production register/deregister/transfer methods, without
        # booting a server or overriding its deployment configuration.
        wrapper = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        wrapper.engine = native
        return wrapper, f"{args.host}:{native.get_rpc_port()}"

    sender, _ = engine()
    receiver, session = engine()
    shape = (2 * args.layers, args.tokens * args.item_bytes)
    src = torch.empty(shape, dtype=torch.uint8, device=args.source_device)
    dst = torch.empty(shape, dtype=torch.uint8, device=args.destination_device)
    for i, buffer in enumerate(src):
        buffer.fill_(i + 1)
    torch.cuda.synchronize(args.source_device)
    sender.register(src.data_ptr(), src.numel())
    receiver.register(dst.data_ptr(), dst.numel())
    blocks = [(s.data_ptr(), d.data_ptr(), s.numel()) for s, d in zip(src, dst)]
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
    ready = torch.cuda.Event()
    busy_done = torch.cuda.Event()
    stream = torch.cuda.default_stream(args.source_device)
    results = []
    rx_manager = SimpleNamespace(
        engine=receiver,
        _remote_attn_send_lock=threading.Lock(),
        kv_args=SimpleNamespace(
            kv_data_ptrs=[d.data_ptr() for d in dst],
            kv_item_lens=[args.item_bytes] * (2 * args.layers),
            kv_layer_ids=[],
        ),
    )
    busy_device = (
        args.source_device if args.busy_device == "source" else args.destination_device
    )
    busy_stream = torch.cuda.default_stream(busy_device)
    with (
        tempfile.TemporaryDirectory() as journal,
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
    ):
        # Warm up the private copy stream and native TCP connection first.
        with torch.cuda.device(args.source_device):
            ready.record(stream)
        for staged in (False, True):
            operation = stager.send if staged else transfer
            kwargs = {"ready_event": ready} if staged else {}
            if executor.submit(operation, session, blocks, **kwargs).result() != 0:
                raise RuntimeError(
                    "warmup transfer failed; buffers retained until exit"
                )
        for staged, busy in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            durations, completed_before_compute = [], []
            for iteration in range(args.iterations):
                rx = adapter = None
                if args.receive_staging:
                    path = Path(journal) / f"{staged}-{busy}-{iteration}"
                    rx = ReceiveStaging(
                        rx_manager,
                        args.destination_device,
                        path,
                        "probe",
                        range(args.tokens),
                        args.layers,
                        300,
                    )
                    source = SimpleNamespace(
                        path=path,
                        generation="probe",
                        index_slice=slice(0, args.tokens),
                        owner=SimpleNamespace(
                            config=SimpleNamespace(drain_timeout=300)
                        ),
                        req=SimpleNamespace(
                            disagg_kv_sender=SimpleNamespace(kv_mgr=manager)
                        ),
                        peers=((SimpleNamespace(mooncake_session_id=session), None),),
                        stager=stager if staged else None,
                    )
                    adapter = HostDestination(
                        source,
                        [(tuple(range(args.layers)), tuple(range(2 * args.layers)))],
                    )
                dst.zero_()
                torch.cuda.synchronize(args.destination_device)
                with torch.cuda.device(args.source_device), torch.cuda.stream(stream):
                    ready.record(stream)
                with torch.cuda.device(busy_device), torch.cuda.stream(busy_stream):
                    if busy:
                        torch.cuda._sleep(args.busy_cycles)
                    busy_done.record(busy_stream)
                operation = stager.send if staged else transfer
                kwargs = {"ready_event": ready} if staged else {}
                if adapter is not None:
                    operation = lambda session, blocks: adapter.transfer(
                        tuple(range(args.layers)), ready, session, blocks
                    )
                    kwargs = {}
                start = time.perf_counter()
                rc = executor.submit(operation, session, blocks, **kwargs).result()
                if rc == 0 and adapter is not None:
                    adapter.complete(tuple(range(args.layers)))
                durations.append((time.perf_counter() - start) * 1000)
                completed_before_compute.append(busy and not busy_done.query())
                if rc != 0:
                    raise RuntimeError(
                        f"transfer failed: {rc}; buffers retained until exit"
                    )
                if rx is not None:
                    rx.close()
                busy_done.synchronize()
                for i, buffer in enumerate(dst):
                    if not torch.all(buffer == i + 1).item():
                        raise AssertionError(f"KV bytes differ in buffer {i}")
            results.append(
                dict(
                    host_staging=staged,
                    recv_host_staging=args.receive_staging,
                    busy_device=args.busy_device,
                    busy_default_stream=busy,
                    bytes=src.numel(),
                    p50_ms=statistics.median(durations),
                    samples_ms=durations,
                    completed_before_compute=completed_before_compute,
                    data_verified=True,
                )
            )
    if not stager.close():
        raise RuntimeError("staging cleanup not proven safe; retained until exit")
    sender.deregister(src.data_ptr())
    receiver.deregister(dst.data_ptr())
    print(json.dumps(results, indent=2))
    if not any(results[-1]["completed_before_compute"]):
        raise SystemExit(
            "Staged transfer never beat the compute tail; inspect a CUDA trace "
            "or increase --busy-cycles if the transfer itself exceeds that tail."
        )


if __name__ == "__main__":
    main()
