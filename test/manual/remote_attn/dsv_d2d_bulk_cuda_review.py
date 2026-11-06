"""D2D safety regressions: real CUDA/TE, injected faults, no model weights.

A lost connection must retain the receive slot without retrying; an unproven
D2H must retain its GPU input. Both cases assert safety after writing the report.
The retry probe never submits a deliberately stale destination pointer to TE.
"""

import argparse
import json
import threading
import time
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
from dsv_d2d_probe_support import Environment
from sglang_remote_attn import dsv_d2d as d2d


def retry_cancel(root):
    e = Environment(root, "cuda", 2, False, kv_layout="v4")
    proceed = threading.Event()
    finish_retry = threading.Event()
    try:
        e.import_(e.freeze())
        assert e.blocked.wait(5)
        source = e.source.d2d_sources[e.epoch]
        target = e.target.d2d_targets[e.epoch]
        client = target.bulk_reader.client
        engine = d2d.manager(e.source).engine
        peer = d2d.manager(e.target).engine
        rt = d2d.remote.runtime()
        started, retried, completed = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        actual_transfer, actual_execute, actual_wait = (
            engine.batch_transfer_sync,
            d2d.execute,
            rt.TargetClient._wait,
        )
        registered = set()
        state = dict(dropped=False, reads=0)
        keep_backing = []
        register, deregister = peer.batch_register, peer.batch_deregister

        def track_register(ptrs, sizes):
            rc = register(ptrs, sizes)
            if rc == 0:
                registered.update(ptrs)
            return rc

        def track_deregister(ptrs):
            rc = deregister(ptrs)
            if rc == 0:
                registered.difference_update(ptrs)
            return rc

        def transfer(session, sources, targets, sizes):
            state["destination"] = targets[0]
            started.set()
            try:
                assert proceed.wait(15)
                if targets[0] not in registered:
                    state["stale_write_prevented_by_probe"] = True
                    return -1
                rc = actual_transfer(session, sources, targets, sizes)
                state["native_write_completed"] = rc == 0
                return rc
            finally:
                completed.set()

        def execute(job, stream, cache):
            if (
                job.header.get("dsv_d2d") == "read_group"
                or job.header.get("dsv_d2d") == "read"
                and job.header.get("kind") == "global"
            ):
                state["reads"] += 1
                if state["reads"] == 2:
                    retried.set()
                    assert finish_retry.wait(15)
            return actual_execute(job, stream, cache)

        def wait(self, fut, deadline):
            if self is client and not state["dropped"]:
                assert started.wait(5)
                # Retain the actual pinned backing as an extra probe safeguard.
                keep_backing.append(target.bulk_reader.bulk.tensor)
                state["dropped"] = True
                raise rt._RemoteDisconnected(
                    "injected connection loss before bulk completion"
                )
            return actual_wait(self, fut, deadline)

        with (
            patch.object(engine, "batch_transfer_sync", transfer),
            patch.object(peer, "batch_register", track_register),
            patch.object(peer, "batch_deregister", track_deregister),
            patch.object(d2d, "execute", execute),
            patch.object(rt.TargetClient, "_wait", wait),
        ):
            e.gate.set()
            assert started.wait(5), state
            deadline = time.monotonic() + 5
            while (
                target.worker.is_alive()
                and not retried.is_set()
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            state["resume"] = e.command(e.source, "d2d_resume")
            finish_retry.set()
            target.worker.join(5)
            slot = target.bulk_reader.bulk
            state.update(
                worker_stopped=not target.worker.is_alive(),
                original_bulk_call_unfinished=not proceed.is_set(),
                receiver_closed=slot.closed,
                receiver_quarantined=slot.unsafe,
                destination_registered=state["destination"] in registered,
                target_error=target.error,
                native_mooncake_setup_transfers=len(e.transport.calls),
                real_pinned_host_memory=keep_backing[0].is_pinned(),
                rpc_retries=client.stats()["rpc_retried"],
            )
            state["unsafe_reproduced"] = (
                state["receiver_closed"]
                and not state["destination_registered"]
                and state["original_bulk_call_unfinished"]
            )
            state["safety_pass"] = (
                state["worker_stopped"]
                and state["receiver_quarantined"]
                and not state["receiver_closed"]
                and state["destination_registered"]
                and state["original_bulk_call_unfinished"]
                and state["reads"] == 1
                and state["rpc_retries"] == 0
                and state["resume"].get("pending_reader") is True
            )
            try:
                e.command(e.target, "d2d_abort")
                state["abort_blocked"] = False
            except d2d.TransportFailure:
                state["abort_blocked"] = True
            state["safety_pass"] &= state["abort_blocked"] and not e.freed
            proceed.set()
            assert completed.wait(8)
            state["safety_pass"] &= state.get("native_write_completed", False)
        return state
    finally:
        finish_retry.set()
        proceed.set()
        e.close()


def d2h_input(root):
    from cuda.bindings import runtime as cuda

    e = Environment(root, "cuda", 2, False, kv_layout="v4")
    try:
        e.freeze()
        sender = e.source.d2d_sources[e.epoch].bulk_sender
        # An actual peer host receive allocation; no TE write is issued in this case.
        from sglang_remote_attn import d2d_bulk as bulk
        from types import SimpleNamespace

        receiver = bulk.Receiver(
            d2d.manager(e.target).engine, SimpleNamespace(), "cuda:0"
        )
        region = receiver.prepare()
        producer = torch.cuda.Stream()
        pending = torch.cuda.Event()
        size = 1024 * 1024
        state = dict(source_allocation_reused=False)

        def submit_then_lose_proof(blocks, ready):
            sender.slot.stream.wait_event(ready)
            with torch.cuda.stream(sender.slot.stream):
                torch.cuda._sleep(300_000_000)
            src, _, length = blocks[0]
            state["source_ptr"] = src
            (rc,) = cuda.cudaMemcpyAsync(
                sender.tensor.data_ptr(),
                src,
                length,
                cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                sender.slot.raw_stream,
            )
            assert int(rc) == 0, rc
            pending.record(sender.slot.stream)
            raise RuntimeError(
                "injected loss of D2H completion proof after real async submission"
            )

        def invoke():
            data = torch.full((size,), 17, dtype=torch.uint8, device="cuda:0")
            producer.synchronize()
            ref = weakref.ref(data)
            try:
                sender.write(data, d2d.dp.event("cuda:0"), region, 1)
            except RuntimeError as exc:
                assert "injected loss" in str(exc)
            return ref

        replacements = []
        with (
            torch.cuda.stream(producer),
            patch.object(sender.slot, "copy", submit_then_lose_proof),
        ):
            ref = invoke()
            state["d2h_pending_after_exception"] = not pending.query()
            state["input_retained_after_exception"] = ref() is not None
            for _ in range(32):
                replacement = torch.empty(size, dtype=torch.uint8, device="cuda:0")
                replacements.append(replacement)
                if replacement.data_ptr() == state["source_ptr"]:
                    replacement.fill_(231)
                    state["source_allocation_reused"] = True
                    break
            producer.synchronize()
        sender.slot.stream.synchronize()
        state.update(
            sender_quarantined=sender.unsafe,
            engine_retains_sender=sender in sender.engine._d2d_bulk_owners,
            host_storage_retained=sender.tensor is not None,
            copied_original=bool(torch.all(sender.tensor[:size] == 17)),
            copied_replacement=bool(torch.all(sender.tensor[:size] == 231)),
        )
        state["unsafe_reproduced"] = (
            state["d2h_pending_after_exception"]
            and not state["input_retained_after_exception"]
        )
        state["safety_pass"] = (
            state["d2h_pending_after_exception"]
            and state["input_retained_after_exception"]
            and not state["source_allocation_reused"]
            and state["copied_original"]
            and not state["copied_replacement"]
            and state["sender_quarantined"]
            and state["engine_retains_sender"]
        )
        receiver.close()
        return state
    finally:
        e.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--case", choices=("retry-cancel", "d2h-input"), required=True)
    args = p.parse_args()
    torch.cuda.set_device(0)
    result = (retry_cancel if args.case == "retry-cancel" else d2h_input)(
        args.source_root
    )
    report = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        case=args.case,
        scope=__doc__,
        result=result,
        status="PASS" if result["safety_pass"] else "FAIL",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    assert result["safety_pass"], f"D2D safety regression failed: {args.case}"


if __name__ == "__main__":
    main()
