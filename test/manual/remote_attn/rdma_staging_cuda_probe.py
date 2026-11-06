"""Real CUDA staging/streams with an explicit in-process fake RDMA NIC boundary.

This does NOT test an RDMA HCA, GPUDirect visibility, or network performance.
The fake NIC uses cudaMemcpyAsync on its own stream and reports completion only
after its event. Native TCP regression is exercised by range_layer_cuda_probe.
"""

import argparse
import json
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from cuda.bindings import runtime as cuda
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import mooncake_backend as mb
from sglang_remote_attn import qwen_ranges
from sglang_remote_attn import remote_data_plane as dp


class FakeRdma:
    peers = {}
    protocol = "rdma"
    d2d_protocol_verified = True

    def __init__(self, name, fail=False, late=False):
        self.name, self.fail = name, fail
        self.regions, self.batches, self.pending = {}, [], set()
        self.lock = threading.Lock()
        self.stream = torch.cuda.Stream()
        self.peers[name] = self
        self.sequence = 0
        self.late = late
        self.late_writes = []
        self.async_calls = 0
        self.sync_calls = 0

    def get_session_id(self):
        return self.name

    def batch_register(self, pointers, sizes):
        for pointer, size in zip(pointers, sizes):
            rc, attr = cuda.cudaPointerGetAttributes(pointer)
            assert (
                int(rc) == 0 and attr.type == cuda.cudaMemoryType.cudaMemoryTypeDevice
            )
            assert pointer not in self.regions
            self.regions[pointer] = size
        return 0

    def batch_deregister(self, pointers):
        for pointer in pointers:
            del self.regions[pointer]
        return 0

    def copy(self, session, sources, targets, sizes):
        with self.lock, torch.cuda.device(0):
            peer = self.peers[session]
            for source, target, size in zip(sources, targets, sizes):
                assert self.regions[source] >= size and peer.regions[target] >= size
                result = cuda.cudaMemcpyAsync(
                    target,
                    source,
                    size,
                    cuda.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                    self.stream.cuda_stream,
                )
                assert int(result[0]) == 0
            self.stream.synchronize()
            self.batches.append(list(sizes))
        return 0

    def batch_transfer_sync(self, *args):
        self.sync_calls += 1
        if self.fail and self.late:
            self.late_writes.append(args)  # Failed first attempt remains live.
            self.copy(*args)  # An internal retry succeeds and hides that attempt.
            return 0
        raise AssertionError("D2D used hidden-retry sync API")

    def batch_transfer_async_write(self, *args):
        self.async_calls += 1
        if self.fail and self.late:
            # Failure in one descriptor leaves a sibling NIC write outstanding.
            # A sync convenience binding could hide this by retrying successfully.
            self.late_writes.append(args)
        else:
            self.copy(*args)
        self.sequence += 1
        self.pending.add(self.sequence)
        return self.sequence

    def get_batch_transfer_status(self, ids):
        if self.fail:
            return -1  # Failed does not mean drained; do not discard pending DMA.
        for bid in ids:
            self.pending.remove(bid)
        return 0

    def flush_late(self):
        assert self.late_writes
        for args in self.late_writes:
            self.copy(*args)  # Must still reference registered, unreused GPU slots.
        self.late_writes.clear()
        self.pending.clear()

    batch_register_memory = batch_register
    batch_unregister_memory = batch_deregister
    batch_transfer_sync_write = batch_transfer_sync


def qwen(fail=False, snapshot=False, late=False, fast_async=True):
    from cuda_handoff_probe import DIM, HEADS, LAYERS, TOKENS, kv_pool, target_state

    engine = FakeRdma(
        f"qwen-{fail}-{snapshot}-{late}-{fast_async}", fail=fail, late=late
    )
    pool = kv_pool(torch.bfloat16)
    indices = torch.arange(TOKENS, device="cuda") * 2 + 1
    req = NS(rid="rdma-cuda", kv=NS(req_pool_idx=0, kv_committed_len=TOKENS))
    scheduler = NS(
        req_to_token_pool=NS(req_to_token=indices.unsqueeze(0)),
        token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
    )
    meta = dict(
        rid=req.rid,
        dtype="bfloat16",
        kv_committed_len=TOKENS,
        kv_heads=HEADS,
        head_dim=DIM,
        chunks=[[TOKENS] for _ in range(LAYERS)],
        d2d_range_protocol="layer-token-range-v1",
        d2d_group_size=3,
        d2d_token_range=257,
    )
    cpu = (
        [
            [(pool.keys[i][indices].cpu(), pool.values[i][indices].cpu())]
            for i in range(LAYERS)
        ]
        if snapshot
        else None
    )
    manager = rt.RegistrationManager(te=engine)
    with (
        patch.object(rt, "_get_te", return_value=engine),
        patch.object(rt, "_REGISTRATION_MANAGER", manager),
        patch.object(rt, "_TRANSFER_REGISTRATIONS", {}),
        patch.object(rt, "_CTL_DIR", ""),
        patch.object(rt, "TARGET_STAGING_SLOTS", 2),
        patch.object(rt, "FAST_ASYNC", fast_async),
    ):
        transfer = qwen_ranges.source(scheduler, req, meta, False, cpu)
        conn, wire, _, _, _, owner, staging, _, _ = rt._recv_transfer_layer_setup(
            transfer
        )
        assert wire["_range_backend"] == mb.GPU and all(t.is_cuda for t in staging)
        keys = [torch.full_like(pool.keys[0], float("nan")) for _ in range(LAYERS)]
        values = [torch.full_like(keys[0], float("nan")) for _ in range(LAYERS)]
        target_pool = NS(
            get_key_buffer=keys.__getitem__, get_value_buffer=values.__getitem__
        )
        target_scheduler = NS(
            token_to_kv_pool_allocator=NS(get_kvcache=lambda: target_pool)
        )
        mapping = torch.arange(TOKENS, device="cuda").flip(0) + 5
        state = target_state()
        state.indices_ready = dp.event("cuda:0")
        qwen_ranges.receive(
            target_scheduler, state, conn, wire, staging, mapping, owner, 0
        )
        req._handoff_transfer_thread.join(10)
        assert not req._handoff_transfer_thread.is_alive()
        if fail:
            assert state.error and state.import_done_at is None
            assert owner in rt._TRANSFER_QUARANTINE and engine.regions
            assert req._handoff_transfer_unsafe
            for value in keys + values:
                assert bool(torch.isnan(value).all())
            try:
                rt.stop_source_transfer(req)
                raise AssertionError("unsafe source reuse")
            except RuntimeError:
                pass
            if late:
                before = dict(engine.regions)
                assert engine.async_calls == 1 and engine.sync_calls == 0
                assert len(engine.late_writes) == 1
                engine.flush_late()
                assert engine.regions == before
                assert req._handoff_transfer_unsafe and owner in rt._TRANSFER_QUARANTINE
                assert state.error and state.import_done_at is None
                for value in keys + values:
                    assert bool(torch.isnan(value).all())
        else:
            assert state.error is None and state.import_done_at is not None
            assert all(e.is_set() for e in state.events)
            for layer in range(LAYERS):
                torch.testing.assert_close(
                    keys[layer][mapping], pool.keys[layer][indices], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    values[layer][mapping], pool.values[layer][indices], rtol=0, atol=0
                )
            assert not engine.regions and not engine.pending
        return dict(
            model="qwen",
            snapshot=snapshot,
            lost_completion=fail,
            late_write_after_cancel=late,
            fast_async=fast_async,
            backend=mb.GPU,
            batches=len(engine.batches),
            pass_=True,
        )


def dsv(root):
    import dsv_d2d_probe_support as support
    from group_layer_cuda_probe import dsv_local

    real = support.Environment
    engines = []

    def environment(*args, **kwargs):
        env = real(*args, **kwargs)
        for controller in env.nodes:
            engine = FakeRdma(f"dsv-{len(engines)}")
            controller.scheduler.disagg_decode_prealloc_queue.kv_manager.engine = engine
            engines.append(engine)
        return env

    with patch.object(support, "Environment", side_effect=environment):
        results = dsv_local(root)
    assert all(not engine.regions for engine in engines)
    return [dict(result, backend=mb.GPU, fake_nic=True) for result in results]


def dsv_failure(root, layout, late=False):
    from dsv_d2d_probe_support import Environment
    from sglang_remote_attn import dsv_d2d as d2d

    env = Environment(root, "cuda", 2, False, kv_layout=layout)
    engines = []
    try:
        for n, controller in enumerate(env.nodes):
            engine = FakeRdma(f"dsv-failure-{layout}-{n}")
            controller.scheduler.disagg_decode_prealloc_queue.kv_manager.engine = engine
            engines.append(engine)
        env.import_(env.reserve_and_freeze())
        assert env.blocked.wait(5)
        target = env.target.d2d_targets[env.epoch]
        source = env.source.d2d_sources[env.epoch]
        engines[0].fail = True
        engines[0].late = late
        submissions = engines[0].async_calls
        env.gate.set()
        target.worker.join(10)
        assert not target.worker.is_alive() and target.error
        assert source.bulk_sender.unsafe and target.bulk_reader.bulk.unsafe
        assert (
            source.bulk_sender.tensor.is_cuda and target.bulk_reader.bulk.tensor.is_cuda
        )
        assert source.bulk_sender.tensor.data_ptr() in engines[0].regions
        assert target.bulk_reader.bulk.tensor.data_ptr() in engines[1].regions
        assert not target.copied and not target.rpc.local_entries
        try:
            env.command(env.target, "d2d_abort")
            raise AssertionError("unsafe target abort")
        except d2d.TransportFailure:
            pass
        try:
            env.command(env.source, "d2d_resume")
            raise AssertionError("unsafe source resume")
        except ValueError:
            pass
        assert not env.freed and target.decode_req.req.kv.holds_kv
        if late:
            before = [dict(engine.regions) for engine in engines]
            assert engines[0].async_calls == submissions + 1
            assert engines[0].sync_calls == 0
            engines[0].flush_late()
            assert [engine.regions for engine in engines] == before
            assert not target.copied and not target.rpc.local_entries
            assert not env.freed and source.bulk_sender.unsafe
        return dict(
            late_write_after_cancel=late,
            layout=layout,
            backend=mb.GPU,
            lost_completion=True,
            source_and_target_retained=True,
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.testing.assert_close(torch.zeros(1), torch.zeros(1))
    visibility = NS(receive=lambda: None)
    # Capability probe still exercises real CUDA allocations and the fake NIC's
    # registration contract. Only external-write ordering is a fake boundary.
    with (
        patch.dict(
            "os.environ",
            {"HANDOFF_D2D_BACKEND": "rdma", "HANDOFF_D2D_TOKEN_RANGE": "17"},
        ),
        patch.object(dp, "Visibility", return_value=visibility),
        patch.object(qwen_ranges, "Visibility", return_value=visibility),
    ):
        cases = [qwen(), qwen(snapshot=True), qwen(fail=True)]
        cases.extend(
            qwen(fail=True, late=True, fast_async=value) for value in (False, True)
        )
        cases.extend(dsv(args.source_root))
        cases.extend(dsv_failure(args.source_root, layout) for layout in ("v4", "v41"))
        cases.extend(
            dsv_failure(args.source_root, layout, late=True) for layout in ("v4", "v41")
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                status="PASS",
                device=torch.cuda.get_device_name(),
                scope=__doc__,
                real_rdma=False,
                cases=cases,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(status="PASS", cases=len(cases), real_rdma=False)))


if __name__ == "__main__":
    main()
