"""Qwen/DSV Q/O/LSE on real CUDA with native TCP or an explicit fake NIC.

No checkpoint is loaded. Qwen uses FlashInfer; DSV uses native byte layouts and
sparse kernels. Fake NIC/visibility cases do not validate a real RDMA HCA/CQ.
"""

import argparse
import json
import os
import sys
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from cuda.bindings import runtime as cuda
from remote_gpu_probe_support import Transport
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import mooncake_backend as mb
from sglang_remote_attn import remote_data_plane as dp


class FakeNIC:
    peers = {}
    protocol = "rdma"
    d2d_protocol_verified = True

    def __init__(self, name):
        self.name = self.hostname = name
        self.regions = {}
        self.pending = {}
        self.next_id = 1
        self.calls = []
        self.sync_calls = 0
        self.fail = None
        self.late = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stream = torch.cuda.Stream()
        self.peers[name] = self

    def get_session_id(self):
        return self.name

    def batch_register(self, addresses, sizes):
        for address, size in zip(addresses, sizes):
            assert address not in self.regions
            self.regions[address] = size
        return 0

    def batch_deregister(self, addresses):
        for address in addresses:
            self.regions.pop(address)
        return 0

    def copy(self, args):
        session, sources, targets, sizes = args
        peer = self.peers[session]
        with torch.cuda.device(0):
            for src, dst, size in zip(sources, targets, sizes):
                assert any(
                    p <= src and src + size <= p + n for p, n in self.regions.items()
                )
                assert any(
                    p <= dst and dst + size <= p + n for p, n in peer.regions.items()
                )
                (rc,) = cuda.cudaMemcpyAsync(
                    dst,
                    src,
                    size,
                    cuda.cudaMemcpyKind.cudaMemcpyDefault,
                    self.stream.cuda_stream,
                )
                assert int(rc) == 0
            self.stream.synchronize()

    def batch_transfer_async_write(self, *args):
        ticket = self.next_id
        self.next_id += 1
        self.calls.append(args)
        self.pending[ticket] = args
        if self.fail in ("late", "cancel"):
            self.late.append(args)
        else:
            self.copy(args)
        return ticket

    def get_batch_transfer_status(self, tickets):
        self.entered.set()
        if self.fail == "cancel":
            assert self.release.wait(8), "test cancellation gate timed out"
        if self.fail == "exception":
            raise RuntimeError("NIC result lost after actual CUDA write")
        if self.fail in ("late", "cancel"):
            return -1
        for ticket in tickets:
            self.pending.pop(ticket)
        return 0

    def batch_transfer_sync_write(self, *args):
        self.sync_calls += 1
        # Model the old bug: first attempt can remain live while retry succeeds.
        self.late.append(args)
        self.copy(args)
        return 0

    def flush_late(self):
        assert self.late
        for args in self.late:
            self.copy(args)
        self.late.clear()


class Fixture:
    def __init__(self, root, model, factory=None):
        self.root, self.model = root, model
        self.stack = ExitStack()
        # Restore only this fixture seam; patch.dict(sys.modules) would remove
        # lazy FlashInfer/CUTLASS imports and re-register native types next case.
        key = "sglang.srt.disaggregation.base.conn"
        previous = sys.modules.get(key)
        sys.modules[key] = NS(KVPoll=NS(Success=4, Failed=0))
        self.stack.callback(
            lambda: (
                sys.modules.__setitem__(key, previous)
                if previous is not None
                else sys.modules.pop(key, None)
            )
        )
        self.folder = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.freed = []
        self.transport = Transport(root) if factory is None else None
        if model == "qwen":
            torch.manual_seed(5)
            self.q = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
            self.k = torch.randn(64, 2, 128, device="cuda", dtype=self.q.dtype)
            self.v = torch.randn_like(self.k)
            pool = NS(
                get_key_buffer=lambda layer: self.k,
                get_value_buffer=lambda layer: self.v,
            )
            self.p = NS(
                scheduler=NS(
                    req_to_token_pool=NS(
                        req_to_token=torch.arange(64, device="cuda")[None]
                    ),
                    token_to_kv_pool_allocator=NS(get_kvcache=lambda: pool),
                )
            )
            self.d = NS(scheduler=NS())
        else:
            from dsv41_remote_probe_support import fixture

            self.sparse = fixture(root, "cuda", kv_layout=model)
            self.p, self.d, self.q = (
                self.sparse.p_controller,
                self.sparse.controller,
                self.sparse.query[0],
            )
            self.p.dsv41_remote = True
        self.engines = []
        for controller, role in ((self.p, "prefill"), (self.d, "decode")):
            if self.transport:
                engine = self.transport.attach(controller, role)
            else:
                engine = factory(role)
                manager = NS(engine=engine, _remote_attn_send_lock=threading.Lock())
                controller.scheduler._pre_layerwise = controller
                controller.gpu_data_plane = dp.Plane(manager, "cuda:0")
            self.engines.append(engine)
        self.hub = rt.RemoteHub(
            self.p.scheduler,
            stream_factory=lambda n: [
                (torch.cuda.Stream(), None, None) for _ in range(n)
            ],
        )
        self.p.scheduler._handoff_hub = self.hub
        self.req = NS(
            kv=NS(req_pool_idx=0),
            cache_request_handle="source",
            disagg_kv_sender=NS(poll=lambda: 3),
        )
        peers = ((NS(mooncake_session_id=self.engines[1].get_session_id()), None),)
        if model == "qwen":
            self.req._pre_layerwise_source = NS(peers=peers)
            self.ctx = self.hub.registry.register(
                "qwen",
                self.req,
                0,
                64,
                64,
                "qwen-token",
                idx_start=0,
                idx_end=64,
                contiguous=True,
                on_release=lambda ctx: self.freed.append(True),
            )
            self.route = dict(
                host=self.hub.host,
                port=self.hub.port,
                handoff_id=self.ctx.handoff_id,
                epoch=self.ctx.owner_epoch,
                token=self.ctx.token,
            )
            self.header = dict(**self.route, layer=0, sm_scale=128**-0.5)
        else:
            from sglang_remote_attn import dsv41_remote as remote

            self.src = NS(
                owner=self.p,
                req=self.req,
                prefix_len=33,
                decode_prefix_len=0,
                generation=model,
                path=Path(self.folder),
                service=None,
                error=None,
                remote_closed=threading.Event(),
                schema=dict(entries=[["kv", 1, 1], ["kv", 2, 2]]),
                peers=peers,
                barrier_ready=True,
                cache_lease=None,
            )
            remote.register(self.src)
            self.route = self.src.service
            self.ctx = self.hub.registry.get(
                self.route["handoff_id"], self.route["epoch"]
            )
            self.ctx.on_release = lambda ctx: self.freed.append(True)
            self.header = dict(
                **self.route,
                dsv41_sparse=True,
                layer=3,
                ratio=2,
                positions=[0, 15],
                scale=512**-0.5,
            )
        self.target_req = NS(kv=NS(req_pool_idx=0))
        self.client = self.d.gpu_data_plane.client(self.route)
        torch.cuda.synchronize()

    def expected(self):
        if self.model == "qwen":
            import flashinfer

            return flashinfer.single_decode_with_kv_cache(
                self.q,
                self.k,
                self.v,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                sm_scale=128**-0.5,
                return_lse=True,
            )
        from sglang_remote_attn import dsv41_sparse_math as math

        f = self.sparse
        ids = torch.tensor([f.p_pages[0] * 4, f.p_pages[3] * 4 + 3], device="cuda")
        keys = math.read_slots(f.p_pool.get_extra_key_buffer(3), ids, 4, f.extra_layout)
        return math.partial(self.q, keys, 512**-0.5)

    def call(self):
        return self.client.call([self.q], [self.header], [self.target_req])

    def close(self, fault=False):
        if not fault:
            self.client.close()
            assert self.hub.registry.release(self.ctx.handoff_id, expected_req=self.req)
            assert self.freed == [True]
        if self.transport:
            self.transport.close()
        else:
            self.d.gpu_data_plane.close()
            self.p.gpu_data_plane.close()
        self.hub.close()
        self.stack.close()


def case(root, model, mode, fault=None, direction="result"):
    stack = ExitStack()
    stack.enter_context(
        patch.dict(
            os.environ,
            {
                "HANDOFF_REMOTE_GPU_TRANSFER": "1",
                "HANDOFF_REMOTE_BACKEND": "rdma" if mode == "gpu" else "auto",
                "HANDOFF_MOONCAKE_HOST": "127.0.0.1",
            },
        )
    )
    engines = []

    def factory(role):
        e = FakeNIC(f"{model}-{mode}-{fault}-{direction}-{role}")
        engines.append(e)
        return e

    if mode in ("gpu", "mixed"):
        stack.enter_context(
            patch.object(dp, "Visibility", return_value=NS(receive=lambda: None))
        )
    if mode == "mixed":
        original_probe = mb.gpu_capability
        stack.enter_context(
            patch.object(
                mb,
                "gpu_capability",
                side_effect=lambda engine, device: (
                    engine.native.name.endswith("prefill")
                    and original_probe(engine, device)
                ),
            )
        )
    f = Fixture(root, model, factory if mode in ("gpu", "mixed") else None)
    try:
        if fault:
            failing = engines[0 if direction == "result" else 1]
            failing.fail = fault
            # Capability registration is pre-transfer; no native write occurs here.
            errors = []

            def call():
                try:
                    torch.cuda.set_device(0)
                    f.call()
                except BaseException as exc:
                    errors.append(repr(exc))

            worker = threading.Thread(target=call)
            worker.start()
            assert failing.entered.wait(8)
            if fault == "cancel":
                assert not dp.ready(f.target_req)
                assert not f.hub.registry.release(f.ctx.handoff_id, expected_req=f.req)
                assert not f.freed
                failing.release.set()
            worker.join(10)
            assert not worker.is_alive() and errors
            assert f.client.slab.quarantined
            with __import__("contextlib").suppress(RuntimeError):
                f.client.close()
            try:
                dp.ready(f.target_req)
                raise AssertionError("unsafe target slot declared reusable")
            except RuntimeError:
                pass
            assert all(e.sync_calls == 0 for e in engines)
            before = [dict(e.regions) for e in engines]
            assert all(before)
            if fault in ("late", "cancel"):
                failing.flush_late()
                assert [e.regions for e in engines] == before
                assert f.client.slab.quarantined
            if direction == "result":
                assert f.req._handoff_transfer_unsafe and not f.freed
            return dict(
                model=model,
                backend=mb.GPU,
                fault=fault,
                direction=direction,
                late_write_retained=True,
                pass_=True,
            )
        expected = f.expected()
        _, out, lse = f.call()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out[0].float(), expected[0].float(), atol=0.015625, rtol=0.01
        )
        torch.testing.assert_close(lse[0], expected[1], atol=0.015625, rtol=0.01)
        assert dp.ready(f.target_req)
        backend = f.client.slab.backend
        wire = f.client.slab.wire
        assert backend == (mb.GPU if mode == "gpu" else mb.TCP)
        assert wire.is_cuda == (mode == "gpu")
        if mode != "gpu":
            assert wire.is_pinned()
        if mode == "mixed":
            assert not any(engine.calls for engine in engines)
            assert mb.GPU in f.p.gpu_data_plane.catalog.engines
            assert mb.GPU not in f.d.gpu_data_plane.catalog.engines
        previous = out.clone()
        _, again, _ = f.call()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, previous)
        torch.testing.assert_close(out, again)
        return dict(
            model=model,
            backend=backend,
            real_cuda=True,
            native_tcp=mode != "gpu",
            mixed_capabilities=mode == "mixed",
            owned_results=True,
            pass_=True,
        )
    finally:
        f.close(fault=bool(fault))
        stack.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "tcp", "faults"), default="all")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    cases = []
    for model in ("qwen", "v4", "v41"):
        if args.suite in ("all", "tcp"):
            cases.append(case(args.source_root, model, "tcp"))
            cases.append(case(args.source_root, model, "mixed"))
        if args.suite in ("all", "faults"):
            cases.append(case(args.source_root, model, "gpu"))
            for fault, direction in (
                ("late", "q"),
                ("late", "result"),
                ("exception", "result"),
                ("cancel", "result"),
            ):
                cases.append(case(args.source_root, model, "gpu", fault, direction))
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
