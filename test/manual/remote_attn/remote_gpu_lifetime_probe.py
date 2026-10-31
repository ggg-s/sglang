"""4060: actual GPU reader fence, native Mooncake payloads and cache/slot lifetime.

Tiny native byte pools and cache lock boundary; no full model or RDMA hardware.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from dsv41_remote_probe_support import fixture
from remote_gpu_probe_support import Transport
from sglang_remote_attn import dsv41_remote as remote
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import remote_data_plane as dp
from sglang_remote_attn.dsv41_layerwise import DSV41Source
from sglang_remote_attn.source_cache import CacheGuard


def main(root, output):
    torch.cuda.set_device(0)
    sys.modules["sglang.srt.disaggregation.base.conn"] = SimpleNamespace(
        KVPoll=SimpleNamespace(Success=4, Failed=0)
    )
    transport = Transport(root)
    f = fixture(root, "cuda")
    f.p_controller.dsv41_remote = True
    transport.attach(f.p_controller, "prefill")
    d_engine = transport.attach(f.controller, "decode")
    hub = rt.RemoteHub(
        f.p_controller.scheduler,
        stream_factory=lambda n: [(torch.cuda.Stream(), None, None) for _ in range(n)],
    )
    f.p_controller.scheduler._handoff_hub = hub
    pins = [0]

    def lock(node):
        pins[0] += 1
        return object()

    def unlock(receipt):
        pins[0] -= 1

    cache = SimpleNamespace(
        lock=lock,
        unlock=unlock,
        checkpoint=lambda *a, **kw: None,
        finish=lambda *a: None,
        reset=lambda: None,
    )
    guard = CacheGuard(cache)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:

        def source(name):
            req = SimpleNamespace(
                kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=8),
                cache_request_handle=name,
                lock=object(),
                last_node="prefix",
                disagg_kv_sender=SimpleNamespace(poll=lambda: 3),
            )
            src = DSV41Source(
                f.p_controller,
                req,
                Path(temp) / name,
                name,
                None,
                None,
                (
                    (
                        SimpleNamespace(mooncake_session_id=d_engine.get_session_id()),
                        None,
                    ),
                ),
                33,
                cache_lease=guard.hold(req),
                schema=dict(entries=[["kv", 3, 2]]),
                barrier_ready=True,
            )
            remote.register(src)
            return src

        src = source("cancel")
        req = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))
        target = remote.Target(
            f.controller, req, src.path, src.generation, src.service, 33
        )
        f.controller.remote_targets[0] = target
        warm = torch.cuda.Stream()
        with torch.cuda.stream(warm):
            torch.cuda._sleep(1_000_000_000)
            gpu_ready = torch.cuda.Event()
            gpu_ready.record(warm)
        ctx = hub.registry.get(src.service["handoff_id"], src.service["epoch"])
        ctx.dsv41_ready = gpu_ready
        errors, results = [], []

        def call():
            try:
                torch.cuda.set_device(0)
                results.append(target.call(f.query[0], [0, 15], 3, 2, 512**-0.5))
            except Exception as exc:
                errors.append(repr(exc))

        worker = threading.Thread(target=call)
        worker.start()
        deadline = time.monotonic() + 5
        while not ctx.inflight:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        assert not dp.ready(req)
        try:
            target.close("too-early")
            raise AssertionError("unproven GPU consumer retired")
        except RuntimeError:
            pass
        assert not hub.registry.release(ctx.handoff_id, expected_req=src.req)
        assert pins == [1] and not src.remote_closed.is_set()
        warm.synchronize()
        worker.join(5)
        assert not worker.is_alive() and not errors, errors
        torch.cuda.synchronize()
        assert dp.ready(req)
        target.close("gpu-drained")
        deadline = time.monotonic() + 5
        while not src.remote_closed.is_set():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        src.cache_lease.retire(lambda: None)
        assert pins == [0]
        old_address = f.controller.gpu_data_plane.client(
            src.service
        ).slab.tensor.data_ptr()
        again = source("reuse")
        req2 = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))
        other = remote.Target(
            f.controller, req2, again.path, again.generation, again.service, 33
        )
        f.controller.remote_targets[0] = other
        got = other.call(f.query[0], [0, 15], 3, 2, 512**-0.5)
        torch.cuda.synchronize()
        torch.testing.assert_close(got[0], results[0][0])
        assert (
            f.controller.gpu_data_plane.client(again.service).slab.tensor.data_ptr()
            == old_address
        )
        other.close("complete")
        assert remote.source_ready(again)
        again.cache_lease.retire(lambda: None)
        assert pins == [0]
        details = transport.close()
        hub.close()
    paths = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py"))
    paths += [
        Path(__file__),
        Path(__file__).with_name("remote_gpu_probe_support.py"),
        Path(__file__).with_name("dsv41_remote_probe_support.py"),
    ]
    paths += [
        root / p
        for p in (
            "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py",
            "python/sglang/kernels/ops/attention/dsv4/kv_layout.py",
            "python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py",
            "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
            "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
        )
    ]
    result = dict(
        device=torch.cuda.get_device_name(),
        gpu_reader_cancel_retained=True,
        early_target_close_rejected=True,
        gpu_consumer_proven=True,
        source_cache_pins_closed=True,
        slot_reused_after_positive_completion=True,
        transport=details,
        scope=__doc__,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root.resolve(), args.output.resolve())
