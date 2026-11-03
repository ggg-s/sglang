"""CUDA/Mooncake TCP bounded admission and quarantined-source recovery.

Real copies/transport, one GPU and two endpoints. Error returns are injected at
the engine-call boundary; this is not a serving, H20, or late-DMA benchmark.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.sglang_root.resolve()
    os.environ.update(
        MC_FORCE_TCP="1",
        MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER="8",
        MC_TCP_MAX_PENDING_ADMISSIONS_PER_PEER="8",
    )
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.disaggregation",
        "sglang.srt.disaggregation.mooncake",
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(root / "python" / name.replace(".", "/"))]
        sys.modules[name] = package
    from mooncake.engine import TransferEngine

    from sglang.srt.disaggregation.mooncake.pinned_staging_core import CorePinnedStager
    from sglang.srt.disaggregation.pinned_staging_buffer import StagingTransportError

    torch.cuda.set_device(0)

    class Engine:
        def __init__(self):
            self.native = TransferEngine()
            assert self.native.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "") == 0
            self.session = f"127.0.0.1:{self.native.get_rpc_port()}"
            self.fail = False
            self.calls = []

        def register(self, ptr, size):
            assert self.native.register_memory(ptr, size) == 0

        def deregister(self, ptr):
            assert self.native.unregister_memory(ptr) == 0

        def batch_transfer_sync(self, peer, src, dst, size):
            self.calls.append(len(src))
            if self.fail:
                self.fail = False
                return -1
            return self.native.batch_transfer_sync_write(peer, src, dst, size)

    sender, receiver = Engine(), Engine()
    src = (torch.arange(16384, dtype=torch.int32, device="cuda") % 251).to(torch.uint8)
    dst = torch.zeros_like(src)
    torch.cuda.synchronize()
    sender.register(src.data_ptr(), src.nbytes)
    receiver.register(dst.data_ptr(), dst.nbytes)
    stager = CorePinnedStager(sender, capacity=65536)
    stager.set_buffers([src], [src.data_ptr()], [8])
    stager.reserve(src.device)
    conn = root / "python/sglang/srt/disaggregation/mooncake/conn.py"
    tree = ast.parse(conn.read_text())
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MooncakeKVManager"
    )
    method = next(n for n in owner.body if getattr(n, "name", None) == "_transfer_data")
    owner.bases, owner.keywords, owner.body = [], [], [method]
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[owner], type_ignores=[])),
            str(conn),
            "exec",
        ),
        namespace,
    )
    manager = namespace[owner.name]()
    manager.engine, manager._core_stager, manager._host_staging_all = (
        sender,
        stager,
        True,
    )
    # 2048 descriptors with queue/admission limits of 8. The real native
    # _transfer_data must submit <=4 descriptors per synchronous engine call.
    blocks = [(src.data_ptr() + i * 8, dst.data_ptr() + i * 8, 8) for i in range(2048)]
    assert manager._transfer_data(receiver.session, blocks) == 0
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src, rtol=0, atol=0)
    bounded_calls = len(sender.calls)
    assert bounded_calls == 512 and max(sender.calls) == 4
    retained = []
    for i in range(8):
        src.fill_(i + 1)
        torch.cuda.synchronize()
        sender.fail = True
        assert manager._transfer_data(receiver.session, blocks[:1]) == -1
        old = stager.buffer.slot
        retained.append((old, old.host[:8].clone()))
        src.fill_(100 + i)
        torch.cuda.synchronize()
        if i == 7:
            try:
                manager._transfer_data(receiver.session, blocks[:1])
            except StagingTransportError:
                pass
            else:
                raise AssertionError("exhausted budget must not send GPU sources")
        else:
            assert manager._transfer_data(receiver.session, blocks[:1]) == 0
        torch.cuda.synchronize()
        if i != 7:
            torch.testing.assert_close(dst[:8], src[:8], rtol=0, atol=0)
        for slot, expected in retained:
            torch.testing.assert_close(slot.host[:8], expected, rtol=0, atol=0)
    assert stager.disabled and len(stager.retired) == 8
    # Exhaustion fails the staged request without any additional engine call.
    src.fill_(219)
    torch.cuda.synchronize()
    calls_before = len(sender.calls)
    try:
        manager._transfer_data(receiver.session, blocks)
    except StagingTransportError:
        pass
    else:
        raise AssertionError("exhausted budget must reject staged sends")
    assert len(sender.calls) == calls_before
    paths = [
        "python/sglang/srt/disaggregation/pinned_staging_buffer.py",
        "python/sglang/srt/disaggregation/mooncake/pinned_staging_core.py",
        "python/sglang/srt/disaggregation/mooncake/conn.py",
        "test/manual/remote_attn/staging_recovery_cuda_probe.py",
    ]
    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        transport="Mooncake TCP",
        queue_limit=8,
        input_descriptors=2048,
        bounded_calls=bounded_calls,
        max_descriptors=4,
        injected_failures=8,
        retired_slots=len(stager.retired),
        retired_bytes=sum(b.slot.capacity for b in stager.retired),
        retained_bytes_unchanged=True,
        budget_rejected_without_send=True,
        exact_data_match=True,
        source_sha256={
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
        },
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))
    # Quarantined registrations intentionally survive until process exit.


if __name__ == "__main__":
    main()
