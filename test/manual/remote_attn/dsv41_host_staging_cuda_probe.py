"""Real CUDA + Mooncake TCP for DSV4.1 native byte plans and P host staging.

One GPU, two TE endpoints; small KV/index fixtures, no model/server or NCCL.
Source-load native methods to avoid importing unrelated model dependencies.
Run --sglang-root ROOT --output result.json with PYTHONPATH=plugins/remote_attn.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from sglang_remote_attn import dsv41_layerwise as ds
from sglang_remote_attn import dsv41_staging as staging


def extract(path, name, methods=None):
    node = next(
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == name
    )
    if methods is not None:
        node.bases, node.keywords = [], []
        node.body = [n for n in node.body if getattr(n, "name", None) in methods]
    namespace = dict(np=np, torch=torch)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["MC_FORCE_TCP"] = "1"
    torch.cuda.set_device(0)
    root = args.sglang_root.resolve()
    native = root / "python/sglang/srt"
    # Only native pinned staging modules are imported; no model import stubs.
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

    class Engine:
        def __init__(self):
            self.native = TransferEngine()
            assert self.native.initialize("127.0.0.1", "P2PHANDSHAKE", "rdma", "") == 0
            self.session = f"127.0.0.1:{self.native.get_rpc_port()}"
            self.calls = []
            self.fail = False

        def register(self, ptr, size):
            assert self.native.register_memory(ptr, size) == 0

        def deregister(self, ptr):
            assert self.native.unregister_memory(ptr) == 0

        def batch_transfer_sync(self, session, sources, targets, sizes):
            self.calls.append(list(zip(sources, targets, sizes)))
            if self.fail:
                return -1  # Failure injection, not a simulated success path.
            return self.native.batch_transfer_sync_write(
                session, sources, targets, sizes
            )

    sender, receiver = Engine(), Engine()
    pool_cls = extract(
        native / "mem_cache/deepseek_v4_memory_pool.py",
        "DeepSeekV4TokenToKVPool",
        {"get_contiguous_buf_infos"},
    )
    index_cls = extract(
        native / "mem_cache/deepseek_v4_memory_pool.py",
        "DeepSeekV4IndexerPool",
        {"contiguous_page_row_buffers"},
    )
    pool = pool_cls()
    pool._unified_kv = pool._unified_kv_fp8 = pool.uniform_fp8 = False
    pool.page_size = 8
    pool.compression_ratios = [0, 0] + [2] * 18 + [1] * 20
    pool._stage_start, pool._stage_end = 0, 40
    pool.sources_by_ratio = {2: [2, 8, 14], 1: [20]}
    pool.kv_pools, pool.index_pools = {}, {}
    tensors = []
    for ratio, sources in pool.sources_by_ratio.items():
        kv = [torch.zeros((8, 256), dtype=torch.uint8, device="cuda") for _ in sources]
        pool.kv_pools[ratio] = types.SimpleNamespace(kv_buffer=kv)
        index = index_cls()
        index.page_size, index.index_k_with_scale_buffer = 2, None
        rows = 8 * pool.page_size // ratio // index.page_size
        index.index_k_payload_buffer = [
            torch.zeros((rows, 16), dtype=torch.uint8, device="cuda") for _ in sources
        ]
        index.index_k_scale_buffer = [
            torch.zeros((rows, 4), dtype=torch.uint8, device="cuda") for _ in sources
        ]
        pool.index_pools[ratio] = index
        tensors += kv + index.contiguous_page_row_buffers()
    ptrs, lengths, strides = pool.get_contiguous_buf_infos()
    assert ptrs == [t.data_ptr() for t in tensors]
    destinations = [
        torch.zeros(t.nbytes, dtype=torch.uint8, device="cuda") for t in tensors
    ]
    for tensor in tensors:
        sender.register(tensor.data_ptr(), tensor.nbytes)
    for tensor in destinations:
        receiver.register(tensor.data_ptr(), tensor.nbytes)
    conn = native / "disaggregation/mooncake/conn.py"
    cls = extract(
        conn,
        "MooncakeKVManager",
        {
            "ensure_host_stager",
            "set_kv_buffer_tensors",
            "_transfer_data",
            "_send_kvcache_generic",
        },
    )
    cls.get_mla_kv_ptrs_with_pp = extract(
        native / "disaggregation/common/conn.py",
        "CommonKVManager",
        {"get_mla_kv_ptrs_with_pp"},
    ).get_mla_kv_ptrs_with_pp
    generic = cls._send_kvcache_generic.__globals__
    group = extract(
        native / "disaggregation/common/utils.py", "group_concurrent_contiguous"
    )
    flat = extract(conn, "_uses_flat_entry_layout")
    state_types = types.SimpleNamespace(DSA="DSA", SWA="SWA")
    flat.__globals__["StateType"] = state_types
    generic.update(
        group_concurrent_contiguous=group,
        _uses_flat_entry_layout=flat,
        StateType=state_types,
        is_elided_entry=extract(native / "disaggregation/utils.py", "is_elided_entry"),
    )
    manager = cls()
    manager.engine, manager._host_staging_all = sender, False
    manager._core_stager = CorePinnedStager(sender, capacity=257)
    manager.kv_args = types.SimpleNamespace(
        kv_data_ptrs=ptrs, kv_data_lens=lengths, kv_item_lens=strides
    )
    manager.is_mla_backend, manager.is_hybrid_mla_backend = True, False
    manager.attn_tp_size, manager.pp_size = 1, 1
    manager.enable_staging = manager.enable_custom_mem_pool = False
    manager.max_transfer_batch_indices = 2
    manager._validate_envelope_kv_layout = lambda *a: None
    dst_ptrs = [t.data_ptr() for t in destinations]
    manager.decode_kv_args_table = {
        receiver.session: types.SimpleNamespace(
            dst_kv_ptrs=dst_ptrs, requires_dcp_relayout=False
        )
    }
    entries = []
    for ratio, sources in pool.sources_by_ratio.items():
        for kind in ("kv", "index", "index"):
            for producer in sources:
                entries.append(
                    [kind, ratio, producer, "fp4", "torch.uint8", strides[len(entries)]]
                )
    manager._dsv41_layerwise_schema = {"entries": entries}
    stager = staging.bind(pool, manager)
    slot = stager.buffer.slot
    assert slot.host.is_pinned()
    assert len(stager.regions) == 12
    assert stager.capacity == 257
    assert slot.stream.cuda_stream != torch.cuda.default_stream().cuda_stream
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    ready = torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(150_000_000)
        for i, tensor in enumerate(tensors):
            tensor.fill_(i + 11)
        ready.record(stream)
    pending = not ready.query()
    assert pending, "producer completed before pending-state check"
    completed = threading.Event()
    source_pages = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    target_pages = np.array([5, 6, 1, 2, 3], dtype=np.int32)

    def worker():
        # The native worker performs exactly this event wait before send_selected.
        ready.synchronize()
        groups = ds.resource_groups(manager._dsv41_layerwise_schema, 1)
        for _, selected in groups:
            rc = ds.send_selected(
                None,
                manager,
                receiver.session,
                source_pages,
                dst_ptrs,
                target_pages,
                None,
                dst_attn_tp_size=1,
                kv_buffer_indices=selected,
                transfer_fn=staging.transfer(manager),
            )
            assert rc == 0
        completed.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        work = executor.submit(worker)
        assert not completed.is_set()
        work.result(timeout=30)
    torch.cuda.synchronize()
    for src, dst, stride in zip(tensors, destinations, strides):
        source = src.view(torch.uint8).reshape(-1)
        for sp, dp in zip(source_pages, target_pages):
            assert torch.equal(
                source[sp * stride : (sp + 1) * stride],
                dst[dp * stride : (dp + 1) * stride],
            )
        assert not dst[:stride].any()
    staged_calls = list(sender.calls)
    assert all(
        slot.ptr <= p and p + n <= slot.ptr + slot.capacity
        for blocks in staged_calls
        for p, _, n in blocks
    )
    assert all(sum(n for _, _, n in blocks) <= 257 for blocks in staged_calls)
    assert stager.buffer.slot is slot
    # Native host-staging-all also reuses this binding for ordinary middle KV.
    manager._host_staging_all = True
    before = len(sender.calls)
    assert (
        manager._transfer_data(receiver.session, [(ptrs[0], dst_ptrs[0], strides[0])])
        == 0
    )
    assert slot.ptr <= sender.calls[before][0][0] < slot.ptr + slot.capacity
    manager._host_staging_all = False
    # Native small state is outside global regions and keeps direct state sends.
    state = torch.full((32,), 99, dtype=torch.uint8, device="cuda")
    state_dst = torch.zeros_like(state)
    sender.register(state.data_ptr(), state.nbytes)
    receiver.register(state_dst.data_ptr(), state_dst.nbytes)
    torch.cuda.synchronize()
    manager._host_staging_all = True
    assert (
        manager._transfer_data(
            receiver.session, [(state.data_ptr(), state_dst.data_ptr(), state.nbytes)]
        )
        == 0
    )
    torch.cuda.synchronize()
    assert torch.equal(state, state_dst)
    assert sender.calls[-1][0][0] == state.data_ptr()
    # Error path must quarantine the original registered slot without fallback.
    sender.fail = True
    before = len(sender.calls)
    rc = manager._transfer_data(
        receiver.session, [(ptrs[0], dst_ptrs[0], strides[0])], stage=True
    )
    assert rc == -1 and stager.poisoned
    assert len(sender.calls) == before + 1
    assert not stager.close()
    assert stager in sender._core_staging_owners and stager.buffer.slot is slot
    failed_again = False
    try:
        manager._transfer_data(
            receiver.session, [(ptrs[0], dst_ptrs[0], strides[0])], stage=True
        )
    except RuntimeError:
        failed_again = True
    assert failed_again and len(sender.calls) == before + 1
    paths = list((root / "plugins/remote_attn/sglang_remote_attn").glob("*.py")) + [
        Path(__file__).resolve()
    ]
    paths += [
        native / p
        for p in (
            "disaggregation/pinned_staging_buffer.py",
            "disaggregation/mooncake/pinned_staging_core.py",
            "disaggregation/mooncake/conn.py",
            "disaggregation/common/conn.py",
            "disaggregation/common/utils.py",
            "disaggregation/utils.py",
            "mem_cache/deepseek_v4_memory_pool.py",
        )
    ]
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        entries=len(tensors),
        native_strides=strides,
        producer_pending=pending,
        bytes_exact=True,
        bounded_host_batches=len(staged_calls),
        slot_capacity=slot.capacity,
        nonblocking_copy_stream=True,
        ordinary_middle_reuses_slot=True,
        final_state_direct=True,
        failed_slot_quarantined=True,
        failed_retry_no_te=True,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        scope="real CUDA+Mooncake TCP, two endpoints on one GPU; toy pool, source-loaded native planners; no DSV4.1 model, full scheduler, NCCL or performance claim",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
