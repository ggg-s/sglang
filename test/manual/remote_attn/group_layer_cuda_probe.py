"""Group-layer CUDA probes; real transport/kernels, explicit scheduler ABI fixtures."""

import argparse
import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import torch


def qwen(root, output):
    from cuda_handoff_probe import LAYERS, transfer_probe

    cases = [
        ("fast-async", "float16", 2, 4, 2, 1),
        ("fast-async", "bfloat16", 3, 3, 3, 1),
        ("fast-sync", "float16", 4, 4, 4, 1),
        ("fast-async", "float16", 1, 3, 4, 1),
        ("fast-async", "bfloat16", 0, 6, 3, 1),
        ("cpu-layer", "bfloat16", 0, 1, 2, 1),
        ("cpu-layer", "float16", 0, 1, 4, 2),
    ]
    results = []
    for index, (mode, dtype, slots, buffers, width, pipeline) in enumerate(cases):
        folder = output.parent / f"qwen-case-{index}"
        folder.mkdir(exist_ok=True)
        os.environ["HANDOFF_TRANSFER_PIPELINE"] = str(pipeline)
        result = transfer_probe(mode, dtype, slots, buffers, folder, group_size=width)
        batches = result["source"]["native_batches"]
        effective = (
            min(width, slots or LAYERS)
            if mode.startswith("fast")
            else min(width, LAYERS)
        )
        assert [b["descriptors"] for b in batches] == [
            2 * min(effective, LAYERS - lo) for lo in range(0, LAYERS, effective)
        ], batches
        result.update(
            requested_group_size=width,
            effective_group_size=effective,
            native_batches_count=len(batches),
            layers=LAYERS,
        )
        results.append(result)
        output.write_text(json.dumps(dict(status="RUNNING", cases=results), indent=2))
    for cancel, remote_drop in ((False, True), (True, False)):
        folder = output.parent / ("qwen-cancel" if cancel else "qwen-remote-drop")
        folder.mkdir(exist_ok=True)
        result = transfer_probe(
            "fast-async",
            "float16",
            2,
            4,
            folder,
            cancel=cancel,
            remote_drop=remote_drop,
            group_size=2,
        )
        results.append(result)
    return results


def dsv_local(root):
    from dsv_d2d_probe_support import Environment
    from sglang_remote_attn import dsv41_receive as receive

    results = []
    for layout in ("v4", "v41"):
        e = Environment(root, "cuda", 2, False, kv_layout=layout)
        release, second = threading.Event(), threading.Event()
        try:
            e.gate.set()
            original, calls = receive.scatter_prefix, []

            def scatter(*a, **kw):
                calls.append(True)
                from sglang_remote_attn.dsv41_staging import buffers
                from sglang_remote_attn.dsv_d2d import manager

                target = e.target.d2d_targets[e.epoch]
                last_entry = target.resources[-1].entry
                last_buffer = buffers(e.target.pool, manager(e.target))[last_entry]
                dst, src, _pages, start = a[:4]
                final_page = (
                    e.prefix + e.target.pool.page_size - 1
                ) // e.target.pool.page_size
                if (
                    dst.data_ptr() == last_buffer.data_ptr()
                    and start + src.shape[0] == final_page
                ):
                    second.set()
                    assert release.wait(20)
                return original(*a, **kw)

            with patch.dict(os.environ, HANDOFF_D2D_GROUP_SIZE="4"):
                meta = e.reserve_and_freeze()
            with patch.object(receive, "scatter_prefix", side_effect=scatter):
                e.import_(meta)
                target = e.target.d2d_targets[e.epoch]
                assert second.wait(15)
                e.command(e.target, "d2d_commit")
                first, last = target.resources
                target.events[first.entry].synchronize()
                assert first.ready_to_finish() and not last.ready_to_finish()
                e.target.decode_overlap.retirement.drain(first)
                assert first.entry in target.rpc.local_entries
                assert last.entry not in target.rpc.local_entries
                assert not target.closed and e.req.kv.holds_kv
                release.set()
                e.wait_copy()
            e.target.drain_targets(required=target)
            source = e.source.d2d_sources[e.epoch]
            deadline = time.monotonic() + 5
            while not source.poll():
                assert time.monotonic() < deadline
                time.sleep(0.001)
            source.retire()
            assert e.freed == ["source"]
            results.append(
                dict(
                    layout=layout,
                    group_size=4,
                    entries_per_frame=target.grouped_entries,
                    independent_entry_local=True,
                    source_free_count=1,
                )
            )
        finally:
            release.set()
            e.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("qwen", "dsv-local"), required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    # Resolve lazy DTensor test helpers before source-loaded ABI fixture imports.
    torch.testing.assert_close(torch.zeros(1), torch.zeros(1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = (
        qwen(args.source_root, args.output)
        if args.suite == "qwen"
        else dsv_local(args.source_root)
    )
    report = dict(
        status="PASS",
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        suite=args.suite,
        cases=cases,
        scope=__doc__,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", suite=args.suite, cases=len(cases))))


if __name__ == "__main__":
    main()
