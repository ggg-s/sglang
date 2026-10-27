"""Measure adaptation overhead, not full-model or distributed throughput.

Uses the production CP scope/view code with synthetic BF16 projections. The
same three GEMMs run in legacy and persistent-view modes. No FlashMLA, MoE,
HiCache DMA or multi-GPU collective cost is represented by these timings.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/registered/unit/multiplex"))
from test_pdmux_cp_decode_attn_tp import fixture


def host_us(fn, loops=5000, repeats=7):
    for _ in range(100):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter_ns() - start) / loops / 1000)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def projection(attention, x):
    groups, rank = attention.n_local_groups, attention.o_lora_rank
    q = torch.nn.functional.linear(x, attention.wq_b.weight)
    q = q.reshape(x.shape[0], groups, -1)
    weight = attention.wo_a.weight.reshape(groups, rank, -1)
    o = torch.einsum("tgd,grd->tgr", q, weight)
    return torch.nn.functional.linear(o.flatten(1), attention.wo_b.weight)


def timed_cuda(fn, iterations):
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    wall = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return {
        "wall_us": (time.perf_counter_ns() - wall) / iterations / 1000,
        "gpu_timeline_us": start.elapsed_time(end) * 1000 / iterations,
    }


def runtime_dispatch_cost():
    """Real runtime getters; no distributed groups or server are initialized."""
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers import dp_attention as dp
    from sglang.srt.layers.cp import utils as cp
    from sglang.srt.runtime_context import ForwardFlags, get_context, get_forward

    ctx = get_context()
    result = {}
    with (
        patch.object(ctx, "prefill_forward", ForwardFlags()),
        patch.object(ps, "_PDMUX_PREFILL_TP_GROUP", object()),
    ):
        ps.set_pdmux_status(False)
        dp.set_dp_buffer_len(1024, 1024, False, [1024])
        result["forward_original_direct"] = host_us(lambda: ctx.forward)
        result["forward_pdmux_decode"] = host_us(get_forward)
        result["dp_original_direct"] = host_us(
            dp._DpGatheredBufferWrapper.get_local_dp_buffer_len
        )
        result["dp_pdmux_decode"] = host_us(dp.get_local_dp_buffer_len)
        ps.set_pdmux_status(True)
        result["forward_pdmux_prefill"] = host_us(get_forward)
        cached_batch = NS(
            seq_lens_cpu=torch.tensor([4096]),
            extend_seq_lens_cpu=[1024],
            attn_cp_metadata=NS(per_rank_actual_token=[512, 512]),
            global_num_tokens_cpu=[1024],
            out_cache_loc=torch.arange(1024, device="cuda"),
        )
        with (
            patch.object(cp, "is_cp_active", lambda _: True),
            patch.object(cp, "get_cp_strategy", lambda: NS()),
        ):
            result["cp_cached_metadata_prepare"] = host_us(
                lambda: cp.prepare_cp_forward(cached_batch)
            )
            result["cp_physical_length_setter_only"] = host_us(
                lambda: dp.set_local_dp_buffer_len(
                    sum(cached_batch.attn_cp_metadata.per_rank_actual_token)
                )
            )
        ps.set_pdmux_status(False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cp-size", type=int, default=2, choices=(2, 4, 8))
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(42)
    result = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "scope": "synthetic projection/scope overhead; no full-model performance claim",
        "cp_size": args.cp_size,
        "dimensions": dict(
            heads=64,
            head_dim=512,
            q_rank=1024,
            groups=8,
            o_rank=1024,
            hidden=4096,
            dtype="bfloat16",
        ),
    }
    # Import the runtime before patch.dict snapshots sys.modules. Otherwise
    # lazy Torch imports inside the fixture can be removed on restoration,
    # causing duplicate c10d operator registration when the runtime imports.
    try:
        result["runtime_dispatch"] = runtime_dispatch_cost()
    except Exception as exc:
        result["runtime_dispatch_unavailable"] = f"{type(exc).__name__}: {exc}"
    model, _, linear_module = fixture(size=args.cp_size)
    model = model.to("cuda")
    model.n_heads = model.n_local_heads = 64
    model.n_groups = model.n_local_groups = 8
    model.o_lora_rank = 1024
    model.attn_mqa.tp_q_head_num = 64
    for name, rows, cols in (
        ("wq_b", 32768, 1024),
        ("wo_a", 8192, 4096),
        ("wo_b", 4096, 8192),
    ):
        layer = getattr(model, name)
        layer.weight = torch.nn.Parameter(
            torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) / 64,
            requires_grad=False,
        )
        if name == "wo_b":
            layer.input_size_per_partition = cols
        else:
            layer.output_size_per_partition = rows
    model.attn_sink = torch.nn.Parameter(
        torch.zeros(64, device="cuda"), requires_grad=False
    )
    mode = [False]
    globals_ = model.maybe_use_decode_attn_tp.__wrapped__.__globals__
    globals_["is_pdmux_enabled"] = lambda: mode[0]
    batch = NS(cp_active=False)

    def scope_only():
        with model.maybe_use_decode_attn_tp(batch) as selected:
            return selected.n_local_heads

    result["cold"] = {}
    with patch.dict(sys.modules, {linear_module.__name__: linear_module}):
        for name, enabled in (("legacy", False), ("persistent_view", True)):
            mode[0] = enabled
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            start = time.perf_counter_ns()
            scope_only()
            torch.cuda.synchronize()
            result["cold"][name] = {
                "wall_us": (time.perf_counter_ns() - start) / 1000,
                "added_cuda_bytes": torch.cuda.memory_allocated() - allocated,
            }
        result["host_scope"] = {}
        for name, enabled in (("legacy", False), ("persistent_view", True)):
            mode[0] = enabled
            allocated = torch.cuda.memory_allocated()
            result["host_scope"][name] = host_us(scope_only)
            result["host_scope"][name]["steady_cuda_bytes_delta"] = (
                torch.cuda.memory_allocated() - allocated
            )

        result["projection"] = {}
        for rows in (1, 8, 32, 128):
            x = torch.randn(rows, 1024, device="cuda", dtype=torch.bfloat16)

            def forward(x=x):
                with model.maybe_use_decode_attn_tp(batch) as selected:
                    return projection(selected, x)

            graphs, reference = {}, None
            for name, enabled in (("legacy", False), ("persistent_view", True)):
                mode[0] = enabled
                for _ in range(10):
                    value = forward()
                torch.cuda.synchronize()
                if reference is None:
                    reference = value.clone()
                else:
                    torch.testing.assert_close(value, reference)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    value = forward()
                graphs[name] = (enabled, graph, value)
            samples = {name: {"eager": [], "graph": []} for name in graphs}
            for repeat in range(args.repeats):
                order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
                for name in order:
                    enabled, graph, value = graphs[name]
                    mode[0] = enabled
                    samples[name]["eager"].append(timed_cuda(forward, args.iterations))
                    samples[name]["graph"].append(
                        timed_cuda(graph.replay, args.iterations)
                    )
                    torch.testing.assert_close(value, reference)
            summary = {}
            for name, kinds in samples.items():
                summary[name] = {
                    kind: {
                        metric: statistics.median(s[metric] for s in values)
                        for metric in ("wall_us", "gpu_timeline_us")
                    }
                    for kind, values in kinds.items()
                }
            result["projection"][rows] = {"median": summary, "samples": samples}
            del graphs, samples, reference, value, graph, x
            gc.collect()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "projection"},
            indent=2,
        )
    )
    for rows, values in result["projection"].items():
        print(rows, json.dumps(values["median"]))


if __name__ == "__main__":
    main()
