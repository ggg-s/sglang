"""Real GDN/QSA/MoE/PLE comparison of eager and BCG prefill dispatch.

Runs production BCG capture, layer markers, interval replay and continuation
executor with explicit TP1 metadata adapters. No server/checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import qwen38_small_decoder as small
import torch
import torch.nn as nn
import validate_qwen38_moe_ple_cuda as real
import validate_qwen38_real_ops_cuda as ops

NS = types.SimpleNamespace
BASE = "python/sglang/srt/model_executor/"


def load_bcg():
    """Compile original weak-ref ATen implementation against this Torch ABI."""
    from torch.utils.cpp_extension import load

    rel = "python/sglang/kernels/aot/csrc/memory/weak_ref_tensor.cpp"
    source = ops.ROOT / rel
    ops.SOURCES[rel] = hashlib.sha256(source.read_bytes()).hexdigest()
    registration = Path(tempfile.gettempdir()) / "pdmux_bcg_weak_ref_registration.cpp"
    registration.write_text(
        "#include <ATen/ATen.h>\n#include <torch/library.h>\n"
        "at::Tensor weak_ref_tensor(const at::Tensor&);\n"
        "TORCH_LIBRARY(pdmux_bcg_validation, m) {\n"
        'm.def("weak_ref_tensor(Tensor x) -> Tensor");\n'
        "}\nTORCH_LIBRARY_IMPL(pdmux_bcg_validation, CUDA, m) {\n"
        'm.impl("weak_ref_tensor", &weak_ref_tensor); }\n'
    )
    load(
        "pdmux_bcg_native_weak_ref",
        sources=[str(source), str(registration)],
        extra_cflags=["-O2"],
        is_python_module=False,
    )
    small.selected(
        "sglang.srt.model_executor.runner_backend_utils.weak_ref_tensor",
        BASE + "runner_backend_utils/weak_ref_tensor.py",
        {},
        dict(
            torch=torch, weak_ref_tensor=torch.ops.pdmux_bcg_validation.weak_ref_tensor
        ),
        functions=("weak_ref_tensors",),
    )
    cu = ops.load(
        "bcg_cuda_utils",
        BASE + "runner_backend_utils/breakable_cuda_graph/cuda_utils.py",
    )
    bcg = ops.load(
        "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph",
        BASE + "runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py",
        checkCudaErrors=cu.checkCudaErrors,
        get_device_module=lambda: torch.cuda,
        is_hip=lambda: False,
        is_musa=lambda: False,
        is_xpu=lambda: False,
    )
    ctx = ops.load(
        "bcg_live_context",
        BASE + "runner_backend_utils/breakable_cuda_graph/context.py",
        Backend=NS(BREAKABLE="breakable"),
        PREFILL_CUDA_GRAPH_CAPTURE_FAILED_MSG="BCG {backend}: {suggestions}",
    )
    executor = ops.load("bcg_continuation", BASE + "runner_utils/pdmux_bcg.py")
    dispatch = ops.load(
        "bcg_split_dispatch",
        BASE + "runner_utils/pdmux.py",
        device_timer_ctx=lambda *a: contextlib.nullcontext(),
    )
    executor.run_split_prefill = dispatch.run_split_prefill
    # CP is excluded by the admitted plain-TP profile. Only this metadata
    # boundary is supplied; dispatch and all model/kernel bodies are original.
    cp = types.ModuleType("sglang.srt.layers.cp.utils")
    cp.is_cp_active = lambda batch: False
    cp.prepare_cp_forward = lambda batch: None
    sys.modules[cp.__name__] = cp
    shape = ops.load("bcg_shape_key", BASE + "runner/shape_key.py")
    small.selected(
        "sglang.srt.model_executor.runner_utils.shared_read_event",
        BASE + "runner_utils/shared_read_event.py",
        {},
        dict(SharedReadEnds=NS(PRE_REPLAY=object())),
        functions=("maybe_publish_prefill_shared_read_done",),
    )
    return bcg, ctx, executor, shape


def build(fp8, bcg, ctx):
    stack, gdn, qsa = small.build_stack()
    qsa.canonical_topk = True
    moe = real.load_moe()
    ple = real.load_ple(fp8, bcg, ctx)
    env = stack.forward.__func__.__globals__
    env.update(
        {
            name: value
            for name, value in ple.src.__dict__.items()
            if name.startswith("_breakable_")
            or name in ("_BreakablePLEBatch", "_prepare_ple_batch", "_commit_ple_batch")
        }
    )
    env.update(
        Qwen2MoeSparseMoeBlock=moe.block.Qwen2MoeSparseMoeBlock,
        is_in_breakable_cuda_graph=ctx.is_in_breakable_cuda_graph,
        _current_capture_var=bcg._current_capture_var,
    )
    stack.has_ple = True
    stack.ple_ngram_size, stack.ple_ngram_eos_token_id = 3, 0
    for i, layer in enumerate(stack.layers):
        shared = layer.mlp
        layer.mlp = real.make_moe(moe, fp8)
        layer.mlp.shared_expert = shared
        layer.mlp.shared_expert_gate = nn.Linear(
            256, 1, bias=False, device="cuda", dtype=torch.bfloat16
        )
        layer.ple = ple.layers[i]
    moe.block.__dict__["fused_gate_sigmoid_mul_add"] = (
        qsa.m.elem.fused_gate_sigmoid_mul_add
    )
    # The production BCG indexer bridge surrounds the same real scoring/top-k
    # kernels as the earlier fixture; complete compressor updates remain out
    # of scope. This tests its live-context eager break as well as captured math.
    selected = small.selected(
        "bcg_qsa_dispatch",
        "python/sglang/srt/models/qwen4_exp.py",
        {"Qwen4ExpAttentionDecoderLayer": ([], {"_compute_qsa_topk_indices"})},
        dict(
            env,
            torch=torch,
            eager_on_graph=bcg.eager_on_graph,
            get_tc_piecewise_forward_context=lambda: NS(
                forward_batch=ple.current.get()
            ),
        ),
        functions=("_breakable_qsa_indexer",),
    )
    layer = stack.layers[1]
    layer._qsa_prefill_topk_bridges = []

    def select_eager(hidden_states, positions, forward_batch):
        return qsa.select(hidden_states, positions, forward_batch)

    layer._compute_qsa_topk_indices_eager = select_eager
    layer._compute_qsa_topk_indices = types.MethodType(
        selected.Qwen4ExpAttentionDecoderLayer._compute_qsa_topk_indices, layer
    )
    fp8_count = real.enable_fp8_linears(stack) if fp8 else 0
    return stack, gdn, qsa, ple, fp8_count


class Fixture:
    def __init__(self, fp8, tokens, bcg, ctx, executor, shape):
        self.bcg, self.ctx, self.api = bcg, ctx, executor
        self.stack, self.gdn, self.qsa, self.ple, self.fp8_count = build(fp8, bcg, ctx)
        self.batches, self.inputs = [], []
        for lane, rows in enumerate((tokens, 1)):
            b = small.prepare_batch(rows, lane)
            b.__dict__.update(
                real.ple_batch(rows, lane == 1, slots=[lane + 1]).__dict__
            )
            b.indices = b.req_pool_indices.int()
            b.input_ids = b.input_ids[:rows].contiguous()
            b.out_cache_loc = b.out_cache_loc[:rows]
            b.split_index, b.spec_info = 0, None
            b.contains_mm_inputs = lambda: False
            self.batches.append(b)
            self.inputs.append(
                torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16) * 0.1
            )
        self.ps, self.ds = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
        ready = torch.cuda.current_stream().record_event()
        self.ps.wait_event(ready)
        self.ds.wait_event(ready)
        self.graphs, self.outputs, self.routes = {}, {}, {}
        self.shape = shape.ShapeKey(size=tokens, stream_idx=1)
        self.mode = "split_bcg"
        self.pools = {}
        self.capture()
        self.variants = [
            (self.batches[0].input_ids.clone(), self.inputs[0].clone()),
            (
                torch.randint(0, 256, self.batches[0].input_ids.shape, device="cuda"),
                torch.randn_like(self.inputs[0]) * 0.1,
            ),
        ]
        # Exercise the production session executor, using explicit constructor,
        # I/O and LM-head boundaries around the real transformer computations.
        self.runner = NS(
            model_runner=NS(model_config=NS(num_hidden_layers=2)),
            raw_num_tokens=tokens,
            model_context=contextlib.nullcontext,
            backend=NS(
                replay_session=ctx.enable_breakable_cuda_graph, replay=self.replay
            ),
            buffer_registry=NS(has_slot=lambda name: False),
            load_batch=lambda b: (b, None),
            _shape_key=lambda size, b: self.shape,
            _prefill_forward_context=self.forward_context,
            device_module=torch.cuda,
            _execute_body_capture=self.final_interval,
            _finalize_execute_output=lambda output: output,
            capture_num_tokens=[tokens],
            _pad_to_bucket=lambda size, buckets: size,
            can_run_graph=lambda batch: True,
        )
        self.continuation = executor.PdmuxBCGExecutor(self.runner)
        self.runner.backend.has_layer_range = lambda key, start, end: bool(
            self.graphs["split_bcg"].layer_segment_range(start, end)
        )
        self.dispatch_runner = NS(
            model_config=NS(num_hidden_layers=2),
            device_timer=None,
            attn_backend=NS(init_forward_metadata=lambda batch: None),
            model=NS(
                forward_split_prefill=lambda ids, pos, batch, interval: self.forward(
                    0, interval
                )
            ),
            prefill_cuda_graph_runner=NS(
                can_run_split=lambda b, count: (
                    self.mode == "split_bcg" and self.continuation.can_run(b, count)
                ),
                execute_split=self.continuation.execute,
            ),
        )

    def reset(self, lane):
        self.gdn.reset(lane + 1)
        self.qsa.reset(lane)
        real.reset_ple(self.ple, self.batches[lane])

    def forward(self, lane, interval=None):
        b = self.batches[lane]
        self.ple.current.set(b)
        return self.stack.forward(
            b.input_ids,
            b.positions,
            b,
            inputs_embeds=self.inputs[lane],
            split_interval=interval,
        )

    def snapshot(self, lane, output, bridge=False, routes=None, selected_override=None):
        b = self.batches[lane]
        selected = b.selected_tokens
        if selected_override is not None:
            selected = selected_override
        if bridge:
            selected = self.stack.layers[1]._qsa_prefill_topk_bridges[0][
                : len(b.input_ids)
            ]
        tensors = [
            *output,
            self.gdn.state[lane + 1],
            self.gdn.conv_state[lane + 1],
            self.qsa.k[lane],
            self.qsa.v[lane],
            selected,
            self.ple.ngram.context[b.req_pool_indices],
            self.ple.ngram.context[b.track_indices],
            self.ple.conv.conv_state[:, b.req_pool_indices],
            self.ple.conv.conv_state[:, b.track_indices],
        ]
        tensors += self.route_views() if routes is None else routes
        return tuple(t.clone() for t in tensors)

    def route_views(self):
        return [
            t
            for layer in self.stack.layers
            for t in (layer.mlp.topk.last.topk_ids, layer.mlp.topk.last.topk_weights)
        ]

    def set_variant(self, index):
        ids, hidden = self.variants[index]
        self.batches[0].input_ids.copy_(ids)
        self.inputs[0].copy_(hidden)

    @contextlib.contextmanager
    def forward_context(self, batch, **kwargs):
        token = self.ple.current.set(batch)
        try:
            yield
        finally:
            self.ple.current.reset(token)

    def capture(self):
        for mode in ("split_bcg", "standard_bcg"):
            with torch.cuda.stream(self.ps), self.ctx.enable_breakable_cuda_graph():
                for _ in range(3):
                    self.reset(0)
                    self.forward(0)
                self.ps.synchronize()
                self.reset(0)
                graph = self.bcg.BreakableCUDAGraph()
                pool = torch.cuda.graph_pool_handle()
                t = time.perf_counter()
                with (
                    (
                        self.api.capture_layer_boundaries(self.stack, 2)
                        if mode == "split_bcg"
                        else contextlib.nullcontext()
                    ),
                    self.bcg.BreakableCUDAGraphCapture(
                        graph, pool=pool, stream=self.ps
                    ),
                ):
                    out = self.forward(0)
                    outputs = out
                    routes = self.route_views()
                self.ps.synchronize()
                self.graphs[mode], self.outputs[mode] = graph, outputs
                self.routes[mode] = routes
                self.pools[mode] = dict(
                    segments=len(graph._segments),
                    capture_ms=(time.perf_counter() - t) * 1000,
                )
        with torch.cuda.stream(self.ds):
            for _ in range(3):
                self.reset(1)
                self.forward(1)
            self.ds.synchronize()
            self.decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.decode_graph, stream=self.ds):
                self.reset(1)
                self.decode_outputs = self.forward(1)
                self.decode_routes = self.route_views()
                self.decode_selected = self.batches[1].selected_tokens
        self.ds.synchronize()

    def replay(self, key, batch, **kwargs):
        interval = kwargs.get("pdmux_layer_interval")
        graph = self.graphs[self.mode]
        if interval is None:
            graph.replay()
        else:
            graph.replay_layers(*interval)
        return self.outputs[self.mode]

    def final_interval(self, batch, static, size, raw, key, **kwargs):
        return self.replay(key, batch, **kwargs)

    def submit(self, mode, verify=True):
        self.mode = mode
        b = self.batches[0]
        prefill_cpu = []
        with torch.cuda.stream(self.ps):
            self.reset(0)
            b.split_index = 0
            t = time.perf_counter()
            if mode == "eager_split":
                self.api.run_split_prefill(self.dispatch_runner, b, forward_count=1)
            elif mode == "split_bcg":
                self.api.run_split_prefill(self.dispatch_runner, b, forward_count=1)
            if mode in ("eager_split", "split_bcg"):
                prefill_cpu.append((time.perf_counter() - t) * 1000)
        with torch.cuda.stream(self.ds):
            t = time.perf_counter()
            self.decode_graph.replay()
            decode_cpu = (time.perf_counter() - t) * 1000
            decode = (
                self.snapshot(
                    1,
                    self.decode_outputs,
                    routes=self.decode_routes,
                    selected_override=self.decode_selected,
                )
                if verify
                else None
            )
        with torch.cuda.stream(self.ps):
            t = time.perf_counter()
            if mode == "eager_split":
                prefill = self.api.run_split_prefill(
                    self.dispatch_runner, b, forward_count=1
                )
            elif mode == "split_bcg":
                prefill = self.api.run_split_prefill(
                    self.dispatch_runner, b, forward_count=1
                )
                assert b.split_prefill_can_run_graph
            elif mode == "standard_eager":
                prefill = self.forward(0)
            else:
                with self.ctx.enable_breakable_cuda_graph(), self.forward_context(b):
                    prefill = self.replay(self.shape, b)
            prefill_cpu.append((time.perf_counter() - t) * 1000)
            if verify:
                prefill = self.snapshot(
                    0,
                    prefill,
                    bridge=mode.endswith("bcg"),
                    routes=self.routes[mode] if mode.endswith("bcg") else None,
                )
        self.last_submit_stats = dict(
            prefill_cpu_ms=sum(prefill_cpu),
            longest_prefill_call_ms=max(prefill_cpu),
            decode_cpu_ms=decode_cpu,
        )
        return prefill, decode

    def run(self, rounds, trials):
        modes = ("eager_split", "split_bcg", "standard_eager", "standard_bcg")
        controls = []
        for variant in range(2):
            self.set_variant(variant)
            pair = []
            for lane in (0, 1):
                self.reset(lane)
                pair.append(self.snapshot(lane, self.forward(lane)))
            controls.append(pair)
        torch.cuda.synchronize()
        results = {}
        for mode in modes:
            records = []
            for i in range(rounds):
                with torch.cuda.stream(self.ps):
                    self.set_variant(i % 2)
                records.append(self.submit(mode))
            torch.cuda.synchronize()
            for i, pair in enumerate(records):
                for actual, expected in zip(pair, controls[i % 2]):
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
            results[mode] = dict(correctness_rounds=rounds, exact=True, samples=[])
        for trial in range(trials):
            for mode in modes[trial % 4 :] + modes[: trial % 4]:
                self.set_variant(0)
                torch.cuda.synchronize()
                begin, finish = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                with torch.cuda.stream(self.ps):
                    begin.record()
                t = time.perf_counter()
                submissions = []
                for _ in range(rounds):
                    self.submit(mode, verify=False)
                    submissions.append(self.last_submit_stats)
                cpu_ms = (time.perf_counter() - t) * 1000 / rounds
                with torch.cuda.stream(self.ps):
                    self.ps.wait_stream(self.ds)
                    finish.record()
                finish.synchronize()
                results[mode]["samples"].append(
                    dict(
                        cpu_submit_ms=cpu_ms,
                        mixed_gpu_ms=begin.elapsed_time(finish) / rounds,
                        **{
                            metric: statistics.mean(s[metric] for s in submissions)
                            for metric in submissions[0]
                        },
                    )
                )
        for result in results.values():
            for metric in result["samples"][0]:
                result[metric] = statistics.median(s[metric] for s in result["samples"])
        return dict(
            cases=results,
            capture=self.pools,
            fp8_projections=self.fp8_count,
            tokens=len(self.batches[0].input_ids),
            decode_cuda_graph=True,
            selected_qsa_tokens_canonical=True,
        )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--dtype", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--tokens", type=int, default=65)
    args = parser.parse_args()
    torch.manual_seed(613)
    ops.bootstrap_fla()
    bcg, ctx, executor, shape = load_bcg()
    fixture = Fixture(args.dtype == "fp8", args.tokens, bcg, ctx, executor, shape)
    result = fixture.run(args.rounds, args.trials)
    ops.SOURCES[str(Path(__file__).relative_to(ops.ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    result.update(
        passed=True,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        dtype=args.dtype,
        source_sha256=ops.SOURCES,
        git_commit=subprocess.check_output(
            [
                "git",
                "-c",
                "safe.directory=" + str(ops.ROOT),
                "-C",
                str(ops.ROOT),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        tp_size=1,
        dynamic_input_variants=2,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = {k: v for k, v in result.items() if k != "source_sha256"}
    summary["cases"] = {
        k: {m: value for m, value in v.items() if m != "samples"}
        for k, v in result["cases"].items()
    }
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
