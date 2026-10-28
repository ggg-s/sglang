"""Fixed-work CUDA scheduling comparison, not a DSV4/DSPARK benchmark.

Runs production standard/layer_split loops with fixture model/cache/TP peers,
and a minimal single-stream reference with the same GEMMs and decode replays.
All modes execute identical GPU arithmetic. There is no FlashMLA, DSpark
proposal/verify, NCCL, CP communication, HiCache DMA or green-context SM cap.
"""

from __future__ import annotations

import argparse
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
from test_pdmux_standard_prefill import (
    _Batch,
    _FakeScheduler,
    _LoopFinished,
    _STREAM_IDX,
)

from sglang.srt.multiplex import multiplexing_mixin as mux


class Scheduler(_FakeScheduler):
    def __init__(self, mode, *, rows, layers, chunks, decode_steps, split_layers):
        self.chunks = [_Batch(f"chunk-{i}") for i in range(chunks)]
        super().__init__(
            max_iterations=100000, enable_overlap=True, prefill_batches=self.chunks
        )
        self.pdmux_standard = mode == "standard"
        self.spec_algorithm = NS(is_none=lambda: False)
        self.ps.attn_dp_size = 1
        self.model_config = NS(num_hidden_layers=layers)
        self.pdmux_config.split_forward_token_budget = rows * split_layers
        self.pdmux_config.split_forward_max_layers = 0
        self._check_pdmux_dp_graph_capability = lambda: None
        self.stream_groups = [
            (torch.cuda.Stream(), torch.cuda.Stream()) for _ in range(3)
        ]
        self.decode_steps = decode_steps
        self.decode_count = self.prefill_calls = self.prefill_layer_count = 0
        self.prefill_submit_ns = 0
        self.weight = torch.eye(256, device="cuda") * 0.5
        self.decode_input = torch.ones((32, 256), device="cuda")
        self.decode_output = torch.empty_like(self.decode_input)
        self.decode_graph = torch.cuda.CUDAGraph()
        stream = self.stream_groups[0][1]
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(self.decode_graph, stream=stream):
            value = self.decode_input
            for _ in range(4):
                value = torch.nn.functional.linear(value, self.weight)
            self.decode_output.copy_(value)
        stream.synchronize()
        for batch in self.chunks:
            batch.extend_num_tokens = rows
            batch.hidden = torch.ones((rows, 256), device="cuda")
        ready = torch.cuda.current_stream().record_event()
        for pair in self.stream_groups:
            for stream in pair:
                stream.wait_event(ready)

    def _recv_requests(self):
        if (
            len(self.processed_prefills) == len(self.chunks)
            and self.decode_count >= self.decode_steps
        ):
            raise _LoopFinished
        return super()._recv_requests()

    def update_running_batch(self, batch):
        batch._empty = self.decode_count >= self.decode_steps
        return batch

    def process_batch_result(self, batch, result):
        result.copy_done.synchronize()
        super().process_batch_result(batch, result)

    def run_batch(self, batch):
        prefill = (
            batch is self._pdmux_prefill_batch or batch is self.split_prefill_batch
        )
        start = time.perf_counter_ns()
        if prefill:
            count = (
                self.model_config.num_hidden_layers
                if self.pdmux_standard
                else batch.split_forward_count
            )
            for _ in range(count):
                batch.hidden = torch.nn.functional.linear(batch.hidden, self.weight)
            self.prefill_calls += 1
            self.prefill_layer_count += count
            self.prefill_submit_ns += time.perf_counter_ns() - start
            payload = batch.hidden
        else:
            self.decode_count += 1
            self.decode_graph.replay()
            payload = self.decode_output
        event = torch.cuda.Event()
        event.record()
        return NS(
            is_prefill=prefill,
            submitted_iteration=self.iteration,
            copy_done=event,
            extra_keep_alive_refs=[batch, payload],
        )


def run(mode, *, rows, layers, chunks, decode_steps, split_layers):
    scheduler = Scheduler(
        mode,
        rows=rows,
        layers=layers,
        chunks=chunks,
        decode_steps=decode_steps,
        split_layers=split_layers,
    )
    _STREAM_IDX[0] = 0
    torch.cuda.synchronize()
    wall = time.perf_counter_ns()
    if mode == "single_stream_reference":
        # Minimal fixed-work reference, not the native SGLang/DSPARK scheduler.
        stream = scheduler.stream_groups[0][1]
        with torch.cuda.stream(stream):
            done = 0
            for i, batch in enumerate(scheduler.chunks):
                start = time.perf_counter_ns()
                for _ in range(layers):
                    batch.hidden = torch.nn.functional.linear(
                        batch.hidden, scheduler.weight
                    )
                scheduler.prefill_submit_ns += time.perf_counter_ns() - start
                scheduler.prefill_calls += 1
                scheduler.prefill_layer_count += layers
                next_done = (i + 1) * decode_steps // chunks
                for _ in range(next_done - done):
                    scheduler.decode_graph.replay()
                    scheduler.decode_count += 1
                done = next_done
    else:
        with (
            patch.object(mux, "get_current_stream_idx", lambda: _STREAM_IDX[0]),
            patch.object(
                mux,
                "set_current_stream_idx",
                lambda idx: _STREAM_IDX.__setitem__(0, idx),
            ),
            patch.object(mux.dist, "get_world_size", lambda group: 2),
            patch.object(torch.cuda, "empty_cache", lambda: None),
        ):
            try:
                scheduler.event_loop_pdmux()
            except _LoopFinished:
                pass
            finally:
                mux.set_pdmux_status(False)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - wall) / 1e6
    assert scheduler.decode_count == decode_steps, scheduler.decode_count
    assert scheduler.prefill_layer_count == layers * chunks
    for batch in scheduler.chunks:
        torch.testing.assert_close(
            batch.hidden, torch.full_like(batch.hidden, 0.5**layers), atol=0, rtol=0
        )
    torch.testing.assert_close(
        scheduler.decode_output, torch.full_like(scheduler.decode_output, 0.5**4)
    )
    return dict(
        wall_ms=wall_ms,
        prefill_submit_ms=scheduler.prefill_submit_ns / 1e6,
        prefill_calls=scheduler.prefill_calls,
        prefill_layers=scheduler.prefill_layer_count,
        decode_steps=scheduler.decode_count,
        scheduler_iterations=scheduler.iteration,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--layers", type=int, default=43)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--split-layers", type=int, default=2)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    modes = ["single_stream_reference", "layer_split", "standard"]
    result = dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        scope=__doc__,
        shape=dict(
            layers=args.layers,
            chunks=args.chunks,
            decode_steps=args.decode_steps,
            split_layers=args.split_layers,
            hidden=256,
            dtype="float32",
        ),
        cases={},
    )
    for rows in args.rows:
        options = dict(
            rows=rows,
            layers=args.layers,
            chunks=args.chunks,
            decode_steps=args.decode_steps,
            split_layers=args.split_layers,
        )
        for mode in modes:
            run(mode, **options)  # discard cold setup/warmup
        samples = {mode: [] for mode in modes}
        for repeat in range(args.repeats):
            shift = repeat % len(modes)
            order = modes[shift:] + modes[:shift]
            if repeat % 2:
                order.reverse()
            for mode in order:
                samples[mode].append(run(mode, **options))
        result["cases"][rows] = dict(
            median={
                mode: {
                    key: statistics.median(s[key] for s in values) for key in values[0]
                }
                for mode, values in samples.items()
            },
            samples=samples,
        )
        print(rows, json.dumps(result["cases"][rows]["median"]), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
