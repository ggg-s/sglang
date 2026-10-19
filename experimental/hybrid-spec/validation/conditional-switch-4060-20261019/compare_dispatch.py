"""IF versus native SWITCH using identical production tails and statistics."""

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
from hybrid_spec.integrations.sglang.compact_graph import replay_compact
from hybrid_spec.integrations.sglang.conditional_graph import (
    ConditionalTailGraph,
    _switch_adapter,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpers", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.helpers)
    sys.path.insert(0, args.probe_dir)
    from ablate_dispatch import Case
    from dispatch_probe import make_record

    class SwitchCase(Case):
        def __init__(self, batch_size, selected):
            super().__init__(batch_size, selected, make_record)
            stream = torch.cuda.Stream()
            self.switch = ConditionalTailGraph(self.record, stream, dispatch="switch")
            self.switch_record = replace(self.record, conditional=self.switch)
            no_stats = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                no_stats, pool=torch.cuda.graph_pool_handle(), stream=stream
            ):
                _switch_adapter().capture_switch(
                    self.record.num_neural.data_ptr(),
                    [leaf.raw_cuda_graph() for leaf in self.switch.leaf_graphs],
                    stream.cuda_stream,
                )
            self.graphs["switch_no_stats"] = no_stats
            self.dispatch["switch_no_stats"] = no_stats.replay
            self.dispatch["switch_stats"] = self.switch.graph.replay
            self.dispatch["switch_production"] = lambda: self.switch.replay(batch_size)
            self.modes.extend(["switch_no_stats", "switch_stats", "switch_production"])

        def run(self, mode):
            if mode == "switch_production":
                replay_compact(self.switch_record, self.counters, self.batch_size)
            else:
                super().run(mode)

        def validate_switch(self):
            b = self.batch_size
            before, mixed_before = self.switch.snapshot()
            histogram = [0] * (b + 1)
            mixed, checked = 0, 0
            for raw_bs in sorted({1, max(1, b - 1), b}):
                for size in range(raw_bs + 1):
                    for reverse in (False, True):
                        rows = (
                            list(range(raw_bs - size, raw_bs))
                            if reverse
                            else list(range(size))
                        )
                        self.record.override.use_retrieval.fill_(True)
                        self.record.override.use_retrieval[rows] = False
                        replay_compact(self.switch_record, self.counters, raw_bs)
                        assert self.record.output.cpu().tolist() == [
                            row * 3 + 7 if row in rows else -1 for row in range(b)
                        ]
                        histogram[size] += 1
                        mixed += 0 < size < raw_bs
                        checked += 1
            after, mixed_after = self.switch.snapshot()
            assert [a - z for a, z in zip(after, before)] == histogram
            assert mixed_after - mixed_before == mixed
            self.record.override.use_retrieval.fill_(True)
            self.record.override.use_retrieval[: self.selected] = False
            self.switch.live_bs.fill_(b)
            self.prepare()
            return checked

        def close(self):
            self.graphs.pop("switch_no_stats").reset()
            self.switch.reset()
            super().close()

        def profile(self, mode, directory):
            result = super().profile(mode, directory)
            # The shared helper counts PyTorch's conditional_handle name only.
            result["switch_handle_kernels_per_step"] = sum(
                kernel["count_per_step"]
                for kernel in result["kernels"]
                if "set_switch_handle" in kernel["name"]
            )
            return result

    directory = Path(args.output_dir)
    directory.mkdir(exist_ok=True)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
    }
    for b, n in ((1, 0), (1, 1), (3, 1), (4, 2), (8, 4), (16, 8), (32, 16)):
        gc.collect()
        gc.disable()
        case = SwitchCase(b, n)
        checked = case.validate_switch()
        gc.enable()
        row = {
            "batch_size": b,
            "selected": n,
            "checked_rounds": checked,
            "timings": case.measure(),
        }
        if b in (1, 8, 32):
            row["profiles"] = {
                mode: case.profile(mode, directory)
                for mode in (
                    "host",
                    "direct",
                    "tree",
                    "production",
                    "switch_no_stats",
                    "switch_production",
                )
            }
        report["cases"].append(row)
        (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            b,
            n,
            {
                m: round(t["dispatch_gpu_us_median"], 3)
                for m, t in row["timings"].items()
                if t["dispatch_gpu_us_median"] is not None
            },
            flush=True,
        )
        case.close()
        del case
    torch.cuda.synchronize()
