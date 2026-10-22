"""Profile old/new verify kernels on identical tensors, outside serving timing."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import bench_gdn_replayssm_verify as bench
import torch

ROOT = Path(__file__).resolve().parent


def load(name, source):
    path = (
        source / "python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_decode.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    old = load("gdn_replay_control_old", ROOT / "source")
    new = load("gdn_replay_control_new", ROOT / "optimized/source")
    folder = ROOT / "controlled"
    folder.mkdir(exist_ok=True)
    rows = []
    for width in (9, 17):
        torch.manual_seed(20261022 + width)
        common = bench.inputs(1, width, 16, torch.float16)
        checkpoint = common["checkpoint_state"].clone()
        jobs = []
        for label, module in (("baseline", old), ("optimized", new)):
            kw = dict(common)
            for name in (
                "out",
                "d_cache",
                "k_cache",
                "g_cache",
                "rawv_cache",
                "rawk_cache",
            ):
                kw[name] = common[name].clone()

            def run(module=module, kw=kw):
                module.gdn_replayssm_spec_decode(**kw)

            for _ in range(5):
                run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(30):
                    run()
            jobs.append((label, kw, graph))
        torch.testing.assert_close(
            jobs[1][1]["out"], jobs[0][1]["out"], rtol=0.02, atol=0.003
        )
        torch.testing.assert_close(
            jobs[1][1]["d_cache"].float() + jobs[1][1]["rawv_cache"].float(),
            jobs[0][1]["d_cache"].float() + jobs[0][1]["rawv_cache"].float(),
            rtol=0.002,
            atol=0.0002,
        )
        assert torch.equal(common["checkpoint_state"], checkpoint)
        for label, kw, graph in jobs:
            for _ in range(20):
                graph.replay()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(10):
                    with torch.profiler.record_function(
                        f"controlled/{label}/width{width}"
                    ):
                        graph.replay()
                torch.cuda.synchronize()
            path = folder / f"{label}-width{width}.trace.json"
            prof.export_chrome_trace(str(path))
            data = path.read_bytes()
            compressed = path.with_suffix(path.suffix + ".gz")
            compressed.write_bytes(gzip.compress(data))
            path.unlink()
            events = json.loads(data)["traceEvents"]
            kernels = [
                e
                for e in events
                if e.get("cat") == "kernel"
                and e.get("name") == "gdn_replayssm_spec_circular_kernel"
            ]
            assert len(kernels) == 300, (label, width, len(kernels))
            row = {
                "version": label,
                "width": width,
                "batch": 1,
                "history": 16,
                "max_spec_len": 17,
                "launches": len(kernels),
                "kernel_mean_us": sum(e["dur"] for e in kernels) / len(kernels),
                "trace": str(compressed.relative_to(ROOT)),
                "state_dtype": "float32",
                "history_dtype": "float16",
                "same_qkv_checkpoint_inputs": True,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    (ROOT / "controlled-summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
