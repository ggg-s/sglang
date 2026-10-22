"""Read compiler launch resources outside every formal timing window."""

import importlib
import json
from pathlib import Path

import bench_gdn_replayssm_verify as bench
import torch

op = importlib.import_module(
    "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode"
)


def main():
    records = []
    original = op.gdn_replayssm_spec_circular_kernel

    class Capture:
        def __getitem__(self, grid):
            launcher = original[grid]

            def launch(*args, **kwargs):
                kernel = launcher(*args, **kwargs)
                records.append(
                    {
                        "grid": grid,
                        "warps": kwargs["num_warps"],
                        "stages": kwargs["num_stages"],
                        "BS": kwargs["BS"],
                        "BV": kwargs["BV"],
                        "BKT": kwargs["BKT"],
                        "registers_per_thread": kernel.n_regs,
                        "spills": kernel.n_spills,
                        "shared_memory_bytes": kernel.metadata.shared,
                    }
                )
                return kernel

            return launch

    op.gdn_replayssm_spec_circular_kernel = Capture()
    try:
        for width in (9, 17):
            kw = bench.inputs(1, width, 16, torch.float16)
            op.gdn_replayssm_spec_decode(**kw)
            torch.cuda.synchronize()
            records[-1]["width"] = width
    finally:
        op.gdn_replayssm_spec_circular_kernel = original
    Path("resources.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
