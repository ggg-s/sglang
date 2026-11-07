"""4060 shared D0/D1 worker c=4/8 partial-Remote D2D ownership regression.

No full model/HTTP throughput or H20/RDMA performance claim. Incoming P bytes
are injected at the native ring boundary; CUDA scatter/events and outbound
Mooncake TCP byte transfers are real.
"""

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
package = types.ModuleType("sglang")
package.__path__ = [str(ROOT / "python/sglang")]
sys.modules.setdefault("sglang", package)

from dsv_d2d_partial_probe_support import SharedEnvironment, exercise_shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for concurrency in (4, 8):
        env = SharedEnvironment(ROOT, concurrency, "cuda")
        try:
            results.append(exercise_shared(env))
        finally:
            env.close()
    paths = [
        "plugins/remote_attn/sglang_remote_attn/dsv_d2d.py",
        "plugins/remote_attn/sglang_remote_attn/dsv_d2d_partial.py",
        "plugins/remote_attn/sglang_remote_attn/dsv_gateway_control.py",
        "plugins/remote_attn/control_plane/p4_dsv_gateway.py",
        "test/manual/remote_attn/dsv_d2d_partial_probe_support.py",
        "test/manual/remote_attn/dsv_d2d_partial_cuda_probe.py",
    ]
    payload = dict(
        device="RTX 4060 Laptop 8GB",
        transport="Mooncake TCP",
        incoming_pd_boundary="injected pinned-ring bytes, native H2D/scatter/events",
        results=results,
        sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
