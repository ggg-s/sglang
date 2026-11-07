"""Compare hot journal scalar updates at the reported 27 x 32768 prompt size.

This measures coordinator serialization/persistence on the same CPU and disk,
not GPU inference or D2D timing. No backend requests are submitted.
"""

import argparse
import importlib.util
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS


def measure(path, count=27, length=32768, samples=10):
    spec = importlib.util.spec_from_file_location("gateway_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as folder:
        gateway = module.DsvGateway(
            None, NS(), folder, prefill_url="http://unused/generate", autostart=False
        )
        try:
            gateway.facts = {r: [dict(instance_id=r)] for r in ("src", "dst")}
            prompt = list(range(length))
            for i in range(count):
                gateway.accept(
                    dict(rid=f"r{i}", prompt_ids=prompt, max_new_tokens=128, cut=32)
                )
            times = []
            for n in range(samples):
                before = time.perf_counter()
                gateway._update("r0", benchmark_counter=n)
                times.append((time.perf_counter() - before) * 1000)
            return dict(
                entries=count,
                input_tokens=length,
                updates=samples,
                scalar_update_ms=times,
                p50_ms=statistics.median(times),
                journal_bytes=gateway.path.stat().st_size,
            )
        finally:
            gateway.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = (
        Path(__file__).resolve().parents[3]
        / "plugins/remote_attn/control_plane/p4_dsv_gateway.py"
    )
    result = dict(scope=__doc__, current=measure(current))
    if args.baseline:
        result["baseline"] = measure(args.baseline)
        result["speedup"] = result["baseline"]["p50_ms"] / result["current"]["p50_ms"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
