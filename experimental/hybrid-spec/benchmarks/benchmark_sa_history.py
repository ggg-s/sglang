"""Compare native long-history SA build/append against a fixed Git revision.

Run from either checkout with a C++ compiler (or the ziglang Python package).
Outputs medians, raw samples and semantic checksums. CUDA is measured separately
by benchmark_sa_history_cuda.py; native CPU timings do not predict GPU TPOT.
"""

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="88f8d971eb")
    parser.add_argument("--lengths", nargs="+", type=int, default=[8192, 65536, 262144])
    parser.add_argument("--vocabs", nargs="+", type=int, default=[1, 2, 64, 8192])
    parser.add_argument("--tail", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compiler-cache", type=Path, help="optional reusable Zig cache directory"
    )
    parser.add_argument(
        "--cpu",
        type=int,
        help="pin benchmark and compiler processes to one logical CPU",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu is not None:
        if args.cpu < 0:
            parser.error("cpu must be nonnegative")
        if os.name == "nt":
            set_affinity = ctypes.windll.kernel32.SetProcessAffinityMask
            set_affinity.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            if args.cpu >= ctypes.sizeof(ctypes.c_size_t) * 8:
                parser.error("cpu exceeds the Windows affinity mask width")
            if not set_affinity(ctypes.c_void_p(-1), 1 << args.cpu):
                raise ctypes.WinError()
        else:
            os.sched_setaffinity(0, {args.cpu})
    if min(*args.lengths, *args.vocabs, args.tail, args.repeats) < 1:
        parser.error("lengths, vocabs, tail and repeats must be positive")
    package = Path(__file__).resolve().parents[1]
    repo = package.parents[1]
    source = package / "src/hybrid_spec/backends/gpu_sa"
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang"):
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        parser.error("requires c++, g++, or the ziglang Python package")
    report = {
        "platform": platform.platform(),
        "compiler": subprocess.check_output(command + ["--version"], text=True),
        "baseline_ref": args.baseline_ref,
        "tail": args.tail,
        "repeats": args.repeats,
        "logical_cpu": args.cpu,
        "scope": "native CPU build / append, allocation excluded; not GPU or TPOT",
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="hybrid-sa-bench-") as directory:
        work = Path(directory)
        baseline = work / "baseline"
        baseline.mkdir()
        header = subprocess.check_output(
            [
                "git",
                "show",
                f"{args.baseline_ref}:experimental/hybrid-spec/src/hybrid_spec/backends/gpu_sa/automaton.h",
            ],
            cwd=repo,
        )
        (baseline / "automaton.h").write_bytes(header)
        executables = {}
        env = os.environ.copy()
        cache = args.compiler_cache.resolve() if args.compiler_cache else work
        env["ZIG_GLOBAL_CACHE_DIR"] = str(cache / "zig-global")
        env["ZIG_LOCAL_CACHE_DIR"] = str(cache / "zig-local")
        for name, include in (("before", baseline), ("after", source)):
            executable = work / (name + (".exe" if os.name == "nt" else ""))
            flags = (
                []
                if b"EDGE_FIELDS" in (include / "automaton.h").read_bytes()
                else ["-DHYBRID_SA_LEGACY_LAYOUT"]
            )
            if b"STATE_FIELDS" in (include / "automaton.h").read_bytes():
                flags.append("-DHYBRID_SA_COMPACT_LAYOUT")
            if b"init_frequency" in (include / "automaton.h").read_bytes():
                flags.append("-DHYBRID_SA_FREQUENCY_LAYOUT")
            subprocess.run(
                command
                + ["-std=c++17", "-O3"]
                + flags
                + [
                    "-I",
                    str(include),
                    str(Path(__file__).with_suffix(".cpp")),
                    "-o",
                    str(executable),
                ],
                check=True,
                env=env,
                timeout=300,
                capture_output=True,
            )
            executables[name] = executable
        for length in args.lengths:
            for vocab in args.vocabs:
                row = {"history_tokens": length, "vocab": vocab}
                checksums = set()
                for name, executable in executables.items():
                    output = subprocess.check_output(
                        [
                            str(executable),
                            str(length),
                            str(vocab),
                            str(args.tail),
                            str(args.repeats),
                        ],
                        text=True,
                        timeout=300,
                    )
                    samples = []
                    for line in output.splitlines():
                        build, append, checksum = line.split(",")
                        checksums.add(checksum)
                        samples.append(
                            {
                                "build_us": float(build),
                                "append_us_per_token": float(append),
                            }
                        )
                    row[name] = {
                        "median": {
                            key: statistics.median(sample[key] for sample in samples)
                            for key in samples[0]
                        },
                        "samples": samples,
                    }
                if len(checksums) != 1:
                    raise RuntimeError(
                        f"semantic state mismatch: length={length}, vocab={vocab}"
                    )
                row["semantic_checksum"] = checksums.pop()
                row["speedup"] = {
                    key: row["before"]["median"][key] / row["after"]["median"][key]
                    for key in row["before"]["median"]
                }
                report["results"].append(row)
                print(json.dumps(row), flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, indent=2) + "\n", encoding="utf-8"
                )


if __name__ == "__main__":
    main()
