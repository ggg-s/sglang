"""Launch private servers sequentially for real-request Mixed scheduling A/B."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from bench_live_mixed_target import fixture_for_model
from mixed_eager_validation import write_json
from run_mixed_eager import wait_for_generation_ready


def serving_settings(base, args):
    chunk = args.chunked_prefill_size
    if chunk is None and base.get("CHUNKED_PREFILL_SIZE") is not None:
        chunk = int(base["CHUNKED_PREFILL_SIZE"])
    max_prefill = args.max_prefill_tokens
    if max_prefill is None:
        max_prefill = int(base.get("MAX_PREFILL_TOKENS", 32768))
    if (chunk is not None and (chunk == 0 or chunk < -1)) or max_prefill <= 0:
        raise ValueError("Invalid Prefill serving limits")
    return dict(
        chunk=None if chunk is None else str(chunk),
        max_prefill=str(max_prefill),
        cpu_overlap=args.cpu_overlap or base.get("CPU_OVERLAP", "1"),
        hicache=args.hicache or base.get("HICACHE", "1"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compat-script", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--swe-workload", type=Path)
    parser.add_argument("--chunked-prefill-size", type=int)
    parser.add_argument("--max-prefill-tokens", type=int)
    parser.add_argument("--cpu-overlap", choices=("0", "1"))
    parser.add_argument("--hicache", choices=("0", "1"))
    parser.add_argument("--profile-service", action="store_true")
    parser.add_argument(
        "--groups", nargs="+", default=["dflash", "pdmux", "repaired_separate", "mixed"]
    )
    args = parser.parse_args()
    manual = Path(__file__).resolve().parent
    repo = manual.parents[2]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.base_config.read_text())
    # All groups inherit the same serving shape. Graph eligibility must never
    # force the scheduler to split an existing Prefill chunk into smaller ones.
    shared_settings = serving_settings(base, args)
    chunk, max_prefill = shared_settings["chunk"], shared_settings["max_prefill"]
    fixture = root / "fixture.json"
    if args.swe_workload is None:
        write_json(fixture, fixture_for_model(args.model_path, args.prefill_tokens))
    outcomes = []
    for group in args.groups:
        if group not in {"dflash", "pdmux", "separate", "repaired_separate", "mixed", "tailfill", "tailfill_native"}:
            raise ValueError(f"Unknown group {group}")
        folder = root / group
        folder.mkdir()
        compat = folder / "compat"
        compat.mkdir()
        (compat / "sitecustomize.py").write_text(
            "import runpy\n"
            f"runpy.run_path({str(args.compat_script.resolve())!r})\n"
            "import live_mixed_target\n"
        )
        effective = base | dict(
            TARGET_MODEL=args.model_path,
            DRAFT_MODEL=args.draft_model_path,
            LIVE_MIXED_ROOT=str(folder),
            LIVE_MIXED_GATES=str(args.gates.resolve()),
            LIVE_MIXED_MODE="native" if group in {"dflash", "pdmux"} else group,
            LIVE_SERVICE_PROFILE="1" if args.profile_service else "0",
            MIXED_OFFLINE_ROOT="",
            PDMUX="1" if group == "pdmux" else "0",
            HICACHE=shared_settings["hicache"],
            CPU_OVERLAP=shared_settings["cpu_overlap"],
            BLOCK_SIZE="8",
            MAX_RUNNING_REQUESTS="4",
            PORT="39928",
            HOST="127.0.0.1",
            PDMUX_CONFIG=str(manual / "qwen35_dflash_overlap_replayssm_a100.yaml"),
        )
        write_json(folder / "config.json", effective)
        env = os.environ | effective | dict(
            PYTHONPATH=os.pathsep.join(
                [str(compat), str(manual), str(repo / "python")]
            )
        )
        process = subprocess.Popen(
            [
                "bash",
                str(manual / "launch_qwen35_awq_dflash_overlap.sh"),
                *([] if chunk is None else ["--chunked-prefill-size", chunk]),
                "--max-prefill-tokens", max_prefill,
            ],
            cwd=repo,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=(folder / "server.log").open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        write_json(folder / "server-process.json", dict(pid=process.pid))
        outcome = dict(group=group, passed=False)
        try:
            wait_for_generation_ready(
                requests, process, "http://127.0.0.1:39928", timeout=900
            )
            info = requests.get("http://127.0.0.1:39928/server_info", timeout=30)
            info.raise_for_status()
            write_json(folder / "server-info.json", info.json())
            phases = (("swe", "0"),) if args.swe_workload else (
                ("joint", "2"), ("decode_only", "0")
            )
            for label, waves in phases:
                command = [
                    sys.executable, str(manual / "bench_live_mixed_target.py"),
                    "--fixture", str(fixture),
                    "--output", str(folder / f"{label}.json"),
                    "--trials", str(args.trials),
                    "--waves", waves,
                    "--label-prefix", label,
                ]
                if args.swe_workload:
                    command = [
                        sys.executable, str(manual / "bench_swe_mixed_target.py"),
                        "--workload", str(args.swe_workload.resolve()),
                        "--stats", str(folder / "mixed-stats.json"),
                        "--output", str(folder / "swe.json"),
                    ]
                    if args.profile_service:
                        command.append("--profile-service")
                # The HTTP client must not install the worker scheduler hook.
                client_env = os.environ | dict(PYTHONPATH=str(manual))
                with (folder / f"{label}.log").open("w") as log:
                    subprocess.run(
                        command, env=client_env, cwd=repo, stdout=log,
                        stderr=subprocess.STDOUT, check=True, timeout=1200,
                    )
                time.sleep(2)
                stats = folder / "mixed-stats.json"
                if not stats.exists():
                    raise RuntimeError("Live scheduler statistics are missing")
                write_json(folder / f"{label}-stats.json", json.loads(stats.read_text()))
            outcome["passed"] = True
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            outcomes.append(outcome)
            write_json(root / "summary.json", outcomes)
        print(outcome, flush=True)
        if not outcome["passed"]:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
