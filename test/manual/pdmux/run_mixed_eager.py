"""Run isolated actual-model mixed-eager validation on Linux/CUDA.

Example (retain the existing baseline environment JSON):
  python test/manual/pdmux/run_mixed_eager.py --base-config baseline-config.json \
    --model-path /models/Qwen3.5-35B-A3B-AWQ --draft-model-path /models/DFlash \
    --output-dir /validation/mixed-eager-001

Exit 2 means the correctness gate did not pass. Diagnostic frozen intermediates
and partial numerical ablations never qualify that gate. The stable candidate
uses fixed FP16 projections and full-K MoE in both separate and mixed branches.
No scheduler option is registered; the private server keeps its original
draft/verify graphs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from mixed_eager_validation import (
    build_jobs,
    source_hashes,
    validate_eager_gate,
    validate_proposals,
    write_json,
)


def prepare_run(args):
    """Validate and freeze the launch plan without torch, CUDA or a server."""
    manual = Path(__file__).resolve().parent
    repo = manual.parents[2]
    jobs = build_jobs(
        smoke=args.smoke, numerics=args.numerics, mixed_graph=args.mixed_graph
    )
    base = json.loads(args.base_config.read_text())
    if not isinstance(base, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in base.items()
    ):
        raise ValueError(
            "Base configuration must be a string-valued launch environment"
        )
    defaults = dict(
        MAX_RUNNING_REQUESTS="4",
        BLOCK_SIZE="8",
        DFLASH="1",
        PDMUX="0",
        DTYPE="float16",
        DETERMINISTIC="0",
        DECODE_GRAPH="full",
        ATTENTION_BACKEND="flashinfer",
        DRAFT_ATTENTION_BACKEND="flashinfer",
        REPLAYSSM_CACHE_LEN="32",
        SGLANG_PDMUX_TEMPORAL_LAYERS="0",
        SGLANG_PDMUX_TEMPORAL_ADAPTIVE="0",
        SGLANG_PDMUX_DECODE_READY="off",
        TEMPORAL_AUDIT_PATH="",
        BOUNDARY_DIAG_PATH="",
    )
    for key, value in defaults.items():
        if base.get(key, value) != value:
            raise ValueError(f"Unchanged offline baseline requires {key}={value}")
    if base.get("REPLAYSSM_SPEC") != "1":
        raise ValueError("Baseline must retain REPLAYSSM_SPEC=1")
    proposal_inputs = {}
    if args.proposals_from:
        for reject in (False, True):
            lane = "rejection" if reject else "natural"
            path = args.proposals_from / f"native-reductions-{lane}-report.json"
            report = json.loads(path.read_text())
            rounds = max(job["rounds"] for job in jobs)
            digest = validate_proposals(report, rounds=rounds)
            proposal_inputs[lane] = (path.resolve(), report, digest)
    if args.compat_script:
        # Validate before creating output or launching a process.
        args.compat_script.read_bytes()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    compat = root / "compat"
    compat.mkdir()
    hook = "import os\n"
    if args.compat_script:
        hook += f"import runpy\nrunpy.run_path({str(args.compat_script.resolve())!r})\n"
    hook += 'if os.environ.get("MIXED_OFFLINE_ROOT"):\n import mixed_eager_control\n'
    (compat / "sitecustomize.py").write_text(hook)
    for job in jobs:
        job["root"] = str(root)
        if job["mixed_graph_candidate"]:
            job["eager_gate_files"] = [
                str(root / (name + "-report.json"))
                for name in ("native-natural", "native-rejection")
            ]
            if not proposal_inputs:
                lane = "rejection" if job["rejection_fixture"] else "natural"
                job["proposals_file"] = str(root / f"native-{lane}-report.json")
                job["proposals_from_eager_gate"] = True
        if proposal_inputs:
            lane = "rejection" if job["rejection_fixture"] else "natural"
            path, report, digest = proposal_inputs[lane]
            frozen = root / f"proposals-{lane}.json"
            write_json(frozen, report)
            job.update(proposals_file=str(frozen), proposals_sha256=digest)
    effective = (
        base
        | defaults
        | dict(
            TARGET_MODEL=args.model_path,
            DRAFT_MODEL=args.draft_model_path,
            REPLAYSSM_SPEC="1",
            MIXED_OFFLINE_ROOT=str(root),
            PORT=str(args.port),
            HOST="127.0.0.1",
        )
    )
    env = (
        os.environ
        | effective
        | dict(
            PYTHONPATH=f"{compat}{os.pathsep}{manual}{os.pathsep}{repo / 'python'}"
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        )
    )
    write_json(root / "base-config.json", base)
    write_json(root / "config.json", effective)
    paths = tuple(source_hashes(repo)) + (
        "python/sglang/srt/speculative/mixed_graph.py",
        "test/manual/pdmux/mixed_eager_numerics.py",
        "test/manual/pdmux/mixed_eager_control.py",
        "test/manual/pdmux/run_mixed_eager.py",
        "test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh",
    )
    write_json(root / "source-sha256.json", source_hashes(repo, paths))
    plan = dict(
        stage="prepared",
        gpu_executed=False,
        correctness_passed=None,
        numerical_path=args.numerics,
        serving_path_changed=False,
        mixed_graph_tested=False,
        jobs=jobs,
        numerical_source_sha256=source_hashes(repo),
        proposal_sources={
            lane: dict(path=str(path), proposals_sha256=digest)
            for lane, (path, _, digest) in proposal_inputs.items()
        },
    )
    write_json(root / "plan.json", plan)
    return root, manual, repo, env, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=39927)
    parser.add_argument("--compat-script", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--numerics", choices=["original", "stable"], default="stable")
    parser.add_argument("--proposals-from", type=Path)
    parser.add_argument("--mixed-graph", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze and check inputs without starting a GPU server",
    )
    args = parser.parse_args()
    if args.mixed_graph and (args.smoke or args.numerics != "stable"):
        parser.error(
            "Mixed Graph requires full stable Eager validation; omit --smoke and use --numerics stable"
        )
    if not args.prepare_only and os.name != "posix":
        raise SystemExit("Run this driver on the Linux A100 host")
    root, manual, repo, env, plan = prepare_run(args)
    if args.prepare_only:
        print(
            f"Prepared {len(plan['jobs'])} jobs: {root / 'plan.json'}; GPU not executed"
        )
        return 0
    import requests

    process = subprocess.Popen(
        ["bash", str(manual / "launch_qwen35_awq_dflash_overlap.sh")],
        cwd=repo,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=(root / "server.log").open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    summaries = []
    terminal_error = None
    passed = False
    try:
        write_json(
            root / "server-process.json",
            dict(pid=process.pid, process_group=process.pid, output_dir=str(root)),
        )
        deadline = time.monotonic() + 600
        url = f"http://127.0.0.1:{args.port}"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Private server exited: {root / 'server.log'}")
            try:
                response = requests.get(url + "/get_server_info", timeout=2)
                if response.ok:
                    (root / "server-info.json").write_text(
                        json.dumps(response.json(), indent=2)
                    )
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("Private validation server did not become ready")
        # server-info may be available before the built-in generation warmup
        # completes. Drain a native generation before injecting offline work.
        requests.get(url + "/health_generate", timeout=120).raise_for_status()
        jobs = plan["jobs"]
        for config in jobs:
            nonce = config["nonce"]
            if config["mixed_graph_candidate"] and not all(
                r["gate_passed"] for r in summaries[:2]
            ):
                break
            if source_hashes(repo) != plan["numerical_source_sha256"]:
                raise RuntimeError(
                    "Numerical source changed after experiment preparation"
                )
            if config.get("proposals_file"):
                digest = validate_proposals(
                    json.loads(Path(config["proposals_file"]).read_text()),
                    rounds=config["rounds"],
                )
                if config.get("proposals_from_eager_gate"):
                    config["proposals_sha256"] = digest
                elif digest != config["proposals_sha256"]:
                    raise RuntimeError("Prepared proposals changed before submission")
            write_json(root / "trigger.json", config)
            path = root / (nonce + "-report.json")
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Private server exited: {root / 'server.log'}")
                try:
                    result = json.loads(path.read_text())
                    if result["stage"] in ("completed", "failed"):
                        break
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(nonce)
            gate = result.get("gate_passed") is True
            gate_error = None
            if (
                gate
                and not args.smoke
                and config["repeat_control"]
                and args.numerics == "stable"
            ):
                try:
                    validate_eager_gate(
                        result,
                        expected_sources=plan["numerical_source_sha256"],
                        rejection=config["rejection_fixture"],
                    )
                except ValueError as exc:
                    gate, gate_error = False, str(exc)
            summaries.append(
                dict(
                    nonce=nonce,
                    stage=result["stage"],
                    gate_passed=gate,
                    gate_error=gate_error,
                    pools_flushed=result.get("pools_flushed") is True,
                )
            )
            if result["stage"] == "failed":
                break
        required = 4 if args.mixed_graph else 2
        passed = (
            len(summaries) == len(jobs)
            and all(r["stage"] == "completed" and r["pools_flushed"] for r in summaries)
            and all(r["gate_passed"] for r in summaries[:required])
        )
    except BaseException as exc:
        terminal_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # A launcher may exit before its GPU workers. Its process group still
        # belongs to this driver because Popen used start_new_session=True.
        cleanup_error = None
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            cleanup_error = str(exc)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
        write_json(
            root / "summary.json",
            dict(
                runs=summaries,
                correctness_passed=passed and cleanup_error is None,
                mixed_graph_gate=(
                    passed and cleanup_error is None if args.mixed_graph else None
                ),
                numerical_path=args.numerics,
                mixed_graph_tested=any(
                    r["nonce"].startswith("graph-") for r in summaries
                ),
                gpu_executed=True,
                serving_path_changed=False,
                diagnostic_timings_are_not_performance_results=True,
                error=terminal_error,
                cleanup_error=cleanup_error,
            ),
        )
    return 0 if passed and cleanup_error is None else 2


if __name__ == "__main__":
    sys.exit(main())
