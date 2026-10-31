"""Run the frozen DFlash and spatial PDMux configurations on LongBench.

Source activate-sglang.sh in the parent shell before invoking this script.
--source selects the original replay-source checkout; --compat selects its
compat directory (or its sitecustomize.py). The server uses that checkout's
launcher. The benchmark client is the sibling bench_longbench_pdmux.py.
This driver never modifies the source snapshot or kills unrelated servers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


OWNER_KEY = "LONGBENCH_PDMUX_PROCESS_OWNER"
EXPERIMENT_PREFIXES = (
    "SGLANG_PDMUX_",
    "PDMUX_",
    "LIVE_MIXED_",
    "LIVE_SERVICE_",
    "MIXED_OFFLINE_",
    "TEMPORAL_",
    "BOUNDARY_",
    "ACCEPT_PROBE",
    "PERF_SCOPE",
    "DIAG_BOUNDARY",
)
EXPERIMENT_KEYS = {
    OWNER_KEY,
    "SGLANG_PLUGINS",
    "HYBRID_SPEC_CONFIG",
    "SGLANG_RAGGED_VERIFY_MODE",
    "SGLANG_SIMULATE_ACC_LEN",
}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cleaned_environment():
    removed = sorted(
        key for key in os.environ
        if key.startswith(EXPERIMENT_PREFIXES) or key in EXPERIMENT_KEYS
    )
    return {key: value for key, value in os.environ.items() if key not in removed}, removed


def launch_environment(args, group, owner):
    env, removed = cleaned_environment()
    overrides = {
        "TARGET_MODEL": args.target_model,
        "DRAFT_MODEL": args.draft_model,
        "DTYPE": "float16",
        "DETERMINISTIC": "0",
        "ATTENTION_BACKEND": "flashinfer",
        "DRAFT_ATTENTION_BACKEND": "flashinfer",
        "DECODE_GRAPH": "full",
        "DFLASH": "1",
        "PDMUX": "1" if group == "pdmux" else "0",
        "PDMUX_CONFIG": str(args.pdmux_config),
        "CPU_OVERLAP": "1",
        "BLOCK_SIZE": "8",
        "MAX_RUNNING_REQUESTS": "4",
        "CHUNKED_PREFILL_SIZE": "32768",
        "MAX_PREFILL_TOKENS": "32768",
        "MAX_TOTAL_TOKENS": "65536",
        "MAMBA_CACHE": "24",
        "MEM_FRACTION": "0.9",
        "HICACHE": "1",
        "HICACHE_SIZE": "16",
        "HICACHE_IO_BACKEND": "kernel",
        "HICACHE_LAYOUT": "page_first",
        "REPLAYSSM_SPEC": "1",
        "REPLAYSSM_CACHE_LEN": "32",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_MAMBA_CONV_DTYPE": "float16",
        "SGLANG_PDMUX_TEMPORAL_LAYERS": "0",
        "SGLANG_PDMUX_TEMPORAL_ADAPTIVE": "0",
        "SGLANG_PDMUX_DECODE_READY": "off",
        "HOST": "127.0.0.1",
        "PORT": str(args.port),
        "PYTHONUNBUFFERED": "1",
        # The original compatibility hook loads its next compatibility layer
        # with runpy. Keep its parent for those original helper imports.
        "PYTHONPATH": os.pathsep.join(
            (str(args.compat), str(args.compat.parent), str(args.source / "python"))
        ),
        OWNER_KEY: owner,
    }
    return env | overrides, overrides, removed


def process_identity(pid):
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        return {
            "pid": pid,
            "state": fields[0],
            "pgid": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (OSError, ValueError, IndexError):
        return None


def has_owner(pid, owner):
    try:
        entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        return f"{OWNER_KEY}={owner}".encode() in entries
    except OSError:
        return False


def owned_processes(owner):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if has_owner(pid, owner):
            identity = process_identity(pid)
            if identity is not None and identity["state"] != "Z":
                found.append(identity)
    return found


def stop_owned(process, owner):
    """Terminate the owned session group and marked children that changed groups."""
    result = {"pid": process.pid, "owner": owner, "signals": []}
    for sig, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        processes = owned_processes(owner)
        if not processes:
            break
        # A surviving marked member proves this is still the original session;
        # it also prevents PGID reuse while the group signal is submitted.
        group_owned = any(
            item["pgid"] == process.pid and item["session"] == process.pid
            for item in processes
        )
        if group_owned:
            try:
                os.killpg(process.pid, sig)
                result["signals"].append({"pgid": process.pid, "signal": sig.name})
            except ProcessLookupError:
                pass
        for item in processes:
            if group_owned and item["pgid"] == process.pid:
                continue
            current = process_identity(item["pid"])
            if (
                current is not None
                and current["start_ticks"] == item["start_ticks"]
                and has_owner(item["pid"], owner)
            ):
                try:
                    os.kill(item["pid"], sig)
                    result["signals"].append({"pid": item["pid"], "signal": sig.name})
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            process.poll()
            if not owned_processes(owner):
                break
            time.sleep(0.25)
    result["survivors"] = owned_processes(owner)
    try:
        result["returncode"] = process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        result["returncode"] = None
    result["success"] = not result["survivors"] and result["returncode"] is not None
    return result


def assert_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def assert_gpu_idle(folder, owner):
    observed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, timeout=15, check=False,
    )
    (folder / "gpu-before.csv").write_text(observed.stdout)
    (folder / "gpu-before.stderr.txt").write_text(observed.stderr)
    if observed.returncode:
        raise RuntimeError("Cannot verify GPU idleness: nvidia-smi failed")
    for line in observed.stdout.splitlines():
        if not line.strip():
            continue
        pid_text = line.split(",", 1)[0].strip()
        if not pid_text.isdigit() or not has_owner(int(pid_text), owner):
            raise RuntimeError(
                "GPU has an existing compute process; leaving it untouched: " + line
            )


def http_get(url, timeout):
    # Do not send localhost requests through a proxy from the login shell.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.status, response.read()


def wait_ready(process, url, timeout, folder):
    started = time.monotonic()
    attempts = []
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness: {process.returncode}")
        attempt = {"elapsed_s": time.monotonic() - started}
        try:
            status, body = http_get(url + "/health_generate", timeout=15)
            attempt.update(http_status=status, body=body.decode(errors="replace"))
            attempts.append(attempt)
            write_json(folder / "readiness.json", attempts)
            if status == 200 and process.poll() is None:
                return
        except (OSError, urllib.error.URLError) as error:
            attempt["error"] = f"{type(error).__name__}: {error}"
            attempts.append(attempt)
            write_json(folder / "readiness.json", attempts)
        time.sleep(2)
    raise TimeoutError(f"Generation readiness timed out after {timeout} seconds")


def get_resolved(info):
    for key in ("server_args", "resolved_server_args"):
        if isinstance(info.get(key), dict):
            return info[key]
    return info


def validate_resolved(info, args, group):
    resolved = get_resolved(info)
    expected = {
        "model_path": args.target_model,
        "speculative_draft_model_path": args.draft_model,
        "dtype": "float16",
        "quantization": "awq_marlin",
        "tp_size": 1,
        "pp_size": 1,
        "context_length": 16384,
        "chunked_prefill_size": 32768,
        "max_prefill_tokens": 32768,
        "max_total_tokens": 65536,
        "max_mamba_cache_size": 24,
        "max_running_requests": 4,
        "mem_fraction_static": 0.9,
        "speculative_algorithm": "DFLASH",
        "speculative_dflash_block_size": 8,
        "attention_backend": "flashinfer",
        "speculative_draft_attention_backend": "flashinfer",
        "linear_attn_prefill_backend": "triton",
        "linear_attn_decode_backend": "triton",
        "mamba_radix_cache_strategy": "extra_buffer",
        "disable_overlap_schedule": False,
        "enable_mixed_chunk": False,
        "enable_deterministic_inference": False,
        "enable_hierarchical_cache": True,
        "hicache_size": 16,
        "hicache_io_backend": "kernel",
        "hicache_mem_layout": "page_first",
        "hicache_write_policy": "write_through",
        "enable_linear_replayssm_spec": True,
        "linear_replayssm_cache_len": 32,
        "enable_metrics": True,
        "enable_pdmux": group == "pdmux",
        "port": args.port,
    }
    if group == "pdmux":
        expected.update(
            pdmux_prefill_mode="layer_split",
            sm_group_num=3,
            pdmux_config_path=str(args.pdmux_config),
        )
    checks = {}
    for key, wanted in expected.items():
        actual = resolved.get(key)
        checks[key] = {"expected": wanted, "actual": actual,
                       "passed": key in resolved and actual == wanted}
    graph_config = resolved.get("cuda_graph_config") or {}
    for phase, wanted in (("decode", "full"), ("prefill", "disabled")):
        key = f"cuda_graph_backend_{phase}"
        actual = resolved.get(key)
        nested = graph_config.get(phase) if isinstance(graph_config, dict) else None
        if isinstance(nested, dict) and nested.get("backend") is not None:
            actual = nested["backend"]
        checks[key] = {"expected": wanted, "actual": actual, "passed": actual == wanted}
    # Some revisions normalize the explicit unquant draft setting to None.
    key = "speculative_draft_model_quantization"
    actual = resolved.get(key)
    checks[key] = {
        "expected": ["unquant", None], "actual": actual,
        "passed": key in resolved and actual in ("unquant", None),
    }
    return resolved, {"success": all(c["passed"] for c in checks.values()), "checks": checks}


def check_inputs(args):
    import yaml

    if sys.platform != "linux" or not Path("/proc").is_dir():
        raise RuntimeError("This experiment driver requires Linux /proc for owned cleanup")
    args.source = args.source.resolve(strict=True)
    args.compat = args.compat.resolve(strict=True)
    if args.compat.is_file():
        if args.compat.name != "sitecustomize.py":
            raise ValueError("--compat file must be named sitecustomize.py")
        args.compat = args.compat.parent
    compat_script = args.compat / "sitecustomize.py"
    launcher = args.source / "test/manual/pdmux/launch_qwen35_awq_dflash_overlap.sh"
    client = Path(__file__).resolve().with_name("bench_longbench_pdmux.py")
    args.workload = args.workload.resolve(strict=True)
    args.pdmux_config = args.pdmux_config.resolve(strict=True)
    for path in (compat_script, launcher, client):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.source / "python/sglang").is_dir():
        raise ValueError("--source must contain python/sglang")
    config = yaml.safe_load(args.pdmux_config.read_text())
    wanted = {
        "sm_group_num": 3,
        "split_forward_token_budget": 32768,
        "split_forward_max_layers": 16,
        "overlap_decode_full_sm": True,
        "manual_divisions": [[88, 0, 1]],
    }
    if config != wanted:
        raise ValueError(f"PDMux configuration differs from original P88 settings: {config}")
    # Validate the workload before allocating any GPU resources. This import
    # does not import aiohttp or sglang.
    from bench_longbench_pdmux import load_workload

    rows, workload = load_workload(args.workload, 16384, 128)
    return launcher, client, {
        "source": str(args.source), "compat": str(args.compat),
        "python": sys.executable, "python_version": sys.version,
        "launcher": str(launcher), "launcher_sha256": sha256(launcher),
        "compat_sha256": sha256(compat_script),
        "client_sha256": sha256(client),
        "pdmux_config": str(args.pdmux_config),
        "pdmux_config_sha256": sha256(args.pdmux_config),
        "pdmux_config_contents": config, "workload": workload,
        "request_count": len(rows),
    }


def main(args):
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    if status_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {status_path}")
    status = {
        "state": "preparing", "success": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "groups": [], "group_order": ["dflash", "pdmux"],
        "trials_per_group": args.trials,
    }
    write_json(status_path, status)
    try:
        launcher, client, manifest = check_inputs(args)
        write_json(root / "source-selection.json", manifest)
        for group in status["group_order"]:
            folder = root / group
            folder.mkdir()
            owner = f"{uuid.uuid4().hex}-{group}"
            env, overrides, removed = launch_environment(args, group, owner)
            write_json(folder / "launch-config.json", {
                "overrides": overrides, "removed_environment_names": removed,
                "command": ["bash", str(launcher)], "cwd": str(args.source),
            })
            outcome = {"group": group, "state": "starting", "success": False}
            status["groups"].append(outcome)
            status["state"] = "running"
            write_json(status_path, status)
            server = client_process = None
            client_owner = owner + "-client"
            cleanup = []
            try:
                assert_port_free(args.port)
                assert_gpu_idle(folder, owner)
                with (folder / "server.log").open("w") as log:
                    server = subprocess.Popen(
                        ["bash", str(launcher)], cwd=args.source, env=env,
                        stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                write_json(folder / "server-process.json", {
                    "pid": server.pid, "pgid": server.pid, "owner": owner,
                    "identity": process_identity(server.pid),
                })
                url = f"http://127.0.0.1:{args.port}"
                wait_ready(server, url, args.startup_timeout, folder)
                _, raw = http_get(url + "/get_server_info", timeout=30)
                (folder / "server-info.raw.json").write_bytes(raw)
                info = json.loads(raw)
                write_json(folder / "server-info.json", info)
                resolved, validation = validate_resolved(info, args, group)
                write_json(folder / "resolved-config.json", resolved)
                write_json(folder / "resolved-validation.json", validation)
                if not validation["success"]:
                    failed = [key for key, value in validation["checks"].items() if not value["passed"]]
                    raise RuntimeError(f"Resolved server settings differ: {failed}")
                outcome["state"] = "benchmarking"
                write_json(status_path, status)
                command = [
                    sys.executable, str(client), "--url", url,
                    "--workload", str(args.workload),
                    "--output", str(folder / "results.json"),
                    "--concurrency", "4", "--output-tokens", "128",
                    "--context-length", "16384", "--trials", str(args.trials),
                    "--warmup-requests", "2", "--capture-server-info",
                    "--capture-metrics",
                ]
                client_env, _ = cleaned_environment()
                client_env.update({
                    OWNER_KEY: client_owner, "PYTHONPATH": str(client.parent),
                    "PYTHONUNBUFFERED": "1",
                })
                write_json(folder / "benchmark-command.json", command)
                with (folder / "benchmark.log").open("w") as log:
                    client_process = subprocess.Popen(
                        command, env=client_env, cwd=client.parent,
                        stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                    )
                returncode = client_process.wait(timeout=args.benchmark_timeout)
                outcome["benchmark_returncode"] = returncode
                if returncode:
                    raise RuntimeError(f"Benchmark client failed with exit code {returncode}")
                result = json.loads((folder / "results.json").read_text())
                if not result.get("success"):
                    raise RuntimeError("Benchmark result did not report complete success")
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during benchmark: {server.returncode}")
                _, raw = http_get(url + "/get_server_info", timeout=30)
                (folder / "server-info.after.raw.json").write_bytes(raw)
                outcome["success"] = True
            except BaseException as error:
                outcome["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                outcome["state"] = "cleaning_up"
                write_json(status_path, status)
                for process, token in ((client_process, client_owner), (server, owner)):
                    if process is not None:
                        try:
                            cleanup.append(stop_owned(process, token))
                        except Exception as error:
                            cleanup.append({"pid": process.pid, "success": False,
                                            "error": f"{type(error).__name__}: {error}"})
                write_json(folder / "cleanup.json", cleanup)
                if any(not item["success"] for item in cleanup):
                    outcome["success"] = False
                    outcome["cleanup_failed"] = True
                outcome["state"] = "complete" if outcome["success"] else "failed"
                write_json(status_path, status)
            if not outcome["success"]:
                raise RuntimeError(f"Group {group} failed; stopping the experiment")
            print(json.dumps(outcome), flush=True)
        status["success"] = True
        status["state"] = "complete"
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
    finally:
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)
    return 0 if status["success"] else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compat", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--pdmux-config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=39931)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--benchmark-timeout", type=float, default=7200)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535")
    if args.trials <= 0 or min(args.startup_timeout, args.benchmark_timeout) <= 0:
        parser.error("Trial count and timeouts must be positive")
    return args


if __name__ == "__main__":
    # Ctrl-C and external TERM both enter the same owned-process cleanup path.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    sys.exit(main(parse_args()))
