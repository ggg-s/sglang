"""Compile the exact automaton used by CUDA as host C++ for differential tests.
This validates the shared algorithm and capacity bounds, not CUDA launches.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_automaton_against_exhaustive_oracle(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang") is not None:
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("requires a C++ compiler or the ziglang package")
    package = Path(__file__).resolve().parents[1]
    executable = tmp_path / ("automaton.exe" if os.name == "nt" else "automaton")
    # Pass only process/toolchain essentials. Test failure tracebacks must not
    # include unrelated credentials from the developer's shell environment.
    env = {
        name: os.environ[name]
        for name in ("SystemRoot", "WINDIR", "PATH", "TEMP", "TMP", "HOME")
        if name in os.environ
    }
    env.update(
        ZIG_GLOBAL_CACHE_DIR=str(tmp_path / "zig-global"),
        ZIG_LOCAL_CACHE_DIR=str(tmp_path / "zig-local"),
    )
    subprocess.run(
        command
        + [
            "-std=c++17",
            "-O2",
            "-I",
            str(package / "src/hybrid_spec/backends/gpu_sa"),
            str(package / "tests/native/automaton_test.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        env=env,
        timeout=300,
    )
    subprocess.run([str(executable)], check=True, timeout=60)
