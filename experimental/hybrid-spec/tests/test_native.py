"""Compile the exact automaton used by CUDA as host C++ for differential tests.
This validates the shared algorithm and capacity bounds, not CUDA launches.
"""

import gc
import importlib.util
import itertools
import os
import shutil
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton
from hybrid_spec.backends.host_sa import NativeHostBuilder
from hybrid_spec.backends.reference import SuffixAutomaton
from hybrid_spec.config import HybridConfig


def native_toolchain(tmp_path):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler:
        command = [compiler]
    elif importlib.util.find_spec("ziglang") is not None:
        command = [sys.executable, "-m", "ziglang", "c++"]
    else:
        pytest.skip("requires a C++ compiler or the ziglang package")
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
    return command, env


def test_native_automaton_against_exhaustive_oracle(tmp_path):
    command, env = native_toolchain(tmp_path)
    package = Path(__file__).resolve().parents[1]
    executable = tmp_path / ("automaton.exe" if os.name == "nt" else "automaton")
    subprocess.run(
        command
        + [
            "-std=c++17",
            "-O2",
            "-UNDEBUG",
            "-I",
            str(package / "src/hybrid_spec/backends/gpu_sa"),
            str(package / "tests/native/automaton_test.cpp"),
            str(package / "src/hybrid_spec/backends/gpu_sa/host_builder.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        env=env,
        timeout=300,
    )
    subprocess.run([str(executable)], check=True, timeout=60)


@pytest.fixture(scope="module")
def host_builder(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("host-builder")
    command, env = native_toolchain(tmp_path)
    package = Path(__file__).resolve().parents[1]
    library = tmp_path / ("host_builder.dll" if os.name == "nt" else "host_builder.so")
    flags = [] if os.name == "nt" else ["-fPIC"]
    subprocess.run(
        command
        + ["-std=c++17", "-O2", "-UNDEBUG", "-shared"]
        + flags
        + [
            str(package / "src/hybrid_spec/backends/gpu_sa/host_builder.cpp"),
            str(package / "tests/native/host_builder_probe.cpp"),
            "-I",
            str(package / "src/hybrid_spec/backends/gpu_sa"),
            "-o",
            str(library),
        ],
        check=True,
        env=env,
        timeout=300,
    )
    return NativeHostBuilder(library, pin_memory=False)


def test_host_native_layout_matches_reference_and_validates_tokens(host_builder):
    """Transferred states must preserve suffix links, clone positions and edges."""
    histories = itertools.chain.from_iterable(
        itertools.product(range(alphabet), repeat=width)
        for alphabet, maximum in ((2, 8), (3, 5), (5, 4))
        for width in range(1, maximum + 1)
    )
    for tokens in histories:
        host = host_builder.build(host_builder.prepare(tokens), 100)
        oracle = SuffixAutomaton()
        oracle.append(tokens)
        length, num_states, num_edges, last, healthy, _, _ = host.meta.tolist()
        assert length == len(tokens) and last == oracle.last and healthy == 1
        assert host.history[:length].tolist() == list(tokens)
        assert num_states == len(oracle.links)
        for i, (depth, link, pos, inline, head) in enumerate(
            host.states[:num_states].tolist()
        ):
            assert (depth, link, pos) == (
                oracle.lengths[i],
                oracle.links[i],
                oracle.first_pos[i],
            )
            edges = {} if inline == -1 else {host.history[pos + 1].item(): inline}
            if head < -1:
                target = ~head
                edges[host.history[host.states[target, 2]].item()] = target
                head = -1
            while head != -1:
                assert 0 <= head < num_edges
                token, target, head = host.edges[head, :3].tolist()
                assert token not in edges and 0 <= target < num_states
                edges[token] = target
            assert edges == oracle.edges[i]
    for tokens in ((-1,), (100,), (2**32 + 3,)):
        with pytest.raises(ValueError, match="target-vocabulary"):
            host_builder.build(host_builder.prepare(tokens), 100)


def test_native_host_builder_releases_gil_during_prefill(host_builder):
    """Holding the GIL would prevent Python from releasing the native gate."""
    builder = NativeHostBuilder(host_builder.library._name, pin_memory=False)
    probe = builder.library.hybrid_sa_probe_build
    probe.argtypes = builder.build_native.argtypes
    probe.restype = builder.build_native.restype
    builder.build_native = probe
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(builder.build, builder.prepare((1, 2, 1)), 100)
        try:
            assert builder.library.hybrid_sa_probe_wait() == 1
            builder.library.hybrid_sa_probe_release()
            assert future.result(timeout=5).history.tolist() == [1, 2, 1]
        finally:
            builder.library.hybrid_sa_probe_release()


def test_host_import_copies_used_prefixes_and_keeps_sources_until_event(
    host_builder, monkeypatch
):
    """CPU copies exercise ownership/bookkeeping, not actual CUDA DMA."""
    backend = object.__new__(GPUSuffixAutomaton)
    backend.config = HybridConfig(3, 2, 1, 32, history_mode="device")
    backend.device = torch.device("cpu")
    backend._check_stream = lambda: None
    backend.history = torch.full((1, 32), -99, dtype=torch.int32)
    backend.states = torch.full((1, 65, 5), -99, dtype=torch.int32)
    backend.edges = torch.full((1, 33, 6), -99, dtype=torch.int32)
    backend.meta = torch.zeros((1, 7), dtype=torch.int32)
    backend.pending_meta = torch.tensor([[9, 5, 1]])
    backend.host_transfers = []

    class Event:
        complete = False

        def record(self, stream):
            pass

        def query(self):
            return self.complete

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    host = host_builder.build(host_builder.prepare((4, 5, 6, 7, 4)), 100)
    _, num_states, num_edges, _, _, _, _ = host.meta.tolist()
    source = weakref.ref(host)
    backend.import_host(0, host)
    assert backend.history[0, :5].tolist() == [4, 5, 6, 7, 4]
    assert (backend.history[0, 5:] == -99).all()
    assert torch.equal(backend.states[0, :num_states], host.states[:num_states])
    assert (backend.states[0, num_states:] == -99).all()
    assert torch.equal(backend.edges[0, :num_edges], host.edges[:num_edges])
    assert (backend.edges[0, num_edges:] == -99).all()
    assert backend.pending_meta.tolist() == [[9, 5, 1]]
    del host
    gc.collect()
    assert source() is not None
    backend.host_transfers[0][0].complete = True
    replacement = host_builder.build(host_builder.prepare((8,)), 100)
    backend.import_host(0, replacement)
    gc.collect()
    assert source() is None
