"""Pinned host SA storage and GIL-free native construction; no CUDA calls here."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch

from hybrid_spec.backends.sa_layout import COMPACT_MAX_HISTORY, state_layout


@dataclass
class HostAutomaton:
    tokens: torch.Tensor
    history: torch.Tensor
    states: torch.Tensor
    edges: torch.Tensor
    meta: torch.Tensor


class NativeHostBuilder:
    def __init__(self, library_path, *, pin_memory=True):
        # CDLL (not PyDLL) releases the GIL for the full native call. The same
        # pure C++ source is compiled into CPU tests and the serving extension.
        self.library = ctypes.CDLL(str(library_path))
        self.build_native = self.library.hybrid_sa_build_host
        self.build_native.argtypes = (
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        self.build_native.restype = ctypes.c_int
        self.build_compact_native = self.library.hybrid_sa_build_host_compact
        self.build_compact_native.argtypes = self.build_native.argtypes
        self.build_compact_native.restype = ctypes.c_int
        self.pin_memory = pin_memory

    def prepare(self, history, *, compact=False):
        # Allocate/snapshot before GPU prefill, so background work does not
        # call the CUDA pinned allocator or access mutable request objects.
        capacity = max(1, len(history))
        if compact and capacity > COMPACT_MAX_HISTORY:
            raise ValueError("compact SA history capacity exceeded")
        kwargs = dict(dtype=torch.int32, device="cpu", pin_memory=self.pin_memory)
        dtype, fields = state_layout(capacity) if compact else (torch.int32, 10)
        return HostAutomaton(
            torch.tensor(history, dtype=torch.int64, device="cpu"),
            torch.empty(capacity, **kwargs),
            torch.empty((2 * capacity + 1, fields), **(kwargs | {"dtype": dtype})),
            torch.empty((capacity + 1, 6), **kwargs),
            torch.empty(7, **kwargs),
        )

    def build(self, host, vocab_size):
        if not 0 < vocab_size <= 2**31 - 1:
            raise ValueError("vocab_size must fit a positive int32")
        build = (
            self.build_compact_native
            if host.states.dtype == torch.uint16
            else self.build_native
        )
        code = build(
            host.tokens.data_ptr(),
            host.tokens.numel(),
            vocab_size,
            host.history.data_ptr(),
            host.states.data_ptr(),
            host.edges.data_ptr(),
            host.meta.data_ptr(),
            host.history.numel(),
        )
        if code == 1:
            raise ValueError("host SA requires valid target-vocabulary tokens")
        if code:
            raise RuntimeError("host SA exceeded its reserved state capacity")
        return host
