"""Per-graph scratch buffers allocated during warmup, never during capture."""

from contextlib import contextmanager
from contextvars import ContextVar

_current_pool = ContextVar("capture_workspace_pool", default=None)


def get_capture_workspace(key, allocate):
    pool = _current_pool.get()
    return allocate() if pool is None else pool.get(key, allocate)


class CaptureWorkspacePool:
    """Keep each call's reduction locks independent across calls and graphs.

    The allocation trace must repeat across warmup and capture. A different
    trace fails before allocating in capture. Captured zeroing resets the locks
    on every replay; no mutable workspace is shared with another graph bank.
    """

    def __init__(self):
        self.entries = []
        self.cursor = 0
        self.capturing = False

    @contextmanager
    def scope(self, *, capturing=False):
        if _current_pool.get() is not None:
            raise RuntimeError("Capture workspace scopes cannot nest")
        self.cursor = 0
        self.capturing = capturing
        token = _current_pool.set(self)
        try:
            yield
            if self.cursor != len(self.entries):
                raise RuntimeError("Capture workspace call count differs from warmup")
        finally:
            _current_pool.reset(token)
            self.capturing = False

    def get(self, key, allocate):
        index = self.cursor
        if index == len(self.entries):
            if self.capturing:
                raise RuntimeError("Capture workspace was not allocated during warmup")
            self.entries.append((key, allocate()))
        expected, workspace = self.entries[index]
        if expected != key:
            raise RuntimeError("Capture workspace request differs from warmup")
        self.cursor += 1
        # This memset is captured, while device-property queries and allocation
        # happen only when the slot is first seen outside stream capture.
        workspace.zero_()
        return workspace
