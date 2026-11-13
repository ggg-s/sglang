"""Wake idle ingress for file controls; all commands still run on the TP thread.

The watcher owns only a pipe and directory names. It never reads requests, calls
Torch/distributed, claims a command, or mutates scheduler/KV state.
"""

import os
import threading
from pathlib import Path

import zmq

POLL_SECONDS = 0.005
MAX_IDLE_MS = 10


class WakePoller:
    def __init__(self, poller, directory):
        self.poller = poller
        self.directory = Path(directory)
        self.reader, self.writer = os.pipe()
        os.set_blocking(self.reader, False)
        os.set_blocking(self.writer, False)
        self.lock = threading.Lock()
        self.closed = self.signalled = False
        self.stop = threading.Event()
        poller.register(self.reader, zmq.POLLIN)
        self.thread = threading.Thread(
            target=self._watch, daemon=True, name="remote-attn-control-wakeup"
        )
        self.thread.start()

    def __getattr__(self, name):
        return getattr(self.poller, name)

    def _watch(self):
        while not self.stop.is_set():
            try:
                pending = (self.directory / "cmd.0.json").exists() or any(
                    self.directory.glob("ctlq.*.json")
                )
                if pending:
                    with self.lock:
                        if not self.closed and not self.signalled:
                            os.write(self.writer, b"c")
                            self.signalled = True
            except OSError:
                pass  # Directory creation/teardown races are retried.
            self.stop.wait(POLL_SECONDS)

    def poll(self, timeout=None, *args, **kwargs):
        # Rank 0 can claim the file before a peer's watcher sees it. Bound
        # the idle wait so that peer can still reach the TP broadcast promptly.
        timeout = (
            MAX_IDLE_MS if timeout is None or timeout < 0 else min(timeout, MAX_IDLE_MS)
        )
        events = self.poller.poll(timeout, *args, **kwargs)
        with self.lock:
            if not self.closed:
                try:
                    while os.read(self.reader, 4096):
                        pass
                except BlockingIOError:
                    pass
                self.signalled = False
        return [(fd, event) for fd, event in events if fd != self.reader]

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.stop.set()
        self.thread.join(timeout=0.2)
        self.poller.unregister(self.reader)
        os.close(self.reader)
        os.close(self.writer)


def sleep(original, scheduler):
    from .handoff_control import _CTL_DIR

    sleeper = scheduler.idle_sleeper
    if _CTL_DIR and sleeper is not None:
        if hasattr(sleeper, "poller"):
            if not isinstance(sleeper.poller, WakePoller):
                sleeper.poller = WakePoller(sleeper.poller, _CTL_DIR)
                scheduler._remote_attn_control_wakeup = sleeper.poller
        if hasattr(sleeper, "timeout_ms"):
            # The Rust ingress parks on its own request ring, not a zmq Poller.
            # Keep its native path, with a bounded interval for file controls.
            sleeper.timeout_ms = min(sleeper.timeout_ms, MAX_IDLE_MS)
    return original(scheduler)


def close(scheduler):
    wake = getattr(scheduler, "_remote_attn_control_wakeup", None)
    if wake is not None:
        wake.close()
        sleeper = scheduler.idle_sleeper
        if sleeper is not None and getattr(sleeper, "poller", None) is wake:
            sleeper.poller = wake.poller
        scheduler._remote_attn_control_wakeup = None
