"""Serialize native TE access with bounded priority for latency-sensitive replies."""

import threading
import time
from collections import deque
from contextlib import contextmanager


class TransferLock:
    def __init__(self, urgent_burst=4):
        self.condition = threading.Condition()
        self.urgent, self.bulk = deque(), deque()
        self.busy = False
        self.burst = 0
        self.urgent_burst = urgent_burst
        self.wait_ns = [0, 0]
        self.acquisitions = [0, 0]

    @contextmanager
    def priority(self):
        self.acquire(urgent=True)
        try:
            yield self
        finally:
            self.release()

    def acquire(self, *, urgent=False):
        queue = self.urgent if urgent else self.bulk
        ticket = object()
        started = time.perf_counter_ns()
        with self.condition:
            queue.append(ticket)
            try:
                while True:
                    selected = (
                        self.urgent
                        if self.urgent
                        and (not self.bulk or self.burst < self.urgent_burst)
                        else self.bulk
                    )
                    if not self.busy and selected and selected[0] is ticket:
                        selected.popleft()
                        self.busy = True
                        self.burst = self.burst + 1 if urgent else 0
                        self.wait_ns[int(urgent)] += time.perf_counter_ns() - started
                        self.acquisitions[int(urgent)] += 1
                        return True
                    self.condition.wait()
            except BaseException:
                queue.remove(ticket)
                self.condition.notify_all()
                raise

    def release(self):
        with self.condition:
            if not self.busy:
                raise RuntimeError("release of an idle transfer lock")
            self.busy = False
            self.condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
