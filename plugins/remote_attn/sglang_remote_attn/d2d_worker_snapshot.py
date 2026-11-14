"""Publish advisory D2D status without issuing a scheduler control command."""

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
ENABLED = os.environ.get("HANDOFF_D2D_STATUS_SNAPSHOT") == "1"
INTERVAL_S = 0.02


class Publisher:
    def __init__(self, path, write):
        self.path, self.write = path, write
        self.cv = threading.Condition()
        self.pending = None
        self.closed = False
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="d2d-status-publisher"
        )
        self.thread.start()

    def publish(self, snapshot):
        with self.cv:
            self.pending = snapshot
            self.cv.notify()

    def _run(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                snapshot, self.pending = self.pending, None
            try:
                self.write(self.path, snapshot)
            except (OSError, TypeError, ValueError):
                # The controller can still use the original status command.
                logger.debug("Could not publish D2D status snapshot", exc_info=True)

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify()
        self.thread.join(timeout=1)


def sample(scheduler):
    if not ENABLED:
        return
    from . import dsv_d2d, dsv_gateway_control, handoff_control

    directory = handoff_control._CTL_DIR
    controller = getattr(scheduler, "_pre_layerwise", None)
    if (
        not directory
        or not getattr(controller, "dsv41_remote", False)
        or scheduler.disaggregation_mode.value != "decode"
    ):
        return

    now = time.monotonic()
    if now < getattr(scheduler, "_d2d_status_next_at", 0):
        return
    scheduler._d2d_status_next_at = now + INTERVAL_S
    try:
        dsv_d2d.validate(controller)
        sampled_at = time.time()
        # The writer never touches request objects, allocators or TP groups.
        status = dsv_gateway_control.worker_status(controller, advisory=True)
        status["sample_ms"] = (time.monotonic() - now) * 1000
        publisher = getattr(scheduler, "_d2d_status_publisher", None)
        if publisher is None:
            publisher = scheduler._d2d_status_publisher = Publisher(
                Path(directory) / f"d2d-status.{controller.rank}.json",
                handoff_control._write_result,
            )
        sequence = getattr(scheduler, "_d2d_status_sequence", 0) + 1
        scheduler._d2d_status_sequence = sequence
        publisher.publish(
            dict(
                version=1,
                pid=os.getpid(),
                sequence=sequence,
                sampled_at=sampled_at,
                status=status,
            )
        )
    except Exception:
        logger.debug("Could not sample D2D worker status", exc_info=True)


def close(scheduler):
    publisher = getattr(scheduler, "_d2d_status_publisher", None)
    if publisher is not None:
        publisher.close()
        scheduler._d2d_status_publisher = None
