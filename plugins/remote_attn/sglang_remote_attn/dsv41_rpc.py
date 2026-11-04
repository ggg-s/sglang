"""Bounded per-endpoint sparse RPC batches, with submission-time read ownership."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from . import remote_data_plane as dp


@dataclass
class Row:
    index: int
    target: object
    query: object
    positions: list
    layer: int
    ratio: int
    scale: float

    def header(self):
        return dict(
            handoff_id=self.target.route["handoff_id"],
            epoch=self.target.route["epoch"],
            token=self.target.route["token"],
            dsv41_sparse=True,
            layer=self.layer,
            ratio=self.ratio,
            positions=self.positions,
            scale=float(self.scale),
        )


def groups(rows):
    by_peer = {}
    for row in rows:
        route = row.target.route
        key = (
            route["host"],
            int(route["port"]),
            row.layer,
            tuple(row.query.shape),
            row.query.dtype,
        )
        by_peer.setdefault(key, []).append(row)
    return [
        part[i : i + 4] for part in by_peer.values() for i in range(0, len(part), 4)
    ]


class Pending:
    def __init__(self, future, rows):
        self.future, self.rows = future, rows

    def result(self):
        out, lse, done = self.future.result()
        if out.is_cuda:
            stream = torch.cuda.current_stream(out.device)
            stream.wait_event(done)
            out.record_stream(stream)
            lse.record_stream(stream)
        for row in self.rows:
            row.target.check()
        return [(row.index, (out[i], lse[i])) for i, row in enumerate(self.rows)]


class Batches:
    def __init__(self, controller):
        self.controller = controller
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="dsv41-rpc"
        )
        self.credits = threading.BoundedSemaphore(16)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.clients = {}
        self.closed = False
        self.stats = dict(batches=0, rows=0, queue_ns=0, rpc_ns=0, failed=0)

    def submit(self, rows):
        if not 1 <= len(rows) <= 4:
            raise ValueError("DSV sparse RPC batches require 1..4 rows")
        for row in rows:
            row.target.check()
            if row.target.closed:
                raise RuntimeError("submitting a retired sparse target")
        # This is the producer event only, recorded before D local attention.
        device = rows[0].query.device
        ready = dp.event(device)
        self.credits.acquire()
        ios = [dp.io_for(row.target.req) for row in rows]
        begun = []
        try:
            with self.lock:
                if self.closed:
                    raise RuntimeError("sparse RPC batch queue is closed")
                for io in ios:
                    io.begin()
                    begun.append(io)
                future = self.executor.submit(
                    self._run, rows, ready, ios, time.perf_counter_ns()
                )
                self.stats["batches"] += 1
                self.stats["rows"] += len(rows)
            return Pending(future, rows)
        except BaseException:
            for io in begun:
                io.poison()
            self.credits.release()
            raise

    def _run(self, rows, ready, ios, queued):
        start = time.perf_counter_ns()
        device = rows[0].query.device
        try:
            stream = None
            if device.type == "cuda":
                if not hasattr(self.local, "stream"):
                    with torch.cuda.device(device):
                        self.local.stream = torch.cuda.Stream(device=device)
                stream = self.local.stream
            with (
                torch.cuda.device(device) if stream is not None else nullcontext(),
                torch.cuda.stream(stream) if stream is not None else nullcontext(),
            ):
                if stream is not None:
                    stream.wait_event(ready)
                for row in rows:
                    row.target.check()
                if dp.enabled():
                    _, out, lse = dp.call(
                        self.controller,
                        rows[0].target.route,
                        [r.query for r in rows],
                        [r.header() for r in rows],
                        [r.target.req for r in rows],
                    )
                else:
                    out, lse = self._tcp(rows)
                expected = (len(rows), *rows[0].query.shape)
                if out.shape != expected or lse.shape != expected[:-1]:
                    raise RuntimeError("invalid batched sparse reply shape")
                done = dp.event(device)
                for row, io in zip(rows, ios):
                    row.target.calls += 1
                    row.target.touched.add(row.layer)
                    io.finish(done)
                return out, lse, done
        except BaseException:
            for io in ios:
                io.poison()
            with self.lock:
                self.stats["failed"] += 1
            raise
        finally:
            with self.lock:
                self.stats["queue_ns"] += start - queued
                self.stats["rpc_ns"] += time.perf_counter_ns() - start
            self.credits.release()

    def _tcp(self, rows):
        from .dsv41_remote import runtime

        route = rows[0].target.route
        key = (route["host"], int(route["port"]))
        with self.lock:
            client = self.clients.get(key)
            if client is None:
                client = self.clients[key] = runtime().TargetClient(*key)
        for row in rows:
            r = row.target.route
            client.attach(r["handoff_id"], r["token"], r["epoch"])
        queries = torch.stack([r.query for r in rows])
        if queries.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("sparse query requires BF16/FP16")
        cpu = queries.detach().to("cpu").contiguous()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        headers = [
            dict(
                r.header(),
                q_shape=list(cpu[i].shape),
                q_dtype=str(cpu.dtype).split(".")[-1],
                q_bytes=cpu[i].numel() * cpu.element_size(),
            )
            for i, r in enumerate(rows)
        ]
        response, payload = client.call_batch(
            dict(rows=headers, total_q_bytes=len(raw)), raw
        )
        if not response.get("ok") or len(response.get("rows", ())) != len(rows):
            raise RuntimeError("invalid sparse batch response")
        if response.get("o_batch_shape"):
            out, lse = payload
        else:
            if len(payload) != len(rows):
                raise RuntimeError("invalid sparse batch tensor count")
            out, lse = (
                torch.stack([p[0] for p in payload]),
                torch.stack([p[1] for p in payload]),
            )
        return out.to(queries.device, non_blocking=True), lse.to(
            queries.device, non_blocking=True
        )

    def detach(self, target):
        key = (target.route["host"], int(target.route["port"]))
        with self.lock:
            client = self.clients.get(key)
        if client is not None:
            client.detach(target.route["handoff_id"])

    def close(self):
        with self.lock:
            self.closed = True
        self.executor.shutdown(wait=True)
        for client in self.clients.values():
            client.close()
        self.clients.clear()


def pool(controller):
    result = getattr(controller, "dsv41_rpc_batches", None)
    if result is None:
        result = controller.dsv41_rpc_batches = Batches(controller)
    return result
