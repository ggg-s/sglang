"""Generation-scoped host destinations for final-chunk layer groups.

One slot per participating request/rank. Native sub-batches fill assigned offsets;
a single group manifest starts H2D, and its GPU-complete ACK permits slot reuse.
The original native GPU registrations and all non-selected sends stay intact.
"""

from __future__ import annotations

import bisect
import functools
import threading
import time

from .pinned_staging import CudaPinnedSlot, PinnedKVStager
from .pre_layerwise import read_json, write_json


def _wait(path, generation, timeout):
    deadline = time.monotonic() + timeout
    while True:
        failure = read_json(path.parent / "rx-error.json")
        if failure and failure.get("generation") == generation:
            raise RuntimeError(f"receive staging failed: {failure['error']}")
        value = read_json(path)
        if value and value.get("generation") == generation:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError(f"receive staging timed out: {path.name}")
        time.sleep(0.002)


def _intervals(blocks):
    """Canonical disjoint byte coverage; reject duplicate or overlapping writes."""
    merged = []
    for dst, size in sorted(blocks):
        if size <= 0 or (merged and dst < merged[-1][1]):
            raise ValueError("invalid or overlapping receive destinations")
        end = dst + size
        if merged and dst == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((dst, end))
    return merged


class ReceiveStaging:
    """D-side lease. Errors retain both pinned storage and destination KV."""

    def __init__(
        self,
        manager,
        device,
        path,
        generation,
        indices,
        layers,
        timeout,
        *,
        slot_factory=CudaPinnedSlot,
    ):
        self.path, self.generation = path, generation
        self.indices = tuple(int(i) for i in indices)
        self.layers, self.timeout = layers, timeout
        self.ptrs = tuple(int(i) for i in manager.kv_args.kv_data_ptrs)
        self.sizes = tuple(int(i) for i in manager.kv_args.kv_item_lens)
        ids = list(manager.kv_args.kv_layer_ids)
        self.layer_ids = ids or list(range(layers)) * 2
        self.storage = PinnedKVStager(manager, device, slot_factory=slot_factory)
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.active = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _publish(self, name, **values):
        write_json(self.path / name, {"generation": self.generation, **values})

    def _run(self):
        try:
            # Unselected offers allocate no pinned memory. The scheduler closes
            # them when native PD completes; allocation starts only on a plan.
            while not self.stop.is_set():
                plan = read_json(self.path / "rx-plan.json")
                if plan and plan.get("generation") == self.generation:
                    break
                self.stop.wait(0.002)
            else:
                return
            self.active = True
            groups = plan["groups"]
            if [i for group in groups for i in group] != list(range(self.layers)):
                raise ValueError("receive plan must partition all layers in order")
            start, end = plan["start"], plan["end"]
            if not 0 <= start < end == len(self.indices):
                raise ValueError("receive plan must cover the final token chunk")
            if (
                len(self.ptrs) != 2 * self.layers
                or len(self.sizes) != len(self.ptrs)
                or any(self.layer_ids.count(layer) != 2 for layer in range(self.layers))
            ):
                raise ValueError("receive staging requires dense MHA K/V buffers")
            # Coalesce token rows once, rather than materializing a Python tuple
            # for every token in every K/V buffer on the prefill critical path.
            token_spans = _intervals([(token, 1) for token in self.indices[start:end]])
            expected, layouts = [], []
            for group in groups:
                blocks = [
                    (ptr + lo * size, (hi - lo) * size)
                    for ptr, size, layer in zip(self.ptrs, self.sizes, self.layer_ids)
                    if layer in group
                    for lo, hi in token_spans
                ]
                expected.append(_intervals(blocks))
                # Keep allocation boundaries even if two KV buffers happen to
                # have adjacent addresses. One H2D must stay inside a buffer.
                layouts.append(sorted(blocks))
            capacity = max(sum(hi - lo for lo, hi in spans) for spans in expected)
            self.storage.reserve(capacity)
            self._publish(
                "rx-slot.json",
                ptr=self.storage.slot.ptr,
                capacity=capacity,
                groups=layouts,
            )
            for index, (layers, spans) in enumerate(zip(groups, expected)):
                message = _wait(
                    self.path / f"rx-group-{index}.json", self.generation, self.timeout
                )
                blocks = [(int(dst), int(n)) for dst, n in message["blocks"]]
                if message.get("layers") != layers or _intervals(blocks) != spans:
                    raise ValueError("receive group does not match reserved prompt KV")
                if sum(n for _, n in blocks) > capacity:
                    raise ValueError("receive group exceeds pinned slot")
                self.storage.slot.copy_to_device(layouts[index])
                # The private-stream event has completed before this ACK. The
                # sender publishes layer proofs only after observing this ACK.
                self._publish(
                    f"rx-ack-{index}.json", layers=layers, h2d_done_at=time.monotonic()
                )
            self.done.set()
        except Exception as exc:
            self.error = str(exc)
            self.storage.poisoned = True
            self._publish("rx-error.json", error=self.error)

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"receive staging unsettled; retaining KV: {self.error}")

    def close(self):
        # Only call after canonical native success, or before releasing KV on
        # abort. A selected transfer cannot be cancelled by freeing its slot.
        self.stop.set()
        self.thread.join(timeout=self.timeout)
        self.check()
        if self.thread.is_alive() or (self.active and not self.done.is_set()):
            self.storage.poisoned = True
            raise RuntimeError("receive staging did not drain; retaining KV")
        if not self.storage.close():
            raise RuntimeError("receive staging registration remains quarantined")


class HostDestination:
    """P-side adapter preserving the exact native GPU destination address plan."""

    def __init__(self, source, groups):
        self.source = source
        self.groups = {tuple(layers): index for index, (layers, _) in enumerate(groups)}
        self.lock = threading.Lock()
        self.offset = 0
        self.blocks = []
        self.next_group = 0
        self.poisoned = False
        write_json(
            source.path / "rx-plan.json",
            {
                "generation": source.generation,
                "start": source.index_slice.start,
                "end": source.index_slice.stop,
                "groups": [list(g) for g, _ in groups],
            },
        )
        slot = _wait(
            source.path / "rx-slot.json",
            source.generation,
            source.owner.config.drain_timeout,
        )
        self.ptr, self.capacity = int(slot["ptr"]), int(slot["capacity"])
        if self.ptr <= 0 or self.capacity <= 0:
            raise ValueError("invalid receive staging slot")
        self.layouts = []
        for blocks in slot["groups"]:
            _intervals(blocks)  # Validate overlap without merging allocations.
            spans = [(lo, lo + size) for lo, size in sorted(blocks)]
            offsets, offset = [], 0
            for lo, hi in spans:
                offsets.append(offset)
                offset += hi - lo
            if offset > self.capacity:
                raise ValueError("receive layout exceeds slot")
            self.layouts.append(([lo for lo, _ in spans], spans, offsets))
        if len(self.layouts) != len(groups):
            raise ValueError("receive layout group count changed")

    def transfer(self, layers, ready_event, session, blocks):
        with self.lock:
            try:
                if self.poisoned or self.groups[tuple(layers)] != self.next_group:
                    raise RuntimeError("receive staging group order changed")
                if session != self.source.peers[0][0].mooncake_session_id:
                    raise RuntimeError("receive staging session changed")
                total = sum(int(n) for _, _, n in blocks)
                if (
                    any(n <= 0 for _, _, n in blocks)
                    or self.offset + total > self.capacity
                ):
                    raise ValueError("native group exceeds receive staging capacity")
                remapped = []
                starts, spans, offsets = self.layouts[self.next_group]
                for src, dst, n in blocks:
                    dst, n = int(dst), int(n)
                    span = bisect.bisect_right(starts, dst) - 1
                    if span < 0 or dst + n > spans[span][1]:
                        raise ValueError(
                            "native destination is outside reserved prompt rows"
                        )
                    host = self.ptr + offsets[span] + dst - starts[span]
                    remapped.append((int(src), host, n))
                    self.blocks.append((dst, n))
                    self.offset += n
                send = self.source.req.disagg_kv_sender.kv_mgr._transfer_data
                if self.source.stager is not None:
                    send = functools.partial(
                        self.source.stager.send, ready_event=ready_event
                    )
                rc = send(session, remapped)
                if rc != 0:
                    self.poisoned = True
                return rc
            except Exception:
                self.poisoned = True
                raise

    def complete(self, layers):
        with self.lock:
            try:
                index = self.groups[tuple(layers)]
                if self.poisoned or index != self.next_group or not self.blocks:
                    raise RuntimeError("receive staging group is incomplete")
                spans = _intervals(self.blocks)
                expected = _intervals(
                    [(lo, hi - lo) for lo, hi in self.layouts[index][1]]
                )
                if spans != expected:
                    raise ValueError("native group does not cover reserved prompt rows")
                write_json(
                    self.source.path / f"rx-group-{index}.json",
                    {
                        "generation": self.source.generation,
                        "layers": list(layers),
                        "blocks": [(lo, hi - lo) for lo, hi in spans],
                    },
                )
                ack = _wait(
                    self.source.path / f"rx-ack-{index}.json",
                    self.source.generation,
                    self.source.owner.config.drain_timeout,
                )
                if ack.get("layers") != list(layers):
                    raise RuntimeError("receive staging ACK identity mismatch")
                self.offset, self.blocks = 0, []
                self.next_group += 1
            except Exception:
                self.poisoned = True
                raise
