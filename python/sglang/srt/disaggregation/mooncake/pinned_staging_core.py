"""Opt-in bounded host staging for native Mooncake KV sends."""

import bisect
import logging
import threading

import torch

logger = logging.getLogger(__name__)


class CorePinnedStager:
    def __init__(self, engine, capacity=128 * 1024 * 1024):
        self.engine = engine
        self.capacity = capacity
        self.local = threading.local()
        self.slots = []
        self.lock = threading.Lock()
        self.regions = []
        self.starts = []
        # A failed TE call may still reference pinned storage after returning.
        owners = getattr(engine, "_core_staging_owners", None)
        if owners is None:
            owners = engine._core_staging_owners = []
        owners.append(self)

    def set_buffers(self, tensors, ptrs, item_lens):
        regions = []
        if len(tensors) == len(ptrs) == len(item_lens):
            for tensor, ptr, stride in zip(tensors, ptrs, item_lens):
                if (
                    tensor.data_ptr() != ptr
                    or not tensor.is_contiguous()
                    or tensor.device.type != "cuda"
                    or stride <= 0
                ):
                    break
                regions.append(
                    (ptr, tensor.numel() * tensor.element_size(), stride, tensor)
                )
            else:
                regions.sort(key=lambda r: r[0])
                if all(a[0] + a[1] <= b[0] for a, b in zip(regions, regions[1:])):
                    self.regions = regions
                    self.starts = [r[0] for r in regions]
                    return
        self.regions = []
        self.starts = []
        logger.warning(
            "Mooncake host staging disabled: KV tensor/address layout mismatch"
        )

    def resolve(self, addr, length):
        i = bisect.bisect_right(self.starts, addr) - 1
        if i < 0:
            return None
        base, size, stride, tensor = self.regions[i]
        offset = addr - base
        if offset % stride or length <= 0 or offset + length > size:
            return None
        return tensor.view(torch.uint8).reshape(-1)[offset : offset + length]

    def _slot(self, device):
        slot = getattr(self.local, "slot", None)
        if slot is None:
            with torch.cuda.device(device):
                slot = dict(
                    host=torch.empty(self.capacity, dtype=torch.uint8, pin_memory=True),
                    stream=torch.cuda.Stream(device=device),
                    poisoned=True,
                    device=device,
                )
            self.local.slot = slot
            with self.lock:
                self.slots.append(slot)
            rc = self.engine.batch_register([slot["host"].data_ptr()], [self.capacity])
            if rc != 0:
                raise RuntimeError("Mooncake core pinned registration failed")
            slot["poisoned"] = False
        if slot["poisoned"] or slot["device"] != device:
            raise RuntimeError(
                "Mooncake core staging slot quarantined or device changed"
            )
        return slot

    def send(self, blocks, transfer):
        # The transfer worker has synchronized the submission-time producer event.
        resolved = [self.resolve(src, length) for src, _, length in blocks]
        if all(t is None for t in resolved):
            return transfer(blocks)
        slot = None
        pending = []
        used = 0

        def flush():
            nonlocal used
            if not pending:
                return 0
            slot["stream"].synchronize()
            rc = transfer(list(pending))
            if rc != 0:
                slot["poisoned"] = True
            pending.clear()
            used = 0
            return rc

        try:
            for (src, dst, length), tensor in zip(blocks, resolved):
                if tensor is None:
                    rc = flush()
                    if rc != 0:
                        return rc
                    rc = transfer([(src, dst, length)])
                    if rc != 0:
                        return rc
                    continue
                slot = self._slot(tensor.device)
                offset = 0
                while offset < length:
                    n = min(length - offset, self.capacity - used)
                    with (
                        torch.cuda.device(slot["device"]),
                        torch.cuda.stream(slot["stream"]),
                    ):
                        slot["host"][used : used + n].copy_(
                            tensor[offset : offset + n], non_blocking=True
                        )
                    pending.append((slot["host"].data_ptr() + used, dst + offset, n))
                    offset += n
                    used += n
                    if used == self.capacity:
                        rc = flush()
                        if rc != 0:
                            return rc
            return flush()
        except Exception:
            if slot is not None:
                slot["poisoned"] = True
            raise

    def close(self):
        # Called only after all transfer workers/executors have stopped.
        for slot in self.slots:
            if slot["poisoned"]:
                continue
            slot["stream"].synchronize()
            if self.engine.batch_deregister([slot["host"].data_ptr()]) != 0:
                slot["poisoned"] = True
                continue
            slot["host"] = None
        if not any(s["poisoned"] for s in self.slots):
            self.engine._core_staging_owners.remove(self)
