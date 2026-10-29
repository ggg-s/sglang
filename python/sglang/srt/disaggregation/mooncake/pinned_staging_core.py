"""Opt-in bounded host staging for native Mooncake KV sends."""

import bisect
import logging
import threading
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.pinned_staging_buffer import PinnedKVStager

logger = logging.getLogger(__name__)


class CorePinnedStager:
    def __init__(self, engine, capacity=128 * 1024 * 1024):
        self.engine = engine
        self.capacity = capacity
        self.buffer = None
        self.device = None
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

    @property
    def poisoned(self):
        return self.buffer is not None and self.buffer.poisoned

    def reserve(self, device):
        """Prepare the single rank-owned slot before layerwise forward."""
        with self.lock:
            self._reserve(device)

    def _reserve(self, device):
        device = (
            torch.device(device)
            if not isinstance(device, int)
            else torch.device("cuda", device)
        )
        if self.buffer is None:
            proxy = SimpleNamespace(
                engine=self.engine, _remote_attn_send_lock=threading.Lock()
            )
            self.buffer = PinnedKVStager(proxy, device.index)
            self.device = device
        if self.device != device:
            raise RuntimeError("Mooncake staging device changed")
        self.buffer.reserve(self.capacity)

    def send(self, blocks, transfer):
        # Worker has waited for the producer event. Serial slot ownership covers
        # D2H AND TE completion; ordinary chunks and layer groups share it.
        with self.lock:
            resolved = [self.resolve(src, length) for src, _, length in blocks]
            if all(t is None for t in resolved):
                return transfer(blocks)
            pending = []
            used = 0

            def flush():
                nonlocal used
                if not pending:
                    return 0
                self.buffer.manager._transfer_data = lambda session, plan: transfer(
                    plan
                )
                try:
                    return self.buffer.send(
                        None, pending, ready_event=None, already_ready=True
                    )
                finally:
                    self.buffer.manager._transfer_data = None
                    pending.clear()
                    used = 0

            for (src, dst, length), tensor in zip(blocks, resolved):
                if tensor is None:
                    rc = flush()
                    if rc != 0:
                        return rc
                    rc = transfer([(src, dst, length)])
                    if rc != 0:
                        return rc
                    continue
                self._reserve(tensor.device)
                offset = 0
                while offset < length:
                    n = min(length - offset, self.capacity - used)
                    pending.append((src + offset, dst + offset, n))
                    offset += n
                    used += n
                    if used == self.capacity:
                        rc = flush()
                        if rc != 0:
                            return rc
            return flush()

    def close(self):
        with self.lock:
            if self.buffer is not None and not self.buffer.close():
                return False
            if self in self.engine._core_staging_owners:
                self.engine._core_staging_owners.remove(self)
            return True
