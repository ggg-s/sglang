"""Pinned staging and explicit copy streams for pre-layerwise Mooncake TCP.

The native manager still builds the address plan, bounds batches and owns peer
failure handling. Only the source addresses change. No tensors are gathered on
the compute stream and no default-stream CUDA copies are issued by this adapter.
"""

from __future__ import annotations

import threading


class CudaPinnedSlot:
    """One pinned allocation and an explicitly nonblocking copy stream."""

    def __init__(self, capacity, device):
        import torch
        from cuda.bindings import runtime as cudart

        self.device = device
        self.capacity = capacity
        self.cudart = cudart
        self.torch = torch
        # Construct and register before prefill, not on the busy transfer path.
        with torch.cuda.device(device):
            self.host = torch.empty(capacity, dtype=torch.uint8, pin_memory=True)
            err, raw = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
            if int(err) != 0:
                raise RuntimeError(f"pre-layerwise copy stream creation failed: {err}")
            self.raw_stream = raw
            try:
                self.stream = torch.cuda.ExternalStream(int(raw), device=device)
                self.done = torch.cuda.Event()
            except Exception:
                cudart.cudaStreamDestroy(raw)
                raise
        self.ptr = self.host.data_ptr()

    def warmup(self):
        """Prime the real registered destination and stream before group sends."""
        with self.torch.cuda.device(self.device), self.torch.cuda.stream(self.stream):
            # A private scratch byte avoids reading KV while a previous chunk
            # might still be writing it. Keep it alive until the D2H completes.
            source = self.torch.empty(
                1, dtype=self.torch.uint8, device=f"cuda:{self.device}"
            )
            # Retain scratch even if a CUDA call raises or reports uncertainty.
            self._warmup_source = source
            (err,) = self.cudart.cudaMemcpyAsync(
                self.ptr,
                source.data_ptr(),
                1,
                self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.raw_stream,
            )
            if int(err) != 0:
                raise RuntimeError(f"pre-layerwise D2H warmup failed: {err}")
            self.done.record(self.stream)
            self.done.synchronize()
            self._warmup_source = None

    def copy(self, blocks, ready_event):
        with self.torch.cuda.device(self.device):
            self.stream.wait_event(ready_event)
            offset = 0
            for src, _dst, length in blocks:
                (err,) = self.cudart.cudaMemcpyAsync(
                    self.ptr + offset,
                    src,
                    length,
                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.raw_stream,
                )
                if int(err) != 0:
                    raise RuntimeError(f"pre-layerwise D2H submission failed: {err}")
                offset += length
            self.done.record(self.stream)
            self.done.synchronize()

    def close(self):
        with self.torch.cuda.device(self.device):
            self.stream.synchronize()
            (err,) = self.cudart.cudaStreamDestroy(self.raw_stream)
            if int(err) != 0:
                raise RuntimeError(
                    f"pre-layerwise copy stream destruction failed: {err}"
                )
        self.stream = self.host = None


class PinnedKVStager:
    """Bounded staging storage owned by one rank's native engine.

    P shares an instance across sends; receive staging uses the native ring.

    The slot lock covers D2H and the synchronous native send. A nonzero return
    or exception is NOT a completion proof: retain both registration and storage
    until process exit, and reject future reuse. The engine owns a strong
    reference even if scheduler teardown or an abort discards the controller.
    """

    def __init__(self, manager, device, *, slot_factory=CudaPinnedSlot):
        self.manager = manager
        self.device = device
        self.slot_factory = slot_factory
        self.lock = threading.Lock()
        self.slot = None
        self.registered = False
        self.poisoned = False
        self.closed = False
        owners = getattr(manager.engine, "_pre_layerwise_staging_owners", None)
        if owners is None:
            owners = manager.engine._pre_layerwise_staging_owners = []
        owners.append(self)

    def _check_live(self):
        if self.poisoned or self.closed:
            raise RuntimeError("pre-layerwise staging is quarantined or closed")

    def _release_slot(self):
        if self.slot is not None:
            if self.registered:
                with self.manager._remote_attn_send_lock:
                    self.manager.engine.deregister(self.slot.ptr)
                self.registered = False
            self.slot.close()
            self.slot = None

    def reserve(self, capacity):
        """Called before forward; never allocate/register in a group send."""
        if capacity <= 0:
            raise ValueError("pre-layerwise staging capacity must be positive")
        with self.lock:
            self._check_live()
            if self.slot is not None and self.slot.capacity >= capacity:
                return
            try:
                self._release_slot()
                self.slot = self.slot_factory(capacity, self.device)
                # A registration exception can have an unknown native outcome.
                self.registered = True
                with self.manager._remote_attn_send_lock:
                    self.manager.engine.register(self.slot.ptr, self.slot.capacity)
                self.slot.warmup()
            except Exception:
                self.poisoned = True
                raise

    def send(self, session, blocks, *, ready_event):
        if not blocks:
            return 0
        with self.lock:
            self._check_live()
            blocks = [(int(src), int(dst), int(n)) for src, dst, n in blocks]
            if ready_event is None or any(n <= 0 for _, _, n in blocks):
                raise ValueError("staged KV requires a ready event and positive blocks")
            if self.slot is None or sum(n for _, _, n in blocks) > self.slot.capacity:
                raise RuntimeError("native KV batch exceeds reserved staging capacity")
            host_blocks = []
            offset = 0
            for _src, dst, length in blocks:
                host_blocks.append((self.slot.ptr + offset, dst, length))
                offset += length
            try:
                self.slot.copy(blocks, ready_event)
                # Keep the existing send-lock hook and the same TE/session.
                rc = self.manager._transfer_data(session, host_blocks)
            except Exception:
                # Copies or writes may still reference storage. Do not free it,
                # unregister it, or fall back to a second GPU-pointer send.
                self.poisoned = True
                raise
            if rc != 0:
                self.poisoned = True
            return rc

    def close(self):
        with self.lock:
            if self.closed:
                return True
            if self.poisoned:
                return False
            try:
                self._release_slot()
            except Exception:
                self.poisoned = True
                return False
            self.closed = True
            self.manager.engine._pre_layerwise_staging_owners.remove(self)
            return True
