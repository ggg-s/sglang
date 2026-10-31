"""Actual Mooncake GPU buffers via source-loaded native SGL engine methods.

4060 uses Mooncake TCP internally; NVLink/RDMA hardware is not emulated.
"""

import logging
import os
import threading
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from dsv41_remote_probe_support import definitions
from sglang_remote_attn import handoff_control as rt
from sglang_remote_attn import remote_data_plane as dp


class Transport:
    def __init__(self, root):
        from mooncake.engine import TransferEngine

        os.environ["MC_FORCE_TCP"] = "1"
        os.environ["HANDOFF_REMOTE_GPU_TRANSFER"] = "1"
        self.Native = TransferEngine
        self.Engine = definitions(
            root,
            "python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py",
            owner="MooncakeTransferEngine",
            members={
                "batch_register",
                "batch_deregister",
                "batch_transfer_sync",
                "get_session_id",
            },
        )
        self.Engine.batch_transfer_sync.__globals__["logger"] = logging.getLogger(
            __name__
        )
        self.planes = []
        self.engines = []
        self.calls = []
        self.stack = ExitStack()
        self.stack.enter_context(
            patch.object(
                rt, "_send_tensor", side_effect=AssertionError("TCP tensor payload")
            )
        )
        self.stack.enter_context(
            patch.object(rt, "_recv_q", side_effect=AssertionError("TCP Q payload"))
        )

    def attach(self, controller, role):
        raw = self.Native()
        assert raw.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        engine = self.Engine()
        engine.engine, engine.session_id = raw, f"127.0.0.1:{raw.get_rpc_port()}"
        native = engine.batch_transfer_sync

        def transfer(session, sources, targets, lengths):
            self.calls.append((role, list(lengths)))
            return native(session, sources, targets, lengths)

        engine.batch_transfer_sync = transfer
        manager = SimpleNamespace(
            engine=engine, _remote_attn_send_lock=threading.Lock(), local_ip="127.0.0.1"
        )
        scheduler = getattr(controller, "scheduler", None)
        if scheduler is None:
            scheduler = controller.scheduler = SimpleNamespace()
        scheduler.disaggregation_mode = SimpleNamespace(value=role)
        queue = (
            "disagg_prefill_bootstrap_queue"
            if role == "prefill"
            else "disagg_decode_prealloc_queue"
        )
        setattr(scheduler, queue, SimpleNamespace(kv_manager=manager))
        scheduler._pre_layerwise = controller
        controller.gpu_data_plane = dp.Plane(manager, "cuda:0")
        canary = torch.full((256,), len(self.engines) + 1.0, device="cuda")
        torch.cuda.synchronize()
        assert (
            engine.batch_register(
                [canary.data_ptr()], [canary.numel() * canary.element_size()]
            )
            == 0
        )
        self.engines.append((engine, canary))
        self.planes.append((role, controller.gpu_data_plane))
        return engine

    def close(self):
        for role in ("decode", "prefill"):
            for owner, plane in self.planes:
                if owner == role:
                    plane.close()
        assert not any(plane.quarantine for _, plane in self.planes)
        source, x = self.engines[0]
        target, y = self.engines[-1]
        assert (
            source.batch_transfer_sync(
                target.get_session_id(),
                [x.data_ptr()],
                [y.data_ptr()],
                [x.numel() * x.element_size()],
            )
            == 0
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(x, y)
        for engine, tensor in self.engines:
            assert engine.batch_deregister([tensor.data_ptr()]) == 0
        self.stack.close()
        return dict(
            native_mooncake_gpu=True,
            physical_transport="mooncake-tcp",
            tcp_tensor_payloads=0,
            q_transfers=sum(role == "decode" for role, _ in self.calls),
            result_transfers=sum(role == "prefill" for role, _ in self.calls) - 1,
            registered_gpu_bytes=True,
            no_quarantined_slots=True,
            native_registration_survives_rpc_teardown=True,
        )
