"""Group-owned NCCL resources used only by conditional graph capture/replay."""

from contextlib import contextmanager

import torch


class ConditionalGraphCommunicator:
    def __init__(self, group):
        self.group = group
        self.communicator = None
        self.capturing = False
        self.replay_stream = None

    def get(self):
        if self.communicator is not None:
            return self.communicator
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        group = self.group
        error = None
        try:
            if group.device.type != "cuda" or group.pynccl_comm is None:
                raise ValueError("conditional graphs require a CUDA PyNccl group")
            if not group.pynccl_comm.available:
                raise ValueError("conditional graphs require an available PyNccl group")
            PyNcclCommunicator.validate_graph_usage_mode(
                1, group.pynccl_comm.nccl_version
            )
        except ValueError as exc:
            error = str(exc)
        # A local capability error must not leave peers inside NCCL init.
        # This control-plane collective runs before stream capture begins.
        errors = [None] * group.world_size
        torch.distributed.all_gather_object(errors, error, group=group.cpu_group)
        if any(errors):
            raise ValueError(f"conditional graph communicator unavailable: {errors}")
        comm = PyNcclCommunicator(
            group.cpu_group,
            group.device,
            library_path=group.pynccl_comm.library_path,
            graph_usage_mode=1,
        )
        if not comm.available:
            raise RuntimeError("conditional graph communicator initialization failed")
        self.communicator = comm
        return comm

    @contextmanager
    def capture_scope(self):
        if self.capturing:
            raise RuntimeError("conditional graph capture scopes cannot be nested")
        comm = self.get()
        # Model collectives and plugin consensus must use the same dedicated
        # communicator. Restore accelerated/fused ordinary paths on exit.
        names = (
            "pynccl_comm",
            "ca_comm",
            "qr_comm",
            "pymscclpp_comm",
            "torch_symm_mem_comm",
            "pcie_ipc_comm",
            "_fi_workspace_hint",
        )
        group = self.group
        previous = {name: getattr(group, name) for name in names}
        self.capturing = True
        try:
            for name in names:
                setattr(group, name, comm if name == "pynccl_comm" else None)
            with comm.change_state(enable=True):
                yield
        finally:
            for name, value in previous.items():
                setattr(group, name, value)
            self.capturing = False

    def check_replay_stream(self, stream):
        # No eager calls use this communicator. Replays must be serialized on
        # one stream; this check is on host metadata, never a device readback.
        if self.replay_stream is not None and self.replay_stream != stream:
            raise RuntimeError("conditional graphs must replay on one stream per group")
        self.replay_stream = stream

    def destroy(self):
        if self.capturing:
            raise RuntimeError(
                "cannot destroy a group during conditional graph capture"
            )
        if self.communicator is not None:
            # Graph owners must retire their graphs before parallel groups.
            self.group.device_module.synchronize(self.group.device)
            self.communicator.destroy()
            self.communicator = None
            self.replay_stream = None
