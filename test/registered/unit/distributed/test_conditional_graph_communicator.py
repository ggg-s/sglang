"""CPU contract checks for conditional graph communicator isolation."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.distributed.conditional_graph import ConditionalGraphCommunicator
from sglang.srt.distributed.device_communicators import pynccl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "symmetric,mode,expected", [(False, None, None), (True, None, 1), (False, 1, 1)]
)
def test_initialization_preserves_default_and_symmetric_paths(
    monkeypatch, symmetric, mode, expected
):
    monkeypatch.delenv("NCCL_GRAPH_MIXING_SUPPORT", raising=False)

    class Library:
        lib = SimpleNamespace(_name="libnccl-test.so")

        def ncclGetRawVersion(self):
            return 23000

        def ncclGetVersion(self):
            return "2.30.0"

        def ncclCommInitRank(self, *args):
            return SimpleNamespace(mode=None)

        def ncclCommInitRankConfig(self, size, uid, rank, config):
            return SimpleNamespace(mode=config.graphUsageMode)

        def ncclAllReduce(self, *args):
            pass

    class HostGroup:
        rank = 1
        world_size = 2

        def broadcast_obj(self, value, src):
            return value

    stream = SimpleNamespace(cuda_stream=1, synchronize=lambda: None)
    cuda = SimpleNamespace(
        device=lambda device: nullcontext(),
        Stream=lambda: stream,
        stream=lambda s: nullcontext(),
    )
    torch_boundary = SimpleNamespace(
        device=torch.device, cuda=cuda, zeros=lambda *a, **k: torch.zeros(1)
    )
    monkeypatch.setattr(pynccl, "torch", torch_boundary)
    monkeypatch.setattr(pynccl, "StatelessProcessGroup", HostGroup)
    monkeypatch.setattr(pynccl, "NCCLLibrary", lambda path: Library())
    monkeypatch.setattr(pynccl, "get_current_device_stream_fast", lambda: stream)
    comm = pynccl.PyNcclCommunicator(
        HostGroup(),
        "cpu",
        is_symmetric_memory_enabled=symmetric,
        graph_usage_mode=mode,
    )
    assert comm.comm.mode == expected
    assert comm.available and comm.disabled
    assert comm.library_path == "libnccl-test.so"


@pytest.mark.parametrize(
    "mode,version,override",
    [
        (1, 22809, None),
        (1, 23000, "1"),
        (1, 23000, "0"),
        (True, 23000, None),
        (3, 23000, None),
    ],
)
def test_reject_ignored_or_overridden_explicit_mode(
    monkeypatch, mode, version, override
):
    monkeypatch.delenv("NCCL_GRAPH_MIXING_SUPPORT", raising=False)
    if override is not None:
        monkeypatch.setenv("NCCL_GRAPH_MIXING_SUPPORT", override)
    with pytest.raises(ValueError):
        pynccl.PyNcclCommunicator.validate_graph_usage_mode(mode, version)
    # The legacy path does not acquire these new restrictions.
    pynccl.PyNcclCommunicator.validate_graph_usage_mode(None, version)


class CpuTransport:
    validate_graph_usage_mode = staticmethod(
        pynccl.PyNcclCommunicator.validate_graph_usage_mode
    )
    available = True
    nccl_version = 23000
    library_path = None

    def __init__(self, *args, graph_usage_mode=None, **kwargs):
        self.mode = graph_usage_mode
        self.disabled = graph_usage_mode is not None
        self.destroyed = False

    @contextmanager
    def change_state(self, enable):
        previous = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = previous

    def all_reduce(self, tensor):
        assert not self.disabled and not self.destroyed
        tensor.add_(10 if self.mode == 1 else 100)

    def destroy(self):
        self.destroyed = True


def make_resources(monkeypatch, peer_error=None):
    monkeypatch.delenv("NCCL_GRAPH_MIXING_SUPPORT", raising=False)

    def gather(output, local, group):
        output[:] = [local, peer_error]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    monkeypatch.setattr(pynccl, "PyNcclCommunicator", CpuTransport)
    group = SimpleNamespace(
        world_size=2,
        device=torch.device("cuda:0"),
        cpu_group=object(),
        pynccl_comm=CpuTransport(),
        ca_comm=object(),
        qr_comm=object(),
        pymscclpp_comm=object(),
        torch_symm_mem_comm=object(),
        pcie_ipc_comm=object(),
        _fi_workspace_hint="tp",
        device_module=SimpleNamespace(synchronize=lambda device: None),
    )
    return group, ConditionalGraphCommunicator(group)


@pytest.mark.parametrize("abort", [False, True])
def test_route_capture_then_restore_all_ordinary_backends(monkeypatch, abort):
    group, resources = make_resources(monkeypatch)
    before = vars(group).copy()
    value = torch.zeros(1)
    group.pynccl_comm.all_reduce(value)
    try:
        with resources.capture_scope():
            dedicated = group.pynccl_comm
            assert dedicated is resources.get() and dedicated.mode == 1
            assert all(
                getattr(group, name) is None
                for name in (
                    "ca_comm",
                    "qr_comm",
                    "pymscclpp_comm",
                    "torch_symm_mem_comm",
                    "pcie_ipc_comm",
                    "_fi_workspace_hint",
                )
            )
            group.pynccl_comm.all_reduce(value)
            with pytest.raises(RuntimeError, match="during conditional graph capture"):
                resources.destroy()
            with pytest.raises(RuntimeError, match="nested"):
                with resources.capture_scope():
                    pass
            if abort:
                raise RuntimeError("capture failure")
    except RuntimeError as exc:
        assert abort and str(exc) == "capture failure"
    assert vars(group) == before
    group.pynccl_comm.all_reduce(value)
    assert value.tolist() == [210]
    assert dedicated.disabled and resources.get() is dedicated
    assert not resources.capturing
    resources.destroy()
    resources.destroy()
    assert dedicated.destroyed and resources.communicator is None
    assert group.pynccl_comm is before["pynccl_comm"]


def test_peer_error_prevents_any_communicator_creation(monkeypatch):
    group, resources = make_resources(monkeypatch, peer_error="NCCL too old")
    ordinary = group.pynccl_comm
    with pytest.raises(ValueError, match="NCCL too old"):
        with resources.capture_scope():
            pass
    assert resources.communicator is None and group.pynccl_comm is ordinary
    assert not resources.capturing


def test_replay_rejects_cross_stream_use(monkeypatch):
    _, resources = make_resources(monkeypatch)
    stream = object()
    resources.check_replay_stream(stream)
    resources.check_replay_stream(stream)
    with pytest.raises(RuntimeError, match="one stream"):
        resources.check_replay_stream(object())
