"""NCCL/graph/stream checks for PDMux TP lane routing, without model weights.

Run with pytest on a CUDA host. The singleton test also runs on a 4060;
the two-rank test needs two distinct GPUs and must not be emulated by sharing
one GPU between NCCL processes. Neither test measures model throughput.
"""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/registered/unit/multiplex"))
from test_pdmux_sxf_port import load_methods
from test_pdmux_tp_overlap import lane_namespace

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL required",
)


def run_rank(rank, world_size, store_path):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=45),
        device_id=device,
    )
    try:
        broadcast = load_methods(
            "distributed/parallel_state.py",
            "GroupCoordinator",
            ["broadcast"],
            torch=torch,
            is_hip=lambda: False,
        )["broadcast"]

        class Group:
            def __init__(self, group):
                self.device_group = group
                self.world_size = world_size
                self.ranks = list(range(world_size))
                self.reductions = 0

            def all_reduce(self, tensor):
                dist.all_reduce(tensor, group=self.device_group)
                self.reductions += 1
                return tensor

        Group.broadcast = broadcast
        decode = Group(dist.group.WORLD)
        prefill = Group(dist.new_group(list(range(world_size)), backend="nccl"))
        lanes = lane_namespace(decode, prefill)
        p_stream, d_stream = torch.cuda.Stream(), torch.cuda.Stream()
        p_stream.wait_stream(torch.cuda.current_stream())
        d_stream.wait_stream(torch.cuda.current_stream())
        decode_input = torch.zeros(32, device=device)
        decode_output = torch.empty_like(decode_input)
        lanes.set_pdmux_status(False)
        with torch.cuda.stream(d_stream):
            # Initialize the communicator before capture.
            lanes.get_attn_tp_group().all_reduce(decode_input)
        d_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=d_stream):
            decode_output.copy_(decode_input)
            lanes.get_attn_tp_group().all_reduce(decode_output)
        d_stream.synchronize()

        p_results, d_results, tokens = [], [], []
        for chunk in range(48):
            # Several segments per chunk, with decode graph replays between
            # them. Prefill's final sample/broadcast also runs on its own lane.
            for segment in range(3):
                lanes.set_pdmux_status(False)
                with torch.cuda.stream(d_stream):
                    decode_input.fill_(rank + chunk)
                    graph.replay()
                    d_results.append((chunk, decode_output.clone()))
                    acceptance = torch.tensor([chunk + rank], device=device)
                    lanes.get_tp_group().broadcast(acceptance)
                    tokens.append((chunk, acceptance))
                lanes.set_pdmux_status(True)
                with torch.cuda.stream(p_stream):
                    p = torch.full(
                        (257,),
                        rank + chunk + segment,
                        device=device,
                        dtype=torch.float32,
                    )
                    lanes.get_attn_tp_group().all_reduce(p)
                    lanes.get_moe_tp_group().all_reduce(p)
                    p_results.append((chunk + segment, p))
                    if segment == 2:
                        sampled = torch.tensor([chunk + rank], device=device)
                        lanes.get_tp_group().broadcast(sampled)
                        tokens.append((chunk, sampled))
        torch.cuda.synchronize()
        rank_sum = world_size * (world_size - 1) // 2
        for chunk, output in d_results:
            torch.testing.assert_close(
                output,
                torch.full_like(output, rank_sum + world_size * chunk),
                rtol=0,
                atol=0,
            )
        for value, output in p_results:
            expected = world_size * (rank_sum + world_size * value)
            torch.testing.assert_close(
                output, torch.full_like(output, expected), rtol=0, atol=0
            )
        for chunk, token in tokens:
            torch.testing.assert_close(
                token, torch.full_like(token, chunk), rtol=0, atol=0
            )
        assert prefill.reductions == 48 * 3 * 2
        assert decode.reductions < prefill.reductions
    finally:
        dist.destroy_process_group()


def test_single_gpu_nccl_lanes_with_decode_graph():
    with tempfile.TemporaryDirectory(prefix="pdmux-tp-cuda-") as tmp:
        run_rank(0, 1, str(Path(tmp) / "store"))


def test_same_group_selection_keeps_cuda_graph_work_queued_and_transition_drains():
    """Real CUDA work and graph replay across repeated chunk selections."""
    state, drains, rebound = [0], [], []
    raw_streams = [torch.cuda.Stream() for _ in range(4)]
    values = [torch.zeros(32, device="cuda") for _ in range(2)]
    graph = torch.cuda.CUDAGraph()
    for stream in raw_streams:
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=raw_streams[1]):
        values[1].add_(1)
    torch.cuda.synchronize()
    values[1].zero_()
    raw_streams[1].wait_stream(torch.cuda.current_stream())

    class Stream:
        def __init__(self, idx):
            self.idx = idx

        def synchronize(self):
            raw_streams[self.idx].synchronize()
            drains.append(self.idx)

    activate = load_methods(
        "multiplex/multiplexing_mixin.py",
        "SchedulerMultiplexMixin",
        ["_activate_pdmux_stream_group"],
        get_current_stream_idx=lambda: state[0],
        set_current_stream_idx=lambda idx: state.__setitem__(0, idx),
    )["_activate_pdmux_stream_group"]
    from types import SimpleNamespace

    scheduler = SimpleNamespace(
        stream_groups=[(Stream(0), Stream(1)), (Stream(2), Stream(3))],
        _update_decode_attn_backends=lambda idx: rebound.append(idx),
    )
    for _ in range(32):
        with torch.cuda.stream(raw_streams[0]):
            values[0].add_(1)
        with torch.cuda.stream(raw_streams[1]):
            graph.replay()
        activate(scheduler, 0)
    assert drains == []
    assert rebound == []
    # A real transition must complete both old lanes before rebinding metadata.
    activate(scheduler, 1)
    assert drains == [0, 1]
    assert rebound == [1]
    assert state == [1]
    for value in values:
        torch.testing.assert_close(value, torch.full_like(value, 32))


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Two distinct CUDA GPUs required"
)
def test_two_gpu_nccl_lanes_with_decode_graph():
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="pdmux-tp2-cuda-") as tmp:
        processes = [
            ctx.Process(target=run_rank, args=(rank, 2, str(Path(tmp) / "store")))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        try:
            for process in processes:
                process.join(timeout=120)
                assert not process.is_alive(), "TP2 NCCL lane test hung"
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
