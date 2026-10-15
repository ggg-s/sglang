"""Real process-group tests; CPU Gloo does not substitute for CUDA/NCCL."""

import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hybrid_spec.backends.reference import ReferenceBackend
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import DraftOverrideView
from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers
from hybrid_spec.integrations.sglang.tp_sync import TPSynchronizer


class Request:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = [1, 2, 3, 4, 1]
        self.output_ids = [2]
        self.multimodal_inputs = None
        self.sampling_params = SimpleNamespace(max_new_tokens=32)

    def finished(self):
        return False


def _check_consensus(group, ranks, device):
    rank = dist.get_rank()
    sync = TPSynchronizer(group, ranks[0])
    for k in (1, 3):
        buffers = GraphBuffers(8, k, device)
        captured = buffers.bind_capture_view(8)
        neural = torch.full((8, 1), 99, dtype=torch.int64, device=device)

        def select():
            return torch.cat(
                [captured.select_step(neural, step) for step in range(k)], dim=1
            )

        graph = None
        if device.type == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = select()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = select()

        # Same call order even for a rank with all retrieval rows disabled.
        for mode in ("mixed", "all_neural", "all_retrieval", "one_rank_neural"):
            tokens = torch.arange(5 * k, device=device).view(5, k)
            mask = torch.ones(5, dtype=torch.bool, device=device)
            if mode == "mixed":
                if rank == ranks[-1]:
                    tokens[1, -1] += 1
                    mask[2] = False
                if rank == ranks[0]:
                    mask[3] = False
            elif mode == "all_neural" or (
                mode == "one_rank_neural" and rank == ranks[-1]
            ):
                mask.zero_()
            view = sync.synchronize(DraftOverrideView(tokens, mask))
            expected_mask = {
                "mixed": [True, len(ranks) == 1, False, False, True],
                "all_neural": [False] * 5,
                "all_retrieval": [True] * 5,
                "one_rank_neural": [False] * 5,
            }[mode]
            assert view.use_retrieval.tolist() == expected_mask
            buffers.stage(view, 5, 8)
            if graph is None:
                output = select()
            else:
                graph.replay()
            expected = torch.full((8, k), 99, device=device, dtype=torch.int64)
            for row, enabled in enumerate(expected_mask):
                if enabled:
                    expected[row] = tokens[row]
            assert torch.equal(output, expected)
            # A smaller batch must erase previous retrieval rows in padding.
            small = sync.synchronize(
                DraftOverrideView(
                    tokens[:1], torch.ones(1, dtype=torch.bool, device=device)
                )
            )
            buffers.stage(small, 1, 8)
            if graph is None:
                output = select()
            else:
                graph.replay()
            assert torch.equal(output[1:], neural[1:].expand(-1, k))

    # Exercise the actual bind path, including seq_lens validation BEFORE
    # consensus, request reordering, lifecycle and metrics AFTER consensus.
    config = HybridConfig(
        2, 2, 2, 32, "reference" if device.type == "cpu" else "gpu_sa"
    )
    if device.type == "cpu":
        backend = ReferenceBackend(config)
    else:
        from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton

        backend = GPUSuffixAutomaton(config, device)
    extension = DraftExtension(config, backend, 100, tp_sync=sync)
    requests = [Request("a"), Request("b")]
    draft_input = SimpleNamespace(draft_override=None)
    for order in (requests, requests[::-1]):
        batch = SimpleNamespace(
            reqs=order,
            is_extend_in_batch=False,
            has_grammar=False,
            sampling_info=SimpleNamespace(is_all_greedy=True),
            seq_lens=torch.tensor([6 if rank == ranks[-1] else 5, 5], device=device),
        )
        with extension.bind(batch, draft_input) as view:
            assert view.use_retrieval.tolist() == [False, True]
            assert view.select_step(
                torch.tensor([90, 91], device=device), 0
            ).tolist() == [90, 3]
        assert draft_input.draft_override is None
        extension.on_verify_result(
            SimpleNamespace(accept_lens=torch.tensor([1, 3], device=device))
        )
    assert extension.metrics.snapshot()["retrieval_requests"] == 2
    extension.close_request("a")
    extension.reset()
    assert not extension.runtime.states


def _worker(rank, world_size, init_method, backend):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        _check_consensus(dist.group.WORLD, list(range(world_size)), device)
        if world_size == 4:
            # Source is global rank 1, not global rank 0 or subgroup rank 1.
            ranks = [1, 3]
            group = dist.new_group(ranks=ranks, timeout=timedelta(seconds=60))
            if rank in ranks:
                _check_consensus(group, ranks, device)
                dist.destroy_process_group(group)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend,world_size", [("gloo", 1), ("gloo", 2), ("gloo", 4), ("nccl", 2)]
)
def test_tp_consensus_and_graph_staging(tmp_path, backend, world_size):
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("requires Gloo")
    if backend == "nccl" and (
        not dist.is_nccl_available() or torch.cuda.device_count() < world_size
    ):
        pytest.skip("requires NCCL and two NVIDIA GPUs, plus nvcc for SA")
    context = mp.spawn(
        _worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), backend),
        nprocs=world_size,
        join=False,
    )
    # Includes time for first-use CUDA extension compilation on GPU runners.
    deadline = time.monotonic() + (300 if backend == "nccl" else 120)
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError("TP workers did not finish")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
