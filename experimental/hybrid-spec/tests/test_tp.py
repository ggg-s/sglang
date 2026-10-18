"""Real process-group tests; CPU Gloo does not substitute for CUDA/NCCL."""

import time
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hybrid_spec.backends.reference import ReferenceBackend
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import DraftOverrideView
from hybrid_spec.integrations.sglang.device_extension import (
    DeviceDraftExtension,
    DeviceGraphBuffers,
    DeviceOverride,
)
from hybrid_spec.integrations.sglang.draft_extension import DraftExtension
from hybrid_spec.integrations.sglang.graph_buffers import GraphBuffers
from hybrid_spec.integrations.sglang.tp_sync import TPSynchronizer
from hybrid_spec.observability.metrics import Metrics


class Request:
    def __init__(self, rid):
        self.rid = rid
        self.origin_input_ids = [1, 2, 3, 4, 1]
        self.output_ids = [2]
        self.multimodal_inputs = None
        self.sampling_params = SimpleNamespace(max_new_tokens=32)

    def finished(self):
        return False

    def get_fill_ids(self):
        return self.origin_input_ids


class GlooCommunicator:
    """Exercise PyNccl's enable/restore/in-place contract with CPU collectives.

    This boundary replacement tests adapter semantics, not NCCL or capture.
    """

    available = True

    def __init__(self, group):
        self.group = group
        self.disabled = True

    @contextmanager
    def change_state(self, enable):
        previous = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = previous

    def all_reduce(self, tensor, op):
        assert not self.disabled
        dist.all_reduce(tensor, op=op, group=self.group)


def _check_device_bridge(sync, ranks, device):
    """Lookup precedes consensus; graph replay and metrics see unanimous masks.

    Only the native lookup/build boundary is replaced. Process groups, request
    leases, extension binding, graph staging, consensus and metrics are real.
    """
    rank = dist.get_rank()
    for k in (1, 3):
        config = HybridConfig(k, 2, 4, 32, history_mode="device")

        class LookupBackend:
            def __init__(self):
                self.config, self.device = config, device
                self.tokens = torch.arange(4 * k, device=device).reshape(4, k) + 10
                self.mask = torch.ones(4, dtype=torch.bool, device=device)
                self.feedback = None

            def prepare_host(self, history):
                return history

            def build_host(self, host, vocab):
                return host

            def import_host(self, slot, host):
                pass

            def initialize_device_slot(self, *args):
                pass

            def stage_feedback(self, *args, use_retrieval=None, counts=None):
                self.feedback = args
                if counts is not None:
                    reference = Metrics()
                    reference.record(use_retrieval, args[4])
                    counts.add_(reference.counts)

            def consume_and_query(self, slots, generations, seq_lens, tokens, mask):
                # GPU-safe test boundary so the NCCL case captures this too.
                tokens.copy_(self.tokens[slots.clamp_min(0)])
                mask.copy_((slots >= 0) & self.mask[slots.clamp_min(0)])

        backend = LookupBackend()
        extension = DeviceDraftExtension(config, backend, 100, tp_sync=sync)
        requests = [Request(str(i)) for i in range(4)]
        batch = SimpleNamespace(
            reqs=requests,
            seq_lens_cpu=torch.full((4,), 5),
            seq_lens=torch.full((4,), 5, device=device),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            is_extend_in_batch=False,
            has_grammar=False,
            sampling_info=SimpleNamespace(is_all_greedy=True),
        )
        draft = SimpleNamespace(draft_override=None)
        try:
            ticket = extension.begin_prefill(batch)
            extension.on_prefill(
                batch,
                SimpleNamespace(next_token_ids=torch.full((4,), 2, device=device)),
                ticket,
            )
            buffers = extension.create_graph_buffers(8)
            captured = buffers.bind_capture_view(8)
            seq_lens = torch.ones(8, dtype=torch.int64, device=device)
            neural = torch.full((8, k), 99, dtype=torch.int64, device=device)

            def run():
                captured.prepare(seq_lens)
                return torch.cat(
                    [captured.select_step(neural[:, :1], step) for step in range(k)],
                    dim=1,
                )

            graph = None
            if device.type == "cuda":
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        output = run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run()

            # State changes happen AFTER capture, including all-false local mask.
            for mode in ("agree", "diverge", "one_rank_neural", "agree"):
                backend.tokens.copy_(
                    torch.arange(4 * k, device=device).reshape(4, k) + 10
                )
                backend.mask.fill_(True)
                if mode == "diverge" and rank == ranks[-1]:
                    backend.tokens[1, -1].add_(1)
                    backend.mask[2] = False
                if mode == "one_rank_neural" and rank == ranks[-1]:
                    backend.mask.zero_()
                # Reorder requests; slots and graph padding must stay row aligned.
                batch.reqs = requests[::-1]
                with extension.bind(batch, draft) as view:
                    # The eager view must carry the same synchronizer as the
                    # graph view. Query again below to test live graph inputs.
                    view.prepare(batch.seq_lens)
                    eager_mask = view.use_retrieval.clone()
                    buffers.stage(view, 4, 8)
                    seq_lens[:4].fill_(5)
                    if graph is None:
                        output = run()
                    else:
                        graph.replay()
                    expected_mask = [True] * 4
                    if mode == "diverge":
                        expected_mask = [True, False, len(ranks) == 1, True]
                    elif mode == "one_rank_neural":
                        expected_mask = [False] * 4
                    assert view.use_retrieval.tolist() == expected_mask
                    assert eager_mask.tolist() == expected_mask
                    root_tokens = torch.arange(4 * k, device=device).reshape(4, k) + 10
                    if mode == "diverge" and len(ranks) == 1:
                        root_tokens[1, -1].add_(1)
                    assert torch.equal(captured.tokens[:4], root_tokens.flip(0))
                    assert not captured.use_retrieval[4:].any()
                    expected = neural.clone()
                    for row, enabled in enumerate(expected_mask):
                        if enabled:
                            expected[row] = backend.tokens[3 - row]
                    assert torch.equal(output, expected)
                assert draft.draft_override is None
                result = SimpleNamespace(
                    next_token_ids=torch.tensor(
                        [20, 21, 22, 23], dtype=torch.int32, device=device
                    ),
                    accept_index=torch.arange(4, dtype=torch.int32, device=device)[
                        :, None
                    ],
                    accept_lens=torch.ones(4, dtype=torch.int32, device=device),
                )
                before = extension.metrics.snapshot()["retrieval_requests"]
                extension.on_device_verify(result, batch)
                assert extension.metrics.snapshot()[
                    "retrieval_requests"
                ] - before == sum(expected_mask)
                assert backend.feedback[2] is result.next_token_ids

            # Free a slot on only one rank; all ranks must still communicate and
            # unanimously fall back for that row, even if lease maps differ.
            if rank == ranks[-1]:
                extension.close_request(requests[3].rid)
            batch.reqs, batch.seq_lens = [requests[3]], batch.seq_lens[:1]
            with extension.bind(batch, draft) as view:
                buffers.stage(view, 1, 8)
                if graph is None:
                    output = run()
                else:
                    graph.replay()
                assert not view.use_retrieval.any()
                assert torch.equal(output, neural)
        finally:
            extension.shutdown()


def _check_device_sync(group, ranks, device):
    if device.type == "cpu":
        for communicator in (None, GlooCommunicator(group)):
            sync = TPSynchronizer(group, communicator)
            _check_device_bridge(sync, ranks, device)
            _check_cpu_history_bridge(sync, ranks, device)
            _check_split_gate(sync, ranks, device)
            if communicator is not None:
                assert communicator.disabled  # eager must restore caller state
    else:
        # Real SGLang PyNccl; no ProcessGroupNCCL collective is captured.
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        cpu_group = dist.new_group(ranks=ranks, backend="gloo")
        communicator = PyNcclCommunicator(cpu_group, device)
        sync = TPSynchronizer(group, communicator)
        _check_device_bridge(sync, ranks, device)
        _check_cpu_history_bridge(sync, ranks, device)
        _check_device_kernel_consensus(sync, ranks, device)
        _check_device_kernel_consensus(sync, ranks, device, mode="split")
        _check_split_gate(sync, ranks, device)
        assert communicator.disabled
        dist.destroy_process_group(cpu_group)


def _check_split_gate(sync, ranks, device):
    for gate in (0, 1, 2, 3, 8):
        for staged in (False, True):
            _check_split_gate_width(sync, ranks, device, gate, staged)


def _check_split_gate_width(sync, ranks, device, gate, staged):
    """Local gate/candidate divergence must never split ranks into different tails.

    CPU exercises real Gloo groups; CUDA captures the query fork/join and real
    PyNccl consensus. Only the SA lookup and neural prediction are boundaries.
    """
    config = HybridConfig(
        8,
        2,
        8,
        128,
        history_mode="device",
        num_neural_gate_tokens=gate,
        draft_graph_mode="split",
    )
    base = torch.arange(64, device=device).reshape(8, 8) + 10
    candidates = base.clone()
    eligible = torch.ones(8, dtype=torch.bool, device=device)
    neural = torch.full_like(base, 99)

    def query(slots, generations, seq_lens, tokens, mask):
        tokens.copy_(candidates)
        mask.copy_(eligible)

    backend = SimpleNamespace(config=config, device=device, consume_and_query=query)
    query_stream = torch.cuda.Stream() if device.type == "cuda" else None
    buffers = (
        GraphBuffers(8, 8, device, sync, gate, defer_prefix=True)
        if staged
        else DeviceGraphBuffers(backend, 8, sync, query_stream)
    )
    view = buffers.bind_capture_view(8)
    seq_lens = torch.ones(8, dtype=torch.int64, device=device)

    def run():
        view.prepare(seq_lens)
        return torch.cat(
            [view.select_step(neural[:, i : i + 1] + 0, i) for i in range(8)], dim=1
        )

    graph = None
    if device.type == "cuda":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = run()
    mask_address = view.use_retrieval.data_ptr()
    for mode in (
        "agree",
        "candidate_disagree",
        "gate_disagree",
        "rank_fallback",
        "agree",
    ):
        candidates.copy_(base)
        neural[:, :gate].copy_(base[:, :gate])
        eligible.fill_(True)
        if dist.get_rank() == ranks[-1]:
            if mode == "candidate_disagree":
                candidates[1, -1].add_(1)
            elif mode == "gate_disagree":
                neural[2, max(0, gate - 1)].add_(1)
            elif mode == "rank_fallback":
                eligible[3] = False
        if staged:
            buffers.stage(DraftOverrideView(candidates, eligible), 8, 8)
        if graph is None:
            output = run()
        else:
            graph.replay()
        expected_mask = [
            True,
            not (mode == "candidate_disagree" and (len(ranks) > 1 or gate == 8)),
            mode != "gate_disagree" or gate == 0,
            mode != "rank_fallback",
        ] + [True] * 4
        assert view.use_retrieval.tolist() == expected_mask
        assert view.use_retrieval.data_ptr() == mask_address
        expected = neural.clone()
        for row, enabled in enumerate(expected_mask):
            if enabled:
                expected[row].copy_(candidates[row])
        assert torch.equal(output, expected)


def _check_device_kernel_consensus(sync, ranks, device, mode="full"):
    """Actual host import, feedback, SA lookup and NCCL captured together."""
    from hybrid_spec.backends.gpu_sa import GPUSuffixAutomaton

    config = HybridConfig(
        3,
        2,
        3,
        64,
        history_mode="device",
        draft_graph_mode=mode,
        num_neural_gate_tokens=2 if mode == "split" else 0,
    )
    backend = GPUSuffixAutomaton(config, device)
    prompts = [[1, 2, 3, 4, 1], [4, 5, 6, 7, 4], [1, 2, 3, 4, 1]]
    if dist.get_rank() == ranks[-1]:
        prompts[1] = [4, 5, 8, 9, 4]  # same version, different continuation
    for slot, prompt in enumerate(prompts):
        backend.initialize_device_slot(slot, 10 + slot, 64)
        backend.import_host(slot, backend.build_host(backend.prepare_host(prompt), 100))
    query_stream = torch.cuda.Stream() if mode == "split" else None
    buffers = DeviceGraphBuffers(backend, 4, sync, query_stream)
    captured = buffers.bind_capture_view(4)
    seq_lens = torch.full((4,), 5, device=device, dtype=torch.int64)
    stream = torch.cuda.Stream()

    def run():
        captured.prepare(seq_lens)
        if mode == "split":
            captured.select_step(torch.full((4, 1), 3, device=device), 0)
            captured.select_step(torch.full((4, 1), 4, device=device), 1)

    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    slots = torch.arange(3, device=device, dtype=torch.int64)
    generations = slots + 10
    backend.stage_feedback(
        slots,
        generations,
        torch.tensor([2, 5, 2], dtype=torch.int32, device=device),
        torch.arange(3, dtype=torch.int32, device=device)[:, None],
        torch.ones(3, dtype=torch.int32, device=device),
        seq_lens[:3],
        False,
        100,
    )
    if dist.get_rank() == ranks[-1]:
        backend.owners[2].add_(1)  # only this rank has a stale generation
    view = DeviceOverride(backend, slots, generations, sync)
    buffers.stage(view, 3, 4)
    graph.replay()
    assert view.use_retrieval.tolist() == [True, False, False]
    assert captured.tokens[0].tolist() == [3, 4, 1]
    assert not captured.use_retrieval[3:].any()
    versions = backend.meta[:, 0].clone()
    graph.replay()
    assert torch.equal(backend.meta[:, 0], versions)  # feedback consumed once

    # Accepted prefix remains on device and drives the next captured query.
    backend.stage_feedback(
        slots[:1],
        generations[:1],
        torch.tensor([3, 4, 1, 2], dtype=torch.int32, device=device),
        torch.arange(4, dtype=torch.int32, device=device)[None, :],
        torch.tensor([4], dtype=torch.int32, device=device),
        seq_lens[:1],
        True,
        100,
    )
    reordered = DeviceOverride(
        backend, slots[:2].flip(0), generations[:2].flip(0), sync
    )
    buffers.stage(reordered, 2, 4)
    seq_lens[:2].copy_(torch.tensor([5, 9], device=device))
    graph.replay()
    assert reordered.use_retrieval.tolist() == [False, True]
    assert backend.history[0, :10].tolist() == [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
    assert not captured.use_retrieval[2:].any()


def _check_consensus(group, ranks, device):
    rank = dist.get_rank()
    sync = TPSynchronizer(group)
    from hybrid_spec.selection.online_verify import DynamicVerify

    dynamic = DynamicVerify(
        HybridConfig(
            3,
            2,
            4,
            32,
            history_mode="device",
            verify_mode="compact",
            dynamic_verify=True,
        ),
        device,
        sync,
    )
    dynamic.model.check_ct.fill_(100)
    dynamic.model.correct_ct.fill_(80)
    view = DraftOverrideView(
        torch.zeros((2, 3), device=device, dtype=torch.long),
        torch.tensor([True, False], device=device),
        query_metadata=torch.tensor(
            [[4 + (rank - ranks[0]) * 16, 0], [0, 0]], device=device
        ),
    )
    for size in (4, 6, 8):
        dynamic.observe_cost(
            (2, 512), size, 0.2, (size if rank == ranks[0] else 8 / size)
        )
    lens = dynamic.select(
        view, torch.tensor([3, 3], device=device), [4, 6, 8], (2, 512)
    )
    copies = [torch.empty_like(lens) for _ in ranks]
    dist.all_gather(copies, lens, group=group)
    assert all(torch.equal(lens, copy) for copy in copies)
    previous_correct = dynamic.model.correct_ct.clone()
    dynamic.feedback(
        torch.ones(2, device=device, dtype=torch.int32) + int(rank != ranks[0])
    )
    assert dynamic.model.check_ct[1, 0].item() == 101
    assert dynamic.model.check_ct[0, 0].item() == 101
    assert torch.equal(dynamic.model.correct_ct, previous_correct)
    # Exact agreement must distinguish extrema and permutations with identical
    # sums; a sum/hash or negation with signed overflow is not sufficient.
    extrema = torch.tensor([[-(2**63), 2**63 - 1], [3, 5], [7, 9]], device=device)
    if len(ranks) > 1 and rank == ranks[-1]:
        extrema[1] = extrema[1].flip(0)
        extrema[2, 1] -= 1
    exact = sync.synchronize(
        DraftOverrideView(extrema, torch.ones(3, dtype=torch.bool, device=device))
    )
    assert exact.use_retrieval.tolist() == [True, len(ranks) == 1, len(ranks) == 1]
    assert exact.tokens[0].tolist() == [-(2**63), 2**63 - 1]
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


def _check_cpu_history_bridge(sync, ranks, device):
    """Local bind is communication-free; consensus runs at the draft consumer.

    Capture starts with all-false masks. Replays must synchronize newly staged
    local masks, preserve unanimous fallback and report post-consensus metrics.
    """
    rank = dist.get_rank()
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
    buffers = extension.create_graph_buffers(4)
    captured = buffers.bind_capture_view(4)
    neural = torch.tensor([90, 91, 92, 93], device=device)
    seq_lens = torch.ones(4, dtype=torch.int64, device=device)

    def select():
        captured.prepare(seq_lens)
        return captured.select_step(neural, 0)

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

    token_address = captured.tokens.data_ptr()
    mask_address = captured.use_retrieval.data_ptr()
    for staged in (False, True):
        for order in (requests, requests[::-1]):
            batch = SimpleNamespace(
                reqs=order,
                is_extend_in_batch=False,
                has_grammar=False,
                sampling_info=SimpleNamespace(is_all_greedy=True),
                seq_lens=torch.tensor(
                    [6 if rank == ranks[-1] else 5, 5], device=device
                ),
            )
            with extension.bind(batch, draft_input) as view:
                # A different local mask survives bind until eager prepare or
                # replay; no host-side consensus/readback chooses this branch.
                assert view.use_retrieval.tolist() == [rank != ranks[-1], True]
                if staged:
                    buffers.stage(view, 2, 4)
                    if graph is None:
                        output = select()
                    else:
                        graph.replay()
                    assert output.tolist() == [90, 3, 92, 93]
                    assert not captured.use_retrieval[2:].any()
                    # Restaging an aliased view must not erase valid rows.
                    buffers.stage(view, 2, 4)
                    assert view.use_retrieval.tolist() == [False, True]
                    assert captured.tokens.data_ptr() == token_address
                    assert captured.use_retrieval.data_ptr() == mask_address
                else:
                    view.prepare(batch.seq_lens)
                    assert view.select_step(neural[:2], 0).tolist() == [90, 3]
                assert view.use_retrieval.tolist() == [False, True]
            assert draft_input.draft_override is None
            extension.on_verify_result(
                SimpleNamespace(accept_lens=torch.tensor([1, 3], device=device))
            )
    assert extension.metrics.snapshot()["retrieval_requests"] == 4
    assert extension.metrics.snapshot()["neural_requests"] == 4
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
        _check_device_sync(dist.group.WORLD, list(range(world_size)), device)
        if world_size == 4:
            # A subgroup excluding global rank zero must remain independent.
            ranks = [1, 3]
            group = dist.new_group(ranks=ranks, timeout=timedelta(seconds=60))
            if rank in ranks:
                _check_consensus(group, ranks, device)
                _check_device_sync(group, ranks, device)
                dist.destroy_process_group(group)
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend,world_size",
    [("gloo", 1), ("gloo", 2), ("gloo", 4), ("gloo", 8), ("nccl", 2), ("nccl", 8)],
)
def test_tp_consensus_and_graph_staging(tmp_path, backend, world_size):
    if backend == "gloo" and not dist.is_gloo_available():
        pytest.skip("requires Gloo")
    if backend == "nccl" and (
        not dist.is_nccl_available() or torch.cuda.device_count() < world_size
    ):
        pytest.skip(f"requires NCCL and {world_size} NVIDIA GPUs, plus nvcc for SA")
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
