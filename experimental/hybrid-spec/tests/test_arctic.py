"""Optional native Arctic backend: lifecycle and independent upstream parity.

Requires the real ArcticInference suffix extension; no suffix-tree substitutes.
"""

from types import SimpleNamespace as NS

import pytest
import torch
from hybrid_spec.backends.arctic import ArcticBackend
from hybrid_spec.config import HybridConfig
from hybrid_spec.core.contracts import RequestKey, RequestSnapshot
from hybrid_spec.integrations.sglang.arctic_extension import (
    ArcticDraftExtension,
    ArcticRuntime,
)

pytest.importorskip("arctic_inference.suffix_decoding")


def config(**changes):
    return HybridConfig(
        **(
            dict(
                num_drafts=4,
                min_match_len=1,
                max_requests=2,
                max_history_tokens=128,
                backend="arctic",
                history_mode="cpu",
                draft_graph_mode="split",
                num_neural_gate_tokens=2,
                arctic_max_spec_factor=4.0,
            )
            | changes
        )
    )


def add(runtime, rid, prompt, output=(), budget=100):
    runtime.backend.set_prompt(rid, prompt)
    runtime.backend.set_confirmed(rid, output)
    return runtime.prepare([(rid, list(prompt) + list(output), budget)])


@pytest.mark.parametrize("mode", ["full", "split", "split_no_skip"])
@pytest.mark.parametrize("shared", [False, True])
def test_native_arctic_candidates_match_upstream_and_history_is_incremental(
    mode, shared
):
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    cfg = config(draft_graph_mode=mode, enable_global_pool=shared)
    backend = ArcticBackend(cfg, "cpu")
    runtime = ArcticRuntime(cfg, backend, 1000)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]
    output = [4]
    view = add(runtime, "a", prompt, output)
    oracle = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=cfg.num_sa_slots if shared else 0
    )
    oracle.start_request("a", prompt)
    oracle.add_active_response("a", output)
    expected = oracle.speculate(
        "a",
        prompt + output,
        max_spec_tokens=4,
        max_spec_factor=4.0,
        use_tree_spec=False,
    )
    assert view.tokens.tolist() == [expected.token_ids] == [[5, 6, 7, 8]]
    assert view.use_retrieval.tolist() == [True]
    assert view.query_metadata.tolist() == [[expected.match_len, 0]]
    state = runtime.states["a"]
    first_key = state.key
    # Replaying the same CPU history must not append it again.
    add(runtime, "a", prompt, output)
    assert backend.slots[state.slot].history == tuple(prompt + output)
    assert runtime.states["a"].key == first_key
    # A changed prefix/retraction evicts the old tree and advances generation.
    view = add(runtime, "a", [80, 81], [82])
    assert runtime.states["a"].key != first_key
    assert first_key not in backend.cache.active_requests
    assert not view.use_retrieval.item()
    runtime.reset()
    assert (
        not backend.cache.active_requests and not backend.global_cache.cached_requests
    )


def test_native_arctic_global_cache_retains_only_responses_and_fifo():
    cfg = config(enable_global_pool=True, max_shared_requests=1)
    backend = ArcticBackend(cfg, "cpu")
    runtime = ArcticRuntime(cfg, backend, 1000)
    add(runtime, "donor", [91, 92, 93, 94, 95], list(range(1, 9)))
    donor = runtime.states["donor"].key
    runtime.close_request("donor", retain=True)
    assert donor not in backend.cache.active_requests
    assert donor in backend.global_cache.cached_requests
    probe = add(runtime, "probe", [1, 2, 3], [4])
    assert probe.tokens.tolist() == [[5, 6, 7, 8]]
    assert probe.query_metadata[:, 1].tolist() == [1]
    view = add(runtime, "probe", [91, 92, 93], [94])
    assert not view.use_retrieval.item()  # Another request's prompt is private.
    runtime.close_request("probe")
    add(runtime, "new", [99], [11, 12, 13, 14, 15, 16])
    runtime.close_request("new", retain=True)
    assert donor not in backend.global_cache.cached_requests
    assert len(backend.completed) == 1
    assert not add(runtime, "probe", [1, 2, 3], [4]).use_retrieval.item()


def test_native_arctic_active_capacity_stale_versions_and_completed_donors():
    cfg = config(max_requests=1, enable_global_pool=True, max_shared_requests=2)
    backend = ArcticBackend(cfg, "cpu")
    runtime = ArcticRuntime(cfg, backend, 1000)
    add(runtime, "old", [99], list(range(1, 9)))
    runtime.close_request("old", retain=True)
    view = add(runtime, "a", [1, 2, 3], [4])
    assert view.use_retrieval.item()  # Completed donor does not occupy active slot.
    state = runtime.states["a"]
    snapshot = RequestSnapshot((state.key,), (state.slot,), (999,), (100,))
    assert not backend.query(snapshot, 4).healthy.item()
    stale = RequestKey("a", state.key.generation - 1)
    snapshot = RequestSnapshot((stale,), (state.slot,), (len(state.history),), (100,))
    assert not backend.query(snapshot, 4).healthy.item()
    assert not add(runtime, "b", [1, 2, 3], [4]).use_retrieval.item()


def test_native_arctic_short_candidate_and_budget_fallback():
    cfg = config(num_drafts=8)
    runtime = ArcticRuntime(cfg, ArcticBackend(cfg, "cpu"), 1000)
    assert not add(runtime, "a", [1, 2, 3, 4, 1, 2], [3]).use_retrieval.item()
    assert not add(
        runtime, "a", [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3], [4], budget=4
    ).use_retrieval.item()
    assert not add(runtime, "a", list(range(129)), [129]).use_retrieval.item()
    assert not runtime.states
    assert not runtime.backend.cache.active_requests


class Request:
    def __init__(self, rid, prompt, output):
        self.rid = rid
        self.origin_input_ids = prompt
        self.output_ids = output
        self.sampling_params = NS(max_new_tokens=100)
        self.multimodal_inputs = None
        self.done = False

    def finished(self):
        return self.done

    def get_fill_ids(self):
        return self.origin_input_ids + self.output_ids


def batch(reqs):
    return NS(
        reqs=reqs,
        is_extend_in_batch=False,
        has_grammar=False,
        sampling_info=NS(is_all_greedy=True),
        seq_lens=torch.tensor(
            [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        ),
    )


def test_arctic_finish_reconciles_final_token_and_stop_trim_before_publication():
    cfg = config(enable_global_pool=True, max_shared_requests=2, num_drafts=2)
    backend = ArcticBackend(cfg, "cpu")
    extension = ArcticDraftExtension(cfg, backend, 1000)
    req = Request("a", [99], [1, 2, 3, 4, 5, 6])
    with extension.bind(batch([req]), NS(draft_override=None)):
        pass
    old_key = extension.runtime.states["a"].key
    # Model progress included tokens later removed by stop handling.
    req.output_ids = [1, 2, 3, 4]
    req.done = True
    extension.close_request("a", retain=True)
    assert old_key not in backend.global_cache.cached_requests
    probe = Request("p", [1, 2], [3])
    with extension.bind(batch([probe]), NS(draft_override=None)) as view:
        assert not view.use_retrieval.item()  # Only [4] remains; [4,5] is invalid.
    extension.close_request("p")
    # A final token not seen by another draft must still be cached.
    req = Request("b", [99], [10, 11, 12])
    with extension.bind(batch([req]), NS(draft_override=None)):
        pass
    req.output_ids += [13, 14]
    req.done = True
    extension.close_request("b", retain=True)
    probe = Request("p2", [10, 11], [12])
    with extension.bind(batch([probe]), NS(draft_override=None)) as view:
        assert view.tokens.tolist() == [[13, 14]]
        assert view.use_retrieval.item()
    extension.reset()
    assert not backend.global_cache.cached_requests
    assert not extension.identities and not extension.runtime.states


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("gate", [0, 1, 3, 4])
def test_arctic_full_graph_stages_native_queries_and_clears_padding(gate):
    cfg = config(draft_graph_mode="full", num_neural_gate_tokens=gate)
    backend = ArcticBackend(cfg, "cuda")
    extension = ArcticDraftExtension(cfg, backend, 1000)
    a = Request("a", [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3], [4])
    b = Request("b", [21, 22, 23, 24, 25, 26, 27, 28, 21, 22, 23], [24])
    buffers = extension.create_graph_buffers(8)
    captured = buffers.bind_capture_view(8)
    neural = torch.full((8, 4), 99, dtype=torch.int64, device="cuda")

    def run():
        return torch.cat(
            [
                captured.select_step(neural[:, step : step + 1], step)
                for step in range(4)
            ],
            dim=1,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    addresses = (captured.tokens.data_ptr(), captured.use_retrieval.data_ptr())
    for order in ([a, b], [b, a], [a]):
        current = batch(order)
        current.seq_lens = current.seq_lens.to("cuda")
        with extension.bind(current, NS(draft_override=None)) as view:
            neural.fill_(99)
            if gate:
                neural[: len(order), :gate].copy_(view.tokens[:, :gate])
            expected = neural.clone()
            for row, req in enumerate(order):
                if req is a:
                    expected[row].copy_(view.tokens[row])
                elif gate:
                    neural[row, gate - 1] += 1
                    expected[row].copy_(neural[row])
                else:
                    # Exercise staged eligibility fallback independently of gate.
                    view.use_retrieval[row] = False
            buffers.stage(view, len(order), 8)
            graph.replay()
            assert torch.equal(output, expected)
            assert view.use_retrieval.tolist() == [r is a for r in order]
            assert not captured.use_retrieval[len(order) :].any().item()
            assert addresses == (
                captured.tokens.data_ptr(),
                captured.use_retrieval.data_ptr(),
            )
    graph.reset()
    extension.reset()


@pytest.fixture(params=["cpu", "cuda"])
def feedback_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires NVIDIA CUDA")
    return request.param


def forward_batch(reqs, lengths, device, middle=None):
    current = batch(reqs)
    current.forward_mode = NS(is_idle=lambda: False)
    current.seq_lens_cpu = torch.tensor(lengths)
    current.seq_lens = current.seq_lens_cpu.to(device)
    current.chunked_req = middle
    return current


def prefill(extension, reqs, anchors, *, middle=None, lengths=None):
    lengths = lengths or [len(r.get_fill_ids()) for r in reqs]
    current = forward_batch(reqs, lengths, extension.device, middle)
    ticket = extension.begin_prefill(current)
    extension.on_prefill(
        current,
        NS(next_token_ids=torch.tensor(anchors, device=extension.device)),
        ticket,
    )
    return ticket


@pytest.mark.parametrize("mode", ["full", "split", "split_no_skip"])
def test_arctic_overlap_uses_gpu_paths_before_cpu_commit_and_after_reorder(
    feedback_device, mode
):
    cfg = config(num_drafts=2, num_neural_gate_tokens=1, draft_graph_mode=mode)
    extension = ArcticDraftExtension(cfg, ArcticBackend(cfg, feedback_device), 1000)
    a = Request("a", [1, 2, 3, 4, 5, 6, 1, 2], [])
    b = Request("b", [21, 22, 23, 24, 25, 26, 21, 22], [])
    prefill(extension, [a, b], [3, 23])
    first = forward_batch([a, b], [8, 8], feedback_device)
    with extension.bind(first, NS(draft_override=None)) as view:
        assert view.tokens.tolist() == [[4, 5], [24, 25]]
        assert view.use_retrieval.tolist() == [True, True]
    # Non-contiguous global node indices, padding, unequal accept counts. CPU
    # still has NO prefill anchor, let alone the verify tokens or bonus.
    extension.on_device_verify(
        NS(
            next_token_ids=torch.tensor([4, 99, 5, 24, 25, 26], device=feedback_device),
            accept_index=torch.tensor([[0, 2, -1], [3, 4, 5]], device=feedback_device),
            accept_lens=torch.tensor([2, 3], device=feedback_device),
        ),
        first,
    )
    second = forward_batch([b, a], [11, 10], feedback_device)
    with extension.bind(second, NS(draft_override=None)):
        assert extension.runtime.states["a"].history == tuple(
            a.origin_input_ids + [3, 4, 5]
        )
        assert extension.runtime.states["b"].history == tuple(
            b.origin_input_ids + [23, 24, 25, 26]
        )
    # Scheduler catches up. Rebinding must not append the same tokens again.
    a.output_ids = [3, 4, 5]
    b.output_ids = [23, 24, 25, 26]
    with extension.bind(second, NS(draft_override=None)):
        assert len(extension.runtime.states["a"].history) == 11
        assert len(extension.runtime.states["b"].history) == 12
    assert extension.metrics.snapshot()["retrieval_accept_tokens"] == 5
    # Remaining budget must follow GPU progress rather than lagging CPU output.
    a.sampling_params.max_new_tokens = 5
    a.output_ids = []
    with extension.bind(
        forward_batch([a], [10], feedback_device), NS(draft_override=None)
    ) as view:
        assert not view.use_retrieval.item()
    extension.reset()
    assert not extension.histories


def test_arctic_chunked_prefill_skips_transient_bonus_and_cancels_stale_epoch(
    feedback_device,
):
    cfg = config(num_drafts=2, num_neural_gate_tokens=1)
    extension = ArcticDraftExtension(cfg, ArcticBackend(cfg, feedback_device), 1000)
    a = Request("a", [1, 2, 3, 4, 5, 6, 1, 2], [])
    b = Request("b", [21, 22, 23, 24, 25, 26, 21, 22], [])
    # Final b can progress while a is still in a middle chunk.
    prefill(extension, [a, b], [999, 23], middle=a, lengths=[4, 8])
    assert "a" not in extension.histories
    with extension.bind(
        forward_batch([b], [8], feedback_device), NS(draft_override=None)
    ) as view:
        assert view.tokens.tolist() == [[24, 25]] and view.use_retrieval.item()
    # Last a chunk snapshots the full prefix, including preceding/cached chunks.
    ticket = prefill(extension, [a], [3], lengths=[8])
    with extension.bind(
        forward_batch([a], [8], feedback_device), NS(draft_override=None)
    ) as view:
        assert view.tokens.tolist() == [[4, 5]] and view.use_retrieval.item()
        assert 999 not in extension.runtime.states["a"].history
    old_key = extension.runtime.states["a"].key
    # Retraction invalidates the old private history, even on a middle chunk.
    prefill(extension, [a], [998], middle=a, lengths=[4])
    assert "a" not in extension.histories and "a" not in extension.runtime.states
    assert old_key not in extension.runtime.backend.cache.active_requests
    prefill(extension, [a], [3], lengths=[8])
    extension.cancel_prefill(ticket)  # Old epoch must not cancel a replacement.
    assert "a" in extension.histories
    current = forward_batch([a], [8], feedback_device)
    ticket = extension.begin_prefill(current)
    extension.cancel_prefill(ticket)
    assert "a" not in extension.histories
    # Idle does not allocate/close leases.
    current.forward_mode = NS(is_idle=lambda: True)
    assert extension.begin_prefill(current) == ()
    extension.on_prefill(current, None, ())
    extension.reset()


def test_arctic_gpu_private_tail_never_enters_shared_cache_before_cpu_confirmation(
    feedback_device,
):
    cfg = config(
        num_drafts=2,
        num_neural_gate_tokens=1,
        enable_global_pool=True,
        max_shared_requests=2,
    )
    backend = ArcticBackend(cfg, feedback_device)
    extension = ArcticDraftExtension(cfg, backend, 1000)
    donor = Request("donor", [99], [1, 2, 3])
    first = forward_batch([donor], [3], feedback_device)
    with extension.bind(first, NS(draft_override=None)):
        pass
    extension.on_device_verify(
        NS(
            next_token_ids=torch.tensor([4, 5, 6], device=feedback_device),
            accept_index=torch.tensor([[0, 1, 2]], device=feedback_device),
            accept_lens=torch.tensor([3], device=feedback_device),
        ),
        first,
    )
    with extension.bind(
        forward_batch([donor], [6], feedback_device), NS(draft_override=None)
    ):
        assert extension.runtime.states["donor"].history == (99, 1, 2, 3, 4, 5, 6)
    probe = Request("probe", [1, 2], [3])
    with extension.bind(
        forward_batch([probe], [2], feedback_device), NS(draft_override=None)
    ) as view:
        assert not view.use_retrieval.item()
    donor.output_ids = [1, 2, 3, 4, 5]
    with extension.bind(
        forward_batch([donor, probe], [6, 2], feedback_device), NS(draft_override=None)
    ) as view:
        assert view.tokens[1].tolist() == [4, 5] and view.use_retrieval[1].item()
    # Stop handling truncates the uncommitted GPU tail (5,6). Only final output
    # is retained, with a fresh generation after the local history shrinks.
    old_key = extension.runtime.states["donor"].key
    donor.output_ids = [1, 2, 3, 4]
    donor.done = True
    extension.close_request("donor", retain=True)
    assert old_key not in backend.global_cache.cached_requests
    with extension.bind(
        forward_batch([probe], [2], feedback_device), NS(draft_override=None)
    ) as view:
        assert not view.use_retrieval.item()
    extension.reset()


@pytest.mark.parametrize("bad", ["version", "index", "count", "token"])
def test_arctic_feedback_mismatch_falls_back_and_request_reuse_isolated(
    feedback_device, bad
):
    cfg = config(num_drafts=2, num_neural_gate_tokens=1, max_requests=1)
    extension = ArcticDraftExtension(cfg, ArcticBackend(cfg, feedback_device), 1000)
    req = Request("a", [1, 2, 3, 4, 5, 6, 1, 2], [3])
    current = forward_batch([req], [8], feedback_device)
    with extension.bind(current, NS(draft_override=None)):
        pass
    if bad == "version":
        current.seq_lens += 1
    extension.on_device_verify(
        NS(
            next_token_ids=torch.tensor(
                [4, 1000 if bad == "token" else 5, 6], device=feedback_device
            ),
            accept_index=torch.tensor(
                [[0, -1 if bad == "index" else 1, 2]], device=feedback_device
            ),
            accept_lens=torch.tensor(
                [4 if bad == "count" else 2], device=feedback_device
            ),
        ),
        current,
    )
    with extension.bind(
        forward_batch([req], [10], feedback_device), NS(draft_override=None)
    ) as view:
        assert not view.use_retrieval.item()
    req.done = True  # Prefill-only/health requests have no finish hook.
    replacement = Request("a", [21, 22, 23, 24, 25, 26, 21, 22], [])
    prefill(extension, [replacement], [23])
    with extension.bind(
        forward_batch([replacement], [8], feedback_device), NS(draft_override=None)
    ) as view:
        assert view.tokens.tolist() == [[24, 25]] and view.use_retrieval.item()
    extension.reset()


def test_arctic_prefill_capacity_and_history_limit_fall_back_without_cpu_anchor(
    feedback_device,
):
    cfg = config(
        num_drafts=2, num_neural_gate_tokens=1, max_requests=1, max_history_tokens=8
    )
    extension = ArcticDraftExtension(cfg, ArcticBackend(cfg, feedback_device), 1000)
    a = Request("a", [1, 2, 3, 4, 5, 6, 1, 2], [])
    # Oversized prompt has no lease and CPU hasn't received its first bonus.
    assert prefill(extension, [a], [3]) == (None,)
    with extension.bind(
        forward_batch([a], [8], feedback_device), NS(draft_override=None)
    ) as view:
        assert not view.use_retrieval.item()
    extension.reset()
    # No capacity for the second row. Its missing CPU anchor is a row fallback,
    # while the first row still consumes its prefill feedback normally.
    a = Request("a", [1, 2, 3, 4, 5, 1], [])
    b = Request("b", [21, 22, 23, 24, 25, 21], [])
    ticket = prefill(extension, [a, b], [2, 22])
    assert ticket[0] is not None and ticket[1] is None
    current = forward_batch([a, b], [6, 6], feedback_device)
    with extension.bind(current, NS(draft_override=None)) as view:
        assert view.use_retrieval.tolist() == [True, False]
    extension.on_device_verify(
        NS(
            next_token_ids=torch.tensor([3, 4, 5, 23, 24, 25], device=feedback_device),
            accept_index=torch.tensor([[0, 1, 2], [3, 4, 5]], device=feedback_device),
            accept_lens=torch.tensor([3, 3], device=feedback_device),
        ),
        current,
    )
    with extension.bind(
        forward_batch([a, b], [9, 9], feedback_device), NS(draft_override=None)
    ) as view:
        assert not view.use_retrieval.any().item()
    assert all(
        len(s.tokens) <= cfg.max_history_tokens for s in extension.histories.values()
    )
    extension.reset()


def test_arctic_separate_trees_preserve_upstream_local_global_score_choice():
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    cfg = config(max_requests=3, enable_global_pool=True, max_shared_requests=2)
    backend = ArcticBackend(cfg, "cpu")
    runtime = ArcticRuntime(cfg, backend, 1000)
    oracle = SuffixDecodingCache(
        max_tree_depth=64, max_cached_requests=cfg.num_sa_slots
    )
    histories = {
        "a": ([99], [1, 2, 3, 4, 5, 6, 7, 8]),
        "b": ([1, 2, 3, 14, 15, 16, 17, 18, 1, 2], [3]),
        "p": ([1, 2], [3]),
    }
    for rid, (prompt, output) in histories.items():
        oracle.start_request(rid, prompt)
        oracle.add_active_response(rid, output)
        add(runtime, rid, prompt, output)
    for rid, (prompt, output) in histories.items():
        expected = oracle.speculate(
            rid,
            prompt + output,
            max_spec_tokens=4,
            max_spec_factor=4.0,
            use_tree_spec=False,
        )
        state = runtime.states[rid]
        snapshot = RequestSnapshot(
            (state.key,), (state.slot,), (len(state.history),), (100,)
        )
        candidate = backend.query(snapshot, 4)
        assert (
            candidate.tokens[0, : len(expected.token_ids)].tolist()
            == expected.token_ids
        )
        assert candidate.num_valid_drafts.item() == len(expected.token_ids)
        assert candidate.match_lens.item() == expected.match_len


def test_arctic_prefill_failure_releases_partial_ticket(feedback_device):
    extension = ArcticDraftExtension(
        config(), ArcticBackend(config(), feedback_device), 1000
    )
    a, b = Request("a", [1, 2], []), Request("b", [3, 4], [])

    def fail():
        raise RuntimeError("prefill snapshot failed")

    b.get_fill_ids = fail
    with pytest.raises(RuntimeError, match="snapshot failed"):
        extension.begin_prefill(forward_batch([a, b], [2, 2], feedback_device))
    assert not extension.histories and not extension.identities
    extension.reset()
