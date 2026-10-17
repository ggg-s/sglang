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
    return runtime.prepare([(rid, list(prompt) + list(output), budget)])


@pytest.mark.parametrize("mode", ["full", "split", "split_no_skip"])
def test_native_arctic_candidates_match_upstream_and_history_is_incremental(mode):
    from arctic_inference.suffix_decoding import SuffixDecodingCache

    cfg = config(draft_graph_mode=mode)
    backend = ArcticBackend(cfg, "cpu")
    runtime = ArcticRuntime(cfg, backend, 1000)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]
    output = [4]
    view = add(runtime, "a", prompt, output)
    oracle = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=0)
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
    assert not backend.cache.active_requests and not backend.cache.cached_requests


def test_native_arctic_global_cache_retains_only_responses_and_fifo():
    cfg = config(enable_global_pool=True, max_shared_requests=1)
    backend = ArcticBackend(cfg, "cpu")
    runtime = ArcticRuntime(cfg, backend, 1000)
    add(runtime, "donor", [91, 92, 93, 94, 95], list(range(1, 9)))
    donor = runtime.states["donor"].key
    runtime.close_request("donor", retain=True)
    assert donor not in backend.cache.active_requests
    assert donor in backend.cache.cached_requests
    assert add(runtime, "probe", [1, 2, 3], [4]).tokens.tolist() == [[5, 6, 7, 8]]
    view = add(runtime, "probe", [91, 92, 93], [94])
    assert not view.use_retrieval.item()  # Another request's prompt is private.
    runtime.close_request("probe")
    add(runtime, "new", [99], [11, 12, 13, 14, 15, 16])
    runtime.close_request("new", retain=True)
    assert donor not in backend.cache.cached_requests
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
    assert old_key not in backend.cache.cached_requests
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
    assert not backend.cache.cached_requests
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
