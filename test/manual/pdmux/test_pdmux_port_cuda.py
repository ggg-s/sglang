"""CUDA smoke checks for the opt-to-latest-main PDMux port.

Run with pytest on a CUDA host. Scheduling methods are loaded unchanged from
production source; tiny CUDA operators stand in for the model and attention
backends. This checks graph/stream and input ownership, not model accuracy.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "test/registered/unit/multiplex/test_pdmux_sxf_port.py"
spec = importlib.util.spec_from_file_location("pdmux_cuda_port_helpers", HELPER)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
load_methods = helpers.load_methods

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def lanes():
    torch.cuda.init()
    torch.empty(1, device="cuda")
    contexts = [torch.cuda.GreenContext.create(num_sms=n) for n in (8, 16)]
    groups = [(torch.cuda.Stream(), c.Stream()) for c in contexts]
    yield groups
    torch.cuda.synchronize()


@pytest.mark.parametrize("draft_extend", [False, True])
def test_draft_graphs_replay_on_their_capture_stream_while_prefill_runs(
    lanes, draft_extend
):
    state = NS(idx=1, prefill=True, pool=torch.cuda.graph_pool_handle())
    original_pool = state.pool
    original_backend = object()
    worker = NS(
        server_args=None,
        draft_runner=None,
        topk=1,
        speculative_num_steps=3,
        seed_dsa_topk_from_draft_extend=False,
        draft_attn_backend=original_backend,
    )
    backend_ct = 0

    def make_backend():
        nonlocal backend_ct
        backend_ct += 1
        return NS(bias=float(backend_ct))

    @contextlib.contextmanager
    def graph_capture(*, stream):
        yield NS(stream=stream)

    capture = load_methods(
        "model_executor/runner/decode_cuda_graph_runner.py",
        "DecodeCudaGraphRunner",
        ["capture"],
        freeze_gc=lambda *a: contextlib.nullcontext(),
        empty_context=contextlib.nullcontext,
        graph_capture=graph_capture,
        get_exec=lambda: NS(graph=NS(enable_cudagraph_gc=False)),
        get_or_create_global_graph_capture_stream=torch.cuda.current_stream,
    )["capture"]

    class TinyRunner:
        def __init__(self, worker, *, capture_stream, **kwargs):
            self.capture_stream = capture_stream
            self.attention = next(iter(kwargs.values()))
            self.capture_bs, self.max_bs = [1], 1
            self.enable_pdmux = self.enable_torch_compile = False
            self.enable_profile_cuda_graph = False
            self.seq_len_fill_value = 1
            self.buffers = NS(
                seq_lens=torch.ones(1, device="cuda"),
                seq_lens_cpu=torch.ones(1),
                reset_index_buffers=lambda: None,
            )
            self.backend = NS(capture_session=lambda stream: contextlib.nullcontext())
            self.inputs = torch.ones(4096, device="cuda")
            self.outputs = torch.empty_like(self.inputs)
            self.graph = torch.cuda.CUDAGraph()
            capture(self)
            assert self.stream.cuda_stream == self.capture_stream.cuda_stream

        def warmup(self):
            self.capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.capture_stream):
                for _ in range(3):
                    self.outputs.copy_(self.inputs * 3 + self.attention.bias)
            self.capture_stream.synchronize()

        def _capture_one_stream(self):
            with torch.cuda.graph(self.graph, pool=state.pool, stream=self.stream):
                self.outputs.copy_(self.inputs * 3 + self.attention.bias)

        def can_run_graph(self, batch):
            return True

        def execute(self, batch):
            assert (
                torch.cuda.current_stream().cuda_stream
                == self.capture_stream.cuda_stream
            )
            self.inputs.copy_(batch.values)
            self.graph.replay()
            return self.outputs.clone()

    methods = load_methods(
        "speculative/pdmux_draft_cuda_graph_runner.py",
        "PDMuxDraftCudaGraphRunner",
        ["__init__", "__getattr__", "can_run_graph", "execute"],
        torch=torch,
        logger=logging.getLogger(__name__),
        get_stream_groups=lambda: lanes,
        get_current_stream_idx=lambda: state.idx,
        set_current_stream_idx=lambda idx: setattr(state, "idx", idx),
        is_pdmux_prefill_enabled=lambda: state.prefill,
        set_pdmux_status=lambda value: setattr(state, "prefill", value),
        get_global_graph_memory_pool=lambda: state.pool,
        set_global_graph_memory_pool=lambda value: setattr(state, "pool", value),
        DraftBackendFactory=lambda *a, **kw: NS(
            create_decode_backend=make_backend,
            create_draft_extend_backend=make_backend,
        ),
    )
    wrapper = type("StreamBoundGraphs", (), methods)(
        worker, TinyRunner, draft_extend=draft_extend
    )
    assert state.idx == 1 and state.prefill and state.pool is original_pool
    assert worker.draft_attn_backend is original_backend
    assert len({id(r.attention) for r in wrapper.runners}) == len(lanes)

    prefill_stream = lanes[0][0]
    prefill_input = torch.arange(4096, device="cuda", dtype=torch.float32)
    prefill_output = torch.empty_like(prefill_input)
    prefill_stream.wait_stream(torch.cuda.current_stream())
    prefill_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(prefill_graph, pool=original_pool, stream=prefill_stream):
        prefill_output.copy_(prefill_input * 5 + 7)
    torch.cuda.synchronize()

    results = []
    for step in range(32):
        idx = step % len(lanes)
        state.idx = idx
        with torch.cuda.stream(prefill_stream):
            prefill_graph.replay()
        with torch.cuda.stream(lanes[idx][1]):
            values = torch.full((4096,), float(step), device="cuda")
            batch = NS(values=values, batch_size=1, can_run_decode_cuda_graph=True)
            assert wrapper.can_run_graph(batch)
            results.append(
                (wrapper.execute(batch), step * 3 + wrapper.runners[idx].attention.bias)
            )
    torch.cuda.synchronize()
    for result, expected in results:
        torch.testing.assert_close(
            result, torch.full_like(result, expected), rtol=0, atol=0
        )
    torch.testing.assert_close(prefill_output, prefill_input * 5 + 7, rtol=0, atol=0)


def test_split_prefill_keeps_cuda_inputs_until_final_mtp_extend(lanes):
    stream = lanes[0][1]
    extend_mode = NS(is_idle=lambda: False)
    split_mode = NS(is_idle=lambda: False)
    forward = load_methods(
        "speculative/eagle_worker_v2.py",
        "EAGLEWorkerV2",
        ["forward_batch_split_prefill"],
        CaptureHiddenMode=NS(FULL="full"),
        ForwardMode=NS(EXTEND=extend_mode),
        speculative_moe_backend_context=contextlib.nullcontext,
        speculative_moe_a2a_backend_context=contextlib.nullcontext,
        spec_stage_span=lambda *a: contextlib.nullcontext(),
    )["forward_batch_split_prefill"]
    load_scheduler_split_branch = load_methods(
        "../../../test/registered/unit/multiplex/test_pdmux_split_input_lifetime.py",
        None,
        ["load_scheduler_split_branch"],
        ast=ast,
        SRT=ROOT / "python/sglang/srt",
    )["load_scheduler_split_branch"]
    observed, drafts = [], []

    def resolve(batch, future_map):
        assert batch.input_ids is None
        batch.input_ids = torch.arange(4096, device="cuda")
        observed.append(batch.input_ids)

    run_split = load_scheduler_split_branch(resolve)
    batch = NS(
        forward_mode=split_mode,
        input_ids=None,
        seq_lens=[],
        seq_lens_cpu=None,
        split_index=0,
        split_forward_count=1,
        req_pool_indices=[],
    )

    def target(batch, **kwargs):
        final = batch.split_index == 2
        logits = (
            NS(hidden_states=batch.input_ids * 2, mm_input_embeds=None)
            if final
            else None
        )
        return NS(
            logits_output=logits,
            next_token_ids=None,
            next_draft_input=None,
            new_seq_lens=None,
            has_sampled_token_ids=False,
        )

    def draft(batch, hidden, tokens, embeds):
        assert batch.input_ids is observed[0]
        assert batch.forward_mode is extend_mode
        drafts.append(hidden + batch.input_ids)
        return NS(hidden_states=drafts[-1])

    worker = NS(
        target_worker=NS(
            model_config=NS(num_hidden_layers=3), forward_batch_split_prefill=target
        ),
        draft_worker=NS(
            draft_runner=NS(tp_group=None),
            draft_tp_context=lambda *a: contextlib.nullcontext(),
            prefill_lane_draft_extend_backend=contextlib.nullcontext,
            _draft_extend_for_prefill=draft,
        ),
    )
    worker.forward_batch_split_prefill = lambda b: forward(worker, b)
    scheduler = NS(
        enable_overlap=False,
        model_worker=worker,
        tp_worker=worker,
        future_map=None,
        model_config=NS(num_hidden_layers=3),
        _relay_forward_payload=lambda *a: None,
        _copy_auxiliary_output_to_cpu=lambda *a: None,
    )
    with torch.cuda.stream(stream):
        for idx in range(3):
            batch.split_index = idx
            run_split(scheduler, batch)
            assert (batch.input_ids is None) == (idx == 2)
            assert len(drafts) == int(idx == 2)
        pressure = [torch.empty(4096, device="cuda") for _ in range(32)]
    stream.synchronize()
    assert len(observed) == 1 and batch.forward_mode is split_mode
    torch.testing.assert_close(drafts[0].cpu(), torch.arange(4096) * 3, rtol=0, atol=0)
    del pressure


def test_whole_prefill_capture_uses_lane_zero_and_restores_state(lanes):
    state = NS(idx=1, prefill=False)
    flags = NS(capturing_prefill_graph=False, prefill_graph_has_dp_gather=False)

    @contextlib.contextmanager
    def graph_capture(*, stream):
        yield NS(stream=stream)

    capture = load_methods(
        "model_executor/runner/prefill_cuda_graph_runner.py",
        "PrefillCudaGraphRunner",
        ["capture"],
        freeze_gc=lambda *a: contextlib.nullcontext(),
        graph_capture=graph_capture,
        get_exec=lambda: NS(graph=NS(enable_cudagraph_gc=False)),
        get_flags=lambda: NS(dp=flags),
        get_current_stream_idx=lambda: state.idx,
        set_current_stream_idx=lambda idx: setattr(state, "idx", idx),
        is_pdmux_prefill_enabled=lambda: state.prefill,
        set_pdmux_status=lambda value: setattr(state, "prefill", value),
        get_stream_groups=lambda: lanes,
        get_or_create_global_graph_capture_stream=torch.cuda.current_stream,
        logger=logging.getLogger(__name__),
    )["capture"]
    inputs = torch.arange(4096, device="cuda", dtype=torch.float32)
    outputs = torch.empty_like(inputs)
    graph = torch.cuda.CUDAGraph()
    runner = NS(
        pdmux_split=True,
        warmup=torch.cuda.synchronize,
        backend=NS(capture_session=lambda stream: contextlib.nullcontext()),
    )

    def capture_shape():
        assert state.idx == 0 and state.prefill and flags.capturing_prefill_graph
        assert runner.stream.cuda_stream == lanes[0][0].cuda_stream
        with torch.cuda.graph(graph, stream=runner.stream):
            outputs.copy_(inputs.square() + 11)

    runner._capture_one_stream = capture_shape
    capture(runner)
    assert state.idx == 1 and not state.prefill and not flags.capturing_prefill_graph
    with torch.cuda.stream(runner.stream):
        graph.replay()
    runner.stream.synchronize()
    torch.testing.assert_close(outputs, inputs.square() + 11, rtol=0, atol=0)

    def fail_capture():
        assert state.idx == 0 and state.prefill and flags.capturing_prefill_graph
        raise RuntimeError("capture boundary failure")

    runner._capture_one_stream = fail_capture
    with pytest.raises(RuntimeError, match="capture boundary failure"):
        capture(runner)
    assert state.idx == 1 and not state.prefill and not flags.capturing_prefill_graph
