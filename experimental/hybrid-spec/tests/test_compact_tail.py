"""CPU recurrence checks for the actual V4 compact-tail implementation.

Only CUDA graph/stream, attention metadata and GPU topk boundaries are replaced.
These checks do not establish real CUDA capture, NCCL or V4 model correctness.
"""

import itertools
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from hybrid_spec.core.contracts import DraftOverrideView
from hybrid_spec.integrations.sglang import compact_graph


@pytest.mark.parametrize(
    "k,gate", [(2, 0), (2, 1), (3, 2), (8, 0), (8, 1), (8, 2), (8, 3)]
)
def test_dsv4_compact_tail_recurrence_refreshes_rows_kv_plan_and_step_metadata(
    monkeypatch, k, gate
):
    current = []

    @contextmanager
    def context(ctx):
        current.append(ctx.attn_backend)
        try:
            yield
        finally:
            current.pop()

    def postprocess(logits, positions, drafts, step):
        tokens = logits.argmax(dim=-1, keepdim=True)
        positions.add_(1)
        drafts[:, step : step + 1].copy_(tokens)
        return torch.ones_like(tokens, dtype=torch.float32), tokens

    modules = {
        "sglang.kernels.ops.speculative.topk1": {
            "draft_topk1_postprocess": postprocess
        },
        "sglang.srt.layers.attention.triton_backend": {
            "TritonMultiStepDraftBackend": object
        },
        "sglang.srt.model_executor.forward_context": {
            "ForwardContext": NS,
            "forward_context": context,
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    stream = NS(wait_stream=lambda other: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda other: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def capture(fn, stream):
        return NS(replay=fn), fn()

    monkeypatch.setattr(compact_graph, "_capture_graph", capture)

    class Child:
        def __init__(self, step):
            self.step = step

        def init_forward_metadata_out_graph(self, batch, in_capture):
            assert batch.seq_lens_cpu is None
            if in_capture:
                self.req = torch.empty_like(batch.req_pool_indices)
                self.seq = torch.empty_like(batch.seq_lens)
                self.raw = torch.empty_like(batch.out_cache_loc)
                self.swa = torch.empty_like(self.raw)
            self.req.copy_(batch.req_pool_indices)
            self.seq.copy_(batch.seq_lens)
            self.raw.copy_(batch.out_cache_loc)

        def init_forward_metadata_in_graph(self, batch):
            self.swa.copy_(batch.out_cache_loc.view(batch.batch_size, k)[:, self.step])

    class NativeAttention:
        decode_attention_backend_str = "dsv4"

        def __init__(self, runner, topk, steps):
            self.attn_backends = [Child(i) for i in range(steps)]

        def init_cuda_graph_state(self, *args):
            pass

        def on_after_cuda_graph_warmup(self):
            pass

    class DerivedAttention(NativeAttention):
        pass

    requests = torch.tensor([3, 0, 2, 1])
    positions = torch.tensor([151, 257, 389, 517])
    initial_positions = positions.clone()
    seq_lens = positions + 5
    cache = requests[None, :] * 1000 + torch.arange(k)[:, None]
    hidden = torch.arange(16).reshape(4, 4).float()
    prefix_tokens = torch.arange(4)[:, None] + 20
    candidates = torch.full((4, k), 71, dtype=torch.int64)
    candidates[:, 0] = torch.arange(4) + 10
    width = max(1, gate)
    candidates[:, width - 1 : width].copy_(prefix_tokens)
    drafts = candidates.clone()
    view = DraftOverrideView(candidates, torch.zeros(4, dtype=torch.bool), gate)
    original_plan = NS(rows=requests, write=cache.T.contiguous().flatten())
    original = NS(
        batch_size=4,
        positions=positions,
        mrope_positions=None,
        seq_lens=seq_lens,
        req_pool_indices=requests,
        extend_seq_lens=torch.ones(4, dtype=torch.int32),
        spec_info=NS(draft_override=view),
        kv_loc_plan=original_plan,
        mark_forward_metadata_ready=lambda: None,
    )
    work = torch.zeros(4, dtype=torch.int64)

    def bind(batch):
        batch.kv_loc_plan = NS(rows=batch.req_pool_indices, write=batch.out_cache_loc)

    def forward(batch):
        child = current[-1]
        assert batch.spec_info.hidden_states.shape == (batch.batch_size, 4)
        plan = batch.kv_loc_plan
        assert plan.rows is batch.req_pool_indices
        write = plan.write.view(batch.batch_size, k)[:, child.step]
        tokens = (
            batch.input_ids
            + batch.spec_info.hidden_states.long().sum(dim=1)
            + batch.positions
            + child.seq
            + child.step
            + child.req
            + child.raw
            + child.swa
            + write
        ) % 97
        work.index_add_(
            0, batch.req_pool_indices, torch.ones(batch.batch_size, dtype=torch.int64)
        )
        logits = torch.full((batch.batch_size, 97), -100.0)
        logits.scatter_(1, tokens[:, None], 1.0)
        return NS(
            logits_output=NS(
                next_token_logits=logits,
                hidden_states=batch.input_ids[:, None].float().expand(-1, 4).clone(),
            )
        )

    runner = NS(
        eagle_worker=NS(hot_token_id=None),
        draft_attn_backend=DerivedAttention(None, 1, k),
        speculative_num_steps=k,
        model_runner=NS(
            kv_index_translator=NS(bind_runner_slots=bind),
            forward=forward,
            model_config=NS(
                model_is_mrope=False,
                hf_config=NS(architectures=["DeepseekV4ForCausalLMNextN"]),
            ),
        ),
    )
    record = compact_graph.CompactCapture(
        None,
        None,
        original,
        view,
        (prefix_tokens, hidden, cache, drafts),
        prefix_width=width,
    )
    compact_graph.prepare_compaction(record, "split")
    for size in range(1, 5):
        tail = compact_graph._capture_tail(runner, record, size, stream)
        assert type(tail.attention) is DerivedAttention
        for iteration, selection in enumerate(itertools.combinations(range(4), size)):
            rows = selection[::-1]
            cache.copy_(
                requests[None, :] * 1000 + torch.arange(k)[:, None] + 100 * iteration
            )
            view.use_retrieval.fill_(True)
            view.use_retrieval[list(rows)] = False
            drafts.copy_(candidates)
            tail.rows.copy_(torch.tensor(rows))
            work.zero_()
            tail.graph.replay()
            expected = candidates.clone()
            for row in rows:
                token = int(prefix_tokens[row, 0])
                h = int(hidden[row].sum())
                for step in range(width - 1, k - 1):
                    nxt = (
                        token
                        + h
                        + int(positions[row])
                        + int(seq_lens[row])
                        + 2 * step
                        + int(requests[row])
                        + 3 * int(cache[step, row])
                    ) % 97
                    expected[row, step + 1] = nxt
                    h, token = token * 4, nxt
            assert torch.equal(drafts, expected)
            assert work.tolist() == [
                k - width if request in requests[list(rows)] else 0
                for request in range(4)
            ]
            assert original.kv_loc_plan is original_plan
            assert torch.equal(original.positions, initial_positions)
