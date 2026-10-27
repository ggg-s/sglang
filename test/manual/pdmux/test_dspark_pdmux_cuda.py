"""DSpark prefill/KV ownership on PDMux streams with small synthetic tensors.

Uses the production worker finalizer, MLA injector and DeepSeek-V4 draft KV
projection. The pool stores unencoded KV; this does not validate FlashMLA
encoding, HiCache DMA, model accuracy, or two-GPU NCCL progress.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import CommitKvProj
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/registered/unit/spec"))
sys.path.insert(0, str(ROOT / "test/registered/unit/multiplex"))
from test_dspark_pdmux import WORKER_MODULE, make_prefill_worker
from test_pdmux_sxf_port import load_methods
from test_pdmux_tp_overlap import lane_namespace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Linear(torch.nn.Module):
    quant_method = None

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


@pytest.mark.parametrize("unified", [False, True])
def test_prefill_finalizes_on_prefill_stream_between_decode_graph_replays(unified):
    torch.manual_seed(32)
    contexts = [torch.cuda.GreenContext.create(num_sms=n) for n in (8, 16)]
    prefill_streams = [ctx.Stream() for ctx in contexts]
    decode_streams = [torch.cuda.Stream() for _ in contexts]
    decode = NS(world_size=2, rank_in_group=0, broadcast=Mock())
    prefill = NS(world_size=2, rank_in_group=0, broadcast=Mock())
    lanes = lane_namespace(decode, prefill)
    sync = SpecTpSync(decode)
    assert sync.enabled(SpecTpSyncSite.DSPARK_TARGET)
    worker, batch, output = make_prefill_worker("cuda", sync)
    hidden = torch.randn(5, 8, device="cuda", dtype=torch.bfloat16)
    batch.out_cache_loc = torch.tensor([1, 2, 3, 4, 5], device="cuda")
    weights = [torch.randn(8, 8, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    stores = [torch.zeros(64, 8, device="cuda", dtype=torch.bfloat16) for _ in weights]
    store_streams = []

    def store(*, layer_id, swa_loc, kv, **kwargs):
        store_streams.append(torch.cuda.current_stream().cuda_stream)
        stores[layer_id].index_copy_(0, swa_loc.long(), kv)

    pool = NS(
        translate_loc_from_full_to_swa=lambda loc: loc + 8,
        set_swa_key_buffer_radix_fused_norm_rope=store,
        set_unified_key_buffer_radix_fused_norm_rope=store,
        unified_swa_ring_size=16,
        unified_swa_window=16,
    )
    model = NS(
        project_target_hidden=lambda value: value,
        stages=[
            NS(
                self_attn=NS(
                    layer_id=i,
                    wkv=Linear(weight),
                    kv_norm=NS(weight=torch.ones(8, device="cuda")),
                    eps=1e-6,
                    freqs_cis=None,
                )
            )
            for i, weight in enumerate(weights)
        ],
    )
    method = load_methods(
        "models/deepseek_v4_dspark.py",
        "DeepseekV4ForCausalLMDSpark",
        ["write_target_hidden_kv"],
        CommitKvProj=CommitKvProj,
        is_unified_kv_triton=lambda: unified,
        torch=torch,
    )["write_target_hidden_kv"]
    model.write_target_hidden_kv = MethodType(method, model)
    worker._kv_injector = TargetHiddenKvInjector(
        draft_model=model,
        draft_model_runner=NS(token_to_kv_pool=pool),
        model_runner=NS(device=torch.device("cuda")),
        device=torch.device("cuda"),
        verify_num_draft_tokens=4,
        block_pos_offsets=torch.arange(4, device="cuda"),
    )
    values = [torch.zeros(32, device="cuda") for _ in contexts]
    graphs = [torch.cuda.CUDAGraph() for _ in contexts]
    for stream, graph, value in zip(decode_streams, graphs, values):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            value.add_(1)
    torch.cuda.synchronize()
    for value in values:
        value.zero_()
    for stream in [*prefill_streams, *decode_streams]:
        stream.wait_stream(torch.cuda.current_stream())

    with (
        patch(
            f"{WORKER_MODULE}.get_parallel", return_value=NS(enable_dp_attention=False)
        ),
        patch(f"{WORKER_MODULE}.get_tp_group", side_effect=lanes.get_tp_group),
        patch(f"{WORKER_MODULE}.is_unified_kv_triton", return_value=unified),
        patch(
            "sglang.srt.speculative.dspark_components.dspark_kv_inject.is_unified_kv_triton",
            return_value=unified,
        ),
    ):
        for step in range(32):
            idx = step % len(contexts)
            # Serialize access to the shared draft pool across lane changes.
            if step:
                prefill_streams[idx].wait_stream(prefill_streams[1 - idx])
            lanes.set_pdmux_status(False)
            with torch.cuda.stream(decode_streams[idx]):
                graphs[idx].replay()
            lanes.set_pdmux_status(True)
            with torch.cuda.stream(prefill_streams[idx]):
                output.logits_output.hidden_states = hidden
                worker._finalize_prefill(batch, output, None)
    torch.cuda.synchronize()
    locations = (
        torch.tensor([18, 19, 20, 55, 56], device="cuda")
        if unified
        else batch.out_cache_loc + 8
    )
    for weight, cache in zip(weights, stores):
        torch.testing.assert_close(
            cache[locations], torch.nn.functional.linear(hidden, weight)
        )
    for value in values:
        torch.testing.assert_close(value, torch.full_like(value, 16))
    assert prefill.broadcast.call_count == 32
    decode.broadcast.assert_not_called()
    assert store_streams == [
        prefill_streams[step % 2].cuda_stream for step in range(32) for _ in weights
    ]
    assert output.logits_output.hidden_states is None
    torch.testing.assert_close(
        batch.seq_lens_cpu, torch.tensor([5, 9], dtype=torch.int32)
    )
