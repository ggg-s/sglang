"""Native Qwen GDN prefix match/CoW + native PD lifetime, on small CUDA pools.

No model weights or HTTP performance claim. Native cache/pools, Conv/GDN kernels,
PD page/state selection, retirement and Mooncake TCP bytes are real. Model token
sampling and the sender/metadata integration boundary are explicit fixtures.
"""

import argparse
import hashlib
import json
import tempfile
from array import array
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
from mooncake.engine import TransferEngine
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.configs.qwen3_5 import Qwen3_5Config
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin as PD
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import acquire_rows, kv_storage_of
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mooncake_contract import write_once
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.tree_slots import tree_slots_of
from sglang_remote_attn import pre_layerwise as pl, qwen_hybrid as hybrid


class Sender:
    def __init__(self, owner):
        self.owner = owner
        self.kv_mgr = owner.manager
        self.state = KVPoll.Transferring
        self.pending, self.sent = [], []
        self.supports_host_tensor_indices = False
        self.cleared = False

    def get_max_transfer_tokens(self):
        return None

    def should_send_kv_chunk(self, count, last):
        return count > 0 or last

    def send(self, pages, states, **kwargs):
        self.pending.append((list(pages), states))
        self.sent.append((list(pages), states))

    def poll(self):
        return self.state

    def clear(self):
        self.cleared = True

    def complete(self):
        for pages, states in self.pending:
            ids = torch.tensor(pages, device="cuda", dtype=torch.long)
            tensors = [
                t.index_select(0, ids).view(torch.uint8).flatten()
                for t in hybrid.kv_buffers(self.owner.pool)
            ]
            if states is not None:
                slot = int(states[0][0])
                tensors.extend(
                    t[slot].view(torch.uint8).flatten()
                    for _, t, _, _ in hybrid.state_entries(self.owner.pool)
                )
            data = torch.cat(tensors)
            size = data.numel()
            assert size < self.owner.tx.numel()
            self.owner.tx[:size].copy_(data.cpu())
            assert (
                write_once(
                    self.owner.engines[0],
                    self.owner.session,
                    [self.owner.tx.data_ptr()],
                    [self.owner.rx.data_ptr()],
                    [size],
                )
                == 0
            )
            received = self.owner.rx[:size].cuda()
            assert torch.equal(data, received)
            self.owner.transfers += 1
        torch.cuda.synchronize()
        self.pending.clear()
        self.state = KVPoll.Success


class Probe(PD):
    def __init__(self, folder):
        config = Qwen3_5Config(
            text_config=dict(
                num_hidden_layers=4,
                full_attention_interval=4,
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=6,
                num_key_value_heads=1,
                head_dim=256,
                linear_num_key_heads=1,
                linear_num_value_heads=3,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
            )
        )
        text = config.text_config
        self.req_to_token_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=24,
            mamba_spec_state_size=8,
            max_context_len=64,
            device="cuda",
            enable_memory_saver=False,
            cache_params=text.mamba2_cache_params,
            mamba_layer_ids=text.linear_layer_ids,
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=0,
        )
        self.pool = HybridLinearKVPool(
            size=128,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=256,
            full_attention_layer_ids=text.full_attention_layer_ids,
            device="cuda",
            enable_memory_saver=False,
            mamba_pool=self.req_to_token_pool.mamba_pool,
        )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=128,
            dtype=torch.bfloat16,
            device="cuda",
            kvcache=self.pool,
            need_sort=False,
        )
        self.token_to_kv_pool_allocator.state = self.req_to_token_pool.state
        self.tree_cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                page_size=1,
                tree_slots=tree_slots_of(
                    (ComponentType.FULL, ComponentType.MAMBA), checkpoint_interval=64
                ),
                tree_core_backend="python",
            )
        )
        self.storage = kv_storage_of(
            self.req_to_token_pool, self.token_to_kv_pool_allocator
        )
        ptrs, sizes, strides = self.pool.get_contiguous_buf_infos()
        sp, sz, ss = self.pool.get_state_buf_infos()
        self.manager = NS(
            enable_staging=False,
            enable_all_cp_ranks_for_transfer=False,
            kv_args=NS(
                kv_data_ptrs=ptrs,
                kv_data_lens=sizes,
                kv_item_lens=strides,
                state_types=[StateType.MAMBA],
                state_data_ptrs=[sp],
                state_data_lens=[sz],
                state_item_lens=[ss],
            ),
        )
        self.disagg_prefill_bootstrap_queue = NS(kv_manager=self.manager)
        self.tp_worker = NS(
            model_runner=NS(
                model_config=NS(hf_config=config, num_hidden_layers=4),
                lora_manager=None,
                model=torch.nn.ModuleList([RadixAttention(6, 256, 256**-0.5, 1, 3)]),
            )
        )
        self.disaggregation_mode = NS(value="prefill")
        self.kv_checksum_computer = None
        self.scheduler_stage_metrics = None
        self.enable_staging = self.enable_overlap = False
        self.disagg_metadata_buffers = NS(set_buf=lambda req: None)
        self.disagg_prefill_pending_chunk_rids = set()
        self.disagg_prefill_transfer_queue = IndexedReqList(ListKind.PD_SENDING)
        self.attn_cp_cpu_group = self.attn_tp_cpu_group = dist.group.WORLD
        self.request_retirer = RequestRetirer(
            tree_cache=self.tree_cache,
            beam_coordinator=NS(retire_group=lambda req: None),
            hisparse_coordinator=None,
            uocr_runtime=None,
            send_abort_output=lambda *a: None,
        )
        self.output_streamer = NS(stream_output=lambda *a: None)
        self.req_to_metadata_buffer_idx_allocator = NS(free=lambda *a: None)
        self._pre_layerwise = hybrid.Controller(
            self, pl.Config("remote", Path(folder) / "p")
        )
        self._pre_layerwise.validate()
        self.transfers = 0
        self.engines = [TransferEngine(), TransferEngine()]
        for engine in self.engines:
            assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        self.session = f"127.0.0.1:{self.engines[1].get_rpc_port()}"
        self.tx = torch.empty(2 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
        self.rx = torch.empty_like(self.tx, pin_memory=True)
        for engine, t in zip(self.engines, (self.tx, self.rx)):
            assert engine.batch_register_memory([t.data_ptr()], [t.nbytes]) == 0
        self.m = self.pool.mamba_pool
        self.m.mamba_cache.conv[0].zero_()
        self.m.mamba_cache.temporal.zero_()
        torch.manual_seed(20261107)
        self.inputs = torch.randn(12, 1, 640, device="cuda", dtype=torch.bfloat16) * 0.1
        self.weight = torch.randn(640, 4, device="cuda", dtype=torch.bfloat16) * 0.1
        self.a = torch.randn(12, 1, 3, device="cuda", dtype=torch.bfloat16)
        self.b = torch.randn_like(self.a)
        self.A_log = torch.randn(3, device="cuda")
        self.dt_bias = torch.randn(3, device="cuda", dtype=torch.bfloat16)

    def new(self, rid, tokens, hit=False):
        req = Req(
            rid, "", array("q", tokens), SamplingParams(temperature=0, max_new_tokens=1)
        )
        req.extra_key = None
        req.full_untruncated_fill_ids = array("q", tokens)
        req.extend_end = len(tokens)
        if hit:
            req.init_next_round_input(self.tree_cache)
            req.lock = self.tree_cache.lock(req.last_node)
            prefix = self.tree_cache.prefix_device_indices(req)
        else:
            req.prefix_len = 0
            req.last_node = self.tree_cache.root_node_handle()
            prefix = torch.empty(0, device="cuda", dtype=torch.int64)
        assert acquire_rows(self.req_to_token_pool, self.req_to_token_pool.state, [req])
        suffix = self.token_to_kv_pool_allocator.alloc(len(tokens) - req.prefix_len)
        row = torch.cat((prefix, suffix))
        self.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), row)
        req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
        batch = NS()
        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])
        if batch.mamba_cow_src_indices is not None:
            self.storage.copy_state(
                self.req_to_token_pool.state_sub_pool,
                batch.mamba_cow_src_indices,
                batch.mamba_cow_dst_indices,
            )
        elif batch.mamba_clear_indices is not None:
            self.m.clear_slots(batch.mamba_clear_indices)
        req.disagg_kv_sender = Sender(self)
        req.start_send_idx = req.disagg_decode_prefix_len = 0
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        req.pending_bootstrap = False
        return req

    def step(self, req, indices):
        slot = (
            self.token_to_kv_pool_allocator.state_ids(
                self.req_to_token_pool.state_sub_pool
            )
            .ids(req.kv.mamba_pool_idx.reshape(1))
            .int()
        )
        outputs = []
        for token in indices:
            for li in range(3):
                mixed = causal_conv1d_update(
                    self.inputs[token].clone(),
                    self.m.mamba_cache.conv[0][li],
                    self.weight,
                    activation="silu",
                    conv_state_indices=slot,
                )
                out = mixed.new_empty((1, 1, 3, 128))
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed,
                    a=self.a[token],
                    b=self.b[token],
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    scale=128**-0.5,
                    initial_state=self.m.mamba_cache.temporal[li],
                    out=out,
                    ssm_state_indices=slot,
                    use_qk_l2norm_in_kernel=True,
                )
                outputs.append(out.clone())
            loc = self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, token]
            for i, tensor in enumerate(hybrid.kv_buffers(self.pool)):
                tensor[loc].fill_(token + i + 1)
        return outputs

    def pd(self, req, cached):
        sender = req.disagg_kv_sender
        if cached:
            self.maybe_send_cached_prefix_chunk(req)
            assert len(sender.sent) == 1 and len(sender.sent[0][0]) == cached
            assert sender.sent[0][1] is None
        self.checkpoint_disagg_prefill(req)
        self._send_kv_chunk(req, last_chunk=True)
        expected = (
            self.token_to_kv_pool_allocator.state_ids(
                self.req_to_token_pool.state_sub_pool
            )
            .ids(
                self.req_to_token_pool.req_index_to_mamba_index_mapping[
                    req.kv.req_pool_idx
                ]
            )
            .cpu()
        )
        assert sender.sent[-1][1][0][0].tolist() == expected.tolist()
        self.disagg_prefill_transfer_queue.join([req])
        row, slot = req.kv.req_pool_idx, req.kv.mamba_pool_idx.clone()
        self.process_disagg_prefill_transfer_queue()
        assert req.kv.holds_kv and req.kv.holds_mamba and req.kv.req_pool_idx == row
        assert torch.equal(req.kv.mamba_pool_idx, slot) and not sender.cleared
        sender.complete()
        guard = self._pre_layerwise.cache_guard
        lease = None if guard is None else guard.by_req.get(id(req))
        if lease is None:
            self.process_disagg_prefill_transfer_queue()
        else:
            # Native sender is complete; the separate PD Remote probe verifies
            # that retirement also waits for the reader/GPU/TP proof.
            lease.retire(self.process_disagg_prefill_transfer_queue)
            assert lease.closed and lease.published_tokens == len(req.origin_input_ids)
        assert req.kv.is_kv_released and sender.cleared
        if req.kv.holds_mamba:
            # A fresh native final insertion donates the working state to the
            # tree; core retains its old numeric id on the retired Req. It is
            # tree-owned, and must not be returned to the free list again.
            donated = self.tree_cache.tree_core.get_component_device_value(
                req.last_node, ComponentType.MAMBA
            )
            assert torch.equal(donated.reshape(-1), slot.reshape(-1))
        else:
            assert not req.kv.holds_mamba
        assert sum(len(pages) for pages, _ in sender.sent) == len(req.origin_input_ids)

    def close(self):
        torch.cuda.synchronize()
        for engine, t in zip(self.engines, (self.tx, self.rx)):
            assert engine.batch_unregister_memory([t.data_ptr()]) == 0
        self.tree_cache.shutdown()


def run(folder):
    probe = Probe(folder)
    try:
        warm = probe.new("warm", range(8))
        probe.step(warm, range(8))
        probe.pd(warm, 0)
        from sglang_remote_attn.source_cache import install

        guard = install(probe.tree_cache, allow_mamba=True, scheduler=probe)
        probe._pre_layerwise.cache_guard = guard
        cached = probe.new("hit", range(10), True)
        lease = guard.hold(cached)
        assert cached.prefix_len == 8
        prefix_slot = probe.tree_cache.tree_core.get_component_device_value(
            cached.last_node, ComponentType.MAMBA
        ).clone()
        cached_bytes = [
            t[int(prefix_slot.item())].clone()
            for _, t, _, _ in hybrid.state_entries(probe.pool)
        ]
        assert int(prefix_slot.item()) != int(cached.kv.mamba_pool_idx.item())
        actual = probe.step(cached, range(8, 10))
        # The mutable working state is now at 10. An older checkpoint must
        # remain deferred and never be replayed with this newer GDN state.
        probe.tree_cache.checkpoint(cached, up_to=8)
        assert lease.pending_checkpoint[0] == 8 and not lease.closed
        cold = probe.new("cold", range(10))
        expected = probe.step(cold, range(10))[-6:]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        for old, (_, t, _, _) in zip(cached_bytes, hybrid.state_entries(probe.pool)):
            assert torch.equal(old, t[int(prefix_slot.item())])
        probe.pd(cached, 8)
        probe.pd(cold, 0)
        again = probe.new("repeat", range(12), True)
        assert again.prefix_len == 10
        probe.step(again, range(10, 12))
        probe.pd(again, 10)
        assert (
            not probe._pre_layerwise.sources
            and not guard.leases
            and lease.closed
            and not lease.replayed_checkpoint
        )
        return dict(
            prefix_hits=[8, 10],
            native_cache=True,
            remote_read_pin_preserves_final_frontier=True,
            cached_gdn_matches_cold_exactly=True,
            cow_preserves_cached_checkpoint=True,
            native_pd_full_prompt_transferred=True,
            native_pd_uses_working_state_slot=True,
            pending_pd_retains_kv_and_gdn=True,
            native_release_after_success=True,
            mooncake_transfers=probe.transfers,
        )
    finally:
        probe.close()


def main():
    import os

    os.environ["HANDOFF_REMOTE_GPU_TRANSFER"] = "1"
    os.environ["MC_FORCE_TCP"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder:
        dist.init_process_group(
            "gloo", init_method="file://" + folder + "/gloo", rank=0, world_size=1
        )
        try:
            with (
                get_context().override_server_args(
                    disaggregation_mode="prefill",
                    language_model_only=True,
                    disable_radix_cache=False,
                    disable_overlap_schedule=True,
                ),
                get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1),
            ):
                result = run(folder)
        finally:
            dist.destroy_process_group()
    root = Path(__file__).resolve().parents[3]
    files = [
        "plugins/remote_attn/sglang_remote_attn/qwen_hybrid.py",
        "test/manual/remote_attn/qwen_hybrid_prefix_cache_cuda_probe.py",
    ]
    result["sha256"] = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
