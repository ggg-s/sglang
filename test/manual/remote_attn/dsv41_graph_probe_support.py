"""Small native-ABI-shaped model fixture; no DSV checkpoint/kernel substitution claim."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang_remote_attn import dsv41_graph as dg


def fixture(
    device="cpu",
    dtype=torch.float32,
    *,
    on_attention=None,
    fused=False,
    bf16_source=False,
):
    controller = SimpleNamespace(
        graph_fast_path=True, dsv41_remote=True, remote_targets={}, targets=[]
    )
    calls, prepared, metadata = [], [], []
    bank = [None]
    projections = []
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "split_qkv_native_support",
        Path(__file__).with_name("dsv41_remote_probe_support.py"),
    )
    support = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support)
    Core = support.definitions(
        root,
        "python/sglang/srt/models/deepseek_v4.py",
        owner="MQALayer",
        members={
            "prepare_qkv",
            "project_decode_qkv",
            "update_decode_state",
            "_compute_q_a",
            "_compute_kv_to_cache",
            "_compute_kv_bf16",
        },
    )
    cache = [torch.full((8, 4), -99.0, device=device, dtype=dtype) for _ in range(3)]
    carry = [torch.zeros(8, device=device, dtype=dtype) for _ in range(3)]

    def norm(kv, weight, eps, freqs, pos):
        # External fused norm/RoPE kernel boundary; preserve in-place semantics.
        kv.mul_(0.5).add_(pos[:, None] * 0.03125)

    def store(*, layer_id, swa_loc, kv, positions, **kw):
        if bank[0] and bank[0].use_remote:
            assert (
                not torch.cuda.is_current_stream_capturing()
                if device == "cuda"
                else True
            )
            prepared.append((attn.current.marker, layer_id))
        value = (kv.float() * 0.5 + positions.float()[:, None] * 0.03125).to(dtype)
        cache[layer_id].index_copy_(0, swa_loc, value)

    pool = SimpleNamespace(set_swa_key_buffer_radix_fused_norm_rope=store)
    globals_ = Core._compute_kv_to_cache.__globals__
    globals_.update(
        envs=SimpleNamespace(
            SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE=SimpleNamespace(
                get=lambda: bf16_source
            )
        ),
        get_token_to_kv_pool=lambda: pool,
        TYPE_CHECKING=False,
        fused_norm_rope_inplace=norm,
        _is_hip=False,
        _is_npu=False,
    )

    class AttentionBackend:
        is_dsv41 = True
        trtllm_attn = mega_attn = False

        def init_forward_metadata_in_graph(self, forward_batch):
            self.current = forward_batch
            if bank[0] and bank[0].use_remote:
                metadata.append(forward_batch.marker)

        def get_swa_out_cache_loc(self, forward_batch):
            assert self.current is forward_batch
            return forward_batch.out_cache_loc

        def store_cache(self, *, layer_id, swa_k, forward_batch):
            if bank[0] and bank[0].use_remote:
                prepared.append((forward_batch.marker, layer_id))
            cache[layer_id].index_copy_(
                0, self.get_swa_out_cache_loc(forward_batch), swa_k
            )

        def forward_low_ratio_sources(
            self, *, layer, x, q_lora, positions, forward_batch
        ):
            assert self.current is forward_batch
            carry[layer.layer_id].index_add_(
                0, forward_batch.out_cache_loc, q_lora.sum(-1)
            )

        def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
            assert k is v
            if bank[0] and bank[0].use_remote:
                assert self.current is forward_batch
                assert forward_batch.marker >= 0
                calls.append((forward_batch.marker, layer.layer_id))
                if on_attention:
                    return on_attention(q, layer, forward_batch)
                if getattr(forward_batch, "fail", None) == layer.layer_id:
                    raise RuntimeError("RPC failed")
                return q + forward_batch.marker
            return q

    attn = AttentionBackend()

    class Layer(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.layer_id = index
            self.side = torch.cuda.Stream() if device == "cuda" else None
            self.is_dsv41 = True
            self.compress_ratio = index
            self.use_mega_attention = self.use_fused_qk_norm_rope = False
            self.fuse_wqa_wkv = fused
            self.q_lora_rank = 4
            self.compressor = self.indexer = object() if index else None
            self.kv_norm = SimpleNamespace(
                weight=torch.ones(4, device=device, dtype=dtype)
            )
            self.eps = 1e-6
            self.freqs_cis = torch.empty(0, device=device)

        def wq_a(self, x):
            projections.append(("qa", self.layer_id))
            return x + 0, None

        def wqkv_a(self, x):
            projections.append(("qkv", self.layer_id))
            return torch.cat((x, x * 3 + 0.125), dim=-1), None

        def wkv(self, x):
            projections.append(("kv", self.layer_id))
            return x * 3 + 0.125, None

        def _normalize_q_lora(self, q):
            return q, q

        def _compute_q_b(self, q, positions, q_out=None):
            projections.append(("qb", self.layer_id))
            value = q * 2 + positions[:, None]
            if q_out is not None:
                q_out.copy_(value)
                return q_out
            return value

        def _forward_prepare(
            self,
            x,
            positions,
            forward_batch,
            attn_backend,
            q_out=None,
            x_quant=None,
            q_rope_out=None,
            k_nope_out=None,
            k_rope_out=None,
        ):
            packed = self.wqkv_a(x)[0] if self.fuse_wqa_wkv else None
            q_lora, q_for_wqb = self._compute_q_a(x, qkv_a=packed)
            q = self._compute_q_b(q_for_wqb, positions, q_out)
            attn_backend.layer_id = self.layer_id
            self._compute_kv_to_cache(
                x, positions, forward_batch, attn_backend, qkv_a=packed
            )
            if self.compress_ratio:
                attn_backend.forward_low_ratio_sources(
                    layer=self,
                    x=x,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                )
            return q, None

        def _forward_prepare_multi_stream(
            self, x, positions, forward_batch, attn_backend, q_out=None, x_quant=None
        ):
            # Boundary returns differ between serial tuple and multi-stream Q.
            if self.side is None:
                return self._native_prepare(x, positions, forward_batch, attn_backend)[
                    0
                ]
            current = torch.cuda.current_stream()
            self.side.wait_stream(current)
            with torch.cuda.stream(self.side):
                result = self._native_prepare(
                    x, positions, forward_batch, attn_backend
                )[0]
            current.wait_stream(self.side)
            result.record_stream(current)
            return result

        def _forward_prepare_low_ratio_multi_stream(
            self, x, positions, forward_batch, attn_backend, q_out=None, x_quant=None
        ):
            if self.side is None:
                return self._native_prepare(x, positions, forward_batch, attn_backend)[
                    0
                ]
            current = torch.cuda.current_stream()
            self.side.wait_stream(current)
            with torch.cuda.stream(self.side):
                result = self._native_prepare(
                    x, positions, forward_batch, attn_backend
                )[0]
            current.wait_stream(self.side)
            result.record_stream(current)
            return result

    for name in (
        "prepare_qkv",
        "project_decode_qkv",
        "update_decode_state",
        "_compute_q_a",
        "_compute_kv_to_cache",
        "_compute_kv_bf16",
    ):
        setattr(Layer, name, getattr(Core, name))
    layers = [Layer(i) for i in range(3)]
    for layer in layers:
        layer._native_prepare = layer._forward_prepare

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(self_attn=l) for l in layers]
            )

        def forward(self, x, positions, fb):
            attn.init_forward_metadata_in_graph(fb)
            for layer in layers:
                q = layer.prepare_qkv(
                    x,
                    positions,
                    fb,
                    attn,
                    enable_multi_stream=layer.layer_id == 1,
                    low_ratio_multi_stream=layer.layer_id == 2,
                )
                if isinstance(q, tuple):
                    q = q[0]
                x = (
                    attn.forward(
                        q,
                        q,
                        q,
                        layer,
                        fb,
                        compress_ratio=layer.layer_id,
                        save_kv_cache=False,
                        attn_sink=None,
                    )
                    + 1
                )
            return x

    model = Model()
    mr = SimpleNamespace(
        model=model,
        model_config=SimpleNamespace(
            num_hidden_layers=3,
            hf_config=SimpleNamespace(
                model_type="deepseek_v41", compress_ratios=[0, 1, 2]
            ),
        ),
        attn_backend=attn,
        _pre_layerwise=controller,
    )
    runner = SimpleNamespace(
        model_runner=mr,
        attn_backend=attn,
        device_module=torch.cuda if device == "cuda" else None,
        cuda_graph_output_capacity_rows=lambda out: out.shape[0],
        cuda_graph_output_rows=lambda out: out.shape[0],
    )

    def batch(size, marker=0, fail=None):
        return SimpleNamespace(
            batch_size=size,
            input_ids=torch.zeros(size, dtype=torch.int64, device=device),
            input_embeds=None,
            replace_embeds=None,
            req_pool_indices_cpu=list(range(size)),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            marker=marker,
            out_cache_loc=torch.arange(size, device=device, dtype=torch.int64)
            + max(0, marker) % max(1, 9 - size),
            fail=fail,
        )

    def attach(backend):
        bank[0] = backend
        runner.backend = backend
        mr._remote_segmented_backend = backend
        dg.install(mr, runner)

    return SimpleNamespace(
        controller=controller,
        model=model,
        runner=runner,
        mr=mr,
        attn=attn,
        layers=layers,
        batch=batch,
        attach=attach,
        calls=calls,
        prepared=prepared,
        metadata=metadata,
        projections=projections,
        cache=cache,
        carry=carry,
        core=Core,
    )
