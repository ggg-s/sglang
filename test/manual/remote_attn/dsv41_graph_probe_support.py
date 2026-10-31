"""Small native-ABI-shaped model fixture; no DSV checkpoint/kernel substitution claim."""

from types import SimpleNamespace

import torch
from sglang_remote_attn import dsv41_graph as dg


def fixture(device="cpu", dtype=torch.float32, *, on_attention=None):
    controller = SimpleNamespace(
        graph_fast_path=True, dsv41_remote=True, remote_targets={}, targets=[]
    )
    calls, prepared, metadata = [], [], []
    bank = [None]

    class AttentionBackend:
        is_dsv41 = True
        trtllm_attn = mega_attn = False

        def init_forward_metadata_in_graph(self, forward_batch):
            self.current = forward_batch
            if bank[0] and bank[0].use_remote:
                metadata.append(forward_batch.marker)

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
            if bank[0] and bank[0].use_remote:
                assert attn_backend.current is forward_batch
                prepared.append((forward_batch.marker, self.layer_id))
            q = x * 2 + positions[:, None]
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
                name = dg.PREPARE[layer.layer_id]
                q = getattr(layer, name)(x, positions, fb, attn)
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
    )
