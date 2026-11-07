"""CUDA split/full equivalence through native Qwen loops and residual boundaries.

Uses the production Qwen3 full/split/decoder methods, split helper, controller
publication, StageBoundary/ResidualStream and RMSNorm; tiny CUDA linear attention
and MLP stand in for model weights/attention and TP layout is single-rank. This
is a lifecycle/continuation probe, not Qwen3-8B inference or a TP accuracy run.
"""

import argparse
import ast
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch


def method(path, cls, name, namespace):
    owner = next(
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", "") == cls
    )
    node = next(n for n in owner.body if getattr(n, "name", "") == name)
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
    from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.model_executor.split_prefill import forward_split_prefill
    from sglang.test.boundary_fixtures import make_test_stages, parallel_of, planning

    qwen = root / "python/sglang/srt/models/qwen3.py"
    ns = dict(
        torch=torch,
        residual_batch=residual_batch,
        AuxHiddenStateList=AuxHiddenStateList,
        _is_npu=False,
    )
    full = method(qwen, "Qwen3Model", "forward", dict(ns))
    split = method(qwen, "Qwen3ForCausalLM", "forward_split_prefill", dict(ns))
    decoder = method(qwen, "Qwen3DecoderLayer", "forward", dict(ns))
    publish = method(
        root / "plugins/remote_attn/sglang_remote_attn/pre_layerwise.py",
        "Controller",
        "forward",
        dict(functools=functools),
    )
    total, width, tokens = 5, 128, 7
    torch.manual_seed(451)
    cases = []

    class Compute(torch.nn.Module):
        def __init__(self, dtype, layer, kind):
            super().__init__()
            self.linear = torch.nn.Linear(
                width, width, bias=False, device="cuda", dtype=dtype
            )
            self.layer, self.kind = layer, kind

        def forward(self, hidden_states=None, forward_batch=None, **kwargs):
            if (
                self.kind == "attention"
                and getattr(forward_batch, "fail_layer", None) == self.layer
            ):
                raise RuntimeError("injected layer failure")
            if self.kind == "attention" and getattr(
                forward_batch, "delay_compute", False
            ):
                torch.cuda._sleep(50_000_000)
            result = torch.tanh(self.linear(hidden_states))
            if self.kind == "mlp":
                forward_batch.layer_outputs[self.layer].copy_(result)
            return result

    class Layer(torch.nn.Module):
        forward = decoder

        def __init__(self, dtype, index):
            super().__init__()
            self.self_attn = Compute(dtype, index, "attention")
            self.mlp = Compute(dtype, index, "mlp")
            stages = make_test_stages(
                attention_norm=RMSNorm(width).cuda().to(dtype),
                ffn_norm=RMSNorm(width).cuda().to(dtype),
                first=index == 0,
                last=index == total - 1,
            )
            self.attn_boundary, self.ffn_boundary = stages.attn, stages.ffn

    def batch(dtype, delay_compute=False):
        return NS(
            input_ids=torch.arange(tokens, device="cuda"),
            layer_outputs=torch.zeros(
                (total, tokens, width), device="cuda", dtype=dtype
            ),
            delay_compute=delay_compute,
            positions=torch.arange(tokens, device="cuda"),
            forward_mode=ForwardMode.EXTEND,
            split_index=0,
            hidden_states=None,
            residual_stream=None,
            attn_cp_metadata=None,
            lora_ids=None,
            spec_info=None,
        )

    with planning(parallel_of(attn_dp=1, attn_tp=1)):
        for dtype in (torch.float16, torch.bfloat16):
            model = NS(
                config=NS(num_hidden_layers=total),
                embed_tokens=torch.nn.Embedding(16, width, device="cuda", dtype=dtype),
                layers=[Layer(dtype, i) for i in range(total)],
                norm=RMSNorm(width).cuda().to(dtype),
                pp_group=NS(is_first_rank=True, is_last_rank=True),
                start_layer=0,
                end_layer=total,
                layers_to_capture=[],
            )
            outer = NS(
                config=model.config,
                model=model,
                lm_head=torch.randn(width, 23, device="cuda", dtype=dtype),
                logits_processor=lambda ids, h, weight, fb: h @ weight,
            )
            outer.forward_split_prefill = functools.partial(split, outer)
            baseline_fb = batch(dtype)
            expected = (
                full(model, baseline_fb.input_ids, baseline_fb.positions, baseline_fb)
                @ outer.lm_head
            )
            assert baseline_fb.residual_stream is None
            for group in (1, 2, 3, 8):
                fb = batch(dtype, delay_compute=True)
                metadata, sends, residual_ids = [], [], []
                ready = torch.cuda.Stream()
                consumer = torch.cuda.Stream()
                copied = []

                def original(runner, current, split_forward_count=1):
                    metadata_before = len(metadata)
                    result = forward_split_prefill(
                        model=outer,
                        attn_backend=NS(
                            init_forward_metadata=lambda b: metadata.append(
                                b.split_index
                            )
                        ),
                        device_timer=None,
                        forward_batch=current,
                        forward_count=split_forward_count,
                    )
                    if metadata_before:
                        assert len(metadata) == metadata_before
                    if current.residual_stream is not None:
                        residual_ids.append(id(current.residual_stream))
                        assert current.residual_stream.pending is not None
                    return result

                def submit(*a, **kw):
                    event = kw["wait_event"]
                    assert not event.query(), (
                        "producer proof must trail queued CUDA work"
                    )
                    layers = list(kw["kv_buffer_indices"])
                    with torch.cuda.stream(consumer):
                        consumer.wait_event(event)
                        copied.append((layers, fb.layer_outputs[layers].clone()))
                    sends.append((fb.split_index, event, layers))

                groups = [
                    (
                        list(range(i, min(i + group, total))),
                        list(range(i, min(i + group, total))),
                    )
                    for i in range(0, total, group)
                ]
                source = NS(
                    req=NS(
                        disagg_kv_sender=NS(
                            kv_mgr=NS(add_transfer_request=submit),
                            bootstrap_room=17,
                            trace_ctx=NS(copy_for_thread=lambda: None),
                        )
                    ),
                    indices=[],
                    index_slice=slice(None),
                    stager=None,
                    native_staging=False,
                    complete=lambda layers: None,
                )
                controller = NS(plan=[(source, groups)], config=NS(mode="layerwise"))
                ready.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(ready):
                    torch.cuda._sleep(250_000_000)
                    actual = publish(controller, original, None, fb)
                assert metadata == [0] and fb.split_index == total
                assert (
                    fb.forward_mode is ForwardMode.EXTEND and fb.residual_stream is None
                )
                assert len(set(residual_ids)) <= 1, (
                    "continuations must retain native residual stream"
                )
                ready.synchronize()
                consumer.synchronize()
                for layers, received in copied:
                    torch.testing.assert_close(
                        received, baseline_fb.layer_outputs[layers], rtol=0, atol=0
                    )
                assert all(event.query() for _, event, _ in sends)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert [end for end, _, _ in sends] == [
                    layers[-1] + 1 for layers, _ in groups
                ]
                cases.append(
                    dict(
                        dtype=str(dtype),
                        group_size=group,
                        groups=len(groups),
                        final_group=len(groups[-1][0]),
                        bitwise_equal=True,
                        metadata_once=True,
                        native_residual_retained=True,
                        producer_event_pending=True,
                        consumer_wait_observed_complete_payload=True,
                    )
                )

            # A failed second group must neither advance its continuation nor
            # enqueue a completion proof for partially computed layer state.
            fb = batch(dtype, delay_compute=True)
            fb.fail_layer = 2
            metadata, sends = [], []
            groups = [([0, 1], [0, 1]), ([2, 3], [2, 3]), ([4], [4])]
            controller = NS(plan=[(source, groups)], config=NS(mode="layerwise"))
            ready.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(ready):
                torch.cuda._sleep(250_000_000)
                try:
                    publish(controller, original, None, fb)
                except RuntimeError as exc:
                    assert str(exc) == "injected layer failure"
                else:
                    raise AssertionError("failed group unexpectedly completed")
            assert fb.split_index == 2 and len(sends) == 1 and metadata == [0]
            assert fb.forward_mode is ForwardMode.EXTEND
            ready.synchronize()
            consumer.synchronize()
            residual_batch.clear(fb)
            cases.append(
                dict(
                    dtype=str(dtype),
                    injected_failure=True,
                    completed_groups=len(sends),
                    split_index=fb.split_index,
                    failed_group_not_published=True,
                )
            )

    files = [
        qwen,
        root / "python/sglang/srt/model_executor/split_prefill.py",
        root / "plugins/remote_attn/sglang_remote_attn/pre_layerwise.py",
        Path(__file__).resolve(),
    ]
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cases=cases,
        scope=__doc__,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
