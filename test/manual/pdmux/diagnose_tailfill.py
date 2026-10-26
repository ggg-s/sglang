"""One-shape native recapture diagnosis; never admits a candidate."""
import json
import re
from pathlib import Path

import torch
from bench_mixed_full_model import SeparateTargetGraph


def diagnose_native_pair(mr, pfb, vfb, before, native, native_state, root):
    features = {}
    hooks = []
    capture = False

    def hook(name):
        def save(module, args, output):
            value = output if isinstance(output, torch.Tensor) else output[0]
            if isinstance(value, torch.Tensor):
                features.setdefault(name, []).append(
                    value.detach().clone() if capture else value.detach().to('cpu', copy=True)
                )
        return save

    def metrics(a, b):
        b = b.detach().cpu()
        err = (a.float() - b.float()).abs()
        return dict(shape=list(a.shape), max_abs=float(err.max()),
                    mismatch=int((~torch.isclose(a, b, rtol=.02, atol=.002)).sum()),
                    rel_l2=float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)))

    for name, module in mr.model.named_modules():
        if re.search(r'\.layers\.[0-3]\.', name) and name.endswith(
            ('.in_proj_qkvz', '.in_proj_ba', '.qkv_proj', '.out_proj', '.o_proj', '.mlp', '.linear_attn')
        ):
            hooks.append(module.register_forward_hook(hook(name)))
    try:
        before.restore()
        repeat_outputs = native()
        conv_tensor, _, native_conv = native_state.tensors['mamba.conv.0']
        repeated_conv = conv_tensor.detach().to('cpu', copy=True)
        repeat_conv = [metrics(native_conv[i], repeated_conv[i]) for i in range(native_conv.shape[0])]
        expected = features
        features = {}
        capture = True
        graph = SeparateTargetGraph(
            mr, pfb, vfb, prefill_capacity=len(pfb.input_ids), restore=before.restore,
            prefill_capacity_limit=32768, variable_batch=True, stable_numerics=False,
        )
        before.restore()
        graph.replay(pfb, vfb)
        torch.cuda.synchronize()
        result = dict(tokens=len(pfb.input_ids), p_requests=pfb.batch_size,
                      v_requests=vfb.batch_size, features={}, conv_by_layer=[], native_repeat_conv=repeat_conv)
        for name, values in expected.items():
            # Native Verify is already captured, hence hooks record Prefill only.
            # Constructor records warm P,V then captured P,V. Capture clones
            # are updated by graph replay; the penultimate entry is captured P.
            actual = features[name][-2]
            result['features'][name] = metrics(values[0], actual)
        tensor, _, expected_conv = native_state.tensors['mamba.conv.0']
        actual_conv = tensor.detach().cpu()
        for i in range(expected_conv.shape[0]):
            result['conv_by_layer'].append(metrics(expected_conv[i], actual_conv[i]))
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root/'native-recapture.json').write_text(json.dumps(result, indent=2))
        print('TAIL_DIAG', json.dumps(result), flush=True)
    finally:
        for handle in hooks:
            handle.remove()
        before.restore()
