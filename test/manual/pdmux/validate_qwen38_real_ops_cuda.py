"""Weight-free *real* GDN/QSA/FP8 kernel checks on two CUDA streams.

No simulated decoder layers. Source loading omits package/runtime imports only;
Triton kernels and their launch wrappers are compiled from unchanged repo files.
FLA prefill uses normal imports through namespace-only package bootstrap.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import logging
import math
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {}


def load(name, rel, **env):
    path = ROOT / rel
    SOURCES[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    tree = ast.parse(path.read_text())
    # Runtime metadata is provided explicitly for this SM89, normal-numerics
    # process. Function/kernel bodies and launch parameters are not rewritten.
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sglang"))
    ]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(env)
    sys.modules[name] = module
    # Preserve the source module's own annotation semantics. Inheriting this
    # harness's future-annotations flag breaks TileLang's nested dynamic shapes.
    exec(compile(tree, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def bootstrap_fla():
    for name in [
        "sglang",
        "sglang.kernels",
        "sglang.kernels.ops",
        "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.fla",
        "sglang.srt",
        "sglang.srt.utils",
    ]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(ROOT / "python" / name.replace(".", "/"))]
        sys.modules[name] = mod
    common = types.ModuleType("sglang.srt.utils.common")
    common.torch_release = tuple(
        int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
    )
    sys.modules[common.__name__] = common
    result = importlib.import_module("sglang.kernels.ops.attention.fla.chunk")
    for path in (ROOT / "python/sglang/kernels/ops/attention/fla").glob("*.py"):
        SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return result.chunk_gated_delta_rule


def pair_test(name, pfn, dfn, references, rounds):
    p0, d0 = pfn(), dfn()  # JIT compilation and independent serial controls
    torch.cuda.synchronize()
    ref_errors = []
    for actual, reference, atol in references(p0, d0):
        error = (actual.float() - reference.float()).abs().max().item()
        torch.testing.assert_close(
            actual.float(), reference.float(), atol=atol, rtol=atol
        )
        ref_errors.append(error)
    ready = torch.cuda.current_stream().record_event()
    ps, ds = torch.cuda.Stream(), torch.cuda.Stream()
    ps.wait_event(ready)
    ds.wait_event(ready)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=ds):
        graph_output = dfn()
    ds.synchronize()
    records = []
    for _ in range(rounds):
        with torch.cuda.stream(ps):
            p = pfn()
        with torch.cuda.stream(ds):
            graph.replay()
            d = tuple(t.clone() for t in graph_output)
            torch.empty((128, 128), device="cuda").fill_(3)
        records.append((p, d))
    ps.synchronize()
    ds.synchronize()
    for p, d in records:
        for actual, control in zip(p + d, p0 + d0):
            torch.testing.assert_close(actual, control, atol=0, rtol=0)
    for actual, reference, atol in references(p0, d0):
        torch.testing.assert_close(
            actual.float(), reference.float(), atol=atol, rtol=atol
        )
    return {
        "name": name,
        "rounds": rounds,
        "decode_cuda_graph": True,
        "dual_stream_exact_match": True,
        "reference_max_abs": ref_errors,
    }


def recurrent_reference(q, k, v, a, b, A, bias, initial):
    state = initial.float().clone()  # [HV,V,K]
    outputs = []
    hv = v.shape[2]
    group = hv // q.shape[2]
    for t in range(q.shape[1]):
        qt = q[0, t].float().repeat_interleave(group, 0) / math.sqrt(q.shape[-1])
        kt = k[0, t].float().repeat_interleave(group, 0)
        vt = v[0, t].float()
        gate = -A.float().exp() * F.softplus(a[t].float() + bias.float())
        beta = b[t].float().sigmoid().to(b.dtype).float()
        state = state * gate.exp()[:, None, None]
        delta = vt - torch.einsum("hvk,hk->hv", state, kt)
        state = state + (delta * beta[:, None])[:, :, None] * kt[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, qt))
    return torch.stack(outputs).unsqueeze(0).to(q.dtype), state


def gdn_case(recurrent, chunk):
    H, HV, K = 2, 6, 128  # Qwen's 3:1 value/key head ratio and real head dims
    pool = (torch.randn((3, 2, HV, K, K), device="cuda") * 0.02)[:, 0]
    assert not pool.is_contiguous()  # envelope-strided state slot pitch
    initial = pool.clone()
    items = []
    refs = []
    for lane, T in enumerate((65, 1)):
        q = F.normalize(torch.randn(1, T, H, K, device="cuda"), dim=-1).bfloat16()
        k = F.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
        v = torch.randn(1, T, HV, K, device="cuda", dtype=torch.bfloat16) * 0.1
        a = torch.randn(T, HV, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A = torch.randn(HV, device="cuda") * 0.1
        bias = torch.randn(HV, device="cuda") * 0.1
        idx = torch.tensor([lane], device="cuda", dtype=torch.int32)
        ref, state = recurrent_reference(q, k, v, a, b, A, bias, initial[lane])
        refs.append((ref, state))

        def fn(lane=lane, q=q, k=k, v=v, a=a, b=b, A=A, bias=bias, idx=idx, T=T):
            pool[lane].copy_(initial[lane])
            if T > 1 and chunk is not None:
                g = (
                    -A.exp()[None, :] * F.softplus(a.float() + bias[None, :])
                ).unsqueeze(0)
                beta = b.float().sigmoid().to(b.dtype).unsqueeze(0)
                out, _, _ = chunk(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state=pool,
                    initial_state_indices=idx,
                    inplace_update=True,
                )
            else:
                out = recurrent(A, a, bias, 1.0, 20.0, q, k, v, b, pool, idx)
            return out, pool[lane].clone()

        items.append(fn)

    def reference(p, d):
        return [
            (p[0], refs[0][0], 0.015),
            (p[1], refs[0][1], 0.015),
            (d[0], refs[1][0], 0.002),
            (d[1], refs[1][1], 0.002),
            (pool[2], initial[2], 0),
        ]

    return items[0], items[1], reference


def fp8_case(fp8):
    N, K = 256, 256
    W = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    ws = torch.rand(N // 128, K // 128, device="cuda") * 0.2 + 0.8
    fns = []
    refs = []
    for M in (65, 1):
        x = (torch.randn(M, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        xs = torch.rand(M, K // 128, device="cuda") * 0.2 + 0.8
        reference = (x.float() * xs.repeat_interleave(128, dim=1)) @ (
            W.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)
        ).T
        refs.append(reference.bfloat16())
        fns.append(lambda x=x, xs=xs: (fp8(x, W, xs, ws, [128, 128], torch.bfloat16),))
    return *fns, lambda p, d: [(p[0], refs[0], 0.003), (d[0], refs[1], 0.003)]


def qsa_case(sparse, kernel):
    fns = []
    refs = []
    shared_k = torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16)
    shared_v = torch.randn_like(shared_k)
    for lane, rows in enumerate((65, 1)):
        length = 128
        heads = 6
        kvheads = 1
        dim = 256
        topk = 32
        q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = shared_k[lane * length : (lane + 1) * length]
        v = shared_v[lane * length : (lane + 1) * length]
        cuq = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        cuk = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        lens = torch.tensor([length], device="cuda", dtype=torch.int32)
        indices = (
            torch.arange(topk, device="cuda", dtype=torch.int32)
            .expand(rows, topk)
            .contiguous()
        )
        refs.append(kernel.qsa_sparse_attention_reference(q, k, v, indices))

        def fn(q=q, k=k, v=v, indices=indices, cuq=cuq, cuk=cuk, lens=lens, rows=rows):
            op = (
                sparse.sparse_gqa_packed_decode_triton
                if rows == 1
                else sparse.sparse_gqa_fwd_interface_triton_ck
            )
            kwargs = {} if rows == 1 else {"max_q": rows}
            return (op(q, k, v, indices, cuq, cuk, lens, dim**-0.5, **kwargs),)

        fns.append(fn)
    return *fns, lambda p, d: [(p[0], refs[0], 0.015), (d[0], refs[1], 0.015)]


def qsa_metadata_case(kernel):
    # Real compressed-key gather + block expansion; fixed request-owned inputs.
    fns = []
    refs = []
    for rows in (65, 1):
        compressed = torch.randn(64, 1, 128, device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(128, device="cuda", dtype=torch.int64).view(1, -1)
        cu = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
        blocks = torch.tensor([[0, 2, 4, 6]], device="cuda", dtype=torch.int32).repeat(
            rows, 1
        )
        positions = torch.full((rows,), 31, device="cuda", dtype=torch.int32)
        lengths = positions + 1
        ref = kernel.torch_expand_qsa_block_indices(blocks, positions, lengths, 4, 16)
        refs.append((compressed[:32].clone(), ref))
        fns.append(
            lambda compressed=compressed, slots=slots, cu=cu, blocks=blocks, positions=positions, lengths=lengths: (
                kernel.gather_qsa_compressed_keys(compressed, slots, cu, 32, 32, 4),
                kernel.expand_qsa_block_indices(blocks, positions, lengths, 4, 16),
            )
        )
    return *fns, lambda p, d: [
        (x, y, 0) for pair, ref in zip((p, d), refs) for x, y in zip(pair, ref)
    ]


def topk_case():
    from tvm_ffi.cpp import load_inline

    header = ROOT / "python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh"
    SOURCES[str(header.relative_to(ROOT))] = hashlib.sha256(
        header.read_bytes()
    ).hexdigest()
    source = "\n".join(
        [
            "#include <tvm/ffi/container/tensor.h>",
            "#include <tvm/ffi/dtype.h>",
            "#include <tvm/ffi/error.h>",
            "#include <tvm/ffi/extra/c_env_api.h>",
            "#include <tvm/ffi/function.h>",
            f'#include "{header}"',
            "namespace sglang {",
            "TVM_FFI_DLL_EXPORT_TYPED_FUNC(fast_topk, (FastTopKKernel<512, false>::run));",
            "}",
        ]
    )
    module = load_inline(
        "qwen38_real_topk512_sm89",
        cuda_sources=source,
        extra_cuda_cflags=[
            "-DSGL_CUDA_ARCH=890",
            "-std=c++20",
            "-O3",
            "-allow-unsupported-compiler",
            "-gencode=arch=compute_89,code=sm_89",
        ],
        extra_include_paths=[str(ROOT / "python/sglang/kernels/jit/include")],
    )
    fns, refs = [], []
    for rows in (65, 1):
        scores = torch.randn(rows, 1031, device="cuda")
        lengths = torch.full((rows,), 1024, device="cuda", dtype=torch.int32)
        if rows > 1:
            lengths[0] = 507
        starts = torch.full_like(lengths, 3)
        reference = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
        for row, length in enumerate(lengths.tolist()):
            reference[row, : min(length, 512)] = torch.topk(
                scores[row, 3 : 3 + length], min(length, 512)
            ).indices.int()
        refs.append(reference.sort(dim=1).values)

        def fn(scores=scores, starts=starts, lengths=lengths, rows=rows):
            output = torch.empty((rows, 512), device="cuda", dtype=torch.int32)
            module.fast_topk(scores, starts, output, lengths)
            # Atomic collection order is explicitly unspecified; selected sets
            # must agree exactly with the serial control and Torch reference.
            return (output.sort(dim=1).values,)

        fns.append(fn)
    return *fns, lambda p, d: [(p[0], refs[0], 0), (d[0], refs[1], 0)]


def registered_gdn_regression():
    path = ROOT / "test/registered/attention/unittests/gdn/test_triton.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if getattr(n, "name", None) == "TestTritonGDNBackendCorrectness"
    )
    method = next(
        n
        for n in cls.body
        if getattr(n, "name", None)
        == "test_chunk_prefill_updates_strided_state_and_track_pools"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace[method.name](unittest.TestCase())
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(48)
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_capability()
    if cap != (8, 9):
        raise RuntimeError("This standalone bootstrap is explicitly for the SM89 4060")
    logging.basicConfig(level=logging.WARNING)
    shared = dict(
        is_cuda=lambda: True,
        is_hip=lambda: False,
        is_gfx95_supported=lambda: False,
        is_sm90_supported=lambda: False,
        get_device_sm=lambda: 89,
        is_arch_support_pdl=lambda: False,
    )
    gdn = load(
        "real_gdn",
        "python/sglang/kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py",
        **shared,
    )
    fp8 = load(
        "real_fp8",
        "python/sglang/kernels/ops/gemm/fp8_kernel.py",
        is_cuda=lambda: True,
        is_gfx1250_supported=lambda: False,
        register_custom_op=lambda **kw: lambda fn: fn,
        get_device_name=torch.cuda.get_device_name,
        get_platform=lambda: types.SimpleNamespace(is_sm90=False),
        log_info_on_rank0=lambda logger, message: logger.info(message),
    )
    qsa = load(
        "real_qsa_meta",
        "python/sglang/srt/layers/attention/qsa/kernel.py",
        is_batch_invariant_mode_enabled=lambda: False,
        is_hip=lambda: False,
    )
    sparse = load(
        "real_qsa_sparse",
        "python/sglang/srt/layers/attention/qsa/sparse_attn.py",
        is_batch_invariant_mode_enabled=lambda: False,
    )
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "scope": "unchanged real kernels/wrappers, explicit SM89 runtime metadata",
        "cases": [],
        "errors": [],
    }
    cases = [
        (
            "gdn_recurrent",
            lambda: gdn_case(gdn.fused_sigmoid_gating_delta_rule_update, None),
        ),
        (
            "gdn_chunk_prefill_recurrent_decode",
            lambda: gdn_case(
                gdn.fused_sigmoid_gating_delta_rule_update, bootstrap_fla()
            ),
        ),
        ("fp8_block128_gemm", lambda: fp8_case(fp8.w8a8_block_fp8_matmul_triton)),
        ("qsa_sparse_prefill_decode", lambda: qsa_case(sparse, qsa)),
        ("qsa_compressed_gather_expand", lambda: qsa_metadata_case(qsa)),
        ("qsa_topk512_native_cuda", topk_case),
    ]
    for name, builder in cases:
        try:
            result = pair_test(name, *builder(), args.rounds)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            report["errors"].append(
                {"name": name, "error": f"{type(exc).__name__}: {exc}"}
            )
    try:
        registered_gdn_regression()
        report["strided_state_track_regression"] = "passed"
    except Exception as exc:
        report["errors"].append(
            {"name": "registered_gdn_regression", "error": str(exc)}
        )
    report["source_sha256"] = SOURCES
    report["passed"] = not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
