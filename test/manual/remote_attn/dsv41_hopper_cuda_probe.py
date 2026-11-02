"""Real CUDA V4 quantize/store/sparse-dequant and C2 ring-pooling correctness.

Runs on RTX4060/Hopper without SM100 kernels. Native Triton kernels and torch
oracles are source-loaded to avoid unrelated serving/model imports. This probe
does not load a checkpoint, run FlashMLA attention or measure serving performance.
"""

import argparse
import ast
import dataclasses
import hashlib
import json
import sys
import types
from pathlib import Path

import torch
import triton
import triton.language as tl
from dsv41_remote_probe_support import load, v4_reference
from sglang_remote_attn.dsv41_sparse_math import read_slots


def source_module(root, relative, name, namespace, names):
    path = root / relative
    nodes = [
        n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names
    ]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    module = types.ModuleType(name)
    module.__dict__.update(namespace)
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


def main(root, output):
    torch.cuda.set_device(0)
    layout_path = "python/sglang/kernels/ops/attention/dsv4/kv_layout.py"
    store_path = "python/sglang/kernels/ops/kvcache/dsv4_main_kv_store.py"
    reference_path = "python/sglang/test/kernels/deepseek_v4/k_cache_reference.py"
    dequant_path = "python/sglang/kernels/ops/attention/dsv4/dequant_k_cache.py"
    pair_path = "python/sglang/kernels/ops/attention/dsv4/c2_decode_pool.py"
    layouts = load(root, layout_path, "hopper_probe_layout").KVLayout
    namespace = dict(
        torch=torch,
        triton=triton,
        tl=tl,
        dataclass=dataclasses.dataclass,
        fp8_dtype=torch.float8_e4m3fn,
        KVLayout=layouts,
        DIM_NOPE=448,
        DIM_ROPE=64,
        NOPE_ROPE_BYTES=576,
        NUM_SCALE_TILES=7,
        PADDED_SCALE_PER_TOKEN=8,
        TILE_SIZE=64,
    )
    store = source_module(
        root,
        store_path,
        "hopper_native_store",
        namespace,
        {
            "NopeFp8RopeBf16Pack",
            "SetKAndS",
            "_set_k_and_s_triton",
            "_set_k_and_s_triton_kernel",
        },
    )
    namespace["NopeFp8RopeBf16Pack"] = store.NopeFp8RopeBf16Pack
    quant = source_module(
        root,
        reference_path,
        "hopper_native_quant",
        namespace,
        {"_quant_k_cache_fused_kernel", "quant_to_nope_fp8_rope_bf16_pack_triton"},
    )
    dequant = source_module(
        root,
        dequant_path,
        "hopper_native_dequant",
        namespace,
        {"_dequantize_k_cache_paged_kernel", "dequantize_k_cache_paged_dsv4"},
    )
    pair = load(root, pair_path, "hopper_native_c2")
    oracle = v4_reference(root)
    torch.manual_seed(72)
    page_cases = []
    for page_size in (1, 4, 64, 128):
        capacity = 3 * page_size
        locations = torch.randperm(capacity, device="cuda")[
            : min(12, capacity)
        ].contiguous()
        values = (torch.randn(locations.numel(), 512, device="cuda") * 2).to(
            torch.bfloat16
        )
        cache = torch.full(
            (3, layouts.V4.page_bytes(page_size)), 253, dtype=torch.uint8, device="cuda"
        )
        before = cache.cpu()
        pack = quant.quant_to_nope_fp8_rope_bf16_pack_triton(values)
        store.SetKAndS.execute(
            types.SimpleNamespace(page_size=page_size), cache, locations, pack
        )
        cpu_pack = store.NopeFp8RopeBf16Pack(
            pack.k_nope_fp8.cpu(), pack.k_rope_bf16.cpu(), pack.scale_k_nope_ue8m0.cpu()
        )
        oracle.set_k_and_s_torch(before, locations.cpu(), cpu_pack, page_size)
        assert torch.equal(cache.cpu(), before), "native V4 store bytes differ"
        selected = torch.cat((locations.flip(0), locations[:2]))
        expected = oracle.dequantize_k_cache_paged_ref(
            before, selected.cpu(), page_size
        ).reshape(-1, 512)
        native = dequant.dequantize_k_cache_paged_dsv4(
            cache, selected, page_size
        ).reshape(-1, 512)
        actual = read_slots(cache.view(torch.bfloat16), selected, page_size, layouts.V4)
        assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
        assert torch.equal(actual.view(torch.int16), native.view(torch.int16))
        page_cases.append(
            dict(
                page_size=page_size,
                selected_slots=selected.numel(),
                store_byte_exact=True,
                native_dequant_bit_exact=True,
            )
        )

    ring = 8
    state = torch.randn(3 * ring + 1, 1024, device="cuda")
    state_kv, state_score = state[:, :512], state[:, 512:]
    req = torch.tensor([1, 0, 2], dtype=torch.int64, device="cuda")
    raw = torch.tensor([9, 0, 17], dtype=torch.int64, device="cuda")
    # Request 0 is a live row elsewhere: a padded graph row must never write
    # it. Consecutive positions exercise odd/even carry and ring wraparound.
    for step in range(12):
        pos = torch.tensor([step, step + 1, step + 2], dtype=torch.int64, device="cuda")
        out_loc = torch.tensor(
            [5 if step % 2 else -1, -1, 9 if step % 2 else -1],
            dtype=torch.int64,
            device="cuda",
        )
        kv, score = (
            torch.randn(3, 512, device="cuda"),
            torch.randn(3, 512, device="cuda"),
        )
        previous = state.clone()
        read = torch.where(
            (raw == 0) | (pos == 0), 3 * ring, req * ring + (pos - 1) % ring
        )
        weights = torch.stack((previous[read, 512:], score), dim=1).softmax(dim=1)
        expected = (torch.stack((previous[read, :512], kv), dim=1) * weights).sum(dim=1)
        pooled, group_pos, slots = pair.c2_decode_pool(
            kv,
            score,
            pos,
            raw,
            out_loc,
            req,
            state_kv,
            state_score,
            3 * ring,
            ring_size=ring,
        )
        torch.testing.assert_close(pooled, expected, atol=0, rtol=0)
        assert torch.equal(group_pos, torch.where(pos % 2 == 1, pos - 1, pos))
        assert torch.equal(slots, out_loc.clamp_min(0))
        expected_state = previous.clone()
        rows = req[[0, 2]] * ring + pos[[0, 2]] % ring
        expected_state[rows, :512], expected_state[rows, 512:] = (
            kv[[0, 2]],
            score[[0, 2]],
        )
        assert torch.equal(state, expected_state), "C2 carry/padding state changed"
    torch.cuda.synchronize()
    paths = [
        layout_path,
        store_path,
        reference_path,
        dequant_path,
        pair_path,
        "test/manual/remote_attn/dsv41_hopper_cuda_probe.py",
        "plugins/remote_attn/sglang_remote_attn/dsv41_sparse_math.py",
    ]
    result = dict(
        device=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        torch=torch.__version__,
        v4_pages=page_cases,
        c2_steps=12,
        c2_pool_bit_exact=True,
        c2_ring_and_padding_exact=True,
        scope=__doc__,
        source_sha256={
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
        },
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_root.resolve(), args.output.resolve())
