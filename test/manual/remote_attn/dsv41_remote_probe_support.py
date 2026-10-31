"""Small source-loaded native layout/pool/metadata fixtures for sparse RPC tests."""

import ast
import dataclasses
import enum
import functools
import importlib.util
import sys
from types import SimpleNamespace

import torch


def load(root, relative, name):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(root, relative, *, owner=None, members=None):
    path = root / relative
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if owner:
        node = next(n for n in nodes if getattr(n, "name", None) == owner)
        node.bases, node.keywords = [], []
        node.body = [n for n in node.body if getattr(n, "name", None) in members]
        nodes = [node]
    namespace = dict(
        torch=torch,
        dataclasses=dataclasses,
        dataclass=dataclasses.dataclass,
        enum=enum,
        functools=functools,
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace[owner] if owner else SimpleNamespace(**namespace)


def encoded(layout, pages, page_size, device):
    raw = torch.zeros(
        (pages, layout.page_bytes(page_size)), dtype=torch.uint8, device=device
    )
    data_stride, base, scale_stride = layout.row_addressing(page_size)
    for row in range(page_size):
        for p in range(pages):
            values = (
                torch.arange(layout.data_bytes, device=device) + row * 7 + p * 11
            ) % (240 if layout.is_fp4 else 80)
            raw[p, row * data_stride : row * data_stride + layout.data_bytes] = (
                values.to(torch.uint8)
            )
            scale_values = torch.tensor(
                [48, 56, 60, 64] if layout.is_fp4 else [126, 127, 128, 129],
                dtype=torch.uint8,
                device=device,
            )
            raw[
                p,
                base + row * scale_stride : base
                + row * scale_stride
                + layout.scale_bytes,
            ] = scale_values.repeat(layout.scale_bytes // 4)
    return raw


def fixture(root, device):
    from sglang_remote_attn import dsv41_remote as remote

    layout = load(
        root,
        "python/sglang/kernels/ops/attention/dsv4/kv_layout.py",
        "dsv41_probe_layout",
    ).KVLayout
    oracle = definitions(
        root, "python/sglang/test/kernels/deepseek_v4/dsv41_kv_quant_reference.py"
    )
    methods = {
        "get_extra_key_buffer",
        "get_extra_key_page_size",
        "get_extra_key_layout",
        "source_layer_of",
        "get_swa_key_buffer_radix",
        "get_swa_key_page_size",
    }
    pool_cls = definitions(
        root,
        "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py",
        owner="DeepSeekV4TokenToKVPool",
        members=methods,
    )
    metadata_cls = definitions(
        root,
        "python/sglang/srt/layers/attention/deepseek_v4_backend.py",
        owner="DSV4AttnMetadata",
        members={"sparse_page_indices", "sparse_topk_lengths"},
    )
    page, prefix = 8, 33
    p_pages, d_pages = [3, 6, 2, 4, 1], [7, 2, 5, 1, 6]

    def pool(physical_pages):
        pool = pool_cls()
        pool.page_size = pool.swa_page_size = page
        pool.kv_layout = layout.V41
        pool.sources_by_ratio = {1: [1], 2: [2]}
        pool.compression_ratios = [0, 1, 2, 2]
        pool.request_window = None
        pool.wait_layer_transfer = lambda layer: None
        pool._swa_local_layer_id = lambda layer: layer
        pool.swa_kv_pool = SimpleNamespace(
            page_size=page, get_key_buffer=lambda layer: swa[layer]
        )
        swa = [encoded(layout.V41, 8, page, device) for _ in range(4)]
        pool.kv_pools = {}
        for ratio in (1, 2):
            cpp = page // ratio
            logical = encoded(layout.V41_FP4, len(physical_pages), cpp, device)
            raw = torch.zeros(
                (8, layout.V41_FP4.page_bytes(cpp)), dtype=torch.uint8, device=device
            )
            for i, physical in enumerate(physical_pages):
                raw[physical].copy_(logical[i])
            pool.kv_pools[ratio] = SimpleNamespace(
                kv_buffer=[raw],
                page_size=cpp,
                kv_layout=layout.V41_FP4,
                get_key_buffer=lambda index, raw=raw: raw,
            )
        pool.layer_mapping = [
            None,
            (1, 0, pool.kv_pools[1]),
            (2, 0, pool.kv_pools[2]),
            (2, 0, pool.kv_pools[2]),
        ]
        row = torch.cat(
            [
                torch.arange(p * page, p * page + page, device=device)
                for p in physical_pages
            ]
        ).reshape(1, -1)
        return pool, row, swa

    p_pool, p_row, p_swa = pool(p_pages)
    d_pool, d_row, d_swa = pool(d_pages)
    config = SimpleNamespace(compress_ratios=[0, 1, 2, 2], index_topk=32)
    controller = SimpleNamespace(
        pool=d_pool,
        dsv41_remote=True,
        layers=4,
        hf_config=config,
        remote_targets={},
        targets=[],
        scheduler=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=d_row)
        ),
    )
    core = metadata_cls()
    core.swa_page_indices = torch.tensor(
        [[13, 14, -1]], dtype=torch.int32, device=device
    )
    core.swa_topk_lengths = torch.tensor([2], dtype=torch.int32, device=device)
    # Completed c2 prefix positions 0, 15 go to P; the just-completed position
    # 16 belongs to D's odd-prefix carry. Keep the repeated 15 as native input.
    core.c2_sparse_page_indices = torch.tensor(
        [[d_pages[0] * 4, d_pages[3] * 4 + 3, d_pages[4] * 4, d_pages[3] * 4 + 3, -1]],
        dtype=torch.int32,
        device=device,
    )
    core.c2_sparse_topk_lengths = torch.tensor([4], dtype=torch.int32, device=device)
    core.c1_sparse_page_indices = torch.tensor(
        [[d_pages[0] * 8, d_pages[4] * 8 + 1, -1]], dtype=torch.int32, device=device
    )
    core.c1_sparse_topk_lengths = torch.tensor([2], dtype=torch.int32, device=device)
    stores = []
    backend = SimpleNamespace(
        _dsv41_remote=controller,
        token_to_kv_pool=d_pool,
        forward_metadata=SimpleNamespace(core_attn_metadata=core),
        softmax_scale=512**-0.5,
        store_cache=lambda layer, k, fb: stores.append(layer),
    )
    fb = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        req_pool_indices=torch.tensor([0], device=device),
        req_pool_indices_cpu=[0],
        seq_lens=torch.tensor([34], device=device),
    )
    query = (
        torch.sin(torch.arange(1024, device=device).float())
        .reshape(1, 2, 512)
        .to(torch.bfloat16)
    )
    sink = torch.tensor([-0.5, 0.2], device=device)
    req = SimpleNamespace(
        rid="sparse", bootstrap_room=7, kv=SimpleNamespace(req_pool_idx=0)
    )
    p_controller = SimpleNamespace(
        pool=p_pool,
        hf_config=config,
        layers=4,
        scheduler=SimpleNamespace(
            req_to_token_pool=SimpleNamespace(req_to_token=p_row)
        ),
    )
    return SimpleNamespace(
        remote=remote,
        layout=layout,
        oracle=oracle,
        page=page,
        prefix=prefix,
        p_pages=p_pages,
        d_pages=d_pages,
        p_pool=p_pool,
        d_pool=d_pool,
        p_row=p_row,
        d_row=d_row,
        p_swa=p_swa,
        d_swa=d_swa,
        controller=controller,
        p_controller=p_controller,
        backend=backend,
        core=core,
        fb=fb,
        query=query,
        sink=sink,
        req=req,
        stores=stores,
    )
