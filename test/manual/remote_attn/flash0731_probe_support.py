"""Native paged-byte and metadata fixtures for C4/C128 Flash Remote."""

from types import SimpleNamespace as NS
import math
import torch
from dsv41_remote_probe_support import fixture, encoded


def flash_fixture(root, device, ratio, prefix=None):
    f = fixture(root, device, kv_layout="v4")
    page = 256
    prefix = prefix if prefix is not None else (66687 if ratio == 128 else 1031)
    pages = math.ceil((prefix + ratio) / page)
    # Native guard is one FULL logical page worth of compressed slots.
    # In compressed physical pages this adds `ratio` pages, not one.
    capacity = pages + ratio
    p_pages = list(range(1, pages + 1))
    d_pages = list(reversed(p_pages))
    cpp = page // ratio
    logical = encoded(f.layout.V4, pages, cpp, device)
    for pool, physical, controller in (
        (f.p_pool, p_pages, f.p_controller),
        (f.d_pool, d_pages, f.controller),
    ):
        pool.page_size = page
        pool.compression_ratios = [0, ratio, ratio, ratio]
        pool.sources_by_ratio = {ratio: [1, 2, 3]}
        pool.kv_pools = {}
        raw = torch.zeros(
            (capacity, f.layout.V4.page_bytes(cpp)), device=device, dtype=torch.uint8
        )
        for i, p in enumerate(physical):
            raw[p].copy_(logical[i])
        compressed = NS(
            kv_buffer=[raw, raw.clone(), raw.clone()],
            page_size=cpp,
            kv_layout=f.layout.V4,
        )
        compressed.get_key_buffer = compressed.kv_buffer.__getitem__
        pool.kv_pools[ratio] = compressed
        pool.layer_mapping = [None] + [(ratio, i, compressed) for i in range(3)]
        row = torch.cat(
            [torch.arange(p * page, (p + 1) * page, device=device) for p in physical]
        ).reshape(1, -1)
        controller.scheduler.req_to_token_pool.req_to_token = row
        controller.hf_config = NS(
            model_type="deepseek_v4",
            compress_ratios=pool.compression_ratios,
            index_topk=512,
        )
        if controller is f.p_controller:
            f.p_row = row
        else:
            f.d_row = row
    # Native c128 attends ALL completed blocks; C4 remains top-k selected.
    count = (prefix + ratio) // ratio
    positions = (
        list(range(count))
        if ratio == 128
        else [0, prefix // ratio - 1, prefix // ratio - 1, prefix // ratio]
    )
    ids = [d_pages[p // cpp] * cpp + p % cpp for p in positions]
    if ratio == 128:
        f.core.c128_page_indices = torch.tensor(
            [ids + [-1]], dtype=torch.int32, device=device
        )
        f.core.c128_topk_lengths_clamp1 = torch.tensor(
            [len(ids)], dtype=torch.int32, device=device
        )
    else:
        f.core.c4_sparse_page_indices = torch.tensor(
            [ids + [-1]], dtype=torch.int32, device=device
        )
        f.core.c4_sparse_topk_lengths = torch.tensor(
            [len(ids)], dtype=torch.int32, device=device
        )
    f.fb.seq_lens = torch.tensor([prefix + ratio], device=device)
    f.fb.seq_lens_cpu = [prefix + ratio]
    f.prefix, f.page, f.extra_layout = prefix, page, f.layout.V4
    f.positions = positions
    f.backend.is_dsv41 = False
    return f
