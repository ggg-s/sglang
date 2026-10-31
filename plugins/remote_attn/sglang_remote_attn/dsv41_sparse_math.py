"""Eager sparse MLA reference operations over native V4.1 paged bytes.

Only selected slots are decoded. Top-k/candidate selection remains native on D;
these operations never independently select, sort, deduplicate or truncate it.
"""

import math


def read_slots(cache, slots, page_size, layout):
    import torch

    encoding = layout.encoding.value
    if encoding not in ("v41", "v41_fp4"):
        raise ValueError("Remote eager requires V4.1 fp8/fp4 cache encoding")
    ids = slots.to(device=cache.device, dtype=torch.int64).reshape(-1)
    raw = cache.view(torch.uint8)
    if ids.numel() == 0:
        return torch.empty((0, 512), dtype=torch.bfloat16, device=cache.device)
    if bool(((ids < 0) | (ids >= raw.shape[0] * page_size)).any()):
        raise ValueError("sparse cache slot outside native pool")
    data_stride, scale_base, scale_stride = layout.row_addressing(page_size)
    pages, rows = ids // page_size, ids % page_size
    data = raw[
        pages[:, None],
        rows[:, None] * data_stride
        + torch.arange(layout.data_bytes, device=raw.device),
    ].contiguous()
    scales = raw[
        pages[:, None],
        scale_base
        + rows[:, None] * scale_stride
        + torch.arange(layout.scale_bytes, device=raw.device),
    ].contiguous()
    if encoding == "v41":
        values = data.view(torch.float8_e4m3fn).float()
        scales = torch.exp2(scales.float() - 127)
    else:
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            device=raw.device,
        )
        nibbles = torch.stack((data & 15, data >> 4), dim=-1).reshape(-1, 512)
        values = lut[nibbles.long()]
        scales = scales.view(torch.float8_e4m3fn).float()
    values = values * scales.repeat_interleave(layout.tile_size, dim=-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite sparse cache payload/scale")
    return values.to(torch.bfloat16)


def partial(query, keys, scale, sink=None):
    """One query, H heads; natural-log LSE, including a zero-value sink once."""
    import torch

    if query.ndim != 2 or query.shape[-1] != 512 or keys.shape[-1] != 512:
        raise ValueError("DSV4.1 eager requires H x 512 query and N x 512 shared KV")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid sparse attention scale")
    logits = query.float() @ keys.float().T * scale
    if sink is not None:
        logits = torch.cat((logits, sink.float().reshape(-1, 1)), dim=1)
        keys = torch.cat((keys, keys.new_zeros((1, 512))), dim=0)
    lse = torch.logsumexp(logits, dim=-1)
    weights = torch.exp(logits - lse[:, None])
    return weights @ keys.float(), lse


def merge(left, right):
    import torch

    a, al = left
    b, bl = right
    total = torch.logaddexp(al, bl)
    wa = torch.where(torch.isfinite(al), torch.exp(al - total), 0.0)
    wb = torch.where(torch.isfinite(bl), torch.exp(bl - total), 0.0)
    return a * wa[:, None] + b * wb[:, None]


def partition(selected, full_pages, prefix_len, full_page_size, ratio):
    """Native D physical compressed slots -> P logical positions / D suffix.

    A ratio-2 pending half-pair belongs to D, even if its raw first token was
    prefetched. Shared producer layers use this same position/ownership map.
    """
    if ratio not in (1, 2) or full_page_size % ratio:
        raise ValueError("unsupported sparse ownership ratio/page")
    cpp = full_page_size // ratio
    inverse = {int(page): logical for logical, page in enumerate(full_pages)}
    if len(inverse) != len(full_pages):
        raise ValueError("aliased request page map")
    remote, local = [], []
    for slot in selected:
        slot = int(slot)
        if slot < 0:
            continue
        page, offset = divmod(slot, cpp)
        if page not in inverse:
            raise ValueError("top-k slot does not belong to this request")
        position = inverse[page] * cpp + offset
        if position < prefix_len // ratio:
            remote.append(position)
        else:
            local.append(slot)
    return remote, local
