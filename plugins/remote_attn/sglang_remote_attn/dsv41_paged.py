"""Paged sparse GPU math and attempt-owned incremental page maps.

CPU execution is an independent Torch reference used by lifecycle tests. CUDA
kernels return checked status flags; bounds failures never become cache reads.
"""

import math

import torch

from . import dsv41_sparse_math as reference


def settings(cache, page, layout):
    raw = cache.view(torch.uint8)
    encoding = {"v4": 0, "v41": 1, "v41_fp4": 2}.get(layout.encoding.value)
    if (
        encoding is None
        or page <= 0
        or raw.ndim != 2
        or not raw.is_contiguous()
        or raw.shape[1] < layout.page_bytes(page)
    ):
        raise ValueError("unsupported sparse cache layout/stride")
    data, base, scale = layout.row_addressing(page)
    return raw, (page, raw.shape[0], raw.stride(0), data, base, scale, encoding)


def partial(query, sources, scale, sink=None):
    if query.ndim != 2 or query.shape[-1] != 512 or not 0 < query.shape[0] <= 512:
        raise ValueError("sparse query must be H x 512")
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("invalid sparse query dtype/scale")
    if not 1 <= len(sources) <= 2:
        raise ValueError("one or two sparse sources required")
    if sink is not None and (
        sink.numel() != query.shape[0] or sink.device != query.device
    ):
        raise ValueError("sink must cover exactly the query heads")
    if not query.is_cuda:
        keys = [
            reference.read_slots(cache, ids[ids >= 0], page, layout)
            for cache, ids, page, layout in sources
        ]
        out, lse = reference.partial(query, torch.cat(keys), scale, sink)
        return out, lse, torch.zeros(1, dtype=torch.int32)
    from . import dsv41_kernels as kernels

    args = []
    pointers = []
    for cache, ids, page, layout in sources:
        if cache.device != query.device or ids.device != query.device:
            raise ValueError("sparse pointers belong to different devices")
        raw, config = settings(cache, page, layout)
        ids = ids.reshape(-1).contiguous()
        pointers.extend((raw, ids))
        args.extend((ids.numel(), *config))
    if len(sources) == 1:
        pointers.extend(pointers[:2])
        args.extend((0, *args[1:8]))
    query = query.contiguous()
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    lse = torch.empty(query.shape[0], dtype=torch.float32, device=query.device)
    error = torch.zeros(1, dtype=torch.int32, device=query.device)
    kernels._partial[(kernels.tr.cdiv(query.shape[0], 16),)](
        query,
        *pointers,
        sink if sink is not None else lse,
        out,
        lse,
        error,
        query.shape[0],
        query.stride(0),
        float(scale),
        *args,
        sink is not None,
        16,
        16,
        num_warps=4,
        num_stages=1,
    )
    return out, lse, error


def check(errors):
    if errors and bool(torch.cat(errors).any().item()):
        raise RuntimeError("invalid sparse GPU index/query/cache payload")


def merge(left, right, dtype):
    a, al = left
    b, bl = right
    if not a.is_cuda:
        return reference.merge(left, right).to(dtype)
    from . import dsv41_kernels as kernels

    if a.shape != b.shape or al.shape != bl.shape or a.ndim != 2:
        raise ValueError("partial attention shapes differ")
    out = torch.empty(a.shape, dtype=dtype, device=a.device)
    kernels._merge[(kernels.tr.cdiv(out.numel(), 256),)](
        a.contiguous(),
        al.contiguous(),
        b.contiguous(),
        bl.contiguous(),
        out,
        a.shape[0],
        a.shape[1],
        256,
    )
    return out


def masked(ids, lengths):
    return torch.where(
        torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths.reshape(-1, 1),
        ids,
        -1,
    )


class PageMap:
    def __init__(self, target, table, page, capacity):
        self.target, self.table, self.page = target, table, page
        self.handle, self.row = target.handle, target.row
        self.inverse = torch.full(
            (capacity,), -1, dtype=torch.int32, device=table.device
        )
        self.error = torch.zeros(1, dtype=torch.int32, device=table.device)
        self.pages = 0

    def extend(self, length):
        self.target.check()
        count = (length + self.page - 1) // self.page
        if (
            self.target.handle != self.handle
            or self.target.row != self.row
            or count < self.pages
            or length > self.table.shape[1]
        ):
            raise RuntimeError("sparse page map attempt/extent changed")
        if count == self.pages:
            return
        row = self.table[self.row]
        if row.is_cuda:
            from . import dsv41_kernels as kernels

            kernels._map[(kernels.tr.cdiv(count - self.pages, 256),)](
                row,
                self.inverse,
                self.error,
                self.pages,
                count,
                self.page,
                self.inverse.numel(),
                256,
            )
        else:
            for i in range(self.pages, count):
                slot = int(row[i * self.page])
                physical = slot // self.page
                if (
                    slot < 0
                    or slot % self.page
                    or physical >= self.inverse.numel()
                    or self.inverse[physical] != -1
                ):
                    raise RuntimeError("invalid/aliased sparse page map")
                self.inverse[physical] = i
        self.pages = count

    def partition(self, ids, length, ratio, local, remote):
        cpp = self.page // ratio
        if ratio not in (1, 2, 4, 128) or self.page % ratio:
            raise ValueError("unsupported sparse ownership geometry")
        lo, hi = self.target.decode_prefix_len // ratio, self.target.prefix_len // ratio
        if ids.is_cuda:
            from . import dsv41_kernels as kernels

            kernels._partition[(kernels.tr.cdiv(ids.numel(), 256),)](
                ids,
                length,
                self.inverse,
                self.error,
                local,
                remote,
                ids.numel(),
                cpp,
                self.inverse.numel(),
                lo,
                hi,
                256,
            )
        else:
            local.fill_(-1)
            remote.fill_(-1)
            n = int(length)
            if not 0 <= n <= ids.numel():
                raise ValueError("invalid sparse length")
            for i, slot in enumerate(ids[:n].tolist()):
                if slot < 0:
                    continue
                physical, offset = divmod(slot, cpp)
                if physical >= self.inverse.numel() or self.inverse[physical] < 0:
                    raise ValueError("top-k slot outside request")
                pos = int(self.inverse[physical]) * cpp + offset
                (remote if lo <= pos < hi else local)[i] = (
                    pos if lo <= pos < hi else slot
                )


def page_map(target, table, page, capacity, length, *, ratio=None):
    if ratio not in (None, 1, 2, 4, 128):
        raise ValueError("unsupported page-map ratio")
    key = (id(table), page, ratio)
    maps = getattr(target, "_sparse_page_maps", None)
    if maps is None:
        maps = target._sparse_page_maps = {}
    result = maps.get(key)
    if result is None:
        if any(k[:2] != key[:2] for k in maps):
            raise RuntimeError("sparse page-map storage changed")
        # Native C4/C128 pools have different guard-page capacities. Keep a
        # request-owned inverse per ratio, never reinterpret a changed pool
        # for the same ratio or change the request's FULL table identity.
        result = maps[key] = PageMap(target, table, page, capacity)
    elif result.inverse.numel() != capacity:
        raise RuntimeError("sparse page-map storage changed")
    result.extend(length)
    return result


def warmup(pool, layers, heads, dtype, role):
    """Compile against real pool geometry with masked IDs; never read/write KV."""
    seen = set()
    for layer in layers:
        cache = pool.get_extra_key_buffer(layer)
        page = pool.get_extra_key_page_size(layer)
        layout = pool.get_extra_key_layout(layer)
        shape = (tuple(cache.shape), page, layout.value)
        if shape in seen:
            continue
        seen.add(shape)
        ids = torch.full((32,), -1, dtype=torch.int64, device=cache.device)
        for h in heads:
            q = torch.zeros((h, 512), dtype=dtype, device=cache.device)
            sink = torch.zeros(h, device=cache.device)
            sources = [(cache, ids, page, layout)]
            if role == "decode":
                sources.insert(
                    0,
                    (
                        pool.get_swa_key_buffer_radix(layer),
                        ids,
                        pool.get_swa_key_page_size(),
                        pool.kv_layout,
                    ),
                )
            out, lse, error = partial(
                q, sources, 512**-0.5, sink if role == "decode" else None
            )
            check([error])
            if role == "decode":
                merge((out, lse), (out, lse), dtype)
        if role == "decode":
            from types import SimpleNamespace

            for index_dtype in (torch.int32, torch.int64):
                table = torch.arange(
                    pool.page_size, dtype=index_dtype, device=cache.device
                ).reshape(1, -1)
                target = SimpleNamespace(
                    row=0,
                    handle="warmup",
                    check=lambda: None,
                    prefix_len=pool.page_size,
                    decode_prefix_len=0,
                )
                mapping = PageMap(target, table, pool.page_size, cache.shape[0])
                mapping.extend(pool.page_size)
                selected = torch.zeros(32, dtype=index_dtype, device=cache.device)
                length = torch.tensor([32], dtype=torch.int32, device=cache.device)
                mapping.partition(
                    selected,
                    length,
                    pool.page_size // page,
                    torch.empty_like(selected),
                    torch.empty_like(selected),
                )
                check([mapping.error])
    if seen:
        torch.cuda.current_stream(cache.device).synchronize()
