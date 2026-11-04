"""DSV decode: per-layer native routing, GPU indices, batched asynchronous RPC."""

import time

import torch

from . import dsv41_paged as paged
from . import dsv41_rpc as rpc


def host_values(batch, name, rows):
    value = getattr(batch, name + "_cpu", None)
    if value is None:
        value = getattr(batch, name).cpu()
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            raise ValueError("CPU metadata is not on CPU")
        value = value.tolist()
    if len(value) != rows:
        raise ValueError("live CPU metadata differs from decode batch")
    return value


def attention(
    original,
    backend,
    q,
    k,
    v,
    layer,
    forward_batch,
    compress_ratio,
    save_kv_cache=True,
    attn_sink=None,
    cos_sin_cache=None,
    **kwargs,
):
    from .dsv41_remote import resource_local

    controller = backend._dsv41_remote
    rows = host_values(forward_batch, "req_pool_indices", q.shape[0])
    targets = [controller.remote_targets.get(int(row)) for row in rows]
    pool = backend.token_to_kv_pool
    local = []
    for target in targets:
        if target is not None:
            target.check()
            if target.closed:
                raise RuntimeError("closed target in live decode")
        local.append(
            target is None
            or not compress_ratio
            or resource_local(target, pool, layer.layer_id, compress_ratio)
        )
    stats = getattr(controller, "sparse_stats", None)
    if stats is None:
        stats = controller.sparse_stats = dict(
            native_layers=0, sparse_layers=0, index_readbacks=0, rpc_wait_ns=0
        )
    if all(local):
        # Only TP-committed local_entries enables this branch, never raw DMA.
        stats["native_layers"] += 1
        return original(
            backend,
            q,
            k,
            v,
            layer,
            forward_batch,
            compress_ratio,
            save_kv_cache=save_kv_cache,
            attn_sink=attn_sink,
            cos_sin_cache=cos_sin_cache,
            **kwargs,
        )
    if cos_sin_cache is not None or attn_sink is None or k is not v:
        raise ValueError("sparse decode requires pre-rotated shared KV and sink")
    if q.ndim not in (3, 4) or q.shape[-1] != 512:
        raise ValueError("invalid sparse query shape")
    stats["sparse_layers"] += 1
    if save_kv_cache:
        backend.store_cache(layer.layer_id, k, forward_batch)
    queries = q.reshape(len(rows), -1, 512)
    core = backend.forward_metadata.core_attn_metadata
    ids = core.sparse_page_indices(compress_ratio).reshape(len(rows), -1)
    lengths = core.sparse_topk_lengths(compress_ratio).reshape(-1)
    local_ids = paged.masked(ids, lengths)
    remote_ids = torch.full_like(ids, -1)
    seqs = host_values(forward_batch, "seq_lens", len(rows))
    errors = []
    for i, (target, is_local) in enumerate(zip(targets, local)):
        if is_local:
            continue
        mapping = paged.page_map(
            target,
            controller.scheduler.req_to_token_pool.req_to_token,
            pool.page_size,
            pool.get_extra_key_buffer(layer.layer_id).shape[0],
            int(seqs[i]),
        )
        mapping.partition(
            ids[i], lengths[i : i + 1], compress_ratio, local_ids[i], remote_ids[i]
        )
        errors.append(mapping.error)
    # One compact readback per layer for control headers + validation status.
    # SWA, local top-k, physical page maps and their lengths stay on the GPU.
    packed = torch.cat(
        (remote_ids.reshape(-1).to(torch.int64), torch.cat(errors).to(torch.int64))
    )
    host = packed.cpu().tolist()
    stats["index_readbacks"] += 1
    n = remote_ids.numel()
    if any(host[n:]):
        raise RuntimeError("invalid sparse page ownership")
    pending_rows = []
    for i, target in enumerate(targets):
        selected = [
            x for x in host[i * ids.shape[1] : (i + 1) * ids.shape[1]] if x >= 0
        ]
        if selected:
            pending_rows.append(
                rpc.Row(
                    i,
                    target,
                    queries[i],
                    selected,
                    layer.layer_id,
                    compress_ratio,
                    backend.softmax_scale,
                )
            )
    jobs = []
    try:
        # All Q producer events are recorded before local partial launches.
        for group in rpc.groups(pending_rows):
            jobs.append(rpc.pool(controller).submit(group))
        swa = paged.masked(
            core.swa_page_indices.reshape(len(rows), -1), core.swa_topk_lengths
        )
        partials, checks = [], []
        for i in range(len(rows)):
            out, lse, error = paged.partial(
                queries[i],
                [
                    (
                        pool.get_swa_key_buffer_radix(layer.layer_id),
                        swa[i],
                        pool.get_swa_key_page_size(),
                        pool.kv_layout,
                    ),
                    (
                        pool.get_extra_key_buffer(layer.layer_id),
                        local_ids[i],
                        pool.get_extra_key_page_size(layer.layer_id),
                        pool.get_extra_key_layout(layer.layer_id),
                    ),
                ],
                backend.softmax_scale,
                attn_sink[: queries.shape[1]],
            )
            partials.append((out, lse))
            checks.append(error)
        started = time.perf_counter_ns()
        remote_results = dict(item for job in jobs for item in job.result())
        stats["rpc_wait_ns"] += time.perf_counter_ns() - started
        paged.check(checks)
        return torch.stack(
            [
                paged.merge(partials[i], remote_results[i], q.dtype)
                if i in remote_results
                else partials[i][0].to(q.dtype)
                for i in range(len(rows))
            ]
        )
    except BaseException:
        # A failed local launch cannot silently retire queued remote work.
        from .remote_data_plane import io_for

        for row in pending_rows:
            io_for(row.target.req).poison()
        raise
