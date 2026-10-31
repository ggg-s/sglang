"""First post-PD decode eager Remote sparse MLA; no early admission.

D owns native top-k/indexer, mutable SWA/carry and newly compressed rows. P owns
only an immutable completed-prefix read lease. Native PD must finish first.
"""

import functools
import inspect
import secrets
import time
from dataclasses import dataclass, field

from . import dsv41_sparse_math as sparse
from .pre_layerwise import _poll_success, read_json, write_json


def runtime():
    from . import handoff_control

    return handoff_control


def validate(controller, runner, role):
    if controller.pool.kv_layout.encoding.value != "v41":
        raise ValueError("DSV4.1 Remote eager requires native V4.1 fp8 SWA layout")
    if role == "prefill":
        return
    backends = [
        getattr(runner, "attn_backend", None),
        getattr(runner, "decode_attn_backend", None),
    ]
    backends.extend(getattr(runner, "decode_attn_backend_group", None) or [])
    backends = [backend for backend in backends if backend is not None]
    if not backends or any(
        not getattr(b, "is_dsv41", False) or b.trtllm_attn or b.mega_attn
        for b in backends
    ):
        raise ValueError(
            "DSV4.1 Remote eager requires native flashmla backend, without mega/TRTLLM"
        )
    for backend in backends:
        backend._dsv41_remote = controller
        if getattr(backend, "_dsv41_eager_bound", False):
            continue
        original = backend._forward_attention
        parameters = inspect.signature(original).parameters
        if (
            not {"compress_ratio", "save_kv_cache", "attn_sink", "cos_sin_cache"}
            <= parameters.keys()
        ):
            raise ValueError(
                "native DSV4.1 attention interface lacks sparse eager inputs"
            )

        def native(_backend, *args, _original=original, **kwargs):
            return _original(*args, **kwargs)

        backend._forward_attention = functools.partial(attention, native, backend)
        backend._dsv41_eager_bound = True


def register(source):
    import torch

    controller = source.owner
    req = source.req
    page = controller.pool.page_size
    # Freeze only a compact request-owned page map; no full KV snapshot/gather.
    pages = (
        controller.scheduler.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : source.prefix_len : page
        ]
        .to("cpu")
        .tolist()
    )
    hub = runtime().get_remote_hub(controller.scheduler)
    ctx = hub.registry.register(
        f"dsv41-{source.generation}",
        req,
        req.kv.req_pool_idx,
        source.prefix_len,
        source.prefix_len,
        secrets.token_hex(16),
        on_release=lambda ctx: source.remote_closed.set(),
    )
    ctx.dsv41_source = source
    ctx.dsv41_handle = req.cache_request_handle
    ctx.dsv41_pages = tuple(int(x) // page for x in pages)
    # register is called after native forward/result GPU proof; readiness is
    # still fenced explicitly before any socket worker may read shared storage.
    ctx.dsv41_ready = torch.cuda.Event()
    ctx.dsv41_ready.record(torch.cuda.current_stream())
    source.service = dict(
        host=hub.host,
        port=hub.port,
        token=ctx.token,
        handoff_id=ctx.handoff_id,
        epoch=ctx.owner_epoch,
    )
    write_json(
        source.path / "remote-service.json",
        dict(
            generation=source.generation,
            route=source.service,
            prefix_len=source.prefix_len,
            owners=dict(
                global_prefix="P-readonly",
                indexer_topk="D-native",
                swa="D-native",
                carry="D-native",
            ),
            boundary="first-post-PD-decode",
            at=time.monotonic(),
        ),
    )


def source_ready(source):
    if source.error:
        raise RuntimeError(f"DSV4.1 remote source failed; retaining KV: {source.error}")
    if source.service is None:
        return source.remote_closed.is_set()
    done = read_json(source.path / "remote-done.json")
    if done and done.get("generation") == source.generation:
        hub = runtime().get_remote_hub(source.owner.scheduler)
        ctx = hub.registry.get(source.service["handoff_id"], source.service["epoch"])
        if ctx is not None:
            hub.registry.release(ctx.handoff_id, expected_req=source.req)
    if getattr(source.req, "_handoff_transfer_unsafe", False):
        raise RuntimeError("Remote GPU read fence failed; retaining P cache/KV")
    return source.remote_closed.is_set()


def execute(job, stream, cache):
    import torch

    ctx = job.ctx
    source = getattr(ctx, "dsv41_source", None)
    if source is None or source.service["epoch"] != job.epoch:
        raise ValueError("not a DSV4.1 sparse read lease")
    if (
        source.req.kv.req_pool_idx != ctx.req_pool_idx
        or source.req.cache_request_handle != ctx.dsv41_handle
    ):
        raise RuntimeError("Remote source request attempt/row changed; retaining KV")
    if source.cache_lease is not None:
        source.cache_lease.check()
    if not _poll_success(source.req.disagg_kv_sender):
        raise RuntimeError("Remote read before native PD completion")
    header = job.header
    layer_id, ratio = int(header["layer"]), int(header["ratio"])
    pool = source.owner.pool
    if layer_id < 0 or layer_id >= source.owner.layers or ratio not in (1, 2):
        raise ValueError("invalid Remote sparse layer/ratio")
    if source.owner.hf_config.compress_ratios[layer_id] != ratio:
        raise ValueError("Remote sparse producer/ratio changed")
    positions = header["positions"]
    if len(positions) > source.owner.hf_config.index_topk or any(
        type(p) is not int or not 0 <= p < source.prefix_len // ratio for p in positions
    ):
        raise ValueError("Remote selection exceeds immutable completed prefix")
    if (
        job.q_cpu.ndim != 2
        or not 0 < job.q_cpu.shape[0] <= 512
        or job.q_cpu.shape[1] != 512
    ):
        raise ValueError("Remote query shape exceeds sparse MLA contract")
    if not bool(torch.isfinite(job.q_cpu).all()):
        raise ValueError("nonfinite Remote sparse query")
    device = pool.get_extra_key_buffer(layer_id).device
    with torch.cuda.device(device), torch.cuda.stream(stream):
        stream.wait_event(ctx.dsv41_ready)
        cpp = pool.page_size // ratio
        ids = torch.tensor(
            [ctx.dsv41_pages[p // cpp] * cpp + p % cpp for p in positions],
            device=device,
        )
        keys = sparse.read_slots(
            pool.get_extra_key_buffer(layer_id),
            ids,
            pool.get_extra_key_page_size(layer_id),
            pool.get_extra_key_layout(layer_id),
        )
        query = job.q_cpu.to(device=device)
        out, lse = sparse.partial(query, keys, float(header["scale"]))
        out, lse = out.cpu(), lse.cpu()  # Real GPU completion before reply.
    return (
        dict(o_shape=list(out.shape), o_dtype="float32", lse_shape=list(lse.shape)),
        out,
        lse,
    )


@dataclass(eq=False)
class Target:
    controller: object
    req: object
    path: object
    generation: str
    route: dict
    prefix_len: int
    client: object = None
    calls: int = 0
    closed: bool = False
    touched: set = field(default_factory=set)
    row: int = field(init=False)
    handle: object = field(init=False)

    def __post_init__(self):
        self.row = int(self.req.kv.req_pool_idx)
        self.handle = getattr(self.req, "cache_request_handle", None)

    def check(self):
        if (
            self.req.kv.req_pool_idx != self.row
            or getattr(self.req, "cache_request_handle", None) != self.handle
        ):
            raise RuntimeError(
                "Remote target attempt/row changed; retaining read lease"
            )

    def call(self, query, positions, layer, ratio, scale):
        import torch

        self.check()
        if self.closed:
            raise RuntimeError("Remote sparse target already retired")
        if self.client is None:
            self.client = runtime().TargetClient(self.route["host"], self.route["port"])
            self.client.attach(
                self.route["handoff_id"], self.route["token"], self.route["epoch"]
            )
        if query.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Remote sparse query requires native BF16/FP16")
        cpu = query.detach().to("cpu").contiguous()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        response, out, lse = self.client.call(
            dict(
                handoff_id=self.route["handoff_id"],
                epoch=self.route["epoch"],
                dsv41_sparse=True,
                layer=layer,
                ratio=ratio,
                positions=positions,
                scale=float(scale),
                q_shape=list(cpu.shape),
                q_dtype="bfloat16" if cpu.dtype == torch.bfloat16 else "float16",
                q_bytes=len(raw),
            ),
            raw,
        )
        if (
            not response.get("ok")
            or out.shape != query.shape
            or lse.shape != query.shape[:1]
        ):
            raise RuntimeError("invalid Remote sparse reply; retaining source lease")
        self.calls += 1
        self.touched.add(layer)
        return out.to(query.device), lse.to(query.device)

    def close(self, reason):
        if self.closed:
            return
        if self.client is not None:
            self.client.detach(self.route["handoff_id"])
            self.client.close()
        write_json(
            self.path / "remote-done.json",
            dict(
                generation=self.generation,
                reason=reason,
                remote_calls=self.calls,
                layers=sorted(self.touched),
                at=time.monotonic(),
            ),
        )
        self.closed = True
        if self.controller.remote_targets.get(self.row) is self:
            self.controller.remote_targets.pop(self.row)
        if getattr(self.req, "_dsv41_remote", None) is self:
            self.req._dsv41_remote = None


def commit(original, queue, dr, *, keep_receiver=False):
    controller = getattr(queue.scheduler, "_pre_layerwise", None)
    result = original(queue, dr, keep_receiver=keep_receiver)
    if not getattr(controller, "dsv41_remote", False):
        return result
    offer = getattr(dr, "_pre_layerwise_offer", None)
    if offer is None:
        return result
    data = read_json(offer.path / "remote-service.json")
    if data is not None and data.get("generation") == offer.generation:
        target = Target(
            controller,
            dr.req,
            offer.path,
            offer.generation,
            data["route"],
            int(data["prefix_len"]),
        )
        dr.req._dsv41_remote = target
        controller.remote_targets[int(dr.req.kv.req_pool_idx)] = target
    return result


def release(req, reason="native-release"):
    target = getattr(req, "_dsv41_remote", None)
    if target is not None:
        target.close(reason)
    else:
        offered = getattr(req, "_dsv41_remote_offer", None)
        if offered is not None:
            path, generation = offered
            write_json(
                path / "remote-done.json", dict(generation=generation, reason=reason)
            )


def forward(controller, original, runner, fb, *args, **kwargs):
    # TP membership was established by native PD completion's existing MIN.
    # A complete first decode consumes the same model-layer sequence on ranks.
    if not controller.remote_targets or not fb.forward_mode.is_decode():
        return original(runner, fb, *args, **kwargs)
    rows = (
        [int(x) for x in fb.req_pool_indices_cpu]
        if getattr(fb, "req_pool_indices_cpu", None) is not None
        else fb.req_pool_indices.cpu().tolist()
    )
    active = [
        controller.remote_targets[x] for x in rows if x in controller.remote_targets
    ]
    result = original(runner, fb, *args, **kwargs)
    for target in active:
        target.close("first-post-PD-eager-decode-complete")
    return result


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
    import torch

    controller = getattr(backend, "_dsv41_remote", None)
    if (
        controller is None
        or not controller.remote_targets
        or not forward_batch.forward_mode.is_decode()
    ):
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
    req_rows = forward_batch.req_pool_indices.cpu().tolist()
    native_rows = [int(rpi) not in controller.remote_targets for rpi in req_rows]
    if not compress_ratio or all(native_rows):
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
    # In a mixed batch retain the native outputs for ordinary requests. Its
    # cache store runs once; only Remote rows replace their attention output.
    native_output = None
    if any(native_rows):
        native_output = original(
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
        raise ValueError("Remote eager requires native pre-rotated shared KV and sink")
    pool = backend.token_to_kv_pool
    core = backend.forward_metadata.core_attn_metadata
    if save_kv_cache and native_output is None:
        backend.store_cache(layer.layer_id, k, forward_batch)
    if q.ndim not in (3, 4) or q.shape[-1] != 512:
        raise ValueError("Remote sparse query layout changed")
    queries = q.reshape(-1, q.shape[-2], 512)
    seq_lens = forward_batch.seq_lens.cpu().tolist()
    if queries.shape[0] != len(req_rows):
        raise ValueError("Remote sparse query/request rows differ")
    swa_ids = core.swa_page_indices.reshape(len(req_rows), -1).cpu().tolist()
    swa_lens = core.swa_topk_lengths.reshape(-1).cpu().tolist()
    extra_ids = (
        core.sparse_page_indices(compress_ratio)
        .reshape(len(req_rows), -1)
        .cpu()
        .tolist()
        if compress_ratio
        else [[] for _ in req_rows]
    )
    extra_lens = (
        core.sparse_topk_lengths(compress_ratio).reshape(-1).cpu().tolist()
        if compress_ratio
        else [0] * len(req_rows)
    )
    outputs = []
    for row, rpi in enumerate(req_rows):
        if native_rows[row]:
            outputs.append(native_output[row])
            continue
        target = controller.remote_targets.get(int(rpi))
        selected = [i for i in extra_ids[row][: extra_lens[row]] if i >= 0]
        local_ids, remote = selected, []
        if target is not None and compress_ratio:
            pages = (
                controller.scheduler.req_to_token_pool.req_to_token[
                    int(rpi), : seq_lens[row] : pool.page_size
                ]
                .cpu()
                .tolist()
            )
            pages = [int(x) // pool.page_size for x in pages]
            remote, local_ids = sparse.partition(
                selected, pages, target.prefix_len, pool.page_size, compress_ratio
            )
        swa = torch.tensor(
            [i for i in swa_ids[row][: swa_lens[row]] if i >= 0], device=q.device
        )
        local_keys = sparse.read_slots(
            pool.get_swa_key_buffer_radix(layer.layer_id),
            swa,
            pool.get_swa_key_page_size(),
            pool.kv_layout,
        )
        if local_ids:
            tail = sparse.read_slots(
                pool.get_extra_key_buffer(layer.layer_id),
                torch.tensor(local_ids, device=q.device),
                pool.get_extra_key_page_size(layer.layer_id),
                pool.get_extra_key_layout(layer.layer_id),
            )
            local_keys = torch.cat((local_keys, tail))
        local = sparse.partial(
            queries[row],
            local_keys,
            backend.softmax_scale,
            attn_sink[: queries.shape[1]],
        )
        if remote:
            remote_result = target.call(
                queries[row],
                remote,
                layer.layer_id,
                compress_ratio,
                backend.softmax_scale,
            )
            out = sparse.merge(local, remote_result)
        else:
            out = local[0]
        outputs.append(out.to(q.dtype))
    return torch.stack(outputs)
