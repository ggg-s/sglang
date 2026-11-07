"""Eager layer-group continuation used by remote-attn prefill transfers."""

from sglang.srt.utils.device_timer import device_timer_ctx


def forward_split_prefill(
    *,
    model,
    attn_backend,
    device_timer,
    forward_batch,
    reinit_attn_backend=False,
    forward_count=1,
):
    start = forward_batch.split_index
    total = model.config.num_hidden_layers
    if (
        isinstance(forward_count, bool)
        or not isinstance(forward_count, int)
        or forward_count <= 0
    ):
        raise ValueError("split prefill requires a positive layer count")
    if not 0 <= start < total:
        raise ValueError("split prefill continuation is outside the model")
    end = min(start + forward_count, total)
    if start == 0 or reinit_attn_backend:
        attn_backend.init_forward_metadata(forward_batch)
    with device_timer_ctx(device_timer, "split_prefill"):
        result = model.forward_split_prefill(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            (start, end),
        )
    # Publish continuation only after a successful group; the caller records
    # the producer event on this same stream before enqueueing its KV transfer.
    forward_batch.split_index = end
    return result
