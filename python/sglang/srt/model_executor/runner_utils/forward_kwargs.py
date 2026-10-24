"""Model inputs shared by eager extend and prefill CUDA graphs."""


def build_extend_forward_kwargs(
    forward_batch, *, pp_kwargs, model, model_dtype, is_generation
) -> dict:
    kwargs = pp_kwargs
    if forward_batch.input_embeds is not None:
        # DFlash supplies the target embedding to its draft runner. Its dtype
        # must match that runner's weights, including FP16 GPTQ deployments.
        kwargs["input_embeds"] = forward_batch.input_embeds.to(dtype=model_dtype)
    if (
        forward_batch.replace_embeds is not None
        and forward_batch.replace_positions is not None
    ):
        if "input_embeds" not in kwargs:
            kwargs["input_embeds"] = model.get_input_embeddings()(
                forward_batch.input_ids
            )
        kwargs["input_embeds"][forward_batch.replace_positions] = (
            forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
        )
    if not is_generation:
        kwargs["get_embedding"] = True
    return kwargs
