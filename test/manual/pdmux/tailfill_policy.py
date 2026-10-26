"""Token-budget policy for filling native Prefill tails with whole Verify blocks."""

import copy


def decode_slots(prefill_tokens, chunk_size, max_prefill_tokens, running, block_size):
    if not chunk_size or chunk_size <= 0 or prefill_tokens <= 0 or block_size <= 0:
        return 0
    remaining = max(0, min(chunk_size, max_prefill_tokens) - prefill_tokens)
    return min(running, remaining // block_size)


def split_running(batch, count):
    """Own mutable sampler/spec metadata before filtering a decode subset."""
    if count >= batch.batch_size():
        return batch, None
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    def subset(indices):
        selected = copy.copy(batch)
        selected.spec_info = copy.copy(batch.spec_info)
        selected.sampling_info = SamplingBatchInfo.from_schedule_batch(
            selected, batch.sampling_info.vocab_size
        )
        selected.filter_batch(keep_indices=indices)
        return selected

    return subset(list(range(count))), subset(list(range(count, batch.batch_size())))
