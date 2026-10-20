"""GPU-only online C128 plans and rollback banks for compact verify.

Plan rows use the existing 16-byte CompressPlan ABI. Unused rows are invalid,
so the physical graph capacity stays fixed without reading a device count.
"""

import torch


def plan_online_verify(seq_lens, verify_lens, req_indices, capacity, state_offset):
    if capacity > 65536:
        raise ValueError("online C128 CompressPlan ragged_id must fit uint16")
    plans = torch.full((2, capacity, 4), -1, dtype=torch.int32, device=seq_lens.device)
    starts = verify_lens.to(torch.int64).cumsum(0) - verify_lens
    if seq_lens.is_cuda:
        import triton

        from .online_c128_ragged_triton import _plan_kernel

        _plan_kernel[(len(verify_lens),)](
            seq_lens,
            verify_lens,
            req_indices,
            starts,
            plans,
            capacity,
            state_offset,
            BLOCK=triton.next_power_of_2(capacity),
        )
    else:
        for row, length in enumerate(verify_lens.tolist()):
            req = int(req_indices[row])
            if req < 0:
                continue
            prefix = int(seq_lens[row])
            for step in range(length):
                end = prefix + step + 1
                close = end % 128 == 0
                if not close and step + 1 != length:
                    continue
                token = int(starts[row]) + step
                size = end - max(prefix, ((end - 1) // 128) * 128)
                plans[0 if close else 1, token] = torch.tensor(
                    [end, token | (size << 16), req + state_offset, req],
                    dtype=torch.int32,
                )
    raw = plans.view(torch.uint8)
    return raw[0], raw[1]


def write_ragged_prefix_states(
    kv_score, seq_lens, req_indices, verify_lens, ape, state, state_stride
):
    starts = verify_lens.to(torch.int64).cumsum(0) - verify_lens
    head_dim = ape.shape[1]
    if kv_score.is_cuda:
        import triton

        from .online_c128_ragged_triton import _write_kernel

        _write_kernel[(len(verify_lens),)](
            kv_score,
            seq_lens,
            req_indices,
            verify_lens,
            starts,
            ape,
            state,
            kv_score.stride(0),
            state.stride(0),
            state_stride,
            D=head_dim,
            BLOCK=triton.next_power_of_2(head_dim),
        )
        return
    for row, length in enumerate(verify_lens.tolist()):
        req, prefix = int(req_indices[row]), int(seq_lens[row])
        if req < 0 or not length:
            continue
        old = state[req].float().reshape(3, head_dim)
        maximum, total, value = old[0].clone(), old[1].clone(), old[2].clone()
        for step in range(length):
            token = int(starts[row]) + step
            pos = (prefix + step) % 128
            kv = kv_score[token, :head_dim]
            score = kv_score[token, head_dim:] + ape[pos]
            if pos == 0:
                maximum, total, value = score, torch.ones_like(score), kv
            else:
                new_max = torch.maximum(maximum, score)
                old_sum = total * (maximum - new_max).exp()
                new_sum = (score - new_max).exp()
                total = old_sum + new_sum
                value = (value * old_sum + kv * new_sum) / total
                maximum = new_max
            if (prefix + step + 1) % 128:
                state[req + (step + 1) * state_stride].copy_(
                    torch.cat((maximum, total, value))
                )


def commit_ragged_pending(seq_lens, req_indices, pending, state, width, state_stride):
    if seq_lens.is_cuda:
        import triton

        from .online_c128_ragged_triton import _commit_kernel

        _commit_kernel[(len(seq_lens),)](
            seq_lens,
            req_indices,
            pending,
            state,
            pending.numel(),
            width,
            state.stride(0),
            state_stride,
            D=state.shape[1],
            BLOCK=triton.next_power_of_2(state.shape[1]),
        )
        return
    for row, prefix in enumerate(seq_lens.tolist()):
        req = int(req_indices[row])
        if not 0 <= req < pending.numel():
            continue
        old = int(pending[req])
        count = max(0, min(prefix - old, width))
        if old >= 0 and count and (old + count) % 128:
            state[req].copy_(state[req + count * state_stride])
