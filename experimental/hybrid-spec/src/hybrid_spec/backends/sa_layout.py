"""SA storage selection; semantic unpacking is for CPU inspection/tests only."""

import torch

COMPACT_MAX_HISTORY = 32767
SA_STORAGE_REVISION = 6


def state_layout(capacity):
    return (torch.uint16, 11) if capacity <= COMPACT_MAX_HISTORY else (torch.int32, 10)


def unpack_states(states):
    if states.dtype == torch.int32:
        return states
    if states.dtype != torch.uint16 or states.shape[-1] != 11:
        raise ValueError("invalid compact SA state storage")
    values = states.to(torch.int32)
    secondary = values[..., 4] | (values[..., 5] << 16)
    return torch.cat(
        (values[..., :4] - 1, secondary[..., None], values[..., 6:] - 1), dim=-1
    )
