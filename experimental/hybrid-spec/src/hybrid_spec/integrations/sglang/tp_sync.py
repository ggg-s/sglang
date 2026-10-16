"""Consensus on retrieval overrides, including device lookup inside graphs."""

from contextlib import nullcontext

import torch
import torch.distributed as dist

from hybrid_spec.core.contracts import DraftOverrideView


class TPOverride(DraftOverrideView):
    """Consensus at the draft consumer, including captured CPU-history views."""

    def __init__(self, tokens, use_retrieval, tp_sync):
        super().__init__(tokens, use_retrieval)
        self.tp_sync = tp_sync

    def prepare(self, seq_lens):
        agreed = self.tp_sync.synchronize(self)
        # Capture/replay and metrics must retain the same storage addresses.
        self.tokens.copy_(agreed.tokens)
        self.use_retrieval.copy_(agreed.use_retrieval)


class TPSynchronizer:
    def __init__(self, group, communicator=None):
        self.group = group
        self.world_size = dist.get_world_size(group)
        self.communicator = communicator
        if communicator is not None and not communicator.available:
            raise ValueError("hybrid-spec TP requires an available PyNccl communicator")

    def synchronize(self, view):
        if self.world_size == 1 or view.tokens.shape[0] == 0:
            return view
        # A single MIN reduction carries token minima, complemented maxima,
        # and eligibility. Bitwise NOT reverses signed-int64 ordering without
        # overflow: min(t) == ~min(~t) iff every rank proposed the same token.
        # This is exact, including invalid int64 values; no hash collisions.
        # SGLang guarantees the same ordered batch and K across this TP group.
        k = view.tokens.shape[1]
        packet = torch.cat(
            (
                view.tokens,
                ~view.tokens,
                view.use_retrieval[:, None].to(torch.int64),
            ),
            dim=1,
        )
        comm = self.communicator
        # PyNccl normally disables itself outside capture. Explicitly enable it
        # for eager too, retaining the same communicator and collective ordering.
        with nullcontext() if comm is None else comm.change_state(enable=True):
            # No local-mask branch, even for empty/padded/ineligible SA slots.
            if comm is None:
                dist.all_reduce(packet, op=dist.ReduceOp.MIN, group=self.group)
            else:
                comm.all_reduce(packet, op=dist.ReduceOp.MIN)
        tokens = packet[:, :k].contiguous()
        agree = packet[:, -1].bool() & (tokens == ~packet[:, k : 2 * k]).all(dim=1)
        return DraftOverrideView(tokens, agree)
