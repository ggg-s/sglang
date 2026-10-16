"""Consensus on retrieval overrides, including device lookup inside graphs."""

from contextlib import nullcontext

import torch
import torch.distributed as dist

from hybrid_spec.core.contracts import DraftOverrideView


class TPSynchronizer:
    def __init__(self, group, source_rank, communicator=None):
        self.group = group
        # torch.distributed expects a global rank, including for subgroups.
        self.source_rank = source_rank
        self.world_size = dist.get_world_size(group)
        self.communicator = communicator
        if communicator is not None:
            if not communicator.available:
                raise ValueError(
                    "device history TP requires an available PyNccl communicator"
                )
            # PyNccl takes a group-relative root; torch.distributed takes global.
            self.source_group_rank = dist.get_process_group_ranks(group).index(
                source_rank
            )

    def synchronize(self, view):
        if self.world_size == 1 or view.tokens.shape[0] == 0:
            return view
        # SGLang's TP scheduler guarantees the same ordered batch and K on
        # all ranks. Each rank independently validates its local SA/history.
        # Broadcast a copy so local candidates remain available for comparison.
        packet = torch.cat(
            (view.tokens, view.use_retrieval[:, None].to(torch.int64)), dim=1
        )
        comm = self.communicator
        # PyNccl normally disables itself outside capture. Explicitly enable it
        # for eager too, retaining the same communicator and collective ordering.
        with nullcontext() if comm is None else comm.change_state(enable=True):
            if comm is None:
                dist.broadcast(packet, src=self.source_rank, group=self.group)
            else:
                comm.broadcast(packet, src=self.source_group_rank)
            tokens = packet[:, :-1].contiguous()
            agree = (
                view.use_retrieval
                & packet[:, -1].bool()
                & (view.tokens == tokens).all(dim=1)
            ).to(torch.int32)
            # No local-mask branch, even for empty/padded/ineligible SA slots.
            if comm is None:
                dist.all_reduce(agree, op=dist.ReduceOp.MIN, group=self.group)
            else:
                comm.all_reduce(agree, op=dist.ReduceOp.MIN)
        return DraftOverrideView(tokens, agree.bool())
