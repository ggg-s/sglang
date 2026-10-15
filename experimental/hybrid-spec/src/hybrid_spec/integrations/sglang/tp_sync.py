"""Consensus on retrieval overrides before eager execution or graph staging."""

import torch
import torch.distributed as dist
from hybrid_spec.core.contracts import DraftOverrideView


class TPSynchronizer:
    def __init__(self, group, source_rank):
        self.group = group
        # torch.distributed expects a global rank, including for subgroups.
        self.source_rank = source_rank
        self.world_size = dist.get_world_size(group)

    def synchronize(self, view):
        if self.world_size == 1 or view.tokens.shape[0] == 0:
            return view
        # SGLang's TP scheduler guarantees the same ordered batch and K on
        # all ranks. Each rank independently validates its local SA/history.
        # Broadcast a copy so local candidates remain available for comparison.
        packet = torch.cat(
            (view.tokens, view.use_retrieval[:, None].to(torch.int64)), dim=1
        )
        dist.broadcast(packet, src=self.source_rank, group=self.group)
        tokens = packet[:, :-1].contiguous()
        agree = (
            view.use_retrieval
            & packet[:, -1].bool()
            & (view.tokens == tokens).all(dim=1)
        ).to(torch.int32)
        # Never branch on a local selection mask: every rank participates even
        # when it has no eligible retrieval rows. MIN makes fallback unanimous.
        dist.all_reduce(agree, op=dist.ReduceOp.MIN, group=self.group)
        return DraftOverrideView(tokens, agree.bool())
