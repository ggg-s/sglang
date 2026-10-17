"""Shared gate and TP decision for staged CPU and graph-resident retrieval."""

import torch

from hybrid_spec.core.contracts import DraftOverrideView


class GatedOverride(DraftOverrideView):
    def __init__(
        self,
        tokens,
        use_retrieval,
        num_neural_gate_tokens=0,
        *,
        tp_sync=None,
        defer_prefix=False,
        query_stream=None,
        neural_prefix=None,
    ):
        super().__init__(tokens, use_retrieval, num_neural_gate_tokens)
        self.tp_sync = tp_sync
        self.defer_prefix = defer_prefix
        self.query_stream = query_stream
        if defer_prefix:
            self.neural_prefix = (
                neural_prefix
                if neural_prefix is not None
                else torch.empty(
                    (tokens.shape[0], max(0, num_neural_gate_tokens - 1)),
                    dtype=torch.int64,
                    device=tokens.device,
                )
            )

    def prepare(self, seq_lens):
        # CPU candidates are already staged. Split consensus follows the gate;
        # full consensus precedes per-step selection, as in the device backend.
        if not self.defer_prefix:
            self._synchronize()

    def _synchronize(self):
        if self.tp_sync is not None:
            agreed = self.tp_sync.synchronize(self)
            # Preserve graph storage and the mask alias used by verify metrics.
            self.tokens.copy_(agreed.tokens)
            self.use_retrieval.copy_(agreed.use_retrieval)

    def select_step(self, neural_tokens, step):
        if not self.defer_prefix:
            return super().select_step(neural_tokens, step)
        if neural_tokens.shape not in (
            (self.tokens.shape[0],),
            (self.tokens.shape[0], 1),
        ):
            raise ValueError("hybrid selection requires one neural token per row")
        gate = self.num_neural_gate_tokens
        if step < gate - 1:
            # Run the complete neural gate prefix without reading SA output.
            self.neural_prefix[:, step].copy_(neural_tokens.reshape(-1))
            return neural_tokens
        if step == max(0, gate - 1):
            if self.query_stream is not None:
                torch.cuda.current_stream().wait_stream(self.query_stream)
            if gate:
                self.use_retrieval.logical_and_(
                    (self.tokens[:, : gate - 1] == self.neural_prefix).all(dim=1)
                )
                self.use_retrieval.logical_and_(
                    self.tokens[:, step] == neural_tokens.reshape(-1)
                )
            # All ranks participate, even with no local hits. Combining the
            # final gate mask with exact candidate consensus guarantees the
            # same skip/compact rows before any rank chooses a tail graph.
            # Keep NCCL on the model stream, after its prefix collectives.
            self._synchronize()
        if step < gate:
            return neural_tokens
        return super().select_step(neural_tokens, step)
