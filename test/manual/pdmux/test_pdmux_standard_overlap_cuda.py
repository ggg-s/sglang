"""Standard chunk scheduling with real CUDA streams and decode graph replay.

Runs on one GPU, including the 4060. Model/HiCache/TP collaborators are small
fixtures; this does not emulate multi-GPU DSV4 attention or measure throughput.
The production standard loop, completion/merge and stream transitions run as-is.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "test/registered/unit/multiplex"))
from test_pdmux_standard_prefill import (
    _Batch,
    _FakeScheduler,
    _LoopFinished,
    _STREAM_IDX,
)

from sglang.srt.distributed.parallel_state import is_pdmux_prefill_enabled
from sglang.srt.multiplex import multiplexing_mixin

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("grammar", [False, True])
def test_standard_whole_chunk_with_decode_graph_and_delayed_completion(grammar):
    class Scheduler(_FakeScheduler):
        def __init__(self):
            self.chunks = [_Batch(f"chunk-{i}") for i in range(3)]
            super().__init__(
                max_iterations=128,
                enable_overlap=True,
                prefill_batches=self.chunks,
                grammar=grammar,
            )
            self.stream_groups = [
                (torch.cuda.Stream(), torch.cuda.Stream()) for _ in range(3)
            ]
            self.decode_input = torch.zeros((32, 32), device="cuda")
            self.decode_output = torch.empty_like(self.decode_input)
            self.decode_graph = torch.cuda.CUDAGraph()
            decode_stream = self.stream_groups[0][1]
            decode_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(self.decode_graph, stream=decode_stream):
                self.decode_output.copy_(self.decode_input + 1)
            decode_stream.synchronize()
            self.weight = torch.eye(32, device="cuda") * 0.5
            self.prefill_outputs = []
            self.decode_outputs = []
            self.forward_counts = {}
            ready = torch.cuda.current_stream().record_event()
            for streams in self.stream_groups:
                for stream in streams:
                    stream.wait_event(ready)

        def _recv_requests(self):
            if len(self.processed_prefills) == len(self.chunks):
                raise _LoopFinished
            return super()._recv_requests()

        def run_batch(self, batch):
            prefill = self._pdmux_prefill_batch is batch
            if prefill:
                assert is_pdmux_prefill_enabled()
                assert torch.cuda.current_stream() == self.pdmux_prefill_stream
                assert batch.forward_mode.is_extend()
                self.forward_counts[batch.name] = (
                    self.forward_counts.get(batch.name, 0) + 1
                )
                self.submitted.append(
                    (self.iteration, batch, self.pdmux_prefill_stream)
                )
                # One worker call executes all three fixture layers for this
                # chunk. Aux captures remain alive across many decode steps.
                hidden = torch.ones((1024, 32), device="cuda")
                aux = []
                for _ in range(3):
                    hidden = torch.nn.functional.linear(hidden, self.weight)
                    aux.append(hidden)
                payload = hidden.clone()
                self.prefill_outputs.append(payload)
                keep_alive = [batch, hidden, *aux]
            else:
                assert not is_pdmux_prefill_enabled()
                assert torch.cuda.current_stream() == self.forward_stream
                self.decode_input.fill_(self.iteration)
                self.decode_graph.replay()
                payload = self.decode_output.clone()
                self.decode_outputs.append((self.iteration, payload))
                keep_alive = [payload]
            event = torch.cuda.Event()
            event.record()
            return SimpleNamespace(
                is_prefill=prefill,
                submitted_iteration=self.iteration,
                extra_keep_alive_refs=keep_alive,
                copy_done=event,
                payload=payload,
            )

        def process_batch_result(self, batch, result):
            # Same dependency as the production result processor. The loop
            # should only wait for N-1; a ready prefill merge drains N once.
            result.copy_done.synchronize()
            super().process_batch_result(batch, result)

        def _allreduce(self, flags, op):
            # Model one slower peer: a ready local prefill cannot finalize on
            # its first three votes. The two-process Gloo test covers the real
            # collective rather than sharing this one GPU between NCCL ranks.
            ready = int(flags[0]) and self.iteration % 4 == 3
            return SimpleNamespace(wait=lambda: flags.fill_(2 if ready else 0))

    scheduler = Scheduler()
    _STREAM_IDX[0] = 0
    with (
        patch.object(
            multiplexing_mixin, "get_current_stream_idx", lambda: _STREAM_IDX[0]
        ),
        patch.object(
            multiplexing_mixin,
            "set_current_stream_idx",
            lambda idx: _STREAM_IDX.__setitem__(0, idx),
        ),
        patch.object(multiplexing_mixin.dist, "get_world_size", lambda group: 2),
    ):
        try:
            scheduler.event_loop_pdmux_standard()
        except _LoopFinished:
            pass
        finally:
            multiplexing_mixin.set_pdmux_status(False)
    torch.cuda.synchronize()
    assert scheduler.forward_counts == {f"chunk-{i}": 1 for i in range(3)}
    assert len(scheduler.processed_prefills) == 3
    assert scheduler.pumps
    for output in scheduler.prefill_outputs:
        torch.testing.assert_close(output, torch.full_like(output, 0.125))
    for step, output in scheduler.decode_outputs:
        torch.testing.assert_close(output, torch.full_like(output, step + 1))
    for batch in scheduler.chunks:
        assert batch.spec_info.future_indices is batch.req_pool_indices
