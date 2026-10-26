"""Private full-model target-stage benchmark, after the mixed correctness gate.

Invoked by mixed_eager_harness with benchmark_trials in a Graph job. Native
Separate retains native Prefill and the original Verify CUDA Graph. Repaired
Separate and Mixed both replay captured full-model graphs. Draft, acceptance,
state commit, and request scheduling are outside these target-stage timings;
these measurements must not be reported as serving throughput.
"""

from __future__ import annotations

import gc
import statistics
import time

import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.mixed_graph import MixedTargetGraph


class SeparateTargetGraph(MixedTargetGraph):
    """Own distinct input buffers and attention metadata for the control graph."""

    def _forward(self):
        outputs = []
        for batch, backend in (
            (self.prefill_batch, self.prefill_backend),
            (self.verify_batch, self.verify_backend),
        ):
            with forward_context(ForwardContext(attn_backend=backend)):
                outputs.append(self.mr.model(batch.input_ids, batch.positions, batch))
        return outputs[0], outputs[1], []


def _summary(values):
    ordered = sorted(values)
    return dict(
        median_ms=statistics.median(ordered),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        samples_ms=values,
    )


@torch.no_grad()
def benchmark_target(
    *,
    model_runner,
    graph,
    prefill_batch,
    verify_batch,
    prefill_backend,
    native_verify,
    restore,
    snapshot,
    trials=12,
):
    if type(trials) is not int or trials < 6:
        raise ValueError("Use at least six balanced trials")

    def native_separate():
        prefill_backend.init_forward_metadata(prefill_batch)
        with forward_context(ForwardContext(attn_backend=prefill_backend)):
            model_runner.model(
                prefill_batch.input_ids, prefill_batch.positions, prefill_batch
            )
        output = native_verify()
        if not output.can_run_cuda_graph:
            raise RuntimeError("Native Verify fell out of its CUDA Graph")

    separate_graph = None
    try:
        print(f"MIXED_TARGET_BENCH P={graph.layout.prefill_tokens} capture", flush=True)
        separate_graph = SeparateTargetGraph(
            model_runner,
            prefill_batch,
            verify_batch,
            prefill_capacity=graph.layout.prefill_capacity,
            restore=restore,
        )

        # Check the added Separate graph, including its target pool writes,
        # against the already-qualified Mixed graph before collecting timings.
        restore()
        separate_outputs = separate_graph.replay(prefill_batch, verify_batch)
        separate_state = snapshot()
        expected = [
            (
                output.next_token_logits.detach().cpu().clone(),
                output.hidden_states.detach().cpu().clone(),
            )
            for output in separate_outputs[:2]
        ]
        restore()
        mixed_outputs = graph.replay(prefill_batch, verify_batch)
        differences = separate_state.compare(snapshot())
        if any(row["mismatch"] for row in differences.values()):
            raise RuntimeError("Repaired Separate/Mixed Graph state differs")
        for (logits, hidden), output in zip(expected, mixed_outputs[:2]):
            torch.testing.assert_close(
                output.next_token_logits.cpu(), logits, rtol=0.02, atol=0.002
            )
            torch.testing.assert_close(
                output.hidden_states.cpu(), hidden, rtol=0.02, atol=0.002
            )
        del separate_state, expected
        print(f"MIXED_TARGET_BENCH P={graph.layout.prefill_tokens} timing", flush=True)

        def separate():
            separate_graph.replay(prefill_batch, verify_batch)

        def mixed():
            graph.replay(prefill_batch, verify_batch)

        actions = dict(
            native_separate=native_separate,
            repaired_separate=separate,
            repaired_mixed=mixed,
        )
        records = {name: dict(gpu=[], wall=[]) for name in actions}
        names = tuple(actions)
        # Rotate and reverse orders. Every path starts from the identical pool
        # snapshot; CPU restoration and final synchronization are outside GPU
        # timing. Wall timing includes input preparation and completion.
        for trial in range(trials + 3):
            order = names[trial % 3 :] + names[: trial % 3]
            if trial % 2:
                order = order[::-1]
            for name in order:
                restore()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                wall_start = time.perf_counter()
                start.record()
                actions[name]()
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter() - wall_start) * 1000
                if trial >= 3:
                    records[name]["gpu"].append(start.elapsed_time(end))
                    records[name]["wall"].append(wall_ms)
        return dict(
            scope="full_model_target_stage_only",
            serving_throughput_measured=False,
            native_verify_graph=True,
            repaired_separate_graph=True,
            repaired_mixed_graph=True,
            separate_graph_state_passed=True,
            trials=trials,
            state_restore_in_timing=False,
            prepare_in_timing=True,
            capture_in_timing=False,
            paths={
                name: {metric: _summary(values) for metric, values in row.items()}
                for name, row in records.items()
            },
        )
    finally:
        restore()
        model_runner.decode_cuda_graph_runner.load_batch(verify_batch)
        del separate_graph
        gc.collect()
