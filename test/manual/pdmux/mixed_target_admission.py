"""Validate/calibrate an actual Prefill chunk without changing its geometry."""

from __future__ import annotations

import gc
import os
import time

import torch
from bench_mixed_full_model import SeparateTargetGraph
from mixed_eager_harness import StateSnapshot

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    get_attention_backend,
)
from sglang.srt.speculative.mixed_graph import MixedTargetGraph


def _candidate_linear_tile():
    choice = os.environ.get("LIVE_MIXED_LINEAR_TILE")
    if choice is None:
        choice = (
            "projection" if os.environ.get("LIVE_MIXED_PROJECTION_TILE") == "1"
            else "legacy"
        )
    choices = {
        "legacy": (16, 32, 4),
        "projection": "projection",
        "wide": (64, 128, 8),
        "pipeline": (128, 128, 4, 3),
    }
    if choice not in choices:
        raise ValueError(f"Unknown private Mixed linear tile: {choice}")
    return choices[choice]


def _measure(function):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    result = function()
    end.record()
    end.synchronize()
    return result, dict(gpu_ms=start.elapsed_time(end), wall_ms=(time.perf_counter()-wall)*1000)


def _same_state(first, second):
    for name, (_, _, expected) in first.tensors.items():
        actual = second.tensors[name][2]
        if torch.equal(expected, actual):
            if expected.is_floating_point():
                assert bool(torch.isfinite(expected).all()), f"Nonfinite state: {name}"
        else:
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002, msg=name)


@torch.no_grad()
def admit_graph(
    scheduler, pfb, vfb, requests, native_verify, *, capacity_limit=32768,
    separate_control=False,
    native_gate=False,
):
    """Return a profitable validated graph, or restore state and reject its shape.

    Only a new shape pays this cost. Native timing retains native eager Prefill
    and the original Verify Graph. Both compared repaired graphs own independent
    buffers/backends. Rejected shapes never change the scheduler's chunk size.
    """
    if native_gate:
        from tailfill_admission import admit_tail_graph

        return admit_tail_graph(
            scheduler, pfb, vfb, requests, native_verify,
            linear_tile=_candidate_linear_tile(),
            moe_block_size=int(os.environ.get("LIVE_MIXED_MOE_BLOCK", "32")),
        )
    mr = scheduler.model_worker.target_worker.model_runner
    count = len(pfb.input_ids)
    free, _ = torch.cuda.mem_get_info()
    required = (512 << 20) + count * (512 << 10)
    if free < required:
        return None, dict(tokens=count, admitted=False, reason="workspace_budget")
    locations = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=mr.device),
            pfb.out_cache_loc, vfb.out_cache_loc,
            *[
                scheduler.req_to_token_pool.req_to_token[
                    req.kv.req_pool_idx, :req.kv.kv_allocated_len
                ] for req in requests
            ],
        ]
    ).unique()
    before = StateSnapshot(scheduler, locations)
    separate = graph = None
    report = dict(tokens=count, capacity=count, admitted=False, graphs_captured=0)
    try:
        p_backend = get_attention_backend(model_runner=mr, init_new_workspace=True)

        def native():
            p_backend.init_forward_metadata(pfb)
            with forward_context(ForwardContext(attn_backend=p_backend)):
                p_out = mr.model(pfb.input_ids, pfb.positions, pfb)
            v_out = native_verify()
            assert v_out.can_run_cuda_graph, "Native Verify must retain its Graph"
            return p_out, v_out.logits_output

        before.restore()
        native()  # Untimed preparation/JIT.
        torch.cuda.synchronize()
        before.restore()
        native_outputs, report["native"] = _measure(native)
        native_ids = [out.next_token_logits.argmax(-1).cpu() for out in native_outputs]
        del native_outputs
        separate = SeparateTargetGraph(
            mr, pfb, vfb, prefill_capacity=count, restore=before.restore,
            prefill_capacity_limit=capacity_limit,
        )
        report["graphs_captured"] += 1
        graph = MixedTargetGraph(
            mr, pfb, vfb, prefill_capacity=count, restore=before.restore,
            prefill_capacity_limit=capacity_limit,
            linear_tile=_candidate_linear_tile(),
            moe_block_size=int(os.environ.get("LIVE_MIXED_MOE_BLOCK", "8")),
        )
        report["candidate_numerical_options"] = graph.numerical_options
        report["graphs_captured"] += 1
        before.restore()
        reference = separate.replay(pfb, vfb)
        reference_state = StateSnapshot(scheduler, locations)
        before.restore()
        outputs = graph.replay(pfb, vfb)
        _same_state(reference_state, StateSnapshot(scheduler, locations))
        for expected, actual, ids in zip(reference[:2], outputs[:2], native_ids):
            torch.testing.assert_close(
                actual.next_token_logits, expected.next_token_logits, rtol=0.02, atol=0.002
            )
            torch.testing.assert_close(
                actual.hidden_states, expected.hidden_states, rtol=0.02, atol=0.002
            )
            assert torch.equal(actual.next_token_logits.argmax(-1).cpu(), ids), (
                "Native/Joint greedy decisions differ"
            )
        report["state_and_output_passed"] = True
        from profile_mixed_target import profile_target

        profile_target(
            count, before, native,
            lambda: separate.replay(pfb, vfb),
            lambda: graph.replay(pfb, vfb),
        )
        before.restore()
        candidate = separate if separate_control else graph
        _, report["candidate"] = _measure(lambda: candidate.replay(pfb, vfb))
        report["candidate_kind"] = "separate" if separate_control else "mixed"
        report["admitted"] = report["candidate"]["wall_ms"] < report["native"]["wall_ms"] * 0.98
        report["reason"] = "profitable" if report["admitted"] else "no_net_gain"
        return candidate if report["admitted"] else None, report
    except AssertionError as exc:
        report.update(reason="numerical_gate", error=str(exc)[:1000])
        return None, report
    finally:
        before.restore()
        mr.decode_cuda_graph_runner.load_batch(vfb)
        del separate
        gc.collect()
