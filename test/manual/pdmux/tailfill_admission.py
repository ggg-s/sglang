"""Qualify a variable tail layout directly against its native target execution."""

import gc
import os
import statistics

import torch

from mixed_eager_harness import StateSnapshot
from bench_mixed_full_model import SeparateTargetGraph
from mixed_target_admission import _measure, _same_state
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import get_attention_backend
from sglang.srt.speculative.mixed_graph import MixedTargetGraph


@torch.no_grad()
def admit_tail_graph(scheduler, pfb, vfb, requests, native_verify, *, linear_tile, moe_block_size):
    mr = scheduler.model_worker.target_worker.model_runner
    count = len(pfb.input_ids)
    report = dict(tokens=count, prefill_requests=pfb.batch_size,
                  verify_requests=vfb.batch_size, verify_tokens=len(vfb.input_ids),
                  admitted=False, graphs_captured=0, gate="native_state_output_and_greedy")
    diagnostic = os.environ.get("LIVE_TAILFILL_DIAG_DIR")
    if diagnostic and getattr(scheduler, "_tail_diagnostic_done", False):
        return None, report | dict(reason="diagnostic_already_done")
    free, _ = torch.cuda.mem_get_info()
    if free < (1024 << 20) + count * (512 << 10):
        return None, report | dict(reason="workspace_budget")
    locations = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=mr.device),
        pfb.out_cache_loc, vfb.out_cache_loc,
        *[scheduler.req_to_token_pool.req_to_token[r.kv.req_pool_idx, :r.kv.kv_allocated_len]
          for r in requests],
    ]).unique()
    before = StateSnapshot(scheduler, locations)
    separate = None
    reference_kind = os.environ.get("LIVE_TAILFILL_REFERENCE", "native")
    if reference_kind not in {"native", "stable_separate"}:
        raise ValueError(f"Unknown tail reference: {reference_kind}")
    try:
        backend = get_attention_backend(model_runner=mr, init_new_workspace=True)

        def native():
            backend.init_forward_metadata(pfb)
            with forward_context(ForwardContext(attn_backend=backend)):
                pout = mr.model(pfb.input_ids, pfb.positions, pfb)
            verify = native_verify()
            assert verify.can_run_cuda_graph, "Native Verify Graph required"
            return pout, verify.logits_output

        native()
        before.restore()
        outputs, report["native"] = _measure(native)
        expected = [(o.next_token_logits.cpu().clone(), o.hidden_states.cpu().clone()) for o in outputs]
        native_ids = [o.next_token_logits.argmax(-1).cpu() for o in outputs]
        native_state = StateSnapshot(scheduler, locations)
        del outputs
        if diagnostic:
            from diagnose_tailfill import diagnose_native_pair
            scheduler._tail_diagnostic_done = True
            diagnose_native_pair(mr, pfb, vfb, before, native, native_state, diagnostic)
            return None, report | dict(reason="diagnostic_only")
        if reference_kind == "stable_separate":
            # Keep this arithmetic contract explicit. Native atomic Marlin is
            # not an exact state oracle; never silently relax the native gate.
            separate = SeparateTargetGraph(
                mr, pfb, vfb, prefill_capacity=count, restore=before.restore,
                prefill_capacity_limit=scheduler.chunked_prefill_size,
                linear_tile=(64, 128, 8), moe_block_size=moe_block_size,
                variable_batch=True,
            )
            report["graphs_captured"] += 1
            before.restore()
            reference = separate.replay(pfb, vfb)
            expected = [(o.next_token_logits.cpu().clone(), o.hidden_states.cpu().clone()) for o in reference[:2]]
            native_state = StateSnapshot(scheduler, locations)
            report["gate"] = "stable_separate_state_output_and_native_greedy"
        native_pair = os.environ.get("LIVE_TAILFILL_NATIVE_PAIR") == "1"
        report["execution"] = "native_pair_graph" if native_pair else "shared_mixed_graph"
        graph_class = SeparateTargetGraph if native_pair else MixedTargetGraph
        graph = graph_class(
            mr, pfb, vfb, prefill_capacity=count, restore=before.restore,
            prefill_capacity_limit=scheduler.chunked_prefill_size,
            linear_tile=linear_tile, moe_block_size=moe_block_size,
            variable_batch=True,
            stable_numerics=not native_pair,
        )
        report["graphs_captured"] += 1
        before.restore()
        outputs = graph.replay(pfb, vfb)
        actual_state = StateSnapshot(scheduler, locations)
        _same_state(native_state, actual_state)
        greedy_mismatches = []
        for (logits, hidden), actual, ids in zip(expected, outputs[:2], native_ids):
            logits_actual = actual.next_token_logits.cpu()
            torch.testing.assert_close(logits_actual, logits, rtol=0.02, atol=0.002)
            torch.testing.assert_close(actual.hidden_states.cpu(), hidden, rtol=0.02, atol=0.002)
            greedy_mismatches.append(int((logits_actual.argmax(-1) != ids).sum()))
        report["state_and_output_passed"] = True
        report["native_greedy_mismatches"] = greedy_mismatches
        report["native_greedy_passed"] = not any(greedy_mismatches)
        if separate is not None:
            from profile_mixed_target import profile_target
            profile_target(count, before, native,
                           lambda: separate.replay(pfb, vfb),
                           lambda: graph.replay(pfb, vfb))
        timings = {"native": [], "candidate": []}
        actions = {"native": native, "candidate": lambda: graph.replay(pfb, vfb)}
        for trial in range(3):
            order = ("native", "candidate") if trial % 2 == 0 else ("candidate", "native")
            for name in order:
                before.restore()
                _, timing = _measure(actions[name])
                timings[name].append(timing)
        for name, samples in timings.items():
            report[name] = {key: statistics.median(row[key] for row in samples)
                            for key in ("gpu_ms", "wall_ms")}
            report[name]["samples"] = samples
        if any(greedy_mismatches):
            return None, report | dict(reason="numerical_gate", error="Native/Joint greedy decisions differ")
        report["admitted"] = report["candidate"]["wall_ms"] < report["native"]["wall_ms"] * 0.98
        report["reason"] = "profitable" if report["admitted"] else "no_net_gain"
        return graph if report["admitted"] else None, report
    except AssertionError as exc:
        return None, report | dict(reason="numerical_gate", error=str(exc)[:1000])
    finally:
        before.restore()
        mr.decode_cuda_graph_runner.load_batch(vfb)
        del separate
        gc.collect()
