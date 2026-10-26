"""Private TP1/greedy scheduler experiment preserving the serving configuration.

Import from a private sitecustomize only with LIVE_MIXED_ROOT set. Modes are
native (unchanged scheduler), separate (paired scheduling, native forwards),
repaired_separate (captured Separate), and mixed (captured Mixed). HiCache and
PDMux are excluded from pairing. This is not a registered serving option.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
import json
import os
import time
from pathlib import Path

from mixed_target_admission import admit_graph
from tailfill_policy import decode_slots, split_running
from profile_live_service import ServiceProfile
from mixed_eager_validation import (
    DIAGNOSTIC_FLAGS,
    GRAPH_CAPACITIES,
    GRAPH_CHUNKS,
    TOLERANCES,
    _state_matches,
    source_hashes,
    validate_eager_gate,
    write_json,
)

from sglang.srt.managers.scheduler import Scheduler, resolve_forward_inputs
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.speculative.mixed_graph import MixedGraphLayoutChanged

ROOT = Path(os.environ["LIVE_MIXED_ROOT"])
MODE = os.environ.get("LIVE_MIXED_MODE", "mixed")
RUN_BATCH = Scheduler.run_batch
ON_IDLE = Scheduler.on_idle
PROCESS_RESULT = Scheduler.process_batch_result


def _qualified_gates(folder):
    expected = source_hashes(Path(__file__).resolve().parents[3])
    for rejection, lane in ((False, "natural"), (True, "rejection")):
        report = json.loads((folder / f"native-{lane}-report.json").read_text())
        validate_eager_gate(report, expected_sources=expected, rejection=rejection)
        report = json.loads((folder / f"graph-{lane}-report.json").read_text())
        if (
            report.get("stage") != "completed"
            or report.get("gate_passed") is not True
            or report.get("pools_flushed") is not True
            or report.get("formal_serving_changed") is not False
            or report.get("tolerances") != TOLERANCES
            or report.get("numerical_source_sha256") != expected
            or report.get("draft_graph_hits") != [True] * 9
            or any(report.get("config", {}).get(flag) for flag in DIAGNOSTIC_FLAGS)
            or len(report.get("cases", [])) != 9
            or not all(
                case.get("p_tokens") == GRAPH_CHUNKS[index]
                and case.get("p_capacity") == GRAPH_CAPACITIES[index]
                and case.get("native_verify_graph") is True
                and _state_matches(case.get("state_comparison"))
                and case.get("safety_checks")
                and all(
                    row and all(value is True for value in row.values())
                    for row in case["safety_checks"]
                )
                and case.get("mixed_graph", {}).get("replayed") is True
                and case.get("acceptance_equal") is True
                and case.get("native_acceptance_equal") is True
                and case.get("safety_passed") is True
                and case.get("state_mismatches") == {}
                for index, case in enumerate(report["cases"])
            )
        ):
            raise ValueError("Complete source-matched Mixed Graph gates are required")


class LiveMixedTarget:
    def __init__(self, scheduler):
        if MODE not in {"native", "separate", "repaired_separate", "mixed", "tailfill", "tailfill_native"}:
            raise ValueError("Unknown LIVE_MIXED_MODE")
        if (
            scheduler.ps.tp_size != 1
            or scheduler.ps.pp_size != 1
            or scheduler.ps.attn_dp_size != 1
        ):
            raise ValueError("Live Mixed experiment supports TP1/PP1 only")
        # Tailfill admits each new variable layout against native state/output.
        # Historical fixed P1+D3 gates cannot qualify these new layouts.
        if MODE not in {"native", "tailfill", "tailfill_native"}:
            _qualified_gates(Path(os.environ["LIVE_MIXED_GATES"]))
        self.scheduler = scheduler
        self.profile = ServiceProfile(ROOT)
        self.worker = scheduler.model_worker
        self.mr = self.worker.target_worker.model_runner
        if self.worker.block_size != 8:
            raise ValueError("Live Mixed experiment requires DFlash block8")
        self.graphs = []
        self.rejected_shapes = set()
        self.min_tokens = int(os.environ.get("LIVE_MIXED_MIN_TOKENS", "1"))
        self.graph_limit = int(os.environ.get("LIVE_MIXED_GRAPH_LIMIT", "4"))
        if self.min_tokens < 1 or not 1 <= self.graph_limit <= 4:
            raise ValueError("Invalid private Mixed admission bounds")
        self.capacity_limit = int(os.environ.get("LIVE_MIXED_CAPACITY_LIMIT", "32768"))
        if not 0 < self.capacity_limit <= 32768:
            raise ValueError("Invalid live Mixed capacity limit")
        self.probing = False
        self.remainder = None
        self.stats = dict(
            mode=MODE,
            pairs=0,
            mixed=0,
            captures=0,
            fallbacks=0,
            prefill_batches=0,
            prefill_tokens=0,
            draft_calls=0,
            draft_graph_hits=0,
            native_verify_calls=0,
            native_verify_graph_hits=0,
            admission_verify_calls=0,
            admission_graph_captures=0,
            unsupported_prefill=0,
            unpaired_prefill=0,
            rejected_prefill=0,
            mixed_prefill_tokens=0,
            shape_decisions=[],
            tail_opportunities=0,
            tail_unused_tokens=0,
            mixed_verify_tokens=0,
            partial_decode_pairs=0,
            native_tail_pairs=0,
            native_tail_verify_tokens=0,
            native_tail_prefill_tokens=0,
        )
        self.last_write = 0.0
        draft_forward = self.worker.draft_model_runner.forward

        def track_draft(*args, **kwargs):
            with self.profile.scope("stage/draft"):
                output = draft_forward(*args, **kwargs)
            self.stats["draft_calls"] += 1
            self.stats["draft_graph_hits"] += int(output.can_run_graph)
            return output

        self.worker.draft_model_runner.forward = track_draft
        target_forward = self.worker.target_worker.forward_batch_generation

        def track_target(*args, **kwargs):
            with self.profile.scope("stage/verify" if kwargs.get("is_verify") else "stage/prefill"):
                output = target_forward(*args, **kwargs)
            if kwargs.get("is_verify"):
                if self.probing:
                    self.stats["admission_verify_calls"] += 1
                else:
                    self.stats["native_verify_calls"] += 1
                    self.stats["native_verify_graph_hits"] += int(output.can_run_cuda_graph)
            return output

        self.worker.target_worker.forward_batch_generation = track_target

    def eligible(self, batch):
        if MODE == "native" or not batch.forward_mode.is_extend_without_speculative():
            return False
        if self.scheduler.enable_pdmux or self.scheduler.enable_hierarchical_cache:
            self.stats["unsupported_prefill"] += 1
            return False
        count = batch.extend_num_tokens
        running = self.scheduler.running_batch
        selected = self.tail_slots(batch) if MODE.startswith("tailfill") else running.batch_size()
        if MODE.startswith("tailfill"):
            if selected == 0:
                return False
            self.stats["tail_opportunities"] += 1
            self.stats["tail_unused_tokens"] += min(
                self.scheduler.chunked_prefill_size, self.scheduler.max_prefill_tokens
            ) - count
        key = (count, batch.batch_size(), selected) if MODE.startswith("tailfill") else count
        if key in self.rejected_shapes and not any(
            graph.layout.prefill_tokens == count for graph in self.graphs
        ):
            self.stats["rejected_prefill"] += 1
            return False
        if not self.min_tokens <= count <= self.capacity_limit or (not MODE.startswith("tailfill") and batch.batch_size() != 1):
            self.stats["unsupported_prefill"] += 1
            return False
        if len(self.graphs) >= self.graph_limit and not any(
            graph.layout.prefill_tokens == count for graph in self.graphs
        ):
            self.stats["unsupported_prefill"] += 1
            return False
        if not MODE.startswith("tailfill") and running.batch_size() != 3:
            self.stats["unpaired_prefill"] += 1
            return False
        return (
            running.check_decode_mem()
            and not batch.return_logprob
            and not running.return_logprob
            and not batch.return_hidden_states
            and not running.return_hidden_states
            and not batch.has_grammar
            and not running.has_grammar
            and batch.input_embeds is None
            and all(
                not req.finished()
                and req.lora_id is None
                and req.multimodal_inputs is None
                and req.beam_group is None
                and req.sampling_params.top_k == 1
                and req.sampling_params.repetition_penalty == 1
                and req.sampling_params.frequency_penalty == 0
                and req.sampling_params.presence_penalty == 0
                and req.custom_logit_processor is None
                for req in (*batch.reqs, *running.reqs)
            )
        )

    def tail_slots(self, batch):
        return decode_slots(
            batch.extend_num_tokens, self.scheduler.chunked_prefill_size,
            self.scheduler.max_prefill_tokens,
            self.scheduler.running_batch.batch_size(), self.worker.block_size,
        )

    def _mixed(self, pfb, vfb, requests, native_verify):
        for graph in self.graphs:
            if graph.layout.prefill_tokens != len(pfb.input_ids):
                continue
            if MODE.startswith("tailfill") and (
                graph.layout.verify_requests != vfb.batch_size
                or graph.prefill_batch.batch_size != pfb.batch_size
            ):
                continue
            try:
                return graph, graph.replay(pfb, vfb)
            except MixedGraphLayoutChanged:
                pass
        if len(self.graphs) >= self.graph_limit:
            return None, None
        self.probing = True
        try:
            graph, decision = admit_graph(
                self.scheduler, pfb, vfb, requests, native_verify,
                capacity_limit=self.capacity_limit,
                separate_control=MODE == "repaired_separate",
                native_gate=MODE.startswith("tailfill"),
            )
        finally:
            self.probing = False
        self.stats["shape_decisions"].append(decision)
        self.stats["admission_graph_captures"] += decision.get("graphs_captured", 0)
        if graph is None:
            self.rejected_shapes.add(
                (len(pfb.input_ids), pfb.batch_size, vfb.batch_size)
                if MODE.startswith("tailfill") else len(pfb.input_ids)
            )
            return None, None
        self.graphs.append(graph)
        self.stats["captures"] += 1
        return graph, graph.replay(pfb, vfb)

    def _finish_pair(self, batch, running, decode_result, kwargs):
        scheduler = self.scheduler
        if getattr(scheduler, "enable_overlap", False):
            # Keep the existing queue depth/order: one logical round owns D+P.
            # The normal loop processes its older result first, then this
            # round's Decode and Prefill in order on the next queue pop.
            owned_decode = running.copy()
            keepalive = scheduler.batch_record_buf[scheduler.batch_record_ct]
            result = RUN_BATCH(scheduler, batch, **kwargs)
            result._mixed_decode_result = (owned_decode, decode_result)
            result._mixed_decode_keepalive = keepalive
            self._merge_remainder(running)
            return result
        scheduler.process_batch_result(running, decode_result)
        self._merge_remainder(running)
        return RUN_BATCH(scheduler, batch, **kwargs)

    def _merge_remainder(self, running):
        remainder = getattr(self, "remainder", None)
        if remainder is not None:
            remainder.merge_batch(running)
            self.scheduler.running_batch = remainder
            self.remainder = None

    def run(self, batch, **kwargs):
        scheduler, worker = self.scheduler, self.worker
        selected = scheduler.running_batch
        if MODE.startswith("tailfill"):
            slots = self.tail_slots(batch)
            if slots == 0:
                return RUN_BATCH(scheduler, batch, **kwargs)
            selected, self.remainder = split_running(selected, slots)
            self.stats["partial_decode_pairs"] += int(self.remainder is not None)
        running = scheduler.update_running_batch(selected)
        scheduler.running_batch = running
        if running.is_empty():
            if self.remainder is not None:
                scheduler.running_batch, self.remainder = self.remainder, None
            return RUN_BATCH(scheduler, batch, **kwargs)
        self.stats["pairs"] += 1
        if MODE in {"separate", "tailfill_native"}:
            if MODE == "tailfill_native":
                self.stats["native_tail_pairs"] += 1
                self.stats["native_tail_verify_tokens"] += running.batch_size() * worker.block_size
                self.stats["native_tail_prefill_tokens"] += batch.extend_num_tokens
            decode_result = RUN_BATCH(scheduler, running)
            return self._finish_pair(batch, running, decode_result, kwargs)
        overlap = getattr(scheduler, "enable_overlap", False)
        with scheduler.forward_stream_ctx if overlap else nullcontext():
            if overlap:
                scheduler.forward_stream.wait_stream(scheduler.schedule_stream)
            resolve_forward_inputs(batch, scheduler.future_map)
            worker.target_worker.set_hicache_consumer(batch.hicache_consumer_index)
            with scheduler._isolate_forward_fields(batch):
                pfb = ForwardBatch.init_new(
                    batch,
                    self.mr,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    return_hidden_states_before_norm=False,
                )
                self.mr._maybe_execute_deferred_mamba_cow_and_clear(pfb)

        original = worker.target_worker.forward_batch_generation
        old_backend = self.mr.attn_backend
        cached_prefill = None

        def target(*args, **target_kwargs):
            nonlocal cached_prefill
            vfb = target_kwargs.get("forward_batch")
            if (
                MODE in {"mixed", "repaired_separate", "tailfill"}
                and target_kwargs.get("is_verify")
                and (MODE.startswith("tailfill") or vfb.batch_size == 3)
            ):
                self.mr._maybe_execute_deferred_mamba_cow_and_clear(vfb)
                graph, outputs = self._mixed(
                    pfb, vfb, [*batch.reqs, *running.reqs],
                    lambda: original(*args, **target_kwargs),
                )
                if graph is not None:
                    pout, vout, _ = outputs
                    cached_prefill = copy.copy(pout)
                    # Native acceptance/commit must use the metadata that wrote
                    # the mixed verify intermediates, until this decode ends.
                    self.mr.attn_backend = graph.verify_backend
                    if MODE in {"mixed", "tailfill", "tailfill_native"}:
                        self.stats["mixed"] += 1
                        self.stats["mixed_prefill_tokens"] += len(pfb.input_ids)
                        self.stats["mixed_verify_tokens"] = self.stats.get("mixed_verify_tokens", 0) + (
                            len(vfb.input_ids) if MODE.startswith("tailfill") else 24
                        )
                    else:
                        key = "repaired_separate_graph_calls"
                        self.stats[key] = self.stats.get(key, 0) + 1
                    return GenerationBatchResult(
                        logits_output=copy.copy(vout), can_run_cuda_graph=True
                    )
                self.stats["fallbacks"] += 1
            if args and args[0] is batch and cached_prefill is not None:
                # Finalization clears hidden_states; never mutate the captured
                # output object. Sampling may modify logits in place as well.
                output = copy.copy(cached_prefill)
                output.next_token_logits = output.next_token_logits.clone()
                return GenerationBatchResult(
                    logits_output=output,
                    next_token_ids=self.mr.sample(output, pfb),
                    can_run_cuda_graph=True,
                )
            return original(*args, **target_kwargs)

        worker.target_worker.forward_batch_generation = target
        try:
            decode_result = RUN_BATCH(scheduler, running)
            self.mr.attn_backend = old_backend
            return self._finish_pair(batch, running, decode_result, kwargs)
        finally:
            self.mr.attn_backend = old_backend
            worker.target_worker.forward_batch_generation = original

    def publish(self):
        if time.monotonic() - self.last_write >= 1:
            self.last_write = time.monotonic()
            write_json(ROOT / "mixed-stats.json", self.stats)


def run_batch(self, batch, **kwargs):
    controller = getattr(self, "_live_mixed_target", None)
    if controller is None:
        controller = self._live_mixed_target = LiveMixedTarget(self)
    controller.profile.maybe_start()
    if batch.forward_mode.is_extend_without_speculative():
        controller.stats["prefill_batches"] += 1
        controller.stats["prefill_tokens"] += batch.extend_num_tokens
    label = (
        f"round/prefill_{batch.extend_num_tokens}"
        if batch.forward_mode.is_extend_without_speculative()
        else f"round/decode_{batch.batch_size()}"
    )
    with controller.profile.scope(label):
        if controller.eligible(batch):
            return controller.run(batch, **kwargs)
        return RUN_BATCH(self, batch, **kwargs)


def on_idle(self, *args, **kwargs):
    controller = getattr(self, "_live_mixed_target", None)
    if controller is not None:
        controller.publish()
        controller.profile.maybe_finish()
    return ON_IDLE(self, *args, **kwargs)


def process_batch_result(self, batch, result):
    paired = getattr(result, "_mixed_decode_result", None)
    if paired is not None:
        del result._mixed_decode_result
        PROCESS_RESULT(self, *paired)
    try:
        return PROCESS_RESULT(self, batch, result)
    finally:
        if hasattr(result, "_mixed_decode_keepalive"):
            del result._mixed_decode_keepalive


Scheduler.run_batch = run_batch
Scheduler.on_idle = on_idle
Scheduler.process_batch_result = process_batch_result
