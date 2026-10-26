"""Offline actual-model validation; imported only by a private worker idle hook.

Run with the ordinary AWQ+DFlash launch config (PDMux disabled). The harness
owns all four requests, never processes external requests, and restores the
same target/draft state before every branch. Timings are diagnostic, not A/B.
"""

from __future__ import annotations

import dataclasses
import gc
import importlib
import json
import time
from array import array
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from mixed_eager_validation import (
    source_hashes,
    validate_eager_gate,
    validate_proposals,
)

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    get_attention_backend,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import mixed_eager

run_mixed_eager = importlib.reload(mixed_eager).run_mixed_eager


@contextmanager
def _fixed_reduction_diagnostic():
    """Offline eager diagnosis only; captured graphs and serving stay unchanged."""
    linear = importlib.import_module("sglang.srt.layers.quantization.marlin_utils")
    moe = importlib.import_module(
        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
    )
    old_linear, old_moe = (
        linear.should_use_atomic_add_reduce,
        moe.is_batch_invariant_mode_enabled,
    )
    linear.should_use_atomic_add_reduce = lambda *a, **kw: False
    moe.is_batch_invariant_mode_enabled = lambda: True
    try:
        yield
    finally:
        linear.should_use_atomic_add_reduce = old_linear
        moe.is_batch_invariant_mode_enabled = old_moe


def _models(mixed, config, mr, pfb, vfb, pb, vb, capacity, frozen=None):
    with (
        (
            _fixed_reduction_diagnostic()
            if config.get("fixed_reduction_diagnostic")
            else nullcontext()
        ),
        (
            importlib.import_module(
                "sglang.srt.speculative.mixed_marlin"
            ).stable_marlin(
                mr.model,
                full_k_moe=config.get("stable_marlin_candidate") != "linears_only",
            )
            if config.get("stable_marlin_candidate")
            else nullcontext()
        ) as stable_dispatch,
        (
            importlib.import_module("mixed_eager_numerics").fp32_router_diagnostic(
                mr.model
            )
            if config.get("fp32_router_diagnostic")
            else nullcontext()
        ),
    ):
        if stable_dispatch is not None:
            config.setdefault("_stable_dispatch", []).append(stable_dispatch)
        if mixed:
            with _freeze_token_local_outputs(mr.model, frozen, capacity):
                return run_mixed_eager(
                    mr.model, pfb, vfb, pb, vb, prefill_capacity=capacity
                )
        with forward_context(ForwardContext(attn_backend=pb)):
            pout = mr.model(pfb.input_ids, pfb.positions, pfb)
        with forward_context(ForwardContext(attn_backend=vb)):
            vout = mr.model(vfb.input_ids, vfb.positions, vfb)
        return pout, vout, []


def _cpu(tensor):
    return tensor.detach().to("cpu", copy=True)


@contextmanager
def _capture_features(model, enabled):
    values, hooks = {}, []
    if enabled:
        for name, module in model.named_modules():
            if not name.endswith(
                (
                    ".in_proj_qkvz",
                    ".in_proj_ba",
                    ".qkv_proj",
                    ".out_proj",
                    ".o_proj",
                    ".mlp",
                )
            ):
                continue

            def hook(owner, args, output, name=name):
                tensor = output if isinstance(output, torch.Tensor) else output[0]
                if isinstance(tensor, torch.Tensor):
                    values.setdefault(name, []).append(_cpu(tensor))

            hooks.append(module.register_forward_hook(hook))
    try:
        yield values
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def _freeze_token_local_outputs(model, reference, capacity):
    """Diagnostic replay only: isolate state routing from quantized GEMM noise.

    This is not a runnable mixed candidate and never qualifies the graph gate.
    Normal mixed execution does not install these hooks.
    """
    hooks = []
    if reference is not None:
        for name, module in model.named_modules():
            if name not in reference or len(reference[name]) != 2:
                continue
            p, v = reference[name]

            def hook(owner, args, output, p=p, v=v):
                actual = output if isinstance(output, torch.Tensor) else output[0]
                joined = actual.new_zeros(actual.shape)
                joined[: len(p)].copy_(p.to(actual.device))
                joined[capacity : capacity + len(v)].copy_(v.to(actual.device))
                return (
                    joined
                    if isinstance(output, torch.Tensor)
                    else (joined, *output[1:])
                )

            hooks.append(module.register_forward_hook(hook))
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def _compare_features(separate, mixed, count, capacity):
    rows = {}
    for name, parts in separate.items():
        if name not in mixed or len(parts) != 2 or len(mixed[name]) != 1:
            continue
        joined = mixed[name][0]
        result = {}
        for label, first, second in [
            ("P", parts[0], joined[:count]),
            ("V", parts[1], joined[capacity : capacity + 24]),
        ]:
            error = (first.float() - second.float()).abs()
            result[label] = dict(
                max_abs=float(error.max()),
                rel_l2=float(
                    torch.linalg.vector_norm(error)
                    / torch.linalg.vector_norm(first.float()).clamp_min(1e-12)
                ),
            )
        rows[name] = result
    return rows


class StateSnapshot:
    """Copy mutable pool data, not weights, graph pointers or allocator state.

    Preparation/allocation occurs once before snapshotting. Capture the physical
    deduplicated Conv backing rather than restoring an overlapping as_strided
    view. Request-keyed replay rings and physical SSM slots are separate arrays.
    """

    def __init__(self, scheduler, locations):
        mr = scheduler.tp_worker.model_runner
        pool = scheduler.req_to_token_pool.mamba_pool
        state = pool.mamba_cache
        self.tensors = {}
        for field in dataclasses.fields(state):
            name, value = field.name, getattr(state, field.name)
            if value is None or name == "intermediate_conv_window":
                continue
            values = value if isinstance(value, list) else [value]
            for i, tensor in enumerate(values):
                self.tensors[f"mamba.{name}.{i}"] = (tensor, None, _cpu(tensor))
        for i, tensor in enumerate(pool._intermediate_conv_window_phys):
            self.tensors[f"mamba.conv_window_physical.{i}"] = (
                tensor,
                None,
                _cpu(tensor),
            )
        for name in (
            "replayssm_spec_write_pos",
            "replayssm_cache_base",
            "replayssm_is_flush",
        ):
            tensor = getattr(pool, name)
            self.tensors[f"mamba.{name}"] = (tensor, None, _cpu(tensor))
        for name in (
            "req_to_token",
            "req_index_to_mamba_index_mapping",
            "req_index_to_mamba_ping_pong_track_buffer_mapping",
        ):
            tensor = getattr(scheduler.req_to_token_pool, name, None)
            if isinstance(tensor, torch.Tensor):
                self.tensors["mapping." + name] = (tensor, None, _cpu(tensor))
        self.locations = locations
        for label, kv in (
            ("target", mr.token_to_kv_pool),
            ("draft", scheduler.draft_worker.draft_model_runner.token_to_kv_pool),
        ):
            layers = getattr(kv, "full_attention_layer_id_mapping", None)
            layers = sorted(layers) if layers is not None else range(kv.layer_num)
            for layer in layers:
                for kind in ("key", "value"):
                    tensor = getattr(kv, f"get_{kind}_buffer")(layer)
                    self.tensors[f"{label}.{kind}.{layer}"] = (
                        tensor,
                        locations,
                        _cpu(tensor.index_select(0, locations)),
                    )
        torch.cuda.synchronize()

    def restore(self):
        for tensor, indices, value in self.tensors.values():
            if indices is None:
                # Copy directly into the existing pool: staging a complete
                # extra SSM tensor consumes 1.46 GiB on this checkpoint.
                tensor.copy_(value)
            else:
                source = value.to(tensor.device)
                tensor.index_copy_(0, indices, source)
        torch.cuda.synchronize()

    def compare(self, other, *, rtol=0.02, atol=0.002):
        metrics = {}
        for name, (_, _, expected) in self.tensors.items():
            actual = other.tensors[name][2]
            if expected.is_floating_point():
                error = (expected.float() - actual.float()).abs()
                close = torch.isclose(
                    expected, actual, rtol=rtol, atol=atol, equal_nan=False
                )
                metrics[name] = dict(
                    max_abs=float(error.max()) if error.numel() else 0,
                    mismatch=int((~close).sum()),
                    elements=expected.numel(),
                    rel_l2=float(
                        torch.linalg.vector_norm(error)
                        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
                    ),
                )
            else:
                metrics[name] = dict(
                    mismatch=int((expected != actual).sum()), elements=expected.numel()
                )
        return metrics


def _prepare(scheduler, reqs):
    batch = ScheduleBatch.init_new(
        reqs,
        scheduler.req_to_token_pool,
        scheduler.token_to_kv_pool_allocator,
        scheduler.tree_cache,
        scheduler.model_config,
        enable_overlap=True,
        spec_algorithm=scheduler.spec_algorithm,
    )
    batch.prepare_for_extend()
    # The public scheduler normally resolves this staging on the forward stream.
    from sglang.srt.managers.scheduler import resolve_forward_inputs

    resolve_forward_inputs(batch, scheduler.future_map)
    return batch


def _new_req(scheduler, label, tokens):
    params = SamplingParams(temperature=0, top_k=1, max_new_tokens=128, ignore_eos=True)
    params.normalize(scheduler.tokenizer)
    req = Req(
        label,
        "",
        array("q", tokens),
        params,
        vocab_size=scheduler.model_config.vocab_size,
    )
    req.init_next_round_input(scheduler.tree_cache)
    req.set_extend_range(0, len(tokens))
    return req


def _advance(batch, result):
    commits = result.accept_lens.cpu().tolist()
    out = result.next_token_ids.view(len(batch.reqs), 8).cpu().tolist()
    lengths = result.new_seq_lens.cpu().tolist()
    for req, count, row, length in zip(batch.reqs, commits, out, lengths):
        req.output_ids.extend(row[:count])
        req.kv.kv_committed_len = length
    batch.spec_info = result.next_draft_input
    batch.seq_lens = result.new_seq_lens.clone()
    batch.seq_lens_cpu = torch.tensor(lengths, dtype=torch.int64)
    batch.seq_lens_sum = sum(lengths)


def _accept_and_commit(worker, batch, draft_input, pfb, vfb, pout, vout, vbackend):
    prefix = vfb.positions.view(3, 8)[:, 0]
    accepted = worker._accept_block(
        candidates=vfb.input_ids.view(3, 8),
        next_token_logits=vout.next_token_logits,
        sampling_info=batch.sampling_info,
        draft_input=draft_input,
        prefix_lens=prefix,
        bs=3,
    )
    accept, commits, bonus, outputs, new_lens, predict = accepted
    # Own the results before the worker's rotating scratch buffers are reused.
    owned = [None if t is None else t.clone() for t in accepted]
    last_steps = commits.to(torch.int64) - 1
    vbackend.update_mamba_state_after_mtp_verify(
        last_correct_step_indices=last_steps,
        mamba_track_indices=None,
        mamba_steps_to_track=None,
        model=worker.target_worker.model_runner.model,
        req_pool_indices=vfb.req_pool_indices,
    )
    worker._append_target_hidden_to_draft_kv_by_loc(
        target_hidden=vout.hidden_states,
        cache_loc=vfb.out_cache_loc,
        cache_loc_2d=vfb.out_cache_loc.view(3, 8),
        positions=vfb.positions,
        commit_lens=commits,
    )
    worker._append_target_hidden_to_draft_kv_by_loc(
        target_hidden=pout.hidden_states,
        cache_loc=pfb.out_cache_loc,
        positions=pfb.positions,
    )
    return dict(
        accept=owned[0].cpu().tolist(),
        commits=owned[1].cpu().tolist(),
        bonus=owned[2].cpu().tolist(),
        outputs=owned[3].cpu().tolist(),
    )


def _rotate_ring(pool, req_indices, shift):
    state = pool.mamba_cache
    length = state.replayssm_d.shape[-2]
    for name in (
        "replayssm_d",
        "replayssm_k",
        "replayssm_rawv",
        "replayssm_rawk",
        "replayssm_g",
    ):
        tensor = getattr(state, name)
        if tensor is None:
            continue
        axis = -1 if name == "replayssm_g" else -2
        tensor[:, req_indices] = torch.roll(tensor[:, req_indices], shift, dims=axis)
    pool.replayssm_cache_base[req_indices] = (
        pool.replayssm_cache_base[req_indices] + shift
    ) % length


def _region_metrics(first, second, scheduler, pfb, vfb):
    regions = {}
    pslots = (
        scheduler.req_to_token_pool.get_mamba_indices(pfb.req_pool_indices).cpu().long()
    )
    vslots = (
        scheduler.req_to_token_pool.get_mamba_indices(vfb.req_pool_indices).cpu().long()
    )
    for name, (_, indices, x) in first.tensors.items():
        y = second.tensors[name][2]
        if not x.is_floating_point():
            continue
        selections = []
        if name.startswith(("target.", "draft.")):
            loc = indices.cpu()
            selections = [
                ("P", torch.isin(loc, pfb.out_cache_loc.cpu()), 0),
                ("V", torch.isin(loc, vfb.out_cache_loc.cpu()), 0),
            ]
        elif name.startswith(("mamba.conv.", "mamba.temporal.")):
            selections = [("P", pslots, 1), ("V", vslots, 1)]
        elif name.startswith("mamba.replayssm_") and x.ndim > 2:
            selections = [
                ("P", pfb.req_pool_indices.cpu().long(), 1),
                ("V", vfb.req_pool_indices.cpu().long(), 1),
            ]
        for label, select, axis in selections:
            if select.dtype == torch.bool:
                select = select.nonzero().flatten()
            a, b = x.index_select(axis, select), y.index_select(axis, select)
            e = (a.float() - b.float()).abs()
            regions[name + "." + label] = dict(
                max_abs=float(e.max()) if e.numel() else 0,
                rel_l2=float(
                    torch.linalg.vector_norm(e)
                    / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
                ),
            )
    return regions


def _safety_checks(before, after, scheduler, pfb, vfb, decisions):
    checks = {}
    pool = scheduler.req_to_token_pool
    req_ids = torch.cat((pfb.req_pool_indices, vfb.req_pool_indices))
    slots = pool.get_mamba_indices(req_ids)
    tracks = pool.req_index_to_mamba_ping_pong_track_buffer_mapping[req_ids].flatten()
    allowed_slots = torch.cat((slots, tracks)).cpu().long().unique()
    draft_writes = torch.cat(
        [
            pfb.out_cache_loc,
            *[
                row[:n]
                for row, n in zip(vfb.out_cache_loc.view(3, 8), decisions["commits"])
            ],
        ]
    ).cpu()
    target_writes = torch.cat((pfb.out_cache_loc, vfb.out_cache_loc)).cpu()
    for name, (_, indices, x) in before.tensors.items():
        y = after.tensors[name][2]
        if name.startswith("mapping."):
            checks[name + ".unchanged"] = torch.equal(x, y)
        elif name.startswith(("mamba.conv.", "mamba.temporal.")):
            unused = torch.arange(x.shape[1])[
                ~torch.isin(torch.arange(x.shape[1]), allowed_slots)
            ]
            checks[name + ".unused_slots_unchanged"] = torch.equal(
                x.index_select(1, unused), y.index_select(1, unused)
            )
        elif name.startswith("mamba.replayssm_"):
            axis = 1 if x.ndim > 2 else 0
            unused = torch.arange(x.shape[axis])[
                ~torch.isin(torch.arange(x.shape[axis]), vfb.req_pool_indices.cpu())
            ]
            checks[name + ".nonverify_rows_unchanged"] = torch.equal(
                x.index_select(axis, unused), y.index_select(axis, unused)
            )
        elif name.startswith(("target.", "draft.")):
            allowed = draft_writes if name.startswith("draft.") else target_writes
            unused = (~torch.isin(indices.cpu(), allowed)).nonzero().flatten()
            a, b = x.index_select(0, unused), y.index_select(0, unused)
            # Snapshot uninitialized capacity may contain NaNs. Compare raw bits.
            checks[name + ".outside_valid_writes_unchanged"] = torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )
    return checks


@torch.no_grad()
def run(scheduler, config):
    # Worker planning buffers grow lazily. Keep them ordinary tensors so the
    # private scheduler's later warmup/health forwards can update them safely.
    gc.collect()
    torch.cuda.empty_cache()
    if config.get("numerical_probe"):
        import mixed_eager_numerics

        importlib.reload(mixed_eager_numerics)
        return mixed_eager_numerics.run(scheduler, config)
    if config.get("stable_marlin_candidate"):
        from sglang.srt.speculative import mixed_marlin

        importlib.reload(mixed_marlin)
    if config.get("mixed_graph_candidate"):
        assert config.get("stable_marlin_candidate") is True
        gate_files = config.get(
            "eager_gate_files",
            [
                str(Path(config["root"]) / (name + "-report.json"))
                for name in ["stable-full-natural", "stable-full-rejection"]
            ],
        )
        assert len(gate_files) == 2
        sources = source_hashes(Path(__file__).resolve().parents[3])
        for rejection, path in zip((False, True), gate_files):
            gate = json.loads(Path(path).read_text())
            validate_eager_gate(gate, expected_sources=sources, rejection=rejection)
        from sglang.srt.speculative import mixed_graph

        importlib.reload(mixed_graph)
    root = Path(config["root"])
    mr, worker = scheduler.tp_worker.model_runner, scheduler.draft_worker
    assert worker.block_size == 8 and len(scheduler.running_batch.reqs) == 0
    assert not worker.use_compact_draft_cache, (
        "Add compact draft snapshot coverage before using it"
    )
    assert mr.ps.tp_size == mr.ps.pp_size == mr.ps.attn_dp_size == 1
    assert mr.decode_cuda_graph_runner is not None
    assert worker.draft_model_runner.decode_cuda_graph_runner is not None
    pbackend = get_attention_backend(model_runner=mr, init_new_workspace=True)
    vbackend = get_attention_backend(model_runner=mr, init_new_workspace=True)
    seed_text = "def solve_problem(values):\n    result = []\n    for value in values:\n        if value > 0:\n            result.append(value * 2)\n    return result\n\n"
    tokens = scheduler.tokenizer.encode(seed_text * 250)
    dreqs = [
        _new_req(scheduler, f"offline-D{i}", tokens[i : 128 + i]) for i in range(3)
    ]
    dbatch = _prepare(scheduler, dreqs)
    with scheduler._isolate_forward_fields(dbatch):
        initial = worker.forward_batch_generation(dbatch)
    for req, token in zip(dreqs, initial.next_token_ids.cpu().tolist()):
        req.output_ids.append(token)
        req.kv.kv_committed_len = 128
    dbatch.spec_info = initial.next_draft_input
    preq = _new_req(scheduler, "offline-P", tokens[:12000])
    lengths = config.get("chunks", [23, 40, 65, 128, 257, 430])
    capacities = config.get("capacities", [32, 64, 128, 128, 512, 512])
    rounds = config.get("rounds", len(lengths))
    replay = None
    if config.get("proposals_file"):
        proposal_report = json.loads(Path(config["proposals_file"]).read_text())
        digest = validate_proposals(proposal_report, rounds=rounds)
        if config.get("proposals_sha256", digest) != digest:
            raise ValueError("Proposal source changed after experiment preparation")
        replay = proposal_report["cases"]
    reports = []
    graph_cache = []
    original = worker.target_worker.forward_batch_generation
    draft_run = worker.draft_model_runner.forward
    draft_graph_hits = []

    def audited_draft(*args, **kwargs):
        value = draft_run(*args, **kwargs)
        draft_graph_hits.append(bool(value.can_run_graph))
        return value

    worker.draft_model_runner.forward = audited_draft
    prefix = 0
    try:
        for turn in range(rounds):
            count, capacity = (
                lengths[turn % len(lengths)],
                capacities[turn % len(capacities)],
            )
            preq.prefix_indices = (
                scheduler.req_to_token_pool.req_to_token[preq.kv.req_pool_idx, :prefix]
                .to(torch.int64)
                .clone()
                if prefix
                else torch.empty(0, dtype=torch.int64, device=mr.device)
            )
            preq.set_extend_range(prefix, prefix + count)
            pbatch = _prepare(scheduler, [preq])
            pfb = ForwardBatch.init_new(
                pbatch,
                mr,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                return_hidden_states_before_norm=False,
            )
            mr._maybe_execute_deferred_mamba_cow_and_clear(pfb)
            if turn in config.get("rotate_rounds", [2]):
                _rotate_ring(
                    scheduler.req_to_token_pool.mamba_pool, dbatch.req_pool_indices, 29
                )
            dbatch.prepare_for_decode()
            draft_input = dbatch.spec_info
            turn_report = dict(
                turn=turn, p_prefix=prefix, p_tokens=count, p_capacity=capacity
            )

            def intercepted(*args, **kwargs):
                vfb = kwargs.get("forward_batch")
                if vfb is None or not kwargs.get("is_verify"):
                    return original(*args, **kwargs)
                assert vfb.batch_size == 3 and len(vfb.input_ids) == 24
                if replay is not None:
                    proposals = vfb.input_ids.view(3, 8)
                    turn_report["unmodified_draft_proposals"] = proposals.cpu().tolist()
                    saved = proposals.new_tensor(replay[turn]["proposal_ids"])
                    assert torch.equal(saved[:, 0], proposals[:, 0]), (
                        "Replayed anchors must match the native committed history"
                    )
                    proposals.copy_(saved)
                    mr.decode_cuda_graph_runner.load_batch(vfb)
                elif config.get("rejection_fixture"):
                    proposals = vfb.input_ids.view(3, 8)
                    turn_report["unmodified_draft_proposals"] = proposals.cpu().tolist()
                    proposals[0, 1] = (
                        proposals[0, 1] + 47
                    ) % scheduler.model_config.vocab_size
                    proposals[1, 4] = (
                        proposals[1, 4] + 47
                    ) % scheduler.model_config.vocab_size
                    mr.decode_cuda_graph_runner.load_batch(vfb)
                ids = torch.cat((pfb.req_pool_indices, vfb.req_pool_indices))
                allocated = [
                    scheduler.req_to_token_pool.req_to_token[
                        r.kv.req_pool_idx, : r.kv.kv_allocated_len
                    ]
                    for r in [preq, *dreqs]
                ]
                locations = torch.cat(
                    [torch.zeros(1, device=mr.device, dtype=torch.int64), *allocated]
                ).unique()
                before = StateSnapshot(scheduler, locations)
                turn_report["proposal_ids"] = vfb.input_ids.view(3, 8).cpu().tolist()
                turn_report["request_rows"] = ids.cpu().tolist()
                pool = scheduler.req_to_token_pool.mamba_pool
                turn_report["ring_before"] = dict(
                    base=pool.replayssm_cache_base.cpu().tolist(),
                    write=pool.replayssm_spec_write_pos.cpu().tolist(),
                )
                rid = vfb.req_pool_indices.cpu().long()
                bases = pool.replayssm_cache_base.cpu()[rid]
                writes = pool.replayssm_spec_write_pos.cpu()[rid]
                turn_report["ring_read_wrap"] = bool(((bases + writes) > 32).any())
                turn_report["ring_write_wrap"] = bool(
                    (((bases + writes) % 32 + 8) > 32).any()
                )
                try:
                    native = original(*args, **kwargs)
                    turn_report["native_verify_graph"] = bool(native.can_run_cuda_graph)
                    native_logits = native.logits_output.next_token_logits.clone()
                    native_accept = worker._accept_block(
                        candidates=vfb.input_ids.view(3, 8),
                        next_token_logits=native_logits,
                        sampling_info=dbatch.sampling_info,
                        draft_input=draft_input,
                        prefix_lens=vfb.positions.view(3, 8)[:, 0],
                        bs=3,
                    )
                    turn_report["native_decisions"] = dict(
                        accept=native_accept[0].cpu().tolist(),
                        commits=native_accept[1].cpu().tolist(),
                        bonus=native_accept[2].cpu().tolist(),
                    )
                    del native_logits
                    branches = []
                    features = []
                    safety = []
                    for mixed in (
                        (False, False, True)
                        if config.get("repeat_control")
                        else (False, True)
                    ):
                        before.restore()
                        pbackend.init_forward_metadata(pfb)
                        vbackend.init_forward_metadata(vfb)
                        started = time.perf_counter()
                        with _capture_features(
                            mr.model,
                            config.get("trace_features")
                            or config.get("freeze_token_local_diagnostic"),
                        ) as values:
                            commit_backend = vbackend
                            if mixed and config.get("mixed_graph_candidate"):
                                reused = False
                                graph = None
                                for candidate in graph_cache:
                                    if (
                                        candidate.layout.prefill_capacity != capacity
                                        or candidate.layout.prefill_tokens != count
                                    ):
                                        continue
                                    try:
                                        outputs = candidate.replay(pfb, vfb)
                                        graph = candidate
                                        reused = True
                                        break
                                    except mixed_graph.MixedGraphLayoutChanged:
                                        pass
                                if graph is None:
                                    graph = mixed_graph.MixedTargetGraph(
                                        mr,
                                        pfb,
                                        vfb,
                                        prefill_capacity=capacity,
                                        restore=before.restore,
                                    )
                                    graph_cache.append(graph)
                                    outputs = graph.replay(pfb, vfb)
                                pout, vout, calls = outputs
                                commit_backend = graph.verify_backend
                                turn_report["mixed_graph"] = dict(
                                    replayed=True,
                                    reused=reused,
                                    captures=len(graph_cache),
                                    instance_replays=graph.replays,
                                    P_valid=count,
                                    P_capacity=capacity,
                                    verify_tokens=24,
                                    original_prefill_ragged=pbackend.full_attn_backend.forward_metadata.use_ragged,
                                    graph_prefill_ragged=graph.prefill_backend.full_attn_backend.forward_metadata.use_ragged,
                                    prefill_tracks=bool(
                                        graph.pinned_metadata[0].has_mamba_track_mask
                                    ),
                                    prefill_state_slots=graph.pinned_metadata[0]
                                    .mamba_cache_indices.cpu()
                                    .tolist(),
                                    verify_state_slots=graph.pinned_metadata[1]
                                    .mamba_cache_indices.cpu()
                                    .tolist(),
                                )
                            else:
                                pout, vout, calls = _models(
                                    mixed,
                                    config,
                                    mr,
                                    pfb,
                                    vfb,
                                    pbackend,
                                    vbackend,
                                    capacity,
                                    frozen=features[0]
                                    if mixed
                                    and config.get("freeze_token_local_diagnostic")
                                    else None,
                                )
                        features.append(values)
                        if mixed:
                            turn_report["mixed_attention_calls"] = calls
                        decisions = _accept_and_commit(
                            worker,
                            dbatch,
                            draft_input,
                            pfb,
                            vfb,
                            pout,
                            vout,
                            commit_backend,
                        )
                        snapshot = StateSnapshot(scheduler, locations)
                        safety.append(
                            _safety_checks(
                                before, snapshot, scheduler, pfb, vfb, decisions
                            )
                        )
                        branches.append((decisions, snapshot))
                        turn_report["mixed" if mixed else "separate"] = dict(
                            decisions=decisions,
                            diagnostic_wall_s=time.perf_counter() - started,
                        )
                        del pout, vout
                    turn_report["state_comparison"] = branches[0][1].compare(
                        branches[-1][1]
                    )
                    turn_report["region_metrics"] = _region_metrics(
                        branches[0][1], branches[-1][1], scheduler, pfb, vfb
                    )
                    if config.get("trace_features") and not config.get(
                        "freeze_token_local_diagnostic"
                    ):
                        turn_report["feature_comparison"] = _compare_features(
                            features[0], features[-1], count, capacity
                        )
                    if len(branches) == 3:
                        turn_report["repeat_control_metrics"] = branches[0][1].compare(
                            branches[1][1]
                        )
                    turn_report["state_mismatches"] = {
                        k: v
                        for k, v in turn_report["state_comparison"].items()
                        if v["mismatch"]
                    }
                    turn_report["safety_checks"] = safety
                    turn_report["safety_passed"] = all(
                        all(row.values()) for row in safety
                    )
                    turn_report["acceptance_equal"] = branches[0][0] == branches[-1][0]
                    turn_report["native_acceptance_equal"] = all(
                        branch[0][key] == turn_report["native_decisions"][key]
                        for branch in branches
                        for key in ("accept", "commits", "bonus")
                    )
                    if not config.get("continue_on_mismatch"):
                        assert turn_report["acceptance_equal"], (
                            "Separate/Mixed acceptance mismatch"
                        )
                        assert turn_report["native_acceptance_equal"], (
                            "Native graph/eager acceptance mismatch"
                        )
                        assert not turn_report["state_mismatches"], (
                            "Separate/Mixed state mismatch"
                        )
                        assert turn_report["safety_passed"], (
                            "Padding/ownership/write-boundary violation"
                        )
                    if config.get("benchmark_trials"):
                        if not config.get("mixed_graph_candidate"):
                            raise ValueError("Performance measurement requires Mixed Graph")
                        from bench_mixed_full_model import benchmark_target

                        turn_report["target_benchmark"] = benchmark_target(
                            model_runner=mr,
                            graph=graph,
                            prefill_batch=pfb,
                            verify_batch=vfb,
                            prefill_backend=pbackend,
                            native_verify=lambda: original(*args, **kwargs),
                            restore=before.restore,
                            snapshot=lambda: StateSnapshot(scheduler, locations),
                            trials=config["benchmark_trials"],
                        )
                finally:
                    before.restore()
                    # Replay metadata after the independent backend experiments.
                    mr.decode_cuda_graph_runner.load_batch(vfb)
                return original(*args, **kwargs)

            worker.target_worker.forward_batch_generation = intercepted
            try:
                with scheduler._isolate_forward_fields(dbatch):
                    result = worker.forward_batch_generation(dbatch)
                _advance(dbatch, result)
                with scheduler._isolate_forward_fields(pbatch):
                    worker.target_worker.forward_batch_generation = original
                    p_result = worker.forward_batch_generation(pbatch)
                preq.kv.kv_committed_len = prefix + count
                prefix += count
                reports.append(turn_report)
                (root / "progress.json").write_text(json.dumps(reports, indent=2))
            except BaseException:
                reports.append(turn_report)
                (root / "progress.json").write_text(json.dumps(reports, indent=2))
                raise
            finally:
                worker.target_worker.forward_batch_generation = original
        return dict(
            cases=reports,
            draft_graph_hits=draft_graph_hits,
            gate_passed=not config.get("freeze_token_local_diagnostic")
            and not config.get("fixed_reduction_diagnostic")
            and not config.get("fp32_router_diagnostic")
            and len(reports) == rounds > 0
            and bool(draft_graph_hits)
            and all(draft_graph_hits)
            and all(
                r["acceptance_equal"]
                and r["native_acceptance_equal"]
                and r["native_verify_graph"]
                and not any(
                    v["mismatch"] for v in r.get("repeat_control_metrics", {}).values()
                )
                and not r["state_mismatches"]
                and r["safety_passed"]
                and (
                    not config.get("mixed_graph_candidate")
                    or r.get("mixed_graph", {}).get("replayed", False)
                )
                for r in reports
            ),
            tolerances=dict(rtol=0.02, atol=0.002),
            formal_serving_changed=False,
        )
    finally:
        worker.target_worker.forward_batch_generation = original
        worker.draft_model_runner.forward = draft_run
