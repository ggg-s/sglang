"""Bounded, intrusive same-state audit; never use its service TPOT as a benchmark.

All proposals still take native target verification. Extra draft replays only
write speculative draft KV beyond the committed prefix. The final production
replay restores the original mode before verify. Raw neural columns two/three
are observed under the SA prefix: they test a longer gate, not an independent
neural continuation after its first disagreement.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

RAW = {}
TRACE = {}
ENABLED = False


def raw_topk(
    result, next_token_logits, positions, draft_tokens=None, draft_token_column=0
):
    if draft_tokens is not None and draft_token_column in (1, 2):
        import torch

        key = draft_tokens.data_ptr()
        if key not in RAW:
            RAW[key] = torch.empty_like(draft_tokens)
        RAW[key][:, draft_token_column : draft_token_column + 1].copy_(result[1])


def counters():
    return {
        "prefix_ct": 0,
        "request_ct": 0,
        "skip_request_ct": 0,
        "tail_request_ct": 0,
        "mixed_ct": 0,
        "decision_wait_ns": 0,
        "tail_ct": 0,
        "skip_ct": 0,
        "tail_batch_histogram": {},
    }


def replay(original, backend, shape_key, static_forward_batch, **kwargs):
    # Query/feedback kernels update persistent SA state. Replaying them is not
    # a snapshot/restore protocol, even without another verify. Execute this
    # prefix once, then restore its candidate output before each tail replay.
    record = backend.records[shape_key]
    prefix = record.prefix
    baseline = None

    def frozen_prefix():
        nonlocal baseline
        if baseline is None:
            prefix.replay()
            baseline = record.output[2].clone()
        else:
            record.output[2].copy_(baseline)

    record.prefix = SimpleNamespace(replay=frozen_prefix)
    try:
        return _replay_audit(
            original, backend, shape_key, static_forward_batch, **kwargs
        )
    finally:
        record.prefix = prefix


def _replay_audit(original, backend, shape_key, static_forward_batch, **kwargs):
    global ENABLED
    if not ENABLED:
        ENABLED = Path(os.environ["SPLIT_AUDIT_ENABLE"]).exists()
    bs = backend.runner.raw_bs
    extension = backend.runner.eagle_worker.draft_extension
    counts = getattr(extension, "_audit_counts", {})
    seen = getattr(extension, "_audit_seen", {})
    seen[bs] = seen.get(bs, 0) + 1
    extension._audit_seen = seen
    limit = int(os.environ.get("SPLIT_AUDIT_LIMIT", "32"))
    if not ENABLED or bs not in (1, 4) or counts.get(bs, 0) >= limit or seen[bs] % 2:
        return original(backend, shape_key, static_forward_batch, **kwargs)

    import torch
    from hybrid_spec.integrations.sglang.compact_graph import replay_compact

    record = backend.records[shape_key]
    assert record.conditional is None
    assert record.prefix_width == 1
    assert extension.config.neural_drafts == extension.config.num_drafts == 8
    assert not extension.config.dynamic_verify
    old = record.no_skip
    measurements = []
    candidates, masks = [], []
    raw = None
    seq_lens = record.inputs.seq_lens[:bs].clone()
    request_rows = record.inputs.req_pool_indices[:bs].clone()
    try:
        # Consume each feedback exactly once before the measured stable replays.
        record.no_skip = False
        replay_compact(record, counters(), bs)
        for repeat in range(3):
            order = (False, True) if (repeat + counts.get(bs, 0)) % 2 else (True, False)
            for no_skip in order:
                record.no_skip = no_skip
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                output = replay_compact(record, counters(), bs)
                end.record()
                candidates.append(output[2][:bs].clone())
                masks.append(record.override.use_retrieval[:bs].clone())
                if no_skip:
                    child = record.tails[bs]
                    buffer = RAW[child.staging[2].data_ptr()]
                    raw = buffer[:bs, 1:3].clone()
                measurements.append(("no_skip" if no_skip else "split", start, end))
        torch.cuda.synchronize()
        expected, mask = candidates[0], masks[0]
        if not torch.equal(record.inputs.seq_lens[:bs], seq_lens) or not torch.equal(
            record.inputs.req_pool_indices[:bs], request_rows
        ):
            raise RuntimeError("tail replay overwrote outer request mapping")
        if not all(torch.equal(item, expected) for item in candidates[1:]):
            failure = {
                "batch_size": bs,
                "seq_lens": seq_lens.cpu().tolist(),
                "modes": [mode for mode, _, _ in measurements],
                "candidates": [item.cpu().tolist() for item in candidates],
                "masks": [item.cpu().tolist() for item in masks],
                "sa": record.override.tokens[:bs].cpu().tolist(),
                "trace": {
                    str(k): [
                        {name: value.cpu().tolist() for name, value in step.items()}
                        for step in steps
                    ]
                    for k, steps in TRACE.items()
                },
                "raw_neural_2_3": raw.cpu().tolist(),
            }
            Path(os.environ["SPLIT_AUDIT_OUTPUT"]).with_name(
                "candidate-mismatch.json"
            ).write_text(json.dumps(failure, indent=2))
            raise RuntimeError("same-state split changed candidate tokens")
        if not all(torch.equal(item, mask) for item in masks[1:]):
            raise RuntimeError("same-state replay changed source mask")
        sa = record.override.tokens[:bs]
        gate2 = raw[:, 0] == sa[:, 1]
        gate3 = gate2 & (raw[:, 1] == sa[:, 2])
        entry = {
            "snapshot": counts.get(bs, 0),
            "batch_size": bs,
            "seq_lens": seq_lens.cpu().tolist(),
            "mask": mask.cpu().tolist(),
            "gate2": gate2.cpu().tolist(),
            "gate3": gate3.cpu().tolist(),
            "raw_neural_2_3": raw.cpu().tolist(),
            "candidates": expected.cpu().tolist(),
            "equal_candidates": True,
            "equal_masks": True,
            "measurements": [
                {"mode": mode, "gpu_ms": a.elapsed_time(b)}
                for mode, a, b in measurements
            ],
        }
        extension._audit_pending = entry
        counts[bs] = counts.get(bs, 0) + 1
        extension._audit_counts = counts
    finally:
        record.no_skip = old
    result = original(backend, shape_key, static_forward_batch, **kwargs)
    if not torch.equal(result[2][:bs], expected):
        raise RuntimeError("final native replay changed audited candidates")
    return result


def verify(result, worker, batch, *args, **kwargs):
    if batch.forward_mode.is_idle():
        return
    extension = worker.draft_worker.draft_extension
    entry = getattr(extension, "_audit_pending", None)
    if entry is None:
        return
    extension._audit_pending = None
    entry["accept_lens"] = result.accept_lens.cpu().tolist()
    entry["rids"] = [req.rid for req in batch.reqs]
    # At SA's first rejection, target's bonus identifies the correct next
    # greedy token. A neural disagreement at that same position can therefore
    # be classified without running a second target verification.
    indices = result.accept_index.gather(1, (result.accept_lens.long() - 1)[:, None])
    entry["bonus_tokens"] = (
        result.next_token_ids.flatten()[indices.flatten().long()].cpu().tolist()
    )
    with Path(os.environ["SPLIT_AUDIT_OUTPUT"]).open("a") as out:
        out.write(json.dumps(entry) + "\n")


def register():
    from hybrid_spec.integrations.sglang import compact_graph

    # hybrid_spec's adapters are plugin-owned and outside HookRegistry's sglang
    # target namespace. Patch only this diagnostic process, before capture.
    from hybrid_spec.integrations.sglang.split_graph import SplitDraftGraphBackend
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    capture_tail = compact_graph._capture_tail

    def trace_tail(runner, record, size, stream, **kwargs):
        forward = runner.model_runner.forward
        observations = []

        def trace_forward(batch, *args, **kw):
            attention = get_attn_backend()
            step = {
                "input": batch.input_ids.clone(),
                "positions": batch.positions.clone(),
                "rows": record.rows.clone(),
                "original_seq": record.inputs.seq_lens.clone(),
                "original_reqs": record.inputs.req_pool_indices.clone(),
                "seq_lens": batch.seq_lens.clone(),
                "reqs": batch.req_pool_indices.clone(),
                "cache": batch.out_cache_loc.clone(),
                "indptr": attention.forward_metadata.kv_indptr.clone(),
            }
            # Copy only used KV entries; the upper bound buffer may be much larger.
            step["indices"] = attention.forward_metadata.kv_indices[:4096].clone()
            result = forward(batch, *args, **kw)
            values, indices = result.logits_output.next_token_logits.topk(4, dim=-1)
            step["logits_top4"] = values.clone()
            step["tokens_top4"] = indices.clone()
            observations.append(step)
            return result

        runner.model_runner.forward = trace_forward
        try:
            result = capture_tail(runner, record, size, stream, **kwargs)
            TRACE[(record.inputs.batch_size, size)] = observations[-7:]
            ptrs = {
                name: [
                    getattr(record.inputs, name).data_ptr(),
                    getattr(result.batch, name).data_ptr(),
                ]
                for name in ("seq_lens", "req_pool_indices", "positions")
            }
            ptrs["rows"] = record.rows.data_ptr()
            with (
                Path(os.environ["SPLIT_AUDIT_OUTPUT"])
                .with_name("pointers.jsonl")
                .open("a") as out
            ):
                out.write(
                    json.dumps(
                        {"batch": record.inputs.batch_size, "size": size, "ptrs": ptrs}
                    )
                    + "\n"
                )
            return result
        finally:
            runner.model_runner.forward = forward

    compact_graph._capture_tail = trace_tail

    native = SplitDraftGraphBackend.replay
    SplitDraftGraphBackend.replay = lambda self, *a, **kw: replay(
        native, self, *a, **kw
    )
    HookRegistry.register(
        "sglang.kernels.ops.speculative.topk1.draft_topk1_postprocess",
        raw_topk,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2.verify",
        verify,
        HookType.AFTER,
    )
