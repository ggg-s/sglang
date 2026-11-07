"""Small CUDA regression for the Qwen3.8 PDMux model loop (no model weights).

Loads production loop bodies through the CPU fixture; substitutes only GPU-layer
collaborators and test infrastructure. Uses real CUDA tensors on two streams.
This does not test the real GDN/QSA/FP8 kernels or model accuracy/throughput.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "test/registered/unit/models/test_qwen38_pdmux.py"


def load_fixture():
    # Avoid importing the complete server dependency tree for a tensor fixture.
    # Production code is loaded from the snapshot by its unmodified load_body.
    tree = ast.parse(FIXTURE.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("sglang.test")
        )
        and not isinstance(node, ast.If)
    ]
    namespace = {
        "__file__": str(FIXTURE),
        "__name__": "qwen38_cuda_fixture",
        "CustomTestCase": unittest.TestCase,
        "register_cpu_ci": lambda **kw: None,
    }
    exec(compile(ast.fix_missing_locations(tree), str(FIXTURE), "exec"), namespace)
    return namespace["TestQwen38Pdmux"]()


def run_case(fixture, *, tokens, quantum, rounds):
    model, forward, journal = fixture.fixture()
    ids, positions = torch.arange(tokens), torch.arange(tokens)
    expected = forward(model, ids, positions, fixture.batch("reference"))
    decode_ids, decode_positions = torch.tensor([17]), torch.tensor([100])
    decode_reference = forward(
        model, decode_ids, decode_positions, fixture.batch("reference_decode")
    )
    ready = torch.cuda.current_stream().record_event()
    prefill_stream, decode_stream = torch.cuda.Stream(), torch.cuda.Stream()
    prefill_stream.wait_event(ready)
    decode_stream.wait_event(ready)
    prefill_outputs, decode_outputs = [], []
    began = time.perf_counter()
    for iteration in range(rounds):
        batch = fixture.batch(f"prefill-{iteration}")
        for start in range(0, 4, quantum):
            # Separate asynchronous lanes; no cross-lane synchronization until
            # validation. Each submitting host call still completes its scope.
            with torch.cuda.stream(prefill_stream):
                result = forward(
                    model,
                    ids,
                    positions,
                    batch,
                    split_interval=(start, min(4, start + quantum)),
                )
            with torch.cuda.stream(decode_stream):
                decode = forward(
                    model, decode_ids, decode_positions, fixture.batch("decode")
                )
                decode_outputs.append(decode)
            if start + quantum < 4:
                assert result is None
                # Pressure the caching allocator while the prefill's per-batch
                # intermediate state must remain live across a decode yield.
                with torch.cuda.stream(decode_stream):
                    torch.empty((256, 256)).fill_(iteration)
        prefill_outputs.append(result)
        assert batch.residual_stream is None
        assert not batch.model_specific_states
    prefill_stream.synchronize()
    decode_stream.synchronize()
    for result in prefill_outputs:
        for actual, reference in zip(result, expected):
            assert actual.is_cuda
            assert actual.dtype == torch.get_default_dtype()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for result in decode_outputs:
        for actual, reference in zip(result, decode_reference):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for iteration in range(rounds):
        assert [event for tag, event in journal if tag == f"prefill-{iteration}"] == [
            "prepare",
            "commit",
        ]
    return {
        "dtype": str(torch.get_default_dtype()),
        "tokens": tokens,
        "quantum": quantum,
        "rounds": rounds,
        "decode_forwards": len(decode_outputs),
        "exact_match": True,
        "harness_seconds": time.perf_counter() - began,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=32)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(0)
    torch.set_default_device("cuda")
    report = {
        "scope": "production model loop with small tensor-layer collaborators",
        "not_validated": [
            "full model",
            "real GDN/QSA/FP8 operators",
            "answer accuracy",
            "serving performance",
            "multi-GPU TP",
        ],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cases": [],
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [ROOT / "python/sglang/srt/models/qwen4_exp.py", FIXTURE]
        },
    }
    fixture = load_fixture()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            torch.set_default_dtype(dtype)
            for tokens in (1, 65, 513, 2048):
                for quantum in (1, 2, 4):
                    case = run_case(
                        fixture, tokens=tokens, quantum=quantum, rounds=args.rounds
                    )
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
        # Remaining CPU fixture tests now allocate real CUDA tensors as well.
        fixture.test_fp8_backend_policy_overrides_auto_and_rejects_shared_scratch()
        fixture.test_ple_prefetch_does_not_touch_another_lanes_pending_state()
        fixture.test_split_wrapper_preserves_mtp_hyperconnection_hidden_states()
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
