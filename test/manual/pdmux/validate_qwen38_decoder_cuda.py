"""Small real Qwen4-Exp decoder / TileLang MQA CUDA-stream regression.

Run only in an isolated SM89 environment; see the companion acceptance report.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

import validate_qwen38_real_ops_cuda as ops


def mqa_pair(mqa, dtype, relative=False):
    heads, dim, page_size = 8, 128, 16
    cache = (torch.randn(120, page_size, 1, dim, device="cuda") * 0.2).to(dtype)
    k = cache[:96].reshape(-1, 1, dim)
    q = (torch.randn(65, heads, dim, device="cuda") * 0.2).to(dtype)
    # One row tile straddles requests; exercise nonzero starts and tail tiles.
    starts = torch.where(torch.arange(65, device="cuda") < 33, 3, 701).int()
    lengths = torch.arange(65, device="cuda").int() * 3 + 65
    ends = starts + lengths
    width = 257
    valid = torch.arange(width, device="cuda")[None, :] < lengths[:, None]
    reference = (
        mqa.torch_qsa_mqa_prefill_relative(q, k, starts, ends, width)
        if relative
        else mqa.torch_qsa_mqa_prefill(q, k, starts, ends)
    )

    def prefill():
        if relative:
            out = mqa.tilelang_qsa_mqa_prefill_relative(q, k, starts, ends, width)
            # Relative output outside each request's span is undefined by API.
            return (out.masked_fill(~valid, -float("inf")),)
        return (mqa.tilelang_qsa_mqa_prefill(q, k, starts, ends),)

    dq = (torch.randn(3, heads - 2, dim, device="cuda") * 0.2).to(dtype)
    table = torch.arange(96, 120, device="cuda", dtype=torch.int32).view(3, 8)
    lens = torch.tensor([127, 65, 1], device="cuda", dtype=torch.int32)
    dref = mqa.torch_qsa_mqa_decode(dq, cache, table, lens, 131)

    def decode():
        return (mqa.tilelang_qsa_mqa_decode(dq, cache, table, lens, 131),)

    def refs(p, d):
        # Avoid inf-inf in numeric-error reporting; compare masks independently.
        values = []
        for actual, expected in ((p[0], reference), (d[0], dref)):
            torch.testing.assert_close(torch.isneginf(actual), torch.isneginf(expected))
            finite = torch.isfinite(expected)
            values.append((actual[finite], expected[finite], 0.003))
        return values

    return prefill, decode, refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use kernel API logging configured by SGLANG_KERNEL_API_*.",
    )
    parser.add_argument("--only", choices=("mqa", "decoder", "all"), default="all")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    torch.manual_seed(147)
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("This standalone bootstrap is explicitly for SM89")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cases": [],
        "errors": [],
    }
    if args.only in ("mqa", "all"):
        mqa = ops.load(
            "real_qsa_mqa",
            "python/sglang/srt/layers/attention/qsa/mqa.py",
            is_hip=lambda: False,
        )
        if not mqa.HAS_TILELANG:
            raise RuntimeError("TileLang is required; no Torch fallback is accepted")
        if args.debug:
            api = ops.load(
                "qwen38_kernel_api", "python/sglang/kernels/kernel_api_logging.py"
            )
            for name in (
                "tilelang_qsa_mqa_prefill",
                "tilelang_qsa_mqa_prefill_relative",
                "tilelang_qsa_mqa_decode",
            ):
                setattr(
                    mqa, name, api.debug_kernel_api(getattr(mqa, name), op_name=name)
                )
        report["tilelang"] = mqa.tilelang.__version__
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for relative in (False, True):
                name = f"tilelang_mqa_{dtype}_{'relative' if relative else 'absolute'}"
                try:
                    result = ops.pair_test(
                        name, *mqa_pair(mqa, dtype, relative), args.rounds
                    )
                    report["cases"].append(result)
                    print(json.dumps(result), flush=True)
                except Exception as exc:
                    traceback.print_exc()
                    report["errors"].append(
                        {"name": name, "error": f"{type(exc).__name__}: {exc}"}
                    )
    if args.only in ("decoder", "all"):
        from qwen38_small_decoder import run_decoder

        try:
            result = run_decoder(args.rounds)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
        except Exception as exc:
            traceback.print_exc()
            report["errors"].append(
                {
                    "name": "production_decoder_stack",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    report["source_sha256"] = ops.SOURCES
    report["passed"] = not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
