"""Write an official SGLang native/hybrid benchmark plan (sa-spec, 2048 cap)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--draft-model-path")
    parser.add_argument(
        "--algorithm", choices=("EAGLE", "EAGLE3", "STANDALONE"), default="EAGLE3"
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--num-drafts", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--port", type=int, default=39921)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    parser.add_argument("--chunked-prefill-size", type=int)
    args = parser.parse_args()
    if min(args.tp_size, args.num_drafts, args.repeats, *args.concurrencies) < 1:
        parser.error("TP size, draft count, repeats and concurrencies must be positive")
    if not 0 < args.mem_fraction_static < 1 or not 0 < args.port < 65536:
        parser.error("invalid memory fraction or port")
    if args.algorithm != "EAGLE" and not args.draft_model_path:
        parser.error("EAGLE3 and STANDALONE require --draft-model-path")
    data = args.data_dir.resolve()
    provenance = json.loads((data / "provenance.json").read_text(encoding="utf-8"))
    prompts_path = data / "sa-spec-prompts.json"
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if provenance.get("template") != "sa-spec" or not prompts:
        parser.error("generate data with --template sa-spec first")
    if provenance["model_path"] != args.model_path:
        parser.error("use the same --model-path for data preparation and serving")
    rows = provenance["prompts"]
    if len(rows) != len(prompts) or any(
        hashlib.sha256(prompt.encode()).hexdigest() != row["prompt_sha256"]["sa-spec"]
        for prompt, row in zip(prompts, rows)
    ):
        parser.error("prompt contents differ from provenance; regenerate the data")
    dataset_path = data / "sa-spec-sglang.jsonl"
    if not dataset_path.is_file():
        parser.error("missing official JSONL; regenerate data with --template sa-spec")
    dataset = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [
        {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": ""},
            ]
        }
        for prompt in prompts
    ]
    if dataset != expected:
        parser.error(
            "official JSONL differs from original prompts; regenerate the data"
        )
    required = max(row["prompt_tokens"]["sa-spec"] for row in rows) + 2048
    if args.context_length < required:
        parser.error(f"--context-length must be at least {required} (prompt + 2048)")
    max_requests = max(args.concurrencies)
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--language-only",
        "--speculative-algorithm",
        args.algorithm,
        "--speculative-num-steps",
        str(args.num_drafts),
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        str(args.num_drafts + 1),
        "--tp-size",
        str(args.tp_size),
        "--context-length",
        str(args.context_length),
        "--max-running-requests",
        str(max_requests),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--random-seed",
        "42",
        "--watchdog-timeout",
        "1800",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if args.draft_model_path:
        command += ["--speculative-draft-model-path", args.draft_model_path]
    if args.disable_overlap_schedule:
        command += ["--disable-overlap-schedule"]
    if args.chunked_prefill_size is not None:
        command += ["--chunked-prefill-size", str(args.chunked_prefill_size)]
    config = dict(
        num_drafts=args.num_drafts,
        min_match_len=4,
        max_requests=max_requests,
        max_history_tokens=args.context_length,
        backend="gpu_sa",
        history_mode="device",
        partial_drafts="zero",
    )
    servers = []
    for repeat in range(1, args.repeats + 1):
        # Alternate ordering between repetitions to expose timing/order effects.
        for name in ("native", "hybrid") if repeat % 2 else ("hybrid", "native"):
            server = dict(
                name=f"{name}-r{repeat}",
                url=f"http://127.0.0.1:{args.port}",
                command=command,
            )
            if name == "hybrid":
                server["hybrid_config"] = config
            servers.append(server)
    plan = dict(
        benchmark="sglang",
        model=args.model_path,
        seed=42,
        servers=servers,
        startup_timeout=1800,
        concurrencies=args.concurrencies,
        workloads=[
            dict(
                name="sa-spec-2048",
                dataset=str(dataset_path),
                num_prompts=len(prompts),
                max_tokens=2048,
                context_length=args.context_length,
            )
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        f"Plan: {args.output}; {len(prompts)} prompts; output cap 2048; K={args.num_drafts}"
    )


if __name__ == "__main__":
    main()
