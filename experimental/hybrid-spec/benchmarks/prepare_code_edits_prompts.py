"""Prepare paired diff/full-file code-edit workloads without answer leakage."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=7168)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    if args.num_prompts < 1:
        parser.error("num-prompts must be positive")
    raw = args.dataset.read_bytes()
    rows = json.loads(raw)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompts = {"diff": [], "rewrite": []}
    provenance = []
    for index, row in enumerate(rows):
        # The reference solution is deliberately never included in the input.
        code, edit = row["code"], row["edit"]
        rendered, lengths = {}, {}
        for name, instruction in (
            ("diff", "Return only a unified diff, without an explanation."),
            (
                "rewrite",
                "Return the complete updated source code. Preserve unaffected code. "
                "Return code only, without an explanation.",
            ),
        ):
            messages = [
                {"role": "system", "content": "You are a coding assistant."},
                {
                    "role": "user",
                    "content": f"Edit request:\n{edit}\n\nOriginal code:\n```\n{code}\n```\n\n{instruction}",
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            rendered[name] = prompt
            lengths[name] = len(tokenizer.encode(prompt, add_special_tokens=False))
        if max(lengths.values()) > args.max_prompt_tokens:
            continue
        for name in prompts:
            prompts[name].append(rendered[name])
        provenance.append(
            dict(
                row_index=index,
                source_sha256=hashlib.sha256(code.encode()).hexdigest(),
                source_tokens=len(tokenizer.encode(code, add_special_tokens=False)),
                prompt_tokens=lengths,
                prompt_sha256={
                    name: hashlib.sha256(prompt.encode()).hexdigest()
                    for name, prompt in rendered.items()
                },
            )
        )
        if len(provenance) == args.num_prompts:
            break
    if len(provenance) != args.num_prompts:
        raise RuntimeError("Not enough complete source files fit the token budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in prompts.items():
        (args.output_dir / f"{name}-prompts.json").write_text(
            json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    report = dict(
        dataset="glaiveai/code_edits_sample",
        revision=args.revision,
        source_file_sha256=hashlib.sha256(raw).hexdigest(),
        dataset_rows=len(rows),
        model_path=args.model_path,
        selection="First rows where both complete source prompts fit; no truncation",
        reference_solution_in_prompt=False,
        prompts=provenance,
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(provenance)} paired code-edit inputs", flush=True)
    for name in prompts:
        sizes = [r["prompt_tokens"][name] for r in provenance]
        print(
            name,
            "tokens min/mean/max:",
            min(sizes),
            sum(sizes) / len(sizes),
            max(sizes),
        )


if __name__ == "__main__":
    main()
