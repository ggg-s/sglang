"""Prepare original sa-spec or paired diff/full-file code-edit prompts."""

import argparse
import hashlib
import json
from pathlib import Path


def sa_spec_prompt(code, edit):
    # Match sa_spec/bench/make_dataset.py at 1cc7e8c86b19, including whitespace.
    # Raw completion prompt: do not apply a chat template or strip the newlines.
    return (
        "\nApply the following edit to the following code:\n\n"
        f"<edit>\n{edit}\n</edit>\n\n\n"
        f"<code>\n{code}\n</code>\n\n\n"
        "Result:\n<edited_code>\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=7168)
    parser.add_argument(
        "--template",
        choices=("paired", "sa-spec"),
        default="paired",
        help="sa-spec uses the original raw prompt and first N rows without filtering",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer

    if args.num_prompts < 1 or args.max_prompt_tokens < 1:
        parser.error("num-prompts and max-prompt-tokens must be positive")
    raw = args.dataset.read_bytes()
    rows = json.loads(raw)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    original = args.template == "sa-spec"
    prompts = {"sa-spec": []} if original else {"diff": [], "rewrite": []}
    provenance = []
    for index, row in enumerate(rows):
        # The reference solution is deliberately never included in the input.
        code, edit = row["code"], row["edit"]
        rendered, lengths = {}, {}
        instructions = (
            ("diff", "Return only a unified diff, without an explanation."),
            (
                "rewrite",
                "Return the complete updated source code. Preserve unaffected code. "
                "Return code only, without an explanation.",
            ),
        )
        if original:
            rendered["sa-spec"] = sa_spec_prompt(code, edit)
        for name, instruction in () if original else instructions:
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
        for name, prompt in rendered.items():
            # /generate encodes raw text with special tokens by default. Paired
            # prompts already contain the model's chat-template special tokens.
            lengths[name] = len(tokenizer.encode(prompt, add_special_tokens=original))
        if max(lengths.values()) > args.max_prompt_tokens:
            if original:
                raise ValueError(
                    f"Row {index} needs {lengths['sa-spec']} prompt tokens, above "
                    f"--max-prompt-tokens={args.max_prompt_tokens}. Increase the "
                    "budget and server context together; no rows were skipped."
                )
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
    if original:
        # Official SGLang custom loader requires two turns. The empty assistant
        # is only a schema placeholder: --sharegpt-output-len supplies the cap,
        # and only the first turn is sent to /generate (no chat template).
        with (args.output_dir / "sa-spec-sglang.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for prompt in prompts["sa-spec"]:
                stream.write(
                    json.dumps(
                        {
                            "conversations": [
                                {"from": "human", "value": prompt},
                                {"from": "gpt", "value": ""},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        with (args.output_dir / "sa-spec-dataset.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for index, prompt in enumerate(prompts["sa-spec"]):
                stream.write(
                    json.dumps(
                        dict(task_id=index, prompt=prompt, output_tokens=2048),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    report = dict(
        dataset="glaiveai/code_edits_sample",
        revision=args.revision,
        source_file_sha256=hashlib.sha256(raw).hexdigest(),
        dataset_rows=len(rows),
        model_path=args.model_path,
        template=args.template,
        chat_template_applied=not original,
        max_prompt_tokens=args.max_prompt_tokens,
        selection=(
            "First N rows in source order; fail on overflow; no truncation"
            if original
            else "First rows where both complete source prompts fit; no truncation"
        ),
        reference_solution_in_prompt=False,
        prompts=provenance,
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(provenance)} {args.template} code-edit inputs", flush=True)
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
