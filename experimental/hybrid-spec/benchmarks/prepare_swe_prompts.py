"""Render fixed SWE-rebench conversation prefixes for inference-only replay.

No tool calls are executed. This is a serving workload, not an agent task score.
"""

import argparse
import hashlib
import json
from pathlib import Path


def without_nulls(value):
    # Arrow unions tool schemas into one struct; remove absent properties.
    if isinstance(value, dict):
        return {k: without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [without_nulls(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=7168)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompts, provenance = [], []
    skipped = 0
    table = pq.ParquetFile(args.dataset)
    for batch in table.iter_batches(
        batch_size=16, columns=["trajectory_id", "trajectory", "tools"]
    ):
        for row in batch.to_pylist():
            messages = without_nulls(row["trajectory"])
            tools = without_nulls(row["tools"])
            # OpenAI trajectories store arguments as JSON strings; Qwen's chat
            # template renders named arguments from a mapping.
            try:
                for message in messages:
                    for call in message.get("tool_calls", []):
                        function = call["function"]
                        arguments = function.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be a JSON object")
                        function["arguments"] = arguments
            except (ValueError, json.JSONDecodeError):
                skipped += 1
                continue
            turns = [
                i for i, m in enumerate(messages) if i and m["role"] == "assistant"
            ]
            # Find the longest whole-turn prefix that fits, without truncating
            # serialized tool calls or dropping system/tool instructions.
            def render(index):
                prompt = tokenizer.apply_chat_template(
                    messages[:index],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                length = len(tokenizer.encode(prompt, add_special_tokens=False))
                return index, prompt, length

            # Tool schemas alone can exceed the budget. Reject those rows
            # before tokenizing many long prefixes from the same trajectory.
            chosen = render(turns[0]) if turns else None
            if chosen is None or chosen[2] > args.max_prompt_tokens:
                skipped += 1
                if skipped % 100 == 0:
                    print(f"Skipped {skipped}; prepared {len(prompts)}", flush=True)
                continue
            low, high = 1, len(turns) - 1
            while low <= high:
                middle = (low + high) // 2
                index, prompt, length = render(turns[middle])
                if length <= args.max_prompt_tokens:
                    chosen = (index, prompt, length)
                    low = middle + 1
                else:
                    high = middle - 1
            if chosen is None:
                skipped += 1
                continue
            index, prompt, length = chosen
            prompts.append(prompt)
            print(f"Prepared prefix {len(prompts)}: {length} tokens", flush=True)
            provenance.append(
                dict(
                    trajectory_id=row["trajectory_id"],
                    assistant_turn_index=index,
                    prompt_tokens=length,
                    sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                )
            )
            if len(prompts) == args.num_prompts:
                break
        if len(prompts) == args.num_prompts:
            break
    if len(prompts) != args.num_prompts:
        raise RuntimeError(f"Only {len(prompts)} trajectories fit the token budget")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(
            dict(
                dataset="nebius/SWE-rebench-openhands-trajectories",
                revision=args.revision,
                model_path=args.model_path,
                skipped=skipped,
                prompts=provenance,
                note="Whole conversation prefixes; tool calls are not executed; no agent quality score.",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Prepared {len(prompts)} SWE prefixes: {args.output}", flush=True)


if __name__ == "__main__":
    main()
