"""Prepare reproducible LongBench v1 token IDs for a fixed-output serving replay.

Requires transformers and the target tokenizer; no GPU or model weights are used.
Input data is a LongBench data.zip, a directory containing task JSONL files, or
one JSONL file with a single --tasks entry. Templates must be the official
dataset2prompt.json. Downloads are deliberately separate from preparation.

This is a 128-token performance workload, not a LongBench answer-quality score.
No prompt is truncated. Length filtering includes the target chat template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TASKS = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def read_task(data_path, task):
    """Resolve exactly one task file; do not extract untrusted zip paths."""
    filename = f"{task}.jsonl"
    if data_path.is_dir():
        matches = sorted(data_path.rglob(filename))
        if len(matches) != 1:
            raise ValueError(f"Expected one {filename} under {data_path}: {matches}")
        source = matches[0]
        raw, name = source.read_bytes(), str(source.relative_to(data_path))
    elif zipfile.is_zipfile(data_path):
        with zipfile.ZipFile(data_path) as archive:
            matches = [
                member
                for member in archive.infolist()
                if not member.is_dir() and Path(member.filename).name == filename
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one {filename} in {data_path}: "
                    f"{[member.filename for member in matches]}"
                )
            raw, name = archive.read(matches[0]), matches[0].filename
    else:
        if data_path.name != filename:
            raise ValueError(f"Single JSONL input must be named {filename}")
        raw, name = data_path.read_bytes(), data_path.name

    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{line_number} must contain an object")
        rows.append((line_number, sha256(line), row))
    return rows, {"member": name, "sha256": sha256(raw), "rows": len(rows)}


def tokenizer_provenance(tokenizer, args, transformers_version):
    tokenizer_path = Path(args.tokenizer).resolve()
    # Hash tokenizer assets, including custom chat templates, without reading
    # model weights. The serialized effective fast tokenizer is hashed too.
    asset_names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
        "tokenizer.model", "spiece.model", "config.json",
    }
    assets = {
        str(path.relative_to(tokenizer_path)): file_sha256(path)
        for path in sorted(tokenizer_path.rglob("*"))
        if path.is_file() and (
            path.name in asset_names or path.suffix in {".jinja", ".jinja2"}
        )
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    return {
        "path": str(tokenizer_path),
        "requested_revision": args.tokenizer_revision,
        "resolved_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "class": type(tokenizer).__name__,
        "transformers_version": transformers_version,
        "trust_remote_code": args.trust_remote_code,
        "asset_sha256": assets,
        "effective_fast_tokenizer_sha256": (
            sha256(backend.to_str().encode("utf-8")) if backend is not None else None
        ),
    }


def make_encoder(tokenizer):
    template = getattr(tokenizer, "chat_template", None)
    has_chat = (
        callable(getattr(tokenizer, "apply_chat_template", None)) and bool(template)
    )
    if has_chat:
        # Resolve a template dictionary consistently with apply_chat_template.
        if callable(getattr(tokenizer, "get_chat_template", None)):
            template = tokenizer.get_chat_template()
        elif not isinstance(template, str):
            raise ValueError("Cannot resolve this tokenizer's chat template dictionary")
        mode = {
            "mode": "chat_template",
            "add_generation_prompt": True,
            "enable_thinking_requested": False,
            "thinking_control": (
                "template_references_enable_thinking"
                if "enable_thinking" in template
                else "unsupported_or_not_declared_by_template"
            ),
            "chat_template_sha256": sha256(template.encode("utf-8")),
        }

        def encode(prompt):
            kwargs = dict(add_generation_prompt=True, enable_thinking=False)
            messages = [{"role": "user", "content": prompt}]
            # Do not silently retry a failing template without its controls.
            # Unsupported tokenizers fail with a concrete preparation error.
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, **kwargs
            )
            return rendered, ids

    else:
        mode = {
            "mode": "raw_prompt",
            "reason": "tokenizer_has_no_apply_chat_template_or_chat_template",
            "add_generation_prompt": None,
            "enable_thinking_requested": False,
            "thinking_control": "unsupported",
            "add_special_tokens": True,
            "chat_template_sha256": None,
        }

        def encode(prompt):
            return prompt, tokenizer.encode(
                prompt, add_special_tokens=True, truncation=False
            )

    if mode["thinking_control"] != "template_references_enable_thinking":
        print(f"Tokenizer formatting limitation: {json.dumps(mode)}", file=sys.stderr)
    return encode, mode


def prepare(args, tokenizer, transformers_version):
    template_raw = args.templates.read_bytes()
    templates = json.loads(template_raw)
    encode, formatting = make_encoder(tokenizer)
    selected, sources, selection = [], {}, {}
    seen_contexts = set()
    for task in args.tasks:
        if task not in templates or not isinstance(templates[task], str):
            raise ValueError(f"No string prompt template for task {task}")
        rows, sources[task] = read_task(args.data, task)
        # Random permutation + rejection sampling gives a deterministic sample
        # without tokenizing every eligible row or using dataset-native lengths.
        rng = random.Random(int(sha256(f"{args.seed}:{task}".encode()), 16))
        rng.shuffle(rows)
        accepted, scanned, too_short, too_long = [], 0, 0, 0
        duplicate_contexts = 0
        for line_number, line_hash, row in rows:
            scanned += 1
            context = row.get("context")
            if not isinstance(context, str):
                raise ValueError(f"{task}:{line_number}: context must be a string")
            context_hash = sha256(context.encode("utf-8"))
            if context_hash in seen_contexts:
                duplicate_contexts += 1
                continue
            prompt = templates[task].format(**row)
            rendered, ids = encode(prompt)
            if not isinstance(rendered, str) or not isinstance(ids, list) or not ids:
                raise ValueError(f"{task}:{line_number}: invalid tokenizer output")
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError(f"{task}:{line_number}: invalid token IDs")
            if len(ids) < args.min_input_tokens:
                too_short += 1
                continue
            if len(ids) > args.max_input_tokens:
                too_long += 1
                continue
            source_id = row.get("_id", row.get("id"))
            accepted.append({
                "id": f"{task}:line:{line_number}",
                "task": task,
                "input_ids": ids,
                "input_tokens": len(ids),
                "prompt_sha256": sha256(rendered.encode("utf-8")),
                "unwrapped_prompt_sha256": sha256(prompt.encode("utf-8")),
                "input_ids_sha256": sha256(canonical_bytes(ids)),
                "source_id": source_id,
                "source_line": line_number,
                "source_line_sha256": line_hash,
                "source_length": row.get("length"),
                "source_member": sources[task]["member"],
                "source_language": row.get("language"),
                "context_sha256": context_hash,
                "max_new_tokens": args.output_tokens,
                "truncated": False,
            })
            # Only accepted contexts consume the global deduplication set: a
            # filtered question must not exclude a later eligible question.
            seen_contexts.add(context_hash)
            if len(accepted) == args.per_task:
                break
        selection[task] = {
            "requested": args.per_task, "selected": len(accepted), "scanned": scanned,
            "scanned_too_short": too_short, "scanned_too_long": too_long,
            "scanned_duplicate_context": duplicate_contexts,
            "exhausted_source": scanned == len(rows),
        }
        print(f"{task}: {json.dumps(selection[task])}", file=sys.stderr, flush=True)
        if len(accepted) < args.per_task and not args.allow_shortfall:
            raise ValueError(
                f"{task}: only {len(accepted)}/{args.per_task} examples have "
                f"{args.min_input_tokens}..{args.max_input_tokens} "
                "target input tokens. "
                "Selected contexts must be unique across all tasks. "
                "No output written. Change the explicit bounds/count or use "
                "--allow-shortfall; prompts are never truncated."
            )
        selected.extend(accepted)
    if not selected:
        raise ValueError("No examples passed target-tokenizer length filtering")
    random.Random(args.seed).shuffle(selected)
    return {
        "provenance": {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "fixed_output_performance_replay_not_longbench_quality_score",
            "dataset": "THUDM/LongBench v1",
            "dataset_revision": args.dataset_revision,
            "data_path": str(args.data.resolve()),
            "data_file_sha256": file_sha256(args.data) if args.data.is_file() else None,
            "source_files": sources,
            "templates_path": str(args.templates.resolve()),
            "templates_revision": args.templates_revision,
            "templates_sha256": sha256(template_raw),
            "tokenizer": tokenizer_provenance(tokenizer, args, transformers_version),
            "formatting": formatting,
            "selection": selection,
            "tasks": args.tasks,
            "per_task": args.per_task,
            "seed": args.seed,
            "min_input_tokens": args.min_input_tokens,
            "max_input_tokens": args.max_input_tokens,
            "context_length": args.context_length,
            "output_tokens": args.output_tokens,
            "allow_shortfall": args.allow_shortfall,
            "truncation": False,
            "context_deduplication": (
                "sha256_of_original_context_utf8_against_selected_records_across_tasks"
            ),
            "sampling": "per_task_seeded_random_permutation_then_length_rejection",
            "record_order_sha256": sha256(canonical_bytes([r["id"] for r in selected])),
            "script_sha256": file_sha256(Path(__file__)),
            "python_version": platform.python_version(),
        },
        "records": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", required=True, help="Local target tokenizer directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument("--min-input-tokens", type=int, default=4096)
    parser.add_argument("--max-input-tokens", type=int, default=16000)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision", help="Exact upstream revision, if known")
    parser.add_argument(
        "--templates-revision", help="Exact official-template commit, if known"
    )
    parser.add_argument(
        "--tokenizer-revision", help="Provenance label for local tokenizer"
    )
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    if not (0 < args.min_input_tokens <= args.max_input_tokens):
        parser.error("Require 0 < min-input-tokens <= max-input-tokens")
    if args.per_task <= 0 or args.output_tokens <= 0:
        parser.error("per-task and output-tokens must be positive")
    if args.max_input_tokens + args.output_tokens > args.context_length:
        parser.error("max-input-tokens + output-tokens exceeds context-length")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must be unique")
    if not Path(args.tokenizer).is_dir():
        parser.error(
            "tokenizer must be a local directory; download it before preparation"
        )
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve provenance")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=args.trust_remote_code
    )
    workload = prepare(args, tokenizer, transformers.__version__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids overwriting a concurrent preparation's artifact.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(workload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output.resolve()), "sha256": file_sha256(args.output),
        "records": len(workload["records"]),
        "input_tokens": sum(r["input_tokens"] for r in workload["records"]),
    }))


if __name__ == "__main__":
    main()
