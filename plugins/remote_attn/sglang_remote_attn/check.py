"""Preflight: python -m sglang_remote_attn.check [--source-only]."""

import argparse
import ast
import importlib.metadata
import inspect
import json
import pkgutil
from pathlib import Path

from .hooks import HOOKS


def check_sources(root):
    """Check the target API without importing CUDA/model dependencies."""
    checked = []
    for target, hook in HOOKS.items():
        parts = target.split(".")
        for index in range(len(parts), 0, -1):
            path = root / "python" / Path(*parts[:index]).with_suffix(".py")
            if path.is_file():
                break
        else:
            raise RuntimeError(f"target module not found: {target}")
        node = ast.parse(path.read_text())
        for name in parts[index:]:
            node = next(
                (child for child in node.body if getattr(child, "name", None) == name),
                None,
            )
            if node is None:
                raise RuntimeError(f"target not found: {target}")
        args = node.args
        positional = [None] * (len(args.posonlyargs) + len(args.args))
        keywords = {arg.arg: None for arg in args.kwonlyargs}
        inspect.signature(hook).bind(None, *positional, **keywords)
        checked.append(target)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument(
        "--sglang-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    if args.source_only:
        checked = check_sources(args.sglang_root)
        print(json.dumps({"source_hooks": len(checked), "root": str(args.sglang_root)}))
        return

    # CUDA wheels may rely on the runtime libraries that PyTorch preloads.
    # Load it before Mooncake, including in this standalone preflight process.
    import torch
    from mooncake.engine import TransferEngine

    from sglang.srt.environ import envs

    eps = importlib.metadata.entry_points(group="sglang.srt.plugins")
    if not any(
        ep.name == "remote_attn" and ep.value == "sglang_remote_attn:register"
        for ep in eps
    ):
        raise RuntimeError(
            "install the plugin: python -m pip install -e plugins/remote_attn"
        )
    if not callable(getattr(TransferEngine, "send_probe", None)):
        raise RuntimeError("Mooncake TransferEngine.send_probe is required")
    envs.SGLANG_MOONCAKE_SEND_LOCK.get()
    for target in HOOKS:
        pkgutil.resolve_name(target)
    from .hooks import register_hooks

    # Registration resolves and signature-checks targets; this subprocess
    # exits without applying hooks or starting a scheduler.
    register_hooks()
    print(
        json.dumps(
            {
                "hooks": len(HOOKS),
                "send_probe": True,
                "plugin": "remote_attn",
                "torch": torch.__version__,
            }
        )
    )


if __name__ == "__main__":
    main()
