"""Check the extracted native dispatch against a supplied pre-extraction revision.

This proves the branch bodies/conditions, allowing only the explicit binding of
q_rope/k_nope/k_rope to the new method's parameters. It does not certify the new
projection/state semantics; those have separate numerical/lifecycle tests.
"""

import argparse
import ast
import subprocess
from pathlib import Path


def verify(root, base):
    relative = "python/sglang/srt/models/deepseek_v4.py"
    before = ast.parse(
        subprocess.check_output(
            ["git", "show", f"{base}:{relative}"], cwd=root, text=True
        )
    )
    after = ast.parse((root / relative).read_text())

    def method(tree, name):
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MQALayer"
        )
        return next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )

    old = next(
        n
        for n in ast.walk(method(before, "forward"))
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "enable_multi_stream"
    )
    new = next(n for n in method(after, "prepare_qkv").body if isinstance(n, ast.If))

    class BindNames(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = {
                "q_rope": "q_rope_out",
                "k_nope": "k_nope_out",
                "k_rope": "k_rope_out",
            }.get(node.id, node.id)
            return node

    assert ast.dump(BindNames().visit(old)) == ast.dump(new), (
        "native prepare dispatch changed"
    )
    print("PASS: native dispatch AST matches the base after explicit parameter binding")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--base", default="d3e2d0eaa6")
    args = parser.parse_args()
    verify(args.source_root, args.base)
