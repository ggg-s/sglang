"""Freeze final source before the optimized capacity-eight servers launch."""

import ast
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
opt = ROOT / "optimized"
matrix = ROOT / "matrix8"
assert (opt / "optimized.exit").read_text().strip() == "0"
assert not (matrix / "optimized-replay-boot1-server.log").exists()
kernel = (
    opt / "source/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_decode.py"
)


def jit_bodies():
    return {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(kernel.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.decorator_list
    }


before = jit_bodies()
(opt / "c1-source-manifest.json").write_bytes(
    (opt / "source-manifest.json").read_bytes()
)
with tarfile.open(ROOT / "gdn-replay-opt-final-overlay.tar.gz") as archive:
    archive.extractall(opt / "source", filter="data")
assert before == jit_bodies(), "Final hardware selector must not change JIT bodies"
manifest = json.loads((opt / "source-manifest.json").read_text())
with tarfile.open(ROOT / "gdn-replay-opt-final-overlay.tar.gz") as archive:
    for name in archive.getnames():
        manifest[name] = hashlib.sha256(
            (opt / "source" / name).read_bytes()
        ).hexdigest()
(opt / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(matrix / "source-opt-manifest.json").write_bytes(
    (opt / "source-manifest.json").read_bytes()
)
(ROOT / "final-source-ok.json").write_text(
    json.dumps(
        {
            "jit_bodies_equal_to_c1_profile_source": True,
            "files": len(manifest),
            "overlay_sha256": hashlib.sha256(
                (ROOT / "gdn-replay-opt-final-overlay.tar.gz").read_bytes()
            ).hexdigest(),
        },
        indent=2,
    )
    + "\n"
)
print("Final source frozen; JIT bodies unchanged")
