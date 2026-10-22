"""Archive raw traces, measurements and source overlays, without source checkouts."""

import hashlib
import importlib.metadata
import json
import os
import tarfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    assert (
        json.loads((ROOT / "final-validation.json").read_text())["status"] == "complete"
    )
    assert json.loads((ROOT / "matrix8/summary.json").read_text())["complete"]
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "triton", "transformers")
    }
    versions.update(torch_build=torch.__version__, cuda=torch.version.cuda)
    (ROOT / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    files = []
    for folder, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ("source", "source-opt", "__pycache__")]
        for name in names:
            path = Path(folder) / name
            if path.is_symlink() or name in (
                "evidence-manifest.json",
                "evidence.tar.gz",
            ):
                continue
            if (
                path.suffix
                in (
                    ".json",
                    ".jsonl",
                    ".log",
                    ".py",
                    ".sh",
                    ".yaml",
                    ".txt",
                    ".csv",
                    ".xml",
                    ".exit",
                    ".diff",
                    ".md",
                )
                or name.endswith(".trace.json.gz")
                or name in ("opt-overlay.tar.gz", "gdn-replay-opt-final-overlay.tar.gz")
            ):
                files.append(path)
    manifest = {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(files)
    }
    (ROOT / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(ROOT / "evidence.tar.gz", "w:gz") as archive:
        for name in sorted(manifest):
            archive.add(ROOT / name, arcname=name)
        archive.add(ROOT / "evidence-manifest.json", arcname="evidence-manifest.json")
    path = ROOT / "evidence.tar.gz"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "evidence.sha256").write_text(sha + "  evidence.tar.gz\n")
    print(
        json.dumps(
            {"raw_files": len(manifest), "bytes": path.stat().st_size, "sha256": sha}
        )
    )


if __name__ == "__main__":
    main()
