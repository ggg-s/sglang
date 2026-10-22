"""Archive completed measurements with a SHA256 manifest."""

import hashlib
import importlib.metadata
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    assert (ROOT / "matrix.exit").read_text().strip() == "0"
    assert json.loads((ROOT / "summary.json").read_text())["complete"]
    audit = json.loads((ROOT / "audit.json").read_text())
    assert audit["port_closed"] and "0 MiB" in audit["gpu_after"]
    versions = {}
    for package in ("torch", "triton", "transformers", "sglang"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "no distribution metadata; source revision recorded separately"
    (ROOT / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    files = sorted(f for f in ROOT.iterdir() if f.is_file() and f.suffix in (
        ".json", ".jsonl", ".log", ".txt", ".csv", ".exit", ".py", ".sh", ".yaml"
    ) and f.name != "evidence-manifest.json")
    files += sorted(f for folder in ROOT.glob("*-sanity") for f in folder.rglob("*") if f.is_file())
    manifest = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (ROOT / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(ROOT / "evidence.tar.gz", "w:gz") as archive:
        for name in sorted(manifest):
            archive.add(ROOT / name, arcname=name)
        archive.add(ROOT / "evidence-manifest.json", arcname="evidence-manifest.json")
    archive_path = ROOT / "evidence.tar.gz"
    sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (ROOT / "evidence.sha256").write_text(sha + "  evidence.tar.gz\n")
    print(json.dumps({"members": len(manifest), "bytes": archive_path.stat().st_size, "sha256": sha}))


if __name__ == "__main__":
    main()
