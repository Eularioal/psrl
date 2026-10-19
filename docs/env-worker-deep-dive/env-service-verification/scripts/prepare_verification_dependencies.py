"""Apply the pinned verl fix to an isolated copy of the installed package."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path


def prepare(repo: Path, output: Path):
    package = Path(importlib.util.find_spec("verl").origin).resolve().parent
    patch = repo.resolve() / "patch/verl/d613affc.patch"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package, text=True).strip()
    if not commit.startswith("d613affc"):
        raise RuntimeError(f"Expected verl d613affc, got {commit}.")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(package, output / "verl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=output, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=output, check=True)
    manifest = {
        "source_package": str(package),
        "base_commit": commit,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "patched_file_sha256": hashlib.sha256((output / "verl/utils/tensordict_utils.py").read_bytes()).hexdigest(),
    }
    (output / "dependency-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo, args.output)
