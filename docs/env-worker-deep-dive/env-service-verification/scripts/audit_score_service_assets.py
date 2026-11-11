"""Audit pinned score task assets without importing candidate code or using a GPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash an asset without loading the whole model into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Validate upstream attestations and report declared fixture lifetimes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tests = args.task.resolve() / "tests"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    package = args.output.with_name(args.output.stem + "-package.json")
    env = dict(os.environ, VERIFIER_TESTS_ROOT=str(tests), VERIFIER_TEST_ROOT=str(tests / "postmerge_tests"))
    validations = []
    for script, extra in [("validate_vendor.py", []), ("validate_upstream_e2e_package.py", [str(package)])]:
        result = subprocess.run(
            [sys.executable, "-I", str(tests / script), *extra],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        validations.append(
            dict(script=script, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        )
    manifest = json.loads((tests / "model_assets_manifest.json").read_text())
    assets = []
    for model in manifest["models"]:
        snapshot = args.hf_cache / Path(model["snapshot_dir"]).relative_to("/hf-cache")
        for name, expected in model["files"].items():
            path = snapshot / name
            actual = sha256(path) if path.is_file() else None
            assets.append(
                dict(repository=model["repository"], revision=model["revision"], file=name, matches=actual == expected)
            )
    fixtures = []
    for path in sorted((tests / "postmerge_tests").rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                item.name: item.lineno
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                and item.name in {"setUp", "tearDown", "setUpClass", "tearDownClass"}
            }
            if methods:
                fixtures.append(dict(file=str(path.relative_to(tests)), class_name=node.name, methods=methods))
    patch = (args.task / "solution/changes.patch").read_text()
    changed_files = [line.split(" b/", 1)[1] for line in patch.splitlines() if line.startswith("diff --git ")]
    passed = all(item["exit_code"] == 0 for item in validations) and all(item["matches"] for item in assets)
    evidence = dict(
        scope="Offline asset and source audit only. No CUDA, service, task reward or scheduling benefit validated.",
        passed=passed,
        validations=validations,
        package=json.loads(package.read_text()) if package.is_file() else None,
        assets=assets,
        fixture_definitions=fixtures,
        oracle_changed_files=changed_files,
    )
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
