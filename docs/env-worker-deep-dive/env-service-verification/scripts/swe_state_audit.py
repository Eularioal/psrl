"""
Audit declared GPU tasks, reference patches and fixture lifetimes without importing tasks.
"""

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import tomllib


def revision(root):
    """
    Record a checkout identity without changing its files.
    """
    return {
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).splitlines(),
    }


def python_fixtures(path, root):
    """
    Record syntactic construction sites without inferring dynamic execution counts.
    """
    source = path.read_text()
    try:
        module = ast.parse(source)
    except SyntaxError as error:
        return {"path": str(path.relative_to(root)), "parse_error": str(error)}
    fixture_names = {"setUpClass", "setUp", "tearDownClass", "tearDown", "setup_class", "setup_method"}
    constructors = {"Engine", "popen_launch_server", "Popen", "from_pretrained", "launch_server"}
    records = []
    classes = [node for node in ast.walk(module) if isinstance(node, ast.ClassDef)]
    for owner in classes:
        for function in owner.body:
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = []
            for call in ast.walk(function):
                if not isinstance(call, ast.Call):
                    continue
                name = ast.unparse(call.func).split(".")[-1]
                if name in constructors:
                    calls.append({"call": ast.unparse(call.func), "line": call.lineno})
            if function.name in fixture_names:
                records.append(
                    {"class": owner.name, "function": function.name, "line": function.lineno, "constructors": calls}
                )
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "fixtures": records,
    }


def task_record(root, row):
    """
    Keep reference edits separate from scored hardware and fixture declarations.
    """
    task = root / "tasks" / row["task_dir"]
    patches = sorted((task / "solution").rglob("*.patch"))
    changed = set()
    for patch in patches:
        changed.update(re.findall(r"^\+\+\+ b/(.+)$", patch.read_text(), re.MULTILINE))
    tests = task / "tests"
    protocol = tests / "upstream_e2e.toml"
    metadata = tomllib.loads(protocol.read_text()) if protocol.exists() else {}
    files = []
    for subtree in ("postmerge_tests", "unit_tests"):
        if (tests / subtree).exists():
            files.extend(sorted((tests / subtree).rglob("*.py")))
    fixtures = [python_fixtures(path, root) for path in files]
    shell = (tests / "test.sh").read_text()
    return {
        "task": row["task_dir"],
        "declared_gpus": int(row["gpus"]),
        "patch_files": [
            {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in patches
        ],
        "changed_source_paths": sorted(changed),
        "changed_srt_paths": sorted(name for name in changed if name.startswith("python/sglang/srt/")),
        "protocol": {
            key: metadata.get(key) for key in ("classification", "hardware", "execution_lane", "upstream_ref")
        },
        "scored_nodes": [item.get("node") for item in metadata.get("tests", [])],
        "test_entrypoint_sha256": hashlib.sha256(shell.encode()).hexdigest(),
        "entrypoint_mentions_group_runner": "run_upstream_e2e_group.py" in shell,
        "python_files": fixtures,
    }


def audit(root):
    """
    Count source evidence while preserving the limits of a static audit.
    """
    with (root / "task_list.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    gpu_rows = [row for row in rows if int(row["gpus"]) > 0]
    records = [task_record(root, row) for row in gpu_rows]
    counts = Counter()
    for record in records:
        counts["with_reference_patch"] += bool(record["patch_files"])
        counts["reference_edits_srt"] += bool(record["changed_srt_paths"])
        counts["with_group_runner_entrypoint"] += record["entrypoint_mentions_group_runner"]
        fixture_functions = [fixture for file in record["python_files"] for fixture in file.get("fixtures", [])]
        counts["with_class_setup"] += any(f["function"] in {"setUpClass", "setup_class"} for f in fixture_functions)
        counts["with_per_test_setup"] += any(f["function"] in {"setUp", "setup_method"} for f in fixture_functions)
        counts["with_constructor_in_class_setup"] += any(
            f["constructors"] and f["function"] in {"setUpClass", "setup_class"} for f in fixture_functions
        )
        counts["with_constructor_in_per_test_setup"] += any(
            f["constructors"] and f["function"] in {"setUp", "setup_method"} for f in fixture_functions
        )
    return {
        "scope": (
            "Static audit of manifest-declared GPU tasks. No imports, test execution, GPU measurements or agent edits."
        ),
        "upstream": revision(root),
        "manifest_tasks": len(rows),
        "declared_gpu_tasks": len(records),
        "counts": dict(counts),
        "limits": [
            "A syntactic constructor site is not an observed construction or cache reuse event.",
            "Class setup may call factories elsewhere, so direct constructor counts are lower bounds on syntax only.",
            "A GPU declaration does not imply every scored node uses a GPU.",
            "Reference patches are maintainer solutions, not autonomous agent trajectories or edit-frequency samples.",
            "Scanned postmerge and unit test files may contain coverage that is not directly scored by the protocol.",
        ],
        "tasks": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.swe_root)
    report["auditor"] = {
        "name": Path(__file__).name,
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))
