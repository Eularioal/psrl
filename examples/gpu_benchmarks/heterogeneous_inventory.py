"""Audit complete upstream scopes without treating metadata as runtime verification."""

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import tomllib
import yaml
from examples.gpu_benchmarks.prepare import provenance


def posttrain_inventory(repo):
    path = repo / "src/harbor_adapter/adapter.py"
    spec = importlib.util.spec_from_file_location("psrl_posttrain_inventory", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    tasks = []
    for benchmark in module.DEFAULT_BENCHMARKS:
        for model, info in module.MODELS.items():
            tasks.append(
                {
                    "id": f"posttrainbench-{benchmark}-{model}",
                    "benchmark": benchmark,
                    "model": info.model_id,
                    "test_data_present": (repo / "src/eval/tasks" / benchmark / "test_data.json").is_file(),
                    "required_judge_credentials": ["OPENAI_API_KEY"],
                    "gated_assets": ([info.model_id] if model == "gemma3-4b" else [])
                    + (["Idavidrein/gpqa"] if benchmark == "gpqamain" else []),
                    "runtime_verified": False,
                }
            )
    return {**provenance(repo), "task_count": len(tasks), "tasks": tasks}


def fastkernels_inventory(repo, captures=None):
    program = (
        "import json,sys; from dataclasses import asdict; sys.path.insert(0,sys.argv[1]); "
        "from fastkernels.list import discover_operator_targets; "
        "print(json.dumps([asdict(t) for t in discover_operator_targets()]))"
    )
    targets = json.loads(subprocess.check_output([sys.executable, "-c", program, str(repo)], text=True, timeout=60))
    observed = set()
    if captures:
        for path in captures.rglob("*.json"):
            report = json.loads(path.read_text())
            for name, entry in report.get("operators", {}).items():
                if entry.get("forward", {}).get("variants"):
                    observed.add(name)
    tasks = []
    for target in targets:
        qualified = f"fastkernels.tasks.baseline.L{target['level']}.{target['name']}:{target['class_name']}"
        tasks.append(
            {**target, "qualified_name": qualified, "has_capture": qualified in observed, "runtime_verified": False}
        )
    files = [path for path in (repo / "fastkernels/tasks/baseline").glob("L*/*.py") if path.name != "__init__.py"]
    scenarios = yaml.safe_load((repo / "fastkernels/scenarios/full.yaml").read_text())["scenarios"]
    dependencies = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["dependencies"]
    return {
        **provenance(repo),
        "source_modules": len(files),
        "modules_per_level": dict(Counter(path.parent.name for path in files)),
        "operator_targets": len(tasks),
        "targets_with_captures": sum(task["has_capture"] for task in tasks),
        "complete_capture_coverage": bool(tasks) and all(task["has_capture"] for task in tasks),
        "runtime_verified_count": 0,
        "full_capture_scenarios": len(scenarios),
        "tensor_parallel_degrees": dict(Counter(item.get("tp", 1) for item in scenarios)),
        "capture_dtypes": dict(Counter(item["dtype"] for item in scenarios)),
        "pinned_core_dependencies": [
            item for item in dependencies if item.startswith(("torch==", "vllm==", "transformers=="))
        ],
        "tasks": tasks,
    }


def swe_inventory(repo):
    requirements = json.loads((repo / "scripts/configs/hf_cache_requirements.json").read_text())
    with (repo / "task_list.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    tasks = []
    for row in rows:
        if not int(row["gpus"]):
            continue
        assets = requirements["tasks"][row["task_dir"]]
        names = assets.get("assets", []) if isinstance(assets, dict) else []
        tasks.append(
            {
                "id": row["task_dir"],
                "models": [requirements["assets"][name]["repository"] for name in names],
                "known_asset_bytes": sum(
                    requirements["assets"][name].get("snapshot_size_bytes") or 0 for name in names
                ),
                "unknown_asset_sizes": [
                    name for name in names if not requirements["assets"][name].get("snapshot_size_bytes")
                ],
                "upstream_images": requirements["task_images"][row["task_dir"]],
            }
        )
    return {**provenance(repo), "upstream_total": len(rows), "gpu_tasks": len(tasks), "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = args.sources.resolve()
    result = {
        "scope": "Source and asset inventory only. Runtime verification requires separate evidence.",
        "swe_serve": swe_inventory(sources / "swe-serve"),
        "posttrainbench": posttrain_inventory(sources / "PostTrainBench"),
        "fastkernels": fastkernels_inventory(sources / "fastkernels", args.captures),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
