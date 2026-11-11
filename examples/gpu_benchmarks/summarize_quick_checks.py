"""Summarize distinct task coverage without promoting bounded checks to full runs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(args):
    inventory = json.loads((args.kernel_suite / "inventory.json").read_text())["tasks"]
    tasks = {item["task_id"]: {"task_id": item["task_id"], "level": item["level"]} for item in inventory}
    for group in ("check0", "check1", "kernel-reuse", "kernel-bounded"):
        profile = "bounded" if group == "kernel-bounded" else "original"
        for path in (args.runs / group).rglob("quick-results.json"):
            for row in json.loads(path.read_text()):
                if row["task_id"] not in tasks:
                    continue
                current = tasks[row["task_id"]]
                if current.get(profile) != "passed":
                    current[profile] = row["status"]
                if row.get("overrides"):
                    current["bounded_overrides"] = row["overrides"]
    for task in tasks.values():
        if task.get("original") == "passed":
            task["status"] = "passed_original"
        elif task.get("bounded") == "passed":
            task["status"] = "passed_bounded"
        elif task["level"] == 4:
            task["status"] = "not_run_pretrained_assets"
        else:
            task["status"] = task.get("bounded", task.get("original", "not_run"))
    flash = []
    for path in (args.runs / "flash-check").rglob("operator-results.json"):
        flash = json.loads(path.read_text())
    mle = {}
    for path in args.mle_runs.rglob("dataset-results.json"):
        mle = json.loads(path.read_text())
    cleanup = {}
    for directory in (args.runs, args.mle_runs):
        for path in directory.rglob("evidence.json"):
            evidence = json.loads(path.read_text())
            after = evidence.get("gpu_after", {})
            cleanup[path.parent.name] = {
                "owned_containers_remaining": evidence.get("containers_remaining"),
                "gpu_compute_processes": len(after["pids"]) if "pids" in after else None,
                "gpu_used_mib": after.get("used_mb"),
                "cleanup_error": evidence.get("cleanup_error"),
            }
    repo = Path(__file__).resolve().parents[2]
    packaged = {}
    for path in args.runs.glob("packaged-local*/evidence.json"):
        evidence = json.loads(path.read_text())
        packaged[path.parent.name] = {"passed": evidence["passed"], "result": evidence.get("result")}
    return {
        "scope": "Quick task integration checks through PSRL EnvRouter and Harbor. No policy training.",
        "date": "2026-11-11",
        "kernel": {"counts": dict(Counter(item["status"] for item in tasks.values())), "tasks": list(tasks.values())},
        "flashinfer": {
            "prepared": 30,
            "evaluations": len(flash),
            "statuses": dict(Counter(row["evaluation"]["status"] for row in flash)),
        },
        "mle_local": mle,
        "packaged_local_checks": packaged,
        "cleanup": cleanup,
        "current_source_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in [
                "examples/kernel_bench/quick_check.py",
                "examples/flashinfer_bench/quick_check.py",
                "examples/flashinfer_bench/verify.py",
                "examples/mle_bench/local_baseline.py",
                "examples/gpu_benchmarks/evaluate.py",
                "examples/gpu_benchmarks/reward.py",
            ]
        },
        "prepared_checker_sha256": {
            path.parent.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.runs.glob("*/tests/quick_check.py")
        },
        "limitations": [
            "Reference controls check integration, not agent optimization ability.",
            "Bounded module dimensions are diagnostic overrides, not canonical KernelBench inputs.",
            "L4 pretrained assets were not prepared for this quick check.",
            "Cold per-task checks were stopped. Successful original checks were preserved and excluded from retries.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runs", "kernel-suite", "mle-runs", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
