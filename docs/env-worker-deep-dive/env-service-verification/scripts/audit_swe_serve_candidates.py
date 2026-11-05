"""Inspect public task metadata without running tasks or claiming runtime compatibility."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    root = args.checkout.resolve()
    candidates, excluded = [], []
    with (root / "task_list.csv").open() as stream:
        tasks = list(csv.DictReader(stream))
    for task in tasks:
        if int(task["gpus"]) == 0:
            continue
        config = tomllib.loads((root / "tasks" / task["task_dir"] / "task.toml").read_text())
        verifier = config.get("verifier", {})
        destination = candidates if verifier.get("f2p_mode") == "correctness" else excluded
        destination.append(task["task_id"])
    print(
        json.dumps(
            {
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "total": len(tasks),
                "gpu_correctness_count": len(candidates),
                "gpu_correctness_candidates": candidates,
                "excluded_gpu_tasks": excluded,
                "boundary": "Metadata filter only. Audit test assertions, timeouts and hardware before execution.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
