"""Require real successful scenarios from the upstream CUDA benchmark worker."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path


def score(report, expected_class):
    cases = report.get("scenarios", [])
    valid = bool(cases) and all(
        case.get("op", "").endswith(":" + expected_class)
        and case.get("status") == "PASSED"
        and math.isfinite(case.get("speedup", float("nan")))
        and case.get("candidate_ms", 0) > 0
        and case.get("baseline_ms", 0) > 0
        for case in cases
    )
    return {
        "reward": min(10.0, sum(case["speedup"] for case in cases) / len(cases)) if valid else 0.0,
        "eval_valid": int(valid),
        "cuda": int(valid),
        "local_evaluation": 1,
        "cases": len(cases),
        "skipped": report.get("skipped", 0),
    }


def main():
    spec = json.loads(Path("/tests/spec.json").read_text())
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH="/upstream",
        FASTKERNELS_CANDIDATE_DIR="/logs/artifacts/candidate",
        FASTKERNELS_RESULTS_DIR="/tmp/fastkernels-results",
        FASTKERNELS_CACHE_DIR="/tmp/fastkernels-cache",
    )
    command = [
        sys.executable,
        "-m",
        "fastkernels.bench",
        "--worker",
        "--op",
        spec["qualified_name"],
        "--level",
        str(spec["level"]),
        "--standalone",
        "--gpus",
        "0",
        "--max-shapes",
        "1",
        "--warmup",
        "1",
        "--iters",
        "2",
        "--rounds",
        "2",
        "--captures",
        "/captures",
    ]
    with (out / "upstream.log").open("w") as stream:
        result = subprocess.run(
            command, env=env, stdout=subprocess.PIPE, stderr=stream, text=True, timeout=150, check=False
        )
    cases = []
    for line in result.stdout.splitlines():
        try:
            case = json.loads(line)
        except ValueError:
            continue
        if isinstance(case, dict) and "status" in case:
            cases.append(case)
    report = {"scenarios": cases, "skipped": sum(case.get("status") == "SKIPPED" for case in cases)}
    (out / "upstream.json").write_text(json.dumps(report, indent=2))
    metrics = score(report, spec["class_name"])
    if result.returncode:
        metrics.update(reward=0.0, eval_valid=0)
    (out / "reward.json").write_text(json.dumps(metrics))
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
