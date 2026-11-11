"""Evaluate submissions in a fresh verifier using the upstream scoring code."""

import json
import math
import shutil
import subprocess
from pathlib import Path


def evaluate(spec):
    artifacts = Path("/logs/artifacts")
    if spec["benchmark"] == "kernel_bench":
        import torch
        from kernelbench.eval import eval_kernel_against_ref

        candidate = artifacts / "submission.py"
        if not candidate.is_file():
            return {"reward": 0.0, "correctness": 0.0}
        reference = Path(spec["reference"]).read_text()
        bounded = spec.get("input_profile") == "bounded_module_dimensions"
        if bounded:
            from quick_check import bounded_source

            reference, _ = bounded_source(reference)
        result = eval_kernel_against_ref(
            reference,
            candidate.read_text(),
            num_correct_trials=1 if bounded else 3,
            num_perf_trials=10,
            measure_performance=not bounded,
            device=torch.device("cuda:0"),
            backend=spec["backend"],
        )
        (Path("/logs/verifier") / "kernel_result.json").write_text(
            json.dumps(result.model_dump(), indent=2, default=str)
        )
        speedup = result.ref_runtime / result.runtime if result.ref_runtime > 0 and result.runtime > 0 else 0.0
        return {
            "reward": float(result.correctness),
            "correctness": float(result.correctness),
            "runtime_ms": result.runtime,
            "speedup": speedup,
            "local_evaluation": int(bounded),
            "bounded_inputs": int(bounded),
        }
    if spec["benchmark"] == "mle_bench":
        from mlebench.grade import grade_csv
        from mlebench.registry import Registry

        competition = Registry(Path("/data/mle")).get_competition(spec["competition"])
        report = grade_csv(artifacts / "submission.csv", competition)
        (Path("/logs/verifier") / "mle_report.json").write_text(json.dumps(report.to_dict(), default=str))
        return {
            "reward": float(report.above_median),
            "valid_submission": float(report.valid_submission),
            "any_medal": float(report.any_medal),
            "raw_score": float(report.score or 0.0),
        }
    candidate = artifacts / "train.py"
    if not candidate.is_file():
        return {"reward": 0.0}
    work = Path("/tmp/autoresearch-replay")
    work.mkdir(exist_ok=False)
    shutil.copy2(candidate, work / "train.py")
    shutil.copy2("/opt/upstream/prepare.py", work / "prepare.py")
    shutil.copy2("/tests/autoresearch_replay.py", work / "replay.py")
    with Path("/logs/verifier/replay.log").open("w") as log:
        result = subprocess.run(
            [spec["python"], "-u", "replay.py"],
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=spec["timeout"] - 30,
        )
    measured = work / "measured.json"
    if result.returncode or not measured.is_file():
        return {"reward": 0.0}
    bpb = float(json.loads(measured.read_text())["val_bpb"])
    if not math.isfinite(bpb) or bpb <= 0:
        return {"reward": 0.0}
    return {"reward": 1 / (1 + bpb), "val_bpb": bpb}


if __name__ == "__main__":
    metrics = evaluate(json.loads(Path("/tests/spec.json").read_text()))
    Path("/logs/verifier/reward.json").write_text(json.dumps(metrics, allow_nan=False))
