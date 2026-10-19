"""Reproduce judge error classification without running submitted code or Docker."""

import importlib.util
import json
from pathlib import Path

repo = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
spec = importlib.util.spec_from_file_location("airs_judge_review", repo / "examples/airs_bench/apps/judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)
operations = []


def request(message):
    # Simulate only the candidate transport. Use the real judge and comparator.
    operations.append(message["op"])
    return {"ok": True, "stdout": '{"not_an_integer": 1}'}


judge.request = request
samples = [{"input_output": json.dumps({"fn_name": "solve", "inputs": [[1]], "outputs": [1]})}]
try:
    result = judge.grade([["bad", "correct", "unused", "unused", "unused"]], samples)
    print(json.dumps({"result": result, "operations": operations}))
except Exception as error:
    print(
        json.dumps(
            {
                "exception": type(error).__name__,
                "message": str(error),
                "operations": operations,
                "later_candidate_attempted": operations.count("begin") > 1,
            }
        )
    )
