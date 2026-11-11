"""Evaluate an agent submission with the official FlashInfer benchmark protocol."""

import json
import shutil
import tempfile
from pathlib import Path


def main(
    submission=Path("/logs/artifacts/solution.json"), output=Path("/logs/verifier"), traces_root=Path("/data/traces")
):
    reward = {"reward": 0.0, "correctness": 0, "local_evaluation": 1, "eval_valid": 1}
    if not submission.is_file():
        (output / "reward.json").write_text(json.dumps(reward))
        return
    definitions = list((traces_root / "definitions").glob("*.json"))
    if len(definitions) != 1:
        raise ValueError("The verifier requires exactly one protected definition.")
    expected_definition = definitions[0].stem
    try:
        candidate = json.loads(submission.read_text())
        if (
            not isinstance(candidate, dict)
            or candidate.get("definition") != expected_definition
            or not candidate.get("sources")
        ):
            raise ValueError("Submit a solution for the protected definition with implementation sources.")
    except (ValueError, UnicodeError) as exc:
        (output / "submission-error.json").write_text(json.dumps({"error": str(exc)}))
        (output / "reward.json").write_text(json.dumps(reward))
        return
    from flashinfer_bench import Benchmark, BenchmarkConfig, TraceSet
    from flashinfer_bench.data import EvaluationStatus, Solution
    from pydantic import ValidationError

    try:
        Solution.model_validate(candidate)
    except ValidationError as exc:
        (output / "submission-error.json").write_text(json.dumps({"error": str(exc)}))
        (output / "reward.json").write_text(json.dumps(reward))
        return
    root = Path(tempfile.mkdtemp(prefix="submitted-traces-"))
    shutil.copytree(traces_root, root, dirs_exist_ok=True)
    for p in (root / "solutions").glob("*.json"):
        p.unlink()
    (root / "solutions/agent.json").write_text(json.dumps(candidate))
    traces = TraceSet.from_path(root)
    expected = sum(len(items) for items in traces.workloads.values())
    benchmark = Benchmark(traces, BenchmarkConfig.default(timeout_seconds=120))
    try:
        result = benchmark.run_all(dump_traces=False)
        rows = [row for values in result.traces.values() for row in values]
        valid = (
            len(rows) == expected
            and expected > 0
            and all(row.evaluation.status == EvaluationStatus.PASSED for row in rows)
        )
        reward.update(reward=float(valid), correctness=int(valid), evaluations=len(rows))
        (output / "official-results.json").write_text(
            json.dumps([row.model_dump(mode="json") for row in rows], indent=2)
        )
    finally:
        benchmark.close()
        shutil.rmtree(root)
    (output / "reward.json").write_text(json.dumps(reward))


if __name__ == "__main__":
    main()
