"""Protect model identity and hidden-data boundaries in local post-training tasks."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from examples.gpu_benchmarks.reward import compute_score
from examples.posttrain_bench import prepare as module
from examples.posttrain_bench.judge_scoring import rubric_score
from examples.posttrain_bench.prepare_judge_data import select_rubrics
from psrl.workers.env_service.harbor_resources import demand_from_task


@pytest.mark.parametrize("contract", ["unknown", "independent_commands"])
def test_complete_public_combinations_keep_private_answers_separate(tmp_path, monkeypatch, contract):
    models = {}
    tasks = []
    assets = tmp_path / "assets"
    assets.mkdir()
    for model in ("qwen-small", "qwen-large", "smol"):
        path = assets / model
        path.mkdir()
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"fixture")
        models[model] = {"path": str(path), "revision": "fixed"}
        for benchmark in module.BENCHMARKS:
            tasks.append({"id": f"{benchmark}-{model}", "model": model, "benchmark": benchmark, "gated_assets": []})
    tasks.append({"id": "gated", "model": "gated-model", "benchmark": "gsm8k", "gated_assets": ["model"]})
    (assets / "models.json").write_text(json.dumps(models))
    (assets / "judge.json").write_text(json.dumps({"path": str(path), "name": "frozen-fixture"}))
    for benchmark in module.BENCHMARKS:
        for part in ("public", "private"):
            directory = assets / "data" / benchmark / part
            directory.mkdir(parents=True)
            (directory / "rows.json").write_text('[{"prompt":"q","answer":"a"}]')
    monkeypatch.setattr(module, "posttrain_inventory", lambda repo: {"revision": "fixed", "tasks": tasks})
    output = tmp_path / "tasks"
    module.prepare(tmp_path, assets, output, "/runtime/bin/python", "fixture", True, gpu_execution_contract=contract)
    frame = pd.read_parquet(output / "tasks.parquet")
    assert len(frame) == 15, "All public model and task combinations must be prepared."
    assert set(frame.data_source) == {"posttrain_bench_local", "posttrain_bench_judged_local"}, (
        "Local scoring must identify its actual protocol."
    )
    for item in json.loads((output / "manifest.json").read_text()):
        target = output / item["task"]
        assert demand_from_task(str(target)).gpu_execution_contract == contract, (
            "Generated task capabilities must reach the same admission demand used by the runner."
        )
        agent = yaml.safe_load((target / "environment/docker-compose.yaml").read_text())["services"]["main"]
        verifier = yaml.safe_load((target / "tests/docker-compose.yaml").read_text())["services"]["main"]
        assert not any(":/data/private" in value for value in agent["volumes"]), "Agent must not see hidden answers."
        assert any(value.endswith(":/data/private:ro") for value in verifier["volumes"]), (
            "Verifier needs private answers."
        )
        assert "psrl-gpu" in (target / "instruction.md").read_text(), "Guided acceptance must use managed GPU jobs."
        is_judged = item["benchmark"] == "posttrain_bench_judged_local"
        assert ("--benchmark" in (target / "tests/test.sh").read_text()) == is_judged, (
            "Likelihood and judge protocols must select their respective evaluator."
        )
        assert not any(":/judge:" in value for value in agent["volumes"]), "Only the verifier receives the judge."
        assert any(value.endswith(":/judge:ro") for value in verifier["volumes"]) == is_judged, (
            "The judge must be frozen and mounted only for judged evaluations."
        )
    Path(models["smol"]["path"], "model.safetensors").unlink()
    rejected = tmp_path / "rejected"
    with pytest.raises(FileNotFoundError):
        module.prepare(tmp_path, assets, rejected, "/runtime/bin/python", "fixture")
    assert not rejected.exists(), "Missing real model assets must fail before publishing tasks."


def test_medical_rubrics_preserve_negative_points():
    rubrics = [
        {"points": 2, "criterion": "Mentions evidence."},
        {"points": -3, "criterion": "Guarantees a cure."},
        {"points": 6, "criterion": "Explains uncertainty."},
        {"points": -1, "criterion": "Gives a harmful instruction."},
    ]
    selected = select_rubrics(rubrics)
    assert [item["points"] for item in selected] == [2, 6, -3, -1], "Signed original points must be preserved."
    assert rubric_score(selected, [1, 1, 0, 0]) == 1, "Meeting all helpful criteria should receive full credit."
    assert rubric_score(selected, [1, 1, 1, 1]) == 0.5, "Harmful content must reduce the reward."
    assert rubric_score(selected, [0, 0, 1, 1]) == 0, "Negative totals must be bounded at zero."
    with pytest.raises(ValueError):
        rubric_score(selected, [1, 1, float("nan"), 0])
    with pytest.raises(ValueError):
        rubric_score(selected, [1])


def test_judged_reward_requires_real_generation_and_judge_evaluation():
    metrics = {
        "reward": 0.25,
        "eval_valid": 1,
        "cuda": 1,
        "local_evaluation": 1,
        "judge_valid": 1,
        "generated_answers": 2,
        "judge_calls": 4,
    }
    for key in ("eval_valid", "cuda", "local_evaluation", "judge_valid", "generated_answers", "judge_calls"):
        invalid = {k: v for k, v in metrics.items() if k != key}
        result = compute_score("posttrain_bench_judged_local", "", "", {"harbor_rewards": invalid})
        assert result["score"] == 0, f"Missing {key!r} must reject the judged reward."
    assert compute_score("posttrain_bench_judged_local", "", "", {"harbor_rewards": metrics})["score"] == 0.25, (
        "A complete judged evaluation must preserve its reward."
    )


@pytest.mark.parametrize("source", ["posttrain_bench_local", "mlgym_nlp_local"])
def test_language_rewards_require_cuda_and_independent_evaluation(source):
    metrics = {"reward": 0.5, "eval_valid": 1, "cuda": 1, "local_evaluation": 1}
    for key in ("eval_valid", "cuda", "local_evaluation"):
        invalid = {k: v for k, v in metrics.items() if k != key}
        result = compute_score(source, "", "", {"harbor_rewards": invalid})
        assert result["score"] == 0, f"Missing {key!r} must not produce a training reward."
    assert compute_score(source, "", "", {"harbor_rewards": metrics})["score"] == 0.5, (
        "Valid reward must be preserved."
    )
