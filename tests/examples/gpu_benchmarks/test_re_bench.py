"""Check RE-Bench data isolation, official reward gates, and offline judge transport."""

import argparse
import asyncio
import json
import threading

import pandas as pd
import pytest
import tomllib
from examples.gpu_benchmarks.reward import compute_score
from examples.re_bench import judge, replicate, run
from examples.re_bench.download import split_prompts
from examples.re_bench.prepare import prepare


def test_official_split_excludes_answers():
    texts = [f"question {i}\n### Response:\nSECRET ANSWER" for i in range(514)]
    public, private = split_prompts(texts)
    assert len(private) == 512 and len(public) == 2, "Keep the official held-out boundary."
    assert public[0].startswith("question 512"), "Do not leak held-out prompts into training."
    assert all("SECRET" not in prompt for prompt in public + private), "Expose prompts without ground truth."


@pytest.mark.parametrize("smoke", [False, True])
def test_native_task_and_score_identity(tmp_path, monkeypatch, smoke):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "assets.json").write_text(json.dumps({"heldout_prompts": 512, "model": "gpt2-xl"}))
    monkeypatch.setattr("examples.re_bench.prepare.provenance", lambda _: {"revision": "test"})
    args = argparse.Namespace(
        repo=tmp_path,
        assets=assets,
        output=tmp_path / "task",
        image="public:local",
        verifier_image="private:local",
        timeout=120,
        gpu_memory_mb=49152,
        smoke=smoke,
    )
    task = prepare(args)
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["environment"]["network_mode"] == "no-network", "Keep native network isolation."
    assert config["verifier"]["environment_mode"] == "separate", "Use an independent verifier."
    expected = "public:local" if smoke else "private:local"
    assert config["verifier"]["environment"]["docker_image"] == expected, "Keep private scoring data separate."
    assert not list(task.rglob("*compose*")), "Do not require a Compose provider."
    row = pd.read_parquet(task.parent / "task.parquet").iloc[0]
    assert row.agent_name == "gpu_benchmark", "Use the Env-required PSRL loop."
    assert row.data_source == ("re_bench_smoke" if smoke else "re_bench"), "Separate smoke and official scores."
    assert row.extra_info["gpu_memory_mb"] == 49152, "Forward the real task reservation."


@pytest.mark.parametrize(
    "source,metrics,expected",
    [
        ("re_bench", {"reward": 0.75, "judge_valid": 1, "official_evaluation": 1}, 0.75),
        ("re_bench", {"reward": 0.75}, 0.0),
        ("re_bench", {"reward": 0.75, "judge_valid": 0, "official_evaluation": 1}, 0.0),
        ("re_bench_smoke", {"reward": 1, "smoke_valid": 1}, 0.0),
    ],
)
def test_only_valid_official_score_trains(source, metrics, expected):
    result = compute_score(source, "", "", {"harbor_rewards": metrics})
    assert result["score"] == expected, "Do not train on missing judges or smoke checks."


def test_bridge_limits_and_ranking():
    inputs = {"prompt": "p", "system_prompt": "s", "temperature": 0.0, "max_tokens": 100}
    assert judge.validate_request({"model": judge.MODEL, "input": inputs}) == inputs, "Keep official inputs."
    with pytest.raises(ValueError, match="official"):
        judge.validate_request({"model": "another-model", "input": inputs})
    with pytest.raises(ValueError, match="sampling"):
        judge.validate_request({"model": judge.MODEL, "input": {**inputs, "max_tokens": 10000}})
    with pytest.raises(ValueError, match="ranking"):
        judge.validate_ranking('[{"model":"model_1","rank":1}]')


@pytest.mark.asyncio
async def test_networkless_transport_and_budget(tmp_path, monkeypatch):
    root = tmp_path / "trial/artifacts/rebench-judge"
    monkeypatch.setenv("REBENCH_JUDGE_DIR", str(root))
    monkeypatch.setenv("REBENCH_JUDGE_TIMEOUT", "5")
    answer = '[{"model":"model_1","rank":1},{"model":"model_2","rank":2}]'
    monkeypatch.setattr(judge, "call_judge", lambda *args: answer)
    stop = threading.Event()
    broker = asyncio.create_task(asyncio.to_thread(judge.serve, tmp_path, "fixture", 1, 1, 1, stop))
    inputs = {"prompt": "p", "system_prompt": "s", "temperature": 0.0, "max_tokens": 100}
    try:
        result = await replicate.async_run(judge.MODEL, inputs)
        assert result == [answer], "Preserve the upstream streaming-output interface."
        before = replicate.failures
        with pytest.raises(RuntimeError, match="budget"):
            await replicate.async_run(judge.MODEL, inputs)
        assert replicate.failures == before + 1, "Make failed feedback visible to the strict evaluator."
    finally:
        stop.set()
        counters = await asyncio.wait_for(broker, 5)
    assert counters == {"submitted": 1, "succeeded": 1, "failed": 1}, "Do not exceed the API budget."


@pytest.mark.asyncio
async def test_missing_credentials_fail_before_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    (tmp_path / "provenance.json").write_text(json.dumps({"benchmark": "re_bench", "mode": "official-evaluation"}))
    args = argparse.Namespace(task=tmp_path, environment_backend="env_worker")
    with pytest.raises(RuntimeError, match="REPLICATE_API_TOKEN"):
        await run.run(args)
