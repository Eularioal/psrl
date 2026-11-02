"""Check resource routing and private-data boundaries of the GPU recipes."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
import tomllib
import yaml
from examples.gpu_benchmarks.prepare import compose, prepare
from examples.gpu_benchmarks.reward import compute_score


def args(tmp_path, benchmark):
    repo = tmp_path / "upstream"
    repo.mkdir()
    return argparse.Namespace(
        repo=repo,
        output=tmp_path / "task",
        benchmark=benchmark,
        image="test:local",
        python="/opt/env/bin/python",
        timeout=120,
        gpu_memory_mb=4096,
        level=1,
        problem="task.py",
        backend="triton",
        data=None,
        competition=None,
        harbor_task=None,
    )


def test_mle_private_data_is_verifier_only(tmp_path, monkeypatch):
    config = args(tmp_path, "mle_bench")
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {"revision": "test"})
    config.data = tmp_path / "data"
    config.competition = "example"
    for split in ["public", "private"]:
        (config.data / "example" / "prepared" / split).mkdir(parents=True)
    desc = config.repo / "mlebench/competitions/example/description.md"
    desc.parent.mkdir(parents=True)
    desc.write_text("Train and submit predictions.")
    task = prepare(config)
    public = (task / "environment/docker-compose.yaml").read_text()
    private = (task / "tests/docker-compose.yaml").read_text()
    assert "/data/mle" not in public, "Private dataset root must not be visible to the agent."
    assert "/data/mle:ro" in private, "The verifier must have read-only grading data."
    assert "/prepared/public:/data/public:ro" in public, "Expose only the public split."
    spec = tomllib.loads((task / "task.toml").read_text())
    assert spec["verifier"]["environment_mode"] == "separate", "Use a fresh grader container."
    row = pd.read_parquet(task.parent / "task.parquet").iloc[0]
    assert row.agent_name == "gpu_benchmark", "Dispatch through the Env-required loop."
    assert row.extra_info["gpus"] == 1, "Declare environment GPU demand."


def test_missing_real_assets_fail_before_writing(tmp_path, monkeypatch):
    config = args(tmp_path, "autoresearch")
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {})
    with pytest.raises(ValueError, match="prepared"):
        prepare(config)
    assert not config.output.exists(), "Do not publish a task without upstream assets."


def test_venv_mount_includes_base_interpreter(tmp_path):
    base = tmp_path / "base"
    venv = tmp_path / "venv"
    for prefix in (base, venv):
        (prefix / "bin").mkdir(parents=True)
    (base / "bin/python").touch()
    (venv / "bin/python").symlink_to(base / "bin/python")
    definition = compose([], str(venv / "bin/python"), mount_python_prefix=True)
    assert set(definition["services"]["main"]["volumes"]) == {
        f"{base}:{base}:ro",
        f"{venv}:{venv}:ro",
    }, "Mount both the isolated packages and their base interpreter."


def test_native_task_uses_distinct_prebuilt_images(tmp_path, monkeypatch):
    config = args(tmp_path, "kernel_bench")
    config.environment_backend = "env_worker"
    config.verifier_image = "verifier:local"
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {})
    reference = config.repo / "KernelBench/level1/task.py"
    reference.parent.mkdir(parents=True)
    reference.write_text("original reference")
    task = prepare(config)
    definition = tomllib.loads((task / "task.toml").read_text())
    assert definition["environment"]["docker_image"] == "test:local"
    assert definition["verifier"]["environment"]["docker_image"] == "verifier:local"
    assert definition["environment"]["gpus"] == 1, "Keep GPU demand for admission."
    assert not list(task.rglob("docker-compose.yaml")), "Native execution cannot interpret Compose."


def test_autoresearch_rejects_incomplete_cache(tmp_path, monkeypatch):
    config = args(tmp_path, "autoresearch")
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {})
    config.data = tmp_path / "data"
    for name in ("tokenizer", "data"):
        (config.data / name).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="tokenizer.pkl"):
        prepare(config)
    assert not config.output.exists(), "Do not publish a task with empty cache directories."


def test_kernel_preserves_reference(tmp_path, monkeypatch):
    config = args(tmp_path, "kernel_bench")
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {"revision": "test"})
    reference = config.repo / "KernelBench/level1/task.py"
    reference.parent.mkdir(parents=True)
    reference.write_text("original reference")
    task = prepare(config)
    assert reference.read_text() == "original reference", "Never resize official benchmark inputs silently."
    assert json.loads((task / "provenance.json").read_text())["reference_sha256"], "Record reference identity."
    assert yaml.safe_load((task / "environment/docker-compose.yaml").read_text()), "Write a valid Compose file."


def test_nonfinite_reward_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        compute_score("kernel_bench", "", "", {"harbor_rewards": {"reward": float("nan")}})


def test_env_cannot_be_disabled():
    from types import SimpleNamespace

    from examples.gpu_benchmarks.agent_loop import GpuBenchmarkAgentLoop
    from omegaconf import OmegaConf

    context = SimpleNamespace(config=OmegaConf.create({"psrl": {"env_service": {"enable": False}}}))
    with pytest.raises(ValueError, match="env_service.enable"):
        GpuBenchmarkAgentLoop(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", ["kernel_bench", "re_bench"])
async def test_session_trajectory_reward_and_cleanup(benchmark):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from examples.gpu_benchmarks.agent_loop import GpuBenchmarkAgentLoop
    from examples.sciaccel_rl.runner import HarborEpisodeResult
    from psrl.workers.agent_loop.loops.utils import TerminateReason

    loop = object.__new__(GpuBenchmarkAgentLoop)
    loop.model_config = SimpleNamespace(path="test-model")
    loop.rollout_config = SimpleNamespace(response_length=32)
    loop.max_turns = 4
    loop.thinking_template = "multi_thinking"
    loop.session_router_url = "http://test-router"
    loop.create_session = AsyncMock(return_value="sid")
    loop.delete_session = AsyncMock()
    loop.get_training_data = AsyncMock(return_value=[{"num_turns": 1, "response_ids": [10, 11]}])
    loop._run_harbor_in_thread = AsyncMock(
        return_value=HarborEpisodeResult(
            task_name=benchmark,
            reward=1.0,
            rewards={"reward": 1.0, "correctness": 1.0, "judge_valid": 1, "official_evaluation": 1},
        )
    )
    loop.build_token_output = lambda item, extra_fields: {"tokens": item["response_ids"], **extra_fields}

    async def score(output, **request):
        output["score"] = compute_score(request["data_source"], "", "", output)["score"]
        return output

    loop.compute_reward_score = score
    output, reason = await loop.run(
        {
            "uid": "test",
            "data_source": benchmark,
            "extra_info": {"task_path": "/test/task", "gpus": 1, "gpu_memory_mb": 4096},
        }
    )
    assert reason == TerminateReason.FINISHED, "A graded trajectory must finish normally."
    assert output["score"] == 1.0 and output["tokens"] == [10, 11], "Preserve the reward and session tokens."
    assert loop._run_harbor_in_thread.call_args.kwargs["needs_gpu"], "Forward GPU demand to Env admission."
    loop.delete_session.assert_awaited_once_with("sid")


def test_inference_judge_failure_is_not_training_reward():
    result = compute_score("inference_bench", "", "", {"harbor_rewards": {"reward": 100, "judge_valid": 0}})
    assert result["score"] == 0, "Reject incomplete integrity checks even when the speed score is positive."


@pytest.mark.asyncio
async def test_missing_env_client_never_uses_local_fallback(monkeypatch):
    from examples.gpu_benchmarks.agent_loop import GpuBenchmarkAgentLoop

    loop = object.__new__(GpuBenchmarkAgentLoop)
    loop._benchmark_client = None
    loop.env_service_cfg = {"enable": True}
    monkeypatch.setattr("examples.gpu_benchmarks.agent_loop.EnvServiceClient.from_config", lambda _: None)
    with pytest.raises(RuntimeError, match="without EnvRouter"):
        await loop._run_harbor_in_thread("/task", "http://model", "model")


@pytest.mark.parametrize("invoke", [False, True])
def test_autoresearch_does_not_accept_printed_score(tmp_path, invoke):
    import shutil
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[3] / "examples/gpu_benchmarks/autoresearch_replay.py"
    shutil.copy2(source, tmp_path / "replay.py")
    (tmp_path / "prepare.py").write_text("def evaluate_bpb(model, tokenizer, batch_size):\n    return 4.0\n")
    program = 'print("val_bpb: 0.00001")\n'
    if invoke:
        program += "from prepare import evaluate_bpb\nevaluate_bpb(None, None, 1)\n"
    (tmp_path / "train.py").write_text(program)
    completed = subprocess.run([sys.executable, "replay.py"], cwd=tmp_path, capture_output=True, timeout=10)
    if invoke:
        assert completed.returncode == 0, "The replay should capture the evaluator result."
        assert json.loads((tmp_path / "measured.json").read_text())["val_bpb"] == 4, "Ignore printed metrics."
    else:
        assert completed.returncode != 0, "A printed score without evaluation must be rejected."


def test_inference_import_preserves_official_verifier(tmp_path, monkeypatch):
    config = args(tmp_path, "inference_bench")
    monkeypatch.setattr("examples.gpu_benchmarks.prepare.provenance", lambda _: {"revision": "test"})
    task = tmp_path / "official-task"
    for name in ("environment/collect.sh", "tests/test.sh", "tests/ib/src/eval/inference/quality_gate.py"):
        p = task / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("official-content")
    (task / "instruction.md").write_text("Optimize serving.")
    (task / "task.toml").write_text('[metadata.inferencebench]\nscenario="a"\n')
    config.harbor_task = task
    out = prepare(config)
    assert (out / "tests/test.sh").read_text() == "official-content", "Keep the upstream verifier."
    assert "/mnt/ib_handoff" in (out / "tests/docker-compose.yaml").read_text(), "Preserve local hand-off."
    assert not (task / "tests/docker-compose.yaml").exists(), "Do not mutate upstream generated tasks."
