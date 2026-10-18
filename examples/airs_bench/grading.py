"""Trusted AIRS evaluation outside the agent's filesystem and Python environment."""

from __future__ import annotations

import json
import math
import shlex
from pathlib import Path
from typing import Any

import yaml
from examples.airs_bench.config import AirsBenchRuntimeConfig
from psrl.workers.env_worker.sandbox import CodeExecutionSpec, SandboxSpec

WORKSPACE = "/home/agent/workspace"
PYTHON = "/home/agent/miniconda3/envs/mlgym_generic/bin/python"


def task_config(runtime: AirsBenchRuntimeConfig, path: str) -> tuple[dict, Path]:
    """Resolve task assets from the prepared config root without changing MLGym globals."""
    root = (Path(runtime.data_root) / "airs_configs").resolve()
    source = Path(path)
    source = source.resolve() if source.is_absolute() else (root / source).resolve()
    if not source.is_relative_to(root):
        raise ValueError("Task config must be inside the prepared config root.")
    task = yaml.safe_load(source.read_text())
    if task.get("task_entrypoint", "CSVSubmissionTasks") != "CSVSubmissionTasks":
        raise ValueError("AIRS remote grading requires CSV submissions.")
    return task, root


def build_verifier_spec(
    runtime: AirsBenchRuntimeConfig,
    dataset_data_path: str,
    task_path: str,
    labels: dict[str, str],
) -> tuple[SandboxSpec, str]:
    """Mount trusted grading assets and hidden labels only into the verifier."""
    task, root = task_config(runtime, task_path)
    paths = task.get("evaluation_paths", [])
    if len(paths) != 1:
        raise ValueError("AIRS grading requires exactly one evaluation script.")
    assets = (root / "data" / task["id"]).resolve()
    script = (assets / paths[0]).resolve()
    if not script.is_relative_to(assets) or not script.is_file():
        raise ValueError("Evaluation script must be a file inside the trusted task assets.")
    if not (Path(dataset_data_path) / "test_with_labels").is_dir():
        raise ValueError("Prepared private test labels are missing.")
    is_apps = task["id"] == "CodeGenerationAPPSPassAt5"
    apps_dir = str(Path(__file__).resolve().parent / "apps")
    code_execution = (
        CodeExecutionSpec(
            python=PYTHON,
            runner_path="/opt/psrl-code/candidate.py",
            runner_host_dir=apps_dir,
            cpus=runtime.apps_candidate_cpus,
            memory=runtime.apps_candidate_memory,
            case_timeout_s=runtime.apps_case_timeout_s,
            candidate_timeout_s=runtime.apps_candidate_timeout_s,
        )
        if is_apps
        else None
    )
    spec = SandboxSpec(
        image=runtime.image,
        cpus=runtime.verifier_cpus,
        memory=runtime.verifier_memory,
        mounts=(
            (str(Path(dataset_data_path).resolve()), f"{WORKSPACE}/data", "ro"),
            (apps_dir, "/opt/psrl-apps", "ro") if is_apps else (str(assets), "/opt/airs-verifier", "ro"),
        ),
        network="none",
        labels={**labels, "psrl.airs_role": "verifier"},
        startup_timeout_s=runtime.startup_timeout_s,
        code_execution=code_execution,
        auxiliary_cpus=runtime.apps_candidate_cpus if is_apps else 0,
        auxiliary_memory=runtime.apps_candidate_memory if is_apps else "0",
    )
    command = (
        f"cd {WORKSPACE} && {PYTHON} /opt/airs-verifier/{shlex.quote(str(script.relative_to(assets)))} "
        f"--submission-file {WORKSPACE}/submission.csv"
    )
    return spec, "" if is_apps else command


def parse_metrics(output: str, metric: str | None) -> dict[str, float]:
    """Accept only finite numeric metrics from the trusted evaluator's final JSON object."""
    decoder = json.JSONDecoder()
    for index in reversed([i for i, char in enumerate(output) if char == "{"]):
        try:
            value, end = decoder.raw_decode(output[index:])
        except ValueError:
            continue
        if output[index + end :].strip() or not isinstance(value, dict) or not value:
            continue
        if metric is not None and metric not in value:
            continue
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value.values()
        ):
            raise ValueError("Evaluator returned invalid numeric metrics.")
        return value
    raise ValueError("Evaluator did not return the required metric.")


async def grade_submission(handle: Any, spec: SandboxSpec, command: str, metric: str | None, timeout_s: float) -> dict:
    """Grade a snapshot without returning private evaluator diagnostics to the agent."""
    result = await handle.evaluate(spec, command, timeout_s)
    if result.timed_out:
        raise RuntimeError("Verifier timed out.")
    if result.exit_code == 2:
        raise ValueError("Submission could not be evaluated. Check the documented CSV format.")
    if result.exit_code != 0:
        raise RuntimeError("Trusted verifier failed. Discard this episode.")
    try:
        return parse_metrics(result.stdout, metric)
    except ValueError as error:
        raise RuntimeError("Trusted verifier returned invalid metrics. Discard this episode.") from error
