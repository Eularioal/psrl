"""Inspect saved science task contracts and call sites without running external services."""

import argparse
import ast
import json
from pathlib import Path

import tomllib


def attribute_calls(path, attribute, class_name=None):
    tree = ast.parse(path.read_text())
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attribute
    ]


def audit(source):
    # Diagram 1: A task's GPU requirement is distinct from the model's GPU placement.
    tasks = {}
    for kind in ("cpu-task", "cuda-task"):
        task = tomllib.loads((source / kind / "task.toml").read_text())
        tasks[kind] = {
            "name": task["task"]["name"],
            "environment_gpus": task["environment"].get("gpus", 0),
            "environment_cpus": task["environment"].get("cpus", 0),
            "verifier_mode": task["verifier"].get("environment_mode"),
            "verifier_gpus": task["verifier"].get("environment", {}).get("gpus", 0),
        }
    # Diagram 3: Attribute call sites are static evidence, not an execution trace.
    return {
        "scope": "Static source audit only. Zero external calls or task execution.",
        "tasks": tasks,
        "exec_gpu_call_lines": {
            "TruncatingTerminus2": attribute_calls(source / "agent.py", "exec_gpu", "TruncatingTerminus2"),
            "CommandPressureAgent": attribute_calls(source / "pressure-agent.py", "exec_gpu", "CommandPressureAgent"),
            "HarborTerminus2": attribute_calls(source / "harbor/terminus_2.py", "exec_gpu"),
            "HarborTmuxSession": attribute_calls(source / "harbor/tmux_session.py", "exec_gpu"),
            "HarborVerifier": attribute_calls(source / "harbor/verifier.py", "exec_gpu"),
        },
        "tmux_send_keys_call_lines": attribute_calls(source / "harbor/terminus_2.py", "send_keys"),
        "boundary": (
            "Absence of a call is not a runtime test. Review tool completion and grading semantics separately."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results/env-design-status-20261021/source",
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.source), indent=2))
