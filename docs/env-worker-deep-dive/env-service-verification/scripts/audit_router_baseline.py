"""Audit the requested Git baseline and replay admission on CPU, without Docker.

The old semaphore function is extracted unchanged from its AST. Separate global
namespaces represent two worker processes; this does not replay a Ray deployment.
Current admission calls the production EnvUnitLedger. No CUDA OOM is simulated.
"""

import argparse
import ast
import asyncio
import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

REPO = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
REV = "0eeead861ffafed6bf31c962d259fd847775dcf9"
sys.path.insert(0, str(REPO))

from psrl.workers.env_service import EnvDemand, EnvUnit, EnvUnitLedger, NodeRules


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


async def replay_old_gate(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_acquire_episode_slot")
    code = compile(ast.Module(body=[function], type_ignores=[]), "baseline-agent-loop", "exec")
    gates = []
    try:
        for _ in range(2):
            # Diagram 9a: separate workers have separate module-global semaphores.
            namespace = {
                "asyncio": asyncio,
                "psrl_logger": logging.getLogger(__name__),
                "_episode_gate": None,
                "_episode_gate_limit": 0,
            }
            exec(code, namespace)
            for _ in range(2):
                gates.append(await asyncio.wait_for(namespace["_acquire_episode_slot"](2), 1))
        return {"workers": 2, "per_worker_limit": 2, "total_admitted": len(gates)}
    finally:
        for gate in gates:
            gate.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [
        "examples/sciaccel_rl/agent_loop.py",
        "examples/sciaccel_rl/runner.py",
        "examples/sciaccel_rl/config/gpu-compose-override.yaml",
        "examples/sciaccel_rl/config/sciaccel_agent_config.yaml",
    ]
    sources, hashes = {}, {}
    for path in paths:
        source = git("show", f"{REV}:{path}")
        target = args.output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        sources[path] = source
        hashes[path] = hashlib.sha256(source.encode()).hexdigest()
    compose = sources[paths[2]]
    devices = re.findall(r"^\s*-\s*(/dev/nvidia\d+):", compose, flags=re.M)
    baseline = asyncio.run(replay_old_gate(sources[paths[0]]))
    baseline["when_static_override_enabled_device_for_each_episode"] = devices
    baseline["shipped_agent_config_override_empty"] = 'gpu_compose_override: ""' in sources[paths[3]]
    assert devices == ["/dev/nvidia0"]
    assert baseline["total_admitted"] == 4

    # Diagram 9b: four requests from any workers share one production ledger.
    units = [EnvUnit(f"gpu-{i}", "192.0.2.10", 16, 32768, (i,), 4, 80 * 1024) for i in (6, 7)]
    ledger = EnvUnitLedger(units, NodeRules.build("any"))
    demand = EnvDemand(4, 8192, True, 51 * 1024)
    decisions = [ledger.try_admit(f"request-{i}", demand) for i in range(4)]
    assert [d.status for d in decisions] == ["granted", "granted", "wait", "wait"]
    current = {
        "decisions": [d.status for d in decisions],
        "allocated_devices": [d.lease.gpu_indices for d in decisions if d.lease],
    }
    for d in decisions[:2]:
        ledger.release(d.lease.lease_id)
    later = [ledger.try_admit(f"request-{i}", demand) for i in (2, 3)]
    assert all(d.status == "granted" for d in later)
    for d in later:
        ledger.release(d.lease.lease_id)
    current["after_release"] = [d.status for d in later]
    assert all(s.containers_running == 0 for s in ledger.snapshot())

    matched_path = args.output.parent / "matched-comparison.json"
    matched = json.loads(matched_path.read_text())
    timings = {}
    for arm, data in matched["arms"].items():
        sums = {
            key: sum(s[f"timing_s/{key}"] for s in data["steps"])
            for key in ("wait_for_gen", "old_log_prob", "update_actor", "step")
        }
        sums["other"] = sums["step"] - sum(sums[k] for k in ("wait_for_gen", "old_log_prob", "update_actor"))
        timings[arm] = sums
    pressure_durations = {}
    for arm in ("off", "on"):
        by_code = {}
        for path in (args.output.parent / arm).glob("pool-*/harbor-jobs/*/*/*/agent/gpu-pressure-command.json"):
            record = json.loads(path.read_text())
            by_code.setdefault(record["command"]["return_code"], []).append(
                record["finished_at"] - record["started_at"]
            )
        pressure_durations[arm] = {
            str(code): {
                "count": len(values),
                "min_seconds": min(values),
                "max_seconds": max(values),
                "mean_seconds": sum(values) / len(values),
            }
            for code, values in by_code.items()
        }
    result = {
        "scope": "CPU admission replay plus historical timing analysis; NOT old-commit GPU training",
        "baseline_commit": REV,
        "current_head": git("rev-parse", "HEAD").strip(),
        "source_sha256": hashes,
        "baseline": baseline,
        "current_ledger": current,
        "historical_all_recorded_pressure_commands_by_exit_code": pressure_durations,
        "historical_current_off_on_first_two_steps_seconds": timings,
    }
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "diff-stat.txt").write_text(git("diff", "--stat", REV, "HEAD"))
    print(json.dumps(result, indent=2))

    # Historical measurements only: do not label these bars as the old baseline.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    left = [0.0, 0.0]
    for key, label, color in [
        ("wait_for_gen", "Wait for rollout results (includes queue/tasks)", "#4477aa"),
        ("old_log_prob", "Old log probability", "#ccbb44"),
        ("update_actor", "Actor update", "#228833"),
        ("other", "Other", "#bbbbbb"),
    ]:
        values = [timings[a][key] for a in ("off", "on")]
        ax.barh(["Current OFF", "Current ON"], values, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, values)]
    ax.set_xlabel("Seconds, first two completed steps")
    ax.set_title("Historical timing breakdown: current OFF vs ON\nNot a GPU run of 0eeead8; ON crashed after step 2")
    ax.legend(loc="lower right", fontsize=8)
    for i, value in enumerate(left):
        ax.text(value + 8, i, f"{value:.1f}s", va="center")
    ax.set_xlim(0, max(left) * 1.15)
    fig.tight_layout()
    fig.savefig(args.output / "historical-timing.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
