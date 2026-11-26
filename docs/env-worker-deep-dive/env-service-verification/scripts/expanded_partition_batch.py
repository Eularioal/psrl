"""Compare static Episode, static Command, and dynamic Command on eight GPUs."""

import argparse
import hashlib
import json
import math
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from analyze_handover_overlap import handovers
from analyze_integrated_partition import analyze
from capacity_validation_pipeline import save
from episode_partition_followup import replace_once
from partition_followup import run_prepared, verify_release

GPU_INDICES = range(8)
LAYOUT = {"training": [0, 1], "rollout": [2, 3, 4], "dedicated_environment": [5, 6, 7]}
GROUPS = 10
VISION_TASKS = ("imageClassificationCifar10", "imageCaptioningCOCO")
ARMS = {
    "episode_static": ("fixed", "episode"),
    "command_static": ("fixed", "command"),
    "command_elastic": ("resident", "command"),
}
EPISODE_COMMAND_CAP = (
    "max_gpu_commands=args.dedicated_cap,",
    'max_gpu_commands=args.dedicated_cap if args.gpu_reservation == "command" else 0,',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_data(args, target):
    """Copy benchmark inputs and calculate the expected environment updates."""
    if args.bench_family == "vision":
        import numpy as np

        payloads = {}
        for task in VISION_TASKS:
            for directory, filename in (
                ("public", "config.json"),
                ("public", "images.npz"),
                ("public", "train_labels.json"),
                ("private", "test_labels.json"),
            ):
                name = f"{task}/{directory}/{filename}"
                payloads[name] = args.data / name
        updates_per_epoch = {}
        for task in VISION_TASKS:
            with np.load(args.data / task / "public/images.npz", allow_pickle=False) as data:
                train_rows, test_rows = len(data["train_x"]), len(data["test_x"])
            labels = json.loads((args.data / task / "public/train_labels.json").read_text())
            held_out = json.loads((args.data / task / "private/test_labels.json").read_text())
            if not train_rows or len(labels) != train_rows or not test_rows or len(held_out) != test_rows:
                raise ValueError(f"Images and labels differ for {task!r}.")
            updates_per_epoch[task] = math.ceil(train_rows / 16)
        groups = [(VISION_TASKS[i % len(VISION_TASKS)], args.command_updates) for i in range(args.steps * args.groups)]
        expected_updates = 2 * sum(updates_per_epoch[task] * epochs for task, epochs in groups)
    else:
        payloads = {f"{split}/rows.json": args.data / split / "rows.json" for split in ("public", "private")}
        public, private = (json.loads(path.read_text()) for path in payloads.values())
        if len(public) < 2 or not private or any(not row["prompt"] or not row["answer"] for row in public + private):
            raise ValueError("PostTrain needs two valid adaptation examples and held-out scoring data.")
        if {r["prompt"] for r in public[:2]} & {r["prompt"] for r in private}:
            raise ValueError("Adaptation and private scoring examples overlap.")
        groups = [
            ("aime2025", args.command_updates * (2 if i % args.groups == args.groups - 1 else 1))
            for i in range(args.steps * args.groups)
        ]
        expected_updates = 2 * sum(updates for _, updates in groups)
    for name, path in payloads.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    hashes = {name: digest(target / name) for name in payloads}
    if args.bench_family != "vision":
        hashes = {name.split("/")[0]: value for name, value in hashes.items()}
    return groups, expected_updates, hashes


def set_vision_protocol(runtime, agent):
    """Reuse the CUDA MLGym vision task with the policy-selected learning rate."""
    vision = runtime / "examples/mlgym_bench/vision.py"
    replace_once(vision, "*, epochs=2, training_hook=None", "*, epochs=2, training_hook=None, learning_rate=3e-3")
    replace_once(vision, "lr=3e-3)", "lr=learning_rate)")
    worker = runtime / "partition_worker.py"
    shutil.copy2(Path(__file__).with_name("partition_vision_worker.py"), worker)
    replace_once(
        agent,
        "f\"{settings['env_model']}:/model:ro\",",
        "",
    )
    replace_once(
        agent,
        '"/benchmark/public" if mode == "train" else "/benchmark/private",',
        """f"/benchmark/{request['extra_info']['benchmark']}/"
                    f"{'public' if mode == 'train' else 'private'}",""",
    )
    replace_once(
        agent,
        '"Choose a learning rate for adapting only the final normalization weights of Qwen3-4B "\n'
        '                    "on two GSM8K training examples. The budget is fixed. '
        'Evaluation uses private answer likelihood. "',
        """f"Choose a learning rate for the MLGym {request['extra_info']['benchmark']} image task. "
                    "The epoch budget is fixed, and evaluation uses held-out targets. " """.rstrip(),
    )
    replace_once(agent, "The adapter completed", "The image model completed")
    replace_once(agent, "held-out answer likelihood", "the independent held-out image score")
    replace_once(
        agent,
        """f"Private answer NLL is {verified['answer_nll']:.6f} "
                        f"and exp(-NLL) is {verified['reward']:.6f}. " """.rstrip(),
        """f"Held-out {verified['metric_name']} is {verified['reward']:.6f}. " """.rstrip(),
    )
    replace_once(agent, "PostTrain command failed", "MLGym vision command failed")
    return [str(path.relative_to(runtime)) for path in (vision, worker)]


def prepare(args):
    """Freeze one workload and identical implementation for the selected arms."""
    import pandas as pd

    reference = args.reference.resolve()
    prior = json.loads((reference / "supervisor.json").read_text())
    design = json.loads((reference / "design.json").read_text())
    if prior["status"] != "complete" or not prior["physical_release_verified"]:
        raise ValueError("The reference must have completed and released its GPUs.")
    manifest = json.loads((reference / "source-manifest.json").read_text())
    if any(digest(reference / "source" / name) != expected for name, expected in manifest.items()):
        raise ValueError("The frozen reference has changed.")
    model_dirs = (args.model,) if args.bench_family == "vision" else (args.model, args.env_model)
    for model_dir in model_dirs:
        path = model_dir / "config.json"
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output.mkdir(parents=True, exist_ok=False)
    runtime = args.output / "frozen"
    shutil.copytree(reference / "source", runtime, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
    runner = runtime / "integrated_partition_training.py"
    old_runner = runner.read_text()
    replace_once(
        runner,
        old_runner[old_runner.index("def gpu_layout(") : old_runner.index("def prepare(")],
        f"def gpu_layout(rollout_replicas):\n"
        f"    if rollout_replicas != {len(LAYOUT['rollout'])}:\n"
        '        raise ValueError("The expanded experiment requires three rollout replicas.")\n'
        f"    return {LAYOUT!r}\n\n\n",
    )
    replace_once(
        runner,
        'EnvLease("probe", "probe", "probe", args.node_ip, (4,),',
        'EnvLease("probe", "probe", "probe", args.node_ip, (5,),',
    )
    code = runner.read_text()
    if code.count("probe.read_gpu_snapshot(range(6))") != 3:
        raise ValueError("Unexpected GPU telemetry paths in the frozen reference.")
    runner.write_text(code.replace("probe.read_gpu_snapshot(range(6))", "probe.read_gpu_snapshot(range(8))"))
    replace_once(
        runner,
        'for gpu in layout["dedicated_environment"] + layout["rollout"]:',
        'for gpu in layout["dedicated_environment"] + (layout["rollout"] if mode == "resident" else []):',
    )
    replace_once(runner, 'gpu_reservation="command",', "gpu_reservation=args.gpu_reservation,")
    replace_once(
        runner,
        'command_gpu_pool="environment",',
        'command_gpu_pool="environment" if args.gpu_reservation == "command" else "",',
    )
    replace_once(runner, *EPISODE_COMMAND_CAP)
    replace_once(
        runner,
        '            "train_actor_rollout_ref.actor.rollout_n=2",',
        '            "train_actor_rollout_ref.actor.rollout_n=2",\n'
        '            "train_actor_rollout_ref.model.use_fused_kernels=False",',
    )
    replace_once(
        runner,
        "            router_name=tag,",
        "            router_name=tag,\n"
        "            episode_timeout_s=args.episode_timeout_s,\n"
        "            gpu_wait_timeout_s=args.gpu_wait_timeout_s,\n"
        "            env_gpu_memory_mb=args.env_gpu_memory_mb,\n"
        "            container_memory_mb=args.container_memory_mb,",
    )
    replace_once(
        runner,
        '"psrl.env_service.default_gpu_memory_mb=16384",',
        'f"psrl.env_service.default_gpu_memory_mb={args.env_gpu_memory_mb}",',
    )
    replace_once(
        runner,
        '"gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.45",',
        'f"gen_actor_rollout_ref.rollout.gpu_memory_utilization={args.rollout_memory_utilization}",',
    )

    agent = runtime / "integrated_partition_agent.py"
    replace_once(agent, "asyncio.timeout(1200)", 'asyncio.timeout(self.settings["episode_timeout_s"])')
    replace_once(agent, "max_wait_s=600", 'max_wait_s=settings["gpu_wait_timeout_s"]')
    replace_once(agent, "wait_timeout_sec=600", 'wait_timeout_sec=settings["gpu_wait_timeout_s"]')
    replace_once(
        agent,
        "EnvDemand(2, 16384, True, 16384)",
        'EnvDemand(2, settings["container_memory_mb"], True, settings["env_gpu_memory_mb"])',
    )
    replace_once(
        agent,
        "cpus=2, memory_mb=16384),",
        'cpus=2, memory_mb=settings["container_memory_mb"]),',
    )
    if args.bench_family == "vision":
        bench_changes = set_vision_protocol(runtime, agent)
    else:
        bench_changes = []
        replace_once(agent, "on two GSM8K training examples", "on two AIME 2025 training examples")
        model_name = args.env_model.name.removesuffix("-Base")
        replace_once(agent, "normalization weights of Qwen3-4B", f"normalization weights of {model_name}")
    replace_once(
        agent,
        '"parent_id": str(request.get("parent_id")),',
        '"parent_id": str(request.get("parent_id")),\n'
        '            "benchmark": request.get("extra_info", {}).get("benchmark", "aime2025"),',
    )
    replace_once(agent, '"mode": "command",', '"mode": handle.lease.gpu_reservation,')
    replace_once(
        agent,
        '            record["host_unit"] = handle.lease.unit_id',
        '            record["host_unit"] = handle.lease.unit_id\n'
        '            record["gpu_reservation"] = handle.lease.gpu_reservation\n'
        '            record["lease_granted_at"] = time.time()\n'
        "            save()",
    )
    replace_once(
        agent,
        '"admission": environment.last_gpu_command,',
        '"admission": {**environment.last_gpu_command, "unit_id": '
        'environment.last_gpu_command.get("unit_id", handle.lease.unit_id)},',
    )
    ledger = runtime / "psrl/workers/env_service/ledger.py"
    replace_once(ledger, "                    self._load(item[1], demand, now),\n", "")

    trial_data = args.output / "datasets"
    groups, expected_updates, data_hashes = freeze_data(args, trial_data)
    prompt = "Adapt a language model and evaluate held-out likelihood."
    if args.bench_family == "vision":
        prompt = "Train an image model and evaluate it on held-out examples."
    rows = [
        {
            "data_source": "sciaccel_rl",
            "agent_name": "integrated_posttrain",
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "updates": updates,
                "task_name": f"{benchmark}-{i}" if args.bench_family == "vision" else f"aime2025-posttrain-{i}",
                "reward_key": "reward",
                **({"benchmark": benchmark} if args.bench_family == "vision" else {}),
            },
        }
        for i, (benchmark, updates) in enumerate(groups)
    ]
    tasks = args.output / "tasks.parquet"
    pd.DataFrame(rows).to_parquet(tasks, index=False)
    changed = sorted(name for name, expected in manifest.items() if digest(runtime / name) != expected)
    expected_changes = [
        "integrated_partition_agent.py",
        "integrated_partition_training.py",
        "psrl/workers/env_service/ledger.py",
        *bench_changes,
    ]
    if changed != sorted(expected_changes):
        raise ValueError(f"Unexpected change relative to the prior frozen code {changed!r}.")
    for name in changed:
        compile((runtime / name).read_text(), name, "exec")
    frozen = {name: digest(runtime / name) for name in manifest}
    requested = {
        "steps": args.steps,
        "trajectories": args.steps * args.groups * 2,
        "commands": args.steps * args.groups * 4,
        "requests": args.steps * args.groups * 6,
        "environment_updates": expected_updates,
    }
    limits = dict(
        design["limits"],
        arm_s=args.arm_timeout_s,
        episode_s=args.episode_timeout_s,
        gpu_wait_s=args.gpu_wait_timeout_s,
    )
    dedicated_cap = min(16, 92160 // args.env_gpu_memory_mb)
    protocol = {
        "reference_commit": design["commit"],
        "reference_changes": changed,
        "gpu_layout": LAYOUT,
        "expected": requested,
        "arms": args.arms,
        "groups_per_step": args.groups,
        "dedicated_cap": dedicated_cap,
        "shared_cap": 2,
        "model": args.model.name,
        "environment_model": args.env_model.name if args.env_model else "MLGym vision models",
        "rollout_memory_utilization": args.rollout_memory_utilization,
        "env_gpu_memory_mib": args.env_gpu_memory_mb,
        "container_memory_mb": args.container_memory_mb,
        "actor_fused_kernels": False,
        "data": args.data.name,
        "benchmarks": sorted({benchmark for benchmark, _ in groups}),
        "benchmark_trajectories": dict(Counter(benchmark for benchmark, _ in groups for _ in range(2))),
        "data_sha256": data_hashes,
        "tasks_sha256": digest(tasks),
        "arm_budget_s": args.arm_timeout_s,
        "limits": limits,
    }
    save(args.output / "protocol.json", protocol)
    for name in args.arms:
        mode, reservation = ARMS[name]
        out = args.output / name
        out.mkdir()
        shutil.copytree(runtime, out / "source")
        shutil.copy2(tasks, out / "tasks.parquet")
        settings = dict(
            design["args"],
            output=str(out),
            model=str(args.model),
            env_model=str(args.env_model) if args.env_model else "",
            data=str(trial_data),
            node_ip=args.node_ip,
            local_python_site=str(args.local_python_site),
            steps=args.steps,
            groups=args.groups,
            dedicated_cap=dedicated_cap,
            rollout_replicas=3,
            mode=mode,
            arm_timeout=args.arm_timeout_s,
            total_timeout=args.arm_timeout_s,
            elastic_partition=mode == "resident",
            gpu_reservation=reservation,
            episode_timeout_s=args.episode_timeout_s,
            gpu_wait_timeout_s=args.gpu_wait_timeout_s,
            env_gpu_memory_mb=args.env_gpu_memory_mb,
            container_memory_mb=args.container_memory_mb,
            rollout_memory_utilization=args.rollout_memory_utilization,
        )
        save(
            out / "design.json",
            dict(
                design,
                args=settings,
                gpu_budget=LAYOUT,
                steps_per_arm=args.steps,
                training_episodes_per_arm=requested["trajectories"],
                benchmark_trajectories=protocol["benchmark_trajectories"],
                limits=limits,
            ),
        )
        save(out / "source-manifest.json", frozen)


def run_arm(out, mode, expected):
    """Collect observed workload and resource state for one arm."""
    design = json.loads((out / "design.json").read_text())
    settings = design["args"]
    source = out / "source"
    if not verify_release(source, out, GPU_INDICES):
        raise RuntimeError("All eight target GPUs must be idle before starting an arm.")
    run_prepared(
        SimpleNamespace(
            output=out,
            mode=mode,
            budget_s=settings["arm_timeout"],
            rollout_replicas=3,
            local_python_site=settings["local_python_site"],
        ),
        source,
        gpu_indices=GPU_INDICES,
    )
    supervisor = json.loads((out / "supervisor.json").read_text())
    if (out / mode / "telemetry.jsonl").is_file():
        result = analyze(out / mode)
        save(out / "analysis.json", result)
        work = dict(
            steps=len(result["steps"]),
            trajectories=result["episodes_completed"],
            commands=result["commands_completed"],
            requests=result["queries_completed"],
            environment_updates=result["environment_updates_completed"],
        )
        records = [json.loads(path.read_text()) for path in (out / mode).glob("episodes/*/episode.json")]
        reservation = settings["gpu_reservation"]
        command_release = reservation == "command"
        holding_verified = len(records) == expected["trajectories"] and all(
            row.get("gpu_reservation") == reservation
            and all(item["admission"]["capacity_released"] == command_release for item in row["commands"])
            for row in records
        )
        layout_verified = mode != "fixed" or set(result["command_placements"]) <= {
            f"gpu-{gpu}" for gpu in LAYOUT["dedicated_environment"]
        }
        benchmark_counts = dict(Counter(row.get("benchmark", "aime2025") for row in records))
        benchmarks_verified = benchmark_counts == design.get("benchmark_trajectories", benchmark_counts)
        supervisor.update(
            work=work,
            hold_mode_verified=holding_verified,
            placement_verified=layout_verified,
            benchmark_trajectories=benchmark_counts,
            benchmarks_verified=benchmarks_verified,
            work_verified=work == expected and holding_verified and layout_verified and benchmarks_verified,
        )
        if mode == "resident":
            save(out / "handover.json", handovers(out))
    return supervisor


def run(output):
    """Execute the paired arms and preserve completion and resource evidence."""
    protocol = json.loads((output / "protocol.json").read_text())
    expected = protocol["expected"]
    state = {"status": "running", "arms": {}, "started_at": time.time()}
    save(output / "status.json", state)
    try:
        for name in protocol["arms"]:
            mode, _ = ARMS[name]
            state["arms"][name] = {"status": "initializing"}
            save(output / "status.json", state)
            supervisor = run_arm(output / name, mode, expected)
            state["arms"][name] = supervisor
            save(output / "status.json", state)
            save_comparison(output, state)
            if supervisor.get("interrupted"):
                raise InterruptedError("The active experiment arm was interrupted.")
            if not supervisor.get("physical_release_verified"):
                raise RuntimeError("The preceding arm has not physically released its devices.")
        state["status"] = (
            "complete"
            if all(arm["status"] == "complete" and arm.get("work_verified") for arm in state["arms"].values())
            else "incomplete"
        )
    except BaseException as exc:
        state.update(status="failed", error=repr(exc))
        raise
    finally:
        state["finished_at"] = time.time()
        save(output / "status.json", state)
        save_comparison(output, state)


def save_comparison(output, state):
    """Preserve training time and cold-start costs after each finished arm."""
    metrics = {}
    for name, supervisor in state["arms"].items():
        entry = {"status": supervisor["status"], "work": supervisor.get("work")}
        if supervisor["status"] == "complete" and supervisor.get("work_verified"):
            arm = output / name
            result = json.loads((arm / "analysis.json").read_text())
            steps = sorted((int(step), row["timing_s/step"]) for step, row in result["steps"].items())
            timings = result["timing_totals"]
            entry.update(
                complete_workload=True,
                initialization_s=result["status"]["initialization_s"],
                training_s=timings["timing_s/step"],
                first_step_s=steps[0][1],
                remaining_steps_median_s=statistics.median(value for _, value in steps[1:])
                if len(steps) > 1
                else None,
                wait_for_gen_s=timings["timing_s/wait_for_gen"],
                training_step_s=steps,
                launch_to_cleanup_s=result["launch_to_cleanup_s"],
                query_p95_s=result["query_latency_s"]["p95"],
                generated_tokens=result["generated_tokens"],
                command_placements=result["command_placements"],
            )
            if (arm / "handover.json").exists():
                transitions = json.loads((arm / "handover.json").read_text())["transitions"]
                entry["borrow_count"] = sum(row["borrowed"] for row in transitions)
        metrics[name] = entry
    comparison = {}
    method = metrics.get("command_elastic", {})
    if method.get("complete_workload"):
        for baseline in ("episode_static", "command_static"):
            ref = metrics.get(baseline, {})
            if ref.get("complete_workload"):
                comparison[baseline] = {
                    metric: 100 * (1 - method[metric] / ref[metric])
                    for metric in ("training_s", "launch_to_cleanup_s", "wait_for_gen_s", "query_p95_s")
                }
    save(output / "comparison.json", {"status": state["status"], "arms": metrics, "dynamic_savings_pct": comparison})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("reference", "output", "model", "data", "local-python-site"):
        parser.add_argument(f"--{key}", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--env-model", type=lambda value: Path(value).resolve())
    parser.add_argument("--node-ip", required=True)
    parser.add_argument(
        "--steps",
        type=int,
        choices=(1, 2, 10, 20),
        default=2,
        help="PSRL steps per arm. Use 2 for short or 10 for long.",
    )
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--arm-timeout-s", type=int, default=2100)
    parser.add_argument("--episode-timeout-s", type=int, default=1200)
    parser.add_argument("--gpu-wait-timeout-s", type=int, default=600)
    parser.add_argument("--bench-family", choices=("posttrain", "vision"), default="posttrain")
    parser.add_argument("--groups", type=int, default=GROUPS)
    parser.add_argument("--command-updates", type=int, default=448, help="PostTrain updates or MLGym vision epochs.")
    parser.add_argument("--rollout-memory-utilization", type=float, default=0.45)
    parser.add_argument("--env-gpu-memory-mb", type=int, default=16384)
    parser.add_argument("--container-memory-mb", type=int, default=16384)
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args()
    if len(options.arms) != len(set(options.arms)):
        parser.error("Each experiment arm must be unique.")
    if options.bench_family == "posttrain" and options.env_model is None:
        parser.error("PostTrain requires --env-model.")
    if min(options.groups, options.command_updates, options.container_memory_mb) <= 0:
        parser.error("Groups, command work, and container memory must be positive.")
    if not 0 < options.env_gpu_memory_mb <= 92160 or not 0 < options.rollout_memory_utilization < 1:
        parser.error("GPU reservations must fit the configured limits.")
    prepare(options)
    if not options.prepare_only:
        run(options.output)
