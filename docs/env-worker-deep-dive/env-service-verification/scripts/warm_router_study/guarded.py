"""Opt into evidence-gated scheduling on the existing PSRL fixed-step services."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

A, B = "Qwen3-4B-Base", "Qwen3-1.7B-Base"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, required=True)
    result.add_argument("--assets", type=Path, required=True)
    result.add_argument("--python-prefix", type=Path, default=Path(sys.prefix))
    result.add_argument("--container-python", required=True)
    result.add_argument("--node-ip", required=True)
    result.add_argument("--gpus", type=int, nargs=2, required=True)
    result.add_argument("--image", required=True)
    result.add_argument("--budget-s", type=float, required=True)
    result.add_argument("--mode", choices=("off", "observe", "auto"), default="off")
    result.add_argument("--target-steps", type=int, default=1024)
    result.add_argument("--blocker-steps", type=int, default=2048)
    result.add_argument("--target-version", required=True, help="Immutable local target weight revision.")
    result.add_argument("--blocker-version", required=True, help="Immutable local blocker weight revision.")
    return result


def validate(args):
    if not 1 <= args.target_steps <= 8192 or not 96 <= args.blocker_steps <= 8192:
        raise ValueError("Target steps require 1..8192 and blocker steps require 96..8192.")
    if not 0 < args.budget_s <= 2700:
        raise ValueError("The work budget must be positive and at most 2700 seconds.")
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        raise ValueError("Two distinct nonnegative physical GPU indices are required.")
    if any(not value.strip() for value in (args.target_version, args.blocker_version)):
        raise ValueError("Immutable weight revisions must be nonempty.")


def freeze_policy(context):
    """Archive the exact imported policy sources separately from historic matrices."""
    folder = context["root"] / "guarded-policy-source"
    folder.mkdir()
    hashes = {}
    for name in (
        "warm_placement",
        "step_predictor",
        "warm_dispatch",
        "command_prediction",
        "predictive_scheduler",
        "managed_command",
    ):
        module = importlib.import_module("psrl.workers.env_service." + name)
        source = Path(module.__file__)
        target = folder / source.name
        shutil.copy2(source, target)
        hashes[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


async def shutdown_services(context):
    """Stop only this harness's containers and confirm their physical GPU release."""
    from experiment import docker
    from runtime_guard import await_cleanup

    deadline = time.monotonic() + 120

    def remaining(limit):
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("Provider physical cleanup exceeded its 120 second budget.")
        return min(limit, left)

    for environment in reversed(context["envs"]):
        await await_cleanup(environment.stop(delete=True), deadline, timeout_s=35)
    owned = await asyncio.to_thread(docker, "ps", "-aq", "--filter", f"label=psrl.actor_id={context['record']['tag']}")
    if owned:
        raise RuntimeError("Owned service containers remain after provider shutdown.")
    devices = tuple(context["record"]["gpu_before"])
    snapshot_code = """import json
import sys
from examples.gpu_benchmarks.run import gpu_snapshot
print(json.dumps({int(value): gpu_snapshot(int(value)) for value in sys.argv[1:]}))
"""
    for _ in range(10):
        # A separate process bounds native GPU telemetry calls as well as Python waits.
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", snapshot_code, *map(str, devices)],
            text=True,
            capture_output=True,
            check=True,
            timeout=remaining(10),
        )
        observations = json.loads(result.stdout)
        if all(not value["pids"] and value["used_mb"] < 128 for value in observations.values()):
            context["record"]["guarded_provider_cleanup"] = {
                "containers_removed": True,
                "physical_release_confirmed": True,
                "lease_cleanup_owner": "outer experiment harness",
            }
            context["save"]()
            return
        await asyncio.sleep(remaining(1))
    raise RuntimeError("Provider shutdown could not confirm physical release within ten observations.")


async def wait_for_arrival(context, blocker, command_id):
    deadline = time.monotonic() + min(30, context["remaining"]())
    while time.monotonic() < deadline:
        if blocker.done():
            blocker.result()
            raise RuntimeError("The blocker completed before the requested arrival boundary.")
        try:
            value = json.loads(context["progress_file"](0).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            value = {}
        if value.get("id") == command_id and value.get("completed_steps", 0) >= 80:
            # Allow the provider's bounded progress polling to consume this report.
            await asyncio.sleep(0.05)
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("No blocker arrival report within the bounded observation window.")


async def study(context, args):
    from prediction_provider import FixedStepServiceProvider
    from psrl.workers.env_service.managed_command import ManagedCommandDispatcher
    from psrl.workers.env_service.predictive_scheduler import (
        CommandCostProfile,
        PlacementConfig,
        PredictionAwareScheduler,
    )

    record, request, save = context["record"], context["request"], context["save"]
    preparation_started = time.monotonic()
    record["guarded_protocol"] = {
        "mode": args.mode,
        "target_steps": args.target_steps,
        "blocker_steps": args.blocker_steps,
        "profile_commands_per_device": 3,
        "calibration_commands": 3,
        "arrival_completed_steps": 80,
        "primary_timing": "before blocker submission through both responses",
        "off_reference": "current managed affinity, not the fixed historical git baseline",
        "scope": "Fixed-step native PSRL services. No LLM rollout or policy training.",
    }
    record["guarded_policy_hashes"] = freeze_policy(context)
    class_names = {value["name"] for value in record["gpu_before"].values()}
    if len(class_names) != 1:
        raise ValueError("This bounded runner requires two matching GPU classes.")
    ready = record["service_ready"]
    versions = {(item["torch"], item["transformers"]) for item in ready}
    if len(versions) != 1:
        raise ValueError("The two service runtime versions differ.")
    server_source = context["root"] / "source-snapshot/warm_router_study/server.py"
    worker_source = context["root"] / "source-snapshot/joint_gpu_study/worker.py"
    code_version = hashlib.sha256(server_source.read_bytes() + worker_source.read_bytes()).hexdigest()
    data = args.assets / "data/gsm8k/public/rows.json"
    input_signature = (
        "first-two-rows;prompt<=192;answer<=64;bf16;eager;norm-only;" + hashlib.sha256(data.read_bytes()).hexdigest()
    )
    # The native harness verifies a single local Ray node and node-local progress files.
    context["progress_clock"] = "host_monotonic"
    provider = FixedStepServiceProvider(
        context,
        device_class=next(iter(class_names)),
        model_versions={A: args.target_version, B: args.blocker_version},
        code_version=code_version,
        runtime_version=repr((next(iter(versions)), record["image_id"])),
        input_signature=input_signature,
        shutdown=lambda: shutdown_services(context),
    )
    # Real independent full commands measure A at exactly the requested work size.
    measurements = {str(index): [] for index in (0, 1)}
    record["guarded_profiles"] = measurements
    for index in (0, 1):
        for ordinal in range(3):
            await request(index, {"op": "clear"})
            started = time.monotonic()
            value = await request(
                index,
                {
                    "id": f"profile-{index}-{ordinal}",
                    "model": A,
                    "steps": args.target_steps,
                    "progress_every": 16,
                },
            )
            elapsed = time.monotonic() - started
            execution_s = elapsed - value["load_s"]
            if execution_s <= 0:
                raise RuntimeError("Profile timing boundaries produced nonpositive execution service.")
            measurements[str(index)].append(
                {
                    "id": value["id"],
                    "rebuild_s": value["load_s"],
                    "execution_s": execution_s,
                    "observed_at_s": time.monotonic(),
                }
            )
            save()
        await request(index, {"op": "clear"})
    await provider.initialize()
    # Observation learns B without using the forecast to place warmup commands.
    scheduler = PredictionAwareScheduler(PlacementConfig(mode="observe"))
    record["guarded_calibration"] = []
    async with ManagedCommandDispatcher(scheduler, provider, close_timeout_s=150) as dispatcher:
        for ordinal in range(3):
            result = await dispatcher.submit(provider.request(f"calibration-{ordinal}", B, args.blocker_steps), {})
            record["guarded_calibration"].append(result.value)
            save()
        # A becomes resident only on the hot service through the same managed path.
        await dispatcher.submit(provider.request("prepare-hot-target", A, args.target_steps), {})
        scheduler.config = dataclasses.replace(scheduler.config, mode=args.mode)
        costs = {
            device: CommandCostProfile(
                provider.workload(A),
                provider.model_id(A),
                args.target_steps,
                provider.model_bytes[A],
                provider.request("profile-footprint", A, args.target_steps).workspace_bytes,
                tuple(item["rebuild_s"] for item in samples),
                tuple(item["execution_s"] for item in samples),
                min(item["observed_at_s"] for item in samples),
            )
            for device, samples in measurements.items()
        }
        record["guarded_preparation_s"] = time.monotonic() - preparation_started
        record["guarded_registered_optimizer_updates"] = 8 * args.target_steps + 4 * args.blocker_steps
        started = time.monotonic()
        async with asyncio.TaskGroup() as tasks:
            blocker = tasks.create_task(
                dispatcher.submit(provider.request("formal-blocker", B, args.blocker_steps), {})
            )
            await wait_for_arrival(context, blocker, "formal-blocker")
            arrival = time.monotonic()
            target = tasks.create_task(
                dispatcher.submit(provider.request("formal-target", A, args.target_steps), costs)
            )
        record["guarded_result"] = {
            "decision": dataclasses.asdict(target.result().decision),
            "target": target.result().value,
            "blocker": blocker.result().value,
            "primary_makespan_s": time.monotonic() - started,
            "target_latency_s": target.result().value["finished_monotonic"] - arrival,
        }
        save()


def main():
    args = parser().parse_args()
    validate(args)
    from experiment import run

    asyncio.run(run(args, lambda context: study(context, args)))


if __name__ == "__main__":
    main()
