"""Compare admission and runtime locality using identical original runners."""

import argparse
import asyncio
import dataclasses
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import tomllib

BASELINE = "0eeead861ffafed6bf31c962d259fd847775dcf9"


def prepare_copy(source, target, gpu, static_gpu_overlay, yaml, runtime_cache=None, runtime_prefix=None):
    """Keep task semantics and container paths while changing the dependency source."""
    shutil.copytree(source, target, symlinks=True)
    config = tomllib.loads((target / "task.toml").read_text())
    if config.get("verifier", {}).get("environment", {}).get("gpus", 0):
        compose = target / "tests/docker-compose.yaml"
        content = yaml.safe_load(compose.read_text()) if compose.exists() else {"services": {"main": {}}}
        main = content["services"]["main"]
        overlay = static_gpu_overlay(gpu)["services"]["main"]
        main["devices"] = overlay["devices"]
        main.setdefault("volumes", []).extend(overlay["volumes"])
        main.setdefault("environment", {})["LD_LIBRARY_PATH"] = "/usr/lib64:/usr/local/cuda/lib64"
        compose.write_text(yaml.safe_dump(content))
    if runtime_cache:
        for relative in ("environment/docker-compose.yaml", "tests/docker-compose.yaml"):
            compose = target / relative
            content = yaml.safe_load(compose.read_text())
            volumes = content["services"]["main"]["volumes"]
            expected = f"{runtime_prefix}:{runtime_prefix}:ro"
            if volumes.count(expected) != 1:
                raise ValueError(f"Require exactly one immutable runtime mount in {compose!s}.")
            volumes[volumes.index(expected)] = f"{runtime_cache}:{runtime_prefix}:ro"
            compose.write_text(yaml.safe_dump(content))


def command(*args):
    return subprocess.check_output(args, text=True, timeout=20).strip()


async def run(args):
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "warm_router_study"))
    from baseline import gpu_snapshot, load_old_runner, static_gpu_overlay

    harbor_type, runtime_type, runner, source = load_old_runner(args.baseline)
    import yaml

    manifest = json.loads(args.manifest.read_text())
    evidence = dict(
        baseline=BASELINE,
        runner_source=str(source),
        runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        patch=command("git", "-C", str(args.baseline), "diff"),
        scope="Real oracle episodes. Both arms use the old runner with an external admission prototype.",
        policies=args.policies,
        runtime_cache=str(args.runtime_cache) if args.runtime_cache else None,
        runtime_prefix=str(args.runtime_prefix) if args.runtime_prefix else None,
        budget_s=args.budget,
        blocks=[],
        source_locations={
            name: str(inspect.getfile(sys.modules[name])) for name in ("examples.sciaccel_rl.runner", "harbor")
        },
        tasks=manifest,
    )

    def save():
        path = args.output / "evidence.tmp"
        path.write_text(json.dumps(evidence, indent=2, default=str))
        path.replace(args.output / "evidence.json")

    overlays = []
    before = [gpu_snapshot(gpu) for gpu in args.gpus]
    evidence["before"] = before
    if any(row["pids"] or row["used_mb"] > 128 for row in before):
        raise RuntimeError("Select physically idle devices.")
    for index, row in enumerate(before):
        path = args.output / f"gpu-{index}.yaml"
        path.write_text(yaml.safe_dump(static_gpu_overlay(row)))
        overlays.append(path)
    for item in manifest:
        task = Path(item["path"])
        config = tomllib.loads((task / "task.toml").read_text())
        for env in (config["environment"], config.get("verifier", {}).get("environment", {})):
            if env.get("docker_image"):
                command("docker", "image", "inspect", env["docker_image"])
        item["task_sha256"] = hashlib.sha256((task / "task.toml").read_bytes()).hexdigest()
    evidence["task_transport_patch"] = "Identical device and driver mappings added to each GPU verifier compose copy."
    for item in manifest:
        item["copies"] = {}
        for policy in dict.fromkeys(args.policies):
            item["copies"][policy] = []
            for slot, gpu in enumerate(before):
                target = args.output / "tasks" / policy / str(slot) / Path(item["path"]).name
                prepare_copy(
                    item["path"],
                    target,
                    gpu,
                    static_gpu_overlay,
                    yaml,
                    args.runtime_cache if policy == "local" else None,
                    args.runtime_prefix,
                )
                item["copies"][policy].append(str(target))
    save()
    started = time.monotonic()
    try:
        async with asyncio.timeout(args.budget):
            for block_index, policy in enumerate(args.policies):
                block = dict(policy=policy, index=block_index, tasks=[])
                evidence["blocks"].append(block)
                available = asyncio.Queue()
                for index in range(len(args.gpus)):
                    available.put_nowait(index)
                t0 = time.monotonic()

                async def execute(index, item, policy=policy, available=available, block=block, t0=t0):
                    gpu_slot = index % len(args.gpus)
                    if policy == "pool":
                        gpu_slot = await available.get()
                    admitted = time.monotonic()
                    tag = "admit-study-" + uuid.uuid4().hex[:10]
                    row = dict(family=item["family"], tag=tag, gpu=args.gpus[gpu_slot], wait_s=admitted - t0)
                    block["tasks"].append(row)
                    config = tomllib.loads((Path(item["path"]) / "task.toml").read_text())
                    runtime = runtime_type(
                        harbor=harbor_type(
                            jobs_dir=str(args.output / "jobs"),
                            agent_name="oracle",
                            override_gpus=0,
                            gpu_compose_override=str(overlays[gpu_slot]),
                            max_concurrent_episodes=8,
                        ),
                        task_timeout_sec=min(300, config["agent"]["timeout_sec"]),
                        verifier_timeout_sec=min(300, config["verifier"]["timeout_sec"]),
                    )
                    try:
                        result = await runner(
                            task_path=item["copies"][policy][gpu_slot],
                            model_base_url="http://127.0.0.1:1/v1",
                            model_name="unused",
                            config=runtime,
                            needs_gpu=True,
                            session_id=tag,
                            actor_id=tag,
                            max_turns=1,
                            regrade_unverified=False,
                        )
                        row["result"] = dataclasses.asdict(result)
                        row["passed"] = (
                            result.exception is None
                            and bool(result.rewards)
                            and result.rewards.get("eval_valid", 1) == 1
                        )
                    except Exception as exc:
                        row["error"] = repr(exc)
                        row["passed"] = False
                    finally:
                        row["service_s"] = time.monotonic() - admitted
                        row["completion_s"] = time.monotonic() - t0
                        row["remaining_containers"] = command(
                            "docker", "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"
                        )
                        if policy == "pool":
                            available.put_nowait(gpu_slot)
                        save()

                tasks = [asyncio.create_task(execute(i, item)) for i, item in enumerate(manifest)]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                block["makespan_s"] = time.monotonic() - t0
                block["after"] = [gpu_snapshot(gpu) for gpu in args.gpus]
                save()
                print(
                    json.dumps(
                        dict(
                            block=block_index,
                            policy=policy,
                            seconds=block["makespan_s"],
                            passed=sum(row.get("passed", False) for row in block["tasks"]),
                        )
                    ),
                    flush=True,
                )
                if any(not row.get("passed") or row["remaining_containers"] for row in block["tasks"]):
                    raise RuntimeError("Task or cleanup failed. Stop the matrix without replacing failed cases.")
                if any(row["pids"] or row["used_mb"] > 128 for row in block["after"]):
                    raise RuntimeError("Physical release is unconfirmed. Stop before the next block.")
    except BaseException as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        evidence["elapsed_s"] = time.monotonic() - started
        evidence["after"] = [gpu_snapshot(gpu) for gpu in args.gpus]
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument(
        "--policies", nargs="+", choices=("static", "pool", "local"), default=["static", "pool", "pool", "static"]
    )
    parser.add_argument("--runtime-cache", type=Path)
    parser.add_argument("--runtime-prefix", type=Path)
    args = parser.parse_args()
    if "local" in args.policies and not (args.runtime_cache and args.runtime_prefix):
        parser.error("Local execution requires both runtime paths.")
    asyncio.run(run(args))
