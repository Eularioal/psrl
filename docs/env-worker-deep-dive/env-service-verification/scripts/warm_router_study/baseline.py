"""Run the identical model services through the fixed historical PSRL runner.

Run this backend in a fresh Python process: it deliberately imports PSRL from the
historical worktree, never the current EnvRouter implementation. The callback is
the same bounded command protocol used by ``experiment.run``. A custom Harbor
agent only holds each episode open. It makes no language-model requests.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from runtime_guard import archive_runtime, await_cleanup, cancel_and_join

BASELINE = "0eeead861ffafed6bf31c962d259fd847775dcf9"
GPU_BUDGET_MB = 20480
HOST_MEMORY_MB = 49152
CPUS = 8
_EPISODES = {}


def docker(*arguments):
    return subprocess.check_output(["docker", *arguments], text=True, timeout=15).strip()


def gpu_snapshot(index):
    import pynvml as nv

    nv.nvmlInit()
    try:
        handle = nv.nvmlDeviceGetHandleByIndex(index)
        memory = nv.nvmlDeviceGetMemoryInfo(handle, version=nv.nvmlMemory_v2)
        return dict(
            uuid=nv.nvmlDeviceGetUUID(handle),
            minor=nv.nvmlDeviceGetMinorNumber(handle),
            total_mb=memory.total / 1024**2,
            used_mb=memory.used / 1024**2,
            pids=[process.pid for process in nv.nvmlDeviceGetComputeRunningProcesses(handle)],
        )
    finally:
        nv.nvmlShutdown()


def snapshot_scripts(root):
    source = Path(__file__).resolve().parent.parent
    target = root / "source-snapshot"
    target.mkdir()
    hashes = {}
    for path in sorted(source.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return target, hashes


def static_gpu_overlay(before):
    """Resolve the actual minor number and matching host driver, without a Router."""
    version_text = Path("/proc/driver/nvidia/version").read_text()
    match = re.search(r"Kernel Module\s+([0-9][0-9.]*)", version_text)
    if not match:
        raise RuntimeError("Could not identify the loaded NVIDIA kernel driver.")
    version = match.group(1)
    directories = ("/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/lib")
    volumes = []
    libraries = (
        ("libcuda", ("libcuda.so.1", "libcuda.so"), True),
        ("libnvidia-ml", ("libnvidia-ml.so.1",), True),
        ("libnvidia-ptxjitcompiler", ("libnvidia-ptxjitcompiler.so.1",), False),
    )
    for name, targets, required in libraries:
        source = next(
            (
                Path(directory) / f"{name}.so.{version}"
                for directory in directories
                if (Path(directory) / f"{name}.so.{version}").is_file()
            ),
            None,
        )
        if source is None and required:
            raise FileNotFoundError(f"No {name}.so.{version} matching the loaded driver.")
        if source is not None:
            volumes.extend(f"{source}:/usr/lib64/{target}:ro" for target in targets)
    if Path("/usr/bin/nvidia-smi").is_file():
        volumes.append("/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro")
    devices = [f"/dev/nvidia{before['minor']}"]
    devices.extend(
        path for path in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools") if Path(path).exists()
    )
    if not all(Path(path).exists() for path in devices):
        raise FileNotFoundError(f"Missing NVIDIA device among {devices!r}.")
    return dict(
        services=dict(
            main=dict(
                devices=[f"{path}:{path}" for path in devices],
                volumes=volumes,
                environment=["LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda/lib64"],
            )
        )
    )


def load_old_runner(repo):
    """Refuse mixed current/historical imports and preserve all local patches."""
    repo = Path(repo).resolve()
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    if head != BASELINE:
        raise ValueError(f"Fixed baseline requires {BASELINE}, received {head!r}.")
    for name in ("psrl", "examples"):
        module = sys.modules.get(name)
        if module is not None:
            location = getattr(module, "__file__", None)
            if location and not Path(location).resolve().is_relative_to(repo):
                raise RuntimeError(f"{name} was already imported from {location!r}; use a fresh baseline process.")
    sys.path.insert(0, str(repo))
    from examples.sciaccel_rl.config import HarborConfig, SciAccelRuntimeConfig
    from examples.sciaccel_rl.runner import run_harbor_episode

    source = Path(inspect.getfile(run_harbor_episode)).resolve()
    if not source.is_relative_to(repo):
        raise RuntimeError(f"Historical runner resolved outside the fixed worktree: {source!s}.")
    # A previously reviewed runtime patch passes override_gpus through to Harbor.
    # No admission, memory, or execution scheduling is introduced by that patch.
    if 'env_kwargs["override_gpus"]' not in source.read_text():
        raise RuntimeError("The fixed worktree needs its disclosed Harbor override_gpus runtime patch.")
    if re.search(r"^\s*await _in_cleanup_pool\(prune_dangling_images\)", source.read_text(), re.M):
        raise RuntimeError("The fixed runner must disable global image pruning on the shared node.")
    return HarborConfig, SciAccelRuntimeConfig, run_harbor_episode, source


def register_agent():
    """Harbor imports this holder in the same Python process as its driver."""
    from harbor.agents.nop import NopAgent

    class StaticResidentAgent(NopAgent):
        async def run(self, instruction, environment, context):
            entry = _EPISODES[json.loads(instruction)["episode_key"]]
            entry["ready"].set_result(environment)
            await entry["stop"].wait()

    module = sys.modules[__name__]
    module.StaticResidentAgent = StaticResidentAgent
    sys.modules["_psrl_fixed_warm_baseline"] = module
    return "_psrl_fixed_warm_baseline:StaticResidentAgent"


async def run(args, experiment=None):
    HarborConfig, RuntimeConfig, runner, runner_source = load_old_runner(args.baseline_repo)
    agent_path = register_agent()
    import yaml

    budget_s = float(getattr(args, "budget_s", 600))
    gpus = tuple(map(int, getattr(args, "gpus", (0, 1))))
    if not 0 < budget_s <= 2700 or len(gpus) != 2 or len(set(gpus)) != 2 or min(gpus) < 0:
        raise ValueError("Use two distinct GPU indices and a positive budget of at most 2700 seconds.")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    assets = Path(args.assets).resolve()
    prefix = Path(getattr(args, "python_prefix", sys.prefix)).resolve()
    container_python = getattr(args, "container_python", None)
    if container_python and not Path(container_python).is_absolute():
        raise ValueError("The Python executable inside the image must be an absolute path.")
    python = str(container_python) if container_python else str(prefix / "bin/python")
    if not (assets / "models").is_dir() or (not container_python and not Path(python).is_file()):
        raise FileNotFoundError("Local model assets and Python prefix must already exist.")
    snapshot, hashes = snapshot_scripts(root)
    for name in ("server.py", "client.py"):
        if not (snapshot / "warm_router_study" / name).is_file():
            raise FileNotFoundError(f"Missing frozen service helper {name!r}.")
    tag = "warm-fixed-" + uuid.uuid4().hex[:10]
    runtime_root = Path(tempfile.mkdtemp(prefix=tag + "-", dir="/tmp"))
    started = time.monotonic()
    deadline = started + budget_s
    baseline_repo = Path(args.baseline_repo).resolve()
    patch = subprocess.check_output(["git", "-C", str(baseline_repo), "diff", "HEAD"], timeout=10)
    (root / "baseline.patch").write_bytes(patch)
    record = dict(
        tag=tag,
        status="starting",
        started=time.time(),
        budget_s=budget_s,
        cleanup_budget_s=180,
        environment_backend="fixed_psrl_harbor_docker",
        git_commit=BASELINE,
        baseline_repo=str(baseline_repo),
        runner=str(runner_source),
        runner_sha256=hashlib.sha256(runner_source.read_bytes()).hexdigest(),
        source_hashes=hashes,
        requests=[],
        service_ready=[],
        episodes=[],
        declared_service_gpu_budget_mb=GPU_BUDGET_MB,
        progress_storage="node_local",
        container_python=python,
        python_source="image" if container_python else "host_prefix",
        image_name=args.image,
        container_python_identity=[],
        scope=(
            "Fixed historical PSRL runner and static per-GPU Compose overlays. "
            "Same local model services and command work. "
            "No EnvRouter, LLM rollout, or PSRL optimizer step."
        ),
        sources={name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "harbor")},
    )
    envs, episodes, entries, active_requests = [], [], [], set()

    def save():
        temporary = root / "status.tmp"
        temporary.write_text(json.dumps(record, indent=2, default=str) + "\n")
        temporary.replace(root / "status.json")

    async def request(index, payload, timeout_s=None):
        task = asyncio.current_task()
        active_requests.add(task)
        began = time.monotonic()
        ordinal = len(record["requests"])
        entry = dict(index=index, payload=payload, submitted_at=time.time(), status="running")
        record["requests"].append(entry)
        save()
        try:
            timeout = min(240 if timeout_s is None else timeout_s, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("The experiment budget expired before submitting a request.")
            command = shlex.join(
                [
                    python,
                    "/study/client.py",
                    "--socket",
                    "/run/service.sock",
                    "--request",
                    json.dumps(payload, allow_nan=False),
                ]
            )
            result = await envs[index].exec(command, timeout_sec=timeout)
            (root / f"request-{ordinal:03d}.stdout").write_text(result.stdout or "")
            (root / f"request-{ordinal:03d}.stderr").write_text(result.stderr or "")
            entry["return_code"] = result.return_code
            if result.return_code:
                raise RuntimeError(f"Historical service request failed: {result.stderr!r}.")
            response = json.loads((result.stdout or "").strip())
            entry.update(status="passed", result=response)
            if response.get("status") in {"failed", "error"}:
                raise RuntimeError(f"Model service rejected a request: {response!r}.")
            return response
        except asyncio.CancelledError as exc:
            entry.update(status="cancel_requested", error=repr(exc))
            raise
        except BaseException as exc:
            entry.update(status="failed", error=repr(exc))
            raise
        finally:
            active_requests.discard(task)
            entry["wall_s"] = time.monotonic() - began
            save()

    save()
    try:
        async with asyncio.timeout_at(deadline):
            record["docker_version"] = docker("info", "--format", "{{.ServerVersion}}")
            record["image_id"] = docker("image", "inspect", args.image, "--format", "{{.Id}}")
            record["gpu_before"] = {index: gpu_snapshot(index) for index in gpus}
            for gpu, before in record["gpu_before"].items():
                if before["pids"] or before["used_mb"] > 128 or before["total_mb"] < GPU_BUDGET_MB:
                    raise RuntimeError(f"GPU {gpu} is occupied or too small: {before!r}.")
            for index, gpu in enumerate(gpus):
                key = f"{tag}-{index}"
                entry = dict(ready=asyncio.get_running_loop().create_future(), stop=asyncio.Event())
                entries.append(entry)
                _EPISODES[key] = entry
                service_root = runtime_root / f"service-{index}"
                service_root.mkdir()
                task_root = root / f"task-{index}"
                (task_root / "environment").mkdir(parents=True)
                (task_root / "tests").mkdir()
                (task_root / "instruction.md").write_text(json.dumps(dict(episode_key=key)))
                (task_root / "task.toml").write_text(
                    'version="1.0"\n[environment]\n'
                    + f"docker_image={json.dumps(record['image_id'])}\n"
                    + f'cpus={CPUS}\nmemory_mb={HOST_MEMORY_MB}\ngpus=1\nnetwork_mode="no-network"\n'
                    + f"[agent]\ntimeout_sec={budget_s}\n[verifier]\ntimeout_sec=10\n"
                )
                verifier = task_root / "tests/test.sh"
                verifier.write_text(
                    "#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\n"
                    "printf '{\"reward\":1.0}\\n' > /logs/verifier/reward.json\n"
                )
                verifier.chmod(0o755)
                gpu_uuid = record["gpu_before"][gpu]["uuid"]
                mounts = [
                    (snapshot / "warm_router_study", "/study", True),
                    (snapshot / "joint_gpu_study", "/joint-study", True),
                    (snapshot, "/study-sources", True),
                    (assets, "/assets", True),
                    (service_root, "/run", False),
                ]
                if not container_python:
                    mounts.append((prefix, str(prefix), True))
                environment = dict(
                    PATH=f"{Path(python).parent}:/usr/local/bin:/usr/bin:/bin",
                    PYTHONPATH="/study:/joint-study:/opt/python-extra" if container_python else "/study:/joint-study",
                    CUDA_VISIBLE_DEVICES=gpu_uuid,
                    EXPECTED_GPU_UUID=gpu_uuid,
                    HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    TOKENIZERS_PARALLELISM="false",
                    OMP_NUM_THREADS="2",
                )
                compose = dict(
                    services=dict(
                        main=dict(
                            image=record["image_id"],
                            volumes=[
                                f"{source}:{target}:{'ro' if readonly else 'rw'}"
                                for source, target, readonly in mounts
                            ],
                            environment=environment,
                            cpus=CPUS,
                            mem_limit=f"{HOST_MEMORY_MB}m",
                            ipc="private",
                            shm_size="64m",
                        )
                    )
                )
                (task_root / "environment/docker-compose.yaml").write_text(yaml.safe_dump(compose))
                overlay = task_root / "gpu-overlay.yaml"
                overlay.write_text(yaml.safe_dump(static_gpu_overlay(record["gpu_before"][gpu])))
                runtime = RuntimeConfig(
                    harbor=HarborConfig(
                        jobs_dir=str(root / "jobs"),
                        agent_name=agent_path,
                        override_gpus=0,
                        gpu_compose_override=str(overlay),
                        max_concurrent_episodes=2,
                    ),
                    task_timeout_sec=budget_s,
                    verifier_timeout_sec=10,
                )
                episode = asyncio.create_task(
                    runner(
                        task_path=str(task_root),
                        model_base_url="http://127.0.0.1:1/v1",
                        model_name="unused",
                        config=runtime,
                        needs_gpu=True,
                        session_id=key,
                        actor_id=tag,
                        regrade_unverified=False,
                    )
                )
                episodes.append(episode)
                done, _ = await asyncio.wait(
                    (entry["ready"], episode),
                    timeout=min(90, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if episode in done:
                    raise RuntimeError(f"Historical episode ended before its service was ready: {episode.result()!r}.")
                if entry["ready"] not in done:
                    raise TimeoutError("Historical episode startup exceeded its budget.")
                environment = entry["ready"].result()
                envs.append(environment)
                cuda = shlex.join(
                    [
                        python,
                        "-c",
                        "import json,sys,torch; x=torch.ones(8,device='cuda'); "
                        "assert x.sum().item()==8, 'CUDA probe failed.'; "
                        "print(json.dumps(dict(python=sys.version, executable=sys.executable, "
                        "torch=torch.__version__, cuda=torch.version.cuda)))",
                    ]
                )
                result = await environment.exec(cuda, timeout_sec=60)
                if result.return_code:
                    raise RuntimeError(f"Historical container CUDA preflight failed: {result.stderr!r}.")
                record["container_python_identity"].append(json.loads(result.stdout.strip()))
                server = shlex.join(
                    [
                        python,
                        "-u",
                        "/study/server.py",
                        "--socket",
                        "/run/service.sock",
                        "--budget-mb",
                        str(GPU_BUDGET_MB),
                    ]
                )
                result = await environment.exec(
                    f"nohup {server} >/run/server.log 2>&1 </dev/null & echo $! >/run/server.pid", timeout_sec=10
                )
                if result.return_code:
                    raise RuntimeError(f"Historical service launch failed: {result.stderr!r}.")
                ready_deadline = min(deadline, time.monotonic() + 90)
                while not (service_root / "service.sock").is_socket():
                    if time.monotonic() >= ready_deadline:
                        raise TimeoutError("Historical service did not create its socket.")
                    await asyncio.sleep(0.25)
                record["service_ready"].append(await request(index, dict(op="ping"), timeout_s=10))
                save()
            record.update(status="running", setup_s=time.monotonic() - started)
            save()
            context = dict(
                envs=envs,
                leases=[],
                request=request,
                record=record,
                save=save,
                root=root,
                router=None,
                python=python,
                sources=record["sources"],
                progress_file=lambda index: runtime_root / f"service-{index}" / "progress.json",
                remaining=lambda: max(0.0, deadline - time.monotonic()),
            )
            if experiment is not None:
                await experiment(context)
            record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        cleanup_started = time.monotonic()
        cleanup_deadline = cleanup_started + record["cleanup_budget_s"]
        work_error = record.get("error")
        errors = []
        phases = record["cleanup_phases"] = dict(
            cancel_requested=False,
            requests_joined=False,
            episodes_finished=False,
            owned_containers_removed=False,
            physical_release_confirmed=False,
            artifacts_archived=False,
        )
        outstanding = [task for task in active_requests if not task.done() and task != asyncio.current_task()]
        phases["cancel_requested"] = bool(outstanding)
        try:
            joined = await cancel_and_join(outstanding, deadline=cleanup_deadline)
            record["client_cancellation"] = joined
            phases["requests_joined"] = joined["joined"]
            if not joined["joined"]:
                errors.append(f"Client cancellation left {joined['pending']} requests unfinished.")
        except BaseException as exc:
            errors.append(f"Client cancellation failed: {exc!r}.")
        for entry in entries:
            entry["stop"].set()
        if episodes:
            try:
                results = await await_cleanup(asyncio.gather(*episodes), cleanup_deadline, timeout_s=80)
                record["episodes"] = [dataclasses.asdict(result) for result in results]
                if any(result.exception is not None or result.reward != 1 for result in results):
                    errors.append("A historical runner episode failed its completion check.")
                else:
                    phases["episodes_finished"] = True
            except BaseException as exc:
                errors.append(f"Historical episode completion failed: {exc!r}.")
                try:
                    joined = await cancel_and_join(episodes, deadline=cleanup_deadline)
                    record["episode_cancellation"] = joined
                    if not joined["joined"]:
                        errors.append(f"Episode cancellation left {joined['pending']} tasks unfinished.")
                except BaseException as cancel_error:
                    errors.append(f"Historical episode cancellation failed: {cancel_error!r}.")
        else:
            phases["episodes_finished"] = True
        for key in list(_EPISODES):
            if key.startswith(tag):
                del _EPISODES[key]
        try:
            owned = (
                await await_cleanup(
                    asyncio.to_thread(docker, "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"),
                    cleanup_deadline,
                    timeout_s=15,
                )
            ).split()
            if owned:
                await await_cleanup(asyncio.to_thread(docker, "rm", "-f", *owned), cleanup_deadline, timeout_s=15)
            record["containers_remaining"] = await await_cleanup(
                asyncio.to_thread(docker, "ps", "-aq", "--filter", f"label=psrl.actor_id={tag}"),
                cleanup_deadline,
                timeout_s=15,
            )
            phases["owned_containers_removed"] = not bool(record["containers_remaining"])
            for _ in range(10):
                record["gpu_after"] = await await_cleanup(
                    asyncio.to_thread(lambda: {index: gpu_snapshot(index) for index in gpus}),
                    cleanup_deadline,
                    timeout_s=10,
                )
                if all(not item["pids"] and item["used_mb"] < 128 for item in record["gpu_after"].values()):
                    break
                await await_cleanup(asyncio.sleep(1), cleanup_deadline)
            if record["containers_remaining"] or any(
                item["pids"] or item["used_mb"] >= 128 for item in record["gpu_after"].values()
            ):
                errors.append("Historical physical resource cleanup has not been confirmed.")
            else:
                phases["physical_release_confirmed"] = True
        except BaseException as exc:
            errors.append(f"Historical physical release check failed: {exc!r}.")
        try:
            await await_cleanup(asyncio.to_thread(archive_runtime, runtime_root, root), cleanup_deadline, timeout_s=15)
            await await_cleanup(asyncio.to_thread(shutil.rmtree, runtime_root), cleanup_deadline, timeout_s=5)
            phases["artifacts_archived"] = True
        except BaseException as exc:
            record["runtime_artifacts_retained"] = True
            errors.append(f"Runtime evidence archival failed: {exc!r}.")
        record.update(
            cleanup_errors=errors,
            cleanup_elapsed_s=time.monotonic() - cleanup_started,
            elapsed_s=time.monotonic() - started,
            finished=time.time(),
        )
        if errors:
            record["status"] = "failed"
        try:
            save()
        except BaseException as exc:
            record["status"] = "failed"
            errors.append(f"Final status archival failed: {exc!r}.")
        if errors and work_error is None:
            raise RuntimeError(f"Historical cleanup or completion failed: {errors!r}.")
    return record


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python-prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--container-python", type=Path)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--budget-s", type=float, default=600)
    return parser


if __name__ == "__main__":
    asyncio.run(run(argument_parser().parse_args()))
