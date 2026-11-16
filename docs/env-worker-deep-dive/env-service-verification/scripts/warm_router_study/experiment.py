"""Host two persistent model services under real PSRL episode GPU reservations."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from examples.gpu_benchmarks.run import gpu_snapshot
from runtime_guard import archive_runtime, await_cleanup, cancel_and_join

GPU_BUDGET_MB = 20480
HOST_MEMORY_MB = 49152
CPUS = 8


def docker(*arguments):
    """Bound each Docker control operation independently."""
    return subprocess.check_output(["docker", *arguments], text=True, timeout=15).strip()


def origins():
    """Identify dependency roots in both the driver and its Ray worker."""
    return {name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "harbor")}


def snapshot_scripts(root):
    """Freeze helper sources so ongoing edits cannot change an active comparison."""
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


async def run(args, experiment=None):
    """
    Invoke a bounded experiment after two native environments are ready.

    Args:
        args (argparse.Namespace): Paths, node identity, two GPU indices and a time budget.
        experiment (Callable | None): Async callback receiving the shared context dictionary.

    Returns:
        dict: Evidence including resource ownership, requests and physical cleanup.
    """
    import ray
    from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths
    from psrl.workers.env_service.client import EnvServiceClient
    from psrl.workers.env_service.container_budget import ContainerBudget
    from psrl.workers.env_service.overlay import build_lease_overlay, read_kernel_driver_version, resolve_driver_libs
    from psrl.workers.env_service.router import EnvRouter
    from psrl.workers.env_service.types import EnvDemand
    from psrl.workers.env_worker.harbor import EnvWorkerEnvironment

    budget_s = float(getattr(args, "budget_s", 600))
    gpus = tuple(map(int, getattr(args, "gpus", (0, 1))))
    if len(gpus) != 2 or len(set(gpus)) != 2 or min(gpus) < 0:
        raise ValueError("Exactly two distinct nonnegative GPU indices are required.")
    if not 0 < budget_s <= 2700:
        raise ValueError("The experiment budget must be positive and no greater than 2700 seconds.")
    if ray.is_initialized():
        raise RuntimeError("Run the harness in a fresh process so it owns its private Ray cluster.")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    assets = Path(args.assets).resolve()
    prefix = Path(getattr(args, "python_prefix", sys.prefix)).resolve()
    container_python = getattr(args, "container_python", None)
    if container_python and not Path(container_python).is_absolute():
        raise ValueError("The container Python executable must have an absolute path.")
    python = str(container_python or prefix / "bin/python")
    if not (assets / "models").is_dir() or (not container_python and not (prefix / "bin/python").is_file()):
        raise FileNotFoundError("Existing local model assets and the selected Python prefix are required.")
    snapshot, hashes = snapshot_scripts(root)
    for name in ("server.py", "client.py"):
        if not (snapshot / "warm_router_study" / name).is_file():
            raise FileNotFoundError(f"Required frozen helper {name!r} is missing.")

    tag = "warm-router-" + uuid.uuid4().hex[:10]
    runtime_root = Path(tempfile.mkdtemp(prefix=tag + "-", dir="/tmp"))
    started = time.monotonic()
    deadline = started + budget_s
    record = dict(
        tag=tag,
        status="starting",
        started=time.time(),
        budget_s=budget_s,
        cleanup_budget_s=180,
        environment_backend="env_worker",
        gpu_reservation="episode",
        per_gpu_reserved_mb=GPU_BUDGET_MB,
        progress_storage="node_local",
        container_python=python,
        python_source="image" if container_python else "host_prefix",
        image_name=args.image,
        host_python_prefix_mounted=not bool(container_python),
        scope="Real Ray EnvRouter and native EnvWorker model services. No LLM agent rollout or PSRL training step.",
        sources=origins(),
        source_hashes=hashes,
        service_ready=[],
        requests=[],
    )
    router = None
    envs, handles, active_requests, owned_gpus = [], [], set(), []

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
            remaining = deadline - time.monotonic()
            timeout = min(240 if timeout_s is None else timeout_s, remaining)
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
                raise RuntimeError(f"Service request failed for environment {index!r}: {result.stderr!r}.")
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
            for index, before in record["gpu_before"].items():
                if before["pids"] or before["used_mb"] > 128 or before["total_mb"] - 1024 < GPU_BUDGET_MB:
                    raise RuntimeError(f"GPU {index!r} is not idle or lacks the declared capacity: {before!r}.")
            record["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=10).strip()
            record["git_status"] = subprocess.check_output(["git", "status", "--short"], text=True, timeout=10).strip()
            ray.init(
                address="local",
                namespace=tag,
                _node_ip_address=args.node_ip,
                _temp_dir=f"/tmp/{tag}",
                num_cpus=4,
                num_gpus=0,
                object_store_memory=134217728,
                include_dashboard=False,
                log_to_driver=False,
            )
            record["ray_nodes"] = ray.nodes()
            alive = [node for node in record["ray_nodes"] if node["Alive"]]
            if len(alive) != 1 or alive[0]["NodeManagerAddress"] != args.node_ip:
                raise RuntimeError("Unexpected private Ray node identity.")

            @ray.remote
            def worker_origins():
                import importlib.util

                return {
                    name: importlib.util.find_spec(name).origin
                    for name in ("psrl", "verl", "transfer_queue", "harbor")
                }

            record["ray_worker_sources"] = await worker_origins.remote()
            if record["ray_worker_sources"] != record["sources"]:
                raise RuntimeError("Driver and Ray worker dependency roots differ.")
            units = [
                dict(
                    unit_id=f"{tag}-{index}",
                    node_ip=args.node_ip,
                    cpu_quota=CPUS,
                    memory_mb=HOST_MEMORY_MB,
                    gpu_indices=[gpu],
                    gpu_memory_mb=int(record["gpu_before"][gpu]["total_mb"]) - 1024,
                    max_containers=1,
                    gpu_reservation="episode",
                )
                for index, gpu in enumerate(gpus)
            ]
            record["units"] = units
            router = EnvRouter.options(name=tag).remote(dict(node_policy="any", units=units, slot_ttl_s=900))
            await asyncio.wait_for(router.ping.remote(), 45)
            client = EnvServiceClient(router, args.node_ip, router_name=tag, heartbeat_interval_s=5)
            driver_libs = resolve_driver_libs(read_kernel_driver_version())
            container_budget = ContainerBudget(CPUS, HOST_MEMORY_MB, 0)
            for index, gpu in enumerate(gpus):
                handle = await client.acquire(
                    f"{tag}-{index}",
                    EnvDemand(CPUS, HOST_MEMORY_MB, True, GPU_BUDGET_MB, container_budget=container_budget),
                    max_wait_s=30,
                )
                handles.append(handle)
                if tuple(handle.lease.gpu_indices) != (gpu,):
                    raise RuntimeError("Router did not assign the expected independent GPU.")
                service_root = runtime_root / f"service-{index}"
                service_root.mkdir()
                environment_root = root / f"environment-{index}"
                environment_root.mkdir()
                mounts = [
                    (snapshot / "warm_router_study", "/study", True),
                    (snapshot / "joint_gpu_study", "/joint-study", True),
                    (snapshot, "/study-sources", True),
                    (assets, "/assets", True),
                    (service_root, "/run", False),
                ]
                if not container_python:
                    mounts.append((prefix, str(prefix), True))
                overlay = build_lease_overlay(handle.lease, driver_libs)
                gpu_uuid = record["gpu_before"][gpu]["uuid"]
                environment = EnvWorkerEnvironment(
                    environment_dir=environment_root,
                    environment_name=f"{tag}-{index}",
                    session_id=f"{tag}-{index}",
                    actor_id=tag,
                    trial_paths=TrialPaths(root / f"trial-{index}"),
                    task_env_config=EnvironmentConfig(
                        docker_image=record["image_id"], cpus=CPUS, memory_mb=HOST_MEMORY_MB
                    ),
                    network_policy=NetworkPolicy(network_mode="no-network"),
                    mounts=[
                        dict(type="bind", source=str(source), target=target, read_only=readonly)
                        for source, target, readonly in mounts
                    ],
                    persistent_env=dict(
                        PATH=f"{Path(python).parent}:/usr/local/bin:/usr/bin:/bin",
                        PYTHONPATH="/study:/joint-study:/opt/python-extra"
                        if container_python
                        else "/study:/joint-study",
                        CUDA_VISIBLE_DEVICES=gpu_uuid,
                        EXPECTED_GPU_UUID=gpu_uuid,
                        HF_HUB_OFFLINE="1",
                        TRANSFORMERS_OFFLINE="1",
                        PYTHONDONTWRITEBYTECODE="1",
                        TOKENIZERS_PARALLELISM="false",
                        OMP_NUM_THREADS="2",
                    ),
                    container_budget=dataclasses.asdict(container_budget),
                    lease_overlay=overlay,
                    gpu_execution=dict(mode="episode", router_name=tag, lease_id=handle.lease.lease_id),
                )
                envs.append(environment)
                owned_gpus.append(gpu)
                await environment.start()
                cuda = shlex.join(
                    [
                        python,
                        "-c",
                        "import json,sys,torch; x=torch.ones(8,device='cuda'); "
                        "assert x.sum().item()==8, 'CUDA probe failed.'; "
                        "print(json.dumps(dict(python_version=sys.version,python_executable=sys.executable,"
                        "torch_version=torch.__version__,cuda_version=torch.version.cuda)))",
                    ]
                )
                preflight = await environment.exec_gpu(cuda, timeout_sec=60, wait_timeout_sec=30)
                record.setdefault("cuda_preflight", []).append(
                    dict(
                        index=index,
                        return_code=preflight.return_code,
                        stdout=preflight.stdout,
                        stderr=preflight.stderr,
                    )
                )
                if preflight.return_code:
                    raise RuntimeError(f"Container CUDA preflight failed: {preflight.stderr!r}.")
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
                launched = await environment.exec(
                    f"nohup {server} >/run/server.log 2>&1 </dev/null & echo $! >/run/server.pid", timeout_sec=10
                )
                if launched.return_code:
                    raise RuntimeError(f"Persistent model service did not launch: {launched.stderr!r}.")
                ready_deadline = min(deadline, time.monotonic() + 150)
                while not (service_root / "service.sock").is_socket():
                    if time.monotonic() >= ready_deadline:
                        raise TimeoutError(f"Service {index!r} did not create its socket within 150 seconds.")
                    await asyncio.sleep(0.25)
                record["service_ready"].append(await request(index, dict(op="ping"), timeout_s=10))
                container_id = await environment._gpu_container_id()
                info = json.loads(docker("inspect", container_id))[0]
                record.setdefault("containers", []).append(
                    dict(id=container_id, labels=info["Config"]["Labels"], host_config=info["HostConfig"])
                )
                save()

            record["leases"] = [dataclasses.asdict(handle.lease) for handle in handles]
            record["ledger_ready"] = [dataclasses.asdict(unit) for unit in await router.snapshot.remote()]
            if sum(unit["declared_gpu_memory_mb"] for unit in record["ledger_ready"]) != 2 * GPU_BUDGET_MB:
                raise RuntimeError("Persistent model services are not fully charged in the Router ledger.")
            record.update(status="running", setup_s=time.monotonic() - started)
            save()
            context = dict(
                envs=envs,
                leases=[handle.lease for handle in handles],
                request=request,
                record=record,
                save=save,
                root=root,
                router=router,
                python=python,
                sources=record["sources"],
                progress_file=lambda index: runtime_root / f"service-{index}" / "progress.json",
                remaining=lambda: max(0.0, deadline - time.monotonic()),
            )
            if experiment is not None:
                await experiment(context)
            record["ledger_before_cleanup"] = [dataclasses.asdict(unit) for unit in await router.snapshot.remote()]
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
            environments_stopped=False,
            leases_released=False,
            router_stopped=False,
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
        environments_stopped = True
        for environment in reversed(envs):
            try:
                await await_cleanup(environment.stop(delete=True), cleanup_deadline, timeout_s=35)
            except BaseException as exc:
                environments_stopped = False
                errors.append(f"Environment stop failed: {exc!r}.")
        phases["environments_stopped"] = environments_stopped
        leases_released = True
        for handle in handles:
            try:
                await await_cleanup(handle.release(), cleanup_deadline, timeout_s=10)
            except BaseException as exc:
                leases_released = False
                errors.append(f"Lease release failed: {exc!r}.")
        phases["leases_released"] = leases_released
        if router is not None:
            try:
                ledger = await await_cleanup(router.snapshot.remote(), cleanup_deadline, timeout_s=10)
                record["ledger_after_cleanup"] = [dataclasses.asdict(unit) for unit in ledger]
                if any(unit["leases_active"] for unit in record["ledger_after_cleanup"]):
                    phases["leases_released"] = False
                    errors.append("Router leases remained active after environment cleanup.")
            except BaseException as exc:
                errors.append(f"Router snapshot failed: {exc!r}.")
            try:
                await await_cleanup(router.shutdown_nodes.remote(), cleanup_deadline, timeout_s=20)
                phases["router_stopped"] = True
            except BaseException as exc:
                errors.append(f"Router node shutdown failed: {exc!r}.")
            try:
                await await_cleanup(
                    asyncio.to_thread(ray.kill, router, no_restart=True), cleanup_deadline, timeout_s=5
                )
            except BaseException as exc:
                phases["router_stopped"] = False
                errors.append(f"Owned Router actor stop failed: {exc!r}.")
        else:
            phases["router_stopped"] = True
        if ray.is_initialized():
            try:
                await await_cleanup(asyncio.to_thread(ray.shutdown), cleanup_deadline, timeout_s=20)
            except BaseException as exc:
                errors.append(f"Private Ray shutdown failed: {exc!r}.")
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
                    asyncio.to_thread(lambda: {index: gpu_snapshot(index) for index in owned_gpus}),
                    cleanup_deadline,
                    timeout_s=10,
                )
                if all(not value["pids"] and value["used_mb"] < 128 for value in record["gpu_after"].values()):
                    break
                await await_cleanup(asyncio.sleep(1), cleanup_deadline)
            if record["containers_remaining"] or any(
                value["pids"] or value["used_mb"] >= 128 for value in record["gpu_after"].values()
            ):
                errors.append("Physical resources have not been confirmed released.")
            else:
                phases["physical_release_confirmed"] = True
        except BaseException as exc:
            errors.append(f"Physical release check failed: {exc!r}.")
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
            raise RuntimeError(f"Physical cleanup could not be confirmed: {errors!r}.")
    return record


def argument_parser():
    """Expose the same options to the experiment callback entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
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
