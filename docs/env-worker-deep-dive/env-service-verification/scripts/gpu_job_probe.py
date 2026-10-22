"""Check GPU jobs with real Harbor containers and a local production scheduler adapter."""

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from examples.sciaccel_rl.gpu_grading import apply_gpu_activity_gate
from examples.sciaccel_rl.gpu_tools import dispatch_gpu_tool
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from psrl.workers.env_service.command_gpu import GpuCommandScheduler
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.ledger import EnvUnitLedger
from psrl.workers.env_service.node import EnvNodeService
from psrl.workers.env_service.node_policy import NodeRules
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.task_telemetry import activity_verified
from psrl.workers.env_service.types import EnvDemand, EnvUnit


class Remote:
    """Exercise production methods without starting a training or Ray cluster."""

    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        return SimpleNamespace(remote=getattr(self.target, name))


async def probe(args, evidence):
    node = EnvNodeService(None, "local")
    try:
        await probe_node(args, evidence, node)
    finally:
        await node.close()
        evidence["node_service_closed"] = True


async def probe_node(args, evidence, node):
    unit = EnvUnit("job-probe", "local", 4, 4096, (args.gpu,), 2, 4096, gpu_reservation="command")
    ledger = EnvUnitLedger([unit], NodeRules.build(policy="any"), lease_ttl_s=600)
    evidence["stage"] = "preflight"
    evidence["workload_started"] = False
    await node.prepare(unit.gpu_indices)
    await node.check_gpu_command_support()
    if not args.allow_unavailable_attribution:
        await node.check_gpu_attribution_support(unit.gpu_indices)
    evidence["attribution_preflight"] = "diagnostic_bypass" if args.allow_unavailable_attribution else "passed"
    if args.preflight_only:
        return
    before = await node.gpu_lending_snapshot(unit.gpu_indices)
    assert before[args.gpu]["used_mb"] < 64 and not before[args.gpu]["pids"], "Select an idle GPU."
    evidence["before"] = before

    async def get_node(_ip):
        return Remote(node)

    scheduler = GpuCommandScheduler(ledger, get_node)
    router = SimpleNamespace(
        **{
            name: SimpleNamespace(remote=getattr(scheduler, target))
            for name, target in {
                "acquire_gpu_command": "acquire",
                "release_gpu_command": "release",
                "begin_gpu_command_return": "begin_return",
                "cancel_gpu_command": "cancel",
                "gpu_job_telemetry": "telemetry",
            }.items()
        }
    )
    environments, leases, paths = [], [], []
    try:
        evidence["stage"] = "workload"
        evidence["workload_started"] = True
        await ManagedDockerEnvironment.prepare_runtime()
        for name in ("a", "b"):
            lease = ledger.try_admit(name, EnvDemand(2, 2048, True, 2048), "local").lease
            leases.append(lease)
            overlay = build_lease_overlay(lease, resolve_driver_libs(read_kernel_driver_version()))
            path = write_overlay(overlay)
            paths.append(path)
            directory = args.output / name
            (directory / "environment").mkdir(parents=True)
            environment = ManagedDockerEnvironment(
                environment_dir=directory / "environment",
                environment_name="gpu-job-probe",
                session_id="gpu-job-" + uuid.uuid4().hex[:12],
                trial_paths=TrialPaths(directory / "trial"),
                task_env_config=EnvironmentConfig(docker_image=args.image, cpus=2, memory_mb=2048),
                extra_docker_compose=[path],
                lease_overlay=overlay,
                gpu_commands={"router_name": "probe", "lease_id": lease.lease_id},
                network_policy=NetworkPolicy(network_mode="no-network"),
            )
            environments.append(environment)
            await environment.start(force_build=False)
            await environment.upload_file(Path(__file__).with_name("cuda_job_workload.py"), "/tmp/workload.py")
        a, b = environments
        command = "python3 /tmp/workload.py --seconds 6"
        with patch("ray.get_actor", return_value=router):
            denied = await a.exec(command, timeout_sec=15)
            assert denied.return_code != 0, "Ordinary terminal CUDA must not bypass GPU admission."
            evidence["outside_grant_denied"] = True
            evidence["latency"] = []
            for mode in () if args.skip_latency else ("serial", "overlap", "overlap", "serial"):
                started = time.monotonic()
                if mode == "serial":
                    result = await a.exec_gpu(command, timeout_sec=30)
                    assert result.return_code == 0, result
                    await asyncio.sleep(6)
                    state = {"state": "completed"}
                else:
                    state = await dispatch_gpu_tool(a, "psrl-gpu " + json.dumps({"op": "start", "command": command}))
                    await asyncio.sleep(6)
                    state = await a.wait_gpu_job(state["job_id"], timeout_sec=30, released=True)
                    assert state["state"] == "completed" and state["exit_code"] == 0, state
                evidence["latency"].append({"mode": mode, "seconds": time.monotonic() - started, "job": state})

            busy = b.start_gpu_job("python3 /tmp/workload.py --seconds 10", tag="busy")
            await asyncio.sleep(2)
            sampler = node._gpu_lease_state(leases[1].lease_id)["command_telemetry"]
            evidence["attribution_debug"] = {
                "cgroup": str(sampler.fence.path),
                "pids": list(sampler.fence.processes()),
                "local_proc_namespace": str(Path("/proc/1/ns/pid").readlink()),
                "raw": node._gpu_probe().read_process_snapshot((args.gpu,), 0),
            }
            observer = a.start_gpu_job(
                "sleep 4; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
                tag="cpu-only",
            )
            observation = await a.wait_gpu_job(observer["job_id"], released=True)
            busy_result = await b.wait_gpu_job(busy["job_id"], released=True)
            assert observation["exit_code"] == 0 and busy_result["exit_code"] == 0
            reports = scheduler.reports(leases[0].lease_id) + scheduler.reports(leases[1].lease_id)
            cpu_report = next(r for r in reports if r["tag"] == "cpu-only")
            busy_report = next(r for r in reports if r["tag"] == "busy")
            evidence["attribution"] = {
                "global_utilization_seen_by_cpu_task": observation["stdout"],
                "cpu_report": cpu_report,
                "busy_report": busy_report,
                "busy_activity_verified": activity_verified(busy_report["telemetry"]),
            }
            assert not activity_verified(cpu_report["telemetry"]), "Neighbor activity must not activate CPU rewards."
            if args.allow_unavailable_attribution:
                assert busy_report["telemetry"]["errors"], "Missing host procfs must be explicit."
                try:
                    apply_gpu_activity_gate({"reward": 1, "reward_gpu": 1}, reports, ["cpu-only"])
                except RuntimeError as exc:
                    evidence["attribution"]["grading_rejected"] = str(exc)
                else:
                    raise AssertionError("Unknown attribution must reject grading.")
            else:
                assert activity_verified(busy_report["telemetry"]), "The actual CUDA owner needs attributed activity."
                evidence["attribution"]["cpu_rewards"] = apply_gpu_activity_gate(
                    {"reward": 1, "reward_gpu": 1}, reports, ["cpu-only"]
                )
            evidence["task_view"] = await b.gpu_job_telemetry(busy["job_id"])

            cancelled = a.start_gpu_job("python3 /tmp/workload.py --seconds 30")
            await asyncio.sleep(2)
            evidence["cancelled"] = await a.cancel_gpu_job(cancelled["job_id"])
            assert evidence["cancelled"]["state"] == "cancelled", evidence["cancelled"]
            assert ledger.snapshot()[0].declared_gpu_memory_mb == 0, "Cancelled work must return GPU capacity."
            followup = a.start_gpu_job("python3 /tmp/workload.py --seconds 1")
            resumed = await a.wait_gpu_job(followup["job_id"], released=True)
            assert resumed["exit_code"] == 0 and resumed["state"] == "completed", resumed
            evidence["reacquisition"] = resumed

            evidence["passed"] = True
    finally:
        with patch("ray.get_actor", return_value=router):
            evidence["cleanup_errors"] = []
            for environment in environments:
                try:
                    await environment.stop(delete=True)
                except Exception as exc:
                    evidence["cleanup_errors"].append(str(exc))
            for lease in leases:
                await scheduler.finish(lease.lease_id, lease.unit_id)
                ledger.release(lease.lease_id)
        for path in paths:
            path.unlink(missing_ok=True)
        evidence["after"] = await node.gpu_lending_snapshot(unit.gpu_indices)
        assert not evidence["after"][args.gpu]["pids"], "Probe CUDA processes must be gone."
        assert evidence["after"][args.gpu]["used_mb"] < 64, "Probe GPU memory must be physically returned."


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    scope = (
        "Production node deployment preflight. No task containers or CUDA workload."
        if args.preflight_only
        else "Real Docker and CUDA with local production scheduler RPC adapter. No model or training."
    )
    evidence = {"scope": scope}
    try:
        async with asyncio.timeout(300):
            await probe(args, evidence)
    except BaseException as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--allow-unavailable-attribution", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and not args.image:
        parser.error("--image is required unless --preflight-only is selected")
    if args.preflight_only and args.allow_unavailable_attribution:
        parser.error("--preflight-only cannot bypass attribution checks")
    asyncio.run(main(args))
