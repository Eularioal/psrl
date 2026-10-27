"""
Exercise production control logic with teaching inputs without Ray, Docker or GPUs.
"""

import ast
import asyncio
import hashlib
import importlib.util
import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from psrl.workers.env_service import Admission, EnvDemand, EnvUnit, EnvUnitLedger, NodeRules  # noqa: E402
from psrl.workers.env_service.health import NodeHealth  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "routing_pause_audit", REPO / "psrl/workers/gen/rollout_coordination/routing_pause.py"
)
pause_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pause_module)
RoutingPauseController = pause_module.RoutingPauseController


def migration_method():
    """
    Compile the actual method while replacing external commands with local fault injection.
    """
    path = REPO / "psrl/workers/gen/rollout_coordination/sync_and_migrate/sync_and_migrate_mixin.py"
    source = path.read_text()
    method = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "check_and_migrate"
    )
    namespace = {
        "logging": logging,
        "psrl_logger": logging.getLogger("env_design_audit"),
        "log_dual_events": lambda *args, **kwargs: nullcontext(),
        "EventType": SimpleNamespace(OTHER="other"),
        "Command": SimpleNamespace,
        "CommandType": SimpleNamespace(ABORT="abort"),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["check_and_migrate"], hashlib.sha256(source.encode()).hexdigest()


class MigrationHarness:
    """
    Model routing state and inject failure at the two awaited migration boundaries.
    """

    def __init__(self, fault):
        self.fault = fault
        self.running = True
        self.transitions = []
        self._pauses = RoutingPauseController(self._routing_status, self._set_routing_loop_running)
        self.config = SimpleNamespace(
            psrl=SimpleNamespace(
                rollout_coordination=SimpleNamespace(
                    sync_and_mig_strategy=SimpleNamespace(mig=SimpleNamespace(enable=True))
                ),
                status_collection=SimpleNamespace(enable=True),
            )
        )

    async def _check_should_migrate(self):
        return ["teaching-instance"]

    async def _routing_status(self):
        return {"paused": not self.running}

    def _routing_pause_controller(self):
        return self._pauses

    async def _set_routing_loop_running(self, running):
        self.running = running
        self.transitions.append(running)

    async def exec_command(self, command, blocking):
        if self.fault == "abort":
            raise RuntimeError("Injected abort failure.")

    async def _wait_interrupted_partial_requests_loop_back(self, instances):
        if self.fault == "drain":
            raise TimeoutError("Injected drain timeout.")


async def migration_cases():
    method, fingerprint = migration_method()
    cases = {}
    for fault in ("none", "abort", "drain"):
        harness = MigrationHarness(fault)
        error = None
        try:
            await method(harness)
        except (RuntimeError, TimeoutError) as exc:
            error = type(exc).__name__
        cases[fault] = {
            "error": error,
            "routing_running": harness.running,
            "transitions": harness.transitions,
        }
    return {"source_sha256": fingerprint, "cases": cases}


def expiry_case(mode):
    """
    Hold physical cleanup unacknowledged while the real ledger processes expiration.
    """
    clock = [0.0]
    gpu = mode != "cpu"
    unit = EnvUnit(
        "unit",
        "192.0.2.10",
        4,
        8192,
        (0,) if gpu else (),
        1,
        16384 if gpu else 0,
        gpu_reservation="command" if mode == "command" else "episode",
    )
    ledger = EnvUnitLedger([unit], NodeRules.build("any"), lease_ttl_s=10, clock=lambda: clock[0])
    demand = EnvDemand(cpu=4, memory_mb=8192, needs_gpu=gpu, gpu_memory_mb=8192 if gpu else 0)
    first = ledger.try_admit("old", demand, unit.node_ip)
    assert first.status == "granted", "The initial teaching request must fit."
    clock[0] = 11.0
    expired = ledger.reap()
    second = ledger.try_admit("new", demand, unit.node_ip)
    return {
        "initial": first.status,
        "expired_leases": len(expired),
        "cleanup_acknowledged": False,
        "cleanup_queue_size": len(ledger._reaped.get(unit.node_ip, [])),
        "replacement_admission": second.status,
        "unit_blocked": unit.unit_id in ledger._blocked_units,
        "scope": "Cleanup is deliberately not executed. No actual container or memory allocation exists.",
    }


def multi_gpu_case():
    """
    Show that an aggregate request carries no constraint describing its single device peak.
    """
    unit = EnvUnit(
        "two-cards",
        "192.0.2.10",
        4,
        8192,
        (0, 1),
        1,
        80 * 1024,
        gpu_memory_per_device_mb=(40 * 1024, 40 * 1024),
    )
    ledger = EnvUnitLedger([unit], NodeRules.build("any"))
    result = ledger.try_admit(
        "single-device-program", EnvDemand(cpu=1, needs_gpu=True, gpu_memory_mb=60 * 1024), unit.node_ip
    )
    return {
        "teaching_device_capacities_gib": [40, 40],
        "declared_total_gib": 60,
        "intended_single_device_peak_gib": 60,
        "admission": result.status,
        "assigned_devices": result.lease.gpu_indices if result.lease else [],
        "scope": "Explicit per-device budgets. No real device capacity or OOM is tested.",
    }


async def router_expiry_cases():
    """
    Exercise the production admission wrapper around ledger expiry and cleanup acknowledgement.
    """
    path = REPO / "psrl/workers/env_service/router.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "RayEnvRouter")
    method = next(node for node in cls.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "try_admit")
    namespace = {"asyncio": asyncio, "Admission": Admission, "replace": replace}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    cases = {}
    for mode in ("pending", "confirmed", "new_reap_after_cleanup"):
        clock = [0.0]
        unit = EnvUnit("cpu", "192.0.2.10", 4, 8192, max_containers=1)
        ledger = EnvUnitLedger([unit], NodeRules.build("any"), lease_ttl_s=10, clock=lambda clock=clock: clock[0])
        demand = EnvDemand(cpu=4, memory_mb=8192)
        ledger.try_admit("old", demand, unit.node_ip)
        clock[0] = 11.0

        async def cleanup(mode=mode, ledger=ledger, node_ip=unit.node_ip):
            if mode != "new_reap_after_cleanup":
                if mode == "confirmed":
                    for lease in ledger.pending_reaped(node_ip):
                        ledger.acknowledge_reaped(node_ip, lease.lease_id)
            return 1 if mode == "pending" else 0

        async def node_service(cleanup=cleanup):
            return SimpleNamespace(cleanup=SimpleNamespace(remote=cleanup)), "teaching-node-url"

        state = NodeHealth()

        async def check(cleanup=cleanup, state=state):
            state.pending_cleanup = bool(await cleanup())

        def health_task(ip, force=False, check=check):
            return asyncio.create_task(check())

        harness = SimpleNamespace(
            _ledger=ledger,
            _lending_unavailable=lambda *args: False,
            _ensure_reaper=lambda: None,
            _node_task=lambda ip, node_service=node_service: node_service(),
            _node_health={unit.node_ip: state},
            _health_task=health_task,
            _health_max_failures=3,
        )
        admission = await namespace["try_admit"](harness, "new", demand, unit.node_ip)
        cases[mode] = {"admission": admission.status, "reason": admission.reason}
    return cases


async def main():
    result = {
        "scope": "CPU mechanism audit using real source and teaching clocks, demands and injected exceptions.",
        "migration": await migration_cases(),
        "expired_lease_before_cleanup": {mode: expiry_case(mode) for mode in ("cpu", "episode", "command")},
        "production_router_cleanup_guard": await router_expiry_cases(),
        "multi_gpu": multi_gpu_case(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
