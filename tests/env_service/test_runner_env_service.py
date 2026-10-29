"""
Tests for the env service path through `examples.sciaccel_rl.runner.run_harbor_episode`.

Covers Gate A as the router's resource-keyed admission: the overlay and `override_gpus: 0`
that realise a lease, release only after the containers are gone, release on every failure
path (a leaked lease would keep heartbeating forever), and Gate B binding for a shared card.

Harbor is replaced by light stubs only when it is not installed. Run with
`PYTHONPATH=. python3 tests/env_service/test_runner_env_service.py` or under pytest.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402
from _light_stubs import load_sciaccel_runner  # noqa: E402
from psrl.workers.env_service import EnvDemand  # noqa: E402
from psrl.workers.env_service.client import EnvServiceClient  # noqa: E402
from psrl.workers.env_service.overlay import DriverLibraryError, DriverLibs  # noqa: E402
from psrl.workers.env_service.probe import GpuDevice  # noqa: E402
from psrl.workers.env_service.router import EnvRouterCore  # noqa: E402
from test_router_client import LocalRouter  # noqa: E402

runner = load_sciaccel_runner()
from examples.sciaccel_rl.config import SciAccelRuntimeConfig  # noqa: E402

NODE = "10.0.0.1"
SESSION = "sess-1"


def _cfg(max_containers=2):
    return {
        "node_policy": "any",
        "default_gpu_memory_mb": 24000,
        "slot_ttl_s": 1800,
        "units": [
            {
                "unit_id": "g0",
                "node_ip": NODE,
                "cpu_quota": 16,
                "memory_mb": 65536,
                "gpu_indices": [0, 1],
                "max_containers": max_containers,
                "gpu_memory_mb": 160000,
                "gpu_memory_per_device_mb": [80000, 80000],
            },
            {"unit_id": "c0", "node_ip": NODE, "cpu_quota": 16, "memory_mb": 65536, "max_containers": 4},
        ],
    }


async def _no_telemetry():
    return {}


def _client(core):
    return EnvServiceClient(
        LocalRouter(core), NODE, telemetry_reader=_no_telemetry, poll_interval_s=0.01, heartbeat_interval_s=3600.0
    )


def _ok_job(on_create=None):
    trial = MagicMock()
    trial.task_name = "t"
    trial.verifier_result = MagicMock()
    trial.verifier_result.rewards = {"reward": 1.0}
    trial.exception_info = None
    result = MagicMock()
    result.trial_results = [trial]
    job = AsyncMock()
    job.run = AsyncMock(return_value=result)
    return job


_LIBS = DriverLibs(
    version="535.1",
    mounts=(("/usr/lib64/libcuda.so.535.1", "/usr/lib64/libcuda.so.1"),),
)


def _run_episode(
    core,
    *,
    needs_gpu,
    create_side_effect=None,
    driver_version="535.1",
    events=None,
    gate_b=None,
    runtime=None,
    task_path="/t",
):
    """
    Run one episode through the env service path with Harbor and Docker mocked.

    Returns `(result, captured)` where `captured` holds what Harbor was handed.
    """
    events = events if events is not None else []
    captured: dict = {}
    client = _client(core)

    async def create(job_config):
        captured["agent"] = job_config.agents[0]
        env = job_config.environment or {}
        if hasattr(env, "model_dump"):
            env = env.model_dump(exclude_unset=True)
        captured["environment"] = env
        captured["overlay_paths"] = [Path(p) for p in env.get("extra_docker_compose", [])]
        captured["overlay_existed_during_job"] = [p.exists() for p in captured["overlay_paths"]]
        if create_side_effect:
            raise create_side_effect
        return _ok_job()

    def removed(_name):
        events.append("containers_removed")
        return []

    original_release = runner.EnvLeaseHandle.release

    async def traced_release(self):
        events.append("lease_released")
        await original_release(self)

    post = gate_b if gate_b is not None else AsyncMock(return_value=None)
    with (
        patch.object(runner, "Job") as job_cls,
        patch("psrl.workers.env_service.docker.ManagedDockerEnvironment.prepare_runtime", new_callable=AsyncMock),
        patch.object(runner, "force_remove_compose_project", side_effect=removed),
        patch.object(runner, "force_remove_compose_images", return_value=0),
        patch.object(runner, "read_kernel_driver_version", return_value=driver_version),
        patch.object(runner, "resolve_driver_libs", return_value=_LIBS),
        patch(
            "psrl.workers.env_service.overlay.read_gpu_devices",
            return_value={i: GpuDevice(i, i, f"GPU-{i}") for i in range(8)},
        ),
        patch.object(runner, "post", post),
        patch.object(runner.EnvLeaseHandle, "release", traced_release),
    ):
        job_cls.create = AsyncMock(side_effect=create)
        result = asyncio.run(
            runner.run_harbor_episode(
                task_path=task_path,
                model_base_url="u",
                model_name="m",
                config=runtime or SciAccelRuntimeConfig(),
                needs_gpu=needs_gpu,
                session_id=SESSION,
                session_router_url="http://router:8200",
                regrade_unverified=False,
                env_client=client,
                env_demand=EnvDemand(needs_gpu=needs_gpu),
            )
        )
    captured["job_create_called"] = job_cls.create.await_count > 0
    captured["gate_b_calls"] = [call.kwargs.get("payload") for call in post.await_args_list]
    return result, captured


def test_borrowed_gpu_wires_the_same_lease_into_agent_and_container_hooks():
    from types import SimpleNamespace

    cfg = _cfg(max_containers=1)
    cfg["units"][0]["rollout_replica_rank"] = 0
    core = EnvRouterCore(cfg)
    binding = SimpleNamespace(phase="model", set_phase=AsyncMock(), finish=AsyncMock())
    core._gpu_bindings["g0"] = binding
    core._ledger._blocked_units.clear()
    result, captured = _run_episode(core, needs_gpu=True)
    assert result.reward == 1.0
    binding.set_phase.assert_awaited_once()
    lease, phase, sequence = binding.set_phase.await_args.args
    assert lease.gpu_lending and (phase, sequence) == ("env", 0)
    assert captured["agent"].name == "examples.sciaccel_rl.agent:TruncatingTerminus2"
    assert captured["environment"]["kwargs"]["gpu_lending"] == captured["agent"].kwargs["gpu_lending"]
    binding.finish.assert_awaited_once_with(lease.lease_id)


def test_command_gpu_wires_explicit_execution_and_skips_model_reply_gate():
    from types import SimpleNamespace

    cfg = _cfg()
    cfg["units"][0]["gpu_reservation"] = "command"
    core = EnvRouterCore(cfg)
    node = SimpleNamespace(
        remove_gpu_lease=SimpleNamespace(remote=AsyncMock()),
        gpu_lending_snapshot=SimpleNamespace(remote=AsyncMock(return_value={0: {"used_mb": 0}})),
    )
    core._gpu_commands.get_node = AsyncMock(return_value=node)
    result, captured = _run_episode(core, needs_gpu=True)
    assert result.reward == 1.0
    commands = captured["environment"]["kwargs"]["gpu_commands"]
    assert commands["lease_id"].startswith("g0-")
    assert commands["router_name"]
    assert all(not item["gpu_device"] for payload in captured["gate_b_calls"] for item in payload), (
        "GPU command ownership must not gate unrelated model replies."
    )
    node.remove_gpu_lease.remote.assert_awaited_once_with(commands["lease_id"])
    assert _ledger_containers(core) == 0


def _ledger_containers(core):
    return sum(snap.leases_active for snap in asyncio.run(core.snapshot()))


def test_deferred_gpu_return_failure_cannot_be_accepted_as_a_successful_episode():
    from types import SimpleNamespace

    cfg = _cfg()
    cfg["units"][0]["gpu_reservation"] = "command"
    core = EnvRouterCore(cfg)
    core._gpu_commands.get_node = AsyncMock(
        return_value=SimpleNamespace(
            remove_gpu_lease=SimpleNamespace(remote=AsyncMock()),
            gpu_lending_snapshot=SimpleNamespace(remote=AsyncMock(return_value={0: {"used_mb": 0}})),
        )
    )
    core.wait_gpu_command_returns = AsyncMock(side_effect=RuntimeError("Deferred GPU return failed."))
    result, _ = _run_episode(core, needs_gpu=True)
    assert result.reward == 0.0 and result.exception == "Deferred GPU return failed."
    core.wait_gpu_command_returns.assert_awaited_once()
    assert _ledger_containers(core) == 0


# --------------------------------------------------------------------------- the GPU lease path


def test_a_gpu_lease_becomes_an_overlay_with_its_own_devices_and_override_gpus_zero():
    core = EnvRouterCore(_cfg())
    result, captured = _run_episode(core, needs_gpu=True)
    assert result.reward == 1.0
    env = captured["environment"]
    # Harbor's local Docker backend declares no GPU capability, so it must be told none is needed.
    assert env["override_gpus"] == 0
    assert captured["overlay_existed_during_job"] == [True], "Harbor needs the overlay while the job runs"
    assert not captured["overlay_paths"][0].exists(), "the temp overlay must be deleted after the episode"


def test_the_overlay_content_carries_the_leases_devices_labels_and_resolved_libs():
    core = EnvRouterCore(_cfg())
    seen: dict = {}

    async def create(job_config):
        env = job_config.environment
        if hasattr(env, "model_dump"):
            env = env.model_dump(exclude_unset=True)
        path = Path(env["extra_docker_compose"][0])
        seen["overlay"] = yaml.safe_load(path.read_text())
        return _ok_job()

    with (
        patch.object(runner, "Job") as job_cls,
        patch("psrl.workers.env_service.docker.ManagedDockerEnvironment.prepare_runtime", new_callable=AsyncMock),
        patch.object(runner, "force_remove_compose_project", return_value=[]),
        patch.object(runner, "force_remove_compose_images", return_value=0),
        patch.object(runner, "read_kernel_driver_version", return_value="535.1"),
        patch.object(runner, "resolve_driver_libs", return_value=_LIBS),
        patch(
            "psrl.workers.env_service.overlay.read_gpu_devices",
            return_value={i: GpuDevice(i, i, f"GPU-{i}") for i in range(8)},
        ),
        patch.object(runner, "post", AsyncMock(return_value=None)),
    ):
        job_cls.create = AsyncMock(side_effect=create)
        asyncio.run(
            runner.run_harbor_episode(
                task_path="/t",
                model_base_url="u",
                model_name="m",
                config=SciAccelRuntimeConfig(),
                needs_gpu=True,
                session_id=SESSION,
                regrade_unverified=False,
                env_client=_client(core),
                env_demand=EnvDemand(needs_gpu=True),
            )
        )
    main = seen["overlay"]["services"]["main"]
    assert "/dev/nvidia0:/dev/nvidia0" in main["devices"] and "/dev/nvidia1:/dev/nvidia1" in main["devices"]
    assert any(label.startswith("psrl.env_lease=g0-") for label in main["labels"])
    assert "psrl.env_unit=g0" in main["labels"]
    assert "/usr/lib64/libcuda.so.535.1:/usr/lib64/libcuda.so.1:ro" in main["volumes"]


def test_the_lease_is_released_only_after_the_containers_are_gone():
    events: list[str] = []
    core = EnvRouterCore(_cfg())
    _run_episode(core, needs_gpu=True, events=events)
    assert events == ["containers_removed", "lease_released"]
    assert _ledger_containers(core) == 0


def test_gate_b_is_bound_with_all_devices_and_shared_then_unbound():
    core = EnvRouterCore(_cfg(max_containers=2))  # >1 container per unit, so the card is shared
    post = AsyncMock(return_value=None)
    _, captured = _run_episode(core, needs_gpu=True, gate_b=post)
    bind, unbind = captured["gate_b_calls"]
    assert bind == [
        {
            "session_id": SESSION,
            "gpu_device": "/dev/nvidia0,/dev/nvidia1",
            "shared": True,
            "node_service_url": "",
            "gpu_indices": [0, 1],
        }
    ]
    assert unbind == [
        {"session_id": SESSION, "gpu_device": None, "shared": False, "node_service_url": "", "gpu_indices": []}
    ]


def test_a_solo_unit_binds_gate_b_unshared():
    core = EnvRouterCore(_cfg(max_containers=1))
    _, captured = _run_episode(core, needs_gpu=True)
    assert captured["gate_b_calls"][0][0]["shared"] is False


# --------------------------------------------------------------------------- the CPU lease path


def test_a_cpu_lease_gets_labels_only_and_no_override_gpus_and_no_gate_b():
    core = EnvRouterCore(_cfg())
    result, captured = _run_episode(core, needs_gpu=False)
    assert result.reward == 1.0
    assert "override_gpus" not in captured["environment"]
    assert captured["gate_b_calls"] == []
    assert _ledger_containers(core) == 0


# --------------------------------------------------------------------------- failure paths never leak a lease


def test_a_job_failure_still_releases_the_lease():
    core = EnvRouterCore(_cfg())
    result, _ = _run_episode(core, needs_gpu=True, create_side_effect=RuntimeError("boom"))
    assert result.exception == "boom"
    assert _ledger_containers(core) == 0


def test_a_driver_mismatch_fails_the_episode_cleanly_and_releases_the_lease():
    core = EnvRouterCore(_cfg())
    events: list[str] = []
    result = _run_with_failing_driver(core, events)
    assert result.exception_type == "DriverLibraryError"
    assert result.exception.startswith("env_lease_setup_failed")
    assert events == ["lease_released"]
    assert _ledger_containers(core) == 0


def _run_with_failing_driver(core, events):
    original_release = runner.EnvLeaseHandle.release

    async def traced_release(self):
        events.append("lease_released")
        await original_release(self)

    with (
        patch.object(runner, "Job") as job_cls,
        patch("psrl.workers.env_service.docker.ManagedDockerEnvironment.prepare_runtime", new_callable=AsyncMock),
        patch.object(runner, "force_remove_compose_project", return_value=[]),
        patch.object(runner, "force_remove_compose_images", return_value=0),
        patch.object(runner, "read_kernel_driver_version", return_value="535.1"),
        patch.object(runner, "resolve_driver_libs", side_effect=DriverLibraryError("no libcuda for 535.1")),
        patch.object(runner, "post", AsyncMock(return_value=None)),
        patch.object(runner.EnvLeaseHandle, "release", traced_release),
    ):
        job_cls.create = AsyncMock()
        result = asyncio.run(
            runner.run_harbor_episode(
                task_path="/t",
                model_base_url="u",
                model_name="m",
                config=SciAccelRuntimeConfig(),
                needs_gpu=True,
                session_id=SESSION,
                regrade_unverified=False,
                env_client=_client(core),
                env_demand=EnvDemand(needs_gpu=True),
            )
        )
        assert job_cls.create.await_count == 0, "Harbor must not start when the lease cannot be realised"
    return result


def test_no_nvidia_kernel_driver_fails_the_gpu_episode_cleanly():
    core = EnvRouterCore(_cfg())
    result, captured = _run_episode(core, needs_gpu=True, driver_version=None)
    assert result.exception.startswith("env_lease_setup_failed")
    assert "no NVIDIA kernel driver" in result.exception
    assert captured["job_create_called"] is False
    assert _ledger_containers(core) == 0


def test_an_impossible_demand_is_an_episode_result_and_starts_nothing():
    core = EnvRouterCore({**_cfg(), "units": [_cfg()["units"][1]]})  # CPU unit only
    result, captured = _run_episode(core, needs_gpu=True)
    assert result.exception.startswith("env_admission_failed")
    assert result.exception_type == "EnvAdmissionRejected"
    assert result.reward == 0.0
    assert captured["job_create_called"] is False


def test_an_overlay_is_not_left_behind_when_the_job_fails():
    core = EnvRouterCore(_cfg())
    _, captured = _run_episode(core, needs_gpu=True, create_side_effect=RuntimeError("boom"))
    assert captured["overlay_paths"] and not captured["overlay_paths"][0].exists()


# --------------------------------------------------------------------------- the legacy path is untouched


def test_without_an_env_client_the_legacy_path_never_touches_the_router():
    with (
        patch.object(runner, "Job") as job_cls,
        patch("psrl.workers.env_service.docker.ManagedDockerEnvironment.prepare_runtime", new_callable=AsyncMock),
        patch.object(runner, "force_remove_compose_project", return_value=[]),
        patch.object(runner, "force_remove_compose_images", return_value=0),
        patch.object(runner, "GpuDeviceGate") as gate_cls,
    ):
        job_cls.create = AsyncMock(return_value=_ok_job())
        result = asyncio.run(
            runner.run_harbor_episode(
                task_path="/t",
                model_base_url="u",
                model_name="m",
                config=SciAccelRuntimeConfig(),
                needs_gpu=True,  # but no gpu_compose_override is configured, so no GPU gate either
                regrade_unverified=False,
            )
        )
    assert result.reward == 1.0
    gate_cls.assert_not_called()
    assert "override_gpus" not in (job_cls.create.await_args.args[0].environment or {})


def test_the_default_job_name_is_generated_when_no_session_id_is_given():
    core = EnvRouterCore(_cfg())
    client = _client(core)
    with (
        patch.object(runner, "Job") as job_cls,
        patch("psrl.workers.env_service.docker.ManagedDockerEnvironment.prepare_runtime", new_callable=AsyncMock),
        patch.object(runner, "force_remove_compose_project", return_value=[]) as removed,
        patch.object(runner, "force_remove_compose_images", return_value=0),
    ):
        job_cls.create = AsyncMock(return_value=_ok_job())
        asyncio.run(
            runner.run_harbor_episode(
                task_path="/t",
                model_base_url="u",
                model_name="m",
                config=SciAccelRuntimeConfig(),
                regrade_unverified=False,
                env_client=client,
            )
        )
    (name,) = removed.call_args.args
    assert name and name != SESSION
    assert _ledger_containers(core) == 0


def test_native_cuda_missing_attribution_fails_before_harbor_starts():
    config = _cfg()
    config["units"][0]["gpu_reservation"] = "command"
    core = EnvRouterCore(config)
    core.require_gpu_attribution = AsyncMock(side_effect=RuntimeError("Host PID procfs is unavailable."))
    core.finish_gpu_command_lease = AsyncMock()
    with patch.object(runner, "is_laps_cuda", return_value=True):
        result, captured = _run_episode(core, needs_gpu=True)
    assert not captured["job_create_called"]
    assert "Host PID procfs" in result.exception
    assert _ledger_containers(core) == 0


def test_command_lease_enables_default_science_gpu_job_tools():
    config = _cfg()
    config["units"][0]["gpu_reservation"] = "command"
    core = EnvRouterCore(config)
    core.finish_gpu_command_lease = AsyncMock()
    result, captured = _run_episode(core, needs_gpu=True)
    assert result.reward == 1
    assert captured["agent"].name == "examples.sciaccel_rl.agent:TruncatingTerminus2"
    assert captured["agent"].kwargs["gpu_jobs"] is True


def test_episode_lease_enables_the_same_gpu_tools_with_session_ownership():
    core = EnvRouterCore(_cfg())
    result, captured = _run_episode(core, needs_gpu=True)
    assert result.reward == 1
    assert captured["agent"].kwargs["gpu_jobs"] is True
    settings = captured["environment"]["kwargs"]
    assert settings["gpu_execution"]["mode"] == "episode"
    assert settings["gpu_execution"]["lease_id"].startswith("g0-")
    assert "gpu_commands" not in settings
    assert _ledger_containers(core) == 0


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)


def test_native_backend_uses_episode_budget_without_compose(tmp_path):
    from types import SimpleNamespace

    from psrl.workers.env_service.container_budget import ContainerBudget

    lease = SimpleNamespace(
        lease_id="native-lease",
        unit_id="cpu",
        gpu_indices=(),
        demand=SimpleNamespace(container_budget=ContainerBudget(2, 512, 0, 2)),
    )
    kwargs = {"import_path": "psrl.workers.env_worker.harbor:EnvWorkerEnvironment"}
    overlays = []
    result = asyncio.run(runner._apply_env_lease(lease, overlays, kwargs, "", "session"))
    assert result is None and not overlays, "Native CPU admission must not create a Compose overlay."
    assert kwargs["kwargs"]["container_budget"]["environments"] == 2, "Both environments must share one episode plan."
    assert kwargs["kwargs"]["lease_labels"]["psrl.env_lease"] == "native-lease", (
        "Containers must retain lease ownership."
    )
