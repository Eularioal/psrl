"""Run bounded PostTrain actions through PSRL sessions and managed GPU commands."""

import asyncio
import importlib.util
import json
import logging
import os
import re
import shlex
import sys
import time
import uuid
from pathlib import Path

import yaml
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from omegaconf import OmegaConf
from psrl.utils.common.http_utils import post
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop, SessionAgentResult
from psrl.workers.agent_loop.loops.utils import register
from psrl.workers.env_service.client import EnvServiceClient
from psrl.workers.env_service.docker import ManagedDockerEnvironment
from psrl.workers.env_service.overlay import (
    build_lease_overlay,
    read_kernel_driver_version,
    resolve_driver_libs,
    write_overlay,
)
from psrl.workers.env_service.types import EnvDemand

psrl_logger = logging.getLogger(__name__)


@register("integrated_posttrain")
class IntegratedPostTrainAgent(SessionAgentLoop):
    """Optimize a bounded adapter and train the policy on private likelihood reward."""

    def __init__(self, context, **kwargs):
        super().__init__(context)
        self.client = EnvServiceClient.from_config(
            OmegaConf.to_container(context.config.psrl.env_service, resolve=True)
        )
        self.settings = json.loads(Path(os.environ["PARTITION_SETTINGS"]).read_text())

    async def run_session(self, request, api_base_url):
        async with asyncio.timeout(720):
            return await self._episode(request, api_base_url)

    async def _episode(self, request, api_base_url):
        settings = self.settings
        folder = Path(settings["output"]) / "episodes" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        artifact = folder / "artifact"
        artifact.mkdir()
        record = {
            "uid": str(request["uid"]),
            "parent_id": str(request.get("parent_id")),
            "version_tag": request.get("version_tag"),
            "started_at": time.time(),
            "queries": [],
            "commands": [],
            "status": "running",
            "session_api_url": api_base_url,
            "source_paths": {
                name: importlib.util.find_spec(name).origin for name in ("psrl", "verl", "transfer_queue", "vllm")
            },
        }

        def save():
            temporary = folder / "episode.tmp"
            temporary.write_text(json.dumps(record, indent=2, default=str))
            temporary.replace(folder / "episode.json")

        messages = [
            {
                "role": "user",
                "content": (
                    "Choose a learning rate for adapting only the final normalization weights of Qwen3-4B "
                    "on two GSM8K training examples. The budget is fixed. Evaluation uses private answer likelihood. "
                    "Choose exactly one of 0.0003, 0.001, 0.003 and begin with LR=<number>. "
                    "Explain your choice and the risk of overfitting in about 150 words."
                ),
            }
        ]
        sampling = self.get_session_sampling_params(request)
        sampling["max_tokens"] = 384

        async def generate():
            started = time.time()
            with self.timer.generation():
                response = await post(
                    f"{api_base_url}/chat/completions",
                    payload={"model": self.model_config.path, "messages": messages, **sampling},
                )
            message = response["choices"][0]["message"]
            messages.append({"role": "assistant", "content": message.get("content") or ""})
            record["queries"].append(
                {"started_at": started, "finished_at": time.time(), "usage": response.get("usage")}
            )
            save()
            return messages[-1]["content"]

        save()
        handle = None
        environment = None
        overlay_path = None
        try:
            answer = await generate()
            match = re.search(r"LR\s*=\s*(0\.0003|0\.001|0\.003)(?!\d)", answer)
            learning_rate = float(match[1]) if match else 0.001
            record.update(learning_rate=learning_rate, action_parsed=bool(match))
            handle = await self.client.acquire(str(folder.name), EnvDemand(2, 16384, True, 16384), max_wait_s=600)
            record["host_unit"] = handle.lease.unit_id
            overlay = build_lease_overlay(handle.lease, resolve_driver_libs(read_kernel_driver_version()))
            overlay_path = write_overlay(overlay)
            mounts = [
                f"{settings['source']}:/study:ro",
                f"{settings['env_model']}:/model:ro",
                f"{settings['data']}:/benchmark:ro",
                f"{artifact}:/artifact:rw",
                f"{settings['bootstrap']}:/bootstrap:ro",
            ]
            for prefix in dict.fromkeys(
                (Path(sys.executable).parent.parent, Path(sys.executable).resolve().parent.parent)
            ):
                mounts.append(f"{prefix}:{prefix}:ro")
            (folder / "docker-compose.yaml").write_text(
                yaml.safe_dump(
                    {
                        "services": {
                            "main": {
                                "volumes": mounts,
                                "environment": {
                                    "PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin",
                                    "PYTHONPATH": "/bootstrap",
                                    "OMP_NUM_THREADS": "2",
                                    "MKL_NUM_THREADS": "2",
                                    "OPENBLAS_NUM_THREADS": "1",
                                    "HF_HUB_OFFLINE": "1",
                                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                },
                            }
                        }
                    }
                )
            )
            environment = ManagedDockerEnvironment(
                environment_dir=folder,
                environment_name=f"{settings['tag']}-{folder.name[:10]}",
                session_id=folder.name,
                trial_paths=TrialPaths(folder / "trial"),
                task_env_config=EnvironmentConfig(docker_image=settings["image"], cpus=2, memory_mb=16384),
                extra_docker_compose=[overlay_path],
                lease_overlay=overlay,
                gpu_execution={
                    "router_name": settings["router_name"],
                    "lease_id": handle.lease.lease_id,
                    "mode": "command",
                },
                network_policy=NetworkPolicy(network_mode="no-network"),
            )
            await environment.start(force_build=False)

            async def command(mode):
                started = time.time()
                args = [
                    sys.executable,
                    "/study/partition_worker.py",
                    "--mode",
                    mode,
                    "--steps",
                    str(request.get("extra_info", {}).get("updates", 448)),
                    "--learning-rate",
                    str(learning_rate),
                    "--data-root",
                    "/benchmark/public" if mode == "train" else "/benchmark/private",
                ]
                result = await environment.exec_gpu(
                    shlex.join(args),
                    timeout_sec=240,
                    wait_timeout_sec=600,
                    overlap_return=False,
                )
                (folder / f"{mode}.stdout").write_text(result.stdout or "")
                (folder / f"{mode}.stderr").write_text(result.stderr or "")
                if result.return_code:
                    raise RuntimeError(f"PostTrain command failed with exit {result.return_code!r}.")
                value = json.loads(
                    next(
                        line.split("=", 1)[1]
                        for line in result.stdout.splitlines()
                        if line.startswith("PARTITION_RESULT=")
                    )
                )
                if value["cuda_readback"] != 528:
                    raise RuntimeError("Container CUDA verification failed.")
                record["commands"].append(
                    {
                        "mode": mode,
                        "started_at": started,
                        "finished_at": time.time(),
                        "result": value,
                        "admission": environment.last_gpu_command,
                    }
                )
                save()
                return value

            trained = await command("train")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The adapter completed {trained['updates']} updates at learning rate {learning_rate}. "
                        "Explain what we should check on held-out answer likelihood before accepting it."
                    ),
                }
            )
            await generate()
            verified = await command("verify")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Private answer NLL is {verified['answer_nll']:.6f} "
                        f"and exp(-NLL) is {verified['reward']:.6f}. "
                        "Summarize the result, its limitations and a possible next experiment."
                    ),
                }
            )
            await generate()
            record.update(status="complete", reward=verified["reward"])
            return SessionAgentResult(
                extra_fields={
                    "harbor_rewards": {"reward": verified["reward"]},
                    "reward_key": "reward",
                    "task_name": "posttrain-bounded-adapter",
                }
            )
        except BaseException as exc:
            record.update(status="failed", error=repr(exc))
            raise
        finally:
            try:
                if environment:
                    await environment.stop(delete=True)
                if handle:
                    await handle.release()
            finally:
                if overlay_path:
                    Path(overlay_path).unlink(missing_ok=True)
                record["finished_at"] = time.time()
                save()
