"""Run concurrent real CIFAR training commands without auxiliary GPU allocations."""

import asyncio
import json
import time

from harbor.agents.nop import NopAgent


class NaturalGpuAgent(NopAgent):
    async def run(self, instruction, environment, context):
        spec = json.loads(instruction)
        record = {"started_at": time.time(), "commands": [], "spec": spec}
        destination = self.logs_dir / "natural-gpu.json"

        async def command(index):
            item = {"index": index, "submitted_at": time.time()}
            record["commands"].append(item)
            cmd = f"{spec['python']} /opt/workload/train.py --index {index}"
            try:
                if spec["mode"] == "on":
                    if spec.get("prepared"):
                        output = await environment.exec_gpu_prepared(
                            "/opt/workload/train.py",
                            {"index": index},
                            python=spec["python"],
                            prepare_timeout_sec=180,
                            timeout_sec=180,
                            wait_timeout_sec=300,
                        )
                    else:
                        output = await environment.exec_gpu(cmd, timeout_sec=180, wait_timeout_sec=600)
                    item["reservation"] = (
                        output.reservation if spec.get("prepared") else dict(environment.last_gpu_command)
                    )
                else:
                    output = await environment.exec(cmd, timeout_sec=180)
                item.update(return_code=output.return_code, stdout=output.stdout, stderr=output.stderr)
            except Exception as exc:  # noqa: BLE001
                item["error"] = repr(exc)
            finally:
                item["finished_at"] = time.time()
                destination.write_text(json.dumps(record, indent=2))

        try:
            async with asyncio.timeout(600):
                await asyncio.gather(*(command(i) for i in range(spec["count"])))
        finally:
            record["finished_at"] = time.time()
            destination.write_text(json.dumps(record, indent=2))
        context.metadata = {"scope": "Real CIFAR training commands, no language model agent"}
