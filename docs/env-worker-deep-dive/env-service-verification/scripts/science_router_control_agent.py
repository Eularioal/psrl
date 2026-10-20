"""Execute equal CUDA work with a CPU interval through the science Harbor runner."""

import json
import time
from pathlib import Path

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.nop import NopAgent


class RouterControlAgent(NopAgent):
    async def run(self, instruction, environment, context):
        spec = json.loads(instruction)
        record = {"started_at": time.time(), "phases": [], "status": "running"}
        destination = self.logs_dir / "router-control.json"
        await environment.upload_file(Path(__file__).with_name("cuda_pressure.py"), "/tmp/router-pressure.py")
        try:
            for phase in range(2):
                command = (
                    f"python3 /tmp/router-pressure.py --gib {spec['gib']} --hold {spec['hold_s']} "
                    f"--output /logs/artifacts/pressure-{phase}.json"
                )
                item = {"phase": phase, "requested_at": time.time()}
                record["phases"].append(item)
                if spec["mode"] == "on":
                    output = await environment.exec_gpu(
                        command, timeout_sec=40, wait_timeout_sec=180, overlap_return=phase == 0
                    )
                    item["reservation"] = environment.last_gpu_command
                else:
                    output = await environment.exec(command, timeout_sec=40)
                item.update(finished_at=time.time(), return_code=output.return_code)
                item["cuda"] = json.loads(output.stdout.strip())
                destination.write_text(json.dumps(record, indent=2))
                if output.return_code:
                    record["status"] = item["cuda"]["status"]
                    raise NonZeroAgentExitCodeError(f"Controlled CUDA command failed: {output.return_code!r}.")
                if phase == 0:
                    record["cpu_started_at"] = time.time()
                    cpu = await environment.exec(f"sleep {spec['cpu_s']}", timeout_sec=30)
                    assert cpu.return_code == 0, "CPU interval failed."
                    record["cpu_finished_at"] = time.time()
                    if spec["mode"] == "on":
                        await environment.wait_for_gpu_return()
            record["status"] = "ok"
            context.metadata = {"scope": "Controlled CUDA and CPU work, no model inference"}
        finally:
            record["finished_at"] = time.time()
            destination.write_text(json.dumps(record, indent=2))
