"""Keep an official inference server alive across separate environment commands."""

import json
import time

from harbor.agents.nop import NopAgent


class ResidentInferenceAgent(NopAgent):
    async def run(self, instruction, environment, context):
        spec = json.loads(instruction)
        record = {"started_at": time.time(), "commands": [], "status": "running"}

        def save():
            (self.logs_dir / "resident.json").write_text(json.dumps(record, indent=2))

        async def execute(command, timeout=90):
            item = {"command": command, "started_at": time.time()}
            record["commands"].append(item)
            result = await environment.exec(command, timeout_sec=timeout)
            item.update(finished_at=time.time(), return_code=result.return_code, stderr=result.stderr)
            save()
            if result.return_code:
                raise RuntimeError(f"Resident command failed: {result.stderr!r} {result.stdout!r}.")
            return result

        try:
            await execute(
                "nohup python /opt/resident/server.py --model /opt/model --dtype bfloat16 "
                "--max-model-len 2048 >/logs/artifacts/server.log 2>&1 </dev/null & echo $! >/tmp/resident.pid"
            )
            await execute("python /opt/resident/client.py --ready", timeout=250)
            record["ready_at"] = time.time()
            save()
            await execute("python /opt/resident/client.py --index 0")
            record["idle_started_at"] = time.time()
            await execute(f"sleep {float(spec['idle_s'])}")
            record["idle_finished_at"] = time.time()
            await execute("python /opt/resident/client.py --index 1")
            record["status"] = "ok"
            context.metadata = {"scope": "Real MMLU-Pro requests, no official quality judge or policy optimizer"}
        except BaseException as exc:
            record.update(status="failed", error=repr(exc))
            raise
        finally:
            try:
                await execute(
                    "test ! -f /tmp/resident.pid || kill -TERM $(cat /tmp/resident.pid) 2>/dev/null || true",
                    timeout=10,
                )
            finally:
                record["finished_at"] = time.time()
                save()
