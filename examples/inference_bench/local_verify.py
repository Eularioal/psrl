"""Replay a submitted inference service and check fixed deterministic responses."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path


def stop_service(service, timeout=10):
    """Reap the service and kill its process group even if the shell has exited."""
    try:
        try:
            os.killpg(service.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            service.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(service.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        service.wait(timeout=timeout)


def main(candidate=Path("/logs/artifacts/serve.sh"), output=Path("/logs/verifier")):
    reward = {"reward": 0.0, "local_evaluation": 1, "eval_valid": 0, "quality_pass": 0}
    if not candidate.is_file():
        reward.update(eval_valid=1, valid_submission=0)
        (output / "reward.json").write_text(json.dumps(reward))
        return
    references = json.loads(Path("/tests/references.json").read_text())
    service = None
    try:
        with (output / "service.log").open("w") as log:
            client_log = Path("/logs/artifacts/server.log")
            client_log.unlink(missing_ok=True)
            client_log.symlink_to(output / "service.log")
            service = subprocess.Popen(
                ["bash", str(candidate)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            Path("/tmp/resident.pid").write_text(str(service.pid))
            subprocess.run(["python", "/opt/resident/client.py", "--ready"], check=True, timeout=250)
            started = time.monotonic()
            actual = []
            for index in range(2):
                path = Path(f"/logs/artifacts/request-{index}.json")
                path.unlink(missing_ok=True)
                subprocess.run(["python", "/opt/resident/client.py", "--index", str(index)], check=True, timeout=90)
                item = json.loads(path.read_text())
                actual.append(item["response"]["choices"][0]["message"]["content"])
            elapsed = time.monotonic() - started
            valid = actual == references
            reward.update(
                reward=1 / (1 + elapsed) if valid else 0, quality_pass=int(valid), eval_valid=1, seconds=elapsed
            )
            (output / "responses.json").write_text(json.dumps(actual))
    finally:
        if service is not None:
            stop_service(service)
        (output / "reward.json").write_text(json.dumps(reward))


if __name__ == "__main__":
    main()
