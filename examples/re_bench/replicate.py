"""Transport official judge requests over artifacts from a networkless container."""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

MODEL = "meta/meta-llama-3-8b-instruct"
failures = 0


async def async_run(model, input):
    global failures
    try:
        return await _async_run(model, input)
    except Exception:
        failures += 1
        raise


async def _async_run(model, input):
    if model != MODEL:
        raise ValueError("Only the official RE-Bench judge model is allowed.")
    root = Path(os.environ.get("REBENCH_JUDGE_DIR", "/logs/artifacts/rebench-judge"))
    requests = root / "requests"
    responses = root / "responses"
    requests.mkdir(parents=True, exist_ok=True)
    responses.mkdir(exist_ok=True)
    request_id = uuid.uuid4().hex
    temporary = requests / f"{request_id}.tmp"
    temporary.write_text(json.dumps({"model": model, "input": input}))
    temporary.replace(requests / f"{request_id}.json")
    deadline = time.monotonic() + float(os.environ.get("REBENCH_JUDGE_TIMEOUT", "180"))
    response = responses / f"{request_id}.json"
    while time.monotonic() < deadline:
        if response.exists():
            payload = json.loads(response.read_text())
            if payload.get("error"):
                raise RuntimeError(f"RE-Bench judge request failed: {payload['error']}.")
            return [payload["output"]]
        await asyncio.sleep(0.1)
    raise TimeoutError("RE-Bench judge did not reply before the deadline.")
