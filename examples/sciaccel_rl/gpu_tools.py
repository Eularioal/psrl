"""Expose explicit GPU jobs through the science agent's existing command schema."""

import json

GPU_JOB_INSTRUCTIONS = """
GPU work must use the managed GPU job tool. Ordinary terminal commands have no GPU grant.
In a command's keystrokes, send one standalone line beginning with psrl-gpu and a JSON object.
Examples:
psrl-gpu {"op":"start","command":"./solver","cwd":"/workspace","tag":"check-name"}
psrl-gpu {"op":"status","job_id":"the returned id"}
psrl-gpu {"op":"wait","job_id":"the returned id","timeout_sec":1,"released":false}
psrl-gpu {"op":"cancel","job_id":"the returned id"}
psrl-gpu {"op":"telemetry","job_id":"the returned id"}
Start returns immediately. Continue independent CPU editing or planning while it runs or queues.
Wait only observes and never cancels on observation timeout. Output readiness can precede GPU return.
Each job runs in a separate bash shell. Pass cwd and env explicitly. Files are shared with
the terminal, but terminal-local variables and directory changes are not inherited by GPU jobs.
Run the entire GPU program in the foreground. Do not detach GPU processes or pass tmux commands.
For a scored GPU check, use its check name as tag and run the check's whole solver in that job.
Collect dependent results only after output_ready is true and exit_code is zero.
Telemetry reports only this container's observed process activity. Missing metrics are unknown.
Do not mark the task complete until its GPU jobs have completed.
"""


async def dispatch_gpu_tool(environment, text):
    """Return a structured observation without evaluating shell text on the host."""
    request = json.loads(text.removeprefix("psrl-gpu").strip())
    if not isinstance(request, dict):
        raise ValueError("GPU job request must be a JSON object.")
    operation = request.pop("op")
    if operation == "start":
        return environment.start_gpu_job(**request)
    if operation == "status":
        return environment.gpu_job_status(**request)
    if operation == "wait":
        return await environment.wait_gpu_job(**request)
    if operation == "cancel":
        return await environment.cancel_gpu_job(**request)
    if operation == "telemetry":
        return await environment.gpu_job_telemetry(**request)
    raise ValueError(f"Unknown GPU job operation {operation!r}.")
