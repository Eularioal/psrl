"""Explicit managed-job instructions for bounded functional acceptance."""

import json


def managed_baseline_guidance(command, cwd="/"):
    start = json.dumps({"op": "start", "command": command, "cwd": cwd, "tag": "baseline"})
    wait = json.dumps({"op": "wait", "job_id": "RETURNED_JOB_ID", "timeout_sec": 180, "released": False})
    return (
        "\nFunctional acceptance runbook, using the direct backend:\n"
        "Do not inspect files, select another backend, change the command or run extra experiments. "
        "Your first response must execute this exact standalone keystrokes line:\n"
        f"psrl-gpu {start}\n"
        "In your next response, replace RETURNED_JOB_ID with the returned job_id and execute:\n"
        f"psrl-gpu {wait}\n"
        "If output_ready is false, repeat the same wait. Once output_ready is true and exit_code is zero, "
        "mark task_complete true. psrl-gpu is intercepted by PSRL, not a shell executable. "
        "Never prefix it with sleep, export or a shell operator. Do not specify gpu_memory_mb: "
        "this direct-backend acceptance already reserves the task GPU."
    )
