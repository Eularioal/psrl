"""Measure connection lifetime costs without allocating GPU memory or starting containers."""

import argparse
import json
import time
from pathlib import Path

from psrl.workers.env_service.probe import nvml_session, read_gpu_devices, read_gpu_snapshot


def measure(gpu, repeats):
    rows = []
    for _ in range(repeats):
        for name, query in (("inventory", read_gpu_devices), ("snapshot", lambda: read_gpu_snapshot((gpu,)))):
            start = time.perf_counter()
            query()
            rows.append({"query": name, "duration_s": time.perf_counter() - start})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"scope": "Read-only real GPU query timing, six queries per condition"}
    result["per_query_connection"] = measure(args.gpu, 3)
    start = time.perf_counter()
    with nvml_session():
        result["connection_start_s"] = time.perf_counter() - start
        result["retained_connection"] = measure(args.gpu, 3)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
