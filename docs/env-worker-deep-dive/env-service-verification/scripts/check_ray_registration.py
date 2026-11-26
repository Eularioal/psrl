"""Verify a bounded registration window on a private CPU only Ray instance."""

import argparse
import json
import time
import uuid
from pathlib import Path

import ray


def check(output, node_ip):
    """Exercise real worker registration without allocating model or GPU workers."""
    result = {"started_at": time.time(), "worker_register_timeout_s": 90, "gpu_budget": 0}
    try:
        info = ray.init(
            address="local",
            num_cpus=1,
            num_gpus=0,
            include_dashboard=False,
            _node_ip_address=node_ip,
            _temp_dir=f"/tmp/register-{uuid.uuid4().hex[:8]}",
            object_store_memory=128 * 1024**2,
            _system_config={"worker_register_timeout_seconds": 90},
        )
        result["ray_tmp"] = info.address_info["session_dir"]

        @ray.remote
        def worker_ready():
            import os

            return {"pid": os.getpid(), "registered": True}

        result["worker"] = ray.get(worker_ready.remote(), timeout=90)
        result["complete"] = True
    finally:
        ray.shutdown()
        result["finished_at"] = time.time()
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--node-ip", required=True)
    args = parser.parse_args()
    check(args.output, args.node_ip)
