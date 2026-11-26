"""Measure real Ray handle imports without loading model weights or using GPUs."""

import argparse
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

import ray
from psrl.workers.env_service.router import EnvRouterCore


class StartupRouter(EnvRouterCore):
    def snapshot(self):
        return {
            "pid": os.getpid(),
            "torch_loaded": "torch" in sys.modules,
            "vllm_server_loaded": "psrl.workers.gen.vllm_async_server" in sys.modules,
            "blocked": sorted(self._ledger._blocked_units),
            "active": sorted(self._gpu_commands.shared_servers),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-seconds", type=float, default=30)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "scope": "CPU Ray mechanism probe. The server inherits the real vLLM class but has no engine or model.",
        "synthetic_weight_preparation_s": args.preparation_seconds,
        "started_at": time.time(),
        "arms": {},
    }

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    save()
    # Import the production actor class before defining the CPU profile stub.
    started = time.monotonic()
    try:
        from psrl.workers.gen.vllm_async_server import PSRL_vLLMHttpServer
    except BaseException as exc:
        record.update(status="failed", error=repr(exc), finished_at=time.time())
        save()
        raise

    record["driver_server_class_import_s"] = time.monotonic() - started

    class ProfileOnlyServer(PSRL_vLLMHttpServer):
        def __init__(self):
            pass

        async def validate_env_sharing(self, headroom_mb, margin_mb):
            if headroom_mb < 8 + margin_mb:
                raise ValueError("Insufficient test headroom.")
            return {"required_headroom_mb": 8 + margin_mb}

        def ready(self):
            return True

    config = {
        "node_policy": "any",
        "gpu_sharing": {"safety_margin_mb": 4},
        "units": [
            {
                "unit_id": "shared",
                "node_ip": "test",
                "cpu_quota": 1,
                "memory_mb": 4096,
                "gpu_indices": [0],
                "gpu_memory_mb": 48000,
                "max_containers": 1,
                "shared_rollout_replica_rank": 0,
                "model_headroom_mb": 35,
                "gpu_reservation": "command",
            }
        ],
    }
    handles = []
    try:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=0,
            include_dashboard=False,
            namespace=f"gpu-control-startup-{uuid.uuid4().hex[:8]}",
            _temp_dir=f"/tmp/gcs-{uuid.uuid4().hex[:6]}",
            object_store_memory=100 * 1024 * 1024,
        )
        server = ray.remote(num_cpus=1)(ProfileOnlyServer).remote()
        handles.append(server)
        ray.get(server.ready.remote(), timeout=150)
        for arm in ("serial", "overlap"):
            router = ray.remote(num_cpus=1)(StartupRouter).remote(config)
            handles.append(router)
            before = ray.get(router.snapshot.remote(), timeout=90)
            started = time.monotonic()
            if arm == "overlap":
                prepared = router.prepare_gpu_bindings.remote({"shared": server}, None)
            # This delay represents independent weight preparation, not real GPU work.
            time.sleep(args.preparation_seconds)
            if arm == "serial":
                prepared = router.prepare_gpu_bindings.remote({"shared": server}, None)
            ray.get(prepared, timeout=150)
            prepared_s = time.monotonic() - started
            staged = ray.get(router.snapshot.remote(), timeout=10)
            ray.get(router.activate_gpu_bindings.remote(), timeout=30)
            after = ray.get(router.snapshot.remote(), timeout=10)
            record["arms"][arm] = {
                "before": before,
                "prepared": staged,
                "after": after,
                "through_prepared_s": prepared_s,
                "through_activated_s": time.monotonic() - started,
            }
            assert before["blocked"] == staged["blocked"] == ["shared"]
            assert not staged["active"] and after["active"] == ["shared"]
            assert not before["torch_loaded"] and staged["torch_loaded"]
            save()
            ray.kill(router, no_restart=True)
            handles.remove(router)
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        for handle in handles:
            ray.kill(handle, no_restart=True)
        ray.shutdown()
        record["finished_at"] = time.time()
        save()


if __name__ == "__main__":

    def timeout_signal(signum, frame):
        raise TimeoutError("CPU startup probe exhausted its external time budget.")

    signal.signal(signal.SIGTERM, timeout_signal)
    main()
