"""Run actual SMG and SessionRouter children, then kill their Ray owner."""

import argparse
import json
import os
import socket
import time
import uuid
from pathlib import Path

import psutil
import ray
from omegaconf import OmegaConf
from psrl.workers.gen.rollout_gateway import RolloutGateway
from psrl.workers.gen.smg_adapter import build_reward_router_args


@ray.remote(num_cpus=0)
class ProbeGateway(RolloutGateway.__ray_metadata__.modified_class):
    def _init_router_args(self):
        # A CPU-only router exercises process lifecycle without a training parameter server.
        return build_reward_router_args(self.config, self.smg_ip, self.smg_port, 0)

    def processes(self):
        return dict(owner=os.getpid(), smg=self.router_process.pid, session=self.session_router_process.pid)


def live(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    actors = []
    pids = []
    evidence = {}
    # Keep the old default occupied to prove neither router depends on it.
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 29000))
        occupied.listen()
        try:
            ray.init(
                address="local",
                num_cpus=4,
                num_gpus=0,
                include_dashboard=False,
                _temp_dir=f"/tmp/evgw-{uuid.uuid4().hex[:8]}",
            )
            for i in range(2):
                config = OmegaConf.create(
                    {"psrl": {"logging_path": str(out / f"gateway-{i}"), "agentic_rl": {}, "rollout_gateway": {}}}
                )
                actor = ProbeGateway.remote(config, "127.0.0.1", 65530)
                actors.append(actor)
                ray.get(actor.launch_router.remote(), timeout=60)
                ray.get(actor.launch_session_router.remote(), timeout=60)
                row = ray.get(actor.processes.remote())
                row["listeners"] = [
                    c.laddr.port
                    for c in psutil.Process(row["smg"]).net_connections(kind="tcp")
                    if c.status == "LISTEN"
                ]
                pids.append(row)
            evidence["running"] = pids
            assert all(live(row[k]) for row in pids for k in ("smg", "session"))
            assert all(29000 not in row["listeners"] for row in pids)
            for actor in actors:
                ray.kill(actor, no_restart=True)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if all(not live(row[k]) for row in pids for k in ("owner", "smg", "session")):
                    break
                time.sleep(0.1)
            evidence["all_processes_exited"] = all(
                not live(row[k]) for row in pids for k in ("owner", "smg", "session")
            )
            assert evidence["all_processes_exited"]
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
            ray.shutdown()
            (out / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output.resolve())
