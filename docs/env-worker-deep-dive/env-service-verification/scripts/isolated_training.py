"""Own an isolated Ray cluster and give every training subprocess its address."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import ray


def pointer():
    path = Path("/tmp/ray/ray_current_cluster")
    if not path.exists():
        return None
    return dict(
        content=path.read_text(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        mtime_ns=path.stat().st_mtime_ns,
    )


def main(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex[:8]
    evidence = dict(started_at=time.time(), original_pointer=pointer())
    child = None
    try:
        info = ray.init(
            address="local",
            num_cpus=32,
            num_gpus=8,
            include_dashboard=False,
            _temp_dir=f"/tmp/evtrain-{token}",
            namespace=f"evtrain-{token}",
            log_to_driver=False,
        )
        address = info.address_info["gcs_address"]
        evidence.update(address=address, node_ip=ray.util.get_node_ip_address())
        env = dict(
            os.environ,
            RAY_ADDRESS=address,
            VERIFY_RAY_ADDRESS=address,
            VERIFY_TIMEOUT_SECONDS=str(args.timeout),
            PYTHONPATH=".",
        )
        if args.dependency_path:
            env["VERIFY_DEPENDENCY_PATH"] = str(args.dependency_path.resolve())
            env["PYTHONPATH"] = f"{args.repo.resolve()}:{args.dependency_path.resolve()}"
        (out / "cluster.json").write_text(json.dumps(evidence, indent=2))
        with (out / "launcher.log").open("w") as log:
            child = subprocess.Popen(
                [
                    "bash",
                    str(Path(__file__).with_name("one_step_science.sh")),
                    str(args.repo.resolve()),
                    str(args.dataset.resolve()),
                    str(args.model.resolve()),
                    str(out),
                    evidence["node_ip"],
                ],
                cwd=args.repo,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            evidence["exit_code"] = child.wait(timeout=args.timeout + 180)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        ray.shutdown()
        evidence.update(finished_at=time.time(), pointer_after=pointer())
        evidence["original_pointer_unchanged"] = evidence["original_pointer"] == evidence["pointer_after"]
        (out / "cluster.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)
    raise SystemExit(evidence["exit_code"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--dependency-path", type=Path)
    main(parser.parse_args())
