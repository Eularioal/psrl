"""Run an owned, bounded checkpoint experiment on one idle private GPU."""

import argparse
import hashlib
import json
import re
import subprocess
import time
import uuid
from pathlib import Path


def launch(args):
    """Limit resources and remove only this experiment container on every exit."""
    args.output.mkdir(parents=True, exist_ok=False)
    token = "memory-state-" + uuid.uuid4().hex[:10]
    evidence = dict(status="running", token=token, budget_s=args.budget_seconds)

    def save():
        temporary = args.output / "controller.tmp"
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(args.output / "controller.json")

    def query(argv):
        return subprocess.check_output(argv, text=True, timeout=10).strip()

    save()
    gpu_rows = query(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
    ).splitlines()
    index, gpu_uuid, used = next(row.split(", ") for row in gpu_rows if int(row.split(",")[0]) == args.gpu)
    if int(used) >= 32:
        raise RuntimeError("The selected device is no longer idle.")
    info = next(
        path.read_text()
        for path in Path("/proc/driver/nvidia/gpus").glob("*/information")
        if gpu_uuid in path.read_text()
    )
    minor = int(re.search(r"Device Minor:\s*(\d+)", info).group(1))
    version = re.search(r"Kernel Module\s+([0-9.]+)", Path("/proc/driver/nvidia/version").read_text()).group(1)
    evidence.update(
        gpu=dict(index=int(index), uuid=gpu_uuid, minor=minor),
        image=args.image,
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")
        },
    )
    base = [
        "docker",
        "run",
        "--rm",
        "--name",
        token,
        "--label",
        f"psrl.memory_study={token}",
        "--network=none",
        "--cpus=8",
        "--memory=16g",
        "--memory-swap=16g",
        "--pids-limit=512",
        "--entrypoint=python3",
        "-e",
        f"CUDA_VISIBLE_DEVICES={gpu_uuid}",
        "-e",
        f"PSRL_EXPECTED_GPU_UUID={gpu_uuid}",
        "-e",
        "LD_LIBRARY_PATH=/usr/local/cuda/compat:/host-driver:/usr/local/cuda/lib64",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "OMP_NUM_THREADS=4",
        "-v",
        f"{Path(__file__).resolve().parent}:/study:ro",
        "-v",
        f"{args.cache}:/hf-cache:ro",
        "-v",
        f"{args.output}:/logs:rw",
    ]
    for device in [f"/dev/nvidia{minor}", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]:
        if Path(device).exists():
            base.extend(["--device", device])
    for library in ("libcuda", "libnvidia-ml", "libnvidia-ptxjitcompiler"):
        path = next(
            (
                Path(directory) / f"{library}.so.{version}"
                for directory in ("/usr/lib64", "/usr/lib/x86_64-linux-gnu")
                if (Path(directory) / f"{library}.so.{version}").is_file()
            ),
            None,
        )
        if path:
            base.extend(["-v", f"{path}:/host-driver/{library}.so.1:ro"])
    started = time.monotonic()
    entry = ["/study/memory_runtime.py", "--output", "/logs/profile.json"]
    if args.plan:
        (args.output / "plan.json").write_bytes(args.plan.read_bytes())
        (args.output / "input-profile.json").write_bytes(args.profile.read_bytes())
        entry = [
            "/study/memory_study.py",
            "--output",
            "/logs/study.json",
            "--profile",
            "/logs/input-profile.json",
            "--plan",
            "/logs/plan.json",
            "--budget-seconds",
            str(args.budget_seconds - 20),
        ]
    save()
    try:
        with (args.output / "runtime.log").open("w") as log:
            process = subprocess.run(
                [*base, args.image, *entry],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.budget_seconds,
            )
        evidence["runtime_exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError("The checkpoint runtime failed. See its retained log.")
        evidence["status"] = "passed"
    except BaseException as error:
        evidence.update(status="failed", error=repr(error))
        raise
    finally:
        subprocess.run(["docker", "rm", "-f", token], capture_output=True, text=True, timeout=20)
        evidence["remaining_containers"] = query(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.memory_study={token}"]
        ).splitlines()
        for attempt in range(6):
            gpu_line = query(
                ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"]
            ).splitlines()
            current = int(next(row.split(", ")[1] for row in gpu_line if row.startswith(gpu_uuid)))
            evidence["final_gpu_used_mib"] = current
            if current < 32:
                break
            if attempt < 5:
                time.sleep(1)
        evidence["elapsed_s"] = time.monotonic() - started
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--budget-seconds", type=int, default=180)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if not 30 <= args.budget_seconds <= 600:
        raise ValueError("The experiment budget must be between 30 and 600 seconds.")
    if args.plan and not args.profile:
        raise ValueError("The scheduling study needs an independent input profile.")
    launch(args)
