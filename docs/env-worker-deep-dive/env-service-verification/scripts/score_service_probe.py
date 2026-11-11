"""Run bounded container readiness checks and the unchanged score task verifier."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time
import uuid
from pathlib import Path

psrl_logger = logging.getLogger(__name__)


def main() -> None:
    """Use one idle device and retain logs for each completed stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--task-budget", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.task_budget <= 900:
        raise ValueError("Task budget must be between 1 and 900 seconds.")
    args.output.mkdir(parents=True, exist_ok=False)
    token = f"score-probe-{uuid.uuid4().hex[:10]}"
    evidence = dict(token=token, status="running", stages={}, scope="Container verifier, not a Router comparison.")

    def save():
        temporary = args.output / "evidence.tmp"
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(args.output / "evidence.json")

    def query(command):
        return subprocess.check_output(command, text=True, timeout=15).strip()

    rows = query(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]).splitlines()
    index, gpu_uuid, used = next(row.split(", ") for row in rows if int(row.split(",")[0]) == args.gpu)
    if int(used) >= 64:
        raise RuntimeError(f"Selected GPU is occupied: {used!r} MiB.")
    info = next(
        path.read_text()
        for path in Path("/proc/driver/nvidia/gpus").glob("*/information")
        if gpu_uuid in path.read_text()
    )
    minor = int(re.search(r"Device Minor:\s*(\d+)", info).group(1))
    version = re.search(r"Kernel Module\s+([0-9.]+)", Path("/proc/driver/nvidia/version").read_text()).group(1)
    evidence.update(gpu=dict(index=int(index), uuid=gpu_uuid, minor=minor), driver=version)
    base = [
        "docker",
        "run",
        "--rm",
        "--label",
        f"psrl.experiment={token}",
        "--network=none",
        "--cpus=8",
        "--memory=64g",
        "--shm-size=8g",
        "--pids-limit=4096",
        "--entrypoint=/bin/bash",
        "--workdir=/code",
        "-e",
        f"CUDA_VISIBLE_DEVICES={index}",
        "-e",
        f"PSRL_EXPECTED_GPU_UUID={gpu_uuid}",
        "-e",
        "LD_LIBRARY_PATH=/usr/local/cuda/compat:/host-driver:/usr/local/cuda/lib64",
        "-e",
        "PYTHONPATH=/code/python",
        "-e",
        "OMP_NUM_THREADS=4",
        "-v",
        f"{args.assets / 'code'}:/code:ro",
        "-v",
        f"{args.assets / 'hf-cache'}:/hf-cache:ro",
        "-v",
        f"{args.assets / 'upstream/tasks/ss-sgl22544-score-multi-item-delimiters/tests'}:/tests:ro",
        "-v",
        f"{args.output}:/logs:rw",
        "-v",
        "/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro",
    ]
    for device in [f"/dev/nvidia{minor}", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]:
        if Path(device).exists():
            base.extend(["--device", device])
    for library in ["libcuda", "libnvidia-ml", "libnvidia-ptxjitcompiler"]:
        path = next(
            (
                Path(directory) / f"{library}.so.{version}"
                for directory in ["/usr/lib64", "/usr/lib/x86_64-linux-gnu"]
                if (Path(directory) / f"{library}.so.{version}").is_file()
            ),
            None,
        )
        if path:
            base.extend(["-v", f"{path}:/host-driver/{library}.so.1:ro"])

    def stage(name, command, budget):
        container = f"{token}-{name}"
        started = time.time()
        result = dict(started=started, budget_s=budget)
        evidence["stages"][name] = result
        save()
        try:
            with (args.output / f"{name}.log").open("w") as output:
                process = subprocess.run(
                    [*base, "--name", container, args.image, "-c", command],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=budget,
                    check=False,
                )
            result["exit_code"] = process.returncode
        except subprocess.TimeoutExpired:
            result["exit_code"] = 124
        finally:
            cleanup = subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            result["cleanup_exit_code"] = cleanup.returncode
            result["elapsed_s"] = time.time() - started
            save()
        if result["exit_code"]:
            raise RuntimeError(f"Stage {name!r} failed with exit {result['exit_code']!r}.")

    cpu = """import json,time
t=time.monotonic()
import torch,transformers,pytest
assert not torch.cuda.is_initialized(), 'Preparation initialized CUDA.'
print(json.dumps(dict(import_s=time.monotonic()-t,torch=torch.__version__,cuda=torch.version.cuda,cuda_initialized=False)))
"""
    cuda = """import json,time,os
print('compat',os.path.exists('/usr/local/cuda/compat/libcuda.so.1'),flush=True)
t=time.monotonic()
import torch
print('torch',torch.__version__,'runtime',torch.version.cuda,flush=True)
x=torch.arange(8,device='cuda')
assert int((x*x).sum())==140, 'CUDA kernel returned incorrect output.'
torch.cuda.synchronize()
actual=str(torch.cuda.get_device_properties(0).uuid)
assert actual.removeprefix('GPU-')==os.environ['PSRL_EXPECTED_GPU_UUID'].removeprefix('GPU-'), 'GPU identity mismatch.'
print(json.dumps(dict(elapsed_s=time.monotonic()-t,device=torch.cuda.get_device_name(0),uuid=actual,allocated=torch.cuda.memory_allocated())))
"""
    try:
        stage("cpu-import", "python3 -I - <<'PY'\n" + cpu + "PY", 90)
        stage("cuda", "python3 -I - <<'PY'\n" + cuda + "PY", 90)
        stage("verifier", "bash /tests/test.sh", args.task_budget)
        reward = args.output / "verifier/reward.txt"
        evidence["reward"] = reward.read_text().strip() if reward.is_file() else None
        evidence["status"] = "passed" if evidence["reward"] in {"1", "1.0"} else "task_failed"
    except Exception as error:
        evidence.update(status="failed", error=str(error))
        psrl_logger.exception("Score service probe failed.")
    finally:
        evidence["remaining_containers"] = query(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.experiment={token}"]
        ).splitlines()
        evidence["final_gpu"] = query(
            ["nvidia-smi", f"--id={gpu_uuid}", "--query-gpu=uuid,memory.used", "--format=csv,noheader"]
        )
        save()
    if evidence["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
