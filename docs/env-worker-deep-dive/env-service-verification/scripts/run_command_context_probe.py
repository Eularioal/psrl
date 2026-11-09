"""Run a bounded, sequential Docker cache characterization on two idle GPUs."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def run(command, timeout=15):
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=True).stdout


def inventory(path):
    return {
        str(p.relative_to(path)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in path.rglob("*")
        if p.is_file()
    }


def devices():
    root = ET.fromstring(run(["nvidia-smi", "-q", "-x"]))
    return [
        {
            "uuid": g.findtext("uuid"),
            "minor": g.findtext("minor_number"),
            "memory_used": g.findtext("fb_memory_usage/used"),
            "processes": [p.findtext("pid") for p in g.findall("processes/process_info")],
        }
        for g in root.findall("gpu")
    ]


def require_idle(selected):
    snapshot = devices()
    for gpu in selected:
        state = next(d for d in snapshot if d["uuid"] == gpu["uuid"])
        if state["processes"] or int(state["memory_used"].split()[0]) > 64:
            raise RuntimeError(f"Selected GPU is occupied: {state!r}.")
    return snapshot


def prepare(root):
    assets = root / "assets"
    relative = Path("sglang/jit_kernel/triton/gdn_fused_proj.py")
    original = assets / "gdn_fused_proj.base.py"
    blob = run(["git", "hash-object", str(original)]).strip()
    if not blob.startswith("d7d07da739"):
        raise RuntimeError(f"Unexpected upstream source blob: {blob!r}.")
    base = root / "versions/base"
    destination = base / relative
    destination.parent.mkdir(parents=True)
    shutil.copy2(original, destination)
    patch = assets / "task/solution/changes.patch"
    run(
        ["git", "apply", "--unsafe-paths", "-p2", "--directory", str(base), "--include", str(destination), str(patch)],
    )
    source = destination.read_text()
    if "def fused_qkv_split_gdn_prefill(" not in source:
        raise RuntimeError("The oracle kernel patch was not applied.")
    variants = {
        "unrelated": source + "\n\ndef unused_host_helper():\n    return 1\n",
        "kernel_edit": source.replace("offsets = tl.arange(0, BLOCK_SIZE)", "offsets = tl.arange(0, BLOCK_SIZE) + 0"),
    }
    if variants["kernel_edit"] == source:
        raise RuntimeError("The controlled kernel edit was not applied.")
    for name, text in variants.items():
        target = root / "versions" / name / relative
        target.parent.mkdir(parents=True)
        target.write_text(text)
    return {
        name: hashlib.sha256((root / "versions" / name / relative).read_bytes()).hexdigest()
        for name in ("base", "unrelated", "kernel_edit")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python-env", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--image", default="qwt-gpu-probe:latest")
    parser.add_argument("--budget-s", type=int, default=900)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    token = f"psrl-context-{os.getpid()}"
    cache_root = Path("/tmp") / token
    cache_root.mkdir()
    result = {"status": "running", "runs": [], "label": token, "cache_root": str(cache_root)}

    def save():
        temporary = root / "summary.tmp"
        temporary.write_text(json.dumps(result, indent=2))
        temporary.replace(root / "summary.json")

    def docker_base(gpu):
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            token,
            "--label",
            f"psrl.context_probe={token}",
            "--network",
            "none",
            "--cpus",
            "4",
            "--memory",
            "8g",
            "--pids-limit",
            "256",
            "--device",
            f"/dev/nvidia{gpu['minor']}",
            "--device",
            "/dev/nvidiactl",
            "--device",
            "/dev/nvidia-uvm",
            "--device",
            "/dev/nvidia-uvm-tools",
            "-v",
            f"{args.python_env}:{args.python_env}:ro",
            "-v",
            f"{root}:/work",
            "-v",
            f"{cache_root}:/cache",
            "-e",
            f"CUDA_VISIBLE_DEVICES={gpu['uuid']}",
            "-e",
            f"PATH={args.python_env}/bin:/usr/local/bin:/usr/bin:/bin",
            "-e",
            "LD_LIBRARY_PATH=/usr/lib64",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        ]
        for name in ("libcuda", "libnvidia-ml", "libnvidia-ptxjitcompiler"):
            source = Path("/usr/lib64") / f"{name}.so.{result['driver_version']}"
            command += ["-v", f"{source.resolve()}:/usr/lib64/{name}.so.1:ro"]
            if name == "libcuda":
                command += ["-v", f"{source.resolve()}:/usr/lib64/libcuda.so:ro"]
        return command

    try:
        result["source_hashes"] = prepare(root)
        result["docker_version"] = run(["docker", "info", "--format", "{{.ServerVersion}}"])
        result["image_id"] = run(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"])
        result["driver_version"] = run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ).splitlines()[0]
        all_devices = devices()
        selected = [all_devices[i] for i in args.gpus]
        result["selected"] = selected
        result["preflight"] = require_idle(selected)
        plan = [
            ("a_cold_original", 0, "a", "base"),
            ("a_warm_same", 0, "a", "base"),
            ("a_warm_other", 1, "a", "base"),
            ("a_cold_other", 1, "a_fresh", "base"),
            ("b_cold_original", 1, "b", "base"),
            ("b_warm_same", 1, "b", "base"),
            ("b_warm_other", 0, "b", "base"),
            ("b_cold_other", 0, "b_fresh", "base"),
            ("unrelated_edit", 0, "a", "unrelated"),
            ("kernel_edit", 1, "a", "kernel_edit"),
            ("kernel_edit_warm_other", 0, "a", "kernel_edit"),
        ]
        result["plan"] = plan
        save()
        for name, gpu_id, cache_name, version in plan:
            remaining = args.budget_s - (time.monotonic() - started)
            if remaining < 20:
                raise TimeoutError("Characterization budget exhausted.")
            require_idle(selected)
            cache = cache_root / cache_name
            cache.mkdir(exist_ok=True)
            before = inventory(cache)
            command = docker_base(selected[gpu_id])
            command += [
                "-e",
                f"TRITON_CACHE_DIR=/cache/{cache_name}/triton",
                "-e",
                f"CUDA_CACHE_PATH=/cache/{cache_name}/cuda",
                "-e",
                f"XDG_CACHE_HOME=/cache/{cache_name}/xdg",
                "--entrypoint",
                str(args.python_env / "bin/python"),
                args.image,
                "/work/command_context_probe.py",
                "--source",
                f"/work/versions/{version}",
                "--tests",
                "/work/assets/task/tests/postmerge_tests/test/registered/kernels",
                "--output",
                f"/work/{name}.json",
            ]
            print(f"Starting {name} on {selected[gpu_id]['uuid']}.", flush=True)
            begin = time.monotonic()
            with (root / f"{name}.log").open("w") as stream:
                process = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, timeout=min(120, remaining), check=False
                )
            elapsed = time.monotonic() - begin
            cleaned = require_idle(selected)
            after = inventory(cache)
            entry = {
                "name": name,
                "gpu": selected[gpu_id]["uuid"],
                "cache": cache_name,
                "version": version,
                "docker_wall_s": elapsed,
                "exit_code": process.returncode,
                "after_release": cleaned,
                "cache_before": before,
                "cache_after": after,
            }
            output = root / f"{name}.json"
            if output.exists():
                entry["probe"] = json.loads(output.read_text())
            result["runs"].append(entry)
            save()
            if process.returncode:
                raise RuntimeError(f"Probe {name!r} failed with exit code {process.returncode!r}.")
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        owned = run(["docker", "ps", "-aq", "--filter", f"label=psrl.context_probe={token}"]).split()
        if owned:
            run(["docker", "rm", "-f", *owned], timeout=30)
        result["remaining_containers"] = run(
            ["docker", "ps", "-aq", "--filter", f"label=psrl.context_probe={token}"]
        ).split()
        result["final_devices"] = devices()
        result["elapsed_s"] = time.monotonic() - started
        save()


if __name__ == "__main__":
    main()
