"""Verify one Docker CUDA process against a temporary BPF collector and NVML."""

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

import pynvml as nvml


def command(*args, timeout=10):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tag = "psrl" + uuid.uuid4().hex[:11]
    name = "psrl-bpf-" + tag
    report = {"tag": tag, "container_name": name, "status": "running", "production_attribution_verified": False}
    collector, created, handle = None, False, None
    nvml.nvmlInit()
    try:
        command("docker", "info", "--format", "{{.ServerVersion}}")
        report["image_id"] = command("docker", "image", "inspect", args.image, "--format", "{{.Id}}")
        selected = None
        for index in range(nvml.nvmlDeviceGetCount()):
            candidate = nvml.nvmlDeviceGetHandleByIndex(index)
            if not nvml.nvmlDeviceGetComputeRunningProcesses(candidate):
                selected, handle = index, candidate
                break
        if selected is None:
            raise RuntimeError("No GPU without compute processes is available.")
        minor = nvml.nvmlDeviceGetMinorNumber(handle)
        report["gpu"] = {"index": selected, "minor": minor, "uuid": nvml.nvmlDeviceGetUUID(handle)}
        report["memory_before_bytes"] = nvml.nvmlDeviceGetMemoryInfo(handle).used
        driver = nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        libraries = {
            "libcuda.so.1": Path(f"/usr/lib64/libcuda.so.{driver}"),
            "libnvidia-ptxjitcompiler.so.1": Path(f"/usr/lib64/libnvidia-ptxjitcompiler.so.{driver}"),
        }
        if not all(path.is_file() for path in libraries.values()):
            raise RuntimeError("Matching driver libraries are unavailable.")
        create = [
            "docker",
            "create",
            "--name",
            name,
            "--label",
            f"psrl.identity.probe={tag}",
            "--network",
            "none",
            "--cpus",
            "1",
            "--memory",
            "512m",
            "--pids-limit",
            "32",
            "--device",
            f"/dev/nvidia{minor}",
            "--device",
            "/dev/nvidiactl",
            "--device",
            "/dev/nvidia-uvm",
            "--env",
            "LD_LIBRARY_PATH=/probe-driver",
            "--env",
            "CUDA_CACHE_DISABLE=1",
            "--mount",
            f"type=bind,src={out},dst=/probe-evidence",
            "--mount",
            f"type=bind,src={Path(__file__).with_name('bpf_cuda_identity.py').resolve()},dst=/probe.py,readonly",
        ]
        for target, source in libraries.items():
            create.extend(["--mount", f"type=bind,src={source},dst=/probe-driver/{target},readonly"])
        create.extend([args.image, "timeout", "45", "python3", "/probe.py", tag, "/probe-evidence"])
        report["container_id"] = command(*create)
        created = True
        (out / "owned-container.txt").write_text(name)
        command("docker", "start", name)
        deadline = time.monotonic() + 20
        while not (out / "cuda-ready.json").exists():
            if command("docker", "inspect", name, "--format", "{{.State.Running}}") != "true":
                raise RuntimeError("The CUDA container exited before readiness. Inspect its saved log.")
            if time.monotonic() >= deadline:
                raise TimeoutError("The CUDA container did not become ready.")
            time.sleep(0.1)
        report["cuda"] = json.loads((out / "cuda-ready.json").read_text())
        report["outer_container_init_pid"] = int(command("docker", "inspect", name, "--format", "{{.State.Pid}}"))
        # The collector records only our marker and detaches after the first sample.
        with (out / "collector-stderr.log").open("w") as errors:
            collector = subprocess.Popen([args.binary, tag], stdout=subprocess.PIPE, stderr=errors, text=True)
            if collector.stdout.readline().strip() != "READY":
                raise RuntimeError("The BPF collector could not attach.")
            (out / "begin").touch()
            stdout, _ = collector.communicate(timeout=23)
            if collector.returncode != 0:
                raise RuntimeError("The BPF collector failed to capture the process.")
        report["bpf"] = json.loads(stdout)
        report["docker_top"] = command("docker", "top", name, "-eo", "pid,comm")
        host_pid = report["bpf"]["host_tgid"]
        samples = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
        report["nvml_processes"] = [{"pid": p.pid, "used_gpu_memory": p.usedGpuMemory} for p in samples]
        matching = [p for p in samples if p.pid == host_pid]
        if len(matching) != 1 or matching[0].usedGpuMemory < report["cuda"]["allocation_bytes"]:
            raise RuntimeError("NVML did not match the captured host PID and CUDA allocation.")
        report["host_nvml_pid_mapping_verified"] = True
        (out / "finish").touch()
        report["container_exit_code"] = int(command("docker", "wait", name, timeout=10))
        if report["container_exit_code"] != 0:
            raise RuntimeError("The CUDA workload did not exit successfully.")
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", repr(exc)
    finally:
        if collector is not None:
            if collector.poll() is None:
                collector.kill()
            collector.wait(timeout=3)
            report["collector_reaped"] = True
        if created:
            logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True, timeout=8)
            (out / "container.log").write_text(logs.stdout + logs.stderr)
            command("docker", "rm", "-f", name)
            remaining = command("docker", "ps", "-aq", "--filter", f"label=psrl.identity.probe={tag}")
            report["container_removed"] = not remaining
        if handle is not None:
            host_pid = report.get("bpf", {}).get("host_tgid")
            report["release_samples"] = []
            for _ in range(20):
                pids = [p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)]
                memory = nvml.nvmlDeviceGetMemoryInfo(handle).used
                report["release_samples"].append({"pids": pids, "memory_bytes": memory})
                if host_pid not in pids and memory <= report["memory_before_bytes"]:
                    report["gpu_release_verified"] = True
                    break
                time.sleep(0.25)
            if not report.get("gpu_release_verified"):
                report["status"] = "cleanup_unverified"
        nvml.nvmlShutdown()
        (out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
