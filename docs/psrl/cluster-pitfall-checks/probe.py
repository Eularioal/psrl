"""Reproduce small cluster failure mechanisms without starting training or Ray."""

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def command(args, *, timeout=12, **kwargs):
    start = time.monotonic()
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kwargs)
        return {
            "code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "seconds": round(time.monotonic() - start, 3),
        }
    except subprocess.TimeoutExpired:
        return {"timeout": timeout, "seconds": round(time.monotonic() - start, 3)}
    except OSError as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--env-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dependency-path", type=Path)
    args = parser.parse_args()
    repo = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    workspace = repo.parent
    started = time.monotonic()
    result = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "repo": str(repo), "python": sys.executable, "checks": {}}
    checks = result["checks"]
    checks["git_head"] = command(["git", "rev-parse", "HEAD"], cwd=repo)
    checks["git_status"] = command(["git", "status", "--short"], cwd=repo)
    checks["activation"] = command(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'export PSRL_WORKSPACE="$1"\n'
            'source "$2" >/dev/null 2>&1\n'
            'printf "expected=%s\\nactual=%s\\n" "$1" "$PSRL_WORKSPACE"',
            "probe",
            str(workspace),
            str(args.env_script),
        ],
        timeout=30,
    )
    import_code = (
        "import importlib.util,json; "
        "print(json.dumps({n:importlib.util.find_spec(n).origin "
        "for n in ('psrl','verl','transfer_queue')}))"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    checks["imports_unpinned"] = command([sys.executable, "-c", import_code], cwd="/tmp", env=env)
    env["PYTHONPATH"] = str(repo)
    checks["imports_pinned"] = command([sys.executable, "-c", import_code], cwd="/tmp", env=env)
    checks["route"] = command(["ip", "-4", "route", "get", "1.1.1.1"])
    addresses = command(["hostname", "-I"])
    checks["hostname_addresses"] = {
        "first": addresses.get("stdout", "").split()[:1],
        "count": len(addresses.get("stdout", "").split()),
    }
    checks["docker_info"] = command(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=6)
    checks["docker_ping"] = command(
        [
            "curl",
            "--silent",
            "--show-error",
            "--max-time",
            "3",
            "--unix-socket",
            "/var/run/docker.sock",
            "http://localhost/_ping",
        ],
        timeout=5,
    )
    processes = command(["ps", "-eo", "pid,ppid,stat,comm"])
    checks["docker_processes"] = [
        line for line in processes.get("stdout", "").splitlines() if "dockerd" in line or "containerd" in line
    ]
    inventory = command(["nvidia-smi", "-q", "-x"], timeout=8)
    if inventory.get("code") == 0:
        checks["gpu_mapping"] = [
            {"index": i, "minor": int(gpu.findtext("minor_number")), "uuid": gpu.findtext("uuid")}
            for i, gpu in enumerate(ET.fromstring(inventory["stdout"]).findall("gpu"))
        ]
    else:
        checks["gpu_mapping"] = inventory
    checks["gpu_usage"] = command(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        timeout=8,
    )
    with tempfile.TemporaryDirectory(prefix="psrl-probe-", dir="/tmp") as scratch:
        long_name = str(Path(scratch) / ("x" * 110))
        with socket.socket(socket.AF_UNIX) as sock:
            try:
                sock.bind(long_name)
                checks["long_unix_socket"] = {"unexpected_success": True}
            except OSError as exc:
                checks["long_unix_socket"] = {"bytes": len(long_name.encode()), "error": str(exc)}
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(Path(scratch) / "short.sock"))
            checks["short_unix_socket"] = "bound successfully"

        import pyarrow as pa
        import pyarrow.parquet as pq

        bad_path = Path(scratch) / "bad.parquet"
        good_path = Path(scratch) / "good.parquet"
        pq.write_table(pa.Table.from_pylist([{"extra_info": '{"index": 7}'}]), bad_path)
        pq.write_table(pa.Table.from_pylist([{"extra_info": {"index": 7}}]), good_path)
        try:
            pq.read_table(bad_path).to_pylist()[0].get("extra_info", {}).get("index", 0)
        except AttributeError as exc:
            checks["json_string_parquet"] = {
                "error": str(exc),
                "fixed_index": pq.read_table(good_path).to_pylist()[0]["extra_info"]["index"],
            }

    from omegaconf import OmegaConf

    runtime = OmegaConf.create({"env_vars": {"RAY_ADDRESS": "127.0.0.1:12345"}})
    OmegaConf.set_struct(runtime, True)
    try:
        runtime["env_vars"]["TRANSFER_QUEUE_ENABLE"] = "1"
    except Exception as exc:
        checks["structured_config"] = {"error_type": type(exc).__name__, "error": str(exc)}
    plain = OmegaConf.to_container(runtime)
    plain["env_vars"]["TRANSFER_QUEUE_ENABLE"] = "1"
    checks["structured_config_fixed"] = plain

    checks["model_configs"] = []
    for config in sorted((workspace / "psrl_demo/models").glob("*/config.json")):
        data = json.loads(config.read_text())
        inner = data.get("text_config", data)
        checks["model_configs"].append(
            {
                "path": str(config),
                "model_type": data.get("model_type"),
                "max_position_embeddings": inner.get("max_position_embeddings"),
                "rope_scaling": inner.get("rope_scaling"),
            }
        )
    checks["gsm8k_schema"] = []
    for dataset in sorted((workspace / "psrl_demo/data/gsm8k").glob("*.parquet")):
        schema = pq.read_schema(dataset)
        checks["gsm8k_schema"].append({"path": str(dataset), "schema": str(schema)})

    sys.path.insert(0, str(repo))
    if args.dependency_path:
        sys.path.insert(1, str(args.dependency_path.resolve()))
    checks["tensor_dependency_origins"] = {
        name: importlib.util.find_spec(name).origin for name in ("verl", "transfer_queue")
    }
    import torch
    from tensordict import TensorDict
    from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager
    from verl.utils.tensordict_utils import index_select_tensor_dict, maybe_fix_3d_position_ids

    rows = [torch.arange(4 * 2407).reshape(4, 2407)]
    packed = AsyncSimpleStorageManager._pack_field_values(rows)
    checks["position_ids_before"] = {"ragged_idx": packed._ragged_idx, "offsets": packed.offsets().tolist()}
    broken = AsyncSimpleStorageManager._pack_field_values(rows)
    broken._ragged_idx = 2
    try:
        broken.unbind()
        checks["position_ids_old_mechanism"] = "unexpected success"
    except RuntimeError as exc:
        checks["position_ids_old_mechanism"] = str(exc)
    batch = TensorDict({"position_ids": packed}, batch_size=[1])
    try:
        maybe_fix_3d_position_ids(batch)
        maybe_fix_3d_position_ids(batch)
        selected = index_select_tensor_dict(batch, [0])["position_ids"]
        checks["position_ids_current"] = {
            "offsets": selected.offsets().tolist(),
            "ragged_idx": selected._ragged_idx,
            "equal": torch.equal(selected.unbind()[0], rows[0]),
        }
    except RuntimeError as exc:
        checks["position_ids_current"] = {"error": str(exc)}
    result["seconds"] = round(time.monotonic() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
