"""Build a PSRL dataset and agent configurations from explicit prepared tasks."""

import argparse
import json
from pathlib import Path

import pandas as pd
import tomllib
import yaml


def prepare(manifest: Path, output: Path):
    """Keep task protocols and container backends explicit in a mixed batch."""
    records = json.loads(manifest.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("The manifest must contain at least one prepared task.")
    rows = []
    backends = set()
    for item in records:
        task = Path(item["task_path"]).resolve()
        config = tomllib.loads((task / "task.toml").read_text())
        provenance = json.loads((task / "provenance.json").read_text())
        backend = item["environment_backend"]
        if backend not in ("docker", "env_worker"):
            raise ValueError("Select a managed Docker or native Env backend.")
        if backend == "env_worker" and list(task.rglob("docker-compose.yaml")):
            raise ValueError("Native tasks must not depend on Compose definitions.")
        source = item["data_source"]
        if "smoke" in source or provenance.get("mode") in ("smoke", "feedback-smoke"):
            raise ValueError("Smoke tasks cannot demonstrate benchmark policy training.")
        benchmark = provenance.get("benchmark")
        if benchmark == "re_bench" and provenance.get("judge", {}).get("backend") == "local":
            benchmark = "re_bench_local"
        if benchmark is not None and source != benchmark:
            raise ValueError("The data source must match the prepared benchmark scoring identity.")
        memory = int(item["gpu_memory_mb"])
        if memory <= 0 or config["environment"].get("gpus", 0) != 1:
            raise ValueError("Each local GPU task must declare one GPU and a positive memory demand.")
        rows.append(
            {
                "data_source": source,
                "agent_name": f"gpu_{backend}",
                "prompt": [{"role": "user", "content": (task / "instruction.md").read_text()}],
                "ability": "gpu-engineering",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "task_path": str(task),
                    "gpus": 1,
                    "gpu_memory_mb": memory,
                    "reward_key": "reward",
                    "benchmark": source,
                },
            }
        )
        backends.add(backend)
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_parquet(output / "tasks.parquet", index=False)
    configs = [
        {
            "name": f"gpu_{backend}",
            "_target_": "examples.gpu_benchmarks.agent_loop.GpuBenchmarkAgentLoop",
            "harbor": {
                "environment_backend": backend,
                "agent_name": "examples.sciaccel_rl.agent:TruncatingTerminus2",
                "jobs_dir": str((output / "jobs").resolve()),
                "max_concurrent_episodes": 1,
            },
            "task_timeout_sec": 900,
            "verifier_timeout_sec": 900,
        }
        for backend in sorted(backends)
    ]
    (output / "agent.yaml").write_text(yaml.safe_dump(configs))
    (output / "manifest.json").write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.output)


if __name__ == "__main__":
    main()
