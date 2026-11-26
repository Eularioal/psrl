"""Compare gated and overlapping retirement with the same live-load observation."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from capacity_validation_pipeline import save
from live_partition_study import summarize_partition
from partition_followup import run_prepared, verify_release

OVERLAYS = (
    "psrl/workers/gen/gpu_handoff.py",
    "psrl/workers/gen/rollout_coordination/env_partition.py",
    "psrl/workers/gen/vllm_async_server.py",
    "psrl/workers/env_service/command_gpu.py",
    "psrl/workers/env_service/router.py",
    "psrl/trainer/config/psrl/env_service.yaml",
    "integrated_partition_training.py",
)


def prepare(reference, output, repo, overlap, local_site):
    """Freeze only the reviewed changes on the completed startup reference."""
    manifest = json.loads((reference / "source-manifest.json").read_text())
    state = json.loads((reference / "supervisor.json").read_text())
    if state["status"] != "complete" or not state.get("physical_release_verified"):
        raise ValueError("Reference did not complete and release its resources.")
    for name, digest in manifest.items():
        if hashlib.sha256((reference / "source" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Reference changed: {name}.")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    shutil.copytree(reference / "source", source, ignore=shutil.ignore_patterns("__pycache__", "outputs"))
    shutil.copy2(reference / "tasks.parquet", output / "tasks.parquet")
    for name in OVERLAYS:
        if name == "integrated_partition_training.py":
            runner = Path(__file__).with_name(name).read_text()
            (source / name).write_text(runner.replace("model_headroom_mb=8192", "model_headroom_mb=16384"))
        else:
            shutil.copy2(repo / name, source / name)
    config = source / "psrl/trainer/config/psrl/env_service.yaml"
    config.write_text(
        config.read_text()
        .replace("handover_overlap: false", f"handover_overlap: {str(overlap).lower()}")
        .replace("gpu_identity_backend: procfs", "gpu_identity_backend: ebpf")
    )
    design = json.loads((reference / "design.json").read_text())
    design["args"].update(output=str(output), local_python_site=str(local_site))
    design.update(
        reference=str(reference),
        comparison="Compare the transition gate with and without concurrent model retirement.",
        handover_overlap=overlap,
        live_frontend_load=True,
        gpu_identity_backend="ebpf",
        startup_preflight="Check module origins and Hydra composition, then load business dependencies in the job.",
    )
    design["limits"].update(arm_s=2700, initialization_s=1200, episode_s=720, attempts_per_arm=1)
    save(output / "design.json", design)
    frozen = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    changed = sorted(name for name in frozen if frozen[name] != manifest.get(name))
    if set(changed) - set(OVERLAYS):
        raise ValueError(f"Unexpected changes: {changed}.")
    save(output / "source-manifest.json", frozen)
    save(output / "reference-audit.json", {"changed": changed, "data_unchanged": True, "handover_overlap": overlap})
    return source


def main(args):
    """Prepare both arms before starting and stop on any failure or unreleased GPU."""
    sources = {}
    site_manifest = json.loads(args.local_site_manifest.read_text())
    local_site = Path(site_manifest["local"])
    for name, expected in site_manifest["sha256"].items():
        if hashlib.sha256((local_site / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Node local dependency changed at {name!r}.")
    for label, overlap in (("gated", False), ("overlap", True)):
        sources[label] = prepare(args.reference, args.output / label, args.repo, overlap, local_site)
    # The treatment switch must be the only difference between the two arms.
    left = json.loads((args.output / "gated/source-manifest.json").read_text())
    right = json.loads((args.output / "overlap/source-manifest.json").read_text())
    differences = [name for name in left if left[name] != right[name]]
    if differences != ["psrl/trainer/config/psrl/env_service.yaml"]:
        raise ValueError(f"Unpaired sources: {differences}.")
    save(
        args.output / "protocol.json",
        {
            "arms": ["gated", "overlap"],
            "only_source_difference": differences,
            "per_arm": {
                "steps": 2,
                "groups_per_step": 6,
                "trajectories_per_group": 2,
                "trajectories": 24,
                "gpu_commands": 48,
                "model_requests": 72,
            },
            "budget_per_arm_s": 2700,
            "attempts_per_arm": 1,
            "gpu_identity_backend": "ebpf",
            "local_site_manifest": str(args.local_site_manifest),
            "purpose": "Test physical overlap and end-to-end latency, not policy learning quality.",
        },
    )
    for label, source in sources.items():
        output = args.output / label
        if not verify_release(source, output):
            raise RuntimeError("Selected devices are occupied; refusing to start.")
        run_prepared(
            argparse.Namespace(
                output=output, mode="resident", budget_s=2700, rollout_replicas=2, local_python_site=local_site
            ),
            source,
        )
        summarize_partition(output, "resident")
        state = json.loads((output / "supervisor.json").read_text())
        if state["status"] != "complete" or not state.get("physical_release_verified"):
            raise RuntimeError(f"Arm {label} failed; no automatic retry or following arm.")
    save(args.output / "status.json", {"status": "complete", "arms": list(sources)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "output", "repo", "local-site-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    main(parser.parse_args())
