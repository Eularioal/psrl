"""Compare conservative and resident-only synchronization in frozen PSRL experiments."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def patch_gate(text):
    """Restrict experimental overlap to validated resident synchronization operations."""
    text = text.replace(
        "async def operation(self, borrower=None, sleeping=False):",
        "async def operation(self, borrower=None, sleeping=False, resident_sync=False):",
    )
    text = text.replace(
        "and not self.shared_commands",
        """and (
                                not self.shared_commands
                                or (
                                    resident_sync
                                    and self.borrower is None
                                    and self.reservation is None
                                    and all(mode == "resident" for mode in self.shared_commands.values())
                                )
                            )""",
    )
    text = text.replace(
        'async with self._gpu_control.operation(sleeping=method.__name__ in ("sleep", "nixl_sleep")) as waited:',
        """overlap = (
            self.psrl_config.env_service.gpu_sharing.get("sync_overlap", False)
            and self.psrl_config.ps_mode == "nixl_cpu"
            and self._gpu_weights_valid
            and method.__name__ in ("pause_for_sync", "pull_model_for_sync", "resume_after_sync", "fail_sync")
        )
        async with self._gpu_control.operation(
            sleeping=method.__name__ in ("sleep", "nixl_sleep"), resident_sync=overlap
        ) as waited:
            shared_at_entry = dict(self._gpu_control.shared_commands)""",
    )
    text = text.replace('"gate_wait_s": waited,', '"gate_wait_s": waited, "shared_at_entry": shared_at_entry,')
    if "resident_sync=overlap" not in text:
        raise ValueError("The frozen gate does not match the reviewed experiment source.")
    return text


def make_runner(root, enabled):
    """Create a reproducible experiment wrapper without editing production modules."""
    scripts = Path(__file__).parent
    runner = root / ("overlap-runner" if enabled else "wait-runner")
    runner.mkdir()
    for name in (
        "integrated_partition_training.py",
        "capacity_validation_pipeline.py",
        "analyze_integrated_partition.py",
    ):
        shutil.copy2(scripts / name, runner / name)
    shutil.copy2(Path(__file__), runner / Path(__file__).name)
    shutil.copy2(scripts / "partition_followup.py", runner / "partition_followup.py")
    return runner


def main(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "arms": {}}
    for enabled, name in ((False, "wait"), (True, "overlap")):
        runner = make_runner(root, enabled)
        command = [
            sys.executable,
            "-u",
            str(runner / "partition_followup.py"),
            "--reference",
            str(args.reference.resolve()),
            "--output",
            str(root / name),
            "--rollout-replicas",
            "2",
            "--mode",
            "resident",
            "--sync-policy",
            "overlap" if enabled else "wait",
            "--headroom-mib",
            "16384",
        ]
        with (root / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status["arms"][name] = {"pid": child.pid}
            (root / "status.json").write_text(json.dumps(status, indent=2))
            try:
                child.wait(timeout=3300)
            except subprocess.TimeoutExpired:
                sys.path.insert(0, str(runner))
                from capacity_validation_pipeline import cleanup_processes

                owned = root / name / "owned-processes.json"
                if owned.exists():
                    cleanup_processes({int(k): v for k, v in json.loads(owned.read_text()).items()})
                child.terminate()
                child.wait(timeout=30)
        state_path = root / name / "supervisor.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "failed"}
        status["arms"][name].update(state)
        if state["status"] != "complete" or not state.get("physical_release_verified"):
            status["status"] = "failed"
            break
    else:
        status["status"] = "complete"
        summary = {name: json.loads((root / name / "analysis.json").read_text()) for name in ("wait", "overlap")}
        (root / "analysis.json").write_text(json.dumps(summary, indent=2))
    (root / "status.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
