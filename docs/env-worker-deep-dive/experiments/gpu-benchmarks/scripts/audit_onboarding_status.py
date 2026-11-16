"""Summarize saved GPU evidence and inspect dependency metadata without running tasks."""

import argparse
import datetime
import hashlib
import json
import re
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(evidence):
    gpu = evidence.get("gpu_after", {})
    return (
        not evidence.get("cleanup_error")
        and not evidence.get("containers_remaining")
        and not gpu.get("pids", ["unknown"])
        and gpu.get("used_mb", float("inf")) < 128
    )


def trial(path, repo):
    evidence = load(path)
    result = evidence.get("result", {})
    return {
        "task": result.get("task_name", path.parent.name),
        "file": str(path.relative_to(repo.parent)),
        "sha256": digest(path),
        "runner_passed": evidence.get("passed", False),
        "physical_cleanup": clean(evidence),
        "gpu_observed": any(x.get("gpu", {}).get("pids") for x in evidence.get("samples", [])),
        "metrics": result.get("rewards", {}),
        "error": result.get("exception_type") or bool(evidence.get("error")),
    }


def accepted(row):
    metrics = row["metrics"]
    return row["runner_passed"] and row["physical_cleanup"] and not row["error"] and metrics.get("eval_valid", 1) == 1


def policy(root, repo):
    progress_path = root / "policy-functional-progress.json"
    progress = load(progress_path)
    text = (root / "policy-training/train.log").read_text(errors="replace")
    step = next(line for line in text.splitlines() if re.search(r"\bstep:1 - ", line))
    metrics = {}
    for name in ("actor/loss", "actor/grad_norm", "termination/trained_tokens", "training/global_step"):
        match = re.search(re.escape(name) + r":(?:np\.float64\()?([-+0-9.e]+)", step)
        metrics[name] = float(match.group(1)) if match else None
    launch = load(root / "policy-training/launch.json")
    return {
        "file": str(progress_path.relative_to(repo.parent)),
        "sha256": digest(progress_path),
        "passed": progress["passed"],
        "coverage": progress["coverage"],
        "optimizer_step": progress["optimizer_step"],
        "version_published": progress["version_published"],
        "metrics": metrics,
        "stop": launch.get("controlled_stop") or launch.get("error") or launch.get("timeout"),
        "exit": launch["exit"],
        "physical_cleanup": load(root / "policy-cleanup.json")["passed"],
        "natural_finish_count": progress.get("natural_finish_count"),
        "budget_stop_count": progress.get("budget_stop_count"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[5]
    runs = repo.parent / "benchmark-runs"
    old = runs / "heterogeneous-onboarding-20261112"
    new = runs / "complete-onboarding-20261112"

    def rows(root, directories):
        paths = []
        for directory in directories:
            target = root / directory
            paths += list(target.glob("*/evidence.json"))
            if (target / "evidence.json").is_file():
                paths.append(target / "evidence.json")
        return [trial(path, repo) for path in sorted(paths)]

    post = rows(new, ["posttrain-verify"])
    nlp = rows(new, ["mlgym-nlp-verify"])
    fast_rows = rows(
        new, ["fast-verify-v2", "fast-verify-v3", "fast-conv-probe", "fast-verify-clip", "fast-verify-clip-fixed"]
    )
    fast = {}
    for row in fast_rows:
        m = row["metrics"]
        if accepted(row) and m.get("cuda") == 1 and m.get("cases", 0) > 0 and m.get("skipped") == 0:
            fast[row["task"]] = row
    original_rows = rows(old, ["swe-sweep", "verify-swe-score-v3"]) + rows(new, ["swe-sweep"])
    originals = {}
    for row in original_rows:
        if accepted(row) and row["metrics"].get("resolved") == 1 and row["metrics"].get("reward") == 1:
            originals[row["task"]] = row
    local_rows = rows(new, ["swe-quick-verify", "swe-bounded-verify", "swe-patch-verify"])
    locals_ = {}
    for row in local_rows:
        if accepted(row) and row["metrics"].get("local_evaluation") == 1 and row["metrics"].get("reward") == 1:
            locals_[row["task"]] = row
    gpu_swe = {r["task"] for r in [*originals.values(), *locals_.values()] if r["gpu_observed"]}
    site = args.main_prefix / "lib/python3.12/site-packages"
    cutoff = datetime.datetime(2026, 11, 9).timestamp()
    metadata = sorted(p for p in site.glob("*.dist-info/*") if p.is_file())
    recent = [
        str(p.relative_to(args.main_prefix)) for p in metadata if max(p.stat().st_mtime, p.stat().st_ctime) >= cutoff
    ]
    metadata_hash = hashlib.sha256()
    for path in metadata:
        metadata_hash.update(str(path.relative_to(args.main_prefix)).encode())
        metadata_hash.update(path.read_bytes())
    categories = {
        old: ["mlgym-rl-v3", "mlgym-vision"],
        new: [
            "mlgym-nlp-tasks",
            "posttrain-tasks",
            "fast-tasks-v2",
            "fast-tasks-v3",
            "fast-tasks-clip-fixed",
            "swe-all/tasks",
        ],
    }
    prefixes = [str(args.main_prefix), str(repo.parent / "benchmark-assets/mlgym-rl-env")]
    mounts = []
    for root, directories in categories.items():
        for directory in directories:
            for path in (root / directory).glob("*/*/docker-compose.yaml"):
                lines = [line.strip()[2:] for line in path.read_text().splitlines() if line.strip().startswith("- ")]
                env_mounts = [line for line in lines if any(line.startswith(prefix + ":") for prefix in prefixes)]
                if env_mounts:
                    mounts.append(
                        {
                            "file": str(path.relative_to(repo.parent)),
                            "environment_mounts": len(env_mounts),
                            "all_readonly": all(line.endswith(":ro") for line in env_mounts),
                        }
                    )
    status = {
        "verified_at": datetime.datetime.now().astimezone().isoformat(),
        "basis": "6aa328d56eff746c2ae6d9fd766b2f3f76b4a670 plus uncommitted working directory changes",
        "scope": (
            "Saved bounded functional experiments. Local protocols, "
            "not official benchmark reproduction or learning-gain evidence."
        ),
        "counts": {
            "base_tasks": 495,
            "mlgym_local_tasks": 9,
            "posttrain_upstream_combinations": 24,
            "posttrain_local_verified": sum(accepted(r) and r["metrics"].get("cuda") == 1 for r in post),
            "mlgym_nlp_local_verified": sum(accepted(r) and r["metrics"].get("cuda") == 1 for r in nlp),
            "fastkernels_upstream_targets": 458,
            "fastkernels_local_verified": len(fast),
            "fastkernels_remaining_unverified": 458 - len(fast),
            "swe_imported_gpu_tasks": 41,
            "swe_original_test_passes": len(originals),
            "swe_original_passes_gpu_observed": sum(r["gpu_observed"] for r in originals.values()),
            "swe_local_test_passes": len(locals_),
            "swe_local_passes_gpu_observed": sum(r["gpu_observed"] for r in locals_.values()),
            "swe_original_or_local_passes": len(set(originals) | set(locals_)),
            "swe_original_or_local_gpu_observed": len(gpu_swe),
            "swe_local_policy_tasks": 1,
        },
        "posttrain": post,
        "mlgym_nlp": nlp,
        "fastkernels": sorted(fast.values(), key=lambda r: r["task"]),
        "swe_original": sorted(originals.values(), key=lambda r: r["task"]),
        "swe_local": sorted(locals_.values(), key=lambda r: r["task"]),
        "swe_attempts_without_complete_pass": [
            r for r in local_rows if not (accepted(r) and r["metrics"].get("reward") == 1)
        ],
        "policy_runs": {"eleven_tasks": policy(new, repo), "two_task_completion": policy(new / "final-policy", repo)},
        "environment_isolation": {
            "metadata_files_checked": len(metadata),
            "metadata_files_changed_since_20261109": recent,
            "current_metadata_sha256": metadata_hash.hexdigest(),
            "conda_history_modified_at": datetime.datetime.fromtimestamp(
                (args.main_prefix / "conda-meta/history").stat().st_mtime
            ).isoformat(),
            "has_before_snapshot": False,
            "mount_definitions_checked": len(mounts),
            "readonly_environment_mounts": all(m["all_readonly"] for m in mounts),
            "mounts": mounts,
            "jax_uses_system_site_packages": "include-system-site-packages = true"
            in (repo.parent / "benchmark-assets/mlgym-rl-env/pyvenv.cfg").read_text(),
            "installation_targets": [
                "benchmark-specific Docker image layers",
                "benchmark-assets/mlgym-rl-env",
                "benchmark-runs/full-onboarding-20261110/deps",
                ".verification-deps",
            ],
            "download_targets": [
                "benchmark-assets/swe-wheels",
                "benchmark-assets/fastkernels-wheels",
                "benchmark-assets/swe-complete-hf",
                "benchmark-assets/posttrain-local",
                "benchmark-assets/mlgym-nlp",
            ],
            "interpretation": (
                "No recent package-metadata installation writes observed. "
                "Host Python is reused read-only by several tasks. "
                "This is not a historical byte-for-byte integrity proof or a VM security boundary."
            ),
        },
        "resource_release": {
            "file": "benchmark-runs/complete-onboarding-20261112/resource-audit.json",
            "sha256": digest(new / "resource-audit.json"),
            "docker_ps_empty": not load(new / "resource-audit.json")["containers"]["output"],
            "gpu_mib": [
                int(line.rsplit(",", 1)[-1])
                for line in load(new / "resource-audit.json")["gpu"]["output"].splitlines()
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "counts": status["counts"],
                "metadata_files_checked": len(metadata),
                "recent_metadata_files": len(recent),
                "mounts_checked": len(mounts),
                "readonly_mounts": status["environment_isolation"]["readonly_environment_mounts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
