"""Build bounded local runtime images for explicitly selected GPU tasks."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from examples.gpu_benchmarks.prepare import provenance


def local_recipe(text, base_image, wheelhouse=False, omit_jit_cache=False):
    lines = []
    for line in text.replace("\\\n", " ").splitlines():
        if line.startswith("FROM "):
            line = f"FROM {base_image}"
            if wheelhouse:
                line += "\nCOPY wheels /opt/psrl-wheels\nENV PIP_NO_INDEX=1 PIP_FIND_LINKS=/opt/psrl-wheels"
        elif line.startswith("RUN git clone "):
            line = "RUN rm -rf /code\nCOPY code /code"
        elif wheelhouse and line.startswith("COPY --from=ghcr.io/astral-sh/uv:"):
            continue
        if wheelhouse:
            line = re.sub(r"uv pip\s+install", "python3 -m pip install --break-system-packages --no-deps", line)
            line = line.replace(" --system", "").replace(" --no-cache ", " --no-cache-dir ")
            line = re.sub(r" --index-url \S+", "", line)
            line = line.replace(
                "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*",
                "RUN command -v git",
            )
        if omit_jit_cache:
            line = re.sub(
                r" &&\s+python3 -m pip install --break-system-packages --no-deps\s+flashinfer-jit-cache==[\w.]+",
                "",
                line,
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


def build(repo, output, base_image, budget, selected_tasks=None, wheelhouse=None, omit_jit_cache=False):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    revision = provenance(repo)
    if wheelhouse is not None:
        wheelhouse = Path(wheelhouse).resolve(strict=True)
        if not wheelhouse.is_dir():
            raise ValueError("Wheelhouse must be a directory.")
    requirements = json.loads((repo / "scripts/configs/hf_cache_requirements.json").read_text())
    with (repo / "task_list.csv").open() as stream:
        tasks = [
            row["task_dir"]
            for row in csv.DictReader(stream)
            if int(row["gpus"])
            and (
                row["task_dir"] in selected_tasks
                if selected_tasks is not None
                else requirements["tasks"][row["task_dir"]] == "no_external_assets"
            )
        ]
    if selected_tasks is not None and set(tasks) != set(selected_tasks):
        raise ValueError("Selection includes unknown or non-GPU task identities.")
    output.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + budget
    records, images = [], {}
    for task in tasks:
        record = {"task": task, "status": "not_started", "local_base_image": base_image}
        records.append(record)
        remaining = deadline - time.monotonic()
        if remaining < 30:
            record["status"] = "budget_exhausted"
            continue
        try:
            source = repo / "tasks" / task / "environment"
            context = output / task
            shutil.copytree(source, context)
            text = (source / "Dockerfile").read_text().replace("\\\n", " ")
            checkout = re.search(r"git checkout\s+([0-9a-f]{12,40})", text)
            if checkout is None:
                raise ValueError("Cannot identify the pinned source commit.")
            commit = checkout.group(1)
            archive = context / "source.tar.gz"
            with urllib.request.urlopen(
                f"https://codeload.github.com/sgl-project/sglang/tar.gz/{commit}", timeout=30
            ) as response:
                archive.write_bytes(response.read())
            record.update(source_revision=commit, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
            with tarfile.open(archive) as source_archive:
                source_archive.extractall(context / "unpacked", filter="data")
            trees = list((context / "unpacked").iterdir())
            if len(trees) != 1 or not trees[0].is_dir():
                raise ValueError("Unexpected source archive layout.")
            trees[0].rename(context / "code")
            record["upstream_base_image"] = next(line[5:] for line in text.splitlines() if line.startswith("FROM "))
            record["omit_flashinfer_jit_cache"] = omit_jit_cache
            if wheelhouse:
                shutil.copytree(wheelhouse, context / "wheels")
                record["wheels"] = {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in wheelhouse.glob("*.whl")
                }
            recipe = local_recipe(text, base_image, wheelhouse is not None, omit_jit_cache)
            (context / "python3").write_text(
                '#!/bin/sh\nexport LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}"\n'
                'exec /usr/bin/python3 "$@"\n'
            )
            recipe += "COPY --chmod=755 python3 /usr/local/bin/python3\nRUN ln -sf python3 /usr/local/bin/python\n"
            (context / "Dockerfile").write_text(recipe)
            (context / ".dockerignore").write_text("source.tar.gz\nunpacked\n")
            image = f"psrl-swe-local:{task.removeprefix('ss-sgl-')}"
            with (context / "build.log").open("w") as log:
                result = subprocess.run(
                    ["docker", "build", "--network=none", "-t", image, str(context)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=max(1, min(120, deadline - time.monotonic())),
                    check=False,
                )
            record["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError("Image build failed.")
            record["image_id"] = subprocess.check_output(
                ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True, timeout=10
            ).strip()
            record["status"] = "built_not_verified"
            images[task] = image
        except Exception as error:
            record.update(status="failed", error=repr(error))
        (output / "images.json").write_text(json.dumps(images, indent=2))
        (output / "manifest.json").write_text(json.dumps({**revision, "tasks": records}, indent=2))
        print(json.dumps(record), flush=True)
    (output / "manifest.json").write_text(json.dumps({**revision, "tasks": records}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--budget", type=int, default=900)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--wheelhouse", type=Path, help="Offline dependencies for the selected base runtime.")
    parser.add_argument(
        "--omit-jit-cache", action="store_true", help="Use prebuilt cubins without optional JIT cache."
    )
    args = parser.parse_args()
    if not 1 <= args.budget <= 1800:
        parser.error("Build budget must be between 1 and 1800 seconds.")
    build(args.repo, args.output, args.base_image, args.budget, args.tasks, args.wheelhouse, args.omit_jit_cache)


if __name__ == "__main__":
    main()
