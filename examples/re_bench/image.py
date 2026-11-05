"""Build separate RE-Bench runtime and verifier image contexts from pinned assets."""

import argparse
import errno
import hashlib
import json
import os
import shutil
from pathlib import Path

from examples.gpu_benchmarks.prepare import provenance


def copy_asset(source, destination):
    source = Path(source).resolve()
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copyfile(source, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scoring-repo", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    args = parser.parse_args()
    upstream = args.repo / "ai_rd_nanogpt_chat_rl/assets"
    context = args.context
    if not (context / "python-extra").is_dir():
        raise ValueError("Install the isolated image dependencies into context/python-extra first.")
    manifest = {
        "upstream": provenance(args.repo),
        "scoring": provenance(args.scoring_repo),
        "assets": json.loads((args.assets / "assets.json").read_text()),
        "base_image": args.base_image,
    }
    for name in ("hf", "tiktoken", "public", "private"):
        shutil.copytree(args.assets / name, context / name, copy_function=copy_asset, symlinks=True)
    solution = context / "solution"
    shutil.copytree(upstream / "solution", solution)
    reference = context / "private/reference_outputs"
    reference.mkdir()
    for name in ("gpt2-xl.jsonl", "gpt2-alpaca.jsonl"):
        shutil.copyfile(upstream / name, reference / name)
    package = context / "rebench"
    package.mkdir()
    for name in ("replicate.py", "strict_evaluation.py", "smoke.py", "feedback_smoke.py", "verify.py"):
        shutil.copyfile(Path(__file__).with_name(name), package / name)
    for name in ("evaluation.py", "score.py", "padding_text.txt"):
        target = "evaluation_official.py" if name == "evaluation.py" else name
        shutil.copyfile(upstream / name, package / target)
    shutil.copytree(args.scoring_repo / "metr", package / "metr")
    manifest["upstream_files_sha256"] = {
        name: hashlib.sha256((upstream / name).read_bytes()).hexdigest()
        for name in ("score.py", "evaluation.py", "solution/model.py", "solution/rl_finetune.py")
    }
    (context / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (context / "test.sh").write_text("#!/bin/bash\nset -euo pipefail\npython /opt/rebench/verify.py\n")
    (context / "Dockerfile").write_text(
        f"FROM {args.base_image} AS runtime\n"
        "RUN rm -rf /opt/upstream /root/.cache/autoresearch /root/.cache/huggingface\n"
        "COPY python-extra /opt/python-extra\nCOPY hf /opt/hf\nCOPY tiktoken /opt/tiktoken\n"
        "COPY solution /home/agent/solution\nCOPY public/prompts.jsonl /home/agent/prompts.jsonl\n"
        "COPY rebench /opt/rebench\nCOPY manifest.json /opt/rebench/manifest.json\n"
        "COPY test.sh /tests/test.sh\nRUN chmod +x /tests/test.sh\n"
        "RUN cp /opt/rebench/strict_evaluation.py /home/agent/evaluation.py && "
        "cp /opt/rebench/padding_text.txt /home/agent/padding_text.txt && "
        "ln -s /logs/artifacts/finetuned_model /home/agent/finetuned_model\n"
        "ENV PYTHONPATH=/opt/rebench:/opt/python-extra:/home/agent/solution:/home/agent "
        "HF_HOME=/opt/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TIKTOKEN_CACHE_DIR=/opt/tiktoken\n"
        "WORKDIR /home/agent\n"
        "FROM runtime AS verifier\nCOPY private /protected\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
