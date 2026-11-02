"""Prepare an offline image context from downloaded official inference assets."""

import argparse
import errno
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "-C", str(args.upstream), "status", "--porcelain"]):
        raise ValueError("The official upstream checkout must be clean.")
    if not (args.context / "python-extra").is_dir():
        raise ValueError("Install resident-requirements.txt into context/python-extra first.")
    model = args.context / "model"
    model.mkdir(exist_ok=False)
    index = json.loads((args.model_snapshot / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    for name in sorted(required):
        source = (args.model_snapshot / name).resolve(strict=True)
        try:
            os.link(source, model / name)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, model / name)
    server = args.upstream / "src/eval/inference/servers/transformers_openai_server.py"
    shutil.copyfile(server, args.context / "server.py")
    shutil.copyfile(Path(__file__).with_name("resident_client.py"), args.context / "client.py")
    samples = args.upstream / "src/eval/inference/baselines/samples/mmlu_pro/21_500/samples.jsonl"
    (args.context / "samples.jsonl").write_text("\n".join(samples.read_text().splitlines()[:2]) + "\n")
    (args.context / "Dockerfile").write_text(
        f"FROM {args.base_image}\nCOPY python-extra /opt/python-extra\nCOPY model /opt/model\n"
        "COPY server.py client.py samples.jsonl /opt/resident/\nENV PYTHONPATH=/opt/python-extra\n"
    )
    manifest = {
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        "server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
        "samples_sha256": hashlib.sha256((args.context / "samples.jsonl").read_bytes()).hexdigest(),
        "model_revision": args.model_snapshot.name,
        "base_image": args.base_image,
        "profile": "Two MMLU-Pro samples, 16 generated tokens, 2048 context, bfloat16",
    }
    (args.context / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
