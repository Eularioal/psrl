"""Download the pinned Alpaca split and GPT-2 XL weights for RE-Bench."""

import argparse
import hashlib
import json
import os
from pathlib import Path

DATA_REVISION = "dce01c9b08f87459cf36a430d809084718273017"


def split_prompts(texts):
    prompts = [text.split("### Response:\n")[0] + "### Response:\n" for text in texts]
    if len(prompts) <= 512:
        raise ValueError("The official split requires 512 held-out prompts and training data.")
    return prompts[512:], prompts[:512]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(root / "tiktoken")
    import tiktoken
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    data = load_dataset("tatsu-lab/alpaca", revision=DATA_REVISION, split="train", cache_dir=str(root / "download"))
    train, heldout = split_prompts(data["text"])
    for name, prompts in (("public", train), ("private", heldout)):
        directory = root / name
        directory.mkdir(exist_ok=True)
        (directory / "prompts.jsonl").write_text("".join(json.dumps({"prompt": p}) + "\n" for p in prompts))
    revision = HfApi().model_info("gpt2-xl").sha
    snapshot_download(
        "gpt2-xl",
        revision=revision,
        cache_dir=str(root / "hf/hub"),
        allow_patterns=["config.json", "model.safetensors", "generation_config.json"],
        max_workers=2,
    )
    # The unchanged upstream loader resolves its model name through the offline main ref.
    refs = root / "hf/hub/models--gpt2-xl/refs"
    refs.mkdir(exist_ok=True)
    (refs / "main").write_text(revision)
    tiktoken.get_encoding("gpt2")
    manifest = {
        "dataset": "tatsu-lab/alpaca",
        "dataset_revision": DATA_REVISION,
        "model": "gpt2-xl",
        "model_revision": revision,
        "training_prompts": len(train),
        "heldout_prompts": len(heldout),
        "sha256": {
            name: hashlib.sha256((root / name / "prompts.jsonl").read_bytes()).hexdigest()
            for name in ("public", "private")
        },
    }
    (root / "assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
