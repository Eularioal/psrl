"""Train CUDA image models and score their weights independently on private labels."""

import argparse
import json
import math
import re
from pathlib import Path


def tokens(text):
    return re.findall(r"[a-z0-9]+|[^\w\s]", text.lower())


def evaluate(mode, data, checkpoint, private, *, epochs=2, training_hook=None):
    import numpy as np
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("Vision tasks require CUDA.")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = json.loads((data / "config.json").read_text())
    captions = config["caption"]
    labels = json.loads((data / "train_labels.json").read_text())
    vocabulary = ["<pad>", "<bos>", "<eos>", "<unk>"]
    if captions:
        vocabulary += sorted({word for group in labels for sentence in group for word in tokens(sentence)})
    word_ids = {word: index for index, word in enumerate(vocabulary)}

    class ImageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            if captions:
                self.embedding = nn.Embedding(len(vocabulary), 32)
                self.decoder = nn.GRU(32, 32, batch_first=True)
                self.head = nn.Linear(32, len(vocabulary))
            else:
                self.head = nn.Linear(32, 10)

        def forward(self, image, words=None):
            encoded = self.encoder(image)
            if not captions:
                return self.head(encoded)
            output, _ = self.decoder(self.embedding(words), encoded.unsqueeze(0))
            return self.head(output)

        def generate(self, image):
            state = self.encoder(image).unsqueeze(0)
            current = torch.ones((len(image), 1), dtype=torch.long, device="cuda")
            generated = []
            for _ in range(16):
                output, state = self.decoder(self.embedding(current), state)
                current = self.head(output).argmax(-1)
                generated.append(current[:, 0])
            return torch.stack(generated, 1).cpu().tolist()

    images = np.load(data / "images.npz", allow_pickle=False)
    model = ImageModel().cuda()
    updates = 0
    if mode == "train":
        x = torch.tensor(images["train_x"], device="cuda", dtype=torch.float32).permute(0, 3, 1, 2) / 255
        if captions:
            encoded = []
            for group in labels:
                sequence = [1] + [word_ids[word] for word in tokens(group[0])[:15]] + [2]
                encoded.append(sequence + [0] * (17 - len(sequence)))
            y = torch.tensor(encoded, device="cuda", dtype=torch.long)
        else:
            y = torch.tensor(labels, device="cuda", dtype=torch.long)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        if training_hook is not None:
            training_hook("prepared", updates)
        for _ in range(epochs):
            for start in range(0, len(x), 16):
                bx, by = x[start : start + 16], y[start : start + 16]
                optimizer.zero_grad()
                if captions:
                    logits = model(bx, by[:, :-1])
                    loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocabulary)), by[:, 1:].reshape(-1))
                else:
                    loss = nn.functional.cross_entropy(model(bx), by)
                if not torch.isfinite(loss):
                    raise ValueError("Vision training loss must be finite.")
                loss.backward()
                optimizer.step()
                updates += 1
                if training_hook is not None:
                    training_hook("updated", updates)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        return {"cuda": 1, "optimizer_updates": updates, "checkpoint_bytes": checkpoint.stat().st_size}

    if not checkpoint.is_file() or checkpoint.stat().st_size > 16 * 1024**2:
        raise ValueError("Missing or oversized model checkpoint.")
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    x = torch.tensor(images["test_x"], device="cuda", dtype=torch.float32).permute(0, 3, 1, 2) / 255
    truth = json.loads((private / "test_labels.json").read_text())
    if len(truth) != len(x):
        raise ValueError("Private labels do not match the evaluation images.")
    with torch.no_grad():
        if captions:
            from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

            generated = model.generate(x)
            predictions = []
            for sequence in generated:
                if 2 in sequence:
                    sequence = sequence[: sequence.index(2)]
                predictions.append([vocabulary[index] for index in sequence if index >= 4])
            references = [[tokens(sentence) for sentence in group] for group in truth]
            score = float(corpus_bleu(references, predictions, smoothing_function=SmoothingFunction().method1))
        else:
            predictions = model(x).argmax(-1).cpu().numpy()
            score = float(np.mean(predictions == np.array(truth)))
    if not math.isfinite(score):
        raise ValueError("Vision evaluation score must be finite.")
    torch.cuda.synchronize()
    return {
        "reward": score,
        "bleu4" if captions else "accuracy": score,
        "eval_valid": 1,
        "local_evaluation": 1,
        "cuda": 1,
        "test_rows": len(truth),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "verify"))
    parser.add_argument("--data", type=Path, default=Path("/data/public"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/logs/artifacts/model.pt"))
    parser.add_argument("--private", type=Path, default=Path("/data/private"))
    args = parser.parse_args()
    result = evaluate(args.mode, args.data, args.checkpoint, args.private)
    if args.mode == "verify":
        output = Path("/logs/verifier/reward.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
