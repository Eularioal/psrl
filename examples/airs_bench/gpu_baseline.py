"""Fit bounded CUDA baselines using public AIRS data and write a submission."""

import csv
import hashlib
import json
import re
from pathlib import Path

import torch


def features(rows, width=128):
    """Hash public features into fixed vectors without reading test labels."""
    result = torch.zeros(len(rows), width)
    for index, row in enumerate(rows):
        text = json.dumps(row, sort_keys=True, default=str)[:16384]
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            result[index, int.from_bytes(digest, "little") % width] += 1
    return torch.nn.functional.normalize(result, dim=-1).cuda()


def predict(spec):
    """Use CUDA learning for numeric targets and similarity search for text."""
    if not torch.cuda.is_available():
        raise RuntimeError("The GPU baseline requires executable CUDA.")
    torch.manual_seed(20261111)
    torch.set_num_threads(1)
    train = features(spec["train_x"])
    test = features(spec["test_x"])
    outputs = spec["train_y"]
    kind = spec["kind"]
    steps = 0
    if kind in ("classification", "regression"):
        if kind == "classification":
            classes = sorted(set(outputs))
            targets = torch.tensor([classes.index(v) for v in outputs], device="cuda")
            model = torch.nn.Linear(train.shape[1], len(classes)).cuda()
        else:
            targets = torch.tensor(outputs, device="cuda", dtype=torch.float32).reshape(len(outputs), -1)
            mean = targets.mean(0)
            scale = targets.std(0).clamp_min(1e-4)
            targets = (targets - mean) / scale
            model = torch.nn.Linear(train.shape[1], targets.shape[1]).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            optimizer.zero_grad()
            prediction = model(train)
            loss = (
                torch.nn.functional.cross_entropy(prediction, targets)
                if kind == "classification"
                else torch.nn.functional.mse_loss(prediction, targets)
            )
            loss.backward()
            optimizer.step()
            steps += 1
        with torch.no_grad():
            predicted = model(test)
            if kind == "classification":
                values = [classes[i] for i in predicted.argmax(-1).tolist()]
            else:
                values = (predicted * scale + mean).cpu().tolist()
                values = [row[0] if len(row) == 1 else row for row in values]
    else:
        ranks = (test @ train.T).topk(min(5, len(outputs)), dim=-1).indices.tolist()
        if kind == "retrieval":
            values = [[outputs[i] for i in row] for row in ranks]
        elif kind == "code":
            values = [[outputs[i][0] for i in row] for row in ranks]
            values = [(row * 5)[:5] for row in values]
        else:
            values = [outputs[row[0]] for row in ranks]
    torch.cuda.synchronize()
    telemetry = {
        "device": torch.cuda.get_device_name(),
        "train_rows": len(outputs),
        "prediction_rows": len(values),
        "optimizer_steps": steps,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "kind": kind,
    }
    return values, telemetry


def main():
    root = Path("/home/agent/workspace")
    spec = json.loads((root / "data/train/psrl-baseline.json").read_text())
    values, telemetry = predict(spec)
    with (root / "submission.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(spec["columns"])
        for index, value in enumerate(values):
            if spec["kind"] == "code":
                writer.writerow(value)
            elif spec["kind"] == "retrieval":
                writer.writerow([spec["test_x"][index]["query"], json.dumps(value)])
            elif spec["kind"] == "duorc":
                writer.writerow([value, not spec["test_x"][index]["no_answer"]])
            else:
                writer.writerow([json.dumps(value) if isinstance(value, list) else value])
    (root / "gpu-baseline.json").write_text(json.dumps(telemetry))
    print(json.dumps(telemetry), flush=True)


if __name__ == "__main__":
    main()
