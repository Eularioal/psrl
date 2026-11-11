"""Train a small CUDA classifier using only public training features and labels."""

import argparse
import csv
from pathlib import Path

import torch


def main(public=Path("/data/public"), artifacts=Path("/logs/artifacts")):
    torch.manual_seed(17)
    artifacts.mkdir(parents=True, exist_ok=True)
    with (public / "train.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    features = [key for key in rows[0] if key not in ("id", "label")]
    x = torch.tensor([[float(row[key]) for key in features] for row in rows], device="cuda")
    y = torch.tensor([int(row["label"]) for row in rows], device="cuda")
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    x = (x - mean) / scale
    model = torch.nn.Sequential(
        torch.nn.Linear(len(features), 128), torch.nn.ReLU(), torch.nn.Linear(128, int(y.max()) + 1)
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(100):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with (public / "test.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    x = torch.tensor([[float(row[key]) for key in features] for row in rows], device="cuda")
    x = (x - mean) / scale
    with torch.no_grad():
        predictions = model(x).argmax(-1).cpu().tolist()
    with (artifacts / "submission.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label"])
        writer.writerows((row["id"], prediction) for row, prediction in zip(rows, predictions, strict=True))
    torch.cuda.synchronize()
    print({"device": torch.cuda.get_device_name(), "steps": 100, "loss": loss.item()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=Path("/data/public"))
    parser.add_argument("--artifacts", type=Path, default=Path("/logs/artifacts"))
    args = parser.parse_args()
    main(args.public, args.artifacts)
