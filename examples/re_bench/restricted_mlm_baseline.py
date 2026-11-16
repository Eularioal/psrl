"""Train the upstream restricted model on a bounded public token split."""

import os
from pathlib import Path

import numpy as np
import torch
from model import Model


def main():
    torch.manual_seed(17)
    config = dict(vocab_size=50257, sequence_length=16, embed_dim=32, layer_sizes=[256])
    model = Model(**config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = np.fromfile("/data/public/train.bin", dtype=np.uint16).astype(np.int64)
    for step in range(3):
        raw = torch.tensor(tokens[step * 32 : (step + 1) * 32].reshape(2, 16), device="cuda")
        mask = torch.rand(raw.shape, device="cuda") < 0.2
        mask[:, 0] = True
        x = torch.where(mask, 50256, raw)
        os.environ["IS_TORCH_RESTRICTED"] = "1"
        logits = model(x)
        del os.environ["IS_TORCH_RESTRICTED"]
        loss = torch.nn.functional.cross_entropy(logits[mask], raw[mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.save({"config": config, "state_dict": model.cpu().state_dict()}, "/logs/artifacts/basic_model.pt")
    Path("/logs/artifacts/model.py").write_text(Path("/data/public/model.py").read_text())


if __name__ == "__main__":
    main()
