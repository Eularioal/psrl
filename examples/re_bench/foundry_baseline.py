"""Local replacement for the upstream LLM Foundry pipeline: two GPT-2 updates.

The fixed model, data, optimizer and update count define the local contract.
A candidate may optimize this pipeline while preserving the resulting weights.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def train(destination):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained("/data/public/model", local_files_only=True).cuda()
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.p = 0
    tokens = np.fromfile("/data/public/train.bin", dtype=np.uint16).astype(np.int64)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for step in range(2):
        ids = torch.from_numpy(tokens[step * 32 : step * 32 + 32]).cuda()[None]
        loss = model(ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    model.save_pretrained(destination)


if __name__ == "__main__":
    train(Path(sys.argv[1]))
