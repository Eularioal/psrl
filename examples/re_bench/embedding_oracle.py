"""Restore the setup permutation as a CUDA control for embedding repair."""

import torch


def main():
    state = torch.load("/data/public/broken_model.pt", map_location="cuda", weights_only=True)
    torch.manual_seed(48228)
    permutation = torch.randperm(state["transformer.wte.weight"].shape[0])
    restored = state["transformer.wte.weight"][torch.argsort(permutation).cuda()]
    state["transformer.wte.weight"] = restored
    state["lm_head.weight"] = restored
    torch.save(
        {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}, "/logs/artifacts/fixed_model.pth"
    )


if __name__ == "__main__":
    main()
