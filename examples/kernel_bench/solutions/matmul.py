"""Bounded Triton reference submission for KernelBench level 1 problem 1."""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul(a, b, c, n: tl.constexpr, block: tl.constexpr):
    rows = tl.program_id(0) * block + tl.arange(0, block)
    cols = tl.program_id(1) * block + tl.arange(0, block)
    inner = tl.arange(0, block)
    acc = tl.zeros((block, block), dtype=tl.float32)
    for base in range(tl.cdiv(n, block)):
        k = base * block + inner
        left = tl.load(a + rows[:, None] * n + k[None, :], (rows[:, None] < n) & (k[None, :] < n), 0)
        right = tl.load(b + k[:, None] * n + cols[None, :], (k[:, None] < n) & (cols[None, :] < n), 0)
        acc += tl.dot(left, right, input_precision="ieee")
    tl.store(c + rows[:, None] * n + cols[None, :], acc, (rows[:, None] < n) & (cols[None, :] < n))


class ModelNew(torch.nn.Module):
    def forward(self, a, b):
        n = a.shape[0]
        result = torch.empty((n, n), device=a.device, dtype=a.dtype)
        matmul[(triton.cdiv(n, 32), triton.cdiv(n, 32))](a, b, result, n, 32)
        return result
