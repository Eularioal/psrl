"""A neighbour that really computes on the card it can see, for `seconds`."""

import sys
import time

import torch

seconds = float(sys.argv[1])
a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
a @ b
torch.cuda.synchronize()
print("READY", flush=True)
end = time.time() + seconds
while time.time() < end:
    for _ in range(20):
        a @ b
    torch.cuda.synchronize()
