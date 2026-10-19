"""One emulated episode: a process that really holds device memory on the card it can see."""

import json
import os
import sys
import time

gib = float(sys.argv[1])
hold_s = float(sys.argv[2])
label = sys.argv[3]
card = os.environ.get("CUDA_VISIBLE_DEVICES")

import torch  # noqa: E402

try:
    buf = torch.empty(int(gib * 2**30), dtype=torch.uint8, device="cuda")
    buf.fill_(1)
    torch.cuda.synchronize()
    print(json.dumps({"label": label, "status": "OK", "card": card, "t_alloc": time.time()}), flush=True)
    time.sleep(hold_s if hold_s >= 0 else 3600)
except Exception as exc:  # noqa: BLE001
    status = "OOM" if "out of memory" in str(exc).lower() else f"ERR:{type(exc).__name__}"
    print(json.dumps({"label": label, "status": status, "card": card, "t_alloc": None}), flush=True)
