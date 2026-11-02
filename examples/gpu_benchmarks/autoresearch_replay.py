"""Capture the upstream evaluation call instead of trusting printed scores."""

import json
import math
import runpy
from pathlib import Path

import prepare

original_evaluate = prepare.evaluate_bpb
measurements = []


def measure(model, tokenizer, batch_size):
    value = float(original_evaluate(model, tokenizer, batch_size))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("The upstream evaluation returned an invalid BPB.")
    measurements.append(value)
    return value


prepare.evaluate_bpb = measure
runpy.run_path("train.py", run_name="__main__")
if len(measurements) != 1:
    raise ValueError("The submitted training script must execute the upstream evaluation exactly once.")
Path("measured.json").write_text(json.dumps({"val_bpb": measurements[0]}))
