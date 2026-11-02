"""Run fixed autoresearch updates with barriers separating preparation and measurement."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
original = args.source.read_text()
assert (
    hashlib.sha256(original.encode()).hexdigest() == "2954175f4ac42ad65164aef40910ef953789abcd05a5cc886ac9ba5a00814414"
)
record = {"pid": os.getpid(), "source_sha256": hashlib.sha256(original.encode()).hexdigest(), "steps": []}


def save(name, value):
    target = args.output / name
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(target)


def barrier(name):
    save(name + "-ready.json", {"time": time.time(), "pid": os.getpid()})
    deadline = time.monotonic() + 180
    while not (args.output / (name + "-go")).exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Training barrier expired.")
        time.sleep(0.02)


code = original[: original.index("# Final eval")]
code = code.replace("DEVICE_BATCH_SIZE = 128", "DEVICE_BATCH_SIZE = 64")
code = code.replace("progress = min(total_training_time / TIME_BUDGET, 1.0)", "progress = min(step / 13, 1.0)")
code = code.replace(
    "while True:\n    torch.cuda.synchronize()",
    """while True:
    if step == 3:
        torch.cuda.synchronize()
        record['warmup_peak_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        barrier('train')
        record['train_start'] = time.time()
    if step == 13:
        torch.cuda.synchronize()
        record['train_end'] = time.time()
        record['loss_after_10'] = train_loss_f
        save('measured.json', record)
        if not args.profile:
            break
        barrier('profile')
        try:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA])
            profiler.start()
        except Exception as exc:
            record['profile_error'] = repr(exc)
            profiler = None
    torch.cuda.synchronize()""",
)
code = code.replace(
    "    dt = t1 - t0",
    "    dt = t1 - t0\n    record['steps'].append({'step': step, 'seconds': dt, 'loss': train_loss_f})",
)
code = code.replace("if step > 10 and total_training_time >= TIME_BUDGET:", "if step >= 15:")
namespace = {
    "__name__": "__main__",
    "args": args,
    "record": record,
    "barrier": barrier,
    "save": save,
    "profiler": None,
}
started = time.time()
try:
    exec(compile(code, str(args.source), "exec"), namespace)
    profiler = namespace.get("profiler")
    if profiler:
        try:
            profiler.stop()
            profiler.export_chrome_trace(str(args.output / "trace.json"))
        except Exception as exc:
            record["profile_error"] = repr(exc)
    record["passed"] = True
finally:
    record["wall_seconds"] = time.time() - started
    save("result.json", record)
