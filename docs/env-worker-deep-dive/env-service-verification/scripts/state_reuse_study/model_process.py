"""Run a pinned CPU model separately from mutable scoring code inside a sandbox."""

import argparse
import hashlib
import json
import os
import resource
import runpy
import socket
import subprocess
import sys
import time
from pathlib import Path


def code_identity(directory):
    """Describe the two files used by this explicit scoring adapter."""
    return hashlib.sha256(
        json.dumps(
            {name: (directory / name).read_text() for name in ("helper.py", "scorer.py")}, sort_keys=True
        ).encode()
    ).hexdigest()


def load_score(directory):
    """Load the candidate in the current process for the monolithic controls."""
    sys.path.insert(0, str(directory))
    return runpy.run_path(str(directory / "scorer.py"))["score"]


def postprocess(directory, value):
    """Execute scoring code in a fresh process that has no model objects."""
    result = subprocess.check_output(
        [sys.executable, "-B", __file__, "post", "--code", str(directory), "--value", str(value)],
        text=True,
        timeout=20,
    )
    return json.loads(result)


class CpuModel:
    """Keep the pinned classification weights in CPU memory for repeated requests."""

    def __init__(self, model_path, config):
        started = time.monotonic()
        import torch
        from transformers import AutoModelForSequenceClassification

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        if torch.cuda.device_count() != 0:
            raise RuntimeError("The experiment must have no visible CUDA device.")
        imported = time.monotonic()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        self.config, self.torch = config, torch
        self.timings = dict(import_s=imported - started, model_load_s=time.monotonic() - imported)
        self.model_identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def forward(self, seed):
        """Run real deterministic inference on short token sequences."""
        torch = self.torch
        tokens = (torch.arange(self.config["length"], dtype=torch.long) + seed + 100).unsqueeze(0)
        started = time.monotonic()
        with torch.inference_mode():
            value = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens)).logits.float().sum().item()
        return value, time.monotonic() - started


def answer(model, request, frozen_score=None, frozen_code=None):
    """Return model output and exactly the scoring code that produced the value."""
    base, inference_s = model.forward(request["seed"])
    code = Path(request["code"])
    if frozen_score is None:
        transformed = postprocess(code, base)
        value, actual_code = transformed["value"], transformed["code_identity"]
    else:
        value, actual_code = frozen_score(base), frozen_code
    gate = request.get("gate")
    if gate:
        Path(gate + ".ready").write_text("ready")
        deadline = time.monotonic() + 15
        while not Path(gate + ".continue").exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("The bounded publication fault gate was not released.")
            time.sleep(0.02)
    return dict(
        value=value,
        base_value=base,
        code_identity=actual_code,
        model_identity=model.model_identity,
        pid=os.getpid(),
        inference_s=inference_s,
        preparation=model.timings,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cuda_initialized=model.torch.cuda.is_initialized(),
    )


def serve(args):
    """Keep one model process until the controller explicitly stops it."""
    model = CpuModel(args.model, json.loads(args.config.read_text()))
    frozen_score = load_score(args.code) if args.frozen else None
    frozen_code = code_identity(args.code) if args.frozen else None
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.bind(str(args.socket))
    endpoint.listen(1)
    args.ready.write_text(json.dumps(dict(pid=os.getpid(), preparation=model.timings)))
    try:
        while True:
            connection, _ = endpoint.accept()
            with connection, connection.makefile("rwb") as stream:
                request = json.loads(stream.readline())
                if request.get("stop"):
                    stream.write(b'{"stopped":true}\n')
                    stream.flush()
                    return
                result = answer(model, request, frozen_score, frozen_code)
                stream.write((json.dumps(result) + "\n").encode())
                stream.flush()
    finally:
        endpoint.close()
        args.socket.unlink(missing_ok=True)


def main():
    """Provide a model service, a lightweight client and a fresh-process control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve", "request", "once", "post"])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--code", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--ready", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--value", type=float)
    parser.add_argument("--frozen", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.mode == "serve":
        serve(args)
    elif args.mode == "post":
        print(json.dumps(dict(value=load_score(args.code)(args.value), code_identity=code_identity(args.code))))
    elif args.mode == "once":
        model = CpuModel(args.model, json.loads(args.config.read_text()))
        result = answer(model, json.loads(args.request.read_text()), load_score(args.code), code_identity(args.code))
        print("STATE_RESULT " + json.dumps(result))
    else:
        endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        endpoint.settimeout(90)
        endpoint.connect(str(args.socket))
        with endpoint, endpoint.makefile("rwb") as stream:
            stream.write(args.request.read_bytes() + b"\n")
            stream.flush()
            print("STATE_RESULT " + stream.readline().decode().strip())


if __name__ == "__main__":
    main()
