"""
Measure unchanged PostTrain verification with fresh or identity checked resident bases.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def digest(value):
    """
    Hash a canonical manifest without relying on file size or timestamps.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_manifest(root):
    """
    Read every base file on every request without modifying shared artifacts.
    """
    root = Path(root).resolve()
    files, total = [], 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ValueError("Directory symlinks are outside the bounded model manifest contract.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Model assets must be readable regular files: {str(path)!r}.")
        content = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                content.update(block)
                size += len(block)
        files.append({"path": str(path.relative_to(root)), "sha256": content.hexdigest(), "bytes": size})
        total += size
    if not files:
        raise ValueError("The model directory must contain real assets.")
    return {"sha256": digest(files), "bytes_hashed": total, "files": files}


def tensor_identity(state, torch):
    """
    Include all candidate keys, shapes, dtypes and exact contiguous tensor bytes.
    """
    members = []
    for name, tensor in sorted(state.items()):
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        members.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return digest(members)


class Slot:
    """
    Retain at most one exact loader identity so model changes cannot accumulate GPU bases.
    """

    def __init__(self):
        self.key, self.value = None, None

    def get(self, key, build, release):
        """
        Release the previous identity before constructing a replacement.
        """
        if key == self.key:
            return self.value, True
        self.key, self.value = None, None
        release()
        self.value = build()
        self.key = key
        return self.value, False

    def clear(self):
        """
        Drop the resident reference before final resource cleanup.
        """
        self.key, self.value = None, None


class Study:
    """
    Instrument the real verify function without changing its training or scoring code.
    """

    def __init__(self, mode, model, source):
        import safetensors.torch
        import torch
        import transformers

        if mode not in {"fresh", "resident"}:
            raise ValueError("Only fresh and resident verification modes are supported.")
        self.mode, self.model, self.source = mode, Path(model).resolve(), Path(source).resolve()
        self.torch, self.transformers, self.safetensors = torch, transformers, safetensors.torch
        self.tokenizer_slot, self.model_slot = Slot(), Slot()
        self.current_model = None
        self.rows, self.observed = [], {}
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def release(self):
        """
        Collect dead model objects while preserving an explicitly resident base.
        """
        gc.collect()
        self.torch.cuda.empty_cache()

    def request(self, data, checkpoint):
        """
        Run one unchanged verify call and audit the fully applied candidate tensor.
        """
        started = time.monotonic()
        self.current_model = None
        data, checkpoint = Path(data).resolve(), Path(checkpoint).resolve()
        candidate = checkpoint / "adapter.safetensors"
        source_bytes, data_bytes, candidate_bytes = self.source.read_bytes(), data.read_bytes(), candidate.read_bytes()
        manifest = file_manifest(self.model)
        identity = {
            "base": manifest["sha256"],
            "source": hashlib.sha256(source_bytes).hexdigest(),
            "versions": self.versions,
        }
        request_id = digest(
            {
                **identity,
                "data": hashlib.sha256(data_bytes).hexdigest(),
                "candidate": hashlib.sha256(candidate_bytes).hexdigest(),
            }
        )
        expected_state = self.safetensors.load(candidate_bytes)
        if set(expected_state) != {"model.norm.weight"}:
            raise ValueError("The bounded verify protocol requires the complete final normalization vector.")
        expected_id = tensor_identity(expected_state, self.torch)
        identity_s = time.monotonic() - started
        namespace = {"__file__": str(self.source), "__name__": "bounded_posttrain_source"}
        exec(compile(source_bytes, str(self.source), "exec"), namespace)
        execute = namespace["execute"]
        phases = {
            "tokenizer_from_pretrained_s": 0.0,
            "model_from_pretrained_s": 0.0,
            "cuda_transfer_s": 0.0,
            "candidate_load_s": 0.0,
        }
        hits, loaded = {}, {}
        tokenizer_class = self.transformers.AutoTokenizer
        model_class = self.transformers.AutoModelForCausalLM
        tokenizer_loader, model_loader = tokenizer_class.from_pretrained, model_class.from_pretrained
        candidate_loader = self.safetensors.load_file
        cuda = self.torch.nn.Module.cuda

        def pretrained(kind, loader, slot, args, kwargs):
            begin = time.monotonic()
            key = digest({**identity, "path": str(Path(args[0]).resolve()), "kwargs": kwargs})
            if self.mode == "resident":
                value, hit = slot.get(key, lambda: loader(*args, **kwargs), self.release)
            else:
                value, hit = loader(*args, **kwargs), False
            phases[f"{kind}_from_pretrained_s"] += time.monotonic() - begin
            hits[kind] = hit
            if kind == "model":
                self.current_model = value
            return value

        def load_candidate(path, *args, **kwargs):
            if Path(path).resolve() != candidate:
                raise ValueError("The bounded verifier attempted to load an undeclared checkpoint.")
            begin = time.monotonic()
            state = candidate_loader(path, *args, **kwargs)
            loaded.update({name: tensor.detach().cpu().clone() for name, tensor in state.items()})
            if tensor_identity(loaded, self.torch) != expected_id:
                raise RuntimeError("The loaded candidate tensors do not match the captured source bytes.")
            phases["candidate_load_s"] += time.monotonic() - begin
            return state

        def transfer(module, *args, **kwargs):
            begin = time.monotonic()
            result = cuda(module, *args, **kwargs)
            self.torch.cuda.synchronize()
            phases["cuda_transfer_s"] += time.monotonic() - begin
            return result

        try:
            execute_started = time.monotonic()
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        tokenizer_class,
                        "from_pretrained",
                        side_effect=lambda *args, **kwargs: pretrained(
                            "tokenizer", tokenizer_loader, self.tokenizer_slot, args, kwargs
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        model_class,
                        "from_pretrained",
                        side_effect=lambda *args, **kwargs: pretrained(
                            "model", model_loader, self.model_slot, args, kwargs
                        ),
                    )
                )
                stack.enter_context(patch.object(self.safetensors, "load_file", side_effect=load_candidate))
                stack.enter_context(patch.object(self.torch.nn.Module, "cuda", new=transfer))
                metrics = execute("verify", str(self.model), str(data), str(checkpoint))
            execute_s = time.monotonic() - execute_started
            if set(loaded) != {"model.norm.weight"}:
                raise RuntimeError("The verify function did not load the declared complete candidate.")
            parameter = dict(self.current_model.named_parameters())["model.norm.weight"]
            expected_applied = {"model.norm.weight": expected_state["model.norm.weight"].to(dtype=parameter.dtype)}
            applied_id = tensor_identity({"model.norm.weight": parameter}, self.torch)
            if applied_id != tensor_identity(expected_applied, self.torch):
                raise RuntimeError("The resident model did not fully replace its candidate normalization vector.")
            if data.read_bytes() != data_bytes or candidate.read_bytes() != candidate_bytes:
                raise RuntimeError("A declared request input changed during verification.")
            previous = self.observed.get(request_id)
            consistent = previous is None or all(
                math.isclose(metrics[key], previous[key], rel_tol=1e-5, abs_tol=1e-7)
                for key in ("answer_nll", "reward")
            )
            if not consistent:
                raise RuntimeError("An identical request changed score after switching candidate states.")
            self.observed[request_id] = metrics
            row = {
                "index": len(self.rows),
                "request_id": request_id,
                "data": str(data),
                "checkpoint": str(checkpoint),
                "identity": identity,
                "candidate_bytes_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
                "candidate_tensor_id": expected_id,
                "applied_norm_tensor_id": applied_id,
                "identity_s": identity_s,
                "execute_s": execute_s,
                "phases": phases,
                "cache_hits": hits,
                "metrics": metrics,
                "repeated_request_consistent": consistent,
                "base_manifest": manifest,
            }
        finally:
            self.current_model = None
            self.release()
        row["total_wall_s"] = time.monotonic() - started
        self.rows.append(row)
        return row

    def close(self):
        """
        Remove retained bases before reporting end to end cleanup time.
        """
        self.current_model = None
        self.tokenizer_slot.clear()
        self.model_slot.clear()
        self.release()
        self.torch.cuda.synchronize()


def main():
    """
    Run a predeclared bounded sequence under an external driver timeout.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fresh", "resident"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data", type=Path)
    data.add_argument("--data-paths", type=Path, nargs="+")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", default=os.environ.get("PSRL_EXPECTED_GPU_UUID"))
    args = parser.parse_args()
    paths = args.data_paths if args.data_paths else [args.data] * len(args.checkpoints)
    if len(paths) != len(args.checkpoints) or len(paths) > 8:
        raise ValueError("Declare one data path per checkpoint and at most eight requests.")
    if not args.expected_gpu_uuid:
        raise ValueError("The driver must declare the expected GPU UUID.")
    started, study = time.monotonic(), None
    report = {
        "status": "running",
        "mode": args.mode,
        "boundary": (
            "Single process verification only. Base files are read only. "
            "Full content hashing does not provide an atomic snapshot under concurrent external writes."
        ),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        study = Study(args.mode, args.model, args.source)
        actual = str(study.torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
        if actual != args.expected_gpu_uuid.removeprefix("GPU-"):
            raise RuntimeError("The worker did not select the declared GPU.")
        sample = study.torch.arange(8, device="cuda")
        if int((sample * sample).sum()) != 140:
            raise RuntimeError("The CUDA preflight kernel returned an incorrect result.")
        del sample
        study.torch.cuda.synchronize()
        report.update(gpu_uuid=actual, versions=study.versions, setup_s=time.monotonic() - started)
        for path, checkpoint in zip(paths, args.checkpoints, strict=True):
            report["rows"].append(study.request(path, checkpoint))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        begin = time.monotonic()
        if study is not None:
            study.close()
        report.update(final_cleanup_s=time.monotonic() - begin, total_wall_s=time.monotonic() - started)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
