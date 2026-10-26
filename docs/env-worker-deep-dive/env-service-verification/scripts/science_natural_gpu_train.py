"""Bound the original MLGym CIFAR Trainer and expose explicit CPU and GPU phases."""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path


def prepare(payload):
    """Prepare images on CPU and defer CUDA probing imports until admission."""
    index = payload["index"]
    record = {"index": index, "pid": os.getpid(), "started_at": time.time(), "phases": {}}
    started = time.perf_counter()
    import numpy as np

    record["phases"]["numpy_s"] = time.perf_counter() - started
    started = time.perf_counter()
    import torch

    record["phases"]["torch_s"] = time.perf_counter() - started
    started = time.perf_counter()
    from datasets import Dataset

    record["phases"]["datasets_s"] = time.perf_counter() - started
    torch.set_num_threads(1)
    started = time.perf_counter()
    data = np.load("/opt/workload/cifar640.npz")
    pixels = data["images"].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    dataset = Dataset.from_dict({"pixel_values": pixels, "label": data["labels"]})
    dataset.set_format("torch")
    record["phases"]["data_s"] = time.perf_counter() - started
    record["prepared_at"] = time.time()

    def execute():
        try:
            record["gpu_phase_started_at"] = time.time()
            started = time.perf_counter()
            spec = importlib.util.spec_from_file_location("cifar_baseline", "/opt/workload/baseline.py")
            baseline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline)
            record["phases"]["baseline_s"] = time.perf_counter() - started
            original_arguments = baseline.TrainingArguments

            def bounded_arguments(**kwargs):
                kwargs.update(
                    output_dir=f"/tmp/cifar-{index}",
                    logging_dir=f"/tmp/cifar-{index}/logs",
                    max_steps=10,
                    eval_strategy="no",
                    save_strategy="no",
                    load_best_model_at_end=False,
                    logging_steps=1,
                    report_to="none",
                )
                return original_arguments(**kwargs)

            baseline.TrainingArguments = bounded_arguments
            baseline.set_seed()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable.")
            record["cuda_device"] = str(torch.cuda.get_device_properties(0))
            record["cuda_started_at"] = time.time()
            trainer, model = baseline.train_model(dataset, dataset)
            torch.cuda.synchronize()
            record.update(
                status="ok",
                steps=trainer.state.global_step,
                log_history=trainer.state.log_history,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                parameter_sum=float(sum(p.detach().float().sum().cpu() for p in model.parameters())),
                cuda_finished_at=time.time(),
            )
        except Exception as exc:  # noqa: BLE001
            record.update(status="error", error=repr(exc))
        record["finished_at"] = time.time()
        encoded = json.dumps(record)
        Path(f"/logs/artifacts/cifar-{index}.json").write_text(encoded)
        print("NATURAL_GPU_RESULT=" + encoded, flush=True)
        if record["status"] != "ok":
            raise SystemExit(1)

    return execute


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    prepare({"index": parser.parse_args().index})()
