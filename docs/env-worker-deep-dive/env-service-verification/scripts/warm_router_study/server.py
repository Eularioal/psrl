"""Serve a controlled fixed-step extension of the local PostTrain GPU protocol."""

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import socketserver
import sys
import time
from pathlib import Path

from metadata_cache import manifest_distributions

sys.path.insert(0, "/joint-study")
from worker import MODELS, Worker  # noqa: E402


class Service(Worker):
    """Retain frozen base models, resetting every command's norm and optimizer."""

    def progress(self, value):
        started = time.monotonic()
        temporary = self.progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, allow_nan=False))
        temporary.replace(self.progress_path)
        return time.monotonic() - started

    def ensure(self, name):
        if name not in MODELS:
            raise ValueError("Unsupported fixed model.")
        size = self.model_bytes(name)
        if size + self.activation > self.budget:
            raise ValueError("Model exceeds reserved capacity.")
        hit = name in self.cache
        if not hit:
            if self.resident_bytes() + size + self.activation > self.budget:
                raise ValueError("This experiment never evicts an existing model implicitly.")
            path = self.assets / "models" / name
            tokenizer = self.tokenizer_class.from_pretrained(path, local_files_only=True)
            model = (
                self.model_class.from_pretrained(
                    path,
                    local_files_only=True,
                    torch_dtype=self.torch.bfloat16,
                    attn_implementation="eager",
                )
                .cuda()
                .eval()
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            norm = dict(model.named_parameters())["model.norm.weight"]
            self.cache[name] = dict(model=model, tokenizer=tokenizer, norm=norm.detach().cpu().clone(), bytes=size)
        self.torch.cuda.synchronize()
        return hit, size

    def execute(self, request):
        torch = self.torch
        name, command_id = request["model"], request["id"]
        steps = request.get("steps", 2)
        if type(steps) is not int or not 1 <= steps <= 8192:
            raise ValueError("Bounded commands require 1..8192 integer updates.")
        progress_every = request.get("progress_every", self.progress_every)
        if type(progress_every) is not int or not 1 <= progress_every <= 256:
            raise ValueError("Progress reports require an integer interval in 1..256 updates.")
        started = time.monotonic()
        preparation_progress_s = self.progress(
            dict(id=command_id, phase="rebuild", total_steps=steps, completed_steps=0)
        )
        torch.cuda.reset_peak_memory_stats()
        hit, model_bytes = self.ensure(name)
        load_s = time.monotonic() - started
        entry = self.cache[name]
        model, tokenizer = entry["model"], entry["tokenizer"]
        data = (self.assets / "data" / "gsm8k" / "public" / "rows.json").resolve()
        raw = data.read_bytes()
        batches = []
        for row in json.loads(raw)[:2]:
            prompt = tokenizer.encode(row["prompt"], add_special_tokens=True)[-192:]
            answer = tokenizer.encode(row["answer"], add_special_tokens=False)[:64]
            inputs = torch.tensor([prompt + answer], device="cuda")
            labels = inputs.clone()
            labels[:, : len(prompt)] = -100
            batches.append((inputs, labels))
        if len(batches) != 2:
            raise ValueError("Exactly two public rows required.")
        parameter = dict(model.named_parameters())["model.norm.weight"]
        parameter.data.copy_(entry["norm"].to(device="cuda"))
        parameter.grad = None
        parameter.requires_grad_(True)
        torch.manual_seed(42)
        optimizer = torch.optim.AdamW([parameter], lr=0.001)
        torch.cuda.synchronize()
        computed, samples, losses = time.monotonic(), [], []
        chunk_started, chunk_completed = computed, 0
        chunk_steps, wall_chunk_s = [], []
        progress_write_s = self.progress(
            dict(
                id=command_id,
                phase="compute",
                total_steps=steps,
                completed_steps=0,
                sample_step_s=[],
                chunk_steps=[],
                wall_chunk_s=[],
                progress_every=progress_every,
                observed_at=computed,
                step_started=computed,
                compute_started=computed,
                compute_elapsed_s=0,
            )
        )
        for index in range(steps):
            tick = time.monotonic()
            inputs, labels = batches[index % 2]
            loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss.")
            losses.append(float(loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            samples.append(elapsed)
            if (index + 1) % progress_every == 0 or index + 1 == steps:
                observed_at = time.monotonic()
                chunk_steps.append(index + 1 - chunk_completed)
                wall_chunk_s.append(observed_at - chunk_started)
                chunk_started, chunk_completed = observed_at, index + 1
                # The next chunk includes this publication's wall time.
                progress_write_s += self.progress(
                    dict(
                        id=command_id,
                        phase="compute",
                        total_steps=steps,
                        completed_steps=index + 1,
                        sample_step_s=samples[-32:],
                        chunk_steps=chunk_steps[-32:],
                        wall_chunk_s=wall_chunk_s[-32:],
                        progress_every=progress_every,
                        observed_at=observed_at,
                        step_started=observed_at,
                        compute_started=computed,
                        compute_elapsed_s=observed_at - computed,
                        progress_write_s=progress_write_s,
                    )
                )
        compute_s = time.monotonic() - computed
        digest = hashlib.sha256(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        parameter.grad = None
        parameter.requires_grad_(False)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        if peak_mb > self.budget / 1024**2:
            raise RuntimeError("Observed allocated memory exceeded reservation.")
        result = dict(
            id=command_id,
            status="passed",
            model=name,
            steps=steps,
            optimizer_updates=steps,
            started_monotonic=started,
            finished_monotonic=time.monotonic(),
            compute_started=computed,
            cache_hit=hit,
            load_s=load_s,
            compute_s=compute_s,
            core_compute_s=sum(samples),
            progress_write_s=progress_write_s,
            progress_every=progress_every,
            chunk_steps=chunk_steps,
            wall_chunk_s=wall_chunk_s,
            wall_chunk_total_s=sum(wall_chunk_s),
            total_s=time.monotonic() - started,
            step_s=samples,
            norm_sha256=digest,
            losses_sha256=hashlib.sha256(json.dumps(losses).encode()).hexdigest(),
            first_loss=losses[0],
            last_loss=losses[-1],
            data_sha256=hashlib.sha256(raw).hexdigest(),
            model_bytes=model_bytes,
            peak_allocated_mb=peak_mb,
            cache_keys=list(self.cache),
            cache_bytes=self.resident_bytes(),
        )
        completion_progress_s = self.progress(
            dict(
                id=command_id,
                phase="complete",
                completed_steps=steps,
                total_steps=steps,
                observed_at=time.monotonic(),
                progress_every=progress_every,
            )
        )
        result["progress_write_total_s"] = preparation_progress_s + progress_write_s + completion_progress_s
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--budget-mb", type=int, required=True)
    parser.add_argument("--assets", type=Path, default=Path("/assets"))
    parser.add_argument("--gpu-uuid", default=os.environ.get("EXPECTED_GPU_UUID"))
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.progress_every <= 256:
        raise ValueError("Progress reports require an integer interval in 1..256 updates.")
    # Limit the import workaround to this process and its startup phase.
    original_mapping = importlib.metadata.packages_distributions
    importlib.metadata.packages_distributions = manifest_distributions
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        worker = Service(args)
    finally:
        importlib.metadata.packages_distributions = original_mapping
        faulthandler.cancel_dump_traceback_later()
    worker.progress_path = args.socket.parent / "progress.json"
    worker.progress_every = args.progress_every
    worker.maximum_budget = worker.budget

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request = json.loads(self.rfile.readline())
            try:
                operation = request.get("op", "run")
                if operation == "ping":
                    result = dict(
                        status="passed",
                        gpu_uuid=worker.gpu_uuid,
                        python=sys.executable,
                        torch=worker.torch.__version__,
                        transformers=importlib.metadata.version("transformers"),
                        cache_keys=list(worker.cache),
                        cache_bytes=worker.resident_bytes(),
                        budget_bytes=worker.budget,
                        workspace_bytes=worker.activation,
                    )
                elif operation == "set_budget":
                    budget_mb = request["budget_mb"]
                    if type(budget_mb) is not int or not 1 <= budget_mb <= worker.maximum_budget // 1024**2:
                        raise ValueError("The service budget must remain within its original reservation.")
                    if worker.resident_bytes() + worker.activation > budget_mb * 1024**2:
                        raise ValueError("Clear retained models before reducing the service budget.")
                    worker.budget = budget_mb * 1024**2
                    result = dict(status="passed", budget_bytes=worker.budget)
                elif operation == "admit":
                    model_bytes = worker.model_bytes(request["model"])
                    required = worker.activation + (0 if request["model"] in worker.cache else model_bytes)
                    available = worker.budget - worker.resident_bytes()
                    result = dict(
                        status="passed",
                        required_bytes=required,
                        available_bytes=available,
                        admits=required <= available,
                    )
                elif operation == "load":
                    started = time.monotonic()
                    hit, size = worker.ensure(request["model"])
                    result = dict(status="passed", cache_hit=hit, model_bytes=size, load_s=time.monotonic() - started)
                elif operation == "clear":
                    elapsed, removed = worker.clear()
                    result = dict(status="passed", clear_s=elapsed, removed=removed)
                else:
                    result = worker.execute(request)
            except Exception as exc:
                result = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.wfile.write((json.dumps(result, allow_nan=False) + "\n").encode())

    args.socket.unlink(missing_ok=True)
    with socketserver.UnixStreamServer(str(args.socket), Handler) as server:
        try:
            server.serve_forever()
        finally:
            worker.clear()


if __name__ == "__main__":
    main()
