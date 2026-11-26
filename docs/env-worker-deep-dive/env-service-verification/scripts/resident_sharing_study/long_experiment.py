"""Compare longer model generation and real PostTrain tools under a fixed resource budget."""

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
import traceback
from pathlib import Path

from experiment import Experiment, wait_json
from protocol import read, save


def choose_steps(seconds_per_step, target_s):
    """Freeze a bounded update count from one isolated calibration."""
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise ValueError("Calibration must provide a finite positive step duration.")
    return max(128, min(4096, 32 * math.ceil(target_s / seconds_per_step / 32)))


class LongExperiment(Experiment):
    """Reuse actual leases and cleanup while lengthening both useful workloads."""

    command_timeout_s = 280
    command_deadline_s = 300
    ready_timeout_s = 240
    environment_memory_mb = 32768
    environment_gpu_mb = 16384
    worker_script = "posttrain_worker.py"
    task_snapshot = "executed-posttrain.py"
    boundary = "Real vLLM generation and frozen-base PostTrain norm tuning. No PPO, MPS, or model offload."

    def __init__(self, args):
        super().__init__(args)
        self.batch_number = 0
        self.service_root = self.local / "batch-0"
        self.trial_root = self.out / "batch-0"

    def task_source(self):
        return Path(self.args.repo) / "examples/posttrain_bench/local_model.py"

    def server_options(self):
        return {"output_tokens": 768, "max_model_len": 2048, "gpu_memory_utilization": 0.25}

    def extra_mounts(self, role):
        return [f"{self.args.env_model}:/model:ro"]

    def study_plan(self):
        return {
            "workload": "PostTrain GSM8K with Qwen3-4B-Base and fixed final-norm AdamW updates",
            "rounds": self.args.rounds,
            "policies": ["serial", "existing", "cooperative"],
            "profiles": ["spaced", "burst"],
            "requests_per_trial": 3,
            "tokens_per_request": 768,
            "calibration_steps": 64,
            "step_bounds": [128, 4096],
            "target_environment_s": self.args.target_env_s,
            "fixed_updates": self.args.updates,
            "tool_preparation": "Train and verifier load concurrently under separate command leases.",
            "trial_timeout_s": 150,
            "total_timeout_s": 1500,
            "command_deadline_s": 300,
            "tool_batch_timeout_s": 260,
            "candidate_pool": "Only the explicitly shared model GPU is configured.",
            "mps": False,
            "model_offload": False,
        }

    async def prepare(self):
        started = time.monotonic()
        await super().prepare()
        self.evidence["initialization_s"] = time.monotonic() - started
        self.evidence["environment_model"] = str(self.args.env_model)
        self.evidence["inference_config"] = self.server_options()
        self.evidence["data_sha256"] = {
            str(path.relative_to(self.args.data)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.args.data.rglob("*.json")
        }
        self.samples = []
        self.evidence["sampling_clock"] = {"realtime_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}
        self.sampling = asyncio.create_task(self.sample_gpu())
        self.record()

    async def start_environment(self, role):
        if role == "train":
            self.batch_started = time.monotonic()
        return await super().start_environment(role)

    async def start_environments(self):
        """Overlap independent model loading within the existing command deadline."""
        return await asyncio.gather(self.start_environment("train"), self.start_environment("verify"))

    async def stop_tools(self):
        for directory in (self.train_control, self.verify_control):
            save(directory / "request.json", {"stop": True})
        for command in self.commands:
            result = await asyncio.wait_for(asyncio.shield(command), 15)
            if result.return_code:
                raise RuntimeError(f"A tool exited with code {result.return_code!r}: {result.stderr!r}.")
        for environment in self.environments:
            await asyncio.wait_for(environment.stop(delete=True), 20)
        for handle in self.handles:
            await asyncio.wait_for(handle.release(), 20)
        owner_pids = set(self.evidence["model_resident"][self.gpu_index]["pids"])
        for _ in range(20):
            remaining = self.probe.read_gpu_snapshot([self.gpu_index])
            if set(remaining[self.gpu_index]["pids"]) <= owner_pids:
                break
            await asyncio.sleep(0.25)
        else:
            raise RuntimeError("A finished tool batch still has GPU processes.")
        self.evidence.setdefault("released_tool_batches", []).append(
            {"batch": self.batch_number, "gpu": remaining, "elapsed_s": time.monotonic() - self.batch_started}
        )
        self.commands.clear()
        self.environments.clear()
        self.handles.clear()
        self.record()

    async def sample_gpu(self):
        since_us = 0
        while True:
            try:
                sample = self.probe.read_process_snapshot([self.gpu_index], since_us)
                self.samples.append({"at_ns": time.monotonic_ns(), "batch": self.batch_number, "gpu": sample})
                timestamps = [row["timestamp_us"] for row in sample[self.gpu_index]["samples"]]
                since_us = max([since_us, *timestamps])
            except Exception as exc:
                self.samples.append({"at_ns": time.monotonic_ns(), "missing": repr(exc)})
            await asyncio.sleep(0.25)

    async def trial(self, policy, steps, profile, repetition, *, warmup=False):
        trial_id = f"{steps}-{profile}-{repetition}-{policy}"
        request = {
            "trial_id": trial_id,
            "steps": steps,
            "checkpoint": trial_id,
            "go": False,
            "pause": False,
            "epoch": 0,
        }
        save(self.train_control / "request.json", request)
        await wait_json(self.train_control / "state.json", lambda value: value.get("trial_id") == trial_id)
        start_ns = time.monotonic_ns()
        queries = []
        if policy != "serial" or warmup:
            save(self.train_control / "request.json", {**request, "go": True})
            request["go"] = True
        for index in range(0 if warmup else 3):
            offset = 0.2 + (index * self.interval_s if profile == "spaced" else 0)
            await asyncio.sleep(max(0, offset - (time.monotonic_ns() - start_ns) / 1e9))
            submitted_ns = time.monotonic_ns()
            receipt = None
            if policy == "cooperative":
                request.update(pause=True, epoch=index + 1)
                save(self.train_control / "request.json", request)
                async with asyncio.timeout(2):
                    while True:
                        state = read(self.train_control / "state.json")
                        finished = read(self.train_control / "result.json")
                        if finished and finished.get("trial_id") == trial_id:
                            receipt = finished
                            break
                        if (
                            state
                            and state.get("trial_id") == trial_id
                            and state.get("state") == "paused"
                            and state.get("epoch") == index + 1
                        ):
                            receipt = state
                            break
                        await asyncio.sleep(0.001)
            admitted_ns = time.monotonic_ns()
            result = await self.server.query.remote(index)
            ended_ns = time.monotonic_ns()
            queries.append(
                {
                    "scheduled_ns": start_ns + int(offset * 1e9),
                    "submitted_ns": submitted_ns,
                    "admitted_ns": admitted_ns,
                    "ended_ns": ended_ns,
                    "receipt": receipt,
                    **result,
                }
            )
            if policy == "cooperative":
                request["pause"] = False
                save(self.train_control / "request.json", request)
                if receipt["state"] == "paused":
                    await wait_json(
                        self.train_control / "state.json",
                        lambda value, epoch=index + 1: (
                            value.get("trial_id") == trial_id
                            and value.get("state") == "resumed"
                            and value.get("epoch") == epoch
                        ),
                        timeout=2,
                    )
        if policy == "serial" and not warmup:
            request["go"] = True
            save(self.train_control / "request.json", request)
        training = await wait_json(self.train_control / "result.json", lambda value: value.get("trial_id") == trial_id)
        complete_ns = max([training["ended_ns"], *(query["ended_ns"] for query in queries)])
        save(self.verify_control / "request.json", {"trial_id": trial_id, "checkpoint": f"/submissions/{trial_id}"})
        verified = await wait_json(
            self.verify_control / "result.json", lambda value: value.get("trial_id") == trial_id
        )
        if training["optimizer_updates"] != steps or any(len(query["token_ids"]) != 768 for query in queries):
            raise RuntimeError("The policies performed different amounts of useful work.")
        row = {
            "id": trial_id,
            "policy": policy,
            "updates": steps,
            "profile": profile,
            "repetition": repetition,
            "warmup": warmup,
            "started_ns": start_ns,
            "makespan_s": (complete_ns - start_ns) / 1e9,
            "training": training,
            "verified": verified,
            "queries": queries,
            "gpu": self.probe.read_gpu_snapshot([self.gpu_index]),
        }
        self.evidence["trials"].append(row)
        self.record()
        print(json.dumps({"trial": trial_id, "seconds": row["makespan_s"], "score": verified["reward"]}), flush=True)
        return row

    async def run_trials(self):
        self.interval_s = 1.5 * statistics.mean(row["engine_duration_s"] for row in self.evidence["warmup_requests"])
        self.evidence["spaced_request_interval_s"] = self.interval_s
        await self.trial("serial", 16, "warmup", 0, warmup=True)
        calibration = await self.trial("serial", 64, "calibration", 0, warmup=True)
        times = calibration["training"]["update_times_ns"]
        seconds_per_step = (times[-1] - times[0]) / (len(times) - 1) / 1e9
        steps = self.args.updates or choose_steps(seconds_per_step, self.args.target_env_s)
        self.evidence["calibration"] = {
            "seconds_per_step": seconds_per_step,
            "selected_updates": steps,
            "predicted_environment_s": steps * seconds_per_step,
        }
        self.record()
        print(json.dumps({"calibration": self.evidence["calibration"]}), flush=True)
        policies = ("serial", "existing", "cooperative")
        for repetition in range(self.args.rounds):
            for profile in ("spaced", "burst"):
                if self.batch_number:
                    self.service_root = self.local / f"batch-{self.batch_number}"
                    self.trial_root = self.out / f"batch-{self.batch_number}"
                    self.train_control, self.verify_control = await self.start_environments()
                    await self.trial("serial", 16, "warmup", self.batch_number, warmup=True)
                remaining_s = 260 - (time.monotonic() - self.batch_started)
                if remaining_s <= 0:
                    raise TimeoutError("Tool preparation exceeded its batch budget.")
                async with asyncio.timeout(remaining_s):
                    for policy in policies[repetition:] + policies[:repetition]:
                        async with asyncio.timeout(150):
                            await self.trial(policy, steps, profile, repetition)
                await self.stop_tools()
                self.batch_number += 1
        rows = [row for row in self.evidence["trials"] if not row["warmup"]]
        self.evidence["correctness"] = {
            "same_tensors": len({row["training"]["tensor_sha256"] for row in rows}) == 1,
            "same_scores": len({row["verified"]["reward"] for row in rows}) == 1,
            "same_tokens": len({tuple(query["token_ids"]) for row in rows for query in row["queries"]}) == 1,
        }
        if not all(self.evidence["correctness"].values()):
            raise RuntimeError("Outputs changed across policies despite fixed useful work.")
        self.evidence["passed"] = True

    async def cleanup(self):
        if hasattr(self, "sampling"):
            self.sampling.cancel()
            await asyncio.gather(self.sampling, return_exceptions=True)
            save(self.out / "gpu-samples.json", self.samples)
        await super().cleanup()


async def main(args):
    experiment = LongExperiment(args)
    started = time.monotonic()
    try:
        async with asyncio.timeout(1500):
            await experiment.prepare()
            await experiment.run_trials()
    except BaseException:
        experiment.evidence["error"] = traceback.format_exc()
        raise
    finally:
        await experiment.cleanup()
        experiment.evidence["elapsed_s"] = time.monotonic() - started
        experiment.record()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--env-model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--target-env-s", type=float, default=20, choices=(20,))
    parser.add_argument("--updates", type=int, choices=range(128, 4097, 32))
    parser.add_argument("--rounds", type=int, default=2, choices=(1, 2))
    asyncio.run(main(parser.parse_args()))
