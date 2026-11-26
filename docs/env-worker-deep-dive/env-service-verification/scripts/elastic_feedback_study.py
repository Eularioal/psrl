"""Measure feedback requests and inference bursts during real PostTrain GPU commands."""

import argparse
import asyncio
import json
import shutil
import time
import traceback
import uuid
from pathlib import Path

from elastic_partition_study import Study
from inference import StudyServer
from psrl.workers.env_service.elastic_partition import ElasticPartition, PartitionDevice


class FeedbackServer(StudyServer):
    """Generate task explanations from public task descriptions and execution results."""

    async def explain(self, prompt):
        from vllm import SamplingParams

        begin = time.monotonic()
        first = None
        result = None
        async for result in self.engine.generate(
            prompt,
            SamplingParams(
                temperature=0, max_tokens=self.output_tokens, min_tokens=self.output_tokens, ignore_eos=True
            ),
            request_id=f"feedback-{uuid.uuid4().hex}",
        ):
            if first is None:
                first = time.monotonic()
        if result is None or not result.finished:
            raise RuntimeError("Task inference did not finish.")
        return dict(
            text=result.outputs[0].text,
            token_ids=list(result.outputs[0].token_ids),
            first_token_s=first - begin,
            service_s=time.monotonic() - begin,
        )


class FeedbackStudy(Study):
    """Keep task work fixed while measuring request return during nonpreemptible commands."""

    server_type = FeedbackServer

    async def calibrate(self):
        server = self.servers[str(self.args.gpus[2])]
        rows = []
        for count in (1, 2, 4, 8, 16):
            begin = time.monotonic()
            results = await asyncio.gather(*(server.explain.remote(self.prompt(0, "initial")) for _ in range(count)))
            if any(len(result["token_ids"]) != self.args.tokens for result in results):
                raise RuntimeError("Calibration output lengths differ.")
            elapsed = time.monotonic() - begin
            rows.append(dict(concurrency=count, elapsed_s=elapsed, tokens_per_s=count * self.args.tokens / elapsed))
            self.evidence["inference_calibration"] = rows
            self.save()
        eligible = [row for row in rows if row["elapsed_s"] <= rows[0]["elapsed_s"] * 1.25]
        self.limit = max(eligible, key=lambda row: row["tokens_per_s"])["concurrency"]
        self.evidence["inference_limit"] = self.limit
        self.evidence["calibration_boundary"] = (
            "One model and fixed prompt/output lengths, largest measured throughput within 25 percent batch latency. "
            "This is a sampled service curve, not an SM saturation measurement or a guaranteed latency bound."
        )
        self.save()

    def prompt(self, job, phase, feedback=None):
        return (
            "You are assisting a bounded post-training experiment. The task uses Qwen3-4B-Base on GSM8K, "
            "updates only model.norm.weight with AdamW at learning rate 0.001 on two public examples, "
            "then independently evaluates a saved adapter. Explain the next checks and interpret available evidence. "
            "Do not invent an accuracy measurement.\n"
            f"Job {job}, phase {phase}. Public execution feedback: {json.dumps(feedback)}\n"
            "Explain your reasoning in detail:\n"
        )

    async def feedback_trial(self, policy, repetition):
        args = self.args
        row = dict(id=f"{repetition}-{policy}", policy=policy, commands=[], queries=[], transitions=[], episodes=[])
        self.evidence["trials"].append(row)
        started = time.monotonic()
        baseline = self.probe.read_gpu_snapshot(args.gpus)

        async def sleep(device):
            begin = time.monotonic()
            await self.servers[device].pause_generation.remote(mode="wait", clear_cache=False)
            await self.servers[device].nixl_sleep.remote(level=1)
            snapshot = self.probe.read_gpu_snapshot([int(device)])[int(device)]
            if snapshot["used_mb"] > baseline[int(device)]["used_mb"] - 1024:
                raise RuntimeError("The model did not release its memory.")
            row["transitions"].append(
                dict(kind="sleep", device=device, at_s=begin - started, duration_s=time.monotonic() - begin)
            )
            self.save()

        async def wake(device):
            begin = time.monotonic()
            snapshot = self.probe.read_gpu_snapshot([int(device)])[int(device)]
            if set(snapshot["pids"]) - set(baseline[int(device)]["pids"]):
                raise RuntimeError("Environment processes remain before model wake.")
            await self.servers[device].nixl_wake_up.remote()
            await self.servers[device].resume_generation.remote()
            row["transitions"].append(
                dict(kind="wake", device=device, at_s=begin - started, duration_s=time.monotonic() - begin)
            )
            self.save()

        controller = ElasticPartition(
            [PartitionDevice(str(i), args.slots, 49152, i in args.gpus[2:]) for i in args.gpus],
            sleep,
            wake,
            dynamic=policy == "dynamic",
            resident_sharing=policy == "resident",
            inference_limit=self.limit,
            transition_timeout_s=270,
        )

        async def query(job, phase, feedback=None):
            begin = time.monotonic()
            async with controller.inference() as device:
                admitted = time.monotonic()
                result = await self.servers[device].explain.remote(self.prompt(job, phase, feedback))
                row["queries"].append(
                    dict(
                        job=job,
                        phase=phase,
                        device=device,
                        at_s=begin - started,
                        wait_s=admitted - begin,
                        latency_s=time.monotonic() - begin,
                        result=result,
                    )
                )
                self.save()
                return result

        async def job(index):
            await query(index, "initial")
            for mode in ("train", "verify"):
                begin = time.monotonic()
                steps = args.slow_steps if index == args.jobs - 1 else args.steps
                async with controller.environment(16384) as device:
                    admitted = time.monotonic()
                    result = await self.command(device, row["id"], index, mode, steps=steps)
                    row["commands"].append(
                        dict(
                            job=index,
                            mode=mode,
                            device=device,
                            at_s=begin - started,
                            wait_s=admitted - begin,
                            service_s=time.monotonic() - admitted,
                            result=result,
                        )
                    )
                    self.save()
                feedback = {
                    key: value
                    for key, value in result.items()
                    if key in ("updates", "tensor_sha256", "reward", "examples")
                }
                await query(index, f"after-{mode}", feedback)
            row["episodes"].append(dict(job=index, finished_s=time.monotonic() - started))
            self.save()

        async def burst():
            await asyncio.sleep(args.burst_at)
            await asyncio.gather(*(query(i, "burst") for i in range(args.burst_requests)))

        async with asyncio.TaskGroup() as group:
            monitor = group.create_task(controller.run())
            jobs = [group.create_task(job(i)) for i in range(args.jobs)]
            burst_task = group.create_task(burst())
            await asyncio.gather(*jobs)
            row["all_episode_results_ready_s"] = time.monotonic() - started
            await burst_task
            while any(d.inference and d.state != "inference" for d in controller.devices):
                if any(d.state == "quarantined" for d in controller.devices):
                    raise RuntimeError("A device was quarantined.")
                await asyncio.sleep(0.1)
            monitor.cancel()
        row["total_with_restore_s"] = time.monotonic() - started
        row["events"] = [{**e, "at_s": e["at"] - started} for e in controller.events]
        row["gpu_after"] = self.probe.read_gpu_snapshot(args.gpus)
        self.save()


async def main(args):
    study = FeedbackStudy(args)
    try:
        async with asyncio.timeout(args.total_timeout):
            await study.prepare()
            shutil.copy2(__file__, study.out / "source" / Path(__file__).name)
            study.evidence["boundary"] = (
                "Real PSRL EnvRouter and model controls with PostTrain train/verify and generated feedback. "
                "The bounded workflow fixes commands and updates. "
                "No free-form agent, SMG, PPO, or optimizer on trajectories."
            )
            await study.calibrate()
            for index, policy in enumerate(args.modes):
                async with asyncio.timeout(args.trial_timeout):
                    await study.feedback_trial(policy, index)
            outputs = []
            for trial in study.evidence["trials"]:
                outputs.append(
                    {
                        f"{c['job']}-{c['mode']}": c["result"].get("tensor_sha256", c["result"].get("reward"))
                        for c in trial["commands"]
                    }
                )
            if not all(value == outputs[0] for value in outputs):
                raise RuntimeError("Task outputs differ across policies.")
            study.evidence.update(status="passed", task_outputs_identical=True)
    except BaseException:
        study.evidence.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        await study.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--env-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=448)
    parser.add_argument("--slow-steps", type=int, default=1792)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--burst-at", type=float, default=55)
    parser.add_argument("--burst-requests", type=int, default=64)
    parser.add_argument(
        "--modes", nargs="+", choices=("fixed", "resident", "dynamic"), default=["fixed", "resident", "dynamic"]
    )
    parser.add_argument("--trial-timeout", type=float, default=540)
    parser.add_argument("--total-timeout", type=float, default=1900)
    asyncio.run(main(parser.parse_args()))
