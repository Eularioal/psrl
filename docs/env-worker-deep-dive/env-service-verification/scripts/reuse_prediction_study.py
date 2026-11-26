"""Measure causal GPU loan forecasts with real vLLM sleep and PostTrain borrowers."""

import argparse
import asyncio
import dataclasses
import hashlib
import os
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path

import ray
from auto_reservation_study import SCRIPT, Study
from psrl.workers.env_service.command_prediction import WorkloadKey
from psrl.workers.env_service.reuse_prediction import ReuseDelayPredictor, ReuseSample, additional_wait
from roofline_sweep import gpu_snapshot

sys.path.insert(0, str(SCRIPT / "resident_sharing_study"))
from inference import StudyServer  # noqa: E402


class PredictionStudy(Study):
    ray_gpu_count = 1

    def __init__(self, args):
        super().__init__(args)
        self.server = None
        self.predictor = ReuseDelayPredictor()

    async def prepare(self):
        await super().prepare()
        for path in (Path(__file__), SCRIPT / "real_gpu_lending.py", SCRIPT / "resident_sharing_study/inference.py"):
            shutil.copy2(path, self.out / "source" / path.name)
        self.code_hash = hashlib.sha256(
            b"".join(
                (self.out / "source" / name).read_bytes()
                for name in ("partition_worker.py", "task.py", "reuse_prediction_study.py", "real_gpu_lending.py")
            )
        ).hexdigest()
        self.data["boundary"] = (
            "One H20, real vLLM sleep/wake and managed PostTrain commands. "
            "Controlled return times, forecast before execution, no live PSRL rollout or trainer."
        )
        runtime = dict(
            env_vars=dict(
                PYTHONPATH=os.pathsep.join(
                    (
                        str(SCRIPT),
                        str(SCRIPT / "resident_sharing_study"),
                        str(self.args.repo),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
                RAY_ADDRESS=ray.get_runtime_context().gcs_address,
            )
        )
        self.server = (
            ray.remote(num_cpus=2, num_gpus=1, max_concurrency=16)(StudyServer)
            .options(runtime_env=runtime)
            .remote(str(self.args.inference_model), output_tokens=64, max_model_len=512)
        )
        self.data["inference_worker"] = await self.server.ready.remote()
        self.data["warm_query"] = await self.server.query.remote("warmup")
        self.resident = gpu_snapshot(self.args.gpu)
        self.env, self.handle, record = await self.create("command", "borrower", "b")
        self.data["borrower"] = record
        self.save()

    async def sleep_model(self):
        await self.server.pause_generation.remote(mode="wait", clear_cache=False)
        await self.server.nixl_sleep.remote(level=1)
        sleeping = gpu_snapshot(self.args.gpu)
        if sleeping["used_mib"] > self.resident["used_mib"] - 1024:
            raise RuntimeError("Model sleep did not release memory.")
        return sleeping

    async def restore_model(self):
        before = gpu_snapshot(self.args.gpu)
        if set(before["pids"]) - set(self.resident["pids"]):
            raise RuntimeError("Borrower GPU processes remain before restoration.")
        await self.server.nixl_wake_up.remote()
        await self.server.resume_generation.remote()

    async def cycle(self, index, profile, gap, *, calibration=False, force=False, proactive=True):
        key = WorkloadKey(
            self.code_hash,
            self.args.inference_model.name + "+" + self.args.model.name,
            f"gsm8k:steps={self.args.steps}:gap-profile={profile}:proactive={proactive}",
            "H20:exclusive-loan",
            "vllm-sleep-level1+managed-direct",
        )
        forecast = self.predictor.predict(key, now_s=time.monotonic(), max_extra_wait_s=1, proactive_restore=proactive)
        lend = calibration or force or (forecast is not None and forecast.release_recommended)
        row = dict(
            index=index,
            profile=profile,
            controlled_gap_s=gap,
            calibration=calibration,
            forced=force,
            proactive=proactive,
            forecast=dataclasses.asdict(forecast) if forecast else None,
            action="lend" if lend else "keep",
            key=dataclasses.asdict(key),
        )
        self.data["trials"].append(row)
        self.save()
        arrived, restored = asyncio.Event(), asyncio.Event()
        started = time.monotonic()
        row["started_s"] = started

        async def request():
            await asyncio.sleep(gap)
            row["request_at_s"] = time.monotonic()
            arrived.set()
            if lend:
                await restored.wait()
            row["service_at_s"] = time.monotonic()
            row["query"] = await self.server.query.remote(index)
            row["request_done_s"] = time.monotonic()

        requesting = asyncio.create_task(request())
        try:
            if not lend:
                # Keep A's reservation through its next request, then service B.
                await requesting
            loan_started = time.monotonic()
            row["sleeping_gpu"] = await self.sleep_model()
            saved = time.monotonic()
            folder = self.out / "borrower/b" / f"cycle-{index}"
            folder.mkdir()
            row["borrower"] = await self.command(self.env, "train", folder)
            returned = time.monotonic()
            if lend and not proactive:
                await arrived.wait()
            restoring = time.monotonic()
            await self.restore_model()
            ready = time.monotonic()
            restored.set()
            await requesting
            sample = ReuseSample(
                saved - loan_started, returned - saved, ready - restoring, row["request_at_s"] - started
            )
            row.update(
                sample=dataclasses.asdict(sample),
                model_ready_s=ready,
                observed_extra_wait_s=row["service_at_s"] - row["request_at_s"],
                cycle_s=time.monotonic() - started,
            )
            if lend:
                # Include measured dispatch slack before the first sleep RPC.
                row["formula_wait_s"] = additional_wait(
                    saved - started,
                    returned - saved,
                    ready - restoring,
                    sample.total_gap_s,
                    proactive_restore=proactive,
                )
                row["formula_error_s"] = row["observed_extra_wait_s"] - row["formula_wait_s"]
                if forecast is not None:
                    row["prediction_error_s"] = row["observed_extra_wait_s"] - forecast.point_wait_s
                    row["envelope_exceeded"] = row["observed_extra_wait_s"] > forecast.envelope_wait_s + 0.05
            if row["query"]["token_ids"] != self.data["warm_query"]["token_ids"]:
                raise RuntimeError("Restored inference produced different deterministic tokens.")
            self.predictor.observe(key, sample, now_s=time.monotonic())
            self.save()
        except BaseException:
            self.predictor.invalidate(key, now_s=time.monotonic())
            row["profile_invalidated"] = True
            self.save()
            raise
        finally:
            if not requesting.done():
                requesting.cancel()
                await asyncio.gather(requesting, return_exceptions=True)

    async def cleanup(self):
        if self.server is not None:
            try:
                await asyncio.wait_for(self.server.stop.remote(), 20)
            except Exception as exc:
                self.data["cleanup_errors"].append(dict(kind="model", error=repr(exc)))
            ray.kill(self.server, no_restart=True)
        await super().cleanup()


async def main(args):
    study = PredictionStudy(args)
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    try:
        async with asyncio.timeout(args.timeout):
            await study.prepare()
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
            study.data["status"] = "running"
            study.save()
            plan = [
                *(dict(profile="short", gap=6, calibration=True) for _ in range(3)),
                *(dict(profile="long", gap=70, calibration=True) for _ in range(3)),
                dict(profile="short", gap=6, force=True),
                dict(profile="long", gap=70),
                dict(profile="long", gap=6),
                dict(profile="long", gap=6),
                dict(profile="on-demand", gap=70, force=True, proactive=False),
            ]
            for index, options in enumerate(plan):
                await study.cycle(index, **options)
            hashes = {r["borrower"]["result"]["tensor_sha256"] for r in study.data["trials"]}
            if len(hashes) != 1:
                raise RuntimeError("Borrower updates differ across scheduling choices.")
            study.data.update(status="passed", hashes=list(hashes))
    except BaseException:
        study.data.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        await study.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "output", "model", "data", "inference-model"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1200)
    asyncio.run(main(parser.parse_args()))
