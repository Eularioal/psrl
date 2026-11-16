"""Run a preregistered longer hot-queue placement comparison through PSRL."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

A, B = "Qwen3-4B-Base", "Qwen3-1.7B-Base"
TARGET_STEPS, BLOCKER_STEPS, PROFILE_STEPS, REPORT_EVERY, ARRIVAL_STEPS = 1536, 2048, 128, 16, 64


def load_source(name, path):
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module, source


def payload(command_id, model, steps):
    return dict(id=command_id, model=model, steps=steps, progress_every=REPORT_EVERY)


async def study(context, args, policy, predictor, policy_source, predictor_source):
    from telemetry import inspect_owned

    request, record, save = context["request"], context["record"], context["save"]
    for name, source in (("placement-policy", policy_source), ("step-predictor", predictor_source)):
        (context["root"] / (name + ".py")).write_bytes(source)
        record[name.replace("-", "_") + "_sha256"] = hashlib.sha256(source).hexdigest()
    record["protocol"] = dict(
        name="controlled PostTrain fixed-step extension",
        target_model=A,
        blocker_model=B,
        target_steps=TARGET_STEPS,
        blocker_steps=BLOCKER_STEPS,
        profile_steps=PROFILE_STEPS,
        progress_every=REPORT_EVERY,
        arrival_after_completed_steps=ARRIVAL_STEPS,
        switch_margin_s=2.0,
        public_examples=2,
        target_parameter="model.norm.weight",
        optimizer="AdamW lr=0.001",
        token_limits=[192, 64],
        primary_timing="driver before blocker submission through both RPC responses",
        note="Controlled repeated epochs, not official default benchmark or an LLM rollout.",
    )
    record.update(round=args.round, profiles=[], blocks=[])
    record["container_inspection_before"] = await inspect_owned(record["tag"])
    for index in (0, 1):
        await request(index, dict(op="clear"))
    for index, model in ((1, A), (0, A), (0, B)):
        result = await request(index, payload(f"profile-{index}-{model}", model, PROFILE_STEPS))
        record["profiles"].append(result)
        save()
    profiles_a = [value for value in record["profiles"] if value["model"] == A]
    rates = [
        elapsed / count
        for value in profiles_a
        for count, elapsed in zip(value["chunk_steps"][1:], value["wall_chunk_s"][1:], strict=True)
    ]
    execution_s = statistics.median(rates) * TARGET_STEPS
    rebuild_s = statistics.median(value["load_s"] for value in profiles_a)
    record["predictor"] = dict(
        method="median complete chunk wall time per step, recent eight chunks, first warmup chunk excluded",
        execution_estimate_s=execution_s,
        rebuild_estimate_s=rebuild_s,
        decision_mode="median_point_with_2s_margin",
        interval_role="audit only, empirical padded quantiles with no coverage guarantee",
    )
    save()
    for label in args.policies:
        if context["remaining"]() < 145:
            raise TimeoutError("Insufficient fixed budget for another long block.")
        await request(1, dict(op="clear"))
        block_id = f"r{args.round}-{label}"
        blocker_id = block_id + "-blocker"
        block_started = time.monotonic()
        blocker_task = asyncio.create_task(request(0, payload(blocker_id, B, BLOCKER_STEPS)))
        try:
            while True:
                if time.monotonic() - block_started > 30:
                    raise TimeoutError("No bounded arrival progress observation.")
                if blocker_task.done():
                    await blocker_task
                    raise RuntimeError("Blocker finished before the intended arrival observation.")
                try:
                    progress = json.loads(context["progress_file"](0).read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    await asyncio.sleep(0.01)
                    continue
                if progress.get("id") == blocker_id and progress.get("completed_steps", 0) >= ARRIVAL_STEPS:
                    break
                await asyncio.sleep(0.01)
            arrival = time.monotonic()
            forecast = predictor.predict_remaining(
                progress["chunk_steps"][1:],
                progress["wall_chunk_s"][1:],
                progress["completed_steps"],
                BLOCKER_STEPS,
                max(0, arrival - progress["observed_at"]),
                report_every_steps=REPORT_EVERY,
            )
            point = policy.DurationInterval
            remaining = point(forecast.estimate_s, forecast.estimate_s) if forecast.estimate_s is not None else None
            placement_request = policy.PlacementRequest(
                A, profiles_a[0]["model_bytes"], point(rebuild_s, rebuild_s), point(execution_s, execution_s)
            )
            devices = (
                policy.DeviceSnapshot("hot", frozenset({A, B}), 1024**3, True, remaining),
                policy.DeviceSnapshot("cold", frozenset(), 19 * 1024**3),
            )
            decision = dataclasses.asdict(policy.choose_placement(placement_request, devices, switch_margin_s=2.0))
            decision["mode"] = "median_point_with_2s_margin"
            selected = int(label == "on" and decision["action"] == "rebuild")
            block = dict(
                id=block_id,
                round=args.round,
                policy=label,
                scenario="long",
                enabled=label == "on",
                selected=selected,
                observed_progress=progress,
                progress_estimate=dataclasses.asdict(forecast),
                decision=decision,
                decision_overhead_s=time.monotonic() - arrival,
            )
            record["blocks"].append(block)
            save()
            target, blocker = await asyncio.gather(
                request(selected, payload(block_id + "-target", A, TARGET_STEPS)), blocker_task
            )
            done = time.monotonic()
            block.update(
                target=target,
                blocker=blocker,
                primary_makespan_s=done - block_started,
                driver_join_both_s=done - arrival,
                target_latency_s=target["finished_monotonic"] - arrival,
                target_wait_s=max(0, target["started_monotonic"] - arrival),
                tail_makespan_s=max(target["finished_monotonic"], blocker["finished_monotonic"]) - arrival,
                server_work_makespan_s=max(target["finished_monotonic"], blocker["finished_monotonic"])
                - block_started,
                blocker_remaining_actual_s=max(0, blocker["finished_monotonic"] - arrival),
            )
            save()
            clear_started = time.monotonic()
            cleared = await request(1, dict(op="clear"))
            block.update(cold_clear_s=cleared["clear_s"], cold_clear_rpc_s=time.monotonic() - clear_started)
            block["closed_cycle_s"] = block["primary_makespan_s"] + block["cold_clear_rpc_s"]
            save()
        finally:
            if not blocker_task.done():
                blocker_task.cancel()
                await asyncio.gather(blocker_task, return_exceptions=True)
    record["container_inspection_after"] = await inspect_owned(record["tag"])
    if len(record["blocks"]) == 2:
        record["outputs_identical"] = all(
            record["blocks"][0][role][field] == record["blocks"][1][role][field]
            for role in ("target", "blocker")
            for field in ("model", "steps", "data_sha256", "norm_sha256", "losses_sha256")
        )
        if not record["outputs_identical"]:
            raise AssertionError("ON/OFF changed training work or outputs.")
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python-prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--container-python", type=Path)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--image", required=True)
    parser.add_argument("--budget-s", type=float, default=480)
    parser.add_argument("--backend", choices=("router", "baseline"), required=True)
    parser.add_argument("--baseline-repo", type=Path)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--policies", nargs="+", choices=("off", "on", "baseline"), required=True)
    parser.add_argument("--policy-source", type=Path, required=True)
    parser.add_argument("--predictor-source", type=Path, required=True)
    args = parser.parse_args()
    policy, policy_source = load_source("frozen_placement_policy", args.policy_source)
    predictor, predictor_source = load_source("frozen_step_predictor", args.predictor_source)

    async def callback(context):
        await study(context, args, policy, predictor, policy_source, predictor_source)

    name = "baseline" if args.backend == "baseline" else "experiment"
    spec = importlib.util.spec_from_file_location("long_study_backend", Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    asyncio.run(module.run(args, callback))


if __name__ == "__main__":
    main()
