"""Execute a fixed placement matrix with typed costs and real PSRL services."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

A, B = "Qwen3-4B-Base", "Qwen3-1.7B-Base"
MIB = 1024**2
STRATEGIES = ("affinity", "always_copy", "predictive")
CASES = (("default", 2, 128), ("short", 1024, 128), ("boundary", 1024, 512), ("long", 1024, 2048))


def plan_cells():
    cells = []
    for backend in ("baseline", "native"):
        for repeat in range(1, 4):
            for ordinal, (case_id, target_steps, blocker_steps) in enumerate(CASES):
                shift = (repeat - 1 + ordinal) % len(STRATEGIES)
                strategies = STRATEGIES[shift:] + STRATEGIES[:shift] if backend == "native" else ("affinity",)
                for strategy in strategies:
                    cells.append(
                        dict(
                            cell_id=f"{backend}-r{repeat}-{case_id}-{strategy}",
                            round=repeat,
                            phase="core",
                            backend=backend,
                            case_id=case_id,
                            strategy=strategy,
                            target_steps=target_steps,
                            blocker_steps=blocker_steps,
                            forecast_scale=1.0,
                            report_age_offset_s=0.0,
                            cold_budget_mb=20480,
                        )
                    )
        if backend == "native":
            for repeat in range(1, 4):
                for case_id, base_case, blocker_steps, scale, age, budget in (
                    ("overpredict_short", "short", 128, 10.0, 0.0, 20480),
                    ("stale_short", "short", 128, 1.0, 10.0, 20480),
                    ("memory_long", "long", 2048, 1.0, 0.0, 6144),
                ):
                    cells.append(
                        dict(
                            cell_id=f"native-r{repeat}-{case_id}-predictive",
                            round=repeat,
                            phase="fault",
                            backend=backend,
                            case_id=case_id,
                            reference_case=base_case,
                            strategy="predictive",
                            target_steps=1024,
                            blocker_steps=blocker_steps,
                            forecast_scale=scale,
                            report_age_offset_s=age,
                            cold_budget_mb=budget,
                        )
                    )
    return cells


def load_policy(directory):
    package = types.ModuleType("frozen_matrix_policy")
    package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    return tuple(
        importlib.import_module(package.__name__ + "." + name)
        for name in ("warm_dispatch", "step_predictor", "warm_placement")
    )


def command(command_id, model, steps):
    return dict(id=command_id, model=model, steps=steps, progress_every=16)


def empirical(samples, multiplier, dispatch, placement):
    point = statistics.median(samples) * multiplier
    return dispatch.TimeEstimate(
        point, placement.DurationInterval(min(samples) * multiplier * 0.8, max(samples) * multiplier * 1.3)
    )


async def observe(context, task, device, command_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if task.done():
            task.result()
            raise RuntimeError("The blocker finished before its intended arrival observation.")
        try:
            progress = json.loads(context["progress_file"](device).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            await asyncio.sleep(0.01)
            continue
        if progress.get("id") == command_id and progress.get("completed_steps", 0) >= 64:
            return progress
        await asyncio.sleep(0.01)
    raise TimeoutError("No fresh arrival observation within thirty seconds.")


async def study(context, args):
    from telemetry import inspect_owned

    dispatch, predictor, placement = load_policy(args.policy_dir)
    request, record, save = context["request"], context["record"], context["save"]
    record.update(
        protocol=dict(
            name="fixed PostTrain placement matrix",
            target_model=A,
            blocker_model=B,
            progress_every=16,
            arrival_after_completed_steps=64,
            profile_steps=128,
            switch_margin_s=2.0,
            decision_mode="point",
            primary_timing="before blocker submission through both RPC responses",
            scope="Controlled repeated public examples, norm-only updates, no LLM rollout or policy training.",
        ),
        backend=args.backend,
        blocks=[],
        profiles=[],
        policy_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.policy_dir.glob("*.py")},
    )
    record["container_inspection_before"] = await inspect_owned(record["tag"])
    cells = [cell for cell in plan_cells() if cell["backend"] == args.backend]
    profile = {}
    for phase in ("core", "fault"):
        selected_cells = [cell for cell in cells if cell["phase"] == phase]
        for repeat in sorted({cell["round"] for cell in selected_cells}):
            hot, cold = (repeat - 1) % 2, repeat % 2
            for index in (0, 1):
                await request(index, dict(op="clear"))
                await request(index, dict(op="set_budget", budget_mb=20480))
            if phase == "core":
                measured = []
                for index, model in ((cold, A), (hot, A), (hot, B)):
                    value = await request(index, command(f"r{repeat}-profile-{index}-{model}", model, 128))
                    value.update(round=repeat, device=index)
                    record["profiles"].append(value)
                    measured.append(value)
                    save()
                profile[repeat] = measured
            else:
                # Restore resident identities without adding unregistered optimizer updates.
                await request(hot, dict(op="load", model=A))
                await request(hot, dict(op="load", model=B))
            measured = profile[repeat]
            pa = [value for value in measured if value["model"] == A]
            pb = [value for value in measured if value["model"] == B][0]
            load_samples = [value["load_s"] for value in pa]
            a_rates = [
                wall / steps
                for value in pa
                for steps, wall in zip(value["chunk_steps"][1:], value["wall_chunk_s"][1:], strict=True)
            ]
            b_rate = statistics.median(
                wall / steps for steps, wall in zip(pb["chunk_steps"][1:], pb["wall_chunk_s"][1:], strict=True)
            )
            for cell in (value for value in selected_cells if value["round"] == repeat):
                expected = cell["blocker_steps"] * b_rate + cell["target_steps"] * statistics.median(a_rates)
                minimum = max(25, expected * 1.4 + max(load_samples) + 10)
                if context["remaining"]() < minimum:
                    record["skipped_budget_cell"] = cell["cell_id"]
                    save()
                    raise TimeoutError("The next registered cell does not fit the remaining work budget.")
                await request(cold, dict(op="clear"))
                await request(cold, dict(op="set_budget", budget_mb=cell["cold_budget_mb"]))
                states = [await request(index, dict(op="ping")) for index in (hot, cold)]
                admission = await request(cold, dict(op="admit", model=A))
                execution = empirical(a_rates, cell["target_steps"], dispatch, placement)
                rebuild = empirical(load_samples, 1, dispatch, placement)
                cost = dispatch.ModelCost(A, pa[0]["model_bytes"], rebuild, execution, states[0]["workspace_bytes"])
                block = dict(**cell, hot_device=hot, cold_device=cold, server_admission=admission, status="running")
                record["blocks"].append(block)
                save()
                started = time.monotonic()
                block["started_unix_s"] = time.time()
                async with asyncio.TaskGroup() as tasks:
                    blocker_task = tasks.create_task(
                        request(hot, command(cell["cell_id"] + "-blocker", B, cell["blocker_steps"]))
                    )
                    progress = await observe(context, blocker_task, hot, cell["cell_id"] + "-blocker")
                    arrival = time.monotonic()
                    age = max(0, arrival - progress["observed_at"])
                    base = predictor.predict_remaining(
                        progress["chunk_steps"][1:],
                        progress["wall_chunk_s"][1:],
                        progress["completed_steps"],
                        cell["blocker_steps"],
                        age,
                        report_every_steps=16,
                    )
                    effective = predictor.predict_remaining(
                        progress["chunk_steps"][1:],
                        progress["wall_chunk_s"][1:],
                        progress["completed_steps"],
                        cell["blocker_steps"],
                        age + cell["report_age_offset_s"],
                        report_every_steps=16,
                    )
                    active = dispatch.estimate_from_forecast(effective)
                    if active is not None and cell["forecast_scale"] != 1:
                        scale = cell["forecast_scale"]
                        active = dispatch.TimeEstimate(
                            active.point_s * scale,
                            placement.DurationInterval(
                                active.interval.lower_s * scale, active.interval.upper_s * scale
                            ),
                        )
                    snapshots = (
                        dispatch.QueueSnapshot(
                            "hot",
                            frozenset(states[0]["cache_keys"]),
                            states[0]["budget_bytes"] - states[0]["cache_bytes"],
                            True,
                            active,
                        ),
                        dispatch.QueueSnapshot(
                            "cold",
                            frozenset(states[1]["cache_keys"]),
                            states[1]["budget_bytes"] - states[1]["cache_bytes"],
                        ),
                    )
                    decision = dispatch.WarmPlacementController().decide(cost, snapshots, strategy=cell["strategy"])
                    if decision.device_id not in {"hot", "cold"}:
                        raise RuntimeError("No legal GPU placement for this registered command.")
                    selected = cold if decision.device_id == "cold" else hot
                    if selected == cold and not admission["admits"]:
                        raise AssertionError("Controller contradicted the actual service admission check.")
                    block.update(
                        decision=dataclasses.asdict(decision),
                        selected_device=decision.device_id,
                        snapshots=[dataclasses.asdict(value) for value in snapshots],
                        service_states=states,
                        observed_progress=progress,
                        progress_forecast=dataclasses.asdict(base),
                        effective_forecast=dataclasses.asdict(effective),
                        effective_active=dataclasses.asdict(active) if active is not None else None,
                        execution_estimate=dataclasses.asdict(execution),
                        rebuild_estimate=dataclasses.asdict(rebuild),
                        decision_overhead_s=time.monotonic() - arrival,
                    )
                    target_task = tasks.create_task(
                        request(selected, command(cell["cell_id"] + "-target", A, cell["target_steps"]))
                    )
                finished = time.monotonic()
                target, blocker = target_task.result(), blocker_task.result()
                block.update(
                    status="passed",
                    finished_unix_s=time.time(),
                    target=target,
                    blocker=blocker,
                    primary_makespan_s=finished - started,
                    target_latency_s=target["finished_monotonic"] - arrival,
                    target_wait_s=max(0, target["started_monotonic"] - arrival),
                    blocker_remaining_actual_s=max(0, blocker["finished_monotonic"] - arrival),
                    driver_join_both_s=finished - arrival,
                )
                clear_started = time.monotonic()
                clear = await request(cold, dict(op="clear"))
                block.update(cold_clear_s=clear["clear_s"], cold_clear_rpc_s=time.monotonic() - clear_started)
                block["closed_cycle_s"] = block["primary_makespan_s"] + block["cold_clear_rpc_s"]
                save()
    record["container_inspection_after"] = await inspect_owned(record["tag"])
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python-prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--container-python", type=Path, required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--image", required=True)
    parser.add_argument("--budget-s", type=float, required=True)
    parser.add_argument("--backend", choices=("native", "baseline"), required=True)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    args = parser.parse_args()
    module = importlib.import_module("experiment" if args.backend == "native" else "baseline")

    async def callback(context):
        await study(context, args)

    asyncio.run(module.run(args, callback))


if __name__ == "__main__":
    main()
