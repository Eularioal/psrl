"""Compare hot queue waiting with model rebuilding using real PSRL environments."""

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

policy_path = Path(__file__).resolve().parents[5] / "psrl/workers/env_service/warm_placement.py"
policy_spec = importlib.util.spec_from_file_location("warm_policy", policy_path)
policy = importlib.util.module_from_spec(policy_spec)
sys.modules[policy_spec.name] = policy
policy_source = policy_path.read_bytes()
exec(compile(policy_source, str(policy_path), "exec"), policy.__dict__)
DeviceSnapshot, DurationInterval = policy.DeviceSnapshot, policy.DurationInterval
PlacementRequest, choose_placement = policy.PlacementRequest, policy.choose_placement
estimate_remaining = policy.estimate_remaining

A, B = "Qwen3-4B-Base", "Qwen3-1.7B-Base"
TARGET_STEPS = 256
SCENARIOS = (("short", 64), ("long", 2048))


def interval(samples, steps=1):
    """Use empirical extrema with padding, without claiming confidence coverage."""
    return DurationInterval(min(samples) * steps * 0.8, max(samples) * steps * 1.5)


async def study(context):
    request, record, save = context["request"], context["record"], context["save"]
    (context["root"] / "placement-policy.py").write_bytes(policy_source)
    record["placement_policy_sha256"] = hashlib.sha256(policy_source).hexdigest()
    record["protocol"] = dict(
        name="controlled PostTrain fixed-step extension",
        target_model=A,
        blocker_model=B,
        target_steps=TARGET_STEPS,
        scenarios=SCENARIOS,
        public_examples=2,
        target_parameter="model.norm.weight",
        optimizer="AdamW lr=0.001",
        token_limits=[192, 64],
        arrival_after_completed_steps=16,
        note=(
            "Repeated epochs on local public examples. This is not the default two-update benchmark or official score."
        ),
    )
    record["profiles"], record["blocks"] = [], []
    await request(0, dict(op="clear"))
    await request(1, dict(op="clear"))
    for device, model, steps in ((1, A, 48), (0, A, 48), (0, B, 64)):
        result = await request(device, dict(id=f"profile-{device}-{model}", model=model, steps=steps))
        record["profiles"].append(result)
        save()
    model_a = [value for value in record["profiles"] if value["model"] == A]
    load = interval([value["load_s"] for value in model_a])
    execution = interval(model_a[-1]["step_s"][8:], TARGET_STEPS)
    model_bytes = model_a[0]["model_bytes"]
    record["predictor"] = dict(
        rebuild=dataclasses.asdict(load),
        execution=dataclasses.asdict(execution),
        method="padded empirical min/max, progress updated every step",
    )
    save()
    for scenario, steps in SCENARIOS:
        # Reverse the order between regimes, so ON is not always measured later.
        policies = (
            (False,) if record.get("baseline_repo") else ((False, True) if scenario == "short" else (True, False))
        )
        for enabled in policies:
            if context["remaining"]() < 100:
                raise TimeoutError("Insufficient experiment budget for another bounded block.")
            await request(1, dict(op="clear"))
            label = f"{scenario}-{'on' if enabled else 'off'}"
            blocker_id = label + "-blocker"
            blocker_task = asyncio.create_task(request(0, dict(id=blocker_id, model=B, steps=steps)))
            tick = time.monotonic()
            progress = None
            prediction = None
            while time.monotonic() - tick < 30:
                try:
                    progress = json.loads(context["progress_file"](0).read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    await asyncio.sleep(0.01)
                    continue
                if progress.get("id") == blocker_id and progress.get("completed_steps", 0) >= 16:
                    if progress["phase"] == "complete":
                        prediction = estimate_remaining([], steps, steps)
                    else:
                        prediction = estimate_remaining(
                            progress["sample_step_s"][4:],
                            progress["completed_steps"],
                            steps,
                            max(0, time.monotonic() - progress["step_started"]),
                        )
                    if prediction.remaining is not None:
                        break
                if blocker_task.done():
                    await blocker_task
                await asyncio.sleep(0.005)
            if prediction is None:
                raise TimeoutError("No progress observation within the bounded arrival gate.")
            arrival = time.monotonic()
            placement_request = PlacementRequest(A, model_bytes, load, execution)
            hot_busy = progress["phase"] != "complete"
            snapshots = (
                DeviceSnapshot(
                    "hot", frozenset({A, B}), 1024**3, hot_busy, prediction.remaining if hot_busy else None
                ),
                DeviceSnapshot("cold", frozenset(), 19 * 1024**3),
            )
            decision = choose_placement(placement_request, snapshots, switch_margin_s=0.25)
            selected = 1 if enabled and decision.action == "rebuild" else 0
            block = dict(
                id=label,
                scenario=scenario,
                enabled=enabled,
                arrival_monotonic=arrival,
                observed_progress=progress,
                progress_estimate=dataclasses.asdict(prediction),
                decision=dataclasses.asdict(decision),
                selected=selected,
                decision_overhead_s=time.monotonic() - arrival,
            )
            record["blocks"].append(block)
            save()
            target_task = asyncio.create_task(
                request(selected, dict(id=label + "-target", model=A, steps=TARGET_STEPS))
            )
            target, blocker = await asyncio.gather(target_task, blocker_task)
            done = time.monotonic()
            actual_residual = max(0, blocker["finished_monotonic"] - arrival)
            remaining = prediction.remaining
            block.update(
                target=target,
                blocker=blocker,
                response_latency_s=done - arrival,
                target_latency_s=target["finished_monotonic"] - arrival,
                target_wait_s=max(0, target["started_monotonic"] - arrival),
                tail_makespan_s=max(target["finished_monotonic"], blocker["finished_monotonic"]) - arrival,
                blocker_remaining_actual_s=actual_residual,
                residual_covered=remaining is not None and remaining.lower_s <= actual_residual <= remaining.upper_s,
            )
            save()
    if record.get("baseline_repo"):
        record["outputs_identical"] = "Requires cross-backend audit against Router evidence."
        save()
        return
    checks = []
    for scenario, _ in SCENARIOS:
        pair = [value for value in record["blocks"] if value["scenario"] == scenario]
        for key in ("target", "blocker"):
            checks.append(
                all(
                    pair[0][key][field] == pair[1][key][field]
                    for field in ("model", "steps", "data_sha256", "norm_sha256", "losses_sha256")
                )
            )
    record["outputs_identical"] = all(checks)
    if not all(checks):
        raise AssertionError("ON/OFF changed inputs or training outputs.")
    record["summary"] = {}
    for scenario, _ in SCENARIOS:
        pair = {value["enabled"]: value for value in record["blocks"] if value["scenario"] == scenario}
        record["summary"][scenario] = dict(
            on_action=pair[True]["decision"]["action"],
            on_device=pair[True]["selected"],
            off_target_s=pair[False]["target_latency_s"],
            on_target_s=pair[True]["target_latency_s"],
            target_improvement_pct=100 * (1 - pair[True]["target_latency_s"] / pair[False]["target_latency_s"]),
            off_tail_s=pair[False]["tail_makespan_s"],
            on_tail_s=pair[True]["tail_makespan_s"],
            tail_improvement_pct=100 * (1 - pair[True]["tail_makespan_s"] / pair[False]["tail_makespan_s"]),
        )
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--python-prefix", type=Path, default=Path(sys.prefix))
    parser.add_argument("--container-python", type=Path)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--image", default="psrl-mlgym:20261112")
    parser.add_argument("--budget-s", type=float, default=600)
    parser.add_argument("--backend", choices=("router", "baseline"), default="router")
    parser.add_argument("--baseline-repo", type=Path)
    args = parser.parse_args()
    if args.backend == "baseline":
        # Baseline helper must be imported in a fresh process with the old repository first.
        spec = importlib.util.spec_from_file_location("warm_static_baseline", Path(__file__).with_name("baseline.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        asyncio.run(module.run(args, study))
    else:
        from experiment import run

        asyncio.run(run(args, study))


if __name__ == "__main__":
    main()
