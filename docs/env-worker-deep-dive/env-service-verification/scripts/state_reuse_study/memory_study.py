"""Measure real GPU validation under a bounded converted host state pool."""

import argparse
import json
import statistics
import time
from pathlib import Path

from memory_policy import FAMILIES, Costs, Scheduler
from memory_replay import Arrivals, workload
from memory_runtime import MODELS, ModelRuntime, memory, trim
from model_process import postprocess
from workspace import Workspace, digest


def initialize_round(workspace, session, turn, families):
    """Separate immutable model dependencies from explicitly declared mutable scoring code."""
    offset = 10 * session + turn
    scale = 1.0 + 0.1 * turn
    config = dict(models=MODELS, families=families, length=8)
    workspace.commit(
        {
            "helper.py": f"OFFSET = {offset}\n",
            "scorer.py": f"from helper import OFFSET\ndef score(value):\n    return value * {scale!r} + OFFSET\n",
            "loader.json": json.dumps(config, sort_keys=True),
        }
    )
    version = workspace.capture()
    return version, offset, scale


def run_block(runtime, costs, scenario, policy, path, references, deadline):
    """Count all selected block work including planning, edits, publication and cache cleanup."""
    for key in list(runtime.prepared):
        runtime.evict(key)
    trim()
    before = memory()
    budget = int(scenario["budget_gib"] * 1024**3)
    agents = workload(scenario["pattern"], scenario["seed"], scenario["gap_s"])
    arrivals = Arrivals(agents)
    scheduler = Scheduler(policy, costs, budget)
    path.mkdir()
    workspaces = [Workspace(path / f"session-{index}") for index in range(len(agents))]
    versions = {}
    trace, outputs = [], []
    preparations, hits, peak, idle = 0, 0, 0, 0.0
    started = time.monotonic()
    while arrivals.ready or arrivals.due:
        now = time.monotonic() - started
        if time.monotonic() > deadline:
            raise TimeoutError("The predeclared GPU study budget was reached.")
        for session, turn in arrivals.issue(now):
            families = agents[session][turn]["families"]
            versions[session, turn] = initialize_round(workspaces[session], session, turn, families)
        if not arrivals.ready:
            delay = min(arrivals.due.values()) - (time.monotonic() - started)
            if delay > 0:
                idle += delay
                time.sleep(min(delay, 10))
            continue
        selected = time.monotonic() - started
        previous = frozenset(runtime.prepared)
        action = scheduler.choose(arrivals.ready, previous, selected)
        job = action.job
        evict_s, prepare_s = 0.0, 0.0
        for key in previous - action.keep:
            begin = time.monotonic()
            runtime.evict(key)
            evict_s += time.monotonic() - begin
        if action.mode == "prepare":
            prepare_s = runtime.prepare(job.family)
            preparations += 1
        hits += action.mode == "reuse"
        payload = costs.bytes(runtime.prepared)
        peak = max(peak, payload)
        if payload > budget or frozenset(runtime.prepared) != action.keep:
            raise RuntimeError("The physical prepared tensors disagreed with the selected budget transition.")
        row = runtime.execute(job.family, prepared=action.mode != "direct", seed=job.seed)
        expected_base = references[job.family, job.seed]["value"]
        if abs(row["value"] - expected_base) > max(1e-4, abs(expected_base) * 1e-5):
            raise RuntimeError("A scheduling policy changed the model validation answer.")
        version, offset, scale = versions[job.group]
        result = postprocess(version["snapshot"], row["value"])
        result["model_identity"] = version["model"]
        expected = expected_base * scale + offset
        if abs(result["value"] - expected) > max(1e-4, abs(expected) * 1e-5):
            raise RuntimeError("The scoring code did not execute its declared version.")
        accepted = workspaces[job.session].publish(
            version, result, path / f"answer-{job.session}-{job.turn}-{job.seed}.json"
        )
        if accepted["status"] != "accepted":
            raise RuntimeError("A current version result was unexpectedly rejected.")
        scheduler.observe(action, previous)
        if policy == "stage" and job.family in runtime.prepared:
            begin = time.monotonic()
            runtime.evict(job.family)
            evict_s += time.monotonic() - begin
        finished = time.monotonic() - started
        arrivals.finish(job, finished)
        trace.append(
            dict(
                session=job.session,
                turn=job.turn,
                seed=job.seed,
                family=job.family,
                mode=action.mode,
                selected_s=selected,
                issued_s=job.issued,
                finished_s=finished,
                predicted_s=action.predicted_s,
                prepare_s=prepare_s,
                evict_s=evict_s,
                payload_bytes=payload,
                memory=memory(),
                code_identity=version["code"],
                model_identity=version["model"],
                result=result["value"],
                **row,
            )
        )
        outputs.append((job.session, job.turn, job.seed, job.family, result["value"]))
    completed_s = time.monotonic() - started
    for key in list(runtime.prepared):
        runtime.evict(key)
    elapsed = time.monotonic() - started
    expected_count = sum(len(turn["families"]) for agent in agents for turn in agent)
    if len(trace) != expected_count:
        raise RuntimeError("A policy omitted a required validation command.")
    return dict(
        scenario=scenario,
        policy=policy,
        status="passed",
        wall_s=elapsed,
        completion_s=completed_s,
        mean_feedback_s=statistics.mean(arrivals.feedback),
        worst_feedback_s=max(arrivals.feedback),
        max_command_wait_s=max(row["selected_s"] - row["issued_s"] for row in trace),
        idle_s=idle,
        planning_cpu_s=scheduler.planning_s,
        commands=len(trace),
        preparations=preparations,
        cache_hits=hits,
        peak_payload_bytes=peak,
        memory_before=before,
        memory_after=memory(),
        work_and_output_identity=digest(sorted(outputs)),
        trace=trace,
    )


def study(args):
    """Use held out scripted arrivals and the same pretrained models for every policy."""
    started = time.monotonic()
    deadline = started + args.budget_seconds
    plan = json.loads(args.plan.read_text())
    profile = json.loads(args.profile.read_text())
    costs = Costs(profile)
    evidence = dict(
        status="running",
        scope="Benchmark-derived GPU correctness workload. Scripted arrivals. Not the original verifier or EnvRouter.",
        blocks=[],
        plan=plan,
    )

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    runtime = ModelRuntime(args.cache)
    evidence.update(
        versions=runtime.versions,
        gpu_uuid=runtime.gpu_uuid,
        payload_bytes=runtime.bytes,
        model_native_load_s=runtime.load_times,
        mapped_memory=memory(),
    )
    if runtime.bytes != costs.size:
        raise RuntimeError("The current model states did not match the independent profile.")
    references = {
        (name, seed): runtime.execute(name, prepared=False, seed=seed) for name in FAMILIES for seed in (0, 1)
    }
    evidence["references"] = {f"{name}:{seed}": row for (name, seed), row in references.items()}
    for name in FAMILIES:
        runtime.prepare(name)
        begin = time.monotonic()
        runtime.evict(name)
        costs.evict[name] = time.monotonic() - begin
    evidence["common_eviction_profile_s"] = costs.evict
    evidence["setup_s"] = time.monotonic() - started
    save()
    try:
        for index, scenario in enumerate(plan["scenarios"]):
            for policy in scenario["policies"]:
                path = args.output.parent / f"block-{index}-{policy}"
                row = run_block(runtime, costs, scenario, policy, path, references, deadline)
                prior = [r for r in evidence["blocks"] if r["scenario"]["name"] == scenario["name"]]
                if prior and row["work_and_output_identity"] != prior[0]["work_and_output_identity"]:
                    raise RuntimeError("The policies did not execute identical logical work and answers.")
                evidence["blocks"].append(row)
                save()
                print(
                    json.dumps({key: row[key] for key in ("policy", "wall_s", "mean_feedback_s", "preparations")}),
                    flush=True,
                )
        evidence["status"] = "passed"
    except BaseException as error:
        evidence.update(status="failed", error=repr(error))
        raise
    finally:
        for key in list(runtime.prepared):
            runtime.evict(key)
        evidence["elapsed_s"] = time.monotonic() - started
        evidence["final_memory"] = memory()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/hf-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=430)
    args = parser.parse_args()
    study(args)
