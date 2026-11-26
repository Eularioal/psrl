"""Compare Episode holding, static commands, and dynamic borrowing on matched traces."""

import argparse
import json
import math
import random
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from psrl.workers.gen.rollout_coordination.env_partition import PartitionPolicy

TICK_S = 1
REQUEST_CAP = 16
ENV_CAP = 5
ENV_MEMORY_GIB = 16
POOL_MEMORY_GIB = 90
HEADROOM_GIB = 16
MODEL_BUDGET_GIB = 43
SLEEP_S = 4
WAKE_AND_SYNC_S = 6
TRIALS = 12
POLICY = {
    "request_capacity": REQUEST_CAP,
    "min_replicas": 1,
    "shrink_fraction": 0.65,
    "restore_fraction": 0.90,
    "interval_s": 2,
    "shrink_window_s": 30,
    "restore_window_s": 6,
    "cooldown_s": 60,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    replicas: int
    environment_gpus: int
    trajectories: int
    model: str
    bench: str
    request_multiplier: float = 1
    shared_slowdown: float = 1.25
    model_budget_gib: int = MODEL_BUDGET_GIB
    env_memory_gib: int = ENV_MEMORY_GIB
    env_concurrency: int = ENV_CAP
    env_duration_scale: float = 1


@dataclass
class Trajectory:
    durations: tuple
    stage: int = 0
    queued_at: int = 0
    remaining: float = 0
    lease: int = -1


def trace_profiles(directory, dedicated_units=None):
    """Remove queue time and optionally exclude co-located tool durations."""
    model, tools = [], []
    for path in sorted(directory.glob("episodes/*/episode.json")):
        row = json.loads(path.read_text())
        if row["status"] != "complete" or len(row["queries"]) != 3 or len(row["commands"]) != 2:
            continue
        model.append(tuple(query["finished_at"] - query["started_at"] for query in row["queries"]))
        if dedicated_units is not None and any(
            command["admission"]["unit_id"] not in dedicated_units for command in row["commands"]
        ):
            continue
        samples = []
        for command in row["commands"]:
            admission = command["admission"]
            start = admission.get("admitted_at", command["started_at"])
            stop = admission.get("released_at") or command["finished_at"]
            samples.append(stop - (start if start is not None else command["started_at"]))
        tools.append(tuple(samples))
    if not model or not tools or any(value <= 0 for profile in model + tools for value in profile):
        raise ValueError(f"No valid completed profiles at {str(directory)!r}.")
    return model, tools


def cifar_profiles(evidence):
    """Use serial CUDA timings plus explicit assumed cold command costs."""
    trials = json.loads(evidence.read_text())["trials"]
    return [
        ((trial["training"]["ended_ns"] - trial["training"]["started_ns"]) / 1e9 + 8, 8.5)
        for trial in trials
        if not trial["warmup"] and trial["policy"] == "serial" and trial["epochs"] == 64
    ]


class Replay:
    """Replay a single batch with ordered requests and physical GPU roles."""

    def __init__(self, scenario, durations, policy):
        self.scenario = scenario
        self.command_mode = policy != "episode_static"
        self.dynamic = policy == "command_elastic"
        self.jobs = [Trajectory(row) for row in durations]
        self.pending_model = deque(range(len(self.jobs)))
        self.pending_env = deque()
        self.model_gpus = list(range(scenario.replicas))
        self.env_gpus = list(range(scenario.replicas, scenario.replicas + scenario.environment_gpus))
        self.model_running = {gpu: set() for gpu in self.model_gpus}
        self.env_running = {gpu: set() for gpu in self.model_gpus + self.env_gpus}
        self.env_reservations = {gpu: 0 for gpu in self.env_gpus}
        self.role = {gpu: "resident" for gpu in self.model_gpus}
        self.env_cap = min(scenario.env_concurrency, POOL_MEMORY_GIB // scenario.env_memory_gib)
        spare = POOL_MEMORY_GIB - scenario.model_budget_gib - HEADROOM_GIB
        self.shared_cap = max(0, min(2, spare // scenario.env_memory_gib))
        self.policy = PartitionPolicy(POLICY)
        self.transition = None
        self.role_events = []
        self.requests = []
        self.env_wait = []
        self.placements = {"dedicated": 0, "borrowed": 0, "shared": 0, "draining_shared": 0}
        self.finished = 0

    def start(self, job, gpu, running):
        record = self.jobs[job]
        record.remaining = record.durations[record.stage]
        running[gpu].add(job)

    def dispatch(self, now):
        resident = [gpu for gpu in self.model_gpus if self.role[gpu] == "resident"]
        while self.pending_model:
            gpu = min(resident, key=lambda i: (len(self.model_running[i]), i))
            if len(self.model_running[gpu]) >= REQUEST_CAP:
                break
            self.start(self.pending_model.popleft(), gpu, self.model_running)

        if self.dynamic:
            borrowed = [gpu for gpu in self.model_gpus if self.role[gpu] == "borrowed"]
            shared = [gpu for gpu in self.model_gpus if self.role[gpu] in ("resident", "draining_shared")]
            eligible = (
                [(gpu, self.env_cap, "dedicated") for gpu in self.env_gpus]
                + [(gpu, self.env_cap, "borrowed") for gpu in borrowed]
                + [
                    (gpu, self.shared_cap, "draining_shared" if self.role[gpu] != "resident" else "shared")
                    for gpu in shared
                ]
            )
        else:
            eligible = [(gpu, self.env_cap, "dedicated") for gpu in self.env_gpus]
        while self.pending_env:
            occupied = (
                {gpu: len(jobs) for gpu, jobs in self.env_running.items()}
                if self.command_mode
                else self.env_reservations
            )
            choice = next(((gpu, kind) for gpu, cap, kind in eligible if occupied[gpu] < cap), None)
            if choice is None:
                break
            job = self.pending_env.popleft()
            gpu, kind = choice
            record = self.jobs[job]
            if not self.command_mode:
                record.lease = gpu
                self.env_reservations[gpu] += 1
            self.env_wait.append(now - record.queued_at)
            self.placements[kind] += 1
            self.start(job, gpu, self.env_running)

    def change_roles(self, now):
        if not self.dynamic:
            return
        if self.transition is not None:
            action, gpu, deadline = self.transition
            if deadline is None:
                blocking = self.model_running[gpu] if action == "borrow" else self.env_running[gpu]
                if not blocking:
                    self.role[gpu] = "switching"
                    cost = SLEEP_S if action == "borrow" else WAKE_AND_SYNC_S
                    self.transition = (action, gpu, now + cost)
            elif now >= deadline:
                self.role[gpu] = "borrowed" if action == "borrow" else "resident"
                self.role_events.append((now, gpu, self.role[gpu]))
                self.transition = None
                self.policy.changed()
                self.policy.last_change = now
            return
        if now % POLICY["interval_s"]:
            return
        active = [gpu for gpu in self.model_gpus if self.role[gpu] == "resident"]
        borrowed = [gpu for gpu in self.model_gpus if self.role[gpu] == "borrowed"]
        load = sum(len(self.model_running[gpu]) for gpu in active)
        pressure = bool(self.pending_env or any(self.env_running[gpu] for gpu in active))
        action = self.policy.observe(now, len(active), len(borrowed), load, len(self.pending_model), pressure)
        if action == "borrow":
            gpu = min(active, key=lambda i: (len(self.model_running[i]), i))
            self.role[gpu] = "draining_shared"
            self.transition = (action, gpu, None)
        elif action == "restore":
            gpu = borrowed[0]
            self.role[gpu] = "restoring"
            self.transition = (action, gpu, None)

    def progress(self, now):
        finished = []
        contention = {gpu for gpu, model in self.model_running.items() if model and self.env_running[gpu]}
        for running in (self.model_running, self.env_running):
            for gpu, jobs in running.items():
                rate = TICK_S / (self.scenario.shared_slowdown if gpu in contention else 1)
                for job in sorted(jobs):
                    self.jobs[job].remaining -= rate
                    if self.jobs[job].remaining <= 1e-9:
                        finished.append((job, gpu, running))
        for job, gpu, running in finished:
            running[gpu].remove(job)
            record = self.jobs[job]
            if record.stage % 2 == 0:
                self.requests.append(now - record.queued_at)
            record.stage += 1
            record.queued_at = now
            if record.stage == 5:
                if record.lease >= 0:
                    self.env_reservations[record.lease] -= 1
                self.finished += 1
            elif record.stage % 2 == 0:
                self.pending_model.append(job)
            elif record.lease >= 0:
                self.placements["dedicated"] += 1
                self.start(job, record.lease, self.env_running)
            else:
                self.pending_env.append(job)

    def run(self):
        """Return the last completed trajectory time and paired diagnostics."""
        for now in range(0, 12_000, TICK_S):
            self.change_roles(now)
            self.dispatch(now)
            self.progress(now + TICK_S)
            if self.finished == len(self.jobs):
                break
        else:
            raise TimeoutError("The simulated batch exceeded its bound.")
        assert len(self.env_wait) == len(self.jobs) * (2 if self.command_mode else 1)
        assert sum(self.placements.values()) == 2 * len(self.jobs)
        assert len(self.requests) == 3 * len(self.jobs)
        return {
            "batch_s": now + TICK_S,
            "p95_request_s": sorted(self.requests)[math.ceil(len(self.requests) * 0.95) - 1],
            "total_gpu_wait_s": sum(self.env_wait),
            "placements": self.placements,
            "role_events": self.role_events,
        }


def compare(scenario, request_profiles, env_profile_groups):
    """Change only policy for each seeded sample of the same scenario."""
    trials = []
    for seed in range(TRIALS):
        rng = random.Random(seed)
        durations = []
        for episode in range(scenario.trajectories):
            q1, q2, q3 = (value * scenario.request_multiplier for value in rng.choice(request_profiles))
            train, verify = rng.choice(env_profile_groups[episode % len(env_profile_groups)])
            durations.append((q1, train * scenario.env_duration_scale, q2, verify * scenario.env_duration_scale, q3))
        baseline = Replay(scenario, durations, "episode_static").run()
        command = Replay(scenario, durations, "command_static").run()
        method = Replay(scenario, durations, "command_elastic").run()
        trials.append(
            {
                "seed": seed,
                "episode_static": baseline,
                "command_static": command,
                "command_elastic": method,
                "change_pct": 100 * (method["batch_s"] / baseline["batch_s"] - 1),
                "change_vs_command_static_pct": 100 * (method["batch_s"] / command["batch_s"] - 1),
            }
        )
    changes = [trial["change_pct"] for trial in trials]
    return {
        "scenario": asdict(scenario),
        "median_episode_s": statistics.median(trial["episode_static"]["batch_s"] for trial in trials),
        "median_command_static_s": statistics.median(trial["command_static"]["batch_s"] for trial in trials),
        "median_dynamic_s": statistics.median(trial["command_elastic"]["batch_s"] for trial in trials),
        "median_paired_change_pct": statistics.median(changes),
        "median_paired_change_vs_command_static_pct": statistics.median(
            trial["change_vs_command_static_pct"] for trial in trials
        ),
        "min_max_paired_change_pct": [min(changes), max(changes)],
        "resident_command_cap": max(
            0, min(2, (POOL_MEMORY_GIB - scenario.model_budget_gib - HEADROOM_GIB) // scenario.env_memory_gib)
        ),
        "median_borrowed_commands": statistics.median(
            trial["command_elastic"]["placements"]["borrowed"] for trial in trials
        ),
        "runs_with_borrow": sum(
            any(role == "borrowed" for _, _, role in trial["command_elastic"]["role_events"]) for trial in trials
        ),
        "runs_with_restore": sum(
            any(role == "resident" for _, _, role in trial["command_elastic"]["role_events"]) for trial in trials
        ),
        "trials": trials,
    }


def main(args):
    model3, gsm8k = trace_profiles(args.gsm8k)
    model8, aime2025 = trace_profiles(args.aime, dedicated_units={"gpu-5", "gpu-6", "gpu-7"})
    envs = {"GSM8K": gsm8k, "AIME 2025": aime2025, "CIFAR-10": cifar_profiles(args.cifar)}
    models = {"3B": model3, "8B": model8}
    scenarios = [
        Scenario("six_cards", 2, 2, 20, "3B", "GSM8K"),
        Scenario("eight_cards", 3, 3, 20, "3B", "GSM8K"),
        Scenario("larger_model", 3, 3, 20, "8B", "GSM8K"),
        Scenario("held_out_bench", 3, 3, 20, "8B", "AIME 2025"),
        Scenario("short_image_commands", 3, 3, 20, "8B", "CIFAR-10"),
        Scenario("twelve_cards", 5, 5, 20, "8B", "AIME 2025"),
        Scenario("twelve_cards_forty_trajectories", 5, 5, 40, "8B", "AIME 2025"),
        Scenario("eight_cards_forty_trajectories", 3, 3, 40, "8B", "AIME 2025"),
        Scenario("eight_cards_requests_3x", 3, 3, 40, "8B", "AIME 2025", request_multiplier=3),
        Scenario("eight_cards_sharing_2x", 3, 3, 40, "8B", "AIME 2025", shared_slowdown=2),
    ]
    if args.suite == "diverse":
        scenarios = [
            Scenario("aime_resident_43", 3, 3, 20, "8B", "AIME 2025"),
            Scenario("aime_resident_62", 3, 3, 20, "8B", "AIME 2025", model_budget_gib=62),
            Scenario("aime_resident_72", 3, 3, 20, "8B", "AIME 2025", model_budget_gib=72),
            Scenario(
                "larger_environment",
                3,
                3,
                12,
                "8B",
                "AIME 2025",
                env_memory_gib=24,
                env_duration_scale=1.4,
            ),
            Scenario(
                "larger_environment_and_resident",
                3,
                3,
                12,
                "8B",
                "AIME 2025",
                env_memory_gib=24,
                env_duration_scale=1.4,
                model_budget_gib=62,
            ),
            Scenario("vision_4g", 3, 3, 20, "8B", "CIFAR-10", env_memory_gib=4, env_concurrency=16),
            Scenario("mixed_43", 3, 3, 20, "8B", "AIME 2025+CIFAR-10"),
            Scenario("mixed_62", 3, 3, 20, "8B", "AIME 2025+CIFAR-10", model_budget_gib=62),
        ]
    report = {
        "measurement": "Replay of five phases through the last required trajectory, before policy update.",
        "inputs": {
            key: str(path)
            for key, path in (("3B/GSM8K", args.gsm8k), ("8B/AIME 2025", args.aime), ("CIFAR-10", args.cifar))
        },
        "profiles": {
            "model_3b": len(model3),
            "model_8b": len(model8),
            "env": {name: len(rows) for name, rows in envs.items()},
        },
        "settings": {
            "tick_s": TICK_S,
            "sleep_s": SLEEP_S,
            "wake_and_sync_s": WAKE_AND_SYNC_S,
            "gpu_budget_gib": POOL_MEMORY_GIB,
            "env_budget_gib": ENV_MEMORY_GIB,
            "model_budget_gib": MODEL_BUDGET_GIB,
            "model_headroom_gib": HEADROOM_GIB,
            "policy": POLICY,
            "cifar_assumed_setup_s": 8,
            "cifar_assumed_verify_s": 8.5,
            "aime_tool_profile": "Keep only trajectories with both tool commands on dedicated GPUs 5 through 7.",
        },
        "pairs": [compare(s, models[s.model], tuple(envs[name] for name in s.bench.split("+"))) for s in scenarios],
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "trials"} for row in report["pairs"]], ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("gsm8k", "aime", "cifar", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--suite", choices=("scale", "diverse"), default="scale")
    main(parser.parse_args())
