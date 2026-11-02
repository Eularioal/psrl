"""Illustrate command placement with known durations, without executing GPU work."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name: str
    ready: float
    duration: float
    home: int = 0


def simulate(dynamic, switch_cost, other_ready=5.0):
    # Diagram C1 and C2: sessions retain their files and host resources.
    available = [0.0, other_ready]
    jobs = [Job("A1", 0, 2), Job("B1", 2, 20), Job("A2", 4, 3)]
    rows = []
    for job in jobs:
        # Diagram P2: this oracle knows durations, unlike a production estimator.
        candidates = range(2) if dynamic and job.name == "A2" else (job.home,)
        options = []
        for gpu in candidates:
            setup = switch_cost if gpu != job.home else 0.0
            start = max(job.ready, available[gpu]) + setup
            options.append((start + job.duration, gpu != job.home, gpu, start, setup))
        finish, _, gpu, start, setup = min(options)
        rows.append((job.name, gpu, start, finish, start - job.ready, setup))
        # Diagram P3 and P4: assume physical return is confirmed at finish.
        available[gpu] = finish
    return rows, max(available)


def main():
    print("Teaching simulation only. Each GPU fits one job. All durations are known.")
    print("C occupies GPU1 until t=5. Files stay on the same host. No GPU is used.")
    cases = [
        ("fixed", False, 0.0, 5.0),
        ("dynamic-cheap", True, 0.2, 5.0),
        ("dynamic-expensive", True, 20.0, 5.0),
        ("dynamic-both-busy", True, 0.2, 40.0),
    ]
    for name, dynamic, cost, other_ready in cases:
        rows, makespan = simulate(dynamic, cost, other_ready)
        print(f"\n{name}: switch={cost:g}s, GPU1 available at t={other_ready:g}")
        print("job  gpu  start  finish  delay-including-setup")
        for job, gpu, start, finish, delay, _ in rows:
            print(f"{job:3}  {gpu:3}  {start:5.1f}  {finish:6.1f}  {delay:5.1f}")
        print(f"all jobs, including C, finished at t={makespan:.1f}")

    print("\nQueue length is not queued work:")
    print("GPU0: one 20s job. GPU1: four 1s jobs. A new 3s job finishes at 23s versus 7s.")
    print("This is an illustrative forecast, not a PSRL benchmark or a fairness simulation.")


if __name__ == "__main__":
    main()
