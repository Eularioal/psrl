"""
Exercise proposed admission and cleanup contracts without running processes or GPUs.

Device permissions and cleanup acknowledgements are simulated inputs. This model
does not implement the production router, Linux cgroups or crash recovery.
"""

import json
from dataclasses import dataclass


@dataclass
class Job:
    """Represent one command inside a shared container in the teaching model."""

    name: str
    budget: int
    state: str = "queued"
    observed: int = 0
    epoch: int = 0


class Model:
    """Reserve pending allocations and retain charges until cleanup is acknowledged."""

    charged_states = {"reserved", "running", "draining", "quarantined"}

    def __init__(self, *, capacity=8192, external=512, headroom=256):
        self.capacity = capacity
        self.external = external
        self.headroom = headroom
        self.jobs = {}
        self.enabled = set()
        self.generation = 0
        self.blocked = False

    def submit(self, name, budget=1024):
        if name in self.jobs or budget <= 0 or budget + self.headroom > self.capacity:
            raise ValueError("Invalid or duplicate job specification.")
        self.jobs[name] = Job(name, budget)

    def charge(self):
        return self.external + sum(
            max(job.budget, job.observed) for job in self.jobs.values() if job.state in self.charged_states
        )

    def admit(self, *, fresh=True):
        # Diagram P3 reserves capacity before the simulated node acknowledgement.
        if not fresh or self.blocked:
            return []
        admitted = []
        for job in self.jobs.values():
            if job.state != "queued":
                continue
            if self.charge() + job.budget + self.headroom > self.capacity:
                break
            self.generation += 1
            job.epoch = self.generation
            job.state = "reserved"
            admitted.append(job.name)
        return admitted

    def owned(self, name, epoch):
        job = self.jobs[name]
        if epoch != job.epoch:
            raise ValueError("Stale command acknowledgement.")
        return job

    def activate(self, name, epoch):
        job = self.owned(name, epoch)
        if job.state != "reserved":
            raise ValueError("Job cannot activate after cancellation or retirement.")
        # Diagram P4 gives this job simulated access without changing sibling grants.
        self.enabled.add(name)
        job.state = "running"

    def cancel(self, name):
        job = self.jobs[name]
        if job.state == "queued":
            job.state = "cancelled"
        elif job.state in {"reserved", "running"}:
            job.state = "draining"
            self.enabled.discard(name)

    def acknowledge_empty(self, name, epoch, *, empty):
        job = self.owned(name, epoch)
        if job.state != "draining":
            raise ValueError("Cleanup acknowledgement requires a draining job.")
        # Diagram P6 treats uncertain physical cleanup as retained ownership.
        if not empty:
            job.state = "quarantined"
            self.blocked = True
            return
        job.observed = 0
        job.state = "released"


def four_jobs(**kwargs):
    model = Model(**kwargs)
    for name in "ABCD":
        model.submit(name)
    return model


def verify():
    """Check the proposal's numeric and state contracts using deterministic examples."""
    results = {}
    low = four_jobs()
    assert low.admit() == list("ABCD"), "Low pressure should admit all four jobs."
    for job in low.jobs.values():
        low.activate(job.name, job.epoch)
    assert len(low.enabled) == 4, "All four simulated job grants should overlap."
    results["low_pressure_grants"] = len(low.enabled)

    high = four_jobs(external=6500)
    assert high.admit() == ["A"], "High pressure should admit only one job."
    high.activate("A", high.jobs["A"].epoch)
    high.cancel("A")
    assert high.admit() == [], "Draining must retain the reservation."
    high.acknowledge_empty("A", high.jobs["A"].epoch, empty=True)
    assert high.admit() == ["B"], "Confirmed cleanup should admit the next job."
    results["high_pressure_grants"] = 1
    results["draining_retains_budget"] = True

    low.cancel("B")
    low.acknowledge_empty("B", low.jobs["B"].epoch, empty=True)
    assert low.enabled == set("ACD"), "Cancelling B must preserve sibling grants."
    results["cancel_preserves_siblings"] = sorted(low.enabled)

    waiting = four_jobs(external=6500)
    waiting.admit()
    waiting.cancel("B")
    assert waiting.jobs["B"].state == "cancelled", "Queued cancellation must not activate the job."
    results["queued_cancel"] = True

    delayed = four_jobs()
    delayed.admit()
    delayed.cancel("A")
    try:
        delayed.activate("A", delayed.jobs["A"].epoch)
    except ValueError:
        results["late_activation_rejected"] = True
    else:
        raise AssertionError("Cancellation must reject late activation.")
    try:
        delayed.acknowledge_empty("A", delayed.jobs["A"].epoch + 100, empty=True)
    except ValueError:
        results["stale_ack_rejected"] = True
    else:
        raise AssertionError("Stale cleanup must not release capacity.")

    uncertain = four_jobs(external=6500)
    uncertain.admit()
    uncertain.cancel("A")
    uncertain.acknowledge_empty("A", uncertain.jobs["A"].epoch, empty=False)
    assert uncertain.admit() == [], "Uncertain cleanup must block new admission."
    assert uncertain.charge() == 7524, "Uncertain cleanup must retain the original charge."
    results["uncertain_cleanup_retains_budget"] = True

    loading = Model(external=6000)
    loading.submit("A", 1500)
    assert loading.admit() == ["A"], "A should fit before its allocation grows."
    loading.jobs["A"].observed = 100
    loading.submit("B", 1000)
    assert loading.admit() == [], "External use and future owned growth must both be charged."
    results["external_plus_loading_reservation"] = True

    stale = four_jobs()
    assert stale.admit(fresh=False) == [], "Missing telemetry must not count as zero use."
    results["stale_telemetry_blocks_admission"] = True
    return results


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
