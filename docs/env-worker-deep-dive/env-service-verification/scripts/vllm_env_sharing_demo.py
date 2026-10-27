"""CPU teaching model for spare capacity, inference dependencies and idle windows.

All GiB, durations and service rates are synthetic, with no production imports.
"""

import json


def memory_admission(new_job_gib):
    """Model diagram 2 with simultaneous, disjoint ownership observations."""
    capacity, measured, margin = 80, 64, 4
    promises = {"model": 60, "environment": 12}
    observed = {"model": 56, "environment": 8}
    remaining = sum(max(0, peak - observed[owner]) for owner, peak in promises.items())
    charged = measured + remaining + new_job_gib + margin
    return {"new_job_gib": new_job_gib, "charged_gib": charged, "admitted": charged <= capacity}


def dependency_orders():
    """Model diagram 1 with one model request enabling a CPU task."""
    model_work, docker_work, cpu_work = 2, 10, 10
    model_first = max(model_work + docker_work, model_work + cpu_work)
    docker_first = docker_work + model_work + cpu_work
    # Each GPU job progresses at half speed while both are active.
    shared_rate = 0.5
    shared_model_end = model_work / shared_rate
    docker_remaining = docker_work - shared_model_end * shared_rate
    assert docker_remaining >= 0, "This example requires the model job to finish first."
    shared_docker_end = shared_model_end + docker_remaining
    shared_end = max(shared_docker_end, shared_model_end + cpu_work)
    return {
        "model_first_makespan_s": model_first,
        "docker_first_makespan_s": docker_first,
        "shared_makespan_s": shared_end,
        "shared_model_response_s": shared_model_end,
    }


def idle_window(predicted_remaining, actual_remaining, sleep_s=3, wake_s=5):
    """Model diagrams 3 to 6 for one nonpreemptible job and one later request."""
    job_s, cleanup_s, safety_s = 30, 2, 5
    model_s, dependent_cpu_s = 2, 10
    required_window = sleep_s + job_s + cleanup_s + wake_s + safety_s
    admitted = required_window <= predicted_remaining
    # Reference order is model, offload, Docker, confirmed return and restore.
    reference_model_end = actual_remaining + model_s
    reference_restore = reference_model_end + sleep_s + job_s + cleanup_s + wake_s
    reference_end = max(reference_restore, reference_model_end + dependent_cpu_s)
    if admitted:
        job_end = sleep_s + job_s
        physical_return = job_end + cleanup_s
        model_ready = physical_return + wake_s
        model_start = max(actual_remaining, model_ready)
        model_end = model_start + model_s
        makespan = max(model_ready, model_end + dependent_cpu_s)
        events = [
            ["offload", 0, sleep_s],
            ["Docker GPU", sleep_s, job_end],
            ["physical return", job_end, physical_return],
            ["restore", physical_return, model_ready],
            ["model", model_start, model_end],
            ["dependent CPU", model_end, model_end + dependent_cpu_s],
        ]
        model_delay = model_start - actual_remaining
    else:
        makespan = reference_end
        events = []
        model_delay = 0
    return {
        "predicted_remaining_s": predicted_remaining,
        "actual_remaining_s": actual_remaining,
        "required_window_s": required_window,
        "admitted": admitted,
        "model_extra_wait_s": model_delay,
        "reference_makespan_s": reference_end,
        "makespan_s": makespan,
        "saved_s": reference_end - makespan,
        "events": events,
    }


def main():
    """Print deterministic counterexamples, without real waits or GPU work."""
    result = {
        "scope": "CPU teaching model with synthetic values, not a production scheduler or GPU benchmark",
        "memory_fits": memory_admission(4),
        "memory_growth_rejected": memory_admission(8),
        "dependencies": dependency_orders(),
        "long_idle": idle_window(60, 60),
        "early_model_request": idle_window(60, 8),
        "short_idle_rejected": idle_window(15, 15),
        "expensive_offload_rejected": idle_window(60, 60, sleep_s=20, wake_s=15),
    }
    assert result["memory_fits"]["admitted"], "The small request should fit the declared peaks."
    assert not result["memory_growth_rejected"]["admitted"], "Model growth must remain reserved."
    assert result["dependencies"]["shared_makespan_s"] == 14, "Sharing must include downstream delay."
    assert result["long_idle"]["saved_s"] == 30, "The long window should overlap the Docker job."
    assert result["early_model_request"]["saved_s"] == -2, "An early request can increase makespan."
    assert not result["short_idle_rejected"]["admitted"], "The full handoff must fit the window."
    assert not result["expensive_offload_rejected"]["admitted"], "Expensive transfers must count."
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
