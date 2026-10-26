"""Teach reply gating versus GPU entry gating with deterministic timelines, without services."""

import json


def timeline(capacity, demand_a, demand_b, cpu_b, hold_reply):
    # Diagram 2: A is already computing until second 10. A's utilization stays above the gate threshold.
    a_end = 10
    reply_at = a_end if hold_reply else 0
    cpu_end = reply_at + cpu_b
    # Diagram 3: The command waits only when the two declarations exceed memory capacity.
    gpu_start = max(cpu_end, a_end) if demand_a + demand_b > capacity else cpu_end
    gpu_end = gpu_start + 2
    return {
        "B_reply_at_s": reply_at,
        "B_cpu_window_s": [reply_at, cpu_end],
        "B_gpu_queue_wait_s": gpu_start - cpu_end,
        "B_gpu_window_s": [gpu_start, gpu_end],
        "both_tasks_finished_at_s": max(a_end, gpu_end),
    }


def main():
    cases = {
        "overlap_cpu_before_gpu": dict(capacity=80, demand_a=51, demand_b=51, cpu_b=8),
        "memory_allows_concurrent_compute": dict(capacity=80, demand_a=20, demand_b=20, cpu_b=0),
    }
    print(
        json.dumps(
            {
                "scope": "Teaching simulation only, not production execution or measured GPU performance.",
                "assumptions": (
                    "A computes for 10s, B GPU work takes 2s. Zero control overhead, no slowdown or foreign work."
                ),
                "cases": {
                    name: {
                        "inputs_gib_and_seconds": inputs,
                        "command_only": timeline(**inputs, hold_reply=False),
                        "command_plus_reply_gate": timeline(**inputs, hold_reply=True),
                    }
                    for name, inputs in cases.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
