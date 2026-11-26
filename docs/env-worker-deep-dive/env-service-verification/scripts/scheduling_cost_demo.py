"""Calculate illustrative admission and residency costs without using a GPU."""

import json
import logging
import math
from pathlib import Path

psrl_logger = logging.getLogger(__name__)


def residency_case(gap, other_work, save_s=1, reload_s=3, owner_work=5):
    """Compare two jobs on an exclusive GPU with proactive restoration."""
    hold_owner = gap + owner_work
    hold_other = hold_owner + other_work
    release_other = save_s + other_work
    release_owner = max(gap, release_other + reload_s) + owner_work
    return dict(
        gap_s=gap,
        other_work_s=other_work,
        hold=dict(owner_finish_s=hold_owner, other_finish_s=hold_other, all_done_s=hold_other),
        release=dict(owner_finish_s=release_owner, other_finish_s=release_other, all_done_s=release_owner),
        owner_delay_s=release_owner - hold_owner,
        makespan_saving_s=hold_other - release_owner,
    )


def fixed_work_case(gap=10, owner_gpu=5, borrower_gpu=40, owner_cpu_tail=0):
    """Compare identical fixed work with free handoff and an independent CPU continuation."""
    hold_gpu_done = gap + owner_gpu
    lend_gpu_done = max(gap, borrower_gpu) + owner_gpu
    hold_done = max(hold_gpu_done + owner_cpu_tail, hold_gpu_done + borrower_gpu)
    lend_done = max(lend_gpu_done + owner_cpu_tail, borrower_gpu)
    return dict(
        gap_s=gap,
        owner_gpu_s=owner_gpu,
        borrower_gpu_s=borrower_gpu,
        owner_cpu_tail_s=owner_cpu_tail,
        hold_done_s=hold_done,
        lend_done_s=lend_done,
        owner_extra_wait_s=lend_gpu_done - hold_gpu_done,
        total_saving_s=hold_done - lend_done,
    )


def main():
    """Keep measured inputs distinct from hypothetical scheduling outcomes."""
    env_curve = {1: 10, 2: 14, 5: 25, 10: 80, 30: 300}
    admission = [
        dict(concurrency=n, wave_s=t, commands_per_s=n / t, all_done_s=math.ceil(30 / n) * t)
        for n, t in env_curve.items()
    ]
    best = min(admission, key=lambda row: row["all_done_s"])
    assert best["concurrency"] == 5, "The illustrative curve should favor five concurrent commands."
    residency = {
        "long_gap": residency_case(20, 10),
        "short_gap": residency_case(2, 10),
        "long_borrower": residency_case(20, 60),
    }
    assert residency["long_gap"]["makespan_saving_s"] == 10, "The long gap should hide the handoff."
    assert residency["short_gap"]["makespan_saving_s"] == -2, "The short gap should favor retaining the GPU."
    assert residency["long_borrower"]["owner_delay_s"] == 44, "The long borrower should delay the owner."
    assert residency["long_borrower"]["makespan_saving_s"] == 16, "Makespan can improve while the owner suffers."
    fixed_work = {"terminal_gpu": fixed_work_case(), "cpu_continuation": fixed_work_case(owner_cpu_tail=100)}
    assert fixed_work["terminal_gpu"]["total_saving_s"] == 10, "Using a gap can improve total completion."
    assert fixed_work["terminal_gpu"]["owner_extra_wait_s"] == 30, "The owner can wait longer in a faster batch."
    assert fixed_work["cpu_continuation"]["total_saving_s"] == -30, "A delayed continuation can lose useful overlap."
    source = Path(__file__).resolve().parents[2] / "experiments/single-gpu-inference-curve-evidence.json"
    measured = json.loads(source.read_text())
    short = {row["concurrency"]: row for row in measured["rows"] if row["input_tokens"] == 128}
    estimates = {
        "scope": "Algebraic estimates from measured isolated waves, not new scheduling experiments.",
        "two_waves_of_128_s": 256 * 512 / short[128]["output_tokens_per_s"],
        "one_wave_of_256_s": 256 * 512 / short[256]["output_tokens_per_s"],
    }
    result = dict(
        scope="CPU teaching calculations. Environment and residency values are hypothetical.",
        admission=admission,
        residency=residency,
        fixed_work=fixed_work,
        measured_wave_estimates=estimates,
    )
    psrl_logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
