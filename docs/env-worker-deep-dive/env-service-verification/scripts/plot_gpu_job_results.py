"""Plot the committed bounded job latency comparison without rerunning GPU work."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parents[2] / "gpu-job-validation"
data = json.loads((directory / "summary.json").read_text())
figure, ax = plt.subplots(figsize=(7, 4))
labels = ["Synchronous GPU call\nthen independent wait", "GPU job + independent wait\nconcurrently"]
values = [data["mean_seconds"][key] for key in ("serial", "overlap")]
bars = ax.bar(labels, values, color=["#65758B", "#16877A"], width=0.55)
for bar, value in zip(bars, values, strict=True):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value:.2f} s", ha="center")
ax.set_ylim(0, max(values) * 1.25)
ax.set_ylabel("Time to finish both activities (seconds)")
ax.set_title("6 s real CUDA + 6 s independent wait, two samples per mode")
ax.spines[["right", "top"]].set_visible(False)
figure.text(
    0.5,
    0.02,
    "Controlled overlap demonstration. No model or training. Not the fixed OFF baseline.",
    ha="center",
    fontsize=9,
)
figure.tight_layout(rect=(0, 0.05, 1, 1))
figure.savefig(directory / "latency.svg")
plt.close(figure)
