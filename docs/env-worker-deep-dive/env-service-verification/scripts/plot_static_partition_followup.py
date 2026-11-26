import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

r = Path("docs/env-worker-deep-dive/env-service-verification/results")
a = json.loads((r / "capacity-design-20261119/analysis.json").read_text())["arms"] + [
    json.loads((r / "sync-followup-20261119/static-three-env/analysis.json").read_text())
]
out = Path("docs/env-worker-deep-dive/experiments")
font = FontProperties(family="Droid Sans Fallback", size=13)
fig, axes = plt.subplots(1, 2, figsize=(8, 4.8), layout="constrained")
labels = ["两环境卡\n静态池", "两环境卡\n推理卡共享", "三环境卡\n静态池"]
for ax, vals, title in [
    (axes[0], [x["timing_totals"]["timing_s/step"] for x in a], "四步训练计时（秒）"),
    (axes[1], [x["query_latency_s"]["p95"] for x in a], "模型请求 P95（秒）"),
]:
    ax.bar(range(3), vals, color=["#6686a6", "#e29c43", "#399b83"])
    ax.set_xticks(range(3), labels, fontproperties=font)
    ax.set_title(title, fontproperties=font)
    ax.set_ylim(0, max(vals) * 1.22)
    ax.tick_params(axis="y", labelsize=12)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.03, f"{v:.2f}", ha="center", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("相同六卡与48条轨迹：更强静态分卡超过此前共享", fontproperties=font)
fig.savefig(out / "static-partition-followup.png", dpi=100)
fig.savefig(out / "static-partition-followup.svg")
summary = []
for label, x in zip(labels, a):
    summary.append(
        {
            "label": label,
            **{
                k: x[k]
                for k in [
                    "launch_to_cleanup_s",
                    "timing_totals",
                    "episodes_completed",
                    "commands_completed",
                    "queries_completed",
                    "environment_updates_completed",
                    "query_latency_s",
                    "environment_wait_s",
                    "generated_tokens",
                    "reward",
                    "actions",
                ]
            },
        }
    )
(r / "sync-followup-20261119/three-way-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(
    "static_vs_shared_reduction", 1 - a[2]["timing_totals"]["timing_s/step"] / a[1]["timing_totals"]["timing_s/step"]
)
