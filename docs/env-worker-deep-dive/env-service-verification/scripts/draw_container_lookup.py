"""Render the measured lookup costs and the boundary of the historical trace."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch


def main():
    root = Path(__file__).resolve().parents[2]
    result = json.loads((root / "env-service-verification/results/container-query-20261126/result.json").read_text())
    font = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(), "font.size": 12})
    figure = plt.figure(figsize=(8, 5.6), dpi=100, facecolor="white")
    figure.suptitle("容器 ID 查询：成功路径变轻，历史超时根因仍未知", fontsize=17, y=0.98)
    chart = figure.add_axes([0.11, 0.53, 0.84, 0.32])
    for index, pair in enumerate(result["query_pairs"]):
        for offset, value, color in (
            (0.17, pair["compose_s"] * 1000, "#bf6947"),
            (-0.17, pair["api"]["elapsed_s"] * 1000, "#287c9d"),
        ):
            chart.barh(index + offset, value, height=0.28, color=color)
            chart.text(value * 1.1, index + offset, f"{value:.2f} ms", va="center", fontsize=12)
    chart.set_xscale("log")
    chart.set_xlim(0.7, 1200)
    chart.set_yticks(range(3), ["配对 1", "配对 2", "配对 3"])
    chart.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    chart.set_xlabel("完整单次查询耗时（毫秒，对数轴）", fontsize=12)
    chart.set_title("棕色：Compose     蓝色：本机 Docker API", fontsize=13, pad=10)
    chart.spines[["top", "right"]].set_visible(False)
    panel = figure.add_axes([0.04, 0.05, 0.92, 0.34])
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    panel.axis("off")
    panel.text(0, 0.93, "旧 trace 的可见边界：子进程已创建，communicate 等待超时", fontsize=13)
    panel.add_patch(FancyBboxPatch((0, 0.39), 0.99, 0.43, boxstyle="round,pad=0.005",
                                  facecolor="#fcf0e9", edgecolor="#bf6947", linestyle="--"))
    for x, text in [(0.02, "启动插件"), (0.27, "读取配置"), (0.52, "请求 daemon"), (0.77, "输出／退出")]:
        panel.text(x, 0.64, text, fontsize=13)
        panel.text(x, 0.47, "历史分段耗时未知", fontsize=10.5, color="#8f4a35")
    panel.text(0, 0.21, "新路径记录：连接 → 等响应头 → 读响应体 → 核对归属", fontsize=13, color="#236882")
    panel.text(0, 0.04, "11-26 单节点、一个 CPU 容器；未复现历史卡顿，不代表训练加速比。", fontsize=11)
    target = root / "experiments/container-lookup-trace"
    figure.savefig(target.with_suffix(".png"), dpi=100)
    figure.savefig(target.with_suffix(".svg"))
    plt.close(figure)


if __name__ == "__main__":
    main()
