"""Render the literature guide figures using CPU only and Matplotlib."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent
BLUE = "#2563a6"
ORANGE = "#c66b21"
TEAL = "#117f78"
PURPLE = "#7756a4"
RED = "#b43b4c"
INK = "#233447"
MUTED = "#5e6e7c"
GRAY = "#e8edf2"
LIGHT = "#f5f8fb"


def setup():
    """Select a Chinese font without downloading dependencies."""
    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "Droid Sans Fallback",
        "SimHei",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    chosen = next((name for name in candidates if name in installed), None)
    if chosen is None:
        raise RuntimeError("Install a Chinese font listed in the figure README.")
    plt.rcParams.update(
        {
            "font.family": chosen,
            "font.size": 13,
            "text.color": INK,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )


def canvas(title, subtitle, height=7):
    """Create a canvas with coordinates measured from its top left."""
    fig, ax = plt.subplots(figsize=(14, height))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.03)
    ax.set(xlim=(0, 100), ylim=(100, 0))
    ax.axis("off")
    ax.text(2, 4, title, fontsize=22, weight="bold", va="top")
    ax.text(2, 12, subtitle, fontsize=12, color=MUTED, va="top")
    return fig, ax


def note(ax, x, y, text, size=13, **kwargs):
    ax.text(x, y, text, fontsize=size, va="center", **kwargs)


def box(ax, x, y, w, h, title, detail="", color=BLUE, dashed=False):
    """Draw one concept with its role explained inside the same card."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.35,rounding_size=1",
            facecolor=LIGHT,
            edgecolor=color,
            linewidth=1.5,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(
        x + w / 2,
        y + h * (0.32 if detail else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=15,
        color=color,
        weight="bold",
    )
    if detail:
        ax.text(x + w / 2, y + h * 0.69, detail, ha="center", va="center", fontsize=12, linespacing=1.55)


def arrow(ax, start, end, label="", color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.4, color=color))
    if label:
        note(
            ax,
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 - 2.5,
            label,
            11,
            ha="center",
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )


def save(fig, name):
    """Save vector and directly viewable raster versions."""
    for suffix in ("svg", "png"):
        fig.savefig(ROOT / f"{name}.{suffix}", dpi=150)
    plt.close(fig)


def timeline(ax, y, label, segments, end, ticks, x0=22, x1=94):
    """Draw one row using the same time scale as the other rows."""
    note(ax, 2, y + 4, label, 13)
    for tick in ticks:
        x = x0 + tick / end * (x1 - x0)
        ax.plot([x, x], [y - 1, y + 9], color="#d5dee8", linewidth=0.6, zorder=0)
    for start, stop, text, color in segments:
        x = x0 + start / end * (x1 - x0)
        w = (stop - start) / end * (x1 - x0)
        ax.add_patch(Rectangle((x, y), w, 8, facecolor=color, edgecolor="white"))
        if text:
            note(ax, x + w / 2, y + 4, text, 12, ha="center", color=INK if color == GRAY else "white")


def axis_ticks(ax, y, end, ticks, unit="秒", x0=22, x1=94):
    for tick in ticks:
        note(ax, x0 + tick / end * (x1 - x0), y, str(tick), 11, ha="center", color=MUTED)
    note(ax, x1, y + 5, f"从任务提交开始计时，单位：{unit}", 11, ha="right", color=MUTED)


def reading_map():
    fig, ax = canvas("先辨认要减少哪种等待，再选择论文", "概念阅读地图。各论文的机制与证据范围在对应正文说明。", 8)
    items = [
        ("模型等慢工具", "RollArt / DynaRL", "独立推进轨迹\n按阶段瓶颈分资源", BLUE),
        ("任务等空闲 GPU", "Firepool / GPU 任务调研", "提前准备环境\n把就绪工作交给执行池", TEAL),
        ("并发使评分失真", "KREX / AI4Infra", "区分普通执行与测量\n保护会改变奖励的阶段", ORANGE),
        ("策略缺少任务信息", "YARN / 模型调度大脑", "应用声明需求\n资源层快速核验与派发", PURPLE),
    ]
    for x, (question, papers, detail, color) in zip((3, 27, 51, 75), items):
        box(ax, x, 25, 21, 17, question, color=color)
        arrow(ax, (x + 10.5, 43), (x + 10.5, 51))
        box(ax, x, 53, 21, 26, papers.replace(" / ", "\n"), detail, color=color)
    note(ax, 50, 91, "最后回到同一个目标：相同工作量与结果质量下，整批任务能否更早完成？", 15, ha="center")
    save(fig, "reading-map")


def borrowing():
    fig, ax = canvas(
        "同一批 A、B：把 A 等模型的空档借给 B",
        "教学假设：一张 GPU，A 两段计算各 2 秒，中间空档 6 秒。B 在第 2 秒就绪，需要 4 秒。",
        8,
    )
    ticks = [0, 2, 6, 8, 10, 14]
    timeline(
        ax,
        26,
        "保留给 A：GPU",
        [(0, 2, "A₁", BLUE), (2, 8, "预留给 A，但无计算", GRAY), (8, 10, "A₂", BLUE), (10, 14, "B", ORANGE)],
        14,
        ticks,
    )
    timeline(ax, 39, "B 的经历", [(2, 10, "排队 8 秒", GRAY), (10, 14, "执行", ORANGE)], 14, ticks)
    timeline(
        ax,
        60,
        "按命令归还：GPU",
        [(0, 2, "A₁", BLUE), (2, 6, "B", ORANGE), (6, 8, "空闲", GRAY), (8, 10, "A₂", BLUE)],
        14,
        ticks,
    )
    timeline(ax, 73, "B 的经历", [(2, 6, "立即执行", ORANGE)], 14, ticks)
    axis_ticks(ax, 87, 14, ticks)
    note(ax, 3, 97, "整批完成：14 秒 → 10 秒。这里忽略归还与恢复成本，实际决策需把它们加回。", 12, color=MUTED)
    save(fig, "borrowing-timeline")


def rollout():
    fig, ax = canvas(
        "轨迹独立推进：A 提早结束，整组仍可能等 B",
        "教学假设：A、B 各两轮。每轮生成 1 秒，环境执行分别为 A 1 秒、B 5 秒，生成资源充足。",
        8,
    )
    ticks = list(range(0, 13, 2))
    gray = (2, 6, "等 B 的环境", GRAY)
    timeline(
        ax,
        24,
        "按轮同步：A",
        [
            (0, 1, "", PURPLE),
            (1, 2, "", TEAL),
            gray,
            (6, 7, "", PURPLE),
            (7, 8, "", TEAL),
            (8, 12, "等 B，整组结束", GRAY),
        ],
        12,
        ticks,
    )
    timeline(
        ax,
        37,
        "按轮同步：B",
        [(0, 1, "", PURPLE), (1, 6, "环境 B₁", ORANGE), (6, 7, "", PURPLE), (7, 12, "环境 B₂", ORANGE)],
        12,
        ticks,
    )
    timeline(
        ax, 58, "独立推进：A", [(0, 1, "", PURPLE), (1, 2, "", TEAL), (2, 3, "", PURPLE), (3, 4, "", TEAL)], 12, ticks
    )
    timeline(
        ax,
        71,
        "独立推进：B",
        [(0, 1, "", PURPLE), (1, 6, "环境 B₁", ORANGE), (6, 7, "", PURPLE), (7, 12, "环境 B₂", ORANGE)],
        12,
        ticks,
    )
    note(ax, 51, 62, "A 第 4 秒结束，可提前评分或接后续工作", 12, color=TEAL)
    axis_ticks(ax, 86, 12, ticks)
    note(ax, 3, 96, "紫色：模型生成。绿色：A 的环境。橙色：B 的环境。整组完成时间两种方式均为 12 秒。", 12)
    save(fig, "rollout-independence")


def bottleneck():
    fig, ax = canvas(
        "动态分卡：一个阶段变慢，整条流水线反而更快",
        "独立教学模型：总共 8 张卡。两个阶段的工作量已换算成相同单位，忽略流水线首尾。",
        7,
    )
    for i, (allocation, first, second) in enumerate([("4 / 4 张", 10, 4), ("3 / 5 张", 9, 5), ("2 / 6 张", 6, 6)]):
        y = 27 + i * 19
        note(ax, 3, y + 5, allocation, 15)
        for delta, rate, color in [(0, first, BLUE), (7, second, ORANGE)]:
            ax.add_patch(Rectangle((22, y + delta), rate * 5, 5, facecolor=color))
            note(ax, 23 + rate * 5, y + delta + 2.5, str(rate), 12, color=color)
        note(ax, 83, y + 5, f"整条速率 {min(first, second)}", 14, color=TEAL, weight="bold")
    note(ax, 23, 86, "蓝色：阶段一速率    橙色：阶段二速率    单位：标准工作单元 / 秒", 12)
    note(ax, 3, 96, "整体取较小值，因为每个工作单元都要通过两阶段。短任务还要判断迁移成本能否收回。", 12, color=MUTED)
    save(fig, "pipeline-bottleneck")


def drain():
    fig, ax = canvas(
        "先关提交门，再等 GPU 排空，最后才能测量",
        "KREX 的 NVIDIA 路径概念。以下时间是教学数字：B 在第 0 ms 提交，工作到第 5 ms 才结束。",
        7,
    )
    ticks = [0, 2, 5, 8]
    timeline(ax, 26, "B 的设备工作", [(0, 5, "B 已提交的工作继续执行", ORANGE)], 8, ticks)
    timeline(ax, 42, "新的竞争提交", [(0, 2, "允许", TEAL), (2, 8, "门关闭，新工作不能进入", GRAY)], 8, ticks)
    timeline(ax, 58, "A 的测量请求", [(2, 5, "隔离等待 3 ms", GRAY), (5, 8, "受保护测量", BLUE)], 8, ticks)
    axis_ticks(ax, 75, 8, ticks, "毫秒")
    note(ax, 3, 90, "暂停 B 的 CPU 无法撤回已提交的 GPU 工作。排空确认到达后，才可冻结竞争者并整理测量状态。", 12)
    note(ax, 3, 97, "图中测量占 3 ms，仅用于展示顺序。该机制未作为 KREX 实现接入 PSRL。", 11, color=MUTED)
    save(fig, "gate-and-drain")


def regions():
    fig, ax = canvas(
        "保护测量区域，把 CPU 准备重叠起来",
        "教学假设：四条命令各准备 90 ms、测量 10 ms。CPU 足够，忽略 GPU 正确性检查。",
        7,
    )
    ticks = [0, 90, 134, 200, 300, 400]
    serial = []
    for i in range(4):
        serial.extend([(i * 100, i * 100 + 90, f"命令 {i + 1} 准备", TEAL), (i * 100 + 90, (i + 1) * 100, "", BLUE)])
    timeline(ax, 28, "整命令串行", serial, 400, ticks)
    parallel = [(0, 90, "四条同时准备", TEAL)]
    for i in range(4):
        parallel.extend([(90 + i * 11, 91 + i * 11, "", GRAY), (91 + i * 11, 101 + i * 11, "", BLUE)])
    timeline(ax, 49, "仅测量串行", parallel, 400, ticks)
    axis_ticks(ax, 66, 400, ticks, "毫秒")
    note(ax, 3, 84, "整命令：4 × (90 + 10) = 400 ms。区域保护：90 + 4 × (1 + 10) = 134 ms。", 14)
    note(
        ax,
        3,
        95,
        "蓝色测量始终逐条进行。收益来自绿色准备阶段重叠，每次保护另计 1 ms，真实隔离成本需要实测。",
        12,
        color=MUTED,
    )
    save(fig, "protected-regions")


def krex_flow():
    fig, ax = canvas(
        "KREX：普通阶段允许重叠，关键测量轮流独占",
        "依据论文重画的 NVIDIA 路径概念图。编号与精读正文的机制表对应，不是 PSRL 已实现调用链。",
        8,
    )
    cards = [
        (3, 24, "[0] 命令与 context 配对", "复用健康 GPU 执行环境\n导入、编译、准备可重叠", TEAL),
        (36, 24, "[1、2] 取得区域锁并关门", "只选一个测量持有者\n阻止新的竞争提交", BLUE),
        (69, 24, "[3] 等待设备排空", "各参与进程协作确认\n已提交的工作真正结束", ORANGE),
        (69, 59, "[4、5] 保护测量条件", "先排空，再冻结竞争者\n整理缓存并绑定 CPU 核", ORANGE),
        (36, 59, "[6] 预热并执行测量", "在规定条件下计时\n得到候选与参考耗时", BLUE),
        (3, 59, "[7、8、9] 恢复与清理", "先解冻，再开门放锁\n命令清理，健康 context 留用", TEAL),
    ]
    for x, y, title, detail, color in cards:
        box(ax, x, y, 28, 25, title, detail, color=color)
    arrow(ax, (32, 36), (35, 36))
    arrow(ax, (65, 36), (68, 36))
    arrow(ax, (83, 50), (83, 58))
    arrow(ax, (68, 71), (65, 71))
    arrow(ax, (35, 71), (32, 71))
    note(
        ax,
        50,
        94,
        "context 是驱动维护的 GPU 执行环境。复用它可少做初始化，候选私有显存仍须按协议清理。",
        12,
        ha="center",
    )
    save(fig, "krex-control-flow")


def scheduling_lanes():
    fig, ax = canvas(
        "把低频语义分析与每条命令的快速派发分开",
        "设计提案：模型提供信息，普通程序检查约束并执行。编号与模型调度大脑正文对应。",
        8,
    )
    box(ax, 3, 25, 27, 20, "[O1] 代码与版本观测", "脚本、差异、执行历史\n标记采集时间和适用版本", color=PURPLE)
    box(ax, 38, 25, 27, 20, "[S1] 模型分析", "低频提取阶段与依赖\n信息不足时可返回未知", color=PURPLE, dashed=True)
    box(ax, 73, 25, 24, 20, "结构化结果缓存", "绑定版本与有效期\n新代码不复用旧解释", color=PURPLE, dashed=True)
    arrow(ax, (31, 35), (37, 35))
    arrow(ax, (66, 35), (72, 35))
    box(ax, 3, 64, 27, 20, "[V1、P1] 快速决策", "结合当前队列和语义缓存\n检查合法性并比较成本", color=BLUE)
    box(ax, 38, 64, 27, 20, "[E1] 准入与执行", "重新检查资源是否可用\n获准后才运行命令", color=TEAL)
    box(ax, 73, 64, 24, 20, "[R1] 归还与反馈", "核实物理释放\n再更新账本与观测", color=TEAL)
    arrow(ax, (85, 46), (17, 63), "缓存建议供派发使用")
    arrow(ax, (31, 74), (37, 74))
    arrow(ax, (66, 74), (72, 74))
    note(
        ax, 50, 95, "上层分析可以提前进行。下层每次派发仍读取现场资源，模型建议本身不占有 GPU 配额。", 12, ha="center"
    )
    save(fig, "scheduling-two-lanes")


def reward():
    fig, ax = canvas(
        "计时进入奖励后，竞争可能让代码好坏被颠倒",
        "教学假设：同一输入、同一参考 12 μs，奖励为参考时间 / 候选时间。柱长只表示候选耗时。",
        7,
    )
    for y, label, a, b in [(27, "相同无干扰条件", 10, 10.1), (53, "只在 A 测量时加入竞争", 11, 10.201)]:
        note(ax, 2, y - 5, label, 14)
        for delta, name, value, color in [(0, "A", a, BLUE), (9, "B", b, ORANGE)]:
            ax.add_patch(Rectangle((23, y + delta), value * 4, 6, facecolor=color))
            note(ax, 19, y + delta + 3, name, 13)
            note(ax, 25 + value * 4, y + delta + 3, f"{value:g} μs，奖励 {12 / value:.3f}", 13, color=color)
    note(ax, 3, 82, "无干扰时 A 更快，奖励更高。竞争后 A 的代码没变，却因测量条件更差而输给 B。", 14)
    note(
        ax,
        3,
        94,
        "测量条件 → 时间 → 奖励 → 下一轮保留的候选或训练信号。这条反馈链使调度影响搜索方向。",
        12,
        color=MUTED,
    )
    save(fig, "reward-distortion")


def benchmarks():
    fig, ax = canvas(
        "选择 benchmark：先确定结果按什么判好坏",
        "阅读分类图，任务名仅是代表。具体题目、评分协议与本地接入范围仍以正文为准。",
        8,
    )
    box(ax, 29, 23, 42, 13, "同一份正确程序受硬件竞争拖慢", color=BLUE)
    arrow(ax, (39, 37), (24, 48))
    arrow(ax, (61, 37), (76, 48))
    box(
        ax,
        3,
        50,
        43,
        24,
        "正确性或固定工作量后的质量评分",
        "例如固定训练步数后的验证分数\n共享可增加延迟，分数需另核对数值行为",
        color=TEAL,
    )
    box(
        ax,
        54,
        50,
        43,
        24,
        "耗时、吞吐或固定时间内的质量评分",
        "例如 KernelBench、训练 speedrun\n竞争会直接改变分数或可完成的工作量",
        color=ORANGE,
    )
    note(ax, 25, 84, "先研究整批时间、排队与资源回收", 13, ha="center", color=TEAL)
    note(ax, 76, 84, "同时控制测量条件与真实工作量", 13, ha="center", color=ORANGE)
    note(
        ax,
        50,
        96,
        "同一个题库也可有不同协议。固定步数与固定时长，会把同一训练任务变成不同的调度问题。",
        12,
        ha="center",
    )
    save(fig, "benchmark-choice")


def model_cost():
    fig, ax = canvas(
        "模型参与决策时，要把它自身的等待也算进去",
        "教学假设：同一条 4 秒 GPU 命令。在原卡排队 12 秒，换到空闲卡需加载 2 秒，忽略其他成本。",
        7,
    )
    ticks = [0, 4, 6, 12, 16, 18]
    timeline(ax, 25, "原卡等待", [(0, 12, "排队 12 秒", GRAY), (12, 16, "执行 4 秒", BLUE)], 18, ticks)
    timeline(ax, 42, "低成本换卡", [(0, 2, "加载", TEAL), (2, 6, "执行", BLUE)], 18, ticks)
    timeline(
        ax,
        59,
        "先等模型再换卡",
        [(0, 12, "模型分析 12 秒", PURPLE), (12, 14, "加载", TEAL), (14, 18, "执行", BLUE)],
        18,
        ticks,
    )
    axis_ticks(ax, 76, 18, ticks)
    note(ax, 3, 90, "换卡原本 6 秒完成，加上同步模型分析却要 18 秒，比原卡等待的 16 秒还慢。", 13)
    note(ax, 3, 98, "低频提前分析代码、缓存语义信息，再用普通程序快速派发，是本文建议检验的路线。", 11, color=MUTED)
    save(fig, "model-decision-cost")


def yarn():
    fig, ax = canvas(
        "YARN 的启发：应用说清需要什么，资源层决定怎样分",
        "左侧是 YARN 的角色，右侧是 PSRL 对应分析。虚线框中的跨候选比较协议仍是提案。",
        9,
    )
    rows = [
        (
            23,
            "ApplicationMaster",
            "知道应用依赖与完成条件",
            "[1] 应用 / 评分协调者",
            "哪些候选要比较，何时需要反馈",
            PURPLE,
            True,
        ),
        (
            43,
            "ResourceManager",
            "统筹集群资源分配",
            "[2] EnvRouter 与账本",
            "已有资源准入，测量窗口策略待补",
            BLUE,
            False,
        ),
        (
            63,
            "NodeManager",
            "管理节点上的启动与回收",
            "[3] 节点服务 / EnvWorker",
            "执行容器操作并核实资源归还",
            TEAL,
            False,
        ),
        (
            83,
            "Container",
            "一次资源分配与执行抽象",
            "[4] GPU 命令 / 评估器",
            "执行并返回结果，统一比较条件待补",
            ORANGE,
            False,
        ),
    ]
    for y, left, desc, right, rdesc, color, dashed in rows:
        box(ax, 2, y, 40, 13, left, desc, color=color)
        box(ax, 56, y, 42, 13, right, rdesc, color=color, dashed=dashed)
        arrow(ax, (43, y + 6), (54, y + 6), "类比")
    for y in (37, 57, 77):
        arrow(ax, (22, y), (22, y + 5))
        arrow(ax, (77, y), (77, y + 5))
    save(fig, "yarn-responsibilities")


def rsi():
    fig, ax = canvas(
        "任务里训练了模型，不等于 Agent 自己换成了那个模型",
        "A：外层 Agent 参数。B：任务里训练的模型参数。上半部是已核对的路径，下半部是接管提案。",
        9,
    )
    box(ax, 2, 24, 27, 16, "外层 Agent A", "生成训练与评估命令", color=BLUE)
    box(ax, 37, 24, 27, 16, "Docker 内训练 B", "任务模型参数发生变化", color=TEAL)
    box(ax, 72, 24, 26, 16, "独立评分 B", "返回分数与交互轨迹", color=ORANGE)
    arrow(ax, (30, 32), (36, 32))
    arrow(ax, (65, 32), (71, 32))
    box(ax, 31, 50, 38, 17, "PSRL 更新 A 并发布新 A", "已有：trainer → 参数服务 → vLLM", color=BLUE)
    arrow(ax, (85, 41), (69, 56))
    arrow(ax, (31, 56), (16, 41), "后续 rollout")
    ax.plot([3, 97], [74, 74], color="#d5dee8", linewidth=1)
    note(ax, 2, 80, "额外的接管提案", 14, color=RED, weight="bold")
    box(ax, 25, 83, 21, 13, "B 满足 Agent 职责", "可恢复且独立评估通过", color=RED, dashed=True)
    box(ax, 52, 83, 21, 13, "新服务加载就绪", "实际接请求并可探活", color=RED, dashed=True)
    box(ax, 79, 83, 19, 13, "切换默认路由", "下一轮由 B 发命令", color=RED, dashed=True)
    arrow(ax, (47, 89), (51, 89), color=RED)
    arrow(ax, (74, 89), (78, 89), color=RED)
    save(fig, "rsi-two-loops")


def task_lifecycle():
    fig, ax = canvas(
        "一项 GPU 环境任务：代码与容器保留，计算按协议申请",
        "science 公共执行关系的简化图。具体 benchmark 与 direct / MPS 分支条件在正文核对。",
        8,
    )
    box(ax, 3, 24, 25, 18, "[1] 任务与资源需求", "输入代码、数据与评分协议", color=BLUE)
    box(ax, 37, 24, 26, 18, "[2] EnvRouter 准入", "返回获准使用的资源租约", color=BLUE)
    box(ax, 72, 24, 25, 18, "[3] Agent 与环境", "改代码、运行、读反馈", color=PURPLE)
    arrow(ax, (29, 33), (36, 33))
    arrow(ax, (64, 33), (71, 33))
    box(ax, 72, 58, 25, 21, "候选实验", "训练模型 / kernel / 服务\nGPU 工作遵守命令持有策略", color=TEAL)
    arrow(ax, (79, 43), (79, 57), "命令")
    arrow(ax, (91, 57), (91, 43), "反馈")
    box(ax, 37, 58, 26, 21, "[4] 独立 verifier", "重检最终提交并评分\n结果进入任务后续处理", color=ORANGE)
    box(ax, 3, 58, 25, 21, "[5] 清理与归还", "停止所属进程与容器\n核实释放后归还预算", color=TEAL)
    arrow(ax, (71, 69), (64, 69), "最终提交")
    arrow(ax, (36, 69), (29, 69))
    note(
        ax,
        50,
        93,
        "租约是使用资源的许可。保留工作文件、允许访问 GPU、设备已物理释放，是三个不同状态。",
        12,
        ha="center",
    )
    save(fig, "gpu-task-lifecycle")


def psrl_loop():
    fig, ax = canvas(
        "PSRL：做题和更新参数通过样本与版本衔接",
        "简化数据流，对应阅读指南中的 1 至 7。重试、验证与可选优化省略，实际训练触发取决于批次配置。",
        8,
    )
    box(ax, 3, 29, 26, 20, "[1、2] Manager / Worker", "派发题目，组织交互\n一个任务可多次请求模型", color=BLUE)
    box(ax, 37, 29, 26, 20, "[3、4] 生成 Router / vLLM", "选择实例，生成下一动作\n生成结果回到原任务", color=PURPLE)
    box(ax, 71, 29, 26, 20, "环境工具（按需调用）", "CPU 或 GPU 执行命令\n工具输出回到原任务", color=TEAL)
    arrow(ax, (30, 36), (36, 36), "请求")
    arrow(ax, (36, 44), (30, 44))
    ax.plot([16, 16, 84, 84], [28, 24, 24, 28], color=TEAL, linewidth=1.4)
    note(ax, 50, 21, "Worker 按需调用环境，接收工具输出", 11, ha="center", color=TEAL)
    box(ax, 3, 62, 26, 20, "[5] 评分与样本存储", "按规则组织轨迹与题目组\nTransferQueue 保存训练字段", color=BLUE)
    box(ax, 37, 62, 26, 20, "[6] 训练端", "所需一批样本满足条件后\n读取数据并更新模型", color=ORANGE)
    box(ax, 71, 62, 26, 20, "[7] 参数服务", "保存并发布新版本\n生成实例按协调策略拉取", color=PURPLE)
    arrow(ax, (16, 50), (16, 61), "完成的轨迹")
    arrow(ax, (30, 72), (36, 72))
    arrow(ax, (64, 72), (70, 72))
    arrow(ax, (84, 61), (50, 50), "新模型参数")
    note(ax, 50, 95, "生成 Router 分发模型请求，EnvRouter 管环境资源。二者名字相近，管理的对象不同。", 12, ha="center")
    save(fig, "psrl-training-loop")


def roofline():
    """Connect the hardware upper bound to an idealized decode batch."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(left=0.07, right=0.92, top=0.76, bottom=0.2, wspace=0.34)
    fig.suptitle("Roofline 与合批：同一个瓶颈，用两种横轴观察", fontsize=22, y=0.97)
    fig.text(
        0.5,
        0.86,
        "教学假设：算力 100 TFLOP/s，带宽 2 TB/s，bf16 权重 6 GB。忽略注意力、通信与其他开销。",
        ha="center",
        fontsize=12,
        color=MUTED,
    )
    intensity = np.geomspace(1, 200, 200)
    perf = np.minimum(100, 2 * intensity)
    left, right = axes
    left.plot(intensity, perf, color=BLUE, linewidth=3)
    left.scatter([50], [100], color=ORANGE, zorder=3)
    left.annotate(
        "交点 I = 100 / 2 = 50",
        xy=(50, 100),
        xytext=(6, 120),
        arrowprops={"arrowstyle": "->", "color": MUTED},
        fontsize=11,
    )
    left.set(
        xscale="log",
        yscale="log",
        xlabel="算术强度 I（FLOP / 字节）",
        ylabel="性能上界（TFLOP/s）",
        ylim=(1, 180),
        title="经典 Roofline：算力与带宽共同约束",
    )
    left.text(2, 25, "带宽限制：2 × I", color=BLUE, fontsize=11)
    left.text(65, 82, "算力限制：100", color=BLUE, fontsize=11)
    batch = np.arange(1, 129)
    latency = np.maximum(3, 0.06 * batch)
    throughput = batch / (latency / 1000)
    (latency_line,) = right.plot(batch, latency, color=BLUE, linewidth=2.5, label="每轮 decode 耗时")
    right.set(
        xlabel="同一轮 decode 的请求数 B", ylabel="单轮耗时（毫秒）", ylim=(0, 9), title="理想合批：延迟与吞吐怎样变化"
    )
    other = right.twinx()
    (throughput_line,) = other.plot(batch, throughput, color=ORANGE, linewidth=2.5, label="每秒完成的请求步数")
    other.set(ylabel="吞吐（请求步 / 秒）", ylim=(0, 20000))
    right.axvline(50, color=MUTED, linestyle="--", linewidth=1)
    right.legend(handles=[latency_line, throughput_line], loc="upper left", fontsize=10)
    right.text(57, 2, "理想拐点 B = 50", fontsize=11, color=MUTED)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.04,
        "一条请求步表示一条请求前进一个 token。这里的拐点来自简化公式，实际服务需测量，不能当作环境命令的并发上限。",
        ha="center",
        fontsize=11,
        color=MUTED,
    )
    save(fig, "roofline-and-batching")


def main():
    """Regenerate the complete teaching figure set."""
    setup()
    renderers = [
        reading_map,
        borrowing,
        rollout,
        bottleneck,
        drain,
        regions,
        krex_flow,
        scheduling_lanes,
        reward,
        benchmarks,
        model_cost,
        yarn,
        rsi,
        task_lifecycle,
        psrl_loop,
        roofline,
    ]
    for render in renderers:
        render()
    names = [
        "reading-map",
        "borrowing-timeline",
        "rollout-independence",
        "pipeline-bottleneck",
        "gate-and-drain",
        "protected-regions",
        "reward-distortion",
        "benchmark-choice",
        "model-decision-cost",
        "yarn-responsibilities",
        "rsi-two-loops",
        "krex-control-flow",
        "scheduling-two-lanes",
        "gpu-task-lifecycle",
        "psrl-training-loop",
        "roofline-and-batching",
    ]
    sheet = Image.new("RGB", (1400, 3360), "#e8edf2")
    for index, name in enumerate(names):
        with Image.open(ROOT / f"{name}.png") as source:
            preview = source.convert("RGB")
        preview.thumbnail((690, 410))
        sheet.paste(preview, ((index % 2) * 700 + 5, (index // 2) * 420 + 5))
    sheet.save(ROOT / "preview.png")
    print(f"Rendered {len(renderers)} figures as SVG and PNG in {ROOT}.")


if __name__ == "__main__":
    main()
