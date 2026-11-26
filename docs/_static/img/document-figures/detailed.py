"""Draw chapter explanations with shared scenarios and explicit causal comparisons."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

INK = "#203149"
MUTED = "#53677c"
BLUE, GREEN, ORANGE, PURPLE, RED = "#3478c5", "#159b8d", "#d68b20", "#8764b8", "#c84f60"
TINT = {BLUE: "#eaf2ff", GREEN: "#e6f6f2", ORANGE: "#fff2db", PURPLE: "#f2edfb", RED: "#fdecef"}
LABELS = []


def label(ax, x, y, value, w=94, h=9, size=16, color=INK, bold=False, center=False):
    """Retain physical bounds so crowded text fails instead of becoming tiny."""
    artist = ax.text(
        x,
        y,
        value,
        ha="center" if center else "left",
        va="center",
        fontsize=max(size, 16.5),
        color=color,
        weight="bold" if bold else "normal",
        linespacing=1.4,
    )
    LABELS.append((artist, ax, (x - w / 2 if center else x, y - h / 2, w, h)))
    return artist


def box(ax, x, y, w, h, color=BLUE, text=None, hatch=None):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=TINT.get(color, "#eff3f7"), edgecolor=color, linewidth=1.3, hatch=hatch)
    )
    if text:
        label(ax, x + w / 2, y + h / 2, text, w=w - 2, h=h - 2, center=True)


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=color))


def note(ax, text, color=BLUE, y=88, h=18):
    box(ax, 2, y - h / 2, 96, h, color)
    label(ax, 5, y, text, w=90, h=h - 2, size=15, color=color, bold=True)


def heading(ax, title):
    ax.set(xlim=(0, 100), ylim=(100, 0))
    ax.axis("off")
    label(ax, 2, 6, title, w=96, h=10, size=18, bold=True)
    ax.plot([2, 98], [14, 14], color="#d8e2ed", lw=1)


def bar(ax, y, parts, start=26, width=68, total=16, h=11):
    """Draw proportional intervals on the panel's common time or capacity axis."""
    cursor = start
    for name, amount, color in parts:
        segment = width * amount / total
        box(ax, cursor, y, segment, h, color)
        if name:
            label(ax, cursor + segment / 2, y + h / 2, name, w=segment - 1, h=h - 1, size=15, center=True)
        cursor += segment


def ticks(ax, y, values, total, start=26, width=68):
    for value in values:
        label(ax, start + width * value / total, y, str(value), w=5, h=8, size=16.5, color=MUTED, center=True)


def memory(axes):
    a, b, c, d = axes
    heading(a, "① 初始状态：实占与承诺会重叠吗？")
    label(a, 3, 24, "同一张卡：容量 90 GiB，外部常驻 70 GiB")
    bar(a, 35, [("外部实占 70", 70, BLUE), ("空余 20", 20, GREEN)], start=4, width=92, total=90, h=15)
    box(a, 4, 59, 43, 19, ORANGE, "A 已获准：预约 15\n尚未分配：实占 0")
    box(a, 53, 59, 43, 19, PURPLE, "B 新申请：15\n现在放行，还是等？")
    note(a, "要保住 A 后续分配的权利，再判断 B。")

    heading(b, "② 对比算法：旧公式为什么会漏算？")
    label(b, 3, 25, "旧：max(整卡实占 70, 已预约 15) = 70", size=15, color=RED)
    bar(b, 37, [("外部 70", 70, BLUE), ("A 14", 14, ORANGE), ("B 14", 14, PURPLE)], start=4, width=90, total=105, h=14)
    boundary = 4 + 90 * 90 / 105
    b.plot([boundary, boundary], [34, 54], color=RED, lw=2, ls="--")
    label(b, boundary - 8, 32, "容量 90", w=17, h=7, size=13.5, color=RED)
    label(b, 4, 61, "错放 B 后：70 + 14 + 14 = 98 > 90\n先获准、后分配的时间差，会触发 OOM。", h=16, size=15)
    note(b, "把实占与预约全加也不对：\nA 已分配的部分会被重复计算。", color=RED)

    heading(c, "③ 正确记账：只加尚未兑现的预约")
    label(c, 4, 25, "U = M + Σ max(Rᵢ − Aᵢ, 0)", size=20, bold=True)
    label(c, 4, 41, "M：整卡本次实占   Rᵢ：任务 i 的预约\nAᵢ：本次可靠核实属于任务 i 的实占", h=16, size=15)
    for y, name, allocated in [(57, "A 未分配", 0), (73, "A 已用 14", 14)]:
        label(c, 4, y + 4, name, w=22, h=10, size=14)
        bar(
            c,
            y,
            [
                ("70", 70, BLUE),
                (str(allocated) if allocated else "", allocated, GREEN),
                ("" if allocated else "15", 15 - allocated, ORANGE),
            ],
            start=28,
            width=57,
            total=90,
            h=10,
        )
        if 15 - allocated:
            box(c, 28 + 57 * (70 + allocated) / 90, y, 57 * (15 - allocated) / 90, 10, ORANGE, hatch="///")
        label(c, 87, y + 5, "85", w=10, h=9, bold=True)
    label(c, 4, 94, "蓝＝外部实占，绿＝A 实占，斜线＝剩余承诺", size=13.5)

    heading(d, "④ 放行与归还：低占用何时才是真空余？")
    label(d, 4, 26, "A 仍持预约：U = 85\nB 需 15：85 + 15 = 100 > 90 → 等待", h=17, size=16)
    label(d, 4, 50, "A 仅释放张量：预约仍在，B 继续等。\nA 确认归还：清预约，U 降到 70。", h=17)
    box(d, 4, 64, 92, 12, GREEN, "此时再放 B：70 + 15 = 85 ≤ 90")
    note(d, "按任务抵扣，拒用历史峰值。\n归属不明时不抵扣，容量判断更保守。", color=GREEN)


def placement(axes):
    a, b, c, d = axes
    heading(a, "① 请求 A 就绪：两张卡准备情况不同")
    box(a, 4, 24, 43, 34, BLUE, "GPU0：热卡\nA 模型已加载\nB 还要运行 12 秒")
    box(a, 53, 24, 43, 34, GREEN, "GPU1：冷卡\n当前空闲\n加载 A 需 5 秒")
    label(a, 4, 69, "A 的执行本身都需 4 秒；从此刻 t=0 计时。\nA = 新请求，B = 热卡上的在途任务。", h=17, size=15)
    note(a, "先确认模型版本、容量与候选设备合格。")

    heading(b, "② 同一时间轴：等待还是在空卡重建？")
    ticks(b, 27, [0, 4, 8, 12, 16], 16)
    label(b, 3, 46, "等热卡", w=21)
    bar(b, 40, [("等 B：12", 12, ORANGE), ("A：4", 4, BLUE)])
    label(b, 3, 65, "转冷卡", w=21)
    bar(b, 59, [("加载 5", 5, PURPLE), ("A：4", 4, GREEN)])
    label(b, 27, 77, "单位：秒。所有横条从 A 就绪时开始。", w=69, size=14)
    note(b, "A 完成：16 秒 → 9 秒，提前 7 秒。", color=GREEN, y=90, h=14)

    heading(c, "③ 决策规则：多快才值得重新加载？")
    label(c, 4, 27, "等热卡 = 队列剩余 Q + A 执行 4\n转冷卡 = 冷卡等待 + 加载 5 + 执行 4", h=19)
    label(c, 4, 51, "教学切换门槛取 2 秒：\n冷卡成本 + 2 < 热卡成本，才切换。", h=17)
    box(c, 4, 64, 92, 12, GREEN, "Q=12：9 + 2 < 16，选冷卡")
    note(c, "只把 Q 改成 1 秒：热卡共 5 秒，\n比冷卡 9 秒更快，此时留在热卡。", color=ORANGE)

    heading(d, "④ 谁移动，谁保留？以及如何看整体收益")
    box(d, 4, 24, 92, 17, BLUE, "同机原容器：文件、CPU、主存继续保留")
    arrow(d, 49, 42, 49, 49)
    box(d, 4, 50, 43, 21, ORANGE, "GPU0 上的 B\n继续执行到 t=12")
    box(d, 53, 50, 43, 21, GREEN, "GPU1 重建 A\n完成于 t=9")
    note(d, "仅比较 A、B：整批由 16 秒变 12 秒。\n收益 4 秒，与 A 提前 7 秒不同。", color=GREEN)


def loan(axes):
    a, b, c, d = axes
    heading(a, "① 同一场景：A 暂停，B 正在等卡")
    box(a, 4, 25, 43, 27, BLUE, "A：原持卡任务\n空档剩余 6 秒\n之后继续计算 4 秒")
    box(a, 53, 25, 43, 27, GREEN, "B：等待任务\n需要 GPU 8 秒\n执行中不抢占")
    label(a, 4, 66, "借出前保存并归还：1 秒\nB 结束后恢复 A：1 秒；简化为串行。", h=18)
    note(a, "t=0 从 A 空档开始计。无其他任务干扰。")

    heading(b, "② 一张卡两种策略：谁何时能完成？")
    ticks(b, 26, [0, 6, 10, 14, 18], 18, start=22, width=72)
    label(b, 2, 43, "保留给 A", w=19, size=14)
    bar(b, 35, [("A 空档 6", 6, ORANGE), ("A 4", 4, BLUE), ("B 8", 8, GREEN)], start=22, width=72, total=18, h=15)
    label(b, 2, 67, "借给 B", w=19, size=14)
    bar(
        b,
        59,
        [("", 1, PURPLE), ("B 8", 8, GREEN), ("", 1, PURPLE), ("A 4", 4, BLUE)],
        start=22,
        width=72,
        total=18,
        h=15,
    )
    label(b, 23, 81, "紫色小段各 1 秒：保存归还、恢复。", w=72, size=14)
    label(b, 4, 94, "借出后 A 在 t=10 才能继续，比原计划晚 4 秒。", size=14)

    heading(c, "③ 额外等待怎样算，为什么取 max？")
    label(c, 4, 27, "卡重新可供 A 使用：1 + 8 + 1 = 10 秒\nA 原本只空闲 6 秒，超出部分需等待。", h=19)
    box(c, 4, 44, 92, 16, BLUE, "A 额外等待 = max(0, 1 + 8 + 1 − 6)\n= 4 秒")
    label(c, 4, 71, "只把 B 改成 2 秒：1 + 2 + 1 = 4 < 6\n借还落在空档内，额外等待为 0。", h=19, size=15)
    note(c, "max 取零下界：提前归还不产生负等待。")

    heading(d, "④ A 晚了，整批仍可能更早结束")
    label(d, 4, 25, "保留：A 于 10 秒结束，B 于 18 秒结束。\n借出：B 于 9 秒结束，A 于 14 秒结束。", h=19)
    box(d, 4, 42, 92, 14, GREEN, "整批：18 → 14 秒，节省 4 秒")
    label(d, 4, 67, "需同时判断：A 多等 4 秒是否可接受？\n保存与恢复成本能否被填空收益覆盖？", h=17, size=15)
    note(d, "有收益仍需安全：关闭新借用 → 确认\nB 物理归还 → 恢复 A。协议范围见正文。", color=RED)


def ledgers(axes):
    a, b, c, d = axes
    heading(a, "① 会话放在哪台机器，命令用哪张卡？")
    box(a, 4, 23, 92, 52, BLUE)
    label(a, 8, 31, "节点 N：一个物理主机", bold=True)
    box(a, 9, 40, 37, 27, BLUE, "会话 S 的容器\n代码、文件\nCPU 与主存")
    box(a, 54, 40, 35, 12, GREEN, "GPU0：首条命令")
    box(a, 54, 57, 35, 12, PURPLE, "GPU1：下条候选")
    arrow(a, 46, 47, 53, 47)
    arrow(a, 46, 63, 53, 63)
    note(a, "同机换卡只改变 GPU 命令的目标，\n容器与主存没有搬到另一台主机。")

    heading(b, "② 两层资源账本：分别记录什么？")
    box(b, 4, 23, 43, 34, BLUE, "会话层\n容器槽位、CPU\n主存、会话身份\n通常保留至会话结束")
    box(b, 53, 23, 43, 34, GREEN, "GPU 命令层\n目标设备、显存预算\n活动命令与授权周期\n获准到物理归还")
    label(b, 4, 70, "unit 是声明资源范围的准入单元。\n账本额度决定能否进，设备遥测核对实占。", h=19, size=15)
    note(b, "额度是资源承诺，不能直接当成计算占用率。")

    heading(c, "③ 同一会话执行两条 GPU 命令")
    ticks(c, 25, [0, 20, 30, 40], 40, start=26, width=68)
    label(c, 3, 40, "容器/主存", w=21, size=14)
    bar(c, 34, [("一直保留在节点 N", 40, BLUE)], total=40)
    label(c, 3, 59, "GPU0", w=21)
    bar(c, 53, [("命令1：20秒", 20, GREEN)], total=40)
    label(c, 3, 77, "GPU1", w=21)
    bar(c, 71, [("", 30, "#aab8c7"), ("命令2", 10, PURPLE)], total=40)
    label(c, 4, 94, "20 到 30 秒为 Agent 思考，命令2仍须重新准入。", size=14)

    heading(d, "④ 实现边界：换卡并不等于状态迁移")
    box(d, 4, 23, 92, 18, GREEN, "可选同机 GPU：必须在配置的候选池内")
    label(
        d,
        4,
        54,
        "原卡忙：比较等待时间与换卡准备时间。\n热模型仍在旧卡：新卡可能需要重新加载。\n后台进程仍占卡：不能先把预算算作已归还。",
        h=27,
        size=15,
    )
    note(d, "跨节点需另做文件、主存与进程状态迁移。\n本图仅解释同机命令调度的资源边界。", color=PURPLE)


def version(axes):
    a, b, c, d = axes
    heading(a, "① 修改发生了，但常驻进程还没变")
    box(a, 4, 25, 41, 27, BLUE, "磁盘\nscore.py：v2\n权重文件：W1")
    box(a, 55, 25, 41, 27, ORANGE, "常驻进程\n评分函数：v1\n已加载权重：W1")
    arrow(a, 46, 38, 54, 38, RED)
    label(a, 4, 67, "新请求要求 v2，但进程可能继续调用 v1。\n写文件改变磁盘内容，不会自动改内存对象。", h=20, size=15)
    note(a, "需要区分可复用的权重与必须更新的评分代码。")

    heading(b, "② 按依赖记录版本，而非清空全部状态")
    box(b, 4, 25, 43, 17, BLUE, "权重 W1 → 内存模型")
    box(b, 53, 25, 43, 17, GREEN, "评分 v2 → 评分函数")
    box(b, 4, 51, 43, 17, PURPLE, "数据 D1 → 已处理输入")
    box(b, 53, 51, 43, 17, ORANGE, "配置 C1 → 服务选项")
    note(b, "适配器声明依赖关系。版本标签是教学简写，\n实际身份需覆盖文件内容与相关配置。")

    heading(c, "③ 这一次请求：先核对，再选择更新动作")
    label(c, 4, 25, "对照期望与实际：W1=W1，D1=D1，C1=C1\n只有评分代码不匹配：期望 v2，实际 v1。", h=20, size=15)
    box(c, 4, 44, 43, 24, GREEN, "保留\n模型权重与输入\n避免重复准备")
    box(c, 53, 44, 43, 24, ORANGE, "重建\n受影响评分状态\n更新后再核对")
    note(c, "确认代码与状态一致后才执行并发布结果。\n无法确认或更新失败，就拒绝本次复用。", color=GREEN)

    heading(d, "④ 评价复用：正确结果与准备代价都要看")
    label(
        d,
        4,
        26,
        "每次新进程：重新 import、加载 W1、加载 v2。\n无检查常驻：省准备，却可能返回 v1 的结果。",
        h=21,
        size=15,
    )
    box(
        d,
        4,
        45,
        92,
        26,
        GREEN,
        "执行前固定本次文件快照\n保留 W1，新评分进程读取本次快照\n发布前再核对版本，检查成本也计时",
    )
    note(d, "若执行中发生相关文件的新修改，\n要求当前版本的请求应拒收旧版本结果。", color=PURPLE)


def residency(axes):
    a, b, c, d = axes
    heading(a, "① 同一会话：两次计算，中间思考 10 秒")
    ticks(a, 25, [0, 20, 30, 40], 40, start=22, width=72)
    label(a, 2, 44, "A 任务", w=18)
    bar(a, 36, [("GPU 20", 20, BLUE), ("空档10", 10, ORANGE), ("GPU10", 10, BLUE)], start=22, width=72, total=40, h=15)
    label(a, 4, 67, "容器和文件贯穿 40 秒，GPU 只计算 30 秒。\nB 正在等卡，能否进入中间这 10 秒？", h=20, size=15)
    note(a, "比较相同任务与空档，不改变 A 的工作量。")

    heading(b, "② 两种模式改变 GPU 使用权的寿命")
    ticks(b, 25, [0, 20, 30, 40], 40, start=26, width=68)
    label(b, 3, 43, "episode", w=21)
    bar(b, 36, [("GPU 预约一直留给 A", 40, BLUE)], total=40, h=14)
    label(b, 3, 66, "command", w=21)
    bar(b, 59, [("A 持有", 20, GREEN), ("可借 B", 10, ORANGE), ("A再申请", 10, GREEN)], total=40, h=14)
    note(b, "command 的空档从物理归还后开始，\nB 未及时结束时，A 下条命令仍可能排队。")

    heading(c, "③ 两种模式都不自动保存退出的 Python")
    box(c, 4, 23, 92, 13, BLUE, "容器文件：代码、输出、检查点可以保留")
    box(c, 4, 44, 43, 28, ORANGE, "命令1进程退出\nPython 内存消失\nGPU 张量随之释放")
    box(c, 53, 44, 43, 28, GREEN, "命令2新进程\n通常重新 import\n按需重载模型")
    note(c, "要保住模型内存，需显式常驻服务。\n保留 GPU 预约本身不会做到这一点。", color=PURPLE)

    heading(d, "④ auto 第一版：先验能力，不靠猜命令")
    label(
        d,
        4,
        26,
        "创建租约时检查适配器的执行能力声明：\n未知能力或常驻状态 → episode\n可独立归还的命令 → 继续检查成本建议",
        h=28,
        size=15,
    )
    box(
        d, 4, 47, 92, 24, GREEN, "有可靠预测：按借还建议选择\n无预测：默认 command\n若配置必须有预测，缺预测则 episode"
    )
    note(d, "模式在本次会话内固定，并记录选择理由。\n安全可归还，还要进一步判断是否值得归还。", color=ORANGE)


def compute(axes):
    a, b, c, d = axes
    heading(a, "① 显存容量：装得下只回答第一道问题")
    label(a, 4, 25, "教学卡容量 90 GiB，每条命令需要 8 GiB。")
    for i in range(10):
        box(a, 4 + (i % 5) * 18.5, 38 + (i // 5) * 17, 17, 14, BLUE if i % 2 == 0 else GREEN, str(i + 1))
    note(a, "10 条共 80 GiB，容量可容纳。\n实际还须预留上下文、峰值与安全余量。")

    heading(b, "② 计算竞争：MPS 也不会增加 SM 总数")
    label(b, 4, 26, "SM 是 GPU 执行计算的单元，网格仅作示意。", size=15)
    for row in range(3):
        for col in range(8):
            color = BLUE if (row * 8 + col) % 3 else GREEN
            box(b, 5 + col * 11.4, 39 + row * 9.5, 10, 8, color)
    label(b, 4, 77, "蓝＝A 工作，绿＝B 工作；可交错或并行。", size=15)
    label(b, 4, 92, "MPS 可支持跨进程共享，不能保证互不减速。", size=14, color=RED)

    heading(c, "③ 同样 10 条命令：总时间由竞争决定")
    values = [10, 14, 30]
    names = ["10 条一起：轻竞争", "分两组，每组 5 条", "10 条一起：重竞争"]
    colors = [GREEN, BLUE, RED]
    for y, value, name, color in zip([30, 51, 72], values, names, colors, strict=True):
        label(c, 4, y - 8, name, size=14)
        box(c, 4, y, value / 30 * 79, 8, color)
        label(c, 7 + value / 30 * 79, y + 4, str(value), w=10, h=8, size=15)
    label(c, 4, 94, "假设每组 5 条用 7 秒。横条＝整批完成秒数。", size=14)

    heading(d, "④ 准入上限：按吞吐与尾部等待实测选择")
    label(d, 4, 26, "只有 10 条一起需 30 秒时，5+5 的 14 秒\n才更好；若一起只需 10 秒，就应一起跑。", h=21, size=15)
    box(
        d,
        4,
        45,
        92,
        24,
        GREEN,
        "固定总工作量，扫描并发 1、2、4、…\n记录整批耗时、每条延迟、吞吐、峰值\n混合推理时，再检查推理和整轮训练完成",
    )
    note(d, "算力饱和只说明竞争可能增加。\n是否限流，要看完成时间实际怎样变化。", color=PURPLE)


def roofline(axes):
    a, b, c, d = axes
    heading(a, "① 两道下界：计算与搬运都必须完成")
    label(a, 4, 26, "教学工作量 F=20 万亿次，累计搬运 Q=1 TB\n硬件算力 C=100 万亿次/秒，带宽 W=2 TB/秒", h=20, size=15)
    bar(a, 43, [("计算至少 0.2 秒", 0.2, BLUE)], start=4, width=92, total=0.6, h=12)
    bar(a, 61, [("搬运至少 0.5 秒", 0.5, GREEN)], start=4, width=92, total=0.6, h=12)
    note(a, "即使两者充分重叠，也至少等 0.5 秒。\n1 TB 是累计流量，不是同时需要 1 TB 显存。")

    heading(b, "② Roofline：性能不能超过更低的上界")
    plot = b.inset_axes([0.12, 0.39, 0.80, 0.40])
    intensity = np.linspace(0, 100, 201)
    plot.plot(intensity, np.minimum(100, 2 * intensity), lw=3, color=BLUE)
    plot.scatter([20, 50], [40, 100], color=[GREEN, ORANGE], s=55, zorder=3)
    plot.axvline(50, color=ORANGE, ls="--", lw=1)
    plot.set(xlim=(0, 100), ylim=(0, 120), xticks=[0, 20, 50, 100], yticks=[0, 40, 100])
    plot.tick_params(labelsize=16)
    plot.grid(alpha=0.15)
    plot.set_xlabel("I：运算次数 / 搬运字节", fontsize=16.5)
    plot.set_ylabel("万亿次 / 秒", fontsize=16.5)
    label(b, 4, 84, "拐点 I=C/W=50；例子 I=20，受带宽限制。", h=8, size=14)
    label(b, 4, 95, "性能 ≤ min(C, W×I) = min(100, 40) = 40", size=14)

    heading(c, "③ 合批怎样改变瓶颈？不要直接套请求数")
    box(c, 4, 24, 43, 29, BLUE, "少量 decode 请求\n读取同一份权重\n可分摊的计算少\n常偏带宽限制")
    box(c, 53, 24, 43, 29, GREEN, "更多请求合批\n一次读权重服务更多\n算术强度提高\n逐渐靠近计算上界")
    label(c, 4, 68, "KV 历史长度、注意力、精度和通信也会变化。\n经典横轴是运算/字节，不是并发请求数。", h=20, size=15)
    note(c, "把拐点映射成请求上限，需要实测服务曲线。", color=PURPLE)

    heading(d, "④ 用于调度：容量通过后，还要比较总耗时")
    box(d, 4, 23, 92, 17, BLUE, "先检查显存峰值与预算 → 确认安全候选")
    label(
        d,
        4,
        55,
        "再测组合：环境并发 × vLLM 请求档位\n观测环境完成、推理延迟与整批完成时间。\n共运行快于排队时才增加并发。",
        h=28,
        size=15,
    )
    note(d, "Roofline 提供瓶颈解释和建模先验。\n最终调度比较应包含排队、准备和竞争代价。", color=GREEN)


DRAWERS = {
    "memory-promises": memory,
    "wait-or-reload": placement,
    "borrower-return-delay": loan,
    "host-and-gpu-ledgers": ledgers,
    "state-version-match": version,
    "command-episode-residency": residency,
    "gpu-capacity-and-compute": compute,
    "roofline-capacity-decision": roofline,
}


def draw_figure(spec):
    """Use a fixed reading order and verify each label at ordinary text size."""
    LABELS.clear()
    fig, grid = plt.subplots(4, 1, figsize=(12, 23.5))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.035, top=0.925, hspace=0.12)
    fig.text(0.04, 0.980, spec["title"], fontsize=24, weight="bold", color=INK)
    fig.text(0.04, 0.958, spec["reading_question"], fontsize=17, color=MUTED)
    fig.text(0.04, 0.940, "阅读顺序：自上而下 ① → ② → ③ → ④", fontsize=16.5, color=MUTED)
    fig.text(
        0.04,
        0.013,
        "图内数字为教学假设。先核角色、计时与同量工作，再核实现范围。实测证据以对应正文为准。",
        fontsize=14.5,
        color=MUTED,
    )
    DRAWERS[spec["id"]](list(grid.flat))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, ax, bounds in LABELS:
        x, y, w, h = bounds
        (x1, y1), (x2, y2) = ax.transData.transform([(x, y), (x + w, y + h)])
        limit_w, limit_h = abs(x2 - x1), abs(y2 - y1)
        bbox = artist.get_window_extent(renderer)
        if bbox.width > limit_w * 1.04 or bbox.height > limit_h * 1.04:
            message = textwrap.shorten(artist.get_text().replace("\n", " / "), width=90)
            raise ValueError(f"Crowded label in {spec['id']!r}: {message!r}.")
    fig.canvas.draw()
    return fig
