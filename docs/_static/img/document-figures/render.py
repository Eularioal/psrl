"""Render reproducible document illustrations and review sheets using CPU only."""

import argparse
import json
import re
import runpy
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INK = "#203149"
MUTED = "#5b6c80"
PALETTE = ["#3478c5", "#159b8d", "#e79730", "#8764b8", "#d45b67", "#57934f"]
TINTS = ["#eaf2ff", "#e6f6f2", "#fff2db", "#f2edfb", "#fdecef", "#eef7e8"]
FONT_PATH = None
TEXT_BOXES = []


def wrap(value, units):
    """Wrap mixed CJK and Latin text without truncating the information."""
    result = []
    for paragraph in str(value).split("\n"):
        line = ""
        width = 0
        for char in re.findall(r"[A-Za-z0-9_.+/-]+|.", paragraph):
            step = sum(2 if unicodedata.east_asian_width(item) in "WF" else 1 for item in char)
            if width + step > units and line:
                result.append(line.rstrip())
                line = ""
                width = 0
            line += char
            width += step
        result.append(line.rstrip())
    return "\n".join(result)


def setup():
    """Use an installed CJK font and keep vector text editable."""
    global FONT_PATH
    candidates = [
        Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    FONT_PATH = next((path for path in candidates if path.is_file()), None)
    if FONT_PATH is None:
        fonts = {font.name: font.fname for font in fm.fontManager.ttflist}
        for name in ("Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "Droid Sans Fallback"):
            if name in fonts:
                FONT_PATH = Path(fonts[name])
                break
    if FONT_PATH is None:
        raise RuntimeError("A CJK font is required to render the figure labels.")
    fm.fontManager.addfont(str(FONT_PATH))
    family = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams.update(
        {
            "font.family": [family, "DejaVu Sans"],
            "font.size": 14,
            "text.color": INK,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
        }
    )


def rect(ax, x, y, w, h, fill="white", edge="#d4dfea", rounded=True, **kwargs):
    patch = (
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1", **kwargs)
        if rounded
        else Rectangle((x, y), w, h, **kwargs)
    )
    patch.set_facecolor(fill)
    patch.set_edgecolor(edge)
    patch.set_linewidth(1.3)
    ax.add_patch(patch)
    return patch


def text(ax, x, y, value, w=None, h=None, size=14, color=INK, bold=False, align="left", **kwargs):
    """Place a bounded label and retain its box for the layout check."""
    value = wrap(value, max(8, int(w * 1.22))) if w else str(value)
    artist = ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        weight="bold" if bold else "normal",
        ha=align,
        va="center",
        linespacing=1.45,
        **kwargs,
    )
    if w and h:
        left = x - w / 2 if align == "center" else x
        TEXT_BOXES.append((artist, ax, (left, y - h / 2, w, h)))
    return artist


def card(ax, x, y, w, h, label, detail="", color=0, size=15):
    rect(ax, x, y, w, h, TINTS[color], PALETTE[color])
    text(
        ax,
        x + w / 2,
        y + h * (0.36 if detail else 0.5),
        label,
        w=w - 3,
        h=h * (0.45 if detail else 0.8),
        size=size,
        color=PALETTE[color],
        bold=True,
        align="center",
    )
    if detail:
        text(ax, x + w / 2, y + h * 0.72, detail, w=w - 3, h=h * 0.37, size=13, align="center")


def arrow(ax, x1, y1, x2, y2, color=MUTED, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )


def disk(ax, x, y, w, h, label, color=0):
    rect(ax, x, y + 2, w, h - 4, TINTS[color], PALETTE[color], rounded=False)
    ax.add_patch(Ellipse((x + w / 2, y + 2), w, 5, facecolor=TINTS[color], edgecolor=PALETTE[color]))
    ax.add_patch(Ellipse((x + w / 2, y + h - 2), w, 5, facecolor=TINTS[color], edgecolor=PALETTE[color]))
    text(ax, x + w / 2, y + h / 2, label, w=w - 2, h=h - 7, size=14, align="center")


def cells(ax, x, y, cols, rows, colors, cw=3.7, ch=5.2, gap=0.65):
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            color = colors[index % len(colors)]
            fill = "#edf1f5" if color is None else PALETTE[color]
            rect(ax, x + col * (cw + gap), y + row * (ch + gap), cw, ch, fill, "white", rounded=False)


def gpu(ax, x, y, w=34, h=39, label="GPU", colors=None):
    rect(ax, x, y, w, h, "#24374b", "#24374b")
    text(ax, x + 2, y + 5, label, w=w - 4, h=7, size=16, color="white", bold=True)
    colors = colors if colors is not None else [0, 0, 1, 1, None, None]
    cw = (w - 8) / 6 - 0.5
    cells(ax, x + 4, y + 11, 6, 3, colors, cw=cw, ch=5, gap=0.5)
    text(ax, x + w / 2, y + h - 5, "计算网格示意", size=12, color="#dce9f4", align="center")


def strip(ax, x, y, w, h, parts, capacity=None):
    total = capacity or sum(value for _, value, _ in parts)
    cursor = x
    for label, value, color in parts:
        width = w * value / total
        rect(ax, cursor, y, width, h, TINTS[color], PALETTE[color], rounded=False)
        if width > 6:
            text(ax, cursor + width / 2, y + h / 2, label, w=width - 1, h=h - 1, size=13, align="center")
        cursor += width
    if cursor < x + w:
        rect(ax, cursor, y, x + w - cursor, h, "#f0f3f7", "#d3dce6", rounded=False)


def canvas(spec):
    TEXT_BOXES.clear()
    fig, ax = plt.subplots(figsize=(14, 7.6))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.99)
    ax.set(xlim=(0, 100), ylim=(100, 0))
    ax.axis("off")
    rect(ax, 0.2, 0.2, 99.6, 99.6, "#fbfcfe", "#e3eaf2")
    text(ax, 3, 6, spec["title"], w=94, h=10, size=21, bold=True)
    status = {
        "teaching": ("教学与机制示意", 0),
        "reported": ("历史数据重绘", 1),
        "navigation": ("阅读地图", 3),
    }[spec["status"]]
    rect(ax, 3, 14, 20, 5.5, TINTS[status[1]], TINTS[status[1]])
    text(ax, 13, 16.75, status[0], size=12, color=PALETTE[status[1]], align="center")
    text(ax, 97, 16.75, "颜色区分对象，位置表示关系", size=12, color=MUTED, align="right")
    text(ax, 3, 96, "条件、实现状态与来源见配套图注和正文。", size=11, color=MUTED)
    return fig, ax


def draw_inode(ax):
    for x, phase in [(4, "有名字，也有打开引用"), (53, "名字删除，引用仍打开")]:
        rect(ax, x, 25, 43, 55, "white")
        text(ax, x + 21.5, 30, phase, size=15, bold=True, align="center")
        disk(ax, x + 25, 43, 14, 24, "inode I\nhello\\n", 0)
        card(ax, x + 3, 60, 16, 11, "进程fd", color=1)
        arrow(ax, x + 19, 65, x + 25, 57, PALETTE[1])
    card(ax, 7, 40, 16, 8, "a.txt", color=0)
    card(ax, 7, 50, 16, 8, "b.txt", color=0)
    arrow(ax, 23, 44, 29, 49, PALETTE[0])
    arrow(ax, 23, 54, 29, 54, PALETTE[0])
    text(ax, 58, 46, "a、b均已unlink", w=19, h=12, color=PALETTE[4])
    text(ax, 6, 86, "硬链接共享I；复制会创建另一个文件对象J。", w=88, h=9, size=15)


def draw_mmap(ax):
    text(ax, 4, 29, "进程中的映射", size=16, bold=True)
    text(ax, 4, 63, "底层文件", size=16, bold=True)
    for i in range(4):
        x = 26 + i * 17
        card(ax, x, 25, 14, 17, "虚拟页" + str(i), color=0 if i < 2 else 4)
        if i < 2:
            disk(ax, x, 58, 14, 20, "文件页" + str(i), 1)
            arrow(ax, x + 7, 43, x + 7, 57, PALETTE[1])
        else:
            rect(ax, x, 58, 14, 20, TINTS[4], PALETTE[4], hatch="///")
            text(ax, x + 7, 68, "已截去", size=13, color=PALETTE[4], align="center")
            arrow(ax, x + 7, 43, x + 7, 57, PALETTE[4], dashed=True)
    text(ax, 28, 86, "前两页仍可读取；访问失去支撑的映射页可能触发SIGBUS。", w=65, h=11, size=14)


def draw_sqlite(ax):
    for x, label, detail, color in [
        (5, "主数据库", "已经合并的数据页", 0),
        (37, "WAL", "新提交的页版本", 2),
        (69, "SHM", "索引与协调信息", 3),
    ]:
        disk(ax, x, 30, 26, 32, label, color)
        text(ax, x + 13, 69, detail, w=26, h=10, align="center")
    arrow(ax, 49, 78, 18, 78, PALETTE[2])
    text(ax, 34, 85, "checkpoint合并", size=14, color=PALETTE[2], align="center")
    text(ax, 82, 85, "帮助定位可见版本", size=13, color=PALETTE[3], align="center")


def draw_shell(ax):
    rect(ax, 4, 25, 92, 18, TINTS[0], PALETTE[0])
    text(ax, 7, 30, "当前Shell：source在这里生效", size=16, bold=True)
    text(ax, 7, 38, "已导出环境：PATH、模型路径、运行参数", size=13)
    for x, label, state, color in [
        (7, "Python进程P1", "自己的sys.modules", 1),
        (55, "Python进程P2", "再次创建模块对象", 3),
    ]:
        arrow(ax, x + 17, 44, x + 17, 52, PALETTE[color])
        card(ax, x, 53, 38, 24, label, state, color)
    card(ax, 23, 83, 54, 10, "磁盘包文件：两个进程可读取同一份", color=2, size=13)


def draw_http(ax):
    labels = [
        (5, 26, "主机", "哪台机器", 0),
        (36, 26, "端口8080", "哪个服务", 1),
        (67, 26, "路径/demo", "哪项处理", 2),
    ]
    for x, y, label, detail, color in labels:
        card(ax, x, y, 28, 18, label, detail, color)
    rect(ax, 5, 55, 27, 29, TINTS[0], PALETTE[0])
    text(ax, 18.5, 69, "客户端\nPOST请求", size=16, bold=True, align="center")
    rect(ax, 68, 55, 27, 29, TINTS[1], PALETTE[1])
    text(ax, 81.5, 69, "服务进程\n监听socket", size=16, bold=True, align="center")
    arrow(ax, 34, 63, 66, 63, PALETTE[2])
    arrow(ax, 66, 77, 34, 77, PALETTE[1])
    text(ax, 50, 57, "方法、路径、请求体", size=12, align="center")
    text(ax, 50, 85, "状态码、响应体", size=12, align="center")


def timeline(ax, y, label, segments, total=14, left=25, width=67):
    text(ax, 3, y + 4, label, w=20, h=13, size=14)
    rect(ax, left, y, width, 9, "#f0f3f7", "#e1e7ef", rounded=False)
    for start, duration, name, color in segments:
        x = left + width * start / total
        w = width * duration / total
        rect(ax, x, y, w, 9, TINTS[color], PALETTE[color], rounded=False)
        text(ax, x + w / 2, y + 4.5, name, w=w - 0.6, h=8, size=12, align="center")


def time_axis(ax, total, y=23, unit="教学时间单位"):
    text(ax, 3, y, unit, size=11, color=MUTED)
    for tick in range(0, total + 1, 2):
        text(ax, 25 + 67 * tick / total, y, str(tick), size=10, color=MUTED, align="center")


def draw_cache(ax):
    card(ax, 5, 27, 37, 13, "主存池：3 GiB", color=0)
    strip(ax, 7, 49, 33, 14, [("A：2 GiB", 2, 0)], capacity=3)
    card(ax, 56, 27, 37, 13, "另一个准备状态", color=1)
    strip(ax, 58, 49, 33, 14, [("B：2 GiB", 2, 1)], capacity=3)
    text(ax, 49, 74, "A+B共4 GiB，无法同时放入3 GiB主存池。", w=86, h=12, size=16, align="center")
    text(ax, 49, 88, "保留谁，还要比较下一次使用机会、重建成本与版本有效性。", w=90, h=10, size=14, align="center")


def draw_parallel(ax):
    labels = ["数据并行 DP", "张量并行 TP", "流水线并行 PP"]
    for i, label in enumerate(labels):
        x = 4 + i * 32
        text(ax, x + 14, 29, label, size=16, bold=True, align="center")
        for j in range(2):
            y = 38 + j * 25
            rect(ax, x, y, 28, 21, "#24374b", "#24374b")
            text(ax, x + 2, y + 4, "GPU" + str(j), size=13, color="white")
            if i == 1:
                card(ax, x + 2, y + 8, 24, 10, "每层权重的" + ("左半" if j == 0 else "右半"), color=j, size=13)
            else:
                for k in range(4 if i == 0 else 2):
                    layer = k + 1 if i == 0 else j * 2 + k + 1
                    card(
                        ax,
                        x + 2 + ((k % 2) * 12.5 if i == 0 else 0),
                        y + 7 + (k // 2 if i == 0 else k) * 7.5,
                        11 if i == 0 else 24,
                        6.8,
                        "层" + str(layer),
                        color=layer - 1,
                        size=11,
                    )
        text(
            ax,
            x + 14,
            91,
            ["不同样本，同步梯度", "同一层，分张量", "不同层，传激活"][i],
            w=29,
            h=9,
            size=13,
            align="center",
        )


def draw_kv(ax):
    text(ax, 4, 28, "共享前缀块", size=16, bold=True)
    for i in range(3):
        card(ax, 27 + i * 15, 25, 12, 12, "P" + str(i), color=0)
    for y, label, color in [(52, "分支甲", 1), (77, "分支乙", 2)]:
        text(ax, 5, y + 6, label, size=16)
        for i in range(3):
            card(ax, 27 + i * 15, y, 12, 12, "引用P" + str(i), color=0, size=13)
        card(ax, 76, y, 18, 12, "自己的新块", color=color, size=13)
    text(ax, 53, 94, "共享只读前缀，修改与增长保留分支自己的状态。", size=13, align="center")


def draw_weights(ax):
    gpu(ax, 4, 32, 26, 38, "训练GPU", [0, 1, 2, 3])
    disk(ax, 38, 31, 26, 40, "CPU权重存储\n版本v11", 2)
    for x, label, color in [(73, "推理副本1", 1), (73, "推理副本2", 3)]:
        y = 27 if color == 1 else 61
        card(ax, x, y, 22, 25, label, "本地权重版本", color)
        arrow(ax, 65, 48 if color == 1 else 59, 72, y + 12, PALETTE[color])
    arrow(ax, 31, 50, 37, 50, PALETTE[0])
    text(ax, 49, 87, "版本与权限是控制信息；模型张量是数据。", size=15, align="center")


def draw_staleness(ax):
    text(ax, 4, 29, "当前训练版本：10    教学允许落后：2版", size=17, bold=True)
    for i, version in enumerate([7, 8, 9, 10]):
        x = 5 + i * 24
        color = 4 if version == 7 else 1
        card(ax, x, 43, 20, 31, "生成版本" + str(version), "版本差=" + str(10 - version), color)
        text(ax, x + 10, 83, "超出窗口" if version == 7 else "窗口内", size=14, color=PALETTE[color], align="center")


def draw_fields(ax):
    text(ax, 4, 28, "样本键 q1", size=18, bold=True)
    labels = [
        ("prompt", "输入消息", 0),
        ("response", "生成结果", 1),
        ("score", "评分反馈", 2),
        ("log-prob / advantage", "训练字段", 3),
    ]
    for i, (key, label, color) in enumerate(labels):
        y = 37 + i * 11
        card(ax, 5, y, 56, 9, key + "：" + label, color=color, size=14)
    for x, y, label, color in [(71, 34, "生成阶段", 1), (71, 54, "评分阶段", 2), (71, 74, "训练阶段", 3)]:
        card(ax, x, y, 24, 13, label, color=color)
        arrow(ax, 69, y + 6, 62, y + 6, PALETTE[color])
    text(ax, 51, 91, "阶段间传递样本引用，按需读写同一样本的字段。", size=14, align="center")


def draw_pipeline(ax):
    time_axis(ax, 12, y=22)
    for i, end in enumerate([3, 5, 6, 8]):
        timeline(ax, 26 + i * 12, "轨迹" + str(i + 1), [(0, end, "生成与工具交互", i % 4)], total=12)
    timeline(ax, 76, "策略训练", [(8, 2, "更新", 4), (10, 1, "同步", 2)], total=12)
    x = 25 + 67 * 8 / 12
    ax.plot([x, x], [23, 87], color=PALETTE[4], linestyle="--", linewidth=1.5)
    text(ax, 55, 91, "教学固定批次：4条轨迹准备好后，开始该次更新。", size=13, align="center")


def draw_advantages(ax):
    text(ax, 5, 28, "同题四次尝试", size=17, bold=True)
    for i, (reward, advantage) in enumerate(zip([1, 1, 0, 0], [1, 1, -1, -1], strict=True)):
        x = 5 + i * 23.5
        color = 1 if reward else 4
        card(
            ax,
            x,
            39,
            20,
            35,
            "尝试" + str(i + 1),
            "奖励=" + str(reward) + "\n优势=" + ("+1" if advantage > 0 else "-1"),
            color,
        )
    text(ax, 50, 84, "均值0.5，标准差0.5：奖励高于组内均值的尝试得到正方向。", w=90, h=12, size=15, align="center")


def draw_tokenloss(ax):
    for y, title, weights in [
        (27, "先按轨迹平均：每条各占一半", [0.5, 0.5]),
        (61, "按全部token平均：分别贡献2/8与6/8", [0.25, 0.75]),
    ]:
        text(ax, 4, y, title, size=15, bold=True)
        for j, length in enumerate([2, 6]):
            cells(ax, 6, y + 7 + j * 9, length, 1, [j], cw=6, ch=6, gap=1)
            text(ax, 61, y + 10 + j * 9, "该回答权重：" + str(weights[j]), size=14)


def draw_context(ax):
    for y, label, parts in [
        (37, "第一轮：8+6=14", [("输入8", 8, 0), ("回答6", 6, 1)]),
        (63, "下一轮：12+6=18", [("累积输入12", 12, 0), ("申请回答6", 6, 2)]),
    ]:
        text(ax, 4, y - 7, label, size=15, bold=True)
        strip(ax, 5, y, 90, 13, parts, capacity=18)
    x = 5 + 90 * 16 / 18
    ax.plot([x, x], [27, 80], color=PALETTE[4], linewidth=2, linestyle="--")
    text(ax, 50, 88, "教学上限16：工具输出并入历史后，留给下一轮回答的空间变小。", w=90, h=10, size=14, align="center")


def draw_nixl(ax):
    for x, label, color in [(5, "发送侧已登记缓冲区", 0), (60, "接收侧已登记缓冲区", 1)]:
        rect(ax, x, 30, 35, 40, TINTS[color], PALETTE[color])
        text(ax, x + 17.5, 36, label, w=32, h=10, size=15, bold=True, align="center")
        cells(ax, x + 4, 48, 5, 2, [color], cw=4.7, ch=6)
    arrow(ax, 42, 51, 58, 51, PALETTE[2])
    text(ax, 50, 42, "传输句柄", size=13, align="center")
    card(ax, 28, 80, 44, 12, "完成确认后，按协议使用新内容", color=2)


def draw_oom(ax):
    for y, label, temporary in [(38, "短序列：60+25=85", 25), (66, "长序列：60+50=110", 50)]:
        text(ax, 4, y - 8, label, size=15, bold=True)
        strip(
            ax, 5, y, 90, 13, [("参数及常驻状态60", 60, 0), ("中间状态" + str(temporary), temporary, 2)], capacity=110
        )
    x = 5 + 90 * 100 / 110
    ax.plot([x, x], [27, 81], color=PALETTE[4], linewidth=2, linestyle="--")
    text(
        ax,
        50,
        89,
        "相对容量上限100：需要观察运行峰值，而非只看启动时的常驻占用。",
        w=92,
        h=11,
        size=14,
        align="center",
    )


def draw_health(ax):
    items = [
        ("样本有效性", "response_mask\n有效token与截断", 0),
        ("组内反馈", "奖励差异\n零方差组比例", 1),
        ("策略分布", "entropy\n生成长度与裁剪", 2),
        ("更新幅度", "loss / grad_norm\n优化器调用", 3),
    ]
    for i, (label, detail, color) in enumerate(items):
        x, y = 5 + (i % 2) * 47, 29 + (i // 2) * 31
        card(ax, x, y, 42, 26, label, detail, color)
    text(ax, 50, 91, "分别核对数据、反馈和更新，避免只凭进程仍在运行作判断。", size=14, align="center")


def draw_preflight(ax):
    rect(ax, 4, 25, 92, 62, "white", "#afc0d3")
    text(ax, 7, 31, "真正执行任务的节点", size=18, bold=True)
    card(ax, 8, 41, 25, 28, "宿主与依赖", "导入来源\nIP / Ray身份", 0)
    card(ax, 38, 41, 25, 28, "Docker与容器", "daemon探活\nCUDA内核", 1)
    card(ax, 68, 41, 23, 28, "资产与资源", "数据 / 模型\nGPU / 清理", 2)
    text(ax, 50, 79, "driver终端检查  ≠  worker与容器检查", size=16, align="center")


def draw_isolation(ax):
    card(ax, 4, 28, 43, 47, "候选容器", "候选程序 + 当前输入\n执行后返回输出\n不接收隐藏答案", 2)
    card(ax, 54, 28, 42, 47, "可信判题器", "隐藏测试与答案\n比较输出\n计算最终分数", 1)
    ax.plot([50, 50], [25, 82], linewidth=2, linestyle="--", color=PALETTE[4])
    text(ax, 50, 89, "测试输入与程序输出可跨边界，隐藏答案留在可信侧。", size=15, align="center")


def draw_two_models(ax):
    rect(ax, 4, 27, 43, 57, TINTS[0], PALETTE[0])
    rect(ax, 55, 27, 41, 57, TINTS[1], PALETTE[1])
    text(ax, 25.5, 34, "Agent策略：参数θ", size=17, bold=True, align="center")
    text(ax, 75.5, 34, "环境任务模型：参数φ", size=16, bold=True, align="center")
    cells(ax, 13, 45, 4, 3, [0], cw=5, ch=5)
    cells(ax, 63, 45, 4, 3, [1], cw=5, ch=5)
    text(ax, 25.5, 75, "生成代码、选择工具\n由策略轨迹更新", size=14, align="center")
    text(ax, 75.5, 75, "训练 / 推理 / 评分\n提交任务产物", size=14, align="center")
    text(ax, 50, 91, "环境产物的评分成为反馈，再用于训练外层Agent策略。", size=14, align="center")


def draw_benchmark(ax):
    for i, (label, detail, color) in enumerate(
        [
            ("固定资产", "数据 / 依赖 / 基础模型", 0),
            ("当前尝试", "候选代码 / 候选权重", 2),
            ("可信评分", "隐藏数据 / 评分规则", 1),
        ]
    ):
        x = 4 + i * 32
        disk(ax, x + 2, 31, 25, 26, label, color)
        text(ax, x + 14.5, 66, detail, w=28, h=13, size=14, align="center")
    strip(ax, 6, 80, 87, 10, [("跨请求准备", 35, 0), ("随尝试变化", 35, 2), ("评分器持有", 30, 1)])


def draw_evidence(ax):
    names = ["CPU机制", "容器CUDA", "任务评分", "优化器更新", "性能对照"]
    points = ["协议与故障处理", "设备真的能计算", "任务反馈可产生", "策略更新链路", "相同工作量与边界"]
    for i, (name, point) in enumerate(zip(names, points, strict=True)):
        x = 4 + i * 19
        card(ax, x, 30, 17, 48, name, point, i % 5, size=15)
        cells(ax, x + 3, 84, 3, 1, [i % 5], cw=3, ch=4)
    text(ax, 50, 93, "每一列回答自己的问题，验证范围由本次目标决定。", size=14, align="center")


def draw_measurement(ax):
    time_axis(ax, 6, y=25)
    timeline(ax, 33, "同一候选单独运行", [(0, 2, "计算2", 0)], total=6)
    timeline(ax, 58, "同一候选遇到竞争", [(0, 4, "计时变成4", 4)], total=6)
    timeline(ax, 73, "另一项GPU工作", [(0, 4, "共享设备计算", 2)], total=6)
    text(ax, 50, 89, "教学时间单位：候选代码没变，测量条件改变了观测结果。", size=14, align="center")


def draw_cleanup(ax):
    for x, label, color in [(5, "逻辑状态", 0), (54, "物理资源", 2)]:
        rect(ax, x, 29, 41, 49, TINTS[color], PALETTE[color])
        text(ax, x + 20.5, 35, label, size=17, bold=True, align="center")
    for i, label in enumerate(["租约已结束", "队列已撤回", "预算已归还"]):
        text(ax, 9, 47 + i * 11, "✓ " + label, size=15, color=PALETTE[0])
    for i, label in enumerate(["所属进程已退出？", "所属容器已清理？", "GPU显存已释放？"]):
        text(ax, 58, 47 + i * 11, label, size=15, color=PALETTE[2])
    text(ax, 50, 89, "交给下一组任务前，需要核对两侧都完成。", size=15, align="center")


def draw_identity(ax):
    card(ax, 5, 29, 36, 22, "容器甲", "内部PID 12", 0)
    card(ax, 5, 64, 36, 22, "容器乙", "内部PID 12", 2)
    card(ax, 61, 29, 34, 22, "宿主进程", "PID 418 + 出生身份", 0)
    card(ax, 61, 64, 34, 22, "宿主进程", "PID 907 + 出生身份", 2)
    arrow(ax, 43, 40, 59, 40, PALETTE[0])
    arrow(ax, 43, 75, 59, 75, PALETTE[2])
    text(ax, 50, 93, "容器内编号相同，不说明是同一个宿主进程。", size=14, align="center")


def draw_writeback(ax):
    card(ax, 5, 30, 26, 25, "Docker线程", "等待清理 / 写回", 4)
    disk(ax, 39, 30, 25, 25, "文件系统对象", 2)
    card(ax, 73, 30, 22, 25, "底层I/O", "写回与锁状态", 0)
    arrow(ax, 32, 43, 38, 43, PALETTE[4])
    arrow(ax, 65, 43, 72, 43, PALETTE[2])
    strip(ax, 6, 74, 88, 13, [("容量已用", 55, 0), ("容量空余", 45, 1)])
    text(ax, 50, 65, "等待链与容量统计需要分开核对。", size=15, align="center")


def draw_backend(ax):
    for x, provider, color in [(4, "Compose创建的容器", 0), (57, "EnvWorker创建的容器", 1)]:
        rect(ax, x, 26, 39, 25, TINTS[color], PALETTE[color])
        text(ax, x + 19.5, 31, provider, size=15, bold=True, align="center")
        card(ax, x + 4, 38, 31, 9, "命令进程 / 文件 / 取消", color=color, size=13)
    arrow(ax, 25, 52, 33, 61, PALETTE[0])
    arrow(ax, 75, 52, 67, 61, PALETTE[1])
    rect(ax, 10, 62, 80, 25, "#24374b", "#24374b")
    text(ax, 50, 68, "公共执行层：记账、执行、归还与回收", size=15, color="white", align="center")
    strip(ax, 16, 75, 68, 8, [("甲的GPU预算", 4, 0), ("乙的GPU预算", 3, 1), ("剩余", 3, 2)])
    text(ax, 50, 92, "容器创建方式可以不同，执行与资源协议共同维护。", size=13, align="center")


def draw_queue(ax):
    rect(ax, 4, 26, 92, 60, "white")
    card(ax, 8, 36, 27, 35, "同机容器", "CPU与主存保留\n下一条GPU命令", 0)
    gpu(ax, 48, 34, 21, 37, "GPU0", [2, 2, 2, 2])
    gpu(ax, 74, 34, 18, 37, "GPU1", [None])
    text(ax, 58.5, 79, "有队列", size=14, color=PALETTE[2], align="center")
    text(ax, 83, 79, "候选空余", size=14, color=PALETTE[1], align="center")
    arrow(ax, 37, 49, 47, 49, PALETTE[2], dashed=True)
    arrow(ax, 37, 74, 83, 74, PALETTE[1])
    arrow(ax, 83, 74, 83, 72, PALETTE[1])
    text(ax, 50, 92, "检查候选范围、容量和准备代价，再决定等还是换。", size=14, align="center")


def draw_config(ax):
    columns = [
        ("默认层", ["cpu: 2", "memory: 8", "mode: episode"], 0),
        ("本次覆盖", ["cpu: 4", "mode: command"], 2),
        ("有效配置", ["cpu: 4", "memory: 8", "mode: command"], 1),
    ]
    for i, (label, lines, color) in enumerate(columns):
        x = 4 + i * 32
        card(ax, x, 29, 28, 52, label, "\n".join(lines), color)
    text(ax, 50, 91, "教学键值：未覆盖的值保留，覆盖过的值使用本次设置。", size=14, align="center")


def draw_rules(ax):
    items = [
        ("Rule：要求", "AGENTS.md\n交付与工作约束", 0),
        ("Memory：背景", "历史决策与进度\n日期、条件、来源", 2),
        ("Skill：方法", "SKILL.md\n步骤与配套工具", 1),
    ]
    for i, (label, detail, color) in enumerate(items):
        x = 4 + i * 32
        rect(ax, x, 32, 28, 45, TINTS[color], PALETTE[color])
        rect(ax, x + 9, 36, 10, 2, PALETTE[color], PALETTE[color])
        text(ax, x + 14, 47, label, size=16, bold=True, align="center")
        text(ax, x + 14, 63, detail, w=25, h=20, size=14, align="center")


def draw_sessions(ax):
    for i in range(4):
        card(ax, 6 + i * 14, 34, 11, 14, str(i + 1), color=0)
    text(ax, 31, 26, "共同历史前缀", size=16, bold=True, align="center")
    for y, label, color in [(28, "分支A", 1), (67, "分支B", 2)]:
        card(ax, 72, y, 22, 22, label, "新消息5、6…", color)
        arrow(ax, 61, 42, 71, y + 11, PALETTE[color])
    text(ax, 32, 73, "前缀相同\n分支之后各自继续", size=16, align="center")


def draw_terminal(ax):
    rect(ax, 5, 27, 90, 59, "#223247", "#223247")
    text(ax, 9, 33, "外层终端窗口", size=16, color="white", bold=True)
    rect(ax, 10, 41, 79, 39, "#2d455e", "#97b9d8")
    text(ax, 14, 47, "tmux窗格与历史", size=15, color="#cde8ff")
    rect(ax, 18, 55, 62, 19, "#345f71", "#8bd6d0")
    text(ax, 22, 64, "应用内部列表与当前焦点", size=15, color="white")
    for x, y, height in [(91, 43, 33), (84, 55, 18)]:
        rect(ax, x, y, 2, height, "#7694ae", "#7694ae", rounded=False)
        rect(ax, x, y + 4, 2, height / 3, "#e6f6f2", "#e6f6f2", rounded=False)
    text(ax, 50, 93, "绘制的教学界面：滚轮到达哪一层，需要现场核对。", size=14, align="center")


def draw_restore(ax):
    for x, label, lines, color in [
        (5, "当前配置", ["model = current", "theme = dark", "setting = current"], 0),
        (54, "备份差异", ["model = old（跳过）", "theme = light", "setting = current"], 2),
    ]:
        rect(ax, x, 30, 41, 48, TINTS[color], PALETTE[color])
        text(ax, x + 4, 36, label, size=17, bold=True)
        for i, line in enumerate(lines):
            text(ax, x + 4, 48 + i * 10, ("✓ " if color == 2 and i == 1 else "  ") + line, size=14)
    text(ax, 50, 89, "先确认文件来源和差异，再选择要恢复的部分。", size=15, align="center")


def draw_assets(ax):
    card(ax, 5, 27, 38, 12, "PNG：直接显示", color=0)
    cells(ax, 10, 46, 6, 5, [0, 0, None, 1, 1, None], cw=4.5, ch=4.7)
    card(ax, 57, 27, 38, 12, "SVG：矢量放大", color=3)
    t = np.linspace(0, 1, 100)
    ax.plot(62 + t * 28, 65 - 12 * np.sin(t * np.pi), color=PALETTE[3], linewidth=4)
    for x, y in [(62, 65), (76, 53), (90, 65)]:
        ax.add_patch(Circle((x, y), 1, facecolor="white", edgecolor=PALETTE[3], linewidth=2))
    text(ax, 50, 87, "源脚本 + 绘图输入 + 正文图注 = 可继续维护的解释图", w=91, h=11, size=16, align="center")


def draw_browser(ax):
    rect(ax, 6, 27, 88, 59, "white", "#a6bad1")
    rect(ax, 7, 28, 86, 10, TINTS[0], TINTS[0], rounded=False)
    text(ax, 11, 33, "标题 / 导航", size=15, bold=True)
    rect(ax, 9, 42, 21, 38, TINTS[3], TINTS[3])
    text(ax, 19.5, 60, "侧栏\n阅读目录", size=14, align="center")
    rect(ax, 35, 42, 54, 25, TINTS[1], TINTS[1])
    text(ax, 62, 54.5, "正文、图片和表格", size=17, align="center")
    card(ax, 60, 71, 26, 9, "交互控件", color=2, size=13)
    text(ax, 50, 92, "模板管结构，CSS管样式，JavaScript管交互。", size=14, align="center")


def draw_publishing(ax):
    rect(ax, 5, 29, 40, 52, "#24374b", "#24374b")
    for i, line in enumerate(["# 文档标题", "## 核心机制", "![解释图](图.png)", "[专题入口](正文.md)"]):
        text(ax, 9, 39 + i * 11, line, w=32, h=8, size=14, color=["#9bc5ff", "#9ae3d0", "#f8d596", "#cfb9ee"][i])
    rect(ax, 55, 29, 40, 52, "white", "#a6bad1")
    text(ax, 59, 38, "文档标题", size=20, bold=True)
    text(ax, 59, 49, "核心机制", size=16, color=PALETTE[0])
    cells(ax, 59, 58, 5, 2, [0, 1, 2], cw=5, ch=5)
    text(ax, 59, 75, "专题入口", size=14, color=PALETTE[0])
    text(ax, 50, 92, "移动文件后，本地链接、图片路径与站点目录一起核对。", size=14, align="center")


def draw_dataset(ax):
    rect(ax, 5, 29, 90, 53, "white", "#b6c7d9")
    fields = [
        ("prompt", "消息列表\n[{role, content}, ...]", 0),
        ("reward_model", "奖励或答案信息\n结构化字段", 1),
        ("extra_info", "任务、环境与资产\n结构化参数", 2),
    ]
    for i, (label, detail, color) in enumerate(fields):
        card(ax, 9 + i * 29, 38, 25, 33, label, detail, color)
    text(ax, 50, 91, "结构化列表与结构体，需要匹配loader的实际输入类型。", size=14, align="center")


def draw_software(ax):
    rect(ax, 5, 27, 41, 55, TINTS[0], PALETTE[0])
    text(ax, 25.5, 33, "Agent工作容器", size=16, bold=True, align="center")
    card(ax, 9, 43, 32, 13, "代码：旧版本 → 修改版", color=0)
    card(ax, 9, 63, 32, 12, "检查代码、运行本地测试", color=2, size=13)
    rect(ax, 57, 27, 38, 55, TINTS[1], PALETTE[1])
    text(ax, 76, 33, "新的评分容器", size=16, bold=True, align="center")
    card(ax, 61, 43, 30, 13, "基准代码 + 提交补丁", color=1, size=13)
    card(ax, 61, 63, 30, 12, "重新运行要求的测试", color=1, size=13)
    arrow(ax, 47, 50, 56, 50, PALETTE[2])
    text(ax, 51, 86, "评分重建提交结果，避免沿用Agent工作容器的临时状态。", w=90, h=10, size=14, align="center")


def draw_reward_model(ax):
    card(ax, 5, 30, 28, 36, "策略模型", "根据题目\n生成回答", 0)
    card(ax, 38, 30, 25, 36, "反馈模型", "评分标准\n或教师目标", 3)
    card(ax, 68, 30, 27, 36, "学习信号", "奖励分数\n或token目标", 1)
    arrow(ax, 34, 47, 37, 47, PALETTE[0])
    arrow(ax, 64, 47, 67, 47, PALETTE[3])
    arrow(ax, 82, 68, 82, 79, PALETTE[1])
    arrow(ax, 82, 79, 19, 79, PALETTE[1])
    arrow(ax, 19, 79, 19, 68, PALETTE[1])
    text(ax, 50, 91, "计划方案：反馈用于策略学习，反馈模型是否更新另行决定。", w=92, h=9, size=14, align="center")


def draw_static_gpu(ax):
    card(ax, 5, 28, 33, 22, "容器甲", "静态映射GPU0", 0)
    card(ax, 5, 65, 33, 22, "容器乙", "静态映射GPU0", 2)
    gpu(ax, 58, 34, 35, 44, "同一张物理GPU0", [0, 2, 0, 2, 4, 4])
    arrow(ax, 40, 39, 57, 46, PALETTE[0])
    arrow(ax, 40, 76, 57, 66, PALETTE[2])
    text(ax, 50, 93, "进程内数量上限，与跨进程设备互斥是不同约束。", size=14, align="center")


def draw_gates(ax):
    time_axis(ax, 10, y=25)
    timeline(ax, 32, "Gate A：设备", [(0, 4, "甲持卡", 0), (4, 4, "乙持卡", 1)], total=10)
    timeline(ax, 59, "Gate B：回复", [(0, 3, "等待条件", 2), (3, 1, "回", 3)], total=10)
    text(ax, 27, 79, "设备是否允许执行，与回复何时交给调用方，属于不同控制点。", w=66, h=17, size=15)


def draw_navigation(ax, spec):
    for i, link in enumerate(spec["links"]):
        x, y = 4 + (i % 3) * 32, 27 + (i // 3) * 31
        color = i % len(PALETTE)
        rect(ax, x + 1, y + 1.4, 28, 26, "#e4ebf3", "#e4ebf3")
        rect(ax, x, y, 28, 26, "white", PALETTE[color])
        rect(ax, x, y, 3, 26, PALETTE[color], PALETTE[color], rounded=False)
        text(ax, x + 6, y + 5, f"{i + 1:02d}", size=14, color=PALETTE[color], bold=True)
        text(ax, x + 16, y + 15.5, link["label"], w=21, h=17, size=14, align="center")
    label = "其余主题入口见下方目录。" if spec["extra_links"] else "完整入口与适用条件见正文导航。"
    text(ax, 50, 92, label, size=14, color=MUTED, align="center")


DRAWERS = {
    "software": draw_software,
    "reward_model": draw_reward_model,
    "inode": draw_inode,
    "mmap": draw_mmap,
    "sqlite": draw_sqlite,
    "shell": draw_shell,
    "http": draw_http,
    "cache": draw_cache,
    "parallel": draw_parallel,
    "kv": draw_kv,
    "weights": draw_weights,
    "staleness": draw_staleness,
    "fields": draw_fields,
    "pipeline": draw_pipeline,
    "advantages": draw_advantages,
    "tokenloss": draw_tokenloss,
    "context": draw_context,
    "nixl": draw_nixl,
    "oom": draw_oom,
    "health": draw_health,
    "preflight": draw_preflight,
    "isolation": draw_isolation,
    "two_models": draw_two_models,
    "benchmark": draw_benchmark,
    "evidence": draw_evidence,
    "measurement": draw_measurement,
    "cleanup": draw_cleanup,
    "identity": draw_identity,
    "writeback": draw_writeback,
    "backend": draw_backend,
    "queue": draw_queue,
    "config": draw_config,
    "rules": draw_rules,
    "sessions": draw_sessions,
    "terminal": draw_terminal,
    "restore": draw_restore,
    "assets": draw_assets,
    "browser": draw_browser,
    "publishing": draw_publishing,
    "dataset": draw_dataset,
    "static_gpu": draw_static_gpu,
    "gates": draw_gates,
}


def draw_bars(fig, spec):
    panels = spec["panels"]
    count = len(panels)
    for i, panel in enumerate(panels):
        ax = fig.add_axes((0.15 + i * 0.44 if count == 2 else 0.22, 0.18, 0.31 if count == 2 else 0.63, 0.53))
        values = panel["values"]
        positions = np.arange(len(values))
        colors = panel.get("colors", [j % 3 for j in range(len(values))])
        ax.barh(positions, values, color=[PALETTE[j] for j in colors], height=0.5)
        ax.set_yticks(positions, [wrap(label, 18) for label in panel["labels"]], fontsize=13)
        ax.invert_yaxis()
        limit = max(values) * 1.25
        ax.set_xlim(0, limit)
        ax.set_xlabel("秒", fontsize=13)
        ax.set_title(wrap(panel["title"], 36), fontsize=16, pad=18)
        for y, value in zip(positions, values, strict=True):
            ax.text(
                value + limit * 0.025,
                y,
                f"{value:.3f}" if max(values) < 30 else f"{value:.2f}",
                va="center",
                fontsize=13,
            )
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)


def draw_ppo(fig):
    ratio = np.linspace(0.4, 1.7, 300)
    for i, advantage in enumerate([1, -1]):
        ax = fig.add_axes((0.09 + i * 0.48, 0.18, 0.36, 0.52))
        raw = ratio * advantage
        clipped = np.minimum(raw, np.clip(ratio, 0.8, 1.2) * advantage)
        ax.plot(ratio, raw, color=PALETTE[2], linestyle="--", label="未裁剪分支")
        ax.plot(ratio, clipped, color=PALETTE[i], linewidth=3, label="最终裁剪目标")
        ax.axvspan(0.8, 1.2, color=TINTS[1], alpha=0.8)
        ax.set_title("优势A=" + ("+1" if advantage > 0 else "-1"), fontsize=17, pad=12)
        ax.set_xlabel("概率比", fontsize=14)
        ax.set_ylabel("待最大化目标", fontsize=13)
        ax.grid(alpha=0.15)
        ax.legend(fontsize=11, loc="upper left" if advantage > 0 else "lower left")
        ax.spines[["top", "right"]].set_visible(False)


def fit_text(fig):
    """Reject label overflow without shrinking text below its chosen size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    for artist, ax, (x, y, w, h) in TEXT_BOXES:
        corners = ax.transData.transform([(x, y), (x + w, y + h)])
        width, height = abs(corners[1][0] - corners[0][0]), abs(corners[1][1] - corners[0][1])
        bbox = artist.get_window_extent(renderer)
        if bbox.width > width * 1.05 or bbox.height > height * 1.05:
            failures.append(artist.get_text())
    if failures:
        raise RuntimeError("Labels exceed their allocated boxes: " + repr(failures))


def review_sheets(specs):
    """Keep two figures per sheet at normal document width for readable review."""
    result = []
    font = ImageFont.truetype(str(FONT_PATH), 24)
    for start in range(0, len(specs), 2):
        group = specs[start : start + 2]
        figures = []
        for spec in group:
            im = Image.open(ROOT / (spec["id"] + ".png")).convert("RGB")
            im = im.resize((900, round(im.height * 900 / im.width)), Image.Resampling.LANCZOS)
            figures.append(im)
        sheet = Image.new("RGB", (920, sum(im.height + 48 for im in figures)), "#e5ebf2")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for spec, im in zip(group, figures, strict=True):
            sheet.paste(im, ((920 - im.width) // 2, y + 40))
            draw.text((15, y + 8), spec["id"], font=font, fill=INK)
            y += im.height + 48
        name = f"review-{start // 2 + 1:02d}.png"
        sheet.save(ROOT / name, optimize=True)
        result.append({"file": name, "figures": [spec["id"] for spec in group]})
    (ROOT / "review-sheets.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Render selected figure identifiers.")
    args = parser.parse_args()
    setup()
    catalog = json.loads((ROOT / "catalog.json").read_text())
    specs = catalog["figures"]
    selected = [spec for spec in specs if not args.only or spec["id"] in args.only]
    detailed = runpy.run_path(str(ROOT / "detailed.py"))["draw_figure"]
    composed = runpy.run_path(str(ROOT / "compose.py"))["draw"]
    chapter = runpy.run_path(str(ROOT / "chapters.py"))["render"]
    for spec in selected:
        if spec["kind"] == "chapter":
            chapter(spec["id"])
            print(spec["id"], flush=True)
            continue
        if spec["kind"] == "explanation":
            fig = detailed(spec)
            fig.savefig(ROOT / (spec["id"] + ".png"), dpi=120)
            fig.savefig(ROOT / (spec["id"] + ".svg"))
            plt.close(fig)
            print(spec["id"], flush=True)
            continue
        fig = composed(spec, globals())
        fig.savefig(ROOT / (spec["id"] + ".png"), dpi=120)
        fig.savefig(ROOT / (spec["id"] + ".svg"))
        plt.close(fig)
        print(spec["id"], flush=True)
    screen = runpy.run_path(str(ROOT / "screen.py"))
    screen_records = []
    for spec in selected:
        if spec.get("screen_preview"):
            screen_records.append(screen["render"](spec))
    screen["persist"](catalog, screen_records)
    if not args.only:
        review_sheets(specs)


if __name__ == "__main__":
    main()
