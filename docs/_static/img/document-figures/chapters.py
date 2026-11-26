"""Draw compact chapter figures with objects, aligned timelines and worked examples."""

import argparse
import html
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INK = "#203149"
MUTED = "#52677c"
BLUE, GREEN, ORANGE, PURPLE, RED = "#276dc5", "#008c7c", "#c8790c", "#7950b4", "#c64760"
TINT = {BLUE: "#eaf2ff", GREEN: "#e4f5f0", ORANGE: "#fff1d9", PURPLE: "#f1ebfa", RED: "#fdebf0"}
FONT = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")


class Scene:
    """Keep text readable and reject overflow rather than shrinking labels."""

    def __init__(self, title, subtitle, height=650):
        self.height = height
        self.nodes = []
        self.raster_labels = []
        self.text(32, 38, title, 26, width=836)
        font = ImageFont.truetype(str(FONT), 17)
        lines, line = [], ""
        for character in subtitle:
            if font.getlength(line + character) > 836:
                lines.append(line)
                line = ""
            line += character
        lines.append(line)
        if len(lines) > 2:
            raise ValueError("The subtitle needs a shorter explanation.")
        self.text(32, 74, "\n".join(lines), 17, MUTED, width=836)
        self.line(32, 95 if len(lines) == 1 else 111, 868, 95 if len(lines) == 1 else 111, "#dbe4ee")

    def text(self, x, y, value, size=18, color=INK, anchor="start", width=None):
        font = ImageFont.truetype(str(FONT), size)
        lines = str(value).split("\n")
        for index, line in enumerate(lines):
            if width is not None and font.getlength(line) > width:
                raise ValueError(f"Label exceeds its width: {line!r}.")
            length = font.getlength(line)
            left = x - (length / 2 if anchor == "middle" else length if anchor == "end" else 0)
            if left < 16 or left + length > 884:
                raise ValueError(f"Label exceeds the figure width: {line!r}.")
            self.nodes.append(
                f'<text x="{x}" y="{y + index * size * 1.5}" text-anchor="{anchor}" '
                f'style="font-size:{size}px" fill="{color}">{html.escape(line)}</text>'
            )
            self.raster_labels.append((x, y + index * size * 1.5, line, size, color, anchor))

    def box(self, x, y, w, h, color=BLUE, fill=None, radius=10, dash=False):
        dashed = ' stroke-dasharray="7 5"' if dash else ""
        self.nodes.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill or TINT.get(color, "white")}" stroke="{color}" stroke-width="1.5"{dashed}/>'
        )

    def line(self, x1, y1, x2, y2, color=MUTED, width=1.5, dash=False, arrow=False):
        extra = ' stroke-dasharray="6 5"' if dash else ""
        extra += ' marker-end="url(#arrow)"' if arrow else ""
        self.nodes.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{extra}/>'
        )

    def circle(self, x, y, r, color=BLUE):
        self.nodes.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def path(self, points, color=BLUE, fill="none", width=3):
        coordinates = " ".join(f"{x},{y}" for x, y in points)
        self.nodes.append(f'<polyline points="{coordinates}" fill="{fill}" stroke="{color}" stroke-width="{width}"/>')

    def footer(self, value, color=GREEN):
        self.box(32, self.height - 80, 836, 60, color)
        self.text(48, self.height - 55, value, 17, color, width=804)

    def timeline(self, top, limit, rows, ticks, unit="秒", left=155, right=840, step=70):
        scale = (right - left) / limit
        self.text(32, top - 20, f"同一时间轴（{unit}）", 16, MUTED)
        for tick in ticks:
            x = left + tick * scale
            self.text(x, top + 8, str(tick), 16, MUTED, "middle")
            self.line(x, top + 18, x, top + 43 + len(rows) * step, "#e2e8f0")
        for index, (name, segments) in enumerate(rows):
            y = top + 35 + index * step
            self.text(32, y + 25, name, 18)
            self.line(left, y + 38, right, y + 38, "#e2e8f0")
            for start, end, label, color in segments:
                x, w = left + start * scale, (end - start) * scale
                self.box(x, y, w, 38, color, radius=3)
                if label:
                    self.text(x + w / 2, y + 25, label, 17, color, "middle", width=w - 8)

    def memory(self, x, y, width, values, labels, colors, total, height=56):
        cursor = x
        for amount, label, color in zip(values, labels, colors, strict=True):
            w = width * amount / total
            self.box(cursor, y, w, height, color, radius=0)
            self.text(cursor + w / 2, y + height / 2 + 6, label, 17, color, "middle", width=w - 6)
            cursor += w

    def save(self, identity, output_dir=ROOT):
        import cairosvg

        start = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="{self.height * 2}" '
            f'viewBox="0 0 900 {self.height}"><defs><marker id="arrow" viewBox="0 0 10 10" '
            'refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#52677c"/></marker></defs>'
            f'<rect width="900" height="{self.height}" fill="white"/>'
            '<g font-family="Droid Sans Fallback">'
        )
        svg = start + "".join(self.nodes) + "</g></svg>"
        (output_dir / f"{identity}.svg").write_text(svg)
        shapes = start + "".join(node for node in self.nodes if not node.startswith("<text")) + "</g></svg>"
        image = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=shapes.encode()))).convert("RGB")
        draw = ImageDraw.Draw(image)
        anchors = {"start": "ls", "middle": "ms", "end": "rs"}
        for x, y, value, size, color, anchor in self.raster_labels:
            font = ImageFont.truetype(str(FONT), size * 2)
            draw.text((x * 2, y * 2), value, font=font, fill=color, anchor=anchors[anchor])
        image.save(output_dir / f"{identity}.png", optimize=True)


def critical_path():
    s = Scene(
        "先让谁用 GPU？要看它能解锁什么工作",
        "教学假设：M 用 GPU 2 秒，随后解锁 CPU 工作 A（10 秒）；B 独立用 GPU 10 秒。",
        740,
    )
    s.timeline(
        130,
        22,
        [
            ("模型优先", [(0, 2, "M", BLUE), (2, 12, "B：10 秒", GREEN)]),
            ("CPU 上的 A", [(2, 12, "A：10 秒，与 B 重叠", PURPLE)]),
            ("环境优先", [(0, 10, "B：10 秒", GREEN), (10, 12, "M", BLUE)]),
            ("CPU 上的 A", [(12, 22, "A：10 秒", PURPLE)]),
        ],
        [0, 2, 12, 22],
    )
    s.box(155, 505, 305, 104, GREEN)
    s.text(175, 538, "模型优先：最后完成于 12 秒", 20, GREEN)
    s.text(175, 574, "M 先结束，A 与 B 才能重叠。", 17)
    s.box(480, 505, 360, 104, ORANGE)
    s.text(500, 538, "环境优先：最后完成于 22 秒", 20, ORANGE)
    s.text(500, 574, "B 虽早结束，A 却晚启动 10 秒。", 17)
    s.footer("关键路径：决定最后结束时刻的依赖链。卡一直忙，也可能让整批更慢。")
    return s


def memory_headroom():
    s = Scene(
        "看见 16 GiB 空闲，为什么只能再放 4 GiB？",
        "教学预算：80 GiB 卡。已实占 64，另有尚未分配的承诺，不能把空闲全借出去。",
        650,
    )
    s.text(32, 140, "① 物理占用：模型 56 + 环境 8 = 64 GiB", 21)
    s.memory(32, 160, 800, [56, 8, 16], ["模型实占 56", "环境 8", "空闲 16"], [BLUE, GREEN, ORANGE], 80)
    s.text(32, 260, "② 把空闲 16 GiB 按用途展开", 21)
    s.memory(
        32,
        282,
        800,
        [4, 4, 4, 4],
        ["模型还可能用 4", "环境已答应 4", "安全余量 4", "新命令最多 4"],
        [BLUE, GREEN, PURPLE, ORANGE],
        16,
        72,
    )
    s.line(752, 219, 432, 272, ORANGE, arrow=True)
    s.text(32, 406, "新命令 4 GiB：64 + 4 + 4 + 4 + 4 = 80，容量允许。", 20, GREEN)
    s.text(32, 451, "新命令 8 GiB：总承诺 84，超过 80，等待。", 20, RED)
    s.text(32, 512, "模型 KV 池内部的空槽仍属于模型，环境不能再次分配这份显存。", 18)
    s.footer("已有占用只计一次，未来增长另留预算。容量允许后，还要判断计算竞争。")
    return s


def idle_window():
    s = Scene(
        "借用空档：要装下整段交接，而不只是环境计算",
        "教学假设：从现在起，模型预计到第 60 秒才有新请求；B 是等待 GPU 的环境命令。",
        670,
    )
    s.timeline(
        130,
        60,
        [
            (
                "卡的用途",
                [
                    (0, 3, "", BLUE),
                    (3, 33, "B 计算：30 秒", GREEN),
                    (33, 35, "", ORANGE),
                    (35, 40, "", PURPLE),
                    (40, 60, "模型已就绪", BLUE),
                ],
            )
        ],
        [0, 3, 33, 35, 40, 60],
    )
    for x, title, body, color in [
        (32, "卸载 3 秒", "0 → 3", BLUE),
        (248, "计算 30 秒", "3 → 33", GREEN),
        (464, "清理 2 秒", "33 → 35", ORANGE),
        (680, "恢复 5 秒", "35 → 40", PURPLE),
    ]:
        s.box(x, 300, 188, 100, color)
        s.text(x + 16, 333, title, 20, color)
        s.text(x + 16, 375, body, 18)
    s.text(32, 454, "保守筛选：3 + 30 + 2 + 5 + 安全余量 5 = 45 秒 ≤ 60 秒。", 20)
    s.text(32, 503, "若请求提前到第 8 秒：模型要等到第 40 秒，多等 32 秒。", 20, RED)
    s.footer("空档预测会错。严格限制等待还需要有界归还能力，普通长命令不能瞬间撤回。")
    return s


def gpu_priority():
    s = Scene(
        "独立卡优先；显式共享卡只做后备",
        "教学例子：新命令要 8 GiB。图中共享余量已扣除模型增长、环境预约及安全余量。",
        650,
    )
    for y, name, free1, free0, result, color in [
        (130, "情况一", 12, 20, "选独立卡", GREEN),
        (260, "情况二", 4, 20, "选已绑定共享卡", BLUE),
        (390, "情况三", 4, 6, "两边都不够，等待", ORANGE),
    ]:
        s.text(32, y + 22, name, 20)
        for x, label, free, c in [(150, "独立 GPU1", free1, GREEN), (400, "共享 GPU0", free0, BLUE)]:
            s.box(x, y, 220, 96, c)
            s.text(x + 16, y + 28, label, 20, c)
            s.box(x + 16, y + 46, 170, 26, "#c4cfda", "#edf1f5", 2)
            s.box(x + 16, y + 46, 170 * free / 24, 26, c, radius=2)
            s.text(x + 204, y + 88, f"余量 {free} GiB", 17, MUTED, "end")
        s.text(650, y + 54, result, 18, color)
    s.footer("仅在配置并完成绑定的候选中选择。显存暂时空闲，不会自动把模型卡加入卡池。")
    return s


def sharing_control():
    s = Scene(
        "普通生成可以共运行，模型控制操作要等环境归还",
        "机制示意：共享票据记录在途环境命令。横向表示先后，不按真实秒数绘制。",
        650,
    )
    stages = [(155, "拿票据", BLUE), (310, "环境执行", GREEN), (505, "物理归还", ORANGE), (700, "放票据", BLUE)]
    for x, name, color in stages:
        s.box(x, 160, 135, 60, color)
        s.text(x + 67, 197, name, 19, color, "middle")
    s.text(32, 198, "环境命令", 19)
    for x1, x2 in [(290, 310), (445, 505), (640, 700)]:
        s.line(x1, 190, x2, 190, arrow=True)
    s.text(32, 300, "普通生成", 19)
    s.box(155, 260, 680, 56, PURPLE)
    s.text(495, 295, "模型醒着时，可与环境直接共运行；速度是否划算另测。", 18, PURPLE, "middle")
    s.text(32, 410, "唤醒 / 更新", 19)
    s.box(310, 366, 390, 60, ORANGE)
    s.text(505, 402, "已排队：等待所有共享票据归还", 18, ORANGE, "middle")
    s.line(768, 220, 768, 445, BLUE, dash=True)
    s.box(700, 366, 135, 60, BLUE)
    s.text(767, 402, "再执行控制", 18, BLUE, "middle")
    s.text(155, 500, "控制操作排队后，停止准入新的环境命令，避免控制一直等下去。", 18)
    s.footer("输出可读、命令返回、显存物理释放是不同事件。放票据前必须完成归还确认。")
    return s


def replica_backlog():
    s = Scene(
        "卡归还了，积压请求为什么还没消失？",
        "教学模型：请求持续以 6 个/秒到来；一个副本服务 4 个/秒，两个副本服务 8 个/秒。",
        650,
    )
    x0, y0, width, height = 100, 415, 700, 240
    for count in [0, 80, 160]:
        y = y0 - count * height / 160
        s.line(x0, y, x0 + width, y, "#e1e8f0")
        s.text(82, y + 6, str(count), 17, MUTED, "end")
    s.line(x0, y0, x0 + width, y0, arrow=True)
    s.text(32, 143, "排队请求数", 18)
    points = [(x0, y0), (x0 + width / 2, y0 - height), (x0 + width, y0)]
    s.path(points, ORANGE)
    for time, x in [(0, x0), (80, x0 + width / 2), (160, x0 + width)]:
        s.circle(x, points[1][1] if time == 80 else y0, 5, ORANGE)
        s.text(x, y0 + 28, str(time), 17, MUTED, "middle")
    s.text(820, y0 + 28, "秒", 17, MUTED)
    s.text(130, 500, "0–80 秒：只留一个副本", 20, ORANGE)
    s.text(130, 533, "净增加 6 − 4 = 2 个/秒", 18)
    s.text(490, 500, "80–160 秒：恢复两个副本", 20, BLUE)
    s.text(490, 533, "净消化 8 − 6 = 2 个/秒", 18)
    s.text(450, 153, "第 80 秒归还，已积压 160 个", 18, ORANGE, "middle")
    s.footer("借卡影响可延续到归还之后。这里假定请求成本相同，真实队列还受合批与长度影响。")
    return s


def finite_concurrency():
    s = Scene(
        "并发更大、单条更慢：整批究竟快不快？",
        "教学数据：固定 30 条同类命令，所有工作都必须完成；每波同时开始、同时结束。",
        710,
    )
    values = [(1, 10, 300), (2, 14, 210), (5, 25, 150), (10, 80, 240), (30, 300, 300)]
    s.text(32, 134, "并发数", 18)
    s.text(145, 134, "每波耗时", 18)
    s.text(300, 134, "同样 30 条的总时间 = 波数 × 每波时间", 18)
    for index, (n, one, total) in enumerate(values):
        y = 165 + index * 72
        c = GREEN if n == 5 else BLUE
        s.text(65, y + 25, str(n), 21, c, "middle")
        s.text(180, y + 25, f"{one} 秒", 19, c, "middle")
        s.box(300, y, 470 * total / 300, 38, c, radius=3)
        s.text(312 + 470 * total / 300, y + 26, f"{total} 秒", 19, c)
    s.text(32, 574, "1 → 5 并发：单条 10 → 25 秒，但整批 300 → 150 秒。", 20, GREEN)
    s.footer("先找固定工作量的最短完成时间。显存装得下 30 条，只给出容量范围。")
    return s


def tail_batch():
    s = Scene(
        "只有 12 条待办，10 并发的吞吐最好也未必该选",
        "教学假设：10 条一波用 100 秒，2 条一波用 30 秒；比较相同 12 条工作。",
        640,
    )
    s.timeline(
        135,
        150,
        [
            ("先 10 后 2", [(0, 100, "10 条：100 秒", BLUE), (100, 130, "2 条：30 秒", GREEN)]),
            ("12 条一起①", [(0, 125, "125 秒：比先 10 后 2 早 5 秒", PURPLE)]),
            ("12 条一起②", [(0, 150, "150 秒：比先 10 后 2 晚 20 秒", ORANGE)]),
        ],
        [0, 50, 100, 130, 150],
    )
    s.text(32, 470, "① 持续吞吐略低，但消除了小尾批，当前这批仍更快。", 20, PURPLE)
    s.text(32, 512, "② 额外竞争太重，超过省掉尾批的收益，分两波更快。", 20, ORANGE)
    s.footer("持续满负载看完成率；有限待办看最后完成时刻。两种问题不能共用一个阈值。")
    return s


def three_limits():
    s = Scene(
        "一个“并发数”不够：会话、准备、GPU 要分别看",
        "教学设计：节点可保留 16 个会话，同时准备 4 项，最多让 2 条命令使用 GPU。",
        680,
    )
    s.text(32, 135, "① 会话驻留：16 个", 21, BLUE)
    for i in range(16):
        x = 35 + i * 51
        s.box(x, 160, 42, 48, BLUE)
        s.text(x + 21, 191, str(i + 1), 17, BLUE, "middle")
    s.text(32, 247, "文件、容器和主存还在；其中许多会话正在等模型或排队。", 18)
    s.text(32, 304, "② CPU 准备：4 项", 21, ORANGE)
    for i, name in enumerate(["导入库", "读权重", "预处理", "编译"]):
        x = 35 + i * 211
        s.box(x, 325, 185, 58, ORANGE)
        s.text(x + 92, 361, name, 20, ORANGE, "middle")
    s.text(32, 441, "③ GPU 使用：2 条", 21, GREEN)
    for i in range(2):
        x = 35 + i * 420
        s.box(x, 465, 395, 60, GREEN)
        s.text(x + 197, 502, f"GPU 命令 {i + 1}：预约与执行", 20, GREEN, "middle")
    s.footer("16、4、2 管不同阶段。独立准备限流是设计示意，当前通用命令未自动拆成三段。")
    return s


def instances_requests():
    s = Scene(
        "一个模型实例接很多请求，不等于很多模型实例",
        "教学布局：每实例权重 6 GiB、KV 池 16 GiB、工作区 4 GiB；不据此推断吞吐。",
        670,
    )
    for x, title, copies, color in [(32, "一个实例，多个请求", 1, BLUE), (468, "两个实例，各有请求", 2, PURPLE)]:
        s.box(x, 130, 400, 345, color, fill="white")
        s.text(x + 200, 164, title, 22, color, "middle")
        for i in range(copies):
            y = 190 + i * 130
            s.box(x + 20, y, 360, 105, color)
            s.text(x + 36, y + 30, f"实例 {i + 1}：一套模型与缓存池", 19, color)
            s.memory(x + 36, y + 48, 328, [6, 16, 4], ["权重", "KV", "临时"], [BLUE, GREEN, ORANGE], 26, 40)
        if copies == 1:
            for i in range(5):
                s.circle(x + 60 + i * 65, 359, 18, GREEN)
            s.text(x + 200, 422, "请求在同一引擎内合批", 20, GREEN, "middle")
    s.text(32, 526, "一个实例：26 GiB；两个实例：约 52 GiB，同卡仍要竞争计算。", 20)
    s.footer("请求数影响一个引擎的批；实例数通常复制权重和 KV 池，实例之间不会自动合批。")
    return s


def prediction_intervals():
    s = Scene(
        "预测不该只报一个秒数，还要告诉调度器可能差多少",
        "教学假设：A 预计空档 20 秒；归还需 2 秒，恢复需 3 秒；B 运行时间随输入变化。",
        680,
    )
    s.timeline(
        140,
        30,
        [
            ("A 可用空档", [(0, 20, "预计 20 秒", BLUE)]),
            ("乐观估计", [(0, 2, "", PURPLE), (2, 14, "B：12 秒", GREEN), (14, 17, "", ORANGE)]),
            ("保守估计", [(0, 2, "", PURPLE), (2, 24, "B：22 秒", GREEN), (24, 27, "", ORANGE)]),
        ],
        [0, 10, 20, 30],
    )
    s.text(32, 480, "点预测：2 + 12 + 3 = 17 秒，判断可放入空档。", 20, GREEN)
    s.text(32, 524, "保守上界：2 + 22 + 3 = 27 秒，A 可能多等 7 秒。", 20, ORANGE)
    s.footer("根据相似任务校准预测区间；遇到陌生输入或误差增大，退回保守策略。")
    return s


def rollout_pipeline():
    s = Scene(
        "一条轨迹结束以后，训练还在等什么？",
        "教学假设：训练需要 2 道题，每题 2 条完整尝试；一条尝试可包含多次模型与工具调用。",
        670,
    )
    for row, label, endings, color in [(0, "题 Q1", [4, 7], BLUE), (1, "题 Q2", [5, 10], GREEN)]:
        y = 137 + row * 120
        s.text(32, y + 25, label, 21, color)
        for i, end in enumerate(endings):
            s.box(160, y + i * 49, end * 42, 38, color, radius=3)
            s.text(170 + end * 42, y + i * 49 + 25, f"R{i + 1}：{end} 秒完成", 17, color)
    s.line(580, 125, 580, 400, ORANGE, dash=True)
    s.text(600, 405, "第 10 秒：本例所需 4 条收齐", 18, ORANGE)
    s.box(160, 455, 240, 82, PURPLE)
    s.text(180, 485, "训练前准备", 21, PURPLE)
    s.text(180, 518, "评分、分组与训练数据准备", 17)
    s.line(405, 497, 468, 497, arrow=True)
    s.box(480, 455, 350, 82, BLUE)
    s.text(500, 485, "优化器更新模型", 21, BLUE)
    s.text(500, 518, "满足配置的批量与版本条件后开始", 17)
    s.footer("累计 4 条轨迹 ≠ 同时 4 条 GPU 命令。真实比例和过滤条件由本次训练配置决定。")
    return s


def pipeline_overlap():
    s = Scene(
        "采样与训练重叠，省掉的是哪段等待？",
        "教学假设：每批采样 12 秒、训练 4 秒，三批工作不变，忽略参数同步成本。",
        700,
    )
    s.timeline(
        130,
        48,
        [
            (
                "串行",
                [
                    (0, 12, "采样1", BLUE),
                    (12, 16, "训1", ORANGE),
                    (16, 28, "采样2", BLUE),
                    (28, 32, "训2", ORANGE),
                    (32, 44, "采样3", BLUE),
                    (44, 48, "训3", ORANGE),
                ],
            ),
            ("重叠：采样", [(0, 12, "采样1", BLUE), (12, 24, "采样2", BLUE), (24, 36, "采样3", BLUE)]),
            ("重叠：训练", [(12, 16, "训1", ORANGE), (24, 28, "训2", ORANGE), (36, 40, "训3", ORANGE)]),
        ],
        [0, 12, 24, 36, 48],
    )
    s.text(32, 475, "完全串行：3 × (12 + 4) = 48 秒。", 21, BLUE)
    s.text(32, 520, "允许重叠：12 + 12 + 12 + 4 = 40 秒，最后一批训练仍要做。", 20, GREEN)
    s.footer("条件：算法允许稍旧参数采样。若下一批必须用刚更新的参数，这种重叠就不合法。")
    return s


def resource_reallocation():
    s = Scene(
        "八张卡怎样分：资源应该帮助较慢的那一段",
        "教学模型：每份工作都先过阶段 A，再过阶段 B；持续供给时由较慢阶段限制速率。",
        650,
    )
    for index, (a, arate, brate) in enumerate([(4, 10, 4), (3, 9, 5), (2, 6, 6)]):
        y = 155 + index * 126
        s.text(32, y + 31, f"A/B = {a}/{8 - a}", 20)
        for i in range(8):
            x = 195 + i * 52
            color = BLUE if i < a else GREEN
            s.box(x, y, 42, 46, color)
            s.text(x + 21, y + 30, "A" if i < a else "B", 20, color, "middle")
        s.text(680, y + 30, f"有效 {min(arate, brate)} 个/秒", 21, GREEN)
        s.text(195, y + 82, f"A 能做 {arate} 个/秒；B 能做 {brate} 个/秒，取较小者。", 18)
    s.footer("迁移成本和剩余工作另算。持续速率更高，不保证只有几项待办时也值得搬卡。")
    return s


def code_versions():
    s = Scene(
        "文件已经改了，进程为什么还会跑旧代码？",
        "机制示意：Python 已把模块读入进程；之后修改磁盘文件，不会自动替换内存对象。",
        670,
    )
    for x, title, file_v, process_v, color in [
        (32, "修改前", "v1", "v1", BLUE),
        (468, "只改文件后", "v2", "v1", ORANGE),
    ]:
        s.box(x, 133, 400, 345, color, fill="white")
        s.text(x + 200, 168, title, 23, color, "middle")
        s.box(x + 28, 199, 145, 115, BLUE)
        s.text(x + 100, 238, "磁盘代码", 20, BLUE, "middle")
        s.text(x + 100, 280, file_v, 29, BLUE, "middle")
        s.box(x + 225, 199, 145, 115, PURPLE)
        s.text(x + 297, 238, "进程内模块", 19, PURPLE, "middle")
        s.text(x + 297, 280, process_v, 29, PURPLE, "middle")
        s.line(x + 177, 257, x + 221, 257, dash=x > 100, arrow=True)
        s.text(x + 200, 376, "首次导入" if x < 100 else "没有重新导入 / 重建", 20, color, "middle")
        s.text(x + 200, 425, "文件与进程一致" if x < 100 else "结果可能属于旧版本", 20, color, "middle")
    s.text(32, 532, "复用前检查状态依赖；代码或数据变了，就更新相应状态，再执行。", 20)
    s.footer("容器常驻只保留文件和进程环境。安全复用需要另管版本与内存状态的对应关系。")
    return s


def state_dependencies():
    s = Scene(
        "改了哪里，就重建依赖它的状态", "教学依赖图：模型权重固定，数据和评分代码可变化。颜色表示不同来源的依赖。", 670
    )
    rows = [
        ("模型权重 v1", "GPU 模型", "改权重 → 重载模型", BLUE),
        ("数据版本 v2", "预处理 / 数据迭代器", "改数据 → 重建数据状态", GREEN),
        ("评分代码 v3", "评分器 / 结果标签", "改评分 → 重建评分状态", PURPLE),
    ]
    for index, (source, state, action, color) in enumerate(rows):
        y = 135 + index * 133
        s.box(32, y, 225, 68, color)
        s.text(144, y + 42, source, 21, color, "middle")
        s.line(265, y + 34, 325, y + 34, arrow=True)
        s.box(335, y, 300, 68, color)
        s.text(485, y + 42, state, 21, color, "middle")
        s.text(335, y + 107, action, 19, color)
    s.text(675, 215, "最终输出", 20)
    s.text(675, 252, "携带本轮版本", 18)
    s.text(675, 289, "拒收旧结果", 18, RED)
    s.footer("只改评分代码时，可以保留不依赖它的模型准备。通用依赖自动识别仍需要执行协议。")
    return s


def prepare_gpu():
    s = Scene(
        "CPU 先准备好，再占 GPU，能省掉哪段卡上空等？",
        "教学假设：4 项任务，各准备 6 秒、GPU 计算 2 秒；CPU 足够并行，环境只有 2 张卡。",
        690,
    )
    s.timeline(
        135,
        16,
        [
            (
                "先拿卡：卡0",
                [(0, 6, "准备1", ORANGE), (6, 8, "算1", GREEN), (8, 14, "准备3", ORANGE), (14, 16, "算3", GREEN)],
            ),
            (
                "先拿卡：卡1",
                [(0, 6, "准备2", ORANGE), (6, 8, "算2", GREEN), (8, 14, "准备4", ORANGE), (14, 16, "算4", GREEN)],
            ),
            ("先准备：CPU", [(0, 6, "四项同时准备", ORANGE)]),
            ("先准备：两卡", [(6, 8, "1+2", GREEN), (8, 10, "3+4", GREEN)]),
        ],
        [0, 6, 8, 10, 16],
        step=65,
    )
    s.text(32, 530, "同样 4 项：16 秒 → 10 秒；提前做准备，没有减少 GPU 工作量。", 20, GREEN)
    s.footer("这是已有 prepared 接口能表达的合作式执行。它不会自动拆解任意 shell 命令。")
    return s


def measurement_ranking():
    s = Scene("重复测得很稳定，也可能稳定地选错代码", "教学假设：A、B 的竞争敏感度不同。横条越短越快，单位毫秒。", 650)
    for index, (title, values) in enumerate(
        [
            ("无竞争目标", [("A", 10.0, BLUE), ("B", 10.2, GREEN)]),
            ("共享环境观测", [("A", 11.0, BLUE), ("B", 10.302, GREEN)]),
        ]
    ):
        y = 150 + index * 180
        s.text(32, y, title, 22)
        for j, (name, value, color) in enumerate(values):
            yy = y + 25 + j * 57
            s.text(55, yy + 27, name, 22, color)
            s.box(105, yy, 620 * value / 12, 38, color, radius=3)
            s.text(120 + 620 * value / 12, yy + 26, f"{value:g} ms", 19, color)
    s.text(32, 542, "目标应选 A；共享时却选 B。多次取中位数也消不掉这种系统偏差。", 19, RED)
    s.footer("时间作为评分时，要控制测量条件；环境吞吐和反馈质量必须分别验证。")
    return s


def local_vs_service_time():
    s = Scene(
        "程序只算了 1 ms，为什么请求要等 24 ms？",
        "教学假设：排队 20 ms、设备排空 3 ms、实际测量 1 ms，其他开销暂忽略。",
        550,
    )
    s.timeline(
        145,
        24,
        [("用户看到", [(0, 20, "排队：20 ms", ORANGE), (20, 23, "3 ms", PURPLE), (23, 24, "", GREEN)])],
        [0, 10, 20, 23, 24],
        "毫秒",
    )
    s.box(32, 325, 400, 105, GREEN)
    s.text(52, 360, "候选程序性能：1 ms", 22, GREEN)
    s.text(52, 400, "回答“代码本身快不快”", 18)
    s.box(468, 325, 400, 105, ORANGE)
    s.text(488, 360, "服务端到端延迟：至少 24 ms", 21, ORANGE)
    s.text(488, 400, "回答“结果何时拿得到”", 18)
    s.footer("两个时钟回答不同问题。只报其中一个，会漏掉代码质量或调度成本。")
    return s


def resource_layers():
    s = Scene(
        "谁分资源，谁懂任务：中央层和应用层怎样配合？",
        "YARN 启发的职责示意。这里描述分工，不声称 PSRL 已实现完整应用级性能调度。",
        660,
    )
    s.box(32, 130, 836, 135, BLUE)
    s.text(52, 165, "中央资源层：知道哪里还有多少资源", 22, BLUE)
    for i, label in enumerate(["节点 A：CPU / 主存 / GPU", "节点 B：CPU / 主存 / GPU", "申请 → 配额 → 回收"]):
        s.text(52 + i * 270, 217, label, 17)
    for x, color, title, body in [
        (32, GREEN, "应用 A：多轮 Agent", "下一轮依赖哪个工具？\n哪些状态下次还要复用？"),
        (468, PURPLE, "应用 B：模型训练", "哪些 worker 必须一起工作？\n何时允许安全切换？"),
    ]:
        s.line(x + 200, 275, x + 200, 325, arrow=True)
        s.box(x, 340, 400, 175, color)
        s.text(x + 20, 377, title, 22, color)
        s.text(x + 20, 423, body, 19)
    s.footer("中央层守住总预算；应用层提供依赖与状态代价。只看显存百分比，无法知道谁更急。")
    return s


def warm_layers():
    s = Scene("“已经预热”到底保留了哪一层？", "机制示意：越往下保留的准备越多，但占用资源和版本管理责任也越多。", 650)
    rows = [
        ("容器已就绪", ["镜像", "依赖", "工作目录"], "仍可能要启动 Python、读权重", BLUE),
        ("CPU 进程常驻", ["已导入库", "数据", "CPU 权重"], "可省部分准备，仍可能要传到 GPU", ORANGE),
        ("GPU 模型常驻", ["权重", "缓存", "执行状态"], "可更快继续，但持续占显存", GREEN),
    ]
    for index, (name, items, note, color) in enumerate(rows):
        y = 130 + index * 145
        s.text(32, y + 28, name, 20, color)
        for i, item in enumerate(items):
            x = 230 + i * 190
            s.box(x, y, 170, 50, color)
            s.text(x + 85, y + 32, item, 19, color, "middle")
        s.text(230, y + 88, note, 18)
    s.footer("容器复用、进程复用、模型复用分别验收。Firepool 的 warm pod 不等于常驻 GPU 模型。")
    return s


def nested_deadlock():
    s = Scene(
        "父任务等子任务，为什么可能谁也走不下去？",
        "教学假设：只有一份 GPU 容量。父任务保留它，同时等待子任务完成评估。",
        640,
    )
    s.box(32, 145, 350, 170, BLUE)
    s.text(52, 183, "父任务 A", 24, BLUE)
    s.text(52, 228, "持有 GPU", 21)
    s.text(52, 273, "等待子任务结果", 21)
    s.box(518, 145, 350, 170, ORANGE)
    s.text(538, 183, "子任务 B", 24, ORANGE)
    s.text(538, 228, "先拿到 GPU，才能评估", 21)
    s.text(538, 273, "等待 A 归还容量", 21)
    s.line(385, 180, 512, 180, arrow=True)
    s.line(518, 290, 389, 290, arrow=True)
    s.text(450, 360, "持有资源的等待形成环", 23, RED, "middle")
    s.box(32, 410, 836, 100, GREEN)
    s.text(52, 445, "可行动作：A 先安全释放 GPU，或为 B 预留独立容量。", 21, GREEN)
    s.text(52, 483, "单纯提高队列优先级无效，因为没有可供 B 使用的资源。", 18)
    s.footer("这是提案中的风险场景，需同时管理父子依赖、资源持有和归还协议。")
    return s


def model_promotion():
    s = Scene(
        "工具里训出模型，不等于它已成为下一代 Agent",
        "机制示意：任务模型产物与当前策略模型属于不同用途。模型接管是单独的提案。",
        680,
    )
    s.box(32, 145, 350, 145, BLUE)
    s.text(52, 181, "当前 Agent：策略版本 v10", 22, BLUE)
    s.text(52, 225, "仍负责生成动作与推进任务", 19)
    s.text(52, 264, "原生训练更新走既有同步路径", 17)
    s.box(518, 145, 350, 145, ORANGE)
    s.text(538, 181, "工具训练出的候选模型", 22, ORANGE)
    s.text(538, 225, "先作为文件 / 权重产物保存", 19)
    s.text(538, 264, "还没获得 Agent 服务身份", 17)
    s.line(695, 295, 695, 330)
    s.line(695, 330, 165, 330)
    s.line(165, 330, 165, 365, arrow=True)
    for i, (title, color) in enumerate(
        [("验证格式与能力", PURPLE), ("独立评估是否通过", GREEN), ("原子切换 + 可回退", BLUE)]
    ):
        x = 32 + i * 284
        s.box(x, 375, 266, 95, color)
        s.text(x + 133, 431, title, 20, color, "middle")
        if i < 2:
            s.line(x + 266, 422, x + 282, 422, arrow=True)
    s.text(32, 533, "切换后：新请求用新模型；在途旧请求、KV 缓存和结果版本仍需分开处理。", 18)
    s.footer("接管闭环尚未接入生产。判断自改进，必须核对产物、评估与服务切换三步。")
    return s


def grouped_placement():
    s = Scene(
        "另一台机器有空卡，为什么这组任务还在本机排队？",
        "当前 Science/Harbor 路径示意：同题 8 条先作为一组派给 worker，之后申请限定本节点。",
        690,
    )
    for x, title, active, waiting, color in [
        (32, "节点 A：本组 worker 在这里", 4, 4, BLUE),
        (468, "节点 B：另有空闲资源", 0, 0, GREEN),
    ]:
        s.box(x, 140, 400, 360, color, fill="white")
        s.text(x + 20, 178, title, 21, color)
        s.text(x + 20, 217, "环境槽位", 18)
        for i in range(4):
            xx = x + 22 + i * 92
            s.box(xx, 239, 78, 62, color if active else "#b5c3d2", fill=None if active else "#f0f4f8")
            s.text(xx + 39, 278, f"R{i + 1}" if active else "空", 21, color, "middle")
        if waiting:
            s.text(x + 20, 351, "R5、R6、R7、R8 等待", 21, ORANGE)
            s.text(x + 20, 394, "重试仍携带 node_ip=A", 19)
            s.text(x + 20, 441, "重新检查 A，不重派到 B", 19)
        else:
            s.text(x + 20, 351, "Router 知道 B 的容量", 21, GREEN)
            s.text(x + 20, 394, "但本次申请的候选只含 A", 19)
            s.text(x + 20, 441, "当前没有跨节点接续重派", 19)
    s.footer("同机命令换卡与跨节点迁移是两回事。前者保留容器，后者还要处理文件和执行位置。")
    return s


def gpu_return_states():
    s = Scene(
        "结果已经打印出来，卡为什么还不能交给下一条命令？",
        "机制示意：输出就绪与物理归还是两个事件；背景 GPU 进程仍可能持有显存。",
        650,
    )
    for state, (x, title, items, color) in enumerate(
        [
            (32, "① 输出可读", ["stdout / 文件可返回", "GPU 子进程可能仍活着"], BLUE),
            (318, "② 撤权并清理", ["阻止新的设备访问", "等待旧句柄与进程退出"], ORANGE),
            (604, "③ 资源可复用", ["核对旧占用已经退出", "再释放账本与共享票据"], GREEN),
        ]
    ):
        s.box(x, 155, 264, 244, color)
        s.text(x + 18, 193, title, 22, color)
        s.text(x + 18, 228, ["旧子进程仍占显存", "清理中：占用逐步退出", "核对完成：旧占用已清空"][state], 16)
        for i in range(4):
            occupied = i < [4, 2, 0][state]
            s.box(
                x + 18 + i * 57,
                241,
                51,
                39,
                color if occupied else "#CBD5E1",
                fill=color if occupied else "white",
                radius=2,
            )
        s.text(
            x + 18, 301, ["尚不能承诺给下一条命令", "尚不能承诺给下一条命令", "此时才可发出新的承诺"][state], 16, color
        )
        for i, item in enumerate(items):
            s.text(x + 18, 344 + i * 29, item, 17)
    s.text(32, 464, "若清理失败：保持隔离并报告失败，不能只把预约数改成零。", 20, RED)
    s.footer("账本释放描述“可以再承诺”；物理释放证明“旧程序已让出资源”。两者顺序很重要。")
    return s


def selective_batching():
    s = Scene(
        "投影可以合批，attention 仍要记住每条请求的边界",
        "教学例子：A、B 各 decode 1 个 token，C prefill 4 个 token，共 6 行投影输入。",
        690,
    )
    colors = [BLUE, GREEN, PURPLE, PURPLE, PURPLE, PURPLE]
    labels = ["A", "B", "C1", "C2", "C3", "C4"]
    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        y = 145 + i * 42
        s.box(50, y, 95, 36, color, radius=2)
        s.text(97, y + 25, label, 19, color, "middle")
    s.text(50, 440, "6 个新 token", 19)
    s.line(155, 266, 230, 266, arrow=True)
    s.box(242, 180, 220, 180, ORANGE)
    s.text(352, 220, "共享投影权重 W", 21, ORANGE, "middle")
    for j in range(4):
        s.line(269 + j * 41, 244, 269 + j * 41, 330, ORANGE)
        s.line(269, 246 + j * 27, 392, 246 + j * 27, ORANGE)
    s.text(242, 408, "沿 token 维拼成一次矩阵计算", 18)
    s.line(468, 266, 525, 266, arrow=True)
    for i, (name, color) in enumerate(
        [("A → 只读 A 的历史 KV", BLUE), ("B → 只读 B 的历史 KV", GREEN), ("C → 只读 C，保留因果遮罩", PURPLE)]
    ):
        y = 150 + i * 100
        s.box(545, y, 320, 70, color)
        s.text(561, y + 42, name, 18, color)
    s.text(32, 527, "合批复用权重读取；请求边界保证 A 不会看到 B 的历史。", 21)
    s.footer("这是算子分工示意，具体内核可融合执行。合成一个矩阵不等于合并会话语义。")
    return s


def kv_copy_on_write():
    s = Scene(
        "三条请求共享前缀，继续写时为什么要复制最后一块？",
        "教学假设：每块 4 个 token，前缀已有 6 个 token。块号只是示意，非实际物理地址。",
        690,
    )
    s.text(32, 136, "共享前缀", 22, BLUE)
    for i, (title, cells) in enumerate(
        [("块 P0：已满", ["1", "2", "3", "4"]), ("块 P1：未满", ["5", "6", "空", "空"])]
    ):
        x = 240 + i * 290
        s.text(x, 136, title, 20, BLUE)
        for j, label in enumerate(cells):
            s.box(x + j * 60, 157, 54, 52, BLUE)
            s.text(x + j * 60 + 27, 190, label, 18, BLUE, "middle")
    s.text(32, 276, "分支各自续写", 22)
    for i, (name, color) in enumerate([("A", GREEN), ("B", PURPLE), ("C", ORANGE)]):
        y = 300 + i * 78
        s.text(48, y + 31, f"请求 {name}", 20, color)
        s.box(240, y, 240, 48, BLUE)
        s.text(360, y + 31, "仍引用同一个 P0", 19, BLUE, "middle")
        for j, label in enumerate(["5", "6", f"{name}7", f"{name}8"]):
            s.box(530 + j * 60, y, 54, 48, color)
            s.text(557 + j * 60, y + 31, label, 18, color, "middle")
    s.footer("已满且不再修改的块可以共享；对共享的未满块续写，要先复制，避免改到其他请求。")
    return s


def continuous_batching():
    s = Scene(
        "连续合批：一条完成，就让新请求补进下一轮",
        "教学格子每列是一轮 decode，表示参与批次，不表示每轮真实耗时相同。",
        610,
    )
    names = ["A", "B", "C", "D"]
    starts, ends = [0, 0, 2, 4], [2, 5, 6, 6]
    colors = [BLUE, GREEN, PURPLE, ORANGE]
    for j in range(6):
        s.text(252 + j * 92, 142, f"轮 {j + 1}", 18, MUTED, "middle")
    for i, name in enumerate(names):
        y = 172 + i * 65
        s.text(50, y + 31, f"请求 {name}", 20, colors[i])
        for j in range(6):
            active = starts[i] <= j < ends[i]
            color = colors[i] if active else "#c6d0da"
            s.box(215 + j * 92, y, 74, 46, color, fill=None if active else "#f3f6f9", radius=3)
            if active:
                s.text(252 + j * 92, y + 31, name, 20, color, "middle")
    s.text(32, 491, "A 在第 2 轮结束；C 从第 3 轮加入，无须等 B 也全部结束。", 20)
    s.footer("一个引擎调度多条请求。实际每轮还受 KV 容量、token 预算、prefill 和内核效率影响。")
    return s


def parallel_communication():
    s = Scene(
        "切开模型以后，省下显存也增加了搬运",
        "教学分工：TP 切同一层，PP 切不同层，DP 复制模型处理不同样本；不表示实际时延比例。",
        690,
    )
    rows = [
        ("TP：同一层", "矩阵左半", "矩阵右半", "交换部分结果 / 归约", BLUE),
        ("PP：不同层", "前半层", "后半层", "把激活传给下一段", GREEN),
        ("DP：不同样本", "完整模型：样本 A", "完整模型：样本 B", "训练时同步梯度", PURPLE),
    ]
    for i, (title, a, b, link, color) in enumerate(rows):
        y = 140 + i * 147
        s.text(32, y + 30, title, 21, color)
        s.box(230, y, 260, 68, color)
        s.box(590, y, 260, 68, color)
        s.text(360, y + 42, a, 20, color, "middle")
        s.text(720, y + 42, b, 20, color, "middle")
        s.line(500, y + 33, 578, y + 33, color, arrow=True)
        s.text(230, y + 109, link, 19)
    s.footer("显存决定放不放得下；互联带宽决定切开后传得够不够快。增加卡数不自动等比提速。")
    return s


def ppo_credit():
    s = Scene(
        "整条回答只有最终得分，前面的 token 怎样得到训练信号？",
        "教学关系图：奖励先转成优势 A，再给生成 token 的概率更新方向；图中不比较算法效果。",
        650,
    )
    s.text(32, 140, "模型生成的动作 / token", 21)
    for i in range(7):
        x = 48 + i * 113
        color = BLUE if i < 6 else GREEN
        s.box(x, 164, 92, 60, color)
        s.text(x + 46, 201, f"t{i + 1}", 22, color, "middle")
    s.box(599, 274, 220, 76, GREEN)
    s.text(709, 305, "最终判题奖励", 21, GREEN, "middle")
    s.text(709, 335, "例如成功 / 失败", 17, GREEN, "middle")
    s.line(709, 350, 709, 397, arrow=True)
    s.box(48, 407, 771, 98, PURPLE)
    s.text(68, 444, "PPO：结合价值估计，把后续收益传回较早动作。", 21, PURPLE)
    s.text(68, 482, "GRPO：比较同题多条回答的奖励，形成组内相对方向。", 20, PURPLE)
    s.footer("优势决定鼓励还是抑制；概率比和裁剪再约束这次更新幅度。它们回答不同问题。")
    return s


def gpu_pool_and_host():
    s = Scene(
        "下一条命令换卡，容器和主存需要一起搬吗？",
        "当前同机命令池的资源边界示意。A 是已有会话，B 是占用 A 原卡的其他命令。",
        660,
    )
    s.box(32, 130, 836, 165, BLUE)
    s.text(52, 168, "节点 N：会话 A 的主机资源不动", 23, BLUE)
    for i, name in enumerate(["容器与工作文件", "CPU 配额", "主存与进程状态"]):
        x = 52 + i * 265
        s.box(x, 197, 245, 60, BLUE, fill="white")
        s.text(x + 122, 234, name, 20, BLUE, "middle")
    for x, title, body, color in [
        (32, "GPU0：原卡", "B 正在执行\nA 可以等待", ORANGE),
        (468, "GPU1：同机候选", "容量允许时\nA 可在此执行下一条命令", GREEN),
    ]:
        s.box(x, 359, 400, 165, color)
        s.text(x + 20, 395, title, 22, color)
        s.text(x + 20, 440, body, 20)
    s.line(260, 299, 665, 349, GREEN, arrow=True)
    s.footer("移动的是下一条命令的设备授权。活 GPU 状态不透明迁移，跨节点还需另管文件与状态。")
    return s


def compute_sharing():
    s = Scene(
        "MPS 提供共运行机会，不会把 GPU 计算能力翻倍",
        "教学资源格：每格代表一份可用计算机会。实际 SM 分配并非下图这种固定切块。",
        670,
    )
    for row, title, busy, color in [(0, "A 没用满：B 可能填空", 8, GREEN), (1, "A 已用满：B 需要竞争", 16, ORANGE)]:
        y = 150 + row * 190
        s.text(32, y, title, 22, color)
        for i in range(16):
            x = 40 + i * 51
            cc = BLUE if i < busy else GREEN
            s.box(x, y + 27, 42, 54, cc, radius=3)
            s.text(x + 21, y + 62, "A" if i < busy else "B", 19, cc, "middle")
        s.text(
            32,
            y + 126,
            "有机会提高总完成率，仍需实测延迟与带宽竞争。"
            if row == 0
            else "B 加入后双方分时或争用，是否划算要比较整批时间。",
            19,
        )
    s.footer("显存占用低、SM 有空闲、显存带宽有空闲是三个条件，不能只凭其中一个判断。")
    return s


def process_reuse():
    s = Scene(
        "保留进程与模型，究竟省的是哪段时间？",
        "教学假设：每次导入库 3 秒、加载模型 5 秒、计算 2 秒，共执行三次相同检查。",
        650,
    )
    segments = []
    for i in range(3):
        t = i * 10
        segments.extend([(t, t + 3, "库", ORANGE), (t + 3, t + 8, "加载", PURPLE), (t + 8, t + 10, "算", GREEN)])
    s.timeline(
        145,
        30,
        [
            ("每次新进程", segments),
            (
                "模型常驻",
                [
                    (0, 3, "库", ORANGE),
                    (3, 8, "加载", PURPLE),
                    (8, 10, "算", GREEN),
                    (10, 12, "算", GREEN),
                    (12, 14, "算", GREEN),
                ],
            ),
        ],
        [0, 8, 14, 20, 30],
        step=90,
    )
    s.text(32, 452, "独立进程：3 × (3 + 5 + 2) = 30 秒。", 21, ORANGE)
    s.text(32, 495, "合法复用：3 + 5 + 3 × 2 = 14 秒。计算工作量没有减少。", 21, GREEN)
    s.footer("省时的前提是状态仍有效。代码、数据或模型变化后，须重建受影响状态再复用。")
    return s


def remaining_steps():
    s = Scene(
        "知道还剩多少小步，才比较容易估计结束时间",
        "教学假设：程序共 100 步，已完成 80 步；近期每步 0.2 秒，剩余收尾预计 3 秒。",
        620,
    )
    s.text(32, 144, "执行进度", 22, BLUE)
    for i in range(20):
        x = 38 + i * 41
        cc = BLUE if i < 16 else ORANGE
        s.box(x, 175, 33, 65, cc, radius=3)
    s.text(38, 280, "已完成 80 步", 20, BLUE)
    s.text(672, 280, "剩余 20 步", 20, ORANGE)
    s.text(32, 355, "从现在到完成：20 × 0.2 + 3 = 7 秒。", 24, GREEN)
    s.text(32, 410, "若刚才的报告已经过期，不能每过一秒就机械地减一秒。", 19)
    s.text(32, 455, "程序可能卡住、下一步变慢，或尚有未报告的保存与显存清理。", 19)
    s.footer("这是基于进度的估计。必须绑定工作量、运行阶段和并发条件，并用实际完成记录校准。")
    return s


def episode_admission():
    s = Scene(
        "8 条尝试都在推进，不等于 8 个环境都已获准",
        "沿用正文配置：两个独立GPU unit，每个能容纳2条episode，各预约20 GiB。",
        690,
    )
    for x, title, groups, waiting, color in [
        (32, "① 先有4条获准", [["R1", "R3"], ["R2", "R4"]], ["R5", "R6", "R7", "R8"], BLUE),
        (468, "② R1归还后，R5获准", [["R5", "R3"], ["R2", "R4"]], ["R6", "R7", "R8"], GREEN),
    ]:
        s.box(x, 135, 400, 400, color, fill="white")
        s.text(x + 20, 175, title, 23, color)
        for i, group in enumerate(groups):
            y = 205 + i * 112
            s.text(x + 20, y + 25, f"GPU{i}", 19, MUTED)
            for j, session in enumerate(group):
                xx = x + 110 + j * 132
                cc = GREEN if session == "R5" else BLUE
                s.box(xx, y, 115, 74, cc)
                s.text(xx + 57, y + 31, session, 23, cc, "middle")
                s.text(xx + 57, y + 61, "20 GiB", 17, cc, "middle")
        s.text(x + 20, 456, "等待申请", 19, ORANGE)
        for j, session in enumerate(waiting):
            xx = x + 115 + j * 65
            s.box(xx, 429, 55, 43, ORANGE)
            s.text(xx + 27, 458, session, 18, ORANGE, "middle")
        s.text(x + 20, 507, "未获准的环境尚未创建受管理容器", 17, MUTED)
    s.footer("每条episode独立清理并释放预算。等待client重试后接续获准，不保证严格FIFO。")
    return s


DRAWERS = {
    "chapter-critical-path": critical_path,
    "chapter-memory-headroom": memory_headroom,
    "chapter-idle-window": idle_window,
    "chapter-gpu-priority": gpu_priority,
    "chapter-sharing-control": sharing_control,
    "chapter-replica-backlog": replica_backlog,
    "chapter-finite-concurrency": finite_concurrency,
    "chapter-tail-batch": tail_batch,
    "chapter-three-limits": three_limits,
    "chapter-instances-requests": instances_requests,
    "chapter-prediction-intervals": prediction_intervals,
    "chapter-rollout-pipeline": rollout_pipeline,
    "chapter-pipeline-overlap": pipeline_overlap,
    "chapter-resource-reallocation": resource_reallocation,
    "chapter-code-versions": code_versions,
    "chapter-state-dependencies": state_dependencies,
    "chapter-prepare-gpu": prepare_gpu,
    "chapter-measurement-ranking": measurement_ranking,
    "chapter-local-service-time": local_vs_service_time,
    "chapter-resource-layers": resource_layers,
    "chapter-warm-layers": warm_layers,
    "chapter-nested-deadlock": nested_deadlock,
    "chapter-model-promotion": model_promotion,
    "chapter-grouped-placement": grouped_placement,
    "chapter-gpu-return-states": gpu_return_states,
    "chapter-selective-batching": selective_batching,
    "chapter-kv-copy-on-write": kv_copy_on_write,
    "chapter-continuous-batching": continuous_batching,
    "chapter-parallel-communication": parallel_communication,
    "chapter-ppo-credit": ppo_credit,
    "chapter-gpu-pool-host": gpu_pool_and_host,
    "chapter-compute-sharing": compute_sharing,
    "chapter-process-reuse": process_reuse,
    "chapter-remaining-steps": remaining_steps,
    "chapter-episode-admission": episode_admission,
}


def render(identity):
    """Render a selected chapter figure while leaving historical sources untouched."""
    DRAWERS[identity]().save(identity)


def main():
    """Render the chapter collection or a selected subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    for identity in args.only or DRAWERS:
        render(identity)
        print(identity, flush=True)


if __name__ == "__main__":
    main()
