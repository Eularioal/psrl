"""Compose explanatory objects, worked state changes and contextual evidence panels."""

import copy
import hashlib
import io
import json
import re
import runpy
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
INK, MUTED = "#203149", "#53677c"
COLORS = ["#3478c5", "#159b8d", "#d68b20", "#8764b8", "#c84f60"]
TINTS = ["#eaf2ff", "#e6f6f2", "#fff2db", "#f2edfb", "#fdecef"]
CASES = json.loads((ROOT / "explanations.json").read_text())
BOUNDS = []
SVG_NS = "{http://www.w3.org/2000/svg}"
RASTER_GRIDS = {
    "paired-latency.png": (2, 2, 0.065),
    "fixed-work-outcomes.png": (2, 3, 0.055),
    "fixed-work-training.png": (1, 3, 0.055),
    "startup-offload.png": (1, 3, 0.11),
}
# Coordinates use a 900-pixel source width. The catalog pins both source hashes.
RASTER_LEGENDS = {
    "historical-timing.png": {
        "rows": [(620, 287 + i * 15, 879, 300 + i * 15) for i in range(4)],
        "labels": [
            "Wait for rollout results (includes queue/tasks)",
            "Old log probability",
            "Actor update",
            "Other",
        ],
    },
    "fixed-work-environment-memory.png": {
        "rows": [(51, y, 225, y + 10) for y in (314, 327, 340, 352)],
        "labels": [
            "GPU 6 physical memory",
            "GPU 7 physical memory",
            "Total physical memory (both GPUs)",
            "Total declared memory (both units)",
        ],
    },
}
LITERATURE_DRAWERS = {
    "reading-map": "reading_map",
    "borrowing-timeline": "borrowing",
    "rollout-independence": "rollout",
    "pipeline-bottleneck": "bottleneck",
    "gate-and-drain": "drain",
    "protected-regions": "regions",
    "krex-control-flow": "krex_flow",
    "scheduling-two-lanes": "scheduling_lanes",
    "reward-distortion": "reward",
    "benchmark-choice": "benchmarks",
    "model-decision-cost": "model_cost",
    "yarn-responsibilities": "yarn",
    "rsi-two-loops": "rsi",
    "gpu-task-lifecycle": "task_lifecycle",
    "psrl-training-loop": "psrl_loop",
    "roofline-and-batching": "roofline",
}


def wrapped(value, width=42):
    """Wrap CJK and Latin text using approximate display columns."""
    import unicodedata

    lines = []
    for paragraph in value.split("\n"):
        line, used = "", 0
        for word in re.findall(r"[A-Za-z0-9_.+/<>=−×]+|.", paragraph):
            amount = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in word)
            if used + amount > width and line:
                lines.append(line)
                line, used = "", 0
            line += word
            used += amount
        lines.append(line)
    return "\n".join(lines)


def label(ax, x, y, text, width=94, height=14, size=20, color=INK, bold=False):
    """Track allocated label bounds for the rendered layout audit."""
    artist = ax.text(
        x, y, text, fontsize=size, color=color, weight="bold" if bold else "normal", va="top", linespacing=1.35
    )
    BOUNDS.append((artist, ax, x, y, width, height))
    return artist


def axis(fig, bounds):
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 100), ylim=(100, 0))
    ax.axis("off")
    return ax


def panel(ax, x, y, width, height, color=0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            facecolor=TINTS[color],
            edgecolor=COLORS[color],
            linewidth=1.2,
        )
    )


def header(spec, subtitle, height=16):
    BOUNDS.clear()
    fig = plt.figure(figsize=(16, height), facecolor="white")
    ax = axis(fig, (0.04, 1 - 1.65 / height, 0.92, 1.45 / height))
    label(ax, 0, 0, wrapped(spec["title"], 94), height=51, size=26, bold=True)
    label(ax, 0, 56, wrapped(subtitle, 100), height=44, size=20, color=MUTED)
    return fig


def footer(fig, result, limit, bounds=None):
    if bounds is None:
        height = fig.get_figheight()
        bounds = (0.04, 0.1 / height, 0.92, 1.55 / height)
    ax = axis(fig, bounds)
    panel(ax, 0, 0, 100, 52, 1)
    label(ax, 1.5, 7, wrapped(result, 98), width=97, height=41, size=20, color=COLORS[1], bold=True)
    label(ax, 0, 63, wrapped(limit, 112), height=35, size=18, color=MUTED)


def state_cards(fig, states, bounds=None):
    if bounds is None:
        height = fig.get_figheight()
        bounds = (0.04, 1.9 / height, 0.92, 4.5 / height)
    ax = axis(fig, bounds)
    for i, (title, rows) in enumerate(states):
        y = i * 34
        panel(ax, 0, y, 100, 30, i)
        label(
            ax,
            2,
            y + 4,
            wrapped(f"{i + 1:02d}  {title}", 25),
            width=25,
            height=24,
            size=21,
            color=COLORS[i],
            bold=True,
        )
        for j, row in enumerate(rows):
            row_y = y + 3 + j * 9
            ax.add_patch(Rectangle((29, row_y + 1.3), 0.6, 3.0, color=COLORS[i]))
            label(ax, 31, row_y, wrapped(row, 70), width=67, height=10.5, size=20)


def mechanism(spec, base):
    case = CASES[spec["kind"]]
    fig = header(spec, case["question"], height=18 if spec["kind"] == "ppo" else 16)
    height = fig.get_figheight()
    top = axis(fig, (0.04, 6.7 / height, 0.92, (height - 9.1) / height))
    base["TEXT_BOXES"].clear()
    if spec["kind"] == "ppo":
        top.remove()
        base["draw_ppo"](fig)
        for i, ax in enumerate(fig.axes[-2:]):
            ax.set_position((0.10, (12.1 if i == 0 else 7.6) / height, 0.83, 2.8 / height))
            ax.tick_params(labelsize=19)
            ax.xaxis.label.set_fontsize(20)
            ax.yaxis.label.set_fontsize(20)
            ax.title.set_fontsize(21)
            for artist in ax.get_legend().get_texts():
                artist.set_fontsize(20)
    else:
        base["DRAWERS"][spec["kind"]](top)
        top.set_ylim(100, 20)
        if spec["kind"] == "pipeline":
            top.set_position((0.04, 6.7 / height, 0.92, (height - 9.5) / height))
        for artist in top.texts:
            artist.set_fontsize(max(artist.get_fontsize(), 20))
        fig.canvas.draw()
        for artist, parent, (x, y, width, box_height) in base["TEXT_BOXES"]:
            columns = int(width / 100 * parent.bbox.width * 144 / (artist.get_fontsize() * fig.dpi) * 0.95)
            artist.set_text(wrapped(artist.get_text(), max(8, columns)))
            BOUNDS.append((artist, parent, x, y, width, box_height))
    ax = axis(fig, (0.04, 1 - 2.25 / height, 0.92, 0.55 / height))
    label(ax, 0, 0, wrapped(case["scene"], 100), height=96, size=20, color=MUTED)
    state_cards(fig, case["states"])
    footer(fig, case["result"], case["limit"])
    return fig


def clean_heading(text):
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[`*_]", "", text).strip()


def navigation_topics(path, text):
    """Prefer real descendant topics over generic index section names."""
    if path.name == "README.md":
        topics = []
        for title, reference in re.findall(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)", text):
            parts = urlsplit(reference)
            if parts.scheme:
                continue
            target = (path.parent / unquote(parts.path)).resolve()
            if target.suffix != ".md" or not target.is_file() or target == path:
                continue
            if not target.is_relative_to(path.parent.resolve()):
                continue
            match = re.search(r"^#\s+(.+)$", target.read_text(), re.M)
            topics.append(clean_heading(match[1] if match else title))
        if topics:
            return list(dict.fromkeys(topics))[:3]
    headings = [clean_heading(t) for t in re.findall(r"^##\s+(.+)$", text, re.M)]
    generic = {
        "目录",
        "文档",
        "Contents",
        "相关文档",
        "参考",
        "References",
        "文件与子目录",
        "直接子目录",
        "文件",
        "子目录",
    }
    headings = [t for t in headings if t not in generic]
    if headings:
        return list(dict.fromkeys(headings))[:3]
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        fields = [clean_heading(t) for t in line.strip("|").split("|")]
        if len(fields) > 1 and (".py" in fields[0] or ".sh" in fields[0]):
            rows.append(fields[0] + "：" + fields[1])
    return rows[:3]


def navigation(spec):
    links = spec["links"]
    rows = (len(links) + 1) // 2
    fig = header(
        spec, "按当前问题选择入口。每张书页列出实际章节，定位主题后到正文点击对应链接。", height=4 + 4.0 * rows
    )
    ax = axis(fig, (0.04, 0.135, 0.92, 0.70))
    ax.set_ylim(rows * 34, 0)
    for i, link in enumerate(links):
        x, y = (i % 2) * 51, (i // 2) * 34
        color = i % 5
        panel(ax, x + 1, y + 1, 47, 30, color)
        ax.add_patch(Rectangle((x + 1, y + 1), 1.6, 30, color=COLORS[color]))
        label(
            ax,
            x + 4,
            y + 3,
            wrapped(f"{i + 1:02d}  " + link["label"], 36),
            width=41,
            height=10,
            size=22,
            color=COLORS[color],
            bold=True,
        )
        path = REPO / link["path"]
        if path.is_dir():
            path = path / "README.md"
        text = path.read_text() if path.exists() else ""
        selected = navigation_topics(path, text)
        if not selected:
            selected = [
                clean_heading(t)
                for t in re.findall(r"\[([^\]]+)\]\([^)]*\.md(?:#[^)]*)?\)", text)
                if t not in {"README.md", "文档总览", "返回上级", "生成方法", "上级导航", "总导航"}
            ][:3]
        cursor = y + 12
        for heading in selected:
            text = wrapped("• " + heading, 42)
            lines = text.count("\n") + 1
            required = lines * 3.6
            if cursor + required > y + 29:
                continue
            label(ax, x + 4, cursor, text, width=40.5, height=required, size=20)
            cursor += required + 0.7
    bottom = axis(fig, (0.04, 0.015, 0.92, 0.10))
    panel(bottom, 0, 0, 100, 90, 1)
    extra = "更多入口在本页完整目录。" if spec.get("extra_links") else "完整路径与适用条件见本页目录。"
    label(
        bottom,
        2,
        12,
        "阅读方法：先选问题 → 看对应章节的场景与图 → 需要核验时再查源码或实验。\n"
        + extra
        + "编号仅用于图中定位，不表示所有文章必须依次阅读。",
        height=74,
        width=96,
        size=19,
    )
    return fig


MEASURED = {
    "measured-flashinfer-preparation": {
        "question": "相同四个评估组合，节省来自少启动命令，还是同一进程复用准备？",
        "states": [
            ["公平比较", ["三组都完成四个评估组合", "图中为两轮平均值", "总耗时单位：秒"]],
            ["逐步减少准备", ["4 命令 / 4 进程：125.55", "1 命令 / 4 进程：102.23", "1 命令 / 1 进程：25.86"]],
            ["主要解释", ["只合并 shell 命令收益有限", "同进程复用准备下降更多", "不是算子本身提速近五倍"]],
        ],
        "result": "复用层次影响收益：容器、命令、Python 进程与模型对象的生命周期要分开分析。",
    },
    "measured-low-load": {
        "question": "工作阶段快了，为什么包含初始化和收尾的短作业仍几乎持平？",
        "states": [
            ["同一两对实验", ["分别统计 OFF 与 ON", "工作阶段是总时间的子区间", "两个范围不能相加"]],
            ["观察差值", ["工作：171.68 → 160.84 s", "驱动：213.96 → 217.15 s", "分别少 10.84、多 3.19 s"]],
            ["正确结论", ["局部节省被其他阶段抵消", "服务复用可能摊薄启动成本", "两对样本不证明稳定加速"]],
        ],
        "result": "评价短作业需要同一完整计时边界。局部加速与整组持平可以同时成立。",
    },
    "measured-mps-boundaries": {
        "question": "MPS 下训练阶段更短，是否等于整个任务也更快？",
        "states": [
            ["共同条件", ["两组 Router 都开启", "每份任务各训练 10 步", "比较 MPS 关闭与开启"]],
            ["两个计时范围", ["训练：22.499 → 20.160 s", "创建至清理：133.150 → 136.010 s", "阶段收益未传递到总时间"]],
            ["证据范围", ["不能当 Router ON/OFF", "不能用训练计时替代总耗时", "并行机制需另看设备证据"]],
        ],
        "result": "MPS 的作用与端到端收益分别验证。更短的计算阶段不足以决定整项任务是否值得。",
    },
    "measured-posttrain-reuse": {
        "question": "保留模型让 B 更快，为什么 A/B/A 全部完成反而不如同进程重载？",
        "states": [
            ["同一请求顺序", ["A → B → A", "比较新进程、重载、常驻", "完整容器时间包含切换"]],
            ["单 B 请求", ["同进程重载：9.58 s", "保留模型：4.36 s", "局部 B 确实更快"]],
            ["完整 A/B/A", ["新进程：274.27 s", "同进程重载：91.96 s", "保留模型：113.84 s"]],
        ],
        "result": "局部请求收益必须放回整组工作。保留更多状态也会产生准备、内存和切换成本。",
    },
    "measured-science-three-steps": {
        "question": "高显存压力下，三步训练的时间差与 OOM、重试有什么关系？",
        "states": [
            ["相同完成目标", ["各完成 3 个策略训练 step", "各接受 12 条轨迹", "版本与压力条件沿用原报告"]],
            ["失败与执行", ["OOM 尝试：18 → 0", "训练合计：1100.80 → 503.69 s", "避免失败有助于完成有效工作"]],
            ["完整边界", ["初始化至清理也计入", "1954.11 → 1415.01 s", "不是本次重新训练的结果"]],
        ],
        "result": "该压力场景下，减少 OOM 与完成三步更快同时出现。历史对照版本不能替换成另一条 baseline。",
    },
}


def measured(spec, base):
    case = MEASURED[spec["id"]]
    fig = header(spec, case["question"])
    base["draw_bars"](fig, spec)
    for i, ax in enumerate(fig.axes[1:]):
        count = len(spec["panels"])
        height = fig.get_figheight()
        ax.set_position(
            (
                0.25,
                ((10.8 if i == 0 else 7.3) if count == 2 else 8.0) / height,
                0.65,
                (2.5 if count == 2 else 4.3) / height,
            )
        )
        for artist in ax.texts:
            artist.set_fontsize(20)
        ax.tick_params(labelsize=19)
        ax.xaxis.label.set_fontsize(20)
        ax.title.set_fontsize(21)
    state_cards(fig, case["states"])
    footer(
        fig,
        case["result"],
        "历史实测重画。本次只升级图解，不新增实验。原始版本、重复次数、工作量与失败证据见对应报告。",
    )
    return fig


def source_image(path):
    if path.suffix == ".svg":
        import cairosvg

        source = Image.open(io.BytesIO(cairosvg.svg2png(url=str(path), output_width=2100))).convert("RGBA")
    else:
        source = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", source.size, "white")
    return Image.alpha_composite(background, source).convert("RGB")


def enlarge_raster_legend(image, regions):
    """Add a readable key while retaining every pixel of the source chart."""
    from matplotlib.font_manager import findfont

    scale = image.width / 900

    def bounds(box):
        return tuple(round(value * scale) for value in box)

    rows = []
    for box in regions["rows"]:
        x1, y1, _, y2 = box
        key = image.crop(bounds((x1, y1, x1 + 22, y2)))
        rows.append(key.resize((round(key.width * 36 / key.height), 36), Image.Resampling.LANCZOS))
    chart = image
    chart = chart.resize((1800, round(chart.height * 1800 / chart.width)), Image.Resampling.LANCZOS)
    combined = Image.new("RGB", (1800, chart.height + 24 + len(rows) * 60), "white")
    combined.paste(chart, (0, 0))
    draw = ImageDraw.Draw(combined)
    font = ImageFont.truetype(findfont("DejaVu Sans"), 36)
    for index, (row, text) in enumerate(zip(rows, regions["labels"], strict=True)):
        y = chart.height + 24 + index * 60
        combined.paste(row, (30, y + 6))
        draw.text((140, y + 24), text, font=font, fill=INK, anchor="lm")
    return combined


def raster_svg_fragment(root, nodes, width=1800):
    """Render selected SVG objects with all shared glyph definitions intact."""
    import cairosvg

    isolated = copy.deepcopy(root)
    group = next(node for node in isolated.iter() if node.get("id") == "figure_1")
    group[:] = copy.deepcopy(nodes)
    identifiers = {node.get("id") for node in isolated.iter() if node.get("id")}
    references = {
        value[1:]
        for node in isolated.iter()
        for attribute, value in node.attrib.items()
        if attribute.rsplit("}", 1)[-1] == "href" and value.startswith("#")
    }
    if missing := references - identifiers:
        raise ValueError(f"Missing SVG definitions: {sorted(missing)!r}.")
    view = [float(value) for value in root.get("viewBox").split()]
    preview = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=ElementTree.tostring(isolated), output_width=1000)))
    bounds = preview.getbbox()
    if bounds is None:
        raise ValueError("The selected SVG objects are empty.")
    x1, y1, x2, y2 = bounds
    unit = view[2] / 1000
    box = [view[0] + x1 * unit - 2, view[1] + y1 * unit - 2, (x2 - x1) * unit + 4, (y2 - y1) * unit + 4]
    isolated.set("viewBox", " ".join(str(value) for value in box))
    isolated.set("width", str(box[2]))
    isolated.set("height", str(box[3]))
    rendered = Image.open(
        io.BytesIO(cairosvg.svg2png(bytestring=ElementTree.tostring(isolated), output_width=width))
    ).convert("RGBA")
    white = Image.new("RGBA", rendered.size, "white")
    return Image.alpha_composite(white, rendered).convert("RGB")


def svg_panels(path):
    """Reflow existing SVG axes without changing data, scales or legend content."""
    parser = ElementTree.XMLParser(target=ElementTree.TreeBuilder(insert_comments=True))
    root = ElementTree.fromstring(path.read_bytes(), parser=parser)
    figure = next((node for node in root.iter() if node.get("id") == "figure_1"), None)
    if figure is None:
        return None
    axes = [node for node in figure if node.get("id", "").startswith("axes_")]
    legends = [node for node in figure.iter() if node.get("id", "").startswith("legend_")]
    if not axes or (len(axes) < 2 and not legends):
        return None
    # Matplotlib puts reusable glyph definitions beside their first text use.
    # Later axes can reference them. Retain every definition before isolating axes.
    definitions = ElementTree.Element(SVG_NS + "defs")
    for node in root.iter(SVG_NS + "defs"):
        definitions.extend(copy.deepcopy(list(node)))
    for parent in root.iter():
        for child in list(parent):
            if child.tag == SVG_NS + "defs":
                parent.remove(child)
    root.insert(0, definitions)
    notes = []
    for node in figure:
        if node.get("id", "").startswith("text_"):
            texts = [
                part.text for part in node.iter() if part.tag == SVG_NS + "text" or part.tag is ElementTree.Comment
            ]
            if texts:
                notes.append(" ".join(text.strip() for text in texts if text and text.strip()))
    panels = []
    for selected in axes:
        content = copy.deepcopy(selected)
        legend_rows = []
        for parent in content.iter():
            for legend in list(parent):
                if not legend.get("id", "").startswith("legend_"):
                    continue
                entry = []
                for index, node in enumerate(legend):
                    is_background = any(
                        "fill: #ffffff" in part.get("style", "") and "stroke-linejoin: miter" in part.get("style", "")
                        for part in node.iter()
                    )
                    if index == 0 and is_background:
                        continue
                    entry.append(node)
                    if node.get("id", "").startswith("text_"):
                        row = raster_svg_fragment(root, entry)
                        row_height = 48
                        row_width = round(row.width * row_height / row.height)
                        if row_width > 1740:
                            row_height = round(row_height * 1740 / row_width)
                            row_width = 1740
                        legend_rows.append(row.resize((row_width, row_height), Image.Resampling.LANCZOS))
                        entry = []
                parent.remove(legend)
        chart = raster_svg_fragment(root, [content])
        if legend_rows:
            combined = Image.new("RGB", (1800, chart.height + 24 + 60 * len(legend_rows)), "white")
            combined.paste(chart, (0, 0))
            for index, row in enumerate(legend_rows):
                combined.paste(row, (30, chart.height + 24 + index * 60))
            chart = combined
        panels.append(chart)
    return panels, notes


def literature_image(path):
    """Re-render existing teaching objects at a common readable text size."""
    namespace = runpy.run_path(str(path.parent / "render.py"))
    captured = []
    namespace[LITERATURE_DRAWERS[path.stem]].__globals__["save"] = lambda fig, name: captured.append(fig)
    namespace[LITERATURE_DRAWERS[path.stem]]()
    fig = captured[0]
    if path.stem == "roofline-and-batching":
        # Keep the decode plot's twin throughput scale on the same panel.
        fig.set_size_inches(12, 15)
        left, right, throughput = fig.axes
        left.set_position((0.12, 0.56, 0.72, 0.29))
        right.set_position((0.12, 0.17, 0.72, 0.28))
        throughput.set_position(right.get_position())
        for ax in fig.axes:
            ax.tick_params(labelsize=17)
            ax.title.set_fontsize(20)
            ax.xaxis.label.set_fontsize(18)
            ax.yaxis.label.set_fontsize(18)
            for artist in ax.texts:
                artist.set_fontsize(18)
                if artist.get_text() == "算力限制：100":
                    artist.set_position((70, 65))
        for artist in right.get_legend().get_texts():
            artist.set_fontsize(17)
        for index, artist in enumerate(fig.texts):
            artist.set_fontsize(24 if index == 0 else 17)
            artist.set_text(wrapped(artist.get_text(), 68 if index == 0 else 100))
            if index == 1:
                artist.set_y(0.91)
    else:
        enlarge_literature_objects(fig)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def enlarge_literature_objects(fig):
    """Reflow original teaching diagrams without changing their objects or data."""
    fig.set_figheight(fig.get_figheight() * 1.5)
    fig.canvas.draw()
    for ax in fig.axes:
        for artist in ax.texts:
            size = 24 if artist.get_fontsize() >= 20 else 18
            artist.set_fontsize(size)
            artist.set_linespacing(1.3)
            x, y = artist.get_position()
            boxes = [
                patch
                for patch in ax.patches
                if isinstance(patch, FancyBboxPatch)
                and patch.get_x() <= x <= patch.get_x() + patch.get_width()
                and patch.get_y() <= y <= patch.get_y() + patch.get_height()
            ]
            if boxes:
                width = min(patch.get_width() for patch in boxes) * 0.92
            elif artist.get_ha() == "center":
                width = 2 * min(x - 1, 99 - x)
            elif artist.get_ha() == "right":
                width = x - 2
            else:
                width = 98 - x
            columns = max(8, int(width / 100 * ax.bbox.width * 144 / (size * fig.dpi) * 0.95))
            artist.set_text(wrapped(artist.get_text(), columns))


def evidence_panels(spec):
    """Keep immutable originals and cache only derived display panels locally."""
    path = REPO / spec["original"]
    vector = path.with_suffix(".svg")
    generator = path.parent / "render.py"
    literature = path.parent.name == "literature-reviews" and path.stem in LITERATURE_DRAWERS
    source = path.read_bytes() + (vector.read_bytes() if vector.exists() else b"")
    source += generator.read_bytes() if literature else b""
    revision = b"typography-5" if vector.exists() and not literature else b"typography-3"
    if path.stem == "roofline-and-batching":
        revision = b"typography-6"
    if path.name in RASTER_LEGENDS:
        revision = b"typography-5"
    key = hashlib.sha256(revision + source)
    key = key.hexdigest()[:20]
    cache = Path("/tmp/psrl-figure-panels") / key
    meta = cache / "panels.json"
    if meta.exists():
        data = json.loads(meta.read_text())
        return [Image.open(cache / name).convert("RGB") for name in data["files"]], data["notes"]
    result = ([literature_image(path)], []) if literature else (svg_panels(vector) if vector.exists() else None)
    if result is None:
        image = source_image(path)
        if path.name in RASTER_LEGENDS:
            image = enlarge_raster_legend(image, RASTER_LEGENDS[path.name])
        if path.name in RASTER_GRIDS:
            rows, cols, top = RASTER_GRIDS[path.name]
            panels = []
            for row in range(rows):
                for col in range(cols):
                    bounds = (
                        round(image.width * col / cols),
                        round(image.height * (top + (1 - top) * row / rows)),
                        round(image.width * (col + 1) / cols),
                        round(image.height * (top + (1 - top) * (row + 1) / rows)),
                    )
                    cropped = image.crop(bounds)
                    ink = ImageChops.difference(cropped, Image.new("RGB", cropped.size, "white")).getbbox()
                    panels.append(cropped.crop(ink) if ink else cropped)
            result = panels, [spec["explanation"]["scope"]]
        else:
            result = [image], []
    cache.mkdir(parents=True, exist_ok=True)
    panels, notes = result
    files = []
    for index, panel_image in enumerate(panels):
        name = f"panel-{index + 1}.png"
        panel_image.save(cache / name)
        files.append(name)
    meta.write_text(json.dumps({"files": files, "notes": notes}, ensure_ascii=False))
    return panels, notes


def evidence(spec):
    case = spec["explanation"]
    images, notes = evidence_panels(spec)
    panel_heights = [max(3.5, 14.7 * image.height / image.width) for image in images]
    note_text = wrapped("\n".join(notes), 110)
    notes_height = (note_text.count("\n") + 1) * 0.32 + 0.2 if note_text else 0
    height = sum(panel_heights) + 9.4 + 0.35 * (len(images) - 1) + notes_height
    fig = header(spec, case["question"], height=height)
    title = axis(fig, (0.04, 1 - 2.3 / height, 0.92, 0.55 / height))
    label(title, 0, 0, wrapped(case["scope"], 106), height=98, size=19, color=MUTED)
    cursor = height - 2.55
    for image, panel_height in zip(images, panel_heights, strict=True):
        ax = fig.add_axes((0.04, (cursor - panel_height) / height, 0.92, panel_height / height))
        ax.imshow(image)
        ax.axis("off")
        cursor -= panel_height + 0.35
    if note_text:
        notes_ax = axis(fig, (0.04, 6.6 / height, 0.92, notes_height / height))
        label(notes_ax, 0, 0, note_text, height=98, size=18, color=MUTED)
    state_cards(fig, case["states"])
    footer(fig, case["result"], case["limit"])
    return fig


def verify_layout(fig, identifier):
    """Reject wrapped context labels that exceed their allotted readable area."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    failures = []
    for artist, ax, x, y, width, height in BOUNDS:
        p1, p2 = ax.transData.transform([(x, y), (x + width, y + height)])
        extent = artist.get_window_extent(renderer)
        if extent.width > abs(p2[0] - p1[0]) * 1.035 or extent.height > abs(p2[1] - p1[1]) * 1.035:
            failures.append(artist.get_text())
    if failures:
        raise ValueError(f"Crowded labels in {identifier!r}: {failures!r}.")


def draw(spec, base):
    if spec["kind"] == "navigation":
        fig = navigation(spec)
    elif spec["kind"] == "bars":
        fig = measured(spec, base)
    elif spec["kind"] == "evidence_explanation":
        fig = evidence(spec)
    else:
        fig = mechanism(spec, base)
    verify_layout(fig, spec["id"])
    return fig
