"""Render screen previews while retaining full explanatory figures and sources."""

import argparse
import base64
import copy
import hashlib
import html
import io
import json
import math
import re
import runpy
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
WIDTH = 800
HEIGHT = 600
MARGIN = 16
FONT = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
NS = "{http://www.w3.org/2000/svg}"


def mode_overview(output_dir):
    """Reflow the mode comparison instead of shrinking a four-panel portrait."""
    chapter = runpy.run_path(str(ROOT / "chapters.py"))
    scene = chapter["Scene"]
    blue, green, orange, purple = (chapter[key] for key in ("BLUE", "GREEN", "ORANGE", "PURPLE"))
    drawing = scene(
        "episode 与 command：空档时，GPU 预算留不留？",
        "教学假设：同一会话 A 计算 20 秒、思考 10 秒、再计算 10 秒；B 正在等 GPU。",
        540,
    )
    drawing.timeline(
        127,
        40,
        [
            ("A 的活动", [(0, 20, "GPU 20 秒", blue), (20, 30, "思考 10 秒", orange), (30, 40, "GPU 10 秒", blue)]),
            ("episode", [(0, 40, "GPU 预算全程留给 A", blue)]),
            (
                "command",
                [(0, 20, "A 申请并使用", green), (20, 30, "归还后可借 B", orange), (30, 40, "A 重新申请", green)],
            ),
        ],
        [0, 20, 30, 40],
        step=61,
    )
    drawing.text(32, 389, "两种模式都保留容器、文件、CPU 和主存；保留预算不会自动保住已退出的模型进程。", 18)
    drawing.text(32, 425, "command 必须先确认物理归还；B 若没结束，A 回来仍需等待，图未计交接成本。", 18)
    drawing.footer("auto 在会话准入时按可信能力与可选成本建议选择一种模式，租约内保持固定。", purple)
    drawing.save("mode-preview", output_dir)


def resource_overview(output_dir):
    """Show the host boundary and GPU candidates in a short resource layout."""
    chapter = runpy.run_path(str(ROOT / "chapters.py"))
    blue, green, orange, purple = (chapter[key] for key in ("BLUE", "GREEN", "ORANGE", "PURPLE"))
    drawing = chapter["Scene"](
        "Env 现状：会话留在主机，GPU 按持有模式调度",
        "已启用 Env Service 的结构示意。候选资源来自配置，推理共享卡还需要显式绑定。",
        560,
    )
    for x, title, detail, color in [
        (32, "Agent / Harbor", "提出环境与命令需求", purple),
        (322, "EnvRouter / 账本", "检查预算，批准或等待", blue),
        (612, "容器执行层", "执行、评分与物理清理", green),
    ]:
        drawing.box(x, 119, 256, 76, color)
        drawing.text(x + 16, 149, title, 21, color)
        drawing.text(x + 16, 178, detail, 17)
    drawing.line(292, 157, 316, 157, arrow=True)
    drawing.line(582, 157, 606, 157, arrow=True)
    drawing.box(32, 218, 384, 193, purple)
    drawing.text(48, 248, "会话 A 的主机资源：留在节点 N", 20, purple)
    drawing.box(48, 269, 136, 113, purple, fill="white")
    drawing.text(64, 299, "持久容器", 19, purple)
    for i, label in enumerate(["代码与文件", "CPU 进程状态"]):
        drawing.text(64, 331 + 27 * i, label, 16)
    drawing.text(208, 289, "CPU 配额", 17)
    for i in range(4):
        drawing.box(208 + i * 42, 302, 32, 27, purple, radius=2)
    drawing.text(208, 354, "主存配额", 17)
    for i in range(8):
        drawing.box(208 + i * 21, 365, 18, 17, purple, radius=0)
    drawing.line(424, 315, 472, 315, blue, arrow=True)
    drawing.box(484, 218, 384, 193, blue)
    drawing.text(500, 248, "同机配置的 GPU 候选池", 20, blue)
    drawing.text(500, 283, "独立卡优先", 18, green)
    drawing.memory(500, 296, 344, [16, 8], ["环境可用预算", "已占用"], [green, orange], 24, 32)
    drawing.text(500, 355, "独立卡紧张，再检查已绑定推理卡", 18, blue)
    drawing.memory(500, 366, 344, [14, 4, 6], ["vLLM 常驻", "余量", "保留"], [blue, green, orange], 24, 32)
    drawing.text(32, 447, "episode：GPU 预算随会话持有。command：逐命令申请，确认归还后释放。", 19)
    drawing.footer("共享需要保留推理余量；不自动触发模型睡眠。同机换卡不搬容器，不透明迁移活模型。")
    drawing.save("mode-preview", output_dir)


def render_recap(spec):
    """Write a bounded raster preview and a matching editable vector."""
    preview = spec["screen_preview"]
    with tempfile.TemporaryDirectory(prefix="psrl-screen-figure-") as temporary:
        directory = Path(temporary)
        if preview["layout"] in ("mode_overview", "resource_overview"):
            {"mode_overview": mode_overview, "resource_overview": resource_overview}[preview["layout"]](directory)
            png, svg = directory / "mode-preview.png", directory / "mode-preview.svg"
        else:
            png, svg = (ROOT / (spec["id"] + suffix) for suffix in (".png", ".svg"))
        with Image.open(png) as original:
            height = round(original.height * WIDTH / original.width)
            if height > 600:
                raise ValueError(f"Reflow the preview instead of shrinking a tall figure: {spec['id']!r}.")
            original.convert("RGB").resize((WIDTH, height), Image.Resampling.LANCZOS).save(
                ROOT / preview["png"], optimize=True
            )
        vector = svg.read_text()
        root = ElementTree.fromstring(vector)
        start = vector.index("<svg ")
        end = vector.index(">", start)
        header = vector[start:end]
        header = header.replace(f'width="{root.attrib["width"]}"', f'width="{WIDTH}"')
        header = header.replace(f'height="{root.attrib["height"]}"', f'height="{height}"')
        (ROOT / preview["svg"]).write_text(vector[:start] + header + vector[end:])


def wrapped(value, font, width):
    """Wrap labels by their measured display width without reducing the font."""
    lines = []
    for paragraph in str(value).split("\n"):
        line = ""
        for word in re.findall(r"[A-Za-z0-9_.+/<>=−×]+|.", paragraph):
            if line and font.getlength(line + word) > width:
                lines.append(line)
                line = ""
            line += word
        lines.append(line)
    return lines


def text_block(values, color="#203149", fill="white", size=16):
    """Reflow existing captions and limits at a fixed readable screen size."""
    font = ImageFont.truetype(str(FONT), size)
    lines = [line for value in values for line in wrapped(value, font, WIDTH - 4 * MARGIN)]
    height = 20 + len(lines) * (size + 7)
    image = Image.new("RGB", (WIDTH - 2 * MARGIN, height), fill)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((MARGIN, 9 + index * (size + 7)), line, font=font, fill=color)
    return image


def ink_bounds(image):
    """Ignore antialiasing noise when locating the visible source content."""
    difference = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white"))
    return difference.convert("L").point(lambda value: 255 if value > 3 else 0).getbbox()


def compact_white_rows(image):
    """Remove only blank vertical spacing while retaining every nonwhite row."""
    pixels = np.asarray(image.convert("RGB"))
    occupied = np.any(pixels < 250, axis=(1, 2))
    runs = []
    start = 0
    while start < image.height:
        end = start + 1
        while end < image.height and occupied[end] == occupied[start]:
            end += 1
        if occupied[start] or end - start <= 18:
            runs.append((start, end))
        else:
            runs.append((start, start + 9))
            runs.append((end - 9, end))
        start = end
    output = Image.new("RGB", (image.width, sum(end - start for start, end in runs)), "white")
    top = 0
    for start, end in runs:
        output.paste(image.crop((0, start, image.width, end)), (0, top))
        top += end - start
    return output, runs


def split_at_gutters(image, limit):
    """Split on blank gutters so text, curves and resource objects remain whole."""
    pixels = np.asarray(image.convert("RGB"))
    occupied = np.any(pixels < 250, axis=(1, 2))
    candidates = [0]
    for row in range(1, image.height):
        if not occupied[row] and occupied[row - 1]:
            end = row
            while end < image.height and not occupied[end]:
                end += 1
            candidates.append((row + end) // 2)
    candidates.append(image.height)
    slices = []
    start = 0
    while image.height - start > limit:
        options = [row for row in candidates if start + 80 <= row <= start + limit]
        if not options:
            raise ValueError(f"An intact panel exceeds the {limit!r}-pixel content height.")
        end = max(options)
        slices.append((start, end))
        start = end
    slices.append((start, image.height))
    return [image.crop((0, start, image.width, end)) for start, end in slices], slices


def legend_counts(spec):
    """Match the original chart axes to their separately enlarged legends."""
    if not spec.get("original"):
        return []
    path = ROOT.parents[3] / spec["original"]
    path = path.with_suffix(".svg")
    if not path.exists():
        return []
    root = ElementTree.parse(path).getroot()
    figure = next((node for node in root.iter() if node.get("id") == "figure_1"), None)
    if figure is None:
        return []
    counts = []
    for axis in figure:
        if axis.get("id", "").startswith("axes_"):
            counts.append(
                sum(
                    child.get("id", "").startswith("text_")
                    for legend in axis.iter()
                    if legend.get("id", "").startswith("legend_")
                    for child in legend
                )
            )
    return counts


def fit_chart(image, limit, legends):
    """Fit a large chart while preserving the independently enlarged legend."""
    boundary = image.height
    if legends:
        occupied = np.any(np.asarray(image.convert("RGB")) < 250, axis=(1, 2))
        target = image.height - 26 * legends
        options = [row for row in range(max(0, target - 45), min(image.height, target + 30)) if not occupied[row]]
        if options:
            boundary = min(options, key=lambda row: abs(row - target))
    tail = image.crop((0, boundary, image.width, image.height))
    available = limit - tail.height - (12 if tail.height else 0)
    if available < 200:
        raise ValueError("The chart needs its own manual screen layout.")
    chart = image.crop((0, 0, image.width, boundary))
    ratio = min(1, available / chart.height)
    chart = chart.resize((round(chart.width * ratio), round(chart.height * ratio)), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (image.width, chart.height + tail.height + (12 if tail.height else 0)), "white")
    output.paste(chart, ((image.width - chart.width) // 2, 0))
    if tail.height:
        output.paste(tail, (0, chart.height + 12))
    return output, {
        "chart_scale": ratio,
        "legend_kept_at_original_display_scale": bool(tail.height),
        "legend_boundary": boundary,
    }


def svg_axis_bounds(root, selected, png_size):
    """Locate a semantic SVG panel, then display pixels from the original PNG."""
    import cairosvg

    isolated = copy.deepcopy(root)
    figure = next(node for node in isolated.iter() if node.get("id") == "figure_1")
    definitions = ElementTree.Element(NS + "defs")
    for node in root.iter(NS + "defs"):
        definitions.extend(copy.deepcopy(list(node)))
    figure[:] = [copy.deepcopy(selected)]
    isolated.insert(0, definitions)
    mask = Image.open(io.BytesIO(cairosvg.svg2png(bytestring=ElementTree.tostring(isolated), output_width=500)))
    box = mask.getbbox()
    if box is None:
        raise ValueError("An SVG panel has no visible content.")
    scale = png_size[0] / 500
    return (
        max(0, int(box[0] * scale) - 8),
        max(0, int(box[1] * scale) - 8),
        min(png_size[0], round(box[2] * scale) + 8),
        min(png_size[1], round(box[3] * scale) + 8),
    )


def bar_panel(panel):
    """Replot registered values with room for labels and one unambiguous axis."""
    width = WIDTH - 2 * MARGIN
    font = ImageFont.truetype(str(FONT), 18)
    small = ImageFont.truetype(str(FONT), 16)
    labels, values = panel["labels"], panel["values"]
    height = 128 + 76 * len(values)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), panel["title"], font=font, fill="#203149")
    left = math.ceil(max(font.getlength(label) for label in labels)) + 24
    right, baseline = width - 80, height - 65
    magnitude = 10 ** math.floor(math.log10(max(values)))
    step = math.ceil(max(values) / magnitude * 2) * magnitude / 10
    upper = step * 5
    for index in range(6):
        value = step * index
        x = left + (right - left) * value / upper
        draw.line((x, 50, x, baseline), fill="#e8edf3")
        draw.text((x, baseline + 8), f"{value:g}", font=small, fill="#203149", anchor="mt")
    palette = ["#337bc4", "#149c8e", "#e79a25", "#8965c5"]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 64 + 76 * index
        color = palette[panel.get("colors", list(range(len(values))))[index]]
        draw.text((left - 12, y + 21), label, font=font, fill="#203149", anchor="rm")
        end = left + (right - left) * value / upper
        draw.rectangle((left, y, end, y + 42), fill=color)
        draw.text((end + 8, y + 21), f"{value:g}", font=font, fill="#203149", anchor="lm")
    draw.line((left, baseline, right, baseline), fill="#53677c")
    draw.text(((left + right) / 2, height - 18), "秒", font=small, fill="#203149", anchor="mm")
    return image


def token_weight_panel():
    """Keep the original eight-token example and place weights inside the page."""
    image = Image.new("RGB", (WIDTH - 2 * MARGIN, 326), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), 18)
    for group, (title, weights) in enumerate(
        [("先按轨迹平均：每条各占一半", ["1/2", "1/2"]), ("按全部 token 平均：共 8 个有效 token", ["2/8", "6/8"])]
    ):
        top = group * 163
        draw.text((16, top + 8), title, font=font, fill="#203149")
        for row, (count, color) in enumerate([(2, "#337bc4"), (6, "#149c8e")]):
            y = top + 50 + row * 49
            for index in range(count):
                draw.rectangle((16 + index * 54, y, 60 + index * 54, y + 30), fill=color)
            draw.text((390, y + 15), f"该回答权重：{weights[row]}", font=font, fill="#203149", anchor="lm")
    return image


def fixed_work_panel(spec, index):
    """Replot existing audit values rather than crop overlapping raster axes."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = (ROOT.parents[3] / spec["original"]).parent / "fixed-work-comparison.json"
    results = json.loads(path.read_text())
    colors = {"off": "#cf7735", "on": "#187f9c"}
    with plt.rc_context({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13}):
        figure, axis = plt.subplots(figsize=(7.68, 4.8), dpi=100, layout="constrained")
        if spec["id"] == "explained-fixed-work-outcomes-c85fca2" and index == 0:
            for mode, result in results.items():
                start = result["first_slot_started_at"]
                accepted = [record for record in result["records"] if record["accepted"]]
                axis.step(
                    [0] + [(record["finished_at"] - start) / 60 for record in accepted],
                    [0] + list(range(1, len(accepted) + 1)),
                    where="post",
                    color=colors[mode],
                    label=result["label"],
                )
            axis.axhline(16, color="gray", linestyle="--", linewidth=1)
            axis.set(
                title="Accepted rollout progress (OOM excluded)",
                xlabel="Minutes since first slot request",
                ylabel="Accepted rollouts",
                ylim=(0, 17),
            )
            axis.legend(fontsize=10, loc="upper left")
        else:
            if spec["id"] == "explained-fixed-work-outcomes-c85fca2":
                key, title, ylabel = [
                    ("cuda_oom_attempts", "OOM attempts, all counted", "Attempts"),
                    (
                        "sampling_seconds_including_retries",
                        "Time to all 16 accepted rollouts",
                        "Seconds incl. OOM retries",
                    ),
                    ("logged_trained_tokens_sum", "Mask-valid training tokens", "Tokens"),
                    ("accepted_mean_reward_repair", "Original repair reward", "Mean reward"),
                    ("wall_seconds", "Full arm wall time", "Seconds incl. init and cleanup"),
                ][index - 1]
                values = [results[mode][key] for mode in ("off", "on")]
                labels = [f"{value:.2f}" if isinstance(value, float) else str(value) for value in values]
            else:
                key, title = [
                    ("actor/entropy", "Entropy"),
                    ("actor/loss", "Policy loss"),
                    ("actor/grad_norm", "Gradient norm"),
                ][index]
                ylabel = ""
                values = [results[mode]["steps"][-1][key] for mode in ("off", "on")]
                labels = [f"{value:.5g}" for value in values]
            axis.bar([0, 1], values, color=[colors[mode] for mode in ("off", "on")])
            for position, (value, label) in enumerate(zip(values, labels, strict=True)):
                axis.annotate(label, (position, value), ha="center", xytext=(0, 5), textcoords="offset points")
            axis.set(title=title, ylabel=ylabel, xticks=[0, 1], xticklabels=["0eeead8", "Router ON"])
            axis.margins(y=0.25)
        axis.spines[["right", "top"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def source_blocks(spec):
    """Preserve source objects and data while reflowing surrounding text."""
    source = Image.open(ROOT / (spec["id"] + ".png")).convert("RGB")
    root = ElementTree.parse(ROOT / (spec["id"] + ".svg")).getroot()
    figure = next((node for node in root.iter() if node.get("id") == "figure_1"), None)
    records = []
    blocks = []
    if figure is None:
        if spec["id"] == "chapter-replica-backlog":
            draw = ImageDraw.Draw(source)
            scale = source.width / float(root.get("viewBox").split()[2])
            for node in root.iter(NS + "text"):
                if "−" not in (node.text or ""):
                    continue
                size = float(re.search(r"font-size:([\d.]+)px", node.get("style"))[1]) * scale
                font = ImageFont.truetype(str(FONT), round(size))
                label = node.text.replace("−", "-")
                x, y = float(node.get("x")) * scale, float(node.get("y")) * scale
                draw.rectangle((x, y - size * 1.4, x + font.getlength(label) + size, y + size * 0.3), fill="white")
                draw.text((x, y), label, font=font, fill=node.get("fill"), anchor="ls")
        bounds = ink_bounds(source)
        image = source.crop(bounds)
        width = WIDTH - 2 * MARGIN
        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        image, rows = compact_white_rows(image)
        return [(image, "diagram")], [{"source_box": bounds, "blank_row_compaction": rows}]
    axes = [node for node in figure if node.get("id", "").startswith("axes_")]
    located = [(svg_axis_bounds(root, axis, source.size), axis) for axis in axes]
    located.sort(key=lambda entry: entry[0][1])
    original_legends = legend_counts(spec)
    image_index = 0
    bar_index = 0
    for bounds, axis in located:
        texts = ["".join(node.itertext()).strip() for node in axis.iter(NS + "text")]
        if spec["title"] in texts or (spec["kind"] != "explanation" and axis.get("id") == "axes_1"):
            records.append({"axis": axis.get("id"), "role": "header_reflowed", "source_box": bounds})
            continue
        height = (bounds[3] - bounds[1]) * 800 / source.width
        if texts and height < 130 and len(list(axis.iter(NS + "path"))) <= 1 and not list(axis.iter(NS + "image")):
            footer = any("18px" in node.get("style", "") for node in axis.iter(NS + "text"))
            image = text_block(texts, fill="#e4f5f0" if footer else "#f3f6fa", size=14 if footer else 16)
            blocks.append((image, "caption"))
            records.append({"axis": axis.get("id"), "role": "caption_reflowed", "source_box": bounds, "text": texts})
            continue
        if spec["kind"] == "bars" and bar_index < len(spec["panels"]):
            panel = spec["panels"][bar_index]
            blocks.append((bar_panel(panel), "diagram"))
            records.append({"axis": axis.get("id"), "role": "source_panel", "registered_data_replotted": panel})
            bar_index += 1
            continue
        if spec["id"] == "token-loss-weighting" and axis.get("id") == "axes_2":
            blocks.append((token_weight_panel(), "diagram"))
            records.append({"axis": axis.get("id"), "role": "source_panel", "layout": "eight_tokens_weights_reflowed"})
            continue
        separate_raw_axes = spec["id"] in {
            "explained-fixed-work-outcomes-c85fca2",
            "explained-fixed-work-training-4f78708",
        } and list(axis.iter(NS + "image"))
        image = fixed_work_panel(spec, image_index) if separate_raw_axes else source.crop(bounds)
        if ink := ink_bounds(image):
            image = image.crop(ink)
        width = WIDTH - 2 * MARGIN
        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        image, rows = compact_white_rows(image)
        blocks.append((image, "diagram"))
        legends = 0
        if list(axis.iter(NS + "image")):
            legends = original_legends[image_index] if image_index < len(original_legends) else 0
            image_index += 1
        records.append(
            {
                "axis": axis.get("id"),
                "role": "source_panel",
                "source_box": bounds,
                "blank_row_compaction": rows,
                "legend_rows": legends,
                "audit_values_replotted": bool(separate_raw_axes),
            }
        )
    for node in figure:
        if node.get("id", "").startswith("text_"):
            texts = ["".join(part.itertext()).strip() for part in node.iter(NS + "text")]
            values = [value for value in texts if value not in {spec["title"], spec.get("reading_question", "")}]
            if values:
                blocks.append((text_block(values, size=14), "caption"))
                records.append({"role": "figure_note_reflowed", "text": values})
    return blocks, records


def header(spec, index=None, count=None):
    """Repeat the topic and reading position above each screen panel."""
    font = ImageFont.truetype(str(FONT), 18)
    lines = wrapped(spec["title"], font, WIDTH - 4 * MARGIN)
    image = Image.new("RGB", (WIDTH - 2 * MARGIN, len(lines) * 25 + 28), "#eaf2ff")
    draw = ImageDraw.Draw(image)
    for row, value in enumerate(lines):
        draw.text((12, 3 + row * 25), value, font=font, fill="#203149")
    badge = f"第 {index} / {count} 张 · 从上到下阅读 · 点击可看完整图"
    draw.text((12, len(lines) * 25 + 5), badge, font=ImageFont.truetype(str(FONT), 14), fill="#53677c")
    return image


def write_page(spec, image, index):
    """Keep previews bounded and reference the unchanged editable source."""
    suffix = "" if index == 1 else f"-{index:02d}"
    name = spec["id"] + "-screen" + suffix
    if image.width != WIDTH or image.height > HEIGHT:
        raise ValueError(f"Screen page exceeds the display budget: {name!r}.")
    image.save(ROOT / (name + ".png"), optimize=True)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    (ROOT / (name + ".svg")).write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{image.height}" '
        f'viewBox="0 0 {WIDTH} {image.height}"><title>{html.escape(spec["title"])}</title>'
        f"<metadata>Raster screen layout. Editable original: {spec['id']}.svg. Generator: screen.py.</metadata>"
        f'<image width="{WIDTH}" height="{image.height}" href="data:image/png;base64,{encoded}"/></svg>'
    )
    return {"png": name + ".png", "svg": name + ".svg", "width": WIDTH, "height": image.height}


def render_panels(spec):
    """Pack intact source panels into a numbered sequence of screen pages."""
    blocks, source_records = source_blocks(spec)
    if spec["kind"] == "chapter" and len(blocks) == 1 and blocks[0][0].height <= HEIGHT - 2 * MARGIN:
        image = blocks[0][0]
        output = Image.new("RGB", (WIDTH, image.height + 2 * MARGIN), "white")
        output.paste(image, (MARGIN, MARGIN))
        return [write_page(spec, output, 1)], source_records
    heading = header(spec, 1, 1)
    limit = HEIGHT - heading.height - 2 * MARGIN - 12
    pieces = []
    question = spec.get("reading_question") or spec.get("explanation", {}).get("question")
    if question:
        pieces.append(text_block([question], size=16, fill="#f3f6fa"))
    panel_records = iter(record for record in source_records if record["role"] == "source_panel")
    for image, role in blocks:
        panel = next(panel_records, {}) if role == "diagram" else {}
        separate_teaching_axes = spec["id"] == "explained-roofline-and-batching-1ddd7ab"
        if (
            spec["kind"] == "evidence_explanation"
            and role == "diagram"
            and image.height > limit
            and not separate_teaching_axes
        ):
            image, fit = fit_chart(image, limit, panel.get("legend_rows", 0))
            panel["screen_chart_fit"] = fit
        try:
            parts, slices = split_at_gutters(image, limit)
        except ValueError:
            if spec["kind"] != "evidence_explanation":
                raise ValueError(f"{spec['id']!r}: {role!r} panel {image.size!r} needs a manual layout.") from None
            image, fit = fit_chart(image, limit, panel.get("legend_rows", 0))
            panel["screen_chart_fit"] = fit
            parts, slices = [image], [(0, image.height)]
        pieces.extend(parts)
        source_records.append({"role": role, "display_slices": slices, "resized_width": image.width})
    packed = []
    current = []
    height = 0
    for image in pieces:
        if current and height + image.height + 12 > limit:
            packed.append(current)
            current, height = [], 0
        current.append(image)
        height += image.height + 12
    if current:
        packed.append(current)
    pages = []
    for index, group in enumerate(packed, 1):
        top = header(spec, index, len(packed))
        height = MARGIN * 2 + top.height + sum(image.height + 12 for image in group)
        output = Image.new("RGB", (WIDTH, height), "white")
        output.paste(top, (MARGIN, MARGIN))
        cursor = MARGIN + top.height
        for image in group:
            output.paste(image, (MARGIN, cursor))
            cursor += image.height + 12
        pages.append(write_page(spec, output, index))
    return pages, source_records


def render(spec):
    """Render every registered screen layout and retain a traceable page list."""
    preview = spec["screen_preview"]
    if preview["layout"] != "panels":
        render_recap(spec)
        with Image.open(ROOT / preview["png"]) as image:
            pages = [{"png": preview["png"], "svg": preview["svg"], "width": image.width, "height": image.height}]
        records = [{"role": preview["layout"]}]
    else:
        pages, records = render_panels(spec)
    preview["pages"] = pages
    preview["source_sha256"] = {
        suffix: hashlib.sha256((ROOT / (spec["id"] + "." + suffix)).read_bytes()).hexdigest()
        for suffix in ("png", "svg")
    }
    return {"id": spec["id"], "pages": pages, "source_records": records, "source_sha256": preview["source_sha256"]}


def persist(catalog, records):
    """Save screen registrations and a searchable gallery alongside the source."""
    (ROOT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
    record_path = ROOT / "screen-layout.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else {"figures": []}
    replaced = {record["id"] for record in records}
    previous["figures"] = [record for record in previous["figures"] if record["id"] not in replaced] + records
    record_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + "\n")
    sections = []
    for spec in catalog["figures"]:
        pages = spec.get("screen_preview", {}).get("pages", [])
        panels = "".join(
            f'<a href="{spec["id"]}.png" title="查看完整图"><img loading="lazy" '
            f'src="{page["png"]}" width="{page["width"]}" height="{page["height"]}" '
            f'alt="{html.escape(spec["title"])}，第{index}张"></a>'
            for index, page in enumerate(pages, 1)
        )
        sections.append(
            f'<details open id="{spec["id"]}"><summary>{html.escape(spec["title"])}'
            f" <small>共{len(pages)}张</small></summary>{panels}</details>"
        )
    document = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>文档配图：屏幕版图集</title>'
        "<style>body{font:16px/1.6 sans-serif;color:#203149;margin:24px auto;padding:0 16px;max-width:820px;"
        "background:#f4f7fb}h1{font-size:26px}input{box-sizing:border-box;width:100%;padding:12px;font-size:16px;"
        "border:1px solid #9eafc3;border-radius:8px}details{margin:24px 0;background:white;padding:10px;"
        "border:1px solid #d8e2ed;border-radius:10px}summary{cursor:pointer;font-size:18px;color:#276dc5;"
        "padding:8px}small{font-size:14px;color:#53677c}img{display:block;max-width:100%;height:auto;"
        "margin:12px auto}a{display:block}details[hidden]{display:none}</style>"
        "<h1>文档配图：屏幕版图集</h1><p>每张不超过800×600像素。多张分图按编号阅读，点击图片可看完整图。"
        '原始图、证据与可编辑来源均保留。</p><input id="search" placeholder="按主题筛选，例如：显存、Roofline、训练">'
        + "".join(sections)
        + '<script>document.getElementById("search").addEventListener("input",function(){'
        'const term=this.value.toLowerCase();document.querySelectorAll("details").forEach(function(section){'
        'section.hidden=!section.querySelector("summary").textContent.toLowerCase().includes(term);});});'
        "</script></html>"
    )
    (ROOT / "screen-gallery.html").write_text(document)


def review_sheets(catalog, directory):
    """Place four native 800-pixel pages on each sheet without resizing them."""
    directory.mkdir(parents=True, exist_ok=True)
    pages = [(spec["id"], page) for spec in catalog["figures"] for page in spec["screen_preview"]["pages"]]
    records = []
    font = ImageFont.truetype(str(FONT), 14)
    for offset in range(0, len(pages), 4):
        sheet = Image.new("RGB", (1600, 1248), "#dbe4ee")
        draw = ImageDraw.Draw(sheet)
        name = f"sheet-{offset // 4 + 1:03d}.png"
        for index, (identity, page) in enumerate(pages[offset : offset + 4]):
            x, y = index % 2 * WIDTH, index // 2 * 624
            draw.text((x + 8, y + 3), page["png"], font=font, fill="#203149")
            with Image.open(ROOT / page["png"]) as image:
                sheet.paste(image, (x, y + 24))
            records.append({"figure": identity, "page": page["png"], "sheet": name, "native_width": 800})
        sheet.save(directory / name, optimize=True)
    (directory / "pages.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    return records


def main():
    """Render registered previews with unchanged full-resolution counterparts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Render selected figure identifiers.")
    parser.add_argument("--review-sheets", type=Path, help="Write native-size review sheets into this directory.")
    parser.add_argument(
        "--review-only", action="store_true", help="Use existing screen pages without rerendering them."
    )
    args = parser.parse_args()
    catalog = json.loads((ROOT / "catalog.json").read_text())
    records = []
    failures = []
    for spec in catalog["figures"]:
        if not args.review_only and spec.get("screen_preview") and (not args.only or spec["id"] in args.only):
            try:
                records.append(render(spec))
                print(spec["id"], len(spec["screen_preview"]["pages"]), flush=True)
            except ValueError as error:
                failures.append({"id": spec["id"], "error": str(error)})
                print("FAILED", spec["id"], str(error), flush=True)
    persist(catalog, records)
    if args.review_sheets:
        review_sheets(catalog, args.review_sheets)
    (ROOT / "screen-failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
