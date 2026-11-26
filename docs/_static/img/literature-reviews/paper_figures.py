"""Extract attributed paper figures and validate their local screen copies."""

import argparse
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
MANIFEST = ROOT / "paper-figures.json"


def digest(path):
    """Identify the exact source edition and generated files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_figure(spec, cache, download):
    """Crop source graphics without changing their objects, numbers or labels."""
    import pymupdf

    source = cache / (spec["paper"] + (".pdf" if spec["format"] == "pdf" else ".gif"))
    if not source.exists() and download:
        import requests

        response = requests.get(spec["source_url"], timeout=45)
        response.raise_for_status()
        cache.mkdir(parents=True, exist_ok=True)
        source.write_bytes(response.content)
    if not source.exists():
        raise FileNotFoundError(f"Download the cited source or provide --download: {source!s}.")
    if spec.get("source_sha256") and digest(source) != spec["source_sha256"]:
        raise ValueError(f"Source edition changed: {spec['paper']!r}.")
    spec["source_sha256"] = digest(source)
    if spec["format"] == "pdf":
        document = pymupdf.open(source)
        page = document[spec["pdf_page"] - 1]

        def crop(box):
            rectangle = pymupdf.Rect(box)
            output = pymupdf.open()
            selected = output.new_page(width=rectangle.width, height=rectangle.height)
            selected.show_pdf_page(selected.rect, document, page.number, clip=rectangle)
            return output

        full_document = crop(spec["box"])
        full = full_document[0]
        (ROOT / (spec["id"] + ".svg")).write_text(full.get_svg_image())
        pixmap = full.get_pixmap(matrix=pymupdf.Matrix(1800 / full.rect.width, 1800 / full.rect.width))
        original = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        panels = []
        for box in spec.get("panels", [spec["box"]]):
            selected_document = crop(box)
            selected = selected_document[0]
            scale = min(798 / selected.rect.width, 598 / selected.rect.height)
            pixmap = selected.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
            panels.append(Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB"))
    else:
        transparent = Image.open(source).convert("RGBA")
        original = Image.new("RGB", transparent.size, "white")
        original.paste(transparent, mask=transparent.getchannel("A"))
        scale = min(800 / original.width, 600 / original.height)
        panels = [
            original.resize((round(original.width * scale), round(original.height * scale)), Image.Resampling.LANCZOS)
        ]
    original.save(ROOT / (spec["id"] + ".png"), optimize=True)
    pages = []
    for index, panel in enumerate(panels, 1):
        output = Image.new("RGB", (800, min(600, panel.height)), "white")
        output.paste(panel, ((800 - panel.width) // 2, 0))
        name = spec["id"] + "-screen" + (f"-{index:02d}" if index > 1 else "") + ".png"
        output.save(ROOT / name, optimize=True)
        pages.append({"png": name, "width": output.width, "height": output.height, "sha256": digest(ROOT / name)})
    spec["pages"] = pages
    spec["full_png_sha256"] = digest(ROOT / (spec["id"] + ".png"))
    if spec["format"] == "pdf":
        spec["full_svg_sha256"] = digest(ROOT / (spec["id"] + ".svg"))


def validate(manifest):
    """Check attribution, screen sizes, file hashes and complete document embeds."""
    errors = []
    for spec in manifest["figures"]:
        for key in (
            "title",
            "authors",
            "source_url",
            "source_page",
            "figure_number",
            "rights",
            "accessed",
            "source_sha256",
        ):
            if not spec.get(key):
                errors.append(f"Missing {key!r}: {spec['id']}.")
        for suffix in ("png", "svg"):
            expected = spec.get(f"full_{suffix}_sha256")
            if expected and digest(ROOT / (spec["id"] + "." + suffix)) != expected:
                errors.append(f"Full figure changed: {spec['id']}.")
        for page in spec.get("pages", []):
            path = ROOT / page["png"]
            if not path.exists():
                errors.append(f"Missing screen file: {path.name}.")
                continue
            with Image.open(path) as image:
                if image.width != 800 or image.height > 600 or image.size != (page["width"], page["height"]):
                    errors.append(f"Invalid screen dimensions: {path.name}.")
            if digest(path) != page["sha256"]:
                errors.append(f"Screen file changed: {path.name}.")
        for placement in spec.get("placements", []):
            text = (REPO / placement["document"]).read_text()
            if any(page["png"] not in text for page in spec["pages"]):
                errors.append(f"Incomplete paper figure: {placement['document']} / {spec['id']}.")
            if spec["source_page"] not in text:
                errors.append(f"Missing source citation: {placement['document']} / {spec['id']}.")
    return errors


def main():
    """Render explicitly selected sources or verify the existing collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/psrl-paper-figures"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if not args.check:
        for spec in manifest["figures"]:
            if not args.only or spec["id"] in args.only:
                render_figure(spec, args.cache, args.download)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    errors = validate(manifest) if args.check else []
    print(
        json.dumps(
            {
                "figures": len(manifest["figures"]),
                "screen_pages": sum(len(spec.get("pages", [])) for spec in manifest["figures"]),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
