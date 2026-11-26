"""Check maintained document figure coverage without third-party dependencies."""

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
EXCLUDED = {"results", "baseline", "_build", "__pycache__", ".cache"}
DECORATIVE = re.compile(r"(?:logo|banner|badge|favicon|PSRL_character)", re.I)


def local_target(document, reference):
    """Resolve MyST site-root assets as well as ordinary relative paths."""
    path = unquote(urlsplit(reference).path)
    if path.startswith("/_static/"):
        return (REPO / "docs" / path.lstrip("/")).resolve()
    return (document.parent / path).resolve()


def image_references(content):
    """Recognize Markdown, HTML and MyST figures outside ordinary code blocks."""
    references = []
    definitions = dict(re.findall(r"^\[([^]]+)\]:\s*(\S+)", content, re.M))
    prose = []
    fence = None
    for line in content.splitlines():
        opening = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if opening:
            token, info = opening.groups()
            if fence is None:
                directive = re.match(r"\{(?:image|figure)\}\s+(.+)", info.strip())
                if directive:
                    references.append(directive[1].strip())
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            prose.append(line)
    prose = "\n".join(prose)
    references.extend(re.findall(r"!\[[^]\n]*\]\(([^\n)]+)\)", prose))
    references.extend(re.findall(r"<img\b[^>]*\bsrc=[\"']([^\"']+)", prose, re.I))
    for label, key in re.findall(r"!\[([^]\n]*)\]\[([^]\n]*)\]", prose):
        if target := definitions.get(key or label):
            references.append(target)
    return [re.sub(r"\s+[\"'][^\"']*[\"']$", "", ref.strip()).strip("<>") for ref in references]


def maintained_documents():
    """Use the same explicit source exclusions as the documentation inventory."""
    output = subprocess.check_output(["rg", "--files", "docs", "-g", "*.md"], cwd=REPO, text=True)
    return sorted(REPO / name for name in output.splitlines() if not EXCLUDED.intersection(Path(name).parts))


def validate_generated(spec):
    """Check local raster and vector integrity, not their semantic quality."""
    errors = []
    for suffix in (".png", ".svg"):
        path = ROOT / (spec["id"] + suffix)
        if not path.is_file():
            errors.append(f"Missing generated image: {path.name}.")
            continue
        try:
            if suffix == ".png":
                header = path.read_bytes()[:24]
                if header[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("Invalid PNG signature.")
                width, height = struct.unpack(">II", header[16:24])
                if width < 1000 or height < 500:
                    raise ValueError("The generated figure is smaller than the required reading size.")
            else:
                vector = ElementTree.parse(path).getroot()
                width = float(vector.attrib["viewBox"].split()[2])
                sizes = [
                    float(size) * 900 / width
                    for node in vector.iter()
                    for size in re.findall(r"font-size:\s*([\d.]+)px", node.get("style", ""))
                ]
                if sizes and min(sizes) < 14:
                    raise ValueError("Vector labels become smaller than 14 pixels at 900-pixel reading width.")
        except (ValueError, struct.error, ElementTree.ParseError) as error:
            errors.append(f"Invalid generated image {path.name}: {error}")
    if not spec.get("caption") or spec.get("status") not in {"teaching", "reported", "navigation"}:
        errors.append(f"Missing caption or evidence status: {spec['id']}.")
    review = spec.get("explanation_review", {})
    if not 3 <= len(review.get("target_questions", [])) <= 6:
        errors.append(f"Missing explanation questions: {spec['id']}.")
    if not review.get("source") or not (ROOT / review["source"]).is_file():
        errors.append(f"Missing explanation source: {spec['id']}.")
    if original := spec.get("original"):
        path = REPO / original
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec["original_sha256"]:
            errors.append(f"Original evidence changed or missing: {original}.")
    for key in ("display_vector", "display_generator"):
        if source := spec.get(key):
            path = REPO / source["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                errors.append(f"Display source changed or missing: {source['path']}.")
    if not (preview := spec.get("screen_preview")) or not preview.get("pages"):
        errors.append(f"Missing screen page registration: {spec['id']}.")
        return errors
    for suffix, expected_hash in preview.get("source_sha256", {}).items():
        if hashlib.sha256((ROOT / (spec["id"] + "." + suffix)).read_bytes()).hexdigest() != expected_hash:
            errors.append(f"Full source changed after screen rendering: {spec['id']}.")
    for page in preview["pages"]:
        for key in ("png", "svg"):
            if not (ROOT / page[key]).is_file():
                errors.append(f"Missing screen page: {page[key]}.")
        try:
            if (path := ROOT / page["png"]).is_file():
                data = path.read_bytes()
                width, height = struct.unpack(">II", data[16:24])
                if data[:8] != b"\x89PNG\r\n\x1a\n" or width != 800 or not 0 < height <= 600:
                    errors.append(f"Screen page exceeds its display budget: {path.name}.")
                if (width, height) != (page["width"], page["height"]):
                    errors.append(f"Screen page dimensions differ from registration: {path.name}.")
            if (path := ROOT / page["svg"]).is_file():
                vector = ElementTree.parse(path).getroot()
                scale = page["width"] / float(vector.attrib["viewBox"].split()[2])
                sizes = [
                    float(size) * scale
                    for node in vector.iter()
                    for size in re.findall(r"font-size:\s*([\d.]+)px", node.get("style", ""))
                ]
                if sizes and min(sizes) < 14:
                    errors.append(f"Screen labels become smaller than 14 pixels: {path.name}.")
        except (ValueError, struct.error, ElementTree.ParseError) as error:
            errors.append(f"Invalid screen page {path.name}: {error}")
    return errors


def main():
    """Report missing figures and unresolved image references with a failing exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "coverage.json")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "catalog.json").read_text())
    errors = []
    records = []
    for path in maintained_documents():
        qualifying = []
        decorative = []
        remote = []
        for reference in image_references(path.read_text()):
            parts = urlsplit(reference)
            if parts.scheme or parts.netloc:
                remote.append(reference)
                continue
            target = local_target(path, reference)
            if not target.is_file():
                errors.append({"document": str(path.relative_to(REPO)), "missing_image": reference})
            elif DECORATIVE.search(target.name):
                decorative.append(reference)
            elif target.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif"}:
                qualifying.append(reference)
        if not qualifying:
            errors.append({"document": str(path.relative_to(REPO)), "error": "No local explanatory image."})
        records.append(
            {
                "document": str(path.relative_to(REPO)),
                "figures": qualifying,
                "decorative_images": decorative,
                "remote_images": remote,
            }
        )
    for spec in catalog["figures"]:
        errors.extend(validate_generated(spec))
    figure_specs = {spec["id"]: spec for spec in catalog["figures"]}
    for placement in catalog["placements"]:
        path = REPO / placement["document"]
        preview = figure_specs[placement["figure"]].get("screen_preview", {})
        expected = {(ROOT / page["png"]).resolve() for page in preview.get("pages", [])}
        actual = {local_target(path, ref) for ref in image_references(path.read_text()) if not urlsplit(ref).scheme}
        if not expected or not expected.issubset(actual):
            errors.append({"document": placement["document"], "missing_placement": placement["figure"]})
    report = {
        "scope": "Maintained Markdown under docs, including READMEs and public site pages.",
        "excluded_directory_names": sorted(EXCLUDED),
        "documents": len(records),
        "documents_with_explanatory_images": sum(bool(record["figures"]) for record in records),
        "generated_figures": len(catalog["figures"]),
        "chapter_explanations_with_question_map": sum(
            bool(spec.get("explanation_review")) for spec in catalog["figures"]
        ),
        "new_image_placements": len(catalog["placements"]),
        "screen_figures": sum(bool(spec.get("screen_preview", {}).get("pages")) for spec in catalog["figures"]),
        "screen_pages": sum(len(spec.get("screen_preview", {}).get("pages", [])) for spec in catalog["figures"]),
        "limits": (
            "This check validates references, coverage and generated file integrity. "
            "Meaning, color use and readability require rendered human review."
        ),
        "errors": errors,
        "coverage": records,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"coverage", "limits"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
