"""Export the PSRL environment GPU scheduling report to a shareable A4 PDF."""

import argparse
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode
from weasyprint import CSS, HTML

try:
    from weasyprint.text.fonts import FontConfiguration
except ModuleNotFoundError:
    from weasyprint.fonts import FontConfiguration


EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"
REPORT = EXPERIMENTS / "12-研究报告_PSRL环境GPU调度的设计与评估.md"
DEFAULT_FONT = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
DEFAULT_OUTPUT = EXPERIMENTS / "psrl-env-gpu-scheduling-report.pdf"

STYLESHEET = """
@page {
    size: A4;
    margin: 38pt 48pt 46pt;
    @bottom-left {
        content: "PSRL  /  TECHNICAL REPORT";
        font: 8pt "DejaVu Sans", sans-serif;
        color: #4c6177;
    }
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font: 8pt "DejaVu Sans", sans-serif;
        color: #4c6177;
    }
}
body {
    font-family: "DejaVu Sans", Report, sans-serif;
    color: #202b38;
    font-size: 10.2pt;
    line-height: 1.42;
    margin: 0;
    orphans: 2;
    widows: 2;
}
h1, h2, h3 { color: #153658; break-after: avoid; }
h1 { font-size: 21pt; line-height: 1.23; margin: 6pt 0 9pt; }
h2 { font-size: 14pt; margin: 12pt 0 5pt; }
h3 { font-size: 11.5pt; margin: 10pt 0 4pt; }
p { margin: 0 0 6pt; }
a { color: #215d89; text-decoration: none; }
strong { color: #153658; }
.edition { color: #41576c; font-size: 9.4pt; margin: 0 0 12pt; }
.abstract {
    background: #f0f5fa;
    border-left: 2pt solid #3a77a7;
    padding: 1pt 12pt 5pt;
    break-inside: avoid;
}
.abstract h2 { margin-top: 8pt; }
.figure { margin: 7pt 0; break-inside: avoid; }
.figure img { display: block; margin: 0 auto 5pt; width: 94%; }
.figure p { font-size: 9.1pt; color: #34475b; margin-bottom: 0; }
.table-block { margin: 5pt 0 10pt; break-inside: avoid; }
.table-block p { font-size: 9.3pt; margin-bottom: 5pt; }
table { width: 100%; border-collapse: collapse; font-size: 9.2pt; line-height: 1.35; }
th { background: #e4edf6; color: #153658; }
th, td { padding: 4pt 5pt; border-bottom: 0.5pt solid #cdd9e6; }
th:first-child, td:first-child { text-align: left; }
blockquote { background: #f0f5fa; margin: 7pt 0 8pt; padding: 6pt 9pt 0.1pt; break-inside: avoid; }
li { margin-bottom: 4pt; break-inside: avoid; }
"""


def report_html(source):
    """Keep public citations in the PDF and present local evidence links as labels."""
    source = re.sub(r"^## 目录\n.*?(?=^## 1 引言)", "", source, flags=re.M | re.S)
    source = re.sub(r"^技术报告 · .*$", r'<p class="edition">\g<0></p>', source, flags=re.M)
    html = markdown.markdown(
        source,
        extensions=["tables", "toc"],
        extension_configs={"toc": {"slugify": slugify_unicode}},
    )
    html = re.sub(r"(<h1[^>]*>[^<]+?)：", r"\1<br>", html, count=1)
    html = re.sub(r'(?s)(<h2 id="摘要">.*?)(?=<h2 )', r'<div class="abstract">\1</div>', html)
    html = re.sub(
        r'(?s)<p><a href="[^"]+">(<img [^>]+/>)</a></p>\s*(<p><strong>图 .*?</p>)',
        r'<div class="figure">\1\2</div>',
        html,
    )
    html = re.sub(
        r'(?s)(<p><strong>表 .*?</p>)\s*(<table>.*?</table>)',
        r'<div class="table-block">\1\2</div>',
        html,
    )
    return re.sub(r'<a href="(?!https://|#)[^"]+">([^<]+)</a>', r'\1', html)


def export(font, output):
    source = REPORT.read_text(encoding="utf-8")
    title = source.splitlines()[0].removeprefix("# ")
    css = f'@font-face {{font-family: Report; src: url("{font.resolve(strict=True).as_uri()}");}}\n' + STYLESHEET
    html = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{title}</title></head>'
    html += f"<body>{report_html(source)}</body></html>"
    font_config = FontConfiguration()
    stylesheet = CSS(string=css, font_config=font_config)
    document = HTML(string=html, base_url=EXPERIMENTS.as_uri() + "/")
    document.write_pdf(str(output), stylesheets=[stylesheet], font_config=font_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT, help="Path to a Chinese TrueType font.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    args = parser.parse_args()
    export(args.font, args.output)
