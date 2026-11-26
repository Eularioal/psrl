# Sphinx configuration for Markdown and reStructuredText documentation.
# Uses the `pydata-sphinx-theme` theme.

import os
import re
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

sys.path.insert(0, os.path.abspath(".."))

# -- Project information --------------------------------------------------

project = "PSRL"
copyright = "2025, PKU & SJTU PSRL Team"
author = "PKU & SJTU PSRL Team"
release = "0.1.0"

# -- General configuration ------------------------------------------------

extensions = [
    # Markdown support via MyST
    "myst_parser",
    # Sphinx built-ins
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    # UI enhancements
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.video",
]

# -- MyST configuration ---------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
    "attrs_inline",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]

# -- Source file suffixes -------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# README.md is the repository knowledge index. The public site has its own
# navigation and excludes the internal knowledge directories below.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "**/README.md",
    "superpowers",
    "psrl",
    "env-worker-deep-dive",
    "tools",
]


def _link_repository_markdown(app, docname, source):
    """Keep local Markdown links usable when their targets are outside the site."""
    docs_root = Path(app.srcdir).resolve()
    repo_root = docs_root.parent
    page_dir = (docs_root / docname).parent
    repo_url = app.config.html_theme_options["github_url"].rstrip("/")
    revision = app.config.html_context.get("github_version", "main")

    def replace_link(match):
        href = match.group(1)
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return match.group(0)
        target = (page_dir / unquote(parsed.path)).resolve()
        if target.suffix != ".md" or not target.is_file() or not target.is_relative_to(repo_root):
            return match.group(0)
        if target.is_relative_to(docs_root):
            target_doc = target.relative_to(docs_root).with_suffix("").as_posix()
            if target_doc in app.env.found_docs:
                return match.group(0)
        target_url = f"{repo_url}/blob/{revision}/{quote(target.relative_to(repo_root).as_posix())}"
        if parsed.fragment:
            target_url += f"#{parsed.fragment}"
        return f"]({target_url})"

    source[0] = re.sub(r"\]\(([^\s)]+)\)", replace_link, source[0])


def setup(app):
    app.connect("source-read", _link_repository_markdown)


# -- Options for HTML output ----------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "PSRL Documentation"

html_theme_options = {
    "logo": {
        "text": "PSRL",
    },
    "github_url": "https://github.com/psrl-project/psrl",
    # Docs live in the code repo, so "Edit this page" resolves to a real source file.
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 1,
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "header_links_before_dropdown": 6,
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
}

html_context = {
    "github_user": "psrl-project",
    "github_repo": "psrl",
    "github_version": "main",
    "doc_path": "docs/",
}

templates_path = ["_templates"]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/relabel.js"]

html_sidebars = {
    "index": ["sidebar-collapse", "sidebar-nav-bs"],
    "**": ["sidebar-brand", "sidebar-collapse", "sidebar-nav-bs"],
}

# -- Copybutton configuration --------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |> "
copybutton_prompt_is_regexp = True

# Intersphinx is disabled for restricted network environments.

# -- Napoleon settings (Google/NumPy docstrings) --------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Mermaid configuration ------------------------------------------------

mermaid_output_format = "raw"
