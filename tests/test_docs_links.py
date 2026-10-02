"""Test that the documentation links in the code point to existing pages.

Help buttons and error messages open ``picasso.docs_url("page.html#anchor")``.
Each such link must name a page in ``docs/`` and an anchor on it, either an
explicit label (``.. _anchor:``) or the id of a section heading. Renaming a
heading or moving a section to another page otherwise breaks the help
buttons silently.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PACKAGE = ROOT / "picasso"

#: ``docs_url("page.html#anchor")`` calls, also split over several lines
DOCS_URL_RE = re.compile(r"""docs_url\(\s*(?:f?["'])([^"']*)["']""")
#: full documentation URLs in comments, docstrings and messages
FULL_URL_RE = re.compile(
    r"picassosr\.readthedocs\.io/en/[\w.]+/([\w/-]+\.html(?:#[\w-]+)?)"
)
LABEL_RE = re.compile(r"^\.\. _([\w-]+):\s*$", re.MULTILINE)
UNDERLINE_RE = re.compile(r"""^([=\-~^"+`#*])\1{2,}\s*$""")


def section_id(title: str) -> str:
    """Return the HTML id docutils gives a section titled ``title``."""
    title = unicodedata.normalize("NFKD", title)
    title = title.encode("ascii", "ignore").decode("ascii").lower()
    title = re.sub(r"[^a-z0-9]+", "-", title)
    return re.sub(r"^[^a-z]+|-+$", "", title)


def anchors(page: Path) -> set[str]:
    """Return the explicit labels and section ids of a docs page."""
    text = page.read_text(encoding="utf-8")
    result = set(LABEL_RE.findall(text))
    lines = text.splitlines()
    for title, underline in zip(lines, lines[1:]):
        title = title.strip()
        if (
            title
            and UNDERLINE_RE.match(underline)
            and len(underline.rstrip()) >= len(title)
            and not UNDERLINE_RE.match(title)
        ):
            result.add(section_id(title))
    return result


def code_links() -> list[tuple[str, str]]:
    """Return ``(source location, link)`` for every docs link in the code."""
    links = []
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for regex in (DOCS_URL_RE, FULL_URL_RE):
            for match in regex.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                where = f"{path.relative_to(ROOT)}:{line}"
                links.append((where, match.group(1)))
    return links


LINKS = [link for link in code_links() if link[1]]


def test_links_found():
    """The scan finds the help-button links at all."""
    assert len(LINKS) > 20


@pytest.mark.parametrize(
    "where, link", LINKS, ids=[f"{w} {lnk}" for w, lnk in LINKS]
)
def test_docs_link_target_exists(where, link):
    page, _, anchor = link.partition("#")
    assert page.endswith(".html"), f"{where}: {link} is not an .html page"
    source = DOCS / (page.removesuffix(".html") + ".rst")
    assert source.exists(), f"{where}: no docs page {source.name}"
    if anchor:
        assert anchor in anchors(source), (
            f"{where}: anchor #{anchor} not found in "
            f"docs/{source.relative_to(DOCS)}"
        )
