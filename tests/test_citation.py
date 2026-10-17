"""``CITATION.cff`` (GitHub's "Cite this repository") and the other
places that ask users to cite Picasso name the same publication, so that
switching to a new paper cannot miss one of them (see the comment at the
top of ``CITATION.cff``).

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def citation() -> dict:
    return yaml.safe_load((ROOT / "CITATION.cff").read_text("utf-8"))


@pytest.fixture(scope="module")
def project() -> dict:
    text = (ROOT / "pyproject.toml").read_text("utf-8")
    return tomllib.loads(text)["project"]


def _section(path: str, start: str, end: str | None = None) -> str:
    """The text of ``path`` from ``start`` to ``end`` (or its end)."""
    text = (ROOT / path).read_text("utf-8")
    assert start in text, f"{path} has no {start!r}"
    text = text[text.index(start) :]
    return text[: text.index(end)] if end else text


#: where users are asked to cite Picasso: (file, start, end)
PLACES = [
    ("readme.rst", "SYNC-START: citing", "SYNC-END: citing"),
    ("readme.rst", "Picasso\n=======", "Installation\n------------"),
    ("docs/index.rst", "Citing Picasso\n--------------", ".. toctree::"),
    ("docs/index.rst", "Picasso is complemented by", "Modules\n-------"),
    (
        "release/one_click_windows_gui/readme.txt",
        "One-click installer",
        "How to install",
    ),
    (
        "release/one_click_windows_gui/readme.txt",
        "SYNC-START: citing",
        "SYNC-END: citing",
    ),
    (
        "release/one_click_macos_gui/readme.txt",
        "One-click installer",
        "How to install",
    ),
    (
        "release/one_click_macos_gui/readme.txt",
        "SYNC-START: citing",
        "SYNC-END: citing",
    ),
]


@pytest.mark.parametrize(
    "path,start,end", PLACES, ids=[f"{p}:{s[:20]}" for p, s, _ in PLACES]
)
def test_names_the_preferred_citation(citation, path, start, end):
    doi = citation["preferred-citation"]["doi"]
    assert doi in _section(path, start, end), (
        f"{path} (from {start!r}) does not name the preferred citation of "
        f"CITATION.cff, DOI {doi}"
    )


def test_pyproject_links_the_preferred_citation(citation, project):
    doi = citation["preferred-citation"]["doi"]
    assert project["urls"]["Publications"] == f"https://doi.org/{doi}"


def test_matches_the_package(citation, project):
    assert citation["cff-version"] == "1.2.0"
    assert citation["license"] == project["license"]
    assert citation["repository-code"] == project["urls"]["Source"]
    # the authors of the package are authors of the software
    families = {author.get("family-names") for author in citation["authors"]}
    for author in project["authors"]:
        assert author["name"].split()[-1] in families, author["name"]
