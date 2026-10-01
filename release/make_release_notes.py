"""Write the GitHub release notes for the current version.

The version is read from ``picasso/version.py`` and the changelog
entries from the matching ``## X.Y.Z`` section of ``changelog.md``
(including any ``###`` subsections). The notes are written as
Markdown, ready to paste into the GitHub release, to
``release/release_notes.txt`` (git-ignored).

Fails if the newest section of ``changelog.md`` is not the version in
``picasso/version.py``, i.e., if one of the two was not updated.

Run from anywhere.

Usage:
    python release/make_release_notes.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = REPO_ROOT / "picasso" / "version.py"
CHANGELOG = REPO_ROOT / "changelog.md"
OUTPUT = REPO_ROOT / "release" / "release_notes.txt"

VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.M)
SECTION_RE = re.compile(r"^## (?P<version>\S+)\s*$", re.M)

NOTES = (
    "⚠️ Please see the readme.txt in the .zip files for important "
    "installation notes (camera config, Windows/macOS security warnings)."
)


def read_version() -> str:
    match = VERSION_RE.search(VERSION_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"{VERSION_FILE}: no __version__ found")
    return match.group(1)


def read_sections() -> list[tuple[str, str]]:
    """Return (version, body) for each ``## `` section, newest first."""
    text = CHANGELOG.read_text(encoding="utf-8")
    matches = list(SECTION_RE.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        sections.append((match.group("version"), body))
    return sections


def main() -> int:
    version = read_version()
    sections = read_sections()
    if not sections:
        raise SystemExit(f"{CHANGELOG}: no '## X.Y.Z' sections found")

    newest, body = sections[0]
    if newest != version:
        raise SystemExit(
            f"version mismatch: {VERSION_FILE.relative_to(REPO_ROOT)} is "
            f"{version}, but the newest section of "
            f"{CHANGELOG.relative_to(REPO_ROOT)} is {newest}"
        )
    if not body:
        raise SystemExit(f"{CHANGELOG}: section {version} is empty")

    notes = (
        f"# Picasso {version} is here\n"
        f"\n"
        f"## Changelog\n"
        f"\n"
        f"{body}\n"
        f"\n"
        f"## Notes\n"
        f"{NOTES}\n"
    )
    OUTPUT.write_text(notes, encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(REPO_ROOT)} (v{version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
