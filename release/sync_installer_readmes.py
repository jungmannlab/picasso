"""Sync shared sections from the main readme into installer readmes.

Sections in the main ``readme.rst`` that are wrapped with::

    .. SYNC-START: <key>
    ...
    .. SYNC-END: <key>

are copied into the matching sentinel pairs in each installer readme.
The installer readmes are plain-text ``.txt`` files (shipped alongside
the installers), so reStructuredText markup in the synced bodies is
converted to a reader-friendly plain-text form before insertion.

Run from the repo root, or via the pre-commit hook.

Usage:
    python release/sync_installer_readmes.py          # rewrite if needed
    python release/sync_installer_readmes.py --check  # exit 1 on drift
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "readme.rst"
TARGETS = [
    REPO_ROOT / "release" / "one_click_macos_gui" / "readme.txt",
    REPO_ROOT / "release" / "one_click_windows_gui" / "readme.txt",
]

SECTION_RE = re.compile(
    r"(\.\. SYNC-START: (?P<key>[\w-]+)\s*\n)"
    r"(?P<body>.*?)"
    r"(\n\.\. SYNC-END: (?P=key))",
    re.DOTALL,
)

# `label <url>`__  or  `label <url>`_   ->   label (url)
RST_LINK_RE = re.compile(r"`([^`<]+?)\s*<([^`>]+)>`_{1,2}")
# ``code`` -> code
RST_CODE_RE = re.compile(r"``([^`]+)``")
# leading "| " of an rst line block
RST_LINE_BLOCK_RE = re.compile(r"^\| ?", re.MULTILINE)


def rst_to_text(body: str) -> str:
    body = RST_LINK_RE.sub(
        lambda m: f"{m.group(1).strip()} ({m.group(2)})", body
    )
    body = RST_CODE_RE.sub(r"\1", body)
    body = RST_LINE_BLOCK_RE.sub("", body)
    # Drop bare "|" lines used as vertical spacers in rst line blocks.
    body = re.sub(r"^\|\s*$", "", body, flags=re.MULTILINE)
    return body


def extract_sections(text: str, path: Path) -> dict[str, str]:
    sections: dict[str, str] = {}
    for match in SECTION_RE.finditer(text):
        key = match.group("key")
        if key in sections:
            raise SystemExit(f"{path}: duplicate SYNC-START for key '{key}'")
        sections[key] = match.group("body")
    return sections


def replace_sections(text: str, sections: dict[str, str]) -> str:
    def _sub(match: re.Match) -> str:
        key = match.group("key")
        if key not in sections:
            raise SystemExit(
                f"key '{key}' present in target but missing from {SOURCE}"
            )
        return match.group(1) + rst_to_text(sections[key]) + match.group(4)

    return SECTION_RE.sub(_sub, text)


def main() -> int:
    check_only = "--check" in sys.argv[1:]

    source_text = SOURCE.read_text(encoding="utf-8")
    sections = extract_sections(source_text, SOURCE)
    if not sections:
        raise SystemExit(f"{SOURCE}: no SYNC-START/SYNC-END markers found")

    drift = False
    for target in TARGETS:
        original = target.read_text(encoding="utf-8")
        target_keys = set(extract_sections(original, target))
        missing = target_keys - set(sections)
        if missing:
            raise SystemExit(
                f"{target}: keys not found in main readme: "
                f"{sorted(missing)}"
            )
        not_in_target = set(sections) - target_keys
        if not_in_target:
            raise SystemExit(
                f"{target}: missing SYNC markers for keys: "
                f"{sorted(not_in_target)}"
            )

        updated = replace_sections(original, sections)
        if updated == original:
            continue

        drift = True
        if check_only:
            print(f"drift: {target.relative_to(REPO_ROOT)}")
        else:
            target.write_text(updated, encoding="utf-8")
            print(f"updated: {target.relative_to(REPO_ROOT)}")

    if check_only and drift:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
