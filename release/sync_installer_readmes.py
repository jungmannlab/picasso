"""Sync shared sections from the main readme into installer readmes.

Sections in the main ``readme.rst`` that are wrapped with::

    .. SYNC-START: <key>
    ...
    .. SYNC-END: <key>

are copied verbatim into the matching sentinel pairs in each installer
readme. Run from the repo root, or via the pre-commit hook.

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
    REPO_ROOT / "release" / "one_click_macos_gui" / "readme.rst",
    REPO_ROOT / "release" / "one_click_windows_gui" / "readme.rst",
]

SECTION_RE = re.compile(
    r"(\.\. SYNC-START: (?P<key>[\w-]+)\s*\n)"
    r"(?P<body>.*?)"
    r"(\n\.\. SYNC-END: (?P=key))",
    re.DOTALL,
)


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
        return match.group(1) + sections[key] + match.group(4)

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
