"""Record the public Python API and the command line in
``tests/data/stability/``, which ``tests/test_stability.py`` compares
against.

Run from the repository root after a deliberate change of the public
names (``__all__`` of a module on the API pages, a command or an
option), e.g. once a deprecated name is due for removal, see
``docs/versioning.rst``::

    python tests/regen_stability_snapshots.py

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "tests" / "data" / "stability"
API_SNAPSHOT = SNAPSHOTS / "api.json"
CLI_SNAPSHOT = SNAPSHOTS / "cli.json"


def public_modules() -> list[str]:
    """The modules on the API pages: those listed in docs/api/index.rst
    and the submodules of the packages listed ``:recursive:``."""
    text = (ROOT / "docs" / "api" / "index.rst").read_text()
    modules = []
    for block in re.split(r"\n\.\. autosummary::", text)[1:]:
        recursive = ":recursive:" in block.split("\n\n")[0]
        for name in re.findall(r"^   (picasso[\w.]*)\s*$", block, re.M):
            modules.append(name)
            if recursive:
                package = ROOT.joinpath(*name.split("."))
                modules.extend(
                    f"{name}.{module.name}"
                    for module in pkgutil.iter_modules([str(package)])
                    if not module.name.startswith("_")
                )
    return sorted(set(modules))


def public_api() -> dict[str, list[str]]:
    """The ``__all__`` of every module on the API pages (empty for a
    package that only holds submodules)."""
    return {
        name: list(getattr(importlib.import_module(name), "__all__", []))
        for name in public_modules()
    }


def _options(parser: argparse.ArgumentParser) -> list[str]:
    """The option strings of a parser, sorted, without ``-h``."""
    return sorted(
        option
        for action in parser._actions
        for option in action.option_strings
        if option not in ("-h", "--help")
    )


def command_line() -> dict[str, list[str]]:
    """Every built-in command (``"plugins enable"`` for a nested one)
    with its options."""
    from picasso.__main__ import _build_parser

    parser, _ = _build_parser()
    commands = {}

    def walk(parser: argparse.ArgumentParser, prefix: str) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    commands[prefix + name] = _options(sub)
                    walk(sub, prefix + name + " ")

    walk(parser, "")
    return dict(sorted(commands.items()))


def main() -> None:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    for path, data in (
        (API_SNAPSHOT, public_api()),
        (CLI_SNAPSHOT, command_line()),
    ):
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
