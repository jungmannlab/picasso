"""Record the public Python API and the command line in
``tests/data/stability/``, which ``tests/test_stability.py`` compares
against.

Run from the repository root after a deliberate change of the public
names (``__all__`` of a module on the API pages, a command or an
option) or of their deprecations, e.g. once a deprecated name is due for
removal, see ``docs/versioning.rst``::

    python tests/regen_stability_snapshots.py

It refuses to record the removal of a public name (a rename removes the
old name) unless the name was recorded as deprecated and this version
may remove it. ``--allow-breaking`` records any removal, for a major
release (e.g., the clean-up of 1.0) or a name that was public by
mistake.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
import sys
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOTS = ROOT / "tests" / "data" / "stability"
API_SNAPSHOT = SNAPSHOTS / "api.json"
CLI_SNAPSHOT = SNAPSHOTS / "cli.json"
DEPRECATED_SNAPSHOT = SNAPSHOTS / "deprecated.json"

#: Packages listed ``:recursive:`` whose submodules the API pages
#: document for reference only (docs/versioning.rst), with the
#: submodules that are public nonetheless
REFERENCE_ONLY = {"picasso.fitting": ["precision"]}


def public_modules() -> list[str]:
    """The public modules: those listed in docs/api/index.rst and the
    submodules of the packages listed ``:recursive:``, except the
    submodules of ``REFERENCE_ONLY`` packages that it leaves out."""
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
                    and (
                        name not in REFERENCE_ONLY
                        or module.name in REFERENCE_ONLY[name]
                    )
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


def public_names(
    api: dict[str, list[str]], cli: dict[str, list[str]]
) -> list[str]:
    """Every public module, name, command and option, written as
    ``picasso.deprecation`` registers them, e.g., ``"picasso.io"``,
    ``"picasso.io.load_locs"``, ``"picasso render"`` and
    ``"picasso render --scaling"``."""
    names = []
    for module, members in api.items():
        names.append(module)
        names.extend(f"{module}.{member}" for member in members)
    for command, options in cli.items():
        names.append(f"picasso {command}")
        names.extend(f"picasso {command} {option}" for option in options)
    return names


def _defined_as(name: str) -> str | None:
    """The name ``picasso.deprecation.deprecated`` registers a public
    function or class under: where it is defined, e.g., in a submodule
    of the package that makes it public."""
    module, _, member = name.rpartition(".")
    obj = getattr(sys.modules.get(module), member, None)
    try:
        return f"{obj.__module__}.{obj.__qualname__}"
    except AttributeError:  # not a function or class
        return None


def deprecated_names(
    api: dict[str, list[str]], cli: dict[str, list[str]]
) -> dict[str, str]:
    """The deprecated ones of ``public_names``, with the version that
    may remove them, from the deprecations registered by the modules
    imported so far."""
    from picasso import deprecation

    remove_in = {entry.name: entry.remove_in for entry in deprecation.REGISTRY}
    deprecated = {}
    for name in public_names(api, cli):
        version = remove_in.get(name)
        if version is None and not name.startswith("picasso "):
            version = remove_in.get(_defined_as(name))
        if version is not None:
            deprecated[name] = version
    return deprecated


def read_snapshots() -> dict | None:
    """The recorded snapshots (``"api"``, ``"cli"`` and
    ``"deprecated"``), None if there are none yet."""
    if not API_SNAPSHOT.exists():
        return None
    deprecated = {}
    if DEPRECATED_SNAPSHOT.exists():
        deprecated = json.loads(DEPRECATED_SNAPSHOT.read_text())
    return {
        "api": json.loads(API_SNAPSHOT.read_text()),
        "cli": json.loads(CLI_SNAPSHOT.read_text()),
        "deprecated": deprecated,
    }


def refused_removals(
    recorded: dict,
    api: dict[str, list[str]],
    cli: dict[str, list[str]],
    version: str,
) -> list[str]:
    """Why the removals from the ``recorded`` snapshots (as
    ``read_snapshots`` returns them) to ``api`` and ``cli`` may not be
    recorded in Picasso ``version``: a removed name must have been
    deprecated, and ``version`` must be (a pre-release of) the version
    that may remove it, or later.

    Returns
    -------
    refused : list of str
        One line per refused removal, empty if all are allowed. A
        removed module or command is one removal, with its names or
        options.
    """
    removed = []
    for module, members in recorded["api"].items():
        if module not in api:
            removed.append(module)
        else:
            removed.extend(
                f"{module}.{member}"
                for member in members
                if member not in api[module]
            )
    for command, options in recorded["cli"].items():
        if command not in cli:
            removed.append(f"picasso {command}")
        else:
            removed.extend(
                f"picasso {command} {option}"
                for option in options
                if option not in cli[command]
            )
    current = Version(version).release[:2]
    refused = []
    for name in removed:
        remove_in = recorded["deprecated"].get(name)
        if remove_in is None:
            refused.append(f"{name} was removed without a deprecation")
        elif current < Version(remove_in).release[:2]:
            refused.append(
                f"{name} is deprecated, but may be removed in Picasso "
                f"{remove_in}, not {version}"
            )
    return refused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--allow-breaking",
        action="store_true",
        help="also record removals that skipped the deprecation cycle",
    )
    args = parser.parse_args()

    from picasso import __version__

    api, cli = public_api(), command_line()
    recorded = read_snapshots()
    if recorded is not None and not args.allow_breaking:
        refused = refused_removals(recorded, api, cli, __version__)
        if refused:
            sys.exit(
                "Nothing recorded: removing or renaming a public name "
                "needs a deprecation cycle (docs/versioning.rst).\n  "
                + "\n  ".join(refused)
                + "\nRestore these names or deprecate them first. For a "
                "major release or a name that was public by mistake, "
                "record the removal with --allow-breaking."
            )
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    for path, data in (
        (API_SNAPSHOT, api),
        (CLI_SNAPSHOT, cli),
        (DEPRECATED_SNAPSHOT, deprecated_names(api, cli)),
    ):
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
