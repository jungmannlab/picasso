"""Write the third-party license notices shipped with the installers.

The one-click installers freeze Picasso with PyInstaller, so they
redistribute the Python interpreter, every Python dependency and the
native libraries these link against. Most of their licenses (BSD, MIT,
Apache, PSF) require that copies distributed in binary form carry the
copyright notice and license text, and PyQt6 (GPL-3.0) and Qt (LGPL-3.0)
require the full license. This script collects all of them from the
build environment into one plain-text file:

- every installed Python distribution (``importlib.metadata``): name,
  version, declared license, source URL and the text of every license
  file it installs. That is its ``*.dist-info/licenses`` folder, any
  ``LICENSE*`` / ``COPYING*`` / ``NOTICE*`` file, and license folders
  inside the package, such as ``imagecodecs/licenses`` for the C
  libraries imagecodecs bundles;
- every conda package (``conda-meta/*.json``), i.e., Python itself and
  the native libraries (OpenSSL, libffi, zlib, ...). Their license
  texts are read from the conda package cache (``info/licenses``).

Build-only tools (pip, build and PyInstaller's helpers) are left out;
PyInstaller itself is listed, as parts of it are frozen into the app.
Run it in the installer environment, after PyInstaller and before
the environment is removed, so the list matches the frozen packages.

Usage:
    python release/collect_third_party_licenses.py OUTPUT
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

#: Packages installed into the build environment only to build the
#: installer; PyInstaller does not freeze them into the app. PyInstaller
#: and pyinstaller-hooks-contrib are not among them: the app contains
#: their bootloader (GPL with an exception that allows this) and their
#: run-time hooks (Apache-2.0, which requires the notice).
BUILD_ONLY = {
    "altgraph",
    "build",
    "macholib",
    "pefile",
    "pip",
    "pyproject-hooks",
    "pywin32-ctypes",
    "wheel",
}

#: File-name prefixes (upper case) that mark a license file.
LICENSE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "COPYRIGHT")
#: Folder names (lower case) whose files are all license files.
LICENSE_DIRS = {"licenses", "licences"}
#: Suffixes of files that are never license texts, even if their name
#: matches (e.g. a ``license.py`` module).
CODE_SUFFIXES = {
    ".py",
    ".pyc",
    ".pyi",
    ".pyx",
    ".pxd",
    ".so",
    ".pyd",
    ".dll",
    ".dylib",
    ".js",
    ".json",
    ".html",
    ".css",
}

RULE = "=" * 78
THIN_RULE = "-" * 78


@dataclass
class Package:
    """One bundled package and its license notices."""

    name: str
    version: str
    license: str
    source: str = ""
    #: ``(file name, text)`` pairs, duplicates removed.
    texts: list[tuple[str, str]] = field(default_factory=list)


def canonical(name: str) -> str:
    """Normalize a distribution name as PEP 503 does."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_license_file(path: Path) -> bool:
    """Return True if ``path`` (relative to site-packages) is a license
    text."""
    if path.suffix.lower() in CODE_SUFFIXES:
        return False
    if any(part.lower() in LICENSE_DIRS for part in path.parts[:-1]):
        return True
    return path.name.upper().startswith(LICENSE_PREFIXES)


def read_text(path: Path) -> str | None:
    """Read a license file; return None for missing, empty or binary
    files."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data.strip() or b"\0" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    lines = [line.rstrip() for line in text.strip("\n").splitlines()]
    return "\n".join(lines)


def add_texts(package: Package, files: list[tuple[str, Path]]) -> None:
    """Append the readable, non-duplicate license files to
    ``package.texts``."""
    seen = {hashlib.sha256(t.encode()).digest() for _, t in package.texts}
    for label, path in files:
        text = read_text(path)
        if text is None:
            continue
        digest = hashlib.sha256(text.encode()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        package.texts.append((label, text))


def license_name(meta: metadata.PackageMetadata) -> str:
    """Declared license of a distribution, as short as available."""
    expression = meta.get("License-Expression")
    if expression:
        return expression.strip()
    declared = (meta.get("License") or "").strip()
    if declared and "\n" not in declared and len(declared) <= 80:
        return declared
    classifiers = [
        c.split(" :: ")[-1]
        for c in meta.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers)
    return "see the license text below" if declared else "not declared"


def source_url(meta: metadata.PackageMetadata) -> str:
    """Source or home page URL of a distribution, if declared."""
    urls = {}
    for entry in meta.get_all("Project-URL") or []:
        label, _, url = entry.partition(",")
        urls[label.strip().lower()] = url.strip()
    for key in ("source", "source code", "repository", "homepage", "home"):
        if key in urls:
            return urls[key]
    return meta.get("Home-page") or next(iter(urls.values()), "")


def python_packages() -> dict[str, Package]:
    """Collect every installed Python distribution, keyed by canonical
    name."""
    packages = {}
    for dist in metadata.distributions():
        meta = dist.metadata
        name = meta.get("Name")
        if not name or canonical(name) in BUILD_ONLY:
            continue
        key = canonical(name)
        if key in packages:  # found twice on sys.path
            continue
        package = Package(
            name=name,
            version=dist.version,
            license=license_name(meta),
            source=source_url(meta),
        )
        files = sorted(
            (f for f in dist.files or [] if is_license_file(Path(f))),
            key=str,
        )
        add_texts(
            package,
            [(Path(f).as_posix(), Path(dist.locate_file(f))) for f in files],
        )
        packages[key] = package
    return packages


def conda_packages(prefix: Path, skip: set[str]) -> dict[str, Package]:
    """Collect the conda packages of the environment at ``prefix``,
    except those in ``skip`` (canonical names already found as Python
    distributions)."""
    packages = {}
    meta_dir = prefix / "conda-meta"
    if not meta_dir.is_dir():
        return packages
    for record_path in sorted(meta_dir.glob("*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = record.get("name")
        if not name:
            continue
        key = canonical(name)
        if key in BUILD_ONLY or key in skip:
            continue
        package = Package(
            name=name,
            version=record.get("version", ""),
            license=record.get("license") or "not declared",
            source=record.get("url", ""),
        )
        package_dir = record.get("extracted_package_dir")
        if package_dir:
            licenses = Path(package_dir) / "info" / "licenses"
            if licenses.is_dir():
                files = sorted(p for p in licenses.rglob("*") if p.is_file())
                add_texts(
                    package,
                    [(p.relative_to(licenses).as_posix(), p) for p in files],
                )
        packages[key] = package
    return packages


def collect(prefix: Path | None = None) -> list[Package]:
    """Collect the packages of the running environment.

    Parameters
    ----------
    prefix : Path, optional
        Root of the conda environment. Defaults to ``sys.prefix``.

    Returns
    -------
    packages : list of Package
        Picasso (``picassosr``) first, then the others sorted by name.
    """
    prefix = Path(sys.prefix) if prefix is None else prefix
    found = python_packages()
    found.update(conda_packages(prefix, skip=set(found)))
    picasso = found.pop("picassosr", None)
    others = sorted(found.values(), key=lambda p: p.name.lower())
    return ([picasso] if picasso else []) + others


def _gpl_statement(packages: list[Package], picasso_version: str) -> list[str]:
    """What PyQt6's GPL means for the installation, if it bundles PyQt6:
    the license of the whole and where its source code is."""
    versions = {canonical(p.name): p.version for p in packages}
    if "pyqt6" not in versions:
        return []
    qt = versions.get("pyqt6-qt6")
    text = (
        "Picasso itself is licensed under the MIT license. Its graphical "
        f"user interface uses PyQt6 {versions['pyqt6']}, which is licensed "
        "under the GNU General Public License version 3 (GPL-3.0), and Qt"
        + (f" {qt}" if qt else "")
        + ", which is licensed under the GNU Lesser General Public License "
        "version 3 (LGPL-3.0); both license texts are reproduced below. "
        "Because of PyQt6, this installation as a whole is distributed "
        "under the terms of the GPL-3.0. The source code of each part is "
        "available: Picasso's at https://github.com/jungmannlab/picasso "
        f"(tag v{picasso_version}), PyQt6's and that of the other Python "
        "packages at https://pypi.org (the versions listed below), and "
        "Qt's at https://download.qt.io. The Qt libraries are separate "
        "files in the installation folder, so they can be replaced with "
        "other builds of the same version."
    )
    return [*textwrap.wrap(text, 70), ""]


def render(packages: list[Package], picasso_version: str) -> str:
    """Format the notices as plain text.

    Parameters
    ----------
    packages : list of Package
        As returned by ``collect``.
    picasso_version : str
        Version of Picasso the notices ship with.

    Returns
    -------
    text : str
        The full contents of ``THIRD-PARTY-LICENSES.txt``.
    """
    title = f"Licenses of the software bundled with Picasso {picasso_version}"
    out = [
        title,
        "=" * len(title),
        "",
        "This installation of Picasso includes the open-source software",
        "listed below. Each entry gives the package, its version, its",
        "license and the license and copyright notices the package ships.",
        "",
    ]
    out += _gpl_statement(packages, picasso_version)
    out += [
        "Packages",
        "--------",
        "",
    ]
    out += [f"  {p.name} {p.version} ({p.license})" for p in packages]
    # a package that ships no license file (e.g. streamlit) points to
    # another entry whose license is declared identically
    reproduced = {}
    for p in packages:
        if p.texts:
            reproduced.setdefault(p.license, p.name)
    for p in packages:
        out += ["", "", RULE, f"{p.name} {p.version}", f"License: {p.license}"]
        if p.source:
            out.append(f"Source: {p.source}")
        out.append(RULE)
        if not p.texts:
            note = "(no license file shipped with the package"
            if p.license in reproduced:
                note += (
                    f"; the {p.license} license text is reproduced under "
                    f"{reproduced[p.license]})"
                )
            else:
                note += ")"
            out += ["", note]
        for label, text in p.texts:
            out += ["", f"--- {label} ".ljust(78, "-"), "", text]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the notices to the path given on the command line."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    packages = collect()
    if not packages or packages[0].name != "picassosr":
        print(
            "ERROR: picassosr is not installed in this environment; run "
            "the script in the installer build environment.",
            file=sys.stderr,
        )
        return 1
    output = Path(argv[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(packages, packages[0].version), encoding="utf-8")
    missing = [p.name for p in packages if not p.texts]
    print(f"Wrote the licenses of {len(packages)} packages to {output}")
    if missing:
        print(
            "WARNING: no license file found for "
            + ", ".join(missing)
            + "; only their declared license is listed.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
