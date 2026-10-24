"""Install Picasso from this repository into the build environment of a
one-click installer, e.g., in ``create_installer_windows_cuda.bat``::

    python release/install_picasso.py installer_cuda

installs ``picassosr[installer_cuda]`` with the Python that runs it.

Picasso is built from a temporary copy of the repository, not from the
repository itself: a build writes ``build``, ``dist`` and
``picassosr.egg-info`` into the folder it builds, so the standard and
the CUDA Windows installers, built at the same time, would overwrite
each other's files. The copy also means that every build has the
current code, never a wheel left over from an earlier build of the same
version.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: copied with the files at the top of the repository (pyproject.toml,
#: the readme and the license), i.e., all that the package is built from
FOLDERS = ("picasso", "LICENSES")
#: compiled files, e.g., the numba cache in picasso/__pycache__
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbi", "*.nbc")


def copy_sources(target: str) -> None:
    os.makedirs(target)
    for name in os.listdir(REPO):
        path = os.path.join(REPO, name)
        if os.path.isfile(path):
            shutil.copy2(path, target)
    for name in FOLDERS:
        shutil.copytree(
            os.path.join(REPO, name),
            os.path.join(target, name),
            ignore=IGNORE,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "extra", help="the optional dependencies, e.g., installer_cuda"
    )
    args = parser.parse_args()
    folder = tempfile.mkdtemp(prefix="picasso_install_")
    try:
        source = os.path.join(folder, "picasso")
        copy_sources(source)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"{source}[{args.extra}]"]
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if result.returncode:
        sys.exit(f"Installing Picasso failed (exit code {result.returncode}).")


if __name__ == "__main__":
    main()
