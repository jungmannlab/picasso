"""Smoke test of an installed Picasso, e.g., the wheel about to be
published (see ``.github/workflows/pypi_release.yml``).

It checks what a broken package would break first: the package and
its GUIs import, the core without Qt; the version and the bundled files
are right; the ``picasso`` command runs; and a short analysis works,
from a simulated movie through ``picasso localize`` to a rendered image.

Run it outside the repository, so that the installed package is tested
rather than the source folder::

    cd /tmp
    python /path/to/picasso/release/pypi/smoke_test.py

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

#: package data the GUIs and Localize read at runtime
BUNDLED_FILES = ("config_template.yaml", "gui/icons/picasso.ico")
CORE_MODULES = ("io", "localize", "postprocess", "render", "spinna")
GUI_MODULES = ("render", "localize", "filter", "simulate", "spinna")


def check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"Smoke test failed: {message}")


def check_imports() -> None:
    import picasso

    for name in CORE_MODULES:
        importlib.import_module(f"picasso.{name}")
    print(f"picasso {picasso.__version__} from {picasso.__file__}")
    version = importlib.metadata.version("picassosr")
    check(
        picasso.__version__ == version,
        f"picasso.__version__ is {picasso.__version__}, the installed "
        f"package {version}",
    )
    check(
        not any(name.startswith("PyQt6") for name in sys.modules),
        "importing the core modules loads PyQt6",
    )
    package = os.path.dirname(picasso.__file__)
    for name in BUNDLED_FILES:
        check(
            os.path.isfile(os.path.join(package, name)),
            f"{name} is missing from the package",
        )


def check_command_line() -> None:
    picasso = shutil.which("picasso")
    check(picasso is not None, "the picasso command is not installed")
    result = subprocess.run(
        [picasso, "--help"], capture_output=True, text=True
    )
    check(
        result.returncode == 0 and "localize" in result.stdout,
        f"picasso --help failed:\n{result.stdout}{result.stderr}",
    )


def _movie(n_frames: int = 20, shape=(32, 32), seed: int = 0) -> np.ndarray:
    """Gaussian spots (sigma 1 px) blinking on a Poisson background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    spots = [(8.3, 9.6), (22.7, 12.1), (15.2, 24.4)]
    frames = []
    for frame in range(n_frames):
        image = np.full(shape, 200.0)
        for i, (x, y) in enumerate(spots):
            if (frame + i) % 3:  # each spot is on in two of three frames
                image += 1500 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2)
        frames.append(rng.poisson(image))
    return np.stack(frames).astype(np.uint16)


def check_analysis(folder: str) -> None:
    from picasso import io, postprocess, render

    movie = _movie()
    path = os.path.join(folder, "movie.raw")
    io.save_raw(
        path,
        movie,
        [
            {
                "Byte Order": "<",
                "Data Type": "uint16",
                "Frames": movie.shape[0],
                "Height": movie.shape[1],
                "Width": movie.shape[2],
            }
        ],
    )
    # the defaults of the camera parameters, given so that no config is
    # needed; -px gives the pixel size Render needs
    command = [
        shutil.which("picasso"),
        "localize",
        path,
        "-b",
        "7",
        "-g",
        "1000",
        "-a",
        "lq",
        "-d",
        "0",
        "-bl",
        "0",
        "-s",
        "1",
        "-ga",
        "1",
        "-px",
        "130",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    check(
        result.returncode == 0,
        f"picasso localize failed:\n{result.stdout}{result.stderr}",
    )
    locs, info = io.load_locs(os.path.join(folder, "movie_locs.hdf5"))
    check(len(locs) > 20, f"only {len(locs)} localizations found")
    linked = postprocess.link(locs, info)
    check(0 < len(linked) <= len(locs), "linking failed")
    _, image = render.render(locs, info, disp_px_size=26)
    check(image.sum() > 0, "the rendered image is empty")
    print(f"{len(locs)} localizations, {len(linked)} binding events")


def check_gui() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for name in GUI_MODULES:
        importlib.import_module(f"picasso.gui.{name}")
    app.quit()


def main() -> None:
    check_imports()
    check_command_line()
    folder = tempfile.mkdtemp(prefix="picasso_smoke_")
    try:
        check_analysis(folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    check_gui()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
