"""Smoke test of an installed Picasso: the package from PyPI (see
``.github/workflows/pypi_release.yml``) or a one-click build (the build
scripts in ``release`` run it before packaging the app).

It checks what a broken package or build breaks first. The ``picasso``
command runs a short analysis, from a simulated movie through
``localize``, ``link`` and ``render``, and every GUI opens its window
(offscreen) and quits cleanly. For the Python package, it also checks
that the core imports without Qt and that the version and the bundled
files are right.

The Python package, run outside the repository so that the installed
package is tested rather than the source folder::

    cd /tmp
    python /path/to/picasso/release/smoke_test.py

A one-click build, with only the Python standard library::

    python release/smoke_test.py --frozen C:\\Picasso\\picasso.exe
    python release/smoke_test.py --frozen Picasso.app/Contents/MacOS/picasso

``--gui-only`` tests only the GUIs, e.g., of the windowed
``picassow.exe``, which writes no output to check.

All commands run with a temporary home folder, so the user settings of
the computer are not touched.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import argparse
import array
import importlib
import importlib.metadata
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

#: package data the GUIs and Localize read at runtime
BUNDLED_FILES = ("config_template.yaml", "gui/icons/picasso.ico")
CORE_MODULES = ("io", "localize", "postprocess", "render", "spinna")
#: the GUIs of the one-click installers, except the browser-based server
GUIS = (
    "design",
    "simulate",
    "localize",
    "filter",
    "render",
    "average",
    "spinna",
    "nanotron",
)
#: seconds a command may take, e.g., the first start of a one-click build
TIMEOUT = 600


def fail(message: str) -> None:
    sys.exit(f"Smoke test failed: {message}")


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def run(command: list[str], env: dict, **kwargs) -> str:
    """Run ``command``; fail with its output unless it succeeds."""
    name = " ".join(os.path.basename(part) for part in command[:2])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=TIMEOUT,
            **kwargs,
        )
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or "") + (error.stderr or "")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        fail(f"{name} did not finish within {TIMEOUT} s:\n{output}")
    check(
        result.returncode == 0,
        f"{name} exited with code {result.returncode}:\n"
        f"{result.stdout}{result.stderr}",
    )
    return result.stdout


def check_package() -> None:
    """The installed Python package: imports, version, bundled files."""
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


def write_movie(path: str, n_frames: int = 20, size: int = 32) -> None:
    """Gaussian spots (sigma 1 px) blinking on a noisy background, as a
    ``.raw`` movie with its ``.yaml`` metadata."""
    rng = random.Random(0)
    spots = [(8.3, 9.6), (22.7, 12.1), (15.2, 24.4)]
    pixels = array.array("H")
    for frame in range(n_frames):
        for y in range(size):
            for x in range(size):
                mean = 200.0
                for i, (sx, sy) in enumerate(spots):
                    if (frame + i) % 3:  # on in two of three frames
                        r2 = (x - sx) ** 2 + (y - sy) ** 2
                        mean += 1500 * math.exp(-r2 / 2)
                value = rng.gauss(mean, math.sqrt(mean))
                pixels.append(min(65535, max(0, round(value))))
    if sys.byteorder != "little":
        pixels.byteswap()
    with open(path, "wb") as file:
        pixels.tofile(file)
    with open(os.path.splitext(path)[0] + ".yaml", "w") as file:
        file.write(
            "Byte Order: <\n"
            "Data Type: uint16\n"
            f"Frames: {n_frames}\n"
            f"Height: {size}\n"
            f"Width: {size}\n"
        )


def check_analysis(picasso: list[str], folder: str, env: dict) -> None:
    """``picasso localize``, ``link`` and ``render`` on a movie."""
    help_text = run([*picasso, "--help"], env)
    check("localize" in help_text, "picasso --help lists no commands")
    movie = os.path.join(folder, "movie.raw")
    write_movie(movie)
    # camera parameters and pixel size given, so no config is needed
    run(
        [
            *picasso,
            "localize",
            movie,
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
        ],
        env,
    )
    locs = os.path.join(folder, "movie_locs.hdf5")
    check(os.path.isfile(locs), "picasso localize wrote no localizations")
    run([*picasso, "link", locs], env)
    check(
        os.path.isfile(os.path.join(folder, "movie_locs_link.hdf5")),
        "picasso link wrote no linked localizations",
    )
    run([*picasso, "render", locs, "-s"], env)
    image = os.path.join(folder, "movie_locs.png")
    check(
        os.path.isfile(image) and os.path.getsize(image) > 0,
        "picasso render wrote no image",
    )


def check_package_results(folder: str) -> None:
    """The localizations of ``check_analysis``, read with the API."""
    from picasso import io, render

    locs, info = io.load_locs(os.path.join(folder, "movie_locs.hdf5"))
    check(len(locs) > 20, f"only {len(locs)} localizations found")
    _, image = render.render(locs, info, disp_px_size=26)
    check(image.sum() > 0, "the rendered image is empty")
    print(f"{len(locs)} localizations")


def check_guis(picasso: list[str], env: dict) -> None:
    """Every GUI opens its window offscreen and quits cleanly (see
    ``PICASSO_QUIT_AFTER_START`` in ``picasso.gui.app.run_gui``)."""
    env = dict(env, QT_QPA_PLATFORM="offscreen", PICASSO_QUIT_AFTER_START="1")
    for name in GUIS:
        run([*picasso, name], env)
        print(f"picasso {name} started")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--frozen",
        metavar="PATH",
        help="executable of a one-click build to test, e.g., picasso.exe",
    )
    parser.add_argument(
        "--gui-only",
        action="store_true",
        help="test only that the GUIs start, e.g., of picassow.exe",
    )
    args = parser.parse_args()
    if args.frozen:
        check(os.path.isfile(args.frozen), f"{args.frozen} does not exist")
        picasso = [os.path.abspath(args.frozen)]
    else:
        check_package()
        executable = shutil.which("picasso")
        check(executable is not None, "the picasso command is not installed")
        picasso = [executable]

    home = tempfile.mkdtemp(prefix="picasso_smoke_home_")
    # no update checks: the test needs no network
    os.makedirs(os.path.join(home, ".picasso"))
    with open(os.path.join(home, ".picasso", "settings.yaml"), "w") as file:
        file.write("Updates:\n  Disabled: true\n")
    folder = tempfile.mkdtemp(prefix="picasso_smoke_")
    env = dict(os.environ, HOME=home, USERPROFILE=home)
    if sys.platform == "win32":
        # Windows resolves the AppData folders (e.g., Numba's cache) inside
        # USERPROFILE and fails if they do not exist
        for name, subfolder in (
            ("APPDATA", "Roaming"),
            ("LOCALAPPDATA", "Local"),
        ):
            env[name] = os.path.join(home, "AppData", subfolder)
            os.makedirs(env[name])
    try:
        if not args.gui_only:
            check_analysis(picasso, folder, env)
            if not args.frozen:
                check_package_results(folder)
        check_guis(picasso, env)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(home, ignore_errors=True)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
