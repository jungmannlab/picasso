"""Test the third-party license notices written for the installers.

``release/collect_third_party_licenses.py`` collects the licenses of
everything the one-click installers bundle. A license file it fails to
recognize, or a conda package whose texts it does not find, would ship
the installer without a notice its license requires.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "release" / "collect_third_party_licenses.py"


@pytest.fixture(scope="module")
def licenses():
    """The release script, imported as a module."""
    spec = importlib.util.spec_from_file_location("licenses", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # registered before running it: dataclasses look the module up there
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules[spec.name]


@pytest.mark.parametrize(
    "path",
    [
        "czifile-2026.6.12.dist-info/licenses/LICENSE",
        "numpy-2.4.4.dist-info/licenses/numpy/random/LICENSE.md",
        "imagecodecs/licenses/LICENSE-libjpeg-turbo",
        "pyqt6_qt6-6.11.0.dist-info/LICENSE",
        "matplotlib/mpl-data/fonts/ttf/LICENSE_DEJAVU",
        "tqdm-4.67.3.dist-info/licenses/LICENCE",
        "foo/COPYING.txt",
        "foo/NOTICE",
    ],
)
def test_license_files_are_recognized(licenses, path):
    assert licenses.is_license_file(Path(path))


@pytest.mark.parametrize(
    "path",
    [
        "foo/license.py",
        "foo/__pycache__/license.cpython-314.pyc",
        "foo/licenses/__init__.py",
        "foo/README.md",
        "foo/core.py",
    ],
)
def test_other_files_are_not_license_files(licenses, path):
    assert not licenses.is_license_file(Path(path))


def write_conda_package(prefix, cache, name, license_text):
    """Write a conda-meta record and its cached license folder."""
    package_dir = cache / f"{name}-1.0-h0"
    if license_text is not None:
        (package_dir / "info" / "licenses").mkdir(parents=True)
        (package_dir / "info" / "licenses" / "LICENSE").write_text(
            license_text
        )
    record = {
        "name": name,
        "version": "1.0",
        "license": "MIT",
        "url": f"https://conda.example/{name}-1.0-h0.conda",
        "extracted_package_dir": str(package_dir),
    }
    (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)
    (prefix / "conda-meta" / f"{name}-1.0-h0.json").write_text(
        json.dumps(record)
    )


def test_conda_packages_read_cached_license_texts(licenses, tmp_path):
    prefix, cache = tmp_path / "env", tmp_path / "pkgs"
    write_conda_package(prefix, cache, "openssl", "Apache text")
    write_conda_package(prefix, cache, "libsqlite", None)  # cache cleaned
    write_conda_package(prefix, cache, "pip", "pip text")  # build only
    write_conda_package(prefix, cache, "numpy", "numpy text")  # also pip

    packages = licenses.conda_packages(prefix, skip={"numpy"})

    assert sorted(packages) == ["libsqlite", "openssl"]
    assert packages["openssl"].texts == [("LICENSE", "Apache text")]
    assert packages["openssl"].license == "MIT"
    assert packages["libsqlite"].texts == []


def test_conda_packages_without_conda_meta(licenses, tmp_path):
    assert licenses.conda_packages(tmp_path, skip=set()) == {}


def test_duplicate_and_empty_files_are_dropped(licenses, tmp_path):
    (tmp_path / "a").write_text("Same text\n")
    (tmp_path / "b").write_text("Same text")
    (tmp_path / "c").write_text("  \n")
    (tmp_path / "d").write_bytes(b"binary\0data")
    package = licenses.Package("foo", "1.0", "MIT")

    licenses.add_texts(
        package, [(n, tmp_path / n) for n in ("a", "b", "c", "d", "missing")]
    )

    assert package.texts == [("a", "Same text")]


def test_render_points_missing_texts_to_a_reproduced_license(licenses):
    packages = [
        licenses.Package("picassosr", "1.0", "MIT", texts=[("L", "MIT")]),
        licenses.Package(
            "msgpack", "1.0", "Apache-2.0", texts=[("L", "Apache")]
        ),
        licenses.Package("streamlit", "1.0", "Apache-2.0"),
        licenses.Package("libsqlite", "1.0", "blessing"),
    ]

    text = licenses.render(packages, "1.0")

    assert "Picasso 1.0" in text.splitlines()[0]
    assert "  streamlit 1.0 (Apache-2.0)" in text
    assert "license text is reproduced under msgpack" in text
    assert text.count("(no license file shipped with the package)") == 1


def test_collect_lists_picasso_first(licenses):
    packages = licenses.collect()

    names = [p.name.lower() for p in packages]
    assert names[0] == "picassosr"
    assert "numpy" in names
    assert not set(names) & licenses.BUILD_ONLY
    numpy = packages[names.index("numpy")]
    assert numpy.texts, "numpy ships its license in its dist-info"


def test_pyinstaller_notices_are_kept(licenses):
    # the app contains PyInstaller's bootloader and run-time hooks
    assert "pyinstaller" not in licenses.BUILD_ONLY
    assert "pyinstaller-hooks-contrib" not in licenses.BUILD_ONLY


def test_each_license_of_adapted_code_is_referenced():
    # every file in LICENSES covers code in Picasso, which names it
    root = Path(__file__).resolve().parent.parent
    sources = "\n".join(
        path.read_text("utf-8")
        for path in [root / "readme.rst", *(root / "picasso").rglob("*.py")]
    )
    for license_file in (root / "LICENSES").glob("*.txt"):
        assert f"LICENSES/{license_file.name}" in sources, license_file.name


def test_gpl_statement_names_the_versions_and_sources(licenses):
    packages = [
        licenses.Package("picassosr", "1.0", "MIT"),
        licenses.Package("PyQt6", "6.11.0", "GPL-3.0-only"),
        licenses.Package("PyQt6-Qt6", "6.11.1", "LGPL-3.0-only"),
    ]
    text = " ".join(licenses.render(packages, "1.0").split())
    assert "PyQt6 6.11.0, which is licensed under the GNU General" in text
    assert "and Qt 6.11.1, which is licensed under the GNU Lesser" in text
    assert "as a whole is distributed under the terms of the GPL-3.0" in text
    assert "github.com/jungmannlab/picasso (tag v1.0)" in text


def test_no_gpl_statement_without_pyqt6(licenses):
    packages = [licenses.Package("picassosr", "1.0", "MIT")]
    assert "GPL" not in licenses.render(packages, "1.0")
