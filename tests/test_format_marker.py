"""The format marker on every file Picasso writes in nm.

Files written before Picasso 1.0 store lengths in camera pixels; the
marker (``picasso_format``, ``io.FORMAT_VERSION_ATTR``) tells a current
file from them, so a file written without it is converted to nm a
second time when it is loaded. Each writer is checked with
``legacy.needs_conversion``, which decides that conversion.

:author: Rafal Kowalewski, 2026
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import json
import os
import pathlib
import re

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

import picasso
from picasso import __main__ as cli
from picasso import io, legacy, lib, spinna, transforms

PS = 130.0  # nm
INFO = [{"Width": 32, "Height": 32, "Frames": 100, "Pixelsize": PS}]


def _locs(n: int = 50, seed: int = 0, scale: float = PS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "frame": rng.integers(0, 100, n).astype(np.uint32),
            "x": (rng.uniform(0, 32, n) * scale).astype(np.float32),
            "y": (rng.uniform(0, 32, n) * scale).astype(np.float32),
            "lpx": np.full(n, 0.1 * scale, np.float32),
            "lpy": np.full(n, 0.1 * scale, np.float32),
            "photons": np.full(n, 1000.0, np.float32),
        }
    )


def _assert_current(path: str, kind: str | None) -> None:
    """``path`` is a file of ``kind`` with the format marker."""
    assert legacy.file_kind(path) == kind
    assert not legacy.needs_conversion(path)


def _write_locs(tmp_path) -> str:
    path = str(tmp_path / "locs.hdf5")
    io.save_locs(path, _locs(), INFO)
    return path


def _write_pick_properties(tmp_path) -> str:
    path = str(tmp_path / "pick_props.hdf5")
    groups = pd.DataFrame({"group": [0, 1], "com_x": [100.0, 200.0]})
    io.save_datasets(path, INFO, groups=groups)
    return path


def _write_picks(tmp_path) -> str:
    path = str(tmp_path / "picks.yaml")
    io.save_picks(path, [(100.0, 200.0)], "Circle", size=150.0)
    return path


def _write_drift(tmp_path) -> str:
    path = str(tmp_path / "drift.txt")
    io.save_drift(path, pd.DataFrame({"x": [0.0, 1.5], "y": [0.0, -2.0]}))
    return path


def _write_z_calibration(tmp_path) -> str:
    path = str(tmp_path / "calib.yaml")
    io.save_calibration(
        path, {"X Coefficients": [0.0, 1.0], "Y Coefficients": [0.0, 1.0]}
    )
    return path


def _write_lateral_calibration(tmp_path) -> str:
    path = str(tmp_path / "chromatic.yaml")
    shift = transforms.TranslationTransform.from_shift((10.0, -5.0))
    io.save_calibration(
        path,
        {
            lib.LATERAL_TRANSFORMS_KEY: [
                {"Type": "chromatic", "Transform": shift.to_dict()}
            ]
        },
    )
    return path


def _write_spline_calibration(tmp_path) -> str:
    path = str(tmp_path / "spline_calib.hdf5")
    io.save_spline_calibration(
        path,
        {"coefficients": np.zeros((2, 64)), "model": "spline-3d"},
    )
    return path


def _write_mask(tmp_path) -> str:
    locs_path = _write_locs(tmp_path)
    mask = spinna.MaskGenerator(locs_path, binsize=200, sigma=300, ndim=2)
    mask.generate_mask()
    path = str(tmp_path / "mask.yaml")
    mask.save_mask_info(path)
    return path


WRITERS = {
    "locs": (_write_locs, "locs"),
    "pick properties": (_write_pick_properties, "locs"),
    "picks": (_write_picks, "picks"),
    "drift": (_write_drift, "drift"),
    "z calibration": (_write_z_calibration, "calibration"),
    "lateral calibration": (_write_lateral_calibration, "calibration"),
    "spline calibration": (_write_spline_calibration, "spline"),
    "SPINNA mask": (_write_mask, "mask"),
}


class TestWriters:
    @pytest.mark.parametrize("name", WRITERS)
    def test_written_with_the_marker(self, name, tmp_path):
        write, kind = WRITERS[name]
        _assert_current(write(tmp_path), kind)

    def test_identifications(self, tmp_path):
        path = str(tmp_path / "ids.hdf5")
        ids = pd.DataFrame({"frame": [0], "x": [3], "y": [4], "n_id": [0]})
        io.save_identifications(path, ids, INFO)
        assert io.get_format_version(path) == (1, 0)


def _write_legacy_locs(tmp_path) -> str:
    path = str(tmp_path / "old_locs.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("locs", data=_locs(scale=1.0).to_records(index=False))
        f.create_dataset("metadata", data=json.dumps(INFO))
    return path


def _write_legacy_yaml(name: str, content: dict):
    def write(tmp_path) -> str:
        path = str(tmp_path / name)
        with open(path, "w") as f:
            yaml.dump(content, f)
        return path

    return write


def _write_legacy_drift(tmp_path) -> str:
    path = str(tmp_path / "old_drift.txt")
    np.savetxt(path, np.tile([0.5, -0.25], (4, 1)))
    return path


def _write_legacy_spline(tmp_path) -> str:
    path = str(tmp_path / "old_spline.hdf5")
    with h5py.File(path, "w") as f:
        f.create_dataset("coefficients", data=np.zeros((2, 64), "f4"))
        f.attrs["metadata"] = json.dumps(
            {"model": "spline-3d", "pixelsize": PS}
        )
    return path


LEGACY_FILES = {
    "locs": _write_legacy_locs,
    "picks": _write_legacy_yaml(
        "old_picks.yaml",
        {"Shape": "Circle", "Centers": [[1.0, 2.0]], "Diameter": 1.0},
    ),
    "drift": _write_legacy_drift,
    "calibration": _write_legacy_yaml(
        "old_calib.yaml",
        {"X Coefficients": [0.0, 1.0], "Y Coefficients": [0.0, 1.0]},
    ),
    "spline": _write_legacy_spline,
    "mask": _write_legacy_yaml(
        "old_mask.yaml",
        {"Binsize (nm)": [100.0, 100.0], "x_min": 0.0, "x_max": 32.0},
    ),
    "nanotron": _write_legacy_yaml(
        "old_model.yaml", {"Pick Diameter": 1.0, "Oversampling": 50}
    ),
}


def test_every_kind_is_covered():
    assert set(LEGACY_FILES) == set(legacy.FILE_KINDS)


@pytest.mark.parametrize("kind", legacy.FILE_KINDS)
def test_converted_files_carry_the_marker(kind, tmp_path):
    path = LEGACY_FILES[kind](tmp_path)
    assert legacy.file_kind(path) == kind
    assert legacy.needs_conversion(path)
    _assert_current(legacy.convert_file(path, pixelsize=PS), kind)


class TestCommandLine:
    def _write_clusters(self, path: str, marker: bool) -> pd.DataFrame:
        """A file of the cluster format that ``link`` updates."""
        clusters = pd.DataFrame(
            {"groups": [0, 1], "com_x": [3.0, 9.0], "com_y": [4.0, 1.0]}
        )
        with h5py.File(path, "w") as f:
            if marker:
                f.attrs[io.FORMAT_VERSION_ATTR] = io.FORMAT_VERSION
            f.create_dataset("clusters", data=clusters.to_records(index=False))
            f.create_dataset("metadata", data=json.dumps(INFO))
        return clusters

    def _link(self, tmp_path) -> None:
        locs = _locs(n=40)
        locs["frame"] = np.arange(40, dtype=np.uint32)
        locs["group"] = np.repeat([0, 1], 20).astype(np.int32)
        path = str(tmp_path / "sample_picked.hdf5")
        io.save_locs(path, locs, INFO)
        cli._link(path, d_max=0.3 * PS, tolerance=1)
        assert os.path.isfile(str(tmp_path / "sample_picked_link.hdf5"))

    @pytest.mark.parametrize("marker", [True, False])
    def test_link_keeps_the_units_of_the_clusters_file(self, marker, tmp_path):
        path = str(tmp_path / "sample_clusters.hdf5")
        clusters = self._write_clusters(path, marker)
        self._link(tmp_path)
        with h5py.File(path, "r") as f:
            assert "linked_n" in f["clusters"].dtype.names
            assert (io.FORMAT_VERSION_ATTR in f.attrs) == marker
            np.testing.assert_allclose(f["clusters"]["com_x"], clusters.com_x)
        loaded = io.load_clusters(path)
        scale = 1.0 if marker else PS
        np.testing.assert_allclose(loaded["com_x"], clusters["com_x"] * scale)

    def test_nneighbor_of_old_clusters_is_nm(self, tmp_path):
        path = str(tmp_path / "old_clusters.hdf5")
        self._write_clusters(path, marker=False)
        cli._nneighbor(path)
        minval = np.loadtxt(str(tmp_path / "old_clusters_minval.txt"))
        np.testing.assert_allclose(minval, np.hypot(6.0, 3.0) * PS)


HDF5_WRITE = re.compile(
    r"h5py\.File\([^)]*[\"'](?:w|w-|x|a)[\"']|\.to_hdf\(|HDFStore\("
)


def test_hdf5_files_are_written_by_io_only():
    """Every HDF5 file Picasso creates goes through ``picasso.io``,
    whose writers add the format marker (PyME export excepted, a format
    of python-microscopy)."""
    package = pathlib.Path(picasso.__file__).parent
    found = []
    for source in package.rglob("*.py"):
        if source.name == "io.py" and source.parent == package:
            continue
        for number, line in enumerate(source.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if HDF5_WRITE.search(code):
                found.append(f"{source.relative_to(package)}:{number}")
    assert found == []


# GUI writers of files that are not written through ``picasso.io``'s
# file writers (Render's FOV, picks and view files are tested in
# test_gui_render_units.py)


def test_3d_view_settings(qt_offscreen, tmp_path):
    from picasso.gui import render as gui_render

    window = gui_render.Window(plugins_loaded=True)
    try:
        locs = _locs()
        locs["z"] = np.zeros(len(locs), np.float32)
        window.view.add(str(tmp_path / "locs.hdf5"), locs, INFO, render_=False)
        window.view.viewport = [(0.0, 0.0), (32 * PS, 32 * PS)]
        window.open_3d_view()
        path = str(tmp_path / "view_3d.png")
        window.window_rot.view_rot.export_current_view_info(path)
        info = io.load_info(str(tmp_path / "view_3d.yaml"))
        assert info[0][io.FORMAT_VERSION_ATTR] == io.FORMAT_VERSION
    finally:
        window.window_rot.view_rot.stop_render_worker()
        window.view.stop_render_worker()
        window.close()


def test_nanotron_model(qt_offscreen, tmp_path, monkeypatch):
    from picasso.gui import nanotron as gui_nanotron

    path = str(tmp_path / "model.sav")
    monkeypatch.setattr(
        gui_nanotron.lib,
        "get_save_filename_ext_dialog",
        lambda *args, **kwargs: (path, ".sav"),
    )
    window = gui_nanotron.Window()
    try:
        dialog = window.train_dialog
        dialog.mlp = {"a": "picklable stand-in for the classifier"}
        dialog.train_log.update(
            {"Pick Diameter (nm)": 2 * PS, "Display pixel size (nm)": 10.0}
        )
        dialog.save_model()
        _assert_current(str(tmp_path / "model.yaml"), "nanotron")
    finally:
        window.close()
